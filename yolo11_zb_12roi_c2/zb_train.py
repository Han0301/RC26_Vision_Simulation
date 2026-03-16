# ===================== 导入依赖（参数区之后） =====================
import os
import cv2
import numpy as np
import torch
import multiprocessing
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
# 修正混合精度导入（兼容新旧版本）
try:
    from torch.amp import GradScaler, autocast  # PyTorch 2.0+ 推荐
except ImportError:
    from torch.cuda.amp import GradScaler, autocast  # 兼容旧版本

from zb_infer import group_separation_loss
from zb_main import process_zbuffer_with_rt_batch  # 导入批量函数
# 自定义模块导入
from zb_dataset import ZBGlobalImageDataset
from model import YOLO11ROIClassifier, calculate_2c_metrics, load_yolo11_pretrained_weights
from loss import YOLO11ROIFocalLoss2C, YOLO11ROICOUNTLOSS


# 设备配置 + GPU优化（新增核心）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 开启cudnn优化，提升卷积速度
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 模型配置
MODEL_SIZE = "s"  # YOLO11尺寸：n(轻量)/s(中等)/l(大型)
ROI_IMG_SIZE = 64  # ROI图像尺寸（固定64x64）
NUM_ROI = 12  # 每个样本的ROI数量（固定12）
NUM_CLASSES = 2  # 分类数（二分类：0=无方块，1=有方块）
PRETRAINED_PATH = "yolo11s.pt"  # YOLO11预训练权重路径

# 训练配置（可根据GPU显存调整）
BATCH_SIZE = 4  # 批次大小（全局图批次，建议16/32）
EPOCHS = 120  # 总训练轮数
VAL_RATIO = 0.2  # 验证集占比
PATIENCE = 12  # 早停耐心值（无提升轮数）
MAX_CYCLES = 7  # 单样本最大循环推理次数
LEARNING_RATE = 1e-4  # 基础学习率
WEIGHT_DECAY = 5e-4  # 权重衰减（防止过拟合）

# 损失函数配置
LOSS_WEIGHT = [4.0, 2.0]  # Focal Loss的类别权重
FOCAL_GAMMA = 1.5  # Focal Loss的gamma值
COUNT_LOSS_WEIGHT = 0.0  # 计数损失权重（暂时关闭）

# 数据路径配置
DATASET_ROOT = r"../global_datasets_150"  # 全局图像数据集根路径
SAVE_DIR = "./zb_checkpoints"  # 模型保存路径

# ===================== 核心后处理函数（保持不变） =====================
def post_process_prob(prob, last_low_indices=None, max_empty=4, history_probs=None):
    """
    完全保留原输入输出，按新逻辑实现：
    1. 第一轮：取概率最低2位，仅置空前9个位置中的对应位置
    2. 第n轮(n≥2)：位置概率加权（轮数为权重），取n+1个最低加权概率置空（第4轮最多4个）
    3. 第四轮强制收敛，循环必在第4轮结束
    输入：prob, last_low_indices=None, max_empty=4, history_probs=None
    输出：exist_boxes_new, is_converged, current_low_indices, k, prob_hist_mean, prob_hist_std, global_low_quantile, low_prob_indices
    """
    # 1. 初始化历史概率 + 确定当前轮数
    history_probs = history_probs if history_probs is not None else []
    all_probs = history_probs + [prob.copy()]
    current_round = len(all_probs)  # 当前轮数：1=第一轮，2=第二轮，3=第三轮，4=第四轮

    # 2. 概率预处理：裁剪极端值
    prob_np = np.clip(prob.copy(), 1e-6, 1 - 1e-6)

    # 3. 初始化返回用的统计变量（保持输出兼容）
    prob_hist_mean = np.zeros_like(prob_np)
    prob_hist_std = np.zeros_like(prob_np)
    global_low_quantile = 0.0
    low_prob_indices = []

    # ===================== 核心逻辑：按轮数执行不同策略 =====================
    if current_round == 1:
        # -------------------- 第一轮逻辑 --------------------
        # 取概率最低的2个位置
        sorted_indices = np.argsort(prob_np)  # 升序排列，前2个是概率最低的
        lowest_2_indices = sorted_indices[:2].tolist()

        # 筛选这2个位置中属于前9个位置（索引0-8）的，作为置空位置
        current_low_indices = [idx for idx in lowest_2_indices if idx < 9]
        # 兜底：如果前9个位置中没有，至少取1个最低的（保证k≥1）
        if len(current_low_indices) == 0:
            current_low_indices = [sorted_indices[0]]

        k = len(current_low_indices)  # 第一轮置空数量
        is_converged = False  # 第一轮不收敛

        # 填充兼容用的统计变量
        prob_hist_mean = prob_np
        global_low_quantile = np.quantile(prob_np, 0.25)
        low_prob_indices = current_low_indices

    else:
        # -------------------- 第2/3/4轮逻辑 --------------------
        # 1. 计算加权概率（权重=轮数：第1轮权重1，第2轮权重2...）
        weights = np.array([i + 1 for i in range(len(all_probs))])  # 轮数作为权重
        weighted_prob = np.average(np.array(all_probs), axis=0, weights=weights)
        # 2. 确定当前轮数n和置空数量
        n = current_round  # 2/3/4
        empty_num = n + 1  # 第二轮取3个，第三轮取4个，第四轮取5个（但最多4个）
        empty_num = min(empty_num, max_empty)  # 第四轮强制限制为4个

        # 3. 取加权概率最低的empty_num个位置作为置空位置
        sorted_weighted_indices = np.argsort(weighted_prob)  # 升序，前empty_num个是最低的
        current_low_indices = sorted_weighted_indices[:empty_num].tolist()
        k = empty_num  # 置空数量

        # 4. 填充兼容用的统计变量
        prob_hist_mean = weighted_prob  # 加权均值作为历史均值
        prob_hist_std = np.std(np.array(all_probs), axis=0)  # 历史方差
        global_low_quantile = np.quantile(weighted_prob, 0.25)
        low_prob_indices = current_low_indices

        # 5. 收敛条件：第四轮强制收敛，确保循环在第4轮结束
        is_converged = True if current_round == 4 else False
        result = group_separation_loss(weighted_prob, current_low_indices, alpha=1.0, beta=0.1)

    # 4. 生成新的exist_boxes（置空对应位置）
    exist_boxes_new = np.ones(12, dtype=int)
    exist_boxes_new[current_low_indices] = 0

    # ===================== 返回值完全不变 =====================
    return exist_boxes_new, is_converged, current_low_indices, k, prob_hist_mean, prob_hist_std, global_low_quantile, low_prob_indices


# ===================== 批量后处理函数（优化：给必填参数加默认值，避免漏传） =====================
def batch_post_process_prob(pred_probs_batch, last_low_indices_batch=None, max_empty=4, history_probs=None):
    """
    批量后处理概率，生成批量exist_boxes
    输入：
        pred_probs_batch: (batch_size, 12) 批量概率
        last_low_indices_batch: list of None/List 批量上一轮低概率索引（新增默认值None）
        max_empty: 最大置空数量（默认4）
        history_probs: list of list 批量历史概率（默认None）
    输出：
        exist_boxes_new_batch: (batch_size, 12) 批量新exist_boxes
        is_converged_batch: (batch_size,) 批量收敛标记
    """
    # 修复1：处理last_low_indices_batch的默认值（避免漏传时报错）
    batch_size = len(pred_probs_batch)
    if last_low_indices_batch is None:
        last_low_indices_batch = [None for _ in range(batch_size)]

    # 修复2：history_probs默认值初始化
    history_probs = history_probs or [[] for _ in range(batch_size)]

    exist_boxes_new_batch = []
    is_converged_batch = []

    for b in range(batch_size):
        exist_boxes_new, is_converged, _, _, _, _, _, _ = post_process_prob(
            pred_probs_batch[b],
            last_low_indices=last_low_indices_batch[b],
            max_empty=max_empty,
            history_probs=history_probs[b]
        )
        exist_boxes_new_batch.append(exist_boxes_new)
        is_converged_batch.append(is_converged)

    return np.stack(exist_boxes_new_batch, axis=0), is_converged_batch

# ===================== 批量ROI预处理函数（新增核心） =====================
def preprocess_roi_images_batch(batch_roi_np, roi_img_size=64, transform=None):
    """
    批量预处理ROI图像
    输入：
        batch_roi_np: (batch_size, 12, 160, 160, 3) 批量ROI图像（np.uint8）
        roi_img_size: 目标尺寸
        transform: 预处理变换
    输出：
        roi_tensor: (batch_size*12, 3, roi_img_size, roi_img_size) 模型输入张量
    """
    batch_size = batch_roi_np.shape[0]
    roi_list = []

    for b in range(batch_size):
        for roi_idx in range(12):
            roi = batch_roi_np[b, roi_idx]
            roi_resized = cv2.resize(roi, (roi_img_size, roi_img_size))
            if transform is not None:
                roi_tensor = transform(roi_resized)
            else:
                roi_tensor = torch.from_numpy(roi_resized).permute(2, 0, 1).float() / 255.0
            roi_list.append(roi_tensor)

    # 堆叠为 (batch_size*12, 3, H, W)，适配模型输入
    roi_tensor = torch.stack(roi_list, dim=0)
    return roi_tensor

# ===================== 批量循环推理函数（修复最终后处理的调用） =====================
def infer_batch_samples(global_imgs_np, rvecs, tvecs, labels_batch, model, transform,
                        device, max_cycles=7, roi_img_size=64):
    """
    批量循环推理：一次处理整个batch，充分利用GPU并行
    输入：
        global_imgs_np: (batch_size, H, W, C) 批量全局图像（np.uint8）
        rvecs: (batch_size, 3, 1) 批量旋转向量（np.float32）
        tvecs: (batch_size, 3, 1) 批量平移向量（np.float32）
        labels_batch: (batch_size, 12) 批量标签（tensor）
        model: 模型实例
        transform: 预处理变换
        device: 设备
        max_cycles: 最大循环次数
        roi_img_size: ROI目标尺寸
    输出：
        批量推理结果字典
    """
    batch_size = len(global_imgs_np)
    # 初始化每个样本的exist_boxes（batch_size, 12）
    exist_boxes_batch = np.ones((batch_size, 12), dtype=int)
    last_low_indices_batch = [None for _ in range(batch_size)]
    prob_history_batch = [[] for _ in range(batch_size)]
    cycle_num = 0

    # 批量指标收集
    batch_cycle_metrics = {
        "acc": [[] for _ in range(batch_size)],
        "f1": [[] for _ in range(batch_size)]
    }
    final_prob_batch = np.zeros((batch_size, 12), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        while cycle_num < max_cycles:
            # 1. 批量生成ROI（核心：一次处理整个batch）
            batch_roi_np = process_zbuffer_with_rt_batch(
                global_imgs_np, rvecs, tvecs, exist_boxes_batch
            )
            # 2. 批量预处理ROI
            roi_tensor_batch = preprocess_roi_images_batch(
                batch_roi_np, roi_img_size, transform
            ).to(device)
            # 3. 模型批量推理（GPU并行处理）
            pred_logits = model(roi_tensor_batch)  # (batch_size*12, 2)
            pred_logits_reshaped = pred_logits.reshape(batch_size, 12, 2)  # (batch_size, 12, 2)
            pred_probs = torch.softmax(pred_logits_reshaped, dim=-1)[:, :, 1].cpu().numpy()  # (batch_size, 12)

            # 4. 批量计算指标
            for b in range(batch_size):
                metrics = calculate_2c_metrics(
                    pred_logits_reshaped[b:b + 1],
                    labels_batch[b:b + 1].to(device)
                )
                batch_cycle_metrics["acc"][b].append(metrics["total_acc"])
                batch_cycle_metrics["f1"][b].append(metrics["pos_metrics"]["f1"])

            # 5. 批量后处理（参数完整传递）
            exist_boxes_new_batch, is_converged_batch = batch_post_process_prob(
                pred_probs,
                last_low_indices_batch=last_low_indices_batch,
                max_empty=4,
                history_probs=prob_history_batch
            )

            # 6. 更新历史概率
            for b in range(batch_size):
                prob_history_batch[b].append(pred_probs[b])

            # 7. 检查收敛
            if all(is_converged_batch) or cycle_num >= max_cycles - 1:
                break

            # 8. 更新状态
            exist_boxes_batch = exist_boxes_new_batch
            last_low_indices_batch = [None] * batch_size
            cycle_num += 1

    # 9. 计算最终概率
    for b in range(batch_size):
        if len(prob_history_batch[b]) > 1:
            weights = np.linspace(0.1, 1.0, len(prob_history_batch[b]))
            weights = weights / weights.sum()
            final_prob_batch[b] = np.average(prob_history_batch[b], axis=0, weights=weights)
        else:
            final_prob_batch[b] = prob_history_batch[b][0] if prob_history_batch[b] else pred_probs[b]

    # 10. 最终后处理（修复：补充参数，或利用新的默认值）
    # 方式1：显式传递last_low_indices_batch（推荐，可读性高）
    final_exist_batch, is_converged_final_batch = batch_post_process_prob(
        final_prob_batch,
        last_low_indices_batch=[None] * batch_size,  # 最终后处理无历史索引，传全None
        max_empty=4
    )
    # 方式2：利用函数默认值（简化写法）
    # final_exist_batch, is_converged_final_batch = batch_post_process_prob(final_prob_batch, max_empty=4)

    # 11. 计算最终指标
    final_pred_logits = torch.tensor(final_prob_batch).unsqueeze(-1).to(device)  # (batch_size, 12, 1)
    final_pred_logits = torch.cat([1 - final_pred_logits, final_pred_logits], dim=-1)  # (batch_size, 12, 2)
    final_metrics_batch = []

    for b in range(batch_size):
        final_metrics = calculate_2c_metrics(
            final_pred_logits[b:b + 1],
            labels_batch[b:b + 1].to(device)
        )
        final_metrics_batch.append(final_metrics)

    # 整理返回结果
    result = {
        "final_prob": final_prob_batch,
        "final_exist": final_exist_batch,
        "cycle_num": cycle_num + 1,
        "cycle_metrics": batch_cycle_metrics,
        "is_converged": is_converged_final_batch,
        "prob_history": prob_history_batch,
        "final_metrics": final_metrics_batch
    }
    return result

# ===================== 批量验证函数（替换原有） =====================
def evaluate(model, val_loader, transform, device, max_cycles=7, roi_img_size=64):
    """批量验证函数：完全基于批量推理"""
    model.eval()
    cycle_num_list = []
    is_converged_list = []
    avg_acc_per_cycle_list = []
    avg_f1_list = []
    final_acc_list = []
    final_f1_list = []
    per_cycle_acc = []
    per_cycle_f1 = []

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            global_imgs = batch_data["global_img"]
            labels = batch_data["labels"].to(device)
            rvecs = batch_data["rvec"]
            tvecs = batch_data["tvec"]

            # 1. 转换为numpy格式（批量）
            global_imgs_np = (global_imgs.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
            rvecs_np = rvecs.cpu().numpy()
            tvecs_np = tvecs.cpu().numpy()

            # 2. 批量推理
            infer_result = infer_batch_samples(
                global_imgs_np=global_imgs_np,
                rvecs=rvecs_np,
                tvecs=tvecs_np,
                labels_batch=labels,
                model=model,
                transform=transform,
                device=device,
                max_cycles=max_cycles,
                roi_img_size=roi_img_size
            )

            # 3. 收集指标
            batch_size = global_imgs.shape[0]
            cycle_num_list.extend([infer_result["cycle_num"]] * batch_size)
            is_converged_list.extend(infer_result["is_converged"])

            for b in range(batch_size):
                # 每轮平均指标
                acc_list = infer_result["cycle_metrics"]["acc"][b]
                f1_list = infer_result["cycle_metrics"]["f1"][b]
                avg_acc = np.mean(acc_list) if acc_list else 0.0
                avg_f1 = np.mean(f1_list) if f1_list else 0.0
                avg_acc_per_cycle_list.append(avg_acc)
                avg_f1_list.append(avg_f1)

                # 最终指标
                final_metrics = infer_result["final_metrics"][b]
                final_acc_list.append(final_metrics["total_acc"])
                final_f1_list.append(final_metrics["pos_metrics"]["f1"])

                # 每轮详细指标
                for cycle_idx in range(len(acc_list)):
                    if cycle_idx >= len(per_cycle_acc):
                        per_cycle_acc.append([])
                        per_cycle_f1.append([])
                    per_cycle_acc[cycle_idx].append(acc_list[cycle_idx])
                    per_cycle_f1[cycle_idx].append(f1_list[cycle_idx])

    # 计算验证集全局指标
    val_metrics = {
        "avg_converge_cycles": np.mean(cycle_num_list) if cycle_num_list else 0.0,
        "converge_rate": np.mean(is_converged_list) if is_converged_list else 0.0,
        "avg_acc_per_cycle": np.mean(avg_acc_per_cycle_list) if avg_acc_per_cycle_list else 0.0,
        "avg_f1": np.mean(avg_f1_list) if avg_f1_list else 0.0,
        "final_total_acc": np.mean(final_acc_list) if final_acc_list else 0.0,
        "final_pos_f1": np.mean(final_f1_list) if final_f1_list else 0.0,
        "per_cycle_avg_acc": [np.mean(acc) for acc in per_cycle_acc] if per_cycle_acc else [],
        "per_cycle_avg_f1": [np.mean(f1) for f1 in per_cycle_f1] if per_cycle_f1 else [],
        "cycle_num_list": cycle_num_list,
        "is_converged_list": is_converged_list,
        "final_acc_list": final_acc_list,
        "final_f1_list": final_f1_list
    }

    return val_metrics

# ===================== 训练循环函数（核心批量优化） =====================
def train_zb_model():
    # ===================== 1. 数据预处理 =====================
    yolo11_mean = [0.485, 0.456, 0.406]
    yolo11_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=(0, 0.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=yolo11_mean, std=yolo11_std)
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=yolo11_mean, std=yolo11_std)
    ])

    # ===================== 2. 加载数据集（优化数据加载） =====================
    print("=== 加载全局图像数据集 ===")
    temp_dataset = ZBGlobalImageDataset(dataset_root=DATASET_ROOT, transform=None)
    dataset_size = len(temp_dataset)
    val_size = int(VAL_RATIO * dataset_size)
    train_size = dataset_size - val_size

    indices = torch.randperm(dataset_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = ZBGlobalImageDataset(dataset_root=DATASET_ROOT, transform=None)
    val_dataset = ZBGlobalImageDataset(dataset_root=DATASET_ROOT, transform=None)

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    # 优化数据加载器：多进程+预取+常驻进程
    num_workers = multiprocessing.cpu_count()
    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    print(f"训练集：{train_size}样本 | 验证集：{val_size}样本")

    # ===================== 3. 模型初始化 =====================
    model = YOLO11ROIClassifier(
        model_size=MODEL_SIZE,
        num_roi=NUM_ROI,
        num_classes=NUM_CLASSES,
        roi_size=ROI_IMG_SIZE
    ).to(DEVICE)

    model = load_yolo11_pretrained_weights(model, model_size=MODEL_SIZE, load_path=PRETRAINED_PATH)

    # ===================== 4. 损失/优化器配置（修复混合精度） =====================
    cls_loss_fn = YOLO11ROIFocalLoss2C(
        num_roi=NUM_ROI,
        num_classes=NUM_CLASSES,
        alpha=LOSS_WEIGHT,
        gamma=FOCAL_GAMMA,
        max_positive=8,
        max_negative=4
    ).to(DEVICE)

    count_loss_fn = YOLO11ROICOUNTLOSS(exist_count=8, weight=COUNT_LOSS_WEIGHT).to(DEVICE)

    # 分层学习率
    param_groups = [
        {"params": [p for n, p in model.backbone.named_parameters() if "layer0" in n or "layer1" in n or "layer2" in n],
         "lr": LEARNING_RATE * 0.001},
        {"params": [p for n, p in model.backbone.named_parameters() if
                    "layer0" not in n and "layer1" not in n and "layer2" not in n],
         "lr": LEARNING_RATE * 0.1},
        {"params": model.neck.parameters(), "lr": LEARNING_RATE * 0.5},
        {"params": model.head.parameters(), "lr": LEARNING_RATE}
    ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # 修复GradScaler初始化（适配PyTorch 2.0+）
    if torch.cuda.is_available():
        try:
            # PyTorch 2.0+ 新API
            scaler = GradScaler('cuda')
        except (TypeError, ImportError):
            # 兼容旧版本
            scaler = GradScaler()
    else:
        scaler = None  # CPU不使用混合精度

    # ===================== 5. 训练循环（批量核心） =====================
    os.makedirs(SAVE_DIR, exist_ok=True)
    best_pos_f1 = 0.0
    no_improve = 0

    print("=== 开始ZBuffer训练（批量模式） ===")
    for epoch in range(EPOCHS):
        # 训练模式
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        # 训练集指标收集
        train_cycle_num_list = []
        train_is_converged_list = []
        train_avg_acc_per_cycle_list = []
        train_avg_f1_list = []
        train_final_acc_list = []
        train_final_f1_list = []
        train_per_cycle_acc = []
        train_per_cycle_f1 = []

        for batch_idx, batch_data in enumerate(train_loader):
            # 1. 提取批次数据
            global_imgs = batch_data["global_img"]
            labels = batch_data["labels"].to(DEVICE)
            rvecs = batch_data["rvec"]
            tvecs = batch_data["tvec"]

            optimizer.zero_grad()
            batch_loss = 0.0

            # 2. 转换为numpy格式（批量）
            global_imgs_np = (global_imgs.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
            rvecs_np = rvecs.cpu().numpy()
            tvecs_np = tvecs.cpu().numpy()

            # 3. 批量循环推理（核心：无逐样本循环）
            infer_result = infer_batch_samples(
                global_imgs_np=global_imgs_np,
                rvecs=rvecs_np,
                tvecs=tvecs_np,
                labels_batch=labels,
                model=model,
                transform=train_transform,
                device=DEVICE,
                max_cycles=MAX_CYCLES,
                roi_img_size=ROI_IMG_SIZE
            )

            # 4. 批量生成最终ROI
            batch_roi_final_np = process_zbuffer_with_rt_batch(
                global_imgs_np, rvecs_np, tvecs_np, infer_result["final_exist"]
            )

            # 5. 批量预处理最终ROI
            roi_tensor_final = preprocess_roi_images_batch(
                batch_roi_final_np, ROI_IMG_SIZE, train_transform
            ).to(DEVICE)

            # 6. 混合精度前向传播（适配新旧API）
            if scaler is not None:
                with autocast('cuda'):  # 显式指定设备
                    pred_logits = model(roi_tensor_final)
                    pred_logits_reshaped = pred_logits.reshape(BATCH_SIZE, 12, 2)
                    cls_loss = cls_loss_fn(pred_logits_reshaped, labels)
                    count_loss = count_loss_fn(pred_logits_reshaped)
                    loss_batch = cls_loss + count_loss

                # 7. 混合精度反向传播
                scaler.scale(loss_batch).backward()

                # 8. 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # 9. 优化器更新
                scaler.step(optimizer)
                scaler.update()
            else:
                # CPU训练（无混合精度）
                pred_logits = model(roi_tensor_final)
                pred_logits_reshaped = pred_logits.reshape(BATCH_SIZE, 12, 2)
                cls_loss = cls_loss_fn(pred_logits_reshaped, labels)
                count_loss = count_loss_fn(pred_logits_reshaped)
                loss_batch = cls_loss + count_loss

                loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # 10. 收集指标
            batch_size_current = global_imgs.shape[0]
            train_cycle_num_list.extend([infer_result["cycle_num"]] * batch_size_current)
            train_is_converged_list.extend(infer_result["is_converged"])

            for b in range(batch_size_current):
                # 每轮平均指标
                acc_list = infer_result["cycle_metrics"]["acc"][b]
                f1_list = infer_result["cycle_metrics"]["f1"][b]
                avg_acc = np.mean(acc_list) if acc_list else 0.0
                avg_f1 = np.mean(f1_list) if f1_list else 0.0
                train_avg_acc_per_cycle_list.append(avg_acc)
                train_avg_f1_list.append(avg_f1)

                # 最终指标
                final_metrics = infer_result["final_metrics"][b]
                train_final_acc_list.append(final_metrics["total_acc"])
                train_final_f1_list.append(final_metrics["pos_metrics"]["f1"])

                # 每轮详细指标
                for cycle_idx in range(len(acc_list)):
                    if cycle_idx >= len(train_per_cycle_acc):
                        train_per_cycle_acc.append([])
                        train_per_cycle_f1.append([])
                    train_per_cycle_acc[cycle_idx].append(acc_list[cycle_idx])
                    train_per_cycle_f1[cycle_idx].append(f1_list[cycle_idx])

            # 累加损失
            batch_loss_val = loss_batch.item() * batch_size_current
            epoch_loss += batch_loss_val
            batch_count += 1

            # 打印批次日志
            if (batch_idx + 1) % 5 == 0:
                current_lr = optimizer.param_groups[-1]['lr']
                # 计算batch级平均指标
                avg_cycle_num = np.mean(train_cycle_num_list[-batch_size_current:]) if train_cycle_num_list else 0.0
                avg_acc_per_cycle = np.mean(train_avg_acc_per_cycle_list[-batch_size_current:]) if train_avg_acc_per_cycle_list else 0.0
                avg_f1_batch = np.mean(train_avg_f1_list[-batch_size_current:]) if train_avg_f1_list else 0.0
                final_acc_batch = np.mean(train_final_acc_list[-batch_size_current:]) if train_final_acc_list else 0.0
                final_f1_batch = np.mean(train_final_f1_list[-batch_size_current:]) if train_final_f1_list else 0.0

                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}] | Batch [{batch_idx + 1}/{len(train_loader)}] | Loss: {batch_loss_val:.4f} | LR: {current_lr:.6f} "
                    f"| Avg Cycle: {avg_cycle_num:.2f} | Avg Acc/Cycle: {avg_acc_per_cycle:.4f} "
                    f"| Avg F1: {avg_f1_batch:.4f} | Final Acc: {final_acc_batch:.4f} | Final F1: {final_f1_batch:.4f}"
                )

        # 更新学习率
        scheduler.step()

        # ===================== 6. 计算训练集全局指标 =====================
        train_metrics = {
            "avg_converge_cycles": np.mean(train_cycle_num_list) if train_cycle_num_list else 0.0,
            "converge_rate": np.mean(train_is_converged_list) if train_is_converged_list else 0.0,
            "avg_acc_per_cycle": np.mean(train_avg_acc_per_cycle_list) if train_avg_acc_per_cycle_list else 0.0,
            "avg_f1": np.mean(train_avg_f1_list) if train_avg_f1_list else 0.0,
            "final_total_acc": np.mean(train_final_acc_list) if train_final_acc_list else 0.0,
            "final_pos_f1": np.mean(train_final_f1_list) if train_final_f1_list else 0.0,
            "per_cycle_avg_acc": [np.mean(acc) for acc in train_per_cycle_acc] if train_per_cycle_acc else [],
            "per_cycle_avg_f1": [np.mean(f1) for f1 in train_per_cycle_f1] if train_per_cycle_f1 else []
        }

        # ===================== 7. 验证集评估 =====================
        print(f"=== Epoch {epoch + 1} 验证集评估 ===")
        val_metrics = evaluate(
            model=model,
            val_loader=val_loader,
            transform=val_transform,
            device=DEVICE,
            max_cycles=MAX_CYCLES,
            roi_img_size=ROI_IMG_SIZE
        )

        # 打印完整指标（训练+验证）
        print("=" * 150)
        print(f"【Epoch {epoch + 1}/{EPOCHS} 训练集全局指标】")
        print(f"总损失：{epoch_loss / batch_count:.4f} | 平均收敛轮数：{train_metrics['avg_converge_cycles']:.2f}")
        print(f"收敛率：{train_metrics['converge_rate']:.4f} | 每轮平均准确率：{train_metrics['avg_acc_per_cycle']:.4f}")
        print(f"平均F1：{train_metrics['avg_f1']:.4f} | 最终总准确率：{train_metrics['final_total_acc']:.4f}")
        print(f"最终正样本F1：{train_metrics['final_pos_f1']:.4f}")
        print(f"每轮平均准确率：{[f'{acc:.4f}' for acc in train_metrics['per_cycle_avg_acc']]}")
        print(f"每轮平均F1：{[f'{f1:.4f}' for f1 in train_metrics['per_cycle_avg_f1']]}")

        print(f"【Epoch {epoch + 1}/{EPOCHS} 验证集核心指标】")
        print(f"平均收敛轮数：{val_metrics['avg_converge_cycles']:.2f} | 收敛率：{val_metrics['converge_rate']:.4f}")
        print(f"每轮平均准确率：{val_metrics['avg_acc_per_cycle']:.4f} | 平均F1：{val_metrics['avg_f1']:.4f}")
        print(f"最终总准确率：{val_metrics['final_total_acc']:.4f} | 最终正样本F1：{val_metrics['final_pos_f1']:.4f}")
        print(f"每轮平均准确率：{[f'{acc:.4f}' for acc in val_metrics['per_cycle_avg_acc']]}")
        print(f"每轮平均F1：{[f'{f1:.4f}' for f1 in val_metrics['per_cycle_avg_f1']]}")
        print("=" * 150)

        # ===================== 8. 早停+保存模型 =====================
        current_pos_f1 = val_metrics["final_pos_f1"]
        if current_pos_f1 > best_pos_f1:
            best_pos_f1 = current_pos_f1
            no_improve = 0
            save_path = os.path.join(SAVE_DIR, f"zb_yolo11_{MODEL_SIZE}_best.pt")
            save_dict = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_pos_f1': best_pos_f1,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }
            # 仅在有scaler时保存其状态
            if scaler is not None:
                save_dict['scaler_state_dict'] = scaler.state_dict()
            torch.save(save_dict, save_path)
            print(f"✅ 保存最优模型 | 最终正样本F1：{best_pos_f1:.4f} | 路径：{save_path}")
        else:
            no_improve += 1
            print(f"⚠️ 正样本F1未提升 | 当前最优：{best_pos_f1:.4f} | 无提升轮数：{no_improve}/{PATIENCE}")
            if no_improve >= PATIENCE:
                print("🚨 早停触发")
                break

        # 保存本轮模型
        epoch_save_path = os.path.join(SAVE_DIR, f"zb_yolo11_{MODEL_SIZE}_epoch_{epoch + 1}.pt")
        torch.save(model.state_dict(), epoch_save_path)

    print("=== ZBuffer训练完成 ===")
    print(f"最优模型路径：{os.path.join(SAVE_DIR, f'zb_yolo11_{MODEL_SIZE}_best.pt')}")
    print(f"最优正样本F1：{best_pos_f1:.4f}")

# ===================== 主函数 =====================
if __name__ == '__main__':
    multiprocessing.freeze_support()
    train_zb_model()