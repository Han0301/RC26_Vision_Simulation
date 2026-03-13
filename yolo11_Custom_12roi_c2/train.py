import multiprocessing
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset     # subset: PyTorch 数据集子集化工具，基于索引提取数据集的一部分
from torch.optim.lr_scheduler import CosineAnnealingLR      # 余弦退火学习率调度器，让学习率按余弦函数周期性衰减
from torchvision import transforms      # torchvision 的图像变换模块，用于数据增强 / 预处理

# 自定义模块
from dataset import ROI12ImageDataset
from model import YOLO11ROIClassifier, calculate_2c_metrics, evaluate, load_yolo11_pretrained_weights
from loss import YOLO11ROIFocalLoss2C,YOLO11ROICOUNTLOSS

if __name__ == '__main__':
    # 解决Windows多进程启动的bootstrap问题（核心修改2）
    multiprocessing.freeze_support()
    # ===================== 1. 核心配置 =====================
    # 1.1 模型本身相关
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ROI_IMG_SIZE = 64       # roi图像大小
    NUM_ROI = 12            # roi数量
    NUM_CLASSES = 2         # 分类数
    MODEL_SIZE = "s"        # 模型尺寸

    # 1.2 数据集和训练相关
    BATCH_SIZE = 16         # 加载图像的批次
    EPOCHS = 100            # 训练总轮数
    VAL_RATIO = 0.2         # 验证集的占比
    patience = 8           # 耐心
    mixup_rate = 0.2        # mixup触发的概率
    mixup_alpha = 0.2       # mixup增强的beta 分布参数
    DATASET_ROOT = r"H:\pycharm\yolov11\yolov11_proj1\datasets_4018"       # 数据集路径
    SAVE_DIR = "./checkpoints"      # 输出的模型路径

    # 1.3 损失函数和优化器相关
    LOSS_WEIGHT= [4.0, 2.0]    # 损失在两个类别上面的权重
    FOCAL_LOSS = 1.5                # 难样本挖掘系数
    LEARNING_RATE = 5e-5 if MODEL_SIZE == "l" else 1e-4 if MODEL_SIZE == "s" else 1e-3          # 学习率
    WEIGHT_DECAY = 5e-4         # 权重衰减（L2 正则），防止模型过拟合
    count_loss_weight = 0.25    # 数量约束损失的权重

    # ===================== 2. 数据预处理 =====================
    # 2.1 归一化和标准差
    yolo11_mean = [0.485, 0.456, 0.406]
    yolo11_std = [0.229, 0.224, 0.225]

    # 2.2 数据增强
    train_transform = transforms.Compose([
        transforms.ToPILImage(),        # 将 numpy 数组 / 张量转为 PIL 图像（因为多数变换仅支持 PIL 格式）。
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=(0,0.1)),      # 颜色的扰动
        transforms.RandomHorizontalFlip(p=0.5),         # 50% 概率随机水平翻转
        transforms.RandomRotation(15),                  # 随机旋转 ±15 度
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.8, 1.2), shear=10),     # 随机仿射变换（平移 / 缩放 / 剪切），degrees=0 表示不旋转
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),       # 随机高斯模糊（核大小 3，sigma 范围 0.1~2.0）
        transforms.ToTensor(),                                          # 将 PIL 图像转为张量
        transforms.Normalize(mean=yolo11_mean, std=yolo11_std)          # 同时将像素值从 [0,255] 归一化到 [0,1]
    ])

    # 验证集仅做基础变换（保证评估准确）
    val_test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=yolo11_mean, std=yolo11_std)
    ])

    # ===================== 3. 加载数据集 =====================
    print("=== 正在加载数据集 ===")
    # 3.1 先创建一个临时数据集 仅用于计算长度和生成索引
    temp_dataset = ROI12ImageDataset(dataset_root=DATASET_ROOT, roi_img_size=ROI_IMG_SIZE, transform=None)
    dataset_size = len(temp_dataset)
    val_size = int(VAL_RATIO * dataset_size)
    train_size = dataset_size - val_size

    # 3.2 生成随机且互斥的索引
    # 注意：这里为了确保可复现性，可以固定一个seed，也可以不固定
    indices = torch.randperm(dataset_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # 3.3 实例化两个完全独立的 Dataset 对象
    # 这样它们的 transform 互不干扰
    train_dataset_full = ROI12ImageDataset(dataset_root=DATASET_ROOT, roi_img_size=ROI_IMG_SIZE, transform=train_transform)
    val_dataset_full = ROI12ImageDataset(dataset_root=DATASET_ROOT, roi_img_size=ROI_IMG_SIZE, transform=val_test_transform)

    # 3.4 使用 Subset 根据索引包装
    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)

    # 3.5 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=False,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=False,
                            drop_last=True)

    print(f"=== 数据集划分完成 ===")
    print(f"训练集：{train_size}样本 | {len(train_loader)}批次")
    print(f"验证集：{val_size}样本 | {len(val_loader)}批次")
    print(f"训练设备：{DEVICE} | 模型尺寸：YOLO11-{MODEL_SIZE.upper()}")
    print("=" * 80)

    # ===================== 4 初始化模型/损失/优化器 =====================
    # 4.1 加载模型
    model = YOLO11ROIClassifier(
        model_size=MODEL_SIZE,
        num_roi=NUM_ROI,
        num_classes=NUM_CLASSES,
        roi_size=ROI_IMG_SIZE
    ).to(DEVICE)

    # 4.2 加载预训练权重
    model = load_yolo11_pretrained_weights(model, model_size=MODEL_SIZE,
                                           load_path="H:\pycharm\yolov11\yolov11.pt\yolo11s.pt")

    # 4.3 加载损失
    cls_loss_fn = YOLO11ROIFocalLoss2C(
        num_roi=NUM_ROI,
        num_classes=NUM_CLASSES,
        alpha=LOSS_WEIGHT,         # 损失在三个类别上面的权重
        gamma=FOCAL_LOSS,          # Focal Loss 难样本挖掘系数
        max_positive=8,
        max_negative=4
    ).to(DEVICE)

    count_loss_fn = YOLO11ROICOUNTLOSS(
        exist_count=8,
        weight=count_loss_weight
    )
    # 4.4 参数的冻结策略
    for name, param in model.backbone.named_parameters():
        if "layer0" in name or "layer1" in name or "layer2" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True


    # 4.5 分层学习率
    param_groups = [
        {"params": [p for n, p in model.backbone.named_parameters() if "layer0" in n or "layer1" in n or "layer2" in n],
         "lr": LEARNING_RATE * 0.001},
        {"params": [p for n, p in model.backbone.named_parameters() if "layer0" not in n and "layer1" not in n and "layer2" not in n],
         "lr": LEARNING_RATE * 0.1},
        {"params": model.neck.parameters(), "lr": LEARNING_RATE * 0.5},
        {"params": model.head.parameters(), "lr": LEARNING_RATE}
    ]

    # 4.6 优化器
    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)      # adamw: 带权重衰减的优化器

    # 4.7 学习率调度器(余弦退火)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # ===================== 5 训练循环 =====================
    # 5.1 创建模型输出路径
    os.makedirs(SAVE_DIR, exist_ok=True)        # exist_ok=True  路径存在也不报错
    best_pos_f1 = 0.0   # 最佳 f1 值
    no_improve = 0      # 无提升早停计数器

    # 5.2 训练循环
    print(f"=== 开始训练（YOLO11-{MODEL_SIZE.upper()}） ===")
    for epoch in range(EPOCHS):
        # 5.2.1 将模型转为训练模式, 初始化指标
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        valid_batch_count = 0       # 用于计算指标的批次

        train_total_acc = 0.0
        train_valid_acc = 0.0
        train_pos_acc, train_pos_precision, train_pos_recall, train_pos_f1 = 0.0, 0.0, 0.0, 0.0
        train_neg_acc, train_neg_precision, train_neg_recall, train_neg_f1 = 0.0, 0.0, 0.0, 0.0

        train_iter = iter(train_loader)             # 将 dataloader 转为迭代器

        # 5.2.2 从dataloader 中取数据并进行前向传播和反向传播
        for batch_idx, (roi_imgs, cls_target, roi_valid_mask) in enumerate(train_loader):
            roi_imgs = roi_imgs.to(DEVICE)
            cls_target = cls_target.to(DEVICE)

            is_mixup = False
            loss = 0.0
            pred_logits = None

            # ---------------------- MixUp增强开始 ----------------------
            if np.random.rand() < mixup_rate:
                is_mixup = True
                # 1. 从train_dataset中随机采样一个单样本
                mixup_idx = np.random.choice(len(train_dataset))
                roi_imgs2, cls_target2, _ = train_dataset[mixup_idx]

                # 2. 关键修复：扩展到批次维度（和原批次B保持一致）
                B = roi_imgs.shape[0]  # 获取当前批次大小
                roi_imgs2 = roi_imgs2.unsqueeze(0).repeat(B, 1, 1, 1, 1).to(DEVICE)  # [12,3,64,64] → [B,12,3,64,64]
                cls_target2 = cls_target2.unsqueeze(0).repeat(B, 1).to(DEVICE)  # [12] → [B,12]

                lam = np.random.beta(mixup_alpha, mixup_alpha)
                roi_imgs_mix = lam * roi_imgs + (1 - lam) * roi_imgs2

                pred_logits = model(roi_imgs_mix)
                cls_loss1 = cls_loss_fn(pred_logits, cls_target)
                cls_loss2 = cls_loss_fn(pred_logits, cls_target2)
                cls_loss = lam * cls_loss1 + (1 - lam) * cls_loss2
                count_loss = count_loss_fn(pred_logits)
            else:
                pred_logits = model(roi_imgs)
                cls_loss = cls_loss_fn(pred_logits, cls_target)
                count_loss = count_loss_fn(pred_logits)

            # ---------------------- MixUp增强结束 ----------------------

            total_loss = cls_loss + count_loss
            optimizer.zero_grad()       # 清空梯度(防止累积)
            total_loss.backward()             # 损失的反向传播
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)            # 梯度裁剪(防止梯度爆炸)
            optimizer.step()            # 更新参数

            epoch_loss += total_loss.item()   # 累加当前的损失
            batch_count += 1

            # 只有在非 MixUp 时才计算指标
            with torch.no_grad():
                if not is_mixup:
                    metrics = calculate_2c_metrics(pred_logits, cls_target)  # 改为二分类指标函数
                    train_total_acc += metrics["total_acc"]
                    train_pos_acc += metrics["pos_metrics"]["acc"]
                    train_pos_precision += metrics["pos_metrics"]["precision"]
                    train_pos_recall += metrics["pos_metrics"]["recall"]
                    train_pos_f1 += metrics["pos_metrics"]["f1"]
                    valid_batch_count += 1

            # 打印日志
            if (batch_idx + 1) % 10 == 0:
                current_lr = optimizer.param_groups[-1]['lr']
                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}] | Batch [{batch_idx + 1}/{len(train_loader)}] | Loss: {total_loss.item():.4f} | LR: {current_lr:.6f}")

        # 5.2.3 更新学习率
        scheduler.step()

        # 计算训练集平均指标（二分类）
        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
        avg_train_total_acc = train_total_acc / valid_batch_count if valid_batch_count > 0 else 0.0
        avg_train_pos_acc = train_pos_acc / valid_batch_count if valid_batch_count > 0 else 0.0
        avg_train_pos_precision = train_pos_precision / valid_batch_count if valid_batch_count > 0 else 0.0
        avg_train_pos_recall = train_pos_recall / valid_batch_count if valid_batch_count > 0 else 0.0
        avg_train_pos_f1 = train_pos_f1 / valid_batch_count if valid_batch_count > 0 else 0.0

        # 验证集评估（适配二分类）
        val_metrics = evaluate(model, val_loader, cls_loss_fn, DEVICE)
        (avg_val_loss, val_roi_avg_loss, avg_val_total_acc,
         avg_val_pos_acc, avg_val_pos_precision, avg_val_pos_recall, avg_val_pos_f1) = val_metrics

        # 打印日志（关键修改：移除三分类相关打印）
        print("=" * 120)
        print(f"【Epoch {epoch + 1}/{EPOCHS} 训练集】")
        print(f"总损失：{avg_epoch_loss:.4f} | 整体准确率：{avg_train_total_acc:.4f}")
        print(f"└─ 有方块（正样本）：准确率={avg_train_pos_acc:.4f} | 精确率={avg_train_pos_precision:.4f} | 召回率={avg_train_pos_recall:.4f} | F1={avg_train_pos_f1:.4f}")

        print(f"【Epoch {epoch + 1}/{EPOCHS} 验证集】")
        print(f"总损失：{avg_val_loss:.4f} | 整体准确率：{avg_val_total_acc:.4f}")
        print(f"└─ 有方块（正样本）：准确率={avg_val_pos_acc:.4f} | 精确率={avg_val_pos_precision:.4f} | 召回率={avg_val_pos_recall:.4f} | F1={avg_val_pos_f1:.4f}")
        print("=" * 120)

        # 5.2.7 早停+保存模型
        if avg_val_pos_f1 > best_pos_f1:
            best_pos_f1 = avg_val_pos_f1
            no_improve = 0
            save_path = os.path.join(SAVE_DIR, f"yolo11_{MODEL_SIZE}_roi_best_2c.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_pos_f1': best_pos_f1,
                'loss': avg_val_loss,
            }, save_path)
            print(f"✅ 保存最优模型 | 有效有方块F1：{avg_val_pos_f1:.4f} | 路径：{save_path}")
        else:
            no_improve += 1
            print(f"⚠️ 正样本F1未提升 | 当前最优：{best_pos_f1:.4f} | 无提升轮数：{no_improve}/{patience}")
            if no_improve >= patience:
                print("🚨 早停触发")
                break

        # 5.2.8 保存本轮模型
        epoch_save_path = os.path.join(SAVE_DIR, f"yolo11_{MODEL_SIZE}_roi_epoch_{epoch + 1}_2c.pt")
        torch.save(model.state_dict(), epoch_save_path)

    print("=== 训练完成 ===")
    print(f"最优模型路径：{os.path.join(SAVE_DIR, f'yolo11_{MODEL_SIZE}_roi_best_2c.pt')}")
    print(f"最优正样本F1：{best_pos_f1:.4f}")