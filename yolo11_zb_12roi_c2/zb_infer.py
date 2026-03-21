import os
import cv2
import json
import numpy as np
import torch
import traceback  # 用于打印详细堆栈信息
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import multiprocessing

# 导入训练脚本中的核心模块和函数
from model import YOLO11ROIClassifier, calculate_2c_metrics
from yolov11_proj3.yolo11Custom_R0.zb_main import process_zbuffer_with_rt
from zb_dataset import ZBGlobalImageDataset


def group_separation_loss(values, idx_A, alpha=1.0, beta=0.1, eps=1e-8):
    """
    计算组间背离+组内一致的复合损失
    :param values: 原始数值数组（1维）
    :param idx_A: 类A的索引列表（从0开始）
    :param alpha: 组内一致性损失权重
    :param beta: 组间背离损失权重
    :param eps: 防止除0的极小值
    :return: 总损失L, 类A均值μₐ, 类B均值μᵦ, 类A方差σₐ², 类B方差σᵦ², 组间均值差Δμ
    """
    # 1. 提取类A和类B的数值
    values = np.array(values, dtype=np.float32)
    X_A = values[idx_A]
    idx_B = [i for i in range(len(values)) if i not in idx_A]
    X_B = values[idx_B]

    # 2. 计算组内统计量（均值、方差）
    mu_A = np.mean(X_A)
    var_A = np.var(X_A)  # 方差（组内一致性）
    mu_B = np.mean(X_B)
    var_B = np.var(X_B)  # 方差（组内一致性）

    # 3. 计算组间统计量（均值差）
    delta_mu = np.abs(mu_B - mu_A)  # 组间背离性

    # 4. 计算复合损失
    intra_loss = alpha * (var_A + var_B)  # 组内一致性损失
    inter_loss = beta / (delta_mu + eps)  # 组间背离损失
    total_loss = intra_loss + inter_loss

    # 返回详细统计量（便于分析）
    return {
        "total_loss": total_loss,
        "mu_A": mu_A,
        "mu_B": mu_B,
        "var_A": var_A,
        "var_B": var_B,
        "delta_mu": delta_mu,
        "intra_loss": intra_loss,
        "inter_loss": inter_loss
    }


# ===================== 核心后处理函数（完全对齐训练版本） =====================
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


# ===================== ROI预处理函数（完全对齐训练版本） =====================
def preprocess_roi_images(roi_imgs, roi_img_size=64, transform=None):
    """和训练代码完全一致的ROI预处理逻辑"""
    roi_list = []
    for roi in roi_imgs:
        roi_resized = cv2.resize(roi, (roi_img_size, roi_img_size))
        if transform is not None:
            roi_tensor = transform(roi_resized)
        else:
            roi_tensor = torch.from_numpy(roi_resized).permute(2, 0, 1).float() / 255.0
        roi_list.append(roi_tensor)
    roi_stack = torch.stack(roi_list, dim=0).unsqueeze(0)
    return roi_stack


# ===================== 单样本推理函数（完全对齐训练逻辑，带详细打印） =====================
def infer_single_sample_verbose(global_img_np, label_np, rvec, tvec, model, transform,
                                device, max_cycles=7, roi_img_size=64):
    """
    单样本推理：核心逻辑100%对齐训练的infer_single_sample函数
    保留详细打印功能，便于调试
    """
    # 初始化（完全对齐训练）
    exist_boxes = np.ones(12, dtype=int)
    last_low_indices = None
    cycle_num = 0
    f1_list = []
    acc_list = []
    prob_history = []
    cycle_details = []

    # 新增：单样本位置正确率统计
    pos_correct_single = [0] * 12
    pos_total_single = [0] * 12

    model.eval()
    print("\n" + "=" * 80)
    print(f"开始单样本循环推理（最大{max_cycles}轮）")
    print("=" * 80)

    with torch.no_grad():
        while cycle_num < max_cycles:
            print(f"\n--- 第 {cycle_num + 1} 轮推理 ---")

            # 1. 生成ROI图像（和训练一致）
            roi_imgs = process_zbuffer_with_rt(global_img_np, rvec, tvec, exist_boxes.tolist())
            print(f"当前exist_boxes: {exist_boxes.tolist()} (1=保留, 0=置空)")

            # 2. ROI预处理+模型推理（和训练一致）
            roi_tensor = preprocess_roi_images(roi_imgs, roi_img_size, transform).to(device)
            pred_logits = model(roi_tensor)
            pred_prob = torch.softmax(pred_logits, dim=-1)[0, :, 1].cpu().numpy()
            prob_history.append(pred_prob)

            # 3. 计算本轮指标（匹配calculate_2c_metrics返回结构）
            label_tensor = torch.from_numpy(label_np).unsqueeze(0).to(device)
            metrics = calculate_2c_metrics(pred_logits, label_tensor)
            acc_list.append(metrics["total_acc"])  # 正确访问总准确率
            f1_list.append(metrics["pos_metrics"]["f1"])  # 正确访问正样本F1

            # 新增：统计本轮每个位置的预测结果
            pred_cls = torch.argmax(pred_logits, dim=-1).squeeze(0).cpu().numpy()
            for pos_idx in range(12):
                pos_total_single[pos_idx] += 1
                if pred_cls[pos_idx] == label_np[pos_idx]:
                    pos_correct_single[pos_idx] += 1

            # 4. 打印本轮概率信息
            print(f"本轮各位置概率: {[f'{p:.4f}' for p in pred_prob]}")
            sorted_prob_indices = np.argsort(pred_prob)
            print(f"概率排序（升序）索引: {sorted_prob_indices.tolist()}")
            print(f"概率排序（升序）值: {[f'{pred_prob[i]:.4f}' for i in sorted_prob_indices]}")

            # 5. 后处理（完全对齐训练的完整逻辑）
            exist_boxes_new, is_converged, current_low_indices, k, prob_hist_mean, prob_hist_std, global_low_quantile, low_prob_indices = post_process_prob(
                pred_prob, last_low_indices, max_empty=4, history_probs=prob_history[:-1]
            )

            # 6. 打印后处理细节（对齐训练的计算逻辑）
            print(f"全局低概率阈值（25分位数）: {global_low_quantile:.4f}")
            print(f"各位置历史均值: {[f'{p:.4f}' for p in prob_hist_mean]}")
            print(f"各位置历史方差: {[f'{p:.4f}' for p in prob_hist_std]}")
            print(f"低概率候选位置: {low_prob_indices}")
            print(f"上一轮置空位置: {last_low_indices if last_low_indices else '无'}")
            print(f"本轮置空数量k: {k}")
            print(f"本轮置空位置: {current_low_indices}")
            print(f"本轮新exist_boxes: {exist_boxes_new.tolist()}")
            print(f"是否收敛: {'是' if is_converged else '否'} (收敛条件：位置一致+方差稳定+置空数达上限)")
            print(f"本轮准确率: {metrics['total_acc']:.4f} | 本轮F1: {metrics['pos_metrics']['f1']:.4f}")

            # 7. 记录本轮细节
            cycle_details.append({
                "cycle": cycle_num + 1,
                "prob": pred_prob.tolist(),
                "empty_indices": current_low_indices,
                "k": k,
                "is_converged": is_converged,
                "acc": metrics["total_acc"],
                "f1": metrics["pos_metrics"]["f1"],
                # 新增：本轮位置预测结果
                "pos_pred_cls": pred_cls.tolist()
            })

            # 8. 收敛/最大轮数判断（和训练一致）
            if is_converged or cycle_num >= max_cycles - 1:
                print(f"\n推理终止条件: {'收敛' if is_converged else '达到最大轮数'}")
                break

            # 9. 更新状态（和训练一致）
            exist_boxes = exist_boxes_new
            last_low_indices = current_low_indices
            cycle_num += 1

    # 10. 综合最终概率（和训练一致的加权方式）
    if len(prob_history) > 1:
        weights = np.linspace(0.1, 1.0, len(prob_history))
        weights = weights / weights.sum()
        final_prob = np.average(prob_history, axis=0, weights=weights)
    else:
        final_prob = prob_history[0] if prob_history else pred_prob

    # 11. 修复：将最终概率转为正确的二分类logits
    final_prob = np.clip(final_prob, 1e-6, 1 - 1e-6)
    neg_logits = np.log(1 - final_prob)
    pos_logits = np.log(final_prob)
    final_logits = np.stack([neg_logits, pos_logits], axis=-1)
    pred_logits_final = torch.tensor(final_logits).unsqueeze(0).to(device)

    # 12. 计算最终指标（和训练一致）
    label_tensor = torch.from_numpy(label_np).unsqueeze(0).to(device)
    final_metrics = calculate_2c_metrics(pred_logits_final, label_tensor)

    # 新增：计算单样本各位置正确率
    pos_acc_single = [pos_correct_single[i] / (pos_total_single[i] + 1e-6) for i in range(12)]
    print("\n" + "=" * 80)
    print("单样本各位置正确率统计（多轮平均）")
    print("=" * 80)
    for pos_idx in range(12):
        print(
            f"位置{pos_idx + 1:2d} | 总预测次数: {pos_total_single[pos_idx]} | 正确次数: {pos_correct_single[pos_idx]} | 正确率: {pos_acc_single[pos_idx]:.4f}")

    # 13. 打印最终结果
    print("\n" + "=" * 80)
    print("最终推理结果（完全对齐训练逻辑）")
    print("=" * 80)
    print(f"实际标签: {label_np.tolist()}")
    print(f"最终概率: {[f'{p:.4f}' for p in final_prob]}")
    print(f"最终exist_boxes: {exist_boxes_new.tolist()}")
    print(f"实际循环轮数: {cycle_num + 1}")
    print(f"平均每轮准确率: {np.mean(acc_list):.4f}" if acc_list else "无")
    print(f"平均每轮F1: {np.mean(f1_list):.4f}" if f1_list else "无")
    print(f"最终总准确率: {final_metrics['total_acc']:.4f}")
    print(f"最终正样本F1值: {final_metrics['pos_metrics']['f1']:.4f}")
    print("=" * 80)

    # 整理返回结果
    metrics_dict = {
        "total_acc": final_metrics["total_acc"],
        "pos_acc": final_metrics["pos_metrics"]["acc"],
        "pos_precision": final_metrics["pos_metrics"]["precision"],
        "pos_recall": final_metrics["pos_metrics"]["recall"],
        "pos_f1": final_metrics["pos_metrics"]["f1"],
        "avg_acc_per_cycle": np.mean(acc_list) if acc_list else 0.0,
        "avg_f1_per_cycle": np.mean(f1_list) if f1_list else 0.0,
        "cycle_num": cycle_num + 1,
        "cycle_details": cycle_details,
        "final_prob": final_prob.tolist(),
        "label": label_np.tolist(),
        # 新增：位置级统计
        "pos_correct": pos_correct_single,
        "pos_total": pos_total_single,
        "pos_acc": pos_acc_single
    }

    return final_prob, exist_boxes_new, cycle_num + 1, metrics_dict


# ===================== 数据集批量推理函数（含运行时+逻辑异常捕获） =====================
def infer_dataset(dataset_root, model_path, model_size="s", roi_img_size=64,
                  max_cycles=7, batch_size=8, device="cuda"):
    """
    数据集批量推理：
    1. 捕获运行时异常样本（代码报错）
    2. 捕获逻辑异常样本（准确率/F1异常）
    3. 即时打印异常信息，最终汇总所有异常
    4. 新增：统计12个位置各自的正确率
    """
    # 逻辑异常判断阈值（可自定义）
    ACC_THRESHOLD = 0.5  # 总准确率低于该值判定为异常
    F1_THRESHOLD = 0.1  # 正样本F1低于该值判定为异常
    PROB_NAN_THRESHOLD = True  # 概率含NaN/Inf判定为异常

    # 1. 数据预处理（完全对齐训练的val_transform）
    yolo11_mean = [0.485, 0.456, 0.406]
    yolo11_std = [0.229, 0.224, 0.225]
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=yolo11_mean, std=yolo11_std)
    ])

    # 2. 加载数据集（和训练一致）
    dataset = ZBGlobalImageDataset(dataset_root=dataset_root, transform=None)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4 if device == "cuda" else 0,
        pin_memory=True if device == "cuda" else False
    )
    print(f"\n加载数据集: {dataset_root} | 样本总数: {len(dataset)}")

    # 3. 加载模型（完全对齐训练的初始化逻辑）
    model = YOLO11ROIClassifier(
        model_size=model_size,
        num_roi=12,
        num_classes=2,
        roi_size=roi_img_size
    ).to(device)

    # 兼容训练的checkpoint格式（含model_state_dict）
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"加载模型: {model_path} | 模型尺寸: {model_size}")

    # 4. 初始化统计变量
    total_samples = 0
    cycle_num_list = []
    is_converged_list = []
    avg_acc_per_cycle_list = []
    avg_f1_list = []
    final_acc_list = []
    final_f1_list = []
    per_cycle_acc = []
    per_cycle_f1 = []

    # 新增：位置级统计变量（核心修改）
    pos_total = [0] * 12  # 每个位置的总预测次数（所有样本+所有轮次）
    pos_correct = [0] * 12  # 每个位置的正确预测次数
    pos_sample_total = [0] * 12  # 每个位置的总样本数（仅最终预测）
    pos_sample_correct = [0] * 12  # 每个位置最终预测的正确数

    # 异常记录变量
    runtime_error_records = []  # 运行时异常（代码报错）
    logic_error_records = []  # 逻辑异常（结果异常）
    total_runtime_errors = 0  # 运行时异常样本数
    total_logic_errors = 0  # 逻辑异常样本数

    print("\n开始数据集批量推理（含异常监控+位置正确率统计）...")
    pbar = tqdm(total=len(dataloader), desc="推理进度")

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            global_imgs = batch_data["global_img"]
            labels = batch_data["labels"].to(device)
            rvecs = batch_data["rvec"]
            tvecs = batch_data["tvec"]

            for b in range(global_imgs.shape[0]):
                sample_idx = batch_idx * batch_size + b  # 全局样本索引
                try:
                    # -------------------- 样本推理核心逻辑 --------------------
                    # 恢复原始图像（和训练一致的反归一化）
                    global_img_tensor = global_imgs[b].cpu()
                    global_img_np = (global_img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    rvec = rvecs[b].cpu().numpy()
                    tvec = tvecs[b].cpu().numpy()
                    label_np = labels[b].cpu().numpy()
                    label_tensor = labels[b].unsqueeze(0).to(device)

                    # 单样本循环推理
                    exist_boxes = np.ones(12, dtype=int)
                    last_low_indices = None
                    cycle_num = 0
                    f1_list = []
                    acc_list = []
                    prob_history = []
                    pos_correct_sample = [0] * 12  # 该样本各位置正确数（多轮）
                    pos_total_sample = [0] * 12  # 该样本各位置总次数（多轮）

                    while cycle_num < max_cycles:
                        # 1. 生成ROI
                        roi_imgs = process_zbuffer_with_rt(global_img_np, rvec, tvec, exist_boxes.tolist())
                        roi_tensor = preprocess_roi_images(roi_imgs, roi_img_size, val_transform).to(device)

                        # 2. 模型推理
                        pred_logits = model(roi_tensor)
                        pred_prob = torch.softmax(pred_logits, dim=-1)[0, :, 1].cpu().numpy()
                        prob_history.append(pred_prob)

                        # 3. 计算本轮指标
                        metrics = calculate_2c_metrics(pred_logits, label_tensor)
                        acc_list.append(metrics["total_acc"])
                        f1_list.append(metrics["pos_metrics"]["f1"])

                        # 新增：更新该样本本轮位置统计
                        pred_cls = torch.argmax(pred_logits, dim=-1).squeeze(0).cpu().numpy()
                        for pos_idx in range(12):
                            pos_total_sample[pos_idx] += 1
                            pos_total[pos_idx] += 1  # 全局位置总次数
                            if pred_cls[pos_idx] == label_np[pos_idx]:
                                pos_correct_sample[pos_idx] += 1
                                pos_correct[pos_idx] += 1  # 全局位置正确数

                        # 4. 后处理
                        exist_boxes_new, is_converged, current_low_indices, _, _, _, _, _ = post_process_prob(
                            pred_prob, last_low_indices, max_empty=4, history_probs=prob_history[:-1]
                        )

                        # 5. 收敛判断
                        if is_converged or cycle_num >= max_cycles - 1:
                            break

                        # 6. 更新状态
                        exist_boxes = exist_boxes_new
                        last_low_indices = current_low_indices
                        cycle_num += 1

                    # -------------------- 统计样本指标 --------------------
                    actual_cycle = cycle_num + 1
                    cycle_num_list.append(actual_cycle)
                    is_converged_list.append(is_converged)
                    avg_acc_per_cycle = np.mean(acc_list) if acc_list else 0.0
                    avg_f1 = np.mean(f1_list) if f1_list else 0.0
                    avg_acc_per_cycle_list.append(avg_acc_per_cycle)
                    avg_f1_list.append(avg_f1)

                    # -------------------- 修复：最终概率转logits --------------------
                    if len(prob_history) > 1:
                        weights = np.linspace(0.1, 1.0, len(prob_history))
                        weights = weights / weights.sum()
                        final_prob = np.average(prob_history, axis=0, weights=weights)
                    else:
                        final_prob = prob_history[0] if prob_history else pred_prob

                    # 转为正确的二分类logits
                    final_prob = np.clip(final_prob, 1e-6, 1 - 1e-6)
                    neg_logits = np.log(1 - final_prob)
                    pos_logits = np.log(final_prob)
                    final_logits = np.stack([neg_logits, pos_logits], axis=-1)
                    pred_logits_final = torch.tensor(final_logits).unsqueeze(0).to(device)

                    # -------------------- 计算最终指标 --------------------
                    final_metrics = calculate_2c_metrics(pred_logits_final, label_tensor)
                    final_acc = final_metrics["total_acc"]
                    final_pos_f1 = final_metrics["pos_metrics"]["f1"]
                    final_acc_list.append(final_acc)
                    final_f1_list.append(final_pos_f1)

                    # 新增：更新最终预测的位置统计（仅最终一轮）
                    final_pred_cls = torch.argmax(pred_logits_final, dim=-1).squeeze(0).cpu().numpy()
                    for pos_idx in range(12):
                        pos_sample_total[pos_idx] += 1
                        if final_pred_cls[pos_idx] == label_np[pos_idx]:
                            pos_sample_correct[pos_idx] += 1

                    # -------------------- 收集每轮指标 --------------------
                    for cycle_idx in range(len(acc_list)):
                        if cycle_idx >= len(per_cycle_acc):
                            per_cycle_acc.append([])
                            per_cycle_f1.append([])
                        per_cycle_acc[cycle_idx].append(acc_list[cycle_idx])
                        per_cycle_f1[cycle_idx].append(f1_list[cycle_idx])

                    # -------------------- 逻辑异常判断 & 即时打印 --------------------
                    logic_error = False
                    error_reasons = []
                    # 条件1：总准确率低于阈值
                    if final_acc < ACC_THRESHOLD:
                        logic_error = True
                        error_reasons.append(f"最终总准确率 {final_acc:.4f} < 阈值 {ACC_THRESHOLD}")
                    # 条件2：正样本F1低于阈值
                    if final_pos_f1 < F1_THRESHOLD:
                        logic_error = True
                        error_reasons.append(f"正样本F1 {final_pos_f1:.4f} < 阈值 {F1_THRESHOLD}")
                    # 条件3：概率含NaN/Inf
                    if np.any(np.isnan(final_prob)) or np.any(np.isinf(final_prob)):
                        logic_error = True
                        error_reasons.append("最终概率包含NaN/Inf异常值")
                    # 条件4：收敛轮数异常（非4轮）
                    if actual_cycle != 4:
                        logic_error = True
                        error_reasons.append(f"收敛轮数 {actual_cycle} 异常（预期4轮）")

                    # 触发逻辑异常：即时打印+记录
                    if logic_error:
                        total_logic_errors += 1
                        logic_error_info = {
                            "sample_idx": sample_idx,
                            "batch_idx": batch_idx,
                            "batch_sample_idx": b,
                            "error_reasons": error_reasons,
                            "final_acc": final_acc,
                            "final_pos_f1": final_pos_f1,
                            "converge_cycles": actual_cycle,
                            "label": label_np.tolist(),
                            "final_prob": final_prob.tolist(),
                            # 新增：该样本位置预测结果
                            "final_pos_pred": final_pred_cls.tolist()
                        }
                        logic_error_records.append(logic_error_info)

                        # 即时打印逻辑异常信息（黄色标注）
                        print("\n" + "=" * 100)
                        print(f"⚠️  【样本逻辑异常】- 全局样本索引: {sample_idx}")
                        print("=" * 100)
                        print(f"📌 错误位置：批次 {batch_idx} | 批次内样本 {b}")
                        print(f"🔍 异常原因：{'; '.join(error_reasons)}")
                        print(f"📝 样本标签: {label_np.tolist()}")
                        print(f"📝 最终概率: {[f'{p:.4f}' for p in final_prob]}")
                        print(f"📝 最终预测: {final_pred_cls.tolist()}")
                        print(f"📝 最终准确率: {final_acc:.4f} | 正样本F1: {final_pos_f1:.4f}")
                        print(f"📝 收敛轮数: {actual_cycle}")
                        print("=" * 100 + "\n")

                    total_samples += 1

                # -------------------- 运行时异常捕获 --------------------
                except Exception as e:
                    total_runtime_errors += 1
                    runtime_error_info = {
                        "sample_idx": sample_idx,
                        "batch_idx": batch_idx,
                        "batch_sample_idx": b,
                        "error_type": type(e).__name__,
                        "error_msg": str(e),
                        "traceback": traceback.format_exc()
                    }
                    runtime_error_records.append(runtime_error_info)

                    # 即时打印运行时异常（红色标注）
                    print("\n" + "=" * 100)
                    print(f"❌ 【样本运行时错误】- 全局样本索引: {sample_idx}")
                    print("=" * 100)
                    print(f"📌 错误位置：批次 {batch_idx} | 批次内样本 {b}")
                    print(f"🔍 异常类型：{runtime_error_info['error_type']}")
                    print(f"📝 异常描述：{runtime_error_info['error_msg'][:100]}...")
                    print(f"📜 完整堆栈：\n{runtime_error_info['traceback']}")
                    print("=" * 100 + "\n")

                    continue

            pbar.update(1)

    pbar.close()

    # ===================== 最终统计报告 =====================
    print("\n" + "=" * 80)
    print("数据集推理统计报告（含异常监控+位置正确率）")
    print("=" * 80)

    # 核心指标
    print("\n【核心指标】")
    print(f"平均收敛轮数: {np.mean(cycle_num_list):.2f}" if cycle_num_list else 0.0)
    print(f"收敛率: {np.mean(is_converged_list):.4f}" if is_converged_list else 0.0)
    print(f"每轮平均准确率: {np.mean(avg_acc_per_cycle_list):.4f}" if avg_acc_per_cycle_list else 0.0)
    print(f"平均F1: {np.mean(avg_f1_list):.4f}" if avg_f1_list else 0.0)
    print(f"最终总准确率: {np.mean(final_acc_list):.4f}" if final_acc_list else 0.0)
    print(f"最终正样本F1: {np.mean(final_f1_list):.4f}" if final_f1_list else 0.0)

    # 新增：位置级统计（核心修改）
    print("\n【12个位置正确率统计（最终预测）】")
    print("-" * 60)
    print(f"{'位置':<6} {'总样本数':<10} {'正确数':<10} {'正确率':<10}")
    print("-" * 60)
    pos_sample_acc = []
    for pos_idx in range(12):
        acc = pos_sample_correct[pos_idx] / (pos_sample_total[pos_idx] + 1e-6)
        pos_sample_acc.append(acc)
        print(f"{pos_idx + 1:<6} {pos_sample_total[pos_idx]:<10} {pos_sample_correct[pos_idx]:<10} {acc:<10.4f}")

    print("\n【12个位置正确率统计（所有轮次平均）】")
    print("-" * 60)
    print(f"{'位置':<6} {'总预测次数':<12} {'正确次数':<12} {'正确率':<10}")
    print("-" * 60)
    pos_total_acc = []
    for pos_idx in range(12):
        acc = pos_correct[pos_idx] / (pos_total[pos_idx] + 1e-6)
        pos_total_acc.append(acc)
        print(f"{pos_idx + 1:<6} {pos_total[pos_idx]:<12} {pos_correct[pos_idx]:<12} {acc:<10.4f}")

    # 循环轮数分布
    print("\n【循环轮数分布】")
    for cycle in range(1, max_cycles + 1):
        count = sum(1 for c in cycle_num_list if c == cycle) if cycle_num_list else 0
        ratio = count / total_samples * 100 if total_samples > 0 else 0
        print(f"第{cycle}轮终止的样本数: {count} ({ratio:.2f}%)")

    # 每轮详细指标
    print("\n【每轮平均准确率/F1】")
    for cycle_idx in range(len(per_cycle_acc)):
        avg_acc = np.mean(per_cycle_acc[cycle_idx]) if per_cycle_acc[cycle_idx] else 0.0
        avg_f1 = np.mean(per_cycle_f1[cycle_idx]) if per_cycle_f1[cycle_idx] else 0.0
        print(f"第{cycle_idx + 1}轮 - 准确率: {avg_acc:.4f} | F1: {avg_f1:.4f}")

    # 异常汇总
    print("\n【❌ 异常汇总】")
    print(
        f"运行时异常样本数: {total_runtime_errors} / {len(dataset)} ({total_runtime_errors / len(dataset) * 100:.2f}%)")
    print(f"逻辑异常样本数: {total_logic_errors} / {len(dataset)} ({total_logic_errors / len(dataset) * 100:.2f}%)")

    # 运行时异常详情
    if total_runtime_errors > 0:
        print("\n【运行时异常详情】")
        for idx, err in enumerate(runtime_error_records):
            print(f"\n错误{idx + 1} - 样本{err['sample_idx']}:")
            print(f"  类型: {err['error_type']} | 描述: {err['error_msg'][:100]}...")

    # 逻辑异常详情
    if total_logic_errors > 0:
        print("\n【逻辑异常详情】")
        for idx, err in enumerate(logic_error_records):
            print(f"\n异常{idx + 1} - 样本{err['sample_idx']}:")
            print(f"  原因: {'; '.join(err['error_reasons'])}")
            print(f"  准确率: {err['final_acc']:.4f} | F1: {err['final_pos_f1']:.4f}")
            print(f"  位置预测: {err['final_pos_pred']}")

    # 返回完整结果（新增位置统计）
    return {
        "core_metrics": {
            "avg_converge_cycles": np.mean(cycle_num_list) if cycle_num_list else 0.0,
            "converge_rate": np.mean(is_converged_list) if is_converged_list else 0.0,
            "avg_acc_per_cycle": np.mean(avg_acc_per_cycle_list) if avg_acc_per_cycle_list else 0.0,
            "avg_f1": np.mean(avg_f1_list) if avg_f1_list else 0.0,
            "final_total_acc": np.mean(final_acc_list) if final_acc_list else 0.0,
            "final_pos_f1": np.mean(final_f1_list) if final_f1_list else 0.0
        },
        # 新增：位置级指标
        "position_metrics": {
            "final_pred": {  # 最终预测的位置统计
                "pos_total": pos_sample_total,
                "pos_correct": pos_sample_correct,
                "pos_acc": pos_sample_acc
            },
            "all_cycles": {  # 所有轮次的位置统计
                "pos_total": pos_total,
                "pos_correct": pos_correct,
                "pos_acc": pos_total_acc
            }
        },
        "runtime_errors": {
            "count": total_runtime_errors,
            "records": runtime_error_records
        },
        "logic_errors": {
            "count": total_logic_errors,
            "records": logic_error_records
        }
    }


# ===================== 二分类指标计算函数 =====================
def calculate_2c_metrics(pred_logits, cls_target):
    """
    计算二分类任务的核心指标（总准确率/正样本精准率/召回率/F1）
    :param pred_logits: 模型输出 → [B,12,2]（logits值）
    :param cls_target: 真实标签 → [B,12]（0=无方块，1=有方块）
    :return: 指标字典 → 包含总准确率、正样本指标
    """
    # 1. 获取预测类别
    pred_cls = torch.argmax(pred_logits, dim=-1)  # [B,12]
    B, num_roi = pred_cls.shape

    # 2. 计算总准确率
    total_correct = (pred_cls == cls_target).sum().item()
    total_acc = total_correct / (cls_target.numel() + 1e-6)

    # 3. 计算正样本（1类：有方块）指标
    pos_target_mask = (cls_target == 1)
    pos_pred_mask = (pred_cls == 1)
    pos_total = pos_target_mask.sum().item()

    # 正样本准确率
    pos_correct = (pred_cls[pos_target_mask] == cls_target[pos_target_mask]).sum().item() if pos_total > 0 else 0.0
    pos_acc = pos_correct / (pos_total + 1e-6)

    # 混淆矩阵
    tp = (pos_pred_mask & pos_target_mask).sum().item()  # 真阳性
    fn = ((~pos_pred_mask) & pos_target_mask).sum().item()  # 假阴性
    fp = (pos_pred_mask & (~pos_target_mask)).sum().item()  # 假阳性

    # 精准率、召回率、F1
    pos_precision = tp / (tp + fp + 1e-6)
    pos_recall = tp / (tp + fn + 1e-6)
    pos_f1 = 2 * pos_precision * pos_recall / (pos_precision + pos_recall + 1e-6)

    # 返回指标
    return {
        "total_acc": total_acc,
        "pos_metrics": {"acc": pos_acc, "precision": pos_precision, "recall": pos_recall, "f1": pos_f1}
    }


# ===================== 主函数 =====================
def main():
    # 全局参数
    MODE = "dataset"  # "dataset"（数据集） / "single"（单样本）
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型参数
    MODEL_SIZE = "s"
    ROI_IMG_SIZE = 64
    MAX_CYCLES = 4
    MODEL_PATH = "H:\pycharm\yolov11\yolov11_proj1\yolo11_Custom_12roi_2\model_pt\yolo11sroi_best_cls2_1.pt"

    # 数据集参数
    DATASET_ROOT = r"H:\pycharm\yolov11\yolov11_proj3\global_tests_100"
    BATCH_SIZE = 16

    # 单样本参数
    IMAGE_PATH = "H:\pycharm\yolov11\yolov11_proj3\global_datasets_150\global_images\images_1.png"
    LABEL_PATH = "H:\pycharm\yolov11\yolov11_proj3\global_datasets_150\labels\label_1.json"

    print(f"使用设备: {DEVICE}")

    if MODE == "single":
        # 加载模型
        model = YOLO11ROIClassifier(
            model_size=MODEL_SIZE,
            num_roi=12,
            num_classes=2,
            roi_size=ROI_IMG_SIZE
        ).to(DEVICE)

        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()

        # 加载图像和标签
        global_img_np = cv2.imread(IMAGE_PATH)
        if global_img_np is None:
            raise ValueError(f"无法加载图像：{IMAGE_PATH}")
        global_img_np = cv2.cvtColor(global_img_np, cv2.COLOR_BGR2RGB)

        if not os.path.exists(LABEL_PATH):
            raise ValueError(f"标签文件不存在：{LABEL_PATH}")

        with open(LABEL_PATH, 'r', encoding='utf-8') as f:
            label_data = json.load(f)

        label_np = np.array(label_data["labels"], dtype=np.float32)
        rvec = np.array(label_data["rvec"], dtype=np.float32).reshape(3, 1)
        tvec = np.array(label_data["tvec"], dtype=np.float32).reshape(3, 1)

        # 打印验证
        print(f"\n从JSON读取的标签: {label_np.tolist()}")
        print(f"从JSON读取的rvec: {rvec.flatten().tolist()}")
        print(f"从JSON读取的tvec: {tvec.flatten().tolist()}")

        # 数据预处理
        yolo11_mean = [0.485, 0.456, 0.406]
        yolo11_std = [0.229, 0.224, 0.225]
        val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=yolo11_mean, std=yolo11_std)
        ])

        # 单样本推理
        infer_single_sample_verbose(
            global_img_np=global_img_np,
            label_np=label_np,
            rvec=rvec,
            tvec=tvec,
            model=model,
            transform=val_transform,
            device=DEVICE,
            max_cycles=MAX_CYCLES,
            roi_img_size=ROI_IMG_SIZE
        )

    elif MODE == "dataset":
        # 数据集推理
        infer_result = infer_dataset(
            dataset_root=DATASET_ROOT,
            model_path=MODEL_PATH,
            model_size=MODEL_SIZE,
            roi_img_size=ROI_IMG_SIZE,
            max_cycles=MAX_CYCLES,
            batch_size=BATCH_SIZE,
            device=DEVICE
        )

        # 可选：保存位置统计结果到JSON
        save_path = "./position_metrics.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(infer_result["position_metrics"], f, ensure_ascii=False, indent=2)
        print(f"\n✅ 位置统计结果已保存至：{save_path}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()