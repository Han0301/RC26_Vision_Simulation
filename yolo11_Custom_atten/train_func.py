"""
evaluate_func.py
    模型评估的功能包, 包含二分类的指标计算函数, 训练时的验证集指标计算函数
"""
import numpy as np
import torch
from tqdm import tqdm

# ===================== 二分类指标计算 =====================
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
    # 正样本真实掩码
    pos_target_mask = (cls_target == 1)
    # 正样本预测掩码
    pos_pred_mask = (pred_cls == 1)
    pos_total = pos_target_mask.sum().item()

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
        "pos_metrics": {"precision": pos_precision, "recall": pos_recall, "f1": pos_f1}
    }

# ===================== 二分类验证函数 =====================
def evaluate(model, val_loader, loss_fn, device):
    """
    模型验证函数：计算验证集的平均损失、二分类指标均值
    :param model: 训练好的YOLO11ROIClassifier模型
    :param val_loader: 验证集数据加载器
    :param loss_fn: 损失函数（YOLO11ROIFocalLoss2C）
    :param device: 计算设备（cpu/cuda）
    :return: 二分类相关平均指标
    """
    model.eval()
    val_epoch_loss = 0.0
    batch_count = 0
    val_roi_loss = np.zeros(12)

    # 初始化二分类指标累加
    total_acc_sum = 0.0
    pos_precision_sum, pos_recall_sum, pos_f1_sum = 0.0, 0.0, 0.0
    pred_cls_all = []
    pred_pos_mean_li = []
    pred_pos_std_li = []
    with torch.no_grad():
        for batch_idx, (roi_imgs, cls_target, conf_weight) in enumerate(tqdm(val_loader,desc="验证中",colour="red")):
            roi_imgs = roi_imgs.to(device)
            cls_target = cls_target.to(device)

            pred_logits = model(roi_imgs)
            loss = loss_fn(pred_logits, cls_target)

            val_epoch_loss += loss.item()
            batch_count += 1
            val_roi_loss += loss_fn.per_roi_loss
            pred_cls = torch.argmax(pred_logits, dim=-1)  # [B,12]
            pred_pos_count = (pred_cls == 1).sum(dim=1).cpu().numpy()  # 每张图正样本数
            pred_cls_all.extend(pred_pos_count)

            pred_pos_mean = np.mean(pred_cls_all)
            pred_pos_std = np.std(pred_cls_all)
            pred_pos_mean_li.append(pred_pos_mean)
            pred_pos_std_li.append(pred_pos_std)

            # 计算二分类指标
            metrics = calculate_2c_metrics(pred_logits, cls_target)
            total_acc_sum += metrics["total_acc"]
            pos_precision_sum += metrics["pos_metrics"]["precision"]
            pos_recall_sum += metrics["pos_metrics"]["recall"]
            pos_f1_sum += metrics["pos_metrics"]["f1"]

    print(f"📊 验证集预测有方块数量：{(sum(pred_pos_mean_li) / len(pred_pos_mean_li)):.2f} ± {(sum(pred_pos_std_li) / len(pred_pos_std_li)):.2f}（目标：8.00）")
    # 计算均值
    avg_val_loss = val_epoch_loss / batch_count if batch_count > 0 else 0.0
    val_roi_avg_loss = val_roi_loss / batch_count if batch_count > 0 else np.zeros(12)
    avg_total_acc = total_acc_sum / batch_count if batch_count > 0 else 0.0
    avg_pos_precision = pos_precision_sum / batch_count if batch_count > 0 else 0.0
    avg_pos_recall = pos_recall_sum / batch_count if batch_count > 0 else 0.0
    avg_pos_f1 = pos_f1_sum / batch_count if batch_count > 0 else 0.0

    # 返回二分类指标
    return (avg_val_loss, val_roi_avg_loss, avg_total_acc,
            avg_pos_precision, avg_pos_recall, avg_pos_f1)