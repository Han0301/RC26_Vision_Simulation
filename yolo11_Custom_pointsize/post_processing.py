import numpy as np
import torch

ROI_GROUPS = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]

def _compute_confidence(point_size,device):
    conf_weight = torch.zeros(12, dtype=torch.float32, device=device)

    for group in ROI_GROUPS:
        group_vals = point_size[group]
        max_val = group_vals.max()
        if max_val < 1e-6:
            conf_weight[group] = 1.0
        else:
            conf_weight[group] = group_vals / max_val
    return conf_weight

# ==============================================
# 方法1：纯GPU版 - 返回 pred_cls + 原生加权得分
# ==============================================
def post_process_only_pred1(pred_logits, point_size, num_1=8):
    pred_probs = torch.softmax(pred_logits, dim=-1)[0, :, :].squeeze(0)
    pred_1 = pred_probs[:, 1]
    weighted_score = pred_1 * point_size
    sorted_idx = torch.argsort(weighted_score, descending=True)
    pred_cls = torch.zeros(12, dtype=torch.int32, device=pred_logits.device)
    pred_cls[sorted_idx[:num_1]] = 1
    # 🔥 修改：返回 预测类别 + 本方法的加权得分
    return pred_cls, weighted_score

# ==============================================
# 方法2：纯GPU版 - 返回 pred_cls + 线性归一化得分
# ==============================================
def post_process_linear_norm(pred_logits, point_size, num_1=8):
    pred_probs = torch.softmax(pred_logits, dim=-1)[0, :, :].squeeze(0)
    pred_0 = pred_probs[:, 0]
    pred_1 = pred_probs[:, 1]
    raw_score = pred_1 - pred_0
    norm_score = (raw_score + 1) / 2
    weighted_score = norm_score * point_size
    sorted_idx = torch.argsort(weighted_score, descending=True)
    pred_cls = torch.zeros(12, dtype=torch.int32, device=pred_logits.device)
    pred_cls[sorted_idx[:num_1]] = 1
    # 🔥 修改：返回 预测类别 + 本方法的加权得分
    return pred_cls, weighted_score

# ==============================================
# 方法3：纯GPU版 - 返回 pred_cls + sigmoid平衡得分
# ==============================================
def post_process_sigmoid_balance(pred_logits, point_size, num_1=8):
    pred_probs = torch.softmax(pred_logits, dim=-1)[0, :, :].squeeze(0)
    pred_0 = pred_probs[:, 0]
    pred_1 = pred_probs[:, 1]
    raw_score = (pred_1 - pred_0) * point_size
    norm_score = torch.sigmoid(5 * raw_score)
    sorted_idx = torch.argsort(norm_score, descending=True)
    pred_cls = torch.zeros(12, dtype=torch.int32, device=pred_logits.device)
    pred_cls[sorted_idx[:num_1]] = 1
    # 🔥 修改：返回 预测类别 + 本方法的加权得分
    return pred_cls, norm_score