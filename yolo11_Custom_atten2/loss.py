"""
loss.py
    损失设计, 包含Focal_loss,Bce_loss和Count_loss
"""
import torch
import torch.nn as nn

# 对全局数量的约束正则损失
class CountLoss(nn.Module):
    def __init__(self, exist_count=8, weight=0.1):
        super().__init__()
        self.exist_count = exist_count
        self.weight = weight

    def forward(self, pred_logits):
        B, N, _ = pred_logits.shape
        # 单ROI时，不计算数量损失（关键修复）
        if N == 1:
            return torch.tensor(0.0, device=pred_logits.device)

        pred = torch.softmax(pred_logits, dim=-1)  # [B, N, 2]
        pred_exist = pred[..., 1]  # [B, N ]
        pred_exist_count = pred_exist.sum(dim=-1)
        count_loss = torch.mean((pred_exist_count - self.exist_count) ** 2)
        return self.weight * count_loss


# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, num_roi=12, num_classes=2,
                 alpha=None, gamma=1.5):
        super().__init__()
        if alpha is None:
            alpha = [2.0, 1.0]
        self.num_roi = num_roi
        self.num_classes = num_classes
        self.alpha = torch.tensor(alpha, dtype=torch.float32)
        self.gamma = gamma
        self.per_roi_loss = None

    def forward(self, pred_logits, cls_target):
        B, N, _ = pred_logits.shape
        device = pred_logits.device
        self.alpha = self.alpha.to(device)

        # 1. Softmax 计算概率
        pred_probs = torch.softmax(pred_logits, dim=-1)  # [B,N,2]
        # 2. 获取真实类别对应的概率
        cls_target_expand = cls_target.unsqueeze(-1)
        p_t = torch.gather(pred_probs, dim=-1, index=cls_target_expand).squeeze(-1)  # [B,N]
        # 3. Focal 权重
        focal_weight = (1 - p_t) ** self.gamma  # [B,N]
        # 4. 类别加权
        alpha_weight = self.alpha[cls_target]  # [B,N]
        # 5. 基础交叉熵损失
        ce_loss = nn.CrossEntropyLoss(reduction='none')(
            pred_logits.reshape(-1, self.num_classes),
            cls_target.reshape(-1)
        ).reshape(B, N)  # [B,N] 动态适配
        # 6. 加权Focal Loss
        focal_loss = ce_loss * alpha_weight * focal_weight  # [B,N]

        # 7. 记录每个ROI的平均损失
        total_elem = torch.numel(focal_loss)
        total_loss = focal_loss.sum() / max(total_elem, 1)
        roi_loss_sum = focal_loss.sum(dim=0)
        roi_sample_count = B
        self.per_roi_loss = (roi_loss_sum / roi_sample_count).detach().cpu().numpy()

        return total_loss

# BCE Loss
class BCELoss(nn.Module):
    def __init__(self, num_roi=12, num_classes=2,
                 alpha=None):
        super().__init__()
        if alpha is None:
            alpha = [2.0, 1.0]
        self.num_roi = num_roi
        self.num_classes = num_classes
        self.alpha = torch.tensor(alpha, dtype=torch.float32)
        self.per_roi_loss = None

    def forward(self, pred_logits, cls_target, conf_weight):
        B = pred_logits.shape[0]
        device = pred_logits.device
        self.alpha = self.alpha.to(device)

        # 1. 提取正类logits
        pred_logits_pos = pred_logits[..., 1]  # [B,12]
        cls_target_float = cls_target.float()
        # 2. 基础BCE损失
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(
            pred_logits_pos.reshape(-1),
            cls_target_float.reshape(-1)
        ).reshape(B, self.num_roi)  # [B,12]
        # 3. 类别加权
        alpha_weight = self.alpha[cls_target]  # [B,12]
        weighted_bce_loss = bce_loss * alpha_weight * conf_weight  # [B,12]

        # 4. 记录每个ROI的平均损失
        total_elem = torch.tensor(torch.numel(weighted_bce_loss), dtype=torch.float32, device=device)
        total_loss = weighted_bce_loss.sum() / torch.clamp(total_elem, min=1.0)
        roi_loss_sum = weighted_bce_loss.sum(dim=0)
        roi_sample_count = B
        self.per_roi_loss = (roi_loss_sum / roi_sample_count).detach().cpu().numpy()

        return total_loss