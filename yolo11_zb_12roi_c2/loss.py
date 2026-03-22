import torch
import torch.nn as nn

# 对全局数量的约束正则损失(二分类, 移除无效类别)
class YOLO11ROICOUNTLOSS(nn.Module):
    def __init__(self, exist_count = 8, weight = 0.1):
        super().__init__()
        self.exist_count = exist_count
        self.weight = weight

    def forward(self, pred_logits):
        pred = torch.softmax(pred_logits, dim=-1)       # [B, 12, 2]
        pred_exist = pred[...,1]        # [B, 12 ] 每个roi落在1类的概率
        pred_exist_count = pred_exist.sum(dim=-1)       # 直接使用概率作为期望数量
        count_loss = torch.mean((pred_exist_count - self.exist_count) ** 2)     # mse 损失
        return self.weight * count_loss

# 移除采样的 Focal Loss 版本（修复 clamp 错误）
class YOLO11ROIFocalLoss2C(nn.Module):
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
        B = pred_logits.shape[0]
        device = pred_logits.device
        self.alpha = self.alpha.to(device)

        # 1. Softmax 计算概率
        pred_probs = torch.softmax(pred_logits, dim=-1)  # [B,12,2]
        # 2. 获取真实类别对应的概率
        cls_target_expand = cls_target.unsqueeze(-1)
        p_t = torch.gather(pred_probs, dim=-1, index=cls_target_expand).squeeze(-1)  # [B,12]
        # 3. Focal 权重
        focal_weight = (1 - p_t) ** self.gamma  # [B,12]
        # 4. 类别加权
        alpha_weight = self.alpha[cls_target]  # [B,12]
        # 5. 基础交叉熵损失
        ce_loss = nn.CrossEntropyLoss(reduction='none')(
            pred_logits.reshape(-1, self.num_classes),
            cls_target.reshape(-1)
        ).reshape(B, self.num_roi)  # [B,12]
        # 6. 加权Focal Loss
        focal_loss = ce_loss * alpha_weight * focal_weight  # [B,12]

        # ========== 修复：将 int 转为 Tensor 后 clamp，或改用 max ==========
        # 方案1（推荐）：用 Python max 避免类型问题，更简洁
        total_elem = torch.numel(focal_loss)
        total_loss = focal_loss.sum() / max(total_elem, 1)

        # 方案2（等价）：转 Tensor 后 clamp（和原逻辑对齐）
        # total_elem = torch.tensor(torch.numel(focal_loss), dtype=torch.float32, device=device)
        # total_loss = focal_loss.sum() / torch.clamp(total_elem, min=1.0)

        # 7. 记录每个ROI的平均损失
        roi_loss_sum = focal_loss.sum(dim=0)
        roi_sample_count = B
        self.per_roi_loss = (roi_loss_sum / roi_sample_count).detach().cpu().numpy()

        return total_loss

# 移除采样的 BCEWithLogitsLoss 版本（修复 clamp 错误）
class YOLO11ROIBCEWithLogitsLoss2C(nn.Module):
    def __init__(self, num_roi=12, num_classes=2,
                 alpha=None):
        super().__init__()
        if alpha is None:
            alpha = [2.0, 1.0]
        self.num_roi = num_roi
        self.num_classes = num_classes
        self.alpha = torch.tensor(alpha, dtype=torch.float32)
        self.per_roi_loss = None

    def forward(self, pred_logits, cls_target):
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
        weighted_bce_loss = bce_loss * alpha_weight  # [B,12]

        # ========== 核心修复：解决 clamp 参数类型错误 ==========
        # 方案1（推荐）：Python max 函数（避免 Tensor/int 类型冲突）
        total_elem = torch.numel(weighted_bce_loss)  # 返回int（比如32*12=384）
        total_loss = weighted_bce_loss.sum() / max(total_elem, 1)

        # 方案2（等价）：将int转为Tensor后调用clamp（和原代码逻辑一致）
        # total_elem = torch.tensor(torch.numel(weighted_bce_loss), dtype=torch.float32, device=device)
        # total_loss = weighted_bce_loss.sum() / torch.clamp(total_elem, min=1.0)

        # 4. 记录每个ROI的平均损失
        roi_loss_sum = weighted_bce_loss.sum(dim=0)
        roi_sample_count = B
        self.per_roi_loss = (roi_loss_sum / roi_sample_count).detach().cpu().numpy()

        return total_loss