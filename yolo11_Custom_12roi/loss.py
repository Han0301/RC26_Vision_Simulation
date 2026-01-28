import torch
import torch.nn as nn
import numpy as np


class YOLO11ROIFocalLoss3C(nn.Module):
    """YOLO11原生Focal Loss（适配三分类+12个ROI）"""

    def __init__(self, num_roi=12, num_classes=3,
                 alpha=[1.0, 2.0, 5.0], gamma=1.5,  # YOLO11默认gamma=1.5
                 max_positive=8, max_negative=4):
        super().__init__()
        self.num_roi = num_roi
        self.num_classes = num_classes
        self.alpha = torch.tensor(alpha, dtype=torch.float32)  # 0/1/2类权重（YOLO11正样本高权重）
        self.gamma = gamma  # Focal Loss聚焦系数（YOLO11原生值）
        self.max_positive = max_positive  # 每批次最多选8个有方块样本
        self.max_negative = max_negative  # 每批次最多选4个无方块样本
        self.per_roi_loss = None  # 记录每个ROI的损失

    def forward(self, pred_logits, cls_target):
        """
        输入：
        - pred_logits: 模型输出 [B,12,3]
        - cls_target: 标签 [B,12]
        输出：YOLO11标准加权Focal Loss（标量）
        """
        B = pred_logits.shape[0]
        device = pred_logits.device
        self.alpha = self.alpha.to(device)

        # 1. YOLO11原生Softmax（分类任务）
        pred_probs = torch.softmax(pred_logits, dim=-1)  # [B,12,3]

        # 2. 获取真实类别概率（Focal Loss核心）
        cls_target_expand = cls_target.unsqueeze(-1)  # [B,12,1]
        p_t = torch.gather(pred_probs, dim=-1, index=cls_target_expand).squeeze(-1)  # [B,12]

        # 3. YOLO11 Focal Loss核心：(1-p_t)^γ 聚焦难样本
        focal_weight = (1 - p_t) ** self.gamma  # [B,12]

        # 4. YOLO11类别加权（解决样本不均衡）
        alpha_weight = self.alpha[cls_target]  # [B,12]

        # 5. YOLO11基础交叉熵损失（reduction='none'保留每个ROI损失）
        ce_loss = nn.CrossEntropyLoss(reduction='none')(
            pred_logits.reshape(-1, self.num_classes),
            cls_target.reshape(-1)
        ).reshape(B, self.num_roi)  # [B,12]

        # 6. 加权Focal Loss（YOLO11最终损失计算）
        focal_loss = ce_loss * alpha_weight * focal_weight  # [B,12]

        # 7. YOLO11难样本采样（避免易分类样本主导损失）
        selected_mask = torch.zeros_like(focal_loss, device=device)
        for b_idx in range(B):
            # 区分各类ROI（YOLO11重点采样正样本）
            valid_neg_idx = torch.where(cls_target[b_idx] == 1)[0]  # 1类（有效无方块）
            valid_pos_idx = torch.where(cls_target[b_idx] == 2)[0]  # 2类（有效有方块）

            # ========== 修复点1：空样本时返回空张量（而非空列表） ==========
            # 正样本采样：保证始终返回tensor（同设备、同类型）
            if len(valid_pos_idx) > 0:
                select_pos = valid_pos_idx[torch.randperm(len(valid_pos_idx))[:self.max_positive]]
            else:
                select_pos = torch.tensor([], dtype=torch.long, device=device)  # 空tensor

            # 负样本采样：保证始终返回tensor（同设备、同类型）
            if len(valid_neg_idx) > 0:
                select_neg = valid_neg_idx[torch.randperm(len(valid_neg_idx))[:self.max_negative]]
            else:
                select_neg = torch.tensor([], dtype=torch.long, device=device)  # 空tensor

            # ========== 修复点2：torch.cat仅拼接tensor，else返回空tensor ==========
            if len(select_pos) + len(select_neg) > 0:
                selected_roi = torch.cat([select_pos, select_neg])
            else:
                selected_roi = torch.tensor([], dtype=torch.long, device=device)  # 空tensor

            # 仅当有选中的ROI时赋值，避免空tensor索引报错
            if len(selected_roi) > 0:
                selected_mask[b_idx, selected_roi] = 1.0

        # 8. 计算总损失（YOLO11数值稳定化）
        filtered_loss = focal_loss * selected_mask
        total_loss = filtered_loss.sum() / torch.clamp(selected_mask.sum(), min=1)

        # 9. 记录每个ROI的平均损失（YOLO11监控用）
        roi_loss_sum = filtered_loss.sum(dim=0)
        roi_sample_count = torch.clamp(selected_mask.sum(dim=0), min=1)
        self.per_roi_loss = (roi_loss_sum / roi_sample_count).detach().cpu().numpy()

        return total_loss
