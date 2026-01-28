# 导入PyTorch核心库：构建神经网络、张量计算的基础
import torch
# 导入PyTorch神经网络模块：包含所有层（Conv/Linear/C2f等）的基类
import torch.nn as nn
# 从ultralytics（YOLO官方库）导入YOLO11核心模块：Conv（卷积层）、C2f（特征融合层）、SPPF（空间金字塔池化）
from ultralytics.nn.modules import Conv, C2f, SPPF
# 导入numpy：用于数值计算（如ROI损失的数组统计）
import numpy as np

# ===================== YOLO11 n/s/l 核心配置字典 =====================
# 定义不同尺寸模型的核心参数，通过model_size动态选择，平衡速度与精度
# 键：模型尺寸（n=nano/s=small/l=large）；值：各模块的通道数、层数、dropout等配置
YOLO11_CONFIGS = {
    # nano：最小模型，通道缩放0.25，速度最快（适配低算力设备）
    "n": {
        # backbone配置：channels=各层输出通道数；c2f_layers=C2f模块的堆叠层数
        "backbone": {"channels": [16, 32, 32, 64, 64, 128, 128, 128], "c2f_layers": [1, 2, 2]},
        # neck配置：channels=特征降维后的通道数
        "neck": {"channels": [64, 32]},
        # head配置：hidden_dim=分类头隐藏层维度
        "head": {"hidden_dim": 16},
        # dropout率：防止过拟合（n/s模型用0.1，l模型用0.2）
        "dropout": 0.1
    },
    # small：中等模型，通道缩放0.5，平衡速度/精度（默认选择）
    "s": {
        "backbone": {"channels": [32, 64, 64, 128, 128, 256, 256, 256], "c2f_layers": [1, 2, 2]},
        "neck": {"channels": [128, 64]},
        "head": {"hidden_dim": 32},
        "dropout": 0.1
    },
    # large：大模型，通道缩放1.0，精度最高（适配高算力设备）
    "l": {
        "backbone": {"channels": [64, 128, 128, 256, 256, 512, 512, 512], "c2f_layers": [2, 3, 3]},
        "neck": {"channels": [256, 128]},
        "head": {"hidden_dim": 64},
        "dropout": 0.2
    }
}


class YOLO11ROIBackbone(nn.Module):
    """动态配置的YOLO11 Backbone（支持n/s/l）：负责提取单个ROI的多尺度语义特征"""

    def __init__(self, model_size="n", ch=3):
        # 继承nn.Module的初始化方法（必须调用）
        super().__init__()
        # 根据模型尺寸获取backbone配置（核心：动态适配不同尺寸）
        cfg = YOLO11_CONFIGS[model_size]["backbone"]
        self.channels = cfg["channels"]  # 各层输出通道数列表（共8层）
        self.c2f_layers = cfg["c2f_layers"]  # C2f模块的堆叠层数（共3个C2f）

        # ===================== 动态构建Backbone网络层 =====================
        # layer0：Conv层（输入通道ch=3，输出通道16/32/64，核3x3，步长2）→ 下采样，通道翻倍
        # 作用：将输入RGB图像（3通道）转为特征图，尺寸从64x64→32x32
        self.layer0 = Conv(ch, self.channels[0], 3, 2)
        # layer1：Conv层（步长2）→ 尺寸32x32→16x16，通道翻倍
        self.layer1 = Conv(self.channels[0], self.channels[1], 3, 2)
        # layer2：C2f层（特征融合，True=使用shortcut）→ 通道/尺寸不变，增强特征表达
        self.layer2 = C2f(self.channels[1], self.channels[2], self.c2f_layers[0], True)
        # layer3：Conv层（步长2）→ 尺寸16x16→8x8，通道翻倍
        self.layer3 = Conv(self.channels[2], self.channels[3], 3, 2)
        # layer4：C2f层→ 通道/尺寸不变
        self.layer4 = C2f(self.channels[3], self.channels[4], self.c2f_layers[1], True)
        # layer5：Conv层（步长2）→ 尺寸8x8→4x4，通道翻倍
        self.layer5 = Conv(self.channels[4], self.channels[5], 3, 2)
        # layer6：C2f层→ 通道/尺寸不变
        self.layer6 = C2f(self.channels[5], self.channels[6], self.c2f_layers[2], True)
        # layer7：SPPF层（空间金字塔池化，核5x5）→ 扩大感受野，适配不同大小目标
        self.layer7 = SPPF(self.channels[6], self.channels[7], 5)

    def forward(self, x):
        """
        Backbone前向传播：输入单个ROI图像，输出高维特征
        :param x: 输入张量 → [B×12, 3, 64, 64]（B=批次，12=ROI数，3=通道，64=尺寸）
        :return: 输出特征 → [B×12, 128/256/512, 4, 4]（根据模型尺寸）
        """
        # 逐层前向计算，数据流向：layer0→layer1→layer2→layer3→layer4→layer5→layer6→layer7
        x = self.layer0(x)  # [B×12,3,64,64] → [B×12,16,32,32]
        x = self.layer1(x)  # → [B×12,32,16,16]
        x = self.layer2(x)  # → [B×12,32,16,16]
        x = self.layer3(x)  # → [B×12,64,8,8]
        x = self.layer4(x)  # → [B×12,64,8,8]
        x = self.layer5(x)  # → [B×12,128,4,4]
        x = self.layer6(x)  # → [B×12,128,4,4]
        x = self.layer7(x)  # → [B×12,128,4,4]（n模型）/ [B×12,256,4,4]（s模型）
        return x


class YOLO11ROINeck(nn.Module):
    """动态配置的YOLO11 Neck（支持n/s/l）：承接Backbone特征，融合+降维为1D向量"""

    def __init__(self, model_size="n"):
        super().__init__()
        # 获取backbone和neck的配置
        bb_cfg = YOLO11_CONFIGS[model_size]["backbone"]
        neck_cfg = YOLO11_CONFIGS[model_size]["neck"]

        # ===================== 动态构建Neck网络层 =====================
        # layer8：C2f层→ 特征融合，通道从128→64（n模型），尺寸4x4不变
        self.layer8 = C2f(bb_cfg["channels"][7], neck_cfg["channels"][0], 1, True)
        # layer9：Conv层（核1x1，步长1）→ 通道降维（64→32），尺寸不变，减少计算量
        self.layer9 = Conv(neck_cfg["channels"][0], neck_cfg["channels"][1], 1, 1)
        # avgpool：自适应平均池化（输出1x1）→ 将4x4特征图转为1x1，保留全局信息
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # flatten：展平→ 将[C,1,1]转为[C]，得到1D特征向量
        self.flatten = nn.Flatten()

    def forward(self, x):
        """
        Neck前向传播：将Backbone的2D特征转为1D向量
        :param x: 输入特征 → [B×12, 128, 4, 4]（n模型）
        :return: 输出向量 → [B×12, 32]（n模型）/ [B×12, 64]（s模型）
        """
        x = self.layer8(x)  # → [B×12,64,4,4]
        x = self.layer9(x)  # → [B×12,32,4,4]
        x = self.avgpool(x)  # → [B×12,32,1,1]
        x = self.flatten(x)  # → [B×12,32]
        return x


class YOLO11ROIHead(nn.Module):
    """动态配置的YOLO11 Head（支持n/s/l）：将1D特征转为12个ROI的三分类结果"""

    def __init__(self, model_size="n", num_roi=12, num_classes=3):
        super().__init__()
        self.num_roi = num_roi  # ROI数量（固定12）
        self.num_classes = num_classes  # 分类数（固定3：0=无效ROI，1=有效无方块，2=有效有方块）
        head_cfg = YOLO11_CONFIGS[model_size]["head"]  # 获取head配置
        dropout = YOLO11_CONFIGS[model_size]["dropout"]  # dropout率
        neck_cfg = YOLO11_CONFIGS[model_size]["neck"]  # 获取neck配置

        # ===================== 动态构建Head网络层 =====================
        self.head = nn.Sequential(
            # Conv层（1x1）：适配Conv的4D输入要求，通道32→16（n模型），尺寸1x1不变
            Conv(neck_cfg["channels"][1], head_cfg["hidden_dim"], 1, 1),
            # Dropout层：随机失活部分神经元，防止过拟合
            nn.Dropout(dropout),
            # Linear层：全连接分类，将16维特征转为3类logits
            nn.Linear(head_cfg["hidden_dim"], num_classes)
        )

    def forward(self, x):
        """
        Head前向传播：将1D特征转为12个ROI的分类结果
        :param x: 输入向量 → [B×12, 32]（n模型）
        :return: 输出logits → [B, 12, 3]（B=批次，12=ROI数，3=分类数）
        """
        # 关键：Conv层要求输入为4D张量（N,C,H,W），因此需要扩展维度
        # unsqueeze(-1).unsqueeze(-1) → [B×12,32] → [B×12,32,1,1]
        x = self.head[0](x.unsqueeze(-1).unsqueeze(-1))
        # squeeze(-1).squeeze(-1) → [B×12,16,1,1] → [B×12,16]（还原为2D张量）
        x = x.squeeze(-1).squeeze(-1)
        x = self.head[1](x)  # Dropout → 维度不变
        x = self.head[2](x)  # Linear → [B×12,3]（每个ROI的3类logits）
        # reshape → [B,12,3]（将B×12个ROI拆分为B批次，每批次12个ROI）
        x = x.reshape(-1, self.num_roi, self.num_classes)
        return x


class YOLO11ROIClassifier(nn.Module):
    """最终模型：整合Backbone+Neck+Head，支持n/s/l三种尺寸，无预训练权重依赖"""

    def __init__(self, model_size="n", num_roi=12, num_classes=3, roi_size=64):
        super().__init__()
        self.model_size = model_size  # 模型尺寸（n/s/l）
        self.num_roi = num_roi  # ROI数量（固定12）
        self.num_classes = num_classes  # 分类数（固定3）
        self.roi_size = roi_size  # ROI图像尺寸（固定64x64）

        # ===================== 组装完整模型 =====================
        self.backbone = YOLO11ROIBackbone(model_size=model_size)  # 特征提取
        self.neck = YOLO11ROINeck(model_size=model_size)  # 特征融合+降维
        self.head = YOLO11ROIHead(model_size=model_size, num_roi=num_roi, num_classes=num_classes)  # 分类头

    def forward(self, roi_imgs):
        """
        模型整体前向传播：输入12个ROI图像，输出分类logits
        :param roi_imgs: 输入张量 → [B, 12, 3, 64, 64]（B=批次，12=ROI数，3=通道，64=尺寸）
        :return: pred_logits → [B, 12, 3]（每个ROI的3类预测logits）
        """
        B = roi_imgs.shape[0]  # 获取批次大小B（如B=8）
        # 关键：将12个ROI展平为批次维度 → [B,12,3,64,64] → [B×12,3,64,64]
        # 作用：让12个ROI共享Backbone，批量提取特征，提升计算效率
        roi_flat = roi_imgs.reshape(-1, 3, self.roi_size, self.roi_size)

        # ===================== 特征提取→融合→分类 =====================
        feat_backbone = self.backbone(roi_flat)  # [B×12,3,64,64] → [B×12,128,4,4]（n模型）
        feat_neck = self.neck(feat_backbone)     # → [B×12,32]（n模型）
        pred_logits = self.head(feat_neck)       # → [B,12,3]

        return pred_logits


# ===================== 指标计算/验证函数（无修改） =====================
def calculate_3c_metrics(pred_logits, cls_target):
    """
    计算三分类任务的核心指标（总准确率/有效准确率/正/负样本精准率/召回率/F1）
    :param pred_logits: 模型输出 → [B,12,3]（logits值）
    :param cls_target: 真实标签 → [B,12]（0=无效ROI，1=有效无方块，2=有效有方块）
    :return: 指标字典 → 包含各类准确率、精准率、召回率、F1
    """
    # 1. 从logits中获取预测类别（argmax取维度-1的最大值索引）→ [B,12]
    pred_cls = torch.argmax(pred_logits, dim=-1)
    B, num_roi = pred_cls.shape  # 获取批次B和ROI数（固定12）

    # 2. 计算总准确率（所有ROI，含无效ROI）
    total_correct = (pred_cls == cls_target).sum().item()  # 总正确数（标量）
    total_acc = total_correct / (cls_target.numel() + 1e-6)  # 总准确率（+1e-6避免除0）

    # 3. 计算有效ROI掩码（仅1/2类为有效ROI，0类为无效）→ [B,12]（bool张量）
    valid_mask = (cls_target != 0)
    valid_total = valid_mask.sum().item()  # 有效ROI总数

    # 4. 处理无有效ROI的边界情况
    if valid_total == 0:
        valid_acc = 0.0  # 有效准确率
        # 正样本（2类）指标初始化
        pos_metrics = {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        # 负样本（1类）指标初始化
        neg_metrics = {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    else:
        # 5. 计算有效ROI准确率
        valid_correct = (pred_cls[valid_mask] == cls_target[valid_mask]).sum().item()
        valid_acc = valid_correct / valid_total

        # 6. 计算正样本（2类：有效有方块）指标
        # 正样本真实掩码 → [B,12]（bool）
        pos_target_mask = (cls_target == 2) & valid_mask
        # 正样本预测掩码 → [B,12]（bool）
        pos_pred_mask = (pred_cls == 2) & valid_mask
        pos_total = pos_target_mask.sum().item()  # 正样本总数

        # 正样本准确率（仅正样本的分类正确率）
        pos_correct = (pred_cls[pos_target_mask] == cls_target[pos_target_mask]).sum().item() if pos_total > 0 else 0.0
        pos_acc = pos_correct / (pos_total + 1e-6)

        # 正样本TP/TN/FP/FN（混淆矩阵）
        tp = (pos_pred_mask & pos_target_mask).sum().item()  # 真阳性（预测2，真实2）
        fn = ((~pos_pred_mask) & pos_target_mask).sum().item()  # 假阴性（预测非2，真实2）
        fp = (pos_pred_mask & (~pos_target_mask)).sum().item()  # 假阳性（预测2，真实非2）
        # 精准率（Precision）：TP/(TP+FP) → 预测为正的样本中，真实为正的比例
        pos_precision = tp / (tp + fp + 1e-6)
        # 召回率（Recall）：TP/(TP+FN) → 真实为正的样本中，预测为正的比例
        pos_recall = tp / (tp + fn + 1e-6)
        # F1分数：2*P*R/(P+R) → 平衡精准率和召回率
        pos_f1 = 2 * pos_precision * pos_recall / (pos_precision + pos_recall + 1e-6)
        pos_metrics = {"acc": pos_acc, "precision": pos_precision, "recall": pos_recall, "f1": pos_f1}

        # 7. 计算负样本（1类：有效无方块）指标
        # 负样本真实掩码 → [B,12]（bool）
        neg_target_mask = (cls_target == 1) & valid_mask
        # 负样本预测掩码 → [B,12]（bool）
        neg_pred_mask = (pred_cls == 1) & valid_mask
        neg_total = neg_target_mask.sum().item()  # 负样本总数

        # 负样本准确率（仅负样本的分类正确率）
        neg_correct = (pred_cls[neg_target_mask] == cls_target[neg_target_mask]).sum().item() if neg_total > 0 else 0.0
        neg_acc = neg_correct / (neg_total + 1e-6)

        # 负样本TN/FN/FP（混淆矩阵）
        tn = (neg_pred_mask & neg_target_mask).sum().item()  # 真阴性（预测1，真实1）
        fn_neg = ((~neg_pred_mask) & neg_target_mask).sum().item()  # 假阴性（预测非1，真实1）
        fp_neg = (neg_pred_mask & (~neg_target_mask)).sum().item()  # 假阳性（预测1，真实非1）
        # 精准率（Precision）：TN/(TN+FP)
        neg_precision = tn / (tn + fp_neg + 1e-6)
        # 召回率（Recall）：TN/(TN+FN)
        neg_recall = tn / (tn + fn_neg + 1e-6)
        # F1分数
        neg_f1 = 2 * neg_precision * neg_recall / (neg_precision + neg_recall + 1e-6)
        neg_metrics = {"acc": neg_acc, "precision": neg_precision, "recall": neg_recall, "f1": neg_f1}

    # 8. 返回所有指标
    return {
        "total_acc": total_acc,          # 总准确率（含无效ROI）
        "valid_acc": valid_acc,          # 有效ROI准确率
        "pos_metrics": pos_metrics,      # 正样本（2类）指标
        "neg_metrics": neg_metrics       # 负样本（1类）指标
    }


def evaluate(model, val_loader, loss_fn, device):
    """
    模型验证函数：计算验证集的平均损失、各ROI平均损失、各类指标均值
    :param model: 训练好的YOLO11ROIClassifier模型
    :param val_loader: 验证集数据加载器 → 输出(roi_imgs, cls_target, roi_valid_mask)
    :param loss_fn: 损失函数（YOLO11ROIFocalLoss3C）
    :param device: 计算设备（cpu/cuda）
    :return: 各类平均指标（损失、准确率、精准率、召回率、F1）
    """
    # 1. 模型切换为验证模式（禁用Dropout/BatchNorm的训练行为）
    model.eval()
    # 2. 初始化统计变量
    val_epoch_loss = 0.0  # 验证集总损失
    batch_count = 0       # 验证集批次数量
    val_roi_loss = np.zeros(12)  # 12个ROI的总损失（初始化为0）

    # 3. 初始化指标累加变量（用于计算均值）
    total_acc_sum = 0.0          # 总准确率累加
    valid_acc_sum = 0.0          # 有效准确率累加
    pos_acc_sum, pos_precision_sum, pos_recall_sum, pos_f1_sum = 0.0, 0.0, 0.0, 0.0  # 正样本指标累加
    neg_acc_sum, neg_precision_sum, neg_recall_sum, neg_f1_sum = 0.0, 0.0, 0.0, 0.0  # 负样本指标累加

    # 4. 禁用梯度计算（验证阶段无需反向传播，节省内存）
    with torch.no_grad():
        # 5. 遍历验证集每个批次
        for batch_idx, (roi_imgs, cls_target, roi_valid_mask) in enumerate(val_loader):
            # 6. 将数据移至指定设备（cpu/cuda）
            roi_imgs = roi_imgs.to(device)
            cls_target = cls_target.to(device)

            # 7. 模型推理 → 输出pred_logits [B,12,3]
            pred_logits = model(roi_imgs)
            # 8. 计算批次损失
            loss = loss_fn(pred_logits, cls_target)

            # 9. 累加损失和批次计数
            val_epoch_loss += loss.item()  # loss.item()转为标量，避免张量累加
            batch_count += 1
            val_roi_loss += loss_fn.per_roi_loss  # 累加12个ROI的损失

            # 10. 计算批次指标并累加
            metrics = calculate_3c_metrics(pred_logits, cls_target)
            total_acc_sum += metrics["total_acc"]
            valid_acc_sum += metrics["valid_acc"]
            pos_acc_sum += metrics["pos_metrics"]["acc"]
            pos_precision_sum += metrics["pos_metrics"]["precision"]
            pos_recall_sum += metrics["pos_metrics"]["recall"]
            pos_f1_sum += metrics["pos_metrics"]["f1"]
            neg_acc_sum += metrics["neg_metrics"]["acc"]
            neg_precision_sum += metrics["neg_metrics"]["precision"]
            neg_recall_sum += metrics["neg_metrics"]["recall"]
            neg_f1_sum += metrics["neg_metrics"]["f1"]

    # 11. 计算所有指标的均值（避免除0）
    avg_val_loss = val_epoch_loss / batch_count if batch_count > 0 else 0.0  # 平均验证损失
    val_roi_avg_loss = val_roi_loss / batch_count if batch_count > 0 else np.zeros(12)  # 12个ROI的平均损失
    avg_total_acc = total_acc_sum / batch_count if batch_count > 0 else 0.0  # 平均总准确率
    avg_valid_acc = valid_acc_sum / batch_count if batch_count > 0 else 0.0  # 平均有效准确率
    # 正样本平均指标
    avg_pos_acc = pos_acc_sum / batch_count if batch_count > 0 else 0.0
    avg_pos_precision = pos_precision_sum / batch_count if batch_count > 0 else 0.0
    avg_pos_recall = pos_recall_sum / batch_count if batch_count > 0 else 0.0
    avg_pos_f1 = pos_f1_sum / batch_count if batch_count > 0 else 0.0
    # 负样本平均指标
    avg_neg_acc = neg_acc_sum / batch_count if batch_count > 0 else 0.0
    avg_neg_precision = neg_precision_sum / batch_count if batch_count > 0 else 0.0
    avg_neg_recall = neg_recall_sum / batch_count if batch_count > 0 else 0.0
    avg_neg_f1 = neg_f1_sum / batch_count if batch_count > 0 else 0.0

    # 12. 返回所有平均指标
    return (avg_val_loss, val_roi_avg_loss, avg_total_acc, avg_valid_acc,
            avg_pos_acc, avg_pos_precision, avg_pos_recall, avg_pos_f1,
            avg_neg_acc, avg_neg_precision, avg_neg_recall, avg_neg_f1)
