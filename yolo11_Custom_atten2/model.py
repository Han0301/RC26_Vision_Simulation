"""
model.py
    定义模型的backbooe,neck,attention,head模块, 并组装模型
"""
import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv, C2f, SPPF

# 定义不同尺寸模型的核心参数，通过model_size动态选择，平衡速度与精度
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
        "dropout": 0.15
    },
    # large：大模型，通道缩放1.0，精度最高（适配高算力设备）
    "l": {
        "backbone": {"channels": [64, 128, 128, 256, 256, 512, 512, 512], "c2f_layers": [2, 3, 3]},
        "neck": {"channels": [256, 128]},
        "head": {"hidden_dim": 64},
        "dropout": 0.2
    }
}

class LocalGrid_Attention(nn.Module):
    """局部网格注意力：建模相邻+指定ROI对的关联"""
    def __init__(self, atten_weight, dim=256, num_heads=4):
        super().__init__()
        # 多头自注意力层，batch_first=True 适配 [B, N, C] 格式
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        # 层归一化：稳定训练，适配注意力输出
        self.norm = nn.LayerNorm(dim)
        self.atten_weight = atten_weight

    def forward(self, roi_feat):
        B, N, C = roi_feat.shape
        # ------------- 动态张量判断（ONNX兼容，不固化维度） -------------
        # 条件：N == 1
        is_single_roi = (N == 1)

        if is_single_roi:
            # 🔴 单ROI：直接返回原始特征，禁用注意力（和你原有训练逻辑完全一致）
            attn_weights = torch.zeros((B, 1, 1), device=roi_feat.device)
            out = self.norm(roi_feat)
            return out, attn_weights
        else:
            # 🟢 12ROI：正常计算注意力（原有逻辑不变）
            attn_feat, attn_weights = self.attn(roi_feat, roi_feat, roi_feat)
            out = self.norm(roi_feat + self.atten_weight * attn_feat)
            return out, attn_weights

class Model_Backbone(nn.Module):
    """提取单个ROI的多尺度语义特征"""

    def __init__(self, model_size="n", ch=3):
        super().__init__()
        cfg = YOLO11_CONFIGS[model_size]["backbone"]
        self.channels = cfg["channels"]
        self.c2f_layers = cfg["c2f_layers"]

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


class Model_Neck(nn.Module):
    """承接Backbone特征，融合+降维为1D向量"""

    def __init__(self, model_size="n"):
        super().__init__()
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


class Model_Head(nn.Module):
    """将1D特征转为12个ROI的二分类结果"""

    def __init__(self, model_size="n", num_roi=12, num_classes=2):
        super().__init__()
        self.num_roi = num_roi  # ROI数量
        self.num_classes = num_classes  # 分类数
        head_cfg = YOLO11_CONFIGS[model_size]["head"]  # 获取head配置
        dropout = YOLO11_CONFIGS[model_size]["dropout"]  # dropout率
        neck_cfg = YOLO11_CONFIGS[model_size]["neck"]  # 获取neck配置

        # ===================== 动态构建Head网络层 =====================
        self.head = nn.Sequential(
            # Conv层（1x1）：适配Conv的4D输入要求，通道32→16（n模型），尺寸1x1不变
            Conv(neck_cfg["channels"][1], head_cfg["hidden_dim"], 1, 1),
            # Dropout层：随机失活部分神经元，防止过拟合
            nn.Dropout(dropout),
            # Linear层：全连接分类，将16维特征转为2类logits
            nn.Linear(head_cfg["hidden_dim"], num_classes)
        )

    def forward(self, x):
        # x: [B,12,C]  注意力输出特征
        B, N, C = x.shape
        # 展平为 [B×12, C] 适配原有流程
        x = x.reshape(B*N, C)
        x = self.head[0](x.unsqueeze(-1).unsqueeze(-1))
        x = x.squeeze(-1).squeeze(-1)
        x = self.head[1](x)
        x = self.head[2](x)
        return x.reshape(B, N, self.num_classes)  # [B,12,2]


class YOLO11ROIClassifier(nn.Module):
    """最终模型：整合Backbone+Neck+Head，支持n/s/l三种尺寸，无预训练权重依赖"""

    def __init__(self, model_size="n", num_roi=12, num_classes=2, roi_size=64, atten_weight=0.15):
        super().__init__()
        self.attn_weights = None
        self.model_size = model_size  # 模型尺寸（n/s/l）
        self.num_roi = num_roi  # ROI数量（固定12）
        self.num_classes = num_classes  # 分类数（固定3）
        self.roi_size = roi_size  # ROI图像尺寸（固定64x64）
        self.atten_weight = atten_weight        # 注意力特征所占的权重
        neck_cfg = YOLO11_CONFIGS[model_size]["neck"]
        attn_dim = neck_cfg["channels"][1]  # 动态适配n/s/l维度

        # ===================== 组装完整模型 =====================
        self.backbone = Model_Backbone(model_size=model_size)  # 特征提取
        self.neck = Model_Neck(model_size=model_size)  # 特征融合+降维
        self.spatial_attention = LocalGrid_Attention(
            atten_weight=self.atten_weight,
            dim=attn_dim,
            num_heads=4 if model_size!="n" else 2  # 参数调整：n模型heads=2，避免维度不匹配
        )
        self.head = Model_Head(model_size=model_size, num_roi=num_roi, num_classes=num_classes)  # 分类头

    def forward(self, roi_imgs):
        """
        模型整体前向传播：输入12个ROI图像，输出分类logits
        :param roi_imgs: 输入张量 → [B, 12, 3, 64, 64]（B=批次，12=ROI数，3=通道，64=尺寸）
        :return: pred_logits → [B, 12, 2]（每个ROI的2类预测logits）
        """
        B = roi_imgs.shape[0]  # 获取批次大小B（如B=8）
        N = roi_imgs.shape[1]
        # 关键：将12个ROI展平为批次维度 → [B,12,3,64,64] → [B×12,3,64,64]
        # 作用：让12个ROI共享Backbone，批量提取特征，提升计算效率
        roi_flat = roi_imgs.reshape(-1, 3, self.roi_size, self.roi_size)

        # ===================== 特征提取→融合→分类 =====================
        feat_backbone = self.backbone(roi_flat)  # [B×12,3,64,64] → [B×12,128,4,4]（n模型）
        feat_neck = self.neck(feat_backbone)     # → [B×12,32]（n模型）
        feat_attn = feat_neck.reshape(B, N, -1)
        feat_attn,self.attn_weights = self.spatial_attention(feat_attn)
        pred_logits = self.head(feat_attn)       # → [B,12,2]

        return pred_logits