# 导入PyTorch核心库：构建神经网络、张量计算的基础
import torch
# 导入PyTorch神经网络模块：包含所有层（Conv/Linear/C2f等）的基类
import torch.nn as nn
from ultralytics import YOLO
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


# ===================== 二分类指标计算（核心修改） =====================
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


# ===================== 二分类验证函数（核心修改） =====================
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
    pos_acc_sum, pos_precision_sum, pos_recall_sum, pos_f1_sum = 0.0, 0.0, 0.0, 0.0
    pred_cls_all = []
    with torch.no_grad():
        for bitch_idx, bitch_data in enumerate(val_loader):
            print(f"len(bitch_data): f{bitch_data}")
        for batch_idx, (roi_imgs, cls_target, roi_valid_mask) in enumerate(val_loader):
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
            print(f"📊 验证集预测有方块数量：{pred_pos_mean:.2f} ± {pred_pos_std:.2f}（目标：8.00）")

            # 计算二分类指标
            metrics = calculate_2c_metrics(pred_logits, cls_target)
            total_acc_sum += metrics["total_acc"]
            pos_acc_sum += metrics["pos_metrics"]["acc"]
            pos_precision_sum += metrics["pos_metrics"]["precision"]
            pos_recall_sum += metrics["pos_metrics"]["recall"]
            pos_f1_sum += metrics["pos_metrics"]["f1"]

    # 计算均值
    avg_val_loss = val_epoch_loss / batch_count if batch_count > 0 else 0.0
    val_roi_avg_loss = val_roi_loss / batch_count if batch_count > 0 else np.zeros(12)
    avg_total_acc = total_acc_sum / batch_count if batch_count > 0 else 0.0
    avg_pos_acc = pos_acc_sum / batch_count if batch_count > 0 else 0.0
    avg_pos_precision = pos_precision_sum / batch_count if batch_count > 0 else 0.0
    avg_pos_recall = pos_recall_sum / batch_count if batch_count > 0 else 0.0
    avg_pos_f1 = pos_f1_sum / batch_count if batch_count > 0 else 0.0

    # 返回二分类指标（移除三分类相关）
    return (avg_val_loss, val_roi_avg_loss, avg_total_acc,
            avg_pos_acc, avg_pos_precision, avg_pos_recall, avg_pos_f1)
def load_yolo11_pretrained_weights(model, model_size, load_path):
    """
    加载YOLO11预训练权重并精准对齐到你的YOLO11ROIClassifier模型
    :param model: 你的YOLO11ROIClassifier实例
    :param model_size: 模型尺寸 "n"/"s"/"l"
    :return: 加载权重后的模型
    """
    # 1. 加载Ultralytics官方YOLO11预训练模型
    print(f"📥 加载YOLO11-{model_size.upper()}预训练权重...")
    yolo11_official = YOLO(load_path)
    official_state_dict = yolo11_official.model.state_dict()

    # 2. 构建精准的键名映射表（核心：官方层编号 → 你的模型层名）
    # 官方层编号: 0→layer0, 1→layer1, ..., 8→layer8, 9→layer9
    mapped_state_dict = {}
    current_model_dict = model.state_dict()

    # 遍历所有官方权重键
    for official_key, official_param in official_state_dict.items():
        # 跳过和分类头相关的权重（官方YOLO11的检测头）
        if any(key in official_key for key in ["detect", "bbox", "cls"]):
            continue

        # 拆分官方键名，例如："model.0.conv.weight" → ["model", "0", "conv", "weight"]
        parts = official_key.split(".")
        if len(parts) < 3 or parts[0] != "model":
            continue  # 跳过非模型层的权重

        # 提取官方层编号（如0,1,2...）
        try:
            official_layer_idx = int(parts[1])
        except ValueError:
            continue  # 跳过非数字编号的层

        # 确定目标模块（backbone/neck）和目标层名
        if 0 <= official_layer_idx <= 7:
            # 官方层0-7 → 你的backbone.layer0-layer7
            target_module = "backbone"
            target_layer_name = f"layer{official_layer_idx}"
        elif 8 <= official_layer_idx <= 9:
            # 官方层8-9 → 你的neck.layer8-layer9
            target_module = "neck"
            target_layer_name = f"layer{official_layer_idx}"
        else:
            continue  # 跳过10及以后的层（官方检测头）

        # 构建你的模型键名（替换前缀）
        # 示例：官方"model.0.conv.weight" → 你的"backbone.layer0.conv.weight"
        target_key_parts = [target_module, target_layer_name] + parts[2:]
        target_key = ".".join(target_key_parts)

        # 检查当前模型是否有该键，且形状匹配
        if target_key in current_model_dict:
            if current_model_dict[target_key].shape == official_param.shape:
                mapped_state_dict[target_key] = official_param
                print(f"✅ 匹配权重: {official_key:40s} → {target_key}")
            else:
                print(f"⚠️ 跳过权重（形状不匹配）: {official_key}")
                print(f"   官方形状: {official_param.shape} | 你的模型形状: {current_model_dict[target_key].shape}")
        else:
            print(f"⚠️ 跳过权重（键不存在）: {official_key} → {target_key}")

    # 3. 加载对齐后的权重（strict=False跳过head等不匹配层）
    print("\n🔧 开始加载权重到模型...")
    missing_keys, unexpected_keys = model.load_state_dict(mapped_state_dict, strict=False)

    # 4. 打印加载结果统计
    print(f"\n📊 权重加载结果:")
    print(f"   ✅ 成功加载权重数: {len(mapped_state_dict)}")
    print(f"   ⚠️  未加载的键（自定义head）: {len(missing_keys)}")
    print(f"   ❌ 意外的键: {len(unexpected_keys)}")

    # 打印关键缺失键（仅前5个，避免刷屏）
    if missing_keys:
        print(f"   主要缺失键（自定义层）: {missing_keys[:5]}")

    # 5. 验证权重是否加载成功（检查backbone第一层卷积）
    print("\n🔍 验证权重加载效果...")
    # 对比官方第一层卷积和你的模型第一层卷积
    try:
        # 修复：直接取layer0.conv.weight，去掉多余的.conv
        your_first_conv = model.backbone.layer0.conv.weight.data.cpu().numpy()
        official_first_conv = official_state_dict.get("model.0.conv.weight", None)

        if official_first_conv is not None:
            official_first_conv_np = official_first_conv.cpu().numpy()
            # 检查前10个值是否接近（允许微小浮点误差）
            is_match = np.allclose(official_first_conv_np[:10], your_first_conv[:10], atol=1e-6)
            print(f"   Backbone第一层卷积权重匹配: {'✅' if is_match else '❌'}")
        else:
            print(f"   ⚠️  无法验证：官方模型无第一层卷积权重")
    except Exception as e:
        print(f"   ⚠️  验证失败: {str(e)}")

    return model
