"""
load_model.py
    加载模型的功能包, 包含加载YOLO的预训练权重到本模型, 加载训练过的模型再次训练
"""
import numpy as np
import torch
from ultralytics import YOLO

def load_YOLO_weights(model, model_size, load_path):
    """
    加载YOLO11预训练权重并精准对齐到你的YOLO11ROIClassifier模型
    :param model: 你的YOLO11ROIClassifier实例
    :param model_size: 模型尺寸 "n"/"s"/"l"
    :return: 加载权重后的模型
    """
    # 1. 加载Ultralytics官方YOLO11预训练模型
    print(f"=== 加载YOLO11-{model_size.upper()}预训练权重 ===")
    yolo11_official = YOLO(load_path)
    official_state_dict = yolo11_official.model.state_dict()

    # 2. 构建精准的键名映射表（核心：官方层编号 → 你的模型层名）
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

    # 3. 加载对齐后的权重（strict=False跳过head等不匹配层）
    missing_keys, unexpected_keys = model.load_state_dict(mapped_state_dict, strict=False)

    # 4. 打印加载结果统计
    print(f"成功加载权重数: {len(mapped_state_dict)} | 未加载的键: {len(missing_keys)} | 意外的键: {len(unexpected_keys)}")

    # 5. 验证权重是否加载成功（检查backbone第一层卷积）
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

def reload_model(model, optimizer, checkpoint_path, device):
    """
    断点续训：加载之前训练的checkpoint，恢复模型、优化器、最优F1、训练轮数
    :param model: 初始化好的YOLO11ROIClassifier模型
    :param optimizer: 初始化好的优化器
    :param checkpoint_path: 之前保存的模型路径（.pt文件）
    :param device: 训练设备
    :return: best_pos_f1(最优F1), start_epoch(起始训练轮数)
    """
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    # 加载优化器状态
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # 加载最优F1
    best_pos_f1 = checkpoint.get('best_pos_f1', 0.0)
    # 加载起始轮数
    start_epoch = checkpoint.get('epoch', 0)

    print(f"=== 断点续训加载成功！===")
    print(f"├─ 恢复模型权重 | 起始轮数：{start_epoch}")
    print(f"└─ 恢复最优正样本F1：{best_pos_f1:.4f}")
    return best_pos_f1, start_epoch