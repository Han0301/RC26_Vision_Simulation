"""
train_main.py
    主要功能函数: train, 训练roi12分类模型
"""
import multiprocessing
import os
import torch
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR      # 余弦退火学习率调度器，让学习率按余弦函数周期性衰减
from tqdm import tqdm

# 自定义模块
from model import YOLO11ROIClassifier
from load_model import load_YOLO_weights,reload_model
from train_func import evaluate,calculate_2c_metrics
from loss import CountLoss, FocalLoss, BCELoss
from dataset_func import load_dataset, load_train_val_datasets
from train_config import model_config,train_config,loss_config,dataset_config   # 相关训练参数

def train(model_config, train_config, dataset_config, loss_config):
    # ===================== 1. 数据预处理 并 加载数据集 =====================
    print("=== 正在加载数据集 ===")
    if dataset_config["load_datasets"]:  # 直接加载数据集
        train_loader, val_loader, train_size, val_size = load_dataset(dataset_config["DATASET_ROOTS"], dataset_config["VAL_RATIO"], train_config["BATCH_SIZE"],
                                                                      model_config["ROI_IMG_SIZE"], train_config["WORKERS"])
    else:  # 直接指定训练集和验证集
        train_loader, val_loader, train_size, val_size = load_train_val_datasets(dataset_config["TRAIN_DATASETS"], dataset_config["VAL_DATASETS"], train_config["BATCH_SIZE"],
                                                                                model_config["ROI_IMG_SIZE"], train_config["WORKERS"])
    print(f"=== 数据集划分完成 ===")
    print(f"训练集：{train_size}样本 | {len(train_loader)}批次 | 验证集：{val_size}样本 | {len(val_loader)}批次")
    print(f"训练设备：{model_config['DEVICE']} | 模型尺寸：{model_config['MODEL_SIZE'].upper()}")

    # ===================== 2 初始化模型/损失/优化器 =====================
    # 2.1 初始化模型
    model = YOLO11ROIClassifier(
        model_size=model_config["MODEL_SIZE"],
        num_roi=model_config["NUM_ROI"],
        num_classes=model_config["NUM_CLASSES"],
        roi_size=model_config["ROI_IMG_SIZE"]
    ).to(model_config["DEVICE"])

    # 2.2 加载损失
    cls_loss_fn = FocalLoss(
        num_roi=model_config["NUM_ROI"],
        num_classes=model_config["NUM_CLASSES"],
        alpha=loss_config["LOSS_WEIGHT"],  # 损失在两个类别上面的权重
        gamma=loss_config["FOCAL_LOSS"],  # Focal Loss 难样本挖掘系数
    ).to(model_config["DEVICE"])

    # cls_loss_fn = BCELoss(
    #     num_roi=model_config["NUM_ROI"],
    #     num_classes=model_config["NUM_CLASSES"],
    #     alpha=loss_config["LOSS_WEIGHT"],         # 损失在两个类别上面的权重
    # ).to(model_config["DEVICE"])

    count_loss_fn = CountLoss(
        exist_count=8,
        weight=loss_config["count_loss_weight"]
    )

    # 2.3 参数的冻结策略
    for name, param in model.backbone.named_parameters():
        if "layer0" in name or "layer1" in name or "layer2" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    for name, param in model.named_parameters():
        if "spatial_attention" in name:
            param.requires_grad = True

    # 2.4 分层学习率
    param_groups = [
        {"params": [p for n, p in model.backbone.named_parameters() if "layer0" in n or "layer1" in n or "layer2" in n],
         "lr": loss_config["LEARNING_RATE"] * 0.001},
        {"params": [p for n, p in model.backbone.named_parameters() if
                    "layer0" not in n and "layer1" not in n and "layer2" not in n],
         "lr": loss_config["LEARNING_RATE"] * 0.1},
        {"params": model.neck.parameters(), "lr": loss_config["LEARNING_RATE"] * 0.5},
        {"params": model.head.parameters(), "lr": loss_config["LEARNING_RATE"]},
        {"params": model.spatial_attention.parameters(), "lr": loss_config["LEARNING_RATE"] * 0.1}
    ]

    # 2.5 优化器, 学习率调度器(余弦退火)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=loss_config["WEIGHT_DECAY"])  # adamw: 带权重衰减的优化器
    scheduler = CosineAnnealingLR(optimizer, T_max=train_config["EPOCHS"], eta_min=1e-6)

    # 2.6 加载权重的方式
    if train_config["RESUME_TRAIN"]:
        best_pos_f1, start_epoch = reload_model(model, optimizer, train_config["CHECKPOINT_PATH"], model_config["DEVICE"])
    else:
        best_pos_f1 = 0.0
        start_epoch = 0
        model = load_YOLO_weights(model, model_size=model_config["MODEL_SIZE"],
                                  load_path=model_config["YOLO_weight_path"])

    # ===================== 3 训练循环 =====================
    # 3.1 创建模型输出路径
    os.makedirs(train_config["SAVE_DIR"], exist_ok=True)  # exist_ok=True  路径存在也不报错
    no_improve = 0  # 无提升早停计数器

    # 3.2 训练循环
    print(f"=== 开始训练 ===")
    for epoch in range(start_epoch, train_config["EPOCHS"]):
        # 3.2.1 将模型转为训练模式, 初始化指标
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        valid_batch_count = 0  # 用于计算指标的批次

        train_total_acc = 0.0
        train_pos_precision, train_pos_recall, train_pos_f1 = 0.0, 0.0, 0.0

        # 3.2.2 从dataloader 中取数据并进行前向传播和反向传播
        for batch_idx, (roi_imgs, cls_target, conf_weight) in enumerate(tqdm(train_loader, desc="训练中", colour="red")):
            roi_imgs = roi_imgs.to(model_config["DEVICE"])
            cls_target = cls_target.to(model_config["DEVICE"])
            conf_weight = conf_weight.to(model_config["DEVICE"])

            is_mixup = False
            loss = 0.0
            pred_logits = None

            # ---------------------- MixUp增强开始 ----------------------
            if np.random.rand() < train_config["mixup_rate"]:
                is_mixup = True
                # 当前批次内随机索引配对
                B = roi_imgs.shape[0]
                # 生成当前批次的随机打乱索引
                indices = torch.randperm(B).to(model_config["DEVICE"])
                # 随机插值系数
                lam = np.random.beta(train_config["mixup_alpha"], train_config["mixup_alpha"])

                # 图像混合 + 标签混合
                roi_imgs_mix = lam * roi_imgs + (1 - lam) * roi_imgs[indices]
                cls_target2 = cls_target[indices]

                # 前向传播
                pred_logits = model(roi_imgs_mix)
                # 混合损失
                cls_loss1 = cls_loss_fn(pred_logits, cls_target)
                cls_loss2 = cls_loss_fn(pred_logits, cls_target2)
                cls_loss = lam * cls_loss1 + (1 - lam) * cls_loss2
                count_loss = count_loss_fn(pred_logits)
            else:
                is_mixup = False
                pred_logits = model(roi_imgs)
                cls_loss = cls_loss_fn(pred_logits, cls_target)
                count_loss = count_loss_fn(pred_logits)
            # ---------------------- MixUp增强结束 ----------------------

            total_loss = cls_loss + count_loss
            optimizer.zero_grad()  # 清空梯度(防止累积)
            total_loss.backward()  # 损失的反向传播
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪(防止梯度爆炸)
            optimizer.step()  # 更新参数

            epoch_loss += total_loss.item()  # 累加当前的损失
            batch_count += 1

            # 只有在非 MixUp 时才计算指标
            with torch.no_grad():
                if not is_mixup:
                    metrics = calculate_2c_metrics(pred_logits, cls_target)  # 改为二分类指标函数
                    train_total_acc += metrics["total_acc"]
                    train_pos_precision += metrics["pos_metrics"]["precision"]
                    train_pos_recall += metrics["pos_metrics"]["recall"]
                    train_pos_f1 += metrics["pos_metrics"]["f1"]
                    valid_batch_count += 1

        # 3.2.3 更新学习率
        scheduler.step()

        # 3.2.4 计算训练集平均指标（二分类）
        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
        avg_train_total_acc = train_total_acc / valid_batch_count if valid_batch_count > 0 else 0.0
        avg_train_pos_precision = train_pos_precision / valid_batch_count if valid_batch_count > 0 else 0.0
        avg_train_pos_recall = train_pos_recall / valid_batch_count if valid_batch_count > 0 else 0.0
        avg_train_pos_f1 = train_pos_f1 / valid_batch_count if valid_batch_count > 0 else 0.0

        # 3.2.5 验证集评估（适配二分类）
        val_metrics = evaluate(model, val_loader, cls_loss_fn, model_config["DEVICE"])
        (avg_val_loss, val_roi_avg_loss, avg_val_total_acc,
         avg_val_pos_precision, avg_val_pos_recall, avg_val_pos_f1) = val_metrics

        # 3.2.6 打印日志（关键修改：移除三分类相关打印）
        print("=" * 120)
        print(f"【Epoch {epoch + 1}/{train_config['EPOCHS']} 训练集】")
        print(f"总损失：{avg_epoch_loss:.4f} | 整体准确率：{avg_train_total_acc:.4f} | 学习率: {optimizer.param_groups[-1]['lr']:.6f}")
        print(
            f"└─ 有方块（正样本）： 精确率={avg_train_pos_precision:.4f} | 召回率={avg_train_pos_recall:.4f} | F1={avg_train_pos_f1:.4f}")

        print(f"【Epoch {epoch + 1}/{train_config['EPOCHS']} 验证集】")
        print(f"总损失：{avg_val_loss:.4f} | 整体准确率：{avg_val_total_acc:.4f}")
        print(
            f"└─ 有方块（正样本）： 精确率={avg_val_pos_precision:.4f} | 召回率={avg_val_pos_recall:.4f} | F1={avg_val_pos_f1:.4f}")
        print("=" * 120)

        # 3.2.7 早停+保存模型
        if avg_val_pos_f1 > best_pos_f1:
            best_pos_f1 = avg_val_pos_f1
            no_improve = 0
            save_path = os.path.join(train_config["SAVE_DIR"], train_config["MODEL_NAME"])
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_pos_f1': best_pos_f1,
                'loss': avg_val_loss,
            }, save_path)
            print(f"✅ 保存最优模型 | 有效有方块F1：{avg_val_pos_f1:.4f} | 路径：{save_path}")
        else:
            no_improve += 1
            print(f"⚠️ 正样本F1未提升 | 当前最优：{best_pos_f1:.4f} | 无提升轮数：{no_improve}/{train_config['patience']}")
            if no_improve >= train_config["patience"]:
                print("🚨 早停触发")
                break

        # 3.2.8 保存本轮模型
        epoch_save_path = os.path.join(train_config["SAVE_DIR"], f"yolo11_{model_config['MODEL_SIZE']}_roi_epoch_{epoch + 1}_2c.pt")
        torch.save(model.state_dict(), epoch_save_path)

    print("=== 训练完成 ===")
    print(f"最优模型路径：{os.path.join(train_config['SAVE_DIR'], train_config['MODEL_NAME'])}")
    print(f"最优正样本F1：{best_pos_f1:.4f}")

# ===================== 调用训练函数 =====================
if __name__ == '__main__':
    multiprocessing.freeze_support()
    train(model_config, train_config, dataset_config, loss_config)