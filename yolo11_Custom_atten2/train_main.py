"""
train_main.py
    主要功能函数: train, 训练roi12分类模型
"""
import multiprocessing
import os
import torch
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from train_func import evaluate
from model import YOLO11ROIClassifier
from load_model import load_YOLO_weights,reload_model

from loss import CountLoss, FocalLoss
# 【修复】正确同级导入，无循环
from dataset_func import load_full_mixed_dataset
from train_config import model_config,train_config,loss_config,dataset_config

def train(model_config, train_config, dataset_config, loss_config):
    device = model_config["DEVICE"]
    best_pos_f1 = 0.0
    no_improve = 0

    # ===================== 1. 加载双数据集 =====================
    full_loader, single_loader, val_full_loader, val_single_loader = load_full_mixed_dataset(
        dataset_config["DATASET_ROOTS"], dataset_config["VAL_RATIO"],
        train_config["BATCH_SIZE"], model_config["ROI_IMG_SIZE"], train_config["WORKERS"]
    )

    # ===================== 2. 模型/损失/优化器 =====================
    model = YOLO11ROIClassifier(
        model_size=model_config["MODEL_SIZE"],
        num_roi=model_config["NUM_ROI"],
        num_classes=model_config["NUM_CLASSES"],
        roi_size=model_config["ROI_IMG_SIZE"],
        atten_weight=model_config["ATTEN_WEIGHT"]
    ).to(device)

    cls_loss_fn = FocalLoss(
        num_classes=model_config["NUM_CLASSES"],
        alpha=loss_config["LOSS_WEIGHT"],
        gamma=loss_config["FOCAL_LOSS"],
    ).to(device)

    count_loss_fn = CountLoss(exist_count=8, weight=loss_config["count_loss_weight"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=loss_config["LEARNING_RATE"], weight_decay=loss_config["WEIGHT_DECAY"])
    scheduler = CosineAnnealingLR(optimizer, T_max=train_config["EPOCHS"], eta_min=1e-6)

    if not train_config["RESUME_TRAIN"]:
        model = load_YOLO_weights(model, model_size=model_config["MODEL_SIZE"], load_path=model_config["YOLO_weight_path"])

    os.makedirs(train_config["SAVE_DIR"], exist_ok=True)

    # ===================== 3. 一轮训练：交替输入 =====================
    print("=== 开始训练（12ROI + 全覆盖单ROI）===")
    for epoch in range(train_config["EPOCHS"]):
        model.train()
        total_loss = 0.0
        iter_full = iter(full_loader)
        iter_single = iter(single_loader)

        # 交替训练：一批12ROI → 一批单ROI
        for _ in range(max(len(full_loader), len(single_loader))):
            # 训练12ROI
            try:
                imgs, target, conf = next(iter_full)
                imgs, target = imgs.to(device), target.to(device)
                pred = model(imgs)
                loss = cls_loss_fn(pred, target) + count_loss_fn(pred)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            except StopIteration:
                pass

            # 训练全覆盖单ROI
            try:
                imgs, target, conf = next(iter_single)
                imgs, target = imgs.to(device), target.to(device)
                pred = model(imgs)
                loss = cls_loss_fn(pred, target) + count_loss_fn(pred)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            except StopIteration:
                pass

        scheduler.step()

        # ===================== 验证 =====================
        model.eval()
        with torch.no_grad():
            _, _, _, _, _, f1_full = evaluate(model, val_full_loader, cls_loss_fn, device)
            _, _, _, _, _, f1_single = evaluate(model, val_single_loader, cls_loss_fn, device)

        avg_f1 = (f1_full + f1_single) / 2
        print(f"Epoch {epoch+1} | Loss: {total_loss:.2f} | 12ROI-F1: {f1_full:.4f} | 单ROI-F1: {f1_single:.4f}")

        # 保存最优模型
        if avg_f1 > best_pos_f1:
            best_pos_f1 = avg_f1
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(train_config["SAVE_DIR"], "best_model.pt"))
            print(f"✅ 最优模型 | 平均F1: {best_pos_f1:.4f}")
        else:
            no_improve += 1
            if no_improve >= train_config["patience"]:
                print("🚨 早停触发")
                break

    print(f"\n训练完成！最优F1: {best_pos_f1:.4f}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    train(model_config, train_config, dataset_config, loss_config)
