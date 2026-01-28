import os
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms

from dataset import ROI12ImageDataset
from model import YOLO11ROIClassifier, calculate_3c_metrics, evaluate
from loss import YOLO11ROIFocalLoss3C

# ===================== 核心配置（支持n/s/l切换） =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROI_IMG_SIZE = 64
NUM_ROI = 12
NUM_CLASSES = 3
MODEL_SIZE = "s"  # 可选："n"（最快）、"s"（平衡）、"l"（最准）
# 根据模型尺寸自适应超参数
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 5e-5 if MODEL_SIZE == "l" else 1e-4 if MODEL_SIZE == "s" else 1e-3
WEIGHT_DECAY = 5e-4
DATASET_ROOT = r"H:\pycharm\yolov11\yolov11_proj1\datasets_16334"
SAVE_DIR = "./checkpoints"
VAL_RATIO = 0.2

# ===================== 数据预处理 =====================
yolo11_mean = [0.485, 0.456, 0.406]
yolo11_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(mean=yolo11_mean, std=yolo11_std)
])

val_test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=yolo11_mean, std=yolo11_std)
])

# ===================== 加载数据集 =====================
full_dataset = ROI12ImageDataset(
    dataset_root=DATASET_ROOT,
    roi_img_size=ROI_IMG_SIZE,
    transform=None
)
val_size = int(VAL_RATIO * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_test_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=False)

print(f"=== 数据集划分完成 ===")
print(f"训练集：{train_size}样本 | {len(train_loader)}批次")
print(f"验证集：{val_size}样本 | {len(val_loader)}批次")
print(f"训练设备：{DEVICE} | 模型尺寸：YOLO11-{MODEL_SIZE.upper()}")
print("=" * 80)

# ===================== 初始化模型/损失/优化器 =====================
model = YOLO11ROIClassifier(
    model_size=MODEL_SIZE,  # 传入模型尺寸
    num_roi=NUM_ROI,
    num_classes=NUM_CLASSES,
    roi_size=ROI_IMG_SIZE
).to(DEVICE)

loss_fn = YOLO11ROIFocalLoss3C(
    num_roi=NUM_ROI,
    num_classes=NUM_CLASSES,
    alpha=[1.0, 3.0, 2.0],
    gamma=1.5,
    max_positive=8,
    max_negative=4
).to(DEVICE)

# 分层学习率（大模型Backbone学习率更低）
lr_scale = 0.005 if MODEL_SIZE == "l" else 0.01 if MODEL_SIZE == "s" else 0.03
param_groups = [
    {"params": model.backbone.parameters(), "lr": LEARNING_RATE * lr_scale},
    {"params": model.neck.parameters(), "lr": LEARNING_RATE},
    {"params": model.head.parameters(), "lr": LEARNING_RATE},
]
optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# ===================== 训练集测试函数 =====================
def test_train_set(model, train_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (roi_imgs, cls_target, roi_valid_mask) in enumerate(train_loader):
            roi_imgs = roi_imgs.to(device)
            cls_target = cls_target.to(device)
            roi_valid_mask = roi_valid_mask.to(device)

            pred_logits = model(roi_imgs)
            pred_cls = torch.argmax(pred_logits, dim=-1)
            pred_cls[~roi_valid_mask] = 0

            correct += (pred_cls == cls_target).sum().item()
            total += cls_target.numel()

            if batch_idx < 3:
                print(f"【训练集测试】Batch {batch_idx}")
                print(f"真实标签：{cls_target[0].cpu().numpy()}")
                print(f"预测标签：{pred_cls[0].cpu().numpy()}")
                print("-" * 50)

    acc = correct / total
    print(f"\n训练集整体准确率：{acc:.4f}")
    model.train()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return acc

# ===================== 训练循环 =====================
os.makedirs(SAVE_DIR, exist_ok=True)
best_pos_f1 = 0.0
patience = 8
no_improve = 0

print(f"=== 开始训练（YOLO11-{MODEL_SIZE.upper()}原生训练策略） ===")
print("\n=== 训练前测试训练集 ===")
test_train_set(model, train_loader, DEVICE)

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    batch_count = 0

    train_total_acc = 0.0
    train_valid_acc = 0.0
    train_pos_acc, train_pos_precision, train_pos_recall, train_pos_f1 = 0.0,0.0,0.0,0.0
    train_neg_acc, train_neg_precision, train_neg_recall, train_neg_f1 = 0.0,0.0,0.0,0.0

    for batch_idx, (roi_imgs, cls_target, roi_valid_mask) in enumerate(train_loader):
        roi_imgs = roi_imgs.to(DEVICE)
        cls_target = cls_target.to(DEVICE)

        pred_logits = model(roi_imgs)
        loss = loss_fn(pred_logits, cls_target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        batch_count += 1

        metrics = calculate_3c_metrics(pred_logits, cls_target)
        train_total_acc += metrics["total_acc"]
        train_valid_acc += metrics["valid_acc"]
        train_pos_acc += metrics["pos_metrics"]["acc"]
        train_pos_precision += metrics["pos_metrics"]["precision"]
        train_pos_recall += metrics["pos_metrics"]["recall"]
        train_pos_f1 += metrics["pos_metrics"]["f1"]
        train_neg_acc += metrics["neg_metrics"]["acc"]
        train_neg_precision += metrics["neg_metrics"]["precision"]
        train_neg_recall += metrics["neg_metrics"]["recall"]
        train_neg_f1 += metrics["neg_metrics"]["f1"]

        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{EPOCHS}] | Batch [{batch_idx + 1}/{len(train_loader)}] | Loss: {loss.item():.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    scheduler.step()

    # 计算训练集平均指标
    avg_epoch_loss = epoch_loss / batch_count if batch_count>0 else 0.0
    avg_train_total_acc = train_total_acc / batch_count if batch_count>0 else 0.0
    avg_train_valid_acc = train_valid_acc / batch_count if batch_count>0 else 0.0
    avg_train_pos_acc = train_pos_acc / batch_count if batch_count>0 else 0.0
    avg_train_pos_precision = train_pos_precision / batch_count if batch_count>0 else 0.0
    avg_train_pos_recall = train_pos_recall / batch_count if batch_count>0 else 0.0
    avg_train_pos_f1 = train_pos_f1 / batch_count if batch_count>0 else 0.0
    avg_train_neg_acc = train_neg_acc / batch_count if batch_count>0 else 0.0
    avg_train_neg_precision = train_neg_precision / batch_count if batch_count>0 else 0.0
    avg_train_neg_recall = train_neg_recall / batch_count if batch_count>0 else 0.0
    avg_train_neg_f1 = train_neg_f1 / batch_count if batch_count>0 else 0.0

    # 验证集评估
    val_metrics = evaluate(model, val_loader, loss_fn, DEVICE)
    (avg_val_loss, val_roi_avg_loss, avg_val_total_acc, avg_val_valid_acc,
     avg_val_pos_acc, avg_val_pos_precision, avg_val_pos_recall, avg_val_pos_f1,
     avg_val_neg_acc, avg_val_neg_precision, avg_val_neg_recall, avg_val_neg_f1) = val_metrics

    # 打印日志
    print("=" * 120)
    print(f"【Epoch {epoch + 1}/{EPOCHS} 训练集（YOLO11-{MODEL_SIZE.upper()}）】")
    print(f"总损失：{avg_epoch_loss:.4f} | 整体准确率：{avg_train_total_acc:.4f} | 有效ROI准确率：{avg_train_valid_acc:.4f}")
    print(f"├─ 有效有方块（2类）：准确率={avg_train_pos_acc:.4f} | 精确率={avg_train_pos_precision:.4f} | 召回率={avg_train_pos_recall:.4f} | F1={avg_train_pos_f1:.4f}")
    print(f"└─ 有效无方块（1类）：准确率={avg_train_neg_acc:.4f} | 精确率={avg_train_neg_precision:.4f} | 召回率={avg_train_neg_recall:.4f} | F1={avg_train_neg_f1:.4f}")

    print(f"【Epoch {epoch + 1}/{EPOCHS} 验证集（YOLO11-{MODEL_SIZE.upper()}）】")
    print(f"总损失：{avg_val_loss:.4f} | 整体准确率：{avg_val_total_acc:.4f} | 有效ROI准确率：{avg_val_valid_acc:.4f}")
    print(f"├─ 有效有方块（2类）：准确率={avg_val_pos_acc:.4f} | 精确率={avg_val_pos_precision:.4f} | 召回率={avg_val_pos_recall:.4f} | F1={avg_val_pos_f1:.4f}")
    print(f"└─ 有效无方块（1类）：准确率={avg_val_neg_acc:.4f} | 精确率={avg_val_neg_precision:.4f} | 召回率={avg_val_neg_recall:.4f} | F1={avg_val_neg_f1:.4f}")
    print("=" * 120)

    # 早停+保存模型
    if avg_val_pos_f1 > best_pos_f1:
        best_pos_f1 = avg_val_pos_f1
        no_improve = 0
        save_path = os.path.join(SAVE_DIR, f"yolo11_{MODEL_SIZE}_roi_best_3c.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_pos_f1': best_pos_f1,
            'loss': avg_val_loss,
        }, save_path)
        print(f"✅ 保存YOLO11-{MODEL_SIZE.upper()}最优模型 | 有效有方块F1：{avg_val_pos_f1:.4f} | 路径：{save_path}")
    else:
        no_improve += 1
        print(f"⚠️ 正样本F1未提升 | 当前最优：{best_pos_f1:.4f} | 无提升轮数：{no_improve}/{patience}")
        if no_improve >= patience:
            print("🚨 早停触发：验证集正样本F1不再提升")
            break

    # 保存本轮模型
    epoch_save_path = os.path.join(SAVE_DIR, f"yolo11_{MODEL_SIZE}_roi_epoch_{epoch + 1}_3c.pt")
    torch.save(model.state_dict(), epoch_save_path)

    # 每10轮测试训练集
    if (epoch + 1) % 10 == 0:
        print(f"\n=== Epoch {epoch + 1} 训练集测试（YOLO11-{MODEL_SIZE.upper()}） ===")
        test_train_set(model, train_loader, DEVICE)

print("=== YOLO11训练完成 ===")
print(f"最优模型路径：{os.path.join(SAVE_DIR, f'yolo11_{MODEL_SIZE}_roi_best_3c.pt')}")
print(f"最优正样本F1：{best_pos_f1:.4f}")
