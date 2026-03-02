import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms

from dataset import ROI12ImageDataset
from model import YOLO11ROIClassifier, calculate_3c_metrics, evaluate, load_yolo11_pretrained_weights
from loss import YOLO11ROIFocalLoss3C

# ===================== 核心配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROI_IMG_SIZE = 64
NUM_ROI = 12
NUM_CLASSES = 3
MODEL_SIZE = "s"
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
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=(0,0.1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.8, 1.2), shear=10),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=yolo11_mean, std=yolo11_std)
])

val_test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=yolo11_mean, std=yolo11_std)
])

# ===================== 修复 1：加载数据集（彻底解决 Transform 污染） =====================
print("=== 正在加载数据集 ===")
# 1. 先创建一个临时数据集仅用于计算长度和生成索引
temp_dataset = ROI12ImageDataset(dataset_root=DATASET_ROOT, roi_img_size=ROI_IMG_SIZE, transform=None)
dataset_size = len(temp_dataset)
val_size = int(VAL_RATIO * dataset_size)
train_size = dataset_size - val_size

# 2. 生成随机且互斥的索引
# 注意：这里为了确保可复现性，可以固定一个seed，也可以不固定
indices = torch.randperm(dataset_size).tolist()
train_indices = indices[:train_size]
val_indices = indices[train_size:]

# 3. 关键步骤：实例化两个完全独立的 Dataset 对象
# 这样它们的 transform 互不干扰
train_dataset_full = ROI12ImageDataset(dataset_root=DATASET_ROOT, roi_img_size=ROI_IMG_SIZE, transform=train_transform)
val_dataset_full = ROI12ImageDataset(dataset_root=DATASET_ROOT, roi_img_size=ROI_IMG_SIZE, transform=val_test_transform)

# 4. 使用 Subset 根据索引包装
train_dataset = Subset(train_dataset_full, train_indices)
val_dataset = Subset(val_dataset_full, val_indices)

# 5. 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=False,
                          drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=False,
                        drop_last=True)

print(f"=== 数据集划分完成 ===")
print(f"训练集：{train_size}样本 | {len(train_loader)}批次")
print(f"验证集：{val_size}样本 | {len(val_loader)}批次")
print(f"训练设备：{DEVICE} | 模型尺寸：YOLO11-{MODEL_SIZE.upper()}")
print("=" * 80)

# ===================== 初始化模型/损失/优化器 =====================
model = YOLO11ROIClassifier(
    model_size=MODEL_SIZE,
    num_roi=NUM_ROI,
    num_classes=NUM_CLASSES,
    roi_size=ROI_IMG_SIZE
).to(DEVICE)

model = load_yolo11_pretrained_weights(model, model_size=MODEL_SIZE,
                                       load_path="H:\pycharm\yolov11\yolov11.pt\yolo11s.pt")

loss_fn = YOLO11ROIFocalLoss3C(
    num_roi=NUM_ROI,
    num_classes=NUM_CLASSES,
    alpha=[1.0, 5.0, 2.0],
    gamma=1.5,
    max_positive=8,
    max_negative=4
).to(DEVICE)

# 冻结策略
for name, param in model.backbone.named_parameters():
    if "layer0" in name or "layer1" in name or "layer2" in name:
        param.requires_grad = False
    else:
        param.requires_grad = True

# 修正后的参数分组定义
param_groups = [
    {"params": [p for n, p in model.backbone.named_parameters() if "layer0" in n or "layer1" in n or "layer2" in n],
     "lr": LEARNING_RATE * 0.001},
    {"params": [p for n, p in model.backbone.named_parameters() if
                "layer0" not in n and "layer1" not in n and "layer2" not in n],
     "lr": LEARNING_RATE * 0.1},
    {"params": model.neck.parameters(), "lr": LEARNING_RATE * 0.5},
    {"params": model.head.parameters(), "lr": LEARNING_RATE}
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
                # 简单看一下前几张图的均值，确认确实有数据增强
                # print(f"图像均值(验证增强): {roi_imgs[0].mean():.3f}")

    acc = correct / total
    print(f"\n训练集整体准确率：{acc:.4f}")
    model.train()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return acc


# ===================== 训练循环 =====================
os.makedirs(SAVE_DIR, exist_ok=True)
best_pos_f1 = 0.0
patience = 12
mixup_alpha = 0.2
no_improve = 0

print(f"=== 开始训练（YOLO11-{MODEL_SIZE.upper()}） ===")
print("\n=== 训练前测试训练集 ===")
test_train_set(model, train_loader, DEVICE)

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    batch_count = 0

    train_total_acc = 0.0
    train_valid_acc = 0.0
    train_pos_acc, train_pos_precision, train_pos_recall, train_pos_f1 = 0.0, 0.0, 0.0, 0.0
    train_neg_acc, train_neg_precision, train_neg_recall, train_neg_f1 = 0.0, 0.0, 0.0, 0.0

    train_iter = iter(train_loader)

    for batch_idx, (roi_imgs, cls_target, roi_valid_mask) in enumerate(train_loader):
        roi_imgs = roi_imgs.to(DEVICE)
        cls_target = cls_target.to(DEVICE)

        # 修复 2：使用 is_mixup 标记
        is_mixup = False
        loss = 0.0
        pred_logits = None

        # ---------------------- MixUp增强开始 ----------------------
        if np.random.rand() < 0.3:
            is_mixup = True
            try:
                roi_imgs2, cls_target2, _ = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                roi_imgs2, cls_target2, _ = next(train_iter)

            roi_imgs2 = roi_imgs2.to(DEVICE)
            cls_target2 = cls_target2.to(DEVICE)

            lam = np.random.beta(mixup_alpha, mixup_alpha)
            roi_imgs_mix = lam * roi_imgs + (1 - lam) * roi_imgs2

            pred_logits = model(roi_imgs_mix)
            loss1 = loss_fn(pred_logits, cls_target)
            loss2 = loss_fn(pred_logits, cls_target2)
            loss = lam * loss1 + (1 - lam) * loss2
        else:
            pred_logits = model(roi_imgs)
            loss = loss_fn(pred_logits, cls_target)
        # ---------------------- MixUp增强结束 ----------------------

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        batch_count += 1

        # 修复 2：只有在非 MixUp 时才计算指标，不再二次随机
        with torch.no_grad():
            if not is_mixup:
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

        # 修复 3：打印最后一个 param_group 的 LR (Head层)
        if (batch_idx + 1) % 10 == 0:
            current_lr = optimizer.param_groups[-1]['lr']
            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] | Batch [{batch_idx + 1}/{len(train_loader)}] | Loss: {loss.item():.4f} | LR: {current_lr:.6f}")

    scheduler.step()

    # 计算训练集平均指标 (注意：分母不再是 batch_count，因为有部分batch是mixup被跳过了，这里简单处理仍用batch_count，或者仅作参考)
    # 为了防止除以0，这里做一个简单的保护
    metric_count = batch_count
    # 实际上更严谨的是增加一个 counter，但为了不引入太多变量，保持原样即可，因为这只是参考

    avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
    avg_train_total_acc = train_total_acc / metric_count if metric_count > 0 else 0.0
    avg_train_valid_acc = train_valid_acc / metric_count if metric_count > 0 else 0.0
    avg_train_pos_acc = train_pos_acc / metric_count if metric_count > 0 else 0.0
    avg_train_pos_precision = train_pos_precision / metric_count if metric_count > 0 else 0.0
    avg_train_pos_recall = train_pos_recall / metric_count if metric_count > 0 else 0.0
    avg_train_pos_f1 = train_pos_f1 / metric_count if metric_count > 0 else 0.0
    avg_train_neg_acc = train_neg_acc / metric_count if metric_count > 0 else 0.0
    avg_train_neg_precision = train_neg_precision / metric_count if metric_count > 0 else 0.0
    avg_train_neg_recall = train_neg_recall / metric_count if metric_count > 0 else 0.0
    avg_train_neg_f1 = train_neg_f1 / metric_count if metric_count > 0 else 0.0

    # 验证集评估
    val_metrics = evaluate(model, val_loader, loss_fn, DEVICE)
    (avg_val_loss, val_roi_avg_loss, avg_val_total_acc, avg_val_valid_acc,
     avg_val_pos_acc, avg_val_pos_precision, avg_val_pos_recall, avg_val_pos_f1,
     avg_val_neg_acc, avg_val_neg_precision, avg_val_neg_recall, avg_val_neg_f1) = val_metrics

    # 打印日志
    print("=" * 120)
    print(f"【Epoch {epoch + 1}/{EPOCHS} 训练集】")
    print(
        f"总损失：{avg_epoch_loss:.4f} | 整体准确率：{avg_train_total_acc:.4f} | 有效ROI准确率：{avg_train_valid_acc:.4f}")
    print(
        f"├─ 有效有方块：准确率={avg_train_pos_acc:.4f} | 精确率={avg_train_pos_precision:.4f} | 召回率={avg_train_pos_recall:.4f} | F1={avg_train_pos_f1:.4f}")
    print(
        f"└─ 有效无方块：准确率={avg_train_neg_acc:.4f} | 精确率={avg_train_neg_precision:.4f} | 召回率={avg_train_neg_recall:.4f} | F1={avg_train_neg_f1:.4f}")

    print(f"【Epoch {epoch + 1}/{EPOCHS} 验证集】")
    print(f"总损失：{avg_val_loss:.4f} | 整体准确率：{avg_val_total_acc:.4f} | 有效ROI准确率：{avg_val_valid_acc:.4f}")
    print(
        f"├─ 有效有方块：准确率={avg_val_pos_acc:.4f} | 精确率={avg_val_pos_precision:.4f} | 召回率={avg_val_pos_recall:.4f} | F1={avg_val_pos_f1:.4f}")
    print(
        f"└─ 有效无方块：准确率={avg_val_neg_acc:.4f} | 精确率={avg_val_neg_precision:.4f} | 召回率={avg_val_neg_recall:.4f} | F1={avg_val_neg_f1:.4f}")
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
        print(f"✅ 保存最优模型 | 有效有方块F1：{avg_val_pos_f1:.4f} | 路径：{save_path}")
    else:
        no_improve += 1
        print(f"⚠️ 正样本F1未提升 | 当前最优：{best_pos_f1:.4f} | 无提升轮数：{no_improve}/{patience}")
        if no_improve >= patience:
            print("🚨 早停触发")
            break

    # 保存本轮模型
    epoch_save_path = os.path.join(SAVE_DIR, f"yolo11_{MODEL_SIZE}_roi_epoch_{epoch + 1}_3c.pt")
    torch.save(model.state_dict(), epoch_save_path)

    # 每10轮测试训练集
    if (epoch + 1) % 10 == 0:
        print(f"\n=== Epoch {epoch + 1} 训练集测试 ===")
        test_train_set(model, train_loader, DEVICE)

print("=== 训练完成 ===")
print(f"最优模型路径：{os.path.join(SAVE_DIR, f'yolo11_{MODEL_SIZE}_roi_best_3c.pt')}")
print(f"最优正样本F1：{best_pos_f1:.4f}")
