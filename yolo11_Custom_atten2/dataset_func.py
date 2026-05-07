"""
dataset_func.py
    用于训练时的数据集功能包, 包含数据集变换, 加载数据集的方式
"""
import torch
from torchvision import transforms
from dataset_main import ROI12ImageDataset
from torch.utils.data import DataLoader, Subset, ConcatDataset, Dataset

# ===================== 【核心新增】全覆盖单ROI数据集（1样本拆12个） =====================
class ROI12SingleFullDataset(Dataset):
    def __init__(self, dataset_roots, roi_img_size=64, transform=None):
        super().__init__()
        self.base_dataset = ROI12ImageDataset(dataset_roots, roi_img_size, transform)
        self.samples = []
        # 每个样本的12个ROI全部覆盖，不随机、不遗漏
        for idx in range(len(self.base_dataset)):
            for pos in range(12):
                self.samples.append((idx, pos))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        base_idx, pos = self.samples[idx]
        imgs, target, conf = self.base_dataset[base_idx]
        return imgs[pos:pos+1], target[pos:pos+1], conf[pos:pos+1]

# ===================== 原有代码（完全不变） =====================
def transform():
    yolo11_mean = [0.485, 0.456, 0.406]
    yolo11_std = [0.224, 0.224, 0.225]
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=(0, 0.1)),
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
    return train_transform,val_test_transform

def load_dataset(DATASET_ROOTS:str, VAL_RATIO:float, BATCH_SIZE = 32,ROI_IMG_SIZE = 64,WORKERS = 0):
    train_transform,val_test_transform = transform()
    print("直接加载数据集")
    train_dataset = ROI12ImageDataset(dataset_roots=DATASET_ROOTS, roi_img_size=ROI_IMG_SIZE, transform=train_transform)
    val_dataset = ROI12ImageDataset(dataset_roots=DATASET_ROOTS, roi_img_size=ROI_IMG_SIZE, transform=val_test_transform)
    dataset_size = len(train_dataset)
    val_size = int(VAL_RATIO * dataset_size)
    train_size = dataset_size - val_size

    indices = torch.randperm(dataset_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=False,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=False,
                            drop_last=True)
    return train_loader,val_loader,train_size,val_size

def load_train_val_datasets(TRAIN_DATASETS:str,VAL_DATASETS:str,BATCH_SIZE = 32,ROI_IMG_SIZE = 64,WORKERS = 0):
    train_transform, val_test_transform = transform()
    print("直接指定训练集和验证集")
    train_dataset = ROI12ImageDataset(dataset_roots=TRAIN_DATASETS, roi_img_size=ROI_IMG_SIZE, transform=train_transform)
    val_dataset = ROI12ImageDataset(dataset_roots=VAL_DATASETS, roi_img_size=ROI_IMG_SIZE, transform=val_test_transform)
    train_size = len(train_dataset)
    val_size = len(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=False,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=False,
                            drop_last=True)
    return train_loader,val_loader,train_size,val_size

# ===================== 【核心新增】加载双数据集（12ROI + 全覆盖单ROI） =====================
def load_full_mixed_dataset(DATASET_ROOTS, VAL_RATIO, BATCH_SIZE=32, ROI_IMG_SIZE=64, WORKERS=0):
    train_transform, val_test_transform = transform()
    print("=== 加载 12ROI + 全覆盖单ROI 数据集 ===")

    # 训练集
    full_ds = ROI12ImageDataset(DATASET_ROOTS, ROI_IMG_SIZE, train_transform)
    single_ds = ROI12SingleFullDataset(DATASET_ROOTS, ROI_IMG_SIZE, train_transform)
    # 验证集
    val_full_ds = ROI12ImageDataset(DATASET_ROOTS, ROI_IMG_SIZE, val_test_transform)
    val_single_ds = ROI12SingleFullDataset(DATASET_ROOTS, ROI_IMG_SIZE, val_test_transform)

    # DataLoader
    full_loader = DataLoader(full_ds, BATCH_SIZE, shuffle=True, num_workers=WORKERS, drop_last=True)
    single_loader = DataLoader(single_ds, BATCH_SIZE, shuffle=True, num_workers=WORKERS, drop_last=True)
    val_full_loader = DataLoader(val_full_ds, BATCH_SIZE, shuffle=False, num_workers=WORKERS, drop_last=True)
    val_single_loader = DataLoader(val_single_ds, BATCH_SIZE, shuffle=False, num_workers=WORKERS, drop_last=True)

    return full_loader, single_loader, val_full_loader, val_single_loader
