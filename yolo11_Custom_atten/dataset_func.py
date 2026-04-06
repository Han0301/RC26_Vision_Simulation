"""
dataset_func.py
    用于训练时的数据集功能包, 包含数据集变换, 加载数据集的方式
"""
import torch
from torchvision import transforms
from dataset_main import ROI12ImageDataset
from torch.utils.data import DataLoader, Subset     # subset: PyTorch 数据集子集化工具，基于索引提取数据集的一部分

def transform():
    # 1 归一化和标准差
    yolo11_mean = [0.485, 0.456, 0.406]
    yolo11_std = [0.229, 0.224, 0.225]
    # 2 数据增强
    train_transform = transforms.Compose([
        transforms.ToPILImage(),  # 将 numpy 数组 / 张量转为 PIL 图像（因为多数变换仅支持 PIL 格式）。
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=(0, 0.1)),  # 颜色的扰动
        transforms.RandomHorizontalFlip(p=0.5),  # 50% 概率随机水平翻转
        transforms.RandomRotation(15),  # 随机旋转 ±15 度
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.8, 1.2), shear=10),
        # 随机仿射变换（平移 / 缩放 / 剪切），degrees=0 表示不旋转
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # 随机高斯模糊（核大小 3，sigma 范围 0.1~2.0）
        transforms.ToTensor(),  # 将 PIL 图像转为张量
        transforms.Normalize(mean=yolo11_mean, std=yolo11_std)  # 同时将像素值从 [0,255] 归一化到 [0,1]
    ])

    # 3 验证集仅做基础变换
    val_test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=yolo11_mean, std=yolo11_std)
    ])
    return train_transform,val_test_transform

def load_dataset(DATASET_ROOTS:str, VAL_RATIO:float, BATCH_SIZE = 32,ROI_IMG_SIZE = 64,WORKERS = 0):
    train_transform,val_test_transform = transform()
    print("直接加载数据集")
    # 3.1 实例化对象
    train_dataset = ROI12ImageDataset(dataset_roots=DATASET_ROOTS, roi_img_size=ROI_IMG_SIZE, transform=train_transform)
    val_dataset = ROI12ImageDataset(dataset_roots=DATASET_ROOTS, roi_img_size=ROI_IMG_SIZE, transform=val_test_transform)
    dataset_size = len(train_dataset)
    val_size = int(VAL_RATIO * dataset_size)
    train_size = dataset_size - val_size

    # 3.2 生成随机且互斥的索引
    indices = torch.randperm(dataset_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # 3.3 通过索引生成train_dataset, val_dataset
    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)

    # 3.4 创建 DataLoader
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