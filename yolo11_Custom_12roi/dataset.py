import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ROI12ImageDataset(Dataset):
    """仅加载12个ROI图像+标签（适配YOLO11预处理，无全局图）"""

    def __init__(self, dataset_root, roi_img_size=64, transform=None):
        self.dataset_root = dataset_root
        self.roi_img_root = os.path.join(dataset_root, "roi_images")  # 12个ROI图目录
        self.label_dir = os.path.join(dataset_root, "labels")  # 标签目录
        self.roi_img_size = roi_img_size
        self.transform = transform

        # 筛选有效样本（仅需ROI图+标签）
        self.valid_samples = []
        for img_idx in range(30000):  # 最大样本数，可根据实际调整
            roi_dir = os.path.join(self.roi_img_root, f"roi_{img_idx}")
            label_path = os.path.join(self.label_dir, f"label_{img_idx}.json")
            if os.path.exists(roi_dir) and os.path.exists(label_path):
                self.valid_samples.append(img_idx)

        # 打印数据集信息
        print(f"=== 数据集初始化完成 ===")
        print(f"ROI图根目录：{self.roi_img_root}")
        print(f"标签文件目录：{self.label_dir}")
        print(f"有效样本数：{len(self.valid_samples)}")
        print(f"ROI目标尺寸：{self.roi_img_size}×{self.roi_img_size}")

    def _load_roi_imgs(self, img_idx):
        """加载12个ROI图像（YOLO11原生格式）"""
        roi_dir = os.path.join(self.roi_img_root, f"roi_{img_idx}")
        roi_imgs = []
        for roi_pos in range(1, 13):  # 12个ROI
            roi_path = os.path.join(roi_dir, f"{roi_pos}.png")
            roi_img = cv2.imread(roi_path)
            roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)  # YOLO默认RGB
            roi_img = cv2.resize(roi_img, (self.roi_img_size, self.roi_img_size))
            roi_imgs.append(roi_img)
        return np.stack(roi_imgs, axis=0)  # [12,64,64,3]

    def _load_label(self, img_idx):
        """加载标签并完成三分类映射（核心逻辑）"""
        label_path = os.path.join(self.label_dir, f"label_{img_idx}.json")
        with open(label_path, "r", encoding="utf-8") as f:
            ann = json.load(f)

        # 强制校验标签格式（YOLO11训练要求严格格式）
        assert "labels" in ann and len(ann["labels"]) == 12, f"label_{img_idx}.json的labels需为12个0/1值"
        assert "roi_valid_mask" in ann and len(
            ann["roi_valid_mask"]) == 12, f"label_{img_idx}.json的roi_valid_mask需为12个bool值"

        # 原始标签转换
        raw_labels = np.array(ann["labels"], dtype=np.int64)  # [12] 0=无方块/1=有方块
        roi_valid_mask = np.array(ann["roi_valid_mask"], dtype=np.bool_)  # [12] True=有效ROI

        # 三分类映射核心逻辑
        cls_target = np.zeros(12, dtype=np.int64)  # 初始化全为0类
        cls_target[roi_valid_mask & (raw_labels == 0)] = 1  # 有效无方块→1类
        cls_target[roi_valid_mask & (raw_labels == 1)] = 2  # 有效有方块→2类

        return cls_target, roi_valid_mask

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        img_idx = self.valid_samples[idx]

        # 1. 加载12个ROI图像
        roi_imgs = self._load_roi_imgs(img_idx)  # [12,64,64,3]

        # 2. YOLO11标准预处理
        if self.transform is not None:
            roi_imgs_list = []
            for roi_img in roi_imgs:
                roi_imgs_list.append(self.transform(roi_img))
            roi_imgs = torch.stack(roi_imgs_list, dim=0)  # [12,3,64,64]
        else:
            # YOLO11原生归一化（0-255→0-1）
            roi_imgs = torch.from_numpy(roi_imgs).permute(0, 3, 1, 2).float() / 255.0

        # 3. 加载并映射标签
        cls_target, roi_valid_mask = self._load_label(img_idx)
        cls_target = torch.from_numpy(cls_target)  # [12] 0/1/2类
        roi_valid_mask = torch.from_numpy(roi_valid_mask)  # [12] bool

        # 仅返回ROI相关数据（无全局图）
        return roi_imgs, cls_target, roi_valid_mask
