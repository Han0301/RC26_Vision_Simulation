import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ROI12ImageDataset(Dataset):
    """仅加载12个ROI图像+标签（适配二分类：直接使用labels字段）"""

    def __init__(self, dataset_root, roi_img_size=64, transform=None):
        self.dataset_root = dataset_root
        self.roi_img_root = os.path.join(dataset_root, "roi_images")
        self.label_dir = os.path.join(dataset_root, "labels")
        self.roi_img_size = roi_img_size
        self.transform = transform

        # 筛选有效样本
        self.valid_samples = []
        for img_idx in range(30000):
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
        """加载12个ROI图像（无修改）"""
        roi_dir = os.path.join(self.roi_img_root, f"roi_{img_idx}")
        roi_imgs = []
        for roi_pos in range(1, 13):
            roi_path = os.path.join(roi_dir, f"{roi_pos}.png")
            roi_img = cv2.imread(roi_path)
            roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
            roi_img = cv2.resize(roi_img, (self.roi_img_size, self.roi_img_size))
            roi_imgs.append(roi_img)
        return np.stack(roi_imgs, axis=0)  # [12,64,64,3]

    def _load_label(self, img_idx):
        """加载标签（核心修改：直接使用labels字段，移除三分类映射）"""
        label_path = os.path.join(self.label_dir, f"label_{img_idx}.json")
        with open(label_path, "r", encoding="utf-8") as f:
            ann = json.load(f)

        # 校验标签格式：labels为12个0/1值，roi_valid_mask保留（兼容原有逻辑）
        assert "labels" in ann and len(ann["labels"]) == 12, f"label_{img_idx}.json的labels需为12个0/1值"
        assert "roi_valid_mask" in ann and len(ann["roi_valid_mask"]) == 12, f"label_{img_idx}.json的roi_valid_mask需为12个bool值"

        # 直接使用labels字段作为二分类标签（0=无方块，1=有方块）
        cls_target = np.array(ann["labels"], dtype=np.int64)  # [12] 0/1
        roi_valid_mask = np.array(ann["roi_valid_mask"], dtype=np.bool_)  # 保留，兼容数据加载逻辑

        return cls_target, roi_valid_mask

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        img_idx = self.valid_samples[idx]

        # 1. 加载12个ROI图像
        roi_imgs = self._load_roi_imgs(img_idx)

        # 2. 预处理
        if self.transform is not None:
            roi_imgs_list = []
            for roi_img in roi_imgs:
                roi_imgs_list.append(self.transform(roi_img))
            roi_imgs = torch.stack(roi_imgs_list, dim=0)
        else:
            roi_imgs = torch.from_numpy(roi_imgs).permute(0, 3, 1, 2).float() / 255.0

        # 3. 加载二分类标签
        cls_target, roi_valid_mask = self._load_label(img_idx)
        cls_target = torch.from_numpy(cls_target)
        roi_valid_mask = torch.from_numpy(roi_valid_mask)

        return roi_imgs, cls_target, roi_valid_mask
