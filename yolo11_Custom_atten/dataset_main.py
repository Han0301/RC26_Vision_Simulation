"""
dataset_main.py
    定义数据集类, 加载数据集
"""
import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

ROI_GROUPS = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]

def _compute_confidence(point_size):
    point_size = np.array(point_size, dtype=np.float32)
    conf_weight = np.zeros(12, dtype=np.float32)

    for group in ROI_GROUPS:
        group_vals = point_size[group]
        max_val = group_vals.max()
        if max_val < 1e-6:
            conf_weight[group] = 1.0
        else:
            conf_weight[group] = group_vals / max_val
    return conf_weight

class ROI12ImageDataset(Dataset):
    def __init__(self, dataset_roots, roi_img_size=64, transform=None):
        self.dataset_roots = dataset_roots if isinstance(dataset_roots, list) else [dataset_roots]
        self.roi_img_size = roi_img_size
        self.transform = transform
        self.valid_samples = []

        # 遍历所有数据集根目录，合并样本
        for root_idx, dataset_root in enumerate(self.dataset_roots):
            roi_img_root = os.path.join(dataset_root, "roi_images")
            label_dir = os.path.join(dataset_root, "labels")

            # 筛选当前根目录下的有效样本
            for img_idx in range(50000):  # 保持原有的最大索引范围
                roi_dir = os.path.join(roi_img_root, f"roi_{img_idx}")
                label_path = os.path.join(label_dir, f"label_{img_idx}.json")
                if os.path.exists(roi_dir) and os.path.exists(label_path):
                    self.valid_samples.append((root_idx, img_idx))

        # 打印合并后数据集信息
        print(f"=== 数据集初始化完成 ===")
        for i, root in enumerate(self.dataset_roots):
            print(f"[{i + 1}] 根目录：{root}")
        print(f"总有效样本数：{len(self.valid_samples)}")
        print(f"ROI目标尺寸：{self.roi_img_size}×{self.roi_img_size}")

    def _load_roi_imgs(self, dataset_root, img_idx):
        """加载12个ROI图像"""
        roi_img_root = os.path.join(dataset_root, "roi_images")
        roi_dir = os.path.join(roi_img_root, f"roi_{img_idx}")
        roi_imgs = []
        for roi_pos in range(1, 13):
            roi_path = os.path.join(roi_dir, f"{roi_pos}.png")
            roi_img = cv2.imread(roi_path)
            roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
            roi_img = cv2.resize(roi_img, (self.roi_img_size, self.roi_img_size))
            roi_imgs.append(roi_img)
        return np.stack(roi_imgs, axis=0)

    def _load_label(self, dataset_root, img_idx):
        label_dir = os.path.join(dataset_root, "labels")
        label_path = os.path.join(label_dir, f"label_{img_idx}.json")
        with open(label_path, "r", encoding="utf-8") as f:
            ann = json.load(f)

        assert "labels" in ann and len(ann["labels"]) == 12, f"label_{img_idx}.json格式错误"
        assert "point_size" in ann and len(ann["point_size"]) == 12

        cls_target = np.array(ann["labels"], dtype=np.int64)
        conf_weight = _compute_confidence(ann["point_size"])

        return cls_target, conf_weight

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        root_idx, img_idx = self.valid_samples[idx]
        dataset_root = self.dataset_roots[root_idx]

        # 1. 加载ROI
        roi_imgs = self._load_roi_imgs(dataset_root, img_idx)

        # 2. 预处理
        if self.transform is not None:
            roi_imgs_list = []
            for roi_img in roi_imgs:
                roi_imgs_list.append(self.transform(roi_img))
            roi_imgs = torch.stack(roi_imgs_list, dim=0)
        else:
            roi_imgs = torch.from_numpy(roi_imgs).permute(0, 3, 1, 2).float() / 255.0

        # 3. 加载标签
        cls_target, conf_weight = self._load_label(dataset_root, img_idx)
        cls_target = torch.from_numpy(cls_target)
        conf_weight = torch.from_numpy(conf_weight).float()

        return roi_imgs, cls_target, conf_weight