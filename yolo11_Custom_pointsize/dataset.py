import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

ROI_GROUPS = [[0,1,2], [3,4,5], [6,7,8], [9,10,11]]

class ROI12ImageDataset(Dataset):
    def __init__(self, dataset_root, roi_img_size=64, transform=None):
        self.dataset_root = dataset_root
        self.roi_img_root = os.path.join(dataset_root, "roi_images")
        self.label_dir = os.path.join(dataset_root, "labels")
        self.roi_img_size = roi_img_size
        self.transform = transform

        # 筛选有效样本
        self.valid_samples = []
        for img_idx in range(50000):
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
        return np.stack(roi_imgs, axis=0)  # 使用stack堆叠数组为张量 [12,64,64,3]

    def _compute_confidence(self, point_size):
        point_size = np.array(point_size, dtype=np.float32)
        conf_weight = np.zeros(12, dtype=np.float32)

        # 逐组计算：当前ROI权重 = 本组point_size / 本组最大值
        for group in ROI_GROUPS:
            group_vals = point_size[group]
            max_val = group_vals.max()
            # 避免除零，组内无点则权重=1.0
            if max_val < 1e-6:
                conf_weight[group] = 1.0
            else:
                conf_weight[group] = group_vals / max_val
        return conf_weight

    def _load_label(self, img_idx):
        label_path = os.path.join(self.label_dir, f"label_{img_idx}.json")
        with open(label_path, "r", encoding="utf-8") as f:
            ann = json.load(f)

        # 校验标签格式：labels为12个0/1值，roi_valid_mask保留（兼容原有逻辑）
        assert "labels" in ann and len(ann["labels"]) == 12, f"label_{img_idx}.json的labels需为12个0/1值"
        assert "point_size" in ann and len(ann["point_size"]) == 12

        # 直接使用labels字段作为二分类标签（0=无方块，1=有方块）
        cls_target = np.array(ann["labels"], dtype=np.int64)  # [12] 0/1
        conf_weight = self._compute_confidence(ann["point_size"])

        return cls_target,conf_weight

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
        cls_target, conf_weight = self._load_label(img_idx)
        cls_target = torch.from_numpy(cls_target)
        conf_weight = torch.from_numpy(conf_weight).float()  # 转为浮点张量

        return roi_imgs, cls_target, conf_weight
