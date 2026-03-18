import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ZBGlobalImageDataset(Dataset):
    """加载全局图像+标签+外参（rvec/tvec）数据集"""

    def __init__(self, dataset_root, transform=None):
        self.dataset_root = dataset_root
        self.global_img_root = os.path.join(dataset_root, "global_images")
        self.label_dir = os.path.join(dataset_root, "labels")
        self.transform = transform

        # 筛选有效样本（匹配images_*.png和label_*.json）
        self.valid_samples = []
        # 遍历所有全局图片
        for img_name in os.listdir(self.global_img_root):
            if img_name.startswith("images_") and img_name.endswith(".png"):
                # 提取索引：images_1.png → 1
                img_idx = int(img_name.split("_")[1].split(".")[0])
                label_path = os.path.join(self.label_dir, f"label_{img_idx}.json")
                if os.path.exists(label_path):
                    self.valid_samples.append(img_idx)

        # 排序保证样本顺序一致
        self.valid_samples.sort()

        # 打印数据集信息
        print(f"=== ZB数据集初始化完成 ===")
        print(f"全局图像根目录：{self.global_img_root}")
        print(f"标签文件目录：{self.label_dir}")
        print(f"有效样本数：{len(self.valid_samples)}")

    def _load_global_image(self, img_idx):
        """加载全局图像"""
        img_path = os.path.join(self.global_img_root, f"images_{img_idx}.png")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转RGB
        return img

    def _load_label_info(self, img_idx):
        """加载标签、外参、有效掩码"""
        label_path = os.path.join(self.label_dir, f"label_{img_idx}.json")
        with open(label_path, "r", encoding="utf-8") as f:
            ann = json.load(f)

        # 校验字段
        required_fields = ["labels", "roi_valid_mask", "rvec", "tvec"]
        for field in required_fields:
            assert field in ann, f"label_{img_idx}.json缺少字段：{field}"

        # 解析数据
        labels = np.array(ann["labels"], dtype=np.int64)  # [12] 0/1
        roi_valid_mask = np.array(ann["roi_valid_mask"], dtype=np.bool_)  # [12]
        rvec = np.array(ann["rvec"], dtype=np.float32).reshape(3, 1)  # (3,1) 适配zb_func
        tvec = np.array(ann["tvec"], dtype=np.float32).reshape(3, 1)  # (3,1)

        # 校验形状
        assert len(labels) == 12, f"labels长度需为12，当前：{len(labels)}"
        assert len(roi_valid_mask) == 12, f"roi_valid_mask长度需为12，当前：{len(roi_valid_mask)}"
        assert rvec.shape == (3, 1) and tvec.shape == (3, 1), "rvec/tvec形状需为(3,1)"

        return labels, roi_valid_mask, rvec, tvec

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        img_idx = self.valid_samples[idx]

        # 1. 加载全局图像
        global_img = self._load_global_image(img_idx)

        # 2. 预处理（如果有）
        if self.transform is not None:
            global_img = self.transform(global_img)
        else:
            # 默认预处理：转张量+归一化
            global_img = torch.from_numpy(global_img).permute(2, 0, 1).float() / 255.0

        # 3. 加载标签信息
        labels, roi_valid_mask, rvec, tvec = self._load_label_info(img_idx)

        # 转张量
        labels = torch.from_numpy(labels)
        roi_valid_mask = torch.from_numpy(roi_valid_mask)
        rvec = torch.from_numpy(rvec)
        tvec = torch.from_numpy(tvec)

        return {
            "global_img": global_img,  # 全局图像 [3, H, W]
            "labels": labels,  # 12个标签 [12]
            "roi_valid_mask": roi_valid_mask,  # 有效掩码 [12]
            "rvec": rvec,  # 旋转向量 (3,1)
            "tvec": tvec,  # 平移向量 (3,1)
            "img_idx": img_idx  # 样本索引
        }