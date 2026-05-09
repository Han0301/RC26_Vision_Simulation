"""
show_atten.py
重构版：对外提供两个标准化函数
1. show_atten_single：单个样本注意力可视化
2. show_atten_datasets：数据集平均注意力可视化
"""
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import cv2
from tqdm import tqdm

from model import YOLO11ROIClassifier

# ===================== 🔥 核心配置：中文显示 + 固定参数 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 固定模型超参数（与原推理代码完全一致，无需修改）
MODEL_SIZE = "s"
NUM_ROI = 12
NUM_CLASSES = 2
ROI_IMG_SIZE = 64
# ========================================================================

# -------------------------- 私有工具函数（内部复用） -------------------------
def _load_model(model_path, device):
    """加载训练好的模型（私有工具函数）"""
    model = YOLO11ROIClassifier(
        model_size=MODEL_SIZE,
        num_roi=NUM_ROI,
        num_classes=NUM_CLASSES,
        roi_size=ROI_IMG_SIZE
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    return model

def _load_single_roi_input(roi_dir, device):
    """加载单个样本的12个ROI图像（私有工具函数）"""
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

    roi_imgs = []
    for roi_pos in range(1, 13):
        roi_path = os.path.join(roi_dir, f"{roi_pos}.png")
        if not os.path.exists(roi_path):
            print(f"⚠️ ROI文件缺失：{roi_path}，使用全黑图替代")
            roi_img = np.zeros((ROI_IMG_SIZE, ROI_IMG_SIZE, 3), dtype=np.uint8)
        else:
            roi_img = cv2.imread(roi_path)
            roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
            roi_img = cv2.resize(roi_img, (ROI_IMG_SIZE, ROI_IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        roi_imgs.append(roi_img)

    roi_imgs = np.stack(roi_imgs, axis=0)
    roi_imgs = torch.from_numpy(roi_imgs).permute(0, 3, 1, 2).float() / 255.0
    roi_imgs = roi_imgs.to(device)
    roi_imgs = (roi_imgs - mean) / std
    roi_imgs = roi_imgs.unsqueeze(0)  # [1,12,3,64,64]
    return roi_imgs

def _get_single_attn_map(model, device, roi_input):
    """提取单个样本的12×12注意力矩阵（私有工具函数）"""
    with torch.no_grad():
        _ = model(roi_input)
        attn_weights = model.attn_weights  # [1,12,12]
        return attn_weights[0].cpu().numpy()

def _plot_attention(attn_map, title, is_show, is_save, save_path, is_print):
    """绘制/打印/保存注意力热力图（私有工具函数）"""
    # 1. 终端打印矩阵
    if is_print:
        print("\n" + "=" * 60)
        print(f"🔥 {title}")
        print("=" * 60)
        np.set_printoptions(precision=3, suppress=True, linewidth=200)
        print(attn_map)

    # 2. 绘制热力图
    plt.figure(figsize=(10, 8))
    im = plt.imshow(attn_map, cmap="Blues", vmin=0, vmax=np.max(attn_map))
    plt.colorbar(im, label="注意力权重 (关注强度)")

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("被关注的 ROI (列)", fontsize=12)
    plt.ylabel("当前查询的 ROI (行)", fontsize=12)

    roi_labels = [f"ROI{i+1}" for i in range(12)]
    plt.xticks(range(12), roi_labels, rotation=45)
    plt.yticks(range(12), roi_labels)

    # 标注数值
    for i in range(12):
        for j in range(12):
            plt.text(j, i, f"{attn_map[i,j]:.3f}", ha="center", va="center", color="black", fontsize=6)

    plt.tight_layout()

    # 3. 保存图片
    if is_save:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        print(f"✅ 热力图已保存：{save_path}")

    # 4. 显示图片
    if is_show:
        plt.show()
    plt.close()

# -------------------------- 对外暴露的核心函数 -------------------------
def show_atten_single(image_path, model_path, is_show, is_save, save_path, is_print):
    """
    显示【单个样本】的ROI注意力权重
    :param image_path: 单个样本的ROI文件夹路径（如 roi_87 的完整路径）
    :param model_path: 模型权重文件路径
    :param is_show: 是否弹出窗口显示热力图
    :param is_save: 是否将热力图保存到本地
    :param save_path: 热力图保存的完整路径（含文件名）
    :param is_print: 是否在终端打印12×12注意力矩阵
    """
    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载模型与ROI数据
    model = _load_model(model_path, device)
    roi_input = _load_single_roi_input(image_path, device)
    print(f"✅ 加载单个ROI样本完成：{image_path}")
    # 提取注意力矩阵
    attn_map = _get_single_attn_map(model, device, roi_input)
    # 可视化
    title = "单个样本 - ROI 12×12空间注意力热力图"
    _plot_attention(attn_map, title, is_show, is_save, save_path, is_print)

def show_atten_datasets(datasets_path, model_path, is_show, is_save, save_path, is_print):
    """
    显示【整个数据集】的ROI平均注意力权重（所有样本取均值）
    :param datasets_path: 数据集根目录（包含 roi_images 子文件夹）
    :param model_path: 模型权重文件路径
    :param is_show: 是否弹出窗口显示热力图
    :param is_save: 是否将热力图保存到本地
    :param save_path: 热力图保存的完整路径（含文件名）
    :param is_print: 是否在终端打印12×12平均注意力矩阵
    """
    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载模型
    model = _load_model(model_path, device)
    # 获取所有ROI样本文件夹
    roi_root = os.path.join(datasets_path, "roi_images")
    if not os.path.exists(roi_root):
        raise FileNotFoundError(f"数据集错误：未找到 roi_images 文件夹 → {roi_root}")

    # 筛选所有 roi_xx 样本文件夹
    roi_dirs = [
        os.path.join(roi_root, d)
        for d in os.listdir(roi_root)
        if os.path.isdir(os.path.join(roi_root, d)) and d.startswith("roi_")
    ]
    if not roi_dirs:
        raise ValueError(f"数据集错误：未找到任何 roi_xx 样本文件夹 → {roi_root}")

    print(f"✅ 数据集共加载 {len(roi_dirs)} 个ROI样本，开始计算平均注意力...")
    # 累加所有样本的注意力矩阵
    total_attn = np.zeros((NUM_ROI, NUM_ROI), dtype=np.float32)
    for roi_dir in tqdm(roi_dirs, desc="处理进度", colour="red", total=len(roi_dirs)):
        roi_input = _load_single_roi_input(roi_dir, device)
        attn_map = _get_single_attn_map(model, device, roi_input)
        total_attn += attn_map

    # 计算均值矩阵
    mean_attn_map = total_attn / len(roi_dirs)
    print(f"✅ 数据集平均注意力矩阵计算完成！")
    # 可视化均值
    title = f"数据集平均 - ROI 12×12空间注意力热力图（{len(roi_dirs)}个样本）"
    _plot_attention(mean_attn_map, title, is_show, is_save, save_path, is_print)

# -------------------------- 使用示例 -------------------------
if __name__ == '__main__':
    # 模型路径（通用）
    MODEL_PATH = r"H:\pycharm\yolov11\yolov11_proj3\yolo11Custom_atten\evlate_pt\yolo11s_roi12_atten_2.pt"

    # ========== 示例1：调用【单个样本】可视化 ==========
    # SINGLE_ROI_PATH = r"H:\pycharm\yolov11\yolov11_proj3\datasets_real_p179\roi_images\roi_87"
    # SINGLE_SAVE_PATH = r"H:\pycharm\yolov11\yolov11_proj3\yolo11Custom_atten\evlate_pt\single_atten.png"
    # show_atten_single(
    #     image_path=SINGLE_ROI_PATH,
    #     model_path=MODEL_PATH,
    #     is_show=True,
    #     is_save=True,
    #     save_path=SINGLE_SAVE_PATH,
    #     is_print=True
    # )

    # ========== 示例2：调用【数据集平均】可视化 ==========
    DATASET_ROOT = r"H:\pycharm\yolov11\yolov11_proj3\datasets_test_2520"
    DATASET_SAVE_PATH = r"H:\pycharm\yolov11\yolov11_proj3\yolo11Custom_atten\atten_png\atten_2_datasets_test_2520.png"
    show_atten_datasets(
        datasets_path=DATASET_ROOT,
        model_path=MODEL_PATH,
        is_show=True,
        is_save=True,
        save_path=DATASET_SAVE_PATH,
        is_print=True
    )