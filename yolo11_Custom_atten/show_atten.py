"""
show_atten.py
    显示图片推理时的注意力权重图
"""
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import cv2

from model import YOLO11ROIClassifier

# ===================== 🔥 核心修复：中文显示配置 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows默认黑体（支持中文）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常
# ================================================================

def load_real_roi_input(img_idx, dataset_root, roi_size=64, device="cuda"):
    """
    🔥 完全复用你INFER代码的逻辑：加载数据集里【真实的12个ROI图像】
    :param img_idx: 数据集样本编号 (和你infer的img_idx一致)
    :param dataset_root: 数据集根目录
    :return: 模型输入tensor [1,12,3,64,64] (完全匹配推理格式)
    """
    # 👇 完全照搬你的infer预处理逻辑
    roi_img_root = os.path.join(dataset_root, "roi_images")
    roi_dir = os.path.join(roi_img_root, f"roi_{img_idx}")

    # 归一化参数（和你推理代码完全一致）
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

    roi_imgs = []
    for roi_pos in range(1, 13):
        roi_path = os.path.join(roi_dir, f"{roi_pos}.png")
        if not os.path.exists(roi_path):
            print(f"⚠️ ROI文件缺失：{roi_path}，使用全黑图替代")
            roi_img = np.zeros((roi_size, roi_size, 3), dtype=np.uint8)
        else:
            roi_img = cv2.imread(roi_path)
            roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
            roi_img = cv2.resize(roi_img, (roi_size, roi_size), interpolation=cv2.INTER_LINEAR)
        roi_imgs.append(roi_img)

    # 格式转换（和推理完全一致）
    roi_imgs = np.stack(roi_imgs, axis=0)
    roi_imgs = torch.from_numpy(roi_imgs).permute(0, 3, 1, 2).float() / 255.0
    roi_imgs = roi_imgs.to(device)
    roi_imgs = (roi_imgs - mean) / std
    roi_imgs = roi_imgs.unsqueeze(0)  # [1,12,3,64,64]

    print(f"✅ 成功加载【真实ROI样本】：{img_idx} | 路径：{roi_dir}")
    return roi_imgs


def visualize_roi_attention(model, device, real_roi_input):
    """
    提取并可视化ROI 12×12注意力权重矩阵（真实ROI输入版）
    :param model: 训练好的YOLO11ROIClassifier
    :param device: 运行设备
    :param real_roi_input: 真实ROI预处理后的tensor [1,12,3,64,64]
    """
    model.eval()
    with torch.no_grad():
        # 🔥 关键：用【真实ROI】替代随机噪声！
        _ = model(real_roi_input)

        # 提取注意力权重：[1, 12, 12]
        attn_weights = model.attn_weights
        print(f"原始权重形状: {attn_weights.shape}")

        # 取第1个样本 → [12, 12]
        attn_map = attn_weights[0].cpu().numpy()

    # ===================== 打印 12×12 矩阵 =====================
    print("\n" + "=" * 60)
    print("🔥 ROI 12×12 注意力权重矩阵（行=当前ROI，列=关注的ROI）")
    print("=" * 60)
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    print(attn_map)

    # ===================== 绘制热力图 =====================
    plt.figure(figsize=(10, 8))
    im = plt.imshow(attn_map, cmap="Blues", vmin=0, vmax=np.max(attn_map))
    plt.colorbar(im, label="注意力权重 (关注强度)")

    plt.title("ROI 12×12 空间注意力热力图（真实ROI输入）", fontsize=14, fontweight='bold')
    plt.xlabel("被关注的 ROI (列)", fontsize=12)
    plt.ylabel("当前查询的 ROI (行)", fontsize=12)

    roi_labels = [f"ROI{i + 1}" for i in range(12)]
    plt.xticks(range(12), roi_labels, rotation=45)
    plt.yticks(range(12), roi_labels)

    for i in range(12):
        for j in range(12):
            text = plt.text(j, i, f"{attn_map[i, j]:.3f}",
                            ha="center", va="center", color="black", fontsize=6)

    plt.tight_layout()

    # 保存路径
    save_path = r"H:\pycharm\yolov11\yolov11_proj3\yolo11Custom_atten\evlate_pt\real_roi_atten.png"
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches='tight',
        pad_inches=0.1
    )
    plt.show()
    print(f"✅ 热力图已保存：{save_path}")


# ===================== 调用函数（真实ROI版） =====================
if __name__ == '__main__':
    # 👇 【必须修改】和你推理代码一致的配置
    MODEL_SIZE = "s"
    NUM_ROI = 12
    NUM_CLASSES = 2
    ROI_IMG_SIZE = 64
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🔥 数据集路径（和你infer的DATASET_ROOT完全一致）
    DATASET_ROOT = r"H:\pycharm\yolov11\yolov11_proj3\datasets_real_p179"
    # 🔥 测试用的真实样本编号（任意选一个，比如1，固定后结果永远不变）
    TEST_REAL_IMG_IDX = 87

    # 1. 初始化模型
    model = YOLO11ROIClassifier(
        model_size=MODEL_SIZE,
        num_roi=NUM_ROI,
        num_classes=NUM_CLASSES,
        roi_size=ROI_IMG_SIZE
    ).to(DEVICE)

    # 2. 加载权重
    checkpoint_path = r"H:\pycharm\yolov11\yolov11_proj3\yolo11Custom_atten\evlate_pt\yolo11s_roi12_atten_1.pt"
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # 3. 🔥 加载【真实ROI】（核心修改）
    real_roi = load_real_roi_input(
        img_idx=TEST_REAL_IMG_IDX,
        dataset_root=DATASET_ROOT,
        roi_size=ROI_IMG_SIZE,
        device=DEVICE
    )

    # 4. 执行可视化（真实输入 → 结果永久固定）
    visualize_roi_attention(model, DEVICE, real_roi)