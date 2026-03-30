from dataset import ROI12ImageDataset  # 直接复用你已有的数据集类
import os
import cv2
import torch
import numpy as np
from torchvision import transforms

def get_datasets_mean_std(dataset_root):
    """
    计算【现实数据集】所有ROI图像的 全局均值 & 标准差
    :param dataset_root: 现实数据集根路径
    :return: mean (torch.Tensor), std (torch.Tensor)  shape=[3] (RGB)
    """
    # 1. 初始化累加变量
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_roi_count = 0  # 总ROI数量（179样本 ×12 = 2148个）

    # 2. 实例化数据集（无transform，加载原始像素）
    dataset = ROI12ImageDataset(dataset_roots=dataset_root, roi_img_size=64, transform=None)
    total_samples = len(dataset)
    print(f"正在计算统计量 | 总样本数：{total_samples} | 总ROI数：{total_samples * 12}")

    # 3. 遍历所有样本 + 所有ROI
    for idx in range(total_samples):
        # 直接用数据集加载12个ROI (numpy数组 [12, 64, 64, 3], RGB格式, 0-255)
        roi_imgs, _, _ = dataset[idx]

        # 转numpy并归一化到 0~1（核心！必须做，否则数值爆炸）
        roi_imgs_np = roi_imgs.permute(0, 2, 3, 1).cpu().numpy()

        # 遍历12个ROI，累加均值/标准差
        for roi_img in roi_imgs_np:
            # 计算单张ROI的均值和标准差 (H,W,C) → 对H、W维度求平均
            img_mean = np.mean(roi_img, axis=(0, 1))  # shape [3]
            img_std = np.std(roi_img, axis=(0, 1))  # shape [3]

            # 累加
            mean += torch.from_numpy(img_mean)
            std += torch.from_numpy(img_std)
            total_roi_count += 1

    # 4. 全局平均（除以总ROI数量）
    mean /= total_roi_count
    std /= total_roi_count

    print(f"✅ 计算完成！")
    print(f"均值 (RGB)：{mean.numpy()}")
    print(f"标准差 (RGB)：{std.numpy()}")
    return mean, std

def denormalize(tensor, mean, std):
    """反归一化：将模型输入的归一化张量 还原为 0~255的可视化图像"""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    img = tensor * std + mean  # 逆变换
    img = torch.clamp(img, 0, 1)  # 截断到0~1
    return (img * 255).byte().numpy()  # 转uint8

def visualize_roi_normalization(roi_folder_path, mean, std, img_size=64):
    """
    测试归一化变换 + 拼接3*4大图显示
    :param roi_folder_path: 存放1.png~12.png的文件夹路径
    :param mean: 归一化均值 (list/np.ndarray)
    :param std: 归一化标准差 (list/np.ndarray)
    :param img_size: ROI尺寸，默认64
    """
    # 1. 定义和训练完全一致的变换
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    processed_imgs = []
    # 2. 加载并处理1~12张ROI图片
    for i in range(1, 13):
        img_path = os.path.join(roi_folder_path, f"{i}.png")
        if not os.path.exists(img_path):
            print(f"警告：{img_path} 不存在，使用黑色占位")
            img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        else:
            # 读取图片 + BGR转RGB
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))

        # 执行变换
        tensor_img = transform(img)  # 归一化后的张量
        # 反归一化，还原为可显示的图像
        vis_img = denormalize(tensor_img, mean, std)
        vis_img = vis_img.transpose(1, 2, 0)  # C,H,W → H,W,C
        processed_imgs.append(vis_img)

    # 3. 拼接 3行×4列 大图
    row1 = np.hstack(processed_imgs[0:4])   # 1-4
    row2 = np.hstack(processed_imgs[4:8])   # 5-8
    row3 = np.hstack(processed_imgs[8:12]) # 9-12
    final_img = np.vstack([row1, row2, row3])

    # 4. 显示结果
    cv2.imshow("ROI 归一化可视化（3×4拼接）", cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ===================== 调用示例 =====================
if __name__ == '__main__':
    # # 你的现实数据集路径
    # REAL_DATASET_PATH = r"H:\pycharm\yolov11\yolov11_proj3\datasets_real_p179"
    # # 计算均值/标准差
    # real_mean, real_std = get_datasets_mean_std(REAL_DATASET_PATH)
    # # 保存结果
    # torch.save({"mean": real_mean, "std": real_std}, REAL_DATASET_PATH + "/real_mean_std.pth")
    # print(f"\n💾 已保存均值标准差到 {REAL_DATASET_PATH}/real_mean_std.pth")

    real_stats = torch.load("real_mean_std.pth")
    REAL_MEAN = real_stats["mean"].numpy()
    REAL_STD = real_stats["std"].numpy()
    YOLO_MEAN = [0.485, 0.456, 0.406]
    YOLO_STD = [0.229, 0.224, 0.225]
    # 2. 输入你的ROI文件夹路径（里面是1~12.png）
    TEST_ROI_FOLDER = r"H:\pycharm\yolov11\yolov11_proj3\Datasets_ROI_map400\roi_images\roi_38"  # 替换成你的路径

    # 3. 执行测试
    visualize_roi_normalization(
        roi_folder_path=TEST_ROI_FOLDER,
        mean=REAL_MEAN,
        std=REAL_STD,
        img_size=64
    )
