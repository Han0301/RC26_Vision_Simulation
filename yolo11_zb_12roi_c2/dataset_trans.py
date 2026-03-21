import os
import cv2
import shutil
import numpy as np
import torch
import threading
from tqdm import tqdm  # 进度条（需安装：pip install tqdm）
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入自定义模块（确保以下文件在脚本同级目录）
from zb_dataset import ZBGlobalImageDataset
from zb_main import process_zbuffer_with_rt

# ===================== 配置参数区（修改这里！）=====================
# 原始ZBGlobal格式数据集根路径（必须包含global_images/labels文件夹）
SRC_DATASET = r"H:\pycharm\yolov11\yolov11_proj3\global_tests_100"
# 输出ROI12Image格式数据集根路径（脚本自动创建roi_images/labels）
DST_DATASET = r"H:\pycharm\yolov11\yolov11_proj1\datasets_global_test100"
# ROI图像输出尺寸（与ROI12ImageDataset默认一致）
ROI_SIZE = 64
# 多线程配置：线程数（IO密集型建议设为CPU核心数*2~4，如8/16/32）
THREAD_NUM = 16
# =================================================================

# 全局锁（保证多线程下计数/打印的线程安全）
lock = threading.Lock()


def create_dataset_dirs(dst_root):
    """创建新数据集目录结构：roi_images + labels"""
    roi_root = os.path.join(dst_root, "roi_images")
    label_root = os.path.join(dst_root, "labels")
    os.makedirs(roi_root, exist_ok=True)
    os.makedirs(label_root, exist_ok=True)
    print(f"✅ 新数据集目录已创建：")
    print(f"   - ROI图像目录：{roi_root}")
    print(f"   - 标签目录：{label_root}")
    return roi_root, label_root


def convert_single_sample(sample, roi_root, roi_size, pbar):
    """处理单个样本：生成ROI图像 + 复制标签文件（适配多线程）"""
    # 1. 提取样本核心数据
    img_idx = sample["img_idx"]  # 样本索引（如3 → roi_3）
    global_img = sample["global_img"]  # 全局图像张量 [3, H, W]
    rvec = sample["rvec"].cpu().numpy()  # 旋转向量 (3,1)
    tvec = sample["tvec"].cpu().numpy()  # 平移向量 (3,1)
    src_label_path = os.path.join(SRC_DATASET, "labels", f"label_{img_idx}.json")

    try:
        # 2. 全局图像张量转np.ndarray（HWC + RGB）
        global_img_np = sample["global_img"].cpu().numpy().transpose(1, 2, 0)
        global_img_np = (global_img_np * 255).astype(np.uint8)  # 反归一化（0-1 → 0-255）

        # 3. 生成ROI图像（exist_boxes全设为1）
        exist_boxes = [1] * 12  # 关键：所有exist_box置为1
        roi_imgs = process_zbuffer_with_rt(global_img_np, rvec, tvec, exist_boxes)
        if len(roi_imgs) != 12:
            raise ValueError(f"生成的ROI数量不为12，实际：{len(roi_imgs)}")

        # 4. 创建当前样本的ROI目录（如roi_images/roi_3）
        sample_roi_dir = os.path.join(roi_root, f"roi_{img_idx}")
        os.makedirs(sample_roi_dir, exist_ok=True)

        # 5. 保存12个ROI图像（1.png ~ 12.png）
        for roi_idx, roi_img in enumerate(roi_imgs, 1):  # roi_idx从1开始
            # ROI图像是RGB格式，cv2保存需转BGR
            roi_img_bgr = cv2.cvtColor(roi_img, cv2.COLOR_RGB2BGR)
            # 调整ROI尺寸（与ROI12ImageDataset一致）
            roi_img_resized = cv2.resize(roi_img_bgr, (roi_size, roi_size))
            # 保存ROI图像
            roi_save_path = os.path.join(sample_roi_dir, f"{roi_idx}.png")
            cv2.imwrite(roi_save_path, roi_img_resized)

        # 6. 复制标签文件到新数据集（标签格式完全兼容）
        dst_label_path = os.path.join(DST_DATASET, "labels", f"label_{img_idx}.json")
        shutil.copyfile(src_label_path, dst_label_path)

        # 线程安全更新进度条
        with lock:
            pbar.update(1)
        return True, img_idx, None

    except Exception as e:
        # 捕获异常并线程安全打印
        error_info = f"\n❌ 样本{img_idx}处理失败：{str(e)}"
        with lock:
            print(error_info)
            pbar.update(1)
        return False, img_idx, str(e)


def main():
    # 1. 初始化：创建目录 + 加载原始数据集
    roi_root, label_root = create_dataset_dirs(DST_DATASET)
    # 加载ZBGlobal数据集（无transform，保留原始数据）
    dataset = ZBGlobalImageDataset(dataset_root=SRC_DATASET, transform=None)
    total_samples = len(dataset)
    print(f"\n📊 开始多线程转换（线程数：{THREAD_NUM}）：共{total_samples}个有效样本")

    # 2. 初始化进度条（多线程安全）
    pbar = tqdm(total=total_samples, desc="转换进度", position=0, leave=True)

    # 3. 多线程处理样本
    success_count = 0
    fail_count = 0
    fail_samples = []  # 记录失败的样本索引和原因

    # 创建线程池
    with ThreadPoolExecutor(max_workers=THREAD_NUM) as executor:
        # 提交所有任务到线程池
        future_to_idx = {
            executor.submit(convert_single_sample, dataset[idx], roi_root, ROI_SIZE, pbar): idx
            for idx in range(total_samples)
        }

        # 遍历完成的任务，统计结果
        for future in as_completed(future_to_idx):
            success, img_idx, error = future.result()
            with lock:  # 保证计数线程安全
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                    fail_samples.append((img_idx, error))

    # 关闭进度条
    pbar.close()

    # 4. 转换完成统计
    print(f"\n🎉 转换完成！")
    print(f"✅ 成功处理：{success_count}个样本")
    print(f"❌ 失败处理：{fail_count}个样本")
    if fail_samples:
        print(f"❌ 失败样本列表：")
        for img_idx, error in fail_samples[:10]:  # 只打印前10个失败样本
            print(f"   - 样本{img_idx}：{error}")
        if len(fail_samples) > 10:
            print(f"   - 更多{len(fail_samples)-10}个失败样本未展示")
    print(f"📁 新数据集路径：{DST_DATASET}")
    print(f"📂 新数据集结构：")
    print(f"   {DST_DATASET}/")
    print(f"   ├── roi_images/  # 每个样本对应roi_*文件夹，内含1-12.png")
    print(f"   └── labels/      # 与原始数据集一致的label_*.json")


if __name__ == "__main__":
    # 强制单线程处理OpenCV相关操作（避免多线程下OpenCV报错）
    os.environ["OPENCV_OPENCL_DEVICE"] = "-1"
    main()