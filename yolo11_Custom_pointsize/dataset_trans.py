import os
import cv2
import shutil
import numpy as np
import torch
import threading
import gc  # 新增：垃圾回收模块
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入自定义模块
from zb_dataset import ZBGlobalImageDataset
from zb_main import process_zbuffer_with_rt

# ===================== 配置参数区（完全不变）=====================
SRC_DATASET = r"H:\pycharm\yolov11\yolov11_proj3\Datasets_Global_map400"
DST_DATASET = r"H:\pycharm\yolov11\yolov11_proj3\Datasets_ROI_map400"
ROI_SIZE = 64
THREAD_NUM = 16
# 新增：分批处理大小（仅控制内存，不影响速度/逻辑）
BATCH_PROCESS_SIZE = 100
# =================================================================

# 全局锁
lock = threading.Lock()
black_count = 0
valid_counter = 0


def is_black_image(img_np: np.ndarray) -> bool:
    return np.all(img_np == 0)


def create_dataset_dirs(dst_root):
    roi_root = os.path.join(dst_root, "roi_images")
    label_root = os.path.join(dst_root, "labels")
    os.makedirs(roi_root, exist_ok=True)
    os.makedirs(label_root, exist_ok=True)
    print(f"✅ 新数据集目录已创建：")
    print(f"   - ROI图像目录：{roi_root}")
    print(f"   - 标签目录：{label_root}")
    return roi_root, label_root


def convert_single_sample(sample, roi_root, roi_size, pbar):
    """处理单个样本：仅新增内存释放逻辑，其余完全不变"""
    img_idx = sample["img_idx"]
    global_img = sample["global_img"]
    rvec = sample["rvec"].cpu().numpy()
    tvec = sample["tvec"].cpu().numpy()
    src_label_path = os.path.join(SRC_DATASET, "labels", f"label_{img_idx}.json")

    try:
        global_img_np = sample["global_img"].cpu().numpy().transpose(1, 2, 0)
        global_img_np = (global_img_np * 255).astype(np.uint8)

        # 全黑图检测
        if is_black_image(global_img_np):
            black_img_path = os.path.join(SRC_DATASET, "global_images", f"{img_idx}.png")
            with lock:
                global black_count
                black_count += 1
            print(f"⚠️  跳过全黑图像：{black_img_path}")
            with lock:
                pbar.update(1)

            # ===================== 内存释放：全黑图分支 =====================
            del global_img, global_img_np, rvec, tvec  # 清理变量
            gc.collect()  # 强制回收
            # =================================================================
            return False, img_idx, "全黑图像，已跳过"

        # 线程安全下标
        with lock:
            global valid_counter
            current_valid_idx = valid_counter
            valid_counter += 1

        # 生成ROI
        exist_boxes = [1] * 12
        roi_imgs = process_zbuffer_with_rt(global_img_np, rvec, tvec, exist_boxes)
        if len(roi_imgs) != 12:
            raise ValueError(f"生成的ROI数量不为12，实际：{len(roi_imgs)}")

        # 保存ROI
        sample_roi_dir = os.path.join(roi_root, f"roi_{current_valid_idx}")
        os.makedirs(sample_roi_dir, exist_ok=True)
        for roi_idx, roi_img in enumerate(roi_imgs, 1):
            roi_img_bgr = cv2.cvtColor(roi_img, cv2.COLOR_RGB2BGR)
            roi_img_resized = cv2.resize(roi_img_bgr, (roi_size, roi_size))
            roi_save_path = os.path.join(sample_roi_dir, f"{roi_idx}.png")
            cv2.imwrite(roi_save_path, roi_img_resized)

        # 复制标签
        dst_label_path = os.path.join(DST_DATASET, "labels", f"label_{current_valid_idx}.json")
        shutil.copyfile(src_label_path, dst_label_path)

        with lock:
            pbar.update(1)

        # ===================== 核心：主动释放所有临时变量 =====================
        del global_img, global_img_np, roi_imgs, rvec, tvec  # 清理张量/数组
        gc.collect()  # 强制垃圾回收
        # =====================================================================
        return True, current_valid_idx, None

    except Exception as e:
        error_info = f"\n❌ 样本{img_idx}处理失败：{str(e)}"
        with lock:
            print(error_info)
            pbar.update(1)

        # ===================== 异常时也释放内存 =====================
        del global_img  # 清理张量
        gc.collect()
        # =============================================================
        return False, img_idx, str(e)


def main():
    roi_root, label_root = create_dataset_dirs(DST_DATASET)
    dataset = ZBGlobalImageDataset(dataset_root=SRC_DATASET, transform=None)
    total_samples = len(dataset)
    print(f"\n📊 开始多线程转换（线程数：{THREAD_NUM}）：共{total_samples}个有效样本")

    success_count = 0
    fail_count = 0
    fail_samples = []

    # ===================== 核心：分批处理任务（避免一次性提交所有样本） =====================
    for start_idx in range(0, total_samples, BATCH_PROCESS_SIZE):
        end_idx = min(start_idx + BATCH_PROCESS_SIZE, total_samples)
        batch_indices = list(range(start_idx, end_idx))

        # 单批次进度条
        pbar = tqdm(total=len(batch_indices), desc=f"转换进度 {start_idx}-{end_idx - 1}", position=0, leave=True)

        # 线程池处理当前批次
        with ThreadPoolExecutor(max_workers=THREAD_NUM) as executor:
            future_to_idx = {
                executor.submit(convert_single_sample, dataset[idx], roi_root, ROI_SIZE, pbar): idx
                for idx in batch_indices
            }

            for future in as_completed(future_to_idx):
                success, img_idx, error = future.result()
                with lock:
                    if success:
                        success_count += 1
                    else:
                        if error != "全黑图像，已跳过":
                            fail_count += 1
                            fail_samples.append((img_idx, error))

        pbar.close()
        # 每批处理完，强制全局内存回收
        gc.collect()
        torch.cuda.empty_cache()  # 清理CUDA缓存（如有GPU）
    # =====================================================================================

    # 完成统计（完全不变）
    print(f"\n🎉 转换完成！")
    print(f"✅ 成功处理：{success_count}个样本")
    print(f"❌ 处理失败：{fail_count}个样本")
    print(f"⚫ 全黑图像（已跳过）：{black_count}个")
    if fail_samples:
        print(f"❌ 失败样本列表：")
        for img_idx, error in fail_samples[:10]:
            print(f"   - 样本{img_idx}：{error}")
        if len(fail_samples) > 10:
            print(f"   - 更多{len(fail_samples) - 10}个失败样本未展示")
    print(f"📁 新数据集路径：{DST_DATASET}")
    print(f"📂 新数据集结构：")
    print(f"   {DST_DATASET}/")
    print(f"   ├── roi_images/  # 有效样本连续命名：roi_0/roi_1/...")
    print(f"   └── labels/      # 有效标签连续命名：label_0.json/...")


if __name__ == "__main__":
    os.environ["OPENCV_OPENCL_DEVICE"] = "-1"
    gc.collect()  # 初始回收
    torch.cuda.empty_cache()
    main()