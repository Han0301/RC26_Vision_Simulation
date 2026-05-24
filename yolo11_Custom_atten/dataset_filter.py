import os
import json
import shutil
from tqdm import tqdm


def filter_invalid_samples(src_root: str, dst_root: str, junk_root: str, threshold: int = 800):
    """
    严格按照你的 dataset_main.py 逻辑读取数据集
    过滤point_size < 800 的无效样本
    ✅ 核心：原始数据集完全不修改、不删除、不移动
    """
    # ===================== 路径定义（和你代码完全一致） =====================
    src_roi_root = os.path.join(src_root, "roi_images")
    src_label_root = os.path.join(src_root, "labels")

    # 目标目录
    dst_roi_root = os.path.join(dst_root, "roi_images")
    dst_label_root = os.path.join(dst_root, "labels")
    junk_roi_root = os.path.join(junk_root, "roi_images")
    junk_label_root = os.path.join(junk_root, "labels")

    # 自动创建文件夹
    for path in [dst_roi_root, dst_label_root, junk_roi_root, junk_label_root]:
        os.makedirs(path, exist_ok=True)

    # ===================== 统计变量 =====================
    total_count = 0
    valid_count = 0
    invalid_count = 0

    print(f"✅ 按照原始数据集逻辑开始过滤 | 阈值：point_size < {threshold}")
    print(f"原始路径：{src_root}")
    print(f"有效样本：{dst_root}")
    print(f"垃圾样本：{junk_root}\n")

    # ===================== 核心：和你代码完全一致的遍历方式 =====================
    # 你原始代码：for img_idx in range(50000)，逐个检查是否存在
    for img_idx in tqdm(range(50000), desc="处理样本进度"):
        roi_dir_name = f"roi_{img_idx}"
        label_file_name = f"label_{img_idx}.json"

        src_roi_dir = os.path.join(src_roi_root, roi_dir_name)
        src_label_path = os.path.join(src_label_root, label_file_name)

        # 严格匹配你代码的判断：必须同时存在 ROI文件夹 和 label文件 才处理
        if not os.path.exists(src_roi_dir) or not os.path.exists(src_label_path):
            continue

        total_count += 1

        # ===================== 读取标签（和你代码校验逻辑一致） =====================
        try:
            with open(src_label_path, "r", encoding="utf-8") as f:
                ann = json.load(f)

            # 完全复刻你代码的断言校验
            assert "labels" in ann and len(ann["labels"]) == 12
            assert "point_size" in ann and len(ann["point_size"]) == 12

            point_size = ann["point_size"]
            # 任意一个point_size小于阈值 → 无效
            is_invalid = any(p < threshold for p in point_size)

        except Exception as e:
            print(f"\n❌ 标签读取失败 {label_file_name}：{str(e)}")
            is_invalid = True

        # ===================== 核心修改：全部使用 复制操作，不修改原数据 =====================
        if is_invalid:
            # 无效样本：复制到垃圾数据集（原文件保留）
            junk_roi = os.path.join(junk_roi_root, roi_dir_name)
            junk_label = os.path.join(junk_label_root, label_file_name)
            shutil.copytree(src_roi_dir, junk_roi)   # 复制文件夹
            shutil.copy(src_label_path, junk_label)  # 复制标签
            invalid_count += 1
        else:
            # 有效样本：复制到新数据集（原文件保留）
            dst_roi = os.path.join(dst_roi_root, roi_dir_name)
            dst_label = os.path.join(dst_label_root, label_file_name)
            shutil.copytree(src_roi_dir, dst_roi)
            shutil.copy(src_label_path, dst_label)
            valid_count += 1

    # ===================== 结果打印 =====================
    print("\n" + "=" * 50)
    print(f"🎯 数据集过滤完成！")
    print(f"总检测样本数：{total_count}")
    print(f"✅ 有效样本（已复制）：{valid_count}")
    print(f"🗑️  无效样本（已复制）：{invalid_count}")
    print(f"🔒 原始数据集未做任何修改！")
    print("=" * 50)


if __name__ == "__main__":
    # 你的路径（直接修改这里即可）
    SOURCE_DATASET = r"I:\datasets_real_blue785"
    NEW_DATASET = r"I:\datasets_real_blue_new785"
    JUNK_DATASET = r"I:\datasets_real_blue_waste785"

    filter_invalid_samples(
        src_root=SOURCE_DATASET,
        dst_root=NEW_DATASET,
        junk_root=JUNK_DATASET,
        threshold=800
    )