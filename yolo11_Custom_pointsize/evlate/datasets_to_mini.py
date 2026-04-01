import os
import re
import random
import shutil
from typing import List


def split_custom_dataset(
        source_root: str,
        output_root: str,
        split_total: int,
        generate_num: int,
        random_seed: int = 114  # 随机种子，固定可复现划分结果
):
    # 1022: seed:1145
    # 2044: seed:114514
    # 3066: seed:114
    # 4088: seed:114
    """
    随机平均划分你的专属数据集（严格配对roi_x和label_x）

    参数：
        source_root: 原始大数据集根路径（包含roi_images、labels）
        output_root: 输出子数据集总路径（会自动创建datasets_1、datasets_2...）
        split_total: 要将数据集**平均划分成的总份数**
        generate_num: 实际要**生成的子数据集份数**（必须 ≤ split_total）
        random_seed: 随机种子，保证划分结果可复现
    """
    # ===================== 1. 输入参数校验 =====================
    if generate_num > split_total:
        raise ValueError(f"生成份数{generate_num}不能大于划分总份数{split_total}")
    if split_total <= 0 or generate_num <= 0:
        raise ValueError("划分份数和生成份数必须为正整数")
    if not os.path.exists(source_root):
        raise FileNotFoundError(f"原始数据集路径不存在：{source_root}")

    # 校验原始数据集核心文件夹
    source_roi_root = os.path.join(source_root, "roi_images")
    source_label_root = os.path.join(source_root, "labels")
    if not os.path.exists(source_roi_root):
        raise FileNotFoundError(f"缺失roi_images文件夹：{source_roi_root}")
    if not os.path.exists(source_label_root):
        raise FileNotFoundError(f"缺失labels文件夹：{source_label_root}")

    # ===================== 2. 获取所有有效样本索引（配对roi和label） =====================
    # 正则匹配roi_x文件夹（兼容roi_0/roi__0，匹配数字x）
    roi_pattern = re.compile(r"roi_+(\d+)")
    valid_sample_indices: List[int] = []

    # 遍历roi_images下的所有子文件夹
    for folder_name in os.listdir(source_roi_root):
        folder_path = os.path.join(source_roi_root, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # 提取样本数字x
        match = roi_pattern.match(folder_name)
        if not match:
            continue
        sample_idx = int(match.group(1))

        # 校验对应的label_x.json是否存在
        label_file = os.path.join(source_label_root, f"label_{sample_idx}.json")
        if os.path.exists(label_file):
            valid_sample_indices.append(sample_idx)

    if not valid_sample_indices:
        raise ValueError("未找到任何配对的roi样本和label文件，请检查数据集格式")

    # 打印样本总数
    total_samples = len(valid_sample_indices)
    print(f"✅ 找到有效样本总数：{total_samples} 个")

    # ===================== 3. 随机打乱样本 + 平均划分 =====================
    random.seed(random_seed)
    random.shuffle(valid_sample_indices)  # 全局随机打乱

    # 平均划分成split_total份
    split_samples = []
    samples_per_split = total_samples // split_total
    for i in range(split_total):
        start = i * samples_per_split
        # 最后一份包含剩余所有样本
        end = start + samples_per_split if i < split_total - 1 else total_samples
        split_samples.append(valid_sample_indices[start:end])

    print(f"✅ 已将{total_samples}个样本平均划分为{split_total}份")
    # 打印每份样本数量
    for idx, samples in enumerate(split_samples):
        print(f"  第{idx + 1}份：{len(samples)}个样本")

    # ===================== 4. 生成子数据集（仅生成前generate_num份） =====================
    for dataset_idx in range(1, generate_num + 1):
        # 子数据集路径：output_root/datasets_1、datasets_2...
        dataset_name = f"datasets_{dataset_idx}"
        dataset_root = os.path.join(output_root, dataset_name)
        dataset_roi_root = os.path.join(dataset_root, "roi_images")
        dataset_label_root = os.path.join(dataset_root, "labels")

        # 创建子数据集文件夹结构
        os.makedirs(dataset_roi_root, exist_ok=True)
        os.makedirs(dataset_label_root, exist_ok=True)

        # 获取当前子数据集的样本索引
        current_samples = split_samples[dataset_idx - 1]
        print(f"\n📂 开始生成 {dataset_name}，共{len(current_samples)}个样本")

        # 复制样本到子数据集
        for sample_idx in current_samples:
            # 源路径
            src_roi_folder = os.path.join(source_roi_root, f"roi_{sample_idx}")
            src_label_file = os.path.join(source_label_root, f"label_{sample_idx}.json")

            # 目标路径
            dst_roi_folder = os.path.join(dataset_roi_root, f"roi_{sample_idx}")
            dst_label_file = os.path.join(dataset_label_root, f"label_{sample_idx}.json")

            # 复制roi文件夹（包含12张图片）
            shutil.copytree(src_roi_folder, dst_roi_folder, dirs_exist_ok=True)
            # 复制label文件
            shutil.copy(src_label_file, dst_label_file)

        print(f"✅ {dataset_name} 生成完成！")

    print(f"\n🎉 所有{generate_num}个子数据集生成完毕！")

# 1. 配置你的参数
SOURCE_DATASET = "H:\pycharm\yolov11\yolov11_proj3\Datasets_ROI_map400"  # 你的原始大数据集路径
OUTPUT_ROOT = r"H:\pycharm\yolov11\yolov11_proj3\yolo11Custom_pointsize\test\mini_datasets_4088"        # 子数据集会生成在这里
SPLIT_TOTAL = 3                                 # 平均划分成5份
GENERATE_NUM = 3                               # 只生成前3份（datasets_1/2/3）

# 2. 执行划分
split_custom_dataset(
    source_root=SOURCE_DATASET,
    output_root=OUTPUT_ROOT,
    split_total=SPLIT_TOTAL,
    generate_num=GENERATE_NUM
)