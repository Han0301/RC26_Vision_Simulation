import os
import re
import numpy as np
import torch
from tqdm import tqdm
# 导入你原有的推理器类
from infer import YOLO11ROIInferencer  # 替换为你的py文件名


def get_valid_sample_indices(dataset_root):
    """获取测试集所有有效样本索引（roi_x + label_x配对）"""
    source_roi_root = os.path.join(dataset_root, "roi_images")
    source_label_root = os.path.join(dataset_root, "labels")
    roi_pattern = re.compile(r"roi_+(\d+)")
    valid_indices = []

    for folder_name in os.listdir(source_roi_root):
        folder_path = os.path.join(source_roi_root, folder_name)
        if not os.path.isdir(folder_path):
            continue
        match = roi_pattern.match(folder_name)
        if not match:
            continue
        idx = int(match.group(1))
        label_path = os.path.join(source_label_root, f"label_{idx}.json")
        if os.path.exists(label_path):
            valid_indices.append(idx)
    return sorted(valid_indices)


def compute_ensemble_logits_mse(
        model_paths: list,
        best_model_path: str,
        test_dataset_root: str,
        target_class: int = 1,
        model_size: str = "s",
        roi_size: int = 64,
        num_roi: int = 12,
        num_classes: int = 2
):
    """
    【修正版】计算【待测模型集合平均】与【最优模型】的平均平方误差
    :param model_paths: 所有待测模型路径列表
    :param best_model_path: 最优模型路径
    :param test_dataset_root: 测试集根路径
    :param target_class: 计算的类别(0无方块/1有方块)
    :return: 最终平均平方误差指标
    """
    # ===================== 1. 获取测试集有效样本 =====================
    sample_indices = get_valid_sample_indices(test_dataset_root)
    total_samples = len(sample_indices)
    num_test_models = len(model_paths)
    if total_samples == 0:
        raise ValueError("测试集无有效样本！")
    if num_test_models == 0:
        raise ValueError("待测模型列表为空！")

    print(f"✅ 测试集样本数: {total_samples}")
    print(f"✅ 待测模型数: {num_test_models}")
    print(f"✅ 计算类别: {target_class} (0=无方块,1=有方块)\n")

    # ===================== 2. 预计算：最优模型所有样本平均logits =====================
    print("🔍 加载最优模型并推理所有样本...")
    best_infer = YOLO11ROIInferencer(
        model_path=best_model_path, dataset_root=test_dataset_root,
        model_size=model_size, roi_size=roi_size, num_roi=num_roi, num_classes=num_classes
    )
    best_logits = []  # 形状: [N,]
    for idx in tqdm(sample_indices, desc="最优模型推理"):
        with torch.no_grad():
            roi_imgs = best_infer.preprocess_roi(idx)
            pred_logits = best_infer.model(roi_imgs)  # [1,12,2] 原始输出
        # 提取目标类别12个logits → 求平均
        cls_logits = pred_logits[0, :, target_class].cpu().numpy()
        best_logits.append(np.mean(cls_logits))
    best_logits = np.array(best_logits)

    # ===================== 3. 预计算：所有待测模型 所有样本平均logits =====================
    print(f"\n🔍 加载{num_test_models}个待测模型并推理所有样本...")
    # 存储所有待测模型结果: 形状 [num_test_models, total_samples]
    test_logits_all = []
    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        test_infer = YOLO11ROIInferencer(
            model_path=model_path, dataset_root=test_dataset_root,
            model_size=model_size, roi_size=roi_size, num_roi=num_roi, num_classes=num_classes
        )
        model_logits = []
        for idx in tqdm(sample_indices, desc=f"模型: {model_name}"):
            with torch.no_grad():
                roi_imgs = test_infer.preprocess_roi(idx)
                pred_logits = test_infer.model(roi_imgs)
            cls_logits = pred_logits[0, :, target_class].cpu().numpy()
            model_logits.append(np.mean(cls_logits))
        test_logits_all.append(model_logits)
    test_logits_all = np.array(test_logits_all)

    # ===================== 4. 核心计算：严格按新公式 =====================
    # 步骤1：所有待测模型 按样本求平均 → [total_samples,]
    test_mean_logits = np.mean(test_logits_all, axis=0)
    # 步骤2：计算平方差 (待测模型平均 - 最优模型平均)²
    squared_diff = (test_mean_logits - best_logits) ** 2
    # 步骤3：所有样本求平均 → 最终指标
    final_score = np.mean(squared_diff)

    # ===================== 5. 输出结果 =====================
    print("\n" + "=" * 60)
    print("📊 最终评估结果")
    print("=" * 60)
    print(f"待测模型数量: {num_test_models}")
    print(f"测试样本数量: {total_samples}")
    print(f"目标类别: {target_class}")
    print(f"最终平均平方误差: {final_score:.8f}")
    print("=" * 60)

    return final_score


def compute_model_vs_ensemble_mean_mse(
        model_paths: list,
        test_dataset_root: str,
        target_class: int = 1,
        model_size: str = "s",
        roi_size: int = 64,
        num_roi: int = 12,
        num_classes: int = 2
):
    # 1. 获取样本
    sample_indices = get_valid_sample_indices(test_dataset_root)
    total_samples = len(sample_indices)
    num_test_models = len(model_paths)

    print(f"✅ 测试集样本数: {total_samples}")
    print(f"✅ 待测模型数: {num_test_models}")
    print(f"✅ 计算类别: {target_class} (0=无方块,1=有方块)\n")

    # 2. 推理所有模型所有样本
    model_names = []
    test_logits_all = []
    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        model_names.append(model_name)

        test_infer = YOLO11ROIInferencer(
            model_path=model_path, dataset_root=test_dataset_root,
            model_size=model_size, roi_size=roi_size, num_roi=num_roi, num_classes=num_classes
        )

        model_logits = []
        for idx in tqdm(sample_indices, desc=f"推理: {model_name}"):
            with torch.no_grad():
                roi_imgs = test_infer.preprocess_roi(idx)
                pred_logits = test_infer.model(roi_imgs)

            cls_logits = pred_logits[0, :, target_class].cpu().numpy()
            model_logits.append(np.mean(cls_logits))

        test_logits_all.append(model_logits)

    test_logits_all = np.array(test_logits_all)
    # 模型集合均值 [N,]
    ensemble_mean = np.mean(test_logits_all, axis=0)

    # 3. 计算每个模型的MSE
    mse_list = []
    print("\n" + "=" * 60)
    print("📊 各模型与整体均值的MSE")
    print("=" * 60)
    for i, name in enumerate(model_names):
        diff_sq = (test_logits_all[i] - ensemble_mean) ** 2
        mse = np.mean(diff_sq)
        mse_list.append(mse)
        print(f"{name:38s} | MSE = {mse:.8f}")

    # 4. 对所有模型的MSE再求平均
    overall_mean_mse = np.mean(mse_list)

    print("\n" + "=" * 60)
    print(f"📌 所有模型MSE的平均值 = {overall_mean_mse:.8f}")
    print("=" * 60)

    return overall_mean_mse


# ===================== 【修改核心：多数据集批量测试】 =====================
if __name__ == "__main__":
    # 1. 配置参数（测试数据集 → 列表格式，可添加任意多个）
    TEST_MODEL_LIST = [
        r"H:\pycharm\yolov11\yolov11_proj3\yolo11Custom_pointsize\test\yolo11_pt\yolo11s_mini1022_1.pt",
        r"H:\pycharm\yolov11\yolov11_proj3\yolo11Custom_pointsize\test\yolo11_pt\yolo11s_mini1022_2.pt",
        r"H:\pycharm\yolov11\yolov11_proj3\yolo11Custom_pointsize\test\yolo11_pt\yolo11s_mini1022_3.pt",
        r"H:\pycharm\yolov11\yolov11_proj3\yolo11Custom_pointsize\test\yolo11_pt\yolo11s_mini1022_4.pt",
        r"H:\pycharm\yolov11\yolov11_proj3\yolo11Custom_pointsize\test\yolo11_pt\yolo11s_mini1022_5.pt",
        r"H:\pycharm\yolov11\yolov11_proj3\yolo11Custom_pointsize\test\yolo11_pt\yolo11s_mini1022_6.pt",
    ]
    BEST_MODEL = r"H:\pycharm\yolov11\yolov11_proj3\yolo11Custom_pointsize\yolo11_pt\yolo11s_roi12_ps_6.pt"

    # ✅ 测试数据集改为【列表】，支持同时传入多个数据集
    TEST_DATASET_LIST = [
        r"H:\pycharm\yolov11\yolov11_proj3\yolo11Custom_pointsize\test\mini_datasets_1022\datasets_1",
        r"H:\pycharm\yolov11\yolov11_proj3\yolo11Custom_pointsize\test\mini_datasets_1022\datasets_2",
        r"H:\pycharm\yolov11\yolov11_proj3\yolo11Custom_pointsize\test\mini_datasets_1022\datasets_3",
        # 继续添加更多数据集...
    ]

    # 2. 初始化结果列表（存储每个数据集的最终指标）
    ensemble_mse_list = []  # 对应：compute_ensemble_logits_mse 结果
    model_ensemble_mse_list = []  # 对应：compute_model_vs_ensemble_mean_mse 结果

    # 3. 批量遍历所有测试集，逐一生成计算
    for i, dataset_path in enumerate(TEST_DATASET_LIST, 1):
        print("\n" + "=" * 80)
        print(f"📂 开始处理 第{i}个测试数据集：{os.path.basename(dataset_path)}")
        print("=" * 80)

        # 计算第一个指标：待测模型集合平均 vs 最优模型
        final_mse = compute_ensemble_logits_mse(
            model_paths=TEST_MODEL_LIST,
            best_model_path=BEST_MODEL,
            test_dataset_root=dataset_path,
            target_class=1
        )
        ensemble_mse_list.append(final_mse)

        # 计算第二个指标：每个模型 vs 模型集合平均 的总平均
        model_avg_mse = compute_model_vs_ensemble_mean_mse(
            model_paths=TEST_MODEL_LIST,
            test_dataset_root=dataset_path,
            target_class=1
        )
        model_ensemble_mse_list.append(model_avg_mse)

    # 4. ✅ 最终汇总输出两个结果列表
    print("\n" + "=" * 80)
    print("📊 【所有数据集汇总结果】")
    print("=" * 80)
    print(f"1. 待测模型集合平均 vs 最优模型 MSE 列表：\n{ensemble_mse_list}")
    print(f"\n2. 所有模型MSE的平均值 列表：\n{model_ensemble_mse_list}")
    print("=" * 80)