import os
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple
# 导入你原有的推理器类
from infer import YOLO11ROIInferencer  # 替换为你的推理脚本文件名


class ModelComparisonEvaluator:
    """双模型对比评估器（基于YOLO11 ROI推理器）"""

    def __init__(
            self,
            model1_path: str,
            model2_path: str,
            dataset_root: str,
            model_size: str = "s",
            roi_size: int = 64,
            num_roi: int = 12,
            num_classes: int = 3,
            conf_threshold: float = 0.7,  # 高置信度阈值
            device: str = "cpu"
    ):
        """
        初始化对比评估器
        :param model1_path: 模型1权重路径（基准模型）
        :param model2_path: 模型2权重路径（对比模型）
        :param dataset_root: 数据集根目录
        :param model_size: 模型尺寸（n/s/l，需与训练一致）
        :param roi_size: ROI尺寸
        :param num_roi: ROI数量（固定12）
        :param num_classes: 分类数（固定3）
        :param conf_threshold: 高置信度阈值（默认0.7）
        :param device: 推理设备（cpu/cuda）
        """
        self.dataset_root = dataset_root
        self.conf_threshold = conf_threshold
        self.num_classes = num_classes
        self.device = torch.device(device) if isinstance(device, str) else device

        # 初始化两个模型的推理器
        print("=== 初始化模型1（基准模型） ===")
        self.inferencer1 = YOLO11ROIInferencer(
            model_path=model1_path,
            dataset_root=dataset_root,
            model_size=model_size,
            roi_size=roi_size,
            num_roi=num_roi,
            num_classes=num_classes
        )

        print("\n=== 初始化模型2（对比模型） ===")
        self.inferencer2 = YOLO11ROIInferencer(
            model_path=model2_path,
            dataset_root=dataset_root,
            model_size=model_size,
            roi_size=roi_size,
            num_roi=num_roi,
            num_classes=num_classes
        )

        # 初始化指标统计容器
        self.model1_metrics = self._init_metrics_dict()
        self.model2_metrics = self._init_metrics_dict()

        # 类别名称映射（便于输出）
        self.class_names = {0: "无效", 1: "无方块", 2: "有方块"}

    def _init_metrics_dict(self) -> Dict:
        """初始化空的指标字典"""
        return {
            # 基础准确率
            "total_valid_roi": 0,
            "total_correct_roi": 0,
            "overall_accuracy": 0.0,
            # 各类别准确率
            "class_accuracy": {i: {"correct": 0, "total": 0, "acc": 0.0} for i in range(self.num_classes)},
            # 置信度相关
            "total_conf_scores": [],  # 所有有效ROI的预测置信度
            "high_conf_count": 0,  # 高置信度（>阈值）的ROI数
            "high_conf_accuracy": 0.0,  # 高置信度样本的准确率
            "class_avg_conf": {i: [] for i in range(self.num_classes)},  # 各类别平均置信度
            # 错误分析
            "error_roi_count": 0,
            "error_conf_scores": [],  # 错误预测的置信度
            "correct_conf_scores": []  # 正确预测的置信度
        }

    def _compute_single_sample_metrics(
            self,
            raw_results: Dict,
            metrics_dict: Dict
    ) -> None:
        """
        计算单个样本的指标并更新到统计字典
        :param raw_results: 推理器返回的原始结果
        :param metrics_dict: 要更新的指标字典
        """
        valid_mask = raw_results["valid_mask"]
        gt_labels = raw_results["gt_labels"]
        pred_cls = raw_results["pred_cls"]
        pred_probs = raw_results["pred_probs"]

        # 遍历每个ROI（仅有效ROI参与计算）
        for i in range(len(valid_mask)):
            if not valid_mask[i]:
                continue  # 跳过无效ROI

            # 基础计数
            metrics_dict["total_valid_roi"] += 1
            gt = gt_labels[i]
            pred = pred_cls[i]
            conf = pred_probs[i][pred]  # 预测类别的置信度

            # 1. 整体准确率
            if pred == gt:
                metrics_dict["total_correct_roi"] += 1
                metrics_dict["correct_conf_scores"].append(conf)
            else:
                metrics_dict["error_roi_count"] += 1
                metrics_dict["error_conf_scores"].append(conf)

            # 2. 各类别准确率
            metrics_dict["class_accuracy"][gt]["total"] += 1
            if pred == gt:
                metrics_dict["class_accuracy"][gt]["correct"] += 1

            # 3. 置信度统计
            metrics_dict["total_conf_scores"].append(conf)
            # 高置信度计数
            if conf >= self.conf_threshold:
                metrics_dict["high_conf_count"] += 1
                # 高置信度准确率
                if pred == gt:
                    metrics_dict["class_accuracy"][gt]["high_conf_correct"] = metrics_dict["class_accuracy"][gt].get(
                        "high_conf_correct", 0) + 1

            # 4. 各类别平均置信度（按预测类别）
            metrics_dict["class_avg_conf"][pred].append(conf)

    def _finalize_metrics(self, metrics_dict: Dict) -> Dict:
        """
        最终计算所有指标（在所有样本推理完成后调用）
        :param metrics_dict: 原始统计字典
        :return: 计算完成的指标字典
        """
        # 1. 整体准确率
        if metrics_dict["total_valid_roi"] > 0:
            metrics_dict["overall_accuracy"] = (
                    metrics_dict["total_correct_roi"] / metrics_dict["total_valid_roi"]
            )
        else:
            metrics_dict["overall_accuracy"] = 0.0

        # 2. 各类别准确率
        for cls_id in metrics_dict["class_accuracy"]:
            cls_metrics = metrics_dict["class_accuracy"][cls_id]
            if cls_metrics["total"] > 0:
                cls_metrics["acc"] = cls_metrics["correct"] / cls_metrics["total"]
            else:
                cls_metrics["acc"] = 0.0

            # 各类别高置信度准确率
            high_conf_total = len([c for c in metrics_dict["total_conf_scores"]
                                   if c >= self.conf_threshold and
                                   metrics_dict["gt_labels"][np.where(metrics_dict["valid_mask"])[0]] == cls_id])
            if high_conf_total > 0:
                cls_metrics["high_conf_acc"] = (
                        cls_metrics.get("high_conf_correct", 0) / high_conf_total
                )
            else:
                cls_metrics["high_conf_acc"] = 0.0

        # 3. 置信度相关指标
        if metrics_dict["total_conf_scores"]:
            # 平均置信度
            metrics_dict["avg_conf"] = np.mean(metrics_dict["total_conf_scores"])
            # 高置信度占比
            metrics_dict["high_conf_ratio"] = (
                    metrics_dict["high_conf_count"] / len(metrics_dict["total_conf_scores"])
            )
            # 高置信度样本的整体准确率
            if metrics_dict["high_conf_count"] > 0:
                high_conf_correct = len([c for c in metrics_dict["correct_conf_scores"] if c >= self.conf_threshold])
                metrics_dict["high_conf_accuracy"] = high_conf_correct / metrics_dict["high_conf_count"]
            else:
                metrics_dict["high_conf_accuracy"] = 0.0

            # 错误/正确样本的平均置信度
            metrics_dict["avg_error_conf"] = np.mean(metrics_dict["error_conf_scores"]) if metrics_dict[
                "error_conf_scores"] else 0.0
            metrics_dict["avg_correct_conf"] = np.mean(metrics_dict["correct_conf_scores"]) if metrics_dict[
                "correct_conf_scores"] else 0.0
        else:
            metrics_dict["avg_conf"] = 0.0
            metrics_dict["high_conf_ratio"] = 0.0
            metrics_dict["high_conf_accuracy"] = 0.0
            metrics_dict["avg_error_conf"] = 0.0
            metrics_dict["avg_correct_conf"] = 0.0

        # 4. 各类别平均置信度
        for cls_id in metrics_dict["class_avg_conf"]:
            if metrics_dict["class_avg_conf"][cls_id]:
                metrics_dict["class_avg_conf"][cls_id] = np.mean(metrics_dict["class_avg_conf"][cls_id])
            else:
                metrics_dict["class_avg_conf"][cls_id] = 0.0

        return metrics_dict

    def evaluate(self, img_idx_list: List[int], save_path: str = "./model_comparison_results") -> Dict:
        """
        执行双模型对比评估
        :param img_idx_list: 要评估的样本索引列表
        :param save_path: 结果保存路径（文件夹）
        :return: 完整的对比结果字典
        """
        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)

        print(f"\n=== 开始对比评估 | 样本数：{len(img_idx_list)} | 高置信阈值：{self.conf_threshold} ===")

        # 遍历所有样本
        for idx, img_idx in enumerate(img_idx_list, 1):
            print(f"\n[{idx}/{len(img_idx_list)}] 评估样本 {img_idx}")

            try:
                # 模型1推理
                res1 = self.inferencer1.infer(img_idx)
                self._compute_single_sample_metrics(res1, self.model1_metrics)

                # 模型2推理
                res2 = self.inferencer2.infer(img_idx)
                self._compute_single_sample_metrics(res2, self.model2_metrics)

            except Exception as e:
                print(f"⚠️ 样本{img_idx}推理失败：{str(e)}")
                continue

        # 最终计算所有指标
        self.model1_metrics = self._finalize_metrics(self.model1_metrics)
        self.model2_metrics = self._finalize_metrics(self.model2_metrics)

        # 生成对比报告
        comparison_report = self._generate_comparison_report()

        # 保存结果
        self._save_results(comparison_report, save_path)

        # 打印对比报告
        self._print_comparison_report(comparison_report)

        return comparison_report

    def _generate_comparison_report(self) -> Dict:
        """生成对比报告字典"""
        report = {
            "config": {
                "dataset_root": self.dataset_root,
                "conf_threshold": self.conf_threshold,
                "num_samples": len(self.model1_metrics["total_conf_scores"]),
                "class_names": self.class_names
            },
            "model1": self.model1_metrics,
            "model2": self.model2_metrics,
            "comparison": {
                # 整体准确率对比
                "overall_accuracy_diff": self.model2_metrics["overall_accuracy"] - self.model1_metrics[
                    "overall_accuracy"],
                # 高置信占比对比
                "high_conf_ratio_diff": self.model2_metrics["high_conf_ratio"] - self.model1_metrics["high_conf_ratio"],
                # 高置信准确率对比
                "high_conf_accuracy_diff": self.model2_metrics["high_conf_accuracy"] - self.model1_metrics[
                    "high_conf_accuracy"],
                # 平均置信度对比
                "avg_conf_diff": self.model2_metrics["avg_conf"] - self.model1_metrics["avg_conf"],
                # 各类别准确率对比
                "class_accuracy_diff": {
                    cls_id: self.model2_metrics["class_accuracy"][cls_id]["acc"] -
                            self.model1_metrics["class_accuracy"][cls_id]["acc"]
                    for cls_id in range(self.num_classes)
                }
            }
        }
        return report

    def _print_comparison_report(self, report: Dict) -> None:
        """打印格式化的对比报告"""
        print("\n" + "=" * 100)
        print("📊 双模型对比评估报告")
        print("=" * 100)

        # 基础信息
        print(f"\n【基础配置】")
        print(f"数据集：{report['config']['dataset_root']}")
        print(f"评估样本数：{report['config']['num_samples']}")
        print(f"高置信度阈值：{report['config']['conf_threshold']}")

        # 整体指标对比
        print(f"\n【整体准确率】")
        print(f"模型1：{report['model1']['overall_accuracy']:.4f}")
        print(f"模型2：{report['model2']['overall_accuracy']:.4f}")
        diff = report['comparison']['overall_accuracy_diff']
        print(f"差异：{diff:+.4f} ({'提升' if diff > 0 else '下降' if diff < 0 else '持平'})")

        # 高置信度指标对比
        print(f"\n【高置信度指标（阈值={self.conf_threshold}）】")
        print(f"{'指标':<20} | {'模型1':<10} | {'模型2':<10} | {'差异':<10}")
        print(f"{'-' * 20}+{'-' * 10}+{'-' * 10}+{'-' * 10}")
        print(
            f"高置信占比       | {report['model1']['high_conf_ratio']:.4f} | {report['model2']['high_conf_ratio']:.4f} | {report['comparison']['high_conf_ratio_diff']:+.4f}")
        print(
            f"高置信准确率     | {report['model1']['high_conf_accuracy']:.4f} | {report['model2']['high_conf_accuracy']:.4f} | {report['comparison']['high_conf_accuracy_diff']:+.4f}")
        print(
            f"平均置信度       | {report['model1']['avg_conf']:.4f} | {report['model2']['avg_conf']:.4f} | {report['comparison']['avg_conf_diff']:+.4f}")

        # 各类别准确率对比
        print(f"\n【各类别准确率】")
        print(f"{'类别':<10} | {'模型1':<10} | {'模型2':<10} | {'差异':<10}")
        print(f"{'-' * 10}+{'-' * 10}+{'-' * 10}+{'-' * 10}")
        for cls_id in range(self.num_classes):
            cls_name = self.class_names[cls_id]
            acc1 = report['model1']['class_accuracy'][cls_id]['acc']
            acc2 = report['model2']['class_accuracy'][cls_id]['acc']
            diff = report['comparison']['class_accuracy_diff'][cls_id]
            print(f"{cls_name:<10} | {acc1:.4f} | {acc2:.4f} | {diff:+.4f}")

        # 错误分析
        print(f"\n【错误分析】")
        print(f"{'指标':<20} | {'模型1':<10} | {'模型2':<10}")
        print(f"{'-' * 20}+{'-' * 10}+{'-' * 10}")
        print(
            f"错误ROI数        | {report['model1']['error_roi_count']:<10} | {report['model2']['error_roi_count']:<10}")
        print(f"错误样本平均置信 | {report['model1']['avg_error_conf']:.4f} | {report['model2']['avg_error_conf']:.4f}")
        print(
            f"正确样本平均置信 | {report['model1']['avg_correct_conf']:.4f} | {report['model2']['avg_correct_conf']:.4f}")

    def _save_results(self, report: Dict, save_path: str) -> None:
        """保存对比结果到文件"""
        # 保存JSON完整报告
        json_path = os.path.join(save_path, "comparison_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # 保存CSV简洁对比表
        csv_data = []
        # 基础指标
        csv_data.append(["整体准确率", report['model1']['overall_accuracy'], report['model2']['overall_accuracy'],
                         report['comparison']['overall_accuracy_diff']])
        csv_data.append(["高置信占比", report['model1']['high_conf_ratio'], report['model2']['high_conf_ratio'],
                         report['comparison']['high_conf_ratio_diff']])
        csv_data.append(["高置信准确率", report['model1']['high_conf_accuracy'], report['model2']['high_conf_accuracy'],
                         report['comparison']['high_conf_accuracy_diff']])
        csv_data.append(["平均置信度", report['model1']['avg_conf'], report['model2']['avg_conf'],
                         report['comparison']['avg_conf_diff']])
        # 各类别指标
        for cls_id in range(self.num_classes):
            cls_name = self.class_names[cls_id]
            csv_data.append([
                f"{cls_name}准确率",
                report['model1']['class_accuracy'][cls_id]['acc'],
                report['model2']['class_accuracy'][cls_id]['acc'],
                report['comparison']['class_accuracy_diff'][cls_id]
            ])

        # 保存CSV
        csv_df = pd.DataFrame(csv_data, columns=["指标", "模型1", "模型2", "差异"])
        csv_path = os.path.join(save_path, "comparison_summary.csv")
        csv_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        print(f"\n✅ 对比结果已保存：")
        print(f"   - 完整报告：{json_path}")
        print(f"   - 简洁对比表：{csv_path}")


# ===================== 测试运行 =====================
if __name__ == "__main__":
    # 配置项
    MODEL1_PATH = r"H:\pycharm\yolov11\yolov11_proj1\yolo11_Custom_12roi\model_pt\yolo11s_roi_16334_new1.pt"  # 基准模型
    MODEL2_PATH = r"H:\pycharm\yolov11\yolov11_proj1\yolo11_Custom_12roi\model_pt\yolo11s_roi_new2.pt"  # 对比模型
    DATASET_ROOT = r"H:\pycharm\yolov11\yolov11_proj1\test_map50"  # 数据集根目录
    MODEL_SIZE = "s"
    CONF_THRESHOLD = 0.7  # 高置信度阈值
    SAVE_PATH = "./model_comparison_results"  # 结果保存路径

    # 要评估的样本索引列表（可自定义）
    # TEST_IDXS = [i for i in range(1, 100)]  # 前100个样本
    TEST_IDXS = [1, 2, 3, 4, 5]  # 测试少量样本

    # 初始化评估器
    evaluator = ModelComparisonEvaluator(
        model1_path=MODEL1_PATH,
        model2_path=MODEL2_PATH,
        dataset_root=DATASET_ROOT,
        model_size=MODEL_SIZE,
        conf_threshold=CONF_THRESHOLD,
        device="cpu"
    )

    # 执行评估
    comparison_results = evaluator.evaluate(
        img_idx_list=TEST_IDXS,
        save_path=SAVE_PATH
    )
