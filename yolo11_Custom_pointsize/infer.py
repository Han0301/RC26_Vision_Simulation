import os
import re
import json
import cv2
import numpy as np
import torch
from pathlib import Path
from torchvision import transforms
from model import YOLO11ROIClassifier  # 确保model.py适配二分类
from post_processing import post_process_linear_norm, post_process_sigmoid_balance, post_process_only_pred1, \
    _compute_confidence


class YOLO11ROIInferencer:
    """YOLO11推理器（二分类）- 多置信度阈值统计 + 文件夹直接推理"""

    def __init__(self, model_path, dataset_root=None, model_size="s", roi_size=64, num_roi=12, num_classes=2):
        """初始化推理器"""
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 核心参数
        self.model_size = model_size
        self.roi_size = roi_size
        self.num_roi = num_roi
        self.num_classes = num_classes
        self.gt_labels = None
        self.point_size = None

        # 数据集路径（原有逻辑，可选）
        self.dataset_root = dataset_root
        if dataset_root is not None:
            self.roi_img_root = os.path.join(dataset_root, "roi_images")
            self.label_dir = os.path.join(dataset_root, "labels")
            if not os.path.exists(self.label_dir):
                raise FileNotFoundError(f"❌ 标签文件夹不存在：{self.label_dir}")

        # 加载二分类模型
        self.model = YOLO11ROIClassifier(
            model_size=self.model_size,
            num_roi=self.num_roi,
            num_classes=self.num_classes,
            roi_size=self.roi_size
        )

        # 加载权重
        try:
            # checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ 加载checkpoint成功 | 最优F1: {checkpoint.get('best_pos_f1', 0.0):.4f}")
            else:
                self.model.load_state_dict(checkpoint)
                print(f"✅ 加载权重成功 | 路径: {model_path}")
        except Exception as e:
            print(f"❌ 加载模型失败：{e}")
            raise

        # 推理模式
        self.model.to(self.device)
        self.model.eval()

        # 归一化参数
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)

    def _extract_idx_from_filename(self, filename):
        """
        从文件名提取idx后的数字（适配格式：idx12cls0conf0.000000.png）
        :param filename: 文件名
        :return: idx后的数字（int），提取失败返回None
        """
        # 正则匹配：idx后紧跟1-2位数字（1-12）
        match = re.search(r'idx(\d{1,2})', filename, re.IGNORECASE)
        if match:
            idx_num = int(match.group(1))
            # 确保数字在1-12范围内
            if 1 <= idx_num <= 12:
                return idx_num
        return None

    def preprocess_roi_from_folder(self, roi_folder):
        """
        从指定文件夹预处理12个ROI（按idx数字1-12排序）
        :param roi_folder: ROI图片所在文件夹路径
        :return: 预处理后的ROI tensor [1,12,3,64,64]
        """
        # 校验文件夹是否存在
        if not os.path.exists(roi_folder):
            raise FileNotFoundError(f"❌ ROI文件夹不存在：{roi_folder}")

        # 步骤1：遍历文件夹，提取所有含idx的图片
        roi_file_dict = {}  # key: idx数字(1-12), value: 文件路径
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        for filename in os.listdir(roi_folder):
            # 筛选图片文件
            if not filename.lower().endswith(valid_extensions):
                continue
            # 提取idx数字
            idx_num = self._extract_idx_from_filename(filename)
            if idx_num is None:
                print(f"⚠️ 文件名{filename}无有效idx数字（需idx1-idx12），忽略该文件")
                continue
            # 去重：保留最后一个同名idx文件
            roi_file_dict[idx_num] = os.path.join(roi_folder, filename)

        # 步骤2：按1-12顺序加载ROI图片（缺失则用全黑图填充）
        roi_imgs = []
        roi_filenames = []  # 记录实际加载的文件名
        for idx_num in range(1, 13):
            if idx_num in roi_file_dict:
                roi_path = roi_file_dict[idx_num]
                roi_filename = os.path.basename(roi_path)
                # 读取图片（容错：无法读取则用全黑图替代）
                roi_img = cv2.imread(roi_path)
                if roi_img is None:
                    print(f"⚠️ ROI文件无法读取：{roi_path}，使用全黑图替代")
                    roi_img = np.zeros((self.roi_size, self.roi_size, 3), dtype=np.uint8)
                else:
                    # 格式转换（BGR→RGB）+ 尺寸调整
                    roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
                    roi_img = cv2.resize(roi_img, (self.roi_size, self.roi_size), interpolation=cv2.INTER_LINEAR)
                print(f"✅ 加载ROI{idx_num}：{roi_filename}")
            else:
                print(f"⚠️ 未找到idx{idx_num}对应的文件，使用全黑图替代")
                roi_img = np.zeros((self.roi_size, self.roi_size, 3), dtype=np.uint8)
                roi_filename = f"idx{idx_num}_missing.png"

            roi_imgs.append(roi_img)
            roi_filenames.append(roi_filename)

        # 步骤3：数据格式转换（与原有逻辑一致）
        roi_imgs = np.stack(roi_imgs, axis=0)  # [12, 64, 64, 3]
        roi_imgs = torch.from_numpy(roi_imgs).permute(0, 3, 1, 2).float() / 255.0  # [12,3,64,64]
        roi_imgs = roi_imgs.to(self.device)  # 部署到目标设备
        roi_imgs = (roi_imgs - self.mean) / self.std  # 归一化
        roi_imgs = roi_imgs.unsqueeze(0)  # 增加batch维度 → [1,12,3,64,64]

        print(f"\n✅ 文件夹ROI预处理完成 | 有效文件数：{len(roi_file_dict)} | 缺失数：{12 - len(roi_file_dict)}")
        return roi_imgs, roi_filenames

    def infer_from_folder(self, roi_folder,point_size=None):
        """
        文件夹直接推理模式：读取指定文件夹内的12张ROI图片并推理
        :param roi_folder: ROI图片所在文件夹路径
        :return: 推理结果字典
        """
        print(f"\n=== 开始文件夹推理 | 路径：{roi_folder} ===")

        # 预处理文件夹内的ROI
        roi_imgs, roi_filenames = self.preprocess_roi_from_folder(roi_folder)

        # 推理（禁用梯度计算）
        with torch.no_grad():
            if point_size is None:
                point_size = np.ones(12, dtype=np.float32)

            pred_logits = self.model(roi_imgs)  # [1,12,2]
            pred_cls = torch.argmax(pred_logits, dim=-1).squeeze(0)  # [12,]
            pred_probs = torch.softmax(pred_logits, dim=-1).squeeze(0)  # [12,2]

        # 转换为numpy数组
        pred_logits_np = pred_logits.squeeze(0).cpu().numpy()
        pred_probs_np = pred_probs.cpu().numpy()
        pred_cls_np = pred_cls.cpu().numpy()
        # 计算每个ROI的预测置信度
        pred_conf_np = np.array([pred_probs_np[i][pred_cls_np[i]] for i in range(12)])

        # 整理推理结果
        raw_results = {
            "roi_folder": roi_folder,
            "roi_filenames": roi_filenames,
            "pred_logits": pred_logits_np,
            "pred_probs": pred_probs_np,
            "pred_cls": pred_cls_np,
            "pred_conf": pred_conf_np,
            # 二分类标签映射
            "pred_label": ["无方块" if cls == 0 else "有方块" for cls in pred_cls_np]
        }

        # 打印格式化结果
        print(f"\n========== 文件夹推理结果（二分类） ==========")
        print("ROI序号 | 文件名                          | 预测类别 | 无方块概率 | 有方块概率 | 预测置信度 | 预测标签")
        print("-" * 100)
        for i in range(12):
            print(
                f"{i + 1:6d} | {roi_filenames[i]:30s} | {pred_cls_np[i]:8d} | {pred_probs_np[i][0]:10.4f} | {pred_probs_np[i][1]:10.4f} | {pred_conf_np[i]:10.4f} | {raw_results['pred_label'][i]}")

        # 保存推理结果到JSON文件
        save_path = os.path.join(roi_folder, "folder_infer_results.json")
        # 转换numpy数组为列表（便于序列化）
        save_data = {
            "roi_folder": roi_folder,
            "roi_filenames": roi_filenames,
            "pred_logits": pred_logits_np.tolist(),
            "pred_probs": pred_probs_np.tolist(),
            "pred_cls": pred_cls_np.tolist(),
            "pred_conf": pred_conf_np.tolist(),
            "pred_label": raw_results["pred_label"]
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 推理结果已保存至：{save_path}")

        return raw_results

    def preprocess_roi(self, img_idx):
        """原有逻辑：从数据集根目录预处理ROI"""
        if self.dataset_root is None:
            raise ValueError("dataset_root未初始化，无法使用原有数据集推理！")

        roi_dir = os.path.join(self.roi_img_root, f"roi_{img_idx}")
        roi_imgs = []
        for roi_pos in range(1, 13):
            roi_path = os.path.join(roi_dir, f"{roi_pos}.png")
            if not os.path.exists(roi_path):
                print(f"⚠️ ROI文件缺失：{roi_path}，使用全黑图替代")
                roi_img = np.zeros((self.roi_size, self.roi_size, 3), dtype=np.uint8)
            else:
                roi_img = cv2.imread(roi_path)
                roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
                roi_img = cv2.resize(roi_img, (self.roi_size, self.roi_size), interpolation=cv2.INTER_LINEAR)
            roi_imgs.append(roi_img)

        # 格式转换
        roi_imgs = np.stack(roi_imgs, axis=0)
        roi_imgs = torch.from_numpy(roi_imgs).permute(0, 3, 1, 2).float() / 255.0
        roi_imgs = roi_imgs.to(self.device)
        roi_imgs = (roi_imgs - self.mean) / self.std
        roi_imgs = roi_imgs.unsqueeze(0)

        # 加载标签（全量使用，不过滤valid_mask）
        label_path = os.path.join(self.label_dir, f"label_{img_idx}.json")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"❌ 标签文件缺失：{label_path}")
        with open(label_path, "r", encoding="utf-8") as f:
            ann = json.load(f)
        self.gt_labels = ann.get("labels", [0] * 12)  # 直接使用所有标签
        self.point_size = torch.from_numpy(np.array(ann["point_size"], dtype=np.float32)).to(self.device)
        return roi_imgs

    def infer(self, img_idx):
        """原有逻辑：单样本推理"""
        roi_imgs = self.preprocess_roi(img_idx)

        with torch.no_grad():
            point_size = _compute_confidence(self.point_size,self.device)
            pred_logits = self.model(roi_imgs)
            # 测试方法1
            pred_cls, weighted_score = post_process_only_pred1(pred_logits, point_size)

            # 测试方法2
            # pred_cls, weighted_score = post_process_linear_norm(pred_logits, point_size)

            # 测试方法3（推荐）
            # pred_cls, weighted_score = post_process_sigmoid_balance(pred_logits, point_size)

            pred_probs = torch.softmax(pred_logits, dim=-1).squeeze(0)

            # ===================== 【直接用返回的得分计算排名】 =====================
            weighted_score_np = weighted_score.cpu().numpy()
            point_size_np = point_size.cpu().numpy()
            sorted_indices = np.argsort(weighted_score_np)[::-1]
            roi_rank = np.zeros(12, dtype=int)
            for rank, idx in enumerate(sorted_indices):
                roi_rank[idx] = rank + 1


        # 转换为numpy
        pred_logits_np = pred_logits.squeeze(0).cpu().numpy()
        pred_probs_np = pred_probs.cpu().numpy()
        pred_cls_np = pred_cls.cpu().numpy()
        gt_labels_np = np.array(self.gt_labels)
        pred_conf_np = np.array([pred_probs_np[i][pred_cls_np[i]] for i in range(12)])

        # 返回结果
        raw_results = {
            "pred_logits": pred_logits_np,
            "pred_probs": pred_probs_np,
            "pred_cls": pred_cls_np,
            "gt_labels": gt_labels_np,
            "pred_conf": pred_conf_np,
            "point_size": point_size_np,  # 新增
            "weighted_score": weighted_score_np,  # 新增
            "rank": roi_rank  # 新增
        }

        # 打印结果
        # ===================== 【修改：打印表头 + 新增列】 =====================
        print(f"\n========== 样本 {img_idx} 推理结果 ==========")
        print("ROI序号 | 真实标签  | 预测类别   | 无方块概率  | 有方块概率  | 预测置信度  | point_size   | 加权得分   | 排名")
        print("-" * 110)
        for i in range(12):
            print(
                f"{i + 1:6d} | {gt_labels_np[i]:8d} | {pred_cls_np[i]:8d} | {pred_probs_np[i][0]:10.4f} | {pred_probs_np[i][1]:10.4f} | {pred_conf_np[i]:10.4f} | {point_size_np[i]:9.4f} | {weighted_score_np[i]:8.4f} | {roi_rank[i]:4d}")
        # ======================================================================
        return raw_results

    def batch_infer_with_multi_thresholds(self, img_idx_list, conf_thresholds=None,
                                          error_log_path="infer_error_log.txt"):
        """原有逻辑：多阈值批量推理 + 新增分类准确率统计 + 新增分类高置信占总预测数比重"""
        # 校验阈值合法性
        if conf_thresholds is None:
            conf_thresholds = [0.6, 0.7, 0.8]
        for th in conf_thresholds:
            if not (0.0 <= th <= 1.0):
                raise ValueError(f"置信度阈值必须在0-1之间！当前值：{th}")

        # 初始化全局统计变量
        total_roi = 0  # 总ROI数
        total_correct = 0  # 总正确数
        error_records = []  # 错误记录

        # ========== 核心新增1：全局预测类别总数（不管置信度） ==========
        pred_cls0_total = 0  # 所有预测为0类的ROI总数
        pred_cls1_total = 0  # 所有预测为1类的ROI总数

        # ========== 核心修改1：扩展阈值统计结构，增加分类统计 ==========
        # 初始化多阈值统计字典（每个阈值对应一套指标）
        threshold_stats = {}
        for th in conf_thresholds:
            threshold_stats[th] = {
                "high_conf_roi": 0,  # 该阈值下高置信度ROI数
                "high_conf_correct": 0,  # 该阈值下高置信度且正确的ROI数
                "high_conf_ratio": 0.0,  # 该阈值下高置信度占比（占总ROI）
                "high_conf_accuracy": 0.0,  # 该阈值下高置信度准确率
                # 新增：按预测类别拆分的高置信度统计
                "cls0": {  # 预测为0类（无方块）
                    "high_conf_roi": 0,  # 预测0类且高置信的ROI数
                    "high_conf_correct": 0,  # 预测0类且高置信且正确的ROI数
                    "high_conf_accuracy": 0.0,  # 预测0类且高置信的准确率
                    "high_conf_ratio_of_total_cls0": 0.0  # 新增：高置信0类数占所有预测0类数的比重
                },
                "cls1": {  # 预测为1类（有方块）
                    "high_conf_roi": 0,  # 预测1类且高置信的ROI数
                    "high_conf_correct": 0,  # 预测1类且高置信且正确的ROI数
                    "high_conf_accuracy": 0.0,  # 预测1类且高置信的准确率
                    "high_conf_ratio_of_total_cls1": 0.0  # 新增：高置信1类数占所有预测1类数的比重
                }
            }

        # 清空日志
        with open(error_log_path, "w", encoding="utf-8") as f:
            f.write("=== YOLO11 ROI推理日志（多置信度阈值）===\n")
            f.write(f"置信度阈值列表：{conf_thresholds}\n")
            f.write(f"推理时间：{os.popen('date /t').read().strip()} {os.popen('time /t').read().strip()}\n")
            f.write("=" * 70 + "\n\n")

        # 遍历推理每个样本
        for img_idx in img_idx_list:
            print(f"\n=== 推理样本 {img_idx} ===")
            try:
                raw_results = self.infer(img_idx)

                # 遍历该样本的所有12个ROI
                for i in range(12):
                    total_roi += 1
                    gt_label = raw_results["gt_labels"][i]
                    pred_cls = raw_results["pred_cls"][i]
                    pred_conf = raw_results["pred_conf"][i]

                    # ========== 核心新增2：统计全局预测类别总数 ==========
                    if pred_cls == 0:
                        pred_cls0_total += 1
                    elif pred_cls == 1:
                        pred_cls1_total += 1

                    # 判断是否正确
                    is_correct = (pred_cls == gt_label)

                    if is_correct:
                        total_correct += 1

                    # ========== 核心修改2：更新分类高置信度统计 ==========
                    # 对每个阈值单独统计高置信度指标
                    for th in conf_thresholds:
                        if pred_conf >= th:
                            # 更新整体高置信统计
                            threshold_stats[th]["high_conf_roi"] += 1
                            if is_correct:
                                threshold_stats[th]["high_conf_correct"] += 1

                            # 更新按预测类别拆分的高置信统计
                            if pred_cls == 0:
                                # 预测为0类（无方块）
                                threshold_stats[th]["cls0"]["high_conf_roi"] += 1
                                if is_correct:
                                    threshold_stats[th]["cls0"]["high_conf_correct"] += 1
                            elif pred_cls == 1:
                                # 预测为1类（有方块）
                                threshold_stats[th]["cls1"]["high_conf_roi"] += 1
                                if is_correct:
                                    threshold_stats[th]["cls1"]["high_conf_correct"] += 1

                    # 记录错误ROI
                    if not is_correct:
                        error_info = {
                            "batch_idx": img_idx,
                            "roi_position": i + 1,
                            "gt_label": int(gt_label),
                            "pred_cls": int(pred_cls),
                            "pred_conf": float(pred_conf),
                            "pred_probs": raw_results["pred_probs"][i].tolist()
                        }
                        error_records.append(error_info)

            except Exception as e:
                error_msg = f"❌ 样本{img_idx}推理失败：{str(e)}"
                print(error_msg)
                with open(error_log_path, "a", encoding="utf-8") as f:
                    f.write(f"【样本 {img_idx}】推理失败：{str(e)}\n\n")

        # ========== 核心修改3：计算分类准确率 + 新增分类高置信占总预测数比重 ==========
        # 计算每个阈值的占比和准确率
        for th in conf_thresholds:
            stats = threshold_stats[th]
            # 整体指标计算
            stats["high_conf_ratio"] = stats["high_conf_roi"] / total_roi if total_roi > 0 else 0.0
            stats["high_conf_accuracy"] = stats["high_conf_correct"] / stats["high_conf_roi"] if stats[
                                                                                                     "high_conf_roi"] > 0 else 0.0

            # 分类指标计算（0类）
            cls0_stats = stats["cls0"]
            cls0_stats["high_conf_accuracy"] = cls0_stats["high_conf_correct"] / cls0_stats["high_conf_roi"] if \
                cls0_stats["high_conf_roi"] > 0 else 0.0
            # 新增：0类高置信数占所有预测0类数的比重
            cls0_stats["high_conf_ratio_of_total_cls0"] = cls0_stats[
                                                              "high_conf_roi"] / pred_cls0_total if pred_cls0_total > 0 else 0.0

            # 分类指标计算（1类）
            cls1_stats = stats["cls1"]
            cls1_stats["high_conf_accuracy"] = cls1_stats["high_conf_correct"] / cls1_stats["high_conf_roi"] if \
                cls1_stats["high_conf_roi"] > 0 else 0.0
            # 新增：1类高置信数占所有预测1类数的比重
            cls1_stats["high_conf_ratio_of_total_cls1"] = cls1_stats[
                                                              "high_conf_roi"] / pred_cls1_total if pred_cls1_total > 0 else 0.0

        # 写入日志
        with open(error_log_path, "a", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("=== 全局统计结果 ===\n")
            f.write(f"总推理样本数：{len(img_idx_list)}\n")
            f.write(f"总ROI数：{total_roi}\n")
            f.write(f"总正确数：{total_correct}\n")
            f.write(f"整体准确率：{total_correct / total_roi:.4f} ({total_correct}/{total_roi})\n")
            f.write(f"错误ROI总数：{len(error_records)}\n")
            # ========== 核心新增3：日志中添加全局预测类别总数 ==========
            f.write(f"全局预测0类（无方块）总数：{pred_cls0_total}\n")
            f.write(f"全局预测1类（有方块）总数：{pred_cls1_total}\n\n")

            # ========== 核心修改4：日志中添加分类统计 ==========
            f.write("=== 各置信度阈值统计结果 ===\n")
            for th in conf_thresholds:
                stats = threshold_stats[th]
                cls0_stats = stats["cls0"]
                cls1_stats = stats["cls1"]
                f.write(f"\n【阈值 {th}】\n")
                # 整体指标
                f.write(f"  ├─ 整体高置信指标 ───────────────────\n")
                f.write(f"  │ 高置信度ROI数：{stats['high_conf_roi']}\n")
                f.write(f"  │ 高置信度正确数：{stats['high_conf_correct']}\n")
                f.write(
                    f"  │ 高置信度占比（总ROI）：{stats['high_conf_ratio']:.4f} ({stats['high_conf_roi']}/{total_roi})\n")
                f.write(
                    f"  │ 高置信度准确率：{stats['high_conf_accuracy']:.4f} ({stats['high_conf_correct']}/{stats['high_conf_roi']})\n")
                # 0类指标
                f.write(f"  ├─ 预测为0类（无方块）高置信指标 ──────\n")
                f.write(f"  │ 高置信度ROI数：{cls0_stats['high_conf_roi']}\n")
                f.write(f"  │ 高置信度正确数：{cls0_stats['high_conf_correct']}\n")
                f.write(
                    f"  │ 高置信度准确率：{cls0_stats['high_conf_accuracy']:.4f} ({cls0_stats['high_conf_correct']}/{cls0_stats['high_conf_roi']})\n")
                # 新增：0类高置信占其总预测数的比重
                f.write(
                    f"  │ 高置信占总预测0类比重：{cls0_stats['high_conf_ratio_of_total_cls0']:.4f} ({cls0_stats['high_conf_roi']}/{pred_cls0_total})\n")
                # 1类指标
                f.write(f"  ├─ 预测为1类（有方块）高置信指标 ──────\n")
                f.write(f"  │ 高置信度ROI数：{cls1_stats['high_conf_roi']}\n")
                f.write(f"  │ 高置信度正确数：{cls1_stats['high_conf_correct']}\n")
                f.write(
                    f"  │ 高置信度准确率：{cls1_stats['high_conf_accuracy']:.4f} ({cls1_stats['high_conf_correct']}/{cls1_stats['high_conf_roi']})\n")
                # 新增：1类高置信占其总预测数的比重
                f.write(
                    f"  │ 高置信占总预测1类比重：{cls1_stats['high_conf_ratio_of_total_cls1']:.4f} ({cls1_stats['high_conf_roi']}/{pred_cls1_total})\n")

        # 控制台打印汇总结果
        print(f"\n" + "=" * 80)
        print(f"=== 批量推理多阈值统计汇总 ===")
        print(f"├─ 全局指标 ──────────────────────────────────────────────────────")
        print(f"│ 总样本数：{len(img_idx_list)} | 总ROI数：{total_roi} | 总正确数：{total_correct}")
        print(f"│ 整体准确率：{total_correct / total_roi:.4f} ({total_correct}/{total_roi})")
        print(f"│ 全局预测0类总数：{pred_cls0_total} | 全局预测1类总数：{pred_cls1_total}")
        print(f"├─ 错误ROI总数：{len(error_records)}")

        # ========== 核心修改5：控制台打印分类统计 ==========
        print(f"├─ 各阈值高置信度整体指标对比 ────────────────────────────────────")
        print(f"│ 阈值   | 高置信ROI数   | 高置信正确数   | 高置信占比    | 高置信准确率")
        print(f"│--------|-------------|--------------|-------------|-------------")
        for th in conf_thresholds:
            stats = threshold_stats[th]
            print(
                f"│ {th:6.2f} | {stats['high_conf_roi']:11d} | {stats['high_conf_correct']:12d} | {stats['high_conf_ratio']:11.4f} | {stats['high_conf_accuracy']:11.4f}")

        # 新增：打印按预测类别拆分的高置信度准确率 + 占总预测数比重
        print(f"├─ 各阈值高置信度分类指标对比 ────────────────────────────────────")
        print(f"│ 阈值   | 0类高置信数 | 0类准确率       | 0类高置信占比   |  1类高置信数  | 1类准确率    | 1类高置信占比 ")
        print(f"│--------|-------------|-------------|---------------|-------------|-------------|---------------")
        for th in conf_thresholds:
            stats = threshold_stats[th]
            cls0 = stats["cls0"]
            cls1 = stats["cls1"]
            print(
                f"│ {th:6.2f} | {cls0['high_conf_roi']:11d} | {cls0['high_conf_accuracy']:11.4f} | {cls0['high_conf_ratio_of_total_cls0']:13.4f} | {cls1['high_conf_roi']:11d} | {cls1['high_conf_accuracy']:11.4f} | {cls1['high_conf_ratio_of_total_cls1']:13.4f}")
        print(f"└─────────────────────────────────────────────────────────────────")
        print(f"\n✅ 错误日志已保存至：{error_log_path}")

        # 整理返回结果
        final_stats = {
            "global": {
                "total_samples": len(img_idx_list),
                "total_roi": total_roi,
                "total_correct": total_correct,
                "overall_accuracy": total_correct / total_roi if total_roi > 0 else 0.0,
                "error_roi_count": len(error_records),
                "pred_cls0_total": pred_cls0_total,  # 新增：全局预测0类总数
                "pred_cls1_total": pred_cls1_total  # 新增：全局预测1类总数
            },
            "thresholds": threshold_stats,
            "conf_thresholds": conf_thresholds
        }

        return final_stats


# 测试入口
if __name__ == "__main__":
    # 核心配置
    MODEL_PATH = r"H:\pycharm\yolov11\yolov11_proj3\yolo11Custom_pointsize\yolo11_pt\yolo11s_roi12_ps_5.pt"
    MODEL_SIZE = "s"
    ROI_SIZE = 64

    # ===================== 模式选择 =====================
    RUN_FOLDER_INFER = False  # 文件夹直接推理模式
    RUN_BATCH_INFER = True  # 原有批量多阈值推理模式

    # 初始化推理器（无需dataset_root，仅加载模型）
    inferencer = YOLO11ROIInferencer(
        model_path=MODEL_PATH,
        dataset_root=None,  # 文件夹模式无需数据集根目录
        model_size=MODEL_SIZE,
        roi_size=ROI_SIZE,
        num_roi=12,
        num_classes=2
    )
    print("\n✅ 推理器初始化成功！")

    # ===================== 1. 文件夹直接推理模式（核心新增） =====================
    if RUN_FOLDER_INFER:
        # 配置：需要推理的ROI文件夹路径
        ROI_FOLDER_PATH = r"H:\pycharm\yolov11\yolov11_proj1\real_tests\roi_3"  # 替换为你的ROI文件夹路径
        # 执行文件夹推理
        folder_results = inferencer.infer_from_folder(ROI_FOLDER_PATH)
        print(f"\n=== 文件夹推理完成 ===")
        # 打印关键结果
        print(f"推理文件夹：{folder_results['roi_folder']}")
        print(f"预测结果汇总：")
        for i in range(12):
            print(f"ROI{i + 1}：{folder_results['pred_label'][i]}（置信度：{folder_results['pred_conf'][i]:.4f}）")

    # ===================== 2. 原有批量多阈值推理模式（可选） =====================
    if RUN_BATCH_INFER:
        CONF_THRESHOLDS = [0.6,0.65,0.7, 0.75, 0.80, 0.85, 0.90, 0.95]
        DATASET_ROOT = r"H:\pycharm\yolov11\yolov11_proj1\datasets_global_test100"
        label_dir = os.path.join(DATASET_ROOT, "labels")
        valid_idxs = []

        # 正则匹配 label_数字.json
        pattern = re.compile(r"label_(\d+)\.json")
        for filename in os.listdir(label_dir):
            match = pattern.match(filename)
            if match:
                img_idx = int(match.group(1))
                valid_idxs.append(img_idx)
        # 排序并生成批量推理列表
        BATCH_TEST_IDXS = sorted(valid_idxs)
        print(f"\n✅ 自动扫描有效样本总数：{len(BATCH_TEST_IDXS)}")

        # 重新初始化推理器（需要dataset_root）
        inferencer_batch = YOLO11ROIInferencer(
            model_path=MODEL_PATH,
            dataset_root=DATASET_ROOT,
            model_size=MODEL_SIZE,
            roi_size=ROI_SIZE,
            num_roi=12,
            num_classes=2
        )

        # 执行批量推理
        batch_stats = inferencer_batch.batch_infer_with_multi_thresholds(
            img_idx_list=BATCH_TEST_IDXS,
            conf_thresholds=CONF_THRESHOLDS,
            error_log_path="./error/infer_error_log_multi_threshold.txt"
        )
        # 保存结果
        save_path = "./error/batch_infer_multi_threshold_stats.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(batch_stats, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 批量推理结果已保存至：{save_path}")