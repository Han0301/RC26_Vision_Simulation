import os
import re
import json
import cv2
import numpy as np
import torch
from pathlib import Path
from torchvision import transforms
from model import YOLO11ROIClassifier  # 确保model.py已更新至支持n/s/l的版本


class YOLO11ROIInferencer:
    """YOLO11风格推理器（仅ROI输入+离线适配+支持n/s/l模型）"""

    def __init__(self, model_path, dataset_root=None, model_size="s", roi_size=64, num_roi=12, num_classes=3):
        """
        初始化推理器
        :param model_path: 训练好的模型权重路径
        :param dataset_root: 原数据集根目录（可选，仅用于原有数据集推理）
        :param model_size: 模型尺寸（n/s/l，必须与训练时一致）
        :param roi_size: ROI图像尺寸（默认64）
        :param num_roi: ROI数量（默认12）
        :param num_classes: 分类数（默认3）
        """
        # 设备配置（优先CPU适配离线环境，避免CUDA依赖）
        self.device = torch.device("cpu")
        # 如需使用CUDA，取消注释下行（需确保模型训练/推理设备兼容）
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 核心参数（必须与训练时完全一致）
        self.model_size = model_size  # 新增：模型尺寸（关键）
        self.roi_size = roi_size
        self.num_roi = num_roi
        self.num_classes = num_classes

        # 原有数据集路径配置（可选）
        self.dataset_root = dataset_root
        if dataset_root is not None:
            self.roi_img_root = os.path.join(dataset_root, "roi_images")
            self.label_dir = os.path.join(dataset_root, "labels")
            # 校验labels文件夹是否存在
            if not os.path.exists(self.label_dir):
                raise FileNotFoundError(f"❌ 标签文件夹不存在：{self.label_dir}")

        # 加载YOLO11模型（修复：删除pretrained_path，添加model_size）
        self.model = YOLO11ROIClassifier(
            model_size=self.model_size,  # 必须指定，与训练时一致
            num_roi=self.num_roi,
            num_classes=self.num_classes,
            roi_size=self.roi_size
        )

        # 加载训练好的模型权重（增强容错+设备兼容）
        try:
            # 强制加载到指定设备，避免CUDA/CPU冲突
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            # 兼容两种权重格式：完整checkpoint / 仅state_dict
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ 加载完整checkpoint成功 | 最优F1: {checkpoint.get('best_pos_f1', 0.0):.4f}")
            else:
                self.model.load_state_dict(checkpoint)
                print(f"✅ 加载模型权重成功 | 路径: {model_path}")
        except FileNotFoundError:
            print(f"❌ 模型文件不存在：{model_path}")
            raise
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print(f"❌ 模型尺寸不匹配！请确认model_size={self.model_size}与训练时一致")
                raise
            else:
                print(f"❌ 加载模型权重失败：{e}")
                raise
        except Exception as e:
            print(f"⚠️ 加载模型异常：{e}")
            raise

        # 模型部署到指定设备+推理模式
        self.model.to(self.device)
        self.model.eval()  # 必须切换到推理模式（禁用Dropout/BatchNorm）

        # 归一化参数（确保和模型在同一设备）
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)

    def preprocess_roi(self, img_idx):
        """YOLO11标准预处理12个ROI（原有逻辑：基于数据集根目录）"""
        if self.dataset_root is None:
            raise ValueError("dataset_root未初始化，无法使用原有数据集推理！")

        # 加载12个ROI图像
        roi_dir = os.path.join(self.roi_img_root, f"roi_{img_idx}")
        roi_imgs = []
        for roi_pos in range(1, 13):
            roi_path = os.path.join(roi_dir, f"{roi_pos}.png")
            # 容错：ROI文件不存在时用全黑图填充
            if not os.path.exists(roi_path):
                print(f"⚠️ ROI文件缺失：{roi_path}，使用全黑图替代")
                roi_img = np.zeros((self.roi_size, self.roi_size, 3), dtype=np.uint8)
            else:
                # 读取+格式转换+尺寸调整（YOLO11标准预处理）
                roi_img = cv2.imread(roi_path)
                roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)  # BGR→RGB
                roi_img = cv2.resize(roi_img, (self.roi_size, self.roi_size), interpolation=cv2.INTER_LINEAR)
            roi_imgs.append(roi_img)

        # 数据格式转换（numpy→tensor+维度调整+归一化）
        roi_imgs = np.stack(roi_imgs, axis=0)  # [12, 64, 64, 3]
        roi_imgs = torch.from_numpy(roi_imgs).permute(0, 3, 1, 2).float() / 255.0  # [12,3,64,64]
        roi_imgs = roi_imgs.to(self.device)  # 部署到目标设备
        roi_imgs = (roi_imgs - self.mean) / self.std  # 归一化（和训练一致）
        roi_imgs = roi_imgs.unsqueeze(0)  # 增加batch维度 → [1,12,3,64,64]

        # 加载ROI有效掩码（valid_mask）
        label_path = os.path.join(self.label_dir, f"label_{img_idx}.json")  # 修正：label{idx}.json（无下划线）
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"❌ 标签文件缺失：{label_path}")
        else:
            with open(label_path, "r", encoding="utf-8") as f:
                ann = json.load(f)
            # 校验字段是否存在
            if "roi_valid_mask" not in ann:
                raise KeyError(f"❌ 标签文件{label_path}缺少roi_valid_mask字段")
            self.roi_valid_mask = torch.tensor(ann["roi_valid_mask"], dtype=torch.bool, device=self.device)
            # 保存labels字段（用于后续指标计算）
            self.gt_labels = ann.get("labels", [0] * 12)

        return roi_imgs

    def _extract_idx_number(self, filename):
        """
        从文件名中提取idx后的数字（核心：适配任意idx+数字的命名）
        :param filename: 文件名（如idx1.png、idx2_test.png、idx3_abc.jpg）
        :return: 提取的数字（int），提取失败返回None
        """
        # 正则匹配：匹配idx后紧跟的数字（忽略大小写）
        match = re.search(r'idx(\d+)', filename, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    def preprocess_custom_roi(self, custom_roi_root, custom_valid_mask):
        """
        新增：预处理自定义目录下的ROI图片（按idx后的数字排序）
        :param custom_roi_root: 自定义ROI图片根目录（文件名含idx+数字，如idx1.png、idx2_test.png）
        :param custom_valid_mask: 自定义有效掩码（12个int的列表，0=无效，1=有效）
        :return: 预处理后的ROI tensor [1,12,3,64,64]
        """
        # 校验参数合法性
        if len(custom_valid_mask) != 12:
            raise ValueError(f"custom_valid_mask长度必须为12！当前长度：{len(custom_valid_mask)}")
        if not all([x in [0, 1] for x in custom_valid_mask]):
            raise ValueError("custom_valid_mask只能包含0或1！")

        # 步骤1：遍历目录，提取所有含idx+数字的图片文件
        roi_file_dict = {}  # key: 数字(1-12), value: 文件路径
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')  # 支持的图片格式
        for filename in os.listdir(custom_roi_root):
            # 筛选图片文件
            if not filename.lower().endswith(valid_extensions):
                continue
            # 提取idx后的数字
            idx_num = self._extract_idx_number(filename)
            if idx_num is None or idx_num < 1 or idx_num > 12:
                print(f"⚠️ 文件名{filename}无有效idx数字（需1-12），忽略该文件")
                continue
            # 去重：如果有多个同数字文件，保留最后一个
            roi_file_dict[idx_num] = os.path.join(custom_roi_root, filename)

        # 步骤2：按1-12顺序加载ROI图片（缺失则用全黑图）
        roi_imgs = []
        roi_filenames = []  # 记录实际加载的文件名
        for roi_pos in range(1, 13):
            if roi_pos in roi_file_dict:
                roi_path = roi_file_dict[roi_pos]
                roi_filename = os.path.basename(roi_path)
                # 容错：文件无法读取时用全黑图
                roi_img = cv2.imread(roi_path)
                if roi_img is None:
                    print(f"⚠️ 自定义ROI文件无法读取：{roi_path}，使用全黑图替代")
                    roi_img = np.zeros((self.roi_size, self.roi_size, 3), dtype=np.uint8)
                else:
                    roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)  # BGR→RGB
                    roi_img = cv2.resize(roi_img, (self.roi_size, self.roi_size), interpolation=cv2.INTER_LINEAR)
                print(f"✅ 加载ROI{roi_pos}：{roi_filename}")
            else:
                print(f"⚠️ 未找到ROI{roi_pos}对应的文件（idx{roi_pos}），使用全黑图替代")
                roi_img = np.zeros((self.roi_size, self.roi_size, 3), dtype=np.uint8)
                roi_filename = f"idx{roi_pos}_missing.png"

            roi_imgs.append(roi_img)
            roi_filenames.append(roi_filename)

        # 步骤3：数据格式转换（与原有逻辑一致）
        roi_imgs = np.stack(roi_imgs, axis=0)  # [12, 64, 64, 3]
        roi_imgs = torch.from_numpy(roi_imgs).permute(0, 3, 1, 2).float() / 255.0  # [12,3,64,64]
        roi_imgs = roi_imgs.to(self.device)  # 部署到目标设备
        roi_imgs = (roi_imgs - self.mean) / self.std  # 归一化（和训练一致）
        roi_imgs = roi_imgs.unsqueeze(0)  # 增加batch维度 → [1,12,3,64,64]

        # 步骤4：转换自定义valid_mask为tensor（int→bool）
        self.roi_valid_mask = torch.tensor(custom_valid_mask, dtype=torch.bool, device=self.device)
        self.roi_filenames = roi_filenames  # 保存文件名，用于推理结果返回
        print(f"\n✅ 自定义valid_mask加载完成：{custom_valid_mask}")
        print(f"✅ 共加载{len(roi_file_dict)}个有效ROI文件，缺失{12 - len(roi_file_dict)}个")

        return roi_imgs

    def infer(self, img_idx):
        """YOLO11风格推理（原有逻辑：基于数据集根目录）"""
        # 预处理ROI（会加载gt_labels和roi_valid_mask）
        roi_imgs = self.preprocess_roi(img_idx)

        # 推理（禁用梯度计算，节省内存）
        with torch.no_grad():
            pred_logits = self.model(roi_imgs)  # [1,12,3]
            pred_cls = torch.argmax(pred_logits, dim=-1).squeeze(0)  # [12,]

        # YOLO11后处理：无效ROI强制设为0类
        pred_cls[~self.roi_valid_mask] = 0

        # 计算置信度（softmax转概率）
        pred_probs = torch.softmax(pred_logits, dim=-1).squeeze(0)  # [12,3]
        max_probs = pred_probs.max(dim=-1)[0]  # [12,]

        # 整理结构化结果
        results = []
        class_names = {0: "无效ROI", 1: "有效无方块", 2: "有效有方块"}
        for i in range(self.num_roi):
            results.append({
                "roi_position": i + 1,
                "is_valid_roi": self.roi_valid_mask[i].item(),
                "predicted_class_id": pred_cls[i].item(),
                "predicted_class_name": class_names[pred_cls[i].item()],
                "confidence": round(max_probs[i].item(), 4),  # 保留4位小数
                "class_probabilities": {
                    class_names[c]: round(pred_probs[i][c].item(), 4) for c in range(self.num_classes)
                },
                "gt_label": self.gt_labels[i]  # 添加真实标签
            })

        return results

    def infer_custom(self, custom_roi_root, custom_valid_mask):
        """
        新增：基于自定义ROI图片和valid_mask的推理（适配任意idx+数字的文件名）
        :param custom_roi_root: 自定义ROI图片根目录
        :param custom_valid_mask: 12个int的列表（0/1）
        :return: 结构化推理结果
        """
        # 预处理自定义ROI
        roi_imgs = self.preprocess_custom_roi(custom_roi_root, custom_valid_mask)

        # 推理（禁用梯度计算，节省内存）
        with torch.no_grad():
            pred_logits = self.model(roi_imgs)  # [1,12,3]
            pred_cls = torch.argmax(pred_logits, dim=-1).squeeze(0)  # [12,]

        # YOLO11后处理：无效ROI强制设为0类
        pred_cls[~self.roi_valid_mask] = 0

        # 计算置信度（softmax转概率）
        pred_probs = torch.softmax(pred_logits, dim=-1).squeeze(0)  # [12,3]
        max_probs = pred_probs.max(dim=-1)[0]  # [12,]

        # 整理结构化结果（包含实际加载的文件名）
        results = []
        class_names = {0: "无效ROI", 1: "有效无方块", 2: "有效有方块"}
        for i in range(self.num_roi):
            results.append({
                "roi_position": i + 1,
                "roi_filename": self.roi_filenames[i],  # 实际加载的文件名
                "is_valid_roi": self.roi_valid_mask[i].item(),
                "predicted_class_id": pred_cls[i].item(),
                "predicted_class_name": class_names[pred_cls[i].item()],
                "confidence": round(max_probs[i].item(), 4),  # 保留4位小数
                "class_probabilities": {
                    class_names[c]: round(pred_probs[i][c].item(), 4) for c in range(self.num_classes)
                }
            })

        return results

    def batch_infer_with_metrics(self, img_idx_list, error_log_path="infer_error_log.txt"):
        """
        批量推理并计算指标（正确率），记录错误信息到TXT
        :param img_idx_list: 批量推理的idx列表
        :param error_log_path: 错误日志保存路径
        :return: 整体正确率、详细统计结果
        """
        # 初始化统计变量
        total_valid_roi = 0  # 总有效ROI数（mask=1）
        total_correct_roi = 0  # 总正确预测数
        error_records = []  # 错误记录列表

        # 清空原有错误日志
        with open(error_log_path, "w", encoding="utf-8") as f:
            f.write("=== YOLO11 ROI推理错误日志 ===\n")
            f.write(f"推理时间：{os.popen('date /t').read().strip()} {os.popen('time /t').read().strip()}\n")
            f.write("=" * 50 + "\n\n")

        # 遍历每个推理样本
        for img_idx in img_idx_list:
            print(f"\n=== 推理样本 {img_idx} ===")
            try:
                # 单样本推理
                results = self.infer(img_idx)
                batch_error_rois = []  # 当前批次错误的ROI

                # 遍历每个ROI计算指标
                for res in results:
                    roi_pos = res["roi_position"]
                    is_valid = res["is_valid_roi"]
                    gt_label = res["gt_label"]  # 真实标签：1=有方块，0=无方块
                    pred_cls_id = res["predicted_class_id"]  # 预测类别：1=无方块，2=有方块，0=无效

                    # 仅有效ROI参与计算
                    if is_valid:
                        total_valid_roi += 1
                        # 映射预测结果到标签体系：
                        # 预测1（无方块）→ 对应标签0；预测2（有方块）→ 对应标签1
                        pred_label = 1 if pred_cls_id == 2 else 0 if pred_cls_id == 1 else -1

                        # 判断是否正确
                        if pred_label == gt_label:
                            total_correct_roi += 1
                        else:
                            # 记录错误ROI信息
                            error_info = {
                                "batch_idx": img_idx,
                                "roi_position": roi_pos,
                                "gt_label": gt_label,
                                "gt_label_desc": "有方块" if gt_label == 1 else "无方块",
                                "pred_cls_id": pred_cls_id,
                                "pred_label_desc": res["predicted_class_name"],
                                "confidence": res["confidence"]
                            }
                            batch_error_rois.append(error_info)

                    # 打印推理结果（带真实标签）
                    is_valid_str = str(is_valid)
                    print(
                        f"ROI{res['roi_position']:2d} | 有效:{is_valid_str:5s} | "
                        f"真实标签:{res['gt_label']} | 预测:{res['predicted_class_name']:8s} | "
                        f"置信度:{res['confidence']:.4f}"
                    )

                # 记录当前批次的错误信息
                if batch_error_rois:
                    error_records.extend(batch_error_rois)
                    # 写入错误日志
                    with open(error_log_path, "a", encoding="utf-8") as f:
                        f.write(f"【批次 {img_idx}】错误ROI列表：\n")
                        for err in batch_error_rois:
                            f.write(
                                f"  ROI{err['roi_position']:2d} | "
                                f"真实标签：{err['gt_label_desc']} | "
                                f"预测结果：{err['pred_label_desc']} | "
                                f"置信度：{err['confidence']:.4f}\n"
                            )
                        f.write("\n")

            except Exception as e:
                error_msg = f"❌ 样本{img_idx}推理失败：{str(e)}"
                print(error_msg)
                # 记录推理失败的批次
                with open(error_log_path, "a", encoding="utf-8") as f:
                    f.write(f"【批次 {img_idx}】推理失败：{str(e)}\n\n")

        # 计算整体正确率
        accuracy = total_correct_roi / total_valid_roi if total_valid_roi > 0 else 0.0

        # 写入统计总结
        with open(error_log_path, "a", encoding="utf-8") as f:
            f.write("=" * 50 + "\n")
            f.write("=== 推理统计总结 ===\n")
            f.write(f"总推理批次：{len(img_idx_list)}\n")
            f.write(f"总有效ROI数：{total_valid_roi}\n")
            f.write(f"总正确预测数：{total_correct_roi}\n")
            f.write(f"整体正确率：{accuracy:.4f} ({total_correct_roi}/{total_valid_roi})\n")
            f.write(f"错误ROI总数：{len(error_records)}\n")

        # 打印统计结果
        print(f"\n=== 批量推理统计结果 ===")
        print(f"总推理批次：{len(img_idx_list)}")
        print(f"总有效ROI数：{total_valid_roi}")
        print(f"总正确预测数：{total_correct_roi}")
        print(f"整体正确率：{accuracy:.4f} ({total_correct_roi}/{total_valid_roi})")
        print(f"错误ROI总数：{len(error_records)}")
        print(f"错误日志已保存至：{error_log_path}")

        # 返回统计结果
        stats = {
            "total_batches": len(img_idx_list),
            "total_valid_roi": total_valid_roi,
            "total_correct_roi": total_correct_roi,
            "accuracy": accuracy,
            "error_roi_count": len(error_records)
        }
        return stats


# YOLO11推理测试（离线适配）
if __name__ == "__main__":
    # ===================== 基础配置项 =====================
    MODEL_PATH = r"H:\pycharm\yolov11\yolov11_proj1\yolo11_Custom_12roi\yolo11s_roi_3c_16334.pt"  # 训练好的模型路径
    MODEL_SIZE = "s"  # 必须与训练时一致（n/s/l）
    ROI_SIZE = 64

    # ===================== 1. 原有数据集推理（可选） =====================
    USE_ORIGINAL_DATASET = True  # 是否使用原有数据集推理
    if USE_ORIGINAL_DATASET:
        DATASET_ROOT = r"H:\pycharm\yolov11\yolov11_proj1\datasets_16334"  # 数据集根目录
        TEST_IMG_IDX = 313  # 单样本测试的索引
        BATCH_TEST_IDXS = [i for i in range(0,16000)]  # 批量测试的索引列表（修改为你需要的idx）

        # 初始化推理器（带数据集根目录）
        inferencer = YOLO11ROIInferencer(
            model_path=MODEL_PATH,
            dataset_root=DATASET_ROOT,
            model_size=MODEL_SIZE,
            roi_size=ROI_SIZE,
            num_roi=12,
            num_classes=3
        )
        print("\n✅ 推理器初始化成功（基于原有数据集）！")

        # 单样本推理
        print(f"\n=== 单样本推理测试（索引：{TEST_IMG_IDX}） ===")
        single_results = inferencer.infer(TEST_IMG_IDX)
        for res in single_results:
            print(f"ROI{res['roi_position']}："
                  f"有效={res['is_valid_roi']} | "
                  f"真实标签={res['gt_label']} | "
                  f"预测={res['predicted_class_name']} | "
                  f"置信度={res['confidence']:.4f} | "
                  f"概率分布={res['class_probabilities']}")

        # 批量推理（带指标计算和错误记录）
        print(f"\n=== 批量推理测试（索引：{BATCH_TEST_IDXS}） ===")
        batch_stats = inferencer.batch_infer_with_metrics(
            img_idx_list=BATCH_TEST_IDXS,
            error_log_path="./infer_error_log.txt"
        )
        # 保存批量推理结果（含指标）
        batch_results_path = "./batch_infer_results_with_metrics.json"
        with open(batch_results_path, "w", encoding="utf-8") as f:
            json.dump({
                "statistics": batch_stats,
                "infer_idx_list": BATCH_TEST_IDXS
            }, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 批量推理结果（含指标）已保存至：{batch_results_path}")

    # ===================== 2. 新增：自定义数据推理（核心） =====================
    USE_CUSTOM_DATA = False  # 是否使用自定义数据推理
    if USE_CUSTOM_DATA:
        # 自定义配置
        CUSTOM_ROI_ROOT = r"H:\pycharm\yolov11\yolov11_proj1\real_datasets\roi_22"  # 自定义ROI图片根目录
        CUSTOM_VALID_MASK = [0, 1, 1,
                             1, 1, 1,
                             1, 1, 0,
                             1, 1, 0]  # 12个int，0=无效ROI，1=有效ROI

        # 初始化推理器（无需数据集根目录）
        inferencer = YOLO11ROIInferencer(
            model_path=MODEL_PATH,
            dataset_root=None,  # 无需原有数据集
            model_size=MODEL_SIZE,
            roi_size=ROI_SIZE,
            num_roi=12,
            num_classes=3
        )
        print("\n✅ 推理器初始化成功（基于自定义数据）！")

        # 自定义数据推理
        print(f"\n=== 自定义数据推理测试 ===")
        print(f"自定义ROI根目录：{CUSTOM_ROI_ROOT}")
        print(f"自定义valid_mask：{CUSTOM_VALID_MASK}")
        custom_results = inferencer.infer_custom(CUSTOM_ROI_ROOT, CUSTOM_VALID_MASK)

        # 打印自定义推理结果
        print("\n=== 自定义推理结果 ===")
        for res in custom_results:
            print(
                f"ROI{res['roi_position']:2d} | 文件:{res['roi_filename']:15s} | 有效:{str(res['is_valid_roi']):5s} | "
                f"预测:{res['predicted_class_name']:8s} | 置信度:{res['confidence']:.4f}"
            )

        # 保存自定义推理结果
        custom_save_path = "./custom_infer_results.json"
        with open(custom_save_path, "w", encoding="utf-8") as f:
            json.dump(custom_results, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 自定义推理结果已保存至：{custom_save_path}")