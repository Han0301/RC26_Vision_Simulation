"""
infer_main.py
    核心推理类, 包含单文件推理, 数据集推理, 并结合相关参数进行统计
"""
import datetime
import os
import json
import cv2
import numpy as np
import torch
from prompt_toolkit.utils import to_str
from tqdm import tqdm
from model import YOLO11ROIClassifier  # 确保model.py适配二分类
from infer_func import place_evaluate,ps_w_evaluate,conf_evaluate,save_results,write_txt

class YOLO11ROIInferencer:
    def __init__(self, model_path, dataset_root=None, model_size="s", roi_size=64, num_roi=12, num_classes=2):
        """初始化推理器"""
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model_name = os.path.splitext( os.path.basename(model_path) )[0]

        # 核心参数
        self.model_size = model_size
        self.roi_size = roi_size
        self.num_roi = num_roi
        self.num_classes = num_classes

        # 数据集路径
        self.dataset_root = dataset_root
        if dataset_root is not None:
            self.roi_img_root = os.path.join(dataset_root, "roi_images")
            self.label_dir = os.path.join(dataset_root, "labels")
            if not os.path.exists(self.roi_img_root):
                raise FileNotFoundError(f"❌ 图像文件夹不存在：{self.roi_img_root}")
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
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
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

        # 推理统计参数
        self.conf_dict = {}
        self.place_acc_count = [0] * 12
        self.ps_count = [0] * 12
        self.ps_acc = [0] * 12

    def pre_roi_folder(self, roi_folder):
        """
        从指定文件夹 预处理 12个ROI
        :param roi_folder: ROI图片所在文件夹路径, 如roi_1,roi_2...子文件夹
        :return: 预处理后的ROI tensor [1,12,3,64,64], roi_filenames 文件夹照片路径列表
        """
        # 1 校验文件夹是否存在
        if not os.path.exists(roi_folder):
            raise FileNotFoundError(f"❌ ROI文件夹不存在：{roi_folder}")

        # 2 获取路径列表和图片列表
        roi_imgs = []       # 待加载的图像列表
        roi_filenames = []  # 记录实际加载的文件名
        for i in range(1,13,1):
            roi_filename = os.path.join(roi_folder , to_str(i) + ".png")

            roi_img = cv2.imread(roi_filename)
            if roi_img is None:
                print(f"⚠️ ROI文件无法读取：{roi_filename}，使用全黑图替代")
                roi_img = np.zeros((self.roi_size, self.roi_size, 3), dtype=np.uint8)
            else:
                # 格式转换（BGR→RGB）+ 尺寸调整
                roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
                roi_img = cv2.resize(roi_img, (self.roi_size, self.roi_size), interpolation=cv2.INTER_LINEAR)

            roi_filenames.append(roi_filename)
            roi_imgs.append(roi_img)

        roi_imgs = np.stack(roi_imgs)       # (12, 64, 64, 3)
        roi_imgs = torch.from_numpy(roi_imgs).permute(0,3,1,2) / 255.0      # torch.Size([12, 3, 64, 64])
        roi_imgs = roi_imgs.to(self.device)
        roi_imgs = (roi_imgs - self.mean) / self.std
        roi_imgs = roi_imgs.unsqueeze(0)    # torch.Size([1, 12, 3, 64, 64])
        return roi_imgs,roi_filenames

    def infer_roi_folder(self,roi_folder_path:str, label_path=None, is_print=False, is_save=False, save_path=None):
        """
        文件夹直接推理模式：读取指定文件夹内的12张ROI图片并推理
        :param roi_folder_path: ROI图片所在文件夹路径
        :param label_path: 标签路径(仅用于打印 和 保存推理结果)
        :param is_print: 是否打印当前的推理结果
        :param is_save: 是否保存推理结果
        :param save_path: 保存路径
        :return: 推理结果
        """
        roi_imgs,roi_filenames = self.pre_roi_folder(roi_folder_path)

        with torch.no_grad():
            pred_logits = self.model(roi_imgs)  # [1,12,2]
            pred_cls = torch.argmax(pred_logits, dim=-1).squeeze(0)  # [12,]
            pred_probs = torch.softmax(pred_logits, dim=-1).squeeze(0)  # [12,2]

            # 转 numpy
            pred_tensor = [pred_cls, pred_probs]
            pred_cls_np, pred_probs_np = [pred.cpu().numpy() for pred in pred_tensor]

            if label_path is None:
                if is_print:
                    print(f"正在推理文件夹路径: {roi_folder_path}")
                    for i in range(12):
                        print(f"位置: {i + 1}, 置信度: [{pred_probs_np[i][0]:.3f}, {pred_probs_np[i][1]:.3f}], 预测类别: {pred_cls_np[i]}")
                if is_save:
                    write_txt(save_path, roi_folder_path, pred_cls_np, pred_probs_np)
                    print(f"json文件已保存至{save_path}")
            else:
                labels, point_size, wrong_place = [], [], []
                with open(label_path, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
                    labels = json_data["labels"]
                    point_size = json_data["point_size"]
                if is_print:
                    print(f"正在推理文件夹路径: {roi_folder_path}")
                    for i in range(12):
                        print(f"位置: {i + 1:2d}, 置信度: [{pred_probs_np[i][0]:.3f}, {pred_probs_np[i][1]:.3f}], 预测类别: {pred_cls_np[i]},真实类别: {labels[i]}, point_size: {point_size[i]}")
                        if pred_cls_np[i] != labels[i]:
                            wrong_place.append(i + 1)
                    print(f"正确数: {12 - len(wrong_place)}, 错误位置: {wrong_place}")
                if is_save:
                    write_txt(save_path, roi_folder_path, pred_cls_np, pred_probs_np)
                    print(f"json文件已保存至{save_path}")
            return pred_cls_np, pred_probs_np

    def infer_datasets(self,datasets_path:str, is_save=False, save_path=None,is_conf=False, conf_list=None, is_point_size_weight=None,ps_w_thods = None, is_place=None):
        roi_images_dirs, labels_dir = datasets_path + r"\roi_images", datasets_path + r"\labels"
        # 相关统计变量
        results = {}
        infer_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 进行推理
        with os.scandir(roi_images_dirs) as roi_dirs:
            total_folders = len([f for f in os.listdir(roi_images_dirs) if os.path.isdir(os.path.join(roi_images_dirs, f))])
            for roi_dir in tqdm(roi_dirs, desc="推理中", colour="red", total=total_folders):
                if roi_dir.is_dir():
                    floder_num = roi_dir.path.split('_')[-1]
                    pred_cls_np, pred_probs_np = self.infer_roi_folder(roi_dir.path)
                    json_path = labels_dir + r"\label_" + to_str(floder_num) + ".json"
                    with open(json_path,"r",encoding="utf-8") as f:
                        json_data = json.load(f)
                        labels, point_size = json_data["labels"],json_data["point_size"]
                    results[to_str(floder_num)] = {
                        "pred_probs_np": pred_probs_np,
                        "pred_cls_np": pred_cls_np,
                        "labels": labels,
                        "point_size":point_size
                    }
        # 统计
        if is_conf and is_conf is not None:
            self.conf_dict = conf_evaluate(results, conf_list)
        if is_place and is_place is not None:
            self.place_acc_count,_ = place_evaluate(results,self.place_acc_count)
        if is_point_size_weight and is_point_size_weight is not None:
            ps_w_evaluate(results,ps_w_thods)
        if is_save:
            save_results(
                results=results,
                infer_time=infer_time,  # 自动传入推理时间
                model_path=self.model_path,  # 自动传入模型路径
                dataset_path=datasets_path,  # 自动传入数据集路径
                is_conf=is_conf, conf_list=conf_list,
                is_place=is_place, place_acc_count=self.place_acc_count,
                is_point_size_weight=is_point_size_weight, ps_w_thods=ps_w_thods,
                is_save=is_save, save_path=save_path
            )

if __name__ == "__main__":
    inferencer = YOLO11ROIInferencer(
        model_path="H:\pycharm\yolov11\yolov11_proj3\yolo11Custom_atten\evlate_pt\yolo11s_roi12_atten_2.pt",
        dataset_root=None,
        model_size="s",
        roi_size=64,
        num_roi=12,
        num_classes=2
    )
    inferencer.infer_roi_folder(r"H:\pycharm\yolov11\yolov11_proj3\datasets_real_p179\roi_images\roi_1",
                                label_path="H:\pycharm\yolov11\yolov11_proj3\datasets_real_p179\labels\label_1.json",
                                is_print=True
                                ,is_save=True,
                                save_path=r"H:\pycharm\yolov11\yolov11_proj3\yolo11Custom_atten\error\pre_error.csv")

    inferencer.infer_datasets(datasets_path=r"H:\pycharm\yolov11\yolov11_proj3\datasets_real_p179",
                                is_conf=True,
                              conf_list=[0.9,0.85,0.80,0.75,0.7,0.65,0.6],
                              is_place=True,
                              is_point_size_weight=True,
                              ps_w_thods=[0.4,0.3,0.2,0.1],
                              is_save=True,
                              save_path=r"H:\pycharm\yolov11\yolov11_proj3\yolo11Custom_atten\error\ " + f"{inferencer.model_name}" + f"_datasets_real_p179"
                              )