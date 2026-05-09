import cv2
import torch
import numpy as np
from torchvision import transforms
from model import YOLO11ROIClassifier

# ===================== 配置项（与训练完全一致，请勿修改）=====================
ROI_SIZE = 64
YOLO11_MEAN = [0.485, 0.456, 0.406]
YOLO11_STD = [0.229, 0.224, 0.225]
MODEL_WEIGHT_PATH = r"H:\pycharm\yolov11\yolov11_proj4\yolo11Custom_atten2\yolo11_pt\best_model.pt"

# ===================== 1. 初始化推理预处理（容器化通用）=====================
infer_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=YOLO11_MEAN, std=YOLO11_STD)
])


# ===================== 2. 容器化函数：批量处理任意数量ROI图片 =====================
def roi_images_to_input(img_paths: list) -> torch.Tensor:
    """
    容器化输入：传入图片路径列表，自动构造模型输入张量
    :param img_paths: 图片路径列表 [path1, path2, ...] （任意长度，1~12张）
    :return: 模型输入张量 [1, N, 3, 64, 64]  N=图片数量
    """
    tensor_list = []
    for path in img_paths:
        # 读取+预处理单张图片
        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resize = cv2.resize(img_rgb, (ROI_SIZE, ROI_SIZE))
        img_tensor = infer_transform(img_resize)
        tensor_list.append(img_tensor)

    # 堆叠为 [N, 3, 64, 64] -> 增加batch维度 -> [1, N, 3, 64, 64]
    roi_tensor = torch.stack(tensor_list, dim=0)
    model_input = roi_tensor.unsqueeze(0)
    return model_input


# ===================== 修复版：逐一推理文件夹下12张照片 =====================
def infer_by_one(path_to_dir):
    """
    逐一推理文件夹下 1~12.png 共12张ROI图片
    :param path_to_dir: ROI图片所在文件夹路径
    """
    # 遍历1-12张图片（原代码range(1,12)只遍历1-11，漏了12）
    for i in range(1, 13):
        # 拼接完整图片路径（统一Windows路径格式，避免斜杠错误）
        img_path = f"{path_to_dir.rstrip('/')}\\{i}.png"
        print(f"\n正在推理第 {i} 张图片: {img_path}")

        # 关键：函数要求传入列表，单张图片包装为 [img_path]
        model_input = roi_images_to_input([img_path])

        # 推理（和主程序一致）
        with torch.no_grad():
            output = model(model_input)

        # 解析结果（单张图片，维度适配）
        pred_probs = torch.softmax(output, dim=-1).squeeze(0).numpy()  # [1, 2]
        pred_classes = torch.argmax(torch.from_numpy(pred_probs), dim=-1).numpy()

        # 打印单张结果
        for idx, (cls_, prob_) in enumerate(zip(pred_classes, pred_probs)):
            print(f"第{i}张ROI | 预测类别: {cls_} | 类别概率: {prob_}")


# ===================== 3. 容器化输入：定义你的ROI图片列表（任意数量）=====================
ROI_IMAGE_CONTAINER = [
    r"H:\pycharm\yolov11\yolov11_proj3\datasets_real_p120\roi_images\roi_2\1.png",
    r"H:\pycharm\yolov11\yolov11_proj3\datasets_real_p120\roi_images\roi_2\2.png",
    r"H:\pycharm\yolov11\yolov11_proj3\datasets_real_p120\roi_images\roi_2\3.png",
    r"H:\pycharm\yolov11\yolov11_proj3\datasets_real_p120\roi_images\roi_2\4.png",
    r"H:\pycharm\yolov11\yolov11_proj3\datasets_real_p120\roi_images\roi_2\5.png",
    r"H:\pycharm\yolov11\yolov11_proj3\datasets_real_p120\roi_images\roi_2\6.png",
    r"H:\pycharm\yolov11\yolov11_proj3\datasets_real_p120\roi_images\roi_2\7.png",
    r"H:\pycharm\yolov11\yolov11_proj3\datasets_real_p120\roi_images\roi_2\8.png",
    r"H:\pycharm\yolov11\yolov11_proj3\datasets_real_p120\roi_images\roi_2\9.png",
    r"H:\pycharm\yolov11\yolov11_proj3\datasets_real_p120\roi_images\roi_2\10.png",
    r"H:\pycharm\yolov11\yolov11_proj3\datasets_real_p120\roi_images\roi_2\11.png",
    r"H:\pycharm\yolov11\yolov11_proj3\datasets_real_p120\roi_images\roi_2\12.png"
]

# ===================== 4. 构造模型输入 =====================
model_input = roi_images_to_input(ROI_IMAGE_CONTAINER)
print(f"模型输入形状: {model_input.shape}")
print(f"容器内ROI数量: {model_input.shape[1]}")

# ===================== 5. 初始化模型 + 加载权重 =====================
model = YOLO11ROIClassifier(model_size="s", num_classes=2, num_roi=12, roi_size=64)
checkpoint = torch.load(MODEL_WEIGHT_PATH, map_location="cpu")
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ 加载checkpoint成功 | 最优F1: {checkpoint.get('best_pos_f1', 0.0):.4f}")
else:
    model.load_state_dict(checkpoint)
    print(f"✅ 加载权重成功 | 路径: {MODEL_WEIGHT_PATH}")
model.eval()

# ===================== 6. 容器化批量推理 =====================
with torch.no_grad():
    output = model(model_input)

# ===================== 7. 解析所有ROI的结果 =====================
print(f"\n模型输出形状: {output.shape}")
pred_probs = torch.softmax(output, dim=-1).squeeze(0).numpy()
pred_classes = torch.argmax(torch.from_numpy(pred_probs), dim=-1).numpy()

print("\n===== 批量推理所有ROI结果 =====")
for idx, (cls_, prob_) in enumerate(zip(pred_classes, pred_probs)):
    print(f"第{idx + 1}张ROI | 预测类别: {cls_} | 类别概率: {prob_}")

# ===================== 调用函数：逐一推理12张图片 =====================
infer_by_one(r"H:\pycharm\yolov11\yolov11_proj3\datasets_real_p120\roi_images\roi_2")
