"""
pt_to_onnx_openvino.py
    将pt权重转换成onnx,openvino格式
    支持导出动态输入的roi模型
"""
import os
import torch
import onnx
import numpy as np
import warnings

# 过滤OpenVINO runtime弃用警告
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*openvino.runtime.*")
# 新版OpenVINO 2024.x+ 正确导入方式
import openvino as ov

from model import YOLO11ROIClassifier

# ===================== 配置参数 =====================
DEVICE = torch.device("cpu")
MODEL_SIZE = "s"
NUM_ROI = 12
NUM_CLASSES = 2
ROI_SIZE = 64
ATTEN_WEIGHT = 0.15  # 新增：模型必需参数

# 文件路径（原始字符串避免转义）
PT_MODEL_PATH = r"H:\pycharm\yolov11\yolov11_proj4\yolo11Custom_atten2\yolo11_pt\best_model.pt"
ONNX_MODEL_PATH = r"H:\pycharm\yolov11\yolov11_proj4\yolo11Custom_atten2\evlate_pt\yolo11n_roi12_atten_21.onnx"
OPENVINO_IR_PATH = r"H:\pycharm\yolov11\yolov11_proj4\yolo11Custom_atten2\evlate_pt\yolo11n_roi12_atten_21"

# 输入配置
INPUT_NAME = "roi_imgs"

# ===================== 1. 加载PyTorch模型 =====================
def load_pytorch_model(pt_path, model_size, num_roi, num_classes, device):
    # 最小修改：补充atten_weight参数
    model = YOLO11ROIClassifier(
        model_size=model_size,
        num_roi=num_roi,
        num_classes=num_classes,
        roi_size=ROI_SIZE,
        atten_weight=ATTEN_WEIGHT
    ).to(device)

    checkpoint = torch.load(pt_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model_weights = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        model_weights = checkpoint["state_dict"]
    else:
        model_weights = checkpoint

    model.load_state_dict(model_weights, strict=False)
    model.eval()
    print(f"✅ PyTorch模型加载完成：{pt_path}")
    return model


# ===================== 2. PT转ONNX =====================
def convert_pt_to_onnx(model, onnx_path, device):
    onnx_dir = os.path.dirname(onnx_path)
    os.makedirs(onnx_dir, exist_ok=True)

    # 最小修改：修复dummy_input语法错误
    dummy_input = torch.randn((1, 1, 3, 64, 64), device=device)

    # 最小修改：标准动态轴（batch + ROI数量 双动态）
    dynamic_axes = {
        INPUT_NAME: {0: "batch", 1: "num_roi"},
        "pred_logits": {0: "batch", 1: "num_roi"}
    }

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=13,
        do_constant_folding=True,
        input_names=[INPUT_NAME],
        output_names=["pred_logits"],
        dynamic_axes=dynamic_axes
    )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"✅ ONNX模型导出并验证完成：{onnx_path}")


# ===================== 3. ONNX转OpenVINO IR =====================
def convert_onnx_to_openvino(onnx_path, ir_path):
    ir_dir = os.path.dirname(ir_path)
    os.makedirs(ir_dir, exist_ok=True)

    ov_model = ov.convert_model(input_model=onnx_path)
    ov.save_model(ov_model, ir_path + ".xml")
    print(f"✅ OpenVINO IR模型转换完成：{ir_path}.xml / {ir_path}.bin")

    core = ov.Core()
    compiled_model = core.compile_model(ov_model, "CPU")
    print(f"✅ OpenVINO模型编译验证完成（设备：CPU）")
    return compiled_model


# ===================== 4. 推理验证 =====================
def inference_verify(compiled_model):
    test_input_1 = np.random.randn(1, 1, 3, 64, 64).astype(np.float32)
    test_input_12 = np.random.randn(1, 12, 3, 64, 64).astype(np.float32)
    test_input_7 = np.random.randn(1, 7, 3, 64, 64).astype(np.float32)

    infer_request = compiled_model.create_infer_request()

    infer_request.infer([test_input_1])
    out1 = infer_request.get_output_tensor(0).data
    infer_request.infer([test_input_12])
    out12 = infer_request.get_output_tensor(0).data
    infer_request.infer([test_input_7])
    out7 = infer_request.get_output_tensor(0).data

    print(f"✅ 动态推理验证完成：")
    print(f"   单ROI输入形状：{test_input_1.shape} | 输出形状：{out1.shape}")
    print(f"   12ROI输入形状：{test_input_12.shape} | 输出形状：{out12.shape}")
    print(f"   7ROI输入形状：{test_input_7.shape} | 输出形状：{out7.shape}")


# ===================== 主函数 =====================
if __name__ == "__main__":
    model = load_pytorch_model(
        pt_path=PT_MODEL_PATH,
        model_size=MODEL_SIZE,
        num_roi=NUM_ROI,
        num_classes=NUM_CLASSES,
        device=DEVICE
    )

    convert_pt_to_onnx(model=model, onnx_path=ONNX_MODEL_PATH, device=DEVICE)
    compiled_ov_model = convert_onnx_to_openvino(onnx_path=ONNX_MODEL_PATH, ir_path=OPENVINO_IR_PATH)
    inference_verify(compiled_ov_model)

    print("\n🎉 所有转换步骤完成！")
    print(f"   ONNX模型：{ONNX_MODEL_PATH}")
    print(f"   OpenVINO IR模型：{OPENVINO_IR_PATH}.xml / {OPENVINO_IR_PATH}.bin")