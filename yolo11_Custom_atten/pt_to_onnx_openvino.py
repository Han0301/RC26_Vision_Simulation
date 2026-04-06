"""
pt_to_onnx_openvino.py
    将pt权重转换成onnx,openvino格式
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

# 文件路径（原始字符串避免转义）
PT_MODEL_PATH = r"H:\pycharm\yolov11\yolov11_proj3\yolo11Custom_atten\evlate_pt\yolo11s_roi12_atten_2.pt"
ONNX_MODEL_PATH = r"H:\pycharm\yolov11\yolov11_proj3\yolo11Custom_atten\evlate_pt\yolo11s_roi12_atten_2.onnx"
OPENVINO_IR_PATH = r"H:\pycharm\yolov11\yolov11_proj3\yolo11Custom_atten\evlate_pt\yolo11s_roi12_atten_2"

# 输入配置
BATCH_SIZE = 1
INPUT_SHAPE = (BATCH_SIZE, NUM_ROI, 3, ROI_SIZE, ROI_SIZE)
INPUT_NAME = "roi_imgs"

# ===================== 1. 加载PyTorch模型 =====================
def load_pytorch_model(pt_path, model_size, num_roi, num_classes, device):
    model = YOLO11ROIClassifier(
        model_size=model_size,
        num_roi=num_roi,
        num_classes=num_classes,
        roi_size=ROI_SIZE
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
def convert_pt_to_onnx(model, onnx_path, input_shape, device):
    # 提前创建目录
    onnx_dir = os.path.dirname(onnx_path)
    os.makedirs(onnx_dir, exist_ok=True)

    dummy_input = torch.randn(input_shape, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=13,
        do_constant_folding=True,
        input_names=[INPUT_NAME],
        output_names=["pred_logits"],
        dynamic_axes=None
    )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"✅ ONNX模型导出并验证完成：{onnx_path}")


# ===================== 3. ONNX转OpenVINO IR =====================
def convert_onnx_to_openvino(onnx_path, ir_path):
    # 提前创建目录
    ir_dir = os.path.dirname(ir_path)
    os.makedirs(ir_dir, exist_ok=True)

    # 核心修复：移除data_type参数（2024.x+版本已移除）
    ov_model = ov.convert_model(
        input_model=onnx_path,
        input={INPUT_NAME: INPUT_SHAPE}
    )

    # 保存IR模型（新版API）
    ov.save_model(ov_model, ir_path + ".xml")
    print(f"✅ OpenVINO IR模型转换完成：{ir_path}.xml / {ir_path}.bin")

    # 验证加载
    core = ov.Core()
    compiled_model = core.compile_model(ov_model, "CPU")
    print(f"✅ OpenVINO模型编译验证完成（设备：CPU）")
    return compiled_model


# ===================== 4. 推理验证（核心修复：set_input_tensor参数） =====================
def inference_verify(compiled_model, input_shape):
    """验证转换后的OpenVINO模型推理正确性（适配2024.x+ API）"""
    # 构造测试输入（numpy数组）
    test_input = np.random.randn(*input_shape).astype(np.float32)

    # 核心修复1：将numpy数组转换为OpenVINO Tensor对象
    input_tensor = ov.Tensor(test_input)

    # 核心修复2：调用set_input_tensor（参数为索引 + Tensor，或直接传Tensor）
    infer_request = compiled_model.create_infer_request()
    infer_request.set_input_tensor(0, input_tensor)  # 0=输入索引，input_tensor=OpenVINO Tensor

    # 执行推理
    infer_request.infer()

    # 获取输出
    output = infer_request.get_output_tensor(0).data  # 0=输出索引
    print(f"✅ 推理验证完成：")
    print(f"   输入形状：{test_input.shape}")
    print(f"   输出形状：{output.shape} (预期：{[BATCH_SIZE, NUM_ROI, NUM_CLASSES]})")


# ===================== 主函数 =====================
if __name__ == "__main__":
    # 加载PT模型
    model = load_pytorch_model(
        pt_path=PT_MODEL_PATH,
        model_size=MODEL_SIZE,
        num_roi=NUM_ROI,
        num_classes=NUM_CLASSES,
        device=DEVICE
    )

    # PT转ONNX
    convert_pt_to_onnx(
        model=model,
        onnx_path=ONNX_MODEL_PATH,
        input_shape=INPUT_SHAPE,
        device=DEVICE
    )

    # ONNX转OpenVINO IR
    compiled_ov_model = convert_onnx_to_openvino(
        onnx_path=ONNX_MODEL_PATH,
        ir_path=OPENVINO_IR_PATH
    )

    # 推理验证（修复后）
    inference_verify(compiled_ov_model, INPUT_SHAPE)

    print("\n🎉 所有转换步骤完成！")
    print(f"   ONNX模型：{ONNX_MODEL_PATH}")
    print(f"   OpenVINO IR模型：{OPENVINO_IR_PATH}.xml / {OPENVINO_IR_PATH}.bin")
