"""
train_config.py
    训练参数, 喂给train函数用于训练
"""
import torch

# 1 模型本身相关
model_config = {
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "ROI_IMG_SIZE": 64,  # roi图像大小
    "NUM_ROI": 12,  # roi数量
    "NUM_CLASSES": 2,  # 分类数
    "MODEL_SIZE": "s",  # 模型尺寸
    "YOLO_weight_path": "H:\pycharm\yolov11\yolov11.pt\yolo11s.pt",  # 加载YOLO预训练权重的路径,尺寸要和模型尺寸一致
    "ATTEN_WEIGHT": 0.15
}

# 2 数据集和训练相关
train_config = {
    "BATCH_SIZE": 32,  # 加载图像的批次
    "EPOCHS": 100,  # 训练总轮数
    "patience": 12,  # 耐心
    "mixup_rate": 0.2,  # mixup触发的概率
    "mixup_alpha": 0.2,  # mixup增强的beta 分布参数

    "WORKERS": 8,
    "RESUME_TRAIN": False,
    # RESUME_TRAIN = False, 从头训练
    "SAVE_DIR": "./yolo11_pt",  # 输出的模型路径
    "MODEL_NAME": "roi12_atten_red17_.pt",
    # RESUME_TRAIN = True, 加载之前的模型继续训练
    "CHECKPOINT_PATH": r"./yolo11_pt/yolo11s_roi12_ps_6.pt"  # 之前保存的模型路径
}

# 3 加载数据集的方式
dataset_config = {
    "load_datasets": True,
    # load_datasets = True 指定数据集
    "DATASET_ROOTS": [  # 数据集路径
        r"H:\pycharm\yolov11\yolov11_proj3\yolo11Custom_pointsize\test\mini_datasets_3066\datasets_1",
        r"H:\pycharm\yolov11\yolov11_proj3\datasets_real_p423",
        r"H:\pycharm\yolov11\yolov11_proj3\datasets_real_p423"
    ],
    "VAL_RATIO": 0.2,  # 验证集的占比
    # load_datasets = False 指定数据集和验证集
    "TRAIN_DATASETS": [r"H:\pycharm\yolov11\yolov11_proj1\datasets_16334"],
    "VAL_DATASETS": [r"H:\pycharm\yolov11\yolov11_proj1\datasets_global_test100"]
}

# 4 损失函数和优化器相关
loss_config = {
    "LOSS_WEIGHT": [2.0, 1.0],  # 损失在两个类别上面的权重
    "FOCAL_LOSS": 1.5,  # 难样本挖掘系数
    # 学习率
    "LEARNING_RATE": 5e-5 if model_config["MODEL_SIZE"] == "l" else 1e-4 if model_config[
                                                                                "MODEL_SIZE"] == "s" else 1e-3,
    "WEIGHT_DECAY": 5e-4,  # 权重衰减（L2 正则），防止模型过拟合
    "count_loss_weight": 0.05  # 数量约束损失的权重
}

loss_config_2 = {
    "LOSS_WEIGHT": [2.0, 1.0],  # 损失在两个类别上面的权重
    "FOCAL_LOSS": 1.5,  # 难样本挖掘系数
    # 学习率
    "LEARNING_RATE": 5e-5 if model_config["MODEL_SIZE"] == "l" else 1e-4 if model_config[
                                                                                "MODEL_SIZE"] == "s" else 1e-3,
    "WEIGHT_DECAY": 5e-4,  # 权重衰减（L2 正则），防止模型过拟合
    "count_loss_weight": 0.05  # 数量约束损失的权重
}

model_config_2 = {
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "ROI_IMG_SIZE": 64,  # roi图像大小
    "NUM_ROI": 12,  # roi数量
    "NUM_CLASSES": 2,  # 分类数
    "MODEL_SIZE": "s",  # 模型尺寸
    "YOLO_weight_path": "H:\pycharm\yolov11\yolov11.pt\yolo11s.pt",  # 加载YOLO预训练权重的路径,尺寸要和模型尺寸一致
    "ATTEN_WEIGHT": 0.30
}

model_config_3 = {
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "ROI_IMG_SIZE": 64,  # roi图像大小
    "NUM_ROI": 12,  # roi数量
    "NUM_CLASSES": 2,  # 分类数
    "MODEL_SIZE": "s",  # 模型尺寸
    "YOLO_weight_path": "H:\pycharm\yolov11\yolov11.pt\yolo11s.pt",  # 加载YOLO预训练权重的路径,尺寸要和模型尺寸一致
    "ATTEN_WEIGHT": 0.50
}

train_config_2 = {
    "BATCH_SIZE": 32,  # 加载图像的批次
    "EPOCHS": 100,  # 训练总轮数
    "patience": 12,  # 耐心
    "mixup_rate": 0.2,  # mixup触发的概率
    "mixup_alpha": 0.2,  # mixup增强的beta 分布参数

    "WORKERS": 8,
    "RESUME_TRAIN": False,
    # RESUME_TRAIN = False, 从头训练
    "SAVE_DIR": "./yolo11_pt",  # 输出的模型路径
    "MODEL_NAME": "roi12_atten_blue17_.pt",
    # RESUME_TRAIN = True, 加载之前的模型继续训练
    "CHECKPOINT_PATH": r"./yolo11_pt/yolo11s_roi12_ps_6.pt"  # 之前保存的模型路径
}

dataset_config_3 = {
    "load_datasets": True,
    # load_datasets = True 指定数据集
    "DATASET_ROOTS": [  # 数据集路径
        r"H:\pycharm\yolov11\yolov11_proj3\yolo11Custom_pointsize\test\mini_datasets_3066\datasets_1",
        r"H:\pycharm\yolov11\yolov11_proj3\datasets_real_p423",
    r"H:\pycharm\yolov11\yolov11_proj3\datasets_real_p423",
    r"H:\pycharm\yolov11\yolov11_proj3\datasets_real_p423"
    ],
    "VAL_RATIO": 0.2,  # 验证集的占比
    # load_datasets = False 指定数据集和验证集
    "TRAIN_DATASETS": [r"H:\pycharm\yolov11\yolov11_proj1\datasets_16334"],
    "VAL_DATASETS": [r"H:\pycharm\yolov11\yolov11_proj1\datasets_global_test100"]
}
train_config_3 = {
    "BATCH_SIZE": 32,  # 加载图像的批次
    "EPOCHS": 100,  # 训练总轮数
    "patience": 12,  # 耐心
    "mixup_rate": 0.2,  # mixup触发的概率
    "mixup_alpha": 0.2,  # mixup增强的beta 分布参数

    "WORKERS": 8,
    "RESUME_TRAIN": False,
    # RESUME_TRAIN = False, 从头训练
    "SAVE_DIR": "./yolo11_pt",  # 输出的模型路径
    "MODEL_NAME": "yolo11n_roi12_atten_15.pt",
    # RESUME_TRAIN = True, 加载之前的模型继续训练
    "CHECKPOINT_PATH": r"./yolo11_pt/yolo11s_roi12_ps_6.pt"  # 之前保存的模型路径
}


dataset_config_4 = {
    "load_datasets": True,
    # load_datasets = True 指定数据集
    "DATASET_ROOTS": [  # 数据集路径
        r"H:\pycharm\yolov11\yolov11_proj3\datasets_blue_mapnew250",
        r"H:\pycharm\yolov11\yolov11_proj3\datasets_blue_mapnew250",
        r"H:\pycharm\yolov11\yolov11_proj3\datasets_blue_mapnew250",
        r"I:\datasets_real_blue_new785",
        r"I:\datasets_real_blue_new785",
        r"I:\datasets_real_blue_new785",
    ],
    "VAL_RATIO": 0.2,  # 验证集的占比
    # load_datasets = False 指定数据集和验证集
    "TRAIN_DATASETS": [r"H:\pycharm\yolov11\yolov11_proj1\datasets_16334"],
    "VAL_DATASETS": [r"H:\pycharm\yolov11\yolov11_proj1\datasets_global_test100"]
}

dataset_config_5 = {
    "load_datasets": True,
    # load_datasets = True 指定数据集
    "DATASET_ROOTS": [  # 数据集路径
        r"H:\pycharm\yolov11\yolov11_proj3\Datasets_ROI_new400",
        r"H:\pycharm\yolov11\yolov11_proj3\Datasets_ROI_new400",
        r"H:\pycharm\yolov11\yolov11_proj3\datasets_test_new2520",
        r"H:\pycharm\yolov11\yolov11_proj3\datasets_test_new2520",
        r"H:\pycharm\yolov11\yolov11_proj3\datasets_test_new2520",
        r"I:\car_red_new",
        r"I:\car_red_new",
        r"I:\car_red_new",
    ],
    "VAL_RATIO": 0.2,  # 验证集的占比
    # load_datasets = False 指定数据集和验证集
    "TRAIN_DATASETS": [r"H:\pycharm\yolov11\yolov11_proj1\datasets_16334"],
    "VAL_DATASETS": [r"H:\pycharm\yolov11\yolov11_proj1\datasets_global_test100"]
}
train_config_4 = {
    "BATCH_SIZE": 32,  # 加载图像的批次
    "EPOCHS": 100,  # 训练总轮数
    "patience": 12,  # 耐心
    "mixup_rate": 0.2,  # mixup触发的概率
    "mixup_alpha": 0.2,  # mixup增强的beta 分布参数

    "WORKERS": 8,
    "RESUME_TRAIN": False,
    # RESUME_TRAIN = False, 从头训练
    "SAVE_DIR": "./yolo11_pt",  # 输出的模型路径
    "MODEL_NAME": "yolo11n_roi12_atten_16.pt",
    # RESUME_TRAIN = True, 加载之前的模型继续训练
    "CHECKPOINT_PATH": r"./yolo11_pt/yolo11s_roi12_ps_6.pt"  # 之前保存的模型路径
}

train_config_5 = {
    "BATCH_SIZE": 32,  # 加载图像的批次
    "EPOCHS": 100,  # 训练总轮数
    "patience": 12,  # 耐心
    "mixup_rate": 0.2,  # mixup触发的概率
    "mixup_alpha": 0.2,  # mixup增强的beta 分布参数

    "WORKERS": 8,
    "RESUME_TRAIN": False,
    # RESUME_TRAIN = False, 从头训练
    "SAVE_DIR": "./yolo11_pt",  # 输出的模型路径
    "MODEL_NAME": "yolo11n_roi12_atten_18.pt",
    # RESUME_TRAIN = True, 加载之前的模型继续训练
    "CHECKPOINT_PATH": r"./yolo11_pt/yolo11s_roi12_ps_6.pt"  # 之前保存的模型路径
}
train_config_6 = {
    "BATCH_SIZE": 32,  # 加载图像的批次
    "EPOCHS": 100,  # 训练总轮数
    "patience": 12,  # 耐心
    "mixup_rate": 0.2,  # mixup触发的概率
    "mixup_alpha": 0.2,  # mixup增强的beta 分布参数

    "WORKERS": 8,
    "RESUME_TRAIN": False,
    # RESUME_TRAIN = False, 从头训练
    "SAVE_DIR": "./yolo11_pt",  # 输出的模型路径
    "MODEL_NAME": "yolo11n_roi12_atten_19.pt",
    # RESUME_TRAIN = True, 加载之前的模型继续训练
    "CHECKPOINT_PATH": r"./yolo11_pt/yolo11s_roi12_ps_6.pt"  # 之前保存的模型路径
}
train_config_7 = {
    "BATCH_SIZE": 32,  # 加载图像的批次
    "EPOCHS": 100,  # 训练总轮数
    "patience": 12,  # 耐心
    "mixup_rate": 0.2,  # mixup触发的概率
    "mixup_alpha": 0.2,  # mixup增强的beta 分布参数

    "WORKERS": 8,
    "RESUME_TRAIN": False,
    # RESUME_TRAIN = False, 从头训练
    "SAVE_DIR": "./yolo11_pt",  # 输出的模型路径
    "MODEL_NAME": "yolo11n_roi12_atten_20.pt",
    # RESUME_TRAIN = True, 加载之前的模型继续训练
    "CHECKPOINT_PATH": r"./yolo11_pt/yolo11s_roi12_ps_6.pt"  # 之前保存的模型路径
}