import multiprocessing
from train_main import train
from train_config import model_config, train_config, loss_config,loss_config_2, train_config_5,dataset_config, model_config_2, train_config_2,dataset_config_3,dataset_config_4,train_config_3,train_config_4
from infer_main import YOLO11ROIInferencer
from show_atten import show_atten_single,show_atten_datasets


if __name__ == '__main__':
    multiprocessing.freeze_support()

    # 第一次训练: 修改 "LOSS_WEIGHT": [2.0, 1.0], 降低了0类损失
    train(model_config, train_config, dataset_config, loss_config)

    # 第二次训练: 修改 "LOSS_WEIGHT": [2.0, 1.0], 降低了0类损失, 修改模型为 n 大小
    train(model_config_2, train_config_2, dataset_config, loss_config)

    # 第三次训练: 修改 数据集, 加入现实数据样本p179
    train(model_config, train_config_3, dataset_config_3, loss_config)

    # 第四次训练: 修改 数据集, 加入现实数据样本p423
    train(model_config, train_config_4, dataset_config_4, loss_config)

    # 第五次训练: 修改 数据集, 加入现实数据样本p423, 修改"LOSS_WEIGHT": [3.0, 1.0]
    train(model_config, train_config_5, dataset_config_4, loss_config_2)

    inferencer_1 = YOLO11ROIInferencer(
        model_path="H:\pycharm\yolov11\yolov11_proj3\yolo11Custom_atten\yolo11_pt\yolo11n_roi12_atten_8.pt",
        dataset_root=None,
        model_size="s",
        roi_size=64,
        num_roi=12,
        num_classes=2
    )

    inferencer_1.infer_datasets(datasets_path=r"H:\pycharm\yolov11\yolov11_proj3\datasets_real_p179",
                                is_conf=True,
                              conf_list=[0.9,0.85,0.80,0.75,0.7,0.65,0.6],
                              is_place=True,
                              is_point_size_weight=True,
                              ps_w_thods=[0.4,0.3,0.2,0.1],
                              is_save=True,
                              save_path=r"H:\pycharm\yolov11\yolov11_proj3\yolo11Custom_atten\error\atten8_real_p179.csv"
                              )

    inferencer_1.infer_datasets(datasets_path=r"H:\pycharm\yolov11\yolov11_proj3\datasets_test_2520",
                                is_conf=True,
                              conf_list=[0.9,0.85,0.80,0.75,0.7,0.65,0.6],
                              is_place=True,
                              is_point_size_weight=True,
                              ps_w_thods=[0.4,0.3,0.2,0.1],
                              is_save=True,
                              save_path=r"H:\pycharm\yolov11\yolov11_proj3\yolo11Custom_atten\error\atten8_test_2520.csv"
                              )

    inferencer_2 = YOLO11ROIInferencer(
        model_path="H:\pycharm\yolov11\yolov11_proj3\yolo11Custom_atten\yolo11_pt\yolo11n_roi12_atten_9.pt",
        dataset_root=None,
        model_size="n",
        roi_size=64,
        num_roi=12,
        num_classes=2
    )

    inferencer_2.infer_datasets(datasets_path=r"H:\pycharm\yolov11\yolov11_proj3\datasets_real_p179",
                                is_conf=True,
                              conf_list=[0.9,0.85,0.80,0.75,0.7,0.65,0.6],
                              is_place=True,
                              is_point_size_weight=True,
                              ps_w_thods=[0.4,0.3,0.2,0.1],
                              is_save=True,
                              save_path=r"H:\pycharm\yolov11\yolov11_proj3\yolo11Custom_atten\error\atten9_real_p179.csv"
                              )

    inferencer_2.infer_datasets(datasets_path=r"H:\pycharm\yolov11\yolov11_proj3\datasets_test_2520",
                                is_conf=True,
                              conf_list=[0.9,0.85,0.80,0.75,0.7,0.65,0.6],
                              is_place=True,
                              is_point_size_weight=True,
                              ps_w_thods=[0.4,0.3,0.2,0.1],
                              is_save=True,
                              save_path=r"H:\pycharm\yolov11\yolov11_proj3\yolo11Custom_atten\error\atten9_test_2520.csv"
                              )

    inferencer_3 = YOLO11ROIInferencer(
        model_path="H:\pycharm\yolov11\yolov11_proj3\yolo11Custom_atten\yolo11_pt\yolo11n_roi12_atten_10.pt",
        dataset_root=None,
        model_size="s",
        roi_size=64,
        num_roi=12,
        num_classes=2
    )

    inferencer_3.infer_datasets(datasets_path=r"H:\pycharm\yolov11\yolov11_proj3\datasets_real_p179",
                                is_conf=True,
                              conf_list=[0.9,0.85,0.80,0.75,0.7,0.65,0.6],
                              is_place=True,
                              is_point_size_weight=True,
                              ps_w_thods=[0.4,0.3,0.2,0.1],
                              is_save=True,
                              save_path=r"H:\pycharm\yolov11\yolov11_proj3\yolo11Custom_atten\error\atten10_real_p179.csv"
                              )

    inferencer_3.infer_datasets(datasets_path=r"H:\pycharm\yolov11\yolov11_proj3\datasets_test_2520",
                                is_conf=True,
                              conf_list=[0.9,0.85,0.80,0.75,0.7,0.65,0.6],
                              is_place=True,
                              is_point_size_weight=True,
                              ps_w_thods=[0.4,0.3,0.2,0.1],
                              is_save=True,
                              save_path=r"H:\pycharm\yolov11\yolov11_proj3\yolo11Custom_atten\error\atten10_test_2520.csv"
                              )

    inferencer_4 = YOLO11ROIInferencer(
        model_path="H:\pycharm\yolov11\yolov11_proj3\yolo11Custom_atten\yolo11_pt\yolo11n_roi12_atten_11.pt",
        dataset_root=None,
        model_size="s",
        roi_size=64,
        num_roi=12,
        num_classes=2
    )

    inferencer_4.infer_datasets(datasets_path=r"H:\pycharm\yolov11\yolov11_proj3\datasets_real_p179",
                                is_conf=True,
                              conf_list=[0.9,0.85,0.80,0.75,0.7,0.65,0.6],
                              is_place=True,
                              is_point_size_weight=True,
                              ps_w_thods=[0.4,0.3,0.2,0.1],
                              is_save=True,
                              save_path=r"H:\pycharm\yolov11\yolov11_proj3\yolo11Custom_atten\error\atten11_real_p179.csv"
                              )


    inferencer_4.infer_datasets(datasets_path=r"H:\pycharm\yolov11\yolov11_proj3\datasets_test_2520",
                                is_conf=True,
                              conf_list=[0.9,0.85,0.80,0.75,0.7,0.65,0.6],
                              is_place=True,
                              is_point_size_weight=True,
                              ps_w_thods=[0.4,0.3,0.2,0.1],
                              is_save=True,
                              save_path=r"H:\pycharm\yolov11\yolov11_proj3\yolo11Custom_atten\error\atten11_test_2520.csv"
                              )

    inferencer_5 = YOLO11ROIInferencer(
        model_path="H:\pycharm\yolov11\yolov11_proj3\yolo11Custom_atten\yolo11_pt\yolo11n_roi12_atten_12.pt",
        dataset_root=None,
        model_size="s",
        roi_size=64,
        num_roi=12,
        num_classes=2
    )

    inferencer_5.infer_datasets(datasets_path=r"H:\pycharm\yolov11\yolov11_proj3\datasets_real_p179",
                                is_conf=True,
                              conf_list=[0.9,0.85,0.80,0.75,0.7,0.65,0.6],
                              is_place=True,
                              is_point_size_weight=True,
                              ps_w_thods=[0.4,0.3,0.2,0.1],
                              is_save=True,
                              save_path=r"H:\pycharm\yolov11\yolov11_proj3\yolo11Custom_atten\error\atten12_real_p179.csv"
                              )


    inferencer_5.infer_datasets(datasets_path=r"H:\pycharm\yolov11\yolov11_proj3\datasets_test_2520",
                                is_conf=True,
                              conf_list=[0.9,0.85,0.80,0.75,0.7,0.65,0.6],
                              is_place=True,
                              is_point_size_weight=True,
                              ps_w_thods=[0.4,0.3,0.2,0.1],
                              is_save=True,
                              save_path=r"H:\pycharm\yolov11\yolov11_proj3\yolo11Custom_atten\error\atten12_test_2520.csv"
                              )




