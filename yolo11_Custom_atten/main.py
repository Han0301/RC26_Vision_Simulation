import multiprocessing
from train_main import train
from train_config import model_config, train_config, loss_config, loss_config_2, train_config_5, dataset_config, \
    model_config_2, train_config_2, dataset_config_3, dataset_config_4, train_config_3, train_config_4, \
    dataset_config_5, model_config_3, train_config_6, train_config_7
from infer_main import YOLO11ROIInferencer
from show_atten import show_atten_single,show_atten_datasets


if __name__ == '__main__':
    multiprocessing.freeze_support()

    # # # 第五次训练: 修改 数据集, p423 * 5
    # train(model_config, train_config, dataset_config_5, loss_config_2)


    # train(model_config, train_config_2, dataset_config_4, loss_config_2)

    inferencer_5 = YOLO11ROIInferencer(
        model_path=r"H:\pycharm\yolov11\yolov11_proj4\yolo11Custom_atten\yolo11_pt\roi12_atten_blue17_.pt",
        dataset_root=None,
        model_size="s",
        roi_size=64,
        num_roi=12,
        num_classes=2
    )

    inferencer_5.infer_datasets(datasets_path=r"I:\datasets_real_blue_new785",
                                is_conf=True,
                              conf_list=[0.9,0.85,0.80,0.75,0.7,0.65,0.6],
                              is_place=True,
                              is_point_size_weight=True,
                              ps_w_thods=[0.4,0.3,0.2,0.1],
                              is_save=True,
                              save_path=r"H:\pycharm\yolov11\yolov11_proj4\yolo11Custom_atten\error\atten17_blue_real_new389.csv"
                              )

    inferencer_5.infer_datasets(datasets_path=r"H:\pycharm\yolov11\yolov11_proj3\datasets_blue_mapnew250",
                                is_conf=True,
                              conf_list=[0.9,0.85,0.80,0.75,0.7,0.65,0.6],
                              is_place=True,
                              is_point_size_weight=True,
                              ps_w_thods=[0.4,0.3,0.2,0.1],
                              is_save=True,
                              save_path=r"H:\pycharm\yolov11\yolov11_proj4\yolo11Custom_atten\error\atten17_blue_test253"
                                        r".csv"
                              )