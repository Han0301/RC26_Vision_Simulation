import cv2
import numpy as np
from zb_func import Init3DBox, OcclusionHandling
from transform import WorldToCamera

# -------------------------- 核心处理函数（完全移除里程计） --------------------------
def process_zbuffer_with_rt(image: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, exist_boxes: list) -> list:
    """
    核心函数：输入图像 + R/T外参 → 3D世界点转2D + ZBuffer处理
    无里程计依赖，直接基于预定义3D世界点计算
    """
    # 1. 输入参数校验
    if image is None or len(image.shape) != 3:
        print("错误：输入图像格式无效，需为HWC格式的np.ndarray")
        return
    if rvec.shape != (3, 1):
        print("错误：旋转矩阵R需为3x3维度")
        return
    if tvec.size != 3:
        print("错误：平移向量T需为3维向量")
        return

    # 2. 初始化核心组件
    camera_trans = WorldToCamera()  # 3D→2D转换类
    init_3d_box = Init3DBox()       # 预定义3D世界点
    occlusion_handler = OcclusionHandling()  # ZBuffer处理

    # 5. 核心转换：3D世界点 → 2D像素点
    camera_trans.world_to_camera(
        init_3d_box.pcl_LM_plum_object_points_,  # 输入：预定义3D世界点
        rvec, tvec,
        init_3d_box.pcl_C_plum_object_points_,  # 输出：相机坐标系3D点
        init_3d_box.object_plum_2d_points_  # 输出：2D像素点
    )

    init_3d_box.pcl_to_C()  # 提取相机坐标系3D点列表

    # 6. ZBuffer遮挡处理（生成ROI）
    occlusion_handler.set_exist_boxes(exist_boxes)
    occlusion_handler.set_box_lists_(
        image,
        init_3d_box.C_object_plum_points_,
        init_3d_box.object_plum_2d_points_,
        init_3d_box.box_lists_
    )

    # 7. 生成调试图像 -------------------可注释-----------------
    # debug_image = occlusion_handler.update_debug_image(
    #     image,
    #     init_3d_box.object_plum_2d_points_
    # )
    # debug_best_roi_image =  np.zeros((480, 640, 3), dtype=np.uint8)
    # occlusion_handler.set_debug_roi_image(
    #     init_3d_box.box_lists_,
    #     debug_best_roi_image
    # )
    #
    # if debug_image.size > 0:
    #     cv2.imshow("3D Points 2D Contour", debug_image)
    # if debug_best_roi_image.size > 0:
    #     cv2.imshow("12 ROI Images (ZBuffer)", debug_best_roi_image)
    #
    # print("提示：按任意键关闭图像窗口")
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # 7. 生成调试图像 -------------------可注释-----------------

    roi_images = []
    for i in range(len(init_3d_box.box_lists_)):
        roi_images.append(init_3d_box.box_lists_[i].roi_image)
    return roi_images


# -------------------------- 批量ROI生成函数（新增核心） --------------------------
def process_zbuffer_with_rt_batch(global_imgs_np, rvecs, tvecs, exist_boxes_batch):
    """
    批量版本的ZBuffer ROI生成
    输入：
        global_imgs_np: (batch_size, H, W, C) 批量全局图像（np.uint8）
        rvecs: (batch_size, 3, 1) 批量旋转向量
        tvecs: (batch_size, 3, 1) 批量平移向量
        exist_boxes_batch: (batch_size, 12) 批量存在标记
    输出：
        batch_roi_imgs: (batch_size, 12, 160, 160, 3) 批量ROI图像
    """
    batch_size = len(global_imgs_np)
    batch_roi_imgs = []

    for b in range(batch_size):
        # 单样本ROI生成（复用原有逻辑）
        roi_imgs = process_zbuffer_with_rt(
            global_imgs_np[b],
            rvecs[b],
            tvecs[b],
            exist_boxes_batch[b].tolist()
        )
        batch_roi_imgs.append(roi_imgs)

    # 转换为numpy数组，形状：(batch_size, 12, 160, 160, 3)
    batch_roi_np = np.stack(batch_roi_imgs, axis=0)
    return batch_roi_np

# -------------------------- 测试主函数 --------------------------
if __name__ == "__main__":
    # 1. 加载图像（替换为你的路径）
    IMAGE_PATH = r"H:\pycharm\yolov11\yolov11_proj3\Datasets_Global_map400\global_images\images_6995.png"
    test_image = cv2.imread(IMAGE_PATH)

    # 2. 输入R/T外参（示例：旋转向量转矩阵 + 平移向量）
    rvec = np.array([1.276089, -1.098547, 1.098540], dtype=np.float32).reshape(3, 1)

    tvec = np.array([-3.783217, 1.300021, -0.331593], dtype=np.float32).reshape(3, 1)
    exist_boxes = [1 * 12]
    # exist_boxes = [1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1]

    # 3. 执行核心处理（无里程计，直接3D→2D）

    roi_imgs = process_zbuffer_with_rt(test_image, rvec, tvec, exist_boxes)