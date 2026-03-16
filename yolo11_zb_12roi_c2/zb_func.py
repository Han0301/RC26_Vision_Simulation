import cv2
import numpy as np
import math
import threading
from typing import List, Tuple
import open3d as o3d

# -------------------------- 全局常量定义 --------------------------
_L_ = 1.2                # 台阶长度
_H_ = 0.2                # 台阶高度
_lx1_ = 0.425            # 台阶到方块的间距
_ly1_ = 0.425            # 台阶到方块的间距
_lh_ = 0.35              # 方块的长度
_X_ = 3.2                # 初始位置到梅花林1号位置边角的x轴距离
_Y_ = -1.2               # 初始位置到梅花林1号位置边角的y轴距离
_offset_x_ = 0           # x方向偏移
_offset_y_ = 0           # y方向偏移
_offset_z_ = 0           # z方向偏移
FLT_MAX = np.finfo(np.float32).max
EPS = 1e-6

# -------------------------- 相机参数定义 --------------------------
class CameraInfo:
        # 相机内参（固定值）
        K_ = np.array([[1012.0711525658555, 0, 960.5],
                            [0, 1012.0711525658555, 540.5],
                            [0, 0, 1]], dtype=np.float64)
        distCoeffs_ = np.zeros((5, 1), dtype=np.float64)  # 无畸变
        extrinsic_ = np.eye(4)  # 外参矩阵（R+T）
        revc_ = np.zeros((3, 1), dtype=np.float32)  # 旋转向量（对应C++ rvec_）
        tevc_ = np.zeros((3, 1), dtype=np.float32)  # 平移向量（对应C++ tvec_）
        R_ = np.eye(3)
        T_ = np.zeros(3)
        mtx_ = threading.Lock()  # 互斥锁（类成员，对应C++ mtx_）

# -------------------------- 数据结构定义 --------------------------
class Surface2DPoint:
    def __init__(self):
        self.idx = 0
        self.left_up = (0.0, 0.0)
        self.right_up = (0.0, 0.0)
        self.right_down = (0.0, 0.0)
        self.left_down = (0.0, 0.0)
        self.surface_depth = 0.0

class Box:
    def __init__(self):
        self.idx = 0
        self.roi_image = np.zeros((160, 160, 3), dtype=np.uint8)
        self.cls = 0
        self.confidence = 0.0
        self.zbuffer_flag = 0  # 0:未处理, 1:已处理, -1:异常
        self.exist_flag = -1  # 0:空, 1:有方块, -1:未处理

# -------------------------- 3D Box初始化类（完全对齐C++的3D点生成） --------------------------
class Init3DBox:
    def __init__(self):
        # 预定义的3D世界点（核心，无需里程计）
        self.W_object_plum_points_ = []
        self.C_object_plum_points_ = []  # 相机坐标系3D点

        # PCL点云（用于转换）
        self.pcl_LM_plum_object_points_ = o3d.geometry.PointCloud()  # 仅作为输入载体
        self.pcl_C_plum_object_points_ = o3d.geometry.PointCloud()

        # 2D像素点（输出）
        self.object_plum_2d_points_ = []

        # Box列表（用于ZBuffer输出）
        self.box_lists_ = [Box() for _ in range(12)]
        for i in range(12):
            self.box_lists_[i].idx = i + 1
            self.box_lists_[i].roi_image = np.zeros((160, 160, 3), dtype=np.uint8)

        # 初始化3D世界点（完全复刻C++逻辑）
        self._init_3d_points()

    def _init_3d_points(self):
        """初始化预定义的3D世界点（完全对齐C++代码）"""
        self.W_object_plum_points_ = []
        _arr = [0.4, 0.2, 0.4, 0.2, 0.4, 0.6, 0.4, 0.6, 0.4, 0.2, 0.4, 0.2]  # 对应C++ arr_

        # -------------------------- 方块3D点（完全复刻C++） --------------------------
        for j in range(4):
            for i in range(3):
                # 计算基础坐标（对齐C++的每个点公式）
                base_x = _X_ + j * _L_ + _lx1_ + _offset_x_
                base_y = _Y_ - i * _L_ - _ly1_ + _offset_y_
                arr_val = _arr[i * 3 + j]  # 核心修正：C++是i*3+j，不是j*3+i
                z_h = arr_val + _lh_ + _offset_z_
                z_base = arr_val + _offset_z_

                # 8个方块点（完全对应C++的8个点顺序）
                self.W_object_plum_points_.extend([
                    (base_x, base_y, z_h),                                    # 0
                    (base_x, base_y - _lh_, z_h),                             # 1 (Y_ -i*L_ -ly1_ -lh_)
                    (base_x, base_y - _lh_, z_base),                          # 2
                    (base_x, base_y, z_base),                                 # 3
                    (base_x + _lh_, base_y, z_h),                             # 4 (X_ +j*L_ +lx1_ +lh_)
                    (base_x + _lh_, base_y - _lh_, z_h),                      # 5
                    (base_x + _lh_, base_y - _lh_, z_base),                   # 6
                    (base_x + _lh_, base_y, z_base)                           # 7
                ])

        # -------------------------- 台阶3D点（完全复刻C++） --------------------------
        for j in range(4):
            for i in range(3):
                # 计算基础坐标
                base_x = _X_ + j * _L_ + _offset_x_
                base_y = _Y_ - i * _L_ + _offset_y_
                arr_val = _arr[i * 3 + j]
                z_val = arr_val + _offset_z_
                z_zero = 0 + _offset_z_

                # 8个台阶点（完全对应C++的8个点顺序）
                self.W_object_plum_points_.extend([
                    (base_x, base_y, z_val),                                 # 0
                    (base_x, base_y - _L_, z_val),                            # 1 (Y_ -i*L_ -L_)
                    (base_x, base_y - _L_, z_zero),                           # 2
                    (base_x, base_y, z_zero),                                 # 3
                    (base_x + _L_, base_y, z_val),                            # 4 (X_ +j*L_ +L_)
                    (base_x + _L_, base_y - _L_, z_val),                       # 5
                    (base_x + _L_, base_y - _L_, z_zero),                      # 6
                    (base_x + _L_, base_y, z_zero)                            # 7
                ])

        # 填充PCL点云（作为转换输入）
        self.pcl_LM_plum_object_points_.points = o3d.utility.Vector3dVector(self.W_object_plum_points_)

        # 初始化2D点列表（96个方块点 + 96个台阶点 = 192个点）
        self.object_plum_2d_points_ = [(0.0, 0.0) for _ in range(192)]

    def pcl_to_C(self):
        """PCL点云转相机坐标系3D点列表"""
        self.C_object_plum_points_ = np.asarray(self.pcl_C_plum_object_points_.points).tolist()


# -------------------------- 遮挡处理类 --------------------------
class OcclusionHandling:
    def __init__(self):
        self.exist_boxes_ = [1] * 12  # 默认所有方块存在
        self.interested_boxes_ = [1] * 12  # 默认所有方块感兴趣
        self.mtx_ = threading.Lock()

    def set_exist_boxes(self, exist_boxes: List[int]):
        with self.mtx_:
            if len(exist_boxes) == 12:
                self.exist_boxes_ = exist_boxes.copy()

    def set_interested_boxes(self, interested_boxes: List[int]):
        with self.mtx_:
            if len(interested_boxes) == 12:
                self.interested_boxes_ = interested_boxes.copy()

    def cal_distance(self, p1: Tuple[float, float, float], p2: Tuple[float, float, float],
                     p3: Tuple[float, float, float], p4: Tuple[float, float, float]) -> float:
        """计算四个3D点的平均深度"""
        def depth(p):
            return math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2)
        total_depth = depth(p1) + depth(p2) + depth(p3) + depth(p4)
        return total_depth / 4.0

    def set_surface_2d_point(self, C_points: List[Tuple[float, float, float]],
                             object_2d: List[Tuple[float, float]],
                             surface_2d: List[Surface2DPoint], label: str):
        """根据深度更新2D表面点（适配C++的点数量）"""
        j = 96 if label == "plum" else 0  # 96个方块点 + 96个台阶点
        for i in range(0, 96, 8):
            idx = i // 8 + 1
            # 计算各面深度
            front_depth = self.cal_distance(
                C_points[j + i], C_points[j + i + 1], C_points[j + i + 2], C_points[j + i + 3]
            )
            back_depth = self.cal_distance(
                C_points[j + i + 4], C_points[j + i + 5], C_points[j + i + 6], C_points[j + i + 7]
            )
            left_depth = self.cal_distance(
                C_points[j + i + 4], C_points[j + i], C_points[j + i + 3], C_points[j + i + 7]
            )
            right_depth = self.cal_distance(
                C_points[j + i + 1], C_points[j + i + 5], C_points[j + i + 6], C_points[j + i + 2]
            )
            up_depth = self.cal_distance(
                C_points[j + i + 4], C_points[j + i + 5], C_points[j + i + 1], C_points[j + i]
            )
            down_depth = self.cal_distance(
                C_points[j + i + 6], C_points[j + i + 7], C_points[j + i + 3], C_points[j + i + 2]
            )

            # 选择更近的面
            surf = Surface2DPoint()
            surf.idx = idx
            if front_depth < back_depth:
                surf.left_up = object_2d[j + i]
                surf.right_up = object_2d[j + i + 1]
                surf.right_down = object_2d[j + i + 2]
                surf.left_down = object_2d[j + i + 3]
                surf.surface_depth = front_depth
            else:
                surf.left_up = object_2d[j + i + 4]
                surf.right_up = object_2d[j + i + 5]
                surf.right_down = object_2d[j + i + 6]
                surf.left_down = object_2d[j + i + 7]
                surf.surface_depth = back_depth
            surface_2d.append(surf)

            surf = Surface2DPoint()
            surf.idx = idx
            if left_depth < right_depth:
                surf.left_up = object_2d[j + i + 4]
                surf.right_up = object_2d[j + i]
                surf.right_down = object_2d[j + i + 3]
                surf.left_down = object_2d[j + i + 7]
                surf.surface_depth = left_depth
            else:
                surf.left_up = object_2d[j + i + 1]
                surf.right_up = object_2d[j + i + 5]
                surf.right_down = object_2d[j + i + 6]
                surf.left_down = object_2d[j + i + 2]
                surf.surface_depth = right_depth
            surface_2d.append(surf)

            surf = Surface2DPoint()
            surf.idx = idx
            if up_depth < down_depth:
                surf.left_up = object_2d[j + i + 4]
                surf.right_up = object_2d[j + i + 5]
                surf.right_down = object_2d[j + i + 1]
                surf.left_down = object_2d[j + i]
                surf.surface_depth = up_depth
            else:
                surf.left_up = object_2d[j + i + 6]
                surf.right_up = object_2d[j + i + 7]
                surf.right_down = object_2d[j + i + 3]
                surf.left_down = object_2d[j + i + 2]
                surf.surface_depth = down_depth
            surface_2d.append(surf)

    def set_all_outside(self, front_2d: Surface2DPoint, side_2d: Surface2DPoint,
                        up_2d: Surface2DPoint, cols: int, rows: int, all_points: List[Tuple[float, float]]) -> bool:
        """判断所有点是否在图像外"""
        all_points.clear()
        all_points.extend([
            front_2d.left_up, front_2d.right_up, front_2d.right_down, front_2d.left_down,
            side_2d.left_up, side_2d.right_up, side_2d.right_down, side_2d.left_down,
            up_2d.left_up, up_2d.right_up, up_2d.right_down, up_2d.left_down
        ])
        for pt in all_points:
            if 0 <= pt[0] < cols and 0 <= pt[1] < rows:
                return False
        return True

    def set_temp(self, front_2d: Surface2DPoint, side_2d: Surface2DPoint,
                 up_2d: Surface2DPoint, temp: np.ndarray):
        """填充深度到临时矩阵"""
        # 构建轮廓
        front_contour = np.array([
            [round(front_2d.left_up[0]), round(front_2d.left_up[1])],
            [round(front_2d.right_up[0]), round(front_2d.right_up[1])],
            [round(front_2d.right_down[0]), round(front_2d.right_down[1])],
            [round(front_2d.left_down[0]), round(front_2d.left_down[1])]
        ], np.int32)
        side_contour = np.array([
            [round(side_2d.left_up[0]), round(side_2d.left_up[1])],
            [round(side_2d.right_up[0]), round(side_2d.right_up[1])],
            [round(side_2d.right_down[0]), round(side_2d.right_down[1])],
            [round(side_2d.left_down[0]), round(side_2d.left_down[1])]
        ], np.int32)
        up_contour = np.array([
            [round(up_2d.left_up[0]), round(up_2d.left_up[1])],
            [round(up_2d.right_up[0]), round(up_2d.right_up[1])],
            [round(up_2d.right_down[0]), round(up_2d.right_down[1])],
            [round(up_2d.left_down[0]), round(up_2d.left_down[1])]
        ], np.int32)

        # 填充深度
        cv2.fillPoly(temp, [front_contour], front_2d.surface_depth)
        cv2.fillPoly(temp, [side_contour], side_2d.surface_depth)
        cv2.fillPoly(temp, [up_contour], up_2d.surface_depth)

    def cal_points_range(self, all_points: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
        """计算点集的像素范围"""
        x_min = min([p[0] for p in all_points])
        y_min = min([p[1] for p in all_points])
        x_max = max([p[0] for p in all_points])
        y_max = max([p[1] for p in all_points])
        return x_min, y_min, x_max, y_max

    def is_update_image(self, box_lists: List[Box], valid_max_points: List[Tuple[float, float]],
                        interested_boxes: List[int], i: int) -> bool:
        """判断是否更新ROI图像"""
        if not valid_max_points or len(valid_max_points) <= 600:
            return False
        if interested_boxes[i // 3] == 0:
            return False
        if box_lists[i // 3].zbuffer_flag == -1:
            return False
        return True

    def set_box_lists_(self, image: np.ndarray, C_points: List[Tuple[float, float, float]],
                       object_2d: List[Tuple[float, float]], box_lists: List[Box]):
        """核心ZBuffer处理：更新box列表"""
        with self.mtx_:
            exist_boxes = self.exist_boxes_.copy()
            interested_boxes = self.interested_boxes_.copy()

        # 生成表面2D点
        object_2d_surf = []
        plum_2d_surf = []
        self.set_surface_2d_point(C_points, object_2d, object_2d_surf, "object")
        self.set_surface_2d_point(C_points, object_2d, plum_2d_surf, "plum")

        if len(object_2d_surf) != 36 or len(plum_2d_surf) != 36:
            print(f"表面2D点数量错误，需为36个，当前object={len(object_2d_surf)}, plum={len(plum_2d_surf)}")
            return

        # 初始化ZBuffer
        zbuffer = np.ones((image.shape[0], image.shape[1]), dtype=np.float32) * FLT_MAX
        object_zbuffer = np.ones((image.shape[0], image.shape[1]), dtype=np.float32) * FLT_MAX

        # 填充台阶深度
        for i in range(0, len(plum_2d_surf), 3):
            p_front = plum_2d_surf[i]
            p_side = plum_2d_surf[i + 1]
            p_up = plum_2d_surf[i + 2]

            # 判断是否全在图像外
            all_points = []
            if self.set_all_outside(p_front, p_side, p_up, image.shape[1], image.shape[0], all_points):
                continue

            # 填充台阶深度到临时矩阵
            plum_temp = np.ones_like(zbuffer) * FLT_MAX
            self.set_temp(p_front, p_side, p_up, plum_temp)

            # 计算像素范围
            x_min, y_min, x_max, y_max = self.cal_points_range(all_points)

            # 更新ZBuffer
            for row in range(max(int(y_min) - 1, 0), min(int(y_max) + 1, image.shape[0])):
                for col in range(max(int(x_min) - 1, 0), min(int(x_max) + 1, image.shape[1])):
                    if plum_temp[row, col] < zbuffer[row, col]:
                        zbuffer[row, col] = plum_temp[row, col]

        # 填充方块深度并裁剪ROI
        for i in range(0, len(object_2d_surf), 3):
            o_front = object_2d_surf[i]
            o_side = object_2d_surf[i + 1]
            o_up = object_2d_surf[i + 2]

            # 判断是否全在图像外
            all_points = []
            if self.set_all_outside(o_front, o_side, o_up, image.shape[1], image.shape[0], all_points):
                continue

            # 填充方块深度到临时矩阵
            object_temp = np.ones_like(zbuffer) * FLT_MAX
            self.set_temp(o_front, o_side, o_up, object_temp)

            # 计算像素范围
            x_min, y_min, x_max, y_max = self.cal_points_range(all_points)

            # 更新ZBuffer并收集深度区域
            depth_regions = {}
            if exist_boxes[int(i / 3)]:
                for row in range(max(int(y_min) - 1, 0), min(int(y_max) + 1, image.shape[0])):
                    for col in range(max(int(x_min) - 1, 0), min(int(x_max) + 1, image.shape[1])):
                        if object_temp[row, col] == FLT_MAX:
                            continue
                        if object_temp[row, col] < zbuffer[row, col]:
                            zbuffer[row, col] = object_temp[row, col]
                            object_zbuffer[row, col] = object_temp[row, col]
                        # 收集深度区域
                        depth = object_zbuffer[row, col]
                        if depth not in depth_regions:
                            depth_regions[depth] = []
                        depth_regions[depth].append((col, row))
            else:
                for row in range(max(int(y_min) - 1, 0), min(int(y_max) + 1, image.shape[0])):
                    for col in range(max(int(x_min) - 1, 0), min(int(x_max) + 1, image.shape[1])):
                        if object_temp[row, col] == FLT_MAX:
                            continue
                        if object_temp[row, col] < zbuffer[row, col]:
                            object_zbuffer[row, col] = object_temp[row, col]
                        # 收集深度区域
                        depth = object_zbuffer[row, col]
                        if depth not in depth_regions:
                            depth_regions[depth] = []
                        depth_regions[depth].append((col, row))

            # 找到有效最大点集
            max_count = -1
            valid_max_points = []
            valid_depths = [o_front.surface_depth, o_up.surface_depth, o_side.surface_depth]
            for depth, points in depth_regions.items():
                if any(abs(depth - vd) < 1e-4 for vd in valid_depths) and len(points) > max_count:
                    max_count = len(points)
                    valid_max_points = points

            # 判断是否更新图像: 有效像素过少, 不感兴趣会不更新图像
            if not self.is_update_image(box_lists, valid_max_points, interested_boxes, i):
                continue

            # 生成ROI掩码
            x_min = min([p[0] for p in valid_max_points])
            x_max = max([p[0] for p in valid_max_points])
            y_min = min([p[1] for p in valid_max_points])
            y_max = max([p[1] for p in valid_max_points])

            roi_mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)
            for (col, row) in valid_max_points:
                if 0 <= row < roi_mask.shape[0] and 0 <= col < roi_mask.shape[1]:
                    roi_mask[int(row), int(col)] = 255

            # 裁剪ROI
            roi_rect = cv2.boundingRect(roi_mask)
            x, y, w, h = roi_rect
            if w <= 0 or h <= 0 or x + w > image.shape[1] or y + h > image.shape[0]:
                continue

            # 提取ROI
            image_roi = image[y:y + h, x:x + w]
            mask_roi = roi_mask[y:y + h, x:x + w]
            crop_roi = np.zeros_like(image_roi)
            crop_roi[mask_roi > 0] = image_roi[mask_roi > 0]

            # 转为正方形
            max_side = max(w, h)
            square_roi = np.zeros((max_side, max_side, 3), dtype=np.uint8)
            x_offset = (max_side - w) // 2
            y_offset = (max_side - h) // 2
            square_roi[y_offset:y_offset + h, x_offset:x_offset + w] = crop_roi

            # 更新box列表
            box_lists[i // 3].roi_image = cv2.resize(square_roi, (160, 160))
            box_lists[i // 3].zbuffer_flag = 1

    def update_debug_image(self, image: np.ndarray, object_2d: List[Tuple[float, float]]) -> np.ndarray:
        """绘制调试图像（绘制3D点对应的2D轮廓）"""
        if image is None or len(image) == 0:
            print("图像为空，跳过绘制")
            return np.array([])

        img = image.copy()
        for i in range(0, len(object_2d), 8):
            if i < 96:  # 仅绘制方块的2D轮廓
                pts = np.array([
                    [round(object_2d[i][0]), round(object_2d[i][1])],
                    [round(object_2d[i + 1][0]), round(object_2d[i + 1][1])],
                    [round(object_2d[i + 2][0]), round(object_2d[i + 2][1])],
                    [round(object_2d[i + 3][0]), round(object_2d[i + 3][1])]
                ], np.int32)
                cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        return img

    def set_debug_roi_image(self, box_lists: List[Box], debug_roi_image: np.ndarray):
        """拼接12个ROI为640x480大图"""
        SINGLE_SIZE = 160
        COL_NUM = 4
        ROW_NUM = 3
        # 初始化12个160x160黑图
        roi_images = [np.zeros((SINGLE_SIZE, SINGLE_SIZE, 3), dtype=np.uint8) for _ in range(12)]

        # 填充有效ROI
        for idx in range(1, 13):
            vec_idx = idx - 1
            for box in box_lists:
                if box.idx == idx and box.roi_image is not None and len(box.roi_image) > 0:
                    roi_images[vec_idx] = cv2.resize(box.roi_image, (SINGLE_SIZE, SINGLE_SIZE))
                    break

        # 拼接成大图
        for row in range(ROW_NUM):
            for col in range(COL_NUM):
                vec_idx = row * COL_NUM + col
                if vec_idx >= 12:
                    break
                x = col * SINGLE_SIZE
                y = row * SINGLE_SIZE
                debug_roi_image[y:y + SINGLE_SIZE, x:x + SINGLE_SIZE] = roi_images[vec_idx]