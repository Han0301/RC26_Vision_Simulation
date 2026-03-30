import cv2
import numpy as np
import threading
from typing import List, Tuple
import open3d as o3d

from zb_func import CameraInfo

def combineRotationAndTranslation(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """正确构建：世界坐标系→相机坐标系的4x4外参矩阵"""
    transform = np.eye(4, dtype=np.float64)
    # transform[:3, :3] = rotation.T  # 旋转矩阵转置（反向）
    # transform[:3, 3] = -np.dot(rotation.T, translation)  # 平移取反（反向）
    transform[:3, :3] = rotation  # 直接用世界→相机的旋转矩阵（不转置）
    transform[:3, 3] = translation  # 直接用世界→相机的平移向量（不取反）
    return transform

# -------------------------- 坐标转换类（简化：直接3D世界→相机→2D） --------------------------
class WorldToCamera:
    def __init__(self):
        self.camerainfo_ = CameraInfo()
        self.mtx_ = threading.Lock()

    def world_to_camera(self, world_cloud: o3d.geometry.PointCloud,
                                      rvec: np.ndarray, tvec: np.ndarray,
                                      camera_cloud: o3d.geometry.PointCloud,
                                      object_2d_points: List[Tuple[float, float]]):

        with self.mtx_:
            # 1. 预处理：确保rvec/tvec形状正确（(3,1)→(3,)）
            rvec = rvec.squeeze()  # 旋转向量转为一维
            tvec = tvec.squeeze()  # 平移向量转为一维

            # 2. 旋转向量转旋转矩阵
            R, _ = cv2.Rodrigues(rvec)

            # 3. 组合4x4外参矩阵（世界→相机）
            extrinsic_matrix = combineRotationAndTranslation(R, tvec)

            # 4. 世界点云 → 相机坐标系点云
            camera_cloud.clear()
            temp_cloud = o3d.geometry.PointCloud(world_cloud)  # 不改变world_cloud
            temp_cloud.transform(extrinsic_matrix)
            camera_cloud.points = temp_cloud.points

            # 5. 提取3D点
            object_3d_points = []
            for point in world_cloud.points:
                obj_x = point[0]
                obj_y = point[1]
                obj_z = point[2]
                object_3d_points.append([obj_x, obj_y, obj_z])

            object_3d_points = np.array(object_3d_points, dtype=np.float32)
            object_2d_points.clear()

            # 6. 3d转2d
            img_pts, _ = cv2.projectPoints(
                object_3d_points,
                rvec,
                tvec,
                self.camerainfo_.K_,
                self.camerainfo_.distCoeffs_
            )

            # 7. 提取2D像素点
            img_pts = img_pts.reshape(-1, 2)
            for pt in img_pts:
                object_2d_points.append((float(pt[0]), float(pt[1])))