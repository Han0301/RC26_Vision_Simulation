#ifndef __POST_PCL_
#define __POST_PCL_

#include <pcl/filters/project_inliers.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/common/transforms.h>

#include "pre_pcl.h"
namespace Ten
{
namespace Plane_FitLocator
{
class Ten_post_pcl
{
public:

    /**
     * @brief 拟合平面的最小外接四边形，结果写入plane_corner
     * @param plane_cloud 已提取的最大平面点云
     * @param plane_coeffs 平面方程系数
     * @param plane_info   面相关信息
     * @param plane_rot_mat 面的旋转矩阵
     */
    void fitPlaneQuadrilateral(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& plane_cloud,
        const pcl::ModelCoefficients::Ptr& plane_coeffs,
        const Plane_Info& plane_info,
        const Eigen::Matrix3d& plane_rot_mat
        )
    {
        // 1. 输入合法性检查
        if (plane_cloud->empty())
        {
            std::cerr << "❌ 四边形拟合失败：点云数量不足！" << std::endl;
            return;
        }

        // 2. 点云投影到拟合平面（消除Z轴偏差，保证严格共面）
        pcl::PointCloud<pcl::PointXYZ>::Ptr projected_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        forced_plane_fitter(plane_cloud,plane_coeffs,projected_cloud);

        // 3. 3D世界坐标 → 平面局部2D坐标（Z=0）
        pcl::PointCloud<pcl::PointXYZ>::Ptr local_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        const Eigen::Vector3d& center = plane_info.plane_center;
        for (const auto& pt : projected_cloud->points)
        {
            Eigen::Vector3d pt_world(pt.x, pt.y, pt.z);
            Eigen::Vector3d pt_local = plane_rot_mat.transpose() * (pt_world - center);
            local_cloud->emplace_back(pt_local.x(), pt_local.y(), 0.0f);
        }

        // 4. 计算最小外接矩形（旋转卡壳法，得到4个角点）
        std::vector<pcl::PointXYZ> obb_corners;
        external_square(local_cloud, obb_corners);

        // 5. 2D局部角点 → 反变换为3D世界坐标
        plane_corner.clear();
        plane_corner.resize(4);
        for (int i = 0; i < obb_corners.size(); i++)
        {
            Eigen::Vector3d pt_local(obb_corners[i].x, obb_corners[i].y, 0.0);
            Eigen::Vector3d pt_world = plane_rot_mat * pt_local + center;
            plane_corner[i] = pt_world;
        }
    }

    std::vector<Eigen::Vector3d> get_plane_corner() const
    {
        return plane_corner;
    }

private:
    std::vector<Eigen::Vector3d> plane_corner;
    // 把「有微小误差的平面点云」，强行精准投影到「数学拟合的完美平面」上，让所有点 100% 严格共面
    void forced_plane_fitter(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& plane_cloud,
        const pcl::ModelCoefficients::Ptr& plane_coeffs,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& projected_cloud)
    {
        pcl::ProjectInliers<pcl::PointXYZ> proj;
        proj.setModelType(pcl::SACMODEL_PLANE);
        proj.setInputCloud(plane_cloud);
        proj.setModelCoefficients(plane_coeffs);
        proj.filter(*projected_cloud);
    }

    // 计算最小外接矩形
    void external_square(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& local_cloud,
        std::vector<pcl::PointXYZ>& obb_corners
    )
    {
        obb_corners.clear();
        obb_corners.resize(4);
        pcl::MomentOfInertiaEstimation<pcl::PointXYZ> extractor;
        extractor.setInputCloud(local_cloud);
        extractor.compute();

        // 定义接口需要的变量
        pcl::PointXYZ min_point;
        pcl::PointXYZ max_point;
        pcl::PointXYZ position;
        Eigen::Matrix3f rot_mat;

        // 调用正确的OBB接口
        extractor.getOBB(min_point, max_point, position, rot_mat);

        // 手动生成2D矩形4个角点（Z=0，平面内）
        obb_corners[0] = pcl::PointXYZ(min_point.x, min_point.y, 0);
        obb_corners[1] = pcl::PointXYZ(max_point.x, min_point.y, 0);
        obb_corners[2] = pcl::PointXYZ(max_point.x, max_point.y, 0);
        obb_corners[3] = pcl::PointXYZ(min_point.x, max_point.y, 0);

    }

};      // class Ten_pre_pcl
}       // namespace Plane_FitLocator
}       // namespace Ten
#endif