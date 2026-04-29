#ifndef __POST_PCL_
#define __POST_PCL_

#include <pcl/filters/project_inliers.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/ModelCoefficients.h>
#include <random>
#include "pre_pcl.h"

#define RadiusSearch 0.03        // 判断为杂点的最小距离， 范围内 没有足够近邻点 则直接删除
#define MinNeighborsInRadius 20  // 最小近邻数
#define ClusterTolerance 0.016   // 判断是否为一堆 的阈值
namespace Ten
{
namespace Plane_FitLocator
{
class Ten_post_pcl
{
public:

    /**
     * @brief 设置到2d平面点云
     * @param local_cloud 已提取的最大平面点云
     * @param plane_coeffs 平面方程系数
     * @param plane_info   面相关信息
     * @param plane_rot_mat 面的旋转矩阵
     * @param 
     */
    void set_2d_cloud(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& plane_cloud,
        const pcl::ModelCoefficients::Ptr& plane_coeffs,
        const Plane_Info& plane_info,
        const Eigen::Matrix3d& plane_rot_mat,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& plane_2d_cloud
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

        removePlaneNoise(local_cloud, plane_2d_cloud);
    }
    /**
     * @brief 拟合平面的最小外接四边形，结果写入plane_corner
     * @param local_cloud 已提取的最大平面点云
     * @param plane_coeffs 平面方程系数
     * @param plane_info   面相关信息
     * @param plane_rot_mat 面的旋转矩阵
     */
    void fit_PlaneSquare(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& local_cloud,
        const pcl::ModelCoefficients::Ptr& plane_coeffs,
        const Plane_Info& plane_info,
        const Eigen::Matrix3d& plane_rot_mat
        )
    {
        obb_corners.clear();
        external_square_doubao(local_cloud, obb_corners);

        // 5. 2D局部角点 → 反变换为3D世界坐标
        plane_corner.clear();
        plane_corner.resize(4);
        const Eigen::Vector3d& center = plane_info.plane_center;
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
    std::vector<pcl::PointXYZ> get_obb_corners() const
    {
        return obb_corners;
    }
private:
    std::vector<Eigen::Vector3d> plane_corner;
    std::vector<pcl::PointXYZ> obb_corners;

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
 
    // 【1. 半径离群滤波：删除孤立散点】 → 【2. 欧式聚类：只保留最大主体簇】
    void removePlaneNoise(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud
    )
    {
        // 步骤1：半径滤波 - 去除孤立散点
        pcl::RadiusOutlierRemoval<pcl::PointXYZ> ror;
        ror.setInputCloud(input_cloud);
        ror.setRadiusSearch(RadiusSearch);    // 3cm范围内
        ror.setMinNeighborsInRadius(MinNeighborsInRadius); // 少于8个邻居的点判定为杂点
        ror.filter(*output_cloud);

        // 步骤2：欧式聚类 - 只保留平面上最大的点簇（目标平面）
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setInputCloud(output_cloud);
        ec.setClusterTolerance(ClusterTolerance); // 2cm间距算同一簇
        ec.extract(cluster_indices);

        if (cluster_indices.empty())
        {
            output_cloud = input_cloud;
            return;
        }

        // 提取最大聚类（目标主体）
        output_cloud->clear();
        for (int idx : cluster_indices[0].indices)
        {
            output_cloud->push_back(input_cloud->points[idx]);
        }
    }

    // 计算最小外接矩形
    void external_square_doubao(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& local_cloud,
        std::vector<pcl::PointXYZ>& obb_corners
    )
    {
        obb_corners.clear();
        if (!local_cloud || local_cloud->size() < 10) return;

        // 1. 计算凸包（包裹所有有效点，抗噪声）
        pcl::ConvexHull<pcl::PointXYZ> hull;
        pcl::PointCloud<pcl::PointXYZ>::Ptr hull_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        hull.setInputCloud(local_cloud);
        hull.reconstruct(*hull_cloud);
        if (hull_cloud->size() < 3) return;

        // 2. 旋转卡壳算法 → 计算2D最小面积矩形（核心！）
        std::vector<Eigen::Vector2d> points2d;
        for (auto& pt : hull_cloud->points) {
            points2d.emplace_back(pt.x, pt.y);
        }

        double min_area = DBL_MAX;
        Eigen::Matrix2d best_rot;
        Eigen::Vector2d best_center;
        Eigen::Vector2d best_size;

        int n = points2d.size();
        for (int i = 0; i < n; ++i) {
            // 取凸包边的方向
            Eigen::Vector2d p1 = points2d[i];
            Eigen::Vector2d p2 = points2d[(i+1)%n];
            Eigen::Vector2d edge = p2 - p1;
            if (edge.norm() < 1e-6) continue;

            // 构建旋转矩阵（对齐当前边）
            double angle = atan2(edge.y(), edge.x());
            Eigen::Matrix2d rot;
            rot << cos(angle), -sin(angle),
                sin(angle),  cos(angle);

            // 旋转所有点，计算轴对齐包围盒
            std::vector<Eigen::Vector2d> rot_pts;
            Eigen::Vector2d min_p(DBL_MAX, DBL_MAX);
            Eigen::Vector2d max_p(-DBL_MAX, -DBL_MAX);
            for (auto& p : points2d) {
                Eigen::Vector2d rp = rot * p;
                min_p = min_p.cwiseMin(rp);
                max_p = max_p.cwiseMax(rp);
            }

            Eigen::Vector2d center = rot.transpose() * ((min_p + max_p) * 0.5);
            Eigen::Vector2d size = max_p - min_p;
            double area = size.x() * size.y();

            // 保留最小面积的矩形
            if (area < min_area) {
                min_area = area;
                best_rot = rot.transpose();
                best_center = center;
                best_size = size;
            }
        }

        // 3. 生成最终4个角点（2D平面，Z=0）
        double w = best_size.x() * 0.5;
        double h = best_size.y() * 0.5;
        std::vector<Eigen::Vector2d> rect_2d = {
            {-w, -h}, {w, -h}, {w, h}, {-w, h}
        };

        for (auto& p : rect_2d) {
            Eigen::Vector2d final_p = best_rot * p + best_center;
            obb_corners.emplace_back(final_p.x(), final_p.y(), 0.0f);
        }
        if (obb_corners.size() == 4)
        {
            // 计算4条边的长度（2D平面距离，Z=0忽略）
            auto dist = [](const pcl::PointXYZ& p1, const pcl::PointXYZ& p2) {
                return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
            };

            double d1 = dist(obb_corners[0], obb_corners[1]); // 边1
            double d2 = dist(obb_corners[1], obb_corners[2]); // 边2
            double d3 = dist(obb_corners[2], obb_corners[3]); // 边3
            double d4 = dist(obb_corners[3], obb_corners[0]); // 边4

            // ROS打印（适配你的ROS环境，保留3位小数）
            ROS_INFO("===== 矩形相邻角点距离 (单位：m) =====");
            ROS_INFO("边1(0→1): %.3f", d1);
            ROS_INFO("边2(1→2): %.3f", d2);
            ROS_INFO("边3(2→3): %.3f", d3);
            ROS_INFO("边4(3→0): %.3f", d4);
            ROS_INFO("矩形长宽: 长=%.3f, 宽=%.3f", std::max(d1,d2), std::min(d1,d2));
        }
    }

};      // class Ten_pre_pcl
}       // namespace Plane_FitLocator
}       // namespace Ten
#endif