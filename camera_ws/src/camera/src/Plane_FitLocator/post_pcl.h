#ifndef __POST_PCL_
#define __POST_PCL_
#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <iostream>
#include <string>
#include <mutex>
#include <unistd.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <unordered_set>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/centroid.h>  // 质心计算
#include <opencv2/core/types.hpp>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/surface/convex_hull.h>
#include <random>

// 半径滤波器参数
#define RadiusSearch 0.03        // 判断为杂点的最小距离， 范围内 没有足够近邻点 则直接删除
#define MinNeighborsInRadius 20  // 最小近邻数

// 欧式聚类
#define ClusterTolerance 0.016   // 判断是否为一堆 的阈值

namespace Ten
{
namespace Plane_FitLocator
{
class Ten_pre_pcl;
struct Plane_Info
{
    Eigen::Vector3d plane_center;
    std::vector<Eigen::Vector3d> plane_corner;
    RPY plane_euler;
    pcl::ModelCoefficients::Ptr plane_coeffs; // 平面方程系数
    Eigen::Matrix3d plane_rot_mat;      // 旋转矩阵

    Plane_Info()
    {
        plane_center = Eigen::Vector3d();
        plane_corner.resize(4);
        for (int i = 0;i < plane_corner.size(); i++)
        {
            plane_corner[i] = Eigen::Vector3d();
        }
        plane_coeffs.reset(new pcl::ModelCoefficients);
    }
};

class Ten_post_pcl
{
public:
    /**
     * @brief 设置到2d平面点云
     * @param plane_cloud 已提取的最大平面点云
     * @param plane_info   面相关信息（确保已填充中心点， 旋转矩阵， 面的方程）
     * @param plane_2d_cloud 2d平面点云
     */
    void set_2d_cloud(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& plane_cloud,
        const Plane_Info& plane_info,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& plane_2d_cloud
    );

    /**
     * @brief 拟合平面的最小外接四边形，并反变换为3D世界坐标， 修正旋转矩阵
     * @param plane_cloud 已提取的最大平面点云
     * @param plane_info   面相关信息
     */
    void fit_PlaneSquare(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& plane_cloud,
        Plane_Info& plane_info
    );

    /**
     * @brief 根据输入的最大平面点云，写入该平面的 中心点，面的旋转矩阵
     * @param input_cloud 输入点云（必须先滤波！）
     * @param plane_info   面相关信息
     */
    void compute_CenterAndNormal(
        pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud,
        Plane_Info& plane_info
    );

    // 填充 plane_info 中的 plane_euler 字段(确保 plane_rot_mat 字段已填充)
    void set_plane_euler(Plane_Info& plane_info);

    std::vector<pcl::PointXYZ> get_obb_corners() const
    {
        return obb_corners;
    }

private:
    std::vector<pcl::PointXYZ> obb_corners;

    // 填充plane_info中的 plane_rot_mat 字段
    void set_rot_mat(
        float nx, float ny, float nz, 
        Plane_Info& plane_info
    )
    {
        Eigen::Vector3d n(nx, ny, nz);
        n.normalize();
        Eigen::Matrix3d& rot = plane_info.plane_rot_mat;

        // 生成平面内正交的X/Y轴（右手坐标系，Z=法向量）
        Eigen::Vector3d x_axis;
        if (std::fabs(n.z()) < 0.999) {
            x_axis = Eigen::Vector3d(1, 0, 0).cross(n).normalized();
        } else {
            x_axis = Eigen::Vector3d(0, 1, 0).cross(n).normalized();
        }
        Eigen::Vector3d y_axis = n.cross(x_axis).normalized();

        // 构建旋转矩阵
        rot.col(0) = x_axis;
        rot.col(1) = y_axis;
        rot.col(2) = n;
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

    // 半径滤波器
    void radius_fitter(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud
    )
    {
        pcl::RadiusOutlierRemoval<pcl::PointXYZ> ror;
        ror.setInputCloud(input_cloud);
        ror.setRadiusSearch(RadiusSearch);    // 3cm范围内
        ror.setMinNeighborsInRadius(MinNeighborsInRadius); // 少于8个邻居的点判定为杂点
        ror.filter(*output_cloud);
    }
    
    // 滤波器，欧式聚类 - 只保留平面上最大的点簇（目标平面）
    void euclidean_filter(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud
    )
    {
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setInputCloud(input_cloud);
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

    // 【1. 半径离群滤波：删除孤立散点】 → 【2. 欧式聚类：只保留最大主体簇】
    void removePlaneNoise(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud
    )
    {
        // 步骤1：半径滤波 - 去除孤立散点
        radius_fitter(input_cloud, output_cloud);

        // 步骤2：欧式聚类 - 只保留平面上最大的点簇（目标平面）
        euclidean_filter(output_cloud,output_cloud);
    }

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

};      // class Ten_post_pcl

    void Ten_post_pcl::fit_PlaneSquare(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& plane_cloud,
        Plane_Info& plane_info
        )
    {
        obb_corners.clear();
        external_square_doubao(plane_cloud, obb_corners);

        // 2D局部角点 → 反变换为3D世界坐标
        plane_info.plane_corner.clear();
        plane_info.plane_corner.resize(4);
        const Eigen::Vector3d& center = plane_info.plane_center;
        for (int i = 0; i < obb_corners.size(); i++)
        {
            Eigen::Vector3d pt_local(obb_corners[i].x, obb_corners[i].y, 0.0);
            Eigen::Vector3d pt_world = plane_info.plane_rot_mat * pt_local + center;
            plane_info.plane_corner[i] = pt_world;
        }

        // 用矩形的 长边/短边 重新生成旋转矩阵
        std::vector<Eigen::Vector3d>& corners = plane_info.plane_corner;
        Eigen::Vector3d z_axis = plane_info.plane_rot_mat.col(2).normalized();

        // 1. 计算矩形两条邻边
        Eigen::Vector3d edge1 = corners[1] - corners[0];
        Eigen::Vector3d edge2 = corners[2] - corners[1];
        double len1 = edge1.norm();
        double len2 = edge2.norm();

        Eigen::Vector3d long_edge, short_edge;
        // 2. 区分长边/短边
        if (len1 >= len2) {
            long_edge = edge1;
            short_edge = edge2;
        } else {
            long_edge = edge2;
            short_edge = corners[0] - corners[1];
        }
        
        Eigen::Vector3d x_axis = long_edge.normalized();
        // 固定方向：如果和世界X轴点积为负，就反转X轴（永远朝一个方向）
        if (x_axis.dot(Eigen::Vector3d(1,0,0)) < 0) {
            x_axis = -x_axis;
        }

        // 3. 右手坐标系生成Y轴（严格垂直X/Z轴，保证TF标准）
        Eigen::Vector3d y_axis = z_axis.cross(x_axis).normalized();

        // 4. 写入稳定的旋转矩阵（永不跳变）
        plane_info.plane_rot_mat.col(0) = x_axis;
        plane_info.plane_rot_mat.col(1) = y_axis;
        plane_info.plane_rot_mat.col(2) = z_axis;
    }

    void Ten_post_pcl::set_2d_cloud(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& plane_cloud,
        const Plane_Info& plane_info,
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
        forced_plane_fitter(plane_cloud,plane_info.plane_coeffs,projected_cloud);

        // 3. 3D世界坐标 → 平面局部2D坐标（Z=0）
        pcl::PointCloud<pcl::PointXYZ>::Ptr local_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        const Eigen::Vector3d& center = plane_info.plane_center;
        for (const auto& pt : projected_cloud->points)
        {
            Eigen::Vector3d pt_world(pt.x, pt.y, pt.z);
            Eigen::Vector3d pt_local = plane_info.plane_rot_mat.transpose() * (pt_world - center);
            local_cloud->emplace_back(pt_local.x(), pt_local.y(), 0.0f);
        }

        removePlaneNoise(local_cloud, plane_2d_cloud);
    }

    void Ten_post_pcl::compute_CenterAndNormal(
        pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud,
        Plane_Info& plane_info
    )
    {
        // 1. 计算【最大平面的质心】（中心点）
        Eigen::Vector4f centroid_float;
        pcl::compute3DCentroid(*input_cloud, centroid_float);
        plane_info.plane_center.x() = centroid_float[0];
        plane_info.plane_center.y() = centroid_float[1];
        plane_info.plane_center.z() = centroid_float[2];

        //  2. 计算【单位法向量】并归一化 
        float a = plane_info.plane_coeffs->values[0];
        float b = plane_info.plane_coeffs->values[1];
        float c = plane_info.plane_coeffs->values[2];
        float norm = sqrt(a*a + b*b + c*c);
        float nx = a / norm;
        float ny = b / norm;
        float nz = c / norm;

        // 强制法向量朝向相机（Z轴正方向）
        if (nz < 0)
        {
            nx = -nx;
            ny = -ny;
            nz = -nz;
        }
        // 3. 填充 plane_rot_mat 字段
        set_rot_mat(nx, ny, nz, plane_info);
    }

    void Ten_post_pcl::set_plane_euler(Plane_Info& plane_info)
    {
        const Eigen::Matrix3d& rot = plane_info.plane_rot_mat;

        // 标准旋转矩阵 → roll-pitch-yaw 欧拉角（ROS TF 标准格式）
        plane_info.plane_euler._roll  = std::atan2(rot(2,1), rot(2,2));
        plane_info.plane_euler._pitch = std::atan2(-rot(2,0), std::hypot(rot(2,1), rot(2,2)));
        plane_info.plane_euler._yaw   = std::atan2(rot(1,0), rot(0,0));
    }
}       // namespace Plane_FitLocator
}       // namespace Ten
#endif