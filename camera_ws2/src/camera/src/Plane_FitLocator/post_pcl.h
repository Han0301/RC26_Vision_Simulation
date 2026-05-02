#ifndef __POST_PCL_
#define __POST_PCL_
#include <iostream>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/centroid.h>  // 质心计算
#include <opencv2/core/types.hpp>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/segmentation/extract_clusters.h>

constexpr double BOX_SIZE = 0.35;      // 35cm固定框
constexpr double HALF_BOX = BOX_SIZE / 2.0;
constexpr double PI = 3.14159265358979323846;

// 半径滤波器参数
#define RadiusSearch 0.03        // 判断为杂点的最小距离， 范围内 没有足够近邻点 则直接删除
#define MinNeighborsInRadius 20  // 最小近邻数

// 欧式聚类
#define ClusterTolerance 0.016   // 判断是否为一堆 的阈值

namespace Ten
{
namespace Plane_FitLocator
{
struct Plane_Info
{
    Eigen::Vector3d plane_center;              // 中心点3d坐标
    RPY plane_euler;                           // 平面的rpy
    pcl::ModelCoefficients::Ptr plane_coeffs;  // 平面方程系数
    Eigen::Matrix3d plane_rot_mat;             // 旋转矩阵

    Plane_Info()
    {
        plane_center = Eigen::Vector3d();
        plane_coeffs.reset(new pcl::ModelCoefficients);
    }
};

class Ten_post_pcl
{
public:

    /**
     * @brief 根据输入的最大平面点云，计算中心+法向量+初始旋转矩阵
     * @param input_cloud 输入点云
     * @param plane_info   面相关信息
     */
    void compute_CenterAndNormal(
        pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud,
        Plane_Info& plane_info
    );

    /**
     * @brief 3D点云 → 平面局部2D点云 + 去噪
     * @param plane_cloud 已提取的最大平面点云
     * @param plane_info   面相关信息（确保已填充中心点， 旋转矩阵， 面的方程）
     * @param plane_2d_cloud 2d平面点云
     */
    void set_2d_cloud(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& plane_cloud,
        Plane_Info& plane_info,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& plane_2d_cloud
    );

    // 搜索最优Yaw → 更新旋转矩阵 → 计算最终RPY
    void set_RPY(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& plane_2d_cloud,
        Plane_Info& plane_info);


private:

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

    // 填充 plane_info 中的 plane_euler 字段(确保 plane_rot_mat 字段已填充)
    void set_plane_euler(Plane_Info& plane_info)
    {
        const Eigen::Matrix3d& rot = plane_info.plane_rot_mat;

        // 标准旋转矩阵 → roll-pitch-yaw 欧拉角（ROS TF 标准格式）
        plane_info.plane_euler._roll  = std::atan2(rot(2,1), rot(2,2));
        plane_info.plane_euler._pitch = std::atan2(-rot(2,0), std::hypot(rot(2,1), rot(2,2)));
        plane_info.plane_euler._yaw   = std::atan2(rot(1,0), rot(0,0));
    }

    /**
     * @brief 绕Z轴旋转2D点云（考虑中心点偏移）
     * @param cloud 输入2D点云 (Z=0)
     * @param angle_deg 旋转角度(度)
     * @return 旋转后的点云
     */
    pcl::PointCloud<pcl::PointXYZ>::Ptr rotatePointCloud2D(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
        double angle_deg)
        {
            // 预分配内存，避免动态扩容
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_rotated(new pcl::PointCloud<pcl::PointXYZ>);
            cloud_rotated->reserve(cloud->size());
            
            double theta = angle_deg * PI / 180.0;
            double c = std::cos(theta);
            double s = std::sin(theta);

            for (const auto& pt : *cloud)
            {
                double x_rot = pt.x * c + pt.y * s;
                double y_rot = -pt.x * s + pt.y * c;
                cloud_rotated->emplace_back(x_rot, y_rot, 0.0);
            }
            return cloud_rotated;
        }

    /**
     * @brief 统计固定轴对齐框内的点数（核心评价函数）
     * @param cloud 旋转后的点云
     * @param center 中心点
     * @return 框内点数
     */
    int countInFixedBox(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
    {
        int count = 0;
        for (const auto& pt : *cloud)
        {
            if (std::fabs(pt.x) < HALF_BOX && std::fabs(pt.y) < HALF_BOX)
                count++;
        }
        return count;
    }

    /**
     * @brief 三轮变步长角度搜索
     * @param plane_cloud_2d 输入2D点云
     * @return best_angle 最优yaw角
     */
    double set_yaw(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& plane_cloud_2d)
    {
        // ==================== 第1轮: 10°大步长 全局粗搜 ====================
        double best_angle = 0.0;
        int max_score = countInFixedBox(rotatePointCloud2D(plane_cloud_2d, best_angle));

        for (double angle = 10.0; angle <= 90.0; angle += 10.0)
        {
            int score = countInFixedBox(rotatePointCloud2D(plane_cloud_2d, angle));
            if (score > max_score)
            {
                max_score = score;
                best_angle = angle;
            }
        }

        // ==================== 第2轮: 1°中步长 ±3°窄范围搜索 ====================
        double start2 = std::max(0.0, best_angle - 3.0);
        double end2 = std::min(90.0, best_angle + 3.0);
        for (double angle = start2; angle <= end2; angle += 1.0)
        {
            int score = countInFixedBox(rotatePointCloud2D(plane_cloud_2d, angle));
            if (score > max_score)
            {
                max_score = score;
                best_angle = angle;
            }
        }

        // ==================== 第3轮: 0.1°小步长 ±0.5°精搜 ====================
        double start3 = std::max(0.0, best_angle - 0.5);
        double end3 = std::min(90.0, best_angle + 0.5);
        for (double angle = start3; angle <= end3; angle += 0.1)
        {
            int score = countInFixedBox(rotatePointCloud2D(plane_cloud_2d, angle));
            if (score > max_score)
            {
                max_score = score;
                best_angle = angle;
            }
        }

        return best_angle;
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

        if(input_cloud->empty()){
            *output_cloud = *input_cloud;
            return;
        }
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
        projected_cloud->reserve(plane_cloud->size());
        pcl::ProjectInliers<pcl::PointXYZ> proj;
        proj.setModelType(pcl::SACMODEL_PLANE);
        proj.setInputCloud(plane_cloud);
        proj.setModelCoefficients(plane_coeffs);
        proj.filter(*projected_cloud);
    }

};      // class Ten_post_pcl

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

    void Ten_post_pcl::set_2d_cloud(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& plane_cloud,
        Plane_Info& plane_info,
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
        local_cloud->reserve(projected_cloud->size());
        const Eigen::Vector3d& center = plane_info.plane_center;
        for (const auto& pt : projected_cloud->points)
        {
            Eigen::Vector3d pt_world(pt.x, pt.y, pt.z);
            Eigen::Vector3d pt_local = plane_info.plane_rot_mat.transpose() * (pt_world - center);
            local_cloud->emplace_back(pt_local.x(), pt_local.y(), 0.0f);
        }

        removePlaneNoise(local_cloud, plane_2d_cloud);
    }

    void Ten_post_pcl::set_RPY(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& plane_2d_cloud,
        Plane_Info& plane_info)
    {
        // 1. 执行三轮搜索，得到最优Yaw角度(度)
        double best_yaw_deg = set_yaw(plane_2d_cloud);
        double best_yaw_rad = best_yaw_deg * PI / 180.0;

        // 2. 保留原有平面法向量(Z轴)，仅更新X/Y轴旋转（关键！不破坏平面姿态）
        Eigen::Vector3d z_axis = plane_info.plane_rot_mat.col(2).normalized();
        double c = cos(best_yaw_rad);
        double s = sin(best_yaw_rad);

        // 构建新的旋转矩阵（仅绕Z轴旋转最优角度）
        Eigen::Matrix3d rot_yaw;
        rot_yaw << c, -s, 0,
                s,  c, 0,
                0,  0, 1;

        // 3. 更新平面旋转矩阵（原有平面姿态 + 最优2D旋转）
        plane_info.plane_rot_mat = plane_info.plane_rot_mat * rot_yaw;

        // 4. ✅ 调用你原有函数，自动填充RPY（完全复用你的代码）
        set_plane_euler(plane_info);
    }
}       // namespace Plane_FitLocator
}       // namespace Ten
#endif