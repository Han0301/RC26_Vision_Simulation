#ifndef __POST_PCL_
#define __POST_PCL_
#include <iostream>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/centroid.h>
#include <opencv2/core/types.hpp>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/segmentation/extract_clusters.h>
#include <Eigen/Dense>

constexpr double BOX_SIZE = 0.35;
constexpr double HALF_BOX = BOX_SIZE / 2.0;
constexpr double PI = 3.14159265358979323846;

#define RadiusSearch 0.03                // 半径滤波搜索半径，值越小过滤越严格
#define MinNeighborsInRadius 20          // 半径滤波最小邻域点数，值越大过滤越严格
#define ClusterTolerance 0.016           // 欧式聚类容差，值越大聚类范围越大

namespace Ten
{
namespace Plane_FitLocator
{

// 存储平面位姿信息
struct Plane_Info
{
    Eigen::Vector3d plane_center;        // 平面3D中心点坐标
    RPY plane_euler;                     // 平面欧拉角姿态
    Eigen::Vector3d plane_normal;        // 平面单位法向量

    Plane_Info()
    {
        plane_center = Eigen::Vector3d::Zero();
        plane_normal = Eigen::Vector3d::UnitZ();
    }
};


// 平面点云后处理核心类
class Ten_post_pcl
{
public:
    /**
     * @brief 计算平面点云质心与初始姿态
     * @param input_cloud 输入平面点云
     * @param plane_info 输出平面信息结构体
     */
    void compute_CenterAndNormal(
        pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud,
        Plane_Info& plane_info
    );

    /**
     * @brief 3D平面点云转换为局部2D点云
     * @param plane_cloud 输入3D平面点云
     * @param plane_info 平面基础信息
     * @param plane_2d_cloud 输出处理后的2D点云
     */
    void set_2d_cloud(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& plane_cloud,
        Plane_Info& plane_info,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& plane_2d_cloud
    );

    /**
     * @brief 优化平面偏航角并更新姿态
     * @param plane_2d_cloud 输入2D局部点云
     * @param plane_info 输出更新后的平面信息
     */
    void set_RPY(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& plane_2d_cloud,
        Plane_Info& plane_info);

private:
    // 基于法向量构建局部正交坐标轴
    void getLocalAxes(const Eigen::Vector3d& n,
                    Eigen::Vector3d& x_axis,
                    Eigen::Vector3d& y_axis);

    // 计算平面初始欧拉角
    void set_plane_euler(Plane_Info& plane_info);

    // 旋转2D点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr rotatePointCloud2D(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
        double angle_deg);

    // 统计方框内点云数量
    int countInFixedBox(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);

    // 搜索最优偏航角
    double set_yaw(const pcl::PointCloud<pcl::PointXYZ>::Ptr& plane_cloud_2d);

    // 半径滤波去噪
    void radius_fitter(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud);

    // 欧式聚类提取主点云
    void euclidean_filter(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud);

    // 组合滤波去除平面噪声
    void removePlaneNoise(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud);

    // 点云投影到指定平面
    void forced_plane_fitter(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& plane_cloud,
        const Plane_Info& plane_info,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& projected_cloud);

};

void Ten_post_pcl::compute_CenterAndNormal(
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud,
    Plane_Info& plane_info)
{
    // 计算点云质心
    Eigen::Vector4f centroid_float;
    pcl::compute3DCentroid(*input_cloud, centroid_float);
    plane_info.plane_center = Eigen::Vector3d(centroid_float[0], centroid_float[1], centroid_float[2]);

    // 调整法向量朝向
    if (plane_info.plane_normal.z() < 0)
    {
        plane_info.plane_normal = -plane_info.plane_normal;
    }

    // 计算初始欧拉角
    set_plane_euler(plane_info);
}

void Ten_post_pcl::set_2d_cloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& plane_cloud,
    Plane_Info& plane_info,
    pcl::PointCloud<pcl::PointXYZ>::Ptr& plane_2d_cloud)
{
    // 点云非空校验
    if (plane_cloud->empty())
    {
        std::cerr << "点云为空！" << std::endl;
        return;
    }

    // 点云平面投影
    pcl::PointCloud<pcl::PointXYZ>::Ptr projected_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    forced_plane_fitter(plane_cloud, plane_info, projected_cloud);

    // 初始化局部点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr local_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    local_cloud->reserve(projected_cloud->size());
    const auto& center = plane_info.plane_center;
    const auto& n = plane_info.plane_normal;
    Eigen::Vector3d x_axis, y_axis;
    getLocalAxes(n, x_axis, y_axis);

    // 坐标转换为局部2D坐标
    for (const auto& pt : projected_cloud->points)
    {
        Eigen::Vector3d pt_world(pt.x, pt.y, pt.z);
        Eigen::Vector3d diff = pt_world - center;
        double x = diff.dot(x_axis);
        double y = diff.dot(y_axis);
        local_cloud->emplace_back(x, y, 0.0f);
    }

    // 点云去噪处理
    removePlaneNoise(local_cloud, plane_2d_cloud);
}

void Ten_post_pcl::set_RPY(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& plane_2d_cloud,
    Plane_Info& plane_info)
{
    // 1 计算最优偏航角
    double best_yaw_deg = set_yaw(plane_2d_cloud);
    double best_yaw_rad = best_yaw_deg * PI / 180.0;

    // 2 构建基础旋转矩阵
    Eigen::Vector3d n = plane_info.plane_normal;
    n.normalize();
    Eigen::Vector3d x_axis, y_axis;
    getLocalAxes(n, x_axis, y_axis);

    Eigen::Matrix3d rot_mat;
    rot_mat.col(0) = x_axis;
    rot_mat.col(1) = y_axis;
    rot_mat.col(2) = n;

    // 3 应用偏航旋转矩阵
    Eigen::Matrix3d rot_yaw;
    rot_yaw << cos(best_yaw_rad), -sin(best_yaw_rad), 0,
               sin(best_yaw_rad),  cos(best_yaw_rad), 0,
               0,                  0,                 1;
    rot_mat = rot_mat * rot_yaw;

    // 4 计算最终欧拉角
    plane_info.plane_euler._roll  = std::atan2(rot_mat(2,1), rot_mat(2,2));
    plane_info.plane_euler._pitch = std::atan2(-rot_mat(2,0), std::hypot(rot_mat(2,1), rot_mat(2,2)));
    plane_info.plane_euler._yaw   = std::atan2(rot_mat(1,0), rot_mat(0,0));
}

void Ten_post_pcl::getLocalAxes(const Eigen::Vector3d& n,
                                Eigen::Vector3d& x_axis,
                                Eigen::Vector3d& y_axis)
{
    // 法向量归一化
    Eigen::Vector3d norm_n = n;
    norm_n.normalize();

    // 计算局部X轴
    if (std::fabs(norm_n.z()) < 0.999)
    {
        x_axis = Eigen::Vector3d(1, 0, 0).cross(norm_n).normalized();
    }
    else
    {
        x_axis = Eigen::Vector3d(0, 1, 0).cross(norm_n).normalized();
    }

    // 计算局部Y轴
    y_axis = norm_n.cross(x_axis).normalized();
}

void Ten_post_pcl::set_plane_euler(Plane_Info& plane_info)
{
    // 构建局部坐标轴
    const Eigen::Vector3d& n = plane_info.plane_normal;
    Eigen::Vector3d x_axis, y_axis;
    getLocalAxes(n, x_axis, y_axis);

    // 构建旋转矩阵
    Eigen::Matrix3d rot;
    rot.col(0) = x_axis;
    rot.col(1) = y_axis;
    rot.col(2) = n;

    // 计算欧拉角
    plane_info.plane_euler._roll  = std::atan2(rot(2,1), rot(2,2));
    plane_info.plane_euler._pitch = std::atan2(-rot(2,0), std::hypot(rot(2,1), rot(2,2)));
    plane_info.plane_euler._yaw   = std::atan2(rot(1,0), rot(0,0));
}

pcl::PointCloud<pcl::PointXYZ>::Ptr Ten_post_pcl::rotatePointCloud2D(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    double angle_deg)
{
    // 初始化旋转点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_rotated(new pcl::PointCloud<pcl::PointXYZ>);
    cloud_rotated->reserve(cloud->size());
    double theta = angle_deg * PI / 180.0;
    double c = std::cos(theta);
    double s = std::sin(theta);

    // 遍历执行2D旋转
    for (const auto& pt : *cloud)
    {
        double x_rot = pt.x * c + pt.y * s;
        double y_rot = -pt.x * s + pt.y * c;
        cloud_rotated->emplace_back(x_rot, y_rot, 0.0);
    }
    return cloud_rotated;
}

int Ten_post_pcl::countInFixedBox(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
    // 统计有效点数量
    int count = 0;
    for (const auto& pt : *cloud)
    {
        if (std::fabs(pt.x) < HALF_BOX && std::fabs(pt.y) < HALF_BOX)
        {
            count++;
        }
    }
    return count;
}

double Ten_post_pcl::set_yaw(const pcl::PointCloud<pcl::PointXYZ>::Ptr& plane_cloud_2d)
{
    // 1 初始化最优角度
    double best_angle = 0.0;
    int max_score = countInFixedBox(rotatePointCloud2D(plane_cloud_2d, best_angle));

    // 2 粗角度搜索
    for (double angle = 10.0; angle <= 90.0; angle += 10.0)
    {
        int score = countInFixedBox(rotatePointCloud2D(plane_cloud_2d, angle));
        if (score > max_score)
        {
            max_score = score;
            best_angle = angle;
        }
    }

    // 3 精角度搜索
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

    // 4 细角度搜索
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

void Ten_post_pcl::radius_fitter(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
    pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud)
{
    // 执行半径滤波
    pcl::RadiusOutlierRemoval<pcl::PointXYZ> ror;
    ror.setInputCloud(input_cloud);
    ror.setRadiusSearch(RadiusSearch);
    ror.setMinNeighborsInRadius(MinNeighborsInRadius);
    ror.filter(*output_cloud);
}

void Ten_post_pcl::euclidean_filter(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
    pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud)
{
    // 执行欧式聚类
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setInputCloud(input_cloud);
    ec.setClusterTolerance(ClusterTolerance);
    ec.extract(cluster_indices);

    // 提取主聚类点云
    if (cluster_indices.empty())
    {
        *output_cloud = *input_cloud;
        return;
    }
    output_cloud->clear();
    for (int idx : cluster_indices[0].indices)
    {
        output_cloud->push_back(input_cloud->points[idx]);
    }
}

void Ten_post_pcl::removePlaneNoise(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
    pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud)
{
    // 点云空值判断
    if(input_cloud->empty())
    {
        *output_cloud = *input_cloud;
        return;
    }

    // 组合滤波去噪
    radius_fitter(input_cloud, output_cloud);
    euclidean_filter(output_cloud,output_cloud);
}

void Ten_post_pcl::forced_plane_fitter(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& plane_cloud,
    const Plane_Info& plane_info,
    pcl::PointCloud<pcl::PointXYZ>::Ptr& projected_cloud)
{
    // 数据类型转换
    const Eigen::Vector3f n = plane_info.plane_normal.cast<float>();
    const Eigen::Vector3f c = plane_info.plane_center.cast<float>();

    // 生成平面方程系数
    pcl::ModelCoefficients::Ptr coeffs(new pcl::ModelCoefficients);
    coeffs->values.resize(4);
    coeffs->values[0] = n.x();
    coeffs->values[1] = n.y();
    coeffs->values[2] = n.z();
    coeffs->values[3] = -(n.x() * c.x() + n.y() * c.y() + n.z() * c.z());

    // 执行平面投影
    pcl::ProjectInliers<pcl::PointXYZ> proj;
    proj.setModelType(pcl::SACMODEL_PLANE);
    proj.setInputCloud(plane_cloud);
    proj.setModelCoefficients(coeffs);
    proj.filter(*projected_cloud);
}

} // namespace Plane_FitLocator
} // namespace Ten

#endif