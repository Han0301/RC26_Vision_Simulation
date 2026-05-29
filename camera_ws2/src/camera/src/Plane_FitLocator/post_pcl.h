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
#include <cmath>
#include "./../method_math.h"

#define BOX_SIZE 0.35
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
    // 计算平面点云质心与初始姿态
    void compute_CenterAndNormal(
        pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud,
        Plane_Info& plane_info
    );

    // 组合滤波去除平面噪声
    void removePlaneNoise(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud);

    // 转3d点云到2dvector容器
    void set_vector_2d(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
        const Plane_Info& plane_info,
        std::vector<cv::Point2f>& output_2d);

    /**
     * @brief 带调试打印的2D离群点过滤（可看平均距离，可自定义范围）
     * @param src 输入原始投影点
     * @param dst 输出过滤后点
     * @param threshold 过滤系数（平均距离的倍数，默认1.5）
     */
    void central_range_filter(
        const std::vector<cv::Point2f>& input_points, 
        std::vector<cv::Point2f>& ouput_points, 
        float threshold = 1.5);

    // 优化平面偏航角并更新姿态
    void set_RPY(
        const std::vector<cv::Point2f>& plane_2d_points,
        Plane_Info& plane_info);

private:
    // 基于法向量构建局部正交坐标轴
    void getLocalAxes(const Eigen::Vector3d& n,
                    Eigen::Vector3d& x_axis,
                    Eigen::Vector3d& y_axis);

    // 计算平面初始欧拉角
    void set_plane_euler(Plane_Info& plane_info);

    // 旋转2D点云
    std::vector<cv::Point2f> rotatePointCloud2D(
        const std::vector<cv::Point2f>& plane_2d_points,
        double angle_deg);

    // 统计方框内点云数量
    int countInFixedBox(const std::vector<cv::Point2f>& points);

    // 搜索最优偏航角
    double set_yaw(const std::vector<cv::Point2f>& plane_2d_points);

    // 欧式聚类提取主点云
    void euclidean_filter(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud);
};      // class Ten_post_pcl

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

void Ten_post_pcl::set_RPY(
    const std::vector<cv::Point2f>& plane_2d_points,
    Plane_Info& plane_info)
{
    // 1 计算最优偏航角
    double best_yaw_deg = set_yaw(plane_2d_points);
    double best_yaw_rad = best_yaw_deg * M_PI / 180.0;

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
    rot_mat = rot_yaw * rot_mat;

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
    // if (std::fabs(norm_n.z()) < 0.999)
    // {
    //     x_axis = Eigen::Vector3d(1, 0, 0).cross(norm_n).normalized();
    // }
    // else
    // {
    //     x_axis = Eigen::Vector3d(0, 1, 0).cross(norm_n).normalized();
    // }

    x_axis = Eigen::Vector3d(1, 0, 0).cross(norm_n).normalized();
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

std::vector<cv::Point2f> Ten_post_pcl::rotatePointCloud2D(
    const std::vector<cv::Point2f>& plane_2d_points,
    double angle_deg)
{
    std::vector<cv::Point2f> rotated;
    rotated.reserve(plane_2d_points.size());
    double theta = angle_deg * M_PI / 180.0;
    float c = static_cast<float>(cos(theta));
    float s = static_cast<float>(sin(theta));

    for (const auto& pt : plane_2d_points)
    {
        float x = pt.x * c - pt.y * s;
        float y = pt.x * s + pt.y * c;
        rotated.emplace_back(x, y);
    }
    return rotated;
}


int Ten_post_pcl::countInFixedBox(const std::vector<cv::Point2f>& points)
{
    int count = 0;
    const float HALF_BOX = static_cast<float>(BOX_SIZE / 2.0);
    for (const auto& pt : points)
    {
        if (fabs(pt.x) < HALF_BOX && fabs(pt.y) < HALF_BOX)
        {
            count++;
        }
    }
    return count;
}

double Ten_post_pcl::set_yaw(const std::vector<cv::Point2f>& plane_2d_points)
{
    // 1 初始化最优角度
    double best_angle = 0.0;
    int max_score = countInFixedBox(rotatePointCloud2D(plane_2d_points, best_angle));

    // 2 粗角度搜索
    for (double angle = 10.0; angle <= 90.0; angle += 10.0)
    {
        int score = countInFixedBox(rotatePointCloud2D(plane_2d_points, angle));
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
        int score = countInFixedBox(rotatePointCloud2D(plane_2d_points, angle));
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
        int score = countInFixedBox(rotatePointCloud2D(plane_2d_points, angle));
        if (score > max_score)
        {
            max_score = score;
            best_angle = angle;
        }
    }
    return best_angle;
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

void Ten_post_pcl::central_range_filter(
    const std::vector<cv::Point2f>& input_points, 
    std::vector<cv::Point2f>& output_points, 
    float threshold)
{
    output_points.clear();
    // 点太少不过滤，直接返回
    if (input_points.size() < 10) 
    { 
        output_points = input_points; 
        return; 
    }

    // 1. 计算点云中心
    cv::Point2f center(0, 0);
    for (const auto& p : input_points) center += p;
    center /= float(input_points.size());

    // 2. 计算每个点到中心的距离
    std::vector<float> distances;
    for (const auto& p : input_points)
    {
        distances.push_back(cv::norm(p - center));
    }

    // 3. 计算【平均距离】（核心调试值）
    float avg_dist = 0.0f;
    for (float d : distances) avg_dist += d;
    avg_dist /= distances.size();

    // // ===================== 【关键：打印平均距离 + 过滤范围】 =====================
    std::cout << "===== 离群点过滤调试信息 =====" << std::endl;
    // std::cout << "点数量: " << input_points.size() << std::endl;
    // std::cout << "点云中心: x=" << center.x << ", y=" << center.y << std::endl;
    std::cout << "平均距离: " << avg_dist << " (像素/单位)" << std::endl;
    // std::cout << "过滤阈值系数: " << threshold << std::endl;
    // std::cout << "最大保留距离 = 平均距离 × 系数 = " << avg_dist * threshold << std::endl;
    // std::cout << "==============================" << std::endl;

    // 4. 过滤：距离 < 平均距离×系数 则保留
    const float max_dist = avg_dist * threshold;
    for (size_t i = 0; i < input_points.size(); ++i)
    {
        if (distances[i] < max_dist)
        {
            output_points.push_back(input_points[i]);
        }
    }

    std::cout << "过滤前点数: " << input_points.size() << ", 过滤后点数: " << output_points.size() << std::endl;
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
    euclidean_filter(input_cloud,output_cloud);
}

void Ten_post_pcl::set_vector_2d(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
    const Plane_Info& plane_info,
    std::vector<cv::Point2f>& output_2d)
{
    // 清空输出
    output_2d.clear();

    // 空指针/空点云校验
    if (!input_cloud || input_cloud->empty())  return;

    const Eigen::Vector3d origin = plane_info.plane_center;
    Eigen::Vector3d n = plane_info.plane_normal;

    Eigen::Vector3d u, v;
    getLocalAxes(n,u,v);
    output_2d.reserve(input_cloud->size());

    for (const auto& p : input_cloud->points)
    {
        const Eigen::Vector3d diff(p.x - origin.x(), p.y - origin.y(), p.z - origin.z());
        const double x_2d = float(diff.dot(u));
        const double y_2d = float(diff.dot(v));
        output_2d.emplace_back(x_2d, y_2d);
    }
}

} // namespace Plane_FitLocator
} // namespace Ten

#endif