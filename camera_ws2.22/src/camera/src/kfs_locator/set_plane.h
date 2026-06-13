#ifndef __SET_PLANE_H_
#define __SET_PLANE_H_
#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <Eigen/Dense>
#include <cmath>
#include "./../method_math.h"

namespace Ten
{
namespace kfs_locator
{
const float BOX_SIZE = 0.35;
const float SIZE_MIN_BIAS = 0.07;   
const float AREA_MIN_BIAS = 0.05; 
const float leaf_size_XY = 0.010f;               // 体素滤波XY尺寸，值越大点云越稀疏
const float leaf_size_Z  = 0.005f;               // 体素滤波Z尺寸，值越大点云越稀疏
const float DistanceThreshold = 0.016f;          // 平面拟合距离阈值，值越大拟合范围越大
const int  MaxIterations = 1000;                 // 平面拟合迭代次数，值越大精度越高
const float ClusterTolerance = 0.012;            // 欧式聚类容差，值越大聚类范围越大


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

class Ten_set_plane
{
public:

    // 点云组合滤波
    bool cloud_filter(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_pclclouds,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& out_pclclouds
    );

    // 平面拟合与点云提取
    bool Plane_fitter(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud,
        Plane_Info& plane_info
    );

    // 计算平面点云质心与初始姿态
    void compute_Center(
        pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud,
        Plane_Info& plane_info,
        bool is_first = true
    );

    // 转3d点云到2dvector容器
    void set_vector_2d(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
        const Plane_Info& plane_info,
        std::vector<cv::Point2f>& output_2d);

    // 形状筛选器
    bool shape_filter(const std::vector<cv::Point2f>& input_points);

    // 2D质心半径过滤
    void central_range_filter(
        const std::vector<cv::Point2f>& input_points,
        std::vector<cv::Point2f>& output_points,
        float distance_max = 0.265f);

    void set_yaw(Plane_Info& plane_info, const std::vector<cv::Point2f>& proj_points);

    void vector2f_to_pcl(
        const std::vector<cv::Point2f>& input_2d,
        const Plane_Info& plane_info,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud
        );

    Eigen::Vector3d cal_center_point(const Plane_Info& plane_info);

private:
    // 体素网格降采样
    void voxel_Downsample(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud
    );

    // 欧式聚类提取主点云
    void euclidean_filter(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud);

    // RANSAC算法拟合平面
    bool ransac_Plane_Segment(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
        pcl::PointIndices::Ptr& plane_inliers,
        Plane_Info& plane_info
    );

    // 提取平面内点云
    void extract_Plane_Cloud(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
        const pcl::PointIndices::Ptr& plane_inliers,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud,
        bool negative = false
    );

    // 基于法向量构建局部正交坐标轴
    void getLocalAxes(const Eigen::Vector3d& n,
                    Eigen::Vector3d& x_axis,
                    Eigen::Vector3d& y_axis);

    // 计算平面初始欧拉角
    void set_plane_euler(Plane_Info& plane_info);

}; // class Ten_set_plane


bool Ten_set_plane::cloud_filter(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_pclclouds,
    pcl::PointCloud<pcl::PointXYZ>::Ptr& out_pclclouds
)
{
    if(input_pclclouds->size() <= 50) return false;

    // 执行体素降采样
    pcl::PointCloud<pcl::PointXYZ>::Ptr mid_pclclouds(new pcl::PointCloud<pcl::PointXYZ>);
    voxel_Downsample(input_pclclouds, mid_pclclouds);
    euclidean_filter(mid_pclclouds,out_pclclouds);
    // 校验点云数量
    if(out_pclclouds->size() <= 50) return false;
    return true;
}

void Ten_set_plane::voxel_Downsample(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
    pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud
)
{
    // 初始化并配置体素滤波器
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(input_cloud);
    vg.setLeafSize(leaf_size_XY, leaf_size_XY, leaf_size_Z);
    vg.filter(*output_cloud);
}

void Ten_set_plane::euclidean_filter(
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

bool Ten_set_plane::ransac_Plane_Segment(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
    pcl::PointIndices::Ptr& plane_inliers,
    Plane_Info& plane_info
)
{
    // 初始化平面拟合参数
    pcl::ModelCoefficients::Ptr coeffs(new pcl::ModelCoefficients);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(DistanceThreshold);
    seg.setMaxIterations(MaxIterations);

    // 执行平面拟合
    seg.setInputCloud(input_cloud);
    seg.segment(*plane_inliers, *coeffs);

    // 计算并赋值平面法向量
    if(!plane_inliers->indices.empty())
    {
        float a = coeffs->values[0];
        float b = coeffs->values[1];
        float c = coeffs->values[2];
        float norm = sqrt(a*a + b*b + c*c);
        plane_info.plane_normal = Eigen::Vector3d(a/norm, b/norm, c/norm);
    }

    return !plane_inliers->indices.empty();
}

void Ten_set_plane::extract_Plane_Cloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
    const pcl::PointIndices::Ptr& plane_inliers,
    pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud,
    bool negative
)
{
    // 初始化并配置点云提取器
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(input_cloud);
    extract.setIndices(plane_inliers);
    extract.setNegative(negative); 
    extract.filter(*output_cloud);
}

bool Ten_set_plane::Plane_fitter(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
    pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud,
    Plane_Info& plane_info
)
{
    // 校验点云有效性
    if (input_cloud->empty() || input_cloud->size() < 50) return false;

    // 执行平面拟合
    pcl::PointIndices::Ptr plane_inliers(new pcl::PointIndices);
    if (!ransac_Plane_Segment(input_cloud, plane_inliers, plane_info))
    {
        std::cerr << "未检测到平面！" << std::endl;
        return false;
    }

    // 提取平面点云
    extract_Plane_Cloud(input_cloud, plane_inliers, output_cloud);

    return true;
}


void Ten_set_plane::compute_Center(
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud,
    Plane_Info& plane_info,
    bool is_first)
{
    // 计算点云质心
    Eigen::Vector4f centroid_float;
    pcl::compute3DCentroid(*input_cloud, centroid_float);
    plane_info.plane_center = Eigen::Vector3d(centroid_float[0], centroid_float[1], centroid_float[2]);

    // 计算初始欧拉角
    if (is_first)
    {   
        set_plane_euler(plane_info);
    }
}

void Ten_set_plane::getLocalAxes(const Eigen::Vector3d& n, Eigen::Vector3d& x, Eigen::Vector3d& y)
{
    Eigen::Vector3d normal = n.normalized();
    Eigen::Vector3d aux = Eigen::Vector3d(0,0,1);
    if(fabs(normal.dot(aux)) > 0.999) aux = Eigen::Vector3d(1,0,0);

    x = aux.cross(normal).normalized();
    y = normal.cross(x).normalized();
}

void Ten_set_plane::set_plane_euler(Plane_Info& plane_info)
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
    plane_info.plane_euler._yaw   = 0.0;
}

void Ten_set_plane::set_yaw(Plane_Info& plane_info, const std::vector<cv::Point2f>& proj_points)
{
    if (proj_points.empty())
    {
        plane_info.plane_euler._yaw = 0.0;
        return;
    }

    // 1. 计算矩形角度
    std::vector<cv::Point2f> hull;
    cv::convexHull(proj_points, hull);
    cv::RotatedRect rect = cv::minAreaRect(hull);
    float ang = rect.angle;
    cv::Size2f sz = rect.size;
    if(sz.width > sz.height) ang += 90.f;
    double yaw_angle = ang * CV_PI / 180.0;

    // 2. 重建基础旋转矩阵（Z轴=法向量，绝对垂直）
    Eigen::Vector3d n = plane_info.plane_normal;
    Eigen::Vector3d x_axis, y_axis;
    getLocalAxes(n, x_axis, y_axis);
    Eigen::Matrix3d rot;
    rot.col(0) = x_axis;
    rot.col(1) = y_axis;
    rot.col(2) = n;

    // 3. 绕【法向量Z轴】旋转
    Eigen::Matrix3d yaw_rot;
    yaw_rot = Eigen::AngleAxisd(yaw_angle, n);  // 绕法向量旋转
    rot = yaw_rot * rot;

    // 4. 重新计算正确RPY
    plane_info.plane_euler._roll = std::atan2(rot(2,1), rot(2,2));
    plane_info.plane_euler._pitch = std::atan2(-rot(2,0), std::hypot(rot(2,1), rot(2,2)));
    plane_info.plane_euler._yaw = std::atan2(rot(1,0), rot(0,0));
}

void Ten_set_plane::set_vector_2d(
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
        const float x_2d = float(diff.dot(u));
        const float y_2d = float(diff.dot(v));
        output_2d.emplace_back(x_2d, y_2d);
    }
}

bool Ten_set_plane::shape_filter(const std::vector<cv::Point2f>& input_points)
{
    if (input_points.size() < 50) return false;

    std::vector<cv::Point2f> hull;
    cv::convexHull(input_points, hull);  // 计算凸包
    const cv::RotatedRect rect = cv::minAreaRect(hull);  // 用凸包算矩形，极稳

    // const cv::RotatedRect rect = cv::minAreaRect(input_points);
    const float faceArea = rect.size.width * rect.size.height;
    // std::cout << "rect.size.width: " << rect.size.width << ", rect.size.height: " << rect.size.height << std::endl;
    // std::cout << "faceArea: " << std::abs(faceArea - BOX_SIZE * BOX_SIZE) << std::endl;
    if (std::abs(rect.size.width - BOX_SIZE) < SIZE_MIN_BIAS
        && std::abs(rect.size.height - BOX_SIZE) < SIZE_MIN_BIAS && std::abs(faceArea - BOX_SIZE * BOX_SIZE) < AREA_MIN_BIAS)
    {
        return true;
    }
    return false;
}

void Ten_set_plane::central_range_filter(
    const std::vector<cv::Point2f>& input_points,
    std::vector<cv::Point2f>& output_points,
    float distance_max)
{
    // 1. 清空输出
    output_points.clear();

    // 2. 空值判断
    if (input_points.empty()) return;

    // 3. 计算2D点质心
    cv::Point2f centroid(0.0f, 0.0f);
    for (const auto& pt : input_points)
    {
        centroid += pt;
    }
    centroid.x /= input_points.size();
    centroid.y /= input_points.size();

    for (const auto& pt : input_points)
    {
        // 计算到质心的欧氏距离
        float dx = pt.x - centroid.x;
        float dy = pt.y - centroid.y;
        float dist = std::sqrt(dx*dx + dy*dy);

        // 核心：保留范围内的点
        if (dist <= distance_max)
        {
            output_points.push_back(pt);
        }
    }
}

void Ten_set_plane::vector2f_to_pcl(
    const std::vector<cv::Point2f>& local_2d_points,
    const Plane_Info& plane_info,
    pcl::PointCloud<pcl::PointXYZ>::Ptr& world_cloud)
{
    world_cloud->clear();
    world_cloud->reserve(local_2d_points.size());

    // 获取平面局部坐标系轴（和投影时完全一致）
    Eigen::Vector3d n = plane_info.plane_normal.normalized();
    Eigen::Vector3d u, v;
    getLocalAxes(n, u, v);
    Eigen::Vector3d center = plane_info.plane_center;

    // 逆转换：局部2D(x,y) → 世界3D点 = 质心 + x*u + y*v
    for (const auto& pt : local_2d_points)
    {
        Eigen::Vector3d world_pt = center + pt.x * u + pt.y * v;
        world_cloud->push_back(pcl::PointXYZ(world_pt.x(), world_pt.y(), world_pt.z()));
    }

    world_cloud->width = world_cloud->size();
    world_cloud->height = 1;
    world_cloud->is_dense = true;
}

Eigen::Vector3d Ten_set_plane::cal_center_point(const Plane_Info& plane_info)
{
    const double offset_distance = BOX_SIZE / 2.0f;

    // 2. 获取平面单位法向量（已归一化，直接使用）
    Eigen::Vector3d normal = plane_info.plane_normal;

    // 3. 获取原始平面中心点
    Eigen::Vector3d plane_center = plane_info.plane_center;

    // 4. 沿法向量**反方向**偏移计算新中心点
    Eigen::Vector3d body_center = plane_center - normal * offset_distance;
    return body_center;
}

} // namespace kfs_locator
} // namespace Ten

#endif