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
#include <pcl/registration/icp.h>
#include "./../method_math.h"

#define BOX_SIZE 0.35
#define SIZE_MIN_BIAS 0.07   
#define AREA_MIN_BIAS 0.05 

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
        float distance_max = 0.25f);

    void set_yaw(Plane_Info& plane_info, const std::vector<cv::Point2f>& proj_points);

    void vector2f_to_pcl(
        const std::vector<cv::Point2f>& input_2d,
        const Plane_Info& plane_info,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud
        );

    Eigen::Vector3d cal_center_point(const Plane_Info& plane_info);
private:
    // 基于法向量构建局部正交坐标轴
    void getLocalAxes(const Eigen::Vector3d& n,
                    Eigen::Vector3d& x_axis,
                    Eigen::Vector3d& y_axis);

    // 计算平面初始欧拉角
    void set_plane_euler(Plane_Info& plane_info);

};      // class Ten_post_pcl

void Ten_post_pcl::compute_Center(
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

void Ten_post_pcl::getLocalAxes(const Eigen::Vector3d& n, Eigen::Vector3d& x, Eigen::Vector3d& y)
{
    Eigen::Vector3d normal = n.normalized();
    Eigen::Vector3d aux = Eigen::Vector3d(0,0,1);
    if(fabs(normal.dot(aux)) > 0.999) aux = Eigen::Vector3d(1,0,0);

    x = aux.cross(normal).normalized();
    y = normal.cross(x).normalized();
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
    plane_info.plane_euler._yaw   = 0.0;
}

// 唯一正确的set_yaw：矩阵级绕法向量旋转，无畸变、轴全对
void Ten_post_pcl::set_yaw(Plane_Info& plane_info, const std::vector<cv::Point2f>& proj_points)
{
    if (proj_points.empty())
    {
        plane_info.plane_euler._yaw = 0.0;
        return;
    }

    // 1. 计算矩形角度（不变）
    std::vector<cv::Point2f> hull;
    cv::convexHull(proj_points, hull);
    cv::RotatedRect rect = cv::minAreaRect(hull);
    float ang = rect.angle;
    cv::Size2f sz = rect.size;
    if(sz.width > sz.height) ang += 90.f;
    double yaw_angle = ang * CV_PI / 180.0;

    // ===================== 核心修复：矩阵旋转，不直接改yaw =====================
    // 2. 重建基础旋转矩阵（Z轴=法向量，绝对垂直）
    Eigen::Vector3d n = plane_info.plane_normal;
    Eigen::Vector3d x_axis, y_axis;
    getLocalAxes(n, x_axis, y_axis);
    Eigen::Matrix3d rot;
    rot.col(0) = x_axis;
    rot.col(1) = y_axis;
    rot.col(2) = n;

    // 3. 绕【法向量Z轴】旋转（唯一不会畸变的方式）
    Eigen::Matrix3d yaw_rot;
    yaw_rot = Eigen::AngleAxisd(yaw_angle, n);  // 绕法向量旋转！
    rot = yaw_rot * rot;

    // 4. 重新计算正确RPY（矩阵→欧拉角，无偏差）
    plane_info.plane_euler._roll = std::atan2(rot(2,1), rot(2,2));
    plane_info.plane_euler._pitch = std::atan2(-rot(2,0), std::hypot(rot(2,1), rot(2,2)));
    plane_info.plane_euler._yaw = std::atan2(rot(1,0), rot(0,0));
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
        const float x_2d = float(diff.dot(u));
        const float y_2d = float(diff.dot(v));
        output_2d.emplace_back(x_2d, y_2d);
    }
}

bool Ten_post_pcl::shape_filter(const std::vector<cv::Point2f>& input_points)
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

void Ten_post_pcl::central_range_filter(
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

void Ten_post_pcl::vector2f_to_pcl(
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

Eigen::Vector3d Ten_post_pcl::cal_center_point(const Plane_Info& plane_info)
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
} // namespace Plane_FitLocator
} // namespace Ten

#endif