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
#define SIZE_MIN_BIAS 0.05   
#define AREA_MIN_BIAS 0.0375   
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

    // 转3d点云到2dvector容器
    void set_vector_2d(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
        const Plane_Info& plane_info,
        std::vector<cv::Point2f>& output_2d);

    // 形状筛选器
    bool shape_filter(const std::vector<cv::Point2f>& input_points);

    // 优化平面偏航角并更新姿态
    void set_RPY(
        const std::vector<cv::Point2f>& plane_2d_points,
        Plane_Info& plane_info);

    // 2D质心半径过滤
    void central_range_filter(
        const std::vector<cv::Point2f>& input_points,
        std::vector<cv::Point2f>& output_points,
        float distance_max = 0.25f);

    void vector2f_to_pcl(
    const std::vector<cv::Point2f>& input_2d,
    const Plane_Info& plane_info,
    pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud
    );
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
    Eigen::Vector3d norm_n = n.normalized();
    // 🔥 固定：永远用世界Z轴构建平面坐标系，绝对不跳变
    Eigen::Vector3d ref_up = Eigen::Vector3d::UnitZ();
    // 正交化：平面X轴 = 世界上方向 × 平面法向
    x_axis = ref_up.cross(norm_n).normalized();
    // 平面Y轴 = 法向 × X轴（保证右手坐标系，正方形永远在平面内）
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
    for (double angle = -90.0; angle <= 90.0; angle += 10.0)
    {
        int score = countInFixedBox(rotatePointCloud2D(plane_2d_points, angle));
        if (score > max_score)
        {
            max_score = score;
            best_angle = angle;
        }
    }

    // 3 精角度搜索
    double start2 = best_angle - 10.0;
    double end2 = best_angle + 10.0;
    for (double angle = start2; angle <= end2; angle += 2.0)
    {
        int score = countInFixedBox(rotatePointCloud2D(plane_2d_points, angle));
        if (score > max_score)
        {
            max_score = score;
            best_angle = angle;
        }
    }

    // 4 细角度搜索
    double start3 = best_angle - 2;
    double end3   = best_angle + 2;
    for (double angle = start3; angle <= end3; angle += 0.2)
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

    const cv::RotatedRect rect = cv::minAreaRect(input_points);
    const float faceArea = rect.size.width * rect.size.height;
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

    // 4. 遍历计算距离 + 筛选 + 统计信息
    float total_dist = 0.0f;
    float max_dist = 0.0f;
    std::vector<float> dist_list;

    for (const auto& pt : input_points)
    {
        // 计算到质心的欧氏距离
        float dx = pt.x - centroid.x;
        float dy = pt.y - centroid.y;
        float dist = std::sqrt(dx*dx + dy*dy);

        dist_list.push_back(dist);
        total_dist += dist;
        if (dist > max_dist) max_dist = dist;

        // 核心：保留范围内的点
        if (dist <= distance_max)
        {
            output_points.push_back(pt);
        }
    }
    // 5. 调试打印（你要的分析信息）
    float avg_dist = total_dist / input_points.size();
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
} // namespace Plane_FitLocator
} // namespace Ten

#endif