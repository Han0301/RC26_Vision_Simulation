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
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& plane_3d_cloud,
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


    Eigen::Matrix3d last_R_ = Eigen::Matrix3d::Identity();  // 上一帧旋转矩阵
    bool has_last_ = false;  // 帧标记
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
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& plane_3d_cloud,
    Plane_Info& plane_info)
{
    if (plane_3d_cloud->empty()) return;

    Eigen::Vector3d center = plane_info.plane_center;
    Eigen::Vector3d n = plane_info.plane_normal.normalized();
    if (n.z() < 0) n = -n;

    std::vector<cv::Point2f> pts_2d;
    set_vector_2d(plane_3d_cloud, plane_info, pts_2d);
    if (pts_2d.empty()) return;

    cv::RotatedRect rect = cv::minAreaRect(pts_2d);
    cv::Point2f verts[4];
    rect.points(verts);
    cv::Point2f e1 = verts[1] - verts[0];
    cv::Point2f e2 = verts[2] - verts[1];
    cv::Point2f long_dir = (cv::norm(e1) > cv::norm(e2)) ? e1 : e2;

    // =========新增：防长短边90°翻转校验=========
    if(has_last_)
    {
        // 当前候选yaw
        Eigen::Vector3d temp_u, temp_v;
        getLocalAxes(n, temp_u, temp_v);
        Eigen::Vector3d tmpX = (long_dir.x * temp_u + long_dir.y * temp_v).normalized();
        double currYawTmp = std::atan2(tmpX.y(),tmpX.x());
        double lastYaw = std::atan2(last_R_.col(0).y(),last_R_.col(0).x());
        double deltaYaw = std::fmod(currYawTmp-lastYaw,2*M_PI);
        if(deltaYaw>M_PI) deltaYaw -= 2*M_PI;
        if(deltaYaw<-M_PI) deltaYaw += 2*M_PI;
        // 差值超45°，判定长短边选反，互换长边短边
        if(std::fabs(deltaYaw) > M_PI*0.25)
        {
            long_dir = (cv::norm(e1) < cv::norm(e2)) ? e1 : e2;
        }
    }

    Eigen::Vector3d temp_u, temp_v;
    getLocalAxes(n, temp_u, temp_v);
    Eigen::Vector3d target_x = (long_dir.x * temp_u + long_dir.y * temp_v).normalized();
    Eigen::Vector3d target_y = n.cross(target_x).normalized();
    Eigen::Matrix3d curr_R;
    curr_R.col(0) = target_x;
    curr_R.col(1) = target_y;
    curr_R.col(2) = n;

    if (has_last_)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr ref_box(new pcl::PointCloud<pcl::PointXYZ>);
        float half_box = BOX_SIZE / 2.0f;
        // =========新增：加密方框采样（边框均匀采点，不再只有4角）=========
        std::vector<cv::Point2f> box_pts;
        float step = 0.05f;
        for(float t=-half_box;t<=half_box;t+=step){box_pts.emplace_back(-half_box,t);}
        for(float t=-half_box;t<=half_box;t+=step){box_pts.emplace_back( half_box,t);}
        for(float t=-half_box;t<=half_box;t+=step){box_pts.emplace_back(t,-half_box);}
        for(float t=-half_box;t<=half_box;t+=step){box_pts.emplace_back(t, half_box);}

        for (auto& p : box_pts) {
            Eigen::Vector3d wp = center + p.x*last_R_.col(0) + p.y*last_R_.col(1);
            ref_box->push_back(pcl::PointXYZ(wp.x(), wp.y(), wp.z()));
        }

        pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
        icp.setInputSource(ref_box);
        icp.setInputTarget(plane_3d_cloud);
        icp.setMaxCorrespondenceDistance(0.12);
        icp.setMaximumIterations(40);

        pcl::PointCloud<pcl::PointXYZ> final_cloud;
        icp.align(final_cloud);

        if (icp.hasConverged()) {
            Eigen::Matrix3d icp_R = icp.getFinalTransformation().block<3,3>(0,0).cast<double>();
            curr_R = icp_R * curr_R;
        }
    }

    // 原始单帧角度
    double r = std::atan2(curr_R(2,1), curr_R(2,2));
    double p = std::atan2(-curr_R(2,0), std::hypot(curr_R(2,1), curr_R(2,2)));
    double y = std::atan2(curr_R(1,0), curr_R(0,0));

    // =========新增：低通滤波平滑小幅抖动 alpha=0.3可调，越小越稳=========
    const double alpha = 0.3;
    if(!has_last_)
    {
        plane_info.plane_euler._roll  = r;
        plane_info.plane_euler._pitch = p;
        plane_info.plane_euler._yaw   = y;
    }
    else
    {
        plane_info.plane_euler._roll  = alpha*r + (1-alpha)*plane_info.plane_euler._roll;
        plane_info.plane_euler._pitch = alpha*p + (1-alpha)*plane_info.plane_euler._pitch;
        plane_info.plane_euler._yaw   = alpha*y + (1-alpha)*plane_info.plane_euler._yaw;
    }

    last_R_.col(0) = Eigen::AngleAxisd(plane_info.plane_euler._yaw,Eigen::Vector3d::UnitZ())
                   *Eigen::Vector3d::UnitX();
    last_R_.col(1) = n.cross(last_R_.col(0)).normalized();
    last_R_.col(2) = n;
    has_last_ = true;

    std::cout << "✅ ICP稳定Yaw：" << plane_info.plane_euler._yaw * 180/M_PI << "°" << std::endl;
}

void Ten_post_pcl::getLocalAxes(const Eigen::Vector3d& n,
                                Eigen::Vector3d& x_axis,
                                Eigen::Vector3d& y_axis)
{
    Eigen::Vector3d normal = n.normalized();
    Eigen::Matrix3d R;
    R.setIdentity();

    if (fabs(normal(2)) < 0.9) {
        R.col(0) = normal.unitOrthogonal();
        R.col(1) = normal.cross(R.col(0)).normalized();
        R.col(2) = normal;
    } else {
        R.col(1) = normal.unitOrthogonal();
        R.col(0) = R.col(1).cross(normal).normalized();
        R.col(2) = normal;
    }

    x_axis = R.col(0);
    y_axis = R.col(1);
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
    std::cout << "rect.size.width: " << rect.size.width << ", rect.size.height: " << rect.size.height << std::endl;
    std::cout << "faceArea: " << std::abs(faceArea - BOX_SIZE * BOX_SIZE) << std::endl;
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