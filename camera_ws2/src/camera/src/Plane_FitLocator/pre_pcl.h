#ifndef __PRE_PCL_H_
#define __PRE_PCL_H_
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
#include <pcl/common/centroid.h>
#include <opencv2/core/types.hpp>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/common/transforms.h>
#include "post_pcl.h"

#define MeanK  40                      // 统计滤波邻域点数，值越大滤波效果越强
#define StddevMulThresh 1.0f            // 统计滤波阈值，值越大保留点云越多
#define leaf_size_XY 0.008f             // 体素滤波XY尺寸，值越大点云越稀疏
#define leaf_size_Z  0.008f             // 体素滤波Z尺寸，值越大点云越稀疏
#define DistanceThreshold 0.02f         // 平面拟合距离阈值，值越大拟合范围越大
#define MaxIterations 600              // 平面拟合迭代次数，值越大精度越高

namespace Ten
{
namespace Plane_FitLocator
{

// 点云预处理类：实现点云滤波、平面拟合、点云提取
class Ten_pre_pcl
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

private:
    // 体素网格降采样
    void voxel_Downsample(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud
    );

    // 统计滤波去除噪声点
    void statistical_filter(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_pclclouds,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& out_pclclouds
    );

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
        pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud
    );
}; // class Ten_pre_pcl

bool Ten_pre_pcl::cloud_filter(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_pclclouds,
    pcl::PointCloud<pcl::PointXYZ>::Ptr& out_pclclouds
)
{
    // 执行体素降采样
    voxel_Downsample(input_pclclouds, out_pclclouds);

    // 执行统计滤波
    statistical_filter(out_pclclouds, out_pclclouds);

    // 校验点云数量
    if(out_pclclouds->size() <= 50)
    {
        return false;
    }

    return true;
}

void Ten_pre_pcl::voxel_Downsample(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
    pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud
)
{
    // 判断点云是否为空
    if (input_cloud->empty())
    {
        return;
    }

    // 初始化并配置体素滤波器
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(input_cloud);
    vg.setLeafSize(leaf_size_XY, leaf_size_XY, leaf_size_Z);
    vg.filter(*output_cloud);
}

void Ten_pre_pcl::statistical_filter(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_pclclouds,
    pcl::PointCloud<pcl::PointXYZ>::Ptr& out_pclclouds
)
{
    // 初始化并配置统计滤波器
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(input_pclclouds);
    sor.setMeanK(MeanK);
    sor.setStddevMulThresh(StddevMulThresh);
    sor.filter(*out_pclclouds);
}

// 平面拟合器
bool Ten_pre_pcl::ransac_Plane_Segment(
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

void Ten_pre_pcl::extract_Plane_Cloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
    const pcl::PointIndices::Ptr& plane_inliers,
    pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud
)
{
    // 初始化并配置点云提取器
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(input_cloud);
    extract.setIndices(plane_inliers);
    extract.filter(*output_cloud);
}

bool Ten_pre_pcl::Plane_fitter(
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

} // namespace Plane_FitLocator
} // namespace Ten

#endif