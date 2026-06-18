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

#define leaf_size_XY 0.010f             // 体素滤波XY尺寸，值越大点云越稀疏
#define leaf_size_Z  0.005f             // 体素滤波Z尺寸，值越大点云越稀疏
#define DistanceThreshold 0.016f         // 平面拟合距离阈值，值越大拟合范围越大
#define MaxIterations 1000              // 平面拟合迭代次数，值越大精度越高
#define ClusterTolerance 0.012           // 欧式聚类容差，值越大聚类范围越大

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
}; // class Ten_pre_pcl

bool Ten_pre_pcl::cloud_filter(
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

void Ten_pre_pcl::voxel_Downsample(
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
void Ten_pre_pcl::euclidean_filter(
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