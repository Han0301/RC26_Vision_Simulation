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
#include <pcl/common/centroid.h>  // 质心计算
#include <opencv2/core/types.hpp>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/common/transforms.h>
#include "./../method_math.h"

#include "post_pcl.h"
// 统计离群滤波 参数
#define MeanK  40                   // 每个点要找的「最近邻点数」， 约小约快， 容易误删正常点，越大计算慢，过度平滑
#define StddevMulThresh 1.0f        // 距离超过「平均值 + 2 倍标准差」的点，全部删掉,数值越小 = 删得越狠, 数值越大 = 删得越少

// 体素下采样 参数
#define leaf_size_XY 0.006f            // 体素xy边长
#define leaf_size_Z  0.010f            // 体素z边长

// 平面拟合 参数
#define DistanceThreshold 0.02f     // 点到 拟合平面的距离上限(m), 默认为 0.02f
#define MaxIterations 1500          // 迭代次数，保证找到最优平面

namespace Ten
{
namespace Plane_FitLocator
{

class Ten_pre_pcl
{
public:

    /**
     * @brief 点云滤波器
     * @param input_pclclouds 输入点云
     * @param out_pclclouds 输出的点云
     * @return bool
    */
    bool cloud_filter(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_pclclouds,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& out_pclclouds
    )
    {
        voxel_Downsample(input_pclclouds,out_pclclouds);
        statistical_filter(out_pclclouds,out_pclclouds);
        if(out_pclclouds->size() <= 50)
        {
            std::cout << "out_pclclouds->size(): " <<out_pclclouds->size() <<std::endl;
            return false;
        }
        return true;
    }

    /**
     * @brief 提取点云中【最大平面】, 填充 plane_info 中 plane_coeffs 字段
     * @param input_cloud   输入点云
     * @param output_cloud  输出点云（拟合的最大平面的点云） 
     * @param plane_info    面的相关信息
     * @return 拟合成功返回true
     */
    bool Plane_fitter(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud,
        Plane_Info& plane_info
    );

private:

    // 体素下采样
    void voxel_Downsample(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud)
    {
        if (input_cloud->empty())
            return;

        // 创建体素网格滤波器
        pcl::VoxelGrid<pcl::PointXYZ> vg;
        vg.setInputCloud(input_cloud);
        // 设置体素大小（核心参数）
        vg.setLeafSize(leaf_size_XY, leaf_size_XY, leaf_size_Z);
        // 执行滤波
        vg.filter(*output_cloud);
    }

    // 统计离群滤波
    void statistical_filter(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_pclclouds,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& out_pclclouds
    )
    {
        // 统计离群滤波
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(input_pclclouds);                   // 输入点云
        sor.setMeanK(MeanK);                                  // 近邻点数（和你原参数一致）
        sor.setStddevMulThresh(StddevMulThresh);              // 标准差系数（和你原参数一致）
        
        // 执行滤波
        sor.filter(*out_pclclouds); 
    }

    // RANSAC拟合平面，仅计算平面内点索引和方程系数
    bool ransac_Plane_Segment(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
        pcl::PointIndices::Ptr& plane_inliers,
        Plane_Info& plane_info
    )
    {
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(DistanceThreshold);
        seg.setMaxIterations(MaxIterations);

        seg.setInputCloud(input_cloud);
        seg.segment(*plane_inliers, *plane_info.plane_coeffs); // 计算索引+系数

        return !plane_inliers->indices.empty();
    }

    // 根据点云索引，提取对应点云
    void extract_Plane_Cloud(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
        const pcl::PointIndices::Ptr& plane_inliers,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud
    )
    {
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(input_cloud);
        extract.setIndices(plane_inliers);
        extract.filter(*output_cloud);
    }

};      // class Ten_pre_pcl

    bool Ten_pre_pcl::Plane_fitter(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud,
        Plane_Info& plane_info
    )
    {
        // 1 输出检查
        if (input_cloud->empty() || input_cloud->size() < 50)
        {
            std::cout << "Plane_fitter: input_cloud->empty() || input_cloud->size() < 50" << std::endl;
            return false;
        }
        // 2. RANSAC 分割【最大平面】
        pcl::PointIndices::Ptr plane_inliers(new pcl::PointIndices);
        if (!ransac_Plane_Segment(input_cloud, plane_inliers, plane_info))
        {
            std::cerr << "未检测到平面！" << std::endl;
            return false;
        }
        // 3. 提取最大平面的点云
        extract_Plane_Cloud(input_cloud, plane_inliers, output_cloud);

        std::cout << "✅ 最大平面点数：" << output_cloud->size() << "/" << input_cloud->size() << std::endl;
        return true;
    }
}       // namespace Plane_FitLocator
}       // namespace Ten
#endif 