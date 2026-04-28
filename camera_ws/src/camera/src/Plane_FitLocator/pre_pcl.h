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
#include "./../method_math.h"

namespace Ten
{
namespace Plane_FitLocator
{
struct Plane_Info
{
    Eigen::Vector3d plane_center = Eigen::Vector3d();
    RPY plane_normal;
};

class Ten_pre_pcl
{
public:

    Ten_pre_pcl()
    {
        plane_coeffs.reset(new pcl::ModelCoefficients);
    }

    /**
     * @brief 点云滤波， 统计离群滤波
     * @param input_pclclouds 输入点云
     * @param out_pclclouds 输出的点云
     * @param MeanK 每个点要找的「最近邻点数」， 约小约快， 容易误删正常点，越大计算慢，过度平滑
     * @param StddevMulThresh 距离超过「平均值 + 2 倍标准差」的点，全部删掉,数值越小 = 删得越狠, 数值越大 = 删得越少
    */
    void cloud_filter(
        pcl::PointCloud<pcl::PointXYZ>::Ptr input_pclclouds,
        pcl::PointCloud<pcl::PointXYZ>::Ptr out_pclclouds,
        const int MeanK = 40,
        const float StddevMulThresh = 1.0f
    )
    {
        // 统计离群滤波
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(input_pclclouds);        // 输入点云
        sor.setMeanK(MeanK);                                  // 近邻点数（和你原参数一致）
        sor.setStddevMulThresh(StddevMulThresh);              // 标准差系数（和你原参数一致）
        
        // 执行滤波
        sor.filter(*out_pclclouds); 
    }

    /**
     * @brief 提取点云中【最大平面】
     * @param input_cloud 输入点云（必须先滤波！）
     * @param output_cloud 输出点云（拟合的最大平面的点云） 
     * @param DistanceThreshold 点到 拟合平面的距离上限(m), 默认为 0.02f
     * @return 拟合成功返回true
     */
    bool Plane_fitter(
        pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud,
        pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud,
        const float DistanceThreshold = 0.01f,
        const int MaxIterations = 1500
    )
    {
        // 1 输出检查
        if (input_cloud->empty() || input_cloud->size() < 50)
        {
            std::cout << "Plane_fitter: input_cloud->empty() || input_cloud->size() < 50" << std::endl;
            return false;
        }

        // 2. RANSAC 分割【最大平面】
        pcl::PointIndices::Ptr plane_inliers(new pcl::PointIndices);          // 平面内点索引

        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true);              // 优化系数，精度更高
        seg.setModelType(pcl::SACMODEL_PLANE);          // 模型：平面
        seg.setMethodType(pcl::SAC_RANSAC);             // 算法：抗噪 RANSAC
        seg.setDistanceThreshold(DistanceThreshold);    // 内点阈值：1cm（深度相机标准）
        seg.setMaxIterations(MaxIterations);            // 迭代次数，保证找到最优平面

        seg.setInputCloud(input_cloud);
        seg.segment(*plane_inliers, *plane_coeffs);

        // 检查是否找到平面
        if (plane_inliers->indices.empty())
        {
            std::cerr << "未检测到平面！" << std::endl;
            return false;
        }
        // 3. 提取最大平面的点云
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(input_cloud);
        extract.setIndices(plane_inliers);
        extract.filter(*output_cloud);

        std::cout << "✅ 最大平面点数：" << output_cloud->size() << "/" << input_cloud->size() << std::endl;
        return true;
    }

    /**
     * @brief 根据输入的最大平面点云，写入该平面的质心(中心点)和单位法向量
     * @param input_cloud 输入点云（必须先滤波！）
     * @param DistanceThreshold 点到 拟合平面的距离上限(m), 默认为 0.02f
     * @return 拟合成功返回true
     */
    void computeCenterAndNormal(
        pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud,
        const float DistanceThreshold = 0.02f
    )
    {
        // 1. 计算【最大平面的质心】（中心点）
        Eigen::Vector4f centroid_float;
        pcl::compute3DCentroid(*input_cloud, centroid_float);
        plane_info.plane_center.x() = centroid_float[0];
        plane_info.plane_center.y() = centroid_float[1];
        plane_info.plane_center.z() = centroid_float[2];

        //  2. 计算【单位法向量】并归一化 
        float a = plane_coeffs->values[0];
        float b = plane_coeffs->values[1];
        float c = plane_coeffs->values[2];
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
        // 3. 笛卡尔法向量 → 转为 RPY 欧拉角（适配你的结构体）
        // vectorToRPY(nx, ny, nz, plane_info.plane_normal);
        createPlaneCoordinate(nx, ny, nz, plane_info.plane_normal);
    }

    /**
     * @brief 取到 plane_info, 内部含有中心点和法向量
    */
    Plane_Info get_plane_info(bool is_pre)
    {
        return plane_info;
    }

private:
    Plane_Info plane_info; 
    pcl::ModelCoefficients::Ptr plane_coeffs; // 平面方程系数

    /**
     * @brief 笛卡尔单位向量 → RPY 欧拉角
     */
    void vectorToRPY(float nx, float ny, float nz, RPY& rpy)
    {
        // 平面法向量转换为姿态角（相机坐标系标准）
        rpy._roll  = atan2(ny, nz);
        rpy._pitch = atan2(-nx, sqrt(ny*ny + nz*nz));
        rpy._yaw   = 0.0; // 平面无偏航角，固定为0
    }

    /**
     * @brief 构建【标准平面坐标系】Z=法向量(垂直平面)，X/Y=平面内(贴合平面)
     */
    void createPlaneCoordinate(float nx, float ny, float nz, RPY& rpy)
    {
        // 1. 平面法向量 = Z轴 (垂直平面)
        Eigen::Vector3d n(nx, ny, nz);
        n.normalize();

        // 2. 构造平面内的X轴（正交于Z轴，绝对贴合平面）
        Eigen::Vector3d up(0, 1, 0);  // 参考方向
        if (fabs(n.dot(up)) > 0.99) up = Eigen::Vector3d(1, 0, 0); // 避免共线
        Eigen::Vector3d x = up.cross(n).normalized();

        // 3. 构造平面内的Y轴（正交于X/Z，贴合平面）
        Eigen::Vector3d y = n.cross(x).normalized();

        // 4. 旋转矩阵 → 转RPY（TF坐标系：X/Y在平面，Z垂直平面）
        Eigen::Matrix3d rot_mat;
        rot_mat.col(0) = x;
        rot_mat.col(1) = y;
        rot_mat.col(2) = n;

        // 矩阵转欧拉角(roll,pitch,yaw) → 赋值给你的RPY结构体
        rpy._roll  = atan2(rot_mat(2,1), rot_mat(2,2));
        rpy._pitch = atan2(-rot_mat(2,0), sqrt(rot_mat(2,1)*rot_mat(2,1) + rot_mat(2,2)*rot_mat(2,2)));
        rpy._yaw   = atan2(rot_mat(1,0), rot_mat(0,0));
    }
};

}       // namespace Plane_FitLocator
}       // namespace Ten
#endif 