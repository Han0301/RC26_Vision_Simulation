#ifndef __SET_PCL_H_
#define __SET_PCL_H_
#include <ros/ros.h>
#include <iostream>
#include <string>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>

namespace Ten
{
namespace Plane_FitLocator
{

class Ten_set_pcl
{
public:

    #define _interval_ 0
    #define _clouddepth_low_ 100
    #define _clouddepth_high_ 1000
    #define _step_uv_ 3

    /**
     * @brief 根据深度图像、内参，提取旋转检测框内的点云
     * @param depth_frame       原生深度帧
     * @param color_intr        彩色相机内参
     * @param obb               旋转检测框参数（Detection结构体）
     * @param pcl_cloud         输出的pcl点云
     * @param interval          宽高单边扩展量（像素）
     * @param CloudDepth_low    最低距离(mm)
     * @param CloudDepth_high   最高距离(mm)
     */
    bool set_Pcl_Cloud(
        const std::shared_ptr<rs2::depth_frame>& depth_frame,
        const rs2_intrinsics& color_intr,
        const Ten::yolo::Detection& obb,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& pcl_cloud,
        const double interval = _interval_,
        const double CloudDepth_low = _clouddepth_low_,
        const double CloudDepth_high = _clouddepth_high_
    )
    {
        pcl_cloud->clear();
        int w = depth_frame->get_width();
        int h = depth_frame->get_height();

        // 双边等比例扩大
        double adjusted_w = obb.w_ + 2 * interval; // 左右各扩interval
        double adjusted_h = obb.h_ + 2 * interval; // 上下各扩interval

        if (adjusted_w <= 0 || adjusted_h <= 0) {
            std::cout << "[错误] 检测框宽高为负！" << std::endl;
            return false;
        }

        // 角度单位转换 + 正确旋转矩阵 
        // 1. 角度转弧度！！！（如果你的obb.angle_已经是弧度，删除这一行）
        double theta_rad = obb.angle_;
        // 2. 正确计算余弦/正弦
        const float cos_theta = static_cast<float>(cos(theta_rad));
        const float sin_theta = static_cast<float>(sin(theta_rad));

        int point_count = 0;
        for (int v = 0; v < h; v += _step_uv_) {
            for (int u = 0; u < w; u += _step_uv_) {
                const float dx = static_cast<float>(u) - static_cast<float>(obb.cx_);
                //  修复1：图像v轴向下，必须对dy取反！ 
                const float dy = -(static_cast<float>(v) - static_cast<float>(obb.cy_));

                // 正确：像素坐标 → OBB局部坐标系（顺时针旋转，匹配视觉绘制）
                const float local_x = dx * cos_theta + dy * sin_theta;
                const float local_y = -dx * sin_theta + dy * cos_theta;

                //  修复2：恢复 /2.0！使用半宽半高判断（核心！） 
                const bool in_obb = (fabs(local_x) <= adjusted_w / 2.0f) && 
                                    (fabs(local_y) <= adjusted_h / 2.0f);
                
                if (!in_obb) continue;

                // 深度获取与过滤（原代码正确，保留）
                float z = depth_frame->get_distance(u, v); 
                double z_mm = z * 1000;
                if (z <= 0 || z_mm < CloudDepth_low || z_mm > CloudDepth_high) continue;

                // 反投影（原代码正确，保留）
                float pixel[2] = {(float)u, (float)v};
                float point3d[3] = {0};
                rs2_deproject_pixel_to_point(point3d, &color_intr, pixel, z);
                
                pcl::PointXYZ p;
                p.x = point3d[0];
                p.y = point3d[1];
                p.z = point3d[2];
                pcl_cloud->push_back(p);
                point_count++;
            }
        }

        if(point_count == 0){
            std::cout << "[错误] 旋转框内未提取到任何点云！" << std::endl;
            return false;
        }
        return true;
    }



private:


};

}       // namespace Plane_FitLocator
}       // namespace Ten
#endif 