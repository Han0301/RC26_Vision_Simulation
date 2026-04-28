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

    /**
     * @brief 根据深度图像， 内参， 直接转成pcl点云
     * @param depth_frame       原生深度帧
     * @param color_intr        彩色相机内参
     * @param pcl_cloud         输出的pcl_cloud点云
     * @param CloudDepth_low    能接受的点云处理的最低距离(mm)， 小于这个距离直接过滤
     * @param CloudDepth_high   能接受的点云处理的最高距离(mm)， 大于这个距离直接过滤
    */
    void set_Pcl_Cloud(
    const std::shared_ptr<rs2::depth_frame>& depth_frame,
        const rs2_intrinsics& color_intr,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& pcl_cloud,
        const int CloudDepth_low = 100,
        const int CloudDepth_high = 1000
    )
    {
        pcl_cloud->clear();
        int w = depth_frame->get_width();
        int h = depth_frame->get_height();

        for (int v = 0; v < h; v++) {
            for (int u = 0; u < w; u++) {
                // 原生精准深度（0.1mm精度，斜视角无误差）
                float z = depth_frame->get_distance(u, v); 
                int z_mm = int(z * 1000);

                // 过滤无效深度
                if (z <= 0 || z_mm < CloudDepth_low || z_mm > CloudDepth_high)
                    continue;

                // 反投影（align后用彩色内参，坐标系100%对齐）
                float pixel[2] = {(float)u, (float)v};
                float point3d[3] = {0};
                rs2_deproject_pixel_to_point(point3d, &color_intr, pixel, z);
                
                pcl::PointXYZ p;
                p.x = point3d[0];
                p.y = point3d[1];
                p.z = point3d[2];
                pcl_cloud->push_back(p);
            }
        }
    }

private:


};

}       // namespace Plane_FitLocator
}       // namespace Ten
#endif 