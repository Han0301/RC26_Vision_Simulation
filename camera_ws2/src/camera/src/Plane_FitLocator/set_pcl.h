#ifndef __SET_PCL_H_
#define __SET_PCL_H_
#include <ros/ros.h>
#include <iostream>
#include <string>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include "../openvino.h"

#define CloudDepth_min 200
#define CloudDepth_max 1800

namespace Ten
{
namespace Plane_FitLocator
{
class Ten_set_pcl
{
public:

    Ten_set_pcl()
        :detector("/home/h/下载/卷轴检测_han2/best","cpu",0,0.5,0.5)
    {}

    cv::Rect set_roi_detect(const cv::Mat &img)
    {
        // 调用worker函数
        cv::Mat image = img.clone();
        std::vector<Ten::Detection> results = detector.worker(image);

        if(results.size() == 0)
        {
            return cv::Rect();
        }

        std::sort(results.begin(), results.end(),
                    [](const Ten::Detection &det1, const Ten::Detection &det2) -> bool
                    {
                        double s1 = det1.w_ * det1.h_;
                        double s2 = det2.w_ * det2.h_;
                        return s1 > s2;
                    });
        Ten::Detection best = results[0];

        // 归一化
        float x1 = best.cx_ - best.w_ / 2;
        float x2 = best.cx_ + best.w_ / 2;
        float y1 = best.cy_ - best.h_ / 2;
        float y2 = best.cy_ + best.h_ / 2;

        return cv::Rect(cv::Point2i(x1, y1), cv::Point2i(x2, y2));
    }

    /**
     * @brief 根据深度图像， 内参， 直接转成pcl点云
     * @param depth_frame       原生深度帧
     * @param color_intr        彩色相机内参
     * @param pcl_cloud         输出的pcl_cloud点云
    */
    void set_Pcl_Cloud(
        const cv::Mat& depth_frame,
        const rs2_intrinsics& color_intr,
        const cv::Rect& roi,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& pcl_cloud
    )
    {
        pcl_cloud->clear();
        int w = depth_frame.cols;
        int h = depth_frame.rows;
        
        for (int v = roi.y; v < roi.y + roi.height; v++) {
        for (int u = roi.x; u < roi.x + roi.width; u++) {
            // 原生精准深度（0.1mm精度，斜视角无误差）
            float z = depth_frame.ptr<uint16_t>(v)[u] * 0.001f;
            int z_mm = int(z * 1000);

            // 过滤无效深度
            if (z <= 0 || z_mm < CloudDepth_min || z_mm > CloudDepth_max)
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
    Ten::Ten_yolo detector;

};

}       // namespace Plane_FitLocator
}       // namespace Ten
#endif 