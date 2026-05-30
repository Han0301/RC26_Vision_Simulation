#ifndef __SET_PCL_H_
#define __SET_PCL_H_
#include <ros/ros.h>
#include <iostream>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
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

    // 设置yolo 目标检测的矩形框
    cv::Rect set_roi_detect(const cv::Mat &image)
    {
        // 1 调用worker函数
        std::vector<Ten::Detection> results = detector.worker(const_cast<cv::Mat&>(image));
        if(results.empty()) return cv::Rect();

        // 2 取最优结果
        std::sort(results.begin(), results.end(),
                    [](const Ten::Detection &det1, const Ten::Detection &det2) -> bool
                    {
                        double s1 = det1.w_ * det1.h_;
                        double s2 = det2.w_ * det2.h_;
                        return s1 > s2;
                    });
        Ten::Detection best = results[0];       // 检测框中心点坐标和宽高

        // 3 计算框边界坐标
        float x1 = best.cx_ - best.w_ / 2;
        float x2 = best.cx_ + best.w_ / 2;
        float y1 = best.cy_ - best.h_ / 2;
        float y2 = best.cy_ + best.h_ / 2;
        
        return cv::Rect(cv::Point2i(cvRound(x1), cvRound(y1)),      // 四舍五入给结果
                        cv::Point2i(cvRound(x2), cvRound(y2)));
    }

    /**
     * @brief 根据深度图像， 内参， 直接转成pcl点云
     * @param depth_frame       原生深度帧
     * @param color_intr        彩色相机内参
     * @param roi               yolo检测处的rect框
     * @param pcl_cloud         输出的pcl_cloud点云
    */
    bool set_Pcl_Cloud(
        const cv::Mat& depth_frame,
        const rs2_intrinsics& color_intr,
        const cv::Rect& roi,
        pcl::PointCloud<pcl::PointXYZ>::Ptr& pcl_cloud
    )
    {
        // 安全性校验和输出重置
        if (roi.empty()) return false;
        if (roi.x < 0 || roi.y < 0 || roi.x + roi.width > depth_frame.cols || roi.y + roi.height > depth_frame.rows) return false;
        pcl_cloud->clear();

        int y_end = roi.y + roi.height;
        int x_end = roi.x + roi.width;
        for (int v = roi.y; v < y_end; v++) 
        {
            for (int u = roi.x; u < x_end; u++) 
            {
                uint16_t z_mm = depth_frame.ptr<uint16_t>(v)[u];

                if (z_mm < CloudDepth_min || z_mm > CloudDepth_max) continue;
                float z = z_mm * 0.001f;

                // 反投影
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
        return true;
    }


private:
    Ten::Ten_yolo detector;

};      // class Ten_set_pcl
}       // namespace Plane_FitLocator
}       // namespace Ten
#endif 