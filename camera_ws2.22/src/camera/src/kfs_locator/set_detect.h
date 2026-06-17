#ifndef __SET_DETECT_H_
#define __SET_DETECT_H_
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include "../yolo/yolo_v5.h"

namespace Ten
{
namespace kfs_locator
{
const int CloudDepth_min = 200;
const int CloudDepth_max = 2000;


class Ten_set_detect
{
public:

    Ten_set_detect()
        :detector("/home/h/下载/卷轴检测red/best","cpu",0.75,0.75,0.75)
    {}

    // 设置yolo 目标检测的矩形框
    std::vector<cv::Rect> set_roi_detect(cv::Mat &image)
    {
        // 1 调用worker函数
        std::vector<Ten::yolo::Detection> results = detector.worker(image);
        if(results.empty()) return {};

        // 2 取最优结果
        std::sort(results.begin(), results.end(),
                    [](const Ten::yolo::Detection &det1, const Ten::yolo::Detection &det2) -> bool
                    {
                        double s1 = det1.w_ * det1.h_;
                        double s2 = det2.w_ * det2.h_;
                        return s1 > s2;
                    });

        // 3 遍历所有检测结果，统一转为 cv::Rect 存入容器
        std::vector<cv::Rect> rect_list;
        for (const auto& det : results)
        {
            float x1 = det.cx_ - det.w_ / 2;
            float x2 = det.cx_ + det.w_ / 2;
            float y1 = det.cy_ - det.h_ / 2;
            float y2 = det.cy_ + det.h_ / 2;

            cv::Rect roi(
                cvRound(x1), cvRound(y1),
                cvRound(x2) - cvRound(x1),   // Rect构造：x,y,width,height
                cvRound(y2) - cvRound(y1)
            );
            rect_list.push_back(roi);
        }

        // 4 返回结果
        return rect_list;
    }

    /**
     * @brief 根据深度图像， 内参， 直接转成pcl点云
     * @param depth_frame       深度帧
     * @param color_intr        彩色相机内参
     * @param rois               yolo检测处的rect框列表
     * @param pcl_clouds         输出的pcl_cloud点云
    */
    bool set_pcl_clouds(
        const cv::Mat& depth_frame,
        const rs2_intrinsics& color_intr,
        const std::vector<cv::Rect>& rois,
        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& pcl_clouds
    )
    {
        // 安全性校验和输出重置
        if (rois.empty()) return false;
        pcl_clouds.clear();

        for (const cv::Rect& roi : rois)
        {
            // 单个ROI越界校验，非法区域直接跳过
            if (roi.x < 0 || roi.y < 0 || roi.x + roi.width  > depth_frame.cols || roi.y + roi.height > depth_frame.rows) continue;
            pcl::PointCloud<pcl::PointXYZ>::Ptr single_cloud(new pcl::PointCloud<pcl::PointXYZ>);

            // // 遍历当前ROI内所有像素
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
                    single_cloud->push_back(p);
                }
            }
            if (!single_cloud->empty())
            {
                pcl_clouds.push_back(single_cloud);
            }
        }
        return true;
    }

private:
    Ten::yolo::yolo_v5 detector;

};      // class Ten_set_detect
}       // namespace kfs_locator
}       // namespace Ten
#endif 