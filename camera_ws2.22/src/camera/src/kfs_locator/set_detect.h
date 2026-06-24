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
        :detector("/home/h/下载/卷轴检测blue/best","cpu",0.75,0.75,0.75)
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

        // 预提取内参，避免循环内重复访问
        const float fx = color_intr.fx;
        const float fy = color_intr.fy;
        const float ppx = color_intr.ppx;
        const float ppy = color_intr.ppy;
        const float inv_fx = 1.0f / fx;
        const float inv_fy = 1.0f / fy;

        for (const cv::Rect& roi : rois)
        {
            // 单个ROI越界校验，非法区域直接跳过
            if (roi.x < 0 || roi.y < 0 || roi.x + roi.width  > depth_frame.cols || roi.y + roi.height > depth_frame.rows) continue;
            pcl::PointCloud<pcl::PointXYZ>::Ptr single_cloud(new pcl::PointCloud<pcl::PointXYZ>);

            const int y_end = roi.y + roi.height;
            const int x_end = roi.x + roi.width;
            // 预分配容量，避免多次 realloc
            single_cloud->reserve(static_cast<size_t>(roi.width) * roi.height);

            for (int v = roi.y; v < y_end; v++)
            {
                const uint16_t* row_ptr = depth_frame.ptr<uint16_t>(v);
                for (int u = roi.x; u < x_end; u++)
                {
                    uint16_t z_mm = row_ptr[u];
                    if (z_mm < CloudDepth_min || z_mm > CloudDepth_max) continue;
                    float z = z_mm * 0.001f;

                    // 手动反投影：比 rs2_deproject_pixel_to_point 快数倍
                    pcl::PointXYZ p;
                    p.x = (u - ppx) * inv_fx * z;
                    p.y = (v - ppy) * inv_fy * z;
                    p.z = z;
                    single_cloud->push_back(p);
                }
            }
            if (!single_cloud->empty())
            {
                pcl_clouds.push_back(single_cloud);
            }
        }
        return !pcl_clouds.empty();
    }

private:
    Ten::yolo::yolo_v5 detector;

};      // class Ten_set_detect
}       // namespace kfs_locator
}       // namespace Ten
#endif 