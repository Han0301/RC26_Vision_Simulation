#ifndef _ZBUFFER_ROI_DEBUG_H_
#define _ZBUFFER_ROI_DEBUG_H_
#include <ros/ros.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>
#include <nav_msgs/Odometry.h>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <numeric>
#include <unordered_set>
#include <mutex>
#include <cmath>
#include <cfloat>
#include <climits>
#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include "methon_math.cpp"
#include "zbuffer_simplify.cpp"
namespace Ten{
class zbuffer_roi_debug{
public:
/**
 * @brief 拼接调试图像
 * @param box_lists 
 * @param debug_best_roi_image 必须为 cv::Mat::zeros(480, 640, CV_8UC3)
 */
void set_debug_roi_image(
    std::vector<Ten::box>box_lists,
    cv::Mat debug_best_roi_image);

// 辅助函数： 取到 debug_best_roi_image
cv::Mat get_debug_best_roi_image(){
    if(!(debug_best_roi_image.empty())){
        return debug_best_roi_image;
    }else{return cv::Mat();}
}

private:
std::unordered_map<int, int> best_roi_count;        // 存储每个idx的最大点集数量

cv::Mat debug_best_roi_image;       //由 best_roi_image 拼接的调试图像

};

void zbuffer_roi_debug::set_debug_roi_image(
    std::vector<Ten::box>box_lists,
    cv::Mat debug_best_roi_image = cv::Mat::zeros(480, 640, CV_8UC3)
){
    // 1. 配置固定参数
    const int SINGLE_SIZE = 160;    // 单个图的目标尺寸（160×160）
    const int COL_NUM = 4;          // 每行列数
    const int ROW_NUM = 3;          // 总行数
    const int TOTAL_IMGS = 12;      // 总图片数（1-12）

    // 2. 初始化12个160×160的全黑图
    std::vector<cv::Mat> roi_images(TOTAL_IMGS, cv::Mat::zeros(SINGLE_SIZE, SINGLE_SIZE, CV_8UC3));

    // 3. 填充有效ROI图（idx1-12）
    for (int idx = 1; idx <= TOTAL_IMGS; ++idx) {
        // 3.1 计算当前idx在vector中的索引（idx1→0，idx12→11）
        int vec_idx = idx - 1;

        // 3.2 检查best_roi_image中是否有该idx的图
        for(const auto& box : box_lists)
        {
            if(box.idx == idx && !box.roi_image.empty())
            {
                const cv::Mat& src_img = box.roi_image;
                // 3.3 校验源图类型
                if (src_img.type() != CV_8UC3) {
                    ROS_WARN("Idx %d image type error (not CV_8UC3), use black image", idx);
                    continue;
                }
                // 3.4 resize为160×160（原正方形，无畸变）
                cv::Mat resized_img;
                cv::resize(src_img, resized_img, cv::Size(SINGLE_SIZE, SINGLE_SIZE), 0, 0, cv::INTER_LINEAR);
                // 3.5 替换初始化的黑图
                roi_images[vec_idx] = resized_img.clone();
                cv::putText(roi_images[vec_idx], std::to_string(box_lists[vec_idx].cls), cv::Point(roi_images[vec_idx].cols - 10 , roi_images[vec_idx].rows - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
                break;
            }
        }
    }

    // 4. 拼接成640×480的大图
    for (int row = 0; row < ROW_NUM; ++row) {
        for (int col = 0; col < COL_NUM; ++col) {
            // 4.1 计算当前小图在vector中的索引
            int vec_idx = row * COL_NUM + col;
            if (vec_idx >= TOTAL_IMGS) break; // 防止越界（理论上不会触发）

            // 4.2 计算当前小图在拼接图中的位置
            int x = col * SINGLE_SIZE;
            int y = row * SINGLE_SIZE;
            cv::Rect roi_rect(x, y, SINGLE_SIZE, SINGLE_SIZE);

            // 4.3 将小图复制到拼接图对应位置
            roi_images[vec_idx].copyTo(debug_best_roi_image(roi_rect));
        }
    }

}

}       // namespace Ten
#endif