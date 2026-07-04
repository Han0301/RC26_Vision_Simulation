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

namespace Ten{
#define _HSV_THRESHOLD_ 300     // 计算像素的hsv得分的阈值，来判断是否是有效像素

struct roi_result{
    int idx;
    cv::Mat roi_image;
    int yolo_result;
};

class zbuffer_roi_debug{
public:
/**
 * @brief 根据图像和zbuffer矩阵，更新最优的roi裁剪结果
 * @param image 图像
 * @param zbuffer 表示遮挡关系的矩阵
 * @param exist_boxes 布尔型容器，表示存在方块的位置
 * @param idx_to_depth_ 下标到深度的映射
 * @param standard_hsv 标准的hsv值
 */
void update_best_roi_image(
    const cv::Mat& image,
    const cv::Mat& zbuffer,
    std::vector<bool> exist_boxes,
    std::unordered_map<int, std::vector<float>> idx_to_depth_,
    const std::vector<int> standard_hsv_
);

/**
 * @brief 输入原图像，在原图像中绘制方块的轮廓
 * @param image 图像
 * @param zbuffer 表示遮挡关系的矩阵
 * @param exist_boxes 布尔型容器，表示存在方块的位置
 * @param idx_to_depth_ 下标到深度的映射
 */
void update_debug_image(
    cv::Mat& image, 
    const cv::Mat& zbuffer, 
    const std::vector<bool> exist_boxes,
    std::unordered_map<int, std::vector<float>> idx_to_depth_
);

/**
 * @brief 为 best_roi_image 拼接调试图像
 * @param best_roi_image 最优的 序号-方块roi画面  
 * @param yolo_result yolo 的检测结果
 */
void update_debug_best_roi_image(
    const std::vector<roi_result>& best_roi_image,
    const std::vector<int> yolo_result
);

// 辅助函数1： 取到 best_roi_image 
std::vector<roi_result> get_best_roi_image(){
    return best_roi_image;
};

// 辅助函数2： 取到 debug_image
cv::Mat get_debug_image(){
    if(!(debug_image.empty())){
        return debug_image;
    }else{return cv::Mat();}
}

// 辅助函数3： 取到 debug_best_roi_image
cv::Mat get_debug_best_roi_image(){
    if(!(debug_best_roi_image.empty())){
        return debug_best_roi_image;
    }else{return cv::Mat();}
}

private:
std::vector<roi_result> best_roi_image;     // 最优的 序号-方块roi画面  
std::unordered_map<int, int> best_roi_count;        // 存储每个idx的最大点集数量

cv::Mat debug_image;        // 在原图像中绘制方块的轮廓的调试图像
cv::Mat debug_best_roi_image;       //由 best_roi_image 拼接的调试图像

// 工具函数，在vector中查找指定idx的roi_result元素，返回迭代器
inline std::vector<roi_result>::iterator find_roi_by_idx(int target_idx)
{
    for(auto it = best_roi_image.begin(); it != best_roi_image.end(); ++it)
    {
        if(it->idx == target_idx)
        {
            return it;
        }
    }
    return best_roi_image.end();
}

};

void zbuffer_roi_debug::update_best_roi_image(
    const cv::Mat& image,
    const cv::Mat& zbuffer,
    std::vector<bool> exist_boxes,
    std::unordered_map<int, std::vector<float>> idx_to_depth_,
    const std::vector<int> standard_hsv_
){
    // 1. 输入检查
    // 1.1 检查 image 和 zbuffer 是否为空
    if (image.empty() || zbuffer.empty()) {
        ROS_WARN("Image or zbuffer is empty, skip ROI process");
        return;
    }
    // 1.2 检查 image 和 zbuffer 是否为指定格式
    if (image.type() != CV_8UC3 || zbuffer.type() != CV_32F) {
        ROS_WARN("Image type or zbuffer type error! Expected CV_8UC3 and cv::32F, got %d, %d", image.type(),zbuffer.type());
        return;
    }
    // 1.3 检查 iamge 和 zbuffer 的尺寸是否一致
    if (image.size() != zbuffer.size()) {
        ROS_WARN("Image size(%dx%d) != Zbuffer size(%dx%d)", 
                  image.cols, image.rows, zbuffer.cols, zbuffer.rows);
        return;
    }

    // 2. 查找深度区域, 填充 depth_regions 
    std::unordered_map<float, std::vector<cv::Point>> depth_regions;        // depth_regions 表示 深度-对应深度的点集
    for (int y = 0; y < zbuffer.rows; ++y) {
        for (int x = 0; x < zbuffer.cols; ++x) {
            float d = zbuffer.at<float>(y, x);
            if (d == FLT_MAX) continue;
            depth_regions[d].emplace_back(x, y);
        }
    }

    // 3. 收集有效深度， 更新 idx_to_depth_, valid_depths, depth_to_idx
    std::unordered_set<float> valid_depths;         // 存储所有方块的深度值
    std::unordered_map<float,int> depth_to_idx;     // 深度到下标的映射表
    std::vector<int> to_erase_idxs;                 // 要删除的idx

    // 3.1 收集有效深度和待删除的idx
    for (const auto& [idx, depths] : idx_to_depth_) {
        if (exist_boxes[idx - 1]) {
            for (float d : depths) {
                valid_depths.insert(d);
                depth_to_idx[d] = idx; 
            }
        } else {
            to_erase_idxs.push_back(idx);  // 收集待删除的idx
        }
    }

    // 3.2 批量删除无效idx
    for (int idx : to_erase_idxs) {
        idx_to_depth_.erase(idx);
    }

    // 4 转hsv, 用于更新 最优
    cv::Mat hsv_image;
    cv::cvtColor(image, hsv_image, cv::COLOR_RGB2HSV);

    // 5. 遍历深度区域, 更新最优roi结果
    for (const auto& [depth, points] : depth_regions) {
        // 5.1 检查当前 depth 和 points 是否有效
        bool is_valid = false;
        float matched_depth = -1.0f;
        for (float valid_d : valid_depths) {
            if (std::abs(depth - valid_d) < 1e-3) {
                is_valid = true;
                matched_depth = valid_d;
                break;
            }
        }
        if (!is_valid) continue;
        if (points.empty()) continue;

        // 5.2 由 depth_to_idx 找到当前正在处理的 深度-区域 对应的下标 now_manage_idx 
        if (depth_to_idx.find(matched_depth) == depth_to_idx.end()) {
            ROS_WARN("Depth %.4f not found in depth_to_idx, skip", matched_depth);
            continue;
        }
        int now_manage_idx = depth_to_idx[matched_depth];

        // 5.3 生成cal_points， valid_points，确定roi的边界，用于后续比较和输出结果
        std::vector<cv::Point> cal_points;      // 用于 计算 最优的点集
        std::vector<cv::Point> valid_points;    // 实际用于创建掩码，生成最终结果的点集
        int x_min = INT_MAX, x_max = INT_MIN;   
        int y_min = INT_MAX, y_max = INT_MIN;

        for (const auto& p : points) {
            // 5.3.1 生成cal_points， valid_points
            int x = p.x < 0 ? 0 : (p.x >= image.cols ? image.cols - 1 : p.x);
            int y = p.y < 0 ? 0 : (p.y >= image.rows ? image.rows - 1 : p.y);
            valid_points.emplace_back(x,y);
            cv::Vec3b hsv_value = hsv_image.at<cv::Vec3b>(y, x);
            if (7 * abs(hsv_value[0] - standard_hsv_[0]) + abs(hsv_value[1] - standard_hsv_[1]) + 2 * abs(hsv_value[2] - standard_hsv_[2]) <= _HSV_THRESHOLD_){
                cal_points.emplace_back(x, y);
            }

            // 5.3.2 更新边界
            x_min = std::min(x_min, x);
            x_max = std::max(x_max, x);
            y_min = std::min(y_min, y);
            y_max = std::max(y_max, y);
        }

        // 5.3.3 边界有效性校验
        if (x_min > x_max || y_min > y_max || x_min < 0 || y_min < 0 || x_max >= image.cols || y_max >= image.rows) {
            ROS_WARN("Invalid ROI boundary: x[%d,%d], y[%d,%d], skip", x_min, x_max, y_min, y_max);
            continue;
        }

        // 5.4 比较点集大小
        int current_count = static_cast<int>(cal_points.size());
        auto count_it = best_roi_count.find(now_manage_idx);
        if (count_it != best_roi_count.end()) {
            if (current_count <= count_it->second) continue;
        }

        // 5.5 创建掩码
        cv::Mat roi_mask = cv::Mat::zeros(image.size(), CV_8UC1);
        for (const auto& p : valid_points) {
            if (p.y >= 0 && p.y < roi_mask.rows && p.x >= 0 && p.x < roi_mask.cols) {
                roi_mask.at<uchar>(p.y, p.x) = 255;
            }
        }

        // 5.6 裁剪ROI
        cv::Rect roi_rect(x_min, y_min, x_max - x_min + 1, y_max - y_min + 1);
        roi_rect.x = std::max(0, roi_rect.x);
        roi_rect.y = std::max(0, roi_rect.y);
        roi_rect.width = std::min(roi_rect.width, image.cols - roi_rect.x);
        roi_rect.height = std::min(roi_rect.height, image.rows - roi_rect.y);
        // 5.6.1 校验roi_rect，避免宽高为负
        if (roi_rect.width <= 0 || roi_rect.height <= 0 || 
            roi_rect.x + roi_rect.width > image.cols || 
            roi_rect.y + roi_rect.height > image.rows) {
            ROS_WARN("Invalid ROI rect: x=%d, y=%d, w=%d, h=%d, skip", 
                      roi_rect.x, roi_rect.y, roi_rect.width, roi_rect.height);
            continue;
        }
        // 5.6.2 生成 image_roi,mask_roi
        cv::Mat image_roi = image(roi_rect).clone();
        cv::Mat mask_roi = roi_mask(roi_rect).clone();

        // 5.7 在 image_roi 中 生成有效区域 mask_roi
        cv::Mat crop_roi = cv::Mat::zeros(image_roi.size(), image_roi.type());
        image_roi.copyTo(crop_roi, mask_roi);

        // 5.8 转为正方形
        int roi_width = crop_roi.cols;
        int roi_height = crop_roi.rows;
        int max_side = std::max(roi_width, roi_height);
        cv::Mat square_roi = cv::Mat::zeros(max_side, max_side, crop_roi.type());
        int x_offset = (max_side - roi_width) / 2;
        int y_offset = (max_side - roi_height) / 2;
        cv::Rect paste_rect(x_offset, y_offset, roi_width, roi_height);

        // 最后一次校验paste_rect
        if (paste_rect.x >= 0 && paste_rect.y >= 0 && 
            paste_rect.x + paste_rect.width <= square_roi.cols && 
            paste_rect.y + paste_rect.height <= square_roi.rows) {
            crop_roi.copyTo(square_roi(paste_rect));
        } else {
            ROS_WARN("Invalid paste rect for square ROI, skip");
            continue;
        }

        // 5.9 更新最优ROI
        best_roi_count[now_manage_idx] = current_count;
        auto roi_iter = find_roi_by_idx(now_manage_idx);
        roi_result roi_data;
        roi_data.idx = now_manage_idx;
        roi_data.roi_image = square_roi.clone();
        roi_data.yolo_result = 0;
        if(roi_iter != best_roi_image.end())
        {
            *roi_iter = roi_data;
        }
        else
        {
            best_roi_image.emplace_back(roi_data);
        }
    }

    // 6 检查 best_roi_image 中是否有错误数据
    std::vector<int> del_indexs;
    for(int i=0; i<best_roi_image.size(); ++i)
    {
        int idx = best_roi_image[i].idx;
        if(idx < 1 || idx > exist_boxes.size() || !exist_boxes[idx - 1])
        {
            del_indexs.push_back(idx);
            best_roi_count.erase(idx);
        }
    }
    std::sort(del_indexs.rbegin(), del_indexs.rend());
    for(int i = best_roi_image.size() - 1; i >= 0; --i)
    {
        int idx = best_roi_image[i].idx;
        if(std::find(del_indexs.begin(), del_indexs.end(), idx) != del_indexs.end())
        {
            best_roi_image.erase(best_roi_image.begin() + i);
        }
    }

}

void zbuffer_roi_debug::update_debug_image(
    cv::Mat& image, 
    const cv::Mat& zbuffer, 
    const std::vector<bool> exist_boxes,
    std::unordered_map<int, std::vector<float>> idx_to_depth_
){
    // 1. 检查输入有效性
    if (image.empty() || zbuffer.empty()) {
        ROS_WARN("Image or zbuffer is empty, skip draw");
        return;
    }

    // 2. 绘制深度区域（非idx_to_depth_中记录的深度）
    std::unordered_map<float, std::vector<cv::Point>> depth_regions;
    for (int y = 0; y < zbuffer.rows; ++y) {
        for (int x = 0; x < zbuffer.cols; ++x) {
            float d = zbuffer.at<float>(y, x);
            if (d == FLT_MAX) continue;
            depth_regions[d].emplace_back(x, y);
        }
    }

    // 2.1 从idx_to_depth_收集所有有效的深度值
    std::unordered_set<float> valid_depths;
    for (const auto& [idx, depths] : idx_to_depth_) {
        if (exist_boxes[idx - 1]){
        for (float d : depths) {
            valid_depths.insert(d);
        }
        }
    }

    // 2.2 遍历深度值的区域，判断是否为有效深度
    for (const auto& [d, points] : depth_regions) {
        // 2.2.1 只处理 idx_to_depth_ 中记录的有效深度
        bool is_valid = false;
        for (float valid_d : valid_depths) {
            if (std::abs(d - valid_d) < 1e-3) {
                is_valid = true;
                break;
            }
        }
        if (!is_valid) continue;

        // 2.2.2 绘制区域轮廓
        cv::Mat region_mask = cv::Mat::zeros(zbuffer.size(), CV_8U);
        for (const auto& p : points) region_mask.at<uchar>(p) = 255;
        
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(region_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (const auto& cnt : contours) {
            if (cnt.size() < 4) continue;
            cv::polylines(image, cnt, true, cv::Scalar(0, 0, 255), 2); // 红色轮廓标记异常区域
        }
    }
    image.copyTo(debug_image);
}

void zbuffer_roi_debug::update_debug_best_roi_image(
    const std::vector<roi_result>& best_roi_image,
    const std::vector<int> yolo_result
){
    // 1. 配置固定参数
    const int SINGLE_SIZE = 160;    // 单个图的目标尺寸（160×160）
    const int STITCH_WIDTH = 640;   // 拼接后总宽度（160×4）
    const int STITCH_HEIGHT = 480;  // 拼接后总高度（160×3）
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
        ROS_INFO("best_roi_image.size(): %lu", best_roi_image.size());
        for(const auto& roi_data : best_roi_image)
        {
            if(roi_data.idx == idx && !roi_data.roi_image.empty())
            {
                const cv::Mat& src_img = roi_data.roi_image;
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
                cv::putText(roi_images[vec_idx], std::to_string(yolo_result[vec_idx]), cv::Point(roi_images[vec_idx].cols - 10 , roi_images[vec_idx].rows - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
                break;
            }
        }
    }

    // 4. 拼接成640×480的大图
    debug_best_roi_image = cv::Mat::zeros(STITCH_HEIGHT, STITCH_WIDTH, CV_8UC3);
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