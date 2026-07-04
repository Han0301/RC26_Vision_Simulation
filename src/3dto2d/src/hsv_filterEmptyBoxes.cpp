#ifndef __HSV_FILTEREMPTYBOXES_H_
#define __HSV_FILTEREMPTYBOXES_H_
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

namespace Ten
{
#define H_BINS_ 180
#define S_BINS_ 256
#define V_BINS_ 256

class hsv_filterEmptyBoxes
{
public:

/**
 * @brief 提取方块HSV并过滤空方块：统计有效像素的HSV众数，计算得分并返回高分局索引
 * @param zbuffer 深度缓冲矩阵
 * @param image 原始图像
 * @param idx_to_depth_ 下标到深度列表的映射
 */
void update_idx_hsv_map_(const cv::Mat& zbuffer, 
    const cv::Mat& image,
    std::unordered_map<int, std::vector<float>> idx_to_depth_
    );

/**
 * @brief 根据方块HSV众数来确定hsv标准值
 * @brief 在启动区初始化时调用一次即可
 * @param idx_hsv_map 当前帧的各个下标和对应的hsv众数值
 */
void update_standard_hsv_(std::unordered_map<int, std::tuple<int, int, int>>idx_hsv_map);

/**
 * @brief 计算单一通道的hsv标准值
 * @param single_channel_result 单一通道的众数，其下标按顺序来
 * @param min_total_single 最小总差值，初始为max int
 * @return 标准的单通道值
 */
int cal_single_standard_hsv(std::vector<int>single_channel_result, int min_total_single = INT_MAX);

/**
 * @brief 筛空,更新 有方块的位置下标-1
 * @param idx_hsv_map 当前帧的各个下标和对应的hsv众数值
 */
void update_exist_boxes(std::unordered_map<int, std::tuple<int, int, int>>idx_hsv_map);

// 辅助函数
// 取到 idx_hsv_map_
std::unordered_map<int, std::tuple<int, int, int>>get_idx_hsv_map_(){
    return idx_hsv_map_;
}

std::vector<int>get_standard_hsv_(){
    return standard_hsv_;
}

std::vector<bool> get_exist_boxes(){
    return exist_boxes;
}

std::vector<bool> get_managed_boxes(){
    return managed_boxes;
}

bool get_is_over(){
    return is_over;
}

private:
// extract_hsv 需要的成员变量
std::unordered_map<int, std::tuple<int, int, int>> idx_hsv_map_;     // 保存每个有效idx的HSV众数
std::vector<int> standard_hsv_;                                      // 标准hsv值
std::vector<bool> exist_boxes = {false,false,false,false,false,false,false,false,false,false,false,false};      // 存在方块的下标会被置为True
std::vector<bool> managed_boxes = {false,false,false,false,false,false,false,false,false,false,false,false};        // 处理过的下标会被置为True
bool is_over = false;       // 标志位，表示可以结束整套筛空功能， 在 update_exist_boxes 函数中更新
};

void hsv_filterEmptyBoxes::update_idx_hsv_map_(
    const cv::Mat& zbuffer, 
    const cv::Mat& image,
    std::unordered_map<int, std::vector<float>> idx_to_depth_
    ){

    // 1. 转换图像到HSV空间
    cv::Mat hsv_image;
    cv::cvtColor(image, hsv_image, cv::COLOR_RGB2HSV);

    // 2. 遍历每个idx的深度数据, 用来填充 idx_hsv_map
    for (const auto& idx_depth_pair : idx_to_depth_) {
        int idx = idx_depth_pair.first;
        const std::vector<float>& depths = idx_depth_pair.second;

        // 2.1 跳过空深度
        if (depths.empty()){
            ROS_INFO("Idx %d's depths not found", idx);
            continue;}

        // 2.2 收集该idx对应的所有像素坐标    
        std::vector<cv::Point> idx_pixels;
        const float depth_tolerance = 0.001f;
        for (float target_depth : depths) {
            // ------------------------------------------------------------------------------------------------------------------------------------------
            for (int y = 0; y < zbuffer.rows; ++y) {
                for (int x = 0; x < zbuffer.cols; ++x) {
                    float current_depth = zbuffer.at<float>(y, x);
                    if (current_depth != FLT_MAX && std::abs(current_depth - target_depth) < depth_tolerance) {
                        idx_pixels.emplace_back(x, y);
                    }
                }
            }
        }

        // 2.3 跳过像素数≤1000的idx
        if (idx_pixels.size() <= 1000) {
            ROS_INFO("Idx %d pixels=%zu(<=1000), skip", idx, idx_pixels.size());

            continue;
        }
        // 2.4 填充 managed_boxes
        if (!(managed_boxes[idx - 1])) {managed_boxes[idx - 1] = true;};


        // 2.5 填充HSV直方图
        cv::Mat h_hist = cv::Mat::zeros(1, H_BINS_, CV_32S);     // HSV直方图
        cv::Mat s_hist = cv::Mat::zeros(1, S_BINS_, CV_32S);
        cv::Mat v_hist = cv::Mat::zeros(1, V_BINS_, CV_32S);
        for (const auto& pt : idx_pixels) {
            cv::Vec3b hsv = hsv_image.at<cv::Vec3b>(pt.y, pt.x);
            h_hist.at<int>(0, hsv[0])++;
            s_hist.at<int>(0, hsv[1])++;
            v_hist.at<int>(0, hsv[2])++;
        }
        // 2.6 计算众数
        cv::Point h_max_loc, s_max_loc, v_max_loc;
        cv::minMaxLoc(h_hist, nullptr, nullptr, nullptr, &h_max_loc);
        cv::minMaxLoc(s_hist, nullptr, nullptr, nullptr, &s_max_loc);
        cv::minMaxLoc(v_hist, nullptr, nullptr, nullptr, &v_max_loc);
        // 2.7 填充idx_hsv_map，构建下标和对应的hsv众数
        int h_mode = h_max_loc.x;
        int s_mode = s_max_loc.x;
        int v_mode = v_max_loc.x;
        idx_hsv_map_[idx] = {h_mode, s_mode, v_mode};
    }
}

int hsv_filterEmptyBoxes::cal_single_standard_hsv(std::vector<int>single_channel_result, int min_total_single){
    int result_channel = -1;
    for(int i = 0;i < single_channel_result.size(); ++i){
        int candidate = single_channel_result[i];
        int total_diff = 0;
        // 计算每一个值与其他所有值的总差值
        for(int j = 0;j < single_channel_result.size(); ++j){      
            total_diff += std::abs(candidate - single_channel_result[j]);
        }
        if(total_diff < min_total_single){
            min_total_single = total_diff;
            result_channel = candidate;
        }
    }
    if (result_channel == -1){
        ROS_WARN("cal_single_standard_hsv: result_channel = -1 !");
    }
    return result_channel;
}

void hsv_filterEmptyBoxes::update_standard_hsv_(std::unordered_map<int, std::tuple<int, int, int>>idx_hsv_map){
    // 1. 填入结果向量
    std::vector<int> idx_h_result, idx_s_result, idx_v_result;
    for (const auto& [idx, hsv_tuple] : idx_hsv_map) {
        idx_h_result.push_back(std::get<0>(hsv_tuple));
        idx_s_result.push_back(std::get<1>(hsv_tuple));
        idx_v_result.push_back(std::get<2>(hsv_tuple));
    }
    // 2. 计算标准hsv值（计算方法： 最小总差值）update_standard_hsv_
    standard_hsv_ = {cal_single_standard_hsv(idx_h_result),cal_single_standard_hsv(idx_s_result), cal_single_standard_hsv(idx_v_result)};
    }

void hsv_filterEmptyBoxes::update_exist_boxes(std::unordered_map<int, std::tuple<int, int, int>> idx_hsv_map){
    std::vector<float> score = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}; 

    // 1.计算得分并填充 下标-得分 列表 idx_score_lists
    std::vector<std::pair<int,float>> idx_score_lists;
    for(const auto& [idx, hsv_tuple] : idx_hsv_map) {
        if (idx < 1 || idx > score.size()) continue;
        int h = std::get<0>(hsv_tuple);
        int s = std::get<1>(hsv_tuple);
        int v = std::get<2>(hsv_tuple);
        
        // 得分公式
        score[idx - 1] = std::max(0.0f,60.0f - abs(h - get_standard_hsv_()[0]) )
                       + std::max(0.0f,10.0f - 0.2f * abs(s - get_standard_hsv_()[1]))
                       + std::max(0.0f,30.0f - 0.5f * abs(v - get_standard_hsv_()[2]));

        if( abs(score[idx - 1] - 0.0f) < 1e-4f ){ 
            continue;
        }
        idx_score_lists.emplace_back(idx, score[idx - 1]);
    }
    
    // 2. 按照得分进行排序
    std::sort(idx_score_lists.begin(), idx_score_lists.end(),
        [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
            return a.second > b.second;
        });

    // 3. 填充 exist_boxes
    std::fill(exist_boxes.begin(), exist_boxes.end(), false);
    if (std::count(managed_boxes.begin(),managed_boxes.end(),false) == 0 ){
        ROS_INFO(" filterEmptyBoxes is over!");
        is_over = true;
    }

    for(size_t i = 0;i < 8 - std::count(managed_boxes.begin(),managed_boxes.end(),false); ++i){
        exist_boxes[idx_score_lists[i].first - 1] = true;
    }
}

}//namespace Ten
#endif