#ifndef __DEBUG_HSV_H_
#define __DEBUG_HSV_H_
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
#include <array>
#include <numeric>
#include <unordered_set>
#include <mutex>
#include <cmath>
#include <cfloat>
#include <climits>
#include <filesystem>
#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <cfloat>
#include "method_math.h"
#include "zbuffer_simplify.h"

namespace Ten{        
struct HistCompareResult {
    int img1;          // 图像编号（1-12）
    int img2;          // 图像编号（1-12）
    double h_sim;      // H通道相似性
    double s_sim;      // S通道相似性
    double v_sim;      // V通道相似性
    double total_sim;  // 加权总相似性（巴氏距离）
};

class Ten_debug_hsv{
public:

/**
 * @brief 可视化并保存单张图片的HSV直方图（输入vector长度3：H/S/V三通道各1个直方图）
 * @param hsv_hist 输入直方图容器：
 *        - 索引0: H通道直方图 
 *        - 索引1: S通道直方图 
 *        - 索引2: V通道直方图
 * @param save_path 保存路径（如："single_img_hsv_hist.jpg"）
 * @param h_bin_num H通道分箱数（默认18，适配0-179范围）
 * @param s_bin_num S通道分箱数（默认32，适配0-255范围）
 * @param v_bin_num V通道分箱数（默认32，适配0-255范围）
 * @param canvas_height 画布高度（默认300像素，影响直方图显示高度）
 */
void save_hsv_hist_visualization(const std::vector<cv::Mat>& hsv_hist, 
                                 const std::string& save_path,
                                 int h_bin_num = 1,  // H通道独立bin数
                                 int s_bin_num = 1,  // S通道独立bin数
                                 int v_bin_num = 1,  // V通道独立bin数
                                 int canvas_height = 640);

// 功能函数 set_hsv_hist: 输入roi_image，生成hsv_list(一张图片的3个通道直方图)
std::vector<cv::Mat> set_hsv_hist(cv::Mat roi_image);


/**
 * @brief 比较两个HSV直方图的相似性（分通道计算+加权平均）
 * @param hist1 第一个HSV直方图（H/S/V三通道）
 * @param hist2 第二个HSV直方图（H/S/V三通道）
 * @param h_sim 输出的 h 通道相似性结果
 * @param s_sim 输出的 s 通道相似性结果
 * @param v_sim 输出的 v 通道相似性结果
 * @param hsv_sim 输出的 hsv 加权相似性结果
 * @param compare_method 比较方法（默认巴氏距离）
 * @param h_weight H通道权重（默认0.3）
 * @param s_weight S通道权重（默认0.4）
 * @param v_weight V通道权重（默认0.3）
 * @return 相似性分数（不同方法取值含义不同，见上表）
 */
void compare_hsv_hist(
    const std::vector<cv::Mat>& hist1,
    const std::vector<cv::Mat>& hist2,
    double &h_sim_out,
    double &s_sim_out,
    double &v_sim_out,
    double &hsv_sim_out,
    int compare_method = cv::HISTCMP_BHATTACHARYYA,
    double h_weight = 0.3,
    double s_weight = 0.4,
    double v_weight = 0.3
);


/**
 * @brief 比较两个HSV直方图的相似性（分通道计算+加权平均）
 * @param hist1 第一个HSV直方图（H/S/V三通道）
 * @param hist2 第二个HSV直方图（H/S/V三通道）
 * @param compare_method 比较方法（默认巴氏距离）
 * @param h_weight H通道权重（默认0.3）
 * @param s_weight S通道权重（默认0.4）
 * @param v_weight V通道权重（默认0.3）
 * @return 相似性分数（不同方法取值含义不同，见上表）
 */
double compare_hsv_hist(
    const std::vector<cv::Mat>& hist1,
    const std::vector<cv::Mat>& hist2,
    int compare_method = cv::HISTCMP_BHATTACHARYYA,
    double h_weight = 0.3,
    double s_weight = 0.4,
    double v_weight = 0.3
);
/**
 * @brief 批量对比12张图像的HSV直方图相似性，并将结果写入TXT
 * @param txt_save_path TXT文件保存路径（如"/home/h/hist_compare_result.txt"）
 * @param hist_36 长度为36的直方图向量，格式：[img1_h, img1_s, img1_v, img2_h, img2_s, img2_v,...img12_h, img12_s, img12_v]
 * @param compare_method 直方图对比方法（默认巴氏距离HISTCMP_BHATTACHARYYA）
 * @param h_weight H通道权重（默认0.3）
 * @param s_weight S通道权重（默认0.4）
 * @param v_weight V通道权重（默认0.3）
 * @return bool 成功返回true，失败返回false
 */
bool batch_compare_hist_and_save(
    const std::string& txt_save_path,
    const std::vector<cv::Mat>& hist_36,
    int compare_method = cv::HISTCMP_BHATTACHARYYA,
    double h_weight = 0.3,
    double s_weight = 0.4,
    double v_weight = 0.3
);

void read_jpgs_by_idx_order(const std::string& img_dir,std::vector<Ten::box>& box_lists);
void cluster_box_images(const std::vector<HistCompareResult>& all_compare_results);
std::tuple<int,int,int> get_hist_mode(std::vector<cv::Mat>& hist);
std::vector<int> bgr_color_analysis(const cv::Mat& img_in);
};

extern Ten::Ten_debug_hsv _DEBUG_HSV_;
}
#endif 