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
#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <cfloat>

#include "occlusion_handing.h"
namespace Ten{

// 表示hsv直方图的信息, hsv topn中主要使用的结构体
struct HistValue {
    int position;  // 直方图的位置（对应H/S/V的取值）
    int count;     // 该位置的像素数量
    float degree_of_promacy;        // 首位度
};

// 计算 hsv 筛空的指标
struct score{
    int idx;
    std::vector<HistValue> top_n_h;
    std::vector<HistValue> top_n_s;
    std::vector<HistValue> top_n_v;
    float hsv_score;
    int valid_count = 0;                      // 有效像素数量
    bool is_managed = true;     
};

struct init_hsv
{
    std::vector<int> standard_hsv_;
    std::vector<score> score_lists_;

    init_hsv()
    {
        standard_hsv_ = {0,0,0};
        score_lists_.resize(12);
    }
};

class Ten_hsv_handing
{
public:
    /**
     * @brief 写入 score_lists ，写入其中的 idx, valid_count, top_n_h, top_n_s, top_n_v, is_managed 字段
     * @param box_lists     用来读取其中的 hsv_iamge 来统计 hsv 分布情况
     * @param top_n         由用户自己选择 获取 hsv 各通道数量前几， 若取 top_n = 1， 即统计众数
     * @param min_manage_points     对于 hsv_image 的最小处理像素值， 小于这个值， 不处理对应的 hsv_image, 并置 score_lists中对应位置的 is_managed 为 false
     * @param score_lists  待写入的各位置 hsv topn和score 列表
    */
    void set_hsv_top_n
    (
        const std::vector<box>& box_lists, 
        const int top_n,
        const int min_manage_points,
        std::vector<score>& score_lists
    );

    /**
     * @brief 写入 score_lists ，调用 set_hsv_top_n 写入其中的 idx, valid_count, top_n_h, top_n_s, top_n_v, is_managed 字段, 并写入 hsv_score 字段
     * @param box_lists     用来读取其中的 hsv_iamge 来统计 hsv 分布情况
     * @param top_n         由用户自己选择 获取 hsv 各通道数量前几， 若取 top_n = 1， 即统计众数
     * @param min_manage_points     对于 hsv_image 的最小处理像素值， 小于这个值， 不处理对应的 hsv_image, 并置 score_lists中对应位置的 is_managed 为 false
     * @param standard_hsv 标准的 hsv 值， 用于统计分数
     * @param score_lists   待写入的各位置 hsv topn和score 列表
    */
    void set_hsv_topn_score
    (
        const std::vector<box>& box_lists, 
        const int top_n,
        const int min_manage_points,
        const std::vector<int> standard_hsv,
        std::vector<score>& score_lists
    );

    /**
     * @brief 写入 score_lists ，生成 hsv_standard, 调用 set_hsv_top_n 写入其中的 idx, valid_count, top_n_h, top_n_s, top_n_v, is_managed 字段
     * @param box_lists     用来读取其中的 hsv_iamge 来统计 hsv 分布情况
     * @param top_n         由用户自己选择 获取 hsv 各通道数量前几， 若取 top_n = 1， 即统计众数
     * @param min_manage_points     对于 hsv_image 的最小处理像素值， 小于这个值， 不处理对应的 hsv_image, 并置 score_lists中对应位置的 is_managed 为 false
     * @param standard_hsv 待写入的标准的 hsv 值， 用于统计分数
     * @param score_lists   待写入的各位置 hsv topn和score 列表
    */
    void set_hsv_topn_stand
    (
        const std::vector<box>& box_lists, 
        const int top_n,
        const int min_manage_points,
        std::vector<int>& standard_hsv,
        std::vector<score>& score_lists
    );

    /**
     * @brief 调试打印函数， 显示 score_lists的信息
     * @param score_lists 得分列表
    */
    void print_score_lists
    (
        const std::vector<score>& score_lists
    );
private:

    // 功能函数：提取hsv 直方图中的前topn， 并以 std::vector<HistValue> 返回
    std::vector<HistValue> getTopNValues
    (
        const cv::Mat& hist, 
        const int top_n
    ) {
        std::vector<HistValue> hist_values;
        
        if (hist.empty() || hist.rows != 1 || top_n <= 0) {
            std::cout << "🤡getTopNValues: invalid hist or top_n!" << std::endl;
            return hist_values; // 返回空vector
        }
        // 1. 提取直方图所有非零值（零值无统计意义）
        for (int col = 0; col < hist.cols; col++) {
            int count = hist.at<int>(0, col);
            if (count > 0) { // 只保留有像素的位置
                hist_values.push_back({col, count});
            }
        }
        
        // 2. 按数量降序排序（数量多的在前）
        std::sort(hist_values.begin(), hist_values.end(), 
            [](const HistValue& a, const HistValue& b) {
                return a.count > b.count;
            });
        
        // 3. 截取前N个（如果总数不足N，取全部）
        if (hist_values.size() > top_n) {
            hist_values.resize(top_n);
        }
        
        // 4. 填充 首位度
        float first_count = static_cast<float>(hist_values[0].count);
        for (size_t i = 0; i < hist_values.size(); i++)
        {   
            float now_count = static_cast<float>(hist_values[i].count);
            hist_values[i].degree_of_promacy = now_count / first_count;
        }
        return hist_values;
    }
};
    extern Ten::Ten_hsv_handing _HSV_HANDING_;
    extern Ten::init_hsv _INIT_HSV_;
}