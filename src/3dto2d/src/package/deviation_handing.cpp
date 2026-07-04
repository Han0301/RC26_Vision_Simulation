#ifndef _Ten_deviation_handing_CPP_
#define _Ten_deviation_handing_CPP_
#include "deviation_handing.h"

namespace Ten{
void Ten_deviation_handing::registration
(
    std::vector<std::vector<cv::Point2f>> pending_points,
    std::vector<std::vector<cv::Point2f>> mask_corner_points,
    registration_results& final_results
)
{
    // std::cout << "---------------" << std::endl;
    for(int i = 0; i < mask_corner_points.size(); i++)
    {
        int target_idx = 0;
        Ten::registration_results target_results;
        target_results.offset_x = FLT_MAX;
        target_results.offset_y = FLT_MAX;

        for(int j = 0; j < pending_points.size(); j++)
        {
            registration_results results;
            cal_offsets(pending_points[j], mask_corner_points[i], results);
            if (abs(results.offset_x) < abs(target_results.offset_x) && abs(results.offset_y) < abs(target_results.offset_y))
            {
                target_idx = j + 1;
                target_results.offset_x = results.offset_x;
                target_results.offset_y = results.offset_y;
            }
            // std::cout << "idx: " << j + 1 << std::endl;
            // for(int z = 0; z < 4; z++)
            // {
            //     std::cout << "pending_points: " <<  pending_points[j][z] << ", mask_corner_points: " << mask_corner_points[i][z] << std::endl;
            // }
            // std::cout << std::endl;
            // std::cout << "results: x: " << results.offset_x << ", y: " << results.offset_y << std::endl;

        }

        std::cout << "target_idx: " << target_idx << std::endl;
        std::cout << "target_result x: " << target_results.offset_x << ", y: " << target_results.offset_y << std::endl;
    }
}


void Ten_deviation_handing::mask_cover
(
    cv::Mat& image, 
    const std::vector<int>& hsv_lower, 
    const std::vector<int>& hsv_upper,
    cv::Mat& hsv_mask,
    std::vector<std::vector<cv::Point2f>>& mask_corner_points
)
{
    createHSVMask(image,hsv_lower,hsv_upper,hsv_mask);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
    cv::morphologyEx(hsv_mask,hsv_mask, cv::MORPH_CLOSE, kernel);

    // 拟合方形角点
    mask_corner_points = fitSquareCornersByApprox(image, hsv_mask);

    
    // 打印并显示 每个方形的四个角点
    for (size_t i = 0; i < mask_corner_points.size(); ++i) {
        std::cout << "方形 " << i << " 四个角点：";
        for (const auto& corner : mask_corner_points[i]) {
            std::cout << "(" << corner.x << ", " << corner.y << ") ";
            cv::circle(image, corner, 5, cv::Scalar(0, 0, 255), -1);
        }
        std::vector<float> side_length = {(mask_corner_points[i][1].x - mask_corner_points[i][0].x), (mask_corner_points[i][2].y - mask_corner_points[i][0].y),
            (mask_corner_points[i][3].y - mask_corner_points[i][1].y),(mask_corner_points[i][3].x - mask_corner_points[i][2].x)
            };
        double variance = calculateVarianceOpt(side_length, false);
        std::cout << "variance: " << variance << std::endl;
        // std::cout << std::endl;
        // std::cout << "length_up: " << mask_corner_points[i][1].x - mask_corner_points[i][0].x << std::endl;
        // std::cout << "length_left: " << mask_corner_points[i][2].y - mask_corner_points[i][0].y << std::endl;
        // std::cout << "length_right: " << mask_corner_points[i][3].y - mask_corner_points[i][1].y << std::endl;
        // std::cout << "length_down: " << mask_corner_points[i][3].x - mask_corner_points[i][2].x << std::endl;

        // 删除边长方差大的，即可能被遮挡，会导致后续匹配失效
        if (variance > 10)       
        {
            mask_corner_points.erase(mask_corner_points.begin() + i);
        }
    }

    for (size_t i = 0; i < mask_corner_points.size(); ++i) {
        std::cout << "删除边长方差大的 方形 " << i << " 四个角点：";
        for (const auto& corner : mask_corner_points[i]) {
            std::cout << "(" << corner.x << ", " << corner.y << ") ";
            // cv::circle(image, corner, 5, cv::Scalar(0, 0, 255), -1);
        }
        std::cout << std::endl;
    }
    // 显示 mask 覆盖的区域
    // for(int i = 0; i < image.cols;i++)
    // {
    //     for (int j = 0; j < image.rows; j++)
    //     {
    //         if (mask.at<uchar>(j,i) == 255)
    //         {
    //             cv::circle(image, cv::Point(i,j), 5, cv::Scalar(0, 0, 255), -1);
    //         }
    //     }
    // }
}

void Ten_deviation_handing::set_low_high_hsv
(
    const std::vector<int>& standard_hsv,
    const int range,
    std::vector<int>& low_hsv,
    std::vector<int>& high_hsv
)
{
    if (standard_hsv.size() != 3) 
    {
        std::cout << "🤡in func set_low_high_hsv: the input standard hsv.size() != 3 " << std::endl;
    }
    low_hsv.resize(3);
    high_hsv.resize(3);

    if(standard_hsv[0] - range >= 0 && standard_hsv[0] + range <= 180)
    {
        low_hsv[0] = standard_hsv[0] - range;
        high_hsv[0] = standard_hsv[0] + range;
    }
    if (standard_hsv[0] - range < 0)
    {
        low_hsv[0] = 180 - (standard_hsv[0] - range);       // 环形区域，直接让low的值在170-180左右，在后续的createHSVMask函数中创建两个掩玛范围
        high_hsv[0] = standard_hsv[0]  + range;
    }
    if (standard_hsv[0] + range > 180)
    {
        low_hsv[0] = standard_hsv[0] - range;
        high_hsv[0] = (standard_hsv[0] + range) - 180;      // 环形区域，直接让high的值在0-10左右，在后续的createHSVMask函数中创建两个掩玛范围
    }

    low_hsv[1] = std::clamp(standard_hsv[1] - range, 0,255);
    high_hsv[1] = std::clamp(standard_hsv[1] + range,0,255);

    low_hsv[2] = std::clamp(standard_hsv[2] - range, 0,255);
    high_hsv[2] = std::clamp(standard_hsv[2] + range,0,255);
}

void Ten_deviation_handing::createZBufferMask(const cv::Mat& object_zbuffer, cv::Mat& mask)
{
    // 1. 输入合法性检查（确保是CV_32F类型）
    if (object_zbuffer.type() != CV_32F) {
        std::cerr << "[Error] object_zbuffer must be CV_32F type!" << std::endl;
        mask = cv::Mat();
        return;
    }

    // 2. 核心逻辑：筛选非FLT_MAX的像素，生成CV_8UC1掩码
    // 步骤1：生成浮点型掩码（非FLT_MAX=1.0，FLT_MAX=0.0）
    cv::Mat float_mask = (object_zbuffer != FLT_MAX);
    // 步骤2：转换为CV_8UC1并缩放（1.0→255，0.0→0），与createHSVMask格式一致
    float_mask.convertTo(mask, CV_8UC1, 255.0);
    
}

void Ten_deviation_handing::createHSVMask(const cv::Mat& image, const std::vector<int>& hsv_lower, const std::vector<int>& hsv_upper, cv::Mat& mask) {

    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    int h_low = hsv_lower[0];
    int h_high = hsv_upper[0];
    int s_low = hsv_lower[1];
    int s_high = hsv_upper[1];
    int v_low = hsv_lower[2];
    int v_high = hsv_upper[2];

    // 拆分H通道为单通道，方便单独处理
    std::vector<cv::Mat> hsv_channels;
    cv::split(hsv, hsv_channels);
    cv::Mat h_channel = hsv_channels[0]; // H通道（0-180）
    cv::Mat s_channel = hsv_channels[1]; // S通道（0-255）
    cv::Mat v_channel = hsv_channels[2]; // V通道（0-255）

    // 1. 处理S和V通道的掩码（常规区间）
    cv::Mat s_mask, v_mask;
    cv::inRange(s_channel, s_low, s_high, s_mask);
    cv::inRange(v_channel, v_low, v_high, v_mask);
    cv::Mat sv_mask = s_mask & v_mask; // S和V同时满足的掩码

    // 2. 处理H通道的环形掩码
    cv::Mat h_mask;
    if (h_low <= h_high) {
        // 常规区间（如H:40-70）：直接取h_low~h_high
        cv::inRange(h_channel, h_low, h_high, h_mask);
    } else {
        // 跨0°区间（如H:177-4）：拆分为h_low~180 和 0~h_high
        cv::Mat h_mask1, h_mask2;
        cv::inRange(h_channel, h_low, 180, h_mask1); // 177-180
        cv::inRange(h_channel, 0, h_high, h_mask2);  // 0-4
        h_mask = h_mask1 | h_mask2; // 合并两个区间
    }

    // 3. 合并H、S、V掩码（同时满足）
    mask = h_mask & sv_mask;
}

    Ten::Ten_deviation_handing _DEVIATION_HANDING_;
    Ten::init_deviation _INIT_DEVIATION_;
}
#endif