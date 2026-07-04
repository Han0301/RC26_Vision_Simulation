#ifndef _Ten_deviation_handing_H_
#define _Ten_deviation_handing_H_

#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

#include "occlusion_handing.h"

namespace Ten{
// 表示 方块正面四个点的 配准结果
struct registration_results
{
    float offset_x;     // x 方向的偏差值， 用于矫正框的x方向
    float offset_y;     // y 方向的偏差值， 用于矫正框的y方向
    // float avg_offset_x;     // 四个点的偏差与 offset_x 的差值， 若配准双方均为方形， 则该值很小
    // float avg_offset_y;     // 四个点的偏差与 offset_y 的差值， 若配准双方均为方形， 则该值很小
};

struct init_deviation
{
    std::vector<int> low_hsv;
    std::vector<int> high_hsv;
    registration_results final_results;
    cv::Mat zb_mask;
    cv::Mat tar_mask;
    cv::Point2f zb_centroid;
    cv::Point2f tar_centroid;
    init_deviation()
    {
        low_hsv.resize(3);
        high_hsv.resize(3);
    }
};
class Ten_deviation_handing
{
public:

/**
 * @brief 通过 有效像素 进行配准
 * @param pending_points 直接由坐标转化的待配准的正面 3d 点组
 * @param mask_corner_points 由有效像素填充的掩玛所得到的方块角点列表
 * @param final_results 最终的配准误差值
*/ 
void registration
(
    std::vector<std::vector<cv::Point2f>> pending_points,
    std::vector<std::vector<cv::Point2f>> mask_corner_points,
    registration_results& final_results
);


/**
 * @brief 填充有效像素的掩玛， 并根据掩玛拟合出方形角点
 * @param image 输入图像
 * @param hsv_lower 有效像素的 hsv 下限
 * @param hsv_upper 有效像素的 hsv 上限
 * @param hsv_mask 生成的有效像素掩玛
 * @param mask_corner_points 待写入的 由有效像素填充的掩玛所得到的方块角点列表
*/
void mask_cover
(
    cv::Mat& image, 
    const std::vector<int>& hsv_lower, 
    const std::vector<int>& hsv_upper,
    cv::Mat& hsv_mask,
    std::vector<std::vector<cv::Point2f>>& mask_corner_points
);

/**
 * @brief 根据标准hsv值，来设置有效像素的上下限
 * @param standard_hsv 输入的标准 hsv
 * @param range 所形成的范围的偏差值
 * @param low_hsv 输出的 hsv 下限
 * @param high_hsv 输出的 hsv 上限
*/
void set_low_high_hsv
(
    const std::vector<int>& standard_hsv,
    const int range,
    std::vector<int>& low_hsv,
    std::vector<int>& high_hsv
);

/**
 * @brief 从ZBuffer生成二值掩码（与createHSVMask输出格式一致）
 * @param object_zbuffer 输入ZBuffer矩阵（CV_32F，FLT_MAX为无效值）
 * @param mask 输出二值掩码（CV_8UC1，255=有效，0=无效）
 */
void createZBufferMask(const cv::Mat& object_zbuffer, cv::Mat& mask);

/**
 * @brief 辅助功能函数：创建适配H通道环形特性的HSV掩码
 * @param image 输入的HSV图像
 * @param hsv_lower HSV下限 [H, S, V]
 * @param hsv_upper HSV上限 [H, S, V]
 * @param mask 输出的二值掩码（CV_8UC1）
 */
void createHSVMask(const cv::Mat& image, const std::vector<int>& hsv_lower, const std::vector<int>& hsv_upper, cv::Mat& mask);

/**
 * @brief 计算单个二值掩码有效区域的质心和边界框
 * @param mask 输入二值掩码（CV_8UC1，255=有效）
 * @param centroid 输出质心坐标（x=水平，y=竖直）
 * @return 是否成功计算（有效像素数>0则返回true）
 */
void getMask_centroid(const cv::Mat& mask, cv::Point2f& centroid) {
    // 1. 提取所有有效像素的坐标
    cv::Mat non_zero_points;
    cv::findNonZero(mask, non_zero_points);
    if (non_zero_points.total() == 0) {
        std::cout << "🤡in func getMask_centroid: non_zero_points.total() == 0!" << std::endl;
        centroid = cv::Point2f(0, 0);
        return;
    }

    // 2. 计算质心（重心）：所有有效像素坐标的平均值
    cv::Scalar mean = cv::mean(non_zero_points);
    centroid = cv::Point2f(mean[0], mean[1]); // mean[0]=x均值，mean[1]=y均值
}

private:

/**
 * @brief 功能函数 多边形逼近：从mask中提取方形区域的四个角点
 * @param image 输入图像（用于绘制验证）
 * @param mask 二值掩码（255为目标区域）
 * @return 每个方形的四个角点（按顺时针排序：左上、右上、右下、左下）
 */
std::vector<std::vector<cv::Point2f>> fitSquareCornersByApprox(cv::Mat& image, const cv::Mat& mask) {
    std::vector<std::vector<cv::Point2f>> squareCorners;

    // 1. 预处理mask：去除噪点、填充小孔
    cv::Mat processedMask = mask.clone();
    cv::Mat close_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(processedMask, processedMask, cv::MORPH_CLOSE, close_kernel);
    cv::Mat open_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(processedMask, processedMask, cv::MORPH_OPEN, open_kernel);

    // 2. 查找轮廓（仅保留最外层轮廓）
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(processedMask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 3. 遍历轮廓，进行多边形逼近
    for (size_t i = 0; i < contours.size(); ++i) {
        if (cv::contourArea(contours[i]) < 200) continue;

        double perimeter = cv::arcLength(contours[i], true);
        double epsilon = 0.12 * perimeter;
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contours[i], approx, epsilon, true);

        // 存储最终要加入结果集的角点
        std::vector<cv::Point2f> final_corners;

        // 分支1：已逼近为4个点
        if (approx.size() == 4) {
            for (const auto& p : approx) {
                final_corners.push_back(cv::Point2f(p.x, p.y));
            }
            // 排序（和补全逻辑保持一致）
            std::sort(final_corners.begin(), final_corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
                return (a.x + a.y) < (b.x + b.y);
            });
            if (final_corners[1].y > final_corners[2].y) std::swap(final_corners[1], final_corners[2]);
            if (final_corners[3].y < final_corners[2].y) std::swap(final_corners[3], final_corners[2]);
        }
        // 分支2：2/3个点，补全为4个点
        else if (approx.size() == 2 || approx.size() == 3) {
            // 补全角点
            std::vector<cv::Point> completed_pts = completeSquarePoints(approx);
            
            if (completed_pts.size() == 4)
            {
                // 转换为Point2f
                for (const auto& p : completed_pts) {
                    final_corners.push_back(cv::Point2f(p.x, p.y));
                }
            } 
            else 
            {
                continue; // 跳过该轮廓
            }
        }
        // 分支3：其他数量的点，跳过
        else  continue;

        // 将处理后的4个点加入最终结果集
        if (!final_corners.empty() && final_corners.size() == 4) {
            squareCorners.push_back(final_corners);
            // 绘制最终角点（红色，标记有效角点）
            for (const auto& corner : final_corners) {
                cv::circle(image, corner, 5, cv::Scalar(0, 0, 255), -1);
            }
        }
    }

    return squareCorners;
}


/**
 * @brief 计算方块四个点的偏差
 * @param pending_points 待配准的方块正面四个点
 * @param target_points 目标的四个点
 * @param results 待写入的对象， 表示偏差
*/
void cal_offsets
(
    const std::vector<cv::Point2f>& pending_points, 
    const std::vector<cv::Point2f>& target_points, 
    registration_results& results
) 
{
    if (pending_points.size() != target_points.size())
    {
        std::cout << "in func registration: pending_points.size() != target_points.size()" << std::endl;
        return;
    }
    float total_offset_x = 0.0f;
    float total_offset_y = 0.0f;
    for(int i = 0; i < pending_points.size(); i++)
    {
        total_offset_x += (target_points[i].x - pending_points[i].x);
        total_offset_y += (target_points[i].y - pending_points[i].y);
    }
    results.offset_x = total_offset_x / static_cast<float>(pending_points.size());
    results.offset_y = total_offset_y / static_cast<float>(pending_points.size());
}

/**
 * @brief 计算一组数值的方差（优化版，一次遍历）
 * @param data 输入数值集合
 * @param is_sample 是否为样本方差
 * @return 方差值
 */
template <typename T>
double calculateVarianceOpt(const std::vector<T>& data, bool is_sample = false) {
    if (data.empty()) {
        throw std::invalid_argument("数据集合不能为空！");
    }
    if (data.size() == 1) {
        return 0.0;
    }

    // 一次遍历：同时计算总和、平方和
    double sum = 0.0, sum_square = 0.0;
    for (const auto& val : data) {
        sum += val;
        sum_square += static_cast<double>(val) * val; // 强制转double避免溢出
    }

    // 计算均值和方差（变形公式）
    size_t n = data.size();
    double mean = sum / n;
    double variance = (sum_square / n) - (mean * mean);

    // 样本方差需修正分母
    if (is_sample) {
        variance = variance * n / (n - 1);
    }

    return variance;
}
    
/**
 * @brief 按方形规则补全角点（size=2/3 → size=4）
 * @param input_pts 输入的原始点集（size=2/3）
 * @return 补全后的4个方形角点（轴对齐优先，按顺时针排序：左上、右上、右下、左下）
 */
std::vector<cv::Point> completeSquarePoints(const std::vector<cv::Point>& input_pts) {
    std::vector<cv::Point> square_pts = input_pts;
    int n = input_pts.size();

    // 合法性检查
    if (n == 4) return square_pts; // 已为4个点，直接返回
    if (n < 2 || n > 3) {
        std::cerr << "[Error] 输入点数量必须为2或3！" << std::endl;
        return {};
    }

    // ------------------- 步骤1：提取坐标特征（轴对齐判断） -------------------
    std::vector<int> xs, ys;
    for (const auto& p : input_pts) {
        xs.push_back(p.x);
        ys.push_back(p.y);
    }
    // 去重后的坐标（判断是否轴对齐）
    std::sort(xs.begin(), xs.end());
    std::sort(ys.begin(), ys.end());
    auto unique_x = std::unique(xs.begin(), xs.end());
    auto unique_y = std::unique(ys.begin(), ys.end());
    int x_unique = std::distance(xs.begin(), unique_x);
    int y_unique = std::distance(ys.begin(), unique_y);

    // ------------------- 步骤2：轴对齐方形补全（核心） -------------------
    if (x_unique <= 2 && y_unique <= 2) { // 轴对齐特征：x/y去重后≤2个值
        int x1 = xs[0], x2 = xs.back();
        int y1 = ys[0], y2 = ys.back();

        // 生成4个角点（轴对齐方形的所有组合）
        std::vector<cv::Point> axis_aligned_pts = {
            cv::Point(x1, y1), // 左上
            cv::Point(x2, y1), // 右上
            cv::Point(x2, y2), // 右下
            cv::Point(x1, y2)  // 左下
        };

        // 过滤已有点，补充缺失点
        for (const auto& p : axis_aligned_pts) {
            bool exists = false;
            for (const auto& q : input_pts) {
                if (p == q) {
                    exists = true;
                    break;
                }
            }
            if (!exists) {
                square_pts.push_back(p);
                if (square_pts.size() == 4) break; // 补全4个点后退出
            }
        }
    } 
    // ------------------- 步骤3：非轴对齐方形补全（向量法） -------------------
    else {
        if (n == 2) {
            // 已知两个点（对角线）：补全另外两个对角点
            cv::Point p1 = input_pts[0], p2 = input_pts[1];
            // 计算中点
            cv::Point2f center((p1.x + p2.x)/2.0f, (p1.y + p2.y)/2.0f);
            // 向量旋转90°（方形对角线垂直且等长）
            cv::Point2f vec(p1.x - center.x, p1.y - center.y);
            cv::Point2f vec_rot(-vec.y, vec.x); // 旋转90°
            // 生成另外两个点
            cv::Point p3(center.x + vec_rot.x, center.y + vec_rot.y);
            cv::Point p4(center.x - vec_rot.x, center.y - vec_rot.y);
            square_pts.push_back(p3);
            square_pts.push_back(p4);
        } else if (n == 3) {
            // 已知三个点：计算前两个点的向量，推导第四个点
            cv::Point p1 = input_pts[0], p2 = input_pts[1], p3 = input_pts[2];
            // 计算向量
            cv::Point vec1 = p2 - p1;
            cv::Point vec2 = p3 - p2;
            // 第四个点 = p3 + (p1 - p2)（保证对边平行）
            cv::Point p4 = p3 + (p1 - p2);
            square_pts.push_back(p4);
        }
    }

    // ------------------- 步骤4：排序（顺时针，左上→右上→右下→左下） -------------------
    if (square_pts.size() == 4) {
        std::sort(square_pts.begin(), square_pts.end(), [](const cv::Point& a, const cv::Point& b) {
            return (a.x + a.y) < (b.x + b.y);
        });
        if (square_pts[1].y > square_pts[2].y) std::swap(square_pts[1], square_pts[2]);
        if (square_pts[3].y < square_pts[2].y) std::swap(square_pts[3], square_pts[2]);
    }

    return square_pts;
}

};
    extern Ten::Ten_deviation_handing _DEVIATION_HANDING_;
    extern Ten::init_deviation _INIT_DEVIATION_;
}
#endif