#ifndef _Ten_edge_handing_CPP_
#define _Ten_edge_handing_CPP_

#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

// 表示 方块正面四个点的 配准结果
struct registration_results
{
    float offset_x;     // x 方向的偏差值， 用于矫正框的x方向
    float offset_y;     // y 方向的偏差值， 用于矫正框的y方向
    // float avg_offset_x;     // 四个点的偏差与 offset_x 的差值， 若配准双方均为方形， 则该值很小
    // float avg_offset_y;     // 四个点的偏差与 offset_y 的差值， 若配准双方均为方形， 则该值很小
};
/**
 * @brief 辅助函数：创建适配H通道环形特性的HSV掩码
 * @param hsv 输入的HSV图像
 * @param hsv_lower HSV下限 [H, S, V]
 * @param hsv_upper HSV上限 [H, S, V]
 * @param mask 输出的二值掩码（CV_8UC1）
 */
void createHSVMask(const cv::Mat& hsv, const std::vector<int>& hsv_lower, const std::vector<int>& hsv_upper, cv::Mat& mask) {
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

/**
 * @brief Sobel算子边缘检测（水平+垂直边缘）
 * @param image 输入输出图像（BGR格式，直接在原图绘制红色轮廓）
 * @param hsv_lower HSV下限 [H, S, V]（std::vector<int>，需3个元素）
 * @param hsv_upper HSV上限 [H, S, V]（std::vector<int>，需3个元素）
 */
void sobelEdgeDetect(cv::Mat& image, const std::vector<int>& hsv_lower, const std::vector<int>& hsv_upper) {
    // 1. 颜色分割（HSV，适配H通道环形特性）
    cv::Mat hsv, mask;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    createHSVMask(hsv, hsv_lower, hsv_upper, mask); // 替换原inRange

    // 2. 形态学去噪（闭合操作：膨胀+腐蚀）
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    // 3. Sobel边缘检测
    cv::Mat sobel_x, sobel_y;
    cv::Sobel(mask, sobel_x, CV_64F, 1, 0, 3);  // 垂直边缘
    cv::Sobel(mask, sobel_y, CV_64F, 0, 1, 3);  // 水平边缘
    // 转换为8位无符号数（取绝对值）
    cv::convertScaleAbs(sobel_x, sobel_x);
    cv::convertScaleAbs(sobel_y, sobel_y);
    // 合并水平+垂直边缘
    cv::Mat sobel_edges;
    cv::addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0, sobel_edges);

    // 4. 查找轮廓并绘制到原图
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(sobel_edges, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::drawContours(image, contours, -1, cv::Scalar(0, 0, 255), 2);  // 红色轮廓，线宽2
}

/**
 * @brief Scharr算子边缘检测（边缘定位更精准）
 * @param image 输入输出图像（BGR格式）
 * @param hsv_lower HSV下限 [H, S, V]
 * @param hsv_upper HSV上限 [H, S, V]
 */
void scharrEdgeDetect(cv::Mat& image, const std::vector<int>& hsv_lower, const std::vector<int>& hsv_upper) {
    // 1. 颜色分割+形态学去噪（适配H通道环形）
    cv::Mat hsv, mask;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    createHSVMask(hsv, hsv_lower, hsv_upper, mask); // 替换原inRange

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    // 2. Scharr边缘检测
    cv::Mat scharr_x, scharr_y;
    cv::Scharr(mask, scharr_x, CV_64F, 1, 0);  // 垂直边缘
    cv::Scharr(mask, scharr_y, CV_64F, 0, 1);  // 水平边缘
    cv::convertScaleAbs(scharr_x, scharr_x);
    cv::convertScaleAbs(scharr_y, scharr_y);
    cv::Mat scharr_edges;
    cv::addWeighted(scharr_x, 0.5, scharr_y, 0.5, 0, scharr_edges);

    // 3. 查找轮廓并绘制
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(scharr_edges, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::drawContours(image, contours, -1, cv::Scalar(0, 0, 255), 2);
}

/**
 * @brief Prewitt算子边缘检测（极简梯度，速度快）
 * @param image 输入输出图像（BGR格式）
 * @param hsv_lower HSV下限 [H, S, V]
 * @param hsv_upper HSV上限 [H, S, V]
 */
void prewittEdgeDetect(cv::Mat& image, const std::vector<int>& hsv_lower, const std::vector<int>& hsv_upper) {
    // 1. 颜色分割+形态学去噪（适配H通道环形）
    cv::Mat hsv, mask;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    createHSVMask(hsv, hsv_lower, hsv_upper, mask); // 替换原inRange

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    // 2. Prewitt卷积核（手动定义）
    cv::Mat kernel_x = (cv::Mat_<float>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
    cv::Mat kernel_y = (cv::Mat_<float>(3, 3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);

    // 3. 卷积计算梯度
    cv::Mat prewitt_x, prewitt_y;
    cv::filter2D(mask, prewitt_x, CV_64F, kernel_x);
    cv::filter2D(mask, prewitt_y, CV_64F, kernel_y);
    cv::convertScaleAbs(prewitt_x, prewitt_x);
    cv::convertScaleAbs(prewitt_y, prewitt_y);
    cv::Mat prewitt_edges;
    cv::addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0, prewitt_edges);

    // 4. 查找轮廓并绘制
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(prewitt_edges, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::drawContours(image, contours, -1, cv::Scalar(0, 0, 255), 2);
}

/**
 * @brief Laplacian算子边缘检测（二阶导数，定位极准）
 * @param image 输入输出图像（BGR格式）
 * @param hsv_lower HSV下限 [H, S, V]
 * @param hsv_upper HSV上限 [H, S, V]
 */
void laplacianEdgeDetect(cv::Mat& image, const std::vector<int>& hsv_lower, const std::vector<int>& hsv_upper) {
    // 1. 颜色分割+形态学去噪（适配H通道环形）
    cv::Mat hsv, mask;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    createHSVMask(hsv, hsv_lower, hsv_upper, mask); // 替换原inRange

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    // 2. Laplacian边缘检测
    cv::Mat laplacian, laplacian_edges;
    cv::Laplacian(mask, laplacian, CV_64F, 3);
    cv::convertScaleAbs(laplacian, laplacian_edges);

    // 3. 查找轮廓并绘制
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(laplacian_edges, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::drawContours(image, contours, -1, cv::Scalar(0, 0, 255), 2);
}

/**
 * @brief LoG算子边缘检测（Laplacian优化版，抗噪）
 * @param image 输入输出图像（BGR格式）
 * @param hsv_lower HSV下限 [H, S, V]
 * @param hsv_upper HSV上限 [H, S, V]
 */
void logEdgeDetect(cv::Mat& image, const std::vector<int>& hsv_lower, const std::vector<int>& hsv_upper) {
    // 1. 颜色分割+形态学去噪（适配H通道环形）
    cv::Mat hsv, mask;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    createHSVMask(hsv, hsv_lower, hsv_upper, mask); // 替换原inRange

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    // 2. LoG：先高斯平滑，再Laplacian
    cv::Mat gauss, log, log_edges;
    cv::GaussianBlur(mask, gauss, cv::Size(3, 3), 0);  // 高斯去噪
    cv::Laplacian(gauss, log, CV_64F, 3);
    cv::convertScaleAbs(log, log_edges);

    // 3. 查找轮廓并绘制
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(log_edges, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (size_t i = 0; i < contours.size(); i ++)
    {
        for(size_t j = 0; j < contours[i].size(); j ++)
        {
            cv::circle(image, contours[i][j], 5, cv::Scalar(0, 0, 255), -1);
        }
    }
    // cv::drawContours(image, contours, -1, cv::Scalar(0, 255, 0), 2);
}

/**
 * @brief 形态学梯度边缘检测（无杂点，适配二值mask）
 * @param image 输入输出图像（BGR格式）
 * @param hsv_lower HSV下限 [H, S, V]
 * @param hsv_upper HSV上限 [H, S, V]
 */
void morphGradientEdgeDetect(cv::Mat& image, const std::vector<int>& hsv_lower, const std::vector<int>& hsv_upper) {
    // 1. 颜色分割（适配H通道环形）
    cv::Mat hsv, mask;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    createHSVMask(hsv, hsv_lower, hsv_upper, mask); // 替换原inRange

    // 2. 形态学去噪
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    // 3. 形态学梯度提取边缘
    cv::Mat morph_edges;
    cv::morphologyEx(mask, morph_edges, cv::MORPH_GRADIENT, kernel);
    // 可选：膨胀边缘，让轮廓更清晰
    cv::dilate(morph_edges, morph_edges, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2)), cv::Point(-1, -1), 1);

    // 4. 查找轮廓并绘制
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(morph_edges, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::drawContours(image, contours, -1, cv::Scalar(0, 0, 255), 2);
}

/**
 * @brief 自适应阈值Canny边缘检测（边缘连续，抗噪）
 * @param image 输入输出图像（BGR格式）
 * @param hsv_lower HSV下限 [H, S, V]
 * @param hsv_upper HSV上限 [H, S, V]
 */
void adaptiveCannyEdgeDetect(cv::Mat& image, const std::vector<int>& hsv_lower, const std::vector<int>& hsv_upper) {
    // 1. 颜色分割+形态学去噪（适配H通道环形）
    cv::Mat hsv, mask;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    createHSVMask(hsv, hsv_lower, hsv_upper, mask); // 替换原inRange

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    // 2. 手动计算mask的中位数（修复不存在的cv::median）
    double median_val = 0.0;
    std::vector<uchar> pixels;
    for (int i = 0; i < mask.rows; ++i) {
        for (int j = 0; j < mask.cols; ++j) {
            uchar pixel = mask.at<uchar>(i, j);
            if (pixel > 0) {
                pixels.push_back(pixel);
            }
        }
    }
    if (!pixels.empty()) {
        std::sort(pixels.begin(), pixels.end());
        int mid_idx = pixels.size() / 2;
        if (pixels.size() % 2 == 0) {
            median_val = (pixels[mid_idx - 1] + pixels[mid_idx]) / 2.0;
        } else {
            median_val = pixels[mid_idx];
        }
    } else {
        median_val = 127.0;
    }

    // 3. 计算自适应Canny阈值
    double lower_thresh = std::max(0.0, 0.6 * median_val);
    double upper_thresh = std::min(255.0, 1.4 * median_val);

    // 4. Canny边缘检测
    cv::Mat canny_edges;
    cv::Canny(mask, canny_edges, lower_thresh, upper_thresh);

    // 5. 查找轮廓并绘制
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(canny_edges, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::drawContours(image, contours, -1, cv::Scalar(0, 0, 255), 2);
}

/**
 * @brief 直接提取轮廓（无需额外边缘检测，适合高质量mask）
 * @param image 输入输出图像（BGR格式）
 * @param hsv_lower HSV下限 [H, S, V]
 * @param hsv_upper HSV上限 [H, S, V]
 */
void findContoursEdgeDetect(cv::Mat& image, const std::vector<int>& hsv_lower, const std::vector<int>& hsv_upper) {
    // 1. 颜色分割+形态学去噪（适配H通道环形）
    cv::Mat hsv, mask;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    createHSVMask(hsv, hsv_lower, hsv_upper, mask); // 替换原inRange

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    // 2. 直接查找轮廓并绘制
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    // cv::drawContours(image, contours, -1, cv::Scalar(0, 0, 255), 2);

    // 可选：轮廓逼近（拟合几何形状，如方块）
    std::vector<std::vector<cv::Point>> approx_contours;
    for (auto& cnt : contours) {
        double epsilon = 0.01 * cv::arcLength(cnt, true);
        std::vector<cv::Point> approx;
        cv::approxPolyDP(cnt, approx, epsilon, true);
        approx_contours.push_back(approx);
    }
    // 拟合后的轮廓绘制为蓝色（可选）
    cv::drawContours(image, approx_contours, -1, cv::Scalar(0, 0, 255), 2);
}

/**
 * @brief Roberts Cross算子边缘检测（检测斜向边缘，速度极快）
 * @param image 输入输出图像（BGR格式）
 * @param hsv_lower HSV下限 [H, S, V]
 * @param hsv_upper HSV上限 [H, S, V]
 */
void robertsEdgeDetect(cv::Mat& image, const std::vector<int>& hsv_lower, const std::vector<int>& hsv_upper) {
    // 1. 颜色分割+形态学去噪（适配H通道环形）
    cv::Mat hsv, mask;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    createHSVMask(hsv, hsv_lower, hsv_upper, mask); // 替换原inRange

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    // 2. Roberts卷积核
    cv::Mat kernel1 = (cv::Mat_<float>(2, 2) << 1, 0, 0, -1);
    cv::Mat kernel2 = (cv::Mat_<float>(2, 2) << 0, 1, -1, 0);

    // 3. 卷积计算
    cv::Mat roberts1, roberts2;
    cv::filter2D(mask, roberts1, CV_64F, kernel1);
    cv::filter2D(mask, roberts2, CV_64F, kernel2);

    // 4. 计算梯度幅值并转换为8位
    cv::Mat roberts_edges;
    cv::magnitude(roberts1, roberts2, roberts_edges);
    cv::convertScaleAbs(roberts_edges, roberts_edges);

    // 5. 查找轮廓并绘制
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(roberts_edges, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::drawContours(image, contours, -1, cv::Scalar(0, 0, 255), 2);
}

/**
 * @brief 多边形逼近法：从mask中提取方形区域的四个角点
 * @param image 输入图像（用于绘制验证）
 * @param mask 二值掩码（255为目标区域）
 * @return 每个方形的四个角点（按顺时针排序：左上、右上、右下、左下）
 */
std::vector<std::vector<cv::Point2f>> fitSquareCornersByApprox(cv::Mat& image, const cv::Mat& mask) {
    std::vector<std::vector<cv::Point2f>> squareCorners;

    // 1. 预处理mask：去除噪点、填充小孔
    cv::Mat processedMask = mask.clone();
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(processedMask, processedMask, cv::MORPH_CLOSE, kernel);  // 填充小孔
    cv::morphologyEx(processedMask, processedMask, cv::MORPH_OPEN, kernel);   // 去除小噪点

    // 2. 查找轮廓（仅保留最外层轮廓）
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(processedMask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 3. 遍历轮廓，进行多边形逼近
    for (size_t i = 0; i < contours.size(); ++i) {
        // 过滤小面积噪点（根据图像分辨率调整阈值）
        if (cv::contourArea(contours[i]) < 100) continue;

        // 计算轮廓周长，设置逼近精度（epsilon为周长的2%，可调整）
        double perimeter = cv::arcLength(contours[i], true);
        double epsilon = 0.02 * perimeter;

        // 多边形逼近
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contours[i], approx, epsilon, true);

        // 筛选四边形（顶点数为4）
        if (approx.size() == 4) {
            // 转换为Point2f并排序（顺时针：左上、右上、右下、左下）
            std::vector<cv::Point2f> corners;
            for (const auto& p : approx) {
                corners.push_back(cv::Point2f(p.x, p.y));
            }

            // 按x+y和排序，调整角点顺序
            std::sort(corners.begin(), corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
                return (a.x + a.y) < (b.x + b.y);
            });
            if (corners[1].y > corners[2].y) std::swap(corners[1], corners[2]);
            if (corners[3].y < corners[2].y) std::swap(corners[3], corners[2]);

            squareCorners.push_back(corners);

            // 验证：在图像上绘制角点（绿色圆点）
            for (const auto& corner : corners) {
                cv::circle(image, corner, 4, cv::Scalar(0, 255, 0), -1);
            }
        }
    }

    return squareCorners;
}


void mask_cover(cv::Mat& image, const std::vector<int>& hsv_lower, const std::vector<int>& hsv_upper)
{
    cv::Mat hsv_image, mask;
    cv::cvtColor(image, hsv_image, cv::COLOR_BGR2HSV);
    createHSVMask(hsv_image,hsv_lower,hsv_upper,mask);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
    cv::morphologyEx(mask,mask, cv::MORPH_CLOSE, kernel);

    // 拟合方形角点
    auto squareCorners = fitSquareCornersByApprox(image, mask);

    // 打印并显示 每个方形的四个角点
    for (size_t i = 0; i < squareCorners.size(); ++i) {
        std::cout << "方形 " << i << " 四个角点：";
        for (const auto& corner : squareCorners[i]) {
            std::cout << "(" << corner.x << ", " << corner.y << ") ";
            cv::circle(image, corner, 5, cv::Scalar(0, 0, 255), -1);
        }
        std::cout << std::endl;
    }

    // 显示 mask 覆盖的区域
    // for(int i = 0; i < hsv_image.cols;i++)
    // {
    //     for (int j = 0; j < hsv_image.rows; j++)
    //     {
    //         if (mask.at<uchar>(j,i) == 255)
    //         {
    //             cv::circle(image, cv::Point(i,j), 5, cv::Scalar(0, 0, 255), -1);
    //         }
    //     }
    // }
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
    registration_results results
) 
{
    if (pending_points.size() != target_points.size())
    {
        std::cout << "in func registration: pending_points.size() != target_points.size()" << std::endl;
        return;
    }
    int total_offset_x = 0;
    int total_offset_y = 0;
    for(int i = 0; i < pending_points.size(); i++)
    {
        total_offset_x += (target_points[i].x - pending_points[i].x);
        total_offset_y += (target_points[i].y - pending_points[i].y);
    }
    results.offset_x = total_offset_x / pending_points.size();
    results.offset_y = total_offset_y / pending_points.size();
}

#endif