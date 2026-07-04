#include <ros/ros.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <image_transport/image_transport.h>
#include <cmath>
#include <sensor_msgs/CameraInfo.h>
#include <tf2_msgs/TFMessage.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h> 
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/Imu.h>
#include <Eigen/Geometry>
#include <cstring>  
#include <iostream>
#include <thread>
#include <stdexcept>
#include <bitset>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <nav_msgs/Odometry.h>
#include <algorithm>
#include <unordered_map>

//点云 cloud map 
//tf cloud->map
//8 判断点的数量
//  订阅雷达话题转化成pcl点云，判断pcl点（坐标变换部分暂时不考虑，假设在一个坐标系中直接比较）的数据是否在指定的区域内
// 实现： 改成8个点的立体区域， 重新定义结构体，判断

#define _L_ 1.2
#define _H_ 0.2
#define _ly1_ 0.425
#define _ly2_ 0.775
#define _lx1_ 0.425
#define _lh_ 0.35 
#define _X_  3.17              //2.17
#define _Y_  1.2             //0.2 

struct G
{
    G()
    {
        // 2. 相机内参矩阵 K
        _K = (cv::Mat_<double>(3,3) <<
            1012.0711525658555, 0, 960.5,
            0, 1012.0711525658555, 540.5,
            0, 0, 1);
        // 3. 畸变系数（假设零畸变）
        _distCoeffs = cv::Mat::zeros(5, 1, CV_64F);

        // 4. 外参：旋转向量和平移向量
        _rvec = (cv::Mat_<double>(3,1) << 0.0, 0.0, 0.0); // 旋转（弧度）
        _tvec = (cv::Mat_<double>(3,1) << 0.0, 0.0, 0.0);   // 平移（米）
        // 1. 定义3D点（世界坐标）
        //std::vector<cv::Point3f> objectPoints;
        //objectPoints.push_back(cv::Point3f(0, 0, 0));   // 原点

        for(int j = 0; j < 4; j++)
        {
            for(int i = 0; i < 3; i++)
            {
                _objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_, _Y_ + i*_L_ + _ly1_, _arr[j*3+i]+_lh_));
                _objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_, _Y_ + i*_L_ + _ly2_, _arr[j*3+i]+_lh_));
                _objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_, _Y_ + i*_L_ + _ly2_, _arr[j*3+i]));
                _objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_, _Y_ + i*_L_ + _ly1_, _arr[j*3+i]));
                //std::cout << "_X_ + j*_L_: " << _X_ + j*_L_ << "_Y_ + i*_L_: " << _Y_ + i*_L_ <<"_X_ + (j + 1)*_L_: " <<  _X_ + (j + 1)*_L_<< "_Y_ + (i + 1)*_L_: "<< _Y_ + (i + 1)*_L_<< std::endl;
            }
        }

        // for(int j = 0; j < 4; j++)
        // {
        //     for(int i = 0; i < 3; i++)
        //     {
        //         side_objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_ + _lh_, _Y_ + i*_L_ + _ly1_, _arr[j*3+i]+_lh_));
        //         side_objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_ + _lh_, _Y_ + i*_L_ + _ly2_, _arr[j*3+i]+_lh_));
        //         side_objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_ + _lh_, _Y_ + i*_L_ + _ly2_, _arr[j*3+i]));
        //         side_objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_ + _lh_, _Y_ + i*_L_ + _ly1_, _arr[j*3+i]));
                
        //         // side_objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_, _Y_ + i*_L_ + _ly2_, _arr[j*3+i]+_lh_));                
        //         // side_objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_ + _lh_, _Y_ + i*_L_ + _ly2_, _arr[j*3+i]+_lh_));                
        //         // side_objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_ + _lh_, _Y_ + i*_L_ + _ly2_, _arr[j*3+i]));
        //         // side_objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_, _Y_ + i*_L_ + _ly2_, _arr[j*3+i]));

             

        //         std::cout << "_X_ + j*_L_: " << _X_ + j*_L_ << "_Y_ + i*_L_: " << _Y_ + i*_L_ <<"_X_ + (j + 1)*_L_: " <<  _X_ + (j + 1)*_L_<< "_Y_ + (i + 1)*_L_: "<< _Y_ + (i + 1)*_L_<< std::endl;
        //     }
        // }        



        for(int i = 0; i < 48; i++)
        {
            //_c_objectPoints.push_back(cv::Point3f(-_objectPoints[i].y, 1.3 - _objectPoints[i].z, _objectPoints[i].x));
            _c_objectPoints.push_back(cv::Point3f(_objectPoints[i].y, 1.3 - _objectPoints[i].z, _objectPoints[i].x));
        };
        // for(int i = 0; i < 48; i++)
        // {
        //     //_c_objectPoints.push_back(cv::Point3f(-_objectPoints[i].y, 1.3 - _objectPoints[i].z, _objectPoints[i].x));
        //     _c_side_objectPoints.push_back(cv::Point3f(side_objectPoints[i].y, 1.3 - side_objectPoints[i].z, side_objectPoints[i].x));
        // };

    }
    struct idx_box{
        int idx;
        float box_x1;
        float box_y1;
        float box_x2;
        float box_y2;
        float distance;
    };
    // struct standard_3d_x_y{
    //     float x_min;
    //     float y_min;
    //     float x_max;
    //     float y_max;
    // };
    float x;
    float y;
    float z;
    //std::vector<struct standard_3d_x_y> standard_3d_x_y_lists;    
    std::vector<struct idx_box> idx_box_lists;
    std::vector<struct idx_box> idx_box_side_lists; 
    std::vector<cv::Point3f> _objectPoints;
    std::vector<cv::Point3f> _c_objectPoints;
    std::vector<cv::Point3f> side_objectPoints;
    std::vector<cv::Point3f> _c_side_objectPoints;

    cv::Mat _K;
    cv::Mat _distCoeffs;
    cv::Mat _rvec;
    cv::Mat _tvec;
    cv::Mat _image;
    std::mutex _mtx_image;
    std::mutex _mtx_tf;
    std::vector<cv::Point2f> imagePoints;
    std::vector<cv::Point2f> side_imagePoints;
    float _arr[12] = {0.4, 0.2, 0.4, 0.2, 0.4, 0.6, 0.4, 0.6, 0.4, 0.2, 0.4, 0.2};

}global;

// 颜色枚举
enum class TargetColor {
    Green,
    Yellow,
    other
};

// 定义目标颜色的BGR阈值范围（OpenCV默认BGR格式）
const std::unordered_map<TargetColor, std::tuple<int, int, int, int, int, int>> COLOR_THRESHOLDS = {
    {TargetColor::Green,  {0, 15, 100, 255, 0, 10}},   // B:100-255, G:100-255, R:0-100
    {TargetColor::Yellow,{-1, 10, 60, 200, 60, 200}},    // B:0-100, G:150-255, R:150-255
    {TargetColor::other, {150,255,150,255,150,255}}
};

// 判断像素是否属于目标颜色
bool isPixelTargetColor(const cv::Vec3b& pixel, TargetColor targetColor) {
    auto it = COLOR_THRESHOLDS.find(targetColor);
    if (it == COLOR_THRESHOLDS.end()) return false;
    
    auto [b_min, b_max, g_min, g_max, r_min, r_max] = it->second;
    return (pixel[0] >= b_min && pixel[0] <= b_max) &&  // B通道
           (pixel[1] >= g_min && pixel[1] <= g_max) &&  // G通道
           (pixel[2] >= r_min && pixel[2] <= r_max);    // R通道
}

std::vector<G::idx_box> filterBoxesByColor(
    const cv::Mat& image,
    const std::vector<G::idx_box>& boxes
) {
    std::vector<G::idx_box> filteredBoxes;

    for (const auto& box : boxes) {
        int greenCount = 0;  // 绿色点数量
        int yellowCount = 0; // 黄色点数量
        int othercount = 0;
        std:: cout << "start color, box.idx: :" << box.idx << std::endl;
        // 计算框的四个角点像素坐标
        std::vector<cv::Point> corners = {
            cv::Point(box.box_x1+14, box.box_y1+14),  // 左上角
            cv::Point(box.box_x1+14, box.box_y2-30),  // 左下角
            cv::Point(box.box_x2-14, box.box_y2-14),  // 右下角
            cv::Point(box.box_x2-14, box.box_y1+14),   // 右上角
            cv::Point(box.box_x2-20, box.box_y2-20),   // 右上角
            cv::Point((box.box_x1 + box.box_x2) / 2,(box.box_y1 + box.box_y2) / 2+ (box.box_y2 - box.box_y1) / 3),
            cv::Point((box.box_x1 + box.box_x2) / 2,(box.box_y1 + box.box_y2) / 2- (box.box_y2 - box.box_y1) / 3)
        };
        // 计算框的中心点像素坐标
        cv::Point center = cv::Point(
            static_cast<int>((box.box_x1 + box.box_x2) / 2),  // 中心X
            static_cast<int>((box.box_y1 + box.box_y2) / 2)   // 中心Y
        );

        const int point_radius = 3;       // 点的半径（像素）
        const cv::Scalar corner_color = cv::Scalar(0, 0, 255);  // 角点颜色：红色（BGR）
        const cv::Scalar center_color = cv::Scalar(255, 0, 0);  // 中心点颜色：蓝色（BGR）

        // 检查四个角点的颜色
        for (const auto& corner : corners) {
            // 避免像素坐标越界（防止框超出图像范围）
            if (corner.x >= 0 && corner.x < image.cols && corner.y >= 0 && corner.y < image.rows) {
                cv::Vec3b pixel = image.at<cv::Vec3b>(corner.y, corner.x);  // BGR格式取像素
                    std::cout << " | BGR: (" << static_cast<int>(pixel[0]) << ", " 
                << static_cast<int>(pixel[1]) << ", " 
                << static_cast<int>(pixel[2]) << ")" << std::endl;
                if (isPixelTargetColor(pixel, TargetColor::Green)) {
                    std::cout << ", green";
                    greenCount++;
                } else if (isPixelTargetColor(pixel, TargetColor::Yellow)) {
                    yellowCount++;
                    std::cout << ", yellow";
                }
                else if (isPixelTargetColor(pixel, TargetColor::other)) {
                    othercount++;
                    std::cout << ", other";
                }
            }
        }
        int center_count = 0; 
        // 4. 检查中心点的颜色
        if (center.x >= 0 && center.x < image.cols && center.y >= 0 && center.y < image.rows) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(center.y, center.x);
                                std::cout << " | BGR: (" << static_cast<int>(pixel[0]) << ", " 
                << static_cast<int>(pixel[1]) << ", " 
                << static_cast<int>(pixel[2]) << ")" << std::endl;
            if (isPixelTargetColor(pixel, TargetColor::Green)) {
                center_count++;
                //std::cout << ", green";
            } else if (isPixelTargetColor(pixel, TargetColor::Yellow)) {
                center_count++;
                //std::cout << ", yellow";
            }else if (isPixelTargetColor(pixel, TargetColor::other)) {
                    center_count++;
                    //std::cout << ", other";
                }
        }

        //绘制四个角点（红色实心圆）
        for (const auto& corner : corners) {
            // 越界保护：避免点超出图像范围
            if (corner.x >= 0 && corner.x < image.cols && corner.y >= 0 && corner.y < image.rows) {
                cv::circle(image, corner, point_radius, corner_color, cv::FILLED);  // FILLED表示实心
            }
        }

        // 绘制中心点（蓝色实心圆）
        if (center.x >= 0 && center.x < image.cols && center.y >= 0 && center.y < image.rows) {
            cv::circle(image, center, point_radius, center_color, cv::FILLED);
        }
        //5. 筛选：绿色/黄色点总数≤1则保留
        //std::cout << "center_count: "<< center_count << "" << <<std::endl;
        if (greenCount + yellowCount <= 2 && center_count <=2) {
            filteredBoxes.push_back(box);
        }
    }

    return filteredBoxes;
}


std::vector<G::idx_box> filterEmptyBoxes(
    const cv::Mat& image,
    const std::vector<G::idx_box>& candidateBoxes,
    float emptyThreshold = 0.08f,  // 第一次检查的阈值
    float center_rate = 0.3f,               // 指定中间区域的范围
    float centerThreshoud = 0.15f,           // 对中间区域进行二次检查的阈值
    float reduce_iamge_Threshold = 0.9f  // 指定图像的roi向内缩进10%
) {
    std::vector<G::idx_box> nonEmptyBoxes;
    if (image.empty()) {
        return nonEmptyBoxes;};

    // 1. 将图像转为HSV空间（更鲁棒的颜色分割）
    cv::Mat hsvImage;
    cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);

    // 2. 定义目标颜色的HSV范围
    // 注：OpenCV的HSV范围是 H:0-180（对应0-360°）、S:0-255、V:0-255
    std::vector<std::pair<cv::Scalar, cv::Scalar>> colorRanges = {
        {cv::Scalar(115, 0, 0), cv::Scalar(160, 255, 255)},  
        //{cv::Scalar(15, 100, 100), cv::Scalar(30, 255, 255)}  
    };

    for (const auto& box : candidateBoxes) {
        // 3. 提取ROI（边界保护，避免越界）
        cv::Mat roi = hsvImage(cv::Rect(box.box_x1 + (box.box_x2 - box.box_x1)*(1 - reduce_iamge_Threshold)/2, 
                                        box.box_y1+(box.box_y2 - box.box_y1)*(1 - reduce_iamge_Threshold)/2, 
                                        (box.box_x2 - box.box_x1) * reduce_iamge_Threshold, 
                                        (box.box_y2 - box.box_y1) * reduce_iamge_Threshold));
        cv::Mat center_roi = hsvImage(cv::Rect(box.box_x1 + (box.box_x2 - box.box_x1)*(1 - center_rate)/2, 
                                            box.box_y1+(box.box_y2 - box.box_y1)*(1 - center_rate)/2, 
                                            (box.box_x2 - box.box_x1) * center_rate, 
                                            (box.box_y2 - box.box_y1) * center_rate
        ));                                
        int totalPixels = roi.rows * roi.cols;  // ROI总像素数
        int center_Pixels = center_roi.rows * center_roi.cols;

        // 4. 生成目标颜色的合并掩码
        cv::Mat colorMask = cv::Mat::zeros(roi.size(), CV_8UC1);
        for (const auto& range : colorRanges) {
            cv::Mat rangeMask;
            cv::inRange(roi, range.first, range.second, rangeMask);
            colorMask |= rangeMask;  
        }
        cv::Mat center_colorMask = cv::Mat::zeros(center_roi.size(), CV_8UC1);
        for (const auto& range : colorRanges) {
            cv::Mat center_rangeMask;
            cv::inRange(center_roi, range.first, range.second, center_rangeMask);
            center_colorMask |= center_rangeMask;  
        }        

        // 5. 统计目标颜色像素数
        int targetPixels = cv::countNonZero(colorMask);
        float targetRatio = static_cast<float>(targetPixels) / totalPixels;
        int center_targetPixels = cv::countNonZero(center_colorMask);
        float center_targetRatio = static_cast<float>(center_targetPixels) / center_Pixels;
        std::cout <<"idx: " << box.idx;
        std::cout << ", total_pixels: " << totalPixels;
        std::cout <<", targetPixels" << targetPixels;
        std::cout <<", targetRatio: " << targetRatio<< std::endl;
        std::cout << "  center_Pixels: " << center_Pixels;
        std::cout <<", center_targetPixels" << center_targetPixels;
        std::cout <<", center_targetRatio: " << center_targetRatio<< std::endl;
        // 6. 判断是否非空：目标像素占比超过阈值
        if (targetRatio > emptyThreshold && center_targetRatio > centerThreshoud) {
            nonEmptyBoxes.push_back(box);
        }
    }
    return nonEmptyBoxes;
}

std::vector<G::idx_box> modifyBoxesByColorBoundary(
    cv::Mat& image,
    std::vector<G::idx_box>& boxes,
    const std::vector<TargetColor>& targetColors  // 颜色集合（如 {Green, Yellow, Other}）
) {

    std::vector<G::idx_box> modifiedBoxes;
    const int min_boundary_pixels = 3; // 边界像素最小数量阈值

    for (auto& box : boxes) {
        std::vector<cv::Point> targetBoundaryPixels; // 存储所有目标颜色的边界像素

        // ---------- 遍历左边界（x=box_x1，y从box_y1到box_y2） ----------
        int original_y2 = box.box_y2;  // 保存原始box_y2，用于初始循环范围
        bool found = false;            // 标记是否找到目标颜色像素
        for (int y = box.box_y1 + 6; y <= box.box_y2 -8; ++y) {
            cv::Point p(box.box_x2 - 8, y);
            if (p.x >= 0 && p.x < image.cols && p.y >= 0 && p.y < image.rows) {
                cv::Vec3b pixel = image.at<cv::Vec3b>(p.y, p.x);
                // 检查像素是否属于任意目标颜色
                for (const auto& color : targetColors) {
                    if (isPixelTargetColor(pixel, color)) {
                        box.box_y2 = p.y;  // 将box_y2设为当前像素的y坐标
                        found = true;      // 标记已找到目标
                        cv::Point point(box.box_x2,box.box_y2);
                        cv::circle(image, point, 3, cv::Scalar(0, 255, 0), -1);
                        break;             
                    }
                }
            }
                if (found) {
                    break;  // 跳出“左边界y循环”
                }
        }
        

        // ---------- 遍历右边界（x=box_x2，y从box_y1到box_y2） ----------
        // found = false;  
        // for (int y = box.box_y1 + 6; y <= original_y2 - 15; ++y) {
        //     cv::Point p(box.box_x2 -5, y);
        //     if (p.x >= 0 && p.x < image.cols && p.y >= 0 && p.y < image.rows) {
        //         cv::Vec3b pixel = image.at<cv::Vec3b>(p.y, p.x);
        //         for (const auto& color : targetColors) {
        //             if (isPixelTargetColor(pixel, color)) {
        //                 box.box_y3 = p.y;  // 将box_y2设为当前像素的y坐标
        //                 found = true;      // 标记已找到目标
        //                 cv::Point point(box.box_x1,box.box_y2);
        //                 cv::circle(image, point, 3, cv::Scalar(0, 255, 0), -1);
        //                 break;
        //             }
        //         }
        //     }
        //         if (found) {
        //             break;  // 跳出“左边界y循环”
        //         }            
        // }

        // ---------- 遍历上边界（y=box_y1，x从box_x1到box_x2） ----------
        // for (int x = box.box_x1; x <= box.box_x2; ++x) {
        //     cv::Point p(x, box.box_y1);
        //     if (p.x >= 0 && p.x < image.cols && p.y >= 0 && p.y < image.rows) {
        //         cv::Vec3b pixel = image.at<cv::Vec3b>(p.y, p.x);
        //         for (const auto& color : targetColors) {
        //             if (isPixelTargetColor(pixel, color)) {
        //                 targetBoundaryPixels.emplace_back(p);
        //                 break;
        //             }
        //         }
        //     }
        // }

        // // ---------- 遍历下边界（y=box_y2，x从box_x1到box_x2） ----------
        // for (int x = box.box_x1; x <= box.box_x2; ++x) {
        //     cv::Point p(x, box.box_y2);
        //     if (p.x >= 0 && p.x < image.cols && p.y >= 0 && p.y < image.rows) {
        //         cv::Vec3b pixel = image.at<cv::Vec3b>(p.y, p.x);
        //         for (const auto& color : targetColors) {
        //             if (isPixelTargetColor(pixel, color)) {
        //                 targetBoundaryPixels.emplace_back(p);
        //                 break;
        //             }
        //         }
        //     }
        // }

        // 若边界像素不足，保留原框
        if (targetBoundaryPixels.size() < min_boundary_pixels) {
            modifiedBoxes.push_back(box);
            continue;
        }

        // 计算边界像素的最小/最大坐标，生成新框
        int min_x = targetBoundaryPixels[0].x;
        int max_x = targetBoundaryPixels[0].x;
        int min_y = targetBoundaryPixels[0].y;
        int max_y = targetBoundaryPixels[0].y;

        for (const auto& p : targetBoundaryPixels) {
            min_x = std::min(min_x, p.x);
            max_x = std::max(max_x, p.x);
            min_y = std::min(min_y, p.y);
            max_y = std::max(max_y, p.y);
        }

        // 生成新框（复制原框idx，更新坐标）
        G::idx_box adjustedBox;
        adjustedBox.idx = box.idx;
        adjustedBox.box_x1 = min_x;
        adjustedBox.box_y1 = min_y;
        adjustedBox.box_x2 = max_x;
        adjustedBox.box_y2 = max_y;
        adjustedBox.distance = box.distance; // 同步距离（若结构体包含该成员）

        modifiedBoxes.push_back(adjustedBox);
    }

    return modifiedBoxes;
}


std::vector<G::idx_box> modifyBoxesByColorBoundary_(
    cv::Mat& image,
    std::vector<G::idx_box>& boxes,
    int window_size = 5,       // 滑动窗口大小，用于平滑噪声
    float gradient_thresh = 30.0f,  // 基础梯度阈值
    float ratio_thresh = 2.5f,      // 梯度比率阈值（当前梯度/窗口平均梯度）
    int min_stable_len = 3,         // 突变后稳定区域最小长度
    int line_count = 3              // 每条边界验证的线数量
) {
    std::vector<G::idx_box> modifiedBoxes;
    if (image.empty() || boxes.empty()) return modifiedBoxes;

    // 计算像素差异的辅助函数（BGR三通道欧式距离）
    auto pixelDiff = [](const cv::Vec3b& a, const cv::Vec3b& b) {
        int db = a[0] - b[0];
        int dg = a[1] - b[1];
        int dr = a[2] - b[2];
        return std::sqrt(db*db + dg*dg + dr*dr);  // 三通道综合差异
    };

    // 对每条边界线检测突变点的函数
    auto detectMutations = [&](const std::vector<cv::Point>& line_pixels) -> std::vector<int> {
        std::vector<int> mutation_indices;
        if (line_pixels.size() < window_size * 2) return mutation_indices;

        // 1. 计算相邻像素梯度
        std::vector<float> gradients;
        for (size_t i = 1; i < line_pixels.size(); ++i) {
            const auto& p1 = line_pixels[i-1];
            const auto& p2 = line_pixels[i];
            cv::Vec3b pix1 = image.at<cv::Vec3b>(p1.y, p1.x);
            cv::Vec3b pix2 = image.at<cv::Vec3b>(p2.y, p2.x);
            gradients.push_back(pixelDiff(pix1, pix2));
        }

        // 2. 滑动窗口计算局部平均梯度（平滑噪声）
        std::vector<float> window_avg(gradients.size(), 0.0f);
        for (size_t i = 0; i < gradients.size(); ++i) {
            int start = std::max(0, (int)i - window_size/2);
            int end = std::min((int)gradients.size()-1, (int)i + window_size/2);
            float sum = 0.0f;
            for (int j = start; j <= end; ++j) sum += gradients[j];
            window_avg[i] = sum / (end - start + 1);
        }

        // 3. 检测梯度突变点（当前梯度远大于局部平均）
        for (size_t i = 0; i < gradients.size(); ++i) {
            if (window_avg[i] < 1e-3) continue;  // 避免除以0
            // 梯度绝对值和相对比率都超过阈值
            if (gradients[i] > gradient_thresh && 
                gradients[i] / window_avg[i] > ratio_thresh) {

                // 4. 验证突变后是否有稳定区域（排除孤立噪声）
                bool has_stable = false;
                int stable_count = 0;
                for (size_t j = i+1; j < std::min(i+min_stable_len+1, gradients.size()); ++j) {
                    if (gradients[j] < window_avg[j] * 1.2f) {  // 后续梯度回归平稳
                        stable_count++;
                    }
                }
                if (stable_count >= min_stable_len) {
                    mutation_indices.push_back(i);  // 记录突变点索引
                }
            }
        }
        return mutation_indices;
    };

    for (auto& box : boxes) {
        // 边界框坐标取整并做越界保护
        int x1 = cvRound(std::max(0.0f, box.box_x1));
        int y1 = cvRound(std::max(0.0f, box.box_y1));
        int x2 = cvRound(std::min((float)image.cols-1, box.box_x2));
        int y2 = cvRound(std::min((float)image.rows-1, box.box_y2));
        if (x1 >= x2 || y1 >= y2) {  // 无效框直接保留
            modifiedBoxes.push_back(box);
            continue;
        }

        // 收集右边界的多条验证线（垂直方向均匀分布line_count条线）
        std::vector<std::vector<cv::Point>> boundary_lines;
        int line_step = std::max(1, (x2 - x1) / (line_count + 1));  // 线之间的水平间隔
        for (int k = 0; k < line_count; ++k) {
            int line_x = x2 - k * line_step;  // 右边界向左偏移的多条线
            line_x = std::max(x1, std::min(x2, line_x));  // 确保在线框内
            // 生成这条线上的所有像素点（垂直方向）
            std::vector<cv::Point> line_pixels;
            int y_step = std::max(1, (y2 - y1) / 50);  // 每50个像素取一个点（减少计算量）
            for (int y = y1 + 5; y <= y2 - 5; y += y_step) {  // 避开边界边缘噪声
                line_pixels.emplace_back(line_x, y);
            }
            if (!line_pixels.empty()) {
                boundary_lines.push_back(line_pixels);
            }
        }

        // 检测所有线的突变点，并收集可能的y坐标
        std::vector<int> candidate_ys;
        for (const auto& line : boundary_lines) {
            auto mutations = detectMutations(line);
            for (int idx : mutations) {
                if (idx + 1 < line.size()) {  // 突变点对应的像素y坐标
                    candidate_ys.push_back(line[idx+1].y);  // 取突变后第一个像素
                }
            }
        }

        // 对候选y坐标进行聚类（取最密集的区域）
        int best_y = y2;  // 默认保留原边界
        if (!candidate_ys.empty()) {
            std::sort(candidate_ys.begin(), candidate_ys.end());
            // 找出现次数最多的y值附近区域（聚类）
            int max_count = 1;
            int current_y = candidate_ys[0];
            int current_count = 1;
            for (size_t i = 1; i < candidate_ys.size(); ++i) {
                if (candidate_ys[i] - current_y < 5) {  // 允许±4像素的误差
                    current_count++;
                    if (current_count > max_count) {
                        max_count = current_count;
                        best_y = (current_y + candidate_ys[i]) / 2;  // 取平均
                    }
                } else {
                    current_y = candidate_ys[i];
                    current_count = 1;
                }
            }
            // 确保突变点在合理范围内（不超出原框太多）
            best_y = std::max(y1 + 10, std::min(y2 - 5, best_y));
        }

        // 绘制突变点标记
        cv::circle(image, cv::Point(x2, best_y), 4, cv::Scalar(0, 255, 0), -1);
        cv::line(image, cv::Point(x2-10, best_y), cv::Point(x2+10, best_y), cv::Scalar(0, 255, 0), 2);

        // 更新边界框
        box.box_y2 = best_y;
        modifiedBoxes.push_back(box);
    }

    return modifiedBoxes;
}

Eigen::Matrix3f createRotationMatrix(float rx, float ry, float rz) {
    // 转换为弧度
    // rx = rx * M_PI / 180.0f; // Roll (绕X轴)
    // ry = ry * M_PI / 180.0f; // Pitch (绕Y轴)
    // rz = rz * M_PI / 180.0f; // Yaw (绕Z轴)
    // 创建绕各轴的旋转矩阵
    Eigen::Matrix3f R_x;
    R_x << 1, 0, 0,
           0, cos(rx), -sin(rx),
           0, sin(rx), cos(rx);
    Eigen::Matrix3f R_y;
    R_y << cos(ry), 0, sin(ry),
           0, 1, 0,
           -sin(ry), 0, cos(ry);
    Eigen::Matrix3f R_z;
    R_z << cos(rz), -sin(rz), 0,
           sin(rz), cos(rz), 0,
           0, 0, 1;
    // 组合旋转矩阵 (Z-Y-X顺序: R = R_z * R_y * R_x)
    return R_z * R_y * R_x;
}

Eigen::Vector3f createTranslationVector(float tx, float ty, float tz) {
    Eigen::Vector3f translation(tx, ty, tz);
    return translation;
}

// 分离和组合现有旋转矩阵与平移向量
Eigen::Matrix4f combineRotationAndTranslation(const Eigen::Matrix3f& rotation, const Eigen::Vector3f& translation) {
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block<3, 3>(0, 0) = rotation;
    transform.block<3, 1>(0, 3) = translation;
    return transform;
}

// 回调函数：处理接收到的 TF 消息
void tfCallback(const tf2_msgs::TFMessage::ConstPtr& msg)
{
    for (const auto& transform : msg->transforms)
    {
        if(transform.header.frame_id != "odom")
        {
            std::cout<< transform.header.frame_id<<std::endl;
            continue;
        }
        // 获取四元数
        geometry_msgs::Quaternion quat = transform.transform.rotation;

        // 使用 tf2 库将四元数转换为欧拉角
        tf2::Quaternion tf_quat;
        tf_quat.setX(quat.x);
        tf_quat.setY(quat.y);
        tf_quat.setZ(quat.z);
        tf_quat.setW(quat.w);

        // 转换为旋转矩阵，然后提取欧拉角
        tf2::Matrix3x3 mat(tf_quat);
        double roll, pitch, yaw;
        mat.getRPY(roll, pitch, yaw);  // 得到的是弧度值

        cv::Mat R = (cv::Mat_<double>(3,3) <<
        cos(yaw), 0, sin(yaw),
        0, 1, 0,
        -sin(yaw), 0, cos(yaw));
        cv::Mat R_inv;
        cv::Mat C_world = (cv::Mat_<double>(3,1) << -transform.transform.translation.y, 0.0, transform.transform.translation.x);
        cv::invert(R, R_inv);
        {
            std::lock_guard<std::mutex> lock(global._mtx_tf);
            cv::Rodrigues(R, global._rvec);
            global._tvec = - (R * C_world);
        }

        std::cout<< "yaw: "<< yaw <<std::endl;
        // double yaw_deg = yaw * 180.0 / M_PI;
        // float _x  = transform.transform.translation.x - 0.28*cos(yaw) + 0.28;
        // float _y = transform.transform.translation.y - 0.28*sin(yaw);

    }
}

void odomCallback(const nav_msgs::Odometry::ConstPtr& msg)
{
    // ROS_INFO_STREAM("odom");
    // ROS_INFO_STREAM("x=" << msg->pose.pose.position.x
    //                << "y=" << msg->pose.pose.position.y
    //                << "z=" << msg->pose.pose.position.z);
    // ROS_INFO_STREAM("x=" << msg->pose.pose.orientation.x
    //                << "y=" << msg->pose.pose.orientation.y
    //                << "z=" << msg->pose.pose.orientation.z
    //                << "w=" << msg->pose.pose.orientation.w);
    global.x = msg->pose.pose.position.x;
    global.y = msg->pose.pose.position.y;
    global.z = msg->pose.pose.position.z;    
    // 获取四元数
    geometry_msgs::Quaternion quat = msg->pose.pose.orientation;

    // 使用 tf2 库将四元数转换为欧拉角
    tf2::Quaternion tf_quat;
    tf_quat.setX(quat.x);
    tf_quat.setY(quat.y);
    tf_quat.setZ(quat.z);
    tf_quat.setW(quat.w);

    // 转换为旋转矩阵，然后提取欧拉角
    tf2::Matrix3x3 mat(tf_quat);
    double roll, pitch, yaw;
    mat.getRPY(roll, pitch, yaw);  // 得到的是弧度值


    cv::Mat R = (cv::Mat_<double>(3,3) <<
        cos(yaw-M_PI/2), 0, sin(yaw-M_PI/2),
        0, 1, 0,
        -sin(yaw-M_PI/2), 0, cos(yaw-M_PI/2));
    cv::Mat R_inv;
    cv::Mat C_world = (cv::Mat_<double>(3,1) << msg->pose.pose.position.x, 0.0, msg->pose.pose.position.y);
    cv::invert(R, R_inv);
    {
        std::lock_guard<std::mutex> lock(global._mtx_tf);
        cv::Rodrigues(R, global._rvec);
        global._tvec = - (R * C_world);
    }

    std::cout<< "yaw: "<< yaw <<std::endl;
}

float calculateIoU(
    float box1_x1, float box1_y1, float box1_x2, float box1_y2,
    float box2_x1, float box2_y1, float box2_x2, float box2_y2) 
{

    float inter_x1 = std::max(box1_x1, box2_x1);
    float inter_y1 = std::max(box1_y1, box2_y1);
    float inter_x2 = std::min(box1_x2, box2_x2);
    float inter_y2 = std::min(box1_y2, box2_y2);


    // 计算交集面积（无交集时为0）
    if (inter_x2 <= inter_x1 || inter_y2 <= inter_y1) {
        return 0.0f;
    }
    float inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1);


    // 计算两个框的面积

    float box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1);
    float box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1);

    // 步骤4：计算并集面积（避免除以0）
    float union_area = box1_area + box2_area - inter_area;
    if (union_area <= 0.0f) return 0.0f; // 极端情况：两框面积均为0

    // 步骤5：返回IoU
    return inter_area / union_area;
}

std::vector<G::idx_box> filterBoxesFromBack(std::vector<G::idx_box>& boxes, float threshold) {
    std::vector<G::idx_box> result;
    
    // 1. 倒序遍历候选框（从最后一个到第一个）
    for (auto it = boxes.rbegin(); it != boxes.rend(); ++it) {
        const auto& current = *it;
        bool shouldKeep = true;
        
        // 2. 检查当前框与已保留框的冲突
        for (auto rit = result.begin(); rit != result.end();) {
            float iou = calculateIoU(
                current.box_x1, current.box_y1, current.box_x2, current.box_y2,
                rit->box_x1, rit->box_y1, rit->box_x2, rit->box_y2
            );
            
            if (iou > threshold) {
                // IOU超阈值：保留距离小的，剔除大的
                if (current.distance > rit->distance) {
                    shouldKeep = false; // 当前框更远，跳过
                    break;
                } else {
                    rit = result.erase(rit); // 已保留的框更远，剔除它
                }
            } else {
                ++rit; // 无冲突，继续检查下一个已保留框
            }
        }
        
        // 3. 保留当前框（无冲突或冲突后仍需保留）
        if (shouldKeep) {
            result.push_back(current);
        }
    }
    
    // 4. 反转结果，恢复原收集顺序（可选：若需保持原顺序）
    std::reverse(result.begin(), result.end());
    
    return result;
}

// std::vector<G::idx_box> filterBoxes(const std::vector<G::idx_box>& boxes, float threshold) {
//     std::vector<G::idx_box> result;
//     for (const auto& current : boxes) {
//         bool shouldKeep = true;
//         // 遍历已保留的框，检查冲突
//         for (auto it = result.begin(); it != result.end();) {
//             // 计算当前框与已保留框的IOU
//             float iou = calculateIoU(
//                 current.box_x1, current.box_y1, current.box_x2, current.box_y2,
//                 it->box_x1, it->box_y1, it->box_x2, it->box_y2
//             );
//             if (iou > threshold) {
//                 // IOU超阈值：保留距离小的，剔除大的
//                 if (current.distance > it->distance) {
//                     shouldKeep = false; // 当前框更远，跳过
//                     break;
//                 } else {
//                     it = result.erase(it); // 已保留的框更远，剔除它
//                 }
//             } else {
//                 ++it; // 无冲突，继续检查下一个
//             }
//         }
//         if (shouldKeep) {
//             result.push_back(current); // 保留当前框
//         }
//     }
//     return result;
// }

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    try
    {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        cv::Mat image_ =  cv_ptr->image;
        global.imagePoints;
        global.side_imagePoints;

        // 投影3D点到2D
        {
            std::lock_guard<std::mutex> lock(global._mtx_tf);
            cv::projectPoints(global._c_objectPoints, global._rvec, global._tvec, global._K, global._distCoeffs, global.imagePoints);
        }
        // {
        //     std::lock_guard<std::mutex> lock(global._mtx_tf);
        //     cv::projectPoints(global._c_side_objectPoints, global._rvec, global._tvec, global._K, global._distCoeffs, global.side_imagePoints);
        // }        

        const float IOU_THRESHOLD = 0.02f; // IOU阈值
        global.idx_box_lists.clear();
        global.idx_box_side_lists.clear();
        std::vector<int> exclude_idxs = {};
        // 1. 收集画面内的框
        for (int i = 0; i < global.imagePoints.size(); i += 4) {                       
            cv::Point3f world_point = global._c_objectPoints[i];
            float dist = std::sqrt(
                world_point.x * world_point.x +
                world_point.y * world_point.y +
                world_point.z * world_point.z
            );
            
            // 检查框的对角点是否在画面内
            bool inFrame = 
                (global.imagePoints[i].x > -60 && global.imagePoints[i].x < image_.cols + 60 && 
                 global.imagePoints[i].y > -60 && global.imagePoints[i].y < image_.rows + 60) &&
                (global.imagePoints[i+2].x > -60 && global.imagePoints[i+2].x < image_.cols + 60 && 
                 global.imagePoints[i+2].y > -60 && global.imagePoints[i+2].y < image_.rows + 60);

            if (inFrame) {
                if (std::find(exclude_idxs.begin(), exclude_idxs.end(), i/4+1) != exclude_idxs.end()) {
                    continue; // 跳过排除的idx
                    }
            
                if (global.imagePoints[i].x < 0){global.imagePoints[i].x = 0;};
                if (global.imagePoints[i].x > image_.cols){global.imagePoints[i].x = image_.cols;}; 
                if (global.imagePoints[i].y < 0){global.imagePoints[i].y = 0;};
                if (global.imagePoints[i].y > image_.rows){global.imagePoints[i].y = image_.rows;}; 
                if (global.imagePoints[i+2].x < 0){global.imagePoints[i+2].x = 0;};
                if (global.imagePoints[i+2].x > image_.cols){global.imagePoints[i+2].x = image_.cols;}; 
                if (global.imagePoints[i+2].y < 0){global.imagePoints[i+2].y = 0;};
                if (global.imagePoints[i+2].y > image_.rows){global.imagePoints[i+2].y = image_.rows;}; 

                global.idx_box_lists.push_back({
                    i/4 + 1,
                    global.imagePoints[i].x,
                    global.imagePoints[i].y,
                    global.imagePoints[i+2].x,
                    global.imagePoints[i+2].y,
                    dist
                });
            }
        }     

        // 2. 筛选正面的框（去除IOU大且远的）
        //global.idx_box_lists = global.filterBoxesByProjected3DXY(global.idx_box_lists);
        global.idx_box_lists = filterBoxesFromBack(global.idx_box_lists , IOU_THRESHOLD);

        global.idx_box_lists = filterEmptyBoxes(image_,global.idx_box_lists);


        //global.idx_box_lists = filterBoxesByColor(image_, global.idx_box_lists);
        //global.idx_box_lists= modifyBoxesByColorBoundary(image_, global.idx_box_lists, {TargetColor::Green, TargetColor::Yellow, TargetColor::other});
        //std::vector<G::idx_box> filtered_boxes = filterBoxesFromBack(global.idx_box_lists , IOU_THRESHOLD);
        
        // 3. 绘制筛选后的框
        for (const auto& box : global.idx_box_lists) {
            
            // 绘制文本（组号+距离）
            // std::string text = "idx:" + std::to_string(box.idx) + " d:" + std::to_string(box.distance).substr(0,4);
            std::string text = "idx:" + std::to_string(box.idx);
            cv::putText(image_, text, 
                        cv::Point(cvRound((box.box_x1 + box.box_x2)/2), cvRound((box.box_y1 + box.box_y2)/2)),
                        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
            cv::line(image_, cv::Point(cvRound(box.box_x1), cvRound(box.box_y1)), cv::Point(cvRound(box.box_x2), cvRound(box.box_y1)), cv::Scalar(0,0,255),2);
            cv::line(image_, cv::Point(cvRound(box.box_x2), cvRound(box.box_y1)), cv::Point(cvRound(box.box_x2), cvRound(box.box_y2)), cv::Scalar(0,0,255),2);
            cv::line(image_, cv::Point(cvRound(box.box_x2), cvRound(box.box_y2)), cv::Point(cvRound(box.box_x1), cvRound(box.box_y2)), cv::Scalar(0,0,255),2);
            cv::line(image_, cv::Point(cvRound(box.box_x1), cvRound(box.box_y2)), cv::Point(cvRound(box.box_x1), cvRound(box.box_y1)), cv::Scalar(0,0,255),2);
            // cv::line(image_, cv::Point(cvRound(box.box_x1-3), cvRound(box.box_y1-5)), cv::Point(cvRound(box.box_x2+5), cvRound(box.box_y1-5)), cv::Scalar(0,0,255),2);
            // cv::line(image_, cv::Point(cvRound(box.box_x2+5), cvRound(box.box_y1-5)), cv::Point(cvRound(box.box_x2+5), cvRound(box.box_y2-3)), cv::Scalar(0,0,255),2);
            // cv::line(image_, cv::Point(cvRound(box.box_x2+5), cvRound(box.box_y2-3)), cv::Point(cvRound(box.box_x1-3), cvRound(box.box_y2-3)), cv::Scalar(0,0,255),2);
            // cv::line(image_, cv::Point(cvRound(box.box_x1-3), cvRound(box.box_y2-3)), cv::Point(cvRound(box.box_x1-3), cvRound(box.box_y1-5)), cv::Scalar(0,0,255),2);
        }

        // 更新全局图像
        {
            std::lock_guard<std::mutex> lock(global._mtx_image);
            image_.copyTo(global._image);
        }
        
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
}

void worker_task1(ros::NodeHandle nh)
{
    ros::Rate sl(50);
    //ros::Subscriber tf_sub = nh.subscribe("/tf", 2, tfCallback);
    ros::Subscriber tf_sub = nh.subscribe("/robot_pose", 2, odomCallback);
    while (ros::ok())
    {
        ros::spinOnce();
        sl.sleep();
    }
}

void worker_task2(ros::NodeHandle nh)
{
    ros::Rate sl(10);
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber image_sub = it.subscribe("/kinect2/hd/image_color_rect", 2, imageCallback);
    while (ros::ok())
    {
        /* code */
        
        ros::spinOnce();
        sl.sleep();
    }
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "t2dto3d_ld_node");
    ros::NodeHandle nh;
    std::vector<std::thread> workers;
    workers.emplace_back(worker_task1, nh);
    workers.emplace_back(worker_task2, nh);
    image_transport::ImageTransport it(nh);
    image_transport::Publisher pub = it.advertise("pub_image_topic", 2);

    ros::Rate rate(10);
    while(ros::ok())
    {
        sensor_msgs::ImagePtr msg;
        {
            std::lock_guard<std::mutex> lock(global._mtx_image);
            msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", global._image).toImageMsg();
        }
        pub.publish(msg);
        std::cout << "publish success" << std::endl;
        rate.sleep();
    }
    return 0;
}