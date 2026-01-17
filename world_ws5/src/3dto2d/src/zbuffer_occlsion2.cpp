#ifndef _ZBUFFER_OCCLSION_H_
#define _ZBUFFER_OCCLSION_H_
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
#include <cfloat>
#include "method_math.cpp"

namespace Ten{

#define pos_eps 1e-3     // xyz的静止判断阈值
#define rot_eps 1e-3     // roll,pitch,yaw的静止判断阈值

struct surface_2d_point {        
    int idx;                       // 对应方块索引
    cv::Point2f left_up;           // 左上2D点
    cv::Point2f right_up;          // 右上2D点
    cv::Point2f right_down;        // 右下2D点
    cv::Point2f left_down;         // 左下2D点
    float surface_depth;           // 该表面的深度值
};

class zbuffer_occlsion{
public:

// 构造函数
zbuffer_occlsion();

/**
 * @brief 更新zbuffer矩阵
 * @param box_2d_points_lists 方块的2d点容器
 * @param plum_2d_points_lists 台阶的2d点容器
 * @param xyzrpy 机器人自身的xyz, roll,pitch,yaw的坐标信息
 * @param iamge 实时的图像信息
 */
void update_zbuffer(
    std::vector<std::vector<surface_2d_point>> object_2d_points_lists,
    std::vector<std::vector<surface_2d_point>> plum_2d_points_lists);

// 辅助函数2： 取到 object_zbuffer 矩阵
cv::Mat get_object_zbuffer(){
    if (object_zbuffer.empty()){
        ROS_WARN("zbuffer is empty!!!");
    }else{return object_zbuffer;}
}

// 辅助函数3： 取到 idx_to_depth_ 映射关系



private:

cv::Mat zbuffer;            // zbuffer 包含台阶，方块的深度矩阵, 用于综合判断遮挡情况
};

void zbuffer_occlsion::update_zbuffer(
    std::vector<std::vector<surface_2d_point>> object_2d_points_lists,
    std::vector<std::vector<surface_2d_point>> plum_2d_points_lists,
    cv::Mat& image){

    // 1. 准备工作
    // 1.1 静止检测，若静止，沿用上一次的 zbuffer 矩阵

    // 1.2 更新 last_zbuffer 
    if (!zbuffer.empty()){
        last_zbuffer = zbuffer.clone();
    }
    
    // 1.3 检查图像有效性
    if (image.empty() || image.cols <= 0 || image.rows <= 0) {
        ROS_WARN("Invalid image size: cols=%d, rows=%d", image.cols, image.rows);
        return;
    }

    // 1.4 清空 idx_to_depth_ 的缓存
    idx_to_depth_.clear();

    // 2. 填充 zbuffer, object_zbuffer 矩阵
    // 2.1 初始化深度缓冲（初始值为最大浮点数，表示无深度）
    zbuffer = cv::Mat::ones(image.rows, image.cols, CV_32F) * FLT_MAX;
    object_zbuffer = cv::Mat::ones(image.rows, image.cols, CV_32F) * FLT_MAX;

    // 2.2 提取2d点坐标
    std::vector<surface_2d_point> object_front_2d = object_2d_points_lists[0];
    std::vector<surface_2d_point> object_side_2d = object_2d_points_lists[1];
    std::vector<surface_2d_point> object_up_2d = object_2d_points_lists[2];
    std::vector<surface_2d_point> plum_front_2d = plum_2d_points_lists[0];
    std::vector<surface_2d_point> plum_side_2d = plum_2d_points_lists[1];
    std::vector<surface_2d_point> plum_up_2d = plum_2d_points_lists[2];
    if (!(object_front_2d.size() == 12 && object_side_2d.size() == 12 && object_up_2d.size() == 12 && plum_front_2d.size() == 12 && plum_side_2d.size() == 12 && plum_up_2d.size() == 12)){
        ROS_WARN("in surface_2d_point, the size is not 12!!!");
        return;
    }

    // 2.3 先填充台阶的深度
    for (size_t i = 0; i < plum_side_2d.size(); i++) {
        auto& p_front = plum_front_2d[i];
        auto& p_side = plum_side_2d[i];
        auto& p_up = plum_up_2d[i];

        // 2.3.1 收集台阶的所有2D点坐标，判断整个台阶的所有点是否都在图像外
        std::vector<cv::Point2f> all_points = {
            p_front.left_up, p_front.right_up, p_front.right_down, p_front.left_down,
            p_side.left_up, p_side.right_up, p_side.right_down, p_side.left_down,
            p_up.left_up, p_up.right_up, p_up.right_down, p_up.left_down

        };
        bool all_outside = true;
        for (const auto& pt : all_points) {
            if (pt.x >= 0 && pt.x < image.cols && pt.y >= 0 && pt.y < image.rows) {
                all_outside = false;
                break;
            }
        }
        if (all_outside) continue; 
        
        // 2.3.2 构建台阶轮廓
        std::vector<cv::Point> front_contour = {
            cv::Point(cvRound(p_front.left_up.x), cvRound(p_front.left_up.y)),
            cv::Point(cvRound(p_front.right_up.x), cvRound(p_front.right_up.y)),
            cv::Point(cvRound(p_front.right_down.x), cvRound(p_front.right_down.y)),
            cv::Point(cvRound(p_front.left_down.x), cvRound(p_front.left_down.y))
        };
        std::vector<cv::Point> side_contour = {
            cv::Point(cvRound(p_side.left_up.x), cvRound(p_side.left_up.y)),
            cv::Point(cvRound(p_side.right_up.x), cvRound(p_side.right_up.y)),
            cv::Point(cvRound(p_side.right_down.x), cvRound(p_side.right_down.y)),
            cv::Point(cvRound(p_side.left_down.x), cvRound(p_side.left_down.y))
        };
        std::vector<cv::Point> up_contour = {
            cv::Point(cvRound(p_up.left_up.x), cvRound(p_up.left_up.y)),
            cv::Point(cvRound(p_up.right_up.x), cvRound(p_up.right_up.y)),
            cv::Point(cvRound(p_up.right_down.x), cvRound(p_up.right_down.y)),
            cv::Point(cvRound(p_up.left_down.x), cvRound(p_up.left_down.y))
        };

        // 2.3.3 填充台阶深度到临时矩阵 plum_tmp
        cv::Mat plum_tmp = cv::Mat::ones(image.rows, image.cols, CV_32F) * FLT_MAX;
        cv::fillPoly(plum_tmp, std::vector<std::vector<cv::Point>>{front_contour}, cv::Scalar(p_front.surface_depth));
        cv::fillPoly(plum_tmp, std::vector<std::vector<cv::Point>>{side_contour}, cv::Scalar(p_side.surface_depth));
        cv::fillPoly(plum_tmp, std::vector<std::vector<cv::Point>>{up_contour}, cv::Scalar(p_up.surface_depth));

        // 2.3.4 计算台阶像素范围
        float plum_x_min = FLT_MAX,plum_y_min = FLT_MAX,plum_x_max = FLT_MIN,plum_y_max = FLT_MIN;
        for(const auto& p : all_points) {
            if(p.x > plum_x_max) plum_x_max = p.x;
            if(p.x < plum_x_min) plum_x_min = p.x;
            if(p.y > plum_y_max) plum_y_max = p.y;
            if(p.y < plum_y_min) plum_y_min = p.y;
        }

        // 2.3.5 写入主zbuffer（台阶深度更近则更新）
        for (int row = int(plum_y_min) - 1; row < int(plum_y_max) + 1; ++row) {
            for (int col = int(plum_x_min) - 1; col < int(plum_x_max) + 1; ++col) {
                if (row < 0 || row >= image.rows || col < 0 || col >= image.cols) continue;
                if (plum_tmp.at<float>(row, col) < zbuffer.at<float>(row, col)) {
                    zbuffer.at<float>(row, col) = plum_tmp.at<float>(row, col);
                }
            }
        }
    }

    // 2.4 再填充方块的深度
    for (size_t i = 0; i < object_side_2d.size(); i++) {
        if (i >= object_front_2d.size() || i >= object_up_2d.size()) break;

        auto& o_front = object_front_2d[i];
        auto& o_side = object_side_2d[i];
        auto& o_up = object_up_2d[i];

        // 2.4.1 收集方块所有2D点坐标， 并判断方块的所有点是否都在图像外
        std::vector<cv::Point2f> all_points = {
            o_front.left_up, o_front.right_up, o_front.right_down, o_front.left_down,
            o_side.left_up, o_side.right_up, o_side.right_down, o_side.left_down,
            o_up.left_up, o_up.right_up, o_up.right_down, o_up.left_down
        };
        bool all_outside = true;
        for (const auto& pt : all_points) {
            if (pt.x >= 0 && pt.x < image.cols && pt.y >= 0 && pt.y < image.rows) {
                all_outside = false;
                break;
            }
        }
        if (all_outside) continue; 

        // 2.4.2 构建方块轮廓
        std::vector<cv::Point> front_contour = {
            cv::Point(cvRound(o_front.left_up.x), cvRound(o_front.left_up.y)),
            cv::Point(cvRound(o_front.right_up.x), cvRound(o_front.right_up.y)),
            cv::Point(cvRound(o_front.right_down.x), cvRound(o_front.right_down.y)),
            cv::Point(cvRound(o_front.left_down.x), cvRound(o_front.left_down.y))
        };
        std::vector<cv::Point> up_contour = {
            cv::Point(cvRound(o_up.left_up.x), cvRound(o_up.left_up.y)),
            cv::Point(cvRound(o_up.right_up.x), cvRound(o_up.right_up.y)),
            cv::Point(cvRound(o_up.right_down.x), cvRound(o_up.right_down.y)),
            cv::Point(cvRound(o_up.left_down.x), cvRound(o_up.left_down.y))
        };
        std::vector<cv::Point> side_contour = {
            cv::Point(cvRound(o_side.left_up.x), cvRound(o_side.left_up.y)),
            cv::Point(cvRound(o_side.right_up.x), cvRound(o_side.right_up.y)),
            cv::Point(cvRound(o_side.right_down.x), cvRound(o_side.right_down.y)),
            cv::Point(cvRound(o_side.left_down.x), cvRound(o_side.left_down.y))
        };

        // 2.4.3 填充方块深度到临时矩阵 plum_tmp
        cv::Mat tmp = cv::Mat::ones(image.rows, image.cols, CV_32F) * FLT_MAX;
        cv::fillPoly(tmp, std::vector<std::vector<cv::Point>>{front_contour}, cv::Scalar(o_front.surface_depth));
        cv::fillPoly(tmp, std::vector<std::vector<cv::Point>>{up_contour}, cv::Scalar(o_up.surface_depth));
        cv::fillPoly(tmp, std::vector<std::vector<cv::Point>>{side_contour}, cv::Scalar(o_side.surface_depth));

        // 2.4.4 写入idx→深度映射
        idx_to_depth_[o_front.idx] = {o_front.surface_depth, o_up.surface_depth, o_side.surface_depth};

        // 2.4.5 计算方块像素范围
        float object_x_min = FLT_MAX,object_y_min = FLT_MAX,object_x_max = FLT_MIN,object_y_max = FLT_MIN;
        for(const auto& p : all_points) {
            if(p.x > object_x_max) object_x_max = p.x;
            if(p.x < object_x_min) object_x_min = p.x;
            if(p.y > object_y_max) object_y_max = p.y;
            if(p.y < object_y_min) object_y_min = p.y;
        }

        // 2.4.6 合并到方块深度缓冲 object_zbuffer + 全局深度缓冲 zbuffer
        for (int row = int(object_y_min) - 1; row < int(object_y_max) + 1; ++row) {
            for (int col = int(object_x_min) - 1; col < int(object_x_max) + 1; ++col) {
                if (row < 0 || row >= image.rows || col < 0 || col >= image.cols) continue;
                if (tmp.at<float>(row, col) < zbuffer.at<float>(row, col)) {
                    zbuffer.at<float>(row, col) = tmp.at<float>(row, col);
                    object_zbuffer.at<float>(row, col) = tmp.at<float>(row, col);
                }
            }
        }
    }    
}


}   // namespace Ten
#endif