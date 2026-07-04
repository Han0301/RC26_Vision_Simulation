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

struct surface_2d_point_ {        
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
    std::vector<std::vector<surface_2d_point_>> object_2d_points_lists,
    std::vector<std::vector<surface_2d_point_>> plum_2d_points_lists,
    cv::Mat& iamge);

/**
 * @brief 静止检测，静止时某些数据不更新，提高效率
 * @param xyzrpy 机器人自身的xyz, roll,pitch,yaw的坐标信息
 */
void update_is_move(XYZRPY xyzrpy);

// 辅助函数1： 取到 is_move 
bool get_is_move(){
    return is_move;
}

// 辅助函数2： 取到 object_zbuffer 矩阵
cv::Mat get_object_zbuffer(){
    if (zbuffer.empty()){
        ROS_WARN("zbuffer is empty!!!");
    }else{return zbuffer;}
}

// 辅助函数3： 取到 idx_to_depth_ 映射关系
std::unordered_map<int, std::vector<float>>get_idx_to_depth_(){
    if (idx_to_depth_.empty()){ROS_WARN("idx_to_depth_ is empty!!!");
    }else{return idx_to_depth_;}
}



private:

XYZRPY last_xyzrpy;     // 用于 static_detection 的上一次的x,y,z,r,p,y值
bool is_move;           // 为true 表示移动， 为false 表示静止

cv::Mat zbuffer;            // zbuffer 包含台阶，方块的深度矩阵, 用于综合判断遮挡情况
cv::Mat object_zbuffer;     // object_zbuffer 只存储方块的深度矩阵， 用作输出
cv::Mat last_zbuffer;       // last_zbuffer 上一次的 object_zbuffer 深度矩阵
std::unordered_map<int, std::vector<float>> idx_to_depth_;      // idx_to_depth_ 表示下标到 存储对应方块深度值的容器 的映射关系， 在 zbuffer 填充的时候顺便填入
};

zbuffer_occlsion::zbuffer_occlsion(){
    // 1. 初始化 用于 static_detection 的上一次的x,y,z,r,p,y值
    last_xyzrpy._xyz._x = DBL_MAX;
    last_xyzrpy._xyz._y = DBL_MAX;
    last_xyzrpy._xyz._z = DBL_MAX;
    last_xyzrpy._rpy._roll = DBL_MAX;
    last_xyzrpy._rpy._pitch = DBL_MAX;
    last_xyzrpy._rpy._yaw = DBL_MAX;
    is_move = true;

}

void zbuffer_occlsion::update_zbuffer(
    std::vector<std::vector<surface_2d_point_>> object_2d_points_lists,
    std::vector<std::vector<surface_2d_point_>> plum_2d_points_lists,
    cv::Mat& image){

    // 1. 准备工作
    // 1.1 静止检测，若静止，沿用上一次的 zbuffer 矩阵
    if (!get_is_move() && !last_zbuffer.empty()){
        return;
    }
    
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
    std::vector<surface_2d_point_> object_front_2d = object_2d_points_lists[0];
    std::vector<surface_2d_point_> object_side_2d = object_2d_points_lists[1];
    std::vector<surface_2d_point_> object_up_2d = object_2d_points_lists[2];
    std::vector<surface_2d_point_> plum_front_2d = plum_2d_points_lists[0];
    std::vector<surface_2d_point_> plum_side_2d = plum_2d_points_lists[1];
    std::vector<surface_2d_point_> plum_up_2d = plum_2d_points_lists[2];
    if (!(object_front_2d.size() == 12 && object_side_2d.size() == 12 && object_up_2d.size() == 12 && plum_front_2d.size() == 12 && plum_side_2d.size() == 12 && plum_up_2d.size() == 12)){
        ROS_WARN("in surface_2d_point_, the size is not 12!!!");
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

void zbuffer_occlsion::update_is_move(XYZRPY xyzrpy){
    // 1. 分别判断位置，位姿是否变化
    bool pos_changed = ((std::abs(xyzrpy._xyz._x - last_xyzrpy._xyz._x)) > pos_eps || (std::abs(xyzrpy._xyz._y - last_xyzrpy._xyz._y) > pos_eps) || (std::abs(xyzrpy._xyz._z - last_xyzrpy._xyz._z) > pos_eps));
    bool rot_changed = (std::abs(xyzrpy._rpy._roll - last_xyzrpy._rpy._roll) > rot_eps) || (std::abs(xyzrpy._rpy._pitch - last_xyzrpy._rpy._pitch) > rot_eps) || (std::abs(xyzrpy._rpy._yaw - last_xyzrpy._rpy._yaw) > rot_eps);
    if (pos_changed || rot_changed) {
        is_move = true;
    }else{is_move = false;}

    // 2. 更新上一次的状态
    last_xyzrpy._xyz._x = xyzrpy._xyz._x;
    last_xyzrpy._xyz._y = xyzrpy._xyz._y;
    last_xyzrpy._xyz._z = xyzrpy._xyz._z;
    last_xyzrpy._rpy._roll = xyzrpy._rpy._roll;
    last_xyzrpy._rpy._pitch = xyzrpy._rpy._pitch;
    last_xyzrpy._rpy._yaw = xyzrpy._rpy._yaw;
}

}   // namespace Ten
#endif