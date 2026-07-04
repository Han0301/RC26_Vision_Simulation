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
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/features2d.hpp>
#include <mutex>
#include <chrono>
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
        // 1. 相机内参矩阵 K
        _K = (cv::Mat_<double>(3,3) <<
            1012.0711525658555, 0, 960.5,
            0, 1012.0711525658555, 540.5,
            0, 0, 1);
        // 2. 畸变系数（假设零畸变）
        _distCoeffs = cv::Mat::zeros(5, 1, CV_64F);

        // 3. 外参：旋转向量和平移向量
        _rvec = (cv::Mat_<double>(3,1) << 0.0, 0.0, 0.0); // 旋转（弧度）
        _tvec = (cv::Mat_<double>(3,1) << 0.0, 0.0, 0.0);   // 平移（米）

        // 4. 定义3D点的世界坐标
        for(int j = 0; j < 4; j++)
        {
            for(int i = 0; i < 3; i++)
            {
                objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_ + _lh_ + 0.1, _Y_ + i*_L_ + _ly1_, _arr[j*3+i]+_lh_));
                objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_ + _lh_ + 0.1, _Y_ + i*_L_ + _ly2_, _arr[j*3+i]+_lh_));
                objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_ + _lh_ + 0.1, _Y_ + i*_L_ + _ly2_, _arr[j*3+i]));
                objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_ + _lh_ + 0.1, _Y_ + i*_L_ + _ly1_, _arr[j*3+i]));

                objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_ + 0.1, _Y_ + i*_L_ + _ly1_, _arr[j*3+i]+_lh_));                
                objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_ + 0.1, _Y_ + i*_L_ + _ly2_, _arr[j*3+i]+_lh_));                
                objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_ + 0.1, _Y_ + i*_L_ + _ly2_, _arr[j*3+i]));
                objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_ + 0.1, _Y_ + i*_L_ + _ly1_, _arr[j*3+i]));
            }
        }        
// ------------------------------------------------------------------------------------------------------------------------
// 雷达坐标系下各个3d点坐标
        for(int j = 0; j < 4; j++)
        {
            for(int i = 0; i < 3; i++)
            {
                lidar_objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_ + _lh_ + 0.1, _Y_ + i*_L_ + _ly1_, _arr[j*3+i]+_lh_));
                lidar_objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_ + _lh_ + 0.1, _Y_ + i*_L_ + _ly2_, _arr[j*3+i]+_lh_));
                lidar_objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_ + _lh_ + 0.1, _Y_ + i*_L_ + _ly2_, _arr[j*3+i]));
                lidar_objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_ + _lh_ + 0.1, _Y_ + i*_L_ + _ly1_, _arr[j*3+i]));

                lidar_objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_ + 0.1, _Y_ + i*_L_ + _ly1_, _arr[j*3+i]+_lh_));                
                lidar_objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_ + 0.1, _Y_ + i*_L_ + _ly2_, _arr[j*3+i]+_lh_));                
                lidar_objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_ + 0.1, _Y_ + i*_L_ + _ly2_, _arr[j*3+i]));
                lidar_objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_ + 0.1, _Y_ + i*_L_ + _ly1_, _arr[j*3+i]));
            }
        }  

        for(int i = 0; i < lidar_objectPoints.size(); i++)
        {
            lidar_objectPoints[i].x = lidar_objectPoints[i].x - x;
            lidar_objectPoints[i].y = lidar_objectPoints[i].y - y;
            lidar_objectPoints[i].z = lidar_objectPoints[i].z - z;
            std::cout << "lidar_objectPoints[" << i << "]: " << lidar_objectPoints[i] << std::endl;
        }



// ------------------------------------------------------------------------------------------------------------------------
        for(size_t i = 0; i < objectPoints.size(); i += 8)
        {
            if(i + 7 >= objectPoints.size()) break;

            // 计算8个点形成的立方体3D空间范围（x,y,z的min和max）
            float x_min = objectPoints[i].x, x_max = objectPoints[i].x;
            float y_min = objectPoints[i].y, y_max = objectPoints[i].y;
            float z_min = objectPoints[i].z, z_max = objectPoints[i].z;

            for(int k = 1; k < 8; k++){
                x_min = std::min(x_min, objectPoints[i+k].x);
                x_max = std::max(x_max, objectPoints[i+k].x);
                y_min = std::min(y_min, objectPoints[i+k].y);
                y_max = std::max(y_max, objectPoints[i+k].y);
                z_min = std::min(z_min, objectPoints[i+k].z);
                z_max = std::max(z_max, objectPoints[i+k].z);
            }

            // 添加到idx_f_b_box_list
            // std::cout << "will push_back: " << (int)(i/8) + 1 << "  x: " << x_min <<" " << x_max << " y: " << y_min << " " << y_max << " z: " << z_min << " " << z_max << std::endl;
            boxes.push_back({
                (int)(i/8) + 1,  // idx从1开始
                x_min, x_max,    // 3D X范围
                y_min, y_max,    // 3D Y范围
                z_min, z_max,     // 3D Z范围
                false,false
            });
        }
    }

    struct box{
        int idx;                      
        float x_min, x_max;    
        float y_min, y_max;     
        float z_min, z_max;   // 方块自身的世界坐标范围
        float r_x_min,r_x_max,r_y_min,r_y_max,r_z_min,r_z_max;      // 相对机器人的x,y坐标 
        float average_x,average_y, average_z;   // 方块的平均坐标
        float front_depth,left_depth,right_depth,side_depth,up_depth;
    };
    struct surface_2d_point{        // 表示方块表面的结构体
        int idx;
        cv::Point2f left_up;
        cv::Point2f right_up;
        cv::Point2f right_down;
        cv::Point2f left_down;
        float surface_depth;
    };

    float x;
    float y;
    float z;
    double yaw;
    float last_camera_x = INFINITY;
    float last_camera_y = INFINITY;
    double last_yaw = INFINITY;
    cv::Mat last_zbuffer;
    std::mutex mtx_last_zbuffer;

    std::vector<struct box> boxes; 
    std::vector<struct surface_2d_point> side_2d_points_lists;
    std::vector<struct surface_2d_point> front_2d_point_lists;
    std::vector<struct surface_2d_point> up_2d_point_lists;

    std::vector<cv::Point3f> objectPoints;
    std::vector<cv::Point3f> lidar_objectPoints;
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
    image_transport::Publisher zbuffer_pub;
}global;

// ---------------------------------------------------------------------------------------------------------------------------------------------- zbufer 相关调试可视化 ---------------
cv::Mat saveZBufferVisualization(const cv::Mat& zbuffer) {
    // 1. 克隆矩阵（避免修改原数据）
    cv::Mat zbuffer_processed = zbuffer.clone();

    // 2. 处理无效值（FLT_MAX）并计算有效深度的最大最小值
    cv::Mat mask = zbuffer_processed != FLT_MAX; // 有效像素掩码（排除无效值）
    double min_val, max_val;
    cv::minMaxLoc(zbuffer_processed, &min_val, &max_val, nullptr, nullptr, mask);

    // 处理全是无效值的极端情况
    if (mask.empty() || cv::countNonZero(mask) == 0) {
        std::cerr << "[ERROR] Z-Buffer中无有效深度值" << std::endl;
    }

    // 将无效值设为0（后续反转后仍为黑色）
    zbuffer_processed.setTo(0, ~mask);

    // 3. 归一化深度值到[0, 255]：原min_val→0，原max_val→255
    cv::Mat zbuffer_vis;
    cv::normalize(zbuffer_processed, zbuffer_vis, 10, 255, cv::NORM_MINMAX, CV_8U, mask);

    // 原min_val（近）→255（白），原max_val（远）→0（黑）
    zbuffer_vis = 255 - zbuffer_vis;

    // 确保无效值仍为黑色（反转后可能被影响，重新覆盖）
    zbuffer_vis.setTo(0, ~mask);

    // 4. 保存图片到指定路径
    return zbuffer_vis;
}



std::vector<std::vector<G::surface_2d_point>>zbuffer_initialization(        // 用于box世界坐标角点到各个面的像素点坐标的初始化
    std::vector<G::box> boxes,
    std::vector<int>allow_idx_lists,
    const float camera_y,
    const float camera_x,     // 机器人的x,y与这相反
    const float camera_z
){
    std::vector<G::surface_2d_point> front_2d_points_lists;
    std::vector<G::surface_2d_point> up_2d_points_lists;
    std::vector<G::surface_2d_point> side_2d_points_lists;
    std::vector<std::vector<G::surface_2d_point>> box_2d_points_lists;

    for (auto& box : boxes) {
    box.r_x_min = std::abs(box.x_min - camera_x);
    box.r_x_max = std::abs(box.x_max - camera_x);
    box.r_y_min = std::abs(box.y_min - camera_y);
    box.r_y_max = std::abs(box.y_max - camera_y);
    box.r_z_min = std::abs(box.z_min - camera_z);
    box.r_z_max = std::abs(box.z_max - camera_z);
    box.average_x = (box.r_x_max + box.r_x_min ) / 2.0f;
    box.average_y = (box.r_y_max + box.r_y_min ) / 2.0f;
    box.average_z = (box.r_z_min + box.r_z_max) / 2.0f;
    box.front_depth = sqrt(box.r_x_min * box.r_x_min + box.average_y * box.average_y + box.average_z * box.average_z);
    box.up_depth = sqrt(box.average_x * box.average_x + box.average_y * box.average_y + box.r_z_max * box.r_z_max);
    box.left_depth = sqrt(box.average_x * box.average_x + box.r_y_min * box.r_y_min + box.average_z * box.average_z);
    box.right_depth = sqrt(box.average_x * box.average_x + box.r_y_max * box.r_y_max + box.average_z * box.average_z);

    // std::cout << "=== 调试信息 - 方块 " << box.idx << " ===" << std::endl;
    // std::cout << "机器人位置: (" << robot_x << ", " << robot_y << ")" << std::endl;
    // std::cout << "方块坐标: x[" << box.x_min << ", " << box.x_max << "], "
    //         << "y[" << box.y_min << ", " << box.y_max << "], "
    //         << "z[" << box.z_min << ", " << box.z_max << "]" << std::endl;
    // std::cout << "相对机器人的坐标: r_x[" << box.r_x_min << ", " << box.r_x_max << "], "
    //         << "r_y[" << box.r_y_min << ", " << box.r_y_max << "]" << std::endl;
    // std::cout << "平均值坐标: (" << box.average_x << ", " << box.average_y << ", " << box.average_z << ")" << std::endl;
    // std::cout << "各方向深度:" << std::endl;
    // std::cout << "  前向深度: " << box.front_depth << std::endl;
    // std::cout << "  左侧深度: " << box.left_depth << std::endl;
    // std::cout << "  右侧深度: " << box.right_depth << std::endl;
    // std::cout << "  顶部深度: " << box.up_depth << std::endl;
    // std::cout << std::endl;

    bool bool_in_allow_idx_lists = false;
    if(std::find(allow_idx_lists.begin(),allow_idx_lists.end(),box.idx) != allow_idx_lists.end()){
        bool_in_allow_idx_lists = true;
    }

    std::vector<cv::Point2f>front_2d_points;
    std::vector<cv::Point3f>front_3d_points;
    front_2d_points.clear();
    front_3d_points.clear();
    front_3d_points.push_back(cv::Point3f(box.y_min,box.r_z_max,box.x_min));
    front_3d_points.push_back(cv::Point3f(box.y_max,box.r_z_max,box.x_min));
    front_3d_points.push_back(cv::Point3f(box.y_max,box.r_z_min,box.x_min));
    front_3d_points.push_back(cv::Point3f(box.y_min,box.r_z_min,box.x_min));
    {
        std::lock_guard<std::mutex> lock(global._mtx_tf); // 保护旋转和平移向量
        cv::projectPoints(front_3d_points, global._rvec, global._tvec, global._K, global._distCoeffs, front_2d_points);
    }
    if (bool_in_allow_idx_lists){
        front_2d_points_lists.push_back({box.idx, cv::Point2f(front_2d_points[0]),cv::Point2f(front_2d_points[1]),cv::Point2f(front_2d_points[2]),cv::Point2f(front_2d_points[3]),box.front_depth});
    };

    std::vector<cv::Point2f>up_2d_points;
    std::vector<cv::Point3f>up_3d_points;
    up_3d_points.clear();
    up_2d_points.clear();
    up_3d_points.push_back(cv::Point3f(box.y_min,box.r_z_max,box.x_max));
    up_3d_points.push_back(cv::Point3f(box.y_max,box.r_z_max,box.x_max));
    up_3d_points.push_back(cv::Point3f(box.y_max,box.r_z_max,box.x_min));
    up_3d_points.push_back(cv::Point3f(box.y_min,box.r_z_max,box.x_min));
    {
        std::lock_guard<std::mutex> lock(global._mtx_tf); // 保护旋转和平移向量
        cv::projectPoints(up_3d_points, global._rvec, global._tvec, global._K, global._distCoeffs, up_2d_points);
    }
    if (bool_in_allow_idx_lists){
        up_2d_points_lists.push_back({box.idx, cv::Point2f(up_2d_points[0]),cv::Point2f(up_2d_points[1]),cv::Point2f(up_2d_points[2]),cv::Point2f(up_2d_points[3]),box.up_depth});
    }

    // 2.判断方块自身侧面的遮挡情况
    if (box.left_depth < box.right_depth){  
        std::vector<cv::Point2f>side_2d_points;
        std::vector<cv::Point3f>side_3d_points;
        side_3d_points.clear();
        side_2d_points.clear();
        side_3d_points.push_back(cv::Point3f(box.y_min,box.r_z_max,box.x_max));
        side_3d_points.push_back(cv::Point3f(box.y_min,box.r_z_max,box.x_min));
        side_3d_points.push_back(cv::Point3f(box.y_min,box.r_z_min,box.x_min));
        side_3d_points.push_back(cv::Point3f(box.y_min,box.r_z_min,box.x_max));
        
        {
            std::lock_guard<std::mutex> lock(global._mtx_tf); // 保护旋转和平移向量
            cv::projectPoints(side_3d_points, global._rvec, global._tvec, global._K, global._distCoeffs, side_2d_points);
        }
        //std::cout << "left: side_2d_points: "<< side_2d_points << std::endl;  
        if (bool_in_allow_idx_lists){
            side_2d_points_lists.push_back({box.idx, cv::Point2f(side_2d_points[0]),cv::Point2f(side_2d_points[1]),cv::Point2f(side_2d_points[2]),cv::Point2f(side_2d_points[3]),box.left_depth});
            }
    }else{
        std::vector<cv::Point2f>side_2d_points;
        std::vector<cv::Point3f>side_3d_points;
        side_3d_points.clear();
        side_2d_points.clear();
        side_3d_points.push_back(cv::Point3f(box.y_max,box.r_z_max,box.x_min));
        side_3d_points.push_back(cv::Point3f(box.y_max,box.r_z_max,box.x_max));
        side_3d_points.push_back(cv::Point3f(box.y_max,box.r_z_min,box.x_max));
        side_3d_points.push_back(cv::Point3f(box.y_max,box.r_z_min,box.x_min));
        
        {
            std::lock_guard<std::mutex> lock(global._mtx_tf); // 保护旋转和平移向量
            cv::projectPoints(side_3d_points, global._rvec, global._tvec, global._K, global._distCoeffs, side_2d_points);
        }
        //std::cout << "right: side_2d_points: "<< side_2d_points << std::endl;
        if (bool_in_allow_idx_lists){
            side_2d_points_lists.push_back({box.idx, cv::Point2f(side_2d_points[0]),cv::Point2f(side_2d_points[1]),cv::Point2f(side_2d_points[2]),cv::Point2f(side_2d_points[3]), box.right_depth});
            }
    };      // 判断方块自身的遮挡情况
}
    box_2d_points_lists.push_back(front_2d_points_lists);
    box_2d_points_lists.push_back(side_2d_points_lists);
    box_2d_points_lists.push_back(up_2d_points_lists);
    return box_2d_points_lists;
};



void draw_zbuffer(cv::Mat& image, const cv::Mat& zbuffer, float depthThreshold = 0.05f) {
    if (zbuffer.empty()) {
        ROS_WARN("Z-buffer is empty, skipping draw_zbuffer.");
        return;
    }
    if (image.empty()) {
        ROS_WARN("Image is empty, skipping draw_zbuffer.");
        return;
    }
    // 1. 预处理：过滤无效深度值（FLT_MAX）并转换为可处理格式
    cv::Mat validDepthMask = zbuffer != FLT_MAX; // 有效深度掩码
    cv::Mat depth8U;
    double minVal, maxVal;
    // 使用 minMaxLoc 函数查找有效深度值中的最小值和最大值
    // validDepthMask 作为掩码，确保只考虑有效区域
    cv::minMaxLoc(zbuffer, &minVal, &maxVal, nullptr, nullptr, validDepthMask);
    zbuffer.convertTo(depth8U, CV_8U, 255.0 / (static_cast<float>(maxVal) - static_cast<float>(minVal)), -255.0 * static_cast<float>(minVal) / (static_cast<float>(maxVal) - static_cast<float>(minVal)));
    depth8U.setTo(0, ~validDepthMask); // 无效区域设为0

    // 2. 遍历所有可能的深度值，提取连通区域
    std::unordered_map<int, std::vector<cv::Point>> depthRegions; // 深度值（量化后）-> 像素点集
    for (int y = 0; y < zbuffer.rows; ++y) {
        for (int x = 0; x < zbuffer.cols; ++x) {
            float depth = zbuffer.at<float>(y, x);
            if (depth == FLT_MAX) continue; // 跳过无效值

            // 量化深度值（按阈值分组，避免浮点精度问题）
            int depthGroup = static_cast<int>(depth / depthThreshold);
            depthRegions[depthGroup].emplace_back(x, y);
        }
    }

    // 3. 对每个深度区域提取轮廓并拟合多边形
    cv::Mat regionMask = cv::Mat::zeros(zbuffer.size(), CV_8U);
    for (const auto& [group, points] : depthRegions) {
        if (points.size() < 4) continue; // 过滤过小区域

        // 生成区域掩码
        regionMask.setTo(0);
        for (const auto& p : points) {
            regionMask.at<uchar>(p) = 255;
        }

        // 提取轮廓
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(regionMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // 拟合多边形并绘制
        for (const auto& contour : contours) {
            if (contour.size() < 4) continue;

            // 多边形逼近（保留原始形状，epsilon根据轮廓大小动态调整）
            double epsilon = 0.01 * cv::arcLength(contour, true);
            std::vector<cv::Point> polygon;
            cv::approxPolyDP(contour, polygon, epsilon, true);

            // 绘制多边形（闭合区域，半透明填充+红色边框）
            cv::polylines(image, polygon, true, cv::Scalar(0, 0, 255), 2); // 红色边框
        }
    }

}



cv::Mat zbuffer_occlusion(
    std::vector<std::vector<G::surface_2d_point>> box_2d_points_lists,
    const float camera_y,
    const float camera_x,     // 机器人的x,y与这相反
    const float camera_z,
    const double yaw,
    const cv::Mat& image
){
    // 函数内部的局部变量
    const int image_width = image.cols;
    const int image_height = image.rows;
    std::vector<G::surface_2d_point> front_2d_points_lists = box_2d_points_lists[0];
    std::vector<G::surface_2d_point> side_2d_points_lists = box_2d_points_lists[1];
    std::vector<G::surface_2d_point> up_2d_points_lists = box_2d_points_lists[2];

    // 调试打印开始
    // ROS_INFO("=== 2D Points Initialization Debug ===");
    // ROS_INFO("Front Points (%zu):", front_2d_points_lists.size());
    // for (const auto& fp : front_2d_points_lists) {
    //     ROS_INFO("  Idx: %d | Depth: %.2f | Corners: [%.2f, %.2f] [%.2f, %.2f] [%.2f, %.2f] [%.2f, %.2f]",
    //             fp.idx, fp.surface_depth,
    //             fp.left_up.x, fp.left_up.y,
    //             fp.right_up.x, fp.right_up.y,
    //             fp.right_down.x, fp.right_down.y,
    //             fp.left_down.x, fp.left_down.y);
    // }

    // ROS_INFO("Side Points (%zu):", side_2d_points_lists.size());
    // for (const auto& sp : side_2d_points_lists) {
    //     ROS_INFO("  Idx: %d | Depth: %.2f | Corners: [%.2f, %.2f] [%.2f, %.2f] [%.2f, %.2f] [%.2f, %.2f]",
    //             sp.idx, sp.surface_depth,
    //             sp.left_up.x, sp.left_up.y,
    //             sp.right_up.x, sp.right_up.y,
    //             sp.right_down.x, sp.right_down.y,
    //             sp.left_down.x, sp.left_down.y);
    // }

    // ROS_INFO("Up Points (%zu):", up_2d_points_lists.size());
    // for (const auto& up : up_2d_points_lists) {
    //     ROS_INFO("  Idx: %d | Depth: %.2f | Corners: [%.2f, %.2f] [%.2f, %.2f] [%.2f, %.2f] [%.2f, %.2f]",
    //             up.idx, up.surface_depth,
    //             up.left_up.x, up.left_up.y,
    //             up.right_up.x, up.right_up.y,
    //             up.right_down.x, up.right_down.y,
    //             up.left_down.x, up.left_down.y);
    // }

    // -----------------------------------------------------------1.画面静止时不更新
    const float eps = 0.001f;  
    const double eps_rot = 0.001;
    bool pos_unchanged = (std::abs(camera_x - global.last_camera_x) < eps && std::abs(camera_y - global.last_camera_y) < eps && std::abs(camera_y - global.last_camera_y) < eps);
    bool yaw_unchanges = (std::abs(yaw - global.last_yaw) < eps_rot);
    if (pos_unchanged && yaw_unchanges) {
        if (!global.last_zbuffer.empty()) {
            ROS_INFO("not move or rotation");
            return global.last_zbuffer.clone();  // 返回上一次的zbuffer
        }
    }
    
    // 更新上一次坐标为当前坐标
    global.last_camera_x = camera_x;
    global.last_camera_y = camera_y;
    global.last_yaw = yaw;

    if (image_width <= 0 || image_height <= 0) {
    std::cerr << "错误:图像尺寸无效(width=" << image_width << ", height=" << image_height << ")" << std::endl;
    return global.last_zbuffer.clone();  // 终止函数，避免创建空矩阵
    }

    //  -----------------------------------------------------------2：计算每个方块的相对坐标和各个面的平均深度


    // 3.初始化Z-buffer（深度缓冲，存储每个像素的最小深度值）
    cv::Mat zbuffer = cv::Mat::ones(image_height, image_width, CV_32F) * FLT_MAX; // 创建一个​​全1矩阵， 初始化为无穷大
    std::cout << "size: " << front_2d_points_lists.size() << "  " << side_2d_points_lists.size() <<  "  " << up_2d_points_lists.size() << std::endl;

    // 5.填入zbuffer
    for (int i = 0; i < side_2d_points_lists.size(); i++) {
        auto& front_point = front_2d_points_lists[i];
        auto& side_point = side_2d_points_lists[i];
        auto& up_point = up_2d_points_lists[i];

        std::vector<cv::Point> front_contour = {
        cv::Point(cvRound(front_point.left_up.x), cvRound(front_point.left_up.y)),
        cv::Point(cvRound(front_point.right_up.x), cvRound(front_point.right_up.y)),
        cv::Point(cvRound(front_point.right_down.x), cvRound(front_point.right_down.y)),
        cv::Point(cvRound(front_point.left_down.x), cvRound(front_point.left_down.y))
        };
        std::vector<cv::Point> up_contour = {
            cv::Point(cvRound(up_point.left_up.x), cvRound(up_point.left_up.y)),
            cv::Point(cvRound(up_point.right_up.x), cvRound(up_point.right_up.y)),
            cv::Point(cvRound(up_point.right_down.x), cvRound(up_point.right_down.y)),
            cv::Point(cvRound(up_point.left_down.x), cvRound(up_point.left_down.y))
        };
        std::vector<cv::Point> side_contour = {
            cv::Point(cvRound(side_point.left_up.x), cvRound(side_point.left_up.y)),
            cv::Point(cvRound(side_point.right_up.x), cvRound(side_point.right_up.y)),
            cv::Point(cvRound(side_point.right_down.x), cvRound(side_point.right_down.y)),
            cv::Point(cvRound(side_point.left_down.x), cvRound(side_point.left_down.y))
        };
        cv::Mat tmp = cv::Mat::ones(image_height, image_width, CV_32F) * FLT_MAX; 
        cv::fillPoly(tmp, std::vector<std::vector<cv::Point>>{front_contour}, cv::Scalar(front_point.surface_depth));
        cv::fillPoly(tmp, std::vector<std::vector<cv::Point>>{up_contour}, cv::Scalar(up_point.surface_depth));
        cv::fillPoly(tmp, std::vector<std::vector<cv::Point>>{side_contour}, cv::Scalar(side_point.surface_depth));

        std::vector<float>x_value = {front_point.left_up.x,front_point.right_up.x,front_point.right_down.x,front_point.left_down.x,
                                    up_point.left_up.x,up_point.right_up.x,up_point.right_down.x,up_point.left_down.x,
                                    side_point.left_up.x,side_point.right_up.x,side_point.right_down.x,side_point.left_down.x};
        std::vector<float>y_value = {front_point.left_up.y,front_point.right_up.y,front_point.right_down.y,front_point.left_down.y,
                                    up_point.left_up.y,up_point.right_up.y,up_point.right_down.y,up_point.left_down.y,
                                    side_point.left_up.y,side_point.right_up.y,side_point.right_down.y,side_point.left_down.y};

        auto x_max = std::max_element(x_value.begin(), x_value.end());
        auto x_min = std::min_element(x_value.begin(), x_value.end());
        auto y_max = std::max_element(y_value.begin(), y_value.end());
        auto y_min = std::min_element(y_value.begin(), y_value.end());

        for (int row = int(*y_min) - 5; row < int(*y_max) + 5; ++row) {       
            for (int col = int(*x_min) - 5; col < int(*x_max) + 5; ++col) {  
                // 读取当前像素值（zbuffer是单通道CV_32F，用at<float>访问）
                float pixel_value = zbuffer.at<float>(row, col);
                if(zbuffer.at<float>(row, col) > tmp.at<float>(row, col))
                {
                    zbuffer.at<float>(row, col) = tmp.at<float>(row, col);
                }
            }
        }
    }

    // 调试保存zbuffer
    //std::string save_path = "/home/h/RC2026/world_ws5/src/3dto2d/zbuffer_.png"; // 替换为你的保存路径
    // bool success = saveZBufferVisualization(zbuffer, save_path);
    // if (!success) {
    //     ROS_WARN("Z-Buffer可视化保存失败");
    // }
    // 调试发布zbuffer到ROS话题
    cv::Mat zbuffer_vis = saveZBufferVisualization(zbuffer);
    if (!zbuffer_vis.empty()) {
        // 转换OpenCV图像为ROS消息
        std_msgs::Header header;
        header.stamp = ros::Time::now(); // 设置时间戳
        header.frame_id = "camera_frame"; // 根据实际情况修改坐标系ID
        
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(
            header, 
            "mono8",  // 单通道8位图像编码
            zbuffer_vis
        ).toImageMsg();
        
        // 发布图像
        global.zbuffer_pub.publish(msg);
        //ROS_INFO("Z-Buffer可视化图像已发布到话题");
    } else {
        //ROS_WARN("Z-Buffer可视化图像为空，无法发布");
    }

    // 6.保存当前zbuffer为last_zbuffer
    {
        std::lock_guard<std::mutex> lock(global.mtx_last_zbuffer);
        zbuffer.copyTo(global.last_zbuffer); // 使用copyTo确保数据被正确复制
    }
    return zbuffer;
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
    std::cout << "x: " << global.x << " y: " << global.y << " z: " << global.z <<std::endl;
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
    double roll, pitch;
    mat.getRPY(roll, pitch, global.yaw);  // 得到的是弧度值


    cv::Mat R = (cv::Mat_<double>(3,3) <<
        cos(global.yaw-M_PI/2), 0, sin(global.yaw-M_PI/2),
        0, 1, 0,
        -sin(global.yaw-M_PI/2), 0, cos(global.yaw-M_PI/2));
    cv::Mat R_inv;
    cv::Mat C_world = (cv::Mat_<double>(3,1) << msg->pose.pose.position.x, 0.0, msg->pose.pose.position.y);
    cv::invert(R, R_inv);
    {
        std::lock_guard<std::mutex> lock(global._mtx_tf);
        cv::Rodrigues(R, global._rvec);
        global._tvec = - (R * C_world);
    }

    std::cout<< "yaw: "<< global.yaw <<std::endl;
}

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    try
    {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        cv::Mat image_ =  cv_ptr->image;
        global.imagePoints;
        global.side_imagePoints;

        const float IOU_THRESHOLD = 0.02f; // IOU阈值
        global.front_2d_point_lists.clear();
        global.side_2d_points_lists.clear(); 
        std::vector<int> exclude_idxs = {};
        // 1. 收集画面内的框
        const int image_width = image_.cols;
        const int iamge_height = image_.rows;

        // 2. 筛选框
        std::vector<int> allow_idx_lists = {1,4,7,8,9,10,11,12};
        std::vector<std::vector<G::surface_2d_point>> box_2d_points_lists = zbuffer_initialization(        // 用于box世界坐标角点到各个面的像素点坐标的初始化
            global.boxes,
            allow_idx_lists,
            global.x,
            global.y,     // 机器人的x,y与这相反
            1.3);

        cv::Mat zbuffer = zbuffer_occlusion(box_2d_points_lists, global.x,global.y,1.3, global.yaw ,image_);
        draw_zbuffer(image_, zbuffer, 0.01f);
        
        // 3. 绘制筛选后的框
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
    ros::init(argc, argv, "zbuffer_orb_node");
    ros::NodeHandle nh;
    std::vector<std::thread> workers;
    workers.emplace_back(worker_task1, nh);
    workers.emplace_back(worker_task2, nh);
    image_transport::ImageTransport it(nh);
    image_transport::Publisher pub = it.advertise("pub_image_topic", 2);
    global.zbuffer_pub = it.advertise("/zbuffer_visualization", 30);

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