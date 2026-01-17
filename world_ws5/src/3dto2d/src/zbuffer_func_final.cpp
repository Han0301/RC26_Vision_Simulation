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
#include <vector>
#include <numeric>
#include <unordered_set>

#define _L_ 1.2
#define _H_ 0.2
#define _ly1_ 0.425
#define _ly2_ 0.775
#define _lx1_ 0.425
#define _lh_ 0.35 
#define _X_  3.17             
#define _Y_  1.2            

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
                plum_blossom_points.push_back(cv::Point3f(_X_ + j*_L_, _Y_ + i*_L_, _arr[j*3+i] ));
                plum_blossom_points.push_back(cv::Point3f(_X_ + j*_L_, _Y_ + i*_L_ + _L_, _arr[j*3+i] ));
                plum_blossom_points.push_back(cv::Point3f(_X_ + j*_L_, _Y_ + i*_L_ + _L_, 0));
                plum_blossom_points.push_back(cv::Point3f(_X_ + j*_L_, _Y_ + i*_L_, 0));
                plum_blossom_points.push_back(cv::Point3f(_X_ + j*_L_ + _L_, _Y_ + i*_L_, _arr[j*3+i] ));
                plum_blossom_points.push_back(cv::Point3f(_X_ + j*_L_ + _L_, _Y_ + i*_L_ + _L_, _arr[j*3+i]));
                plum_blossom_points.push_back(cv::Point3f(_X_ + j*_L_ + _L_, _Y_ + i*_L_ + _L_, 0 ));
                plum_blossom_points.push_back(cv::Point3f(_X_ + j*_L_ + _L_, _Y_ + i*_L_, 0));
            }
        }    
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
            boxes.push_back({
                (int)(i/8) + 1,  // idx从1开始
                x_min, x_max,    // 3D X范围
                y_min, y_max,    // 3D Y范围
                z_min, z_max,     // 3D Z范围
                false,false
            });
        }
        for(size_t i = 0; i < plum_blossom_points.size(); i += 8)
        {
            if(i + 7 >= plum_blossom_points.size()) break;
            // 计算8个点形成的立方体3D空间范围（x,y,z的min和max）
            float x_min = plum_blossom_points[i].x, x_max = plum_blossom_points[i].x;
            float y_min = plum_blossom_points[i].y, y_max = plum_blossom_points[i].y;
            float z_min = plum_blossom_points[i].z, z_max = plum_blossom_points[i].z;
            for(int k = 1; k < 8; k++){
                x_min = std::min(x_min, plum_blossom_points[i+k].x);
                x_max = std::max(x_max, plum_blossom_points[i+k].x);
                y_min = std::min(y_min, plum_blossom_points[i+k].y);
                y_max = std::max(y_max, plum_blossom_points[i+k].y);
                z_min = std::min(z_min, plum_blossom_points[i+k].z);
                z_max = std::max(z_max, plum_blossom_points[i+k].z);
            }
            // 添加到idx_f_b_box_list
            plum_boxes.push_back({
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
    std::mutex video_mtx;  // 保护视频写入的互斥锁
    cv::Mat last_zbuffer;
    std::mutex mtx_last_zbuffer;
    std::vector<float> surface_depths;
    std::mutex mtx_surface_depths; 
    std::vector<struct box> boxes; 
    std::vector<struct box> plum_boxes;
    std::vector<cv::Point3f> objectPoints;
    std::vector<cv::Point3f> plum_blossom_points;
    std::vector<cv::Point3f> plum_blossom_points_lists;
    std::unordered_map<int,std::vector<float>> idx_to_depth;

    bool is_move = true;
    int k = 0;


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

std::vector<std::vector<G::surface_2d_point>>zbuffer_initialization(
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
    global.surface_depths.clear();

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

        bool bool_in_allow_idx_lists = false;
        if(std::find(allow_idx_lists.begin(),allow_idx_lists.end(),box.idx) != allow_idx_lists.end()){
            bool_in_allow_idx_lists = true;
        }

        // 前表面
        std::vector<cv::Point2f>front_2d_points;
        std::vector<cv::Point3f>front_3d_points;
        front_3d_points.push_back(cv::Point3f(box.y_min,box.r_z_max,box.x_min));
        front_3d_points.push_back(cv::Point3f(box.y_max,box.r_z_max,box.x_min));
        front_3d_points.push_back(cv::Point3f(box.y_max,box.r_z_min,box.x_min));
        front_3d_points.push_back(cv::Point3f(box.y_min,box.r_z_min,box.x_min));
        {
            std::lock_guard<std::mutex> lock(global._mtx_tf);
            cv::projectPoints(front_3d_points, global._rvec, global._tvec, global._K, global._distCoeffs, front_2d_points);
        }
        if (bool_in_allow_idx_lists){
            front_2d_points_lists.push_back({box.idx, cv::Point2f(front_2d_points[0]),cv::Point2f(front_2d_points[1]),cv::Point2f(front_2d_points[2]),cv::Point2f(front_2d_points[3]),box.front_depth});
            std::lock_guard<std::mutex> lock(global.mtx_surface_depths);
            global.surface_depths.push_back(box.front_depth);
        }

        // 上表面
        std::vector<cv::Point2f>up_2d_points;
        std::vector<cv::Point3f>up_3d_points;
        up_3d_points.push_back(cv::Point3f(box.y_min,box.r_z_max,box.x_max));
        up_3d_points.push_back(cv::Point3f(box.y_max,box.r_z_max,box.x_max));
        up_3d_points.push_back(cv::Point3f(box.y_max,box.r_z_max,box.x_min));
        up_3d_points.push_back(cv::Point3f(box.y_min,box.r_z_max,box.x_min));
        {
            std::lock_guard<std::mutex> lock(global._mtx_tf);
            cv::projectPoints(up_3d_points, global._rvec, global._tvec, global._K, global._distCoeffs, up_2d_points);
        }
        if (bool_in_allow_idx_lists){
            up_2d_points_lists.push_back({box.idx, cv::Point2f(up_2d_points[0]),cv::Point2f(up_2d_points[1]),cv::Point2f(up_2d_points[2]),cv::Point2f(up_2d_points[3]),box.up_depth});
            std::lock_guard<std::mutex> lock(global.mtx_surface_depths);
            global.surface_depths.push_back(box.up_depth);
        }

        // 侧表面（左/右）
        if (box.left_depth < box.right_depth){  
            std::vector<cv::Point2f>side_2d_points;
            std::vector<cv::Point3f>side_3d_points;
            side_3d_points.push_back(cv::Point3f(box.y_min,box.r_z_max,box.x_max));
            side_3d_points.push_back(cv::Point3f(box.y_min,box.r_z_max,box.x_min));
            side_3d_points.push_back(cv::Point3f(box.y_min,box.r_z_min,box.x_min));
            side_3d_points.push_back(cv::Point3f(box.y_min,box.r_z_min,box.x_max));
            {
                std::lock_guard<std::mutex> lock(global._mtx_tf);
                cv::projectPoints(side_3d_points, global._rvec, global._tvec, global._K, global._distCoeffs, side_2d_points);
            }
            if (bool_in_allow_idx_lists){
                side_2d_points_lists.push_back({box.idx, cv::Point2f(side_2d_points[0]),cv::Point2f(side_2d_points[1]),cv::Point2f(side_2d_points[2]),cv::Point2f(side_2d_points[3]),box.left_depth});
                std::lock_guard<std::mutex> lock(global.mtx_surface_depths);
                global.surface_depths.push_back(box.left_depth);
            }
        }else{
            std::vector<cv::Point2f>side_2d_points;
            std::vector<cv::Point3f>side_3d_points;
            side_3d_points.push_back(cv::Point3f(box.y_max,box.r_z_max,box.x_min));
            side_3d_points.push_back(cv::Point3f(box.y_max,box.r_z_max,box.x_max));
            side_3d_points.push_back(cv::Point3f(box.y_max,box.r_z_min,box.x_max));
            side_3d_points.push_back(cv::Point3f(box.y_max,box.r_z_min,box.x_min));
            {
                std::lock_guard<std::mutex> lock(global._mtx_tf);
                cv::projectPoints(side_3d_points, global._rvec, global._tvec, global._K, global._distCoeffs, side_2d_points);
            }
            if (bool_in_allow_idx_lists){
                side_2d_points_lists.push_back({box.idx, cv::Point2f(side_2d_points[0]),cv::Point2f(side_2d_points[1]),cv::Point2f(side_2d_points[2]),cv::Point2f(side_2d_points[3]), box.right_depth});
                std::lock_guard<std::mutex> lock(global.mtx_surface_depths);
                global.surface_depths.push_back(box.right_depth);
            }
        }
    }

    box_2d_points_lists.push_back(front_2d_points_lists);
    box_2d_points_lists.push_back(side_2d_points_lists);
    box_2d_points_lists.push_back(up_2d_points_lists);
    return box_2d_points_lists;
};


void draw_zbuffer(cv::Mat& image, 
                const cv::Mat& zbuffer, 
                 const std::vector<G::surface_2d_point>& front_2d,  // 新增参数：前面2D点列表
                 const std::vector<G::surface_2d_point>& side_2d,  // 新增参数：侧面2D点列表
                 const std::vector<G::surface_2d_point>& up_2d,    // 新增参数：上面2D点列表
                 float depthThreshold = 0.05f
) {
    if (zbuffer.empty()) {
        ROS_WARN("Z-buffer is empty, skipping draw_zbuffer.");
        return;
    }
    if (image.empty()) {
        ROS_WARN("Image is empty, skipping draw_zbuffer.");
        return;
    }

    // 1. 预处理：过滤无效深度值（FLT_MAX）并转换为可处理格式
    cv::Mat validDepthMask = zbuffer != FLT_MAX;
    cv::Mat depth8U;
    double minVal, maxVal;
    cv::minMaxLoc(zbuffer, &minVal, &maxVal, nullptr, nullptr, validDepthMask);
    zbuffer.convertTo(depth8U, CV_8U, 255.0 / (static_cast<float>(maxVal) - static_cast<float>(minVal)), -255.0 * static_cast<float>(minVal) / (static_cast<float>(maxVal) - static_cast<float>(minVal)));
    depth8U.setTo(0, ~validDepthMask);

    // 2. 提取深度连通区域并绘制多边形（保留原有逻辑）
    std::unordered_map<float, std::vector<cv::Point>> depthRegions;
    for (int y = 0; y < zbuffer.rows; ++y) {
        for (int x = 0; x < zbuffer.cols; ++x) {
            float depth = zbuffer.at<float>(y, x);
            if (depth == FLT_MAX) continue;
            depthRegions[depth].emplace_back(x, y);
        }
    }

    cv::Mat regionMask = cv::Mat::zeros(zbuffer.size(), CV_8U);
    for (const auto& [current_depth, points] : depthRegions) {
        if (points.size() < 4) continue;
        regionMask.setTo(0);
        
        bool is_valid_depth = false;
        {
            std::lock_guard<std::mutex> lock(global.mtx_surface_depths);
            for (float valid_depth : global.surface_depths) {
                if (std::abs(current_depth - valid_depth) < 0.0001 ) {
                    is_valid_depth = true;
                    break;
                }
            }
        }
        if (is_valid_depth) continue;

        for (const auto& p : points) {
            regionMask.at<uchar>(p) = 255;
        }

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(regionMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (const auto& contour : contours) {
            if (contour.size() < 4) continue;
            double epsilon = 0.01 * cv::arcLength(contour, true);
            std::vector<cv::Point> polygon;
            cv::approxPolyDP(contour, polygon, epsilon, true);
            cv::polylines(image, polygon, true, cv::Scalar(0, 0, 255), 2);
        }
    }

    // 3. 收集所有面的点并按idx分组，确保每个idx只绘制一次
    std::unordered_map<int, std::vector<cv::Point2f>> idx_points_map;
    auto collect_points = [&](const std::vector<G::surface_2d_point>& surface_points) {
        for (const auto& sp : surface_points) {
            idx_points_map[sp.idx].push_back(sp.left_up);
            idx_points_map[sp.idx].push_back(sp.right_up);
            idx_points_map[sp.idx].push_back(sp.right_down);
            idx_points_map[sp.idx].push_back(sp.left_down);
        }
    };

    // 收集三个面的所有点
    collect_points(front_2d);
    collect_points(side_2d);
    collect_points(up_2d);

    // 4. 绘制每个idx的文本（只绘制一次）
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.8;
    cv::Scalar textColor(0, 255, 0);  // 绿色文本
    int thickness = 2;
    int lineType = cv::LINE_AA;

    for (const auto& [idx, points] : idx_points_map) {
        // 计算所有点的平均位置作为文本绘制中心
        cv::Point2f avg(0, 0);
        for (const auto& p : points) {
            avg += p;
        }
        avg.x /= points.size();
        avg.y /= points.size();

        // 确保文本位置在图像范围内
        int text_x = std::max(10, std::min(image.cols - 50, static_cast<int>(avg.x)));
        int text_y = std::max(20, std::min(image.rows - 10, static_cast<int>(avg.y)));
        cv::Point textOrg(text_x, text_y);

        // 绘制idx
        cv::putText(image, std::to_string(idx), textOrg, fontFace, fontScale, textColor, thickness, lineType);
    }
}

cv::Mat zbuffer_occlusion(
    std::vector<std::vector<G::surface_2d_point>> box_2d_points_lists,
    std::vector<std::vector<G::surface_2d_point>> plum_box_2d_points_lists,
    const float camera_y,
    const float camera_x,
    const float camera_z,
    const double yaw,
    const cv::Mat& image
){
    // 函数内部局部变量
    const int image_width = image.cols;
    const int image_height = image.rows;
    std::vector<G::surface_2d_point> front_2d_points_lists = box_2d_points_lists[0];
    std::vector<G::surface_2d_point> side_2d_points_lists = box_2d_points_lists[1];
    std::vector<G::surface_2d_point> up_2d_points_lists = box_2d_points_lists[2];
    std::vector<G::surface_2d_point> plum_front_2d_points_lists = plum_box_2d_points_lists[0];
    std::vector<G::surface_2d_point> plum_side_2d_points_lists = plum_box_2d_points_lists[1];
    std::vector<G::surface_2d_point> plum_up_2d_points_lists = plum_box_2d_points_lists[2];

    // 画面静止时返回上次的zbuffer
    const float eps = 0.01f;  
    const double eps_rot = 0.01;
    bool pos_unchanged = (std::abs(camera_x - global.last_camera_x) < eps && std::abs(camera_y - global.last_camera_y) < eps);
    bool yaw_unchanges = (std::abs(yaw - global.last_yaw) < eps_rot);
    if (pos_unchanged && yaw_unchanges) {
        if (!global.last_zbuffer.empty()) {
            //ROS_INFO("not move or rotation");
            global.is_move = true;
            return global.last_zbuffer.clone();
        }
    }else{global.is_move = false;}

    // 更新上次坐标和姿态
    global.last_camera_x = camera_x;
    global.last_camera_y = camera_y;
    global.last_yaw = yaw;

    // 图像尺寸有效性检查
    if (image_width <= 0 || image_height <= 0) {
        std::cerr << "错误:图像尺寸无效(width=" << image_width << ", height=" << image_height << ")" << std::endl;
        return global.last_zbuffer.clone();
    }

    // 初始化主深度缓冲（合并台阶和方块的深度）
    cv::Mat zbuffer = cv::Mat::ones(image_height, image_width, CV_32F) * FLT_MAX;
    cv::Mat box_zbuffer = cv::Mat::ones(image_height, image_width, CV_32F) * FLT_MAX;
    //std::cout << "方块2D点数量: " << front_2d_points_lists.size() << " " << side_2d_points_lists.size() << " " << up_2d_points_lists.size() << std::endl;
    //std::cout << "台阶2D点数量: " << plum_front_2d_points_lists.size() << " " << plum_side_2d_points_lists.size() << " " << plum_up_2d_points_lists.size() << std::endl;

    // -------------------------- 第一步：处理台阶，写入深度缓冲 --------------------------
    for (size_t i = 0; i < plum_side_2d_points_lists.size(); i++) {
        if (i >= plum_front_2d_points_lists.size() || i >= plum_up_2d_points_lists.size()) break;

        auto& plum_front_point = plum_front_2d_points_lists[i];
        auto& plum_side_point = plum_side_2d_points_lists[i];
        auto& plum_up_point = plum_up_2d_points_lists[i];

        // 构建台阶3个面的轮廓
        std::vector<cv::Point> plum_front_contour = {
            cv::Point(cvRound(plum_front_point.left_up.x), cvRound(plum_front_point.left_up.y)),
            cv::Point(cvRound(plum_front_point.right_up.x), cvRound(plum_front_point.right_up.y)),
            cv::Point(cvRound(plum_front_point.right_down.x), cvRound(plum_front_point.right_down.y)),
            cv::Point(cvRound(plum_front_point.left_down.x), cvRound(plum_front_point.left_down.y))
        };
        std::vector<cv::Point> plum_up_contour = {
            cv::Point(cvRound(plum_up_point.left_up.x), cvRound(plum_up_point.left_up.y)),
            cv::Point(cvRound(plum_up_point.right_up.x), cvRound(plum_up_point.right_up.y)),
            cv::Point(cvRound(plum_up_point.right_down.x), cvRound(plum_up_point.right_down.y)),
            cv::Point(cvRound(plum_up_point.left_down.x), cvRound(plum_up_point.left_down.y))
        };
        std::vector<cv::Point> plum_side_contour = {
            cv::Point(cvRound(plum_side_point.left_up.x), cvRound(plum_side_point.left_up.y)),
            cv::Point(cvRound(plum_side_point.right_up.x), cvRound(plum_side_point.right_up.y)),
            cv::Point(cvRound(plum_side_point.right_down.x), cvRound(plum_side_point.right_down.y)),
            cv::Point(cvRound(plum_side_point.left_down.x), cvRound(plum_side_point.left_down.y))
        };

        // 填充台阶深度到临时矩阵
        cv::Mat plum_tmp = cv::Mat::ones(image_height, image_width, CV_32F) * FLT_MAX;
        cv::fillPoly(plum_tmp, std::vector<std::vector<cv::Point>>{plum_front_contour}, cv::Scalar(plum_front_point.surface_depth));
        cv::fillPoly(plum_tmp, std::vector<std::vector<cv::Point>>{plum_up_contour}, cv::Scalar(plum_up_point.surface_depth));
        cv::fillPoly(plum_tmp, std::vector<std::vector<cv::Point>>{plum_side_contour}, cv::Scalar(plum_side_point.surface_depth));

        // 计算台阶像素范围
        std::vector<float> plum_x_value = {
            plum_front_point.left_up.x, plum_front_point.right_up.x, plum_front_point.right_down.x, plum_front_point.left_down.x,
            plum_up_point.left_up.x, plum_up_point.right_up.x, plum_up_point.right_down.x, plum_up_point.left_down.x,
            plum_side_point.left_up.x, plum_side_point.right_up.x, plum_side_point.right_down.x, plum_side_point.left_down.x
        };
        std::vector<float> plum_y_value = {
            plum_front_point.left_up.y, plum_front_point.right_up.y, plum_front_point.right_down.y, plum_front_point.left_down.y,
            plum_up_point.left_up.y, plum_up_point.right_up.y, plum_up_point.right_down.y, plum_up_point.left_down.y,
            plum_side_point.left_up.y, plum_side_point.right_up.y, plum_side_point.right_down.y, plum_side_point.left_down.y
        };
        auto plum_x_max = std::max_element(plum_x_value.begin(), plum_x_value.end());
        auto plum_x_min = std::min_element(plum_x_value.begin(), plum_x_value.end());
        auto plum_y_max = std::max_element(plum_y_value.begin(), plum_y_value.end());
        auto plum_y_min = std::min_element(plum_y_value.begin(), plum_y_value.end());

        // 写入主zbuffer（台阶深度更近则更新）
        for (int row = int(*plum_y_min) - 5; row < int(*plum_y_max) + 5; ++row) {
            for (int col = int(*plum_x_min) - 5; col < int(*plum_x_max) + 5; ++col) {
                if (row < 0 || row >= image_height || col < 0 || col >= image_width) continue;
                if (plum_tmp.at<float>(row, col) < zbuffer.at<float>(row, col)) {
                    zbuffer.at<float>(row, col) = plum_tmp.at<float>(row, col);
                }
            }
        }
    }

    global.idx_to_depth.erase(global.idx_to_depth.begin(), global.idx_to_depth.end());

    // -------------------------- 第二步：处理方块，基于深度判断遮挡 --------------------------
    for (size_t i = 0; i < side_2d_points_lists.size(); i++) {
        if (i >= front_2d_points_lists.size() || i >= up_2d_points_lists.size()) break;

        auto& front_point = front_2d_points_lists[i];
        auto& side_point = side_2d_points_lists[i];
        auto& up_point = up_2d_points_lists[i];

        // 构建方块3个面的轮廓
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

        // 填充方块深度到临时矩阵
        cv::Mat tmp = cv::Mat::ones(image_height, image_width, CV_32F) * FLT_MAX;
        cv::fillPoly(tmp, std::vector<std::vector<cv::Point>>{front_contour}, cv::Scalar(front_point.surface_depth));
        cv::fillPoly(tmp, std::vector<std::vector<cv::Point>>{up_contour}, cv::Scalar(up_point.surface_depth));
        cv::fillPoly(tmp, std::vector<std::vector<cv::Point>>{side_contour}, cv::Scalar(side_point.surface_depth));
        global.idx_to_depth[i + 1] = {front_point.surface_depth, up_point.surface_depth, side_point.surface_depth };

        // ROS_INFO("insert idx_to_depth: idx=%d, depth_list =[front:%.3f, up:%.3f, side:%.3f]",
        //  i + 1,
        //  front_point.surface_depth,
        //  up_point.surface_depth,
        //  side_point.surface_depth);


        // 计算方块像素范围
        std::vector<float> x_value = {
            front_point.left_up.x, front_point.right_up.x, front_point.right_down.x, front_point.left_down.x,
            up_point.left_up.x, up_point.right_up.x, up_point.right_down.x, up_point.left_down.x,
            side_point.left_up.x, side_point.right_up.x, side_point.right_down.x, side_point.left_down.x
        };
        std::vector<float> y_value = {
            front_point.left_up.y, front_point.right_up.y, front_point.right_down.y, front_point.left_down.y,
            up_point.left_up.y, up_point.right_up.y, up_point.right_down.y, up_point.left_down.y,
            side_point.left_up.y, side_point.right_up.y, side_point.right_down.y, side_point.left_down.y
        };
        auto x_max = std::max_element(x_value.begin(), x_value.end());
        auto x_min = std::min_element(x_value.begin(), x_value.end());
        auto y_max = std::max_element(y_value.begin(), y_value.end());
        auto y_min = std::min_element(y_value.begin(), y_value.end());

        // 写入主zbuffer（方块深度更近则更新，不被台阶/其他方块遮挡）
        for (int row = int(*y_min) - 5; row < int(*y_max) + 5; ++row) {
            for (int col = int(*x_min) - 5; col < int(*x_max) + 5; ++col) {
                if (row < 0 || row >= image_height || col < 0 || col >= image_width) continue;
                if (tmp.at<float>(row, col) < zbuffer.at<float>(row, col)) {
                    zbuffer.at<float>(row, col) = tmp.at<float>(row, col);
                    box_zbuffer.at<float>(row, col) = tmp.at<float>(row, col);
                }
            }
        }
    }

    {
        std::lock_guard<std::mutex> lock(global.mtx_last_zbuffer);
        box_zbuffer.copyTo(global.last_zbuffer);
    }

    return box_zbuffer;
}

// -------------------------------------------------------------------------- determine_square_hsv  -----------------------------------------------
// 1.提取方块的标准hsv值, 2.根据标准hsv值筛选方块
std::vector<int> extract_square_hsv_filterEmptyBoxes(const cv::Mat& zbuffer, const cv::Mat& image,std::unordered_set<int>& processed_idxs) {

    // 1.1 转换图像为HSV色彩空间
    cv::Mat hsv_image;
    cv::cvtColor(image, hsv_image, cv::COLOR_RGB2HSV);

    // 1.2 统计HSV值出现频率（使用直方图统计众数）
    const int h_bins = 180; // H通道范围0-179
    const int s_bins = 256; // S通道范围0-255
    const int v_bins = 256; // V通道范围0-255

    // 1.3 记录每个idx对应的hsv众数值
    static std::unordered_map<int,std::tuple<int,int,int>> idx_hsv_map;

    // 2 根据深度表来查找对应深度的像素并进行众数统计
    for (const auto& idx_depth_pair : global.idx_to_depth) {
        int idx = idx_depth_pair.first;
        const std::vector<float>& depths = idx_depth_pair.second;

        if (depths.empty()) continue;
        // 2.1 每个方块只处理一次
        if (processed_idxs.count(idx)) {
            //ROS_INFO("Idx %d processed", idx);
            continue;
        }

        // 2.2 收集该idx对应的所有像素坐标
        std::vector<cv::Point> idx_pixels;

        // 2.3 针对每个表面深度值查找对应像素（允许微小误差）
        const float depth_tolerance = 0.001f; // 深度容差，可根据实际情况调整
        for (float target_depth : depths) {
            for (int y = 0; y < zbuffer.rows; ++y) {
                for (int x = 0; x < zbuffer.cols; ++x) {
                    float current_depth = zbuffer.at<float>(y, x);
                    if (current_depth != FLT_MAX && 
                        std::abs(current_depth - target_depth) < depth_tolerance) {
                        idx_pixels.emplace_back(x, y);
                    }
                }
            }
        }
        // 2.4 筛选掉像素过少的情况
        if (idx_pixels.size() <= 800) {
            ROS_INFO("Idx %d pixels=%zu(<=800),continue", idx, idx_pixels.size());
            continue;
        }

        // 2.5 制表统计众数
        cv::Mat h_hist = cv::Mat::zeros(1, h_bins, CV_32S);
        cv::Mat s_hist = cv::Mat::zeros(1, s_bins, CV_32S);
        cv::Mat v_hist = cv::Mat::zeros(1, v_bins, CV_32S);

        for (const auto& pt : idx_pixels) {
            cv::Vec3b hsv = hsv_image.at<cv::Vec3b>(pt.y, pt.x);
            int h = hsv[0];
            int s = hsv[1];
            int v = hsv[2];
            h_hist.at<int>(0, h)++;
            s_hist.at<int>(0, s)++;
            v_hist.at<int>(0, v)++;
        }

        // 2.6 找到众数对应的HSV值（绝大多数像素的取值）
        cv::Point h_max_loc, s_max_loc, v_max_loc;
        cv::minMaxLoc(h_hist, nullptr, nullptr, nullptr, &h_max_loc);
        cv::minMaxLoc(s_hist, nullptr, nullptr, nullptr, &s_max_loc);
        cv::minMaxLoc(v_hist, nullptr, nullptr, nullptr, &v_max_loc);

        // 2.7 提取众数（Point的x坐标对应直方图的bin索引）
        int h_mode = h_max_loc.x;
        int s_mode = s_max_loc.x;
        int v_mode = v_max_loc.x;
        idx_hsv_map[idx] = {h_mode, s_mode, v_mode};
        std::cout << "will push_back into idx_hsv_map : " << idx << ",h: " <<  h_mode << ",s: " <<  s_mode << ",v: " <<  v_mode << std::endl;
        processed_idxs.insert(idx);
    }

    // 3 确定标准值
    int result_h,result_s,result_v;     // 标准值
    int min_total_h_difference_value = INT_MAX,min_total_s_difference_value = INT_MAX,min_total_v_difference_value = INT_MAX;   // 最小总差值

    // 3.1 分别填入h,s,v的列表
    std::vector<int> idx_h_result, idx_s_result, idx_v_result;
    for (const auto& [idx, hsv_tuple] : idx_hsv_map) {
        idx_h_result.push_back(std::get<0>(hsv_tuple));
        idx_s_result.push_back(std::get<1>(hsv_tuple));
        idx_v_result.push_back(std::get<2>(hsv_tuple));
        // 打印结果
        // ROS_INFO("Idx %d: Majority HSV = (%d, %d, %d)",
        //          idx, std::get<0>(hsv_tuple), std::get<1>(hsv_tuple), std::get<2>(hsv_tuple));
    }
    // 3.2 分别计算h,s,v的标准值(根据与其他的方块的h,s,v的 差值累加的 总差值最小来确定)
    for(int i = 0;i < idx_h_result.size();i++){
        int total_h_difference_value = 0;
        for (int j = 0;j < idx_h_result.size();j++){
            total_h_difference_value += abs(idx_h_result[i] - idx_h_result[j]);
        }
        if(total_h_difference_value < min_total_h_difference_value){
            min_total_h_difference_value = total_h_difference_value;
            result_h = idx_h_result[i];
        }
    }

    for(int i = 0;i < idx_s_result.size();i++){
        int total_s_difference_value = 0;
        for (int j = 0;j < idx_s_result.size();j++){
            total_s_difference_value += abs(idx_s_result[i] - idx_s_result[j]);
        }
        if(total_s_difference_value < min_total_s_difference_value){
            min_total_s_difference_value = total_s_difference_value;
            result_s = idx_s_result[i];
        }
    }

    for(int i = 0;i < idx_v_result.size();i++){
        int total_v_difference_value = 0;
        for (int j = 0;j < idx_v_result.size();j++){
            total_v_difference_value += abs(idx_v_result[i] - idx_v_result[j]);
        }
        if(total_v_difference_value < min_total_v_difference_value){
            min_total_v_difference_value = total_v_difference_value;
            result_v = idx_v_result[i];
        }
    }
    //ROS_INFO("h_result: %d, s_result: %d, v_result:%d",result_h,result_s,result_v);

    // 4 根据标准值计算每个方块的得分
    std::vector<float> score = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}; 
    for (const auto& [idx, hsv_tuple] : idx_hsv_map) {
        if (idx < 1 || idx > score.size()) { // 防止idx超出score数组范围
            continue;
        }
        int h_mode = std::get<0>(hsv_tuple);
        int s_mode = std::get<1>(hsv_tuple);
        int v_mode = std::get<2>(hsv_tuple);
        
        score[idx - 1] = std::max(0.0f,75.0f - abs(h_mode - result_h) )
                       + std::max(0.0f,10.0f - 0.2f * abs(s_mode - result_s))
                       + std::max(0.0f,15.0f - 0.2f * abs(v_mode - result_v));
    }

    // 5 返回结果
    static std::vector<int> result_idx;
    for (int i = 0; i < score.size(); i++){
        if (score[i] >= 95){
            result_idx.push_back(i + 1);
            std::cout << " " << i + 1 << " ";
        }
        std::cout << "i + 1: " << i + 1 <<  ", score: " << score[i] << std::endl;
    }
    return result_idx;
}

Eigen::Matrix3f createRotationMatrix(float rx, float ry, float rz) {
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
    return R_z * R_y * R_x;
}

Eigen::Vector3f createTranslationVector(float tx, float ty, float tz) {
    Eigen::Vector3f translation(tx, ty, tz);
    return translation;
}

Eigen::Matrix4f combineRotationAndTranslation(const Eigen::Matrix3f& rotation, const Eigen::Vector3f& translation) {
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block<3, 3>(0, 0) = rotation;
    transform.block<3, 1>(0, 3) = translation;
    return transform;
}

void tfCallback(const tf2_msgs::TFMessage::ConstPtr& msg)
{
    for (const auto& transform : msg->transforms)
    {
        if(transform.header.frame_id != "odom")
        {
            std::cout<< transform.header.frame_id<<std::endl;
            continue;
        }
        geometry_msgs::Quaternion quat = transform.transform.rotation;
        tf2::Quaternion tf_quat;
        tf_quat.setX(quat.x);
        tf_quat.setY(quat.y);
        tf_quat.setZ(quat.z);
        tf_quat.setW(quat.w);
        tf2::Matrix3x3 mat(tf_quat);
        double roll, pitch, yaw;
        mat.getRPY(roll, pitch, yaw);
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
    }
}

void odomCallback(const nav_msgs::Odometry::ConstPtr& msg)
{
    global.x = msg->pose.pose.position.x;
    global.y = msg->pose.pose.position.y;
    global.z = msg->pose.pose.position.z;    
    //std::cout << "x: " << global.x << " y: " << global.y << " z: " << global.z <<std::endl;

    geometry_msgs::Quaternion quat = msg->pose.pose.orientation;
    tf2::Quaternion tf_quat;
    tf_quat.setX(quat.x);
    tf_quat.setY(quat.y);
    tf_quat.setZ(quat.z);
    tf_quat.setW(quat.w);
    tf2::Matrix3x3 mat(tf_quat);
    double roll, pitch;
    mat.getRPY(roll, pitch, global.yaw);
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
    //std::cout<< "yaw: "<< global.yaw <<std::endl;
}

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    try
    {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        cv::Mat image_ =  cv_ptr->image;

        {
            std::lock_guard<std::mutex> lock(global.mtx_surface_depths);
            global.surface_depths.clear();
        }
        std::vector<int> exclude_idxs = {};

        // 筛选方块和台阶的允许索引
        // std::vector<int> allow_idx_lists = {1,4,7,8,9,10,11,12};
        // std::vector<int> allow_idx_lists = {2,3,5,6,7,9,11};
        std::vector<int> allow_idx_lists = {1,2,3,4,5,6,7,8,9,10,11,12};
        std::vector<std::vector<G::surface_2d_point>> box_2d_points_lists = zbuffer_initialization(
            global.boxes, allow_idx_lists, global.x, global.y, 1.3);
        std::vector<std::vector<G::surface_2d_point>> plum_box_2d_points_lists = zbuffer_initialization(
            global.plum_boxes, {1,2,3,4,5,6,7,8,9,10,11,12}, global.x, global.y, 1.3);
        //check_one_point_hsv(image_);
        // 计算遮挡并生成zbuffer
        cv::Mat zbuffer = zbuffer_occlusion(box_2d_points_lists, plum_box_2d_points_lists, global.x, global.y, 1.3, global.yaw, image_);
        //filterEmptyBoxes(zbuffer,image_,global.is_move);
        static std::unordered_set<int> processed_idxs;
        //std::cout << "processed_idxs.size(): " << processed_idxs.size() << std::endl;
        if (processed_idxs.size() < 12){ 
            std::vector<int> result = extract_square_hsv_filterEmptyBoxes(zbuffer,image_,processed_idxs);
        }

        draw_zbuffer(
            image_, 
            zbuffer, 
            box_2d_points_lists[0],  // 前面的2D点列表
            box_2d_points_lists[1],  // 侧面的2D点列表
            box_2d_points_lists[2],  // 上面的2D点列表
            0.01f
        );

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
        ros::spinOnce();
        sl.sleep();
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "zbuffer_func_final_node");
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
        // std::cout << "publish success" << std::endl;
        rate.sleep();
    }

    for (auto& worker : workers) {
        worker.join();
    }
    return 0;
}