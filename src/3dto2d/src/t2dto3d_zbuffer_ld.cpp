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
#include <dirent.h>    // Linux目录遍历
#include <sys/stat.h>  // Linux文件状态


// 点云 cloud map 
// tf cloud->map
// 8 判断点的数量
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
    std::vector<struct surface_2d_point> side_2d_points_lists;
    std::vector<struct surface_2d_point> front_2d_point_lists;
    std::vector<struct surface_2d_point> up_2d_point_lists;
    std::vector<cv::Point3f> objectPoints;
    std::vector<cv::Point3f> lidar_objectPoints;
    std::vector<cv::Point3f> plum_blossom_points;
    std::vector<cv::Point3f> plum_blossom_points_lists;
    std::unordered_map<int,std::vector<float>> idx_to_depth;
    std::unordered_map<int,std::vector<bool>>record_idx_score;

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

// ------------------------------------------------------------------------------------------- saveZBufferVisualization ---------------
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
        return zbuffer_processed;
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
    // 4. 返回可视化图像
    return zbuffer_vis;
}

// ----------------------------------------------------------------------------------------------  zbuffer_initialization  ---------------------------
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
// --------------------------------------------------------------------------------------------------  save_zbuffer_image  -------------------------
int getMaxImageNumber(const std::string& dir_path) {
    int max_num = -1;  // 初始值-1（无文件时返回-1）

    // 打开目录
    DIR* dir = opendir(dir_path.c_str());
    if (!dir) {
        // 目录不存在或打开失败（已在createDirectoryIfNotExists中创建，此处仅警告）
        ROS_WARN("Directory open failed: %s, err: %s", dir_path.c_str(), strerror(errno));
        return max_num;
    }

    struct dirent* entry = nullptr;
    // 遍历目录内所有文件/文件夹
    while ((entry = readdir(dir)) != nullptr) {
        // 跳过 "." 和 ".." 目录
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }

        // 检查是否为普通文件（避免处理子目录）
        std::string full_path = dir_path + "/" + entry->d_name;
        struct stat st;
        if (stat(full_path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) {
            continue;
        }

        // 解析文件名：必须是 "image_数字.png" 格式
        std::string filename = entry->d_name;
        std::string prefix = "image_";
        std::string suffix = ".png";

        // 1. 检查前缀和后缀是否匹配
        if (filename.find(prefix) != 0 || filename.rfind(suffix) != filename.size() - suffix.size()) {
            continue;
        }

        // 2. 提取中间的数字字符串
        std::string num_str = filename.substr(
            prefix.size(), 
            filename.size() - prefix.size() - suffix.size()
        );

        // 3. 字符串转整数（避免非数字字符崩溃）
        try {
            int num = std::stoi(num_str);
            if (num > max_num) {
                max_num = num;
            }
        } catch (const std::exception& e) {
            ROS_WARN("Invalid number in filename: %s, err: %s", filename.c_str(), e.what());
            continue;
        }
    }

    closedir(dir);  // 关闭目录
    return max_num;
}

void save_zbuffer_image(cv::Mat& image, 
                const cv::Mat& zbuffer, 
                 const std::string& save_dir = "/home/h/RC2026/save_image/8") {  // 新增：保存文件夹参数
    if (zbuffer.empty()) {
        ROS_WARN("Z-buffer is empty, skipping draw_zbuffer.");
        return;
    }
    if (image.empty()) {
        ROS_WARN("Image is empty, skipping draw_zbuffer.");
        return;
    }

    // 1. 预处理：过滤无效深度值并转换格式
    cv::Mat validDepthMask = zbuffer != FLT_MAX;
    cv::Mat depth8U;
    double minVal, maxVal;
    cv::minMaxLoc(zbuffer, &minVal, &maxVal, nullptr, nullptr, validDepthMask);
    zbuffer.convertTo(depth8U, CV_8U, 255.0 / (static_cast<float>(maxVal) - static_cast<float>(minVal)), -255.0 * static_cast<float>(minVal) / (static_cast<float>(maxVal) - static_cast<float>(minVal)));
    depth8U.setTo(0, ~validDepthMask);

    // 2. 提取深度连通区域并处理多边形
    std::unordered_map<float, std::vector<cv::Point>> depthRegions;
    for (int y = 0; y < zbuffer.rows; ++y) {
        for (int x = 0; x < zbuffer.cols; ++x) {
            float depth = zbuffer.at<float>(y, x);
            if (depth == FLT_MAX) continue;
            depthRegions[depth].emplace_back(x, y);
        }
    }

    cv::Mat regionMask = cv::Mat::zeros(zbuffer.size(), CV_8U);
    int quad_count = getMaxImageNumber(save_dir);  // 四边形计数器，用于保存文件名
    std::cout << "quad_count : " << quad_count << std::endl;

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

            // 处理四边形区域
            if (polygon.size() == 4) {  // 确保是四边形
                // 计算四边形的最小外接矩形（ROI）
                cv::Rect roi = cv::boundingRect(polygon);
                // 确保ROI在图像范围内
                roi.x = std::max(0, roi.x);
                roi.y = std::max(0, roi.y);
                roi.width = std::min(image.cols - roi.x, roi.width);
                roi.height = std::min(image.rows - roi.y, roi.height);
                if (roi.width <= 0 || roi.height <= 0) continue;

                // 裁剪ROI
                cv::Mat roi_img = image(roi).clone();
                //  resize为640x640
                cv::Mat resized_img;
                cv::resize(roi_img, resized_img, cv::Size(640, 640));

                // 保存到指定文件夹（文件名格式：quad_{序号}.png）
                std::string save_path = save_dir + "/image_" + std::to_string(quad_count++) + ".png";
                cv::imwrite(save_path, resized_img);
                ROS_INFO("Saved quadrilateral to: %s", save_path.c_str());
            }
        }
    }
}


// --------------------------------------------------------------------------------------------------  draw_zbuffer  -------------------------
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

    // 发布zbuffer可视化图像
    // cv::Mat zbuffer_vis = saveZBufferVisualization(box_zbuffer);
    // if (!zbuffer_vis.empty()) {
    //     std_msgs::Header header;
    //     header.stamp = ros::Time::now();
    //     header.frame_id = "camera_frame";
    //     sensor_msgs::ImagePtr msg = cv_bridge::CvImage(header, "mono8", zbuffer_vis).toImageMsg();
    //     global.zbuffer_pub.publish(msg);
    // } else {
    //     ROS_WARN("Z-Buffer可视化图像为空，无法发布");
    // }

    // 保存当前zbuffer为下次使用
    {
        std::lock_guard<std::mutex> lock(global.mtx_last_zbuffer);
        box_zbuffer.copyTo(global.last_zbuffer);
    }

    return box_zbuffer;
}

void check_one_point_hsv(cv::Mat &image){
    int h_min = 80, h_max = 120, s_min = 0, s_max = 255,  v_min = 0, v_max = 255;
    if ( image.empty()) {
        ROS_WARN("Invalid input for filterEmptyBoxes");
    }
    cv::Mat hsv_image;
    cv::cvtColor(image, hsv_image, cv::COLOR_RGB2HSV);
    cv::Point2f point(800,700);
    cv::Vec3b hsv = hsv_image.at<cv::Vec3b>(point.y, point.x);
    int h = hsv[0], s = hsv[1], v = hsv[2];
    std::cout << "pt.y: " << point.y << " , pt.x: " << point.x << std::endl;
    std::cout << "h: " << h << ", s: " << s << ", v: " << v  << std::endl;
    cv::circle(image, point, 5, (255, 0, 0), -1);  // 半径 2，-1 表示填充圆

}
// --------------------------------------------------------------------------------- filterEmptyBoxes -----------------------------
std::vector<int> filterEmptyBoxes(
    const cv::Mat& zbuffer, 
    const cv::Mat& image,
    bool is_move
) {
    // 初始化临时存储结构（每次处理重置）
    std::unordered_map<int, std::vector<float>> idx_score; // 每个idx的单次得分列表
    std::unordered_map<int, std::vector<float>> idx_score_with_record; // 每个idx的[平均得分, 历史前八占比]
    static std::unordered_map<int, std::vector<bool>> record_idx_score; // 长期存储的前八记录（超过100条时截断）

    // std::vector<std::pair<cv::Scalar, cv::Scalar>> colorRanges = {
    //     {cv::Scalar(115, 0, 0), cv::Scalar(160, 255, 255)},  // 示例颜色范围
    // };
    int h_min = 1, h_max = 15, s_min = 0, s_max = 255,  v_min = 0, v_max = 40;
    // 检查输入有效性
    if (zbuffer.empty() || image.empty()) {
        //ROS_WARN("Invalid input form filterEmptyBoxes");
        return {};
    }
    
    if (is_move) {
        //ROS_WARN("not move form filterEmptyBoxes");
        return {};
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

    // 将图像转换为HSV格式（提高效率，避免多次转换）
    cv::Mat hsv_image;
    cv::cvtColor(image, hsv_image, cv::COLOR_RGB2HSV);

    // 1. 处理每个四边形轮廓，计算得分
    for (const auto& contour : contours) {
        // 多边形逼近（确保是四边形）
        double epsilon = 0.01 * cv::arcLength(contour, true);
        std::vector<cv::Point> polygon;
        cv::approxPolyDP(contour, polygon, epsilon, true);

        if (polygon.size() == 4) { // 只处理四边形
            for (int i = 0; i < polygon.size(); ++i) {
                cv::Point pt = polygon[i];
                // 检查点是否在图像范围内
                if (pt.x < 0 || pt.x >= zbuffer.cols || pt.y < 0 || pt.y >= zbuffer.rows) {
                    continue;
                }
                
                // 提取zbuffer中的深度值
                float depth = zbuffer.at<float>(pt.y, pt.x);
                // std::cout << "depth: "  << depth << std::endl;
                if (depth == FLT_MAX) { // 无效深度
                    continue;
                }

                // 反查深度对应的idx（考虑浮点数误差）
                int target_idx = -1;
                const float depth_eps = 0.001f; // 深度比较误差阈值
                for (const auto& [idx, depths] : global.idx_to_depth) {
                    for (float d : depths) {
                        if (std::abs(d - depth) < depth_eps) {
                            target_idx = idx;
                            break;
                        }
                    }
                    if (target_idx != -1) break;
                }
                if (target_idx == -1) { // 未找到对应idx
                    continue;
                }

                // 提取HSV值
                cv::Vec3b hsv = hsv_image.at<cv::Vec3b>(pt.y, pt.x);
                int h = hsv[0], s = hsv[1], v = hsv[2];
                // std::cout << "pt: " << pt.x << " * " << pt.y << std::endl;
                // std::cout << "h: " << h << ", s: " << s << ", v: " << v  << std::endl;
                // cv::circle(image, pt, 2, (255, 0, 0), -1);  // 半径 2，-1 表示填充圆
                // 计算HSV得分（H:80分, S:10分, V:10分）
                float score = 0.0f;

                // H通道得分（80分）
                int h_mid = (h_min + h_max) / 2;
                int h_range = h_max - h_min;
                if (h >= h_min && h <= h_max && h_range > 0) {
                    float h_ratio = 1.0f - std::abs(h - h_mid) / static_cast<float>(h_range);
                    score += 80.0f * std::max(0.0f, h_ratio);
                }

                // S通道得分（10分）
                int s_mid = (s_min + s_max) / 2;
                int s_range = s_max - s_min;
                if (s >= s_min && s <= s_max && s_range > 0) {
                    float s_ratio = s/255;
                    score += 10.0f * std::max(0.0f, s_ratio);
                }

                // V通道得分（10分）
                int v_mid = (v_min + v_max) / 2;
                int v_range = v_max - v_min;
                if (v >= v_min && v <= v_max && v_range > 0) {
                    float v_ratio = 1.0f - std::abs(v - v_mid) / static_cast<float>(v_range);
                    score += 10.0f * std::max(0.0f, v_ratio);
                }

                // 将得分存入idx_score
                // std::cout << "score: " << score << std::endl;
                idx_score[target_idx].push_back(score);
                // std::cout << "idx: " << target_idx <<  ", score: " << score << std::endl;
            }


        cv::Point center;
        center.x = (polygon[0].x + polygon[1].x + polygon[2].x + polygon[3].x) / 4;
        center.y = (polygon[0].y + polygon[1].y + polygon[2].y + polygon[3].y) / 4;

        // 2. 检查中心点是否在图像/zbuffer范围内（避免越界访问）
        if (center.x >= 0 && center.x < zbuffer.cols && center.y >= 0 && center.y < zbuffer.rows) {
            // 3. 提取中心点的深度值
            float center_depth = zbuffer.at<float>(center.y, center.x);
            if (center_depth != FLT_MAX) { // 深度有效
                // 4. 反查中心点深度对应的idx（与顶点逻辑一致）
                int center_target_idx = -1;
                const float depth_eps = 0.001f;
                for (const auto& [idx, depths] : global.idx_to_depth) {
                    for (float d : depths) {
                        if (std::abs(d - center_depth) < depth_eps) {
                            center_target_idx = idx;
                            break;
                        }
                    }
                    if (center_target_idx != -1) break;
                }

                if (center_target_idx != -1) { // 找到对应idx
                    // 5. 提取中心点的HSV值
                    cv::Vec3b center_hsv = hsv_image.at<cv::Vec3b>(center.y, center.x);
                    int h = center_hsv[0], s = center_hsv[1], v = center_hsv[2];

                    // 6. 计算中心点的得分（与顶点评分逻辑完全一致）
                    float center_score = 0.0f;

                    // H通道得分（80分）
                    int h_mid = (h_min + h_max) / 2;
                    int h_range = h_max - h_min;
                    if (h >= h_min && h <= h_max && h_range > 0) {
                        float h_ratio = 1.0f - std::abs(h - h_mid) / static_cast<float>(h_range);
                        center_score += 80.0f * std::max(0.0f, h_ratio);
                    }

                    // S通道得分（10分）
                    int s_mid = (s_min + s_max) / 2;
                    int s_range = s_max - s_min;
                    if (s >= s_min && s <= s_max && s_range > 0) {
                        float s_ratio = s/255;
                        center_score += 10.0f * std::max(0.0f, s_ratio);
                    }

                    // V通道得分（10分）
                    int v_mid = (v_min + v_max) / 2;
                    int v_range = v_max - v_min;
                    if (v >= v_min && v <= v_max && v_range > 0) {
                        float v_ratio = 1.0f - std::abs(v - v_mid) / static_cast<float>(v_range);
                        center_score += 10.0f * std::max(0.0f, v_ratio);
                    }

                    // 7. 将中心点得分存入idx_score（与顶点得分累积）
                    idx_score[center_target_idx].push_back(center_score);
                    idx_score[center_target_idx].push_back(center_score);
                    idx_score[center_target_idx].push_back(center_score);
                    idx_score[center_target_idx].push_back(center_score);
                }
            }
        }
}


    }
    }

    // 2. 计算每个idx的平均得分，初始化idx_score_with_record
    std::vector<std::pair<int, float>> idx_avg_scores; // 用于排序的(idx, 平均得分)
    for (const auto& [idx, scores] : idx_score) {
        if (scores.empty()) continue;
        float avg = std::accumulate(scores.begin(), scores.end(), 0.0f) / scores.size();

        // std::cout << "idx: " << idx << std::endl;
        // std::cout << " -------------scores: " << std::endl;
        // for (auto& s : scores ){
        //     std::cout << s << ", ";
        // }
        // std::cout << std::endl;

        idx_score_with_record[idx].push_back(avg); // 第一个元素：平均得分
        idx_avg_scores.emplace_back(idx, avg);
    }
    
// ----------
        static std::vector<int> must_be_true = {0,0,0,0,0,0,0,0,0,0,0,0};
        static std::vector<int> must_be_false = {0,0,0,0,0,0,0,0,0,0,0,0};
        static std::vector<int> maybe_true = {0,0,0,0,0,0,0,0,0,0,0,0};
        static std::vector<float> result_score_list = {0,0,0,0,0,0,0,0,0,0,0,0};
        std::vector<std::pair<float, int>> score_with_idx;
        static int total_count = 0;
        total_count += 1;
        const float EPS = 1e-6f;

        for(const auto& [idx, avg] : idx_avg_scores){
            std::cout << "idx: "  << idx << ", avg: " << avg << std::endl;
            
            if (avg < 1.0f - EPS) { 
                if (idx-1 >= 0 && idx-1 < must_be_false.size()) {
                    must_be_false[idx-1] += 1;
                    std::cout << "must_be_false " << std::endl;
                }
            }
            else if (avg > 35.0f + EPS) {
                if (idx-1 >= 0 && idx-1 < must_be_true.size()) {
                    must_be_true[idx-1] += 1;
                    std::cout << "must_be_true " << std::endl;
                }
            }
            else { 
                if (idx-1 >= 0 && idx-1 < maybe_true.size()) {
                    maybe_true[idx-1] += 1;
                    std::cout << "maybe_true " << std::endl;
                }
            }
        }

        if (total_count > 12){
            for (int i = 0;i < must_be_true.size();i++){
                float result_score = 
                  (static_cast<float>(must_be_true[i]) / total_count) * 120 
                   + (static_cast<float>(maybe_true[i]) / total_count) * 30 
                   - (static_cast<float>(must_be_false[i]) / total_count) * 50;

                result_score_list[i] = result_score;
                std::cout << "i + 1: " << i + 1 << ", result_score: " << result_score << std::endl;
                score_with_idx.emplace_back(result_score,i + 1);
            }
        }

        std::sort(score_with_idx.begin(), score_with_idx.end(),
                [](const auto& a, const auto& b) {
                    return a.first > b.first; // 降序规则
                });

        auto new_end = std::remove_if(score_with_idx.begin(), score_with_idx.end(),
                                    [](const auto& pair) {
                                        // 筛选条件：得分 < 38.0f（保留 >=38.0f 的元素）
                                        return pair.first < 35.0f;
                                    });
        score_with_idx.erase(new_end, score_with_idx.end());    
                           
        
        // 4. 提取前八下标（不足8个则取全部）
        std::vector<int> result;
        int take_num = std::min(8, (int)score_with_idx.size());
        for (int i = 0; i < take_num; ++i) {
            result.push_back(score_with_idx[i].second); // 存入原始下标（int 型）
        }





// ----------
    // // 3. 确定本次前八的idx，更新record_idx_score
    // // 排序获取前八的idx
    // std::sort(idx_avg_scores.begin(), idx_avg_scores.end(), 
    //           [](const auto& a, const auto& b) { return a.second > b.second; });
    // std::unordered_set<int> top8_current;
    // int top_n = std::min(8, (int)idx_avg_scores.size());
    // for (int i = 0; i < top_n; ++i) {
    //     top8_current.insert(idx_avg_scores[i].first);
    // }

    // // 更新所有参与评分的idx的记录
    // for (const auto& [idx, _] : idx_score) {
    //     bool is_top8 = top8_current.count(idx) > 0;
    //     record_idx_score[idx].push_back(is_top8);
    //     // 保持记录长度不超过100
    //     if (record_idx_score[idx].size() > 30) {
    //         record_idx_score[idx].erase(record_idx_score[idx].begin());
    //     }
    // }

    // // 4. 计算历史前八占比，完善idx_score_with_record
    // for (auto& [idx, vec] : idx_score_with_record) {
    //     if (record_idx_score.find(idx) == record_idx_score.end() || record_idx_score[idx].empty()) {
    //         vec.push_back(0.0f); // 无历史记录，占比0
    //         continue;
    //     }
    //     // 计算true的占比
    //     int true_count = std::count(record_idx_score[idx].begin(), record_idx_score[idx].end(), true);
    //     float ratio = static_cast<float>(true_count) / record_idx_score[idx].size();
    //     vec.push_back(ratio); // 第二个元素：历史前八占比
    //     // std::cout << "idx: " << idx << ", ratio: " << ratio << std::endl;
    // }

    // // 5. 计算最终得分（平均得分 * 历史占比），取前八输出
    // std::vector<std::pair<int, float>> final_scores;
    // for (const auto& [idx, vec] : idx_score_with_record) {
    //     if (vec.size() < 2) continue; // 确保数据完整
    //     float final_score = vec[0] * vec[1];
    //     final_scores.emplace_back(idx, final_score);
    // }

    // std::cout << " -------------final_scores: " << std::endl;
    // for (auto& [idx, vec] : final_scores ){
    //     std::cout << idx << ": " << vec << ",  ";
    // }
    // std::cout << " -------------" << std::endl;
 

    // // 排序取前八
    // std::sort(final_scores.begin(), final_scores.end(),
    //           [](const auto& a, const auto& b) { return a.second > b.second; });

    // std::vector<int> result;
    // int output_n = std::min(8, (int)final_scores.size());
    // for (int i = 0; i < output_n; ++i) {
    //     result.push_back(final_scores[i].first);
    // }



    //std::sort(result.begin(), result.end());
    std::cout << " -------------result: " << std::endl;
    for (auto& re : result ){
        std::cout << re << "  ";
    }
    std::cout << " -------------" << std::endl;
    return result;
}


// std::vector<int> filterEmptyBoxes(
//     const cv::Mat& zbuffer, 
//     const cv::Mat& image

// ) {
//     std::vector<int> allow_idx_lists; 
//     if (zbuffer.empty()) {
//         ROS_WARN("Z-buffer is empty, skipping draw_zbuffer.");
//         return;
//     }
//     if (image.empty()) {
//         ROS_WARN("Image is empty, skipping draw_zbuffer.");
//         return;
//     }

//     // 1. 将图像转为HSV空间（更鲁棒的颜色分割）
//     cv::Mat hsvImage;
//     cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);

//     // 2. 定义目标颜色的HSV范围（OpenCV的HSV范围：H:0-180, S:0-255, V:0-255）
//     std::vector<std::pair<cv::Scalar, cv::Scalar>> colorRanges = {
//         {cv::Scalar(115, 0, 0), cv::Scalar(160, 255, 255)},  // 示例颜色范围
//     };

//     // 1. 预处理：过滤无效深度值（FLT_MAX）并转换为可处理格式
//     cv::Mat validDepthMask = zbuffer != FLT_MAX;
//     cv::Mat depth8U;
//     double minVal, maxVal;
//     cv::minMaxLoc(zbuffer, &minVal, &maxVal, nullptr, nullptr, validDepthMask);
//     zbuffer.convertTo(depth8U, CV_8U, 255.0 / (static_cast<float>(maxVal) - static_cast<float>(minVal)), -255.0 * static_cast<float>(minVal) / (static_cast<float>(maxVal) - static_cast<float>(minVal)));
//     depth8U.setTo(0, ~validDepthMask);

//     // 2. 提取深度连通区域并绘制多边形（保留原有逻辑）
//     std::unordered_map<float, std::vector<cv::Point>> depthRegions;
//     for (int y = 0; y < zbuffer.rows; ++y) {
//         for (int x = 0; x < zbuffer.cols; ++x) {
//             float depth = zbuffer.at<float>(y, x);
//             if (depth == FLT_MAX) continue;
//             depthRegions[depth].emplace_back(x, y);
//         }
//     }

//     cv::Mat regionMask = cv::Mat::zeros(zbuffer.size(), CV_8U);
//     for (const auto& [current_depth, points] : depthRegions) {
//         if (points.size() < 4) continue;
//         regionMask.setTo(0);
        
//         bool is_valid_depth = false;
//         {
//             std::lock_guard<std::mutex> lock(global.mtx_surface_depths);
//             for (float valid_depth : global.surface_depths) {
//                 if (std::abs(current_depth - valid_depth) < 0.0001 ) {
//                     is_valid_depth = true;
//                     break;
//                 }
//             }
//         }
//         if (is_valid_depth) continue;

//         for (const auto& p : points) {
//             regionMask.at<uchar>(p) = 255;
//         }

//         std::vector<std::vector<cv::Point>> contours;
//         cv::findContours(regionMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
//         std::unordered_map<int,std::vector<int>> idx_score;

//         std::unordered_map<int,std::vector<float>> idx_score_with_record;

//         for (const auto& contour : contours) {
//             if (contour.size() < 4) continue;
//             double epsilon = 0.01 * cv::arcLength(contour, true);
//             std::vector<cv::Point> polygon;
//             cv::approxPolyDP(contour, polygon, epsilon, true);
//             if (polygon.size() == 4) { // 我们只对四边形感兴趣
//                 // polygon 这个 vector 里的四个元素 (polygon[0], polygon[1], polygon[2], polygon[3])
//                 // 就是这个四边形的四个角点
//                 for (int i = 0; i < polygon.size(); i++) {                
//                 }
//             }
//         }
//     }
//     return allow_idx_lists;
// }




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

        const float IOU_THRESHOLD = 0.02f;
        global.front_2d_point_lists.clear();
        global.side_2d_points_lists.clear(); 
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
        filterEmptyBoxes(zbuffer,image_,global.is_move);

        // if (global.k == 0){
        // save_zbuffer_image(image_, 
        //         zbuffer
        //         ); };      
        // if (global.k == 100){
        // save_zbuffer_image(image_, 
        //         zbuffer
        //         ); };
        // if (global.k == 200){
        // save_zbuffer_image(image_, 
        //         zbuffer
        //         ); };
        // if (global.k == 300){
        // save_zbuffer_image(image_, 
        //         zbuffer
        //         ); };
        // if (global.k == 400){
        // save_zbuffer_image(image_, 
        //         zbuffer
        //     ); };
        // if (global.k == 500){
        // save_zbuffer_image(image_, 
        //         zbuffer
        //         ); };
        // if (global.k == 600){
        // save_zbuffer_image(image_, 
        //         zbuffer
        //         ); };
        // if (global.k == 700){
        // save_zbuffer_image(image_, 
        //         zbuffer
        //         ); };
        // global.k += 1;
        // std::cout << global.k << std::endl;

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
    ros::init(argc, argv, "t2dto3d_zbuffer_ld_node");
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