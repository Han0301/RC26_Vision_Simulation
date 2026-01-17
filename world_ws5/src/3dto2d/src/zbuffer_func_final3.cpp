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

#include "hsv_filterEmptyBoxes.cpp"
#include "zbuffer_occlsion.cpp"
#include "zbuffer_roi_debug2.cpp"
#include "method_math.cpp"

#include "zbuffer_simplify.cpp"
#define _L_ 1.2
#define _H_ 0.2
#define _ly1_ 0.425
#define _ly2_ 0.775
#define _lx1_ 0.425
#define _lh_ 0.35 
#define _X_  3.17             
#define _Y_  1.2            

Ten::hsv_filterEmptyBoxes hsv_fliter;
Ten::zbuffer_occlsion zbuffer_updater;
Ten::zbuffer_roi_debug zbuffer_roi_debuger;

Ten::zbuffer_simplify zbuffer_simplier;
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
        for(int i =0;i <12;i ++){
            box_lists.push_back({
                i + 1,
                cv::Mat(),
                0,
                0,
                -1
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


    std::vector<Ten::box> box_lists;



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

std::vector<std::vector<Ten::surface_2d_point>>zbuffer_initialization(
    std::vector<G::box> boxes,
    std::vector<int>allow_idx_lists,
    const float camera_y,
    const float camera_x,     // 机器人的x,y与这相反
    const float camera_z
){
    std::vector<Ten::surface_2d_point> front_2d_points_lists;
    std::vector<Ten::surface_2d_point> up_2d_points_lists;
    std::vector<Ten::surface_2d_point> side_2d_points_lists;
    std::vector<std::vector<Ten::surface_2d_point>> box_2d_points_lists;
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

cv::Mat update_debug_image(
    cv::Mat image,
    const std::vector<std::vector<Ten::surface_2d_point>>& object_2d_points_lists
){
    // 1. 检查输入有效性
    if (image.empty()) {
        ROS_WARN("Image is empty, skip draw");
        return cv::Mat();
    }

    std::vector<Ten::surface_2d_point> object_front_2d = object_2d_points_lists[0];
    std::vector<Ten::surface_2d_point> object_side_2d = object_2d_points_lists[1];
    std::vector<Ten::surface_2d_point> object_up_2d = object_2d_points_lists[2];

    for (size_t i = 0; i < object_side_2d.size(); i++) {
        if (i >= object_front_2d.size() || i >= object_up_2d.size()) break;
        
        auto& o_front = object_front_2d[i];
        auto& o_side = object_side_2d[i];
        auto& o_up = object_up_2d[i];

        cv::line(image, o_front.left_up, o_front.right_up, cv::Scalar(0,255,0), 2, cv::LINE_AA);
        cv::line(image, o_front.right_up, o_front.right_down, cv::Scalar(0,255,0), 2, cv::LINE_AA);
        cv::line(image, o_front.right_down, o_front.left_down, cv::Scalar(0,255,0), 2, cv::LINE_AA);
        cv::line(image, o_front.left_down, o_front.left_up, cv::Scalar(0,255,0), 2, cv::LINE_AA);

        cv::line(image, o_side.left_up, o_side.right_up, cv::Scalar(0,255,0), 2, cv::LINE_AA);
        cv::line(image, o_side.right_up, o_side.right_down, cv::Scalar(0,255,0), 2, cv::LINE_AA);
        cv::line(image, o_side.right_down, o_side.left_down, cv::Scalar(0,255,0), 2, cv::LINE_AA);
        cv::line(image, o_side.left_down, o_side.left_up, cv::Scalar(0,255,0), 2, cv::LINE_AA);

        cv::line(image, o_up.left_up, o_up.right_up, cv::Scalar(0,255,0), 2, cv::LINE_AA);
        cv::line(image, o_up.right_up, o_up.right_down, cv::Scalar(0,255,0), 2, cv::LINE_AA);
        cv::line(image, o_up.right_down, o_up.left_down, cv::Scalar(0,255,0), 2, cv::LINE_AA);
        cv::line(image, o_up.left_down, o_up.left_up, cv::Scalar(0,255,0), 2, cv::LINE_AA);
}
return image;

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

        std::vector<int> allow_idx_lists = {1,2,3,4,5,6,7,8,9,10,11,12};
        std::vector<std::vector<Ten::surface_2d_point>>  box_2d_points_lists = zbuffer_initialization(
            global.boxes, allow_idx_lists, global.x, global.y, 1.3);
        std::vector<std::vector<Ten::surface_2d_point>>  plum_box_2d_points_lists = zbuffer_initialization(
            global.plum_boxes, {1,2,3,4,5,6,7,8,9,10,11,12}, global.x, global.y, 1.3);




        // int exist_boxes[12] = {1,0,0,1,0,0,1,1,1,1,1,1};
        int exist_boxes[12] = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
        zbuffer_simplier.set_exist_boxes(exist_boxes);

        auto start1 = std::chrono::high_resolution_clock::now();
        zbuffer_simplier.set_box_lists_(image_,box_2d_points_lists,plum_box_2d_points_lists,global.box_lists);
        auto end1 = std::chrono::high_resolution_clock::now();

        auto start2 = std::chrono::high_resolution_clock::now();
        cv::Mat debug_best_roi_image = cv::Mat::zeros(480, 640, CV_8UC3);
        zbuffer_simplier.set_debug_roi_image(global.box_lists,debug_best_roi_image);
        auto end2 = std::chrono::high_resolution_clock::now();

        auto start3 = std::chrono::high_resolution_clock::now();
        static bool is_get_sta_hsv = false;
        if (!is_get_sta_hsv){
            is_get_sta_hsv = true;
            zbuffer_simplier.set_standard_hsv_(global.box_lists);
        }

        zbuffer_simplier.set_HSV_exist_boxes_(global.box_lists);
        auto end3 = std::chrono::high_resolution_clock::now();
        
        std::cout << "zbuffer_flag: ";
        for (int i = 0; i< 12; i++){
            std::cout << global.box_lists[i].zbuffer_flag << "  ";
        }
        std::cout << "exist_flag: ";
        for (int i = 0; i< 12; i++){
            std::cout << global.box_lists[i].exist_flag << "  ";
        }
        std::cout << std::endl;
        auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds> (end1 - start1);
        auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds> (end2 - start2);
        auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds> (end2 - start3);
        ROS_INFO("set_box_lists_.count: %ldms", duration1);
        ROS_INFO("set_debug_roi_image.count: %ldms", duration2);
        ROS_INFO("set_HSV_exist_boxes_.count: %ldms", duration3);

        
        cv::Mat debug_image = update_debug_image(image_,box_2d_points_lists);

        // 更新全局图像
        {
            std::lock_guard<std::mutex> lock(global._mtx_image);
            debug_image.copyTo(global._image);
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
    ros::init(argc, argv, "zbuffer_func_final3_node");
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