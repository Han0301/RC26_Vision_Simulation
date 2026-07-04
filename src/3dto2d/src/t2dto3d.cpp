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

// std::vector<idx_box> filterBoxesByProjected3DXY(const std::vector<idx_box>& boxes) {
//     std::vector<idx_box> filtered;
    
//     // 提取相机内参
//     double fx = _K.at<double>(0, 0);  // 焦距X
//     double fy = _K.at<double>(1, 1);  // 焦距Y
//     double cx = _K.at<double>(0, 2);  // 主点X
//     double cy = _K.at<double>(1, 2);  // 主点Y

//     // 锁定获取**相机→世界**的最新位姿
//     cv::Mat rvec, tvec;
//     {
//         std::lock_guard<std::mutex> lock(_mtx_tf);
//         rvec = _rvec.clone();  // 相机→世界的旋转向量
//         tvec = _tvec.clone();  // 相机→世界的平移向量
//     }

//     // 将旋转向量转为旋转矩阵
//     cv::Mat R_cam_world;
//     cv::Rodrigues(rvec, R_cam_world);  


//     for (const auto& box : boxes) {
//         int idx = box.idx;
//         // 计算框的像素中心
//         float u_center = (box.box_x1 + box.box_x2) / 2.0f;  
//         float v_center = (box.box_y1 + box.box_y2) / 2.0f;  


//         // 获取框对应的3D点组
//         int start_idx = (idx - 1) * 4;  
//         std::vector<cv::Point3f> world_group_points;
//         for (int i = 0; i < 4; i++) {
//             world_group_points.push_back(_objectPoints[start_idx + i]);
//         }

//         // 计算相机坐标系的平均深度（世界点→相机点）
//         double avg_depth = 0.0;
//         for (const auto& wp : world_group_points) {
//             cv::Mat wp_mat = (cv::Mat_<double>(3, 1) << wp.x, wp.y, wp.z);
//             // 公式：P_camera = R_cam_world * P_world + tvec（世界点→相机点）
//             cv::Mat cp_mat = R_cam_world * wp_mat + tvec;  
//             avg_depth += cp_mat.at<double>(2);  // 相机坐标系的Z值（深度，应为正）
//         }
//         avg_depth /= 4.0;  // 平均深度（相机坐标系）


//         // 反投影像素中心到相机坐标系的三维点
//         // 公式：X_camera = (u - cx) * avg_depth / fx；Y_camera = (v - cy) * avg_depth / fy
//         double X_camera = (u_center - cx) * avg_depth / fx;
//         double Y_camera = (v_center - cy) * avg_depth / fy;
//         double Z_camera = avg_depth;  // 相机坐标系的Z值（深度）

//         // 转换回世界坐标系（相机点→世界点）
//         cv::Mat camera_pt_mat = (cv::Mat_<double>(3, 1) << X_camera, Y_camera, Z_camera);
//         // 公式：P_world = R_cam_worldᵀ * (P_camera - tvec)（相机点→世界点）
//         cv::Mat world_pt_mat = R_cam_world.t() * (camera_pt_mat - tvec);  
//         double X_world = world_pt_mat.at<double>(0);  // 世界坐标系X
//         double Y_world = world_pt_mat.at<double>(1);  // 世界坐标系Y
//         double Z_world = world_pt_mat.at<double>(2);  // 世界坐标系Z

//         // 步骤6：检查世界坐标系的X/Y是否在范围内
//         const auto& range = standard_3d_x_y_lists[idx - 1];  
//         bool in_range = (X_world >= range.x_min && X_world <= range.x_max) && 
//                         (Y_world >= range.y_min && Y_world <= range.y_max);

//         // 调试打印（增加深度验证）
//         std::cout << "[Filter] Box idx: " << idx 
//                   << " | Pixel Center: (" << std::fixed << std::setprecision(2) << u_center << ", " << v_center << ")"
//                   << " | Camera Depth: " << avg_depth << " | World Center: (" << X_world << ", " << Y_world << ", " << Z_world << ")"
//                   << " | Range: x[" << range.x_min << ", " << range.x_max << "] y[" << range.y_min << ", " << range.y_max << "]"
//                   << " | Result: " << (in_range ? "Keep" : "Filter") << std::endl;


//         // 保留符合条件的框
//         if (in_range) {
//             filtered.push_back(box);
//         }
//     }

//     return filtered;
// }
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
        // 4. 检查中心点的颜色
        if (center.x >= 0 && center.x < image.cols && center.y >= 0 && center.y < image.rows) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(center.y, center.x);
                                std::cout << " | BGR: (" << static_cast<int>(pixel[0]) << ", " 
                << static_cast<int>(pixel[1]) << ", " 
                << static_cast<int>(pixel[2]) << ")" << std::endl;
            if (isPixelTargetColor(pixel, TargetColor::Green)) {
                greenCount++;
                std::cout << ", green";
            } else if (isPixelTargetColor(pixel, TargetColor::Yellow)) {
                yellowCount++;
                std::cout << ", yellow";
            }else if (isPixelTargetColor(pixel, TargetColor::other)) {
                    othercount++;
                    std::cout << ", other";
                }
        }

        // 绘制四个角点（红色实心圆）
        // for (const auto& corner : corners) {
        //     // 越界保护：避免点超出图像范围
        //     if (corner.x >= 0 && corner.x < image.cols && corner.y >= 0 && corner.y < image.rows) {
        //         cv::circle(image, corner, point_radius, corner_color, cv::FILLED);  // FILLED表示实心
        //     }
        // }

        // // 绘制中心点（蓝色实心圆）
        // if (center.x >= 0 && center.x < image.cols && center.y >= 0 && center.y < image.rows) {
        //     cv::circle(image, center, point_radius, center_color, cv::FILLED);
        // }
        // 5. 筛选：绿色/黄色点总数≤1则保留
        if (greenCount + yellowCount < 1) {
            filteredBoxes.push_back(box);
        }
    }

    return filteredBoxes;
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
    ROS_INFO_STREAM("odom");
    ROS_INFO_STREAM("x=" << msg->pose.pose.position.x
                   << "y=" << msg->pose.pose.position.y
                   << "z=" << msg->pose.pose.position.z);
    ROS_INFO_STREAM("x=" << msg->pose.pose.orientation.x
                   << "y=" << msg->pose.pose.orientation.y
                   << "z=" << msg->pose.pose.orientation.z
                   << "w=" << msg->pose.pose.orientation.w);
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

std::vector<G::idx_box> filterBoxesFromBack(const std::vector<G::idx_box>& boxes, float threshold) {
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
                (global.imagePoints[i].x > 0 && global.imagePoints[i].x < image_.cols + 40 && 
                 global.imagePoints[i].y > 0 && global.imagePoints[i].y < image_.rows + 30) &&
                (global.imagePoints[i+2].x > 0 && global.imagePoints[i+2].x < image_.cols + 40 && 
                 global.imagePoints[i+2].y > 0 && global.imagePoints[i+2].y < image_.rows +30);
            
            if (inFrame) {
                if (std::find(exclude_idxs.begin(), exclude_idxs.end(), i/4+1) != exclude_idxs.end()) {
                    continue; // 跳过排除的idx
                    }
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

        // 检查侧面的框的对角点是否在画面内        
        // for (int i = 0; i < global.side_imagePoints.size(); i += 4) {                       
        //     cv::Point3f world_side_point = global._c_side_objectPoints[i];
        //     float dist = std::sqrt(
        //         world_side_point.x * world_side_point.x +
        //         world_side_point.y * world_side_point.y +
        //         world_side_point.z * world_side_point.z
        //     );
            
            // 检查框的对角点是否在画面内
        //     bool inFrame = 
        //         (global.side_imagePoints[i].x > 0 && global.side_imagePoints[i].x < image_.cols + 40&& 
        //          global.side_imagePoints[i].y > 0 && global.side_imagePoints[i].y < image_.rows + 30) &&
        //         (global.side_imagePoints[i+2].x > 0 && global.side_imagePoints[i+2].x < image_.cols +40 && 
        //          global.side_imagePoints[i+2].y > 0 && global.side_imagePoints[i+2].y < image_.rows + 30);
            
        //     if (inFrame) {
        //         if (std::find(exclude_idxs.begin(), exclude_idxs.end(), i/4+1) != exclude_idxs.end()) {
        //             continue; // 跳过排除的idx
        //             }
        //         global.idx_box_side_lists.push_back({
        //             i/4 + 1,
        //             global.side_imagePoints[i].x,
        //             global.side_imagePoints[i].y,
        //             global.side_imagePoints[i+2].x,
        //             global.side_imagePoints[i+2].y,
        //             dist
        //         });
        //     }
        // }        

        // 2. 筛选正面的框（去除IOU大且远的）
        //global.idx_box_lists = global.filterBoxesByProjected3DXY(global.idx_box_lists);
        //global.idx_box_lists = filterBoxesByColor(image_, global.idx_box_lists);
        //std::vector<G::idx_box> filtered_boxes = filterBoxesFromBack(global.idx_box_lists, IOU_THRESHOLD);

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

        // for (const auto& box : global.idx_box_side_lists) {
            
        //     // 绘制文本（组号+距离）
        //     // std::string text = "idx:" + std::to_string(box.idx) + " d:" + std::to_string(box.distance).substr(0,4);
        //     std::string text = "side_idx:" + std::to_string(box.idx);
        //     cv::putText(image_, text, 
        //                 cv::Point(cvRound((box.box_x1 + box.box_x2)/2), cvRound((box.box_y1 + box.box_y2)/2)),
        //                 cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
        //     cv::line(image_, cv::Point(cvRound(box.box_x1), cvRound(box.box_y1)), cv::Point(cvRound(box.box_x2), cvRound(box.box_y1)), cv::Scalar(0,0,255),2);
        //     cv::line(image_, cv::Point(cvRound(box.box_x2), cvRound(box.box_y1)), cv::Point(cvRound(box.box_x2), cvRound(box.box_y2)), cv::Scalar(0,0,255),2);
        //     cv::line(image_, cv::Point(cvRound(box.box_x2), cvRound(box.box_y2)), cv::Point(cvRound(box.box_x1), cvRound(box.box_y2)), cv::Scalar(0,0,255),2);
        //     cv::line(image_, cv::Point(cvRound(box.box_x1), cvRound(box.box_y2)), cv::Point(cvRound(box.box_x1), cvRound(box.box_y1)), cv::Scalar(0,0,255),2);
        //     // cv::line(image_, cv::Point(cvRound(box.box_x1-3), cvRound(box.box_y1-5)), cv::Point(cvRound(box.box_x2+5), cvRound(box.box_y1-5)), cv::Scalar(0,0,255),2);
        //     // cv::line(image_, cv::Point(cvRound(box.box_x2+5), cvRound(box.box_y1-5)), cv::Point(cvRound(box.box_x2+5), cvRound(box.box_y2-3)), cv::Scalar(0,0,255),2);
        //     // cv::line(image_, cv::Point(cvRound(box.box_x2+5), cvRound(box.box_y2-3)), cv::Point(cvRound(box.box_x1-3), cvRound(box.box_y2-3)), cv::Scalar(0,0,255),2);
        //     // cv::line(image_, cv::Point(cvRound(box.box_x1-3), cvRound(box.box_y2-3)), cv::Point(cvRound(box.box_x1-3), cvRound(box.box_y1-5)), cv::Scalar(0,0,255),2);
        // }

        //std::cout << "size:" << global.idx_box_lists.size() << "  " << global.idx_box_side_lists.size() << std::endl;
        // for (size_t i = 0; i < global.idx_box_lists.size(); ++i) {
        //     // 1. 取两个容器的对应框（同一idx的主框与侧面框）
        //     const auto& main_box = global.idx_box_lists[i];      // 主框（来自filter后的结果）
        //     const auto& side_box = global.idx_box_side_lists[i]; // 侧面框（来自另一容器）

        //     // 2. 绘制合并文本（可选：显示两个框的idx）
        //     std::string text = "idx:" + std::to_string(main_box.idx) + "_side:" + std::to_string(side_box.idx);
        //     cv::putText(image_, text, 
        //                 cv::Point(cvRound((main_box.box_x1 + main_box.box_x2)/2), 
        //                         cvRound((main_box.box_y1 + main_box.box_y2)/2)),
        //                 cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);


        // }




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
    ros::init(argc, argv, "t3dto2d_node");
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