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
#include <map>

#include "package/method_math.h"
#include "package/occlusion_handing.h"     
#include "package/world_to_camera.h"  
#include "package/BaseMoveController.h"

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

        // 设置工作空间目录 -> 清空txt文档 -> 确定sh文件内容 —> 开始自动录制数据集
        workspace_path = "/home/h/RC2026/world_ws13.5";
        datasets_path = "/home/h/视频/datasets_blue_200";

        num = Ten::_OCCLUSION_HANDING_.get_txt_flag(workspace_path + "/src/zwei/map1_add");
        Ten::_OCCLUSION_HANDING_.write_txt_flag(num, workspace_path + "/src/zwei/map1_add");
    }
    std::string workspace_path;
    std::string datasets_path;

    std::vector<Ten::box> box_lists;

    cv::Mat _K;
    cv::Mat _distCoeffs;

    cv::Mat _image;
    cv::Mat debug_image;
    cv::Mat debug_best_roi_image = cv::Mat::zeros(480, 640, CV_8UC3);
    std::mutex _mtx_image;

    image_transport::Publisher zbuffer_pub;

    std::string num;

    nav_msgs::Odometry::ConstPtr robot_pose;  // 缓存位姿数据
    bool pose_updated = false;              // 位姿更新标记
    bool image_updated = false;             // 图像更新标记
    std::mutex data_mutex;                  // 互斥锁，防止数据竞争

    std::vector<cv::Point2d> save_datasets_pos;     // 记录保存图像时点的位置， 以防止在同一个位置 重复拍摄
}global;

void zbuffer_process(bool is_recording_dataset = false)
{
    if (!global.pose_updated || !global.image_updated)
    {
        ROS_DEBUG("数据未更新，跳过处理");
        return;
    }

    Ten::XYZRPY tf;
    {
        std::lock_guard<std::mutex> lock(global.data_mutex);
        tf = Ten::Nav_Odometrytoxyzrpy(*global.robot_pose);
        tf._xyz._z = tf._xyz._z + 0.05;
    }

    // 雷达到相机的固定变换
    Ten::XYZRPY wt;
    wt._xyz._z = 1.3;
    wt._rpy._roll = - M_PI / 2;
    wt._rpy._pitch = M_PI / 2;
    Eigen::Matrix4d lidar_to_camera = worldtocurrent(wt._xyz, wt._rpy);

    Ten::_CAMERA_TRANSFORMATION_.camerainfo_.set_Extrinsic_Matrix(lidar_to_camera);        // 设置雷达到相机外参
    Ten::_CAMERA_TRANSFORMATION_.camerainfo_.set_K(global._K);

    Ten::XYZRPY world2toworld1;
    world2toworld1._rpy._yaw = - M_PI / 2;
    Ten::_CAMERA_TRANSFORMATION_.set_world2toworld1(world2toworld1);
    Ten::_CAMERA_TRANSFORMATION_.set_worldtolidar(tf);

    Ten::Ten_camerainfo cccc;

    Eigen::Matrix4d world_to_camera = Ten::_CAMERA_TRANSFORMATION_.pcl_transform_world_to_camera(Ten::_INIT_3D_BOX_.pcl_LM_plum_object_points_, 
            Ten::_INIT_3D_BOX_.pcl_C_plum_object_points_, Ten::_INIT_3D_BOX_.object_plum_2d_points_);
    cccc.set_Extrinsic_Matrix(world_to_camera);

    Ten::_INIT_3D_BOX_.pcl_to_C();

    int exist_boxes[12] = {1,1,1,1,1,1,1,1,1,1,1,1};
    int interested_boxes[12] = {1,1,1,1,1,1,1,1,1,1,1,1};
    Ten::_OCCLUSION_HANDING_.set_exist_boxes(exist_boxes);
    Ten::_OCCLUSION_HANDING_.set_interested_boxes(interested_boxes);

    global.debug_image = Ten::_OCCLUSION_HANDING_.update_debug_image(
        global._image,
        Ten::_INIT_3D_BOX_.object_plum_2d_points_
    );

    Ten::_OCCLUSION_HANDING_.set_debug_roi_image(Ten::_INIT_3D_BOX_.box_lists_,global.debug_best_roi_image);
    
    // 录制数据集部分
    if (is_recording_dataset)
    {
        static bool print_num = true;
        if (print_num)
        {
            print_num = false;
            std::cout << "atoi(global.num.c_str()) : " << atoi(global.num.c_str()) << std::endl;
        }

        static int save_count_sta = Ten::_OCCLUSION_HANDING_.getMaxImageNumber(global.datasets_path + "/global_images");
        
        static Ten::XYZRPY last_tf;         // 上一帧的tf
        static Ten::XYZRPY total_tf;        // 累积的tf差值
        bool place_update = Ten::_OCCLUSION_HANDING_.is_pos_update(tf, last_tf, total_tf);      // 判断位置是否移动到一定程度
        bool tf_update = Ten::_OCCLUSION_HANDING_.checkPointDistance(global.save_datasets_pos,cv::Point2d(tf._xyz._x, tf._xyz._y));     // 判断是否在同一个位置 重复拍摄

        static int start_update = 0;        // 起始处等5帧， 防止保存空图像
        if (place_update && tf_update && start_update > 5)
        { 
            static int save_count = 1;
            bool is_update = true;      // set_box_lists_ 内部判断是否所有方块都在画面中
            std::cout << "-------------------------------------------------------" << std::endl;
            Ten::_OCCLUSION_HANDING_.set_box_lists_(global._image,  Ten::_INIT_3D_BOX_.C_object_plum_points_, 
            Ten::_INIT_3D_BOX_.object_plum_2d_points_ ,Ten::_INIT_3D_BOX_.box_lists_,is_update);
            std::cout << "-------------------------------------------------------" << std::endl;
            if (is_update)
            {
                global.save_datasets_pos.push_back(cv::Point2d(tf._xyz._x, tf._xyz._y));
                Ten::_OCCLUSION_HANDING_.save_dataset(
                    Ten::_INIT_3D_BOX_.box_lists_,
                    global._image,
                    Ten::_OCCLUSION_HANDING_.processMapFile(global.workspace_path + "/src/zwei/map1_add/txt",atoi(global.num.c_str())),
                    global.datasets_path,
                    cccc.rvec(),
                    cccc.tvec(),
                    save_count_sta + save_count
                );
                save_count += 1;
            }
        }
        start_update += 1;
    }
}

// 回调函数1：处理/robot_pose话题
void odomCallback(const nav_msgs::Odometry::ConstPtr& msg)
{
    std::lock_guard<std::mutex> lock(global.data_mutex); // 加锁保证线程安全
    global.robot_pose = msg;
    global.pose_updated = true; // 标记位姿已更新
}

// 回调函数2：处理/kinect2/hd/image_color_rect话题
void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    std::lock_guard<std::mutex> lock(global.data_mutex); // 加锁保证线程安全
    try
    {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        global._image =  cv_ptr->image;
        global.image_updated = true; // 标记图像已更新
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

void pub_color_image
(
    const cv::Mat& color_image,
    const std::string topic_name = "/debug_images"
)
{
    static std::map<std::string, ros::Publisher> pubs;
    auto it = pubs.find(topic_name);
    if (it == pubs.end())
    {
        ros::NodeHandle nh;
        it = pubs.emplace(topic_name,
            nh.advertise<sensor_msgs::Image>(topic_name, 10)).first;
    }
    if (color_image.empty() || color_image.channels() != 3) return;

    cv_bridge::CvImage cv_msg;
    cv_msg.header.stamp = ros::Time::now();
    cv_msg.encoding = sensor_msgs::image_encodings::BGR8;
    cv_msg.image = color_image;
    it->second.publish(cv_msg.toImageMsg());
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "zbuffer_func_node");
    ros::NodeHandle nh("~");

    // 读取 ROS 参数（从 launch 文件中配置）
    bool auto_mode = false;
    nh.param("auto_mode", auto_mode, false);

    ROS_INFO("auto_mode: %s", auto_mode ? "true (运动+录制)" : "false (仅感知)");

    ros::NodeHandle nh_public;

    // 运动控制（auto_mode 时启用）
    std::unique_ptr<BaseMoveController> move_controller;
    if (auto_mode)
    {
        move_controller = std::make_unique<BaseMoveController>(nh_public);
        if (!move_controller->start()) {
            ROS_ERROR("Failed to start BaseMoveController. Shutting down.");
            return 1;
        }
        ROS_INFO("BaseMoveController started successfully.");
    }
    else
    {
        ROS_INFO("Auto mode disabled — perception only.");
    }

    ros::Publisher cmd_vel_pub = nh_public.advertise<geometry_msgs::Twist>("/cmd_vel", 10);

    std::vector<std::thread> workers;
    
    workers.emplace_back(worker_task1, nh_public);
    workers.emplace_back(worker_task2, nh_public);

    image_transport::ImageTransport it(nh_public);

    ros::Rate rate(30);
    while(ros::ok())
    {
        // auto_mode 下遍历完成则退出
        if (auto_mode && move_controller->isCompleted()) {
            ros::shutdown();
            break;
        }

        zbuffer_process(auto_mode);

        // 调试发布（在 zbuffer_process 之后，确保图像已更新）
        {
            std::lock_guard<std::mutex> lock(global._mtx_image);
            pub_color_image(global.debug_image, "pub_image_topic");
            pub_color_image(global.debug_best_roi_image, "/zbuffer_visualization");
        }
        rate.sleep();
    }

    for (auto& worker : workers) {
        worker.join();
    }
    return 0;
}