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
        num = Ten::_OCCLUSION_HANDING_.get_txt_flag("/home/h/RC2026/world_ws13/src/zwei/map1_add");
        Ten::_OCCLUSION_HANDING_.write_txt_flag(num, "/home/h/RC2026/world_ws13/src/zwei/map1_add");
    }

    std::vector<Ten::box> box_lists;

    bool is_move = true;

    cv::Mat _K;
    cv::Mat _distCoeffs;

    cv::Mat _image;
    cv::Mat debug_image;
    cv::Mat debug_best_roi_image = cv::Mat::zeros(480, 640, CV_8UC3);;
    std::mutex _mtx_image;

    image_transport::Publisher zbuffer_pub;

    std::string num;

    nav_msgs::Odometry::ConstPtr robot_pose;  // 缓存位姿数据
    bool pose_updated = false;              // 位姿更新标记
    bool image_updated = false;             // 图像更新标记
    std::mutex data_mutex;                  // 互斥锁，防止数据竞争

    std::vector<cv::Point2d> save_datasets_pos;

    // 🔥 新增：长时间静止检测变量
    cv::Point2d last_check_pos;       // 上一次检测时的位置
    std::chrono::steady_clock::time_point static_start_time; // 开始静止的时间
    bool first_pos_recorded = false;   // 是否记录了第一个位置
    const double MAX_STATIC_TIME = 16.0; // 最大静止时间：60秒
    const double STATIC_DIST_THRESHOLD = 0.1; // 静止距离阈值：0.1米
}global;

void zbuffer_process()
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
        tf._xyz._z = tf._xyz._z - 0.05;
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

    static bool print_num = true;
    if (print_num)
    {
        print_num = false;
        std::cout << "atoi(global.num.c_str()) : " << atoi(global.num.c_str()) << std::endl;
    }
    static int save_count_sta = Ten::_OCCLUSION_HANDING_.getMaxImageNumber("/home/h/视频/Datasets_new/global_images");
    
    static Ten::XYZRPY last_tf;         // 上一帧的tf
    static Ten::XYZRPY total_tf;        // 累积的tf差值
    bool place_update = Ten::_OCCLUSION_HANDING_.is_pos_update(tf, last_tf, total_tf);      // 判断位置是否移动到一定程度
    bool tf_update = Ten::_OCCLUSION_HANDING_.checkPointDistance(global.save_datasets_pos,cv::Point2d(tf._xyz._x, tf._xyz._y));

    static int start_update = 0;
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
                Ten::_OCCLUSION_HANDING_.processMapFile("/home/h/RC2026/world_ws13/src/zwei/map1_add/txt",atoi(global.num.c_str())),
                "/home/h/视频/Datasets_new",
                cccc.rvec(),
                cccc.tvec(),
                save_count_sta + save_count
            );
            save_count += 1;
        }
    }
    start_update += 1;
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

int main(int argc, char **argv)
{
    ros::init(argc, argv, "zbuffer_func_node");
    ros::NodeHandle nh;

    BaseMoveController move_controller(nh);
    if (!move_controller.start()) {
        ROS_ERROR("Failed to start BaseMoveController. Shutting down.");
        return 1;
    }
    ROS_INFO("BaseMoveController started successfully.");
    ros::Publisher cmd_vel_pub = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 10);

    std::vector<std::thread> workers;
    
    workers.emplace_back(worker_task1, nh);
    workers.emplace_back(worker_task2, nh);

    image_transport::ImageTransport it(nh);
    // image_transport::Publisher debug_image_pub = it.advertise("pub_image_topic", 2);
    // image_transport::Publisher debug_roi_pub = it.advertise("/zbuffer_visualization", 30);

    ros::Rate rate(10);
    while(ros::ok())
    {
        std::cout << "move_controller.isCompleted(): " << move_controller.isCompleted() << std::endl;
        if (move_controller.isCompleted()) {
            ros::shutdown();
            break;
        }

        // sensor_msgs::ImagePtr msg;
        // sensor_msgs::ImagePtr roi_msg;
        // {
        //     std::lock_guard<std::mutex> lock(global._mtx_image);
        //     msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", global.debug_image).toImageMsg();
        //     roi_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", global.debug_best_roi_image).toImageMsg();
        // }
        
        // debug_image_pub.publish(msg);
        // debug_roi_pub.publish(roi_msg);
        zbuffer_process();
        rate.sleep();
    }

    for (auto& worker : workers) {
        worker.join();
    }
    return 0;
}