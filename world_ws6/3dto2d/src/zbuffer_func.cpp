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
#include "package/zbuffer_simplify.h"     
#include "package/world_to_camera.h"  

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


    nav_msgs::Odometry::ConstPtr robot_pose;  // 缓存位姿数据
    bool pose_updated = false;              // 位姿更新标记
    bool image_updated = false;             // 图像更新标记
    std::mutex data_mutex;                  // 互斥锁，防止数据竞争
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
        float x = tf._xyz._x;
        tf._xyz._x = -tf._xyz._y;
        tf._xyz._y = x;
    }

    Ten::XYZRPY wt;
    wt._xyz._z = 1.25;
    wt._rpy._roll = - M_PI / 2;
    //wt._rpy._yaw = -M_PI / 2;
    Eigen::Matrix4d transform_matrix = worldtocurrent(wt._xyz, wt._rpy);
    
    Ten::_CAMERA_TRANSFORMATION_.camerainfo_.set_Extrinsic_Matrix(transform_matrix);
    Ten::_CAMERA_TRANSFORMATION_.camerainfo_.set_K(global._K);

    std::cout << "tf.x: " << tf._xyz._x  << std::endl;
    std::cout << "tf.y: " << tf._xyz._y  << std::endl;
    std::cout << "tf.z: " << tf._xyz._z  << std::endl;

    Ten::_CAMERA_TRANSFORMATION_.set_worldtolidar(tf);
    Ten::_CAMERA_TRANSFORMATION_.pcl_transform_world_to_camera(Ten::_INIT_3D_BOX_.pcl_LM_plum_object_points_, 
    Ten::_INIT_3D_BOX_.pcl_C_plum_object_points_, Ten::_INIT_3D_BOX_.object_plum_2d_points_);
    Ten::_INIT_3D_BOX_.pcl_to_C();

    Ten::_ZBUFFER_SIMPLIFY_.set_box_lists_(global._image,  Ten::_INIT_3D_BOX_.C_object_plum_points_, 
        Ten::_INIT_3D_BOX_.object_plum_2d_points_ ,Ten::_INIT_3D_BOX_.box_lists_);

    
    global.debug_image = Ten::_ZBUFFER_SIMPLIFY_.update_debug_image(
        global._image,
        Ten::_INIT_3D_BOX_.object_plum_2d_points_
    );

    Ten::_ZBUFFER_SIMPLIFY_.set_debug_roi_image(Ten::_INIT_3D_BOX_.box_lists_, global.debug_best_roi_image);
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
    std::vector<std::thread> workers;
    workers.emplace_back(worker_task1, nh);
    workers.emplace_back(worker_task2, nh);

    image_transport::ImageTransport it(nh);
    image_transport::Publisher debug_image_pub = it.advertise("pub_image_topic", 2);
    image_transport::Publisher debug_roi_pub = it.advertise("/zbuffer_visualization", 30);

    ros::Rate rate(10);
    while(ros::ok())
    {
        sensor_msgs::ImagePtr msg;
        sensor_msgs::ImagePtr roi_msg;
        {
            std::lock_guard<std::mutex> lock(global._mtx_image);
            msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", global.debug_image).toImageMsg();
            roi_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", global.debug_best_roi_image).toImageMsg();
        }
        debug_image_pub.publish(msg);
        debug_roi_pub.publish(roi_msg);
        zbuffer_process();
        // std::cout << "publish success" << std::endl;
        rate.sleep();
    }

    for (auto& worker : workers) {
        worker.join();
    }
    return 0;
}