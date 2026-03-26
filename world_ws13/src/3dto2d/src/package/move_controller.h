#ifndef _MOVE_CONTROLLER_H_
#define _MOVE_CONTROLLER_H_

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
#include <regex>
#include <filesystem>

namespace Ten
{
    class Ten_move_controller
    {
    public:
        void move_controller(ros::Publisher &cmd_vel_pub);
        void move_controller2(ros::Publisher &cmd_vel_pub);

    private:
    };

    extern Ten::Ten_move_controller _MOVE_CONTROLLER_;

}

#endif