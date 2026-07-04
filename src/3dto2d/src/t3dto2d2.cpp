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
                // _objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_, -_Y_ - i*_L_ - _ly1_, _arr[j*3+i]+_lh_));
                // _objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_, -_Y_ - i*_L_ - _ly2_, _arr[j*3+i]+_lh_));
                // _objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_, -_Y_ - i*_L_ - _ly2_, _arr[j*3+i]));
                // _objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_, -_Y_ - i*_L_ - _ly1_, _arr[j*3+i]));
                // objectPoints是真实世界的3D点
                _objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_, _Y_ + i*_L_ + _ly1_, _arr[j*3+i]+_lh_));
                _objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_, _Y_ + i*_L_ + _ly2_, _arr[j*3+i]+_lh_));
                _objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_, _Y_ + i*_L_ + _ly2_, _arr[j*3+i]));
                _objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_, _Y_ + i*_L_ + _ly1_, _arr[j*3+i]));
            }
        }

        for(int i = 0; i < 48; i++)
        {
            //_c_objectPoints.push_back(cv::Point3f(-_objectPoints[i].y, 1.3 - _objectPoints[i].z, _objectPoints[i].x));
            _c_objectPoints.push_back(cv::Point3f(_objectPoints[i].y, 1.3 - _objectPoints[i].z, _objectPoints[i].x));
        }

    }
    std::vector<cv::Point3f> _objectPoints;
    std::vector<cv::Point3f> _c_objectPoints;
    cv::Mat _K;
    cv::Mat _distCoeffs;
    cv::Mat _rvec;
    cv::Mat _tvec;
    cv::Mat _image;
    // 图像和tf的互斥锁，用于多线程
    std::mutex _mtx_image;
    std::mutex _mtx_tf;
    float _arr[12] = {0.4, 0.2, 0.4, 0.2, 0.4, 0.6, 0.4, 0.6, 0.4, 0.2, 0.4, 0.2};      // z坐标信息
    int _mask[12] = {0};        // 用于判断点的有效性

}global;



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
        //cv::Mat C_world = (cv::Mat_<double>(3,1) << -transform.transform.translation.y, 0.0, transform.transform.translation.x);
        cv::Mat C_world = (cv::Mat_<double>(3,1) << -transform.transform.translation.y, transform.transform.translation.z, transform.transform.translation.x);
        std::cout << "transform.transform.translation.z" << transform.transform.translation.z << std::endl;
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
    cv::Mat C_world = (cv::Mat_<double>(3,1) << msg->pose.pose.position.x, -msg->pose.pose.position.z + 0.02, msg->pose.pose.position.y);
    std::cout<<"msg->pose.pose.position.z"<<msg->pose.pose.position.z<<std::endl;
    cv::invert(R, R_inv);
    {
        std::lock_guard<std::mutex> lock(global._mtx_tf);
        cv::Rodrigues(R, global._rvec);
        global._tvec = - (R * C_world);
        std::cout<< "-------------------------------------" << std::endl;
        for(int i = 0;  i < 12; i++)
        {
            for(int j = 0; j < 4; j++)
            {
                float x = global._objectPoints[i*4 + j].y - msg->pose.pose.position.x;
                float y = global._objectPoints[i*4 + j].x - msg->pose.pose.position.y;
                Eigen::Matrix3d mat;
                mat << cos(-(yaw-M_PI/2)), -sin(-(yaw-M_PI/2)), 0,
                sin(-(yaw-M_PI/2)), cos(-(yaw-M_PI/2)), 0,
                0, 0, 1;
                Eigen::Vector3d vec(x, y, 0);
                Eigen::Vector3d result = mat * vec;
                // std::cout<<"*******"<< i*4 + j << std::endl;
                // std::cout<< "-(yaw-M_PI/2) rad" << -(yaw-M_PI/2) << std::endl;
                // std::cout<< "-(yaw-M_PI/2) degree" << -(yaw-M_PI/2) * 180 / M_PI << std::endl;
                // std::cout<< "x" << x << std::endl;
                // std::cout<< "y" << y << std::endl;
                // std::cout<< "result[0]" << result[0] << std::endl;
                // std::cout<< "result[1]" << result[1] << std::endl;
                double k = 1000;
                if(result[0] != 0)
                {
                    k = result[1] / result[0];
                }


                if(result[1] <= 0)
                {
                    //std::cout<< "result[1] <= 0" <<std::endl;
                    global._mask[i] = 1;
                }
                else if(k < 0.2 && k > -0.2)
                {
                    //std::cout<< "k < 0.05 && k > -0.05" <<std::endl;
                    global._mask[i] = 1;
                }
                else
                {
                    global._mask[i] = 0;
                }
                
            }
        }
        std::cout<< "-------------------------------------" << std::endl;
    }

    //std::cout<< "yaw: "<< yaw <<std::endl;

}



void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    try
    {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        cv::Mat image_ =  cv_ptr->image;
        std::vector<cv::Point2f> imagePoints;
        // {
        //     std::lock_guard<std::mutex> lock(global._mtx_tf);
        //     cv::projectPoints(global._c_objectPoints, global._rvec, global._tvec, global._K, global._distCoeffs, imagePoints);
        // }
   

        {
            std::lock_guard<std::mutex> lock(global._mtx_tf);
            cv::projectPoints(global._c_objectPoints, global._rvec, global._tvec, global._K, global._distCoeffs, imagePoints);
            std::cout<<"imagePoints.size()"<<imagePoints.size()<<std::endl;
            
            for (int i = 0; i < imagePoints.size(); i += 4) 
            {
                if(global._mask[i / 4] == 1)
                {
                    std::cout<<"global._mask[i]:" <<i<<std::endl;
                    continue;
                }
                cv::line(image_, cv::Point(cvRound(imagePoints[i].x), cvRound(imagePoints[i].y)),cv::Point(cvRound(imagePoints[i+1].x), cvRound(imagePoints[i+1].y)),cv::Scalar(0, 0, 255), 2);
                cv::line(image_, cv::Point(cvRound(imagePoints[i+1].x), cvRound(imagePoints[i+1].y)),cv::Point(cvRound(imagePoints[i+2].x), cvRound(imagePoints[i+2].y)),cv::Scalar(0, 0, 255), 2);
                cv::line(image_, cv::Point(cvRound(imagePoints[i+2].x), cvRound(imagePoints[i+2].y)),cv::Point(cvRound(imagePoints[i+3].x), cvRound(imagePoints[i+3].y)),cv::Scalar(0, 0, 255), 2);
                cv::line(image_, cv::Point(cvRound(imagePoints[i+3].x), cvRound(imagePoints[i+3].y)),cv::Point(cvRound(imagePoints[i].x), cvRound(imagePoints[i].y)),cv::Scalar(0, 0, 255), 2);
            }
        }
        {
            std::lock_guard<std::mutex> lock(global._mtx_image);
            image_.copyTo(global._image);
        }


    }
    catch (cv_bridge::Exception& e)
    {
        // 处理转换异常 
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
    ros::init(argc, argv, "t3dto2d2_node");
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