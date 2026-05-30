#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <stdexcept>
#include <ros/ros.h>
#include "camera.h"
#include "./Plane_FitLocator/debug_pcl.h"
#include "./Plane_FitLocator/post_pcl.h"
#include "./Plane_FitLocator/pre_pcl.h"
#include "./Plane_FitLocator/set_pcl.h"

void test1(ros::NodeHandle& nh)
{
    Ten::Ten_camera& _CAMERA_ = Ten::Ten_camera::GetInstance();
    _CAMERA_.reset_camera_depth(640, 480,30);

    Ten::Plane_FitLocator::Ten_debug_pcl _DEBUG_PCL_;
    Ten::Plane_FitLocator::Ten_pre_pcl _PRE_PCL_;
    Ten::Plane_FitLocator::Ten_set_pcl _SET_PCL_;
    Ten::Plane_FitLocator::Ten_post_pcl _POST_PCL_;
    Ten::Plane_FitLocator::Plane_Info plane_info;

    // 点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr filter_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane_cloud_fited(new pcl::PointCloud<pcl::PointXYZ>); 

    std::vector<cv::Point2f> plane_points_2d;
    std::vector<cv::Point2f> plane_points_flited;
    // 参数
    rs2_intrinsics color_intr = _CAMERA_.get_color_intrinsics();    // 彩色内参 → 绘图用

    cv::Mat depth_show;
    cv::Mat debug_image;

    while (ros::ok())
    {
        // 设置输入图像
        Ten::camera_frame frame = _CAMERA_.camera_read_depth();
        if (frame.bgr_image.empty() || frame.depth_image.empty())
        {
            std::cout << "frame.bgr_image.empty() || frame.depth_image.empty()" << std::endl;
            continue;
        }
        // 设置点云
        cv::Rect roi = _SET_PCL_.set_roi_detect(frame.bgr_image);
        bool is_set = _SET_PCL_.set_Pcl_Cloud(frame.depth_image, color_intr, roi,input_cloud);
        if (!is_set) 
        {
            std::cout << "set_Pcl_Cloud error!" << std::endl;
            continue;
        }


        // 滤波器
        bool is_filtted = _PRE_PCL_.cloud_filter(input_cloud,filter_cloud);      // 设置点云
        if (!is_filtted)
        {
            std::cout << "cloud_filter error!" << std::endl;
            continue;
        }
        // 提取平面和中心点，法向量
        bool is_plane_flitted = _PRE_PCL_.Plane_fitter(filter_cloud, plane_cloud, plane_info);
        if (!is_plane_flitted)
        {
            std::cout << "Plane_fitter error!" << std::endl;
            continue;
        }
        

        // 方形拟合
        _POST_PCL_.compute_CenterAndNormal(plane_cloud,plane_info);
        _POST_PCL_.removePlaneNoise(plane_cloud, plane_cloud_fited);
        _POST_PCL_.set_vector_2d(plane_cloud_fited,plane_info,plane_points_2d);
        _POST_PCL_.central_range_filter(plane_points_2d,plane_points_flited);
        _POST_PCL_.set_RPY(plane_points_flited,plane_info);

        std::cout << "bias: " << -plane_info.plane_center.y() 
                  << ", yaw: " << plane_info.plane_euler._yaw << std::endl;

        // debug 部分
        cv::normalize(frame.depth_image, depth_show, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        _DEBUG_PCL_.set_debug_plane_quadrilateral(frame.bgr_image,plane_info, color_intr,debug_image);

        _DEBUG_PCL_.pub_debug_image(depth_show, "depth_show");
        _DEBUG_PCL_.pub_debug_image(debug_image, "debug_images");

        _DEBUG_PCL_.publish_pointcloud(plane_cloud);          // 发布点云
        _DEBUG_PCL_.publish_PlaneTF(plane_info);


        char key = cv::waitKey(1);
        if (key == 27)
        {
            break;
        }
        ros::spinOnce();
    }

    cv::destroyAllWindows();
}



int main(int argc, char** argv)
{
    // 初始化ROS节点
    ros::init(argc, argv, "test_node");
    ros::NodeHandle nh;
    
    test1(nh);
    return 0;
}