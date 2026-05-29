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
    pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // 参数
    rs2_intrinsics color_intr = _CAMERA_.get_color_intrinsics();    // 彩色内参 → 绘图用

    while (ros::ok())
    {
        Ten::camera_frame frame = _CAMERA_.camera_read_depth();

        if (frame.bgr_image.empty() || frame.depth_image.empty())
        {
            std::cout << "frame.bgr_image.empty() || frame.depth_image.empty()" << std::endl;
            continue;
        }

        // 设置点云
        cv::Rect roi = _SET_PCL_.set_roi_detect(frame.bgr_image);
        _SET_PCL_.set_Pcl_Cloud(frame.depth_image, color_intr, roi,input_cloud);
        std::cout << "input_cloud->size()" << input_cloud->size() << std::endl;

        // 滤波器
        _PRE_PCL_.cloud_filter(input_cloud,output_cloud);      // 设置点云
        // 提取平面和中心点，法向量
        bool ret = _PRE_PCL_.Plane_fitter(output_cloud, plane_cloud, plane_info);

        // 方形拟合
        pcl::PointCloud<pcl::PointXYZ>::Ptr plane_2d_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        _POST_PCL_.compute_CenterAndNormal(plane_cloud,plane_info);
        _POST_PCL_.set_2d_cloud(plane_cloud,plane_info, plane_2d_cloud);
        _POST_PCL_.set_RPY(plane_2d_cloud,plane_info);

        std::cout << "bias: " << -plane_info.plane_center.y() << std::endl;

        // debug
        cv::Mat depth_show;
        cv::Mat debug_image;

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
