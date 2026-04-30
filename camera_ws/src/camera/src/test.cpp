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

    // 窗口
    cv::namedWindow("bgr_frame", cv::WINDOW_NORMAL);
    cv::resizeWindow("bgr_frame", 640, 480);
    cv::namedWindow("depth_frame", cv::WINDOW_NORMAL);
    cv::resizeWindow("depth_frame", 640, 480);
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

        cv::Mat draw_bgr, draw_depth;

        cv::Mat depth_show;
        cv::normalize(frame.depth_image, depth_show, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::imshow("depth_frame", depth_show);

        // 设置点云
        _SET_PCL_.set_Pcl_Cloud(frame.raw_depth_frame, color_intr, input_cloud);
        // 滤波器
        _PRE_PCL_.cloud_filter(input_cloud,output_cloud);      // 设置点云
        // 提取平面和中心点，法向量
        bool ret = _PRE_PCL_.Plane_fitter(output_cloud, plane_cloud, plane_info);

        // 方形拟合
        pcl::PointCloud<pcl::PointXYZ>::Ptr plane_2d_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        _POST_PCL_.compute_CenterAndNormal(plane_cloud,plane_info);
        _POST_PCL_.set_2d_cloud(plane_cloud,plane_info, plane_2d_cloud);
        _POST_PCL_.fit_PlaneSquare(plane_2d_cloud,plane_info);
        _POST_PCL_.set_plane_euler(plane_info);

        cv::Mat debug_image;
        // _DEBUG_PCL_.debug_rgb_image(frame.bgr_image,plane_info, color_intr,debug_image);
        _DEBUG_PCL_.debug_plane_quadrilateral(frame.bgr_image,plane_info, color_intr,debug_image);
        cv::imshow("bgr_frame", debug_image);

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
