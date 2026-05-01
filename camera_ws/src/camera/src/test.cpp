#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <stdexcept>
#include <ros/ros.h>
#include "camera.h"
#include "./Plane_FitLocator/pose_resolving.h"
#include "./Plane_FitLocator/debug_pcl.h"

void test1(ros::NodeHandle& nh)
{
    Ten::Ten_camera& _CAMERA_ = Ten::Ten_camera::GetInstance();
    Ten::Plane_FitLocator::Ten_debug_pcl DEBUG_PCL_;
    pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> debug_cloud;

    // 参数
    rs2_intrinsics color_intr = _CAMERA_.get_color_intrinsics();    // 彩色内参 → 绘图用

    Ten::Plane_FitLocator::pose_resolving pose_resolve_(color_intr);
    while (ros::ok())
    {
        // 图像输入
        Ten::camera_frame frame = _CAMERA_.camera_read_depth();

        if (frame.bgr_image.empty() || frame.depth_image.empty())
        {
            std::cout << "frame.bgr_image.empty() || frame.depth_image.empty()" << std::endl;
            continue;
        }

        // 处理
        bool ret = pose_resolve_.process(frame,debug_cloud);
        Ten::Plane_FitLocator::Plane_Info plane_info = pose_resolve_.get_plane_info(ret);

        // 调试输出
        // 深度图
        cv::Mat depth_show;
        cv::normalize(frame.depth_image, depth_show, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::imshow("depth_frame", depth_show);
        // 彩色图
        cv::Mat debug_image;
        // _DEBUG_PCL_.debug_rgb_image(frame.bgr_image,plane_info, color_intr,debug_image);
        DEBUG_PCL_.debug_plane_quadrilateral(frame.bgr_image,plane_info, color_intr,debug_image);
        cv::imshow("bgr_frame", debug_image);
        // 点云
        DEBUG_PCL_.publish_pointcloud(debug_cloud);          // 发布点云
        DEBUG_PCL_.publish_PlaneTF(plane_info);

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
