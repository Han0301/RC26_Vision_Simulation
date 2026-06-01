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
#include <rosbag/bag.h>
#include <rosbag/view.h>

rosbag::Bag g_bag;
rosbag::View* g_view = nullptr;
rosbag::View::iterator g_msg_iter;

bool init_bag_player(const std::string& bag_path)
{
    try
    {
        g_bag.open(bag_path, rosbag::bagmode::Read);
        std::vector<std::string> topics = {"/debug_images","/depth_show"};
        g_view = new rosbag::View(g_bag, rosbag::TopicQuery(topics));
        g_msg_iter = g_view->begin();

        std::cout << "✅ bag初始化成功" << std::endl;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "❌ 初始化失败：" << e.what() << std::endl;
        return false;
    }
}

Ten::camera_frame get_next_frame_from_bag()
{
    Ten::camera_frame frame;
    cv::Mat color_img, depth_img;

    while (g_msg_iter != g_view->end())
    {
        rosbag::MessageInstance const msg = *g_msg_iter;
        ++g_msg_iter;

        // 彩色图（不变）
        if (msg.getTopic() == "/debug_images")
        {
            auto img_msg = msg.instantiate<sensor_msgs::Image>();
            if (img_msg) color_img = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8)->image;
        }
        // 深度图（不变）
        if (msg.getTopic() == "/depth_show")
        {
            auto img_msg = msg.instantiate<sensor_msgs::Image>();
            if (img_msg) depth_img = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::TYPE_16UC1)->image;
        }

        // 仅赋值 cv::Mat，删除 raw_depth_frame 相关代码
        if (!color_img.empty() && !depth_img.empty())
        {
            frame.bgr_image = color_img;
            frame.depth_image = depth_img;
            return frame;
        }
    }

    std::cout << "🔄 bag循环播放" << std::endl;
    g_msg_iter = g_view->begin();
    return frame;
}

rs2_intrinsics createManualIntrinsics()
{
    rs2_intrinsics intr;
    // 1. 核心参数：直接填写你的相机内参
    intr.fx  = 553.7294;   // 焦距x
    intr.fy  = 553.7891;   // 焦距y
    intr.ppx = 317.2345;   // 主点x
    intr.ppy = 239.7654;   // 主点y

    // 2. 固定默认参数（不影响你的相机矩阵，无需修改）
    intr.width  = 640;  // 图像宽
    intr.height = 480;  // 图像高
    intr.model  = RS2_DISTORTION_BROWN_CONRADY; // 畸变模型（默认）
    for (int i = 0; i < 5; i++) intr.coeffs[i] = 0; // 畸变系数全0

    return intr;
}

void test1()
{
    ros::NodeHandle nh;
    // Ten::Ten_camera& _CAMERA_ = Ten::Ten_camera::GetInstance();
    // _CAMERA_.reset_camera_depth(640, 480,30);

    std::string bag_path = "/home/h/2026-06-01-11-05-56.bag";
    if (!init_bag_player(bag_path)) return;

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
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane_cloud_vec(new pcl::PointCloud<pcl::PointXYZ>); 

    std::vector<cv::Point2f> plane_points_2d;
    std::vector<cv::Point2f> plane_points_flited;
    // 参数
    // rs2_intrinsics color_intr = _CAMERA_.get_color_intrinsics();    // 彩色内参 → 绘图用
    rs2_intrinsics color_intr = createManualIntrinsics();

    cv::Mat depth_show;
    cv::Mat debug_image;
    bool save_enabled = false; 
    std::string save_path = "/home/h/RC2026/camera_ws2/debug/0.8mchange.txt";

    ros::Rate loop_rate(30);
    while (ros::ok())
    {
        // 设置输入图像
        // Ten::camera_frame frame = _CAMERA_.camera_read_depth();
        Ten::camera_frame frame = get_next_frame_from_bag();
        if (frame.bgr_image.empty() || frame.depth_image.empty())
        {
            std::cout << "frame.bgr_image.empty() || frame.depth_image.empty()" << std::endl;
            continue;
        }
        // // debug 部分
        cv::normalize(frame.depth_image, depth_show, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        _DEBUG_PCL_.set_debug_plane_quadrilateral(frame.bgr_image,plane_info, color_intr,debug_image);

        _DEBUG_PCL_.pub_depth_image(frame.depth_image, "depth_show");
        _DEBUG_PCL_.pub_color_image(frame.bgr_image, "debug_images");

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
        // bool is_plane_flitted = _PRE_PCL_.Plane_fitter_(filter_cloud, plane_cloud, plane_info);

        if (!is_plane_flitted)
        {
            std::cout << "Plane_fitter error!" << std::endl;
            continue;
        }

        // 方形拟合
        _POST_PCL_.compute_CenterAndNormal(plane_cloud,plane_info);
        _POST_PCL_.set_vector_2d(plane_cloud,plane_info,plane_points_2d);
        _POST_PCL_.central_range_filter(plane_points_2d,plane_points_flited);
        bool is_shape_ok = _POST_PCL_.shape_filter(plane_points_flited);
        if (!is_shape_ok)
        {
            std::cout << "shape_filter error!" << std::endl;
            continue;
        }

        _POST_PCL_.vector2f_to_pcl(plane_points_flited,plane_info,plane_cloud_vec);
        _POST_PCL_.set_RPY(plane_cloud_vec,plane_info);
        std::cout << "plane_cloud_vec->size()" << plane_cloud_vec->size() << std::endl;
        _POST_PCL_.compute_CenterAndNormal(plane_cloud_vec,plane_info);

        _DEBUG_PCL_.publish_PlaneTF(plane_info);
        _DEBUG_PCL_.publish_pointcloud(plane_cloud_vec);          // 发布点云
        // _DEBUG_PCL_.pub_value(-plane_info.plane_center.y() );
        std::cout << "bias: " << -plane_info.plane_center.y() 
                  << ", yaw: " << plane_info.plane_euler._yaw << std::endl;

        // 保存用， 可全注释-------------------
        // cv::imshow("depth_view", depth_show);
        // cv::imshow("debug_view", debug_image);

        // if (save_enabled)
        // {
        //     _DEBUG_PCL_.save_bias(-plane_info.plane_center.y(), save_path);
        // }

        // char key = cv::waitKey(1);
        // if (key == 27)
        // {
        //     break;
        // }
        // else if (key == 's' || key == 'S') // 按 S 启动保存
        // {
        //     save_enabled = true;
        //     std::cout << "========== 已开启自动保存bias ==========" << std::endl;
        // }
        // else if (key == 'e' || key == 'E') // 按 E 停止保存
        // {
        //     save_enabled = false;
        //     std::cout << "========== 已停止自动保存bias ==========" << std::endl;
        // }
        // else if (key == 'o' || key == 'O') // 按 E 停止保存
        // {
        //     std::map<std::string, double> results = _DEBUG_PCL_.read_bias("/home/h/RC2026/camera_ws2/debug/0.8mchange.txt");
        //     std::cout << "/home/h/RC2026/camera_ws2/debug/0.8mchange.txt" << std::endl;
        //     // ========== 格式化打印所有统计值 ==========
        //     std::cout << "=========================================" << std::endl;
        //     std::cout << "            Bias 统计结果" << std::endl;
        //     std::cout << "=========================================" << std::endl;
        //     std::cout << std::fixed << std::setprecision(5);  // 固定6位小数
        //     std::cout << "最大值 (max):\t\t" << results["max"] << std::endl;
        //     std::cout << "最小值 (min):\t\t" << results["min"] << std::endl;
        //     std::cout << "平均值 (avg):\t\t" << results["avg"] << std::endl;
        //     std::cout << "标准差 (standard_bias):\t" << results["standard_bias"] << std::endl;
        //     std::cout << "90%上分位 (90%bias_max):\t" << results["90%bias_max"] << std::endl;
        //     std::cout << "90%下分位 (90%bias_min):\t" << results["90%bias_min"] << std::endl;
        //     std::cout << "=========================================" << std::endl;

        // }
        ros::spinOnce();
        loop_rate.sleep();
        // 保存用， 可全注释-------------------
    }

    cv::destroyAllWindows();
}



int main(int argc, char** argv)
{
    // 初始化ROS节点
    ros::init(argc, argv, "test_node");
    
    test1();
    delete g_view;
    g_bag.close();
    return 0;
}