#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <stdexcept>
#include <ros/ros.h>
#include "camera.h"
#include "./Plane_FitLocator/plane_fit.h"
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

void set_key(Ten::Plane_FitLocator::Ten_debug_pcl& debug_pcl, bool &save_enabled)
{
    char key = cv::waitKey(1);
    if (key == 's' || key == 'S') // 按 S 启动保存
    {
        save_enabled = true;
        std::cout << "========== 已开启自动保存bias ==========" << std::endl;
    }
    else if (key == 'e' || key == 'E') // 按 E 停止保存
    {
        save_enabled = false;
        std::cout << "========== 已停止自动保存bias ==========" << std::endl;
    }
    else if (key == 'o' || key == 'O') // 按 E 停止保存
    {
        std::map<std::string, double> results = debug_pcl.read_bias("/home/h/RC2026/camera_ws2/debug/0.8mchange.txt");
        std::cout << "/home/h/RC2026/camera_ws2/debug/0.8mchange.txt" << std::endl;
        // ========== 格式化打印所有统计值 ==========
        std::cout << "=========================================" << std::endl;
        std::cout << "            Bias 统计结果" << std::endl;
        std::cout << "=========================================" << std::endl;
        std::cout << std::fixed << std::setprecision(5);  // 固定6位小数
        std::cout << "最大值 (max):\t\t" << results["max"] << std::endl;
        std::cout << "最小值 (min):\t\t" << results["min"] << std::endl;
        std::cout << "平均值 (avg):\t\t" << results["avg"] << std::endl;
        std::cout << "标准差 (standard_bias):\t" << results["standard_bias"] << std::endl;
        std::cout << "90%上分位 (90%bias_max):\t" << results["90%bias_max"] << std::endl;
        std::cout << "90%下分位 (90%bias_min):\t" << results["90%bias_min"] << std::endl;
        std::cout << "=========================================" << std::endl;
    }
}

void test1_frombag()
{
    ros::NodeHandle nh;
    std::string bag_path = "/home/h/2026-06-01-11-05-56.bag";
    if (!init_bag_player(bag_path)) return;

    // 参数
    rs2_intrinsics color_intr = createManualIntrinsics();    // 彩色内参 → 绘图用
    Ten::Plane_FitLocator::plane_fit plane_fiter(color_intr);

    ros::Rate loop_rate(30);
    while (ros::ok())
    {
        // 设置输入图像
        Ten::camera_frame frame = get_next_frame_from_bag();
        bool is_ok = plane_fiter.process(frame);
        std::cout << "state: " << plane_fiter.get_state() << std::endl;
        if (is_ok)
        {
            plane_fiter.publish(frame);
        }

        ros::spinOnce();
        loop_rate.sleep();
    }
}

void test1_fromframe()
{
    Ten::Ten_camera& _CAMERA_ = Ten::Ten_camera::GetInstance();
    _CAMERA_.reset_camera_depth(640, 480,30);

    // 参数
    rs2_intrinsics color_intr = _CAMERA_.get_color_intrinsics();    // 彩色内参 → 绘图用
    Ten::Plane_FitLocator::plane_fit plane_fiter(color_intr);

    while (ros::ok())
    {
        // 设置输入图像
        Ten::camera_frame frame = _CAMERA_.camera_read_depth();
        bool is_ok = plane_fiter.process(frame);
        if (is_ok)
        {
            plane_fiter.publish(frame);
        }
    }
}

void set_bias(
    Ten::Plane_FitLocator::plane_fit& plane_fiter, 
    Ten::Plane_FitLocator::Ten_debug_pcl& debug_pcl,
    std::string& save_path, 
    bool& save_enabled)
{
    set_key(debug_pcl,save_enabled);
    if (save_enabled)
    {
        debug_pcl.save_bias(-plane_fiter.get_plane_info().plane_center.y(), save_path);
    }
}
// 测量静态偏差
void test2()
{
    ros::NodeHandle nh;
    Ten::Ten_camera& _CAMERA_ = Ten::Ten_camera::GetInstance();
    _CAMERA_.reset_camera_depth(640, 480,30);

    std::string bag_path = "/home/h/2026-06-01-11-05-56.bag";
    if (!init_bag_player(bag_path)) return;

    // 参数
    rs2_intrinsics color_intr = _CAMERA_.get_color_intrinsics();    // 彩色内参 → 绘图用
    Ten::Plane_FitLocator::plane_fit plane_fiter(color_intr);
    Ten::Plane_FitLocator::Ten_debug_pcl debug_pcl;
    bool save_enabled = false; 
    std::string save_path = "/home/h/RC2026/camera_ws2/debug/0.8mchange.txt";
    cv::Mat debug_image;

    ros::Rate loop_rate(30);
    while (ros::ok())
    {
        // 设置输入图像
        Ten::camera_frame frame = _CAMERA_.camera_read_depth();
        plane_fiter.process(frame);
        debug_pcl.set_debug_plane_quadrilateral(frame.bgr_image, plane_fiter.get_plane_info(), color_intr,debug_image);
        cv::imshow("debug_view", debug_image);
        set_bias(plane_fiter,debug_pcl,save_path,save_enabled);

        ros::spinOnce();
        loop_rate.sleep();
    }
    cv::destroyAllWindows();

}
int main(int argc, char** argv)
{
    // 初始化ROS节点
    ros::init(argc, argv, "test_node");
    
    test1_frombag();
    delete g_view;
    g_bag.close();
    return 0;
}