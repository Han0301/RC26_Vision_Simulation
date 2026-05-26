#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <stdexcept>
#include <ros/ros.h>
#include "camera.h"
#include "./PnP/pnp_main.h"

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <librealsense2/rs.hpp>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <boost/foreach.hpp>

rosbag::Bag g_bag;
rosbag::View* g_view = nullptr;
rosbag::View::iterator g_msg_iter;

void save_native_frames(
    ros::Publisher& color_pub,
    ros::Publisher& depth_pub,
    const Ten::camera_frame& frame
)
{
    // 直接判断 cv::Mat 是否为空
    if (frame.bgr_image.empty() || frame.depth_image.empty())
    {
        std::cout << "[ERROR] 帧数据为空" << std::endl;
        return;
    }
    ros::Time stamp = ros::Time::now();

    // 发布彩色图（不变）
    cv_bridge::CvImage color_msg;
    color_msg.header.stamp = stamp;
    color_msg.encoding = sensor_msgs::image_encodings::BGR8;
    color_msg.image = frame.bgr_image.clone();
    color_pub.publish(color_msg.toImageMsg());

    // 发布深度图 → 直接用 frame.depth_image（cv::Mat），无 memcpy
    cv_bridge::CvImage depth_msg;
    depth_msg.header.stamp = stamp;
    depth_msg.encoding = sensor_msgs::image_encodings::TYPE_16UC1;
    depth_msg.image = frame.depth_image.clone(); // 直接赋值
    depth_pub.publish(depth_msg.toImageMsg());
}

bool init_bag_player(const std::string& bag_path)
{
    try
    {
        g_bag.open(bag_path, rosbag::bagmode::Read);
        std::vector<std::string> topics = {"/camera/color/image_raw","/camera/depth/image_raw"};
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
        if (msg.getTopic() == "/camera/color/image_raw")
        {
            auto img_msg = msg.instantiate<sensor_msgs::Image>();
            if (img_msg) color_img = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8)->image;
        }
        // 深度图（不变）
        if (msg.getTopic() == "/camera/depth/image_raw")
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
    return get_next_frame_from_bag();
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "test_node");
    ros::NodeHandle nh;
    ros::Publisher color_image_pub = nh.advertise<sensor_msgs::Image>("/camera/color/image_raw", 30);
    ros::Publisher depth_image_pub = nh.advertise<sensor_msgs::Image>("/camera/depth/image_raw", 30);

    Ten::Ten_camera& _CAMERA_ = Ten::Ten_camera::GetInstance();
    _CAMERA_.reset_camera_depth(640, 480, 30);
    rs2_intrinsics color_intr = _CAMERA_.get_color_intrinsics();
    Ten::KFS::kfsLocator pnp_hander(color_intr);

    std::string bag_path = "/home/h/camera_native_dat.bag";
    if (!init_bag_player(bag_path)) return -1;

    ros::Rate loop_rate(30);
    while (ros::ok())
    {
        // 读取frame的方式
        Ten::camera_frame frame = get_next_frame_from_bag();
        // Ten::camera_frame frame = _CAMERA_.camera_read_depth();

        pnp_hander.processOneFrame(frame.bgr_image, frame.depth_image);

        // save_native_frames(color_image_pub, depth_image_pub, frame);
        ros::spinOnce();
        loop_rate.sleep();
    }

    delete g_view;
    g_bag.close();
    return 0;
}
