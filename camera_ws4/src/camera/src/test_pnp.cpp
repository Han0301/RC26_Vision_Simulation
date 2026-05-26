#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <stdexcept>
#include <ros/ros.h>
#include "camera.h"
#include "./PnP/pnp_main.h"
#include "./PnP/pnp_debug.h"

// ==================== 独立Lab调试工具（无namespace） ====================
static cv::Mat global_lab_image;         
static std::string global_lab_text;

// 鼠标回调：点击获取Lab值
static void mouse_callback_lab(int event, int x, int y, int flags, void* userdata)
{
    if (event == cv::EVENT_LBUTTONDOWN && !global_lab_image.empty())
    {
        cv::Vec3b lab_value = global_lab_image.at<cv::Vec3b>(y, x);
        int L = lab_value[0];
        int a = lab_value[1];
        int b = lab_value[2];
        global_lab_text = cv::format("Click Lab -> L:%d, a:%d, b:%d", L, a, b);
    }
}

// 主调试函数：输入BGR图像，鼠标点击显示Lab值
void show_lab_value_on_click(cv::Mat& bgr_image)
{
    if (bgr_image.empty())
        return;

    // BGR转Lab色彩空间
    cv::cvtColor(bgr_image, global_lab_image, cv::COLOR_BGR2Lab);
    cv::Mat display_image = bgr_image.clone();

    // 初始化窗口与鼠标回调（仅执行一次）
    static bool window_initialized = false;
    if (!window_initialized)
    {
        cv::namedWindow("Lab_Debug", cv::WINDOW_NORMAL);
        cv::setMouseCallback("Lab_Debug", mouse_callback_lab, nullptr);
        window_initialized = true;
    }

    // 在图像左上角绘制Lab数值
    if (!global_lab_text.empty())
    {
        cv::putText(
            display_image,
            global_lab_text,
            cv::Point(20, 40),
            cv::FONT_HERSHEY_SIMPLEX,
            0.8,
            cv::Scalar(0, 255, 0),
            1.2
        );
    }

    cv::imshow("Lab_Debug", display_image);
    cv::waitKey(1);
}
// ==================== 调试工具结束 ====================


void test_pnp(ros::NodeHandle& nh)
{
    Ten::Ten_camera& _CAMERA_ = Ten::Ten_camera::GetInstance();
    _CAMERA_.reset_camera_depth(640, 480, 30);
    
    rs2_intrinsics color_intr = _CAMERA_.get_color_intrinsics();

    Ten::KFS::kfsLocator pnp_hander(color_intr);
    Ten::KFS::DebugDrawer pnp_debug;

    pnp_debug.init();  

    while (ros::ok())
    {
        Ten::camera_frame frame = _CAMERA_.camera_read_depth();
        if (frame.bgr_image.empty() || frame.depth_image.empty())
        {
            std::cout << "图像为空！" << std::endl;
            continue;
        }
        show_lab_value_on_click(frame.bgr_image);

        pnp_hander.processOneFrame(frame.bgr_image, frame.depth_image);
        ros::spinOnce();
    }

    pnp_debug.shutdown();
    cv::destroyAllWindows();
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "test_pnp_node");
    ros::NodeHandle nh;
    test_pnp(nh);
    return 0;
}