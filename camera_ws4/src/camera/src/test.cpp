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

#include <sys/stat.h>
#include <fstream>

#include <dirent.h>   // 文件夹遍历
#include <cctype>     // 数字判断
#include <climits>    // 数值限制

rosbag::Bag g_bag;
rosbag::View* g_view = nullptr;
rosbag::View::iterator g_msg_iter;

bool g_save_flag = false;
int g_save_count = 1;
// 保存路径
const std::string IMG_SAVE_DIR = "/home/h/视频/datasets_blue_kfs/images/";
const std::string TXT_SAVE_DIR = "/home/h/视频/datasets_blue_kfs/labels/";

int getMaxFileNumber(const std::string& dir_path, const std::string& suffix)
{
    int max_num = 0;
    DIR* dir = opendir(dir_path.c_str());
    
    // 文件夹不存在，直接返回0
    if (!dir)
    {
        std::cout << "ℹ️ 文件夹不存在，初始编号从 1 开始" << std::endl;
        return 0;
    }

    struct dirent* entry;
    // 遍历文件夹所有文件
    while ((entry = readdir(dir)) != nullptr)
    {
        std::string filename = entry->d_name;
        // 跳过 . 和 ..
        if (filename == "." || filename == "..") continue;

        // 匹配后缀（.png / .txt）
        size_t suffix_pos = filename.find(suffix);
        if (suffix_pos == std::string::npos || suffix_pos != filename.length() - suffix.length())
            continue;

        // 提取纯数字文件名（去掉后缀）
        std::string num_str = filename.substr(0, suffix_pos);
        // 判断是否为纯数字
        bool is_all_digit = true;
        for (char c : num_str)
        {
            if (!isdigit(c))
            {
                is_all_digit = false;
                break;
            }
        }
        if (!is_all_digit) continue;

        // 转换为数字，更新最大值
        int current_num = stoi(num_str);
        if (current_num > max_num)
            max_num = current_num;
    }

    closedir(dir);
    std::cout << "ℹ️ 文件夹最大编号：" << max_num << "，下次保存从 " << max_num + 1 << " 开始" << std::endl;
    return max_num;
}
void onMouse(int event, int x, int y, int flags, void* param)
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        g_save_flag = true;
        std::cout << "🖱️ 左键触发保存，计数：" << g_save_count << std::endl;
    }
}

bool makeDir(const std::string& path)
{
    return mkdir(path.c_str(), 0777) == 0 || errno == EEXIST;
}

// 保存当前帧图像 + ROI归一化坐标TXT
void saveFrameAndRoi(const cv::Mat& bgr_img, const cv::Rect& roi, int count)
{
    if (bgr_img.empty() || roi.area() <= 0)
    {
        std::cout << "[ERROR] 无有效图像/ROI，保存失败" << std::endl;
        return;
    }

    // 文件名
    std::string img_name = IMG_SAVE_DIR + std::to_string(count) + ".png";
    std::string txt_name = TXT_SAVE_DIR + std::to_string(count) + ".txt";

    // 1. 保存PNG
    cv::imwrite(img_name, bgr_img);

    // 2. 计算ROI四个角点（左上、右上、右下、左下）
    int cols = bgr_img.cols;
    int rows = bgr_img.rows;
    float x1 = (float)roi.x / cols;
    float y1 = (float)roi.y / rows;
    float x2 = (float)(roi.x + roi.width) / cols;
    float y2 = (float)roi.y / rows;
    float x3 = (float)(roi.x + roi.width) / cols;
    float y3 = (float)(roi.y + roi.height) / rows;
    float x4 = (float)roi.x / cols;
    float y4 = (float)(roi.y + roi.height) / rows;

    // 3. 写入TXT：0 x1 y1 x2 y2 x3 y3 x4 y4
    std::ofstream ofs(txt_name);
    ofs << "0 " 
        << x1 << " " << y1 << " "
        << x2 << " " << y2 << " "
        << x3 << " " << y3 << " "
        << x4 << " " << y4;
    ofs.close();

    std::cout << "✅ 保存成功：" << img_name << " | " << txt_name << std::endl;
}

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

int main(int argc, char** argv)
{
    ros::init(argc, argv, "test_node");
    ros::NodeHandle nh;
    ros::Publisher color_image_pub = nh.advertise<sensor_msgs::Image>("/camera/color/image_raw", 30);
    ros::Publisher depth_image_pub = nh.advertise<sensor_msgs::Image>("/camera/depth/image_raw", 30);

    // Ten::Ten_camera& _CAMERA_ = Ten::Ten_camera::GetInstance();
    // _CAMERA_.reset_camera_depth(640, 480, 30);
    // rs2_intrinsics color_intr = _CAMERA_.get_color_intrinsics();
    rs2_intrinsics color_intr = createManualIntrinsics();
    
    Ten::KFS::kfsLocator pnp_hander(color_intr);

    std::string bag_path = "/home/h/camera_native_dat.bag";
    if (!init_bag_player(bag_path)) return -1;

    // 保存数据集部分
    cv::namedWindow("debug", cv::WINDOW_NORMAL);
    cv::setMouseCallback("debug", onMouse, nullptr);
    makeDir(IMG_SAVE_DIR);
    makeDir(TXT_SAVE_DIR);
    int max_num = getMaxFileNumber(IMG_SAVE_DIR, ".png");
    g_save_count = max_num + 1;  // 续号保存

    ros::Rate loop_rate(30);
    while (ros::ok())
    {
        // 读取frame的方式
        Ten::camera_frame frame = get_next_frame_from_bag();
        // Ten::camera_frame frame = _CAMERA_.camera_read_depth();

        pnp_hander.processOneFrame(frame.bgr_image, frame.depth_image,true);

        show_lab_value_on_click(frame.bgr_image);
        // 保存数据集部分
        // if (g_save_flag)
        // {
        //     cv::Rect current_roi = pnp_hander.getCurrentRoi__();
        //     saveFrameAndRoi(frame.bgr_image, current_roi, g_save_count);
        //     g_save_count++;
        //     g_save_flag = false; // 重置标志
        // }
        // save_native_frames(color_image_pub, depth_image_pub, frame);
        ros::spinOnce();
        loop_rate.sleep();
    }

    delete g_view;
    g_bag.close();
    return 0;
}