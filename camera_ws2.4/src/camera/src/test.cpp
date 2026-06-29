#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <limits>
#include <string>
#include <vector>
#include <utility>
#include <cmath>
#include <cstdlib>
#include <ros/ros.h>
#include "camera.h"

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include "./kfs_locator/set_result.h"
#include "./kfs_locator/debug_pcl.h"

rosbag::Bag g_bag;
rosbag::View* g_view = nullptr;
rosbag::View::iterator g_msg_iter;

bool init_bag_player(const std::string& bag_path)
{
    try
    {
        g_bag.open(bag_path, rosbag::bagmode::Read);
        std::vector<std::string> topics = {"/bgr_image","/depth_show"};
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
        if (msg.getTopic() == "/bgr_image")
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

void test1_frombag(const std::string& bag_path)
{
    if (!init_bag_player(bag_path)) return;

    // 参数
    rs2_intrinsics color_intr = createManualIntrinsics();
    Ten::kfs_locator::Ten_set_result plane_fiter(color_intr);
    Ten::XYZRPY bias;
    Ten::kfs_locator::Ten_debug_pcl debug_pcl;

    // 临时文件路径（根据 bag 文件名生成唯一目录）
    std::string bag_name = bag_path;
    size_t slash = bag_name.find_last_of("/\\");
    if (slash != std::string::npos) bag_name = bag_name.substr(slash + 1);
    size_t dot = bag_name.find_last_of(".");
    if (dot != std::string::npos) bag_name = bag_name.substr(0, dot);
    const std::string tmp_dir = "/tmp/kfs_" + bag_name + "/";
    system(("mkdir -p " + tmp_dir).c_str());
    const std::string path_x     = tmp_dir + "x.txt";
    const std::string path_y     = tmp_dir + "y.txt";
    const std::string path_z     = tmp_dir + "z.txt";
    const std::string path_angle = tmp_dir + "angle.txt";
    const std::string path_pcl   = tmp_dir + "pcl_count.txt";

    // 清空旧临时文件
    for (const auto* p : {&path_x, &path_y, &path_z, &path_angle, &path_pcl})
        std::ofstream(*p, std::ios::trunc).close();

    // 预计算有效帧对数量
    int bag_frame_pairs = 0;
    for (auto it = g_view->begin(); it != g_view->end(); ++it)
        if (it->getTopic() == "/bgr_image") ++bag_frame_pairs;
    g_msg_iter = g_view->begin();

    int total_frames   = 0;
    int success_frames = 0;

    // ── 逐帧处理（只跑一轮） ──
    while (ros::ok() && total_frames < bag_frame_pairs)
    {
        Ten::camera_frame frame = get_next_frame_from_bag();
        ++total_frames;

        debug_pcl.pub_depth_image(frame.depth_image, "depth_show");
        debug_pcl.pub_color_image(frame.bgr_image, "bgr_image");
        bool is_pre_ok = plane_fiter.preprocess(frame);
        std::cout << "state: " <<  plane_fiter.get_state() << std::endl;

        if (is_pre_ok)
        {
            std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> input_clouds = plane_fiter.get_input_clouds();
            debug_pcl.publish_pointcloud(plane_fiter.get_plane_cloud());
            bool is_post_ok = plane_fiter.postprocess();
            std::cout << "state: " <<  plane_fiter.get_state() << std::endl;
            if (is_post_ok)
            {
                Ten::kfs_locator::result result = plane_fiter.set_result(bias);
                ++success_frames;

                debug_pcl.save_bias(result.x,                        path_x);
                debug_pcl.save_bias(result.y,                        path_y);
                debug_pcl.save_bias(result.z,                        path_z);
                debug_pcl.save_bias(result.bia_radian * 180.0 / M_PI, path_angle);
                debug_pcl.save_bias(static_cast<double>(plane_fiter.get_filter_cloud()->size()), path_pcl);
            }
        }

        // 调试发布
        cv::Mat drawn = Ten::kfs_locator::Ten_debug_pcl::draw_rois(frame.bgr_image, plane_fiter.get_rois());
        debug_pcl.pub_color_image(drawn, "drown_rois");
        ros::spinOnce();
        ros::Duration(0.05).sleep();  // 给 rqt 订阅者连接时间
    }

    // 关闭 bag
    g_bag.close();
    delete g_view;
    g_view = nullptr;

    // ── 打印统计 ──
    double success_rate = (total_frames > 0) ? 100.0 * success_frames / total_frames : 0.0;

    std::cout << "\n=========================================" << std::endl;
    std::cout << "   静态跳变统计结果  (" << bag_path << ")" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "总帧数: " << total_frames
              << " | 成功帧数: " << success_frames
              << " | 成功率: " << std::fixed << std::setprecision(1)
              << success_rate << "%" << std::endl;
    std::cout << "=========================================" << std::endl;

    auto print_stats = [&](const std::string& label, const std::string& path) {
        auto s = debug_pcl.read_bias(path);
        std::cout << "\n── " << label << " ──" << std::endl;
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  平均值 (avg):       " << s["avg"] << std::endl;
        std::cout << "  标准差 (std):       " << s["standard_bias"] << std::endl;
        std::cout << "  跳变幅度 (Δ):       " << s["delta"] << std::endl;
        std::cout << "  90%范围 (P95-P5):   " << s["90%_range"] << std::endl;
        std::cout << "  最大值 (max):       " << s["max"] << std::endl;
        std::cout << "  最小值 (min):       " << s["min"] << std::endl;
    };

    print_stats("X (m)",        path_x);
    print_stats("Y (m)",        path_y);
    print_stats("Z (m)",        path_z);
    print_stats("Angle (deg)",  path_angle);
    print_stats("滤波后点数",    path_pcl);

    std::cout << "\n=========================================" << std::endl;
    cv::destroyAllWindows();
}

void test1_fromframe()
{
    Ten::Ten_camera& _CAMERA_ = Ten::Ten_camera::GetInstance();
    _CAMERA_.reset_camera_depth(640, 480,30);

    // 参数
    rs2_intrinsics color_intr = _CAMERA_.get_color_intrinsics();    // 彩色内参 → 绘图用
    Ten::kfs_locator::Ten_set_result plane_fiter(color_intr);
    Ten::kfs_locator::Ten_debug_pcl debug_pcl;
    Ten::XYZRPY bias;
    while (ros::ok())
    {
        // 设置输入图像
        Ten::camera_frame frame = _CAMERA_.camera_read_depth();
        // 处理流程
        bool is_pre_ok = plane_fiter.preprocess(frame);
        std::cout << "state: " <<  plane_fiter.get_state() << std::endl;
        if (is_pre_ok)
        {
            // 设置输入点云列表
            std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> input_clouds = plane_fiter.get_input_clouds();
            std::cout << "input_clouds size: " << input_clouds.size() << std::endl;
            // 将输出点云直接给postprocess函数
            bool is_post_ok = plane_fiter.postprocess();        // 置空输入点云，使用点云列表中的第一个点云
            std::cout << "state: " <<  plane_fiter.get_state() << std::endl;
            if (is_post_ok)
            {
                Ten::kfs_locator::result out = plane_fiter.set_result(bias);
                double angle = out.bia_radian;
                std::cout << "angle: " <<  (angle * 180.0 / M_PI) << std::endl;
                std::cout << "res.x: " <<  out.x << std::endl;
                std::cout << "res.y: " <<  out.y << std::endl;
                std::cout << "res.z: " <<  out.z << std::endl;
            }
        }

        // 调试发布

        // pcl::PointCloud<pcl::PointXYZ>::Ptr merged(new pcl::PointCloud<pcl::PointXYZ>);
        // for (const auto& cloud : plane_fiter.get_input_clouds())
        // {
        //     *merged += *cloud;
        // }
        // debug_pcl.publish_pointcloud(merged);
        // // 画框并发布
        // cv::Mat drawn = Ten::kfs_locator::Ten_debug_pcl::draw_rois(frame.bgr_image, plane_fiter.get_rois());
        // debug_pcl.pub_color_image(drawn, "drown_rois");
        debug_pcl.pub_depth_image(frame.depth_image, "depth_show");
        debug_pcl.pub_color_image(frame.bgr_image, "bgr_image");
        ros::spinOnce();
    }
}

// 测量静态偏差：逐帧处理 bag，写入 x/y/z/angle 到 4 个文件，结束后打印统计
// 支持批量处理：传入 bag 路径列表，依次处理每个 bag，汇总结果写入文件
void test2_frombag()
{
    // ── 1. 定义待处理的 bag 列表 ──
    // 每个元素: {bag文件路径, 标签(用于输出文件夹命名)}
    const std::vector<std::pair<std::string, std::string>> bag_list = {
        {"/home/h/1.05m距离面静态1.bag", "1.05m距离面静态1"},
        {"/home/h/1.2m距离面静态1.bag",  "1.2m距离面静态1"},
        {"/home/h/1.2m距离面静态2.bag",  "1.2m距离面静态2"},
        {"/home/h/1.2m距离面静态3.bag",  "1.2m距离面静态3"},
        {"/home/h/1.2m距离面静态4.bag",  "1.2m距离面静态4"},
        {"/home/h/1.35m距离面静态1.bag", "1.35m距离面静态1"},
        {"/home/h/1.35m距离面静态2.bag", "1.35m距离面静态2"},
        {"/home/h/1.35m距离面静态3.bag", "1.35m距离面静态3"},
        {"/home/h/1.35m距离面静态4.bag", "1.35m距离面静态4"},
        {"/home/h/1.5m距离面静态1.bag",  "1.5m距离面静态1"},
        {"/home/h/1.5m距离面静态2.bag",  "1.5m距离面静态2"},
        {"/home/h/1.5m距离面静态3.bag",  "1.5m距离面静态3"},
        {"/home/h/1.5m距离面静态4.bag",  "1.5m距离面静态4"},
        {"/home/h/1.8m距离面静态1.bag",  "1.8m距离面静态1"},
        {"/home/h/1.8m距离面静态2.bag",  "1.8m距离面静态2"},
        {"/home/h/1.8m距离面静态3.bag",  "1.8m距离面静态3"},
        {"/home/h/1.8m距离面静态4.bag",  "1.8m距离面静态4"},
        {"/home/h/2.1m距离面静态1.bag",  "2.1m距离面静态1"},
        {"/home/h/2.1m距离面静态2.bag",  "2.1m距离面静态2"},
        {"/home/h/2.1m距离面静态3.bag",  "2.1m距离面静态3"},
        {"/home/h/2.1m距离面静态4.bag",  "2.1m距离面静态4"},
        {"/home/h/2.4m距离面静态1.bag",  "2.4m距离面静态1"},
        {"/home/h/2.4m距离面静态2.bag",  "2.4m距离面静态2"},
        {"/home/h/2.4m距离面静态3.bag",  "2.4m距离面静态3"},
        {"/home/h/2.4m距离面静态4.bag",  "2.4m距离面静态4"},
        {"/home/h/2.65m距离面静态1.bag", "2.65m距离面静态1"},
        {"/home/h/2.65m距离面静态2.bag", "2.65m距离面静态2"},
        {"/home/h/2.65m距离面静态3.bag", "2.65m距离面静态3"},
        {"/home/h/2.65m距离面静态4.bag", "2.65m距离面静态4"},
    };

    // 输出根目录
    const std::string output_root = "/home/h/RC2026/camera_ws2.4/debug/";
    // 汇总统计文件
    const std::string summary_path = output_root + "summary_stats.txt";

    // ── 2. 打开汇总文件（覆盖模式） ──
    std::ofstream summary_file(summary_path, std::ios::trunc);
    if (!summary_file.is_open())
    {
        std::cerr << "❌ 无法创建汇总文件：" << summary_path << std::endl;
        return;
    }
    summary_file << std::fixed << std::setprecision(4);
    summary_file << "================================================\n";
    summary_file << "         多 Bag 静态跳变统计汇总\n";
    summary_file << "================================================\n";
    summary_file << "生成时间: " << __DATE__ << " " << __TIME__ << "\n\n";

    // ── 3. 逐 bag 处理 ──
    for (size_t bag_idx = 0; bag_idx < bag_list.size(); ++bag_idx)
    {
        const std::string& bag_path = bag_list[bag_idx].first;
        const std::string& bag_label = bag_list[bag_idx].second;

        std::cout << "\n🔵 [" << (bag_idx + 1) << "/" << bag_list.size()
                  << "] 开始处理: " << bag_label
                  << "  (" << bag_path << ")" << std::endl;

        if (!init_bag_player(bag_path))
        {
            std::cerr << "⚠️  跳过失败的 bag: " << bag_path << std::endl;
            summary_file << "⚠️  [" << bag_label << "] 初始化失败，跳过\n\n";
            continue;
        }

        // 参数（每个 bag 独立创建）
        rs2_intrinsics color_intr = createManualIntrinsics();
        Ten::kfs_locator::Ten_set_result plane_fiter(color_intr);
        Ten::XYZRPY bias;
        Ten::kfs_locator::Ten_debug_pcl debug_pcl;

        // 每个 bag 独立的输出目录
        const std::string bag_output_dir = output_root + bag_label + "/";
        // 用 mkdir 创建目录（不存在则创建）
        if (system(("mkdir -p " + bag_output_dir).c_str()) != 0) {
            ROS_WARN("Failed to create directory: %s", bag_output_dir.c_str());
        }

        const std::string path_x     = bag_output_dir + "x.txt";
        const std::string path_y     = bag_output_dir + "y.txt";
        const std::string path_z     = bag_output_dir + "z.txt";
        const std::string path_angle = bag_output_dir + "angle.txt";

        // 清空旧文件
        for (const auto* p : {&path_x, &path_y, &path_z, &path_angle})
            std::ofstream(*p, std::ios::trunc).close();

        // 预计算 bag 中有效帧对数量
        int bag_frame_pairs = 0;
        for (auto it = g_view->begin(); it != g_view->end(); ++it)
            if (it->getTopic() == "/bgr_image") ++bag_frame_pairs;

        g_msg_iter = g_view->begin();

        int total_frames   = 0;
        int success_frames = 0;

        // ── 逐帧处理 ──
        while (ros::ok() && total_frames < bag_frame_pairs)
        {
            Ten::camera_frame frame = get_next_frame_from_bag();
            ++total_frames;

            bool is_pre_ok = plane_fiter.preprocess(frame);
            if (is_pre_ok)
            {
                bool is_post_ok = plane_fiter.postprocess();
                if (is_post_ok)
                {
                    Ten::kfs_locator::result result = plane_fiter.set_result(bias);
                    ++success_frames;

                    debug_pcl.save_bias(result.x,          path_x);
                    debug_pcl.save_bias(result.y,          path_y);
                    debug_pcl.save_bias(result.z,          path_z);
                    debug_pcl.save_bias(result.bia_radian * 180.0 / M_PI, path_angle);
                    debug_pcl.pub_color_image(frame.bgr_image);
                }
            }

            ros::spinOnce();
        }

        // 关闭当前 bag
        g_bag.close();
        delete g_view;
        g_view = nullptr;

        // ── 打印并写入统计 ──
        double success_rate = (total_frames > 0)
            ? 100.0 * success_frames / total_frames
            : 0.0;

        // 控制台输出
        std::cout << "\n=========================================" << std::endl;
        std::cout << "   [" << bag_label << "] 静态跳变统计结果" << std::endl;
        std::cout << "=========================================" << std::endl;
        std::cout << "总帧数: " << total_frames
                  << " | 成功帧数: " << success_frames
                  << " | 成功率: " << std::fixed << std::setprecision(1)
                  << success_rate << "%" << std::endl;
        std::cout << "=========================================" << std::endl;

        // 汇总文件输出
        summary_file << "────────────────────────────────────────────────\n";
        summary_file << "  Bag: " << bag_label << "\n";
        summary_file << "  路径: " << bag_path << "\n";
        summary_file << "  总帧数: " << total_frames
                     << " | 成功帧数: " << success_frames
                     << " | 成功率: " << success_rate << "%\n";

        // 通用打印 lambda（控制台 + 文件双写）
        auto print_stats = [&](const std::string& label, const std::string& path) {
            auto s = debug_pcl.read_bias(path);

            // 控制台
            std::cout << "\n── " << label << " ──" << std::endl;
            std::cout << std::fixed << std::setprecision(4);
            std::cout << "  平均值 (avg):       " << s["avg"] << std::endl;
            std::cout << "  标准差 (std):       " << s["standard_bias"] << std::endl;
            std::cout << "  跳变幅度 (Δ):       " << s["delta"] << std::endl;
            std::cout << "  90%范围 (P95-P5):   " << s["90%_range"] << std::endl;
            std::cout << "  最大值 (max):       " << s["max"] << std::endl;
            std::cout << "  最小值 (min):       " << s["min"] << std::endl;

            // 文件
            summary_file << "\n  " << label << ":\n";
            summary_file << "    平均值 (avg):       " << s["avg"] << "\n";
            summary_file << "    标准差 (std):       " << s["standard_bias"] << "\n";
            summary_file << "    跳变幅度 (Δ):       " << s["delta"] << "\n";
            summary_file << "    90%范围 (P95-P5):   " << s["90%_range"] << "\n";
            summary_file << "    最大值 (max):       " << s["max"] << "\n";
            summary_file << "    最小值 (min):       " << s["min"] << "\n";
        };

        print_stats("X (m)",        path_x);
        print_stats("Y (m)",        path_y);
        print_stats("Z (m)",        path_z);
        print_stats("Angle (deg)",  path_angle);

        std::cout << "\n=========================================" << std::endl;
        summary_file << "\n";
    }

    // ── 4. 收尾 ──
    summary_file << "================================================\n";
    summary_file << "           处理完成（共 " << bag_list.size() << " 个 bag）\n";
    summary_file << "================================================\n";
    summary_file.close();

    std::cout << "\n✅ 汇总统计已保存至: " << summary_path << std::endl;
    cv::destroyAllWindows();
}

int main(int argc, char** argv)
{
    // 初始化ROS节点--------------------------------------------------------------
    ros::init(argc, argv, "test_node");

    if (argc < 2)
    {
        std::cerr << "用法: test_node <bag_path>" << std::endl;
        return 1;
    }
    
    test1_frombag(argv[1]);
    
    delete g_view;
    g_bag.close();
    return 0;
}