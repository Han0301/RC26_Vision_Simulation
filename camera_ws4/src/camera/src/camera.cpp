#ifndef __CAMERA_CPP_
#define __CAMERA_CPP_
#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <iostream>
#include <string>
#include <mutex>
#include <unistd.h>
#include "camera.h"

namespace Ten
{
    std::once_flag Ten_camera::camera_flag_;

    Ten_camera& Ten_camera::GetInstance(size_t w, size_t h, size_t fps)
    {
        static std::unique_ptr<Ten_camera> ten_camera = nullptr;
        std::call_once(camera_flag_, [w, h, fps]() 
        {
            ten_camera = create(w, h, fps);
            std::cout << "init_camera" << std::endl;
        });
        return *ten_camera;
    } 

    cv::Mat Ten_camera::camera_read()
    {
        std::lock_guard<std::mutex> lock(read_mtx_);
        // 等待并获取帧数据
        rs2::frameset frames = pipe.wait_for_frames();
        // 获取彩色帧
        rs2::frame color_frame = frames.get_color_frame();
        // 转换为OpenCV矩阵格式
        cv::Mat color_image(cv::Size(_w, _h), CV_8UC3, 
                           (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
        return color_image;
    }

    camera_frame Ten_camera::camera_read_depth()
    {
        std::lock_guard<std::mutex> lock(read_mtx_);
        camera_frame frame;

        // 等待并获取帧数据
        rs2::frameset frames = pipe.wait_for_frames();

        rs2::frameset aligned_frames = align_to_color_.process(frames);

        // 获取彩色帧
        rs2::frame color_frame = aligned_frames.get_color_frame();
        // 转换为OpenCV矩阵格式
        frame.bgr_image = cv::Mat (cv::Size(_w, _h), CV_8UC3, 
                           (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);

        // 获取深度帧
        rs2::depth_frame depth_frame = aligned_frames.get_depth_frame();
        frame.depth_image = cv::Mat(cv::Size(_w, _h), CV_16UC1, 
                    (void*)depth_frame.get_data(), cv::Mat::AUTO_STEP);
        // 保存原生深度帧
        frame.raw_depth_frame = std::make_shared<rs2::depth_frame>(depth_frame);
        return frame;
    }


    void Ten_camera::reset_camera(size_t w, size_t h, size_t fps)
    {
        std::lock_guard<std::mutex> lock(read_mtx_);
        
        // 1. 停止当前管道
        pipe.stop();
        
        // 2. 重置配置对象 (创建新的 config 以清除旧设置)
        config = rs2::config();
        
        // 3. 使能新的流配置
        config.enable_stream(RS2_STREAM_COLOR, w, h, RS2_FORMAT_BGR8, fps);
        
        // 4. 重新启动管道
        pipe.start(config);
        
        // 5. 更新成员变量
        _w = w;
        _h = h;
        
        std::cout << "camera reset to " << w << "x" << h << "@" << fps << std::endl;
    }

    void Ten_camera::reset_camera_depth(size_t w, size_t h, size_t fps)
    {
        std::lock_guard<std::mutex> lock(read_mtx_);
        
        // 1. 停止当前管道
        pipe.stop();
        
        // 2. 重置配置对象 (创建新的 config 以清除旧设置)
        config = rs2::config();
        
        // 3. 使能新的流配置
        config.enable_stream(RS2_STREAM_COLOR, w, h, RS2_FORMAT_BGR8, fps);    // RGB
        config.enable_stream(RS2_STREAM_DEPTH, w, h, RS2_FORMAT_Z16, fps);     // 深度图（16位）
        
        // 4. 重新启动管道
        auto active_profile = pipe.start(config);
        _w = w;
        _h = h;
        // ====================== 新增：减少深度噪声的配置 ======================
        auto depth_sensor = active_profile.get_device().first<rs2::depth_sensor>();
    
        // 1. 开启官方「高精度模式」（D400全系支持，最关键！）
        if (depth_sensor.supports(RS2_OPTION_VISUAL_PRESET))
        {
            // RS2_RS400_VISUAL_PRESET_HIGH_ACCURACY = 高精准模式（减少斜视角噪声）
            depth_sensor.set_option(RS2_OPTION_VISUAL_PRESET, 3.0f); 
        }
        // 2. 关闭自动曝光（固定曝光，斜视角深度更稳定）
        if (depth_sensor.supports(RS2_OPTION_AUTO_EXPOSURE_MODE))
        {
            depth_sensor.set_option(RS2_OPTION_AUTO_EXPOSURE_MODE, 0.0f);
            depth_sensor.set_option(RS2_OPTION_EXPOSURE, 7500.0f); 
        }
        // ====================================================================

        auto color_profile = active_profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
        color_intr_ = color_profile.get_intrinsics();                // 彩色内参

        std::cout << "camera reset to " << w << "x" << h << "@" << fps << std::endl;
    }

    // 获取彩色相机内参
    rs2_intrinsics Ten_camera::get_color_intrinsics() {
        return color_intr_;
    }
}
#endif