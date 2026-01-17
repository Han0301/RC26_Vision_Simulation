#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <mutex>
#include <thread>
#include <chrono>

// 全局配置与数据存储
struct GlobalData {
    cv::Mat processed_image;
    std::mutex mtx_image;
    cv::Ptr<cv::ORB> orb_detector;
    image_transport::Publisher feature_pub;

    // 计时变量
    double total_time = 0.0;
    int count = 0;

    GlobalData() {
        // 初始化ORB特征检测器
        orb_detector = cv::ORB::create(
            150,    // 最大特征点数量
            1.2f,   // 尺度因子
            8,      // 金字塔层数
            30,     // 边缘阈值
            0,      // 第一个特征点不使用
            2,      //  WTA_K
            cv::ORB::HARRIS_SCORE, // 评分方法
            31,     // patch大小
            20      //  fast阈值
        );
    }
};

GlobalData global_data;


// 合并重复角点
std::vector<cv::KeyPoint> mergeCloseKeypoints(const std::vector<cv::KeyPoint>& keypoints, double threshold) {
    if (keypoints.empty()) return {};

    std::vector<cv::KeyPoint> merged;
    std::vector<bool> merged_flag(keypoints.size(), false); // 标记是否已合并

    for (size_t i = 0; i < keypoints.size(); ++i) {
        if (merged_flag[i]) continue;

        // 初始化聚类中心为当前点
        cv::Point2f sum_pt = keypoints[i].pt;
        int count = 1;
        merged_flag[i] = true;

        // 寻找所有距离小于阈值的点，归为一类
        for (size_t j = i + 1; j < keypoints.size(); ++j) {
            if (merged_flag[j]) continue;

            // 计算欧氏距离
            double dx = keypoints[i].pt.x - keypoints[j].pt.x;
            double dy = keypoints[i].pt.y - keypoints[j].pt.y;
            double dist = std::sqrt(dx*dx + dy*dy);

            if (dist < threshold) {
                sum_pt += keypoints[j].pt;
                count++;
                merged_flag[j] = true;
            }
        }

        // 取平均位置作为合并后的点
        cv::KeyPoint merged_kp = keypoints[i]; // 保留原始点的其他属性（大小、角度等）
        merged_kp.pt = sum_pt / count; // 平均位置
        merged.push_back(merged_kp);
    }

    return merged;
}


/**
 * @brief 封装的ORB特征检测与绘制函数
 * @param input 输入的彩色图像
 * @param output 输出的、已绘制特征点的彩色图像
 */
void detectAndDrawORB(const cv::Mat& input, cv::Mat& output) {
    auto start = std::chrono::high_resolution_clock::now();

    // --------------------------
    // 1. 定义ROI区域（可根据需求修改）
    // --------------------------
    // 参数：x（左上角x）、y（左上角y）、width（宽度）、height（高度）
    // 注意：需确保ROI在图像范围内（x+width <= input.cols, y+height <= input.rows）
    cv::Rect roi(950, 650, 300, 250); // 示例：左上角(200,150)，宽600，高400
    
    // 检查ROI有效性（避免越界）
    if (roi.x < 0 || roi.y < 0 || 
        roi.x + roi.width > input.cols || 
        roi.y + roi.height > input.rows) {
        ROS_WARN("ROI超出图像范围！使用全图检测");
        roi = cv::Rect(0, 0, input.cols, input.rows); // 越界时使用全图
    }

    // 提取ROI子图像（只在该区域检测特征）
    cv::Mat roi_image = input(roi);

    // --------------------------
    // 2. 在ROI内检测ORB特征
    // --------------------------
    std::vector<cv::KeyPoint> roi_keypoints; // 存储ROI内的特征点（坐标相对ROI）
    cv::Mat descriptors;
    global_data.orb_detector->detectAndCompute(roi_image, cv::Mat(), roi_keypoints, descriptors);

    // --------------------------
    // 3. 转换特征点坐标到原图（ROI相对坐标 -> 原图绝对坐标）
    // --------------------------
    std::vector<cv::KeyPoint> image_keypoints; // 存储原图坐标的特征点
    for (auto& kp : roi_keypoints) {
        // 加上ROI左上角偏移量，转换为原图坐标
        kp.pt.x += roi.x;
        kp.pt.y += roi.y;
        image_keypoints.push_back(kp);
    }

    // 合并重复角点
    double merge_threshold = 20.0; // 距离阈值（可根据实际场景调整，单位：像素）
    std::vector<cv::KeyPoint> merged_keypoints = mergeCloseKeypoints(image_keypoints, merge_threshold);
    ROS_INFO("合并前角点数量: %d, 合并后: %d", (int)image_keypoints.size(), (int)merged_keypoints.size());
    // --------------------------
    // 4. 打印特征点坐标（每帧打印前5个，避免输出过多）
    // --------------------------
    ROS_INFO("point count  %d ",
             image_keypoints.size());
    int print_num = std::min(15, (int)image_keypoints.size()); // 最多打印5个
    for (int i = 0; i < print_num; ++i) {
        ROS_INFO("key point %d: (x=%.1f, y=%.1f)", 
                 i+1, 
                 image_keypoints[i].pt.x, 
                 image_keypoints[i].pt.y);
        }

    // --------------------------
    // 5. 绘制特征点和ROI区域
    // --------------------------
    input.copyTo(output); // 复制原图到输出
    // 绘制ROI区域（蓝色矩形，线宽2）
    cv::rectangle(output, roi, cv::Scalar(255, 0, 0), 2);
    // 绘制特征点（绿色）
    cv::drawKeypoints(
        output, 
        merged_keypoints, 
        output, 
        cv::Scalar(0, 255, 0), 
        cv::DrawMatchesFlags::DRAW_OVER_OUTIMG | cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
    );

    // --------------------------
    // 6. 计时统计
    // --------------------------
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    global_data.total_time += elapsed.count();
    global_data.count++;

    if (global_data.count % 10 == 0) {
        ROS_INFO("ORB平均耗时: %.4f秒（共%d帧）",
                 global_data.total_time / global_data.count,
                 global_data.count);
    }
}


// 基于点的orb范围检测，并在原图绘制范围、圆心和检测点
void detectORBInCircle(cv::Mat& image, const cv::Point2f& center, int radius) {
    if (image.empty()) {
        ROS_WARN("输入图像为空，无法检测角点");
        return;
    }

    // 检查圆心是否在图像范围内
    if (center.x < 0 || center.x >= image.cols || center.y < 0 || center.y >= image.rows) {
        ROS_WARN("圆心坐标超出图像范围");
        return;
    }

    // 检查半径是否有效
    if (radius <= 0) {
        ROS_WARN("半径必须为正数");
        return;
    }

    // 创建掩码：仅圆形区域为有效区域（白色），其他区域无效（黑色）
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);  // 单通道掩码，初始全黑
    cv::circle(mask, center, radius, cv::Scalar(255), -1);  // 绘制填充的白色圆形

    // 在掩码限制的区域内检测ORB角点
    std::vector<cv::KeyPoint> circle_keypoints;
    global_data.orb_detector->detect(image, circle_keypoints, mask);  // 使用掩码进行检测

    // 打印检测到的角点数量和坐标
    ROS_INFO("在圆形区域内（圆心: (%.1f, %.1f), 半径: %d）检测到 %d 个ORB角点",
             center.x, center.y, radius, (int)circle_keypoints.size());

    // 打印部分角点坐标
    int print_limit = 20;  // 最多打印20个
    int print_count = std::min(print_limit, (int)circle_keypoints.size());
    for (int i = 0; i < print_count; ++i) {
        ROS_INFO("圆形区域角点 %d: (x=%.1f, y=%.1f)",
                 i + 1,
                 circle_keypoints[i].pt.x,
                 circle_keypoints[i].pt.y);
    }
    if (circle_keypoints.size() > print_limit) {
        ROS_INFO("... 还有 %d 个角点未显示", (int)circle_keypoints.size() - print_limit);
    }

    // --------------------------
    // 绘制圆形范围、圆心和检测到的特征点
    // --------------------------
    // 1. 绘制圆形范围（蓝色边界，线宽2）
    cv::circle(image, center, radius, cv::Scalar(255, 0, 0), 2);

    // 2. 绘制圆心（红色实心圆，半径5）
    cv::circle(image, center, 5, cv::Scalar(0, 0, 255), -1);

    // 3. 绘制检测到的特征点（绿色）
    cv::drawKeypoints(
        image, 
        circle_keypoints, 
        image, 
        cv::Scalar(0, 255, 0), 
        cv::DrawMatchesFlags::DRAW_OVER_OUTIMG | cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
    );
}




// 图像回调函数：处理图像并调用ORB特征提取函数
void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
    try {
        // 将ROS图像消息转换为OpenCV格式
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(
            msg, 
            sensor_msgs::image_encodings::BGR8
        );
        cv::Mat image = cv_ptr->image;

        if (image.empty()) {
            ROS_WARN("Received empty image");
            return;
        }

        cv::Mat result_image;
        // 调用封装好的ORB特征提取与绘制函数
        //detectAndDrawORB(image, result_image);
        detectORBInCircle(image, cv::Point2f(200,200),80);
        // 保护共享图像数据，并更新全局结果
        {
            std::lock_guard<std::mutex> lock(global_data.mtx_image);
            result_image.copyTo(global_data.processed_image);
        }

    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    } catch (cv::Exception& e) {
        ROS_ERROR("OpenCV exception: %s", e.what());
    }
}

// 图像处理线程
void worker_task(ros::NodeHandle nh) {
    ros::Rate rate(30);  // 30Hz处理频率
    image_transport::ImageTransport it(nh);
    
    // 订阅输入图像话题
    image_transport::Subscriber image_sub = it.subscribe(
        "/kinect2/hd/image_color_rect", 
        2, 
        imageCallback
    );

    while (ros::ok()) {
        ros::spinOnce();
        rate.sleep();
    }
}

int main(int argc, char**argv) {
    // 初始化ROS节点
    ros::init(argc, argv, "orb_feature_detector_node");
    ros::NodeHandle nh;

    // 初始化图像发布器
    image_transport::ImageTransport it(nh);
    global_data.feature_pub = it.advertise("orb_features_image", 2);

    // 启动工作线程
    std::thread worker(worker_task, nh);

    // 主循环：发布处理后的图像
    ros::Rate pub_rate(10);  // 10Hz发布频率
    while (ros::ok()) {
        std_msgs::Header header;
        header.stamp = ros::Time::now();
        header.frame_id = "camera_frame";

        // 发布处理后的图像
        {
            std::lock_guard<std::mutex> lock(global_data.mtx_image);
            if (!global_data.processed_image.empty()) {
                sensor_msgs::ImagePtr msg = cv_bridge::CvImage(
                    header, 
                    "bgr8", 
                    global_data.processed_image
                ).toImageMsg();
                global_data.feature_pub.publish(msg);
            }
        }

        pub_rate.sleep();
    }

    worker.join();
    return 0;
}
