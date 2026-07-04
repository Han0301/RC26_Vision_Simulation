#include <ros/ros.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <image_transport/image_transport.h>
#include <cmath>
#include <sensor_msgs/CameraInfo.h>
#include <tf2_msgs/TFMessage.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h> 
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/Imu.h>
#include <Eigen/Geometry>
#include <cstring>  
#include <iostream>
#include <thread>
#include <stdexcept>
#include <bitset>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <nav_msgs/Odometry.h>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <numeric>
#include <unordered_set>

#include "package/method_math.h"
#include "package/world_to_camera.h"  
#include "package/hsv_handing.h"
#include "package/occlusion_handing.h"
#include "package/edge_handing.cpp"
#include "package/deviation_handing.h"

void read_jpgs_by_idx_order(const std::string& img_dir,std::vector<Ten::box>& box_lists) {
    // 2. 检查目录合法性
    if (!std::filesystem::exists(img_dir) || !std::filesystem::is_directory(img_dir)) {
        ROS_ERROR("图片目录不存在或不是有效目录：%s", img_dir.c_str());
        return; // 返回全黑图的vector
    }

    // 3. 遍历文件夹中所有.jpg文件
    int success_count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(img_dir)) {
        // 过滤非文件/非jpg的项
        if (!entry.is_regular_file() || entry.path().extension() != ".png") {
            continue;
        }

        // 4. 提取文件名（如"idx9cls19conf0.053458.jpg"）
        std::string filename = entry.path().filename().string();
        
        // 5. 解析文件名中的idx数字（核心：定位idx和cls的位置）
        size_t idx_pos = filename.find("idx");
        size_t cls_pos = filename.find("cls");
        if (idx_pos == std::string::npos || cls_pos == std::string::npos || cls_pos <= idx_pos + 3) {
            //ROS_WARN("文件名格式错误，跳过：%s", filename.c_str());
            continue;
        }

        // 6. 截取idx后的数字字符串（如"idx9cls..." → 截取"9"）
        std::string idx_str = filename.substr(idx_pos + 3, cls_pos - (idx_pos + 3));
        int img_idx = -1;
        try {
            img_idx = std::stoi(idx_str);
        } catch (...) {
            //ROS_WARN("idx数字转换失败，跳过：%s", filename.c_str());
            continue;
        }

        // 7. 验证idx范围（必须1-12）
        if (img_idx < 1 || img_idx > 12) {
            //ROS_WARN("idx超出1-12范围，跳过：%s (idx=%d)", filename.c_str(), img_idx);
            continue;
        }

        // 8. 读取图片（BGR格式，匹配你的roi_image）
        cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_UNCHANGED);
        if (img.empty()) {
            //ROS_WARN("图片读取失败，跳过：%s", entry.path().c_str());
            continue;
        }

        // 9. 统一resize为160×160（和你代码中roi_image尺寸一致）
        //cv::resize(img, img, cv::Size(160, 160), 0, 0, cv::INTER_LINEAR);

        // 10. 按idx顺序存入vector（idx1→索引0，idx12→索引11）
        int vec_idx = img_idx - 1;
        box_lists[vec_idx].roi_image = img.clone(); // 深拷贝，避免数据共享
        success_count += 1;
        //ROS_INFO("成功读取idx=%d的图片：%s", img_idx, entry.path().c_str());
    }
    std::cout << "success_count : " << success_count << std::endl; 
}
struct G
{
    G()
    {
        // 1. 相机内参矩阵 K
        _K = (cv::Mat_<double>(3,3) <<
            1012.0711525658555, 0, 960.5,
            0, 1012.0711525658555, 540.5,
            0, 0, 1);
        // 2. 畸变系数（假设零畸变）
        _distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
    }

    std::vector<Ten::box> box_lists;

    bool is_move = true;

    cv::Mat _K;
    cv::Mat _distCoeffs;

    cv::Mat _image;
    cv::Mat debug_image;
    cv::Mat debug_best_roi_image = cv::Mat::zeros(480, 640, CV_8UC3);;
    std::mutex _mtx_image;

    image_transport::Publisher zbuffer_pub;


    nav_msgs::Odometry::ConstPtr robot_pose;  // 缓存位姿数据
    bool pose_updated = false;              // 位姿更新标记
    bool image_updated = false;             // 图像更新标记
    std::mutex data_mutex;                  // 互斥锁，防止数据竞争
}global;

void zbuffer_process()
{
    if (!global.pose_updated || !global.image_updated)
    {
        ROS_DEBUG("数据未更新，跳过处理");
        return;
    }

    Ten::XYZRPY tf;
    {
        std::lock_guard<std::mutex> lock(global.data_mutex);
        tf = Ten::Nav_Odometrytoxyzrpy(*global.robot_pose);
        float x = tf._xyz._x;
        tf._xyz._x = -tf._xyz._y;
        tf._xyz._y = x;
    }

    Ten::XYZRPY wt;
    wt._xyz._z = 1.25;
    wt._rpy._roll = - M_PI / 2;
    //wt._rpy._yaw = -M_PI / 2;
    Eigen::Matrix4d transform_matrix = worldtocurrent(wt._xyz, wt._rpy);
    
    Ten::_CAMERA_TRANSFORMATION_.camerainfo_.set_Extrinsic_Matrix(transform_matrix);
    Ten::_CAMERA_TRANSFORMATION_.camerainfo_.set_K(global._K);

    // std::cout << "tf.x: " << tf._xyz._x  << std::endl;
    // std::cout << "tf.y: " << tf._xyz._y  << std::endl;
    // std::cout << "tf.z: " << tf._xyz._z  << std::endl;

    Ten::_CAMERA_TRANSFORMATION_.set_worldtolidar(tf);
    Ten::_CAMERA_TRANSFORMATION_.pcl_transform_world_to_camera(Ten::_INIT_3D_BOX_.pcl_LM_plum_object_points_, 
    Ten::_INIT_3D_BOX_.pcl_C_plum_object_points_, Ten::_INIT_3D_BOX_.object_plum_2d_points_);
    Ten::_INIT_3D_BOX_.pcl_to_C();

    int exist_boxes[12] = {1,1,1,1,1,1,1,1,1,1,1,1};
    int interested_boxes[12] = {1,1,1,1,1,1,1,1,1,1,1,1};
    Ten::_OCCLUSION_HANDING_.set_exist_boxes(exist_boxes);
    Ten::_OCCLUSION_HANDING_.set_interested_boxes(interested_boxes);

    Ten::_OCCLUSION_HANDING_.set_box_lists_(global._image,  Ten::_INIT_3D_BOX_.C_object_plum_points_, 
    Ten::_INIT_3D_BOX_.object_plum_2d_points_ ,Ten::_INIT_3D_BOX_.box_lists_, Ten::_INIT_3D_BOX_.object_zbuffer);

    // std::string img_dir = "/home/h/rc26_log/log/2026_1_10/10/image/image10";
    // read_jpgs_by_idx_order(img_dir,Ten::_INIT_3D_BOX_.box_lists_);   

    Ten::_OCCLUSION_HANDING_.set_debug_roi_image(Ten::_INIT_3D_BOX_.box_lists_,global.debug_best_roi_image);

 
    Ten::_HSV_HANDING_.set_hsv_topn_stand(Ten::_INIT_3D_BOX_.box_lists_,5,600,Ten::_INIT_HSV_.standard_hsv_,Ten::_INIT_HSV_.score_lists_);
    std::cout << "set_hsv_topn_stand(5): " << Ten::_INIT_HSV_.standard_hsv_[0] << "  " << Ten::_INIT_HSV_.standard_hsv_[1] << "  " << Ten::_INIT_HSV_.standard_hsv_[2] << "  "<< std::endl;

    Ten::_HSV_HANDING_.set_hsv_topn_score(Ten::_INIT_3D_BOX_.box_lists_,5,600,Ten::_INIT_HSV_.standard_hsv_,Ten::_INIT_HSV_.score_lists_);

    // Ten::_HSV_HANDING_.print_score_lists(Ten::_INIT_HSV_.score_lists_);


    // 测试 边缘检测

    // sobelEdgeDetect(global._image, low_hsv, high_hsv);
    // scharrEdgeDetect(global._image, low_hsv, high_hsv);
    // prewittEdgeDetect(global._image, low_hsv, high_hsv);
    // laplacianEdgeDetect(global._image, low_hsv, high_hsv);
    // logEdgeDetect(global._image, low_hsv, high_hsv);
    // morphGradientEdgeDetect(global._image, low_hsv, high_hsv);
    // adaptiveCannyEdgeDetect(global._image, low_hsv, high_hsv);

    // findContoursEdgeDetect(global._image, low_hsv, high_hsv);
    // robertsEdgeDetect(global._image, low_hsv, high_hsv);


    // 掩玛覆盖测试
    std::vector<std::vector<cv::Point2f>> pending_points;
    std::vector<std::vector<cv::Point2f>> target_points;

    // 根据 standard_hsv 生成 low,high hsv
    Ten::_DEVIATION_HANDING_.set_low_high_hsv(Ten::_INIT_HSV_.standard_hsv_, 6, Ten::_INIT_DEVIATION_.low_hsv, Ten::_INIT_DEVIATION_.high_hsv);

    // 填充 pending_points
    for(int i = 0; i < 96; i+=8)
    {
        std::vector<cv::Point2f> points = { Ten::_INIT_3D_BOX_.object_plum_2d_points_[i], 
                                            Ten::_INIT_3D_BOX_.object_plum_2d_points_[i + 1],
                                            Ten::_INIT_3D_BOX_.object_plum_2d_points_[i + 2],
                                            Ten::_INIT_3D_BOX_.object_plum_2d_points_[i + 3]};
        pending_points.push_back(points);
    }


    // 填充 target_points
    Ten::_DEVIATION_HANDING_.mask_cover(global._image, Ten::_INIT_DEVIATION_.low_hsv, Ten::_INIT_DEVIATION_.high_hsv,Ten::_INIT_DEVIATION_.tar_mask, target_points);

    // 测试 四个点的 偏差配准
    Ten::_DEVIATION_HANDING_.registration(pending_points, target_points,Ten::_INIT_DEVIATION_.final_results);



    // 测试 整个掩玛的偏差配准
    // Ten::_DEVIATION_HANDING_.createZBufferMask(Ten::_INIT_3D_BOX_.object_zbuffer, Ten::_INIT_DEVIATION_.zb_mask);
    // Ten::_DEVIATION_HANDING_.createHSVMask(global._image, Ten::_INIT_DEVIATION_.low_hsv, Ten::_INIT_DEVIATION_.high_hsv,Ten::_INIT_DEVIATION_.tar_mask);
    // Ten::_DEVIATION_HANDING_.getMask_centroid(Ten::_INIT_DEVIATION_.zb_mask,Ten::_INIT_DEVIATION_.zb_centroid);
    // Ten::_DEVIATION_HANDING_.getMask_centroid(Ten::_INIT_DEVIATION_.tar_mask,Ten::_INIT_DEVIATION_.tar_centroid);

    // std::cout << "zb_centroid: " << Ten::_INIT_DEVIATION_.zb_centroid << std::endl;
    // std::cout << "tar_centroid: " << Ten::_INIT_DEVIATION_.tar_centroid << std::endl;

    // // 显示 mask 覆盖的区域
    // for(int i = 0; i < global._image.cols;i++)
    // {
    //     for (int j = 0; j < global._image.rows; j++)
    //     {
    //         if (Ten::_INIT_DEVIATION_.zb_mask.at<uchar>(j,i) == 255)
    //         {
    //             cv::circle(global._image, cv::Point(i,j), 5, cv::Scalar(0, 0, 255), -1);
    //         }
    //     }
    // }



    global.debug_image = Ten::_OCCLUSION_HANDING_.update_debug_image(
        global._image,
        Ten::_INIT_3D_BOX_.object_plum_2d_points_
    );
}

// 回调函数1：处理/robot_pose话题
void odomCallback(const nav_msgs::Odometry::ConstPtr& msg)
{
    std::lock_guard<std::mutex> lock(global.data_mutex); // 加锁保证线程安全
    global.robot_pose = msg;
    global.pose_updated = true; // 标记位姿已更新
}

// 回调函数2：处理/kinect2/hd/image_color_rect话题
void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    std::lock_guard<std::mutex> lock(global.data_mutex); // 加锁保证线程安全
    try
    {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        global._image =  cv_ptr->image;
        global.image_updated = true; // 标记图像已更新
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
}
void worker_task1(ros::NodeHandle nh)
{
    ros::Rate sl(50);
    ros::Subscriber tf_sub = nh.subscribe("/robot_pose", 2, odomCallback);
    while (ros::ok())
    {
        ros::spinOnce();
        sl.sleep();
    }
}
void worker_task2(ros::NodeHandle nh)
{
    ros::Rate sl(10);
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber image_sub = it.subscribe("/kinect2/hd/image_color_rect", 2, imageCallback);
    while (ros::ok())
    {
        ros::spinOnce();
        sl.sleep();
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "zbuffer_func_node");
    ros::NodeHandle nh;
    std::vector<std::thread> workers;
    workers.emplace_back(worker_task1, nh);
    workers.emplace_back(worker_task2, nh);

    image_transport::ImageTransport it(nh);
    image_transport::Publisher debug_image_pub = it.advertise("pub_image_topic", 2);
    image_transport::Publisher debug_roi_pub = it.advertise("/zbuffer_visualization", 30);

    ros::Rate rate(10);
    while(ros::ok())
    {
        sensor_msgs::ImagePtr msg;
        sensor_msgs::ImagePtr roi_msg;
        {
            std::lock_guard<std::mutex> lock(global._mtx_image);
            msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", global.debug_image).toImageMsg();
            roi_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", global.debug_best_roi_image).toImageMsg();
        }
        debug_image_pub.publish(msg);
        debug_roi_pub.publish(roi_msg);
        zbuffer_process();
        // std::cout << "publish success" << std::endl;
        rate.sleep();
    }

    for (auto& worker : workers) {
        worker.join();
    }
    return 0;
}