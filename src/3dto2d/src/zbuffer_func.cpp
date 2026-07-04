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
#include "package/zbuffer_simplify.h"     
#include "package/world_to_camera.h"  
#include "package/debug_hsv.h"


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

    std::cout << "tf.x: " << tf._xyz._x  << std::endl;
    std::cout << "tf.y: " << tf._xyz._y  << std::endl;
    std::cout << "tf.z: " << tf._xyz._z  << std::endl;

    Ten::_CAMERA_TRANSFORMATION_.set_worldtolidar(tf);
    Ten::_CAMERA_TRANSFORMATION_.pcl_transform_world_to_camera(Ten::_INIT_3D_BOX_.pcl_LM_plum_object_points_, 
    Ten::_INIT_3D_BOX_.pcl_C_plum_object_points_, Ten::_INIT_3D_BOX_.object_plum_2d_points_);
    Ten::_INIT_3D_BOX_.pcl_to_C();

    int exist_boxes[12] = {1,1,1,1,1,1,1,1,1,1,1,1};
    int interested_boxes[12] = {1,1,1,1,1,1,1,1,1,1,1,1};
    Ten::_ZBUFFER_SIMPLIFY_.set_exist_boxes(exist_boxes);
    Ten::_ZBUFFER_SIMPLIFY_.set_interested_boxes(interested_boxes);

    Ten::_ZBUFFER_SIMPLIFY_.set_box_lists_(global._image,  Ten::_INIT_3D_BOX_.C_object_plum_points_, 
        Ten::_INIT_3D_BOX_.object_plum_2d_points_ ,Ten::_INIT_3D_BOX_.box_lists_);

    global.debug_image = Ten::_ZBUFFER_SIMPLIFY_.update_debug_image(
        global._image,
        Ten::_INIT_3D_BOX_.object_plum_2d_points_
    );

    Ten::_ZBUFFER_SIMPLIFY_.set_debug_roi_image(Ten::_INIT_3D_BOX_.box_lists_, Ten::_INIT_3D_BOX_.score_lists_,global.debug_best_roi_image);

    std::string img_dir = "/home/h/rc26_log/log/2026_1_12/23/image/image1";
    Ten::_DEBUG_HSV_.read_jpgs_by_idx_order(img_dir,Ten::_INIT_3D_BOX_.box_lists_);   

    static int image_num = 0;
    std::vector<cv::Mat> hist_lists_36;
    for (int i = 0; i < Ten::_INIT_3D_BOX_.box_lists_.size(); i ++)
    {
        //保存直方图
        // cv::imshow("img",Ten::_INIT_3D_BOX_.box_lists_[i].roi_image);
        // cv::waitKey(0);
        //std::vector<cv::Mat> hist_lists = Ten::_DEBUG_HSV_.set_hsv_hist(Ten::_INIT_3D_BOX_.box_lists_[i].roi_image);
        //std::vector<int> hsv_mode_honye= Ten::_DEBUG_HSV_.bgr_color_analysis(Ten::_INIT_3D_BOX_.box_lists_[i].roi_image);
        //std::cout << "hsv_houye: " << hsv_mode_honye[0]<<"  "  <<  hsv_mode_honye[1]<<"  " << hsv_mode_honye[2]<<"  "<< std::endl; 
        // std::string save_path = "/home/h/RC2026/merge_ws20/log/2026_1_9/14/image/image4/" + std::to_string(image_num) + "_hist" + std::to_string(i + 1) + ".jpg";
        // Ten::_DEBUG_HSV_.save_hsv_hist_visualization(hist_lists,save_path,90,128,128);
        // std::string image_save_path = "/home/h/RC2026/world_ws7/src/3dto2d/debug/" + std::to_string(image_num) + "_img" + std::to_string(i + 1) + ".jpg";

        // 保存图像
        // bool success = cv::imwrite(image_save_path,Ten::_INIT_3D_BOX_.box_lists_[i].roi_image);
        // if (!success)
        // {
        //     std::cout << "图像保存失败： " << save_path << std::endl;
        // }
        // std::tuple<int,int,int>hsv_mode = Ten::_DEBUG_HSV_.get_hist_mode(hist_lists);
        // std::cout << "idx: " << i + 1 << ", hsv_mode: " << std::get<0>(hsv_mode) << "  "<<std::get<1>(hsv_mode) << "  " << std::get<2>(hsv_mode) << "  ";
    // hist_lists_36.push_back(hist_lists[0]);
    // hist_lists_36.push_back(hist_lists[1]);
    // hist_lists_36.push_back(hist_lists[2]);
    }
    image_num += 1;


    // 通过众数比较分类
    //Ten::_ZBUFFER_SIMPLIFY_.set_hsv_mode(Ten::_INIT_3D_BOX_.box_lists_, Ten::_INIT_3D_BOX_.score_lists_);
    
    Ten::_ZBUFFER_SIMPLIFY_.set_hsv_topn_stand(Ten::_INIT_3D_BOX_.box_lists_, Ten::_INIT_3D_BOX_.score_lists_,5);
    std::cout << "set_hsv_topn_stand(5): " << Ten::_ZBUFFER_SIMPLIFY_.get_standard_hsv_()[0] << "  " << Ten::_ZBUFFER_SIMPLIFY_.get_standard_hsv_()[1] << "  " << Ten::_ZBUFFER_SIMPLIFY_.get_standard_hsv_()[2] << "  "<< std::endl;
    
    // Ten::_ZBUFFER_SIMPLIFY_.set_hsv_topn_stand(Ten::_INIT_3D_BOX_.box_lists_, Ten::_INIT_3D_BOX_.score_lists_,10);
    // std::cout << "set_hsv_topn_stand(10): " << Ten::_ZBUFFER_SIMPLIFY_.get_standard_hsv_()[0] << "  " << Ten::_ZBUFFER_SIMPLIFY_.get_standard_hsv_()[1] << "  " << Ten::_ZBUFFER_SIMPLIFY_.get_standard_hsv_()[2] << "  "<< std::endl;

    // std::cout << "-----use mode--------------" << std::endl;
    // for(int i = 0;i < Ten::_INIT_3D_BOX_.score_lists_.size(); i ++)
    // {
    //     std::cout << "idx : "<< Ten::_INIT_3D_BOX_.score_lists_[i].idx << ",  hsv_mode: " << std::get<0>(Ten::_INIT_3D_BOX_.score_lists_[i].hsv_mode) << " " <<  std::get<1>(Ten::_INIT_3D_BOX_.score_lists_[i].hsv_mode) << " " << std::get<2>(Ten::_INIT_3D_BOX_.score_lists_[i].hsv_mode) << std::endl; 
    //     std::cout <<"      " << "score: " << Ten::_INIT_3D_BOX_.score_lists_[i].hsv_score << std::endl; 

    //     std::cout << " " << std::endl;
    // }

    //Ten::_ZBUFFER_SIMPLIFY_.set_hsv_top_n(Ten::_INIT_3D_BOX_.box_lists_, Ten::_INIT_3D_BOX_.score_lists_,5);
    Ten::_ZBUFFER_SIMPLIFY_.set_hsv_topn_score(Ten::_INIT_3D_BOX_.box_lists_, Ten::_INIT_3D_BOX_.score_lists_,5);
    std::cout << "---------🎃❌🌚" << std::endl;
    // std::cout << "-----use top5--------------" << std::endl;
    // for(int i = 0;i < Ten::_INIT_3D_BOX_.score_lists_.size(); i ++)
    // {
    //     std::cout << "idx : "<< Ten::_INIT_3D_BOX_.score_lists_[i].idx << ", score: " << Ten::_INIT_3D_BOX_.score_lists_[i].hsv_score << std::endl; 

    //     if (Ten::_INIT_3D_BOX_.score_lists_[i].hsv_score  < 0){
    //         continue;
    //     }
    //     for(int j = 0; j < 5; j++ )
    //     {
    //         std::cout << "    j:"<< j << std::endl;
    //         std::cout << "   top_n_h: "<<Ten::_INIT_3D_BOX_.score_lists_[i].top_n_h[j].position << ", " << Ten::_INIT_3D_BOX_.score_lists_[i].top_n_h[j].count
    //                 << ", " << Ten::_INIT_3D_BOX_.score_lists_[i].top_n_h[j].degree_of_promacy << std::endl;
    //         std::cout << "   top_n_s: "<<Ten::_INIT_3D_BOX_.score_lists_[i].top_n_s[j].position << ", " << Ten::_INIT_3D_BOX_.score_lists_[i].top_n_s[j].count
    //                 << ", " << Ten::_INIT_3D_BOX_.score_lists_[i].top_n_s[j].degree_of_promacy << std::endl;
    //         std::cout << "   top_n_v: "<<Ten::_INIT_3D_BOX_.score_lists_[i].top_n_v[j].position << ", " << Ten::_INIT_3D_BOX_.score_lists_[i].top_n_v[j].count
    //                 << ", " << Ten::_INIT_3D_BOX_.score_lists_[i].top_n_v[j].degree_of_promacy << std::endl;
    //     }
    //     std::cout << " " << std::endl;
    // }

    // Ten::_ZBUFFER_SIMPLIFY_.set_hsv_topn_score(Ten::_INIT_3D_BOX_.box_lists_, Ten::_INIT_3D_BOX_.score_lists_,10);
    // std::cout << "-----use top10--------------" << std::endl;
    // for(int i = 0;i < Ten::_INIT_3D_BOX_.score_lists_.size(); i ++)
    // {
    //     std::cout << "idx : "<< Ten::_INIT_3D_BOX_.score_lists_[i].idx << ", score: " << Ten::_INIT_3D_BOX_.score_lists_[i].hsv_score << std::endl; 

    //     // for(int j = 0; j < 5; j++ )
    //     // {
    //     //     std::cout << "    j:"<< j << std::endl;
    //     //     std::cout << "   top_n_h: "<<Ten::_INIT_3D_BOX_.score_lists_[i].top_n_h[j].position << ", " << Ten::_INIT_3D_BOX_.score_lists_[i].top_n_h[j].count
    //     //             << ", " << Ten::_INIT_3D_BOX_.score_lists_[i].top_n_h[j].degree_of_promacy << std::endl;
    //     //     std::cout << "   top_n_s: "<<Ten::_INIT_3D_BOX_.score_lists_[i].top_n_s[j].position << ", " << Ten::_INIT_3D_BOX_.score_lists_[i].top_n_s[j].count
    //     //             << ", " << Ten::_INIT_3D_BOX_.score_lists_[i].top_n_s[j].degree_of_promacy << std::endl;
    //     //     std::cout << "   top_n_v: "<<Ten::_INIT_3D_BOX_.score_lists_[i].top_n_v[j].position << ", " << Ten::_INIT_3D_BOX_.score_lists_[i].top_n_v[j].count
    //     //             << ", " << Ten::_INIT_3D_BOX_.score_lists_[i].top_n_v[j].degree_of_promacy << std::endl;
    //     // }
    //     std::cout << " " << std::endl;
    // }


    // std::cout <<std::endl;

    // 通过平均数比较
    // Ten::_ZBUFFER_SIMPLIFY_.set_hsv_average(Ten::_INIT_3D_BOX_.box_lists_, Ten::_INIT_3D_BOX_.score_lists_);
    // for(int i = 0;i < Ten::_INIT_3D_BOX_.score_lists_.size(); i ++)
    // {
    //     std::cout << "idx : "<<i + 1 << ",  hsv_average: " << std::get<0>(Ten::_INIT_3D_BOX_.score_lists_[i].hsv_average) << " " <<  std::get<1>(Ten::_INIT_3D_BOX_.score_lists_[i].hsv_average) << " " << std::get<2>(Ten::_INIT_3D_BOX_.score_lists_[i].hsv_average) << std::endl; 
    //     //std::cout << "         "  << Ten::_INIT_3D_BOX_.score_lists_[i].hsv_score << std::endl; 
    // }
    
    // 直方图的相似性比较分类
    // std::vector<Ten::HistCompareResult> sim_compare_;
    // for (int i = 0;i < 12; i++)
    // {
    //     std::vector<cv::Mat> hist_lists_i= Ten::_DEBUG_HSV_.set_hsv_hist(Ten::_INIT_3D_BOX_.box_lists_[i].roi_image);
    //     for (int j = i + 1;j < 12; j ++)
    //     {
    //         std::vector<cv::Mat> hist_lists_j = Ten::_DEBUG_HSV_.set_hsv_hist(Ten::_INIT_3D_BOX_.box_lists_[j].roi_image);
    //         //std::cout << "i + 1: " << i + 1 << ", j + 1: " << j + 1 << std::endl;
    //         double h_sim,s_sim,v_sim,hsv_sim;
    //         Ten::_DEBUG_HSV_.compare_hsv_hist(hist_lists_i,hist_lists_j,h_sim,s_sim,v_sim,hsv_sim,cv::HISTCMP_BHATTACHARYYA);
    //         sim_compare_.push_back({
    //             i + 1,
    //             j + 1,
    //             h_sim,
    //             s_sim,
    //             v_sim,
    //             hsv_sim
    //         });
    //     }
    // }

    // Ten::_DEBUG_HSV_.cluster_box_images(sim_compare_);

    // 相似性比较打印
    //std::string txt_path = "/home/h/RC2026/world_ws7/src/3dto2d/debug/" + std::to_string(image_num) + "_compare" + ".txt";
    //bool save_txt = Ten::_DEBUG_HSV_.batch_compare_hist_and_save(txt_path,hist_lists_36,cv::HISTCMP_BHATTACHARYYA,0.3,0.3,0.4);
    // if (!save_txt)
    // {
    //     std::cout << "直方图比较结束，txt保存失败： " << txt_path << std::endl;
    // }



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