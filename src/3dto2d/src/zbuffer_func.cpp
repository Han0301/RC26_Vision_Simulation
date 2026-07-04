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
#include "package/occlusion_handing.h"     
#include "package/world_to_camera.h"  
#include "package/move_controller.h"

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
        num = Ten::_OCCLUSION_HANDING_.get_txt_flag("/home/h/RC2026/world_ws10/src/zwei/map1_add");
        Ten::_OCCLUSION_HANDING_.write_txt_flag(num, "/home/h/RC2026/world_ws10/src/zwei/map1_add");
        // num += 1;

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

    std::string num;

    nav_msgs::Odometry::ConstPtr robot_pose;  // 缓存位姿数据
    bool pose_updated = false;              // 位姿更新标记
    bool image_updated = false;             // 图像更新标记
    std::mutex data_mutex;                  // 互斥锁，防止数据竞争
}global;

void zbuffer_process()
{
    std::vector<std::vector<int>> data = {
        {1,1,1,1,0,1,0,1,0,1,1,0},   // 第1组
        {1,1,0,1,1,0,1,1,0,0,1,1},   // 第2组
        {1,1,1,1,1,0,0,1,1,1,0,0},   // 第3组
        {1,1,0,0,1,0,1,1,1,1,1,0},   // 第4组
        {1,1,1,0,0,1,0,1,1,0,1,1},   // 第5组
        {1,0,0,1,0,1,0,1,1,1,1,1},   // 第6组
        {0,1,1,0,1,1,1,0,1,1,0,1},   // 第7组
        {0,1,1,1,1,1,0,1,0,0,1,1},   // 第8组
        {1,1,1,0,1,1,1,1,1,0,0,0},   // 第9组
        {1,1,0,1,1,1,1,0,1,0,0,1},   // 第10组
        {1,0,0,1,1,1,0,1,1,1,0,1},   // 第11组
        {1,1,1,1,0,1,0,0,1,0,1,1},   // 第12组
        {0,1,1,1,1,0,0,1,1,0,1,1},   // 第13组
        {1,0,1,1,1,1,0,1,0,1,0,1},   // 第14组
        {0,1,1,1,1,1,1,1,1,0,0,0},   // 第15组
        {1,0,1,1,1,0,1,1,1,0,0,1},   // 第16组
        {1,1,1,1,1,0,1,0,0,1,0,1},   // 第17组
        {1,0,1,1,1,0,1,1,1,1,0,0},   // 第18组
        {1,1,1,0,1,1,0,0,1,1,0,1},   // 第19组
        {0,0,1,0,1,1,1,1,0,1,1,1},   // 第20组
        {1,1,0,1,0,1,0,1,1,0,1,1},   // 第21组
        {0,1,1,1,1,1,0,0,1,0,1,1},   // 第22组
        {1,1,1,1,1,1,1,0,0,0,1,0},   // 第23组
        {0,1,1,0,1,1,1,0,1,1,0,1},   // 第24组
        {0,1,0,0,1,1,0,1,1,1,1,1},   // 第25组
        {1,1,0,1,0,1,1,1,0,0,1,1},   // 第26组
        {1,1,1,0,0,1,0,0,1,1,1,1},   // 第27组
        {1,1,1,1,0,0,1,0,0,1,1,1},   // 第28组
        {1,1,0,1,1,1,1,0,1,1,0,0},   // 第29组
        {1,1,1,0,0,0,1,1,0,1,1,1},   // 第30组
        {0,1,1,1,1,1,0,1,0,1,0,1},   // 第31组
        {1,1,0,1,0,1,1,1,0,0,1,1},   // 第32组
        {1,1,1,1,0,1,1,0,0,0,1,1},   // 第33组
        {0,0,1,0,0,1,1,1,1,1,1,1},   // 第34组
        {0,1,1,1,0,0,0,1,1,1,1,1},   // 第35组
        {1,1,1,1,1,1,1,0,0,0,0,1},   // 第36组
        {1,0,0,1,1,0,1,1,1,1,0,1},   // 第37组
        {1,0,1,1,0,1,0,1,1,0,1,1},   // 第38组
        {0,1,1,1,1,1,1,0,1,0,0,1},   // 第39组
        {0,1,1,1,1,1,1,1,0,1,0,0},   // 第40组
        {0,0,1,1,1,1,1,1,0,1,1,0},   // 第41组
        {1,1,1,0,1,1,0,1,1,1,0,0},   // 第42组
        {1,1,1,0,0,1,1,0,1,1,0,1},   // 第43组
        {1,1,0,1,1,1,0,0,1,1,0,1},   // 第44组
        {1,0,1,1,1,0,1,1,0,0,1,1},   // 第45组
        {1,1,0,1,0,1,1,0,0,1,1,1},   // 第46组
        {1,1,1,0,0,1,1,0,0,1,1,1},   // 第47组
        {1,1,0,1,1,0,1,0,1,1,1,0},   // 第48组
        {1,1,1,1,1,0,1,0,0,1,1,0},   // 第49组
        {0,1,1,1,0,0,1,1,1,1,1,0},   // 第50组
        {1,1,1,1,0,1,0,0,1,1,0,1},   // 第51组
        {1,0,0,0,1,1,1,0,1,1,1,1},   // 第52组
        {0,1,1,0,1,1,1,0,1,1,0,1},   // 第53组
        {1,1,1,1,1,1,0,1,0,1,0,0},   // 第54组
        {1,1,1,1,1,0,1,0,1,0,1,0},   // 第55组
        {0,1,1,1,1,1,1,1,1,0,0,0},   // 第56组
        {1,0,0,1,0,1,1,1,1,0,1,1},   // 第57组
        {1,0,0,0,1,1,1,1,1,0,1,1},   // 第58组
        {1,1,1,0,1,1,1,0,0,1,1,0},   // 第59组
        {1,0,1,1,1,0,1,1,0,0,1,1},   // 第60组
        {0,1,0,1,0,1,1,0,1,1,1,1},   // 第61组
        {1,1,1,1,1,1,1,0,0,0,0,1},   // 第62组
        {0,1,1,1,1,1,1,0,0,0,1,1},   // 第63组
        {1,1,0,1,0,1,1,1,0,1,1,0},   // 第64组
        {1,0,1,1,1,0,1,0,0,1,1,1},   // 第65组
        {0,1,1,1,0,1,1,1,1,0,1,0},   // 第66组
        {0,1,1,1,1,1,1,0,1,1,0,0},   // 第67组
        {0,0,1,1,1,1,1,1,0,1,0,1},   // 第68组
        {1,1,1,0,1,0,1,1,0,0,1,1},   // 第69组
        {1,1,1,0,0,0,1,1,1,1,0,1},   // 第70组
        {1,1,0,1,1,1,1,0,0,0,1,1},   // 第71组
        {1,1,1,0,1,1,1,0,1,0,0,1},   // 第72组
        {1,0,0,0,1,1,0,1,1,1,1,1},   // 第73组
        {0,1,0,1,0,1,0,1,1,1,1,1},   // 第74组
        {1,0,1,0,1,1,0,1,1,1,0,1},   // 第75组
        {0,1,1,1,0,1,1,0,1,1,0,1},   // 第76组
        {1,0,1,1,0,0,1,1,1,1,1,0},   // 第77组
        {1,1,0,1,1,1,1,0,0,0,1,1},   // 第78组
        {1,1,1,1,1,0,1,1,0,0,0,1},   // 第79组
        {1,0,1,0,0,1,1,0,1,1,1,1},   // 第80组
        {1,1,1,0,1,1,0,0,1,1,0,1},   // 第81组
        {1,1,1,0,1,1,0,1,0,1,1,0},   // 第82组
        {0,1,1,1,1,0,1,0,1,1,1,0},   // 第83组
        {1,1,1,0,1,0,1,1,0,0,1,1},   // 第84组
        {1,1,1,1,0,0,1,1,1,0,1,0},   // 第85组
        {1,1,1,1,0,1,1,0,1,0,1,0},   // 第86组
        {1,0,1,1,1,1,1,0,1,0,1,0},   // 第87组
        {1,1,1,1,1,1,0,0,0,0,1,1},   // 第88组
        {1,1,1,0,1,0,1,1,1,1,0,0},   // 第89组
        {0,1,1,1,1,0,0,0,1,1,1,1},   // 第90组
        {1,0,0,1,1,0,1,1,1,1,0,1},   // 第91组
        {1,1,0,0,1,1,1,0,1,1,0,1},   // 第92组
        {0,1,1,0,1,1,1,1,0,1,0,1},   // 第93组
        {1,1,1,1,0,1,0,1,0,0,1,1},   // 第94组
        {0,1,1,1,0,1,0,1,1,1,1,0},   // 第95组
        {1,0,0,1,1,0,1,1,1,1,1,0},   // 第96组
        {1,1,1,1,1,1,0,0,0,0,1,1},   // 第97组
        {1,1,0,0,1,0,1,1,1,1,1,0},   // 第98组
        {1,0,1,1,1,1,1,1,0,0,0,1},   // 第99组
        {1,1,0,0,0,1,0,1,1,1,1,1},   // 第100组
        {1,0,1,1,0,1,1,1,1,1,0,0},   // 第101组
        {1,1,1,1,0,1,1,0,0,1,1,0},   // 第102组
        {0,1,1,0,0,1,1,1,1,1,0,1},   // 第103组
        {1,1,1,1,1,0,0,1,0,0,1,1},   // 第104组
        {1,1,0,1,0,0,1,1,0,1,1,1},   // 第105组
        {1,1,0,0,1,1,0,1,1,0,1,1},   // 第106组
        {1,1,1,0,1,1,0,0,1,1,1,0},   // 第107组
        {1,1,0,0,0,1,1,0,1,1,1,1},   // 第108组
        {1,0,0,1,1,1,1,1,0,1,0,1},   // 第109组
        {1,0,1,1,1,1,1,0,0,0,1,1},   // 第110组
        {1,1,0,1,0,1,1,1,0,1,0,1},   // 第111组
        {1,0,1,1,1,0,1,1,1,0,1,0},   // 第112组
        {1,1,1,1,0,1,0,1,0,1,0,1},   // 第113组
        {1,1,0,1,0,0,1,0,1,1,1,1},   // 第114组
        {1,0,1,0,1,0,1,1,1,0,1,1},   // 第115组
        {1,0,0,0,0,1,1,1,1,1,1,1},   // 第116组
        {0,1,1,1,0,0,1,1,1,1,1,0},   // 第117组
        {1,0,1,1,0,1,0,1,1,1,1,0},   // 第118组
        {0,1,1,1,1,0,1,1,1,0,0,1},   // 第119组
        {1,1,0,1,0,1,0,1,1,1,1,0},   // 第120组
        {1,1,0,1,1,1,0,1,0,1,1,0},   // 第121组
        {1,1,1,1,1,1,0,0,0,0,1,1},   // 第122组
        {1,1,1,1,1,0,0,0,0,1,1,1},   // 第123组
        {0,1,0,1,0,0,1,1,1,1,1,1},   // 第124组
        {1,0,1,0,1,0,1,1,0,1,1,1},   // 第125组
        {1,0,0,1,0,0,1,1,1,1,1,1},   // 第126组
        {1,1,0,0,1,1,1,0,1,1,0,1},   // 第127组
        {0,0,1,1,0,1,0,1,1,1,1,1},   // 第128组
        {0,1,1,1,1,1,1,0,1,0,1,0},   // 第129组
        {0,1,1,1,0,0,1,0,1,1,1,1},   // 第130组
        {1,0,1,1,1,1,0,1,0,0,1,1},   // 第131组
        {1,1,1,1,0,1,0,1,0,1,1,0},   // 第132组
        {0,1,1,1,0,1,1,0,1,0,1,1},   // 第133组
        {1,1,1,0,0,1,1,1,1,0,0,1},   // 第134组
        {0,1,1,1,1,0,1,1,0,1,0,1},   // 第135组
        {0,1,0,0,1,1,1,1,1,1,1,0},   // 第136组
        {1,1,1,0,0,1,1,0,1,0,1,1},   // 第137组
        {1,1,1,1,1,1,0,0,1,0,0,1},   // 第138组
        {1,1,0,1,1,1,1,1,0,0,0,1},   // 第139组
        {1,0,0,1,1,1,1,1,1,1,0,0},   // 第140组
        {1,1,1,0,1,1,1,0,0,0,1,1},   // 第141组
        {1,1,0,1,0,1,1,1,1,1,0,0},   // 第142组
        {0,1,0,0,1,1,1,1,1,1,1,0},   // 第143组
        {1,0,1,1,0,1,0,1,0,1,1,1},   // 第144组
        {0,1,0,0,0,1,1,1,1,1,1,1},   // 第145组
        {1,0,1,0,1,0,0,1,1,1,1,1},   // 第146组
        {1,1,0,1,1,0,1,0,1,1,0,1},   // 第147组
        {1,1,0,1,0,1,0,1,1,1,1,0},   // 第148组
        {1,1,1,0,0,1,1,1,1,0,0,1},   // 第149组
        {1,1,1,1,0,0,1,0,1,1,0,1}    // 第150组
    };


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
    Ten::_OCCLUSION_HANDING_.set_exist_boxes(exist_boxes);
    Ten::_OCCLUSION_HANDING_.set_interested_boxes(interested_boxes);

    // Ten::_OCCLUSION_HANDING_.set_box_lists_(global._image,  Ten::_INIT_3D_BOX_.C_object_plum_points_, 
    // Ten::_INIT_3D_BOX_.object_plum_2d_points_ ,Ten::_INIT_3D_BOX_.box_lists_);

    // Ten::_OCCLUSION_HANDING_.set_enclosure_box_(global._image,  Ten::_INIT_3D_BOX_.C_object_plum_points_, 
    // Ten::_INIT_3D_BOX_.object_plum_2d_points_ ,Ten::_INIT_3D_BOX_.box_lists_);

    // global.debug_image = Ten::_OCCLUSION_HANDING_.update_debug_image(
    //     global._image,
    //     Ten::_INIT_3D_BOX_.object_plum_2d_points_
    // );

    // std::string input;
    // std::cin >> input;
    
    // if (input == "s")
    // {
    //     std::cout << "-------------------------------------------------------" << std::endl;
    //     Ten::_OCCLUSION_HANDING_.set_box_lists_(global._image,  Ten::_INIT_3D_BOX_.C_object_plum_points_, 
    //     Ten::_INIT_3D_BOX_.object_plum_2d_points_ ,Ten::_INIT_3D_BOX_.box_lists_);

    //     Ten::_OCCLUSION_HANDING_.save_dataset(
    //         Ten::_INIT_3D_BOX_.box_lists_,
    //         global._image,
    //         {1,1,0,
    //          1,1,0,
    //          1,1,1,
    //          0,1,0},
    //         "/home/h/视频/tests"
    //     );
    // }

    // std::cout << "atoi(global.num.c_str()) : " << std::stoi(global.num) << std::endl;

    std::cout << "atoi(global.num.c_str()) : " << global.num << std::endl;

    static int save_count_sta = Ten::_OCCLUSION_HANDING_.getMaxImageNumber("/home/h/视频/tests/datasets_real");


    static int count = 0;
    if (count % 5 == 0)
    { 
        static int save_count = 1;
        
        std::cout << "-------------------------------------------------------" << std::endl;
        Ten::_OCCLUSION_HANDING_.set_box_lists_(global._image,  Ten::_INIT_3D_BOX_.C_object_plum_points_, 
        Ten::_INIT_3D_BOX_.object_plum_2d_points_ ,Ten::_INIT_3D_BOX_.box_lists_);

        Ten::_OCCLUSION_HANDING_.save_dataset(
            Ten::_INIT_3D_BOX_.box_lists_,
            global._image,
            data[atoi(global.num.c_str()) - 1],
            "/home/h/视频/datasets_real",
            save_count_sta + save_count
        );
        save_count += 1;

    }
    count += 1;



    // Ten::_OCCLUSION_HANDING_.set_debug_roi_image(Ten::_INIT_3D_BOX_.box_lists_,global.debug_best_roi_image);

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
void worker_task3(ros::NodeHandle nh)
{
    ros::Publisher cmd_vel_pub = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 10);
    Ten::_MOVE_CONTROLLER_.move_controller2(cmd_vel_pub);
}
int main(int argc, char **argv)
{
    ros::init(argc, argv, "zbuffer_func_node");
    ros::NodeHandle nh;

    std::vector<std::thread> workers;
    workers.emplace_back(worker_task1, nh);
    workers.emplace_back(worker_task2, nh);
    workers.emplace_back(worker_task3, nh);

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