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

//点云 cloud map 
//tf cloud->map
//8 判断点的数量
//  订阅雷达话题转化成pcl点云，判断pcl点（坐标变换部分暂时不考虑，假设在一个坐标系中直接比较）的数据是否在指定的区域内
// 实现： 改成8个点的立体区域， 重新定义结构体，判断

#define _L_ 1.2
#define _H_ 0.2
#define _ly1_ 0.425
#define _ly2_ 0.775
#define _lx1_ 0.425
#define _lh_ 0.35 
#define _X_  3.17              //2.17
#define _Y_  1.2             //0.2 

struct G
{
    G()
    {
        // 2. 相机内参矩阵 K
        _K = (cv::Mat_<double>(3,3) <<
            1012.0711525658555, 0, 960.5,
            0, 1012.0711525658555, 540.5,
            0, 0, 1);
        // 3. 畸变系数（假设零畸变）
        _distCoeffs = cv::Mat::zeros(5, 1, CV_64F);

        // 4. 外参：旋转向量和平移向量
        _rvec = (cv::Mat_<double>(3,1) << 0.0, 0.0, 0.0); // 旋转（弧度）
        _tvec = (cv::Mat_<double>(3,1) << 0.0, 0.0, 0.0);   // 平移（米）
        // 1. 定义3D点（世界坐标）
        //std::vector<cv::Point3f> objectPoints;
        //objectPoints.push_back(cv::Point3f(0, 0, 0));   // 原点

        for(int j = 0; j < 4; j++)
        {
            for(int i = 0; i < 3; i++)
            {
                _objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_, _Y_ + i*_L_ + _ly1_, _arr[j*3+i]+_lh_));
                _objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_, _Y_ + i*_L_ + _ly2_, _arr[j*3+i]+_lh_));
                _objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_, _Y_ + i*_L_ + _ly2_, _arr[j*3+i]));
                _objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_, _Y_ + i*_L_ + _ly1_, _arr[j*3+i]));
                //std::cout << "_X_ + j*_L_: " << _X_ + j*_L_ << "_Y_ + i*_L_: " << _Y_ + i*_L_ <<"_X_ + (j + 1)*_L_: " <<  _X_ + (j + 1)*_L_<< "_Y_ + (i + 1)*_L_: "<< _Y_ + (i + 1)*_L_<< std::endl;
            }
        }

        for(int j = 0; j < 4; j++)
        {
            for(int i = 0; i < 3; i++)
            {
                side_objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_ + _lh_, _Y_ + i*_L_ + _ly1_, _arr[j*3+i]+_lh_));
                side_objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_ + _lh_, _Y_ + i*_L_ + _ly2_, _arr[j*3+i]+_lh_));
                side_objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_ + _lh_, _Y_ + i*_L_ + _ly2_, _arr[j*3+i]));
                side_objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_ + _lh_, _Y_ + i*_L_ + _ly1_, _arr[j*3+i]));

                side_objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_, _Y_ + i*_L_ + _ly2_, _arr[j*3+i]+_lh_));                
                side_objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_ + _lh_, _Y_ + i*_L_ + _ly2_, _arr[j*3+i]+_lh_));                
                side_objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_ + _lh_, _Y_ + i*_L_ + _ly2_, _arr[j*3+i]));
                side_objectPoints.push_back(cv::Point3f(_X_ + j*_L_ + _lx1_, _Y_ + i*_L_ + _ly2_, _arr[j*3+i]));

            }
        }        
        //std::cout << side_objectPoints <<std::endl;
        for(int i = 0; i < 48; i++)
        {
            //_c_objectPoints.push_back(cv::Point3f(-_objectPoints[i].y, 1.3 - _objectPoints[i].z, _objectPoints[i].x));
            _c_objectPoints.push_back(cv::Point3f(_objectPoints[i].y, 1.3 - _objectPoints[i].z, _objectPoints[i].x));
        };
        std::cout << _c_objectPoints <<std::endl;
        for(int i = 0; i < 96; i++)
        {
            //_c_objectPoints.push_back(cv::Point3f(-_objectPoints[i].y, 1.3 - _objectPoints[i].z, _objectPoints[i].x));
            _c_side_objectPoints.push_back(cv::Point3f(side_objectPoints[i].x, side_objectPoints[i].y, 1.3 - side_objectPoints[i].z));
        };
        //std::cout << _c_side_objectPoints <<std::endl;
        
        for(size_t i = 0; i < _c_side_objectPoints.size(); i += 8)
        {
            if(i + 7 >= _c_side_objectPoints.size()) break;

            // 计算8个点形成的立方体3D空间范围（x,y,z的min和max）
            float x_min = _c_side_objectPoints[i].x, x_max = _c_side_objectPoints[i].x;
            float y_min = _c_side_objectPoints[i].y, y_max = _c_side_objectPoints[i].y;
            float z_min = _c_side_objectPoints[i].z, z_max = _c_side_objectPoints[i].z;

            // 遍历8个点更新min/max
            for(int k = 1; k < 8; k++){
                x_min = std::min(x_min, _c_side_objectPoints[i+k].x);
                x_max = std::max(x_max, _c_side_objectPoints[i+k].x);
                y_min = std::min(y_min, _c_side_objectPoints[i+k].y);
                y_max = std::max(y_max, _c_side_objectPoints[i+k].y);
                z_min = std::min(z_min, _c_side_objectPoints[i+k].z);
                z_max = std::max(z_max, _c_side_objectPoints[i+k].z);
            }

            // 添加到idx_f_b_box_list
            std::cout << "will push_back: " << (int)(i/8) + 1 << "  x: " << x_min <<" " << x_max << " y: " << y_min << " " << y_max << " z: " << z_min << " " << z_max << std::endl;
            idx_f_b_box_list.push_back({
                (int)(i/8) + 1,  // idx从1开始
                x_min, x_max,    // 3D X范围
                y_min, y_max,    // 3D Y范围
                z_min, z_max,     // 3D Z范围
                false,false
            });
        }

    }
    struct idx_box{
        int idx;
        float box_x1;
        float box_y1;
        float box_x2;
        float box_y2;
        float distance;
    };

    struct idx_f_b_box{
        int idx;                      
        float x_min, x_max;    
        float y_min, y_max;     
        float z_min, z_max;   
        float r_x_min,r_x_max,r_y_min,r_y_max;      // 相对机器人的x,y坐标 
        float average_x,average_y, average_z;
        float front_depth,left_depth,right_depth,side_depth,up_depth;
        bool left_occluders,right_occluders;       // 左右面是否被遮挡，被遮挡了，值为false
    };

    struct front_2d_point{
        int idx;
        cv::Point2f left_up;
        cv::Point2f right_up;
        cv::Point2f right_down;
        cv::Point2f left_down;
        float front_depth;
    };
    struct side_2d_point{
        int idx;
        cv::Point2f left_up;
        cv::Point2f right_up;
        cv::Point2f right_down;
        cv::Point2f left_down;
        float side_depth;
    };
    struct up_2d_point{
        int idx;
        cv::Point2f left_up;
        cv::Point2f right_up;
        cv::Point2f right_down;
        cv::Point2f left_down;
        float up_depth;
    };
    
    float x;
    float y;
    float z;
    double yaw;
    float last_robot_x = INFINITY;
    float last_robot_y = INFINITY;
    double last_yaw = INFINITY;

    //std::vector<struct standard_3d_x_y> standard_3d_x_y_lists;    
    std::vector<struct idx_box> idx_box_lists;
    std::vector<struct idx_f_b_box> idx_f_b_box_list; 
    std::vector<struct side_2d_point> side_2d_points_lists;
    std::vector<struct front_2d_point> front_2d_point_lists;
    std::vector<struct up_2d_point> up_2d_point_lists;
    std::vector<cv::Point3f> _objectPoints;
    std::vector<cv::Point3f> _c_objectPoints;
    std::vector<cv::Point3f> side_objectPoints;
    std::vector<cv::Point3f> _c_side_objectPoints;

    cv::Mat _K;
    cv::Mat _distCoeffs;
    cv::Mat _rvec;
    cv::Mat _tvec;
    cv::Mat _image;
    std::mutex _mtx_image;
    std::mutex _mtx_tf;
    std::vector<cv::Point2f> imagePoints;
    std::vector<cv::Point2f> side_imagePoints;
    float _arr[12] = {0.4, 0.2, 0.4, 0.2, 0.4, 0.6, 0.4, 0.6, 0.4, 0.2, 0.4, 0.2};
    image_transport::Publisher zbuffer_pub;
}global;

// ---------------------------------------------------------------------------------------------------------------------------------------------- zbufer 相关调试可视化 ---------------
// void saveZBufferVisualization(const cv::Mat& zbuffer, const std::string& save_path) {
cv::Mat saveZBufferVisualization(const cv::Mat& zbuffer) {
    // 1. 克隆矩阵（避免修改原数据）
    cv::Mat zbuffer_processed = zbuffer.clone();

    // 2. 处理无效值（FLT_MAX）并计算有效深度的最大最小值
    cv::Mat mask = zbuffer_processed != FLT_MAX; // 有效像素掩码（排除无效值）
    double min_val, max_val;
    cv::minMaxLoc(zbuffer_processed, &min_val, &max_val, nullptr, nullptr, mask);

    // 处理全是无效值的极端情况
    if (mask.empty() || cv::countNonZero(mask) == 0) {
        std::cerr << "[ERROR] Z-Buffer中无有效深度值" << std::endl;
    }

    // 将无效值设为0（后续反转后仍为黑色）
    zbuffer_processed.setTo(0, ~mask);

    // 3. 归一化深度值到[0, 255]：原min_val→0，原max_val→255
    cv::Mat zbuffer_vis;
    cv::normalize(zbuffer_processed, zbuffer_vis, 0, 255, cv::NORM_MINMAX, CV_8U, mask);

    // 关键修改：反转像素值，实现“近白远黑”
    // 反转后：原min_val（近）→255（白），原max_val（远）→0（黑）
    zbuffer_vis = 255 - zbuffer_vis;

    // 确保无效值仍为黑色（反转后可能被影响，重新覆盖）
    zbuffer_vis.setTo(0, ~mask);

    // 4. 保存图片到指定路径
    //bool success = cv::imwrite(save_path, zbuffer_vis);
    //return success;
    return zbuffer_vis;
}


// 生成任意两顶点间线段上的所有整数像素点
std::vector<cv::Point> generate_edge_pixels(const cv::Point2f& vtx1, const cv::Point2f& vtx2) {
    std::vector<cv::Point> pixels;
    //std::cout << "input: " << vtx1 << "  " << vtx2 << std::endl;
    // 计算方向向量与长度平方（避免开根号）
    float dx = vtx2.x - vtx1.x;
    float dy = vtx2.y - vtx1.y;
    float len_sq = dx*dx + dy*dy;
    
    // 两点重合，直接返回
    if (len_sq < 1e-12f) return pixels;
    
    // 计算步数：取dx/dy的最大绝对值，确保覆盖整个线段
    int steps = static_cast<int>(std::max(std::abs(dx), std::abs(dy))) + 1;
    steps = std::max(steps, 2); // 至少2个点（避免除以0）
    
    // 线性插值生成像素点
    for (int i = 0; i < steps; ++i) {
        float t = static_cast<float>(i) / (steps - 1); // 参数t∈[0,1]
        float x = vtx1.x + t * dx;
        float y = vtx1.y + t * dy;
        cv::Point point(int(x),int(y));
        
        pixels.emplace_back(std::round(x), std::round(y)); // 四舍五入到整数像素
    }
    //std::cout << "pixels: " << pixels;
     
    // 去重（避免重复点，如dx/dy为0时）
    std::sort(pixels.begin(), pixels.end(), [](const cv::Point& a, const cv::Point& b) {
    return a.x < b.x || (a.x == b.x && a.y < b.y);
    });
    //std::sort(pixels.begin(), pixels.end());
    auto last = std::unique(pixels.begin(), pixels.end());
    pixels.erase(last, pixels.end());
    
    return pixels;
}

    void zbuffer_occlusion(
    std::vector<G::idx_f_b_box> boxes,
    std::vector<G::front_2d_point>front_2d_point_lists,
    const float robot_y,
    const float robot_x,     // 机器人的x,y与这相反
    const double yaw,
    const int image_width,
    const int image_height
){
    // -----------------------------------------------------------1.画面静止时不更新
    const float eps = 0.001f;  
    const double eps_rot = 0.001;
    bool pos_unchanged = (std::abs(robot_x - global.last_robot_x) < eps && std::abs(robot_y - global.last_robot_y) < eps);
    bool yaw_unchanges = (std::abs(yaw - global.last_yaw) < eps_rot);
    if (pos_unchanged && yaw_unchanges) {
        ROS_INFO("not move or rotation");
        return;
    }
    
    // 更新上一次坐标为当前坐标
    global.last_robot_x = robot_x;
    global.last_robot_y = robot_y;
    global.last_yaw = yaw;

    if (image_width <= 0 || image_height <= 0) {
    std::cerr << "错误:图像尺寸无效(width=" << image_width << ", height=" << image_height << ")" << std::endl;
    return;  // 终止函数，避免创建空矩阵
    }
    global.up_2d_point_lists.clear();
    global.side_2d_points_lists.clear();

    //  -----------------------------------------------------------2：计算每个方块的相对坐标和各个面的平均深度
    std::vector<int> allow_idx_list;
    allow_idx_list.clear();
    for(int i = 0;i < front_2d_point_lists.size();i++){
        allow_idx_list.push_back(front_2d_point_lists[i].idx);
        //std::cout << "front_2d_point_lists[i].idx" << front_2d_point_lists[i].idx << std::endl;
    }
    for (auto& box : boxes) {
        box.r_x_min = std::abs(box.x_min - robot_x);
        box.r_x_max = std::abs(box.x_max - robot_x);
        box.r_y_min = std::abs(box.y_min - robot_y);
        box.r_y_max = std::abs(box.y_max - robot_y);
        box.average_x = (box.r_x_max + box.r_x_min ) / 2.0f;
        box.average_y = (box.r_y_max + box.r_y_min ) / 2.0f;
        box.average_z = (box.z_min + box.z_max) / 2.0f;
        box.front_depth = sqrt(box.r_x_min * box.r_x_min + box.average_y * box.average_y + box.average_z * box.average_z);
        if(std::find(allow_idx_list.begin(),allow_idx_list.end(),box.idx) != allow_idx_list.end()){
            //std::cout << "allow: " <<  box.idx << "  " <<  std::find(allow_idx_list.begin(),allow_idx_list.end(),box.idx) - allow_idx_list.begin() << std::endl;
            front_2d_point_lists[std::find(allow_idx_list.begin(),allow_idx_list.end(),box.idx) - allow_idx_list.begin()].front_depth = box.front_depth;
        }
        box.up_depth = sqrt(box.average_x * box.average_x + box.average_y * box.average_y + box.z_max * box.z_max);
        box.left_depth = sqrt(box.average_x * box.average_x + box.r_y_min * box.r_y_min + box.average_z * box.average_z);
        box.right_depth = sqrt(box.average_x * box.average_x + box.r_y_max * box.r_y_max + box.average_z * box.average_z);
        // std::cout << "=== 调试信息 - 方块 " << box.idx << " ===" << std::endl;
        // std::cout << "机器人位置: (" << robot_x << ", " << robot_y << ")" << std::endl;
        // std::cout << "方块坐标: x[" << box.x_min << ", " << box.x_max << "], "
        //         << "y[" << box.y_min << ", " << box.y_max << "], "
        //         << "z[" << box.z_min << ", " << box.z_max << "]" << std::endl;
        // std::cout << "相对机器人的坐标: r_x[" << box.r_x_min << ", " << box.r_x_max << "], "
        //         << "r_y[" << box.r_y_min << ", " << box.r_y_max << "]" << std::endl;
        // std::cout << "平均值坐标: (" << box.average_x << ", " << box.average_y << ", " << box.average_z << ")" << std::endl;
        // std::cout << "各方向深度:" << std::endl;
        // std::cout << "  前向深度: " << box.front_depth << std::endl;
        // std::cout << "  左侧深度: " << box.left_depth << std::endl;
        // std::cout << "  右侧深度: " << box.right_depth << std::endl;
        // std::cout << "  顶部深度: " << box.up_depth << std::endl;
        // std::cout << std::endl;
            std::vector<cv::Point2f>up_2d_points;
            std::vector<cv::Point3f>up_3d_points;
            up_3d_points.clear();
            up_2d_points.clear();
            up_3d_points.push_back(cv::Point3f(box.y_min,box.z_min,box.x_max));
            up_3d_points.push_back(cv::Point3f(box.y_max,box.z_min,box.x_max));
            up_3d_points.push_back(cv::Point3f(box.y_max,box.z_min,box.x_min));
            up_3d_points.push_back(cv::Point3f(box.y_min,box.z_min,box.x_min));
            {
                std::lock_guard<std::mutex> lock(global._mtx_tf); // 保护旋转和平移向量
                cv::projectPoints(up_3d_points, global._rvec, global._tvec, global._K, global._distCoeffs, up_2d_points);
            }
            global.up_2d_point_lists.push_back({box.idx, cv::Point2f(up_2d_points[0]),cv::Point2f(up_2d_points[1]),cv::Point2f(up_2d_points[2]),cv::Point2f(up_2d_points[3]),box.up_depth});
        // 2.判断方块自身侧面的遮挡情况
        if (box.left_depth < box.right_depth){
            box.right_occluders = true;    
            std::vector<cv::Point2f>side_2d_points;
            std::vector<cv::Point3f>side_3d_points;
            side_3d_points.clear();
            side_2d_points.clear();
            side_3d_points.push_back(cv::Point3f(box.y_min,box.z_min,box.x_max));
            side_3d_points.push_back(cv::Point3f(box.y_min,box.z_min,box.x_min));
            side_3d_points.push_back(cv::Point3f(box.y_min,box.z_max,box.x_min));
            side_3d_points.push_back(cv::Point3f(box.y_min,box.z_max,box.x_max));
            
            {
                std::lock_guard<std::mutex> lock(global._mtx_tf); // 保护旋转和平移向量
                cv::projectPoints(side_3d_points, global._rvec, global._tvec, global._K, global._distCoeffs, side_2d_points);
            }
            //std::cout << "left: side_2d_points: "<< side_2d_points << std::endl;  
            global.side_2d_points_lists.push_back({box.idx, cv::Point2f(side_2d_points[0]),cv::Point2f(side_2d_points[1]),cv::Point2f(side_2d_points[2]),cv::Point2f(side_2d_points[3]),box.left_depth});
        }else{box.left_occluders = true;
            std::vector<cv::Point2f>side_2d_points;
            std::vector<cv::Point3f>side_3d_points;
            side_3d_points.clear();
            side_2d_points.clear();
            side_3d_points.push_back(cv::Point3f(box.y_max,box.z_min,box.x_min));
            side_3d_points.push_back(cv::Point3f(box.y_max,box.z_min,box.x_max));
            side_3d_points.push_back(cv::Point3f(box.y_max,box.z_max,box.x_max));
            side_3d_points.push_back(cv::Point3f(box.y_max,box.z_max,box.x_min));
            
            {
                std::lock_guard<std::mutex> lock(global._mtx_tf); // 保护旋转和平移向量
                cv::projectPoints(side_3d_points, global._rvec, global._tvec, global._K, global._distCoeffs, side_2d_points);
            }
            //std::cout << "right: side_2d_points: "<< side_2d_points << std::endl;
            global.side_2d_points_lists.push_back({box.idx, cv::Point2f(side_2d_points[0]),cv::Point2f(side_2d_points[1]),cv::Point2f(side_2d_points[2]),cv::Point2f(side_2d_points[3]), box.right_depth});
        
        };      // 判断方块自身的遮挡情况
    }

    // 3.初始化Z-buffer（深度缓冲，存储每个像素的最小深度值）
    cv::Mat zbuffer = cv::Mat::ones(image_height, image_width, CV_32F) * FLT_MAX; // 创建一个​​全1矩阵， 初始化为无穷大

    // 4.排除掉空的情况
    std::vector<int> invalid_idxs = {1,2,3,4,5,6,7,8,9,10,11,12};
    for (const auto& box : front_2d_point_lists) {
        if (std::find(invalid_idxs.begin(),invalid_idxs.end(),box.idx) != invalid_idxs.end()){
            invalid_idxs.erase(std::find(invalid_idxs.begin(), invalid_idxs.end(), box.idx));
        }
    };
    // std::cout << "invalid_idxs: [";
    // std::copy(invalid_idxs.begin(), invalid_idxs.end(), 
    // std::ostream_iterator<int>(std::cout, ", "));
    // std::cout << "\b\b]" << std::endl;

    for (auto it = global.side_2d_points_lists.begin(); it != global.side_2d_points_lists.end(); ) {
        if (std::find(invalid_idxs.begin(), invalid_idxs.end(), it->idx) != invalid_idxs.end()) {
            it = global.side_2d_points_lists.erase(it); // erase返回下一个有效迭代器
        } else {
            ++it;
        }
    }
    for (auto it = global.up_2d_point_lists.begin(); it != global.up_2d_point_lists.end(); ) {
        if (std::find(invalid_idxs.begin(), invalid_idxs.end(), it->idx) != invalid_idxs.end()) {
            it = global.up_2d_point_lists.erase(it); // erase返回下一个有效迭代器
        } else {
            ++it;
        }
    }
    //std::cout << "size: " << front_2d_point_lists.size() << "  " << global.side_2d_points_lists.size() <<  "  " << boxes.size() << std::endl;

    // 5.填入zbuffer
    for (int i = 0; i < global.side_2d_points_lists.size(); i++) {
        auto& front_point = front_2d_point_lists[i];
        auto& side_point = global.side_2d_points_lists[i];
        auto& up_point = global.up_2d_point_lists[i];

        std::vector<cv::Point> front_contour = {
        cv::Point(cvRound(front_point.left_up.x), cvRound(front_point.left_up.y)),
        cv::Point(cvRound(front_point.right_up.x), cvRound(front_point.right_up.y)),
        cv::Point(cvRound(front_point.right_down.x), cvRound(front_point.right_down.y)),
        cv::Point(cvRound(front_point.left_down.x), cvRound(front_point.left_down.y))
        };
        std::vector<cv::Point> up_contour = {
            cv::Point(cvRound(up_point.left_up.x), cvRound(up_point.left_up.y)),
            cv::Point(cvRound(up_point.right_up.x), cvRound(up_point.right_up.y)),
            cv::Point(cvRound(up_point.right_down.x), cvRound(up_point.right_down.y)),
            cv::Point(cvRound(up_point.left_down.x), cvRound(up_point.left_down.y))
        };
        std::vector<cv::Point> side_contour = {
            cv::Point(cvRound(side_point.left_up.x), cvRound(side_point.left_up.y)),
            cv::Point(cvRound(side_point.right_up.x), cvRound(side_point.right_up.y)),
            cv::Point(cvRound(side_point.right_down.x), cvRound(side_point.right_down.y)),
            cv::Point(cvRound(side_point.left_down.x), cvRound(side_point.left_down.y))
        };
        cv::Mat tmp = cv::Mat::ones(image_height, image_width, CV_32F) * FLT_MAX; 
        cv::fillPoly(tmp, std::vector<std::vector<cv::Point>>{front_contour}, cv::Scalar(front_point.front_depth));
        cv::fillPoly(tmp, std::vector<std::vector<cv::Point>>{up_contour}, cv::Scalar(up_point.up_depth));
        cv::fillPoly(tmp, std::vector<std::vector<cv::Point>>{side_contour}, cv::Scalar(side_point.side_depth));

        std::vector<float>x_value = {front_point.left_up.x,front_point.right_up.x,front_point.right_down.x,front_point.left_down.x,
                                    up_point.left_up.x,up_point.right_up.x,up_point.right_down.x,up_point.left_down.x,
                                    side_point.left_up.x,side_point.right_up.x,side_point.right_down.x,side_point.left_down.x};
        std::vector<float>y_value = {front_point.left_up.y,front_point.right_up.y,front_point.right_down.y,front_point.left_down.y,
                                    up_point.left_up.y,up_point.right_up.y,up_point.right_down.y,up_point.left_down.y,
                                    side_point.left_up.y,side_point.right_up.y,side_point.right_down.y,side_point.left_down.y};

        auto x_max = std::max_element(x_value.begin(), x_value.end());
        auto x_min = std::min_element(x_value.begin(), x_value.end());
        auto y_max = std::max_element(y_value.begin(), y_value.end());
        auto y_min = std::min_element(y_value.begin(), y_value.end());

        for (int row = int(*y_min) - 5; row < int(*y_max) + 5; ++row) {       
            for (int col = int(*x_min) - 5; col < int(*x_max) + 5; ++col) {  
                // 读取当前像素值（zbuffer是单通道CV_32F，用at<float>访问）
                float pixel_value = zbuffer.at<float>(row, col);
                if(zbuffer.at<float>(row, col) > tmp.at<float>(row, col))
                {
                    zbuffer.at<float>(row, col) = tmp.at<float>(row, col);
                }
            }
        }
    }

    // 调试保存zbuffer
    //std::string save_path = "/home/h/RC2026/world_ws5/src/3dto2d/zbuffer_.png"; // 替换为你的保存路径
    // bool success = saveZBufferVisualization(zbuffer, save_path);
    // if (!success) {
    //     ROS_WARN("Z-Buffer可视化保存失败");
    // }
    // 调试发布zbuffer到ROS话题
    cv::Mat zbuffer_vis = saveZBufferVisualization(zbuffer);
    if (!zbuffer_vis.empty()) {
        // 转换OpenCV图像为ROS消息
        std_msgs::Header header;
        header.stamp = ros::Time::now(); // 设置时间戳
        header.frame_id = "camera_frame"; // 根据实际情况修改坐标系ID
        
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(
            header, 
            "mono8",  // 单通道8位图像编码
            zbuffer_vis
        ).toImageMsg();
        
        // 发布图像
        global.zbuffer_pub.publish(msg);
        //ROS_INFO("Z-Buffer可视化图像已发布到话题");
    } else {
        //ROS_WARN("Z-Buffer可视化图像为空，无法发布");
    }
    //printSide2DPoints(global.side_2d_points_lists);

}
// -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// 颜色枚举
enum class TargetColor {
    Green,
    Yellow,
    other
};

// 定义目标颜色的BGR阈值范围（OpenCV默认BGR格式）
const std::unordered_map<TargetColor, std::tuple<int, int, int, int, int, int>> COLOR_THRESHOLDS = {
    {TargetColor::Green,  {0, 15, 100, 255, 0, 10}},   // B:100-255, G:100-255, R:0-100
    {TargetColor::Yellow,{-1, 10, 60, 200, 60, 200}},    // B:0-100, G:150-255, R:150-255
    {TargetColor::other, {150,255,150,255,150,255}}
};

// 判断像素是否属于目标颜色
bool isPixelTargetColor(const cv::Vec3b& pixel, TargetColor targetColor) {
    auto it = COLOR_THRESHOLDS.find(targetColor);
    if (it == COLOR_THRESHOLDS.end()) return false;
    
    auto [b_min, b_max, g_min, g_max, r_min, r_max] = it->second;
    return (pixel[0] >= b_min && pixel[0] <= b_max) &&  // B通道
           (pixel[1] >= g_min && pixel[1] <= g_max) &&  // G通道
           (pixel[2] >= r_min && pixel[2] <= r_max);    // R通道
}

std::vector<G::front_2d_point> filterEmptyBoxes(
    const cv::Mat& image,
    const std::vector<G::front_2d_point>& candidatePoints,  // 输入改为front_2d_point列表
    float emptyThreshold = 0.08f,   // 第一次检查的阈值
    float center_rate = 0.3f,       // 指定中间区域的范围
    float centerThreshoud = 0.15f,  // 对中间区域进行二次检查的阈值
    float reduce_iamge_Threshold = 0.9f  // 图像ROI向内缩进比例
) {
    std::vector<G::front_2d_point> nonEmptyPoints;  // 返回值改为front_2d_point列表
    if (image.empty() || candidatePoints.empty()) {
        return nonEmptyPoints;
    }

    // 1. 将图像转为HSV空间（更鲁棒的颜色分割）
    cv::Mat hsvImage;
    cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);

    // 2. 定义目标颜色的HSV范围（OpenCV的HSV范围：H:0-180, S:0-255, V:0-255）
    std::vector<std::pair<cv::Scalar, cv::Scalar>> colorRanges = {
        {cv::Scalar(115, 0, 0), cv::Scalar(160, 255, 255)},  // 示例颜色范围
    };

    for (const auto& point : candidatePoints) {  // 遍历每个front_2d_point
        // 3. 从四个角点计算边界框（x_min, x_max, y_min, y_max）
        float x_min = std::min({point.left_up.x, point.right_up.x, point.right_down.x, point.left_down.x});
        float x_max = std::max({point.left_up.x, point.right_up.x, point.right_down.x, point.left_down.x});
        float y_min = std::min({point.left_up.y, point.right_up.y, point.right_down.y, point.left_down.y});
        float y_max = std::max({point.left_up.y, point.right_up.y, point.right_down.y, point.left_down.y});

        // 边界保护（避免超出图像范围）
        x_min = std::max(0.0f, x_min);
        x_max = std::min(static_cast<float>(image.cols - 1), x_max);
        y_min = std::max(0.0f, y_min);
        y_max = std::min(static_cast<float>(image.rows - 1), y_max);

        // 计算原始ROI宽高（确保为正）
        float roi_width = x_max - x_min;
        float roi_height = y_max - y_min;
        if (roi_width <= 0 || roi_height <= 0) {
            continue;  // 无效框跳过
        }

        // 4. 计算缩进后的ROI（向内缩进10%）
        int roi_x = static_cast<int>(x_min + roi_width * (1 - reduce_iamge_Threshold) / 2);
        int roi_y = static_cast<int>(y_min + roi_height * (1 - reduce_iamge_Threshold) / 2);
        int roi_w = static_cast<int>(roi_width * reduce_iamge_Threshold);
        int roi_h = static_cast<int>(roi_height * reduce_iamge_Threshold);

        // 5. 计算中心区域ROI
        int center_x = static_cast<int>(x_min + roi_width * (1 - center_rate) / 2);
        int center_y = static_cast<int>(y_min + roi_height * (1 - center_rate) / 2);
        int center_w = static_cast<int>(roi_width * center_rate);
        int center_h = static_cast<int>(roi_height * center_rate);

        // 确保ROI在图像范围内
        cv::Rect roi_rect(roi_x, roi_y, roi_w, roi_h);
        roi_rect &= cv::Rect(0, 0, image.cols, image.rows);  // 与图像边界取交集
        cv::Rect center_rect(center_x, center_y, center_w, center_h);
        center_rect &= cv::Rect(0, 0, image.cols, image.rows);

        // 提取ROI（为空则跳过）
        cv::Mat roi = hsvImage(roi_rect);
        cv::Mat center_roi = hsvImage(center_rect);
        if (roi.empty() || center_roi.empty()) {
            continue;
        }

        // 6. 生成目标颜色的合并掩码
        cv::Mat colorMask = cv::Mat::zeros(roi.size(), CV_8UC1);
        for (const auto& range : colorRanges) {
            cv::Mat rangeMask;
            cv::inRange(roi, range.first, range.second, rangeMask);
            colorMask |= rangeMask;
        }

        cv::Mat center_colorMask = cv::Mat::zeros(center_roi.size(), CV_8UC1);
        for (const auto& range : colorRanges) {
            cv::Mat center_rangeMask;
            cv::inRange(center_roi, range.first, range.second, center_rangeMask);
            center_colorMask |= center_rangeMask;
        }

        // 7. 统计目标颜色像素比例
        int totalPixels = roi.rows * roi.cols;
        int targetPixels = cv::countNonZero(colorMask);
        float targetRatio = totalPixels > 0 ? static_cast<float>(targetPixels) / totalPixels : 0.0f;

        int centerPixels = center_roi.rows * center_roi.cols;
        int centerTargetPixels = cv::countNonZero(center_colorMask);
        float centerRatio = centerPixels > 0 ? static_cast<float>(centerTargetPixels) / centerPixels : 0.0f;

        // 8. 满足阈值条件则保留该front_2d_point
        if (targetRatio > emptyThreshold && centerRatio > centerThreshoud) {
            nonEmptyPoints.push_back(point);
        }
    }

    return nonEmptyPoints;
}

cv::Mat drawRegions(
    const cv::Mat& input_image,
    const cv::Mat& zbuffer,
    const std::vector<G::front_2d_point>& front_points,
    const std::vector<G::side_2d_point>& side_points,
    const std::vector<G::up_2d_point>& up_points,
    float depth_threshold = 0.01f  // 深度值比较的容差阈值
) {
    if (input_image.empty() || zbuffer.empty()) {
        ROS_WARN("输入图像或zbuffer为空，无法绘制");
        return input_image.clone();
    }

    cv::Mat output_image = input_image.clone();
    if (output_image.channels() == 1) {
        cv::cvtColor(output_image, output_image, cv::COLOR_GRAY2BGR);  // 确保为彩色图像
    }

    // 定义不同面的绘制颜色
    const cv::Scalar front_color(0, 0, 255);    // 红色-正面
    const cv::Scalar side_color(0, 255, 0);     // 绿色-侧面
    const cv::Scalar up_color(255, 0, 0);       // 蓝色-顶面

    // 绘制正面区域
    for (const auto& fp : front_points) {
        // 1. 创建深度掩码：找到zbuffer中与当前正面深度接近的像素
        cv::Mat mask = cv::Mat::zeros(zbuffer.size(), CV_8UC1);
        for (int row = 0; row < zbuffer.rows; ++row) {
            for (int col = 0; col < zbuffer.cols; ++col) {
                float z_val = zbuffer.at<float>(row, col);
                if (z_val != FLT_MAX && std::abs(z_val - fp.front_depth) < depth_threshold) {
                    mask.at<uchar>(row, col) = 255;  // 标记匹配的像素
                }
            }
        }

        // 2. 提取轮廓
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // 3. 绘制轮廓（取最大面积的轮廓作为目标区域）
        if (!contours.empty()) {
            // 找到最大轮廓
            auto max_contour = *std::max_element(contours.begin(), contours.end(),
                [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                    return cv::contourArea(a) < cv::contourArea(b);
                });

            // 绘制多边形轮廓
            cv::polylines(output_image, max_contour, true, front_color, 2);
            // 在区域中心标注深度值
            cv::Moments mom = cv::moments(max_contour);
            cv::Point center(mom.m10 / mom.m00, mom.m01 / mom.m00);
            cv::putText(output_image, std::to_string(fp.idx), center, 
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, front_color, 2);
        }
    }

    // 绘制侧面区域（逻辑同正面）
    for (const auto& sp : side_points) {
        cv::Mat mask = cv::Mat::zeros(zbuffer.size(), CV_8UC1);
        for (int row = 0; row < zbuffer.rows; ++row) {
            for (int col = 0; col < zbuffer.cols; ++col) {
                float z_val = zbuffer.at<float>(row, col);
                if (z_val != FLT_MAX && std::abs(z_val - sp.side_depth) < depth_threshold) {
                    mask.at<uchar>(row, col) = 255;
                }
            }
        }

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        if (!contours.empty()) {
            auto max_contour = *std::max_element(contours.begin(), contours.end(),
                [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                    return cv::contourArea(a) < cv::contourArea(b);
                });

            cv::polylines(output_image, max_contour, true, side_color, 2);
            cv::Moments mom = cv::moments(max_contour);
            cv::Point center(mom.m10 / mom.m00, mom.m01 / mom.m00);
            cv::putText(output_image, std::to_string(sp.idx), center, 
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, side_color, 2);
        }
    }

    // 绘制顶面区域（逻辑同正面）
    for (const auto& up : up_points) {
        cv::Mat mask = cv::Mat::zeros(zbuffer.size(), CV_8UC1);
        for (int row = 0; row < zbuffer.rows; ++row) {
            for (int col = 0; col < zbuffer.cols; ++col) {
                float z_val = zbuffer.at<float>(row, col);
                if (z_val != FLT_MAX && std::abs(z_val - up.up_depth) < depth_threshold) {
                    mask.at<uchar>(row, col) = 255;
                }
            }
        }

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        if (!contours.empty()) {
            auto max_contour = *std::max_element(contours.begin(), contours.end(),
                [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                    return cv::contourArea(a) < cv::contourArea(b);
                });

            cv::polylines(output_image, max_contour, true, up_color, 2);
            cv::Moments mom = cv::moments(max_contour);
            cv::Point center(mom.m10 / mom.m00, mom.m01 / mom.m00);
            cv::putText(output_image, std::to_string(up.idx), center, 
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, up_color, 2);
        }
    }

    return output_image;
}

Eigen::Matrix3f createRotationMatrix(float rx, float ry, float rz) {
    // 转换为弧度
    // rx = rx * M_PI / 180.0f; // Roll (绕X轴)
    // ry = ry * M_PI / 180.0f; // Pitch (绕Y轴)
    // rz = rz * M_PI / 180.0f; // Yaw (绕Z轴)
    // 创建绕各轴的旋转矩阵
    Eigen::Matrix3f R_x;
    R_x << 1, 0, 0,
           0, cos(rx), -sin(rx),
           0, sin(rx), cos(rx);
    Eigen::Matrix3f R_y;
    R_y << cos(ry), 0, sin(ry),
           0, 1, 0,
           -sin(ry), 0, cos(ry);
    Eigen::Matrix3f R_z;
    R_z << cos(rz), -sin(rz), 0,
           sin(rz), cos(rz), 0,
           0, 0, 1;
    // 组合旋转矩阵 (Z-Y-X顺序: R = R_z * R_y * R_x)
    return R_z * R_y * R_x;
}

Eigen::Vector3f createTranslationVector(float tx, float ty, float tz) {
    Eigen::Vector3f translation(tx, ty, tz);
    return translation;
}

// 分离和组合现有旋转矩阵与平移向量
Eigen::Matrix4f combineRotationAndTranslation(const Eigen::Matrix3f& rotation, const Eigen::Vector3f& translation) {
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block<3, 3>(0, 0) = rotation;
    transform.block<3, 1>(0, 3) = translation;
    return transform;
}

// 回调函数：处理接收到的 TF 消息
void tfCallback(const tf2_msgs::TFMessage::ConstPtr& msg)
{
    for (const auto& transform : msg->transforms)
    {
        if(transform.header.frame_id != "odom")
        {
            std::cout<< transform.header.frame_id<<std::endl;
            continue;
        }
        // 获取四元数
        geometry_msgs::Quaternion quat = transform.transform.rotation;

        // 使用 tf2 库将四元数转换为欧拉角
        tf2::Quaternion tf_quat;
        tf_quat.setX(quat.x);
        tf_quat.setY(quat.y);
        tf_quat.setZ(quat.z);
        tf_quat.setW(quat.w);

        // 转换为旋转矩阵，然后提取欧拉角
        tf2::Matrix3x3 mat(tf_quat);
        double roll, pitch, yaw;
        mat.getRPY(roll, pitch, yaw);  // 得到的是弧度值

        cv::Mat R = (cv::Mat_<double>(3,3) <<
        cos(yaw), 0, sin(yaw),
        0, 1, 0,
        -sin(yaw), 0, cos(yaw));
        cv::Mat R_inv;
        cv::Mat C_world = (cv::Mat_<double>(3,1) << -transform.transform.translation.y, 0.0, transform.transform.translation.x);
        cv::invert(R, R_inv);
        {
            std::lock_guard<std::mutex> lock(global._mtx_tf);
            cv::Rodrigues(R, global._rvec);
            global._tvec = - (R * C_world);
        }

        std::cout<< "yaw: "<< yaw <<std::endl;
        // double yaw_deg = yaw * 180.0 / M_PI;
        // float _x  = transform.transform.translation.x - 0.28*cos(yaw) + 0.28;
        // float _y = transform.transform.translation.y - 0.28*sin(yaw);

    }
}

void odomCallback(const nav_msgs::Odometry::ConstPtr& msg)
{
    // ROS_INFO_STREAM("odom");
    // ROS_INFO_STREAM("x=" << msg->pose.pose.position.x
    //                << "y=" << msg->pose.pose.position.y
    //                << "z=" << msg->pose.pose.position.z);
    // ROS_INFO_STREAM("x=" << msg->pose.pose.orientation.x
    //                << "y=" << msg->pose.pose.orientation.y
    //                << "z=" << msg->pose.pose.orientation.z
    //                << "w=" << msg->pose.pose.orientation.w);
    global.x = msg->pose.pose.position.x;
    global.y = msg->pose.pose.position.y;
    global.z = msg->pose.pose.position.z;    
    std::cout << "x: " << global.x << " y: " << global.y << std::endl;
    // 获取四元数
    geometry_msgs::Quaternion quat = msg->pose.pose.orientation;

    // 使用 tf2 库将四元数转换为欧拉角
    tf2::Quaternion tf_quat;
    tf_quat.setX(quat.x);
    tf_quat.setY(quat.y);
    tf_quat.setZ(quat.z);
    tf_quat.setW(quat.w);

    // 转换为旋转矩阵，然后提取欧拉角
    tf2::Matrix3x3 mat(tf_quat);
    double roll, pitch;
    mat.getRPY(roll, pitch, global.yaw);  // 得到的是弧度值


    cv::Mat R = (cv::Mat_<double>(3,3) <<
        cos(global.yaw-M_PI/2), 0, sin(global.yaw-M_PI/2),
        0, 1, 0,
        -sin(global.yaw-M_PI/2), 0, cos(global.yaw-M_PI/2));
    cv::Mat R_inv;
    cv::Mat C_world = (cv::Mat_<double>(3,1) << msg->pose.pose.position.x, 0.0, msg->pose.pose.position.y);
    cv::invert(R, R_inv);
    {
        std::lock_guard<std::mutex> lock(global._mtx_tf);
        cv::Rodrigues(R, global._rvec);
        global._tvec = - (R * C_world);
    }

    std::cout<< "yaw: "<< global.yaw <<std::endl;
}

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    try
    {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        cv::Mat image_ =  cv_ptr->image;
        global.imagePoints;
        global.side_imagePoints;

        // 投影3D点到2D
        {
            std::lock_guard<std::mutex> lock(global._mtx_tf);
            cv::projectPoints(global._c_objectPoints, global._rvec, global._tvec, global._K, global._distCoeffs, global.imagePoints);
        }
        // {
        //     std::lock_guard<std::mutex> lock(global._mtx_tf);
        //     cv::projectPoints(global._c_side_objectPoints, global._rvec, global._tvec, global._K, global._distCoeffs, global.side_imagePoints);
        // }        

        const float IOU_THRESHOLD = 0.02f; // IOU阈值
        global.idx_box_lists.clear();
        global.front_2d_point_lists.clear();
        global.side_2d_points_lists.clear(); 
        std::vector<int> exclude_idxs = {};
        // 1. 收集画面内的框
        const int image_width = image_.cols;
        const int iamge_height = image_.rows;
        for (int i = 0; i < global.imagePoints.size(); i += 4) {                       
            cv::Point3f world_point = global._c_objectPoints[i];
            float dist = std::sqrt(
                world_point.x * world_point.x +
                world_point.y * world_point.y +
                world_point.z * world_point.z
            );


            // 检查框的对角点是否在画面内
            bool inFrame = 
                (global.imagePoints[i].x > -60 && global.imagePoints[i].x < image_width + 60 && 
                 global.imagePoints[i].y > -60 && global.imagePoints[i].y < iamge_height + 60) &&
                (global.imagePoints[i+2].x > -60 && global.imagePoints[i+2].x < image_width + 60 && 
                 global.imagePoints[i+2].y > -60 && global.imagePoints[i+2].y < iamge_height + 60);

            if (inFrame) {
                if (std::find(exclude_idxs.begin(), exclude_idxs.end(), i/4+1) != exclude_idxs.end()) {
                    continue; // 跳过排除的idx
                    }
            
                if (global.imagePoints[i].x < 0){global.imagePoints[i].x = 0;};
                if (global.imagePoints[i].x > image_width){global.imagePoints[i].x = image_width;}; 
                if (global.imagePoints[i].y < 0){global.imagePoints[i].y = 0;};
                if (global.imagePoints[i].y > iamge_height){global.imagePoints[i].y = iamge_height;}; 
                if (global.imagePoints[i+2].x < 0){global.imagePoints[i+2].x = 0;};
                if (global.imagePoints[i+2].x >image_width){global.imagePoints[i+2].x = image_width;}; 
                if (global.imagePoints[i+2].y < 0){global.imagePoints[i+2].y = 0;};
                if (global.imagePoints[i+2].y > iamge_height){global.imagePoints[i+2].y = iamge_height;}; 

                global.idx_box_lists.push_back({
                    i/4 + 1,
                    global.imagePoints[i].x,
                    global.imagePoints[i].y,
                    global.imagePoints[i+2].x,
                    global.imagePoints[i+2].y,
                    dist
                });
                global.front_2d_point_lists.push_back({
                    i/4 + 1,
                    cv::Point2f(global.imagePoints[i].x,global.imagePoints[i].y),
                    cv::Point2f(global.imagePoints[i+1].x,global.imagePoints[i+1].y),
                    cv::Point2f(global.imagePoints[i+2].x,global.imagePoints[i+2].y),
                    cv::Point2f(global.imagePoints[i+3].x,global.imagePoints[i+3].y)
                });
            }
        }     

        // 2. 筛选正面的框
        global.front_2d_point_lists = filterEmptyBoxes(image_,global.front_2d_point_lists);
        zbuffer_occlusion(global.idx_f_b_box_list, global.front_2d_point_lists, global.x,global.y, global.yaw ,image_width, iamge_height);
        
        // 3. 绘制筛选后的框
        for (const auto& side_point : global.side_2d_points_lists) {
            // 检查点是否在图像范围内
            bool points_valid = true;
            std::vector<cv::Point2f> points = {
                side_point.left_up,
                side_point.right_up,
                side_point.right_down,
                side_point.left_down
            };
            
            for (const auto& point : points) {
                if (point.x < 0 || point.x >= image_.cols || point.y < 0 || point.y >= image_.rows) {
                    points_valid = false;
                    break;
                }
            }
            
            if (!points_valid) {
                continue; // 跳过无效的点
            }
            
            // 设置绘制参数
            cv::Scalar line_color(0, 0, 255);  // 绿色线条
            cv::Scalar point_color(255, 0, 0);  // 蓝色点
            int line_thickness = 3;
            int point_radius = 5;
            
            // 绘制四个点之间的连线（形成四边形）
            cv::line(image_, points[0], points[1], line_color, line_thickness);  // 左上到右上
            cv::line(image_, points[1], points[2], line_color, line_thickness);  // 右上到右下
            cv::line(image_, points[2], points[3], line_color, line_thickness);  // 右下到左下
            cv::line(image_, points[3], points[0], line_color, line_thickness);  // 左下到左上
            
            // 绘制四个角点
            for (const auto& point : points) {
                cv::circle(image_, point, point_radius, point_color, -1); // 实心圆
            }
        }

        for (const auto& side_point : global.front_2d_point_lists) {
            // 检查点是否在图像范围内
            bool points_valid = true;
            std::vector<cv::Point2f> points = {
                side_point.left_up,
                side_point.right_up,
                side_point.right_down,
                side_point.left_down
            };
            
            for (const auto& point : points) {
                if (point.x < 0 || point.x >= image_.cols || point.y < 0 || point.y >= image_.rows) {
                    points_valid = false;
                    break;
                }
            }
            
            if (!points_valid) {
                continue; // 跳过无效的点
            }
            
            // 设置绘制参数
            cv::Scalar line_color(0, 0, 255);  // 绿色线条
            cv::Scalar point_color(255, 0, 0);  // 蓝色点
            int line_thickness = 3;
            int point_radius = 5;
            
            // 绘制四个点之间的连线（形成四边形）
            cv::line(image_, points[0], points[1], line_color, line_thickness);  // 左上到右上
            cv::line(image_, points[1], points[2], line_color, line_thickness);  // 右上到右下
            cv::line(image_, points[2], points[3], line_color, line_thickness);  // 右下到左下
            cv::line(image_, points[3], points[0], line_color, line_thickness);  // 左下到左上
            
            // 绘制四个角点
            for (const auto& point : points) {
                cv::circle(image_, point, point_radius, point_color, -1); // 实心圆
            }
        }
        // 更新全局图像
        {
            std::lock_guard<std::mutex> lock(global._mtx_image);
            image_.copyTo(global._image);
        }
        
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        
    }
}

void worker_task1(ros::NodeHandle nh)
{
    ros::Rate sl(50);
    //ros::Subscriber tf_sub = nh.subscribe("/tf", 2, tfCallback);
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
        /* code */
        
        ros::spinOnce();
        sl.sleep();
    }
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "t2dto3d_zbuffer_node");
    ros::NodeHandle nh;
    std::vector<std::thread> workers;
    workers.emplace_back(worker_task1, nh);
    workers.emplace_back(worker_task2, nh);
    image_transport::ImageTransport it(nh);
    image_transport::Publisher pub = it.advertise("pub_image_topic", 2);
    global.zbuffer_pub = it.advertise("/zbuffer_visualization", 30);

    ros::Rate rate(10);
    while(ros::ok())
    {
        sensor_msgs::ImagePtr msg;
        {
            std::lock_guard<std::mutex> lock(global._mtx_image);
            msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", global._image).toImageMsg();
        }
        pub.publish(msg);
        std::cout << "publish success" << std::endl;
        rate.sleep();
    }
    return 0;
}