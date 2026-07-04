#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PointStamped.h>
#include <thread>
#include "PID.cpp"
#include <geometry_msgs/TransformStamped.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <std_msgs/Bool.h>
#include <cmath>
#include <tf2_msgs/TFMessage.h>
#include <mutex>
#include <chrono> 
#include <vector> 


struct 
{
    float _lidar_x;      
    float _lidar_y;     
    float _lidar_yaw;  
    std::vector<std::pair<float, float>> _target_list;  // 目标点列表
    int _current_target_index = 0;  
    float _target_x = 0;  
    float _target_y = 0;  
    float _target_yaw = 0; 

    bool _use_visual = true; 

    float _visual_x = 5.2; // visual点坐标，机器人会一直看这个点
    float _visual_y = 1.6; 

    bool _completed = false;
    int _flag = 0;         // 全局控制标志位
    
    // 互斥锁
    std::mutex mtx_lidar;  
    std::mutex mtx_target;
    std::mutex mtx_visual;
    std::mutex mtx_flag;
    std::mutex mtx_target_list;
}global;

// 调试计数器
int debug_counter = 0;
const int DEBUG_PRINT_INTERVAL = 100; 

// 到达阈值
const float ARRIVAL_THRESHOLD = 0.05; 



// 目标点
void initializeTargetList() {
    global._target_list.push_back(std::make_pair(0.0, 0.0));
    global._target_list.push_back(std::make_pair(0.95, 1.0));
    global._target_list.push_back(std::make_pair(0.0, 0.0));
    global._target_list.push_back(std::make_pair(0.0, 1.2));
    global._target_list.push_back(std::make_pair(0.95, 1.6)); 
    global._target_list.push_back(std::make_pair(0.0, 1.2));
    global._target_list.push_back(std::make_pair(0.9, -0.65));
    global._target_list.push_back(std::make_pair(0.0, -0.65));    
    global._target_list.push_back(std::make_pair(1.7, 1.6));  
    global._target_list.push_back(std::make_pair(1.7, 3.0));  
    global._target_list.push_back(std::make_pair(1.4, 3.0));  
    global._target_list.push_back(std::make_pair(1.7, -0.4));  
    global._target_list.push_back(std::make_pair(2.1, 1.7));  
    // 设置第一个目标点
    if (!global._target_list.empty()) {
        global._target_x = global._target_list[0].first;
        global._target_y = global._target_list[0].second;
    }
    
    // ROS_INFO("Target list initialized with %zu points", global._target_list.size());
    // for (size_t i = 0; i < global._target_list.size(); i++) {
    //     ROS_INFO("  Target %zu: (%.2f, %.2f)", i, global._target_list[i].first, global._target_list[i].second);
    // }
}

// 检查是否到达当前目标点
bool checkArrival(float current_x, float current_y, float target_x, float target_y) {
    float dx = target_x - current_x;
    float dy = target_y - current_y;
    float distance = sqrt(dx*dx + dy*dy);
    return distance < ARRIVAL_THRESHOLD;
}

// 切换到下一个目标点
bool switchToNextTarget() {
    std::lock_guard<std::mutex> lock(global.mtx_target_list);
    
    if (global._target_list.empty()) {
        ROS_WARN("Target list is empty!");
        return false;
    }
    
    global._current_target_index++;
    
    // 检查是否已到达最后一个目标点
    if (global._current_target_index >= global._target_list.size()) {
        ROS_INFO("Reached all target points! Stopping.");
        global._flag = 1; // 设置停止标志
        global._completed = true;
        return false;
    }
    

    // 更新当前目标点
    global._target_x = global._target_list[global._current_target_index].first;
    global._target_y = global._target_list[global._current_target_index].second;
    
    // ROS_INFO("Switched to target %d: (%.2f, %.2f)", 
    //          global._current_target_index, 
    //          global._target_x, 
    //          global._target_y);
    
    return true;
}

// Visual点回调函数
void visualCallback(const geometry_msgs::PointStamped::ConstPtr& msg)
{
    std::lock_guard<std::mutex> lock(global.mtx_visual);
    global._visual_x = msg->point.x;
    global._visual_y = msg->point.y;
    global._use_visual = true;
    
    static int visual_counter = 0;
    // visual_counter++;
    // if (visual_counter % 20 == 0) {  // 每20次visual回调打印一次
    //     ROS_INFO("Visual Callback - Received visual point: x=%.3f, y=%.3f", 
    //              global._visual_x, global._visual_y);
    // }
}

// TF回调函数：从/tf话题获取机器人底盘的实际位姿
void tfCallback(const tf2_msgs::TFMessage::ConstPtr& msg)
{
    static int tf_counter = 0;
    bool found_body = false;
    
    for (const auto& transform : msg->transforms)
    {
        if(transform.child_frame_id != "base_footprint") // 只处理机器人本体的TF变换
        {
            continue;
        }
        found_body = true;
        
        // 获取位置
        float x = transform.transform.translation.x;
        float y = transform.transform.translation.y;
        
        // 获取四元数姿态并转换为欧拉角，得到航向角yaw（弧度）
        geometry_msgs::Quaternion quat = transform.transform.rotation;
        tf2::Quaternion tf_quat(quat.x, quat.y, quat.z, quat.w);
        tf2::Matrix3x3 mat(tf_quat);
        double roll, pitch, yaw;
        mat.getRPY(roll, pitch, yaw);
        
        // 将弧度转换为度
        double yaw_deg = yaw * 180.0 / M_PI;
        
        // 坐标变换：从lidar坐标系转换到机器人中心
        float _x  = x - 0.28*cos(yaw) + 0.28;
        float _y = y - 0.28*sin(yaw);

        // 调试输出：定期打印接收到的TF数据
        // tf_counter++;
        // if (tf_counter % 50 == 0) {  // 每50次TF回调打印一次
        //     ROS_INFO("TF Callback - Frame: %s -> %s", transform.header.frame_id.c_str(), transform.child_frame_id.c_str());
        //     ROS_INFO("TF Raw - x: %.3f, y: %.3f, yaw(rad): %.3f, yaw(deg): %.3f", x, y, yaw, yaw_deg);
        //     ROS_INFO("TF Adjusted - _x: %.3f, _y: %.3f", _x, _y);
        // }

        // 使用互斥锁安全地更新全局状态
        std::lock_guard<std::mutex> lock(global.mtx_lidar);
        global._lidar_x = _x;
        global._lidar_y = _y;
        global._lidar_yaw = yaw_deg;
        
        break; // 找到body后跳出循环
    }
    
    if (!found_body && tf_counter % 100 == 0) {
        ROS_WARN("TF Callback: No 'base_footprint' frame found in TF message!");
        for (const auto& transform : msg->transforms) {
            ROS_WARN("Available TF: %s -> %s", transform.header.frame_id.c_str(), transform.child_frame_id.c_str());
        }
    }
}

// 工作线程1：持续订阅TF话题，更新机器人位姿
void worker_task1(ros::NodeHandle nh)
{
    ROS_INFO("Worker Task 1 started - TF subscription");
    ros::Rate sl(1000);
    ros::Subscriber tf_sub = nh.subscribe("/tf", 10, tfCallback);
    ros::Subscriber visual_sub = nh.subscribe("/visual_point", 10, visualCallback);
    while (ros::ok())
    {   
        ros::spinOnce();
        sl.sleep();
    }  
}

// 计算机器人到目标点的航向角
float calculateYawToPoint(float robot_x, float robot_y, float target_x, float target_y)
{
    float dx = target_x - robot_x;
    float dy = target_y - robot_y;
    
    // 使用atan2计算角度，返回角度值（度）
    float yaw_rad = atan2(dy, dx);
    float yaw_deg = yaw_rad * 180.0 / M_PI;
    
    // 将角度归一化到[-180, 180]范围
    while (yaw_deg > 180.0) yaw_deg -= 360.0;
    while (yaw_deg < -180.0) yaw_deg += 360.0;
    
    return yaw_deg;
}

int main(int argc, char** argv)
{
    // 初始化ROS节点
    ros::init(argc, argv, "basemove2_node");
    ros::NodeHandle nh;
    
    ROS_INFO("basemove2_node started");
    
    // 初始化目标点列表
    initializeTargetList();
    

    std::vector<std::thread> workers;
    workers.emplace_back(worker_task1, nh);

    
    // 创建Publisher，发布控制指令到/cmd_vel话题
    ros::Publisher cmd_vel_pub = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 10);
    ROS_INFO("Publisher created for /cmd_vel");
    
    ros::Rate loop_rate(11); // 控制频率：11Hz

    
    // 初始化PID控制器（X、Y位置控制和航向角控制）
    PIDcontroler pid_x;
    PIDcontroler pid_y;
    PIDcontroler pid_yaw(0.31, 0.3, 0.02, 0.08, 0.0, 0.0);

    
    int flag = 0; // 本地标志位副本
    auto start = std::chrono::steady_clock::now(); // 用于PID计算的时间戳
    
    // 添加目标点切换计时器，防止频繁切换
    auto last_target_switch_time = std::chrono::steady_clock::now();
    const double MIN_SWITCH_INTERVAL = 2.0; // 最小切换间隔2秒
    
    // 主控制循环
    while (ros::ok())
    {
        geometry_msgs::Twist vel_msg; // 速度控制消息
        
        // 获取当前控制标志
        {
            std::lock_guard<std::mutex> lock(global.mtx_lidar);
            flag = global._flag;
        }
        
        // 定期打印调试信息
        debug_counter++;
        if (debug_counter % DEBUG_PRINT_INTERVAL == 0) {
            // ROS_INFO("=== Debug Info (Cycle: %d) ===", debug_counter);
            // ROS_INFO("Control flag: %d (0=move, 1=stop)", flag);
            // ROS_INFO("Current target index: %d / %zu", 
            //          global._current_target_index, 
            //          global._target_list.size() - 1);
            
            // 读取当前状态
            float current_x, current_y, current_yaw;
            float target_x, target_y, target_yaw;
            float visual_x, visual_y;
            bool use_visual;
            {
                std::lock_guard<std::mutex> lock(global.mtx_lidar);
                current_x = global._lidar_x;
                current_y = global._lidar_y;
                current_yaw = global._lidar_yaw;
            }
            {
                std::lock_guard<std::mutex> lock(global.mtx_target);
                target_x = global._target_x;
                target_y = global._target_y;
                target_yaw = global._target_yaw;
            }
            {
                std::lock_guard<std::mutex> lock(global.mtx_visual);
                visual_x = global._visual_x;
                visual_y = global._visual_y;
                use_visual = global._use_visual;
            }
            
            // ROS_INFO("Current Pose: x=%.3f, y=%.3f, yaw=%.3f°", current_x, current_y, current_yaw);
            // ROS_INFO("Target Pose:  x=%.3f, y=%.3f, yaw=%.3f°", target_x, target_y, target_yaw);
            // ROS_INFO("Visual Point: x=%.3f, y=%.3f, use_visual=%d", visual_x, visual_y, use_visual);
            
            float error_x = target_x - current_x;
            float error_y = target_y - current_y;
            float error_yaw = target_yaw - current_yaw;
            float distance_to_target = sqrt(error_x*error_x + error_y*error_y);
            // ROS_INFO("Errors: dx=%.3f, dy=%.3f, dyaw=%.3f°, distance=%.3f", 
            //          error_x, error_y, error_yaw, distance_to_target);
            
            // 检查是否到达当前目标点
            bool arrived = checkArrival(current_x, current_y, target_x, target_y);
            if (arrived) {
                ROS_INFO("ARRIVED at target %d! Distance: %.3f < %.3f (threshold)", 
                         global._current_target_index, distance_to_target, ARRIVAL_THRESHOLD);
            }
        }
        
        // 检查是否到达当前目标点，并切换到下一个目标点
        auto current_time = std::chrono::steady_clock::now();
        auto time_since_last_switch = std::chrono::duration<double>(current_time - last_target_switch_time).count();
        
        if (!global._completed && time_since_last_switch > MIN_SWITCH_INTERVAL) {
            float current_x, current_y;
            float target_x, target_y;
            {
                std::lock_guard<std::mutex> lock(global.mtx_lidar);
                current_x = global._lidar_x;
                current_y = global._lidar_y;
            }
            {
                std::lock_guard<std::mutex> lock(global.mtx_target);
                target_x = global._target_x;
                target_y = global._target_y;
            }
            
            if (checkArrival(current_x, current_y, target_x, target_y)) {
                ROS_INFO("Switching to next target point...");
                if (switchToNextTarget()) {
                    last_target_switch_time = current_time;
                }
            }
        }
        
        // 根据标志位决定是否进行移动控制
        if(flag == 0) // 允许运动
        {
            auto end = std::chrono::steady_clock::now();
            auto duration = end - start;
            double seconds_double = static_cast<double>(duration.count()) / 1e9; // 计算时间间隔（秒）
            
            float vel_x = 0, vel_y = 0, vel_yaw = 0;
            float pid_output_x = 0, pid_output_y = 0, pid_output_yaw = 0;
            
            // 锁定并读取当前状态和目标状态，计算控制量
            {
                std::lock_guard<std::mutex> lock(global.mtx_lidar);
                std::lock_guard<std::mutex> lock2(global.mtx_target);
                
                // 读取visual点信息
                float visual_x, visual_y;
                bool use_visual;
                {
                    std::lock_guard<std::mutex> lock3(global.mtx_visual);
                    visual_x = global._visual_x;
                    visual_y = global._visual_y;
                    use_visual = global._use_visual;
                }
                
                // 计算目标航向角
                float dynamic_target_yaw = global._target_yaw;
                if (use_visual) {
                    dynamic_target_yaw = calculateYawToPoint(global._lidar_x, global._lidar_y, visual_x, visual_y);
                }
                
                // 计算全局坐标系下的位置误差控制量
                pid_output_x = pid_x.PIDcalculate(global._target_x, global._lidar_x, seconds_double);
                pid_output_y = pid_y.PIDcalculate(global._target_y, global._lidar_y, seconds_double);
                pid_output_yaw = pid_yaw.PIDcalculate(dynamic_target_yaw, global._lidar_yaw, seconds_double);
                
                // 将全局坐标系下的控制量转换到机器人本体坐标系
                float yaw_rad = global._lidar_yaw * M_PI / 180.0; // 度转弧度
                vel_msg.linear.x = pid_output_x * std::cos(yaw_rad) + pid_output_y * std::sin(yaw_rad);
                vel_msg.linear.y = -pid_output_x * std::sin(yaw_rad) + pid_output_y * std::cos(yaw_rad);
                vel_msg.angular.z = pid_output_yaw;
                
                vel_x = vel_msg.linear.x;
                vel_y = vel_msg.linear.y;
                vel_yaw = vel_msg.angular.z;
                
                // 调试输出：显示动态计算的目标航向角
                // if (debug_counter % DEBUG_PRINT_INTERVAL == 0) {
                //     ROS_INFO("Dynamic target yaw: %.3f° (use_visual=%d)", dynamic_target_yaw, use_visual);
                // }
            }
            
            // 定期打印控制量信息
            // if (debug_counter % DEBUG_PRINT_INTERVAL == 0) {
            //     ROS_INFO("PID Outputs: x=%.3f, y=%.3f, yaw=%.3f", pid_output_x, pid_output_y, pid_output_yaw);
            //     ROS_INFO("Velocity Command: vx=%.3f, vy=%.3f, vyaw=%.3f", vel_x, vel_y, vel_yaw);
            //     ROS_INFO("Time interval: %.6f s", seconds_double);
            // }
        }
        else 
        {
            vel_msg.linear.x = 0;
            vel_msg.linear.y = 0;
            vel_msg.angular.z = 0;
            
            if (debug_counter % DEBUG_PRINT_INTERVAL == 0) {
                ROS_WARN("Robot stopped due to flag=%d", flag);
            }
        }
        
        // 更新时间戳，用于下一次PID计算
        start = std::chrono::steady_clock::now();
        
        // 发布速度控制指令
        cmd_vel_pub.publish(vel_msg);
        
        // if (debug_counter % (DEBUG_PRINT_INTERVAL*2) == 0) {
        //     ROS_INFO("Published cmd_vel: linear.x=%.3f, linear.y=%.3f, angular.z=%.3f", 
        //             vel_msg.linear.x, vel_msg.linear.y, vel_msg.angular.z);
        // }
        
        ros::spinOnce();
        loop_rate.sleep();
    }
    
    for (auto& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    
    ROS_INFO("basemove2_node shutdown");
    return 0;
}
