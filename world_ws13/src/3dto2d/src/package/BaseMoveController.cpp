#include "BaseMoveController.h"
#include "PID.cpp"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <chrono>
#include <cmath>

BaseMoveController::BaseMoveController(ros::NodeHandle& nh)
    : nh_(nh) {
    pid_x_ = std::make_unique<PIDcontroler>();
    pid_y_ = std::make_unique<PIDcontroler>();
    pid_yaw_ = std::make_unique<PIDcontroler>(0.31, 0.3, 0.02, 0.08, 0.0, 0.0);

    initializeTargetList();
    cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 10);
    state_.static_start_time = std::chrono::steady_clock::now();
}

BaseMoveController::~BaseMoveController() {
    stop();
}

bool BaseMoveController::start() {
    if (running_) {
        ROS_WARN("BaseMoveController is already running.");
        return false;
    }
    running_ = true;
    completed_ = false;
    {
        std::lock_guard<std::mutex> lock(state_.mtx_target_list);
        state_.current_target_index = 0;
        if (!state_.target_list.empty()) {
            state_.target_x = state_.target_list[0].first;
            state_.target_y = state_.target_list[0].second;
        }
        state_.flag = 0;
        state_.last_lidar_x = 0.0f;
        state_.last_lidar_y = 0.0f;
        state_.static_start_time = std::chrono::steady_clock::now();
    }
    control_thread_ = std::thread(&BaseMoveController::controlLoop, this);
    ROS_INFO("BaseMoveController started.");
    return true;
}

void BaseMoveController::stop() {
    running_ = false;
    if (control_thread_.joinable()) {
        control_thread_.join();
    }
    geometry_msgs::Twist stop_msg;
    cmd_vel_pub_.publish(stop_msg);
    ROS_INFO("BaseMoveController stopped.");
}

bool BaseMoveController::isCompleted() const {
    return completed_;
}

void BaseMoveController::setTargetList(const std::vector<std::pair<float, float>>& targets) {
    std::lock_guard<std::mutex> lock(state_.mtx_target_list);
    state_.target_list = targets;
    state_.current_target_index = 0;
    if (!state_.target_list.empty()) {
        state_.target_x = state_.target_list[0].first;
        state_.target_y = state_.target_list[0].second;
    }
    completed_ = false;
    ROS_INFO("Target list updated with %zu points.", state_.target_list.size());
}

void BaseMoveController::waitForCompletion() {
    while (running_ && !completed_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void BaseMoveController::initializeTargetList() {
    std::lock_guard<std::mutex> lock(state_.mtx_target_list);
    state_.target_list.clear();
    state_.target_list.push_back(std::make_pair(0.0, 0.0));
    state_.target_list.push_back(std::make_pair(0.95, 1.0));
    state_.target_list.push_back(std::make_pair(0.0, 0.0));
    state_.target_list.push_back(std::make_pair(0.0, 1.2));
    state_.target_list.push_back(std::make_pair(0.95, 1.6));
    state_.target_list.push_back(std::make_pair(0.0, 1.2));
    state_.target_list.push_back(std::make_pair(0.9, -0.65));
    state_.target_list.push_back(std::make_pair(0.0, -0.65));
    state_.target_list.push_back(std::make_pair(1.7, 1.6));
    state_.target_list.push_back(std::make_pair(1.7, 3.0));
    state_.target_list.push_back(std::make_pair(1.4, 3.0));
    state_.target_list.push_back(std::make_pair(1.7, -0.4));
    state_.target_list.push_back(std::make_pair(2.1, 1.7));

    if (!state_.target_list.empty()) {
        state_.target_x = state_.target_list[0].first;
        state_.target_y = state_.target_list[0].second;
    }
    ROS_INFO("Target list initialized with %zu points", state_.target_list.size());
}

void BaseMoveController::tfCallback(const tf2_msgs::TFMessage::ConstPtr& msg) {
    bool found = false;
    for (const auto& transform : msg->transforms) {
        if(transform.child_frame_id == "base_footprint") {
            float x = transform.transform.translation.x;
            float y = transform.transform.translation.y;
            geometry_msgs::Quaternion quat = transform.transform.rotation;
            tf2::Quaternion tf_quat(quat.x, quat.y, quat.z, quat.w);
            tf2::Matrix3x3 mat(tf_quat);
            double roll, pitch, yaw;
            mat.getRPY(roll, pitch, yaw);
            float yaw_deg = yaw * 180.0 / M_PI;
            float adjusted_x = x - 0.28*cos(yaw) + 0.28;
            float adjusted_y = y - 0.28*sin(yaw);

            std::lock_guard<std::mutex> lock(state_.mtx_lidar);
            state_.lidar_x = adjusted_x;
            state_.lidar_y = adjusted_y;
            state_.lidar_yaw = yaw_deg;
            found = true;
            break;
        }
    }
    if(!found) ROS_WARN_THROTTLE(1, "TF: No base_footprint found!");
}

void BaseMoveController::visualCallback(const geometry_msgs::PointStamped::ConstPtr& msg) {
    std::lock_guard<std::mutex> lock(state_.mtx_visual);
    state_.visual_x = msg->point.x;
    state_.visual_y = msg->point.y;
    state_.use_visual = true;
}

bool BaseMoveController::checkArrival(float current_x, float current_y, float target_x, float target_y) const {
    float dx = target_x - current_x;
    float dy = target_y - current_y;
    float distance = sqrt(dx*dx + dy*dy);
    return distance < ARRIVAL_THRESHOLD;
}

// 核心修复：切换目标时重置PID，解决不动问题
bool BaseMoveController::switchToNextTarget() {
    std::lock_guard<std::mutex> lock(state_.mtx_target_list);
    if (state_.target_list.empty()) {
        ROS_WARN("Target list is empty!");
        return false;
    }

    // 重置所有PID控制器，清空积分饱和
    pid_x_->reset();
    pid_y_->reset();
    pid_yaw_->reset();

    state_.current_target_index++;
    if (state_.current_target_index >= state_.target_list.size()) {
        ROS_INFO("Reached all target points! Stopping.");
        state_.flag = 1;
        completed_ = true;
        return false;
    }

    state_.target_x = state_.target_list[state_.current_target_index].first;
    state_.target_y = state_.target_list[state_.current_target_index].second;
    ROS_INFO("Switched to target %d: (%.2f, %.2f)",
             state_.current_target_index, state_.target_x, state_.target_y);

    // 重置静止检测
    std::lock_guard<std::mutex> lock_lidar(state_.mtx_lidar);
    state_.static_start_time = std::chrono::steady_clock::now();
    state_.last_lidar_x = state_.lidar_x;
    state_.last_lidar_y = state_.lidar_y;
    return true;
}

float BaseMoveController::calculateYawToPoint(float robot_x, float robot_y, float target_x, float target_y) const {
    float dx = target_x - robot_x;
    float dy = target_y - robot_y;
    float yaw_rad = atan2(dy, dx);
    float yaw_deg = yaw_rad * 180.0 / M_PI;
    while (yaw_deg > 180.0) yaw_deg -= 360.0;
    while (yaw_deg < -180.0) yaw_deg += 360.0;
    return yaw_deg;
}

// 8秒静止超时检测
void BaseMoveController::checkStaticTimeout(float current_x, float current_y) {
    if (completed_ || state_.flag == 1) return;

    std::lock_guard<std::mutex> lock(state_.mtx_lidar);
    float dx = current_x - state_.last_lidar_x;
    float dy = current_y - state_.last_lidar_y;
    float move_dist = sqrt(dx*dx + dy*dy);

    if (move_dist < state_.STATIC_MOVE_THRESHOLD) {
        auto now = std::chrono::steady_clock::now();
        double static_time = std::chrono::duration<double>(now - state_.static_start_time).count();

        if (static_time >= state_.STATIC_TIMEOUT) {
            ROS_WARN("Robot static for 8s! Force switch to next target.");
            switchToNextTarget();
        }
    } else {
        state_.static_start_time = std::chrono::steady_clock::now();
        state_.last_lidar_x = current_x;
        state_.last_lidar_y = current_y;
    }
}

void BaseMoveController::controlLoop() {
    ros::NodeHandle nh_local;
    tf_sub_ = nh_local.subscribe("/tf", 10, &BaseMoveController::tfCallback, this);
    visual_sub_ = nh_local.subscribe("/visual_point", 10, &BaseMoveController::visualCallback, this);

    ros::Rate loop_rate(11);
    auto last_target_switch_time = std::chrono::steady_clock::now();
    auto start = std::chrono::steady_clock::now();

    while (running_ && ros::ok()) {
        geometry_msgs::Twist vel_msg;
        int flag_local;
        float current_x = 0, current_y = 0;

        {
            std::lock_guard<std::mutex> lock(state_.mtx_lidar);
            flag_local = state_.flag;
            current_x = state_.lidar_x;
            current_y = state_.lidar_y;
        }

        // 执行静止检测
        checkStaticTimeout(current_x, current_y);

        if(flag_local == 0) {
            auto end = std::chrono::steady_clock::now();
            double seconds_double = std::chrono::duration<double>(end - start).count();

            float pid_output_x = 0, pid_output_y = 0, pid_output_yaw = 0;
            {
                std::lock_guard<std::mutex> lock_lidar(state_.mtx_lidar);
                std::lock_guard<std::mutex> lock_target(state_.mtx_target);

                float visual_x, visual_y;
                bool use_visual;
                {
                    std::lock_guard<std::mutex> lock_visual(state_.mtx_visual);
                    visual_x = state_.visual_x;
                    visual_y = state_.visual_y;
                    use_visual = state_.use_visual;
                }

                float dynamic_target_yaw = state_.target_yaw;
                if (use_visual) {
                    dynamic_target_yaw = calculateYawToPoint(state_.lidar_x, state_.lidar_y, visual_x, visual_y);
                }

                pid_output_x = pid_x_->PIDcalculate(state_.target_x, state_.lidar_x, seconds_double);
                pid_output_y = pid_y_->PIDcalculate(state_.target_y, state_.lidar_y, seconds_double);
                pid_output_yaw = pid_yaw_->PIDcalculate(dynamic_target_yaw, state_.lidar_yaw, seconds_double);

                float yaw_rad = state_.lidar_yaw * M_PI / 180.0;
                vel_msg.linear.x = pid_output_x * cos(yaw_rad) + pid_output_y * sin(yaw_rad);
                vel_msg.linear.y = -pid_output_x * sin(yaw_rad) + pid_output_y * cos(yaw_rad);
                vel_msg.angular.z = pid_output_yaw;
            }

            // 到达目标点判断
            auto current_time = std::chrono::steady_clock::now();
            double time_since_switch = std::chrono::duration<double>(current_time - last_target_switch_time).count();
            if (!completed_ && time_since_switch > MIN_SWITCH_INTERVAL) {
                float target_x, target_y;
                {
                    std::lock_guard<std::mutex> lock_target(state_.mtx_target);
                    target_x = state_.target_x;
                    target_y = state_.target_y;
                }
                if (checkArrival(current_x, current_y, target_x, target_y)) {
                    ROS_INFO("Arrived target! Switch next.");
                    if (switchToNextTarget()) {
                        last_target_switch_time = current_time;
                    }
                }
            }
        } else {
            vel_msg.linear.x = 0;
            vel_msg.linear.y = 0;
            vel_msg.angular.z = 0;
        }

        start = std::chrono::steady_clock::now();
        cmd_vel_pub_.publish(vel_msg);
        ros::spinOnce();
        loop_rate.sleep();
    }

    geometry_msgs::Twist stop_msg;
    cmd_vel_pub_.publish(stop_msg);
    ROS_INFO("BaseMoveController control loop finished.");
}