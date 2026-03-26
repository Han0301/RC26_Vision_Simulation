#ifndef BASE_MOVE_CONTROLLER_H
#define BASE_MOVE_CONTROLLER_H

#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PointStamped.h>
#include <tf2_msgs/TFMessage.h>
#include <std_msgs/Bool.h>
#include <mutex>
#include <thread>
#include <vector>
#include <atomic>
#include <functional>
#include <chrono>

class PIDcontroler;

class BaseMoveController {
public:
    BaseMoveController(ros::NodeHandle& nh);
    ~BaseMoveController();

    bool start();
    void stop();
    bool isCompleted() const;
    void setTargetList(const std::vector<std::pair<float, float>>& targets);
    void waitForCompletion();

private:
    void controlLoop();
    void tfCallback(const tf2_msgs::TFMessage::ConstPtr& msg);
    void visualCallback(const geometry_msgs::PointStamped::ConstPtr& msg);
    void initializeTargetList();
    bool checkArrival(float current_x, float current_y, float target_x, float target_y) const;
    bool switchToNextTarget();
    float calculateYawToPoint(float robot_x, float robot_y, float target_x, float target_y) const;
    void checkStaticTimeout(float current_x, float current_y);

    ros::NodeHandle& nh_;
    ros::Subscriber tf_sub_;
    ros::Subscriber visual_sub_;
    ros::Publisher cmd_vel_pub_;

    std::thread control_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> completed_{false};

    struct {
        float lidar_x = 0.0f;
        float lidar_y = 0.0f;
        float lidar_yaw = 0.0f;
        std::vector<std::pair<float, float>> target_list;
        int current_target_index = 0;
        float target_x = 0.0f;
        float target_y = 0.0f;
        float target_yaw = 0.0f;
        bool use_visual = true;

        float visual_x = 5.2;
        float visual_y = 1.6;
        int flag = 0;
        std::mutex mtx_lidar;
        std::mutex mtx_target;
        std::mutex mtx_visual;
        std::mutex mtx_target_list;

        float last_lidar_x = 0.0f;
        float last_lidar_y = 0.0f;
        std::chrono::steady_clock::time_point static_start_time;
        const float STATIC_TIMEOUT = 8.0f;
        const float STATIC_MOVE_THRESHOLD = 0.02f;
    } state_;

    std::unique_ptr<PIDcontroler> pid_x_;
    std::unique_ptr<PIDcontroler> pid_y_;
    std::unique_ptr<PIDcontroler> pid_yaw_;

    const float ARRIVAL_THRESHOLD = 0.05f;
    const double MIN_SWITCH_INTERVAL = 2.0;
    int force_move_counter_ = 0;
};

#endif