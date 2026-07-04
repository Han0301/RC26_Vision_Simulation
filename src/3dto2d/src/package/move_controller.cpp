#include "move_controller.h"

namespace Ten
{
    void Ten::Ten_move_controller::move_controller(ros::Publisher &cmd_vel_pub)
    {
        static float linear_vel = 0.3;
        static float angular_vel = 0.3;

        geometry_msgs::Twist base_cmd;
        base_cmd.linear.x = 0;
        base_cmd.linear.y = 0;
        base_cmd.angular.z = 0;

        int k_vel = 2; // 与键盘控制相同的速度倍数

        printf("开始自动移动控制：\n");

        printf("刹车停止\n");
        base_cmd.linear.x = 0;
        base_cmd.linear.y = 0;
        base_cmd.angular.z = 0;
        cmd_vel_pub.publish(base_cmd);
        ros::Duration(1).sleep();

        // 第一阶段：向前移动1秒（仿照按'w'键）
        printf("向前移动1秒\n");
        base_cmd.linear.x = linear_vel * k_vel; // 直接达到最大速度，仿照持续按w键的效果
        cmd_vel_pub.publish(base_cmd);
        ros::Duration(1.0).sleep(); // 持续1秒
        base_cmd.linear.x = 0;
        base_cmd.linear.y = 0;
        base_cmd.angular.z = 0;
        cmd_vel_pub.publish(base_cmd);

        // 向左运动
        printf("向左移动5秒\n");
        base_cmd.linear.y = linear_vel * k_vel; // 直接达到最大速度，仿照持续按w键的效果
        cmd_vel_pub.publish(base_cmd);
        ros::Duration(5.0).sleep();
        base_cmd.linear.x = 0;
        base_cmd.linear.y = 0;
        base_cmd.angular.z = 0;
        cmd_vel_pub.publish(base_cmd);

        // 向前
        base_cmd.linear.x = linear_vel * k_vel; // 直接达到最大速度，仿照持续按w键的效果
        cmd_vel_pub.publish(base_cmd);
        ros::Duration(3.0).sleep();
        base_cmd.linear.x = 0;
        base_cmd.linear.y = 0;
        base_cmd.angular.z = 0;
        cmd_vel_pub.publish(base_cmd);

        printf("向右移动5秒\n");
        base_cmd.linear.y = -linear_vel * k_vel; // 直接达到最大速度，仿照持续按w键的效果
        cmd_vel_pub.publish(base_cmd);
        ros::Duration(5.0).sleep(); // 持续1秒
        base_cmd.linear.x = 0;
        base_cmd.linear.y = 0;
        base_cmd.angular.z = 0;
        cmd_vel_pub.publish(base_cmd);

        // 左
        base_cmd.linear.y = linear_vel * k_vel * 1.2; // 直接达到最大速度，仿照持续按w键的效果
        base_cmd.angular.z = linear_vel * k_vel / 2;
        cmd_vel_pub.publish(base_cmd);
        ros::Duration(2.5).sleep(); // 持续1秒
        base_cmd.linear.x = 0;
        base_cmd.linear.y = 0;
        base_cmd.angular.z = 0;
        base_cmd.linear.y = linear_vel * k_vel * 1.2;
        base_cmd.angular.z = -linear_vel * k_vel / 2;
        cmd_vel_pub.publish(base_cmd);
        ros::Duration(2.5).sleep();
        base_cmd.linear.x = 0;
        base_cmd.linear.y = 0;
        base_cmd.angular.z = 0;

        printf("向右移动5秒\n");
        base_cmd.angular.z = -linear_vel * k_vel / 2;
        base_cmd.linear.y = -linear_vel * k_vel * 4 / 5; // 直接达到最大速度，仿照持续按w键的效果
        base_cmd.linear.x = linear_vel * k_vel;
        cmd_vel_pub.publish(base_cmd);
        ros::Duration(3.0).sleep(); // 持续1秒
        base_cmd.linear.x = 0;
        base_cmd.linear.y = 0;
        base_cmd.angular.z = 0;
        cmd_vel_pub.publish(base_cmd);
        ros::Duration(1).sleep();

        printf("移动结束！\n");
        ros::shutdown();
    };

    void Ten::Ten_move_controller::move_controller2(ros::Publisher &cmd_vel_pub)
    {

    static float linear_vel = 0.3;
    static float angular_vel = 0.3;

    geometry_msgs::Twist base_cmd;
    
    int k_vel = 2; // 与键盘控制相同的速度倍数
    
    printf("开始自动移动控制：\n");

    printf("刹车停止\n");
    base_cmd.linear.x = 0;
    base_cmd.linear.y = 0;
    base_cmd.angular.z = 0;
    cmd_vel_pub.publish(base_cmd);
    ros::Duration(1).sleep();
    
    // 第一阶段：向前移动5秒（仿照按'w'键）
    printf("向前移动5秒\n");
    base_cmd.linear.x = linear_vel * k_vel; // 直接达到最大速度，仿照持续按w键的效果
    cmd_vel_pub.publish(base_cmd);
    ros::Duration(4.0).sleep(); // 持续1秒
    base_cmd.linear.x = 0;
    base_cmd.linear.y = 0;
    base_cmd.angular.z = 0;
    cmd_vel_pub.publish(base_cmd);

    //向右运动
    printf("向右移动2秒\n");
    base_cmd.linear.y = -linear_vel * k_vel; // 直接达到最大速度，仿照持续按w键的效果
    cmd_vel_pub.publish(base_cmd);
    ros::Duration(2.0).sleep();
    base_cmd.linear.x = 0;
    base_cmd.linear.y = 0;
    base_cmd.angular.z = 0;
    cmd_vel_pub.publish(base_cmd);

    
    //向左
    printf("向左移动5秒\n");
    base_cmd.linear.y = linear_vel * k_vel; // 直接达到最大速度，仿照持续按w键的效果
    cmd_vel_pub.publish(base_cmd);
    ros::Duration(8.0).sleep();
    base_cmd.linear.x = 0;
    base_cmd.linear.y = 0;
    base_cmd.angular.z = 0;
    cmd_vel_pub.publish(base_cmd);

    //向右
    printf("旋转\n");
    base_cmd.angular.z = -angular_vel;
    cmd_vel_pub.publish(base_cmd);
    ros::Duration(2.8).sleep();
    base_cmd.linear.x = 0;
    base_cmd.linear.y = 0;
    base_cmd.angular.z = 0;
    cmd_vel_pub.publish(base_cmd);

    printf("弧线\n");
    base_cmd.angular.z = -angular_vel;
    base_cmd.linear.y = -linear_vel * k_vel;
    cmd_vel_pub.publish(base_cmd);
    ros::Duration(1.5).sleep();

    base_cmd.angular.z = 0;
    base_cmd.angular.z = angular_vel;
    cmd_vel_pub.publish(base_cmd);
    ros::Duration(2.3).sleep();

    base_cmd.linear.x = -linear_vel;
    cmd_vel_pub.publish(base_cmd);
    ros::Duration(2.5).sleep();

    base_cmd.linear.y = 0;
    base_cmd.angular.z = 0;
    base_cmd.linear.x  = linear_vel * k_vel;
    cmd_vel_pub.publish(base_cmd);
    ros::Duration(3.6).sleep();

    base_cmd.linear.x  = 0 ;
    base_cmd.angular.z = -angular_vel;
    cmd_vel_pub.publish(base_cmd);
    ros::Duration(1).sleep();



    base_cmd.linear.x = 0;
    base_cmd.linear.y = 0;
    base_cmd.angular.z = 0;
    cmd_vel_pub.publish(base_cmd);

    printf("移动结束！\n");
    ros::shutdown();
    }
    Ten::Ten_move_controller _MOVE_CONTROLLER_;
}