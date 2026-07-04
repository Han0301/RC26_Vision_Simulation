#!/usr/bin/env python3
import rospy
import math
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

class VectorMovement:
    def __init__(self):
        rospy.init_node('vector_movement')
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/odom', Odometry, self.odom_cb)
        self.current_pose = Point()
        self.initial_yaw = None
        self.current_yaw = 0.0
        self.rate = rospy.Rate(20)
        self.tolerance = 0.05
        
        # 定义移动向量序列
        self.movement_vectors = [
            (1.2, 0.0),    # 沿X轴前进1米
            (0.0, 1.6),   # 沿Y轴前进1.6米
            (1.2, 0.0)     # 沿X轴前进1.4米
        ]
        self.current_vector = 0
        self.target_position = Point()
        self.initial_position = None

    def odom_cb(self, msg):
        self.current_pose = msg.pose.pose.position
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, self.current_yaw = euler_from_quaternion(orientation_list)
        
        if self.initial_yaw is None:
            self.initial_yaw = self.current_yaw
        
        if self.initial_position is None:
            self.initial_position = Point()
            self.initial_position.x = self.current_pose.x
            self.initial_position.y = self.current_pose.y
            self.target_position.x = self.initial_position.x
            self.target_position.y = self.initial_position.y

    def move_to_target(self):
        twist = Twist()
        arrived = False
        
        # 计算当前位置与目标位置的偏差
        dx = self.target_position.x - self.current_pose.x
        dy = self.target_position.y - self.current_pose.y
        distance = math.sqrt(dx**2 + dy**2)
        
        # 保持初始朝向
        angle_error = self.initial_yaw - self.current_yaw
        if angle_error > math.pi:
            angle_error -= 2 * math.pi
        elif angle_error < -math.pi:
            angle_error += 2 * math.pi
        twist.angular.z = 1.0 * angle_error
        
        # 如果距离小于容差，则标记为到达
        if distance < self.tolerance:
            arrived = True
        else:
            # 计算移动方向
            target_angle = math.atan2(dy, dx)
            
            # 计算速度
            linear_speed = min(0.5 * distance, 0.5)
            
            # 设置线速度
            twist.linear.x = linear_speed * math.cos(target_angle)
            twist.linear.y = linear_speed * math.sin(target_angle)
        
        return twist, arrived

    def navigate(self):
        rospy.loginfo("Starting vector movement...")
        
        # 等待初始位置设置
        while self.initial_position is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        
        # 执行所有移动向量
        while not rospy.is_shutdown() and self.current_vector < len(self.movement_vectors):
            # 计算新目标位置
            dx, dy = self.movement_vectors[self.current_vector]
            self.target_position.x += dx
            self.target_position.y += dy
            
            rospy.loginfo(f"Executing vector {self.current_vector+1}: move ({dx}, {dy})")
            rospy.loginfo(f"Target position: ({self.target_position.x:.2f}, {self.target_position.y:.2f})")
            
            # 移动到目标位置
            arrived = False
            while not rospy.is_shutdown() and not arrived:
                twist, arrived = self.move_to_target()
                self.cmd_vel_pub.publish(twist)
                self.rate.sleep()
            
            if arrived:
                rospy.loginfo(f"Completed vector {self.current_vector+1}")
                self.current_vector += 1
                rospy.sleep(1)
        
        # 最终停止
        stop_twist = Twist()
        for _ in range(10):
            self.cmd_vel_pub.publish(stop_twist)
            self.rate.sleep()
        
        rospy.loginfo("Vector movement completed!")

if __name__ == '__main__':
    try:
        navigator = VectorMovement()
        rospy.sleep(1)
        navigator.navigate()
    except rospy.ROSInterruptException:
        pass