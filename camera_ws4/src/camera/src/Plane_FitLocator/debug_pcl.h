#ifndef __DEBUG_PCL_H_
#define __DEBUG_PCL_H_
#include <ros/ros.h>
#include <iostream>
#include <string>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include "pre_pcl.h"

namespace Ten
{
namespace Plane_FitLocator
{
class Ten_debug_pcl
{
public:
    // 构造函数， 初始化节点
    Ten_debug_pcl()
    {
        pcl_pub_ = nh.advertise<sensor_msgs::PointCloud2>(topic_name, 10);
    }

    void publish_pointcloud(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& pcl_cloud,
        std::string frame_id = "map"
    )
    {
        if (!pcl_pub_ || pcl_cloud->empty()) return;

        // 转ROS消息并发布
        sensor_msgs::PointCloud2 ros_cloud;
        pcl::toROSMsg(*pcl_cloud, ros_cloud);
        ros_cloud.header.frame_id = frame_id;
        ros_cloud.header.stamp = ros::Time::now();
        pcl_pub_.publish(ros_cloud);
    }

    /**
     * @brief 发布平面位姿到 TF 话题
     * @param plane_info 面的相关信息
     * @param parent_frame 父坐标系（相机坐标系：camera_color_optical_frame / camera_link）
     * @param child_frame  子坐标系（自定义：detected_plane）
     * @param stamp        时间戳（默认 ros::Time::now()）
     */
    void publish_PlaneTF(
        const Plane_FitLocator::Plane_Info& plane_info,
        const std::string& parent_frame = "camera_color_optical_frame",
        const std::string& child_frame = "detected_plane",
        const ros::Time& stamp = ros::Time::now()
    )
    {
        // 1. 静态 TF 广播器（类内全局，只初始化一次）
        static tf2_ros::TransformBroadcaster tf_broadcaster;

        // 2. 创建 TF 消息
        geometry_msgs::TransformStamped tf_msg;

        // 3. 填充坐标系信息
        tf_msg.header.stamp = stamp;
        tf_msg.header.frame_id = parent_frame;   // 父坐标系（相机）
        tf_msg.child_frame_id = child_frame;     // 子坐标系（平面）

        // 4. 填充平移：平面质心（你的 Plane_Info 中心）
        tf_msg.transform.translation.x = plane_info.plane_center.x();
        tf_msg.transform.translation.y = plane_info.plane_center.y();
        tf_msg.transform.translation.z = plane_info.plane_center.z();

        // 5. 旋转：RPY 转 四元数（TF 强制要求四元数）
        tf2::Quaternion q;
        // 传入：roll, pitch, yaw（你的结构体 RPY）
        q.setRPY(
            plane_info.plane_euler._roll,
            plane_info.plane_euler._pitch,
            plane_info.plane_euler._yaw
        );
        // 填充四元数到 TF 消息
        tf_msg.transform.rotation.x = q.x();
        tf_msg.transform.rotation.y = q.y();
        tf_msg.transform.rotation.z = q.z();
        tf_msg.transform.rotation.w = q.w();

        // 6. 发布 TF
        tf_broadcaster.sendTransform(tf_msg);
    }

    /**
     * @brief 调试彩色图像
     * @param iuput_image 输入图像
     * @param plane_info 面的相关信息
     * @param color_intr 彩色相机内参信息
     * @param output_image 输出调试图像
    */
    void debug_rgb_image(
        const cv::Mat& input_image,
        const Plane_Info& plane_info,
        const rs2_intrinsics& color_intr, 
        cv::Mat& output_image
    )
    {
        output_image = input_image.clone();
        Eigen::Vector3d center_3d = plane_info.plane_center;

        // 1. 3D相机坐标 → 投影为2D像素坐标 (u, v)
        float point3d[3] = {(float)center_3d.x(), (float)center_3d.y(), (float)center_3d.z()};
        float pixel[2] = {0};
        rs2_project_point_to_pixel(pixel, &color_intr, point3d);

        int u = cvRound(pixel[0]);
        int v = cvRound(pixel[1]);

        // 2. OpenCV绘制：红色实心圆 (图像，圆心，半径，颜色，厚度)
        if (u >= 0 && u < output_image.cols && v >= 0 && v < output_image.rows)
        {
            cv::circle(output_image, cv::Point(u, v), 8, cv::Scalar(0, 0, 255), -1);
            cv::putText(output_image, "Plane Center", cv::Point(u+10, v), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);

            char coord_text[100];
            sprintf(coord_text, "X:%.3f Y:%.3f Z:%.3f", 
                    center_3d.x(), center_3d.y(), center_3d.z());
            
            // 绘制在中心点下方，避免重叠
            cv::putText(output_image, coord_text, cv::Point(u+10, v + 20), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 255), 2);
        }
    }

    /**
     * @brief 调试图像：根据中心点+RPY(旋转矩阵) 绘制35cm标准平面正方形（无角点依赖）
     * @param input_image 输入RGB图像
     * @param plane_info 包含平面中心、旋转矩阵、RPY的结构体
     * @param color_intr 彩色相机内参
     * @param output_image 输出调试图像
     */
    void debug_plane_quadrilateral(
        const cv::Mat& input_image,
        const Plane_Info& plane_info,
        const rs2_intrinsics& color_intr, 
        cv::Mat& output_image
    )
    {
        // ===================== 核心参数：35cm正方形固定尺寸 =====================
        const double SQUARE_SIZE = 0.35;    
        const double HALF_SIZE = SQUARE_SIZE / 2.0;

        output_image = input_image.clone();
        std::vector<cv::Point> pixel_points;

        // ===================== 修复：正确4个正方形角点，不要写错 =====================
        std::vector<Eigen::Vector3d> local_corners = {
            Eigen::Vector3d(-HALF_SIZE, -HALF_SIZE, 0.0),  // 左下
            Eigen::Vector3d( HALF_SIZE, -HALF_SIZE, 0.0),  // 右下
            Eigen::Vector3d( HALF_SIZE,  HALF_SIZE, 0.0),  // 右上
            Eigen::Vector3d(-HALF_SIZE,  HALF_SIZE, 0.0)   // 左上
        };

        const Eigen::Vector3d& center = plane_info.plane_center;

        // 1. 用法向量构建基础坐标系
        const Eigen::Vector3d& n = plane_info.plane_normal;
        Eigen::Vector3d norm_n = n;
        norm_n.normalize();
        Eigen::Vector3d x_axis, y_axis;
        if (std::fabs(norm_n.z()) < 0.999) {
            x_axis = Eigen::Vector3d(1, 0, 0).cross(norm_n).normalized();
        } else {
            x_axis = Eigen::Vector3d(0, 1, 0).cross(norm_n).normalized();
        }
        y_axis = norm_n.cross(x_axis).normalized();

        // 2. 初始旋转矩阵
        Eigen::Matrix3d rot_mat;
        rot_mat.col(0) = x_axis;
        rot_mat.col(1) = y_axis;
        rot_mat.col(2) = norm_n;

        // 3. 叠加最优Yaw，和TF对齐
        double yaw = plane_info.plane_euler._yaw;
        Eigen::Matrix3d rot_yaw;
        rot_yaw << cos(yaw), -sin(yaw), 0,
                sin(yaw),  cos(yaw), 0,
                0,         0,        1;
        rot_mat = rot_mat * rot_yaw;

        // 四角点转世界坐标
        std::vector<Eigen::Vector3d> world_corners;
        for (const auto& local_pt : local_corners)
        {
            Eigen::Vector3d world_pt = rot_mat * local_pt + center;
            world_corners.push_back(world_pt);
        }

        // 投影到像素
        for (const auto& pt_3d : world_corners)
        {
            float point3d[3] = {(float)pt_3d.x(), (float)pt_3d.y(), (float)pt_3d.z()};
            float pixel[2] = {0};
            rs2_project_point_to_pixel(pixel, &color_intr, point3d);
            
            int u = cvRound(pixel[0]);
            int v = cvRound(pixel[1]);
            pixel_points.emplace_back(u, v);
        }

        // 绘制方框
        for (int i = 0; i < pixel_points.size(); i++)
        {
            cv::Point p = pixel_points[i];
            if (p.x < 0 || p.x >= output_image.cols || p.y < 0 || p.y >= output_image.rows)
                continue;

            cv::circle(output_image, p, 6, cv::Scalar(255, 0, 0), -1);
            cv::putText(output_image, std::to_string(i+1), cv::Point(p.x+5, p.y),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);

            int next_idx = (i + 1) % pixel_points.size();
            cv::Point p_next = pixel_points[next_idx];
            if (p_next.x >= 0 && p_next.x < output_image.cols && p_next.y >= 0 && p_next.y < output_image.rows)
            {
                cv::line(output_image, p, p_next, cv::Scalar(0, 255, 0), 2);
            }
        }

        // 绘制中心点文字
        Eigen::Vector3d center_3d = plane_info.plane_center;
        float c_point3d[3] = {(float)center_3d.x(), (float)center_3d.y(), (float)center_3d.z()};
        float c_pixel[2] = {0};
        rs2_project_point_to_pixel(c_pixel, &color_intr, c_point3d);
        cv::Point center_p(cvRound(c_pixel[0]), cvRound(c_pixel[1]));
        int u = c_pixel[0];
        int v = c_pixel[1];
        if (center_p.x >=0 && center_p.x < output_image.cols && center_p.y >=0 && center_p.y < output_image.rows)
        {
            cv::circle(output_image, center_p, 8, cv::Scalar(0, 0, 255), -1);
            cv::putText(output_image, "Plane Center", cv::Point(u+10, v), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);

            char coord_text[100];
            sprintf(coord_text, "X:%.3f Y:%.3f Z:%.3f", 
                    center_3d.x(), center_3d.y(), center_3d.z());
            
            cv::putText(output_image, coord_text, cv::Point(u+10, v + 20), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 255), 2);
        }
    }


private:
ros::NodeHandle nh;
ros::Publisher pcl_pub_;
std::string topic_name = "/camera/pointcloud";
};

}       // namespace Plane_FitLocator
}       // namespace Ten
#endif 