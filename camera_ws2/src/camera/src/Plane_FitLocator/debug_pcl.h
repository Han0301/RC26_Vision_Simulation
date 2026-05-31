#ifndef __DEBUG_PCL_H_
#define __DEBUG_PCL_H_
#include <ros/ros.h>
#include <iostream>
#include <string>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float64.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2/LinearMath/Quaternion.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include "pre_pcl.h"

#define BOX_SIZE 0.35
namespace Ten
{
namespace Plane_FitLocator
{
class Ten_debug_pcl
{
public:
    void publish_pointcloud(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& pcl_cloud,
        const std::string frame_id = "camera_color_optical_frame",
        const std::string topic_name = "/camera/pointcloud"
    )
    {
        static ros::Publisher pcl_pub_;
        if (!pcl_pub_)
        {
            ros::NodeHandle nh;
            pcl_pub_ = nh.advertise<sensor_msgs::PointCloud2>(topic_name, 10);
        }

        if (pcl_cloud->empty()) return;

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

    // 通用的 发布图像话题 函数
    void pub_debug_image
    (
        const cv::Mat& debug_image,
        const std::string debug_topic_name = "/kfs/debug_image"
    )
    {
        static ros::Publisher debug_img_pub;
        if (!debug_img_pub)
        {
            ros::NodeHandle nh;
            debug_img_pub = nh.advertise<sensor_msgs::Image>(debug_topic_name, 10);
        }

        if (debug_image.empty())return;

        cv_bridge::CvImage cv_msg;
        cv_msg.header.stamp = ros::Time::now();
        cv_msg.encoding = sensor_msgs::image_encodings::BGR8;
        cv_msg.image = debug_image;

        sensor_msgs::ImagePtr msg = cv_msg.toImageMsg();
        debug_img_pub.publish(msg);
    }

    // 通用的数值发布函数
    void pub_value(
        const double input_value,
        const std::string debug_topic_name = "/kfs/debug_value")
    {
        static ros::Publisher debug_value_pub;
        if (!debug_value_pub)
        {
            ros::NodeHandle nh;
            debug_value_pub = nh.advertise<std_msgs::Float64>(debug_topic_name, 10);
        }
        std_msgs::Float64 mag_value;
        mag_value.data = input_value;
        debug_value_pub.publish(mag_value);
    }

    // 设置调试图像：35cm标准平面正方形，中心点，法向量
    void set_debug_plane_quadrilateral(
        const cv::Mat& input_image,
        const Plane_Info& plane_info,
        const rs2_intrinsics& color_intr, 
        cv::Mat& output_image
    )
    {  
        const double HALF_SIZE = BOX_SIZE / 2.0;

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
        Eigen::Vector3d n = plane_info.plane_normal;
        n.normalize();
        Eigen::Vector3d x_axis, y_axis;
        // 【和set_vector_2d/set_RPY用完全一样的坐标系函数】
        getLocalAxes(n, x_axis, y_axis); 

        Eigen::Matrix3d rot_mat;
        rot_mat.col(0) = x_axis;
        rot_mat.col(1) = y_axis;
        rot_mat.col(2) = n;

        // 叠加最终yaw（和set_RPY完全一致）
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
        // for (int i = 0; i < pixel_points.size(); i++)
        // {
        //     cv::Point p = pixel_points[i];
        //     if (p.x < 0 || p.x >= output_image.cols || p.y < 0 || p.y >= output_image.rows)
        //         continue;

        //     cv::circle(output_image, p, 6, cv::Scalar(255, 0, 0), -1);
        //     cv::putText(output_image, std::to_string(i+1), cv::Point(p.x+5, p.y),
        //                 cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);

        //     int next_idx = (i + 1) % pixel_points.size();
        //     cv::Point p_next = pixel_points[next_idx];
        //     if (p_next.x >= 0 && p_next.x < output_image.cols && p_next.y >= 0 && p_next.y < output_image.rows)
        //     {
        //         cv::line(output_image, p, p_next, cv::Scalar(0, 255, 0), 2);
        //     }
        // }

        // 绘制中心点文字
        Eigen::Vector3d center_3d = plane_info.plane_center;
        Eigen::Vector3d normal_3d = plane_info.plane_normal; // 提取法向量
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

        // ===================== 新增：绘制平面法向量（完全仿照 set_debug_rgb_image） =====================
        // const double NORMAL_LENGTH = 0.2; // 法向量长度0.2米，可自行调整
        // const cv::Scalar COLOR_RED = cv::Scalar(0, 0, 255);

        // // 1. 计算法向量终点3D坐标
        // Eigen::Vector3d normal_end_3d = center_3d + normal_3d * NORMAL_LENGTH;

        // // 2. 投影法向量终点到2D像素
        // float n_end_point3d[3] = {(float)normal_end_3d.x(), (float)normal_end_3d.y(), (float)normal_end_3d.z()};
        // float n_end_pixel[2] = {0};
        // rs2_project_point_to_pixel(n_end_pixel, &color_intr, n_end_point3d);
        // cv::Point normal_end_p(cvRound(n_end_pixel[0]), cvRound(n_end_pixel[1]));

        // // 3. 安全检查 + 绘制法向量线段与文字
        // if (center_p.x >= 0 && center_p.x < output_image.cols &&
        //     center_p.y >= 0 && center_p.y < output_image.rows &&
        //     normal_end_p.x >= 0 && normal_end_p.x < output_image.cols &&
        //     normal_end_p.y >= 0 && normal_end_p.y < output_image.rows)
        // {
        //     // 绘制法向量线段
        //     cv::line(output_image, center_p, normal_end_p, COLOR_RED, 2);
        //     // 标注法向量文字
        //     cv::putText(output_image, "Plane Normal", cv::Point(center_p.x + 10, center_p.y - 5),
        //                 cv::FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED, 1);
        // }
    }

    void save_bias(double data, const std::string& save_path)
    {
        // 以 追加模式 打开文件，不存在则自动创建
        std::ofstream file(save_path, std::ios::app | std::ios::out);
        
        if (!file.is_open())
        {
            throw std::runtime_error("无法打开文件：" + save_path);
        }

        // 写入数据，每行一个，保留6位小数（精度可调）
        file << std::fixed << std::setprecision(6) << data << std::endl;
        file.close();
    }

    std::map<std::string, double> read_bias(const std::string& read_path)
    {
        std::map<std::string, double> result;
        std::vector<double> data_list;

        // 1. 打开并读取文件所有数据
        std::ifstream file(read_path);
        if (!file.is_open())
        {
            throw std::runtime_error("文件不存在：" + read_path);
        }

        double val;
        while (file >> val)
        {
            data_list.push_back(val);
        }
        file.close();

        // 2. 空数据判断
        if (data_list.empty())
        {
            throw std::runtime_error("文件中无有效数据：" + read_path);
        }

        size_t n = data_list.size();
        // 3. 排序（用于分位、最值计算）
        std::vector<double> sorted_data = data_list;
        std::sort(sorted_data.begin(), sorted_data.end());

        // 4. 计算 最大值 & 最小值
        double max_val = sorted_data.back();
        double min_val = sorted_data[0];

        // 5. 计算 平均值(avg)
        double sum = 0.0;
        for (double d : data_list) sum += d;
        double avg = sum / n;

        // 6. 计算 标准差(standard_bias) → 总体标准差
        double sum_sq = 0.0;
        for (double d : data_list)
        {
            sum_sq += (d - avg) * (d - avg);
        }
        double standard_bias = std::sqrt(sum_sq / n);

        // 7. 计算 90%上分位、10%下分位（线性插值法，工业标准）
        auto get_percentile = [&](double percent) -> double
        {
            double pos = percent * (n - 1);
            size_t idx = static_cast<size_t>(pos);
            double frac = pos - idx;
            if (idx + 1 >= n) return sorted_data.back();
            return sorted_data[idx] + frac * (sorted_data[idx + 1] - sorted_data[idx]);
        };

        double percentile_90 = get_percentile(0.9);  // 90%上分位
        double percentile_10 = get_percentile(0.1);  // 90%下分位

        // 8. 存入字典返回
        result["max"] = max_val;
        result["min"] = min_val;
        result["avg"] = avg;
        result["standard_bias"] = standard_bias;
        result["90%bias_max"] = percentile_90;
        result["90%bias_min"] = percentile_10;

        return result;
    }

private:

    void getLocalAxes(const Eigen::Vector3d& n,
                                    Eigen::Vector3d& x_axis,
                                    Eigen::Vector3d& y_axis)
    {
        Eigen::Vector3d norm_n = n.normalized();
        // 🔥 固定：永远用世界Z轴构建平面坐标系，绝对不跳变
        Eigen::Vector3d ref_up = Eigen::Vector3d::UnitZ();
        // 正交化：平面X轴 = 世界上方向 × 平面法向
        x_axis = ref_up.cross(norm_n).normalized();
        // 平面Y轴 = 法向 × X轴（保证右手坐标系，正方形永远在平面内）
        y_axis = norm_n.cross(x_axis).normalized();
    }
};

}       // namespace Plane_FitLocator
}       // namespace Ten
#endif 