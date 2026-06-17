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
#include "set_plane.h"

namespace Ten
{
namespace kfs_locator
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

    void pub_color_image
    (
        const cv::Mat& color_image,
        const std::string topic_name = "/debug_images"
    )
    {
        static ros::Publisher pub;
        if (!pub)
        {
            ros::NodeHandle nh;
            pub = nh.advertise<sensor_msgs::Image>(topic_name, 10);
        }
        if (color_image.empty() || color_image.channels() != 3) return;

        cv_bridge::CvImage cv_msg;
        cv_msg.header.stamp = ros::Time::now();
        cv_msg.encoding = sensor_msgs::image_encodings::BGR8; // 固定彩色格式
        cv_msg.image = color_image;
        pub.publish(cv_msg.toImageMsg());
    }

    // --------------------- 发布 深度图（16UC1，相机原生） ---------------------
    void pub_depth_image
    (
        const cv::Mat& depth_image,
        const std::string topic_name = "/depth_show"
    )
    {
        static ros::Publisher pub;
        if (!pub)
        {
            ros::NodeHandle nh;
            pub = nh.advertise<sensor_msgs::Image>(topic_name, 10);
        }
        if (depth_image.empty() || depth_image.type() != CV_16UC1) return;

        cv_bridge::CvImage cv_msg;
        cv_msg.header.stamp = ros::Time::now();
        cv_msg.encoding = sensor_msgs::image_encodings::TYPE_16UC1; // 固定深度格式
        cv_msg.image = depth_image;
        pub.publish(cv_msg.toImageMsg());
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

    void getLocalAxes(const Eigen::Vector3d& n, Eigen::Vector3d& x, Eigen::Vector3d& y)
    {
        Eigen::Vector3d normal = n.normalized();
        Eigen::Vector3d aux = Eigen::Vector3d(0,0,1);
        if(fabs(normal.dot(aux)) > 0.999) aux = Eigen::Vector3d(1,0,0);

        x = aux.cross(normal).normalized();
        y = normal.cross(x).normalized();
    }

};

}       // namespace kfs_locator
}       // namespace Ten
#endif 