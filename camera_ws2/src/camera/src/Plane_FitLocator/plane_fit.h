#include "debug_pcl.h"
#include "post_pcl.h"
#include "pre_pcl.h"
#include "set_pcl.h"
#include "../camera.h"

namespace Ten
{
namespace Plane_FitLocator
{

struct result
{
    double deviation_angle = 0.0;     // 目标面的法向量到预设面的法向量的角度
    double bias = 0.0;                // 体中心点到预设面上预设直线的投影偏差距离
    double distance = 0.0;            // 体中心点到预设面的距离
};


class plane_fit
{
public:
    plane_fit(rs2_intrinsics color_intr)
    :   input_cloud(new pcl::PointCloud<pcl::PointXYZ>),
        filter_cloud(new pcl::PointCloud<pcl::PointXYZ>),
        plane_cloud(new pcl::PointCloud<pcl::PointXYZ>),
        plane_cloud_fited(new pcl::PointCloud<pcl::PointXYZ>),
        plane_cloud_vec(new pcl::PointCloud<pcl::PointXYZ>)
    {
        color_intr_ = color_intr;
        depth_show = cv::Mat();
        debug_image = cv::Mat();
    }

    // 核心处理函数
    bool process(Ten::camera_frame& frame);
    // 发布调试图像和tf， 点云
    void publish(Ten::camera_frame& frame);
    // 取接口
    const Ten::Plane_FitLocator::Plane_Info get_plane_info() 
    {
        return plane_info;
    }
    const std::string get_state() 
    {
        return state;
    }
    /**
     * @brief 设置结果（确保key_center已被设置）
     * @param target_plane 预设平面方程
     * @param line_point   预设直线上一点
     * @param line_dir     预设直线方程
    */
    result set_result(
        const Eigen::Vector3d& target_plane,        
        const Eigen::Vector3d& line_point, 
        const Eigen::Vector3d& line_dir);

private:
    // 参数
    rs2_intrinsics color_intr_;
    // 处理器
    Ten::Plane_FitLocator::Ten_debug_pcl DEBUG_PCL_;
    Ten::Plane_FitLocator::Ten_pre_pcl PRE_PCL_;
    Ten::Plane_FitLocator::Ten_set_pcl SET_PCL_;
    Ten::Plane_FitLocator::Ten_post_pcl POST_PCL_;
    // 中间变量
    Ten::Plane_FitLocator::Plane_Info plane_info;
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr filter_cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane_cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane_cloud_fited; 
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane_cloud_vec; 
    std::vector<cv::Point2f> plane_points_2d;
    std::vector<cv::Point2f> plane_points_flited;
    Eigen::Vector3d key_center;     // 体中心点
    // 调试相关
    cv::Mat depth_show;
    cv::Mat debug_image;
    std::string state = "off";

    // 功能函数 计算偏差角度， 确保plane_info已被写入信息
    double calc_deviation_angle(const Eigen::Vector3d& target_plane);
};      // class plane_fit

inline bool plane_fit::process(Ten::camera_frame& frame)
{
    if (frame.bgr_image.empty() || frame.depth_image.empty())
    {
        state = "frame.bgr_image.empty() || frame.depth_image.empty()";
        return false;
    }

    // 设置点云
    cv::Rect roi = SET_PCL_.set_roi_detect(frame.bgr_image);
    bool is_set = SET_PCL_.set_Pcl_Cloud(frame.depth_image, color_intr_, roi,input_cloud);
    if (!is_set) 
    {
        state = "set_Pcl_Cloud error!";
        return false;
    }

    // 滤波器
    bool is_filtted = PRE_PCL_.cloud_filter(input_cloud,filter_cloud);      // 设置点云
    if (!is_filtted)
    {
        state = "cloud_filter error!";
        return false;
    }
    // 提取平面和中心点，法向量
    bool is_plane_flitted = PRE_PCL_.Plane_fitter(filter_cloud, plane_cloud, plane_info);
    if (!is_plane_flitted)
    {
        state = "Plane_fitter error!";
        return false;
    }

    // 方形拟合
    POST_PCL_.compute_Center(plane_cloud,plane_info);
    POST_PCL_.set_vector_2d(plane_cloud,plane_info,plane_points_2d);
    POST_PCL_.central_range_filter(plane_points_2d,plane_points_flited);
    bool is_shape_ok = POST_PCL_.shape_filter(plane_points_flited);
    if (!is_shape_ok)
    {
        state = "shape_filter error!";
        return false;
    }
    POST_PCL_.set_yaw(plane_info, plane_points_flited);
    POST_PCL_.vector2f_to_pcl(plane_points_flited,plane_info,plane_cloud_vec);
    POST_PCL_.compute_Center(plane_cloud_vec,plane_info,false);
    // 赋值体中心点
    key_center = POST_PCL_.cal_center_point(plane_info);
    state = "ok";
    return true;
}

inline void plane_fit::publish(Ten::camera_frame& frame)
{
    if (state == "ok")
    {
        cv::normalize(frame.depth_image, depth_show, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        DEBUG_PCL_.set_debug_plane_quadrilateral(frame.bgr_image,plane_info, color_intr_,debug_image);
        DEBUG_PCL_.pub_depth_image(frame.depth_image, "depth_show");
        DEBUG_PCL_.pub_color_image(frame.bgr_image, "bgr_image");
        DEBUG_PCL_.pub_color_image(debug_image, "debug_image");
        DEBUG_PCL_.publish_PlaneTF(plane_info);
        DEBUG_PCL_.publish_pointcloud(plane_cloud_vec);
    }
}

inline double plane_fit::calc_deviation_angle(const Eigen::Vector3d& target_plane)
{
    Eigen::Vector3d normal = plane_info.plane_normal;
    normal.normalize();

    // 计算向量点积
    double dot_product = normal.dot(target_plane);
    // 修复浮点误差，限制在 [-1, 1] 避免 acos 出现 NaN
    dot_product = std::max(-1.0, std::min(1.0, dot_product));

    // 原始夹角（弧度，范围 0 ~ M_PI）
    double rad_angle = std::acos(dot_product);

    // 1. 取最小夹角，约束到 0 ~ M_PI/2（对应 0° ~ 90°）
    rad_angle = std::min(rad_angle, M_PI - rad_angle);

    // 2. 大于 45°(M_PI/4) 则使用 90°(M_PI/2) - 当前弧度
    if (rad_angle > M_PI / 4)
    {
        rad_angle = M_PI / 2 - rad_angle;
    }

    return rad_angle;
}

inline result plane_fit::set_result(
        const Eigen::Vector3d& target_plane,        
        const Eigen::Vector3d& line_point, 
        const Eigen::Vector3d& line_dir)
    {
        result out;
        Eigen::Vector3d target_plane_normal = target_plane.normalized();
        Eigen::Vector3d target_line_dir = line_dir.normalized();
        // 点垂直投影到目标平面
        Eigen::Vector3d proj_center = key_center - target_plane_normal.dot(key_center) * target_plane_normal;

        // 计算投影点到目标直线的距离
        Eigen::Vector3d vec = proj_center - line_point;
        out.bias = vec.cross(target_line_dir).norm();

        // 点到目标平面的垂直距离
        double plane_dist = (key_center - line_point).dot(target_plane_normal);
        out.distance = std::fabs(plane_dist);

        out.deviation_angle = calc_deviation_angle(target_plane);
        return out;
    }

}       // namespace Plane_FitLocator
}       // namespace Ten