#include "set_plane.h"
#include "set_detect.h"
#include "set_plane.h"
#include "debug_pcl.h"
#include "../camera.h"

namespace Ten
{
namespace kfs_locator
{

struct result
{
    double deviation_angle = 0.0;     // 目标面的法向量到预设面的法向量的角度
    double bias = 0.0;                // 体中心点到预设面上预设直线的投影偏差距离
    double distance = 0.0;            // 体中心点到预设面的距离
};


class Ten_set_result
{
public:
    Ten_set_result(rs2_intrinsics color_intr)
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

    // 执行流程 ---------------------------------------------------------------
    // 1 设置初始点云列表
    bool preprocess(Ten::camera_frame& frame);

    // 2 get_input_clouds()

    // 3 核心提取面函数
    bool postprocess(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud_ptr = nullptr);

    // 4 设置结果
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
    // 执行流程 ---------------------------------------------------------------

    // 调试： 发布调试图像和tf， 点云
    void publish(Ten::camera_frame& frame);

    // 取接口
    const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> get_input_clouds()
    {
        return  input_clouds;
    }
    const Ten::kfs_locator::Plane_Info get_plane_info() 
    {
        return plane_info;
    }
    const std::string get_state() 
    {
        return state;
    }
    // 写接口
    void set_input_cloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cl)
    {
        input_cloud = input_cl;
    }

private:
    // 参数
    rs2_intrinsics color_intr_;
    // 处理器
    Ten::kfs_locator::Ten_debug_pcl DEBUG_PCL_;
    Ten::kfs_locator::Ten_set_detect SET_DETECT_;
    Ten::kfs_locator::Ten_set_plane SET_PLANE_;
    // 中间变量
    Ten::kfs_locator::Plane_Info plane_info;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> input_clouds;
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
};      // class Ten_set_result

inline bool Ten_set_result::preprocess(Ten::camera_frame& frame)
{
    if (frame.bgr_image.empty() || frame.depth_image.empty())
    {
        state = "frame.bgr_image.empty() || frame.depth_image.empty()";
        return false;
    }

    // 设置点云
    std::vector<cv::Rect> rois = SET_DETECT_.set_roi_detect(frame.bgr_image);
    bool is_set = SET_DETECT_.set_pcl_clouds(frame.depth_image, color_intr_, rois, input_clouds);
    if (!is_set) 
    {
        state = "set_pcl_clouds error!";
        return false;
    }
    return true;
}

inline bool Ten_set_result::postprocess(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud_ptr)
{
    if (input_cloud_ptr)
    {
        input_cloud = input_cloud_ptr;
    }
    else
    {
        if (input_clouds.empty())
        {
            state = "postprocess: input_clouds is empty";
            return false;
        }

        const auto& front_cloud = input_clouds.front();
        if (!front_cloud)
        {
            state = "postprocess: input_clouds front cloud is nullptr";
            return false;
        }
        input_cloud = front_cloud;
    }

    // 滤波器
    bool is_filtted = SET_PLANE_.cloud_filter(input_cloud,filter_cloud);      // 设置点云
    if (!is_filtted)
    {
        state = "cloud_filter error!";
        return false;
    }
    // 提取平面和中心点，法向量
    bool is_plane_flitted = SET_PLANE_.Plane_fitter(filter_cloud, plane_cloud, plane_info);
    if (!is_plane_flitted)
    {
        state = "Plane_fitter error!";
        return false;
    }

    // 方形拟合
    SET_PLANE_.compute_Center(plane_cloud,plane_info);
    SET_PLANE_.set_vector_2d(plane_cloud,plane_info,plane_points_2d);
    SET_PLANE_.central_range_filter(plane_points_2d,plane_points_flited);
    bool is_shape_ok = SET_PLANE_.shape_filter(plane_points_flited);
    if (!is_shape_ok)
    {
        state = "shape_filter error!";
        return false;
    }
    SET_PLANE_.set_yaw(plane_info, plane_points_flited);
    SET_PLANE_.vector2f_to_pcl(plane_points_flited,plane_info,plane_cloud_vec);
    SET_PLANE_.compute_Center(plane_cloud_vec,plane_info,false);
    // 赋值体中心点
    key_center = SET_PLANE_.cal_center_point(plane_info);
    state = "ok";
    return true;
}

inline void Ten_set_result::publish(Ten::camera_frame& frame)
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

inline double Ten_set_result::calc_deviation_angle(const Eigen::Vector3d& target_plane)
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

inline result Ten_set_result::set_result(
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

}       // namespace kfs_locator
}       // namespace Ten