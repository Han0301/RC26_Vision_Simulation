#ifndef __PNP_MAIN_H_
#define __PNP_MAIN_H_

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include "../camera.h"
#include "pnp_func.h"
#include "../method_math.h"
#include "pnp_debug.h"

namespace Ten
{
namespace KFS
{

// 相机坐标系 → 世界坐标系 
static const Eigen::Matrix3f kCamToWorld = []()
{
    Eigen::Matrix3f R;
    R << 0.0f, 0.0f, 1.0f,
         -1.0f, 0.0f, 0.0f,
         0.0f, -1.0f, 0.0f;
    return R;
}();
static const Eigen::Quaternionf kCamToWorldQ(kCamToWorld);

class kfsLocator
{
public:
    explicit kfsLocator(const rs2_intrinsics& color_intr)
        : solver_(kfsPnpConfig(), color_intr),
          color_intr_(color_intr)
    {
        cameraMatrix_ = (cv::Mat_<double>(3,3) <<
            color_intr.fx, 0, color_intr.ppx,
            0, color_intr.fy, color_intr.ppy,
            0, 0, 1);
        distCoeffs_ = cv::Mat::zeros(5,1,CV_64F);
        pnp_debug.init();  
    }

    void processOneFrame(const cv::Mat color,const cv::Mat depth_frame)
    {
        kfsPnpOutput out = solver_.process(color,color_intr_,depth_frame);

        set_lastest_center(out);
        Ten::XYZRPY center = get_lastest_center();

        if (out.status != "ok")
        {
            std::cout << "status: " << out.status << std::endl;
        }

        // debug 部分
        pnp_debug.draw(color, out, color_intr_);
        char key = cv::waitKey(1);
        if (key == 27) return;
        pnp_debug.publish_pointcloud(out.cloudFiltered);  

        std::cout << "center_bias: " << out.center.x() << std::endl;
        // std::cout << "解算结果：" << center._xyz._x << " " 
        //     << center._xyz._y << " " 
        //     << center._xyz._z << std::endl;
    }



    // 取到中心点位姿
    Ten::XYZRPY get_lastest_center() const
    {
        return lastest_center;
    }

    void setWorldBias(double x, double y, double z)
    {
        world_bias_x_ = x;
        world_bias_y_ = y;
        world_bias_z_ = z;
    }

private:
    kfsPnpSolver solver_;
    rs2_intrinsics color_intr_;
    cv::Mat cameraMatrix_;
    cv::Mat distCoeffs_;
    Ten::KFS::DebugDrawer pnp_debug;

    Ten::XYZRPY lastest_center;

    double world_bias_x_ = 0.0;
    double world_bias_y_ = 0.08995 -0.0175 -0.015;
    double world_bias_z_ = 0.07470 - 0.0125;

    void set_lastest_center(const kfsPnpOutput& input)
    {
        // 坐标转换
        Eigen::Vector3f worldCenter = kCamToWorld * input.center;
        Eigen::Quaternionf worldOrient = kCamToWorldQ * input.orientation;

        lastest_center._xyz._x = static_cast<double>(worldCenter.x()) + world_bias_x_;
        lastest_center._xyz._y = static_cast<double>(worldCenter.y()) + world_bias_y_;
        lastest_center._xyz._z = static_cast<double>(worldCenter.z()) + world_bias_z_;
        lastest_center._rpy._yaw = static_cast<double>(
            std::atan2(2.0 * (worldOrient.w() * worldOrient.z() + worldOrient.x() * worldOrient.y()),
                    1.0 - 2.0 * (worldOrient.y() * worldOrient.y() + worldOrient.z() * worldOrient.z())));
    }
};

} // namespace KFS
} // namespace Ten
#endif