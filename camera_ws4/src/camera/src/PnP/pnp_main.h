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

#define _camera_x_bias_ 0
#define _camera_y_bias_ 0
#define _camera_z_bias_ 0

#define max_bias 0.2    // 约束 -max_bias 到 max_bias
namespace Ten
{
namespace KFS
{
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
        center_bias = 0.0;
    }

    double processOneFrame(const cv::Mat color_frame,const cv::Mat depth_frame, bool debug_mode = false)
    {
        kfsPnpOutput out = solver_.process(color_frame,depth_frame);

        if (out.status != "ok")
        {
            std::cout << "status: " << out.status << std::endl;
        }

        // debug 部分
        if (debug_mode)
        {
            // pnp_debug.draw(color_frame, out, color_intr_);
            // char key = cv::waitKey(1);
            // if (key == 27) return 0;
            pnp_debug.publish_pointcloud(out.cloudFiltered);  
            pnp_debug.publishPnpDebugImage(color_frame, out, color_intr_);
            current_roi_ = out.roi;
        }

        // 设置偏差值
        set_camera_bias(out);
        center_bias = double(-out.center.y());

        if (-max_bias < center_bias && center_bias < max_bias)
        {
            center_bias = 0;
        }

        return center_bias;
    }

    cv::Rect getCurrentRoi__() const {
        return current_roi_;
    }
    
private:
    kfsPnpSolver solver_;
    cv::Mat cameraMatrix_;
    cv::Mat distCoeffs_;
    Ten::KFS::DebugDrawer pnp_debug;

    cv::Rect current_roi_;
    rs2_intrinsics color_intr_;

    double center_bias;

    void set_camera_bias (kfsPnpOutput& out)
    {
        out.center.x() = out.center.x() + _camera_x_bias_;
        out.center.y() = out.center.y() + _camera_y_bias_;
        out.center.z() = out.center.z() + _camera_z_bias_;
    }
};

} // namespace KFS
} // namespace Ten
#endif