#ifndef __POSE_RESOLVING_H_
#define __POSE_RESOLVING_H_
#include "./pre_pcl.h"
#include "./set_pcl.h"
#include "./post_pcl.h"
#include "./../yolo/yolo_26obb.h"
#include "./../camera.h"

namespace Ten
{
    namespace Plane_FitLocator
    {
        #define _model_path_ "/home/maple/study3/li/best_openvino_model_op13/model"
        #define _xpu_ "cpu"

        class pose_resolving
        {
        public:
            /**
             * @brief 初始化函数
             * @param color_intr: 相机参数
             */
            pose_resolving(rs2_intrinsics color_intr)
                :yobb_(_model_path_, _xpu_)
            {
                color_intr_ = color_intr;
            }

            /**
             * @brief 处理图片，返回位姿
             * @param cf: rgb图片和深度图
             * @param debug_cloud: 返回的结果
             * @return bool:是否处理成功
             */
            bool process(Ten::camera_frame cf, pcl::PointCloud<pcl::PointXYZ>::Ptr debug_cloud = nullptr)
            {
                Ten::yolo::Detection det;
                // 检测矩形框
                if(!run_yolo(cf.bgr_image, det))
                {
                    return false;
                }
                // 提取其中点云
                pcl::PointCloud<pcl::PointXYZ>::Ptr raw_cloud(new pcl::PointCloud<pcl::PointXYZ>);
                if(!set_pcl_.set_Pcl_Cloud(cf.raw_depth_frame, color_intr_, det, raw_cloud))
                {
                    return false;
                }

                // 滤波器
                pcl::PointCloud<pcl::PointXYZ>::Ptr filter_cloud(new pcl::PointCloud<pcl::PointXYZ>);
                if(!pre_pcl_.cloud_filter(raw_cloud, filter_cloud))
                {
                    return false;
                }

                // 提取平面点云
                pcl::PointCloud<pcl::PointXYZ>::Ptr plane_cloud(new pcl::PointCloud<pcl::PointXYZ>);
                if(!pre_pcl_.Plane_fitter(filter_cloud, plane_cloud, plane_info))
                {
                    return false;
                }

                // 转2d，提取中心点法向量， 填充plane_info
                pcl::PointCloud<pcl::PointXYZ>::Ptr plane_2d_cloud(new pcl::PointCloud<pcl::PointXYZ>);
                post_pcl_.compute_CenterAndNormal(plane_cloud,plane_info);
                post_pcl_.set_2d_cloud(plane_cloud,plane_info, plane_2d_cloud);
                post_pcl_.fit_PlaneSquare(plane_2d_cloud,plane_info);
                post_pcl_.set_plane_euler(plane_info);
                return true;
            }

            Plane_Info get_plane_info() const
            {
                return plane_info;
            }

        private:
        Ten::yolo::yolo_26obb yobb_;
        rs2_intrinsics color_intr_;
        Ten_set_pcl set_pcl_;
        Ten_pre_pcl pre_pcl_;
        Ten_post_pcl post_pcl_;
        Plane_Info plane_info;

            /**
             * @brief 返回最有检测结果
             * @param image: 输入图片
             * @param result: 输出结果
             * @return bool:处理是否成功
             */
            bool run_yolo(cv::Mat image, Ten::yolo::Detection& result)
            {
                std::vector<Ten::yolo::Detection> results = yobb_.worker(image);
                //调试，可不用
                drawDetections(image, results);
                if(results.size() == 0)
                {
                    return false;
                }
                result = results[0];
                return true;
            }
        };


    }       
}       
#endif 