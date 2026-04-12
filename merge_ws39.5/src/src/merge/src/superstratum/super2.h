#ifndef __SUPPER2_H_
#define __SUPPER2_H_

#include"super.h"
#include"./../yolo/yolo_han2.h"
#include <iomanip>      // 打印限制浮点数

namespace Ten
{
    namespace superstratum
    {
        // 用于存储多次在不同位置的 set_img 的记录
        struct super_init{

            std::vector<std::vector<box>> time_box_lists_;
            std::vector<std::vector<float>> time_ps_w_;                 // 有效点的权重， 由该位置 point_size / 该行最大像素值 计算得到

            std::vector<float> set_ps_w(
                std::vector<box>& box_lists
            )
            {
                std::vector<float>ps_w(12,0.00f);
                if (box_lists.size() != 12)
                {
                    std::cout << "[Error]from func set_ps_w: box_lists.size() != 12, make sure box_lists is set" << std::endl;
                    return ps_w;
                }
                // 找到每行的最大值
                std::vector<int>max_point_size = {0,0,0,0};
                for(int i = 0;i < 4; i++)
                {
                    for (int j = 0;j < 3;j++)
                    {
                        if (box_lists[i * 3 + j].point_size > max_point_size[i])
                        {
                            max_point_size[i] = box_lists[i * 3 + j].point_size;
                        }
                    }
                }
                // 填充 point_size_weight 字段
                for(int i = 0;i < 4; i++)
                {
                    for (int j = 0;j < 3;j++)
                    {
                        ps_w[i * 3 + j] = (float)box_lists[i * 3 + j].point_size / max_point_size[i];
                    }
                }
                return ps_w;
            }

        };


        class supper2
        {
        public:
            supper2()
            {
                //设置稳态误差
                Ten::XYZRPY xyzrpy_error;
                xyzrpy_error._xyz._x = 0;
                xyzrpy_error._xyz._y = 0;
                xyzrpy_error._xyz._z = 0;
                xyzrpy_error._rpy._roll = 0;
                xyzrpy_error._rpy._pitch = 0;
                xyzrpy_error._rpy._yaw = 0;
                //coordinate_transformation_.set_stead_state_error(xyzrpy_error);
                _CAMERA_TRANSFORMATION_.set_error(xyzrpy_error);
                lidar_to_camera_transform_matrix_ << 
                -0.0293067,  -0.999359,  -0.0205564,  0.0519757,  
                0.0195515,  0.0199882,  -0.999609,  0.47424,  
                0.999379,  -0.0296971,  0.0189532,  0.336381,  
                0.0         ,  0.0        ,  0.0        ,  1.0; 
                _CAMERA_TRANSFORMATION_.camerainfo_.set_Extrinsic_Matrix(lidar_to_camera_transform_matrix_);
               
            }

            /**
             * @brief 输入batch_images
             * @param batch_images 当前批次的两张全局图像和r,t
             * @param model_path 用于初始化模型
             * @param place 待写入的位置信息列表， 内部会初始化
             * @param per_loss 每个位置的损失
             * @param device 推理设备， 默认为cpu
             * @param min_ps_w 能容忍的最小point_size_weight 权重大小， 默认为0.2
             * @param is_print 是否打印的标志位
            */
            void manage_roi12(
                const std::vector<Ten::ORB::debug_orb_exhaust_element>& batch_images,
                const std::string model_path,
                std::vector<int>& place,
                std::vector<float>& per_loss,
                const std::string device = "cpu",
                const float min_ps_w = 0.2,
                const bool is_print = false
            )
            {
                Ten::superstratum::supper2 supper2_;
                std::vector<int> place_1(12,-1);
                std::vector<int> place_2(12,-1);
                place.assign(12,-1);
                std::vector<Ten::yolo::han2> det_1;
                std::vector<Ten::yolo::han2> det_2;
                std::vector<float>per_loss_1;
                std::vector<float>per_loss_2;
                per_loss.assign(12,-1.0f);
                float sure_loss_1;
                float sure_loss_2;
                float sure_loss;

                // 设置图片
                supper2_.set_img(batch_images);

                // 进行推理处理
                supper2_.process_img(model_path,det_1,device,min_ps_w);
                supper2_.postprocess_dets(det_1,place_1,per_loss_1,sure_loss_1);

                // retry
                if (sure_loss_1 > 0.06)
                {
                    int exist[12];  
                    for(int i = 0; i < 12; i++) {
                        exist[i] = place_1[i];
                    }

                    // 设置图片
                    supper2_.set_img(batch_images,exist);

                    // 进行推理处理
                    supper2_.process_img(model_path,det_2,device,min_ps_w);
                    supper2_.postprocess_dets(det_2,place_2,per_loss_2,sure_loss_2);
                    
                    // 综合处理这两次的数据， 返回place,per_loss,sure_loss
                    postprocess_retry_dets(place_1,place_2,det_1,det_2,per_loss_1,per_loss_2,sure_loss_1,sure_loss_2,place,per_loss,sure_loss);
                }
                else    // 没有第二次尝试， 直接使用第一次的结果
                {
                    per_loss = per_loss_1;
                    place = place_1;
                }

                // 打印和调试部分
                if (is_print)
                {
                    manage_roi12_print(place_1,place_2,det_1,det_2,sure_loss_1,sure_loss_2,place,per_loss,sure_loss);
                }
            }

            super_init super_init_;
        private:
            Eigen::Matrix4d lidar_to_camera_transform_matrix_ = Eigen::Matrix4d::Identity(); //雷达到相机
            Ten::Ten_worldtocamera camera_transformation_; //坐标点转换器，用于将世界坐标系下的点变换到当前坐标系，以及像素坐标系
            Ten::Ten_occlusion_handing zbuffer_; //zb处理器，用于处理遮挡关系

            /**
             * @brief 输入图像和r,t信息， 设置12个roi图像, 填充当前的box_lists_, 并保存到 time_box_lists_, 用于输入的 筛选
             * @param batch_images 一定批次的全局图像
             * @param exist_boxes int12数组, 表示存在的列表
            */
            void set_img(
                const std::vector<Ten::ORB::debug_orb_exhaust_element>& batch_images,
                int *exist_boxes = nullptr)
            {
                for(size_t j = 0; j < batch_images.size(); j++)
                {
                    //世界点和box_list的类对象，对里面数据进行处理
                    Ten::init_3d_box world_point;
                    camera_transformation_.camerainfo_.set_RT(batch_images[j].oee.rvec_, batch_images[j].oee.tvec_);
                    camera_transformation_.pcl_transform_world_to_camera(world_point.pcl_LM_plum_object_points_, world_point.pcl_C_plum_object_points_, world_point.object_plum_2d_points_);
                    world_point.pcl_to_C();

                    if (j == 0)
                    {
                        super_init_.time_box_lists_.clear();
                        super_init_.time_ps_w_.clear();
                    }
                    if (exist_boxes == nullptr) 
                    {
                        int exist_boxes2[12] = {1,1,1,1,1,1,1,1,1,1,1,1};
                        exist_boxes = exist_boxes2;
                    }
                    zbuffer_.set_exist_boxes(exist_boxes);
                    zbuffer_.set_box_lists_(batch_images[j].oee.image_, world_point.C_object_plum_points_, world_point.object_plum_2d_points_, world_point.box_lists_);

                    super_init_.time_ps_w_.push_back(super_init_.set_ps_w(world_point.box_lists_));
                    super_init_.time_box_lists_.push_back(world_point.box_lists_);
                }
            }

            /**
             * @brief 输入多组box_lists， 写入一组当前最优(结合ps_w筛选)的图像，用于输入的 预处理
             * @param model_path 用于初始化模型
             * @param det        待写入的检测结果
             * @param device    推理设备， 默认为cpu
             * @param min_ps_w 能容忍的最小point_size_weight 权重大小， 默认为0.1
            */
            void process_img(
                const std::string model_path,
                std::vector<Ten::yolo::han2>& det,
                const std::string device = "cpu",
                const float min_ps_w = 0.2f
            )
            {
                Ten::yolo::yolo_han2 yolo_han2_(model_path,device);
                std::vector<cv::Mat>roi_images(12, cv::Mat::zeros(64, 64, CV_8UC3));

                std::vector<int> is_filled(12, 0);     // 表示是否填充图像列表的标志位
                // 表示各位置最大ps_w
                std::vector<float> max_ps_w(12, min_ps_w);     
                
                if (super_init_.time_ps_w_.size() != super_init_.time_box_lists_.size()) {
                    std::cout << "[Error] time_ps_w.size() != time_box_lists.size()" << std::endl;
                    return;
                }

                for (int time = 0; time < super_init_.time_box_lists_.size(); time++)
                {
                    std::vector<box>& box_lists = super_init_.time_box_lists_[time];
                    if (box_lists.size() != 12 || super_init_.time_ps_w_[time].size() != 12)
                    {
                        std::cout << "[Error]from func supper2::process_img: box_lists.size() != 12 || time_ps_w[time].size() != 12" << std::endl;
                        return;
                    }
                    for (int place = 0; place < 12; place++)
                    {
                        if (super_init_.time_ps_w_[time][place] > min_ps_w && super_init_.time_ps_w_[time][place] >= max_ps_w[place])
                        {
                            roi_images[place] = box_lists[place].roi_image.clone();
                            is_filled[place] = 1;
                            max_ps_w[place] = super_init_.time_ps_w_[time][place];
                        }
                    }
                }

                bool full_filled = true;
                std::vector<int> isnt_filled_idx;
                for (int i = 0;i < is_filled.size(); i++)
                {
                    if (!is_filled[i])
                    {
                        full_filled = false;
                        isnt_filled_idx.push_back(i + 1);
                    }
                }
                if (!full_filled)
                {
                    std::cout << "[Error]from func supper2::process_img: isn't full_filled" << std::endl;
                    std::cout << "isnt_filled: ";
                    for (int i = 0; i < isnt_filled_idx.size();i++)
                    {
                        std::cout << isnt_filled_idx[i] << " ";
                    }
                    std::cout << std::endl;
                }
                det = yolo_han2_.worker(roi_images);
            }

            void postprocess_dets(
                std::vector<Ten::yolo::han2>& dets,
                std::vector<int>& sure,
                std::vector<float>& per_loss,
                float& sure_loss
            )
            {
                sure_loss = 0.0f;
                int sure_1 = 0;
                per_loss.assign(12,0.0f);    // 衡量可信程度， loss在【0，1】，越小越好
                sure.assign(12,-1);                 // 基本可信的列表

                for (int place = 0; place < dets.size(); place++)
                {
                    // 填充 loss 列表
                    if (dets[place].valid_exist_ > 0.5)
                    {
                        per_loss[place] = 2 * (1 - dets[place].valid_exist_);
                    }
                    else
                    {
                        per_loss[place] = 2 * dets[place].valid_exist_;
                    }
                    // 填充 sure 列表
                    if (per_loss[place] < 0.4)
                    {
                        sure[place] = (dets[place].valid_exist_ > 0.5)? 1 : -1;
                        if (dets[place].valid_exist_ > 0.5)
                        {
                            sure_loss += per_loss[place];
                            sure_1 += 1;
                        }
                    }
                }
                while (true)
                {
                    if (sure_1 >= 8)
                    {
                        break;
                    }
                    // 找到当前的最大1类阈值
                    int max_place = -1;
                    float max_1_conf = 0.0f;
                    for (int place = 0; place < dets.size(); place++)
                    {
                        if (sure[place] == 1) continue;
                        if (dets[place].valid_exist_ > max_1_conf)
                        {
                            max_1_conf = dets[place].valid_exist_;
                            max_place = place;
                        }
                    }
                    // 把目前认为最有可能为1 类的填充进sure
                    if (max_place != -1)
                    {
                        sure[max_place] = 1;
                        sure_1 += 1;
                        sure_loss += per_loss[max_place];
                    } 
                }

                // 赋0类
                for (int place = 0; place < sure.size(); place++)
                {
                    if (sure[place] != 1)
                    {
                        sure[place] = 0;
                    }
                }
                sure_loss = sure_loss / 8;
            }

            void postprocess_retry_dets(
                const std::vector<int>& place_1,
                const std::vector<int>& place_2,
                const std::vector<Ten::yolo::han2>& det_1,
                const std::vector<Ten::yolo::han2>& det_2,
                const std::vector<float>& per_loss_1,
                const std::vector<float>& per_loss_2,
                const float& sure_loss_1,
                const float& sure_loss_2,
                std::vector<int>& place,
                std::vector<float>& per_loss,
                float& sure_loss
            )
            {
                place.assign(12,-1);
                sure_loss = 0.0f;
                int set_count = 0;

                for (int i = 0;i < 12;i ++)
                {
                    if (per_loss_1[i] < per_loss_2[i])
                    {
                        per_loss[i] = per_loss_1[i];
                        sure_loss += per_loss_1[i];
                        place[i] = place_1[i];
                    }
                    else
                    {
                        per_loss[i] = per_loss_2[i];
                        sure_loss += per_loss_2[i];
                        place[i] = place_2[i];
                    }
                }
                sure_loss = sure_loss / 12;
            }

            void manage_roi12_print(
                const std::vector<int>& place_1,
                const std::vector<int>& place_2,
                const std::vector<Ten::yolo::han2>& det_1,
                const std::vector<Ten::yolo::han2>& det_2,
                const float& sure_loss_1,
                const float& sure_loss_2,
                const std::vector<int>& place,
                const std::vector<float>& per_loss,
                const float& sure_loss
            )
            {
                std::cout << "-----------first--------------" << std::endl;
                // 各个位置置信度
                for(auto e : det_1)
                {
                    std::cout << "exist: " << e.valid_exist_ << ", empty: " << e.valid_empty_ << std::endl;
                } 
                // 各个位置后处理后给的标签
                std::cout << "place_1: ";
                for(auto& e : place_1)
                {
                    std::cout << e << " ";
                }
                std::cout << std::endl;
                std::cout << "sure_loss: " << sure_loss_1 << std::endl;  

                if (sure_loss_1 > 0.06)
                {
                    std::cout << "-----------retry--------------" << std::endl;
                    // 各个位置置信度
                    for(auto e : det_2)
                    {
                        std::cout << "exist: " << e.valid_exist_ << ", empty: " << e.valid_empty_ << std::endl;
                    } 
                    // 各个位置后处理后给的标签
                    std::cout << "place_2: ";
                    for(auto& e : place_2)
                    {
                        std::cout << e << " ";
                    }
                    std::cout << std::endl;
                    std::cout << "retry: sure_loss: " << sure_loss_2 << std::endl;  
                }
                // 最终的结果
                std::cout << "-----------result--------------" << std::endl;
                std::cout << "place: ";
                for(auto& e : place)
                {
                    std::cout << std::fixed << std::setprecision(3) << e << " ";
                }
                std::cout << std::endl;
                std::cout << "per_loss: ";
                for(auto& e : per_loss)
                {
                    std::cout << e << " ";
                }
                std::cout << std::endl;
                std::cout << "sure_loss: " << sure_loss << std::endl;

            }
        };
        
    }


}


#endif
