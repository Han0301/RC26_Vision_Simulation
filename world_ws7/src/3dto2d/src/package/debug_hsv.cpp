#ifndef __DEBUG_HSV_CPP_
#define __DEBUG_HSV_CPP_

#include "debug_hsv.h"

namespace Ten{
    
void Ten_debug_hsv::save_hsv_hist_visualization(const std::vector<cv::Mat>& hsv_hist, 
                                 const std::string& save_path,
                                 int h_bin_num,  // H通道独立bin数
                                 int s_bin_num,  // S通道独立bin数
                                 int v_bin_num,  // V通道独立bin数
                                 int canvas_height) {
    // -------------------------- 1. 输入合法性校验 --------------------------
    if (hsv_hist.size() != 3) {
        std::cerr << "[ERROR] 输入直方图容器长度必须为3（H/S/V三通道），当前长度：" << hsv_hist.size() << std::endl;
        return;
    }
    if (save_path.empty()) {
        std::cerr << "[ERROR] 保存路径不能为空！" << std::endl;
        return;
    }
    // 校验bin数合法性
    if (h_bin_num <= 0 || s_bin_num <=0 || v_bin_num <=0) {
        std::cerr << "[ERROR] 分箱数必须为正整数！当前：H=" << h_bin_num << ", S=" << s_bin_num << ", V=" << v_bin_num << std::endl;
        return;
    }

    // -------------------------- 2. 定义绘图参数 --------------------------
    const int CHANNEL_NUM = 3;               // 通道数（H/S/V）
    const int CHANNEL_WIDTH = 640;           // 每个通道的绘图宽度
    const int CANVAS_W = CHANNEL_NUM * CHANNEL_WIDTH;  // 整体画布宽度（750）
    const int CANVAS_H = canvas_height;      // 整体画布高度（默认300）
    const cv::Scalar COLOR_H = cv::Scalar(0, 0, 255);    // H通道绘图颜色（红色）
    const cv::Scalar COLOR_S = cv::Scalar(0, 255, 0);    // S通道绘图颜色（绿色）
    const cv::Scalar COLOR_V = cv::Scalar(255, 0, 0);    // V通道绘图颜色（蓝色）
    const cv::Scalar COLOR_TEXT = cv::Scalar(0, 0, 0);   // 文字颜色（黑色）
    const cv::Scalar COLOR_BG = cv::Scalar(255, 255, 255);// 背景色（白色）
    const cv::Scalar COLOR_BORDER = cv::Scalar(180, 180, 180); // 通道分隔线颜色

    // 创建整体画布
    cv::Mat canvas(CANVAS_H, CANVAS_W, CV_8UC3, COLOR_BG);

    // -------------------------- 3. 定义各通道基础信息 --------------------------
    // 通道配置：bin数、绘图颜色、通道名称、直方图数据
    struct ChannelConfig {
        int bin_num;
        cv::Scalar color;
        std::string name;
        const cv::Mat& hist;
    };
    std::vector<ChannelConfig> channel_configs = {
        {h_bin_num, COLOR_H, "H Channel (bin=" + std::to_string(h_bin_num) + ")", hsv_hist[0]},
        {s_bin_num, COLOR_S, "S Channel (bin=" + std::to_string(s_bin_num) + ")", hsv_hist[1]},
        {v_bin_num, COLOR_V, "V Channel (bin=" + std::to_string(v_bin_num) + ")", hsv_hist[2]}
    };

    // -------------------------- 4. 遍历3个通道绘制直方图 --------------------------
    for (int ch_idx = 0; ch_idx < CHANNEL_NUM; ch_idx++) {
        // 4.1 计算当前通道的绘图区域
        int x_start = ch_idx * CHANNEL_WIDTH;  // 通道左上角X坐标
        int x_end = (ch_idx + 1) * CHANNEL_WIDTH; // 通道右下角X坐标
        int draw_height = CANVAS_H - 50;      // 绘图高度（预留文字空间）
        int bin_width = CHANNEL_WIDTH / channel_configs[ch_idx].bin_num; // 每个bin的宽度

        // 4.2 获取当前通道配置
        const auto& config = channel_configs[ch_idx];
        const cv::Mat& hist = config.hist;

        // 4.3 绘制通道名称
        cv::putText(canvas, config.name, 
                    cv::Point(x_start + 10, 30), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1);

        // 4.4 绘制通道分隔线
        cv::line(canvas, cv::Point(x_start, 0), cv::Point(x_start, CANVAS_H), COLOR_BORDER, 1);



        // 4.5 绘制直方图（空/无效直方图则绘制警告框）
        if (!hist.empty() && hist.channels() == 1) {
        // 归一化直方图到绘图高度（0 ~ draw_height）
        cv::Mat hist_float;
        hist.convertTo(hist_float, CV_32F); // 转成float型
        cv::Mat hist_norm;
        cv::normalize(hist_float, hist_norm, 0, draw_height, cv::NORM_MINMAX);

            // 遍历每个bin绘制矩形
            for (int bin = 0; bin < config.bin_num; bin++) {
                // 跳过宽度为0的bin（避免绘图异常）
                if (bin_width <= 0) break;

                // 计算bin的像素值和绘制位置（从下往上绘制）
                float bin_val = hist_norm.at<float>(0,bin);
                int bin_val_int = cvRound(bin_val);
                int bin_x_start = x_start + bin * bin_width;
                int bin_y_start = CANVAS_H - 20 - bin_val_int;  // 底部预留20像素
                int bin_x_end = bin_x_start + bin_width - 1;
                int bin_y_end = CANVAS_H - 20;

                // 绘制bin矩形（填充）
                cv::rectangle(canvas,
                              cv::Point(bin_x_start, bin_y_start),
                              cv::Point(bin_x_end, bin_y_end),
                              config.color, -1);
                // 绘制bin边框（增强可读性）
                cv::rectangle(canvas,
                              cv::Point(bin_x_start, bin_y_start),
                              cv::Point(bin_x_end, bin_y_end),
                              cv::Scalar(0,0,0), 0.5);
            }
        } else {
            // 直方图为空/无效，绘制警告框
            cv::rectangle(canvas,
                          cv::Point(x_start + 20, 50),
                          cv::Point(x_end - 20, CANVAS_H - 20),
                          config.color, 2);
            cv::putText(canvas, "INVALID HIST", 
                        cv::Point(x_start + 40, CANVAS_H/2), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, config.color, 2);
        }
    }

    // 绘制最后一条分隔线
    cv::line(canvas, cv::Point(CANVAS_W-1, 0), cv::Point(CANVAS_W-1, CANVAS_H), COLOR_BORDER, 1);

    // -------------------------- 5. 保存图像 --------------------------
    if (!cv::imwrite(save_path, canvas)) {
        std::cerr << "[ERROR] 保存图像失败！路径：" << save_path << std::endl;
    } else {
        //std::cout << "[INFO] HSV直方图可视化图像已保存至：" << save_path << std::endl;
    }
}

std::vector<cv::Mat> Ten_debug_hsv::set_hsv_hist(cv::Mat roi_image){
    std::vector<cv::Mat> hist;
    hist.resize(3);
    cv::Mat hsv_image;
    cv::cvtColor(roi_image, hsv_image, cv::COLOR_BGR2HSV);
    //  填充HSV直方图
    cv::Mat h_hist = cv::Mat::zeros(1, 180, CV_32S);     // HSV直方图
    cv::Mat s_hist = cv::Mat::zeros(1, 256, CV_32S);
    cv::Mat v_hist = cv::Mat::zeros(1, 256, CV_32S);

    for(int i = 0; i < hsv_image.rows; i ++){
        for(int j = 0;j < hsv_image.cols;j++){
            cv::Vec3b hsv = hsv_image.at<cv::Vec3b>(i,j);
            if(hsv[2] == 0)continue;
            h_hist.at<int>(0, hsv[0])++;
            s_hist.at<int>(0, hsv[1])++;
            v_hist.at<int>(0, hsv[2])++;
        }
    }
    hist[0] = h_hist;
    hist[1] = s_hist;
    hist[2] = v_hist;
    return hist;
}

std::tuple<int,int,int> Ten_debug_hsv::get_hist_mode(std::vector<cv::Mat>& hist) {
    cv::Mat &h_list = hist[0];
    cv::Mat &s_list = hist[1];
    cv::Mat &v_list = hist[2];

    int h_max_count = 0;    // 最大计数值
    int s_max_count = 0;    // 最大计数值
    int v_max_count = 0;    // 最大计数值
    int h_mode_idx = 0;     // 众数对应的索引（初始为0）
    int s_mode_idx = 0;     // 众数对应的索引（初始为0）
    int v_mode_idx = 0;     // 众数对应的索引（初始为0）


    for (int i = 0; i < 180; i++) {
        int h_current_count = h_list.at<int>(0, i);
        if (h_current_count > h_max_count) {
            h_max_count = h_current_count;
            h_mode_idx = i;
        }
    }
    
    // 2. 遍历所有bin，找到初始众数（计数值最大的索引）
    for (int i = 0; i < 255; i++) {

        int s_current_count = s_list.at<int>(0, i);
        int v_current_count = v_list.at<int>(0, i);
        // 找到更大的计数值，更新众数

        if (s_current_count > s_max_count) {
            s_max_count = s_current_count;
            s_mode_idx = i;
        }
        if (v_current_count > v_max_count) {
            v_max_count = v_current_count;
            v_mode_idx = i;
        }
    }
    return std::tuple<int,int,int>(h_mode_idx,s_mode_idx,v_mode_idx);
}

void Ten_debug_hsv::compare_hsv_hist(
    const std::vector<cv::Mat>& hist1,
    const std::vector<cv::Mat>& hist2,
    double &h_sim_out,
    double &s_sim_out,
    double &v_sim_out,
    double &hsv_sim_out,
    int compare_method,
    double h_weight,
    double s_weight,
    double v_weight
) {
    // 1. 输入合法性校验
    if (hist1.size() != 3 || hist2.size() != 3) {
        std::cerr << "[ERROR] 直方图必须包含H/S/V三个通道！" << std::endl;
        return;
    }
    for (int i = 0; i < 3; i++) {
        if (hist1[i].empty() || hist2[i].empty() || 
            hist1[i].size() != hist2[i].size()) {
            std::cerr << "[ERROR] 第" << i << "通道直方图为空或尺寸不匹配！" << std::endl;
            return;
        }
    }

    // 2. 转换为float型（OpenCV要求输入为CV_32F/CV_64F）
    std::vector<cv::Mat> hist1_float, hist2_float;
    for (int i = 0; i < 3; i++) {
        cv::Mat tmp1, tmp2;
        hist1[i].convertTo(tmp1, CV_32F);
        hist2[i].convertTo(tmp2, CV_32F);
        hist1_float.push_back(tmp1);
        hist2_float.push_back(tmp2);
    }

    // 3. 分通道计算相似性
    h_sim_out = cv::compareHist(hist1_float[0], hist2_float[0], compare_method);
    s_sim_out = cv::compareHist(hist1_float[1], hist2_float[1], compare_method);
    v_sim_out = cv::compareHist(hist1_float[2], hist2_float[2], compare_method);

    // 4. 加权平均（可根据业务调整权重）
    hsv_sim_out = (h_sim_out * h_weight) + (s_sim_out * s_weight) + (v_sim_out * v_weight);

    // 调试输出：分通道相似性
    //std::cout << "[DEBUG] 分通道相似性：H=" << h_sim_out << ", S=" << s_sim_out << ", V=" << v_sim_out << std::endl;
    //std::cout << "[DEBUG] 加权平均相似性：" << hsv_sim_out << "（方法：" << compare_method << "）" << std::endl;

}

double Ten_debug_hsv::compare_hsv_hist(
    const std::vector<cv::Mat>& hist1,
    const std::vector<cv::Mat>& hist2,
    int compare_method,
    double h_weight,
    double s_weight,
    double v_weight
) {
    // 1. 输入合法性校验
    if (hist1.size() != 3 || hist2.size() != 3) {
        std::cerr << "[ERROR] 直方图必须包含H/S/V三个通道！" << std::endl;
        return -1.0;
    }
    for (int i = 0; i < 3; i++) {
        if (hist1[i].empty() || hist2[i].empty() || 
            hist1[i].size() != hist2[i].size()) {
            std::cerr << "[ERROR] 第" << i << "通道直方图为空或尺寸不匹配！" << std::endl;
            return -1.0;
        }
    }

    // 2. 转换为float型（OpenCV要求输入为CV_32F/CV_64F）
    std::vector<cv::Mat> hist1_float, hist2_float;
    for (int i = 0; i < 3; i++) {
        cv::Mat tmp1, tmp2;
        hist1[i].convertTo(tmp1, CV_32F);
        hist2[i].convertTo(tmp2, CV_32F);
        hist1_float.push_back(tmp1);
        hist2_float.push_back(tmp2);
    }

    // 3. 分通道计算相似性
    double h_sim = cv::compareHist(hist1_float[0], hist2_float[0], compare_method);
    double s_sim = cv::compareHist(hist1_float[1], hist2_float[1], compare_method);
    double v_sim = cv::compareHist(hist1_float[2], hist2_float[2], compare_method);

    // 4. 加权平均（可根据业务调整权重）
    double total_sim = (h_sim * h_weight) + (s_sim * s_weight) + (v_sim * v_weight);

    // 调试输出：分通道相似性
    std::cout << "[DEBUG] 分通道相似性：H=" << h_sim << ", S=" << s_sim << ", V=" << v_sim << std::endl;
    std::cout << "[DEBUG] 加权平均相似性：" << total_sim << "（方法：" << compare_method << "）" << std::endl;

    return total_sim;
}

bool Ten_debug_hsv::batch_compare_hist_and_save(
    const std::string& txt_save_path,
    const std::vector<cv::Mat>& hist_36,
    int compare_method,
    double h_weight,
    double s_weight,
    double v_weight
) {
    // -------------------------- 1. 输入合法性校验 --------------------------
    // 校验直方图向量长度是否为36（12张图×3通道）
    if (hist_36.size() != 36) {
        std::cerr << "[ERROR] 直方图向量长度必须为36！当前长度：" << hist_36.size() << std::endl;
        return false;
    }

    // 校验每个直方图Mat是否有效（非空、单通道）
    for (int i = 0; i < 36; i++) {
        if (hist_36[i].empty() || hist_36[i].channels() != 1) {
            std::cerr << "[ERROR] 第" << i << "个直方图Mat为空或非单通道！" << std::endl;
            return false;
        }
    }

    // -------------------------- 2. 拆分36个直方图为12组（每组3个：H/S/V） --------------------------
    std::vector<std::vector<cv::Mat>> hist_groups; // 12组，每组size=3（h/s/v）
    for (int img_idx = 0; img_idx < 12; img_idx++) {
        int base_idx = 3 * img_idx; // 每张图的起始索引：img1→0, img2→3,...img12→33
        std::vector<cv::Mat> single_img_hist = {
            hist_36[base_idx],    // 第img_idx+1张图的H通道
            hist_36[base_idx + 1],// 第img_idx+1张图的S通道
            hist_36[base_idx + 2] // 第img_idx+1张图的V通道
        };
        hist_groups.push_back(single_img_hist);
    }

    // -------------------------- 3. 打开TXT文件准备写入 --------------------------
    std::ofstream out_file(txt_save_path, std::ios::out | std::ios::trunc);
    if (!out_file.is_open()) {
        std::cerr << "[ERROR] 无法打开TXT文件！路径：" << txt_save_path << std::endl;
        return false;
    }

    // 写入TXT头部（说明格式）
    out_file << "===== 12张图像HSV直方图相似性对比结果（巴氏距离，越接近0越相似）=====\n";
    out_file << "对比图像对\tH通道相似性\tS通道相似性\tV通道相似性\t加权总相似性\n";
    out_file << "-------------------------------------------------------------\n";

    // -------------------------- 4. 两两对比（i<j，避免重复） --------------------------
    Ten::Ten_debug_hsv hist_compare; // 实例化对比类
    for (int i = 0; i < 12; i++) {          // 第i+1张图
        for (int j = i + 1; j < 12; j++) {  // 第j+1张图（只对比一次）
            try {
                // 调用compare_hsv_hist计算相似性
                double total_sim = hist_compare.compare_hsv_hist(
                    hist_groups[i],
                    hist_groups[j],
                    compare_method,
                    h_weight,
                    s_weight,
                    v_weight
                );

                // 单独计算分通道相似性（便于写入TXT）
                cv::Mat h1 = hist_groups[i][0], h2 = hist_groups[j][0];
                cv::Mat s1 = hist_groups[i][1], s2 = hist_groups[j][1];
                cv::Mat v1 = hist_groups[i][2], v2 = hist_groups[j][2];
                // 转换为float型（compareHist要求）
                cv::Mat h1_f, h2_f, s1_f, s2_f, v1_f, v2_f;
                h1.convertTo(h1_f, CV_32F); h2.convertTo(h2_f, CV_32F);
                s1.convertTo(s1_f, CV_32F); s2.convertTo(s2_f, CV_32F);
                v1.convertTo(v1_f, CV_32F); v2.convertTo(v2_f, CV_32F);
                // 分通道对比
                double h_sim = cv::compareHist(h1_f, h2_f, compare_method);
                double s_sim = cv::compareHist(s1_f, s2_f, compare_method);
                double v_sim = cv::compareHist(v1_f, v2_f, compare_method);

                // 写入TXT（格式：图像i+1 vs 图像j+1\tH值\tS值\tV值\t总权重值）
                out_file << "图像" << i+1 << " vs 图像" << j+1 << "\t"
                         << std::fixed << std::setprecision(6) // 保留6位小数
                         << h_sim << "\t"
                         << s_sim << "\t"
                         << v_sim << "\t"
                         << total_sim << "\n";
            } catch (const std::exception& e) {
                std::cerr << "[ERROR] 对比图像" << i+1 << "和" << j+1 << "失败：" << e.what() << std::endl;
                out_file << "图像" << i+1 << " vs 图像" << j+1 << "\t对比失败：" << e.what() << "\n";
            }
        }
    }

    // -------------------------- 5. 关闭文件 --------------------------
    out_file.close();
    std::cout << "[INFO] 直方图对比结果已写入TXT！路径：" << txt_save_path << std::endl;
    return true;
}

void Ten_debug_hsv::cluster_box_images(const std::vector<HistCompareResult>& all_compare_results) {
    // 1. 识别纯黑图像 + 提取非纯黑图像（原有逻辑不变）
    std::unordered_set<int> black_imgs;
    std::unordered_map<int, int> black_count;
    std::unordered_set<int> all_imgs;
    for (const auto& res : all_compare_results) {
        all_imgs.insert(res.img1);
        all_imgs.insert(res.img2);
        if (res.total_sim == 1.0) {
            black_count[res.img1]++;
            black_count[res.img2]++;
        }
    }
    int total_img_num = all_imgs.size();
    for (const auto& pair : black_count) {
        if (pair.second == total_img_num - 1) {
            black_imgs.insert(pair.first);
            std::cout << "[调试] 识别纯黑图像：图像" << pair.first << std::endl;
        }
    }
    std::vector<int> non_black_imgs;
    for (int img : all_imgs) {
        if (!black_imgs.count(img)) non_black_imgs.push_back(img);
    }
    std::cout << "[调试] 非纯黑图像列表：";
    for (int img : non_black_imgs) std::cout << img << " ";
    std::cout << "\n" << std::endl;

    // 2. 构建特征矩阵（原有逻辑不变）
    int n = non_black_imgs.size();
    cv::Mat feature_mat(n, n, CV_32F, cv::Scalar(1.0));
    std::unordered_map<int, int> img_to_idx;
    for (int i = 0; i < n; i++) img_to_idx[non_black_imgs[i]] = i;
    for (const auto& res : all_compare_results) {
        if (black_imgs.count(res.img1) || black_imgs.count(res.img2)) continue;
        int i = img_to_idx[res.img1];
        int j = img_to_idx[res.img2];
        feature_mat.at<float>(i, j) = res.total_sim;
        feature_mat.at<float>(j, i) = res.total_sim;
    }
    for (int i = 0; i < n; i++) feature_mat.at<float>(i, i) = 0.0f;

    // 3. K-Means聚类（修复后）
    cv::Mat data = feature_mat.clone();
    cv::Mat labels, centers;
    double sse = cv::kmeans(
        data,
        2,
        labels,
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 100, 0.001),
        10,
        cv::KMEANS_PP_CENTERS, // 替换为K-Means++，更优
        centers
    );
    std::cout << "[调试] 聚类SSE（簇内误差平方和）：" << sse << std::endl;

    // 4. 查询阵营归属
    std::unordered_map<int, int> idx_to_img;
    for (auto& pair : img_to_idx) idx_to_img[pair.second] = pair.first;
    std::vector<int> camp0, camp1;
    for (int i = 0; i < n; i++) {
        int label = labels.at<int>(i, 0);
        int img_num = idx_to_img[i];
        if (label == 0) camp0.push_back(img_num);
        else camp1.push_back(img_num);
    }

    // 6. 查询聚类中心
    std::cout << "\n[调试] 阵营0聚类中心：\n";
    for (int j = 0; j < centers.cols; j++) {
        std::cout << std::fixed << std::setprecision(3) << centers.at<float>(0, j) << " ";
    }
    std::cout << "\n[调试] 阵营1聚类中心：\n";
    for (int j = 0; j < centers.cols; j++) {
        std::cout << std::fixed << std::setprecision(3) << centers.at<float>(1, j) << " ";
    }

    // 7. 计算成员得分（欧氏距离 + 比值得分）
    std::unordered_map<int, double> img_dist_score, img_ratio_score;
    for (int i = 0; i < n; i++) {
        int img_num = idx_to_img[i];
        int label = labels.at<int>(i, 0);
        cv::Mat sample = data.row(i);
        double dist_self = cv::norm(sample, centers.row(label), cv::NORM_L2);
        double dist_other = cv::norm(sample, centers.row(1-label), cv::NORM_L2);
        img_dist_score[img_num] = dist_self;
        img_ratio_score[img_num] = dist_other / (dist_self + 1e-8);
    }

    // 8. 输出得分
    std::cout << "\n[调试] 阵营0成员得分（欧氏距离，越小越好）：\n";
    for (int img : camp0) {
        std::cout << "图像" << img << "：" << std::fixed << std::setprecision(3) << img_dist_score[img] << std::endl;
    }
    std::cout << "\n[调试] 阵营1成员得分（欧氏距离，越小越好）：\n";
    for (int img : camp1) {
        std::cout << "图像" << img << "：" << std::fixed << std::setprecision(3) << img_dist_score[img] << std::endl;
    }
    std::cout << "\n[调试] 阵营0成员比值得分（越大越好）：\n";
    for (int img : camp0) {
        std::cout << "图像" << img << "：" << std::fixed << std::setprecision(3) << img_ratio_score[img] << std::endl;
    }

    // 9. 最终判定（原有逻辑不变）
    std::vector<int> has_box_camp = camp0.size() >= camp1.size() ? camp0 : camp1;
    std::vector<int> no_box_camp = camp0.size() < camp1.size() ? camp0 : camp1;
    std::cout << "\n===== 最终判定结果 =====\n";
    std::cout << "纯黑图像（无法判断）：";
    for (int img : black_imgs) std::cout << "图像" << img << " ";
    std::cout << "\n有方块阵营（数量多）：";
    for (int img : has_box_camp) std::cout << "图像" << img << " ";
    std::cout << "\n无方块阵营（数量少）：";
    for (int img : no_box_camp) std::cout << "图像" << img << " ";
    std::cout << "\n";
}


// 函数功能：读取文件夹中所有jpg，按idx1-12顺序存入box_lists
// 输入：img_dir - 图片文件夹绝对路径
// 输出：按idx1→idx12顺序排列的Mat向量（索引0=idx1，索引11=idx12），未找到的idx填充160×160黑图
void Ten_debug_hsv::read_jpgs_by_idx_order(const std::string& img_dir,std::vector<Ten::box>& box_lists) {
    // 2. 检查目录合法性
    if (!std::filesystem::exists(img_dir) || !std::filesystem::is_directory(img_dir)) {
        ROS_ERROR("图片目录不存在或不是有效目录：%s", img_dir.c_str());
        return; // 返回全黑图的vector
    }

    // 3. 遍历文件夹中所有.jpg文件
    int success_count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(img_dir)) {
        // 过滤非文件/非jpg的项
        if (!entry.is_regular_file() || entry.path().extension() != ".png") {
            continue;
        }

        // 4. 提取文件名（如"idx9cls19conf0.053458.jpg"）
        std::string filename = entry.path().filename().string();
        
        // 5. 解析文件名中的idx数字（核心：定位idx和cls的位置）
        size_t idx_pos = filename.find("idx");
        size_t cls_pos = filename.find("cls");
        if (idx_pos == std::string::npos || cls_pos == std::string::npos || cls_pos <= idx_pos + 3) {
            //ROS_WARN("文件名格式错误，跳过：%s", filename.c_str());
            continue;
        }

        // 6. 截取idx后的数字字符串（如"idx9cls..." → 截取"9"）
        std::string idx_str = filename.substr(idx_pos + 3, cls_pos - (idx_pos + 3));
        int img_idx = -1;
        try {
            img_idx = std::stoi(idx_str);
        } catch (...) {
            //ROS_WARN("idx数字转换失败，跳过：%s", filename.c_str());
            continue;
        }

        // 7. 验证idx范围（必须1-12）
        if (img_idx < 1 || img_idx > 12) {
            //ROS_WARN("idx超出1-12范围，跳过：%s (idx=%d)", filename.c_str(), img_idx);
            continue;
        }

        // 8. 读取图片（BGR格式，匹配你的roi_image）
        cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_UNCHANGED);
        if (img.empty()) {
            //ROS_WARN("图片读取失败，跳过：%s", entry.path().c_str());
            continue;
        }

        // 9. 统一resize为160×160（和你代码中roi_image尺寸一致）
        //cv::resize(img, img, cv::Size(160, 160), 0, 0, cv::INTER_LINEAR);

        // 10. 按idx顺序存入vector（idx1→索引0，idx12→索引11）
        int vec_idx = img_idx - 1;
        box_lists[vec_idx].roi_image = img.clone(); // 深拷贝，避免数据共享
        success_count += 1;
        //ROS_INFO("成功读取idx=%d的图片：%s", img_idx, entry.path().c_str());
    }
    std::cout << "success_count : " << success_count << std::endl; 
}

std::vector<int> Ten_debug_hsv::bgr_color_analysis(const cv::Mat& img_in)
        {
            cv::Mat img;
            cv::cvtColor(img_in, img, cv::COLOR_BGR2HSV);
            std::vector<std::vector<int>> calculate;
            calculate.resize(3);
            for(size_t i = 0; i < calculate.size(); i++)
            {
                calculate[i].resize(256, 0);
            }
            for (int row = 0; row < img.rows; row++) {
                for (int col = 0; col < img.cols; col++) {
                    // 获取当前像素的BGR值
                    const cv::Vec3b& pixel = img.at<cv::Vec3b>(row, col); // 用引用避免拷贝，提升效率
                    
                    // // 访问每个通道的值
                    // uchar blue = pixel[0];   // 蓝通道
                    // uchar green = pixel[1];  // 绿通道
                    // uchar red = pixel[2];    // 红通道
                    if(pixel[2] == 0)
                    continue;

                    calculate[0][pixel[0]]++; 
                    calculate[1][pixel[1]]++;
                    calculate[2][pixel[2]]++;
                }
            }

            //size_t arr[calculate.size()] = {0};
            std::vector<int> arr;
            arr.resize(3,0);
            for(size_t i = 0; i < calculate.size(); i++)
            {
                auto max = std::max_element(calculate[i].begin(), calculate[i].end());
                arr[i] = max - calculate[i].begin();
            }


            float total = static_cast<float>(img.rows * img.cols);
            cv::Mat result(256, 256*3, CV_8UC3, cv::Scalar(255, 255, 255));

            for(size_t j = 0; j < 256*3; j++)
            {
                //size_t high = 256 - static_cast<size_t>((float)calculate[j / 256][j % 256] / total * 256);
                if(calculate[j / 256][arr[j / 256]] == 0)
                {
                    continue;
                }
                int high = 256 - static_cast<int>((float)calculate[j / 256][j % 256] / (float)calculate[j / 256][arr[j / 256]] * 256);
                // std::cout << "calculate[" << j / 256 << "][" << j % 256 << "]" << calculate[j / 256][j % 256] << std::endl;
                // std::cout << "arr[" << j / 256 << "]" << arr[j / 256] << std::endl;
                for(int i = 256 - 1; i >=0 && i >= high - 1; i--)
                {
                    if(j < 256)
                    result.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 0, 0);
                    else if(j < 256*2)
                    result.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 255, 0);
                    else if(j < 256*3)
                    result.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255);
                }
            }


            cv::putText(result, std::to_string(arr[0]), cv::Point(256/2, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 1);
            cv::putText(result, std::to_string(arr[1]), cv::Point(256 + 256/2, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 1);
            cv::putText(result, std::to_string(arr[2]), cv::Point(256*3 - 256/2, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 1);
            
            return arr;
        }

Ten::Ten_debug_hsv _DEBUG_HSV_;

}
#endif 