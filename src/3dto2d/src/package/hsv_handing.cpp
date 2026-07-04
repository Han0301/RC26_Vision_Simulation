#ifndef _Ten_hsv_handing_CPP_
#define _Ten_hsv_handing_CPP_
#include "hsv_handing.h"

namespace Ten{

void Ten_hsv_handing::set_hsv_top_n
(
    const std::vector<box>& box_lists, 
    const int top_n,
    const int min_manage_points,
    std::vector<score>& score_lists
)
{
    // 1 清空上一帧的缓存
    score_lists.clear();
    score_lists.resize(12);

    // 2 填充 score_lists 
    for (const auto& box: box_lists)
    {
        cv::Mat hsv_image;
        cv::cvtColor(box.roi_image, hsv_image, cv::COLOR_BGR2HSV);
        //  2.1 填充HSV直方图
        cv::Mat h_hist = cv::Mat::zeros(1, 180, CV_32S);     // HSV直方图
        cv::Mat s_hist = cv::Mat::zeros(1, 256, CV_32S);
        cv::Mat v_hist = cv::Mat::zeros(1, 256, CV_32S);

        int valid_points = 0;
        for(int i = 0; i < hsv_image.rows; i ++)
        {
            for(int j = 0;j < hsv_image.cols;j++)
            {
                cv::Vec3b hsv = hsv_image.at<cv::Vec3b>(i,j);
                if(hsv[2] == 0 || hsv[1] < 15)continue;     // 过滤掉 黑色无效像素 和 白色干扰像素
                h_hist.at<int>(0, hsv[0])++;
                s_hist.at<int>(0, hsv[1])++;
                v_hist.at<int>(0, hsv[2])++;
                valid_points++;
            }
        }
        // 2.2 写入 idx, valid_count 字段
        score_lists[box.idx - 1].idx = box.idx;
        score_lists[box.idx - 1].valid_count = valid_points;

        // 🈲2.3 小于 min_manage_points 跳出不更新， 写入 is_managed 为 false
        if (valid_points < min_manage_points) 
        {
            std::cout << "🤡in func: set_hsv_top_n 2.3, box idx= " << box.idx << ",valid_points < min_manage_points, skip cal hsv topn" << std::endl;

            score_lists[box.idx - 1].is_managed = false;
            continue;
        }
        else{score_lists[box.idx - 1].is_managed = true;}

        // 2.4 写入 top_n_h, top_n_s, top_n_v 字段
        std::vector<HistValue> h_topn = getTopNValues(h_hist, top_n);
        std::vector<HistValue> s_topn = getTopNValues(s_hist, top_n);
        std::vector<HistValue> v_topn = getTopNValues(v_hist, top_n);

        score_lists[box.idx - 1].top_n_h = h_topn;
        score_lists[box.idx - 1].top_n_s = s_topn;
        score_lists[box.idx - 1].top_n_v = v_topn;
    }
}

void Ten_hsv_handing::set_hsv_topn_score
(
    const std::vector<box>& box_lists, 
    const int top_n,
    const int min_manage_points,
    const std::vector<int> standard_hsv,
    std::vector<score>& score_lists
)
{
    // 1 写入其中的 idx, valid_count, top_n_h, top_n_s, top_n_v, is_managed 字段
    set_hsv_top_n(box_lists, top_n,min_manage_points,score_lists);

    // 2 统计分数， 写入 hsv_score 字段
    std::vector<float> score = {-1.0f,-1.0f,-1.0f,-1.0f,-1.0f,-1.0f,-1.0f,-1.0f,-1.0f,-1.0f,-1.0f,-1.0f}; 
    for(size_t i = 0; i < 12; i++) 
    {
        // 🈲2.1 跳出 set_hsv_top_n 函数中未写入hsv 位置， 其score 取默认值 -1.0
        if (!(score_lists[i].is_managed))
        {
            score_lists[i].hsv_score = score[i];
            continue;
        }
        // 2.2 累加各位置的首位度， 以作为 后面累加各 topn 得分 的权重
        float total_degree_h = 0.0f;
        float total_degree_s = 0.0f;
        float total_degree_v = 0.0f;

        for(size_t j = 0; j < top_n; j++)
        {
            total_degree_h += score_lists[i].top_n_h[j].degree_of_promacy;
            total_degree_s += score_lists[i].top_n_s[j].degree_of_promacy;
            total_degree_v += score_lists[i].top_n_v[j].degree_of_promacy;
        }
        // 2.3 计算得分
        for(size_t j = 0; j < top_n; j++)
        {   
            // 2.3.1 获取 topn 中的 hsv 值
            int h = score_lists[i].top_n_h[j].position;
            int s = score_lists[i].top_n_s[j].position;
            int v = score_lists[i].top_n_v[j].position;
            // 2.3.2 计算 h 通道的损失 h_diff_score
            float h_diff = std::min( std::abs(h - standard_hsv[0]), (std::min(h,standard_hsv[0]) + 180 - std::max(h,standard_hsv[0]))   );
            float h_diff_score = 0.0f;

            if (h_diff < 5)
            {
                h_diff_score = 1.0f * h_diff;
            }
            else if (5 < h_diff < 10)
            {
                h_diff_score = 5.0f + 3.0f * (h_diff - 5);
            }
            else
            {
                h_diff_score = 20.0f + 5.0f * (h_diff - 10);
            }
            // 2.3.3 计算该 topn 的得分 并累加到最终得分
            float now_score = (score_lists[i].top_n_h[j].degree_of_promacy / total_degree_h) * 
                                std::max( 0.0f, 70.0f - h_diff_score)
                    + (score_lists[i].top_n_s[j].degree_of_promacy / total_degree_s) * std::max(0.0f,10.0f - 0.2f * std::abs(s - standard_hsv[1]))
                    + (score_lists[i].top_n_v[j].degree_of_promacy / total_degree_v) * std::max(0.0f,15.0f - 0.3f * std::abs(v - standard_hsv[2]));
            score[i] += now_score;
        } 
        score[i] = score[i] / 100;
        score_lists[i].hsv_score = score[i];
    }

    // 2.3.4 对 score_lists 重新排序
    //❗❗❗ 注： 该操作打乱了 score_lists 中的顺序， 原本按照位置排序， 现在按照 得分进行降序排序
    std::sort(score_lists.begin(),score_lists.end(),
    [](const Ten::score&a, const Ten::score&b){
        return a.hsv_score > b.hsv_score;
    });
}

void Ten_hsv_handing::set_hsv_topn_stand
(
    const std::vector<box>& box_lists, 
    const int top_n,
    const int min_manage_points,
    std::vector<int>& standard_hsv,
    std::vector<score>& score_lists
)
{
    // 1 写入其中的 idx, valid_count, top_n_h, top_n_s, top_n_v, is_managed 字段
    set_hsv_top_n(box_lists, top_n,min_manage_points,score_lists);

    // 2 统计累加落在hsv直方图 各位置的首位度
    std::unordered_map<int,float> total_h_pos_degree;
    std::unordered_map<int,float> total_s_pos_degree;
    std::unordered_map<int,float> total_v_pos_degree;
    for (int i = 0; i < 12; i++)
    {
        if (!(score_lists[i].is_managed))
        {
            continue;
        }           
        for(int j = 0; j < top_n; j++)
        {
            total_h_pos_degree[score_lists[i].top_n_h[j].position] += score_lists[i].top_n_h[j].degree_of_promacy;
            total_s_pos_degree[score_lists[i].top_n_s[j].position] += score_lists[i].top_n_s[j].degree_of_promacy;
            total_v_pos_degree[score_lists[i].top_n_v[j].position] += score_lists[i].top_n_v[j].degree_of_promacy;
        }
    }

    // 3 转化成列表， 进行降序排序
    std::vector<std::pair<int, float>> h_pos_degree(total_h_pos_degree.begin(), total_h_pos_degree.end());
    std::vector<std::pair<int, float>> s_pos_degree(total_s_pos_degree.begin(), total_s_pos_degree.end());
    std::vector<std::pair<int, float>> v_pos_degree(total_v_pos_degree.begin(), total_v_pos_degree.end());
    
    std::sort(h_pos_degree.begin(), h_pos_degree.end(),
        [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
            return a.second > b.second;
        });

    std::sort(s_pos_degree.begin(), s_pos_degree.end(),
        [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
            return a.second > b.second;
        });
    std::sort(v_pos_degree.begin(), v_pos_degree.end(),
        [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
            return a.second > b.second;
        });

    // 4 获取各通道列表的 first 即落在 hsv直方图的位置， 填充 standard_hsv
    standard_hsv = {h_pos_degree[0].first, s_pos_degree[0].first, v_pos_degree[0].first};
}

void Ten_hsv_handing::print_score_lists
(
    const std::vector<score>& score_lists
)
{
    for(int i = 0; i < Ten::_INIT_HSV_.score_lists_.size(); i++)
    {
        std::cout << "-----------------idx: " << Ten::_INIT_HSV_.score_lists_[i].idx << std::endl;
        // for(int j = 0; j < Ten::_INIT_HSV_.score_lists_[i].top_n_h.size(); j++)
        // {
        //     std::cout << "j: " << std::endl;
        //     std::cout << "position: " << Ten::_INIT_HSV_.score_lists_[i].top_n_h[j].position << 
        //                  ", count: " << Ten::_INIT_HSV_.score_lists_[i].top_n_h[j].count << std::endl;
        // }
        std::cout << "----score: " << Ten::_INIT_HSV_.score_lists_[i].hsv_score << std::endl;
    }
}

    Ten::Ten_hsv_handing _HSV_HANDING_;
    Ten::init_hsv _INIT_HSV_;
}
#endif