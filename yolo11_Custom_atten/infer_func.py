"""
infer_func.py
    推理用的功能包, 包含结合conf,point_size_weight,place的统计函数, 写入csv表格的函数
"""
import os
from prompt_toolkit.utils import to_str
from tqdm import tqdm
from dataset_main import _compute_confidence
import csv

def conf_evaluate(results, conf_list, is_print=True):
    # 初始化
    conf_dict = {
        0: {"count": 0},
        1: {"count": 0}
    }
    for conf in conf_list:
        conf_dict[0][f"{conf}_count"] = 0
        conf_dict[0][f"{conf}_acc_count"] = 0
        conf_dict[1][f"{conf}_count"] = 0
        conf_dict[1][f"{conf}_acc_count"] = 0
    for result in tqdm(results.values(), desc="统计中(conf)", colour="red"):
        for i in range(12):
            conf_dict[result["pred_cls_np"][i]]["count"] += 1
            for conf in conf_list:
                if result["pred_probs_np"][i].max() > conf:
                    conf_dict[result["pred_cls_np"][i]][to_str(conf) + "_count"] += 1
                    if result["pred_cls_np"][i] == result["labels"][i]:
                        conf_dict[result["pred_cls_np"][i]][to_str(conf) + "_acc_count"] += 1
    if is_print:
        count_0, count_1 = conf_dict[0]["count"], conf_dict[1]["count"]
        print(f"0类总数: {count_0:5d}, 1类总数: {count_1:5d}")
        print("     0类进该conf数, 占比, 正确数量, 正确率  |  1类进该conf数, 占比, 正确数量, 正确率")
        for conf in conf_list:
            conf_count_0, conf_count_1 = conf_dict[0][to_str(conf) + "_count"], conf_dict[1][to_str(conf) + "_count"]
            conf_rate_0, conf_rate_1 = conf_dict[0][to_str(conf) + "_count"] / count_0 * 100, conf_dict[1][to_str(conf) + "_count"] / count_1 * 100
            acc_count_0, acc_count_1 = conf_dict[0][to_str(conf) + "_acc_count"], conf_dict[1][to_str(conf) + "_acc_count"]
            acc_rate_0, acc_rate_1 = acc_count_0 / conf_count_0 * 100, acc_count_1 / conf_count_1 * 100
            print(
                f"conf: {conf} :{conf_count_0: 5d}, {conf_rate_0:.2f}%, {acc_count_0:5d}, {acc_rate_0:.2f}% |     {conf_count_1: 5d}, {conf_rate_1:.2f}%, {acc_count_1:5d}, {acc_rate_1:.2f}%")
    return conf_dict

def place_evaluate(results,place_acc_count, is_print=True):
    place_acc_count = [0] * 12
    for result in tqdm(results.values(), desc="统计中(place)", colour="red"):
        for place in range(12):
            if result["pred_cls_np"][place] == result["labels"][place]:
                place_acc_count[place] += 1
    place_acc_rate = [acc_count / len(results) * 100 for acc_count in place_acc_count]
    if is_print:
        print("各位置下标  ", end=": ")
        for i in range(12):
            print(f"{i + 1:5d}", end=" | ")
        print()
        print("各位置正确数", end=": ")
        for i in range(12):
            print(f"{place_acc_count[i]: 5d}", end=" | ")
        print()
        print(f"各位置正确率", end=": ")
        for i in range(12):
            print(f"{place_acc_rate[i]:.2f}", end=" | ")
        print()
    return place_acc_count,place_acc_rate

def ps_w_evaluate(results, ps_w_thods:list, is_print=True):
    ps_dict = {}
    for ps_w_thod in ps_w_thods:
        ps_dict[to_str(ps_w_thod)] = {}
        ps_dict[to_str(ps_w_thod)]["count"] = [0] * 12
        ps_dict[to_str(ps_w_thod)]["acc"] = [0] * 12
    for result in tqdm(results.values(), desc="统计中(point_size)", colour="red"):
        ps_w = _compute_confidence(result["point_size"])
        for place in range(12):
            for ps_w_thod in ps_w_thods:
                if ps_w[place] > ps_w_thod:
                    ps_dict[to_str(ps_w_thod)]["count"][place] += 1
                    if result["pred_cls_np"][place] == result["labels"][place]:
                        ps_dict[to_str(ps_w_thod)]["acc"][place] += 1
    if is_print:
        places = [i + 1 for i in range(12)]
        for ps_w_thod in ps_w_thods:
            print(f"在ps_w_thod = {ps_w_thod}统计结果: ")

            print("各位置",end=": ")
            for place in range(12):
                print(f"{places[place]: 5d}",end=" | ")
            print()

            print("统计数",end=": ")
            for place in range(12):
                count = ps_dict[to_str(ps_w_thod)]["count"][place]
                print(f"{count: 5d}",end=" | ")
            print()

            print("数占比",end=": ")
            for place in range(12):
                count_rate = ps_dict[to_str(ps_w_thod)]["count"][place] / len(results)
                print(f"{count_rate:.3f}",end=" | ")
            print()

            print("正确数",end=": ")
            for place in range(12):
                acc = ps_dict[to_str(ps_w_thod)]["acc"][place]
                print(f"{acc: 5d}",end=" | ")
            print()

            print("正确率",end=": ")
            for place in range(12):
                acc_rate = ps_dict[to_str(ps_w_thod)]["acc"][place] / len(results)
                print(f"{acc_rate:.3f}",end=" | ")
            print()
    return ps_dict

def write_txt(save_path, folder_path, pred_cls_np, pred_probs_np, labels=None, point_size=None, wrong_place=None):
    """
    【CSV表格版】单文件夹推理结果保存 | 自动创建/追加 | 参数完全不变
    """
    # 强制指定CSV后缀，保证格式正确
    base, ext = os.path.splitext(save_path)
    if ext.lower() != ".csv":
        save_path = base + ".csv"

    # 自动创建文件夹
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 判断文件是否存在，不存在则需要写入表头
    file_exists = os.path.exists(save_path)

    # 追加模式打开文件，utf-8-sig兼容Excel
    with open(save_path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)

        # 首次创建文件：写入表头
        if not file_exists:
            writer.writerow([
                "文件夹路径", "位置", "类别0置信度", "类别1置信度",
                "预测类别", "真实类别", "point_size"
            ])

        # 写入12个位置的推理数据
        for i in range(12):
            conf0 = round(pred_probs_np[i][0], 3)
            conf1 = round(pred_probs_np[i][1], 3)
            pred_cls = pred_cls_np[i]
            true_cls = labels[i] if labels is not None else ""
            ps = point_size[i] if point_size is not None else ""

            writer.writerow([
                folder_path, i + 1, conf0, conf1, pred_cls, true_cls, ps
            ])

        # 写入错误位置统计（可选）
        if wrong_place is not None:
            correct_num = 12 - len(wrong_place)
            writer.writerow([
                folder_path, "统计", "", "",
                f"正确数: {correct_num}",
                f"错误位置: {wrong_place}",
                ""
            ])

        # 分隔行，区分不同文件夹结果
        writer.writerow(["-" * 100, "", "", "", "", "", ""])

def save_results(
        results: dict,
        infer_time: str,
        model_path: str,
        dataset_path: str,
        is_conf: bool = False, conf_list: list = None,
        is_place: bool = False, place_acc_count: list = None,
        is_point_size_weight: bool = False, ps_w_thods: list = None,
        is_save: bool = False, save_path: str = None
):
    if not is_save or not save_path:
        return
    if not save_path.endswith(".csv"):
        save_path = os.path.splitext(save_path)[0] + ".csv"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    total_folders = len(results)
    total_roi = total_folders * 12

    try:
        with open(save_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)

            # 【分行显示：推理基础信息】
            writer.writerow(["=" * 60])
            writer.writerow(["📊 推理基础信息"])
            writer.writerow(["=" * 60])
            writer.writerow(["推理时间", infer_time])
            writer.writerow(["模型路径", model_path])
            writer.writerow(["数据集路径", dataset_path])
            writer.writerow([])
            writer.writerow(["📊 数据集统计总览"])
            writer.writerow(["总文件夹数", total_folders])
            writer.writerow(["总ROI数", total_roi])
            writer.writerow(["=" * 60])
            writer.writerow([])

            # 1. 置信度统计
            if is_conf and conf_list is not None:
                conf_dict = conf_evaluate(results, conf_list, is_print=False)
                count_0, count_1 = conf_dict[0]["count"], conf_dict[1]["count"]
                writer.writerow(["置信度阈值统计结果"])
                writer.writerow(["0类总数", count_0, "1类总数", count_1])
                writer.writerow([
                    "置信度阈值",
                    "0类-数量", "0类-占比(%)", "0类-正确率(%)",
                    "1类-数量", "1类-占比(%)", "1类-正确率(%)"
                ])
                for conf in conf_list:
                    conf_count_0 = conf_dict[0][f"{conf}_count"]
                    conf_count_1 = conf_dict[1][f"{conf}_count"]
                    conf_rate_0 = conf_count_0 / count_0 * 100 if count_0 != 0 else 0
                    conf_rate_1 = conf_count_1 / count_1 * 100 if count_1 != 0 else 0
                    acc_count_0 = conf_dict[0][f"{conf}_acc_count"]
                    acc_count_1 = conf_dict[1][f"{conf}_acc_count"]
                    acc_rate_0 = acc_count_0 / conf_count_0 * 100 if conf_count_0 != 0 else 0
                    acc_rate_1 = acc_count_1 / conf_count_1 * 100 if conf_count_1 != 0 else 0
                    writer.writerow([
                        f"{conf:.1f}",
                        f"{conf_count_0}", f"{conf_rate_0:.2f}", f"{acc_rate_0:.2f}",
                        f"{conf_count_1}", f"{conf_rate_1:.2f}", f"{acc_rate_1:.2f}"
                    ])
                writer.writerow([])
                writer.writerow(["-" * 80])
                writer.writerow([])

            # 2. 位置准确率
            if is_place and place_acc_count is not None:
                place_acc_count, place_acc_rate = place_evaluate(results, place_acc_count, is_print=False)
                writer.writerow(["12个位置准确率统计结果"])
                writer.writerow(["位置", "正确数", "正确率(%)"])
                for i in range(12):
                    writer.writerow([i+1, place_acc_count[i], f"{place_acc_rate[i]:.2f}"])
                writer.writerow([])
                writer.writerow(["-" * 80])
                writer.writerow([])

            # 3. 点尺寸权重
            if is_point_size_weight and ps_w_thods is not None:
                ps_dict = ps_w_evaluate(results, ps_w_thods, is_print=False)
                writer.writerow(["point_size权重(ps_w)统计结果"])
                for ps_w_thod in ps_w_thods:
                    writer.writerow([f"ps_w_thod = {ps_w_thod:.1f}"])
                    writer.writerow(["位置", "统计数", "数占比", "正确数", "正确率"])
                    counts = ps_dict[str(ps_w_thod)]["count"]
                    accs = ps_dict[str(ps_w_thod)]["acc"]
                    for i in range(12):
                        c, a = counts[i], accs[i]
                        rate_c = c / total_folders
                        rate_a = a / total_folders
                        writer.writerow([i+1, c, f"{rate_c:.3f}", a, f"{rate_a:.3f}"])
                    writer.writerow([])
                writer.writerow(["-" * 80])

        print(f"\n✅ 强制覆写成功！CSV已保存至：{save_path}")

    # 捕获权限错误（文件被Excel/记事本打开时）
    except PermissionError:
        print(f"\n❌ 错误：文件被其他程序占用，请关闭 Excel/记事本 后重试！")
        print(f"🔒 占用文件：{save_path}")