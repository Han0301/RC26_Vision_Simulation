import os
import json
import cv2
import shutil
import pandas as pd
import re

# ===================== 【请修改这里的所有路径】 =====================
# 数据集根路径（包含 roi_images + labels 文件夹）
DATASET_ROOT = r"I:\datasets_real_blue_new785"
# 按空格删除的样本存放路径（垃圾数据集）
FINAL_JUNK_ROOT = r"I:\datasets_real_blue_waste785"
# 你的错误明细CSV文件路径
ERROR_CSV_PATH = r"I:\datasets_real_blue_new785\error.csv"
# 图片显示尺寸
DISPLAY_SIZE = (640, 640)
# ==================================================================

# 全局变量
click_result = None
is_clicked = False
space_clicked = False
exit_flag = False


def mouse_callback(event, x, y, flags, param):
    """鼠标回调：左键=1，右键=0"""
    global click_result, is_clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        click_result = 1
        is_clicked = True
    elif event == cv2.EVENT_RBUTTONDOWN:
        click_result = 0
        is_clicked = True


def delete_whole_sample(img_idx):
    """
    移动整个样本到垃圾数据集（按空格触发）
    :param img_idx: 样本编号 整数
    :return: True=移动成功
    """
    roi_dir_name = f"roi_{img_idx}"
    label_file_name = f"label_{img_idx}.json"

    # 原路径
    src_roi = os.path.join(DATASET_ROOT, "roi_images", roi_dir_name)
    src_label = os.path.join(DATASET_ROOT, "labels", label_file_name)

    # 目标垃圾路径
    dst_roi_root = os.path.join(FINAL_JUNK_ROOT, "roi_images")
    dst_label_root = os.path.join(FINAL_JUNK_ROOT, "labels")
    os.makedirs(dst_roi_root, exist_ok=True)
    os.makedirs(dst_label_root, exist_ok=True)

    try:
        shutil.move(src_roi, os.path.join(dst_roi_root, roi_dir_name))
        shutil.move(src_label, os.path.join(dst_label_root, label_file_name))
        print(f"✅ 样本 {roi_dir_name} 已移动至垃圾数据集")
        return True
    except Exception as e:
        print(f"❌ 删除失败：{str(e)}")
        return False


def load_error_csv(csv_path):
    """
    读取CSV，修复：浮点数样本编号转整数
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    header_line = -1
    for i, line in enumerate(lines):
        if "错误文件夹路径" in line and "位置" in line:
            header_line = i
            break

    if header_line == -1:
        print("❌ 未找到错误明细表头")
        return []

    df = pd.read_csv(csv_path, skiprows=header_line)
    df = df.dropna(subset=["错误文件夹路径"]).reset_index(drop=True)
    df = df[~df["错误文件夹路径"].str.contains("---")].reset_index(drop=True)

    # 🔥 核心修复：提取样本编号并强制转整数（解决100.0 → 100）
    def extract_img_idx(folder_path):
        match = re.search(r'roi_(\d+)', str(folder_path))
        if match:
            return int(match.group(1))  # 直接转整数！
        return None

    df["img_idx"] = df["错误文件夹路径"].apply(extract_img_idx)
    df = df.dropna(subset=["img_idx"]).reset_index(drop=True)

    df["位置"] = df["位置"].astype(int)
    df["真实类别(正确)"] = df["真实类别(正确)"].astype(int)
    df["预测类别(错误)"] = df["预测类别(错误)"].astype(int)

    error_list = []
    for idx, row in df.iterrows():
        error_list.append({
            "img_idx": row["img_idx"],
            "roi_pos": row["位置"],
            "real_label": row["真实类别(正确)"],
            "pred_label": row["预测类别(错误)"],
            "point_size": row["point_size"]
        })

    print(f"✅ 成功读取CSV，共 {len(error_list)} 条错误数据")
    return error_list


def process_csv_errors():
    global click_result, is_clicked, space_clicked, exit_flag

    error_list = load_error_csv(ERROR_CSV_PATH)
    if not error_list:
        print("❌ 无错误数据需要处理")
        return

    total_errors = len(error_list)
    processed = 0
    modified = 0
    deleted = 0
    skipped = 0

    print("=" * 80)
    print("🎮 操作说明")
    print("  鼠标左键 → 将对应位置标签修改为 1")
    print("  鼠标右键 → 将对应位置标签修改为 0")
    print("  空格键 → 删除整个样本（移动到垃圾数据集）")
    print("  ESC键 → 退出程序")
    print("=" * 80)

    for error_data in error_list:
        if exit_flag:
            break

        # 🔥 这里已经是整数：100 而非 100.0
        img_idx = int(error_data["img_idx"])
        roi_pos = error_data["roi_pos"]
        real_label = error_data["real_label"]
        pred_label = error_data["pred_label"]
        point_size = error_data["point_size"]

        processed += 1
        print(f"\n📌 处理进度：{processed}/{total_errors}")
        print(f"样本：roi_{img_idx} | 错误ROI位置：{roi_pos}")
        print(f"真实标签：{real_label} | 预测标签：{pred_label} | point_size：{point_size}")

        # 路径拼接（100%匹配你的文件夹）
        roi_dir = os.path.join(DATASET_ROOT, "roi_images", f"roi_{img_idx}")
        label_path = os.path.join(DATASET_ROOT, "labels", f"label_{img_idx}.json")
        img_path = os.path.join(roi_dir, f"{roi_pos}.png")

        # 调试：打印真实路径
        print(f"🖼️ 图片路径：{img_path}")
        print(f"📄 标签路径：{label_path}")

        if not os.path.exists(img_path) or not os.path.exists(label_path):
            print(f"⚠️  文件不存在，跳过")
            skipped += 1
            continue

        # 读取标签
        try:
            with open(label_path, "r", encoding="utf-8") as f:
                ann = json.load(f)
            labels = ann["labels"]
            error_idx = roi_pos - 1
            current_label = labels[error_idx]
        except Exception as e:
            print(f"❌ 标签读取失败：{str(e)}")
            skipped += 1
            continue

        # 显示图片
        img = cv2.imread(img_path)
        img_display = cv2.resize(img, DISPLAY_SIZE)
        window_title = f"样本:{img_idx} | ROI:{roi_pos} | 当前标签:{current_label}"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.imshow(window_title, img_display)
        cv2.setMouseCallback(window_title, mouse_callback)

        click_result = None
        is_clicked = False
        space_clicked = False

        # 等待操作
        while True:
            key = cv2.waitKey(10)
            if key == 27:
                exit_flag = True
                break
            if key == 32:
                space_clicked = True
                break
            if is_clicked:
                break

        cv2.destroyAllWindows()
        if exit_flag:
            break

        # 删除样本
        if space_clicked:
            if delete_whole_sample(img_idx):
                deleted += 1
            continue

        # 修改标签
        if click_result is not None:
            labels[error_idx] = click_result
            ann["labels"] = labels
            with open(label_path, "w", encoding="utf-8") as f:
                json.dump(ann, f, ensure_ascii=False, indent=2)
            modified += 1
            print(f"✅ 标签已修改：{current_label} → {click_result}")

    print("\n" + "=" * 80)
    print(f"🎯 处理完成！总错误：{total_errors} | 已修正：{modified} | 已删除：{deleted} | 跳过：{skipped}")
    print("=" * 80)


if __name__ == "__main__":
    process_csv_errors()