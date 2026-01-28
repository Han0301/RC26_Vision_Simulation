import os
import re
import json
import cv2
import numpy as np
import argparse
from pathlib import Path


def extract_idx_number(filename):
    """从文件名中提取idx后的数字（适配idx1/Idx1等格式）"""
    match = re.search(r'idx(\d+)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def load_roi_images(roi_folder):
    """加载单个roi_x文件夹下的12张ROI图片，按idx数字排序（1-12）"""
    # 初始化12个位置的图片（缺失用黑图填充，统一尺寸200x200）
    roi_imgs = [np.zeros((200, 200, 3), dtype=np.uint8) for _ in range(12)]
    roi_filenames = [f"idx{i + 1}_missing.png" for i in range(12)]

    # 遍历文件夹下的png文件
    for filename in os.listdir(roi_folder):
        if not filename.lower().endswith('.png'):
            continue
        idx_num = extract_idx_number(filename)
        if idx_num is None or idx_num < 1 or idx_num > 12:
            print(f"⚠️ 跳过无效文件：{filename}（无有效idx数字）")
            continue

        # 加载图片
        img_path = os.path.join(roi_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ 无法读取图片：{filename}，用黑图替代")
            continue

        # 调整图片尺寸为统一大小（方便拼接）
        img = cv2.resize(img, (200, 200))
        roi_imgs[idx_num - 1] = img
        roi_filenames[idx_num - 1] = filename

    return roi_imgs, roi_filenames


def stitch_roi_images(roi_imgs):
    """将12张ROI图片拼接成3行4列的全局图"""
    # 分3行，每行4张
    row1 = np.hstack(roi_imgs[0:4])
    row2 = np.hstack(roi_imgs[4:8])
    row3 = np.hstack(roi_imgs[8:12])
    # 拼接成整体
    stitched_img = np.vstack([row1, row2, row3])
    return stitched_img


def save_stitched_image(stitched_img, save_path):
    """保存拼接后的图片到指定路径"""
    cv2.imwrite(save_path, stitched_img)
    print(f"📷 拼接后的ROI图片已保存至：{save_path}")


def get_user_input(prompt, input_name):
    """获取用户输入并校验（12个0/1数字，空格分隔）"""
    while True:
        user_input = input(prompt).strip()
        # 分割成列表
        input_list = user_input.split()
        # 校验长度
        if len(input_list) != 12:
            print(f"❌ 错误：{input_name}必须输入12个数字！当前输入了{len(input_list)}个")
            continue
        # 校验是否为0/1
        try:
            input_int = [int(num) for num in input_list]
            if not all([num in [0, 1] for num in input_int]):
                print(f"❌ 错误：{input_name}只能包含0或1！")
                continue
            return input_int
        except ValueError:
            print(f"❌ 错误：{input_name}必须是数字！")
            continue


def extract_roi_number(folder_name):
    """从roi_x文件夹名中提取数字x"""
    match = re.search(r'roi_(\d+)', folder_name)
    if match:
        return int(match.group(1))
    return None


def main(input_folder, output_folder):
    # 1. 创建输出文件夹和临时图片目录（不存在则创建）
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_img_dir = output_dir / "temp_stitched_images"
    temp_img_dir.mkdir(parents=True, exist_ok=True)

    # 2. 遍历输入文件夹下的roi_x子文件夹，按数字排序
    roi_folders = []
    for item in os.listdir(input_folder):
        item_path = os.path.join(input_folder, item)
        if os.path.isdir(item_path) and item.startswith('roi_'):
            roi_num = extract_roi_number(item)
            if roi_num is not None:
                roi_folders.append((roi_num, item_path))

    # 按roi数字升序排序
    roi_folders.sort(key=lambda x: x[0])
    if not roi_folders:
        print("❌ 未找到任何roi_x格式的子文件夹！")
        return

    # 3. 处理每个roi_x文件夹
    for roi_num, roi_path in roi_folders:
        print(f"\n==================== 处理 roi_{roi_num} ====================")

        # 加载12张ROI图片
        print(f"📸 加载{roi_path}下的ROI图片...")
        roi_imgs, roi_filenames = load_roi_images(roi_path)

        # 拼接图片并保存（替代显示窗口）
        stitched_img_path = temp_img_dir / f"roi_{roi_num}_stitched.png"
        stitched_img = stitch_roi_images(roi_imgs)
        save_stitched_image(stitched_img, str(stitched_img_path))

        # 提示用户查看图片后输入
        print(f"\n请先查看拼接图片：{stitched_img_path}")
        input("确认查看完成后，按Enter键继续输入标签...")

        # 获取用户输入labels
        print("\n请输入labels（12个数字，0/1，空格分隔）：")
        labels = get_user_input("labels: ", "labels")

        # 获取用户输入roi_valid_mask
        print("\n请输入roi_valid_mask（12个数字，0/1，空格分隔）：")
        roi_valid_mask = get_user_input("roi_valid_mask: ", "roi_valid_mask")

        # 4. 生成json文件
        json_content = {
            "labels": labels,
            "roi_valid_mask": roi_valid_mask
        }
        json_path = output_dir / f"label_{roi_num}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_content, f, ensure_ascii=False, indent=4)

        print(f"✅ 成功生成标签文件：{json_path}")

    print("\n🎉 所有文件夹处理完成！")
    print(f"📁 标签文件保存至：{output_dir}")
    print(f"🖼️ 拼接图片保存至：{temp_img_dir}（可手动删除）")


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="交互式生成ROI标签JSON文件（无GUI依赖）")
    parser.add_argument("--input", required=True, help="输入文件夹路径（包含roi_1/roi_2等子文件夹）")
    parser.add_argument("--output", required=True, help="输出文件夹路径（生成label_1.json等文件）")
    args = parser.parse_args()

    # 校验输入文件夹是否存在
    if not os.path.isdir(args.input):
        print(f"❌ 输入文件夹不存在：{args.input}")
    else:
        main(args.input, args.output)