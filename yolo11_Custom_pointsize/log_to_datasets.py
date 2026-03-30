import os
import shutil
from typing import List

def get_deepest_folders(root_dir: str) -> List[str]:
    """
    遍历根目录，获取所有【最深层文件夹】（无下级子文件夹的叶节点目录）
    :param root_dir: 源文件夹根路径
    :return: 所有最深层文件夹的绝对路径列表
    """
    deepest_folders = []
    # 遍历所有目录（深度优先）
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 关键判断：没有子文件夹 → 最深层
        if not dirnames:
            deepest_folders.append(dirpath)
    return deepest_folders

def process_files(source_folder: str, output_folder: str):
    """
    主处理函数：提取、重命名、复制文件
    :param source_folder: 输入源文件夹路径
    :param output_folder: 输出目标文件夹路径
    """
    # 1. 路径标准化（处理相对路径/绝对路径）
    source_dir = os.path.abspath(source_folder)
    output_dir = os.path.abspath(output_folder)

    # 2. 校验源文件夹是否存在
    if not os.path.isdir(source_dir):
        print(f"❌ 错误：源文件夹不存在 → {source_dir}")
        return

    # 3. 创建输出文件夹结构
    image_output_dir = os.path.join(output_dir, "global_images")
    label_output_dir = os.path.join(output_dir, "labels")
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(label_output_dir, exist_ok=True)
    print(f"✅ 输出目录创建完成：\n图片 → {image_output_dir}\n标签 → {label_output_dir}")

    # 4. 获取所有最深层文件夹
    deepest_dirs = get_deepest_folders(source_dir)
    if not deepest_dirs:
        print("⚠️ 未找到任何最深层文件夹")
        return
    print(f"🔍 找到 {len(deepest_dirs)} 个最深层文件夹")

    # 5. 遍历处理每个最深层文件夹
    success_count = 0
    for index, folder_path in enumerate(deepest_dirs):
        # 定义两个目标文件路径
        img_src = os.path.join(folder_path, "image.png")
        txt_src = os.path.join(folder_path, "rt.txt")

        # 校验文件是否存在
        if not os.path.isfile(img_src) or not os.path.isfile(txt_src):
            print(f"⏭️  跳过：文件夹 {folder_path} 缺少必要文件")
            continue

        # 定义目标文件路径（按数字下标命名）
        img_dst = os.path.join(image_output_dir, f"{index}.png")
        txt_dst = os.path.join(label_output_dir, f"{index}.txt")

        # 复制文件（覆盖已存在文件）
        shutil.copy(img_src, img_dst)
        shutil.copy(txt_src, txt_dst)

        success_count += 1
        print(f"✅ 处理完成：下标 {index} | 来源 → {folder_path}")

    # 最终统计
    print(f"\n🎉 任务完成！成功处理 {success_count} 组文件")

if __name__ == "__main__":
    # ========== 【用户配置】直接修改这里的路径 ==========
    INPUT_FOLDER = r"输入你的源文件夹路径"  # 示例：D:\data\input
    OUTPUT_FOLDER = r"输入你的输出文件夹路径"  # 示例：D:\data\output
    # ================================================

    # 运行主程序
    process_files(INPUT_FOLDER, OUTPUT_FOLDER)