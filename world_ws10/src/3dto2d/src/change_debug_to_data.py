import os
import shutil
import argparse
import re

def find_image_folders(root_dir):
    """
    递归遍历根目录，找到所有名称为image1、image2...的最深层文件夹
    :param root_dir: 待遍历的根目录
    :return: 列表，元素为image文件夹路径（按遍历顺序）
    """
    image_folders = []
    # 递归遍历所有目录
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 匹配当前目录名是否为image+数字（如image1、image100）
        dir_name = os.path.basename(dirpath)
        match = re.match(r'^image(\d+)$', dir_name)
        if match:
            # 检查是否是最深层文件夹（子目录为空）
            if not dirnames:
                image_folders.append(dirpath)
    return image_folders

def get_max_roi_index(output_root):
    """
    读取输出文件夹下已有的roi_数字文件夹，返回最大序号（无则返回0）
    :param output_root: 输出根目录
    :return: 最大序号（int），如已有roi_5、roi_10则返回10；无则返回0
    """
    # 输出文件夹不存在则先创建，返回0
    if not os.path.exists(output_root):
        os.makedirs(output_root, exist_ok=True)
        return 0
    
    max_idx = 0
    # 匹配roi_数字的正则（如roi_1、roi_100）
    roi_pattern = re.compile(r'^roi_(\d+)$')
    
    # 遍历输出文件夹下的所有子文件夹
    for item in os.listdir(output_root):
        item_path = os.path.join(output_root, item)
        if os.path.isdir(item_path):  # 仅处理文件夹
            match = roi_pattern.match(item)
            if match:
                # 提取数字并更新最大值
                current_idx = int(match.group(1))
                if current_idx > max_idx:
                    max_idx = current_idx
    return max_idx

def copy_images(image_folder, global_idx, output_root):
    """
    将image文件夹中idx开头、png结尾的图片复制到output_root/roi_全局序号文件夹
    :param image_folder: image文件夹路径
    :param global_idx: 全局累加序号（接续已有最大值后的序号）
    :param output_root: 输出根目录
    """
    # 构建唯一的输出文件夹路径：roi_全局累加序号（如roi_6、roi_7...）
    output_folder_name = f'roi_{global_idx}'
    output_folder = os.path.join(output_root, output_folder_name)
    # 创建输出文件夹（如果不存在，递归创建；存在则忽略）
    os.makedirs(output_folder, exist_ok=True)
    
    # 筛选符合条件的文件：idx开头 + .png结尾
    pattern = re.compile(r'^idx.*\.png$', re.IGNORECASE)  # ignorecase兼容大写PNG
    copied_count = 0
    
    for filename in os.listdir(image_folder):
        if pattern.match(filename):
            # 源文件完整路径
            src_path = os.path.join(image_folder, filename)
            # 目标文件完整路径
            dst_path = os.path.join(output_folder, filename)
            
            # 复制文件（可选：取消注释避免覆盖已有文件）
            # if os.path.exists(dst_path):
            #     print(f"文件已存在，跳过：{dst_path}")
            #     continue
            try:
                shutil.copy2(src_path, dst_path)  # copy2保留文件元数据
                copied_count += 1
                print(f"成功复制: {src_path} -> {dst_path}")
            except Exception as e:
                print(f"复制失败: {src_path} 错误: {str(e)}")
    
    print(f"\n【全局序号{global_idx}】文件夹 {image_folder} 处理完成：共复制 {copied_count} 张图片到 {output_folder}\n")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='筛选imageN文件夹中idx开头的png图片，接续输出文件夹已有roi序号生成新文件夹')
    parser.add_argument('--input', '-i', required=True, help='输入根文件夹路径（存储图像的多级文件夹）')
    parser.add_argument('--output', '-o', required=True, help='输出根文件夹路径（保存roi_累加序号文件夹）')
    args = parser.parse_args()
    
    # 检查输入文件夹是否存在
    if not os.path.isdir(args.input):
        print(f"错误：输入文件夹 {args.input} 不存在！")
        return
    
    # 1. 读取输出文件夹已有roi的最大序号
    max_exist_idx = get_max_roi_index(args.output)
    print(f"输出文件夹 {args.output} 中已有roi文件夹的最大序号：{max_exist_idx}")
    print(f"新的roi文件夹将从 {max_exist_idx + 1} 开始编号\n")
    
    # 2. 找到所有imageN最深层文件夹（按遍历顺序）
    print(f"开始遍历输入文件夹：{args.input}")
    image_folders = find_image_folders(args.input)
    if not image_folders:
        print("未找到任何image1/image2/...的最深层文件夹！")
        return
    print(f"共找到 {len(image_folders)} 个目标image文件夹，按遍历顺序生成接续序号：")
    # 打印遍历顺序（方便核对）
    for idx, folder in enumerate(image_folders, 1):
        new_idx = max_exist_idx + idx
        print(f"  接续序号{new_idx} → {folder}")
    print("")
    
    # 3. 按接续序号处理每个image文件夹
    for idx, folder in enumerate(image_folders, 1):
        global_idx = max_exist_idx + idx  # 新序号 = 已有最大值 + 遍历顺序
        copy_images(folder, global_idx, args.output)
    
    print("所有文件夹处理完成！")
    print(f"本次生成的roi文件夹列表：roi_{max_exist_idx + 1} ~ roi_{max_exist_idx + len(image_folders)}")

if __name__ == '__main__':
    main()