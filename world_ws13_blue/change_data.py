import re
import os

def process_urdf_log(input_file: str, output_file: str):
    """
    读取URDF生成日志文件，提取随机数并转换为C++二维vector（0不变，非0转1）
    :param input_file: 输入日志txt路径
    :param output_file: 输出C++ vector格式txt路径
    """
    # 正则表达式：匹配 第x组随机数：(数字,数字,...) 中的数字部分
    pattern = re.compile(r'第\d+组随机数：\((.*?)\)')
    binary_matrix = []

    try:
        # 读取输入文件
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 遍历每一行提取数据
        for line in lines:
            match = pattern.search(line.strip())
            if match:
                # 提取括号内的数字字符串，分割并转为整数
                num_str = match.group(1)
                num_list = [int(num.strip()) for num in num_str.split(',')]
                
                # 处理数字：0不变，非0转为1
                binary_row = [0 if num == 0 else 1 for num in num_list]
                binary_matrix.append(binary_row)

        if not binary_matrix:
            print("❌ 未提取到任何随机数数据！")
            return

        # 生成 C++ 二维vector 格式字符串
        cpp_vector = "// 自动生成的二进制二维数组（0不变，非0=1）\n"
        cpp_vector += "std::vector<std::vector<int>> data = {\n"
        
        # 拼接每一行数据
        for row in binary_matrix:
            row_str = "    {" + ", ".join(map(str, row)) + "},\n"
            cpp_vector += row_str
        
        # 去除最后一行多余的逗号，保证语法正确
        cpp_vector = cpp_vector.rstrip(',\n') + "\n};\n"

        # 写入输出文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(cpp_vector)

        print(f"✅ 处理完成！")
        print(f"📊 共提取 {len(binary_matrix)} 组数据，每组 {len(binary_matrix[0])} 个数字")
        print(f"💾 C++ vector 已保存到：{os.path.abspath(output_file)}")

    except FileNotFoundError:
        print(f"❌ 错误：未找到文件 {input_file}，请检查路径是否正确！")
    except Exception as e:
        print(f"❌ 处理失败：{str(e)}")

if __name__ == "__main__":
    print("="*50)
    print("URDF日志转C++二维Vector工具")
    print("="*50)
    
    # 获取用户输入
    input_path = "/home/h/RC2026/world_ws13/src/zwei/map1_add/label_400.txt"
    output_path = "/home/h/RC2026/world_ws13/src/zwei/map1_add/01_400.txt"
    
    # 执行处理
    process_urdf_log(input_path, output_path)