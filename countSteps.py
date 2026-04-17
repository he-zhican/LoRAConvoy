import os

# ---------------------- 核心配置（请根据实际路径修改） ----------------------
# 文本文件所在根目录（替换为你的position_error_*.txt文件实际路径）
TXT_ROOT_PATH = r"D:/projects/LoRAConvoy/results/llama70B_decision_result"
# 文件编号范围（0-49）
FILE_NUM_RANGE = range(4)  # 0到49
# 是否过滤空行（True=统计非空行，False=统计所有行，包括空行）
FILTER_EMPTY_LINES = True


# ---------------------- 统计函数 ----------------------
def count_file_lines(file_path, filter_empty):
    """
    统计单个文件的行数
    :param file_path: 文件路径
    :param filter_empty: 是否过滤空行
    :return: 行数
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if filter_empty:
                # 过滤空行（去除换行符后为空的行）
                lines = [line for line in lines if line.strip() != ""]
            return len(lines)
    except Exception as e:
        print(f"❌ 读取文件失败 {file_path}：{str(e)}")
        return -1  # 标记读取失败


def calculate_average_lines():
    # 初始化统计变量
    total_files = 0  # 有效文件数（成功读取的文件）
    total_lines = 0  # 所有文件总行数
    file_line_dict = {}  # 记录每个文件的行数 {文件名: 行数}

    # 遍历0-49号position_error文件
    for file_num in FILE_NUM_RANGE:
        # 拼接文件路径
        file_name = f"position_error_{file_num}.txt"
        file_path = os.path.join(TXT_ROOT_PATH, file_name)

        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"⚠️ 文件不存在，跳过：{file_name}")
            file_line_dict[file_name] = "不存在"
            continue

        # 统计当前文件行数
        line_count = count_file_lines(file_path, FILTER_EMPTY_LINES)
        if line_count == -1:
            file_line_dict[file_name] = "读取失败"
            continue

        # 更新统计变量
        total_files += 1
        total_lines += line_count
        file_line_dict[file_name] = line_count

        # 打印单个文件行数（可选）
        print(f"📄 {file_name}：{line_count} 行")

    # 计算平均值
    if total_files == 0:
        avg_lines = 0
        print("\n⚠️ 无有效文件可统计！")
    else:
        avg_lines = total_lines / total_files

    # 输出汇总结果
    print("\n" + "=" * 50)
    print("📊 position_error_*.txt 文件行数统计汇总")
    print("=" * 50)
    print(f"文件总数（0-49）：{len(FILE_NUM_RANGE)}")
    print(f"有效文件数（成功读取）：{total_files}")
    print(f"所有有效文件总行数：{total_lines}")
    print(f"平均行数：{avg_lines:.2f} 行（保留2位小数）")
    print("=" * 50)

    # 可选：输出所有文件的行数明细（方便核对）
    # print("\n📋 各文件行数明细：")
    # for fname, lcount in file_line_dict.items():
    #     print(f"{fname:20s}：{lcount}")


# ---------------------- 执行统计 ----------------------
if __name__ == "__main__":
    calculate_average_lines()