import pandas as pd
import json

# CSV文件路径
csv_file_path = r"E:\Deepseek\data_sorce\HZ\Data\第29批2801-2900\所有报告_汇总.csv"
output_json_path = r"E:\Deepseek\data_sorce\HZ\Data\第29批2801-2900\output_from_csv.json"

# 读取CSV文件
df = pd.read_csv(csv_file_path, encoding='utf-8')

# 创建字典来存储数据
all_data = {}
empty_records = []  # 用于记录空字段的信息

# 遍历每一行数据
for idx, row in df.iterrows():
    # 提取"意见"和"提示"字段
    question = str(row['意见']).strip() if pd.notna(row['意见']) else ""
    response = str(row['提示']).strip() if pd.notna(row['提示']) else ""

    # 获取文件名
    filename = str(row['文件名']) if pd.notna(row['文件名']) else f"未知文件_{idx + 1}"

    # 检查是否有空字段
    if not question or not response:
        empty_info = {
            "文件名": filename,
            "序号": idx + 1,
            "空字段": []
        }
        if not question:
            empty_info["空字段"].append("意见(question)为空")
        if not response:
            empty_info["空字段"].append("提示(response)为空")
        empty_records.append(empty_info)

    # 创建序号键（001, 002, 003...）
    key = str(idx + 1).zfill(3)

    # 存储数据
    all_data[key] = {
        "question": question,
        "response": response
    }

# 保存为JSON文件
with open(output_json_path, 'w', encoding='utf-8') as json_file:
    json.dump(all_data, json_file, ensure_ascii=False, indent=4)

print(f"CSV文件已处理完成，结果已保存到 {output_json_path}")
print(f"总共处理了 {len(all_data)} 条记录")

# 如果有空字段，输出详细信息
if empty_records:
    print(f"\n⚠️ 发现 {len(empty_records)} 条记录存在空字段：")
    print("=" * 80)
    for record in empty_records:
        print(f"文件名: {record['文件名']}")
        print(f"序号: {record['序号']}")
        print(f"问题: {', '.join(record['空字段'])}")
        print("-" * 80)
else:
    print("\n✓ 所有记录的question和response字段均不为空")