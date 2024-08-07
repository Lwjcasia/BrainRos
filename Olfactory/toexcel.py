import pandas as pd

# 读取文本文件
file_path = 'data.txt'  # 替换为您的文件路径
data = []

with open(file_path, 'r') as file:
    for line in file:
        # 忽略空行
        if line.strip():
            # 分解每行并提取数值
            parts = line.split()
            values = [int(part) for part in parts if ":" not in part]
            # 添加额外的0列
            values.append(1)
            data.append(values)

# 创建DataFrame
df = pd.DataFrame(data, columns=['MQ138', 'MQ135', 'MQ8', 'MQ3', 'No Gas'])

# 将DataFrame写入Excel文件
excel_path = 'output.xlsx'  # 您希望保存的Excel文件名
df.to_excel(excel_path, index=False)
