
import re
import pandas as pd

# 假设你的文本文件路径是'text_data.txt'
text_file_path = 'data.txt'

# 读取文本文件
with open(text_file_path, 'r') as file:
    text_data = file.read()

# 使用正则表达式找到所有"F = "后的数字
f_values = re.findall(r'F = (\d+)', text_data)

# 将找到的数字转换为int类型
f_values_int = [int(value) for value in f_values]

# 转换为DataFrame
df = pd.DataFrame(f_values_int, columns=['F'])

# 指定Excel文件路径
excel_file_path = 'data.xlsx'

# 保存到Excel文件
df.to_excel(excel_file_path, index=False, engine='openpyxl')

# 输出提示信息
print(f'数据已保存到Excel文件: {excel_file_path}')