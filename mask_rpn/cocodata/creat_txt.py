import os

def extract_filename(path):
    """Extracts the file name without extension from a given path."""
    # 获取路径中的文件名部分（包含扩展名）
    filename_with_extension = os.path.basename(path)
    # 分离文件名和扩展名
    filename, _ = os.path.splitext(filename_with_extension)
    return filename

def process_file(input_file, output_file):
    """Reads lines from the input file, processes each line, and writes the results to the output file."""
    with open(input_file, 'r') as file_in, open(output_file, 'w') as file_out:
        for line in file_in:
            # 处理每一行，提取文件名
            filename = extract_filename(line.strip())
            file_out.write(filename + '\n')

# 指定输入和输出文件的路径
input_file_path = '/disk1/ybh/data/train.txt'
output_file_path = '/disk1/ybh/datasets/COCO2VOC/VOCdevkit/COCO2014/ImageSets/Main/train.txt'

# 调用函数处理文件
process_file(input_file_path, output_file_path)
