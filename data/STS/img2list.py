# 将slices文件夹中所有文件名去掉后缀写入到一个train_slices.list文件中
import os

def write_filenames_without_extension(folder_path, output_file):
    # 创建一个空列表来存储文件名
    filenames = []

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 去除文件后缀
        base_name = os.path.splitext(filename)[0]
        if "Label" in base_name:
            filenames.append(base_name)
    
    for filename in os.listdir(folder_path):
        # 去除文件后缀
        base_name = os.path.splitext(filename)[0]
        if "Unlabel" in base_name:
            filenames.append(base_name)

    # 将文件名写入到输出文件中
    with open(output_file, 'w') as f:
        for name in filenames:
            f.write(name + '\n')

# 指定文件夹路径和输出文件路径
folder_path = 'volumes'
output_file = 'val_test.list'

# 调用函数
write_filenames_without_extension(folder_path, output_file)

print(f"File names without extensions have been written to {output_file}")