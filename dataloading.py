import os
import shutil

# 定义原始文件夹路径和目标文件夹路径
original_folder = 'data3/Animal/elephant'  # 原始文件夹路径
target_folder = 'data/train/elephant1'  # 目标文件夹路径

# 获取原始文件夹中的所有子文件夹
sub_folders = [f.path for f in os.scandir(original_folder) if f.is_dir()]

# 初始化文件计数器
file_count = 1

# 遍历每个子文件夹
for sub_folder in sub_folders:
    # 获取子文件夹中的所有图片文件
    image_files = [f.path for f in os.scandir(sub_folder) if f.is_file() and f.name.endswith('.jpg')]

    # 遍历每个图片文件
    for image_file in image_files:
        # 构建目标文件路径
        target_file = os.path.join(target_folder, f'{file_count}.jpg')

        # 将图片文件复制到目标文件夹，并按照数字重新命名
        shutil.copy(image_file, target_file)

        # 更新文件计数器
        file_count += 1