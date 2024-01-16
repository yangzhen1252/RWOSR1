import os
from shutil import copy
import random

# 如果file不存在，创建file
def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)


# 获取data文件夹下所有除.txt文件以外所有文件夹名（即需要分类的类名）
# os.listdir()：用于返回指定的文件夹包含的文件或文件夹的名字的列表
# file_path = 'D:/pycharm/AlexNet/data_name'
file_path = './data'
pet_class = [cla for cla in os.listdir(file_path) if ".txt" not in cla]

# 创建训练集train文件夹，并由类名在其目录下创建子目录
mkfile('data/train')
for cla in pet_class:
    mkfile('data/train/' + cla)

# 创建验证集val文件夹，并由类名在其目录下创建子目录
mkfile('data/val')
for cla in pet_class:
    mkfile('data/val/' + cla)

# 划分比例，训练集 : 验证集 = 8 : 2
split_rate = 0.2

# 遍历所有类别的图像并按比例分成训练集和验证集
for cla in pet_class:
    # 某一类别的子目录
    cla_path = file_path + '/' + cla + '/'
    # iamges列表存储了该目录下所有图像的名称
    images = os.listdir(cla_path)
    num = len(images)
    # 从images列表中随机抽取k个图像名称
    # random.sample：用于截取列表的指定长度的随机数，返回列表
    # eval_index保存验证集val的图像名称
    eval_index = random.sample(images, k=int(num * split_rate))
    for index, image in enumerate(images):
        if image in eval_index:
            image_path = cla_path + image
            new_path = 'data/val/' + cla
            # copy()：将源文件的内容复制到目标文件或目录
            copy(image_path, new_path)

        # 其余图像保存在训练集train中
        else:
            image_path = cla_path + image
            new_path = 'data/train/' + cla
            copy(image_path, new_path)
        # '\r'回车，回到当前行的行首，而不会换到下一行，如果接着输出，本行以前的内容会被逐一覆盖
        # <模板字符串>.format(<逗号分隔的参数>)
        # end=""：将print自带的换行用end中指定的str代替
        print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")
    print()

print("processing done!")