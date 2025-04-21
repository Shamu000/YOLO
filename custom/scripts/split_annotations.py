import os
import shutil

# 设置文件夹路径
source_dir = "/home/marshal_kim/下载/0420蓝3拍摄"  # 数据集文件夹路径
images_dir = os.path.join(source_dir, "images")
labels_dir = os.path.join(source_dir, "labels")

# 创建目标文件夹
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# 定义图片和文本的后缀
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
txt_extension = ".txt"

# 遍历文件夹并分类文件
for filename in os.listdir(source_dir):
    filepath = os.path.join(source_dir, filename)
    if os.path.isfile(filepath):  # 判断是否是文件
        file_ext = os.path.splitext(filename)[-1].lower()
        if file_ext in image_extensions:
            shutil.move(filepath, os.path.join(images_dir, filename))  # 移动图片
        elif file_ext == txt_extension:
            shutil.move(filepath, os.path.join(labels_dir, filename))  # 移动txt文件

print("文件分类完成")
