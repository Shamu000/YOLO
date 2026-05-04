import os
import random
import shutil

def ensure_directories_exist(base_dir):
    """确保 images/train, images/val, labels/train, labels/val 目录存在"""
    for subdir in ["train", "val"]:
        os.makedirs(os.path.join(base_dir, "images", subdir), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "labels", subdir), exist_ok=True)

def split_data(base_dir, split_ratio=0.8):
    """按比例划分数据并复制到对应文件夹"""
    images_dir = os.path.join(base_dir, "images")
    labels_dir = os.path.join(base_dir, "labels")
    images_train = os.path.join(images_dir, "train")
    images_val = os.path.join(images_dir, "val")
    labels_train = os.path.join(labels_dir, "train")
    labels_val = os.path.join(labels_dir, "val")

    # 确保 train 和 val 目录存在
    ensure_directories_exist(base_dir)

    # 获取图片和标签
    images = sorted([f for f in os.listdir(images_dir) if f.endswith((".png", ".jpg", ".jpeg",".PNG",".JPEG",".JPG"))])
    labels = sorted([f for f in os.listdir(labels_dir) if f.endswith(".txt")])

    if len(images) != len(labels):
        raise ValueError("图片和标注文件数量不一致，请检查文件目录")

    # 确保文件名一一对应
    data = list(zip(images, labels))
    random.shuffle(data)  # 打乱顺序

    # 按比例划分
    train_size = int(len(data) * split_ratio)
    train_data = data[:train_size]
    val_data = data[train_size:]

    # 复制数据到对应文件夹
    for image, label in train_data:
        shutil.move(os.path.join(images_dir, image), os.path.join(images_train, image))
        shutil.move(os.path.join(labels_dir, label), os.path.join(labels_train, label))

    for image, label in val_data:
        shutil.move(os.path.join(images_dir, image), os.path.join(images_val, image))
        shutil.move(os.path.join(labels_dir, label), os.path.join(labels_val, label))
    
    print(f"数据已成功按 {split_ratio*100:.0f}% 训练集 / {100 - split_ratio*100:.0f}% 验证集 划分")

# 设置数据集根目录
base_dir = "/home/hithcat/Code/CRT/TraditionalLabel/datas/labeled/home/5.2"  # 替换为你的数据集路径

# 执行数据划分
split_data(base_dir, split_ratio=0.8)
