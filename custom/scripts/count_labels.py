import os

# 类别名称映射，类别的编号（0到26）对应具体的类别名称
CATEGORY_NAMES = {
    0: 'Blue Sentinel', 1: 'Blue Hero 1', 2: 'Blue Engineer 2', 3: 'Blue Unit 3',
    4: 'Blue Unit 4', 5: 'Blue Unit 5', 6: 'Blue Outpost', 7: 'Blue Base',
    8: 'Blue Base Armor', 9: 'Red Sentinel', 10: 'Red Hero 1', 11: 'Red Engineer 2',
    12: 'Red Unit 3', 13: 'Red Unit 4', 14: 'Red Unit 5', 15: 'Red Outpost',
    16: 'Red Base', 17: 'Red Base Armor', 18: 'Off Sentinel', 19: 'Off Hero 1',
    20: 'Off Engineer 2', 21: 'Off Unit 3', 22: 'Off Unit 4', 23: 'Off Unit 5',
    24: 'Off Outpost', 25: 'Off Base', 26: 'Off Base Armor'
}


def count_labels_in_dataset(label_dir):
    """
    统计每个标签出现的次数

    参数:
        label_dir (str): 标签文件所在的目录

    返回:
        dict: 每个类别出现的次数
    """
    label_counts = {i: 0 for i in range(27)}  # 初始化一个包含0到26类别的计数器，27个类别
    for label_file in os.listdir(label_dir):
        label_file_path = os.path.join(label_dir, label_file)

        if os.path.isfile(label_file_path) and label_file.endswith('.txt'):  # 确保是标签文件
            with open(label_file_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    # 获取每行标签的类别（第一个数字）
                    category = int(line.strip().split()[0])  # 假设标签格式是每行数字，以空格分隔
                    if category in label_counts:
                        label_counts[category] += 1

    return label_counts


def main(dataset_root):
    # 检查是否为特殊的 "测试集"
    if dataset_root == "/home/wyc/Yolo11/datasets/测试集":
        # 如果是测试集，只需要处理 labels 文件夹
        test_label_dir = os.path.join(dataset_root, "labels")  # 测试集标签路径
        label_counts = count_labels_in_dataset(test_label_dir)

        print("测试集标签统计:")
        for category, count in label_counts.items():
            if count > 0:
                print(f"{CATEGORY_NAMES.get(category, 'Unknown Category')} : {count} 次")
    else:
        # 自动补全训练集和验证集标签路径
        train_label_dir = os.path.join(dataset_root, "labels/train")  # 训练集标签路径
        val_label_dir = os.path.join(dataset_root, "labels/val")  # 验证集标签路径

        # 统计训练集和验证集的标签
        train_label_counts = count_labels_in_dataset(train_label_dir)
        val_label_counts = count_labels_in_dataset(val_label_dir)

        print("训练集标签统计:")
        for category, count in train_label_counts.items():
            if count > 0:
                print(f"{CATEGORY_NAMES.get(category, 'Unknown Category')} : {count} 次")

        print("\n验证集标签统计:")
        for category, count in val_label_counts.items():
            if count > 0:
                print(f"{CATEGORY_NAMES.get(category, 'Unknown Category')} : {count} 次")


if __name__ == "__main__":
    dataset_root = "/home/wyc/Yolo11/datasets/测试集"  # 提示用户输入数据集根路径
    main(dataset_root)
