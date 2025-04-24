import os

# 类别名称映射
CATEGORY_NAMES = {
    0: 'Blue Sentinel', 1: 'Blue Hero 1', 2: 'Blue Engineer 2', 3: 'Blue Unit 3',
    4: 'Blue Unit 4', 5: 'Blue Unit 5', 6: 'Blue Outpost', 7: 'Blue Small Base',
    8: 'Blue Large Base', 9: 'Red Sentinel', 10: 'Red Hero 1', 11: 'Red Engineer 2',
    12: 'Red Unit 3', 13: 'Red Unit 4', 14: 'Red Unit 5', 15: 'Red Outpost',
    16: 'Red Small Base', 17: 'Red Large Base', 18: 'Off Sentinel', 19: 'Off Hero 1',
    20: 'Off Engineer 2', 21: 'Off Unit 3', 22: 'Off Unit 4', 23: 'Off Unit 5',
    24: 'Off Outpost', 25: 'Off Base', 26: 'Off Base Armor'
}


def count_labels_in_dataset(label_dir):
    """
    统计每个标签文件中类别编号的出现次数。
    """
    label_counts = {i: 0 for i in range(27)}
    for label_file in os.listdir(label_dir):
        label_path = os.path.join(label_dir, label_file)
        if os.path.isfile(label_path) and label_file.endswith('.txt'):
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        category = int(line.split()[0])
                        if category in label_counts:
                            label_counts[category] += 1
                    except Exception as e:
                        print(f"⚠️ 错误解析 {label_path}: {line} -> {e}")
    return label_counts


def main(datasets_root):
    # 总类别统计
    total_counts = {i: 0 for i in range(27)}

    # 对数据集文件夹排序
    for dataset_name in sorted(os.listdir(datasets_root)):
        dataset_path = os.path.join(datasets_root, dataset_name)
        if not os.path.isdir(dataset_path):
            continue

        print(f"\n📦 数据集：{dataset_name}")
        dataset_total = {i: 0 for i in range(27)}

        # 统计训练集
        train_labels = os.path.join(dataset_path, "labels/train")
        if os.path.exists(train_labels):
            train_counts = count_labels_in_dataset(train_labels)
            print("  🟢 训练集标签统计：")
            for i, count in train_counts.items():
                if count > 0:
                    print(f"    {CATEGORY_NAMES[i]}: {count}")
                    dataset_total[i] += count
                    total_counts[i] += count

        # 统计验证集
        val_labels = os.path.join(dataset_path, "labels/val")
        if os.path.exists(val_labels):
            val_counts = count_labels_in_dataset(val_labels)
            print("  🔵 验证集标签统计：")
            for i, count in val_counts.items():
                if count > 0:
                    print(f"    {CATEGORY_NAMES[i]}: {count}")
                    dataset_total[i] += count
                    total_counts[i] += count

    # 输出全部数据总计
    print("\n📊 所有数据集类别总计：")
    for i, count in total_counts.items():
        if count > 0:
            print(f"  {CATEGORY_NAMES[i]}: {count}")


if __name__ == "__main__":
    datasets_root = "/media/zjy/TOSHIBA EXT/330_RMUL_上午"
    main(datasets_root)
