import os


def find_labels_with_class(label_dir, target_classes):
    """
    遍历指定目录下所有标签文件，找到包含指定类别的文件名

    :param label_dir: 标签文件所在的目录
    :param target_classes: 目标类别列表，例如 [7, 8, 25]
    :return: 包含目标类别的标签文件名列表
    """
    matched_files = []

    for filename in os.listdir(label_dir):
        file_path = os.path.join(label_dir, filename)

        # 只处理文件
        if os.path.isfile(file_path):
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.strip() == "":
                        continue
                    cls_id = int(line.split()[0])
                    if cls_id in target_classes:
                        matched_files.append(filename)
                        break  # 找到一个就可以，不需要继续读这个文件

    return matched_files


if __name__ == "__main__":
    # 指定你的标签目录
    label_directory = "/home/wyc/Yolo11/datasets/0422测试集拍摄2/labels/train"  # <<< 修改成你的实际路径

    # 指定你想找的类别
    target_class_ids = [14]

    result_files = find_labels_with_class(label_directory, target_class_ids)

    # 打印结果
    print("包含指定类别的标签文件有：")
    for file in result_files:
        print(file)
