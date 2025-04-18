import os

def convert_label(label_line):
    # 分割输入行
    parts = label_line.split()
    if len(parts) != 10:  # 确保是 8 个角点坐标 + 类别 ID + 颜色 ID
        raise ValueError(f"Invalid label format: {label_line}")

    # 解析输入数据
    keypoints = list(map(float, parts[:8]))  # 前八位为角点的 x 和 y 坐标
    category = int(parts[8])  # 倒数第二位为类别 ID
    color = int(parts[9])     # 最后一位为颜色 ID

    # 新类别 = 颜色 * 9 + 类别
    new_category = color * 9 + category

    # 计算外接矩形的中心点和半宽高
    x_coords = keypoints[::2]  # 角点的 x 坐标
    y_coords = keypoints[1::2]  # 角点的 y 坐标
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    half_width = (x_max - x_min) 
    half_height = (y_max - y_min) 

    # 构造新标注行
    new_label = f"{new_category} {x_center:.6f} {y_center:.6f} {half_width:.6f} {half_height:.6f} " + \
                " ".join(f"{k:.6f}" for k in keypoints)
    return new_label

def process_label_file(file_path):
    # 读取文件内容
    with open(file_path, "r") as f:
        lines = f.readlines()
    # 打开同一个文件覆盖写入转换后的标注
    with open(file_path, "w") as f:
        for line in lines:
            line = line.strip()
            if line:  # 跳过空行
                new_line = convert_label(line)
                f.write(new_line + "\n")

def batch_process_labels(label_dir):
    # 遍历文件夹中所有 .txt 文件，直接在原文件上修改
    for filename in os.listdir(label_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(label_dir, filename)
            process_label_file(file_path)
    print("所有标注文件已直接在原文件上转换完成")

# 主程序
label_dir = "/mnt/d/netgroup/数据集/烧饼1剩余_已标_未处理/1/labels"  # 替换为你的标注文件夹路径
batch_process_labels(label_dir)
