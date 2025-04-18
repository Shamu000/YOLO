import os
import cv2
import numpy as np

def visualize_annotations(image_folder, annotation_folder):
    # 定义类别映射字典
    category_mapping = {
        0: "BG",
        1: "B1",
        2: "B2",
        3: "B3",
        4: "B4",
        5: "B5",
        6: "BO",
        7: "BBs",
        8: "BBb",
        9: "RG",
        10: "R1",
        11: "R2",
        12: "R3",
        13: "R4",
        14: "R5",
        15: "RO",
        16: "RBs",
        17: "RBb",
        18: "NG",
        19: "N1",
        20: "N2",
        21: "N3",
        22: "N4",
        23: "N5",
        24: "NO",
        25: "NBs",
        26: "NBb"
    }

    # 获取所有图片文件名
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    for image_file in image_files:
        # 加载图片
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_file}")
            continue

        # 加载对应的标注文件（要求图片和标注文件同名，只是扩展名不同）
        annotation_file = os.path.join(annotation_folder, os.path.splitext(image_file)[0] + '.txt')
        if not os.path.exists(annotation_file):
            print(f"Annotation file not found for image: {image_file}")
            continue

        with open(annotation_file, 'r') as f:
            lines = f.readlines()

        height, width, _ = image.shape

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 13:
                print(f"Invalid annotation format in file: {annotation_file}")
                continue

            # 提取类别ID并通过映射字典得到标签
            try:
                category_id = int(parts[0])
            except:
                print(f"Invalid category id in file: {annotation_file}")
                continue
            label_str = category_mapping.get(category_id, "Unknown")
            
            # 根据类别标签首字母确定绘制颜色：B-蓝、R-红、N-绿，否则白色
            if label_str.startswith("B"):
                box_color = (255, 0, 0)  # 蓝色 (BGR: 蓝)
            elif label_str.startswith("R"):
                box_color = (0, 0, 255)  # 红色 (BGR: 红)
            elif label_str.startswith("N"):
                box_color = (0, 255, 0)  # 绿色 (BGR: 绿)
            else:
                box_color = (255, 255, 255)  # 默认白色

            # 提取归一化后的角点坐标（后8个数）并转换为像素坐标
            corners = np.array(parts[5:], dtype=float).reshape(-1, 2)  # 4个角点
            corners[:, 0] *= width
            corners[:, 1] *= height

            # 绘制角点（可选，用来辅助查看角点位置）
            for corner in corners:
                cv2.circle(image, tuple(corner.astype(int)), 5, box_color, -1)

            # 绘制四边形：按顺序连接角点
            for i in range(4):
                start_point = tuple(corners[i].astype(int))
                end_point = tuple(corners[(i + 1) % 4].astype(int))
                cv2.line(image, start_point, end_point, box_color, 2)

            # 计算边界框左上角（最小的x和y）作为文本显示的位置
            x_min = int(np.min(corners[:, 0]))
            y_min = int(np.min(corners[:, 1]))
            cv2.putText(image, label_str, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

        # 显示图片
        cv2.imshow('Visualization', image)
        key = cv2.waitKey(0)  # 等待按键
        if key == 27:  # Esc 键退出
            break

    cv2.destroyAllWindows()

# 替换为你的图片和标注文件夹路径喵~
image_folder = '/mnt/d/netgroup/数据集/烧饼1剩余_已标_未处理/1/images/train'
annotation_folder = '/mnt/d/netgroup/数据集/烧饼1剩余_已标_未处理/1/labels/train'
visualize_annotations(image_folder, annotation_folder)