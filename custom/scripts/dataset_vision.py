import os
import cv2
from pathlib import Path
import yaml

# ==========================================
# 👇 验证开关：如果发现点上下对称反了，请改为 True
FLIP_Y = False
# ==========================================

def load_category_names():
    """从 datasets.yaml 读取类别名映射"""
    yaml_path = Path(__file__).resolve().parent.parent / "cfg" / "datasets.yaml"
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg.get('names', {})

CATEGORY_NAMES = load_category_names()

def draw_yolo_pose(image_path, label_path):
    """
    极简模式：读取YOLO-Pose并绘制小点与序号
    """
    # 1. 读取图片
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[警告] 无法读取图片: {image_path}")
        return
    h, w, _ = img.shape

    # 2. 读取并解析标注文件
    if not os.path.exists(label_path):
        print(f"[警告] 未找到对应的标注文件: {label_path}")
        return

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        # YOLO-Pose 至少需要 1个类ID + 4个边界框值 + 2个关键点坐标值 = 7
        if len(parts) < 7: 
            continue

        # 提取边界框 (归一化坐标: cx, cy, w, h)
        class_id = int(parts[0])
        cx = float(parts[1])
        cy = float(parts[2])
        bw = float(parts[3])
        bh = float(parts[4])

        # 转换为绝对像素坐标
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        # 绘制检测框
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # 标注类别 ID
        cls_name = CATEGORY_NAMES.get(class_id, f"cls:{class_id}")
        cv2.putText(img, f"{cls_name}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # 提取关键点 (从索引 5 开始，每2个数字为一组 x, y)
        num_keypoints = (len(parts) - 5) // 2
        
        for i in range(num_keypoints):
            x_idx = 5 + i * 2
            y_idx = 5 + i * 2 + 1
            
            x = float(parts[x_idx])
            y = float(parts[y_idx])
            
            # 【核心验证逻辑】如果开启开关，沿水平中线翻转 Y 坐标
            if FLIP_Y:
                y = 1.0 - y
            
            # 转换为绝对像素坐标
            px = int(x * w)
            py = int(y * h)
            
            # 【极简绘制】只画一个小实心点
            cv2.circle(img, (px, py), 2, (255, 255, 255), -1)
            
            # 【极简标注】直接写序号，无背景框
            text = str(i + 1)
            cv2.putText(img, text, (px + 3, py - 3), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # 3. 显示结果
    window_name = f"Pose Viewer: {image_path.name}"
    cv2.imshow(window_name, img)
    print(f"[信息] 正在显示: {image_path.name} (按任意键切换下一张，按 'q' 退出)")
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyWindow(window_name)
    return key == ord('q')

def main(dataset_dir):
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        print(f"[错误] 数据集路径不存在: {dataset_dir}")
        return

    image_files = sorted(list(dataset_path.glob("*.png")))
    if not image_files:
        print(f"[信息] 在 {dataset_dir} 中未找到 .png 文件")
        return

    for img_path in image_files:
        txt_path = img_path.with_suffix(".txt")
        should_quit = draw_yolo_pose(img_path, txt_path)
        if should_quit:
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    DATASET_PATH = "/home/hithcat/Code/CRT/TraditionalLabel/datas/26labeled/正式比赛增亮"
    main(DATASET_PATH)