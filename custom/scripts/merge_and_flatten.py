import os
import shutil
from glob import glob

def move_and_merge_flat(src_img_dir, src_lbl_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(src_img_dir, f'*{ext}')))

    for img_path in image_files:
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        lbl_name = base_name + '.txt'

        src_lbl_path = os.path.join(src_lbl_dir, lbl_name)
        dst_img_path = os.path.join(dst_dir, img_name)
        dst_lbl_path = os.path.join(dst_dir, lbl_name)

        # 移动图像
        shutil.move(img_path, dst_img_path)

        # 移动标签（如果存在）
        if os.path.exists(src_lbl_path):
            shutil.move(src_lbl_path, dst_lbl_path)
        else:
            print(f'[WARNING] 标签不存在: {src_lbl_path}')

def main():
    # ✅ 在这里指定你的数据集根目录
    dataset_root = "/home/hithcat/Code/CRT/TraditionalLabel/frames/temp"  # TODO: 替换为你的路径，例如 "/home/user/dataset"

    dst_dir = dataset_root  # 混合后文件统一放在这个目录中

    img_train_dir = os.path.join(dataset_root, 'images', 'train')
    img_val_dir = os.path.join(dataset_root, 'images', 'val')
    lbl_train_dir = os.path.join(dataset_root, 'labels', 'train')
    lbl_val_dir = os.path.join(dataset_root, 'labels', 'val')

    print("[INFO] 正在处理训练集...")
    move_and_merge_flat(img_train_dir, lbl_train_dir, dst_dir)

    print("[INFO] 正在处理验证集...")
    move_and_merge_flat(img_val_dir, lbl_val_dir, dst_dir)

    print("[INFO] 删除原始目录...")
    shutil.rmtree(os.path.join(dataset_root, 'images'), ignore_errors=True)
    shutil.rmtree(os.path.join(dataset_root, 'labels'), ignore_errors=True)

    print("[SUCCESS] 图像与标签已混合并统一放入根目录。")

if __name__ == "__main__":
    main()
