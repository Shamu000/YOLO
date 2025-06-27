import os

def remove_unpaired_files(image_folder, label_folder):
    # 获取图片和标注文件夹内的所有文件
    image_files = os.listdir(image_folder)
    label_files = os.listdir(label_folder)

    # 获取所有图片的基名（支持 .jpg 和 .png 格式，不区分大小写）
    image_bases = set()
    for f in image_files:
        if f.lower().endswith('.jpg') or f.lower().endswith('.png') or f.lower().endswith('.jpeg') or f.lower().endswith('.PNG') or f.lower().endswith('.JPG') or f.lower().endswith('.JPEG'):
            base = os.path.splitext(f)[0]
            image_bases.add(base)

    # 获取所有标注文件的基名（假设标注文件格式为 .txt）
    txt_files = set(f[:-4] for f in label_files if f.endswith('.txt'))

    # 找出只有图片或只有标注的基名
    unpaired_images = image_bases - txt_files  # 图片存在，但标注缺失
    unpaired_txt = txt_files - image_bases       # 标注存在，但图片缺失

    # 删除未配对的图片文件（检查同时删除 .jpg 和 .png 文件）
    for base in unpaired_images:
        for ext in [".png", ".jpg", ".jpeg",".PNG",".JPEG",".JPG"]:
            image_path = os.path.join(image_folder, base + ext)
            if os.path.exists(image_path):
                print(f"Removing unpaired image: {image_path}")
                os.remove(image_path)

    # 删除未配对的标注文件
    for base in unpaired_txt:
        txt_path = os.path.join(label_folder, base + '.txt')
        if os.path.exists(txt_path):
            print(f"Removing unpaired TXT: {txt_path}")
            os.remove(txt_path)

# 使用示例
image_folder = '/home/wyc/Yolo11/datasets/省赛内录/0330省赛内录/images'
label_folder = '/home/wyc/Yolo11/datasets/省赛内录/0330省赛内录/labels'
remove_unpaired_files(image_folder, label_folder)
