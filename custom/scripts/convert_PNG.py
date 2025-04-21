import os
from PIL import Image


def convert_images_to_png(input_dir, output_dir):
    # 检查输出目录是否存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        # 获取文件路径
        file_path = os.path.join(input_dir, filename)

        # 只处理文件，不处理子目录
        if os.path.isfile(file_path):
            # 获取文件扩展名并将其转化为小写
            ext = filename.split('.')[-1].lower()

            # 过滤非图片格式的文件
            if ext in ['jpg', 'jpeg', 'bmp', 'gif', 'tiff', 'webp']:
                # 打开图片
                try:
                    img = Image.open(file_path)

                    # 设置输出文件路径（替换扩展名为 png）
                    output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.png")

                    # 保存为 png 格式
                    img.save(output_path, 'PNG')
                    print(f"Converted: {filename} to PNG")
                except Exception as e:
                    print(f"Error converting {filename}: {e}")


if __name__ == "__main__":
    # 指定输入和输出路径
    input_dir = '/home/marshal_kim/下载/0420蓝3拍摄'  # 输入路径，替换为你的路径
    output_dir = '/home/marshal_kim/下载/0420蓝3拍摄PNG'  # 输出路径，替换为你的路径

    # 调用转换函数
    convert_images_to_png(input_dir, output_dir)
