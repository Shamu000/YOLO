import os
from PIL import Image
from multiprocessing import Pool, cpu_count


def convert_to_png(task):
    file_path, output_dir = task
    filename = os.path.basename(file_path)
    ext = filename.split('.')[-1].lower()

    if ext not in ['jpg', 'jpeg', 'bmp', 'gif', 'tiff', 'webp']:
        return None  # 不是支持的图片格式

    try:
        img = Image.open(file_path)
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.png")
        img.save(output_path, 'PNG')
        print(f"Converted: {filename}")
    except Exception as e:
        print(f"Error converting {filename}: {e}")


def convert_images_to_png_parallel(input_dir, output_dir, num_processes=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tasks = []
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if os.path.isfile(file_path):
            tasks.append((file_path, output_dir))

    # 设置进程数（默认使用 CPU 核心数）
    num_processes = num_processes or cpu_count()

    with Pool(num_processes) as pool:
        pool.map(convert_to_png, tasks)


if __name__ == "__main__":
    input_dir = '/home/wyc/Downloads/0422测试集拍摄'
    output_dir = '/home/wyc/Downloads/0422测试集拍摄PNG'

    convert_images_to_png_parallel(input_dir, output_dir)
