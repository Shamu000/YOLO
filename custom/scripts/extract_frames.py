import os
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def extract_frames_single_video(video_path, output_dir, interval=1, start_idx=0):
    """
    单个视频提取帧
    :param video_path: 输入视频路径
    :param output_dir: 保存图片的统一输出目录
    :param interval: 每隔多少帧保存一次
    :param start_idx: 保存图片的起始编号（防止重复）
    :return: 保存的图片数量
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ 无法打开视频文件: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    saved_count = 0

    pbar = tqdm(total=total_frames, unit="frame", desc=f"切帧 {os.path.basename(video_path)}", position=start_idx)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0:
            img_name = f"{os.path.splitext(os.path.basename(video_path))[0]}_frame_{frame_count:06d}.png"
            img_path = os.path.join(output_dir, img_name)
            cv2.imwrite(img_path, frame)
            saved_count += 1

        frame_count += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    print(f"✅ 完成 {os.path.basename(video_path)}，提取 {saved_count} 张图片")
    return saved_count

def batch_extract_frames_multithread(input_dir, output_dir, interval=1, max_workers=4):
    """
    批量多线程提取整个目录下所有视频
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_extensions = ('.avi', '.mp4', '.mov', '.mkv', '.flv')
    video_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(video_extensions)]

    futures = []
    total_saved = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, video_path in enumerate(video_paths):
            future = executor.submit(extract_frames_single_video, video_path, output_dir, interval, idx)
            futures.append(future)

        for future in as_completed(futures):
            total_saved += future.result()

    print(f"🎉 全部处理完成，总共提取 {total_saved} 张图片。")

if __name__ == "__main__":
    # === 配置部分 ===
    input_dir = "/home/hithcat/Code/CRT/测试集/原视频"      # 视频目录
    output_dir = "/home/hithcat/Code/CRT/测试集/标注"    # 图片保存目录
    interval = 5     # 每隔多少帧保存一张
    max_workers = 16  # 线程数，建议 = CPU核心数 或 核心数x2

    batch_extract_frames_multithread(input_dir, output_dir, interval, max_workers)
