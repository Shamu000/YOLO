import os
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def extract_frames_single_video(video_path, output_dir, interval=1, start_idx=0, start_frame=0, end_frame=0):
    """
    单个视频提取帧
    :param video_path: 输入视频路径
    :param output_dir: 保存图片的统一输出目录
    :param interval: 每隔多少帧保存一次
    :param start_idx: 保存图片的起始编号（防止重复）
    :param start_frame: 从视频的第几帧开始提取(默认0)
    :param end_frame: 提取到视频的第几帧结束(默认0,表示提取到结尾)
    :return: 保存的图片数量
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ 无法打开视频文件: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = end_frame if end_frame > 0 else total_frames
    # clamp start_frame 到有效范围并从该帧开始读取
    start_frame = max(0, min(start_frame, total_frames))
    frame_count = start_frame
    saved_count = 0

    # 将视频读取位置跳到 start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # 进度条总量按实际要读取的帧数计算
    pbar_total = max(0, end_frame - start_frame + 1)
    pbar = tqdm(total=pbar_total, unit="frame", desc=f"切帧 {os.path.basename(video_path)}", position=start_idx)

    while frame_count <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0 and start_frame <= frame_count <= end_frame:
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

def batch_extract_frames_multithread(input_dir, output_dir, interval=1, max_workers=4, start_frame=0, end_frame=0):
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
            future = executor.submit(extract_frames_single_video, video_path, output_dir, interval, idx, start_frame, end_frame)
            futures.append(future)

        for future in as_completed(futures):
            total_saved += future.result()

    print(f"🎉 全部处理完成，总共提取 {total_saved} 张图片。")

if __name__ == "__main__":
    # === 配置部分 ===
    input_dir = "/home/hithcat/Code/CRT/TraditionalLabel/videos"      # 视频目录
    output_dir = "/home/hithcat/Code/CRT/TraditionalLabel/frames"    # 图片保存目录
    interval = 5     # 每隔多少帧保存一张
    max_workers = 16  # 线程数，建议 = CPU核心数 或 核心数x2
    start_frame = 3130   # 从视频的第几帧开始提取
    end_frame = 3335     # 提取到视频的第几帧结束，0表示提取到结尾

    batch_extract_frames_multithread(input_dir, output_dir, interval, max_workers, start_frame, end_frame)
