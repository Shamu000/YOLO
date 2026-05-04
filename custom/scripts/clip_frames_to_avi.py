import cv2
import os
from tqdm import tqdm


def extract_clip_to_avi(video_path, output_path, start_frame=0, end_frame=0):
    """
    根据帧区间裁剪视频并保存为AVI文件

    :param video_path: 输入视频路径
    :param output_path: 输出AVI路径（包含文件名）
    :param start_frame: 起始帧
    :param end_frame: 结束帧（0表示到结尾）
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return

    # === 读取视频信息 ===
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # === 处理帧范围 ===
    end_frame = end_frame if end_frame > 0 else total_frames
    start_frame = max(0, min(start_frame, total_frames))
    end_frame = min(end_frame, total_frames)

    if start_frame >= end_frame:
        print("起始帧必须小于结束帧")
        return

    # === 创建输出目录 ===
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # === 创建 VideoWriter ===
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # type: ignore[attr-defined]  # avi编码
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # === 跳转到起始帧 ===
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    saved_frames = 0

    pbar_total = end_frame - start_frame + 1
    pbar = tqdm(total=pbar_total, desc="导出AVI", unit="frame")

    # === 逐帧写入 ===
    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        out.write(frame)
        saved_frames += 1

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

    print(f"   导出完成: {output_path}")
    print(f"   帧范围: {start_frame} ~ {end_frame}")
    print(f"   总帧数: {saved_frames}")
    print(f"   FPS: {fps}")


if __name__ == "__main__":
    # ===== 配置 =====
    video_path = "/home/hithcat/Code/CRT/TraditionalLabel/videos/unlabeled/qusai/24_qusai_hik2_2.avi"

    output_path = "/home/hithcat/Code/CRT/TraditionalLabel/videos/unlabeled/qusai/24_qusai_hik2_2_cliped.avi"

    start_frame = 1968
    end_frame = 0   # 0表示到结尾

    # ===== 执行 =====
    extract_clip_to_avi(video_path, output_path, start_frame, end_frame)