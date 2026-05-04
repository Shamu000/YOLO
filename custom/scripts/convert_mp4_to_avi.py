import os

import cv2
from tqdm import tqdm


def convert_mp4_to_avi(mp4_path, avi_path, fourcc_str="XVID"):
    """
    Convert an MP4 video to AVI format.

    :param mp4_path: input MP4 path
    :param avi_path: output AVI path (including filename)
    :param fourcc_str: codec string, default XVID
    """
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        print(f"Failed to open video file: {mp4_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(avi_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)  # type: ignore[attr-defined]
    out = cv2.VideoWriter(avi_path, fourcc, fps, (width, height))

    frame_count = 0
    pbar = tqdm(desc="MP4 to AVI", unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_count += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

    print(f"   Converted: {avi_path}")
    print(f"   Total frames: {frame_count}")
    print(f"   FPS: {fps}")


if __name__ == "__main__":
    # ===== Config =====
    mp4_path = "/home/hithcat/Code/CRT/TraditionalLabel/videos/unlabled/qusai/24_qusai_hik_2.mp4"
    avi_path = "/home/hithcat/Code/CRT/TraditionalLabel/videos/unlabled/qusai/24_qusai_hik_2.avi"

    # ===== Run =====
    convert_mp4_to_avi(mp4_path, avi_path)
