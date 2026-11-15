from pathlib import Path

import cv2

try:
    from tqdm import tqdm
except ImportError:  # Fallback when tqdm is unavailable
    tqdm = None
from ultralytics import YOLO


# =========================
# 直接在这里修改配置即可
# =========================
MODEL_PATH = "/home/hithcat/Code/CRT/yolo/runs/pose/v2-50%数据增强-401代/weights/last.pt"
AVI_PATH = "/home/hithcat/Code/CRT/TraditionalLabel/datas/26unlabeled/tooltrain/2026-05-30_09-31-30_raw.avi"
IMGSZ = 640
CONF = 0.7
IOU = 0.8
KPT_CONF = 0.0  # 0.0 表示返回的关键点全部绘制
DEVICE = "cuda"  # 可改为 "cpu"
RADIUS = 4
SHOW_WINDOW = False
SAVE_OUTPUT = True
OUTPUT_AVI_PATH = "/home/hithcat/下载/output3.avi"


def draw_keypoints(frame, result, kpt_conf_threshold: float, radius: int) -> None:
    keypoints = result.keypoints
    if keypoints is None or keypoints.xy is None:
        return

    xy = keypoints.xy.cpu().numpy()  # (num_det, num_kpt, 2)
    conf = keypoints.conf.cpu().numpy() if keypoints.conf is not None else None

    # 获取每个检测的类别，用于 R 类占位关键点过滤
    boxes = result.boxes
    det_cls = boxes.cls.cpu().numpy().astype(int) if boxes is not None else None
    if det_cls is not None and len(det_cls) == 0:
        det_cls = None

    num_dets = xy.shape[0]
    num_kpts = xy.shape[1] if xy.ndim >= 2 else 0

    for det_i in range(num_dets):
        # R 类(1,3) 只绘制 kpt[0](R标记)，跳过占位关键点
        cls_i = det_cls[det_i] if det_cls is not None else -1
        start_kpt = 1 if cls_i in {1, 3} else 0

        for kpt_i in range(start_kpt, num_kpts):
            if conf is not None and conf[det_i, kpt_i] < kpt_conf_threshold:
                continue

            x, y = xy[det_i, kpt_i]
            px, py = int(round(x)), int(round(y))

            # 采用固定 6 色循环，便于区分 6 个关键点
            color = (
                (37 * (kpt_i + 1)) % 255,
                (17 * (kpt_i + 3)) % 255,
                (29 * (kpt_i + 5)) % 255,
            )
            cv2.circle(frame, (px, py), radius, color, -1)
            cv2.putText(
                frame,
                f"{kpt_i}:{(conf[det_i, kpt_i] if conf is not None else 1.0):.2f}",
                (px + 4, py - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                lineType=cv2.LINE_AA,
            )


def main() -> None:
    model_path = Path(MODEL_PATH)
    source_path = Path(AVI_PATH)
    output_path = Path(OUTPUT_AVI_PATH)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not source_path.exists():
        raise FileNotFoundError(f"Video file not found: {source_path}")
    if source_path.suffix.lower() != ".avi":
        raise ValueError(f"Input must be an AVI file, got: {source_path}")

    model = YOLO(str(model_path)).to(DEVICE)

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {source_path}")

    window_name = "Pose Keypoints Visualization"
    use_window = SHOW_WINDOW
    writer = None

    if SAVE_OUTPUT:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter.fourcc(*"XVID")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to create output video: {output_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = tqdm(total=total_frames, unit="frame") if tqdm is not None else None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model.predict(
                source=frame,
                imgsz=IMGSZ,
                conf=CONF,
                iou=IOU,
                verbose=False,
            )

            if results:
                draw_keypoints(frame, results[0], KPT_CONF, RADIUS)

            if writer is not None:
                writer.write(frame)

            if progress is not None:
                progress.update(1)

            if use_window:
                try:
                    cv2.imshow(window_name, frame)
                    key = cv2.waitKey(0) & 0xFF
                    if key in (27, ord("q")):
                        break
                except cv2.error:
                    # 当前 OpenCV 无 highgui 支持，自动降级为仅保存输出
                    use_window = False
                    print("OpenCV highgui is not available, switch to headless mode (save video only).")
    finally:
        if progress is not None:
            progress.close()
        cap.release()
        if writer is not None:
            writer.release()
        if use_window:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


if __name__ == "__main__":
    main()
