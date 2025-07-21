from pathlib import Path
import cv2
import numpy as np

# =========================================
IMG_DIR = Path(r"/home/marshal_kim/桌面/Yolo11/datasets/模拟赛内录/0703模拟赛步4内录/images")   # ← 图片目录
LBL_DIR = Path(r"/home/marshal_kim/桌面/Yolo11/datasets/模拟赛内录/0703模拟赛步4内录/labels")   # ← 标签 txt 目录
SAVE_DIR = None  # 若想保存结果图，填一个目录，如 Path("annotated")
SHOW = True      # True→逐张弹窗；False→只保存
# ==========================================

# 关键点顺序索引（根据你自己的顺序调整）
IDX_LU, IDX_LD, IDX_RD, IDX_RU = 0, 1, 2, 3   # 左上 左下 右下 右上
IDX_L = (IDX_LU, IDX_LD)                      # 左灯条两点
IDX_R = (IDX_RU, IDX_RD)                      # 右灯条两点

if SAVE_DIR:
    SAVE_DIR = Path(SAVE_DIR)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

def parse_line(line: str):
    vals = list(map(float, line.strip().split()))
    if len(vals) != 1 + 4 + 8:  # cls + cxcywh + 4*(x,y)
        raise ValueError("每行应为 cls cx cy w h + 8 个关键点坐标")
    cls, rest = int(vals[0]), vals[1:]
    cx, cy, w, h, *kps = rest
    kps = np.array(kps, dtype=np.float32).reshape(-1, 2)  # (4,2)
    return cls, (cx, cy, w, h), kps                      # 均为 0~1

def norm2pix(kps, img_w, img_h):
    kps_pix = kps.copy()
    kps_pix[:, 0] *= img_w
    kps_pix[:, 1] *= img_h
    return kps_pix

# ---------- 主循环 ----------
for img_path in sorted(IMG_DIR.glob("*")):
    if img_path.suffix.lower() not in {".jpg", ".png", ".jpeg", ".bmp"}:
        continue
    lbl_path = LBL_DIR / (img_path.stem + ".txt")
    if not lbl_path.exists():
        print(f"[WARN] 找不到标签: {lbl_path.name}")
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[ERR ] 读图失败: {img_path}")
        continue
    H, W = img.shape[:2]

    with open(lbl_path, "r") as f:
        lines = f.readlines()

    for ln in lines:
        _, _, kps_norm = parse_line(ln)
        kps = norm2pix(kps_norm, W, H)  # (4,2)

        # ---- 计算左右灯条长度比 ----
        ptL1, ptL2 = kps[list(IDX_L)]   # 左灯条
        ptR1, ptR2 = kps[list(IDX_R)]   # 右灯条
        lenL = np.linalg.norm(ptL2 - ptL1)
        lenR = np.linalg.norm(ptR2 - ptR1)
        k_lr = lenR / (lenL + 1e-6)

        # ---- 绘制 ----
        cv2.polylines(img, [kps.astype(int)], True, (0,255,0), 2)
        cx, cy = kps.mean(axis=0).astype(int)
        cv2.putText(img, f"k={k_lr:.2f}", (cx-30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    if SAVE_DIR:
        cv2.imwrite(str(SAVE_DIR / img_path.name), img)
    if SHOW:
        cv2.imshow("k_lr visualize", img)
        k = cv2.waitKey(0)
        if k == 27:  # ESC 退出
            break

cv2.destroyAllWindows()