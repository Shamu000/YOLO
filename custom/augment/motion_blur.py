import cv2
import numpy as np
import matplotlib.pyplot as plt

# === 模糊函数 ===
def motion_blur_x(img, size=21):
    kernel = np.zeros((size, size))
    kernel[size // 2, :] = 1
    kernel /= size
    return cv2.filter2D(img, -1, kernel)

def motion_blur_y(img, size=21):
    kernel = np.zeros((size, size))
    kernel[:, size // 2] = 1
    kernel /= size
    return cv2.filter2D(img, -1, kernel)

def motion_blur_on_bright(img, direction='x', size=21, thresh=200):
    """
    仅对图像中亮度大于阈值的区域施加运动模糊
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > thresh  # 高亮区域布尔掩码
    mask = mask.astype(np.uint8)

    if direction == 'x':
        blurred = motion_blur_x(img, size=size)
    elif direction == 'y':
        blurred = motion_blur_y(img, size=size)
    else:
        raise ValueError("direction must be 'x' or 'y'")

    # 使用掩码将模糊区域融合到原图中
    result = img.copy()
    for c in range(3):  # 对每个颜色通道应用
        result[:, :, c] = np.where(mask == 1, blurred[:, :, c], img[:, :, c])

    return result

# === 加载图像 ===
img_path = '/home/marshal_kim/桌面/Yolo11/datasets/交流赛/0724哈威交流步3内录/images/train/sentry_2025-07-24_21-22-38_frame_003360.png'
img = cv2.imread(img_path)
if img is None:
    raise ValueError("图片读取失败，请检查路径是否正确。")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# === 全图模糊 ===
img_x = motion_blur_x(img, size=21)
img_y = motion_blur_y(img, size=21)

# === 亮度阈值区域模糊 ===
img_x_bright = motion_blur_on_bright(img, direction='x', size=33, thresh=200)
img_y_bright = motion_blur_on_bright(img, direction='y', size=33, thresh=200)

# === 转换为 RGB 便于显示 ===
img_x_rgb = cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB)
img_y_rgb = cv2.cvtColor(img_y, cv2.COLOR_BGR2RGB)
img_x_bright_rgb = cv2.cvtColor(img_x_bright, cv2.COLOR_BGR2RGB)
img_y_bright_rgb = cv2.cvtColor(img_y_bright, cv2.COLOR_BGR2RGB)

# === 可视化 ===
plt.figure(figsize=(18, 10))

plt.subplot(2, 3, 1)
plt.imshow(img_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(img_x_rgb)
plt.title("X-direction Blur (Full Image)")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(img_y_rgb)
plt.title("Y-direction Blur (Full Image)")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(img_x_bright_rgb)
plt.title("X-direction Blur (Bright Only)")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(img_y_bright_rgb)
plt.title("Y-direction Blur (Bright Only)")
plt.axis("off")

plt.tight_layout()
plt.show()