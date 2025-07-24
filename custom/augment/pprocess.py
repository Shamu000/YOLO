import cv2
import numpy as np
import random

# === 光照扰动（亮度不均匀） ===
def uneven_illumination(img, gamma_range=(0.5, 1.5)):
    gamma = random.uniform(*gamma_range)
    img_flt = (img.astype(np.float32) / 255) ** gamma * 255
    return np.clip(img_flt, 0, 255).astype(np.uint8)

# === 彩色椒盐噪声 ===
def colorful_sp_noise(img, prob=0.001):
    noisy = img.copy()
    h, w, _ = img.shape
    num_noise = int(prob * h * w)
    noise_colors = [
        (255, 0, 0),      # 红
        (0, 0, 255),      # 蓝
        (0, 0, 0),        # 黑
        (255, 255, 255)   # 白
    ]
    for _ in range(num_noise):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        noisy[y, x] = random.choice(noise_colors)
    return noisy

# === 高亮运动模糊 ===
# def motion_blur_on_bright(img, direction='x', size=None, thresh=200):
#     """
#     仅对图像中高亮区域施加定向运动模糊（模拟灯条模糊）
    
#     :param img: 输入 BGR 图像 (H, W, 3)
#     :param direction: 'x' 或 'y' 方向
#     :param size: 模糊核大小，可为 None 随机选
#     :param thresh: 灰度亮度阈值，高于该值的区域将被模糊
#     :return: 加了局部模糊的图像
#     """
#     if size is None:
#         size = random.choice([7, 15, 23])

#     # 灰度图和高亮区域 mask 提取
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     mask = (gray > thresh).astype(np.uint8) * 255  # 二值高亮区域

#     # 方向性模糊核
#     if direction == 'x':
#         kernel = np.zeros((1, size))
#         kernel[0, :] = 1
#     elif direction == 'y':
#         kernel = np.zeros((size, 1))
#         kernel[:, 0] = 1
#     else:
#         raise ValueError("direction must be 'x' or 'y'")
#     kernel /= size

#     # 整图模糊结果（准备混合）
#     blurred = cv2.filter2D(img, -1, kernel)

#     # 扩展 mask 为 3 通道
#     mask3 = cv2.merge([mask, mask, mask])

#     # 将亮区域换成模糊结果
#     result = np.where(mask3 == 255, blurred, img)
#     return result

# === xy运动模糊 ===
def motion_blur(img, direction='x', size=None):
    """
    对整张图像施加 X 或 Y 方向运动模糊

    :param img: 输入图像 (H, W, 3)
    :param direction: 模糊方向，'x' 或 'y'
    :param size: 模糊核大小,None 表示随机选择
    :return: 模糊图像
    """
    if size is None:
        size = random.choice([7, 15, 23])

    if direction == 'x':
        kernel = np.zeros((1, size), dtype=np.float32)
        kernel[0, :] = 1
    elif direction == 'y':
        kernel = np.zeros((size, 1), dtype=np.float32)
        kernel[:, 0] = 1
    else:
        raise ValueError("direction must be 'x' or 'y'")

    kernel /= size  # 归一化核
    return cv2.filter2D(img, -1, kernel)

# === 随机遮挡（Cutout） ===
def random_cutout(img, num_holes=1, size_range=(20, 40)):
    """
    对图像应用随机遮挡块。
    :param img: 输入图像
    :param num_holes: 遮挡块数量
    :param size_range: 遮挡块边长范围 (min, max)
    :return: 遮挡后的图像
    """
    h, w = img.shape[:2]
    min_size, max_size = size_range
    for _ in range(num_holes):
        size = random.randint(min_size, max_size)
        x = random.randint(0, w - size)
        y = random.randint(0, h - size)
        img[y:y+size, x:x+size] = 0
    return img

# === 主函数 ===
def process(data):
    img = data['img']

    # === 控制增强是否执行 ===
    prob_no_aug = 0.15
    if random.random() < prob_no_aug:
        data['img'] = img
        return data

    # === 各类增强模块 ===
    if random.random() < 0.2:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    if random.random() < 0.5:
        img = uneven_illumination(img)

    if random.random() < 0.4:
        img = colorful_sp_noise(img, prob=0.001)

    if random.random() < 0.5:
        direction = random.choices(['x', 'y'], weights=[0.4, 0.6])[0]
        img = motion_blur(img, direction=direction)

    # 遮挡块增强
    if random.random() < 0.3:  # 30% 概率加入遮挡
        img = random_cutout(img, num_holes=random.randint(1, 3), size_range=(20, 40))

    data['img'] = img
    return data
