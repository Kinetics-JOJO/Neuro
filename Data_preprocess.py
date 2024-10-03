import json
import numpy as np
import imageio
import matplotlib.pyplot as plt
from glob import glob

# Load the images
files = sorted(glob('images/*.tiff'))
imgs = np.array([imageio.imread(f) for f in files])  # 使用 imageio 读取图像
dims = imgs.shape[1:]  # 获取图像的维度

# Load the regions from regions.json
with open('regions/regions.json') as f:
    regions = json.load(f)

# Function to generate mask from coordinates
def tomask(coords, dims):
    mask = np.zeros(dims)  # 创建全零的mask
    mask[tuple(zip(*coords))] = 1  # 使用 zip(*coords) 将坐标解压为适合 NumPy 索引的格式
    return mask

# 生成每个区域的 mask 并合并
masks = np.array([tomask(region['coordinates'], dims) for region in regions])
summed_mask = np.max(masks, axis=0).astype(np.uint8)  # 使用最大值合并所有 mask

# 显示并保存叠加后的 images
plt.figure()
plt.imshow(imgs.sum(axis=0), cmap='gray')  # 叠加后的图像
plt.axis('off')  # 去掉坐标轴
plt.savefig('summed_images_high_res.png', bbox_inches='tight', pad_inches=0, dpi=300)  # 高分辨率保存
plt.close()

# 显示并保存 mask
plt.figure()
plt.imshow(summed_mask, cmap='gray')  # 显示合并后的 mask
plt.axis('off')  # 去掉坐标轴
plt.savefig('generated_mask_high_res.png', bbox_inches='tight', pad_inches=0, dpi=300)  # 高分辨率保存
plt.close()

print("Images and mask saved successfully.")
