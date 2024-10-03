import os
import json
import numpy as np
import imageio
import matplotlib.pyplot as plt
from glob import glob

# 批次处理函数
def process_directory(base_dir):
    # 获取脚本当前目录，并创建两个文件夹来存储生成的结果
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sum_dir = os.path.join(script_dir, 'neurofinder_sum')
    mask_dir = os.path.join(script_dir, 'neurofinder_mask')

    # 如果文件夹不存在，创建它们
    os.makedirs(sum_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # 获取所有主目录下的子目录
    subdirectories = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for subdirectory in subdirectories:
        # 处理路径中的两个相同的子目录名（如 neurofinder.00.00/neurofinder.00.00）
        inner_dir = os.path.join(base_dir, subdirectory, subdirectory)
        image_dir = os.path.join(inner_dir, 'images')
        region_file = os.path.join(inner_dir, 'regions', 'regions.json')

        if not os.path.exists(image_dir):
            print(f"Skipping {subdirectory}: Missing images directory.")
            continue
        
        # 读取 images 目录中的 tiff 文件
        image_files = sorted(glob(os.path.join(image_dir, '*.tiff')))
        if not image_files:
            print(f"No TIFF files found in {image_dir}")
            continue
        
        # 读取所有 tiff 图像
        imgs = np.array([imageio.imread(f) for f in image_files])
        dims = imgs.shape[1:]  # 获取图像的维度

        # 检查 regions.json 文件是否存在
        if not os.path.exists(region_file):
            print(f"Skipping {subdirectory}: Missing regions.json.")
            continue

        # 读取 regions.json 文件并生成 mask
        with open(region_file, 'r') as f:
            regions = json.load(f)

        # Function to generate mask from coordinates
        def tomask(coords, dims):
            mask = np.zeros(dims)  # 创建全零的mask
            mask[tuple(zip(*coords))] = 1  # 使用 zip(*coords) 将坐标解压为适合 NumPy 索引的格式
            return mask

        # 生成每个区域的 mask 并合并
        masks = np.array([tomask(region['coordinates'], dims) for region in regions])
        summed_mask = np.max(masks, axis=0).astype(np.uint8)  # 使用最大值合并所有 mask

        # 保存图像和 mask
        sum_output_path = os.path.join(sum_dir, f"{subdirectory}_sum.tiff")
        mask_output_path = os.path.join(mask_dir, f"{subdirectory}_mask.tiff")
        
        # 显示并保存叠加后的 images
        plt.figure()
        plt.imshow(imgs.sum(axis=0), cmap='gray')  # 叠加后的图像
        plt.axis('off')  # 去掉坐标轴
        plt.savefig(sum_output_path, bbox_inches='tight', pad_inches=0, dpi=300)  # 高分辨率保存
        plt.close()

        # 显示并保存 mask
        plt.figure()
        plt.imshow(summed_mask, cmap='gray')  # 显示合并后的 mask
        plt.axis('off')  # 去掉坐标轴
        plt.savefig(mask_output_path, bbox_inches='tight', pad_inches=0, dpi=300)  # 高分辨率保存
        plt.close()

        print(f"Processed {subdirectory}: Saved {sum_output_path} and {mask_output_path}")

# 主目录路径
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Training group')

# 执行批量处理
process_directory(base_dir)
