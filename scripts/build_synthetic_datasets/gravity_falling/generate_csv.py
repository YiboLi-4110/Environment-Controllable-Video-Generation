#!/usr/bin/env python3
import os
import json
import csv
import random
import argparse
from tqdm import tqdm


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='生成包含视频信息的CSV文件')
    parser.add_argument('video_dir', help='视频文件目录路径')
    parser.add_argument('backgrounds_json', help='backgrounds.json文件路径')
    parser.add_argument('output_csv', help='输出CSV文件路径')
    return parser.parse_args()


def extract_info(filename):
    """从文件名中提取信息"""
    # 移除.mp4后缀
    name_without_ext = os.path.splitext(filename)[0]
    
    # 提取gravity值
    parts = name_without_ext.split('_')
    gravity = None
    background_name = None
    
    for i, part in enumerate(parts):
        if part == 'g' and i + 1 < len(parts):
            gravity = parts[i + 1]
            # 提取背景名称（从gravity后面开始，直到文件名结束）
            background_name = '_'.join(parts[i + 2:])
            break
    
    return gravity, background_name


def get_random_caption(background_data):
    """根据概率随机选择一个提示"""
    prompts = [
        background_data.get('optimized_prompt_1', ''),
        background_data.get('optimized_prompt_2', ''),
        background_data.get('optimized_prompt_3', '')
    ]
    # 概率分布：0.4, 0.4, 0.2
    weights = [0.4, 0.4, 0.2]
    return random.choices(prompts, weights=weights, k=1)[0]


def main():
    args = parse_args()
    video_dir = args.video_dir
    backgrounds_json = args.backgrounds_json
    output_csv = args.output_csv
    
    # 加载backgrounds.json文件
    with open(backgrounds_json, 'r', encoding='utf-8') as f:
        backgrounds_data = json.load(f)
    
    # 准备CSV文件
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['video', 'gravity', 'caption', 'width', 'height', 'fps', 'frame']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # 写入表头
        writer.writeheader()
        
        # 遍历视频文件目录
        for filename in tqdm(os.listdir(video_dir)):
            if filename.endswith('.mp4'):
                # 提取信息
                gravity, background_name = extract_info(filename)
                # 保留原始文件名（包含.mp4后缀）
                video_name = filename
                
                # 获取caption
                caption = ""
                if background_name and background_name in backgrounds_data:
                    caption = get_random_caption(backgrounds_data[background_name])
                
                # 写入行
                writer.writerow({
                    'video': video_name,
                    'gravity': gravity,
                    'caption': caption,
                    'width': 832,
                    'height': 480,
                    'fps': 16,
                    'frame': 81
                })
    
    print(f"CSV文件已生成：{output_csv}")


if __name__ == "__main__":
    main()