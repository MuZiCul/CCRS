import cv2
import numpy as np
import os
import logging
from typing import Tuple, List, Dict
from config import Config

def load_image(image_path: str) -> np.ndarray:
    """安全地加载并预处理单个图像"""
    try:
        # 检查文件是否存在
        if not os.path.exists(image_path):
            logging.error(f"图像文件不存在: {image_path}")
            return None
            
        # 检查文件大小
        if os.path.getsize(image_path) == 0:
            logging.error(f"图像文件为空: {image_path}")
            return None
            
        # 使用完整路径读取图像
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            logging.error(f"无法解码图像: {image_path}")
            return None
            
        # 检查图像尺寸
        if image.shape[0] == 0 or image.shape[1] == 0:
            logging.error(f"图像尺寸无效: {image_path}, shape: {image.shape}")
            return None
            
        # 转换为灰度图
        if Config.CHANNELS == 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        # 调整图像大小
        try:
            image = cv2.resize(image, (Config.IMAGE_SIZE[1], Config.IMAGE_SIZE[0]))
        except Exception as e:
            logging.error(f"调整图像大小失败 {image_path}: {str(e)}, 原始尺寸: {image.shape}")
            return None
        
        # 标准化像素值
        image = image.astype(np.float32) / 255.0
        
        # 根据通道数调整维度
        if Config.CHANNELS == 1:
            image = np.expand_dims(image, axis=-1)
            
        return image
        
    except Exception as e:
        logging.error(f"处理图片出错 {image_path}: {str(e)}")
        return None

def load_dataset(dataset_path, existing_char_to_label=None):
    """
    加载数据集
    
    Args:
        dataset_path: 数据集路径
        existing_char_to_label: 可选，已存在的字符映射表
    
    Returns:
        data: 图像数据
        labels: 标签
        char_to_label: 字符到标签的映射
        label_to_char: 标签到字符的映射
    """
    images = []
    labels = []
    char_to_label = {}
    label_to_char = {}
    current_label = 0
    
    # 确保数据集路径存在
    if not os.path.exists(dataset_path):
        raise ValueError(f"数据集路径不存在: {dataset_path}")
    
    # 记录处理的文件数量
    total_files = 0
    successful_loads = 0
    
    # 遍历数据集目录
    for char_folder in os.listdir(dataset_path):
        char_path = os.path.join(dataset_path, char_folder)
        if not os.path.isdir(char_path):
            continue
            
        # 为每个字符创建标签映射
        if char_folder not in char_to_label:
            char_to_label[char_folder] = current_label
            label_to_char[current_label] = char_folder
            current_label += 1
        
        # 处理该字符下的所有图像
        image_files = [f for f in os.listdir(char_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        total_files += len(image_files)
        
        for image_file in image_files:
            image_path = os.path.join(char_path, image_file)
            image = load_image(image_path)
            
            if image is not None:
                images.append(image)
                labels.append(char_to_label[char_folder])
                successful_loads += 1
    
    logging.info(f"总文件数: {total_files}")
    logging.info(f"成功加载: {successful_loads}")
    logging.info(f"失败数量: {total_files - successful_loads}")
    
    if not images:
        raise ValueError("没有成功加载任何图像")
    
    if existing_char_to_label is not None:
        # 使用已有的映射
        char_to_label = existing_char_to_label
        label_to_char = {v: k for k, v in char_to_label.items()}
        # 将标签转换为已有映射中的索引
        labels = [char_to_label[char] for char in labels]
    else:
        # 创建新的映射
        unique_chars = sorted(list(set(labels)))
        char_to_label = {char: idx for idx, char in enumerate(unique_chars)}
        label_to_char = {idx: char for char, idx in char_to_label.items()}
        labels = [char_to_label[char] for char in labels]
    
    return np.array(images), np.array(labels), char_to_label, label_to_char