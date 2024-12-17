import cv2
import numpy as np

def preprocess_image(image_path, target_size):
    # 读取图片
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 调整大小
    img = cv2.resize(img, target_size)
    
    # 归一化
    img = img.astype('float32') / 255.0
    
    # 增加通道维度
    img = np.expand_dims(img, axis=-1)
    
    return img

def augment_data(image):
    """数据增强"""
    augmented = []
    # 添加原始图片
    augmented.append(image)
    
    # 旋转
    augmented.append(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))
    
    # 缩放
    augmented.append(cv2.resize(cv2.resize(image, (32, 32)), (64, 64)))
    
    return augmented 