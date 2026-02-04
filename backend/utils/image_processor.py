"""
图像预处理模块
负责图像的加载、调整大小、归一化和张量转换
"""
import torch
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
from typing import Union, Tuple
import numpy as np


class ImageProcessor:
    """图像处理器类，用于图像预处理和转换"""
    
    def __init__(self):
        """初始化图像处理器，设置预处理变换"""
        # 定义图像预处理变换
        self.transform = transforms.Compose([
            transforms.Resize(256),  # 调整图像大小为 256x256
            transforms.CenterCrop(224),  # 中心裁剪为 224x224
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(  # 归一化
                mean=[0.485, 0.456, 0.406],  # ImageNet 均值
                std=[0.229, 0.224, 0.225]   # ImageNet 标准差
            )
        ])
    
    def load_image_from_bytes(self, image_bytes: bytes) -> Image.Image:
        """
        从字节数据加载图像
        
        Args:
            image_bytes: 图像的字节数据
            
        Returns:
            Image.Image: PIL 图像对象
        """
        image_stream = BytesIO(image_bytes)
        image = Image.open(image_stream)
        
        # 转换为 RGB 格式（如果图像是 RGBA 或其他格式）
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        return image
    
    def load_image_from_path(self, image_path: str) -> Image.Image:
        """
        从文件路径加载图像
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            Image.Image: PIL 图像对象
        """
        image = Image.open(image_path)
        
        # 转换为 RGB 格式（如果图像是 RGBA 或其他格式）
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        return image
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        预处理图像，转换为模型输入张量
        
        Args:
            image: PIL 图像对象
            
        Returns:
            torch.Tensor: 预处理后的图像张量
        """
        # 应用预处理变换
        image_tensor = self.transform(image)
        
        # 添加批次维度
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
    
    def process_from_bytes(self, image_bytes: bytes) -> torch.Tensor:
        """
        从字节数据处理图像
        
        Args:
            image_bytes: 图像的字节数据
            
        Returns:
            torch.Tensor: 预处理后的图像张量
        """
        image = self.load_image_from_bytes(image_bytes)
        return self.preprocess_image(image)
    
    def process_from_path(self, image_path: str) -> torch.Tensor:
        """
        从文件路径处理图像
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            torch.Tensor: 预处理后的图像张量
        """
        image = self.load_image_from_path(image_path)
        return self.preprocess_image(image)
    
    def resize_image(self, image: Image.Image, size: Tuple[int, int]) -> Image.Image:
        """
        调整图像大小
        
        Args:
            image: PIL 图像对象
            size: 目标大小 (width, height)
            
        Returns:
            Image.Image: 调整大小后的图像
        """
        return image.resize(size)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        归一化图像像素值到 [0, 1] 范围
        
        Args:
            image: numpy 数组格式的图像
            
        Returns:
            np.ndarray: 归一化后的图像
        """
        return image.astype(np.float32) / 255.0


# 创建全局图像处理器实例
image_processor = ImageProcessor()
