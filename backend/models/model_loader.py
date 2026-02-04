"""
模型加载模块
负责加载和管理预训练的 MobileNetV2 模型
"""
import torch
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights
from typing import Dict, List, Tuple
import numpy as np


class ModelLoader:
    """模型加载器类，用于加载和管理 MobileNetV2 预训练模型"""
    
    def __init__(self):
        """初始化模型加载器"""
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = self._load_class_names()
    
    def _load_class_names(self) -> List[str]:
        """
        加载 ImageNet 类别名称
        
        Returns:
            List[str]: ImageNet 1000 个类别名称列表
        """
        # 使用 torchvision 提供的 ImageNet 类别名称
        weights = MobileNet_V2_Weights.DEFAULT
        return weights.meta["categories"]
    
    def load_model(self) -> None:
        """
        加载 MobileNetV2 预训练模型
        
        将模型设置为评估模式并移动到指定设备
        """
        if self.model is None:
            # 加载预训练的 MobileNetV2 模型
            weights = MobileNet_V2_Weights.DEFAULT
            self.model = models.mobilenet_v2(weights=weights)
            
            # 设置为评估模式
            self.model.eval()
            
            # 移动到设备
            self.model.to(self.device)
            
            print(f"模型已加载到设备: {self.device}")
    
    def predict(self, image_tensor: torch.Tensor) -> Tuple[str, float]:
        """
        使用模型对图像进行预测
        
        Args:
            image_tensor: 预处理后的图像张量
            
        Returns:
            Tuple[str, float]: 预测的类别名称和置信度分数
        """
        if self.model is None:
            self.load_model()
        
        # 确保模型在评估模式
        self.model.eval()
        
        # 移动到设备
        image_tensor = image_tensor.to(self.device)
        
        # 进行预测
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # 获取最高概率的类别
        top_prob, top_class = torch.topk(probabilities, 1)
        
        # 获取类别名称和置信度
        class_name = self.class_names[top_class.item()]
        confidence = top_prob.item()
        
        return class_name, confidence
    
    def predict_top_k(self, image_tensor: torch.Tensor, k: int = 5) -> List[Dict[str, float]]:
        """
        使用模型对图像进行预测，返回前 k 个最可能的类别
        
        Args:
            image_tensor: 预处理后的图像张量
            k: 返回前 k 个预测结果
            
        Returns:
            List[Dict[str, float]]: 包含类别名称和置信度的字典列表
        """
        if self.model is None:
            self.load_model()
        
        # 确保模型在评估模式
        self.model.eval()
        
        # 移动到设备
        image_tensor = image_tensor.to(self.device)
        
        # 进行预测
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # 获取前 k 个最高概率的类别
        top_probs, top_classes = torch.topk(probabilities, k)
        
        # 构建结果列表
        results = []
        for i in range(k):
            class_name = self.class_names[top_classes[i].item()]
            confidence = top_probs[i].item()
            results.append({
                "class_name": class_name,
                "confidence": confidence
            })
        
        return results


# 创建全局模型加载器实例
model_loader = ModelLoader()
