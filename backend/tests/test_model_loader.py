"""
模型加载器测试模块

该模块包含对 ModelLoader 类的单元测试，包括：
- 模型加载功能
- 单标签预测功能
- Top-K 预测功能
- 类别名称加载

使用 pytest 和 unittest.mock 进行测试
"""
import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# 添加父目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_loader import ModelLoader, model_loader


@pytest.fixture
def model_loader_instance():
    """
    创建 ModelLoader 实例的 fixture
    
    Returns:
        ModelLoader: 新的模型加载器实例
    """
    return ModelLoader()


@pytest.fixture
def sample_image_tensor():
    """
    创建示例图像张量的 fixture
    
    Returns:
        torch.Tensor: 形状为 (1, 3, 224, 224) 的随机图像张量
    """
    return torch.randn(1, 3, 224, 224)


class TestModelLoaderInitialization:
    """模型加载器初始化测试类"""
    
    def test_initialization(self, model_loader_instance):
        """
        测试 ModelLoader 初始化
        
        Args:
            model_loader_instance: ModelLoader 实例
            
        Expected:
            - model 属性初始为 None
            - device 属性正确设置 (cuda 或 cpu)
            - class_names 已加载且包含 1000 个类别
        """
        assert model_loader_instance.model is None
        assert model_loader_instance.device in [
            torch.device("cuda"),
            torch.device("cpu")
        ]
        assert model_loader_instance.class_names is not None
        assert len(model_loader_instance.class_names) == 1000
    
    def test_load_class_names(self, model_loader_instance):
        """
        测试类别名称加载
        
        Args:
            model_loader_instance: ModelLoader 实例
            
        Expected:
            - 返回列表类型
            - 包含 1000 个类别名称
            - 所有类别名称为字符串类型
        """
        class_names = model_loader_instance._load_class_names()
        
        assert isinstance(class_names, list)
        assert len(class_names) == 1000
        assert all(isinstance(name, str) for name in class_names)


class TestModelLoading:
    """模型加载功能测试类"""
    
    @patch("models.model_loader.models.mobilenet_v2")
    @patch("models.model_loader.MobileNet_V2_Weights")
    def test_load_model(self, mock_weights, mock_mobilenet, model_loader_instance):
        """
        测试模型加载功能
        
        Args:
            mock_weights: 模拟的 MobileNet_V2_Weights
            mock_mobilenet: 模拟的 mobilenet_v2 模型
            model_loader_instance: ModelLoader 实例
            
        Expected:
            - 模型被成功加载
            - 模型设置为评估模式
            - 模型移动到正确设备
        """
        # 设置模拟对象
        mock_model = MagicMock()
        mock_mobilenet.return_value = mock_model
        mock_weights.DEFAULT.meta = {"categories": [f"class_{i}" for i in range(1000)]}
        
        # 加载模型
        model_loader_instance.load_model()
        
        # 验证模型加载
        assert model_loader_instance.model is not None
        mock_model.eval.assert_called_once()
        mock_model.to.assert_called_once_with(model_loader_instance.device)
    
    @patch("models.model_loader.models.mobilenet_v2")
    @patch("models.model_loader.MobileNet_V2_Weights")
    def test_load_model_only_once(self, mock_weights, mock_mobilenet, model_loader_instance):
        """
        测试模型只加载一次
        
        Args:
            mock_weights: 模拟的 MobileNet_V2_Weights
            mock_mobilenet: 模拟的 mobilenet_v2 模型
            model_loader_instance: ModelLoader 实例
            
        Expected:
            - 多次调用 load_model 只实际加载一次模型
        """
        # 设置模拟对象
        mock_model = MagicMock()
        mock_mobilenet.return_value = mock_model
        mock_weights.DEFAULT.meta = {"categories": [f"class_{i}" for i in range(1000)]}
        
        # 加载模型两次
        model_loader_instance.load_model()
        model_loader_instance.load_model()
        
        # 验证 mobilenet_v2 只被调用一次
        mock_mobilenet.assert_called_once()


class TestPredictFunction:
    """单标签预测功能测试类"""
    
    @patch("models.model_loader.models.mobilenet_v2")
    @patch("models.model_loader.MobileNet_V2_Weights")
    def test_predict(self, mock_weights, mock_mobilenet, model_loader_instance, sample_image_tensor):
        """
        测试单标签预测功能
        
        Args:
            mock_weights: 模拟的 MobileNet_V2_Weights
            mock_mobilenet: 模拟的 mobilenet_v2 模型
            model_loader_instance: ModelLoader 实例
            sample_image_tensor: 示例图像张量
            
        Expected:
            - 返回类别名称字符串
            - 返回置信度浮点数
            - 置信度在 0 到 1 之间
        """
        # 设置模拟对象
        mock_model = MagicMock()
        mock_mobilenet.return_value = mock_model
        
        # 模拟模型输出 (1000 个类别的 logits)
        mock_output = torch.randn(1, 1000)
        mock_model.return_value = mock_output
        mock_model.to.return_value = mock_model
        
        mock_weights.DEFAULT.meta = {"categories": [f"class_{i}" for i in range(1000)]}
        
        # 加载模型并执行预测
        model_loader_instance.load_model()
        class_name, confidence = model_loader_instance.predict(sample_image_tensor)
        
        # 验证结果
        assert isinstance(class_name, str)
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1
    
    @patch("models.model_loader.models.mobilenet_v2")
    @patch("models.model_loader.MobileNet_V2_Weights")
    def test_predict_auto_load_model(self, mock_weights, mock_mobilenet, model_loader_instance, sample_image_tensor):
        """
        测试预测时自动加载模型
        
        Args:
            mock_weights: 模拟的 MobileNet_V2_Weights
            mock_mobilenet: 模拟的 mobilenet_v2 模型
            model_loader_instance: ModelLoader 实例
            sample_image_tensor: 示例图像张量
            
        Expected:
            - 未手动加载模型时，predict 方法自动加载模型
        """
        # 设置模拟对象
        mock_model = MagicMock()
        mock_mobilenet.return_value = mock_model
        mock_model.to.return_value = mock_model
        
        mock_output = torch.randn(1, 1000)
        mock_model.return_value = mock_output
        
        mock_weights.DEFAULT.meta = {"categories": [f"class_{i}" for i in range(1000)]}
        
        # 不手动加载模型，直接预测
        assert model_loader_instance.model is None
        model_loader_instance.predict(sample_image_tensor)
        
        # 验证模型已自动加载
        assert model_loader_instance.model is not None
        mock_mobilenet.assert_called_once()
    
    @patch("models.model_loader.models.mobilenet_v2")
    @patch("models.model_loader.MobileNet_V2_Weights")
    def test_predict_model_eval_mode(self, mock_weights, mock_mobilenet, model_loader_instance, sample_image_tensor):
        """
        测试预测时模型处于评估模式
        
        Args:
            mock_weights: 模拟的 MobileNet_V2_Weights
            mock_mobilenet: 模拟的 mobilenet_v2 模型
            model_loader_instance: ModelLoader 实例
            sample_image_tensor: 示例图像张量
            
        Expected:
            - 预测时模型调用 eval 方法
        """
        # 设置模拟对象
        mock_model = MagicMock()
        mock_mobilenet.return_value = mock_model
        mock_model.to.return_value = mock_model
        
        mock_output = torch.randn(1, 1000)
        mock_model.return_value = mock_output
        
        mock_weights.DEFAULT.meta = {"categories": [f"class_{i}" for i in range(1000)]}
        
        # 加载模型并预测
        model_loader_instance.load_model()
        model_loader_instance.predict(sample_image_tensor)
        
        # 验证 eval 被调用
        mock_model.eval.assert_called()


class TestPredictTopKFunction:
    """Top-K 预测功能测试类"""
    
    @patch("models.model_loader.models.mobilenet_v2")
    @patch("models.model_loader.MobileNet_V2_Weights")
    def test_predict_top_k(self, mock_weights, mock_mobilenet, model_loader_instance, sample_image_tensor):
        """
        测试 Top-K 预测功能
        
        Args:
            mock_weights: 模拟的 MobileNet_V2_Weights
            mock_mobilenet: 模拟的 mobilenet_v2 模型
            model_loader_instance: ModelLoader 实例
            sample_image_tensor: 示例图像张量
            
        Expected:
            - 返回列表类型
            - 列表长度为 k
            - 每个结果包含 class_name 和 confidence
        """
        # 设置模拟对象
        mock_model = MagicMock()
        mock_mobilenet.return_value = mock_model
        mock_model.to.return_value = mock_model
        
        mock_output = torch.randn(1, 1000)
        mock_model.return_value = mock_output
        
        mock_weights.DEFAULT.meta = {"categories": [f"class_{i}" for i in range(1000)]}
        
        # 加载模型并执行 Top-K 预测
        model_loader_instance.load_model()
        k = 5
        results = model_loader_instance.predict_top_k(sample_image_tensor, k=k)
        
        # 验证结果
        assert isinstance(results, list)
        assert len(results) == k
        
        for result in results:
            assert "class_name" in result
            assert "confidence" in result
            assert isinstance(result["class_name"], str)
            assert isinstance(result["confidence"], float)
            assert 0 <= result["confidence"] <= 1
    
    @patch("models.model_loader.models.mobilenet_v2")
    @patch("models.model_loader.MobileNet_V2_Weights")
    def test_predict_top_k_different_k_values(self, mock_weights, mock_mobilenet, model_loader_instance, sample_image_tensor):
        """
        测试不同的 K 值
        
        Args:
            mock_weights: 模拟的 MobileNet_V2_Weights
            mock_mobilenet: 模拟的 mobilenet_v2 模型
            model_loader_instance: ModelLoader 实例
            sample_image_tensor: 示例图像张量
            
        Expected:
            - 不同的 K 值返回相应数量的结果
        """
        # 设置模拟对象
        mock_model = MagicMock()
        mock_mobilenet.return_value = mock_model
        mock_model.to.return_value = mock_model
        
        mock_output = torch.randn(1, 1000)
        mock_model.return_value = mock_output
        
        mock_weights.DEFAULT.meta = {"categories": [f"class_{i}" for i in range(1000)]}
        
        model_loader_instance.load_model()
        
        # 测试不同的 K 值
        for k in [1, 3, 5, 10]:
            results = model_loader_instance.predict_top_k(sample_image_tensor, k=k)
            assert len(results) == k
    
    @patch("models.model_loader.models.mobilenet_v2")
    @patch("models.model_loader.MobileNet_V2_Weights")
    def test_predict_top_k_order(self, mock_weights, mock_mobilenet, model_loader_instance, sample_image_tensor):
        """
        测试 Top-K 结果按置信度降序排列
        
        Args:
            mock_weights: 模拟的 MobileNet_V2_Weights
            mock_mobilenet: 模拟的 mobilenet_v2 模型
            model_loader_instance: ModelLoader 实例
            sample_image_tensor: 示例图像张量
            
        Expected:
            - 结果按 confidence 从高到低排列
        """
        # 设置模拟对象
        mock_model = MagicMock()
        mock_mobilenet.return_value = mock_model
        mock_model.to.return_value = mock_model
        
        mock_output = torch.randn(1, 1000)
        mock_model.return_value = mock_output
        
        mock_weights.DEFAULT.meta = {"categories": [f"class_{i}" for i in range(1000)]}
        
        model_loader_instance.load_model()
        
        results = model_loader_instance.predict_top_k(sample_image_tensor, k=5)
        confidences = [result["confidence"] for result in results]
        
        # 验证降序排列
        assert confidences == sorted(confidences, reverse=True)


class TestGlobalInstance:
    """全局实例测试类"""
    
    def test_global_model_loader_exists(self):
        """
        测试全局 model_loader 实例存在
        
        Expected:
            - model_loader 是 ModelLoader 的实例
        """
        assert isinstance(model_loader, ModelLoader)
    
    def test_global_model_loader_singleton(self):
        """
        测试全局 model_loader 是单例
        
        Expected:
            - 多次导入得到的是同一个实例
        """
        from models.model_loader import model_loader as model_loader_2
        assert model_loader is model_loader_2


class TestDeviceHandling:
    """设备处理测试类"""
    
    def test_device_selection(self, model_loader_instance):
        """
        测试设备选择逻辑
        
        Args:
            model_loader_instance: ModelLoader 实例
            
        Expected:
            - 优先选择 CUDA (如果可用)
            - 否则选择 CPU
        """
        expected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert model_loader_instance.device == expected_device
