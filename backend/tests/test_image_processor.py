"""
图像处理器测试模块

该模块包含对 ImageProcessor 类的单元测试，包括：
- 图像从字节/路径加载
- 图像预处理（调整大小、裁剪、归一化）
- 图像格式转换（RGB、张量）
- 各种图像格式的处理

使用 pytest 和 PIL 进行测试
"""
import pytest
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import sys
import os

# 添加父目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.image_processor import ImageProcessor, image_processor


@pytest.fixture
def processor():
    """
    创建 ImageProcessor 实例的 fixture
    
    Returns:
        ImageProcessor: 新的图像处理器实例
    """
    return ImageProcessor()


@pytest.fixture
def sample_rgb_image():
    """
    创建示例 RGB 图像的 fixture
    
    Returns:
        Image.Image: 300x200 像素的红色 RGB 图像
    """
    return Image.new("RGB", (300, 200), color="red")


@pytest.fixture
def sample_rgba_image():
    """
    创建示例 RGBA 图像的 fixture
    
    Returns:
        Image.Image: 300x200 像素的半透明红色 RGBA 图像
    """
    return Image.new("RGBA", (300, 200), color=(255, 0, 0, 128))


@pytest.fixture
def sample_jpeg_bytes():
    """
    创建示例 JPEG 字节数据的 fixture
    
    Returns:
        bytes: JPEG 格式的图像字节数据
    """
    image = Image.new("RGB", (300, 200), color="blue")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def sample_png_bytes():
    """
    创建示例 PNG 字节数据的 fixture
    
    Returns:
        bytes: PNG 格式的图像字节数据（RGBA）
    """
    image = Image.new("RGBA", (300, 200), color=(0, 255, 0, 200))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class TestImageProcessorInitialization:
    """图像处理器初始化测试类"""
    
    def test_initialization(self, processor):
        """
        测试 ImageProcessor 初始化
        
        Args:
            processor: ImageProcessor 实例
            
        Expected:
            - transform 属性已设置
            - transform 是 Compose 对象
        """
        assert processor.transform is not None
        assert isinstance(processor.transform, torch.nn.Module) or hasattr(processor.transform, 'transforms')


class TestLoadImageFromBytes:
    """从字节加载图像测试类"""
    
    def test_load_jpeg_from_bytes(self, processor, sample_jpeg_bytes):
        """
        测试从字节加载 JPEG 图像
        
        Args:
            processor: ImageProcessor 实例
            sample_jpeg_bytes: JPEG 图像字节数据
            
        Expected:
            - 返回 PIL Image 对象
            - 图像模式为 RGB
        """
        image = processor.load_image_from_bytes(sample_jpeg_bytes)
        
        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"
    
    def test_load_png_from_bytes(self, processor, sample_png_bytes):
        """
        测试从字节加载 PNG 图像（RGBA 转 RGB）
        
        Args:
            processor: ImageProcessor 实例
            sample_png_bytes: PNG 图像字节数据
            
        Expected:
            - 返回 PIL Image 对象
            - 图像模式转换为 RGB
        """
        image = processor.load_image_from_bytes(sample_png_bytes)
        
        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"
    
    def test_load_invalid_bytes(self, processor):
        """
        测试加载无效字节数据时的错误处理
        
        Args:
            processor: ImageProcessor 实例
            
        Expected:
            - 抛出异常（OSError 或 IOError）
        """
        invalid_bytes = b"invalid image data"
        
        with pytest.raises((OSError, IOError)):
            processor.load_image_from_bytes(invalid_bytes)


class TestLoadImageFromPath:
    """从路径加载图像测试类"""
    
    def test_load_image_from_path(self, processor, tmp_path):
        """
        测试从文件路径加载图像
        
        Args:
            processor: ImageProcessor 实例
            tmp_path: pytest 临时路径 fixture
            
        Expected:
            - 成功加载图像文件
            - 返回 RGB 模式的 PIL Image
        """
        # 创建临时图像文件
        image_path = tmp_path / "test_image.jpg"
        image = Image.new("RGB", (100, 100), color="green")
        image.save(image_path)
        
        # 加载图像
        loaded_image = processor.load_image_from_path(str(image_path))
        
        assert isinstance(loaded_image, Image.Image)
        assert loaded_image.mode == "RGB"
        assert loaded_image.size == (100, 100)
    
    def test_load_rgba_from_path_converts_to_rgb(self, processor, tmp_path):
        """
        测试从路径加载 RGBA 图像时转换为 RGB
        
        Args:
            processor: ImageProcessor 实例
            tmp_path: pytest 临时路径 fixture
            
        Expected:
            - RGBA 图像自动转换为 RGB 模式
        """
        # 创建临时 RGBA 图像文件
        image_path = tmp_path / "test_image.png"
        image = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        image.save(image_path)
        
        # 加载图像
        loaded_image = processor.load_image_from_path(str(image_path))
        
        assert loaded_image.mode == "RGB"
    
    def test_load_nonexistent_file(self, processor):
        """
        测试加载不存在的文件
        
        Args:
            processor: ImageProcessor 实例
            
        Expected:
            - 抛出 FileNotFoundError 异常
        """
        with pytest.raises(FileNotFoundError):
            processor.load_image_from_path("/nonexistent/path/image.jpg")


class TestPreprocessImage:
    """图像预处理测试类"""
    
    def test_preprocess_output_shape(self, processor, sample_rgb_image):
        """
        测试预处理后的输出形状
        
        Args:
            processor: ImageProcessor 实例
            sample_rgb_image: 示例 RGB 图像
            
        Expected:
            - 输出张量形状为 (1, 3, 224, 224)
        """
        tensor = processor.preprocess_image(sample_rgb_image)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 224, 224)
    
    def test_preprocess_output_type(self, processor, sample_rgb_image):
        """
        测试预处理后的输出类型
        
        Args:
            processor: ImageProcessor 实例
            sample_rgb_image: 示例 RGB 图像
            
        Expected:
            - 输出为 torch.Tensor 类型
            - 数据类型为 torch.float32
        """
        tensor = processor.preprocess_image(sample_rgb_image)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32
    
    def test_preprocess_normalization(self, processor):
        """
        测试图像归一化
        
        Args:
            processor: ImageProcessor 实例
            
        Expected:
            - 预处理后的像素值经过归一化
            - 值范围符合 ImageNet 均值和标准差
        """
        # 创建纯色图像
        image = Image.new("RGB", (300, 300), color=(128, 128, 128))
        tensor = processor.preprocess_image(image)
        
        # 验证张量值
        assert isinstance(tensor, torch.Tensor)
        # 归一化后的值应该在合理范围内
        assert not torch.isnan(tensor).any()
        assert not torch.isinf(tensor).any()


class TestProcessFromBytes:
    """从字节处理图像测试类"""
    
    def test_process_from_bytes(self, processor, sample_jpeg_bytes):
        """
        测试从字节数据直接处理图像
        
        Args:
            processor: ImageProcessor 实例
            sample_jpeg_bytes: JPEG 图像字节数据
            
        Expected:
            - 返回形状为 (1, 3, 224, 224) 的张量
        """
        tensor = processor.process_from_bytes(sample_jpeg_bytes)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 224, 224)
    
    def test_process_rgba_from_bytes(self, processor, sample_png_bytes):
        """
        测试从字节处理 RGBA 图像
        
        Args:
            processor: ImageProcessor 实例
            sample_png_bytes: PNG 图像字节数据
            
        Expected:
            - RGBA 图像正确处理并转换为张量
        """
        tensor = processor.process_from_bytes(sample_png_bytes)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 224, 224)


class TestProcessFromPath:
    """从路径处理图像测试类"""
    
    def test_process_from_path(self, processor, tmp_path):
        """
        测试从文件路径直接处理图像
        
        Args:
            processor: ImageProcessor 实例
            tmp_path: pytest 临时路径 fixture
            
        Expected:
            - 返回形状为 (1, 3, 224, 224) 的张量
        """
        # 创建临时图像文件
        image_path = tmp_path / "test_image.jpg"
        image = Image.new("RGB", (400, 300), color="purple")
        image.save(image_path)
        
        # 处理图像
        tensor = processor.process_from_path(str(image_path))
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 224, 224)


class TestResizeImage:
    """图像调整大小测试类"""
    
    def test_resize_image(self, processor, sample_rgb_image):
        """
        测试图像调整大小功能
        
        Args:
            processor: ImageProcessor 实例
            sample_rgb_image: 示例 RGB 图像
            
        Expected:
            - 返回调整大小后的图像
            - 图像尺寸符合目标大小
        """
        target_size = (150, 100)
        resized = processor.resize_image(sample_rgb_image, target_size)
        
        assert isinstance(resized, Image.Image)
        assert resized.size == target_size
    
    def test_resize_to_square(self, processor, sample_rgb_image):
        """
        测试将图像调整为正方形
        
        Args:
            processor: ImageProcessor 实例
            sample_rgb_image: 示例 RGB 图像（300x200）
            
        Expected:
            - 返回正方形图像
        """
        target_size = (224, 224)
        resized = processor.resize_image(sample_rgb_image, target_size)
        
        assert resized.size == (224, 224)


class TestNormalizeImage:
    """图像归一化测试类"""
    
    def test_normalize_image(self, processor):
        """
        测试 numpy 数组图像归一化
        
        Args:
            processor: ImageProcessor 实例
            
        Expected:
            - 像素值归一化到 [0, 1] 范围
            - 返回 float32 类型数组
        """
        # 创建测试图像数组 (0-255)
        image_array = np.array([[[0, 128, 255]]], dtype=np.uint8)
        
        normalized = processor.normalize_image(image_array)
        
        assert isinstance(normalized, np.ndarray)
        assert normalized.dtype == np.float32
        assert normalized[0, 0, 0] == 0.0
        assert normalized[0, 0, 1] == 128.0 / 255.0
        assert normalized[0, 0, 2] == 1.0
    
    def test_normalize_preserves_shape(self, processor):
        """
        测试归一化保持图像形状
        
        Args:
            processor: ImageProcessor 实例
            
        Expected:
            - 归一化后数组形状不变
        """
        image_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        normalized = processor.normalize_image(image_array)
        
        assert normalized.shape == (100, 100, 3)


class TestGlobalInstance:
    """全局实例测试类"""
    
    def test_global_image_processor_exists(self):
        """
        测试全局 image_processor 实例存在
        
        Expected:
            - image_processor 是 ImageProcessor 的实例
        """
        assert isinstance(image_processor, ImageProcessor)
    
    def test_global_image_processor_singleton(self):
        """
        测试全局 image_processor 是单例
        
        Expected:
            - 多次导入得到的是同一个实例
        """
        from utils.image_processor import image_processor as processor_2
        assert image_processor is processor_2


class TestImageModeConversion:
    """图像模式转换测试类"""
    
    def test_l_mode_conversion(self, processor):
        """
        测试灰度图像（L 模式）转换为 RGB
        
        Args:
            processor: ImageProcessor 实例
            
        Expected:
            - L 模式图像转换为 RGB 模式
        """
        # 创建灰度图像
        gray_image = Image.new("L", (100, 100), color=128)
        buffer = BytesIO()
        gray_image.save(buffer, format="PNG")
        
        # 加载并验证转换
        loaded = processor.load_image_from_bytes(buffer.getvalue())
        assert loaded.mode == "RGB"
    
    def test_p_mode_conversion(self, processor):
        """
        测试调色板模式（P 模式）转换为 RGB
        
        Args:
            processor: ImageProcessor 实例
            
        Expected:
            - P 模式图像转换为 RGB 模式
        """
        # 创建调色板模式图像
        p_image = Image.new("P", (100, 100), color=1)
        buffer = BytesIO()
        p_image.save(buffer, format="PNG")
        
        # 加载并验证转换
        loaded = processor.load_image_from_bytes(buffer.getvalue())
        assert loaded.mode == "RGB"


class TestVariousImageSizes:
    """各种图像尺寸测试类"""
    
    @pytest.mark.parametrize("size", [
        (50, 50),      # 小图像
        (224, 224),    # 标准尺寸
        (500, 500),    # 大图像
        (100, 300),    # 纵向图像
        (400, 100),    # 横向图像
    ])
    def test_various_input_sizes(self, processor, size):
        """
        测试各种输入尺寸的图像处理
        
        Args:
            processor: ImageProcessor 实例
            size: 图像尺寸元组 (width, height)
            
        Expected:
            - 各种尺寸的图像都能正确处理
            - 输出始终为 (1, 3, 224, 224)
        """
        image = Image.new("RGB", size, color="yellow")
        tensor = processor.preprocess_image(image)
        
        assert tensor.shape == (1, 3, 224, 224)
