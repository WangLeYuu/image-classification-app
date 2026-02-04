"""
FastAPI 主应用端点测试模块

该模块包含对 FastAPI 应用各端点的单元测试，包括：
- 根路径端点 (/)
- 健康检查端点 (/health)
- 图像分类端点 (/classify)

使用 pytest 和 TestClient 进行测试
"""
import pytest
from fastapi.testclient import TestClient
from io import BytesIO
from PIL import Image
import sys
import os

# 添加父目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app


@pytest.fixture
def client():
    """
    创建测试客户端的 fixture
    
    Returns:
        TestClient: FastAPI 测试客户端实例
    """
    return TestClient(app)


@pytest.fixture
def sample_image_bytes():
    """
    创建示例图像字节数据的 fixture
    
    Returns:
        bytes: RGB 格式的示例 JPEG 图像字节数据
    """
    image = Image.new("RGB", (224, 224), color="red")
    image_bytes = BytesIO()
    image.save(image_bytes, format="JPEG")
    return image_bytes.getvalue()


@pytest.fixture
def sample_png_image():
    """
    创建示例 PNG 图像字节数据的 fixture
    
    Returns:
        bytes: RGBA 格式的示例 PNG 图像字节数据
    """
    image = Image.new("RGBA", (300, 300), color=(255, 0, 0, 128))
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    return image_bytes.getvalue()


class TestRootEndpoint:
    """根路径端点测试类"""
    
    def test_root_endpoint(self, client):
        """
        测试根路径端点返回正确的欢迎信息
        
        Args:
            client: FastAPI 测试客户端
            
        Expected:
            - HTTP 状态码: 200
            - 响应包含 message 和 status 字段
            - message 包含 "图像分类 API"
            - status 为 "running"
        """
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "status" in data
        assert "图像分类 API" in data["message"]
        assert data["status"] == "running"
    
    def test_root_endpoint_content_type(self, client):
        """
        测试根路径端点返回正确的 Content-Type
        
        Args:
            client: FastAPI 测试客户端
            
        Expected:
            - Content-Type 包含 application/json
        """
        response = client.get("/")
        
        assert "application/json" in response.headers["content-type"]


class TestHealthEndpoint:
    """健康检查端点测试类"""
    
    def test_health_check_success(self, client):
        """
        测试健康检查端点返回健康状态
        
        Args:
            client: FastAPI 测试客户端
            
        Expected:
            - HTTP 状态码: 200
            - 响应包含 status 字段且值为 "healthy"
        """
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_health_check_method_not_allowed(self, client):
        """
        测试健康检查端点不接受 POST 请求
        
        Args:
            client: FastAPI 测试客户端
            
        Expected:
            - HTTP 状态码: 405 (Method Not Allowed)
        """
        response = client.post("/health")
        
        assert response.status_code == 405


class TestClassifyEndpoint:
    """图像分类端点测试类"""
    
    def test_classify_jpeg_image(self, client, sample_image_bytes):
        """
        测试使用 JPEG 图像进行分类
        
        Args:
            client: FastAPI 测试客户端
            sample_image_bytes: 示例 JPEG 图像字节数据
            
        Expected:
            - HTTP 状态码: 200
            - 响应包含 success, filename, prediction, top_k 字段
            - prediction 包含 class_name 和 confidence
            - top_k 为包含 5 个结果的列表
        """
        response = client.post(
            "/classify",
            files={"file": ("test_image.jpg", BytesIO(sample_image_bytes), "image/jpeg")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["filename"] == "test_image.jpg"
        assert "prediction" in data
        assert "class_name" in data["prediction"]
        assert "confidence" in data["prediction"]
        assert "top_k" in data
        assert len(data["top_k"]) == 5
    
    def test_classify_png_image(self, client, sample_png_image):
        """
        测试使用 PNG 图像进行分类
        
        Args:
            client: FastAPI 测试客户端
            sample_png_image: 示例 PNG 图像字节数据
            
        Expected:
            - HTTP 状态码: 200
            - 响应包含正确的分类结果
        """
        response = client.post(
            "/classify",
            files={"file": ("test_image.png", BytesIO(sample_png_image), "image/png")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["filename"] == "test_image.png"
    
    def test_classify_unsupported_format(self, client):
        """
        测试上传不支持的文件格式
        
        Args:
            client: FastAPI 测试客户端
            
        Expected:
            - HTTP 状态码: 400
            - 响应包含错误信息
        """
        response = client.post(
            "/classify",
            files={"file": ("test_file.txt", BytesIO(b"invalid content"), "text/plain")}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "不支持的文件格式" in data["detail"]
    
    def test_classify_no_file(self, client):
        """
        测试未上传文件时的错误处理
        
        Args:
            client: FastAPI 测试客户端
            
        Expected:
            - HTTP 状态码: 422 (Unprocessable Entity)
        """
        response = client.post("/classify")
        
        assert response.status_code == 422
    
    def test_classify_confidence_value_range(self, client, sample_image_bytes):
        """
        测试返回的置信度值在有效范围内
        
        Args:
            client: FastAPI 测试客户端
            sample_image_bytes: 示例图像字节数据
            
        Expected:
            - confidence 值在 0 到 1 之间
            - top_k 中所有结果的 confidence 都在 0 到 1 之间
        """
        response = client.post(
            "/classify",
            files={"file": ("test_image.jpg", BytesIO(sample_image_bytes), "image/jpeg")}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # 验证主预测结果的置信度
        confidence = data["prediction"]["confidence"]
        assert 0 <= confidence <= 1
        
        # 验证 top_k 结果的置信度
        for result in data["top_k"]:
            assert 0 <= result["confidence"] <= 1
    
    def test_classify_top_k_order(self, client, sample_image_bytes):
        """
        测试 top_k 结果按置信度降序排列
        
        Args:
            client: FastAPI 测试客户端
            sample_image_bytes: 示例图像字节数据
            
        Expected:
            - top_k 结果按 confidence 从高到低排列
        """
        response = client.post(
            "/classify",
            files={"file": ("test_image.jpg", BytesIO(sample_image_bytes), "image/jpeg")}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # 验证 top_k 按置信度降序排列
        confidences = [result["confidence"] for result in data["top_k"]]
        assert confidences == sorted(confidences, reverse=True)
    
    def test_classify_various_extensions(self, client, sample_image_bytes):
        """
        测试各种支持的文件扩展名
        
        Args:
            client: FastAPI 测试客户端
            sample_image_bytes: 示例图像字节数据
            
        Expected:
            - 支持的扩展名: .jpg, .jpeg, .png, .bmp, .gif, .webp
            - 所有支持的扩展名都能成功处理
        """
        extensions = [".jpg", ".jpeg", ".JPG", ".JPEG"]
        
        for ext in extensions:
            response = client.post(
                "/classify",
                files={"file": (f"test_image{ext}", BytesIO(sample_image_bytes), "image/jpeg")}
            )
            
            assert response.status_code == 200, f"扩展名 {ext} 应该被支持"


class TestCORS:
    """CORS 跨域测试类"""
    
    def test_cors_headers_present(self, client):
        """
        测试 CORS 响应头是否存在
        
        Args:
            client: FastAPI 测试客户端
            
        Expected:
            - 响应包含 access-control-allow-origin 头
        """
        response = client.get("/health")
        
        # CORS 中间件应该添加相应的响应头
        assert "access-control-allow-origin" in response.headers
    
    def test_cors_preflight_request(self, client):
        """
        测试 CORS 预检请求
        
        Args:
            client: FastAPI 测试客户端
            
        Expected:
            - OPTIONS 请求返回 200 状态码
        """
        response = client.options("/health")
        
        assert response.status_code == 200
