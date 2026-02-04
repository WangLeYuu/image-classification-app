"""
FastAPI 主应用模块
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

# 添加父目录到 Python 路径，以便导入 models 和 utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_loader import model_loader
from utils.image_processor import image_processor


def create_app() -> FastAPI:
    """
    创建并配置 FastAPI 应用实例
    
    Returns:
        FastAPI: 配置好的应用实例
    """
    app = FastAPI(
        title="图像分类 API",
        description="基于 PyTorch 的图像分类服务",
        version="1.0.0"
    )
    
    # 配置 CORS 中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app


# 创建应用实例
app = create_app()


@app.get("/")
async def root():
    """
    根路径端点
    
    Returns:
        dict: 欢迎消息
    """
    return {
        "message": "欢迎使用图像分类 API",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """
    健康检查端点
    
    Returns:
        dict: 健康状态
    """
    return {
        "status": "healthy"
    }


@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    """
    图像分类端点
    
    接收上传的图像文件，使用 MobileNetV2 模型进行分类，
    返回预测的类别名称和置信度分数
    
    Args:
        file: 上传的图像文件
        
    Returns:
        dict: 包含分类结果的字典
            - class_name: 预测的类别名称
            - confidence: 置信度分数 (0-1)
            - top_k: 前5个预测结果列表
            
    Raises:
        HTTPException: 当文件格式不支持或处理出错时
    """
    # 验证文件类型
    allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {file_ext}。支持的格式: {', '.join(allowed_extensions)}"
        )
    
    try:
        # 读取图像文件内容
        image_bytes = await file.read()
        
        # 预处理图像
        image_tensor = image_processor.process_from_bytes(image_bytes)
        
        # 使用模型进行预测
        class_name, confidence = model_loader.predict(image_tensor)
        
        # 获取前5个预测结果
        top_k_results = model_loader.predict_top_k(image_tensor, k=5)
        
        # 返回分类结果
        return {
            "success": True,
            "filename": file.filename,
            "prediction": {
                "class_name": class_name,
                "confidence": round(confidence, 4)
            },
            "top_k": [
                {
                    "class_name": result["class_name"],
                    "confidence": round(result["confidence"], 4)
                }
                for result in top_k_results
            ]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"图像处理失败: {str(e)}"
        )
