import os
from openai import OpenAI

client = OpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key="08a4779d-df62-4c28-a9ff-081cfc6670c8",
)

prompt = """系统架构图，展示前后端分离的图像分类应用：
- 前端：Vue.js 用户界面，包含图片上传区域、图片预览、分类结果显示
- 后端：FastAPI RESTful API，提供 /predict 接口
- 模型：PyTorch MobileNetV2 预训练模型，进行图像分类推理
- 数据流：用户上传图片 -> 前端预览 -> 发送到后端 API -> 模型推理 -> 返回分类结果 -> 前端展示
- 技术栈标注：Vue.js, Axios, FastAPI, Uvicorn, PyTorch, MobileNetV2
- 清晰的箭头指示数据流向
- 现代简洁的设计风格，专业架构图"""

imagesResponse = client.images.generate(
    model="ep-20251205103128-7pcc4",
    prompt=prompt,
    size="2K",
    response_format="url",
    extra_body={
        "watermark": False,
    },
)

image_url = imagesResponse.data[0].url
print(f"Generated image URL: {image_url}")

import urllib.request
urllib.request.urlretrieve(image_url, "architecture_diagram.png")
print("Downloaded architecture_diagram.png")
