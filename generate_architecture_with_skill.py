"""
使用 Seedream Skill 生成系统架构图
"""
from openai import OpenAI
import requests


def generate_architecture_diagram():
    """
    生成图像分类应用的系统架构图
    """
    client = OpenAI(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key="08a4779d-df62-4c28-a9ff-081cfc6670c8",
    )

    # 架构图提示词
    prompt = """专业软件系统架构图，展示前后端分离的图像分类应用。

架构包含三个主要层次：

1. 前端层（蓝色主题）：
   - Vue.js 3 + TypeScript
   - Element Plus UI 组件库
   - 图像上传组件（支持拖拽）
   - 图片预览区域
   - 分类结果展示面板
   - Axios HTTP 客户端

2. 后端层（绿色主题）：
   - FastAPI 框架
   - RESTful API 端点 (/classify, /health)
   - CORS 中间件
   - 图像处理器 (ImageProcessor)
   - 模型加载器 (ModelLoader)

3. 模型层（橙色主题）：
   - PyTorch 深度学习框架
   - MobileNetV2 预训练模型
   - ImageNet 1000 类别分类

数据流向：
用户上传图片 → 前端 Vue.js → HTTP POST 请求 → FastAPI → 图像预处理 → MobileNetV2 推理 → 返回 JSON 结果 → 前端展示

风格要求：
- 专业架构图风格，类似 AWS/Azure 架构图
- 清晰的层次结构，使用不同颜色区分层次
- 包含箭头表示数据流向
- 现代、简洁、专业的设计
- 白色背景
- 中文标注
"""

    try:
        images_response = client.images.generate(
            model="ep-20251205103128-7pcc4",
            prompt=prompt,
            size="2K",
            response_format="url",
            extra_body={
                "watermark": False,
            },
        )

        image_url = images_response.data[0].url
        print(f"架构图已生成: {image_url}")

        # 下载图片
        response = requests.get(image_url, timeout=30)
        if response.status_code == 200:
            output_path = "/Users/mima0000/code/pytorchproject/image-classification-app/architecture_diagram_ai.png"
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"架构图已保存: {output_path}")
            return output_path
        else:
            print(f"下载图片失败: {response.status_code}")
            return None

    except Exception as e:
        print(f"生成架构图失败: {e}")
        return None


if __name__ == "__main__":
    generate_architecture_diagram()
