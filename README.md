# 图像分类应用

一个前后端分离的图像分类 Web 应用，基于 Vue.js + FastAPI + MobileNetV2 实现。

## 功能特性

- 🖼️ **图像上传**：支持拖拽和点击上传图片
- 👁️ **图片预览**：实时预览上传的图像
- 🤖 **智能分类**：使用 MobileNetV2 预训练模型进行图像分类
- 📊 **结果展示**：显示 Top-5 预测结果及置信度
- 📱 **响应式设计**：支持桌面和移动设备
- 🧪 **完整测试**：包含 50+ 单元测试

## 技术栈

### 前端
- Vue.js 3 + TypeScript
- Element Plus UI 组件库
- Axios HTTP 客户端
- Vite 构建工具

### 后端
- FastAPI 框架
- PyTorch + TorchVision
- MobileNetV2 预训练模型
- Pillow 图像处理

## 项目结构

```
image-classification-app/
├── backend/                 # FastAPI 后端
│   ├── app/
│   │   └── main.py         # API 主文件
│   ├── models/
│   │   └── model_loader.py # 模型加载器
│   ├── utils/
│   │   └── image_processor.py # 图像处理器
│   ├── tests/              # 单元测试
│   └── requirements.txt    # Python 依赖
├── frontend/               # Vue.js 前端
│   ├── src/
│   │   ├── views/
│   │   │   └── ImageClassificationView.vue
│   │   ├── api/
│   │   │   └── classification.ts
│   │   └── types/
│   │       └── classification.ts
│   └── package.json
└── architecture_diagram.png # 系统架构图
```

## 快速开始

### 环境要求
- Python 3.8+
- Node.js 16+
- npm 或 yarn

### 后端启动

```bash
cd backend

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# macOS/Linux:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 启动服务
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
```

后端服务将在 http://localhost:8001 运行

### 前端启动

```bash
cd frontend

# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

前端服务将在 http://localhost:5173 运行

## API 接口

### 健康检查
```
GET /health
Response: {"status": "healthy"}
```

### 图像分类
```
POST /classify
Content-Type: multipart/form-data

Parameters:
- file: 图像文件 (jpg, png, bmp, gif, webp)

Response:
{
  "success": true,
  "filename": "image.jpg",
  "prediction": {
    "class_name": "golden retriever",
    "confidence": 0.95
  },
  "top_k": [
    {"class_name": "golden retriever", "confidence": 0.95},
    {"class_name": "Labrador retriever", "confidence":