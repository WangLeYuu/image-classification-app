"""
生成系统架构图

使用 Graphviz 生成图像分类应用的系统架构图
"""
from graphviz import Digraph


def create_architecture_diagram():
    """
    创建系统架构图
    
    展示前端、后端和模型之间的交互关系
    """
    # 创建有向图
    dot = Digraph(
        name='image_classification_architecture',
        format='png',
        engine='dot'
    )
    
    # 设置全局属性
    dot.attr(
        rankdir='TB',  # 从上到下布局
        bgcolor='white',
        fontname='Microsoft YaHei',
        fontsize='20',
        label='图像分类应用系统架构图'
    )
    
    # 定义节点样式
    dot.attr('node', shape='box', style='rounded,filled', fontname='Microsoft YaHei', fontsize='12')
    dot.attr('edge', fontname='Microsoft YaHei', fontsize='10')
    
    # 前端层
    with dot.subgraph(name='cluster_frontend') as frontend:
        frontend.attr(
            label='前端层 (Vue.js + Element Plus)',
            style='rounded,filled',
            color='#E3F2FD',
            bgcolor='#E3F2FD'
        )
        frontend.node('ui', '用户界面\n(图像上传、预览)', fillcolor='#2196F3', fontcolor='white')
        frontend.node('api_client', 'API 客户端\n(Axios)', fillcolor='#64B5F6')
    
    # 后端层
    with dot.subgraph(name='cluster_backend') as backend:
        backend.attr(
            label='后端层 (FastAPI)',
            style='rounded,filled',
            color='#E8F5E9',
            bgcolor='#E8F5E9'
        )
        backend.node('api', 'RESTful API\n(/classify, /health)', fillcolor='#4CAF50', fontcolor='white')
        backend.node('cors', 'CORS 中间件', fillcolor='#81C784')
        backend.node('img_processor', '图像处理器\n(ImageProcessor)', fillcolor='#A5D6A7')
    
    # 模型层
    with dot.subgraph(name='cluster_model') as model:
        model.attr(
            label='模型层 (PyTorch)',
            style='rounded,filled',
            color='#FFF3E0',
            bgcolor='#FFF3E0'
        )
        model.node('model_loader', '模型加载器\n(ModelLoader)', fillcolor='#FF9800', fontcolor='white')
        model.node('mobilenet', 'MobileNetV2\n预训练模型', fillcolor='#FFB74D')
        model.node('imagenet', 'ImageNet\n1000类别', fillcolor='#FFCC80')
    
    # 添加边（连接）
    # 用户到前端
    dot.node('user', '用户', shape='ellipse', fillcolor='#F44336', fontcolor='white')
    dot.edge('user', 'ui', label='上传图像')
    
    # 前端内部
    dot.edge('ui', 'api_client', label='调用')
    
    # 前端到后端
    dot.edge('api_client', 'api', label='HTTP POST /classify', color='#2196F3', penwidth='2')
    
    # 后端内部
    dot.edge('api', 'cors', style='dashed', color='gray')
    dot.edge('api', 'img_processor', label='预处理图像')
    
    # 后端到模型
    dot.edge('img_processor', 'model_loader', label='输入张量')
    
    # 模型内部
    dot.edge('model_loader', 'mobilenet', label='加载')
    dot.edge('mobilenet', 'imagenet', label='分类', style='dashed')
    
    # 返回结果
    dot.edge('model_loader', 'api', label='预测结果', color='#4CAF50', penwidth='2')
    dot.edge('api', 'api_client', label='JSON 响应', color='#4CAF50', penwidth='2')
    dot.edge('api_client', 'ui', label='显示结果')
    dot.edge('ui', 'user', label='展示分类结果')
    
    # 添加图例
    with dot.subgraph(name='cluster_legend') as legend:
        legend.attr(
            label='图例',
            style='rounded,filled',
            color='#F5F5F5',
            bgcolor='#F5F5F5'
        )
        legend.node('frontend_color', '前端组件', fillcolor='#2196F3', fontcolor='white')
        legend.node('backend_color', '后端组件', fillcolor='#4CAF50', fontcolor='white')
        legend.node('model_color', '模型组件', fillcolor='#FF9800', fontcolor='white')
    
    # 保存图表
    output_path = '/Users/mima0000/code/pytorchproject/image-classification-app/architecture_diagram'
    dot.render(output_path, cleanup=True)
    print(f'架构图已生成: {output_path}.png')
    
    return dot


if __name__ == '__main__':
    create_architecture_diagram()
