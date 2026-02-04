"""
pytest 共享配置和 fixtures 模块

该模块包含所有测试模块共享的 pytest 配置和 fixtures
"""
import pytest
import sys
import os

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


# 可以在这里添加更多共享 fixtures
