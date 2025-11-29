#!/bin/bash

# PointNet2环境配置脚本 (Linux/Mac)
echo -e "\033[1;36m正在开始配置PointNet2项目环境...\033[0m"

# 检查Python是否已安装
if ! command -v python3 &> /dev/null; then
    echo -e "\033[1;31m错误: 未找到Python3。请先安装Python 3.6或更高版本。\033[0m"
    exit 1
fi

# 检查Python版本
PYTHON_VERSION=$(python3 --version 2>&1)
echo -e "\033[1;32m当前Python版本: $PYTHON_VERSION\033[0m"

# 创建虚拟环境
echo -e "\033[1;33m正在创建虚拟环境 'pointnet2_env'...\033[0m"
python3 -m venv pointnet2_env

# 激活虚拟环境
echo -e "\033[1;33m正在激活虚拟环境...\033[0m"
source pointnet2_env/bin/activate

# 升级pip
echo -e "\033[1;33m正在升级pip...\033[0m"
pip install --upgrade pip

# 安装依赖包
echo -e "\033[1;33m正在安装项目依赖包...\033[0m"
if [ -f "./requirements.txt" ]; then
    pip install -r requirements.txt
else
    # 如果requirements.txt不存在，直接安装所需包
    pip install torch>=1.6.0 torchvision>=0.7.0 numpy>=1.19.0 tqdm>=4.50.0 opencv-python>=4.4.0 plyfile>=0.7.4
fi

# 检查CUDA环境（可选）
echo -e "\033[1;33m正在检查CUDA环境...\033[0m"
python3 -c "import torch; print('CUDA可用:', torch.cuda.is_available())" || echo -e "\033[1;33m警告: 未检测到CUDA支持，将使用CPU模式。若要使用GPU，请确保安装了与CUDA兼容的PyTorch版本。\033[0m"

# 创建必要的目录结构
echo -e "\033[1;33m正在创建必要的目录结构...\033[0m"
mkdir -p ./log/classification

# 验证安装
echo -e "\033[1;33m正在验证安装...\033[0m"
python3 -c "import torch; import numpy; import tqdm; import cv2; import plyfile; print('所有依赖包安装成功！')"
if [ $? -eq 0 ]; then
    echo -e "\033[1;32m环境配置完成！\033[0m"
    echo -e "\033[1;37m使用方法：\033[0m"
    echo -e "\033[1;37m1. 每次使用前请激活虚拟环境：source pointnet2_env/bin/activate\033[0m"
    echo -e "\033[1;37m2. 训练模型示例：python train_cls.py --model pointnet2_cls_ssg --batch_size 16\033[0m"
    echo -e "\033[1;37m3. 测试模型示例：python test_cls.py --model pointnet2_cls_ssg --batch_size 16\033[0m"
else
    echo -e "\033[1;31m错误: 依赖包安装不完整，请检查错误信息。\033[0m"
    exit 1
fi
