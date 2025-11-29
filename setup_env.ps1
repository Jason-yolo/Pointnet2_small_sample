# PointNet2环境配置脚本 (Windows PowerShell)
Write-Host "正在开始配置PointNet2项目环境..." -ForegroundColor Cyan

# 检查Python是否已安装
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "错误: 未找到Python。请先安装Python 3.6或更高版本。" -ForegroundColor Red
    exit 1
}

# 检查Python版本
$pythonVersion = python --version 2>&1
Write-Host "当前Python版本: $pythonVersion" -ForegroundColor Green

# 创建虚拟环境
Write-Host "正在创建虚拟环境 'pointnet2_env'..." -ForegroundColor Yellow
python -m venv pointnet2_env

# 激活虚拟环境
Write-Host "正在激活虚拟环境..." -ForegroundColor Yellow
.\pointnet2_env\Scripts\Activate.ps1

# 升级pip
Write-Host "正在升级pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# 安装依赖包
Write-Host "正在安装项目依赖包..." -ForegroundColor Yellow
if (Test-Path ".\requirements.txt") {
    python -m pip install -r requirements.txt
} else {
    # 如果requirements.txt不存在，直接安装所需包
    python -m pip install torch>=1.6.0 torchvision>=0.7.0 numpy>=1.19.0 tqdm>=4.50.0 opencv-python>=4.4.0 plyfile>=0.7.4
}

# 检查CUDA环境（可选）
try {
    python -c "import torch; print('CUDA可用:', torch.cuda.is_available())"
    $cudaResult = $?
} catch {
    $cudaResult = $false
}

if (-not $cudaResult) {
    Write-Host "警告: 未检测到CUDA支持，将使用CPU模式。若要使用GPU，请确保安装了与CUDA兼容的PyTorch版本。" -ForegroundColor Yellow
}

# 创建必要的目录结构
Write-Host "正在创建必要的目录结构..." -ForegroundColor Yellow
if (-not (Test-Path ".\log")) { New-Item -ItemType Directory -Path ".\log" }
if (-not (Test-Path ".\log\classification")) { New-Item -ItemType Directory -Path ".\log\classification" }

# 验证安装
Write-Host "正在验证安装..." -ForegroundColor Yellow
try {
    python -c "import torch; import numpy; import tqdm; import cv2; import plyfile; print('所有依赖包安装成功！')"
    Write-Host "环境配置完成！" -ForegroundColor Green
    Write-Host "使用方法：" -ForegroundColor White
    Write-Host "1. 每次使用前请激活虚拟环境：.\pointnet2_env\Scripts\Activate.ps1" -ForegroundColor White
    Write-Host "2. 训练模型示例：python train_cls.py --model pointnet2_cls_ssg --batch_size 16" -ForegroundColor White
    Write-Host "3. 测试模型示例：python test_cls.py --model pointnet2_cls_ssg --batch_size 16" -ForegroundColor White
} catch {
    Write-Host "错误: 依赖包安装不完整，请检查错误信息。" -ForegroundColor Red
    exit 1
}
