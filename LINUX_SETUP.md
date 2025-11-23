# Linux环境配置指南

本文档说明如何在Linux系统上配置和运行PointNet2项目。

## 📋 目录

1. [环境要求](#环境要求)
2. [安装步骤](#安装步骤)
3. [代码修改说明](#代码修改说明)
4. [路径配置](#路径配置)
5. [Visualizer编译](#visualizer编译)
6. [常见问题](#常见问题)

---

## 🔧 环境要求

### 系统要求
- **操作系统**: Ubuntu 18.04+ / CentOS 7+ / Debian 10+ 或其他Linux发行版
- **Python**: 3.6 - 3.9 (推荐 3.7 或 3.8)
- **CUDA**: 10.0+ (如果使用GPU，推荐 CUDA 11.0+)
- **内存**: 至少 8GB RAM (推荐 16GB+)
- **存储**: 至少 5GB 可用空间

### 必需软件
- Python 3.x
- pip 或 conda
- g++ 编译器 (用于编译visualizer)
- CUDA Toolkit (如果使用GPU)

---

## 📦 安装步骤

### 1. 安装系统依赖

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-dev build-essential
sudo apt-get install -y libopenblas-dev liblapack-dev
```

#### CentOS/RHEL
```bash
sudo yum install -y python3 python3-pip python3-devel gcc gcc-c++
sudo yum install -y openblas-devel lapack-devel
```

### 2. 安装CUDA (如果使用GPU)

访问 [NVIDIA CUDA官网](https://developer.nvidia.com/cuda-downloads) 下载并安装对应版本的CUDA Toolkit。

验证CUDA安装:
```bash
nvcc --version
nvidia-smi
```

### 3. 创建Python虚拟环境 (推荐)

```bash
# 使用venv
python3 -m venv pointnet2_env
source pointnet2_env/bin/activate

# 或使用conda
conda create -n pointnet2 python=3.7
conda activate pointnet2
```

### 4. 安装PyTorch

根据你的CUDA版本安装PyTorch。访问 [PyTorch官网](https://pytorch.org/) 获取正确的安装命令。

**CUDA 11.0 示例:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu110
```

**CPU版本 (无GPU):**
```bash
pip install torch torchvision torchaudio
```

### 5. 安装项目依赖

```bash
# 进入项目根目录
cd /path/to/Pointnet2_small_sample

# 安装依赖
pip install -r requirements.txt
```

### 6. 验证安装

```bash
python3 -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"
python3 -c "import numpy; import tqdm; import cv2; print('依赖包安装成功')"
```

---

## 🔄 代码修改说明

### 已修复的问题

#### 1. ✅ train_cls.py - 硬编码路径问题
**修改位置**: 第40-43行

**原代码 (Windows路径)**:
```python
parser.add_argument('--shape_names_file', type=str, default='D:\研究生_study\...', ...)
parser.add_argument('--train_file', type=str, default='D:\研究生_study\...', ...)
parser.add_argument('--test_file', type=str, default='D:\研究生_study\...', ...)
parser.add_argument('--data_path', type=str, default='D:\研究生_study\...', ...)
```

**已修改为 (相对路径)**:
```python
parser.add_argument('--shape_names_file', type=str, default='data/modelnet5_normal_resampled/modelnet5_shape_names.txt', ...)
parser.add_argument('--train_file', type=str, default='data/modelnet5_normal_resampled/modelnet5_train.txt', ...)
parser.add_argument('--test_file', type=str, default='data/modelnet5_normal_resampled/modelnet5_test.txt', ...)
parser.add_argument('--data_path', type=str, default='data/modelnet5_normal_resampled', ...)
```

#### 2. ✅ test_cls.py - 路径拼接问题
**修改位置**: 多处路径拼接

**原代码**:
```python
experiment_dir = 'log/classification/' + args.log_dir
checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth', ...)
```

**已修改为**:
```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
experiment_dir = os.path.join(BASE_DIR, 'log', 'classification', args.log_dir)
checkpoint_path = os.path.join(experiment_dir, 'checkpoints', 'best_model.pth')
checkpoint = torch.load(checkpoint_path, ...)
```

---

## 📂 路径配置

### 数据路径配置

项目现在使用相对路径，确保数据目录结构如下:

```
Pointnet2_small_sample/
├── data/
│   └── modelnet5_normal_resampled/
│       ├── modelnet5_shape_names.txt
│       ├── modelnet5_train.txt
│       ├── modelnet5_test.txt
│       ├── airplane/
│       ├── bathtub/
│       ├── bed/
│       ├── bench/
│       └── bookshelf/
├── train_cls.py
└── test_cls.py
```

### 自定义路径

如果数据在其他位置，可以通过命令行参数指定:

```bash
python train_cls.py \
    --data_path /path/to/your/data/modelnet5_normal_resampled \
    --shape_names_file /path/to/your/data/modelnet5_shape_names.txt \
    --train_file /path/to/your/data/modelnet5_train.txt \
    --test_file /path/to/your/data/modelnet5_test.txt \
    --model pointnet2_cls_msg \
    --normal \
    --log_dir pointnet2_cls_msg
```

---

## 🔨 Visualizer编译

`visualizer` 目录包含C++扩展，需要在Linux上编译。

### 编译步骤

1. **进入visualizer目录**:
```bash
cd visualizer
```

2. **编译C++扩展**:
```bash
# 修改build.sh中的编译选项（如需要）
chmod +x build.sh
./build.sh
```

3. **如果编译失败，手动编译**:
```bash
g++ -std=c++11 render_balls_so.cpp -o render_balls_so.so -shared -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=0
```

**注意**: 
- 如果使用较新的GCC版本，可能需要移除 `-D_GLIBCXX_USE_CXX11_ABI=0` 参数
- 确保安装了g++编译器: `sudo apt-get install g++`

### Windows vs Linux 动态库扩展名

- **Linux**: `.so` (shared object)
- **Windows**: `.dll` (dynamic link library)
- **macOS**: `.dylib` (dynamic library)

代码中的 `np.ctypeslib.load_library()` 会根据平台自动选择合适的扩展名。

---

## 🚀 运行项目

### 训练模型

```bash
# 基础训练命令
python train_cls.py --model pointnet2_cls_msg --normal --log_dir pointnet2_cls_msg

# 完整参数示例
python train_cls.py \
    --model pointnet2_cls_msg \
    --normal \
    --log_dir pointnet2_cls_msg \
    --batch_size 8 \
    --epoch 200 \
    --gpu 0 \
    --num_point 1024 \
    --num_workers 8
```

### 测试模型

```bash
# 测试模型
python test_cls.py \
    --log_dir pointnet2_cls_msg \
    --normal \
    --num_votes 3

# 测试单个文件
python test_cls.py \
    --log_dir pointnet2_cls_msg \
    --normal \
    --single_file "data/modelnet5_normal_resampled/airplane/airplane_0001.txt" \
    --num_votes 3
```

### 无GPU运行 (CPU模式)

如果系统没有GPU，需要修改代码中的 `.cuda()` 调用，或者设置环境变量:

```bash
# 方法1: 设置CUDA_VISIBLE_DEVICES为空
CUDA_VISIBLE_DEVICES="" python train_cls.py --model pointnet2_cls_msg --normal

# 方法2: 修改代码（需要手动修改，将所有.cuda()改为.cpu()）
```

---

## ⚠️ 常见问题

### 1. 路径错误

**问题**: `FileNotFoundError: [Errno 2] No such file or directory`

**解决方法**:
- 确保在项目根目录运行脚本
- 检查数据路径是否正确
- 使用绝对路径或相对于项目根目录的相对路径

```bash
# 检查当前目录
pwd
# 应该显示: /path/to/Pointnet2_small_sample

# 检查数据目录
ls -la data/modelnet5_normal_resampled/
```

### 2. 权限问题

**问题**: `Permission denied` 或无法创建日志目录

**解决方法**:
```bash
# 确保有写入权限
chmod -R 755 .
# 或
chmod -R u+w .
```

### 3. CUDA版本不匹配

**问题**: `RuntimeError: CUDA error` 或 PyTorch找不到CUDA

**解决方法**:
- 检查CUDA版本: `nvcc --version`
- 安装匹配的PyTorch版本
- 确保CUDA_HOME环境变量正确设置

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### 4. 依赖包版本冲突

**问题**: `ImportError` 或版本不兼容

**解决方法**:
```bash
# 创建新的虚拟环境
python3 -m venv fresh_env
source fresh_env/bin/activate

# 重新安装依赖
pip install -r requirements.txt --upgrade
```

### 5. 编译Visualizer失败

**问题**: `g++: command not found` 或编译错误

**解决方法**:
```bash
# 安装编译器
sudo apt-get install build-essential g++

# 检查GCC版本
g++ --version

# 如果仍有问题，尝试修改编译参数
cd visualizer
g++ -std=c++11 render_balls_so.cpp -o render_balls_so.so -shared -fPIC -O2
```

### 6. 中文路径问题

**问题**: 路径中包含中文字符导致错误

**解决方法**:
- 避免在路径中使用中文字符
- 如果必须使用，确保系统locale正确设置:

```bash
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
```

### 7. num_workers问题 (Windows vs Linux)

**问题**: DataLoader的`num_workers`在Windows上可能有问题

**解决方法**:
- Linux上可以使用较大的`num_workers` (如8)
- Windows上建议使用较小的值 (如0或2)
- 代码已经支持通过命令行参数调整

### 8. 内存不足

**问题**: `RuntimeError: CUDA out of memory`

**解决方法**:
```bash
# 减小批次大小
python train_cls.py --batch_size 4 --num_point 512 ...

# 减少num_workers
python train_cls.py --num_workers 2 ...
```

---

## 📝 环境变量配置 (可选)

创建 `setup_env.sh` 脚本:

```bash
#!/bin/bash
# 设置CUDA路径
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 设置locale (如果使用中文路径)
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

echo "环境变量配置完成"
```

使用:
```bash
source setup_env.sh
```

---

## ✅ 验证清单

在运行训练前，请确认:

- [ ] Python 3.6+ 已安装
- [ ] PyTorch 已安装并可正常导入
- [ ] CUDA已安装 (如果使用GPU)
- [ ] CUDA 版本与PyTorch版本匹配
- [ ] 所有依赖包已安装 (`pip list`)
- [ ] 数据目录结构正确
- [ ] 有足够的磁盘空间存储日志和模型
- [ ] 在项目根目录运行脚本

---

## 🔗 相关资源

- [PyTorch官方文档](https://pytorch.org/docs/stable/index.html)
- [CUDA Toolkit文档](https://docs.nvidia.com/cuda/)
- [PointNet论文](https://arxiv.org/abs/1612.00593)
- [PointNet++论文](https://arxiv.org/abs/1706.02413)

---

## 📧 问题反馈

如果遇到其他问题，请检查:
1. Python版本: `python --version`
2. PyTorch版本: `python -c "import torch; print(torch.__version__)"`
3. CUDA版本: `nvcc --version` (如果使用GPU)
4. 系统信息: `uname -a`
5. 依赖包版本: `pip list`

---

**最后更新**: 2024年
**适用版本**: PointNet2_small_sample

