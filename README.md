# PointNet2 点云处理项目

## 项目简介

这是一个基于 PyTorch 实现的 PointNet2 点云处理项目，支持以下三种任务：

1. **点云分类（Classification）** - 使用 ModelNet40 数据集进行3D物体分类
2. **点云部件分割（Part Segmentation）** - 使用 ShapeNet 数据集进行3D物体部件分割
3. **点云语义分割（Semantic Segmentation）** - 使用 S3DIS 数据集进行室内场景语义分割

## 项目结构

```
Pointnet2/
├── models/              # 模型定义文件
│   ├── pointnet2_cls_msg.py      # PointNet2分类模型（MSG版本）
│   ├── pointnet2_cls_ssg.py      # PointNet2分类模型（SSG版本）
│   ├── pointnet2_part_seg_msg.py # PointNet2部件分割模型
│   ├── pointnet2_sem_seg.py      # PointNet2语义分割模型
│   └── pointnet_util.py          # PointNet工具函数
├── data_utils/          # 数据加载器
│   ├── ModelNetDataLoader.py     # ModelNet数据加载
│   ├── ShapeNetDataLoader.py     # ShapeNet数据加载
│   └── S3DISDataLoader.py        # S3DIS数据加载
├── data/                # 数据集目录
│   ├── modelnet40_normal_resampled/      # ModelNet40数据集
│   └── shapenetcore_partanno_segmentation_benchmark_v0_normal/  # ShapeNet数据集
├── train_cls.py         # 分类训练脚本
├── train_partseg.py     # 部件分割训练脚本
├── train_semseg.py      # 语义分割训练脚本
├── test_cls.py          # 分类测试脚本
├── test_partseg.py      # 部件分割测试脚本
├── test_semseg.py       # 语义分割测试脚本
├── provider.py          # 数据增强工具
└── visualizer/          # 可视化工具
```

## 环境配置

### 系统要求

- Python 3.6+
- CUDA 10.0+ (推荐，用于GPU加速)
- Windows/Linux/macOS

### 安装步骤

1. **创建虚拟环境（推荐）**
   ```bash
   conda create -n pointnet2 python=3.7
   conda activate pointnet2
   ```

2. **安装 PyTorch**
   
   根据你的CUDA版本安装PyTorch，访问 [PyTorch官网](https://pytorch.org/) 获取安装命令。
   
   例如，CUDA 11.0:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu110
   ```
   
   或者CPU版本:
   ```bash
   pip install torch torchvision torchaudio
   ```

3. **安装其他依赖**
   ```bash
   pip install -r requirements.txt
   ```

## 快速开始

### 第一步：检查环境

1. **检查Python版本**
   ```bash
   python --version
   # 应该显示 Python 3.6 或更高版本
   ```

2. **检查PyTorch安装**
   ```bash
   python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
   ```

3. **检查数据集**
   ```bash
   # 确认数据集文件夹存在
   dir data\modelnet40_normal_resampled
   dir data\shapenetcore_partanno_segmentation_benchmark_v0_normal
   ```

### 第二步：安装依赖（如果尚未安装）

```bash
# 安装PyTorch（根据你的CUDA版本，访问 https://pytorch.org/ 获取正确命令）
pip install torch torchvision torchaudio

# 安装其他依赖
pip install -r requirements.txt
```

### 第三步：运行项目

## 使用方法

### 1. 点云分类（Classification）

**训练模型：**
```bash
# 基础训练命令
python train_cls.py --model pointnet2_cls_msg --normal --log_dir pointnet2_cls_msg

# 完整参数示例（自定义批次大小和训练轮数）
python train_cls.py --model pointnet2_cls_msg --normal --log_dir pointnet2_cls_msg --batch_size 16 --epoch 100 --gpu 0 --num_point 1024

# 使用SSG版本模型
python train_cls.py --model pointnet2_cls_ssg --normal --log_dir pointnet2_cls_ssg
```

**测试模型：**
```bash
# 测试分类模型（需要先训练或下载预训练模型）
python test_cls.py --log_dir pointnet2_cls_msg --normal --num_votes 3
```

**参数说明：**
- `--model`: 模型名称（pointnet2_cls_msg 或 pointnet2_cls_ssg）
- `--normal`: 使用法向量信息（推荐开启）
- `--batch_size`: 批次大小（默认：8，GPU内存不足时可减小）
- `--epoch`: 训练轮数（默认：200）
- `--gpu`: GPU设备编号（默认：0，多GPU时可指定）
- `--num_point`: 点云点数（默认：1024）
- `--learning_rate`: 学习率（默认：0.001）
- `--log_dir`: 日志和模型保存目录

### 2. 点云部件分割（Part Segmentation）

**训练模型：**
```bash
# 基础训练命令
python train_partseg.py --model pointnet2_part_seg_msg --normal --log_dir pointnet2_part_seg_msg

# 完整参数示例
python train_partseg.py --model pointnet2_part_seg_msg --normal --log_dir pointnet2_part_seg_msg --batch_size 8 --epoch 251 --npoint 2048
```

**测试模型：**
```bash
python test_partseg.py --model pointnet2_part_seg_msg --normal --log_dir pointnet2_part_seg_msg
```

### 3. 点云语义分割（Semantic Segmentation）

**训练模型：**
```bash
# 基础训练命令
python train_semseg.py --model pointnet2_sem_seg --log_dir pointnet2_sem_seg --test_area 5

# 完整参数示例
python train_semseg.py --model pointnet2_sem_seg --log_dir pointnet2_sem_seg --batch_size 16 --epoch 128 --npoint 4096 --test_area 5
```

**测试模型：**
```bash
python test_semseg.py --model pointnet2_sem_seg --log_dir pointnet2_sem_seg --test_area 5
```

## 运行示例

### 示例1：首次运行分类任务（推荐）

```bash
# 1. 进入项目目录
cd D:\研究生_study\cloudpoint_learn\Pointnet2

# 2. 运行训练（使用较小的批次大小，避免内存不足）
python train_cls.py --model pointnet2_cls_msg --normal --log_dir my_first_training --batch_size 4 --epoch 10

# 3. 训练完成后，查看日志
# 日志保存在：log/classification/my_first_training/logs/
```

### 示例2：快速测试（使用已有模型）

```bash
# 如果有预训练模型，可以直接测试
python test_cls.py --log_dir pointnet2_cls_msg --normal
```

### 示例3：CPU运行（无GPU）

如果没有GPU，需要修改代码：
1. 打开训练脚本（如 `train_cls.py`）
2. 将所有 `.cuda()` 替换为 `.cpu()`
3. 或者创建CPU版本的脚本

**或者使用环境变量：**
```bash
# Windows PowerShell
$env:CUDA_VISIBLE_DEVICES=""
python train_cls.py --model pointnet2_cls_msg --normal --log_dir pointnet2_cls_msg
```

## 训练输出说明

训练过程中会显示：
- **Train Instance Accuracy**: 训练集实例准确率
- **Test Instance Accuracy**: 测试集实例准确率
- **Class Accuracy**: 类别准确率
- **Best Model**: 最佳模型会自动保存到 `log/分类/任务名称/checkpoints/best_model.pth`

## 常见问题

### 1. CUDA out of memory（GPU内存不足）

**解决方法：**
- 减小批次大小：`--batch_size 4` 或 `--batch_size 2`
- 减少点云数量：`--num_point 512`（默认1024）
- 关闭其他占用GPU的程序

### 2. 数据集路径错误

**解决方法：**
- 确认 `data/modelnet40_normal_resampled/` 目录存在
- 确认 `data/modelnet40_normal_resampled/modelnet40_train.txt` 文件存在
- 检查文件路径中是否有中文字符（可能导致问题）

### 3. 模块导入错误

**解决方法：**
```bash
# 确保在项目根目录运行
cd D:\研究生_study\cloudpoint_learn\Pointnet2

# 检查Python路径
python -c "import sys; sys.path.append('.'); from data_utils.ModelNetDataLoader import ModelNetDataLoader; print('OK')"
```

### 4. 训练速度慢

**解决方法：**
- 使用GPU加速（确保CUDA和PyTorch GPU版本正确安装）
- 增加 `num_workers` 参数（在DataLoader中，默认是4）
- 减小 `num_point` 参数

### 5. 没有GPU怎么办？

**解决方法：**
- 可以使用CPU训练，但速度很慢
- 建议使用Google Colab等免费GPU平台
- 或者减小模型和数据规模进行测试

## 数据集

项目已包含以下数据集：

1. **ModelNet40**: 用于3D物体分类，包含40个类别
2. **ShapeNet**: 用于部件分割，包含16个物体类别和50个部件类别
3. **S3DIS**: 用于语义分割（如果已下载）

数据集路径：
- ModelNet40: `data/modelnet40_normal_resampled/`
- ShapeNet: `data/shapenetcore_partanno_segmentation_benchmark_v0_normal/`

## 注意事项

1. **GPU要求**: 项目默认使用CUDA，如果没有GPU，需要修改代码中的`.cuda()`为`.cpu()`
2. **内存要求**: 建议至少8GB内存，训练时可能需要更多
3. **数据路径**: 确保数据集路径正确，否则会报错
4. **批次大小**: 如果GPU内存不足，可以减小`batch_size`参数

## 参考

- PointNet论文: [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593)
- PointNet++论文: [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413)

## 许可证

本项目仅供学习研究使用。

