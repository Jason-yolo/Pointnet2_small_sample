# PointNet2 快速开始指南

## 🚀 5分钟快速运行

### 方式一：使用批处理文件（Windows）

1. **双击运行** `run_training.bat`
2. **选择任务**（1-3）
3. **等待训练完成**

### 方式二：命令行运行

#### 步骤1：打开命令行（PowerShell或CMD）

```powershell
# 进入项目目录
cd D:\研究生_study\cloudpoint_learn\Pointnet2
```

#### 步骤2：检查环境

```bash
# 检查Python
python --version

# 检查PyTorch
python -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available())"
```

#### 步骤3：运行训练

**点云分类（推荐新手）：**
```bash
python train_cls.py --model pointnet2_cls_msg --normal --log_dir my_training --batch_size 4 --epoch 10
```

**参数说明：**
- `--batch_size 4`: 小批次，避免内存不足
- `--epoch 10`: 只训练10轮，快速测试
- `--log_dir my_training`: 自定义日志目录

## 📝 完整训练命令

### 点云分类（完整训练）

```bash
python train_cls.py --model pointnet2_cls_msg --normal --log_dir pointnet2_cls_msg --batch_size 8 --epoch 200 --gpu 0 --num_point 1024
```

### 点云部件分割

```bash
python train_partseg.py --model pointnet2_part_seg_msg --normal --log_dir pointnet2_part_seg_msg --batch_size 4 --epoch 251 --npoint 2048
```

### 点云语义分割

```bash
python train_semseg.py --model pointnet2_sem_seg --log_dir pointnet2_sem_seg --test_area 5 --batch_size 16 --epoch 128 --npoint 4096
```

## 🔍 查看训练结果

训练完成后，结果保存在：
```
log/
├── classification/      # 分类任务日志
│   └── [log_dir]/
│       ├── checkpoints/    # 模型文件
│       │   └── best_model.pth
│       ├── logs/           # 训练日志
│       │   └── pointnet2_cls_msg.txt
│       └── eval.txt        # 测试结果（运行test_cls.py后生成）
├── part_seg/           # 部件分割日志
└── sem_seg/            # 语义分割日志
```

## 🧪 测试训练好的模型

训练完成后，使用 `test_cls.py` 评估模型性能：

```bash
# 基础测试命令
python test_cls.py --log_dir pointnet2_cls_msg --normal

# 输出示例：
# Test Instance Accuracy: 0.908765, Class Accuracy: 0.897654
```

**测试结果说明：**
- **Test Instance Accuracy**: 测试集实例准确率（最重要指标，通常 > 90%）
- **Class Accuracy**: 类别平均准确率
- 结果会保存到 `log/classification/[log_dir]/eval.txt`

**详细说明请查看：** `测试结果说明.md`

## ⚙️ 常见参数调整

### 如果GPU内存不足

```bash
# 减小批次大小
--batch_size 2

# 减少点云数量
--num_point 512
```

### 如果训练速度太慢

```bash
# 减少训练轮数（用于测试）
--epoch 10

# 使用SSG模型（比MSG快）
--model pointnet2_cls_ssg
```

### 如果只有CPU

修改代码中的 `.cuda()` 为 `.cpu()`，或使用环境变量：
```bash
# PowerShell
$env:CUDA_VISIBLE_DEVICES=""
python train_cls.py --model pointnet2_cls_msg --normal --log_dir pointnet2_cls_msg
```

## 🐛 常见错误解决

### 错误1：CUDA out of memory
**解决：** 减小 `--batch_size` 到 2 或 4

### 错误2：ModuleNotFoundError
**解决：** 运行 `pip install -r requirements.txt`

### 错误3：数据集路径错误
**解决：** 确认 `data/modelnet40_normal_resampled/` 文件夹存在

### 错误4：训练很慢
**解决：** 
- 检查是否使用了GPU：`torch.cuda.is_available()`
- 减小 `--num_point` 参数
- 减少 `--epoch` 参数用于测试

## 📊 监控训练过程

训练时会实时显示：
- Epoch进度
- Train Instance Accuracy（训练准确率）
- Test Instance Accuracy（测试准确率）
- Best Model保存提示

## 🎯 下一步

1. **查看训练日志**：`log/classification/[log_dir]/logs/`
2. **测试模型**：`python test_cls.py --log_dir [log_dir] --normal`
3. **调整超参数**：修改学习率、批次大小等
4. **可视化结果**：使用 `visualizer/` 中的工具

## 💡 提示

- 首次运行建议使用小参数测试（`--epoch 10 --batch_size 4`）
- 训练完整模型需要较长时间（几小时到几天）
- 建议使用GPU训练，CPU训练非常慢
- 定期查看日志文件了解训练进度

