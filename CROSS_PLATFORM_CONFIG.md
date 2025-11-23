# 跨平台配置总结

本文档总结从Windows环境迁移到Linux环境所需的主要配置和修改。

## ✅ 已完成的修复

### 1. train_cls.py - 硬编码路径修复

**问题**: 第40-43行使用了Windows绝对路径
```python
# 原代码 (Windows路径)
default='D:\研究生_study\cloudpoint_learn\Pointnet2_small_sample\data\...'
```

**修复**: 改为相对路径，自动适配项目根目录
```python
# 修复后 (相对路径)
default='data/modelnet5_normal_resampled/modelnet5_shape_names.txt'
```

### 2. test_cls.py - 路径拼接修复

**问题**: 使用字符串拼接路径，不跨平台
```python
# 原代码
experiment_dir = 'log/classification/' + args.log_dir
checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
```

**修复**: 使用 `os.path.join()` 进行跨平台路径拼接
```python
# 修复后
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
experiment_dir = os.path.join(BASE_DIR, 'log', 'classification', args.log_dir)
checkpoint_path = os.path.join(experiment_dir, 'checkpoints', 'best_model.pth')
```

## 🔧 Linux系统需要配置的内容

### 1. 环境依赖

```bash
# 系统依赖
sudo apt-get install python3 python3-pip python3-dev build-essential g++

# Python依赖
pip install -r requirements.txt

# CUDA (如果使用GPU)
# 安装匹配的CUDA版本和PyTorch
```

### 2. Visualizer编译

```bash
cd visualizer
chmod +x build.sh
./build.sh
```

### 3. 数据路径

项目现在使用相对路径，确保数据目录结构正确:
```
Pointnet2_small_sample/
├── data/
│   └── modelnet5_normal_resampled/
│       ├── modelnet5_shape_names.txt
│       ├── modelnet5_train.txt
│       ├── modelnet5_test.txt
│       └── [类别文件夹]/
└── [脚本文件]
```

### 4. 运行命令

```bash
# 训练
python train_cls.py --model pointnet2_cls_msg --normal --log_dir pointnet2_cls_msg

# 测试
python test_cls.py --log_dir pointnet2_cls_msg --normal
```

## 📋 主要差异

| 项目 | Windows | Linux |
|------|---------|-------|
| 路径分隔符 | `\` 或 `/` | `/` |
| 动态库扩展名 | `.dll` | `.so` |
| 路径处理 | 字符串拼接可能有问题 | 使用 `os.path.join()` |
| num_workers | 建议 0-2 | 可以使用 8+ |
| 编译工具 | Visual Studio | g++ |

## ⚠️ 注意事项

1. **路径**: 避免硬编码绝对路径，使用相对路径或 `os.path.join()`
2. **编码**: 避免在路径中使用特殊字符（如中文），或设置正确的locale
3. **权限**: 确保有读写权限
4. **CUDA**: 确保CUDA版本与PyTorch版本匹配
5. **依赖**: 使用虚拟环境隔离依赖

## 📚 详细文档

- **Linux配置**: 参见 `LINUX_SETUP.md`
- **项目说明**: 参见 `README.md`

## ✅ 验证清单

在Linux环境运行前，确认:

- [x] 硬编码路径已修复
- [x] 路径拼接使用 `os.path.join()`
- [ ] Python 3.6+ 已安装
- [ ] PyTorch 已安装
- [ ] CUDA 已配置 (如果使用GPU)
- [ ] 数据目录结构正确
- [ ] Visualizer已编译 (如果需要)
- [ ] 在项目根目录运行脚本

---

**更新日期**: 2024年
**版本**: 1.0

