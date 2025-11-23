#!/usr/bin/env python3
"""
检测PointNet/PointNet++训练和测试代码的运行环境和文件完整性
"""

import os
import sys
import importlib
import subprocess
from pathlib import Path


# 设置颜色输出
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    RESET = '\033[0m'


def print_success(msg):
    print(f"{Colors.GREEN}[✓] {msg}{Colors.RESET}")


def print_warning(msg):
    print(f"{Colors.YELLOW}[!] {msg}{Colors.RESET}")


def print_error(msg):
    print(f"{Colors.RED}[✗] {msg}{Colors.RESET}")


def print_info(msg):
    print(f"{Colors.BLUE}[i] {msg}{Colors.RESET}")


def check_file_exists(file_path):
    """检查文件是否存在"""
    if os.path.exists(file_path):
        print_success(f"找到文件: {file_path}")
        return True
    else:
        print_error(f"缺失文件: {file_path}")
        return False


def check_directory_exists(dir_path):
    """检查目录是否存在"""
    if os.path.isdir(dir_path):
        print_success(f"找到目录: {dir_path}")
        return True
    else:
        print_warning(f"缺失目录: {dir_path} (将自动创建)")
        os.makedirs(dir_path, exist_ok=True)
        return False


def check_dependencies():
    """检查Python依赖包"""
    print_info("\n=== 检查Python依赖包 ===")

    required_packages = [
        'torch', 'numpy', 'tqdm', 'matplotlib', 'scipy', 'h5py'
    ]

    missing_packages = []

    for pkg in required_packages:
        try:
            importlib.import_module(pkg)
            print_success(f"{pkg} 已安装")
        except ImportError:
            print_error(f"{pkg} 未安装")
            missing_packages.append(pkg)

    return missing_packages


def check_pytorch_cuda():
    """检查PyTorch和CUDA"""
    print_info("\n=== 检查PyTorch和CUDA ===")

    try:
        import torch

        print_success(f"PyTorch版本: {torch.__version__}")

        if torch.cuda.is_available():
            print_success(f"CUDA可用: {torch.cuda.get_device_name(0)}")
            print_success(f"GPU数量: {torch.cuda.device_count()}")
            return True
        else:
            print_warning("CUDA不可用，将使用CPU训练（速度较慢）")
            return False
    except ImportError:
        print_error("PyTorch未安装")
        return False


def check_dataset():
    """检查数据集"""
    print_info("\n=== 检查数据集 ===")

    data_path = "data/modelnet5_normal_resampled"

    if os.path.isdir(data_path):
        # 检查关键文件
        required_files = [
            f"{data_path}/modelnet5_shape_names.txt",
            f"{data_path}/modelnet5_train.txt",
            f"{data_path}/modelnet5_test.txt"
        ]

        all_files_exist = True
        for file in required_files:
            if not os.path.exists(file):
                print_error(f"数据集文件缺失: {file}")
                all_files_exist = False
            else:
                print_success(f"数据集文件存在: {file}")

        if all_files_exist:
            print_success("数据集完整")
            return True
        else:
            print_warning("数据集不完整，请检查ModelNet5文件")
            return False
    else:
        print_error(f"数据集目录不存在: {data_path}")
        print_info("请下载ModelNet数据集并解压到该目录")
        return False


def check_code_files():
    """检查代码文件"""
    print_info("\n=== 检查代码文件 ===")

    required_files = [
        "train_cls.py",  # 训练代码
        "test_cls.py",  # 测试代码
        "data_utils/ModelNetDataLoader.py",  # 数据加载器
    ]

    required_dirs = [
        "models",
        "data_utils",
        "log/classification"
    ]

    all_files_ok = True

    # 检查文件
    for file in required_files:
        if not check_file_exists(file):
            all_files_ok = False

    # 检查目录
    for dir_path in required_dirs:
        check_directory_exists(dir_path)

    # 检查models目录下的文件
    if os.path.isdir("models"):
        model_files = [f for f in os.listdir("models") if f.endswith('.py')]
        if model_files:
            print_success(f"找到模型文件: {', '.join(model_files)}")
        else:
            print_warning("models目录下未找到模型文件")
            print_info("请确保包含pointnet2_cls_msg.py, pointnet_util.py等文件")

    return all_files_ok


def test_code_execution():
    """测试代码是否能运行"""
    print_info("\n=== 测试代码执行 ===")

    # 测试训练代码是否能解析参数
    try:
        result = subprocess.run(
            [sys.executable, "train_cls.py", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            print_success("训练代码参数解析正常")
        else:
            print_error(f"训练代码参数解析失败: {result.stderr}")
            return False
    except Exception as e:
        print_error(f"训练代码测试失败: {e}")
        return False

    # 测试测试代码是否能解析参数
    try:
        result = subprocess.run(
            [sys.executable, "test_cls.py", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            print_success("测试代码参数解析正常")
            return True
        else:
            print_error(f"测试代码参数解析失败: {result.stderr}")
            return False
    except Exception as e:
        print_error(f"测试代码测试失败: {e}")
        return False


def main():
    print_info("=== PointNet/PointNet++ 环境检测工具 ===\n")

    # 1. 检查代码文件
    code_ok = check_code_files()

    # 2. 检查依赖
    missing_pkgs = check_dependencies()

    # 3. 检查PyTorch和CUDA
    cuda_ok = check_pytorch_cuda()

    # 4. 检查数据集
    dataset_ok = check_dataset()

    # 5. 测试代码执行
    execution_ok = code_ok and test_code_execution()

    # 总结
    print_info("\n=== 检测总结 ===")

    if code_ok and not missing_pkgs and execution_ok:
        print_success("代码环境基本就绪！")

        if not dataset_ok:
            print_warning("但数据集缺失，请先下载ModelNet数据集")
        else:
            print_success("所有检查通过，可以开始训练！")
    else:
        print_error("存在问题需要解决，请根据上面的提示修复")

        if missing_pkgs:
            print_info(f"\n请安装缺失的包: pip install {' '.join(missing_pkgs)}")


if __name__ == "__main__":
    main()