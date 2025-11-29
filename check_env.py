#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境依赖检查工具

该脚本用于检查Python环境中的依赖包是否已正确安装，支持从requirements.txt读取
或在代码中定义依赖列表，并提供清晰的检查结果输出和安装建议。
"""

import os
import sys
import importlib
import pkg_resources
import argparse
from typing import Dict, List, Tuple, Optional, Set


class ColorOutput:
    """用于提供彩色输出的工具类"""
    
    # ANSI颜色代码
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    
    @classmethod
    def setup_colors(cls):
        """根据操作系统设置颜色支持"""
        # Windows系统需要特殊处理颜色
        if os.name == 'nt':
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            except:
                # 如果无法启用颜色，将所有颜色代码设置为空字符串
                cls.GREEN = cls.RED = cls.YELLOW = cls.BLUE = cls.ENDC = ''
    
    @classmethod
    def print_success(cls, message: str):
        """打印成功消息（绿色）"""
        print(f"{cls.GREEN}[✓] {message}{cls.ENDC}")
    
    @classmethod
    def print_error(cls, message: str):
        """打印错误消息（红色）"""
        print(f"{cls.RED}[✗] {message}{cls.ENDC}")
    
    @classmethod
    def print_warning(cls, message: str):
        """打印警告消息（黄色）"""
        print(f"{cls.YELLOW}[!] {message}{cls.ENDC}")
    
    @classmethod
    def print_info(cls, message: str):
        """打印信息消息（蓝色）"""
        print(f"{cls.BLUE}[i] {message}{cls.ENDC}")


class DependencyChecker:
    """依赖检查器类"""
    
    def __init__(self):
        # 设置颜色输出
        ColorOutput.setup_colors()
        # 定义PointNet2项目的默认依赖（从之前查看的requirements.txt中提取）
        self.default_dependencies = [
            'torch>=1.6.0',
            'torchvision>=0.7.0', 
            'numpy>=1.19.0',
            'tqdm>=4.50.0',
            'opencv-python>=4.4.0',
            'plyfile>=0.7.4'
        ]
    
    def parse_requirements_file(self, file_path: str) -> List[str]:
        """从requirements.txt文件解析依赖列表"""
        dependencies = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # 跳过空行和注释
                    if line and not line.startswith('#'):
                        dependencies.append(line)
            return dependencies
        except FileNotFoundError:
            ColorOutput.print_error(f"找不到requirements.txt文件: {file_path}")
            return []
        except Exception as e:
            ColorOutput.print_error(f"读取requirements.txt文件时出错: {str(e)}")
            return []
    
    def parse_dependency(self, dep_str: str) -> Tuple[str, Optional[str]]:
        """解析单个依赖字符串，返回包名和版本要求"""
        # 处理带有版本要求的依赖
        if '>=' in dep_str:
            parts = dep_str.split('>=', 1)
            return parts[0].strip(), parts[1].strip()
        elif '==' in dep_str:
            parts = dep_str.split('==', 1)
            return parts[0].strip(), parts[1].strip()
        elif '<=' in dep_str:
            parts = dep_str.split('<=', 1)
            return parts[0].strip(), parts[1].strip()
        elif '>' in dep_str and not dep_str.startswith('http'):
            parts = dep_str.split('>', 1)
            return parts[0].strip(), f'>{parts[1].strip()}'
        elif '<' in dep_str and not dep_str.startswith('http'):
            parts = dep_str.split('<', 1)
            return parts[0].strip(), f'<{parts[1].strip()}'
        else:
            # 只有包名，没有版本要求
            return dep_str.strip(), None
    
    def check_dependency(self, package_name: str, version_requirement: Optional[str]) -> Tuple[bool, Optional[str]]:
        """检查单个依赖是否已安装并满足版本要求"""
        try:
            # 使用pkg_resources获取已安装的包信息
            installed_version = pkg_resources.get_distribution(package_name).version
            
            # 如果有版本要求，进行版本比较
            if version_requirement:
                try:
                    # 使用pkg_resources的版本比较功能
                    if '>=' in version_requirement:
                        required_version = version_requirement.replace('>=', '')
                        is_satisfied = pkg_resources.parse_version(installed_version) >= pkg_resources.parse_version(required_version)
                    elif '==' in version_requirement:
                        required_version = version_requirement.replace('==', '')
                        is_satisfied = pkg_resources.parse_version(installed_version) == pkg_resources.parse_version(required_version)
                    elif '<=' in version_requirement:
                        required_version = version_requirement.replace('<=', '')
                        is_satisfied = pkg_resources.parse_version(installed_version) <= pkg_resources.parse_version(required_version)
                    elif '>' in version_requirement and version_requirement.startswith('>'):
                        required_version = version_requirement.replace('>', '')
                        is_satisfied = pkg_resources.parse_version(installed_version) > pkg_resources.parse_version(required_version)
                    elif '<' in version_requirement and version_requirement.startswith('<'):
                        required_version = version_requirement.replace('<', '')
                        is_satisfied = pkg_resources.parse_version(installed_version) < pkg_resources.parse_version(required_version)
                    else:
                        is_satisfied = True
                    
                    return is_satisfied, installed_version
                except Exception:
                    # 版本解析错误，认为版本不满足要求
                    return False, installed_version
            else:
                # 没有版本要求，只要安装了就算满足
                return True, installed_version
        except pkg_resources.DistributionNotFound:
            # 包未安装
            return False, None
        except Exception as e:
            # 其他错误，记录并返回未安装
            ColorOutput.print_warning(f"检查包 {package_name} 时出错: {str(e)}")
            return False, None
    
    def get_install_command(self, package_name: str, version_requirement: Optional[str]) -> str:
        """生成安装命令"""
        if version_requirement:
            return f"pip install {package_name}{version_requirement}"
        else:
            return f"pip install {package_name}"
    
    def check_all_dependencies(self, dependencies: List[str]) -> Tuple[Dict, Dict, Dict]:
        """检查所有依赖并返回结果"""
        satisfied = {}
        version_mismatch = {}
        missing = {}
        
        for dep in dependencies:
            try:
                package_name, version_req = self.parse_dependency(dep)
                is_satisfied, installed_version = self.check_dependency(package_name, version_req)
                
                if is_satisfied:
                    satisfied[package_name] = {
                        'required': version_req,
                        'installed': installed_version
                    }
                elif installed_version:
                    # 已安装但版本不匹配
                    version_mismatch[package_name] = {
                        'required': version_req,
                        'installed': installed_version
                    }
                else:
                    # 未安装
                    missing[package_name] = {
                        'required': version_req
                    }
            except Exception as e:
                ColorOutput.print_error(f"处理依赖 {dep} 时出错: {str(e)}")
        
        return satisfied, version_mismatch, missing
    
    def print_results(self, satisfied: Dict, version_mismatch: Dict, missing: Dict):
        """打印检查结果"""
        print("\n" + "="*60)
        print("环境依赖检查结果")
        print("="*60)
        
        # 打印已满足的依赖
        if satisfied:
            ColorOutput.print_info(f"已成功安装的依赖 ({len(satisfied)}):")
            for pkg, info in sorted(satisfied.items()):
                if info['required']:
                    ColorOutput.print_success(f"  {pkg} ({info['installed']}) [满足要求: {pkg}{info['required']}]")
                else:
                    ColorOutput.print_success(f"  {pkg} ({info['installed']}) [无版本要求]")
        else:
            ColorOutput.print_warning("没有找到已满足要求的依赖")
        
        print()
        
        # 打印版本不匹配的依赖
        if version_mismatch:
            ColorOutput.print_warning(f"版本不匹配的依赖 ({len(version_mismatch)}):")
            for pkg, info in sorted(version_mismatch.items()):
                ColorOutput.print_warning(f"  {pkg}: 已安装 {info['installed']}, 需要 {info['required']}")
        else:
            ColorOutput.print_info("没有发现版本不匹配的依赖")
        
        print()
        
        # 打印缺失的依赖
        if missing:
            ColorOutput.print_error(f"未安装的依赖 ({len(missing)}):")
            for pkg, info in sorted(missing.items()):
                if info['required']:
                    ColorOutput.print_error(f"  {pkg}{info['required']}")
                else:
                    ColorOutput.print_error(f"  {pkg}")
        else:
            ColorOutput.print_success("所有依赖都已正确安装")
        
        print()
        
        # 打印安装建议
        if missing or version_mismatch:
            ColorOutput.print_info("安装建议:")
            
            # 缺失的依赖
            for pkg, info in sorted(missing.items()):
                cmd = self.get_install_command(pkg, info['required'])
                ColorOutput.print_info(f"  {cmd}")
            
            # 版本不匹配的依赖
            for pkg, info in sorted(version_mismatch.items()):
                cmd = self.get_install_command(pkg, info['required'])
                ColorOutput.print_info(f"  {cmd}")
            
            # 生成一键安装命令
            all_missing_deps = [f"{pkg}{info['required']}" if info['required'] else pkg for pkg, info in missing.items()]
            all_mismatch_deps = [f"{pkg}{info['required']}" for pkg, info in version_mismatch.items()]
            all_commands = all_missing_deps + all_mismatch_deps
            
            if all_commands:
                print()
                ColorOutput.print_info("一键安装命令:")
                # 使用引号包裹每个依赖，避免空格问题
                cmd = "pip install " + " ".join([f'"{dep}"' for dep in all_commands])
                ColorOutput.print_info(f"  {cmd}")
        
        print("="*60)
    
    def run(self, requirements_file: Optional[str] = None):
        """运行依赖检查"""
        try:
            # 确定使用的依赖列表
            if requirements_file:
                ColorOutput.print_info(f"从文件 {requirements_file} 读取依赖列表...")
                dependencies = self.parse_requirements_file(requirements_file)
                if not dependencies:
                    ColorOutput.print_warning("未找到有效依赖，使用默认依赖列表")
                    dependencies = self.default_dependencies
            else:
                ColorOutput.print_info("使用默认依赖列表")
                dependencies = self.default_dependencies
            
            # 显示要检查的依赖数量
            ColorOutput.print_info(f"开始检查 {len(dependencies)} 个依赖...")
            
            # 执行检查
            satisfied, version_mismatch, missing = self.check_all_dependencies(dependencies)
            
            # 显示结果
            self.print_results(satisfied, version_mismatch, missing)
            
            # 返回状态码
            if missing or version_mismatch:
                return 1  # 有缺失或不匹配的依赖
            else:
                return 0  # 所有依赖都满足
                
        except KeyboardInterrupt:
            print("\n")
            ColorOutput.print_warning("检查被用户中断")
            return 1
        except Exception as e:
            ColorOutput.print_error(f"检查过程中发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return 1


def main():
    """主函数"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Python环境依赖检查工具')
    parser.add_argument('-r', '--requirements', help='指定requirements.txt文件路径')
    parser.add_argument('--no-color', action='store_true', help='禁用彩色输出')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 如果禁用彩色输出，清空颜色代码
    if args.no_color:
        ColorOutput.GREEN = ColorOutput.RED = ColorOutput.YELLOW = ColorOutput.BLUE = ColorOutput.ENDC = ''
    
    # 创建检查器并运行
    checker = DependencyChecker()
    exit_code = checker.run(args.requirements)
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()