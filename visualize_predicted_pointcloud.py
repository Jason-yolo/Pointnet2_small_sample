#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
点云分类结果可视化工具
将test_cls.py预测的点云文件进行可视化显示，根据预测类别着色
"""

import os
import sys
import numpy as np
import argparse
import torch
import torch.nn.parallel
import torch.utils.data
import importlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 添加项目根目录到Python路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'visualizer'))

# 导入分类相关模块和函数
from test_cls import classify_single_file

# ModelNet40类别名称
class_names = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
               'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
               'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
               'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
               'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

# 为每个类别分配不同的颜色
colors = np.array([
    [1, 0, 0],      # 红色 - airplane
    [0, 1, 0],      # 绿色 - bathtub
    [0, 0, 1],      # 蓝色 - bed
    [1, 1, 0],      # 黄色 - bench
    [1, 0, 1],      # 紫色 - bookshelf
    [0, 1, 1],      # 青色 - bottle
    [1, 0.5, 0],    # 橙色 - bowl
    [0.5, 1, 0],    # 黄绿色 - car
    [0, 0.5, 1],    # 深蓝色 - chair
    [0.5, 0, 1],    # 蓝紫色 - cone
    [1, 0, 0.5],    # 粉红色 - cup
    [0, 1, 0.5],    # 青绿色 - curtain
    [0.5, 0.5, 0],  # 橄榄色 - desk
    [0.5, 0, 0.5],  # 紫色 - door
    [0, 0.5, 0.5],  # 蓝绿色 - dresser
    [1, 0.5, 0.5],  # 浅红色 - flower_pot
    [0.5, 1, 0.5],  # 浅绿色 - glass_box
    [0.5, 0.5, 1],  # 浅蓝色 - guitar
    [1, 1, 0.5],    # 浅黄色 - keyboard
    [1, 0.5, 1],    # 浅紫色 - lamp
    [0.5, 1, 1],    # 浅青色 - laptop
    [0.8, 0.2, 0.2],# 暗红色 - mantel
    [0.2, 0.8, 0.2],# 暗绿色 - monitor
    [0.2, 0.2, 0.8],# 暗蓝色 - night_stand
    [0.8, 0.8, 0.2],# 暗黄色 - person
    [0.8, 0.2, 0.8],# 暗紫色 - piano
    [0.2, 0.8, 0.8],# 暗青色 - plant
    [0.8, 0.5, 0.2],# 暗橙色 - radio
    [0.2, 0.8, 0.5],# 暗青绿色 - range_hood
    [0.5, 0.2, 0.8],# 暗蓝紫色 - sink
    [0.8, 0.2, 0.5],# 暗粉红色 - sofa
    [0.2, 0.5, 0.8],# 暗蓝色 - stairs
    [0.5, 0.8, 0.2],# 暗黄绿色 - stool
    [0.2, 0.5, 0.5],# 暗蓝绿色 - table
    [0.5, 0.2, 0.5],# 暗紫色 - tent
    [0.5, 0.8, 0.5],# 深绿色 - toilet
    [0.8, 0.5, 0.8],# 深紫色 - tv_stand
    [0.5, 0.5, 0.5],# 灰色 - vase
    [0.8, 0.8, 0.8],# 浅灰色 - wardrobe
    [0.2, 0.2, 0.2] # 黑色 - xbox
])


def visualize_pointcloud_with_prediction(pc_path, model, num_points=2048, normal_channel=False):
    """
    使用matplotlib可视化点云文件并根据分类结果着色
    
    参数:
        pc_path: 点云文件路径
        model: 加载好的分类模型
        num_points: 采样点数量
        normal_channel: 是否使用法向量通道
    """
    print(f"处理文件: {pc_path}")
    
    # 创建一个适配函数，确保数据在CPU上运行
    def classify_single_file_cpu(model, file_path, num_point=2048, normal_channel=False):
        """适配test_cls.py的分类函数，确保在CPU上运行"""
        import torch
        
        # 读取点云数据
        point_set = np.loadtxt(file_path, delimiter=',').astype(np.float32)
        
        # 如果点云包含法向量，只取xyz坐标
        if normal_channel and point_set.shape[1] > 3:
            point_set = point_set[:, :6]  # 取xyz和法向量
        else:
            point_set = point_set[:, :3]  # 只取xyz
        
        # 采样到指定点数
        choice = np.random.choice(point_set.shape[0], num_point, replace=True)
        point_set = point_set[choice, :]
        
        # 数据增强
        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist
        
        # 转换为tensor
        point_set = torch.from_numpy(point_set.astype(np.float32))
        point_set = point_set.unsqueeze(0)  # 添加批次维度
        point_set = point_set.transpose(2, 1)  # 转换为B×C×N格式
        
        # 模型推理
        with torch.no_grad():
            # 确保模型在CPU上
            model = model.to('cpu')
            pred, _ = model(point_set)
            pred_choice = pred.data.max(1)[1]
            pred_prob = torch.nn.functional.softmax(pred, dim=1)
        
        # 获取预测类别和概率
        pred_class = pred_choice.item()
        pred_class_name = class_names[pred_class]
        max_prob = pred_prob[0, pred_class].item()
        all_probs = pred_prob[0].cpu().numpy()
        
        return pred_class, pred_class_name, max_prob, all_probs
    
    # 使用适配的CPU版本函数获取预测结果
    pred_class, pred_class_name, pred_prob, all_probs = classify_single_file_cpu(
        model, pc_path, num_point=num_points, normal_channel=normal_channel
    )
    
    # 获取Top 5预测结果
    top5_indices = np.argsort(all_probs)[-5:][::-1]
    top5_preds = [(idx, all_probs[idx]) for idx in top5_indices]
    
    # 读取点云数据
    point_cloud = np.loadtxt(pc_path, delimiter=',').astype(np.float32)
    
    # 如果点云包含法向量，只取xyz坐标
    if point_cloud.shape[1] > 3:
        point_cloud = point_cloud[:, :3]
    
    # 采样到指定点数
    choice = np.random.choice(point_cloud.shape[0], num_points, replace=True)
    point_cloud = point_cloud[choice, :]
    
    # 为点云生成颜色 (根据预测类别)
    color = colors[pred_class]  # matplotlib使用0-1范围的颜色
    
    # 打印预测结果
    print(f"预测类别: {class_names[pred_class]} (置信度: {pred_prob:.4f})")
    print("Top 5 预测:")
    for i, (cls_idx, prob) in enumerate(top5_preds):
        print(f"  {i+1}. {class_names[cls_idx]}: {prob:.4f}")
    
    # 使用matplotlib进行可视化
    print("\nmatplotlib可视化窗口已打开:")
    print("  - 拖动鼠标: 旋转视角")
    print("  - 鼠标滚轮: 缩放")
    print("  - 's'键: 保存当前视图")
    print("  - 'q'键: 关闭窗口")
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制点云
    scatter = ax.scatter(
        point_cloud[:, 0], 
        point_cloud[:, 1], 
        point_cloud[:, 2], 
        c=[color],  # 使用预测类别颜色
        s=10,  # 点的大小
        alpha=0.8  # 透明度
    )
    
    # 设置坐标轴和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'预测类别: {class_names[pred_class]} (置信度: {pred_prob:.4f})')
    
    # 设置坐标轴范围相等
    max_range = np.max([
        point_cloud[:, 0].max() - point_cloud[:, 0].min(),
        point_cloud[:, 1].max() - point_cloud[:, 1].min(),
        point_cloud[:, 2].max() - point_cloud[:, 2].min()
    ]) * 0.5
    
    mid_x = (point_cloud[:, 0].max() + point_cloud[:, 0].min()) * 0.5
    mid_y = (point_cloud[:, 1].max() + point_cloud[:, 1].min()) * 0.5
    mid_z = (point_cloud[:, 2].max() + point_cloud[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 显示图例
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                              label=f'{class_names[pred_class]}')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()


def visualize_with_matplotlib(pc_path, model, num_points=2048, normal_channel=False):
    """
    使用matplotlib可视化点云文件并根据分类结果着色（备选方案）
    
    参数:
        pc_path: 点云文件路径
        model: 加载好的分类模型
        num_points: 采样点数量
        normal_channel: 是否使用法向量通道
    """
    # 直接调用主可视化函数
    visualize_pointcloud_with_prediction(pc_path, model, num_points, normal_channel)


def load_model(model_name, model_path, normal_channel=False, device='cpu'):
    """
    加载预训练模型
    
    参数:
        model_name: 模型名称
        model_path: 模型路径
        normal_channel: 是否使用法向量通道
        device: 运行设备 (默认使用CPU)
    
    返回:
        加载好的模型
    """
    # 动态导入模型模块
    MODEL = importlib.import_module(model_name)
    
    # 根据数据集类型确定类别数量 - 使用ModelNet5数据集
    num_class = 5
    
    print(f"加载模型: {model_name}")
    print(f"使用设备: {device}")
    print(f"使用法向量通道: {normal_channel}")
    print(f"类别数量: {num_class}")
    
    # 创建模型实例
    model = MODEL.get_model(num_class, normal_channel=normal_channel)
    model.to(device)
    
    # 加载预训练权重
    print(f"加载预训练权重: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # 处理不同格式的checkpoint
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        print("模型加载成功")
        return model
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("尝试加载时忽略不匹配的层...")
        
        # 尝试加载时忽略不匹配的层
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # 获取当前模型的state_dict
        model_state_dict = model.state_dict()
        
        # 只保留匹配的参数
        matched_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}
        
        # 更新模型的state_dict
        model_state_dict.update(matched_state_dict)
        model.load_state_dict(model_state_dict)
        model.eval()
        
        # 打印未加载的参数
        unmatched = set(state_dict.keys()) - set(matched_state_dict.keys())
        if unmatched:
            print(f"未加载的参数数量: {len(unmatched)}")
            print("部分参数可能不匹配，但模型已尝试加载")
        
        return model

def main():
    """
    主函数，处理命令行参数并执行可视化
    """
    parser = argparse.ArgumentParser(description='点云分类结果可视化工具')
    parser.add_argument('--file', type=str, required=True, help='要可视化的点云文件路径')
    parser.add_argument('--model', type=str, default='pointnet2_cls_ssg', help='使用的模型名称')
    parser.add_argument('--model_path', type=str, default='log/classification/pointnet2_cls_ssg/checkpoints/best_model.pth', help='预训练模型路径')
    parser.add_argument('--num_points', type=int, default=2048, help='采样点数量')
    parser.add_argument('--normal', action='store_true', help='是否使用法向量通道')
    # 移除不使用的参数
    # parser.add_argument('--ball_radius', type=int, default=10, help='show3d_balls中点的半径')
    parser.add_argument('--use_matplotlib', action='store_true', default=True, help='使用matplotlib进行可视化')
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.file):
        print(f"错误: 文件 {args.file} 不存在")
        sys.exit(1)
    
    # 加载模型
    print(f"加载模型: {args.model}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model, args.model_path, normal_channel=args.normal, device=device)
    
    # 执行可视化
    try:
        visualize_pointcloud_with_prediction(args.file, model, args.num_points, args.normal)
    except Exception as e:
        print(f"可视化过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
