from data_utils.ModelNetDataLoader import ModelNetDataLoader, pc_normalize, farthest_point_sample
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib
import matplotlib
# 自动检测并设置matplotlib后端
import sys
import os

# 检查是否在headless环境中
is_headless = False
try:
    # 检查是否存在DISPLAY环境变量(Linux/Unix系统)
    if sys.platform.startswith('linux') and 'DISPLAY' not in os.environ:
        is_headless = True
    # 检查是否设置了PYTEST_CURRENT_TEST(测试环境)
    elif 'PYTEST_CURRENT_TEST' in os.environ:
        is_headless = True
    # 检查是否在CI环境中
    elif any(key in os.environ for key in ['CI', 'CONTINUOUS_INTEGRATION', 'JENKINS_URL']):
        is_headless = True
except:
    is_headless = False

# 尝试设置合适的后端
if is_headless:
    # 在headless环境中，直接使用非交互式后端
    matplotlib.use('Agg')
    print("注意: 当前环境为无图形界面(headless)模式，将使用Agg后端")
    print("可视化结果将自动保存为图像文件")
elif sys.platform.startswith('linux'):
    # 在Linux上，先尝试使用TkAgg后端
    try:
        matplotlib.use('TkAgg')
    except ImportError:
        # 如果TkAgg不可用，尝试Qt5Agg
        try:
            matplotlib.use('Qt5Agg')
        except ImportError:
            # 如果都不可用，使用Agg后端(非交互式，但至少可以保存图像)
            matplotlib.use('Agg')
            print("警告: 未找到可用的交互式后端，将使用Agg后端(仅能保存图像，无法显示窗口)")
            print("如需在Linux上显示交互式窗口，请安装python3-tk或pyqt5")
else:
    # 在Windows和其他系统上使用默认后端
    pass

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 自动检测CUDA是否可用
def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

"""
配置参数：
--normal 
--log_dir pointnet2_cls_msg
"""
# python test_cls.py --single_file "数据文件路径" --log_dir test_200_5cls --normal 
# python test_cls.py --log_dir test_200_5cls --normal --single_file "data/modelnet5_normal_resampled/bathtub/bathtub_0002.txt" --num_votes 3
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--log_dir', type=str, default='test_200_5cls', help='Experiment root')              # default='pointnet2_ssg_normal'
    parser.add_argument('--normal', dest='normal', action='store_true', default=True, help='Whether to use normal information [default: True]')
    parser.add_argument('--no_normal', dest='normal', action='store_false', help='Disable normal information')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting [default: 3]')
    parser.add_argument('--single_file', type=str, default='D:/研究生_study/cloudpoint_learn/Pointnet2_small_sample/data/test/airplane_0501.txt', help='Path to a single point cloud file for classification')
    parser.add_argument('--uniform', action='store_true', default=False, help='Whether to use farthest point sampling for single file [default: False]')
    parser.add_argument('--shape_names_file', type=str, default='modelnet5_shape_names.txt', help='shape names file [default: modelnet5_shape_names.txt]')
    parser.add_argument('--train_file', type=str, default='modelnet5_train.txt', help='training file list [default: modelnet5_train.txt]')
    parser.add_argument('--test_file', type=str, default='modelnet5_test.txt', help='test file list [default: modelnet5_test.txt]')
    parser.add_argument('--data_path', type=str, default='data/modelnet5_normal_resampled/', help='data directory path [default: data/modelnet5_normal_resampled/]')
    # 添加可视化相关参数
    parser.add_argument('--visualize', action='store_true', default=True, help='Whether to visualize the point cloud after classification [default: True]') # 是否可视化分类结果
    parser.add_argument('--save_results', action='store_true', default=False, help='Whether to save classification results to file [default: False]')              # 是否保存分类结果到文件
    parser.add_argument('--visualization_dir', type=str, default='visualization_results', help='Directory to save visualization results [default: visualization_results]')  # 可视化结果保存目录
    return parser.parse_args()

def test(model, loader, num_class=5, vote_num=1):
    mean_correct = []
    class_acc = np.zeros((num_class,3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        # 自动选择设备
        device = get_device()
        points, target = points.to(device), target.to(device)
        classifier = model.eval()
        # 自动选择设备
        device = get_device()
        vote_pool = torch.zeros(target.size()[0], num_class).to(device)
        for _ in range(vote_num):
            pred, _ = classifier(points)
            vote_pool += pred
        pred = vote_pool/vote_num
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
    class_acc = np.mean(class_acc[:,2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


def classify_single_file(model, file_path, num_point=1024, normal_channel=True, vote_num=1, uniform=False, class_names=None):
    """
    对单张点云文件进行分类
    
    Args:
        model: 训练好的模型
        file_path: 点云文件路径 (.txt文件，每行格式：x,y,z,nx,ny,nz 或 x,y,z)
        num_point: 采样点数
        normal_channel: 是否使用法向量信息
        vote_num: 投票次数
        uniform: 是否使用最远点采样
        class_names: 类别名称列表
    
    Returns:
        pred_class: 预测的类别索引
        pred_class_name: 预测的类别名称
        confidence: 预测置信度
        all_predictions: 所有类别的概率
    """
    # 加载点云文件
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Point cloud file not found: {file_path}")
    
    point_set = np.loadtxt(file_path, delimiter=',').astype(np.float32)
    
    # 处理点云数据
    if uniform:
        point_set = farthest_point_sample(point_set, num_point)
    else:
        # 如果点数大于num_point，取前num_point个点
        if point_set.shape[0] > num_point:
            point_set = point_set[0:num_point, :]
        # 如果点数小于num_point，重复采样
        elif point_set.shape[0] < num_point:
            indices = np.random.choice(point_set.shape[0], num_point, replace=True)
            point_set = point_set[indices, :]
    
    # 标准化坐标
    point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
    
    # 如果不使用法向量，只保留xyz坐标
    if not normal_channel:
        point_set = point_set[:, 0:3]
    
    # 转换为torch tensor并添加batch维度
    point_set = torch.from_numpy(point_set).unsqueeze(0).float()
    point_set = point_set.transpose(2, 1)  # [B, 3/6, N]
    # 自动选择设备
    device = get_device()
    point_set = point_set.to(device)
    
    # 模型推理（投票）
    model.eval()
    num_classes = len(class_names) if class_names else 40
    # 自动选择设备
    device = get_device()
    vote_pool = torch.zeros(1, num_classes).to(device)
    
    with torch.no_grad():
        for _ in range(vote_num):
            pred, _ = model(point_set)
            vote_pool += pred
    
    # 计算平均预测
    pred = vote_pool / vote_num
    
    # 获取预测结果
    pred_choice = pred.data.max(1)[1].item()
    confidence = torch.softmax(pred, dim=1).data.max(1)[0].item()
    
    # 获取所有类别的概率
    all_probs = torch.softmax(pred, dim=1).cpu().numpy()[0]
    
    # 获取类别名称
    pred_class_name = class_names[pred_choice] if class_names else f"Class_{pred_choice}"
    
    # 返回结果，包括点云数据用于可视化
    return pred_choice, pred_class_name, confidence, all_probs, point_set


def save_classification_results(file_path, pred_class, pred_class_name, confidence, all_probs, class_names, save_dir='results'):
    """
    保存分类结果到文件
    
    Args:
        file_path: 点云文件路径
        pred_class: 预测的类别索引
        pred_class_name: 预测的类别名称
        confidence: 预测置信度
        all_probs: 所有类别的概率
        class_names: 类别名称列表
        save_dir: 结果保存目录
    """
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 生成结果文件名
    file_name = os.path.basename(file_path).replace('.txt', '_results.txt')
    result_path = os.path.join(save_dir, file_name)
    
    # 保存结果
    with open(result_path, 'w') as f:
        f.write(f"Point Cloud File: {file_path}\n")
        f.write(f"Predicted Class: {pred_class_name} (ID: {pred_class})\n")
        f.write(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)\n")
        f.write("\nClass Probabilities:\n")
        
        # 保存所有类别的概率
        for i, prob in enumerate(all_probs):
            class_name = class_names[i] if i < len(class_names) else f"Class_{i}"
            f.write(f"{class_name}: {prob:.4f} ({prob*100:.2f}%)\n")
    
    return result_path

def visualize_pointcloud_with_prediction(point_cloud, file_path, pred_class, pred_class_name, confidence, class_names, save_path=None, visualization_dir='visualization_results'):
    """
    使用matplotlib可视化点云，并显示分类结果
    
    Args:
        point_cloud: 点云数据 (tensor 格式)
        file_path: 点云文件路径
        pred_class: 预测的类别索引
        pred_class_name: 预测的类别名称
        confidence: 预测置信度
        class_names: 类别名称列表
        save_path: 是否保存可视化结果，如果为None则不保存
        visualization_dir: 可视化结果保存目录
    """
    # 为每个类别分配不同的颜色
    colors = np.array([
        [1, 0, 0],      # 红色
        [0, 1, 0],      # 绿色
        [0, 0, 1],      # 蓝色
        [1, 1, 0],      # 黄色
        [1, 0, 1],      # 紫色
        [0, 1, 1],      # 青色
        [1, 0.5, 0],    # 橙色
        [0.5, 1, 0],    # 黄绿色
        [0, 0.5, 1],    # 深蓝色
        [0.5, 0, 1],    # 蓝紫色
        [1, 0, 0.5],    # 粉红色
        [0, 1, 0.5],    # 青绿色
        [0.5, 0.5, 0],  # 橄榄色
        [0.5, 0, 0.5],  # 紫色
        [0, 0.5, 0.5]   # 蓝绿色
    ])
    
    # 将tensor转换为numpy数组
    if isinstance(point_cloud, torch.Tensor):
        point_cloud = point_cloud.cpu().numpy()
    
    # 调整点云形状 (B×C×N -> N×C)
    if len(point_cloud.shape) == 3:
        point_cloud = point_cloud[0].transpose(1, 0)  # 转为 N×C
    
    # 只取xyz坐标
    if point_cloud.shape[1] > 3:
        point_cloud = point_cloud[:, :3]
    
    # 确保颜色索引不会越界
    color_idx = pred_class % len(colors)
    color = colors[color_idx]
    
    # 创建可视化图形
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
    ax.set_title(f'File: {os.path.basename(file_path)}\nPredicted: {pred_class_name} (Confidence: {confidence:.4f})')
    
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
                              label=f'{pred_class_name}')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    # 检查是否在headless环境中 - 最高优先级检查
    try:
        # 检查是否设置了is_headless变量
        if 'is_headless' in globals() and is_headless:
            # 在headless环境中，确保创建可视化目录并保存图像
            if not os.path.exists(visualization_dir):
                os.makedirs(visualization_dir, exist_ok=True)
            
            # 如果没有指定保存路径，使用默认路径
            if not save_path:
                save_path = os.path.join(
                    visualization_dir,
                    os.path.basename(file_path).replace('.txt', '_visualization.png')
                )
            
            # 保存图像
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"当前环境为无图形界面(headless)模式，已自动保存图像: {save_path}")
            plt.close()
            return
    except:
        # 如果检查失败，继续执行
        pass
    
    # 检查当前后端是否支持交互式显示
    is_interactive = matplotlib.get_backend() not in ['Agg', 'Cairo', 'pdf', 'svg', 'ps']
    
    # 无论环境如何，首先确保创建可视化目录
    if not os.path.exists(visualization_dir):
        os.makedirs(visualization_dir, exist_ok=True)
    
    # 如果需要保存或在非交互式环境中，确保图像被保存
    if save_path or not is_interactive:
        if not save_path:
            save_path = os.path.join(
                visualization_dir,
                os.path.basename(file_path).replace('.txt', '_visualization.png')
            )
        
        # 确保保存目录存在
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
        # 保存图像
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存到: {save_path}")
    
    # 只有在交互式后端且不需要保存的情况下才显示窗口
    if is_interactive:
        print("\nmatplotlib可视化窗口已打开:")
        print("  - 拖动鼠标: 旋转视角")
        print("  - 鼠标滚轮: 缩放")
        print("  - 's'键: 保存当前视图")
        print("  - 'q'键: 关闭窗口")
        try:
            plt.show()
        except Exception as e:
            # 如果显示失败，至少确保图像已保存
            if not save_path:
                save_path = os.path.join(
                    visualization_dir,
                    os.path.basename(file_path).replace('.txt', '_visualization.png')
                )
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"显示窗口失败: {str(e)}")
                print(f"已保存图像到: {save_path}")
            plt.close()
    else:
        # 非交互式后端，关闭窗口
        plt.close()
        if not save_path:  # 这种情况不应该发生，因为上面已经处理了保存逻辑
            print(f"注意: 当前使用非交互式后端({matplotlib.get_backend()})，无法显示窗口")

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    experiment_dir = os.path.join(BASE_DIR, 'log', 'classification', args.log_dir)   

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(experiment_dir, 'eval.txt'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    # 如果data_path是相对路径，基于BASE_DIR解析
    if not os.path.isabs(args.data_path):
        DATA_PATH = os.path.join(BASE_DIR, args.data_path)
    else:
        DATA_PATH = args.data_path
    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='test', normal_channel=args.normal,
                                      shape_names_file=args.shape_names_file,
                                      train_file=args.train_file,
                                      test_file=args.test_file)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    # 从数据加载器中动态获取类别数量
    num_class = len(TEST_DATASET.classes)
    log_string('Number of classes: %d' % num_class)
    logs_dir = os.path.join(experiment_dir, 'logs')
    model_name = os.listdir(logs_dir)[0].split('.')[0]
    MODEL = importlib.import_module(model_name)

    # 根据环境自动选择设备
    device = get_device()
    classifier = MODEL.get_model(num_class, normal_channel=args.normal).to(device)

    checkpoint_path = os.path.join(experiment_dir, 'checkpoints', 'best_model.pth')
    # 自动根据环境加载模型
    device = get_device()
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    classifier.load_state_dict(checkpoint['model_state_dict'])

    # 加载类别名称
    catfile = os.path.join(DATA_PATH, args.shape_names_file)
    class_names = [line.rstrip() for line in open(catfile)]

    # 如果指定了单张文件，进行分类
    if args.single_file:
        log_string('=' * 50)
        log_string('Classifying single point cloud file: %s' % args.single_file)
        log_string('=' * 50)
        
        try:
            # 加载原始点云数据用于可视化（保留原始点云）
            raw_point_cloud = np.loadtxt(args.single_file, delimiter=',').astype(np.float32)
            
            # 调用分类函数
            pred_class, pred_class_name, confidence, all_probs, point_set = classify_single_file(
                classifier, 
                args.single_file, 
                num_point=args.num_point,
                normal_channel=args.normal,
                vote_num=args.num_votes,
                uniform=args.uniform,
                class_names=class_names
            )
            
            log_string('=' * 50)
            log_string('Classification Result:')
            log_string('  Predicted Class: %s (ID: %d)' % (pred_class_name, pred_class))
            log_string('  Confidence: %.4f (%.2f%%)' % (confidence, confidence * 100))
            log_string('=' * 50)
            log_string('Top 5 Predictions:')
            
            # 显示Top 5预测结果
            top5_indices = np.argsort(all_probs)[-5:][::-1]
            for i, idx in enumerate(top5_indices):
                log_string('  %d. %s: %.4f (%.2f%%)' % (i+1, class_names[idx], all_probs[idx], all_probs[idx]*100))
            log_string('=' * 50)
            
            # 保存分类结果（如果启用）
            if args.save_results:
                result_path = save_classification_results(
                    args.single_file, 
                    pred_class, 
                    pred_class_name, 
                    confidence, 
                    all_probs, 
                    class_names,
                    save_dir=args.visualization_dir
                )
                log_string(f'分类结果已保存到: {result_path}')
            
            # 可视化点云（如果启用）
            if args.visualize:
                log_string('开始可视化点云...')
                
                # 确定保存路径（如果需要保存可视化结果）
                save_path = None
                if args.save_results:
                    save_path = os.path.join(
                        args.visualization_dir, 
                        os.path.basename(args.single_file).replace('.txt', '_visualization.png')
                    )
                
                # 调用可视化函数
                visualize_pointcloud_with_prediction(
                    point_set,  # 使用处理后的点云进行可视化
                    args.single_file, 
                    pred_class, 
                    pred_class_name, 
                    confidence, 
                    class_names,
                    save_path=save_path,
                    visualization_dir=args.visualization_dir
                )
                log_string('可视化完成')
            
        except Exception as e:
            log_string('Error during classification: %s' % str(e))
            import traceback
            log_string(traceback.format_exc())
    else:
        # 批量测试模式
        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class, vote_num=args.num_votes)
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))



if __name__ == '__main__':
    args = parse_args()
    main(args)
