from data_utils.ModelNetDataLoader import ModelNetDataLoader, pc_normalize, farthest_point_sample
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

"""
配置参数：
--normal 
--log_dir pointnet2_cls_msg
"""
# python test_cls.py --log_dir test_200_5cls --normal --single_file "data/modelnet5_normal_resampled/bathtub/bathtub_0002.txt" --num_votes 3
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--log_dir', type=str, default='my_test', help='Experiment root')              # default='pointnet2_ssg_normal'
    parser.add_argument('--normal', dest='normal', action='store_true', default=True, help='Whether to use normal information [default: True]')
    parser.add_argument('--no_normal', dest='normal', action='store_false', help='Disable normal information')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting [default: 3]')
    parser.add_argument('--single_file', type=str, default=None, help='Path to a single point cloud file for classification')
    parser.add_argument('--uniform', action='store_true', default=False, help='Whether to use farthest point sampling for single file [default: False]')
    parser.add_argument('--shape_names_file', type=str, default='modelnet5_shape_names.txt', help='shape names file [default: modelnet5_shape_names.txt]')
    parser.add_argument('--train_file', type=str, default='modelnet5_train.txt', help='training file list [default: modelnet5_train.txt]')
    parser.add_argument('--test_file', type=str, default='modelnet5_test.txt', help='test file list [default: modelnet5_test.txt]')
    parser.add_argument('--data_path', type=str, default='data/modelnet5_normal_resampled/', help='data directory path [default: data/modelnet5_normal_resampled/]')
    return parser.parse_args()

def test(model, loader, num_class=5, vote_num=1):
    mean_correct = []
    class_acc = np.zeros((num_class,3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        vote_pool = torch.zeros(target.size()[0],num_class).cuda()
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
    point_set = point_set.cuda()
    
    # 模型推理（投票）
    model.eval()
    num_classes = len(class_names) if class_names else 40
    vote_pool = torch.zeros(1, num_classes).cuda()
    
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
    
    return pred_choice, pred_class_name, confidence, all_probs


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

    classifier = MODEL.get_model(num_class,normal_channel=args.normal).cuda()

    checkpoint_path = os.path.join(experiment_dir, 'checkpoints', 'best_model.pth')
    checkpoint = torch.load(checkpoint_path, weights_only=False)
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
            pred_class, pred_class_name, confidence, all_probs = classify_single_file(
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
