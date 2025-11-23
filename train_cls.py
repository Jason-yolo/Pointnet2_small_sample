from data_utils.ModelNetDataLoader import ModelNetDataLoader
import argparse
import numpy as np
import os
import torch
import datetime 
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import provider
import importlib
import shutil

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR
sys.path.append(str(ROOT_DIR / 'models'))
"""
需要配置的参数：
--model pointnet2_cls_msg 
--normal 
--log_dir pointnet2_cls_msg
"""

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size in training [default: 16]')
    parser.add_argument('--model', default='pointnet2_cls_msg', help='model name [default: pointnet_cls]')
    parser.add_argument('--epoch',  default=2, type=int, help='number of epoch in training [default: 200]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--num_point', type=int, default=512, help='Point Number [default: 1024]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
    parser.add_argument('--log_dir', type=str, default="pointnet_test1", help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    parser.add_argument('--normal', dest='normal', action='store_true', default=True, help='Whether to use normal information [default: True]')
    parser.add_argument('--no_normal', dest='normal', action='store_false', help='Disable normal information')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for data loading [default: 8]')
    parser.add_argument('--shape_names_file', type=str, default='data/modelnet5_normal_resampled/modelnet5_shape_names.txt', help='shape names file [default: modelnet5_shape_names.txt]')
    parser.add_argument('--train_file', type=str, default='data/modelnet5_normal_resampled/modelnet5_train.txt', help='training file list [default: modelnet5_train.txt]')
    parser.add_argument('--test_file', type=str, default='data/modelnet5_normal_resampled/modelnet5_test.txt', help='test file list [default: modelnet5_test.txt]')
    parser.add_argument('--data_path', type=str, default='data/modelnet5_normal_resampled', help='data directory path [default: data/modelnet5_normal_resampled]')
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    """Return absolute Path, resolving relative path against BASE_DIR."""
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return path

def test(model, loader, num_class=5):                     # num_class=5
    mean_correct = []
    class_acc = np.zeros((num_class,3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        pred, _ = classifier(points)
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


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # 启用 CuDNN Benchmark（加速卷积操作）
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    '''CREATE DIR'''
    timestr = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    log_root = resolve_path('log/classification')
    log_root.mkdir(parents=True, exist_ok=True)
    run_dir_name = args.log_dir or timestr
    experiment_dir = log_root / run_dir_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = experiment_dir / 'checkpoints'
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    log_dir = experiment_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir / f'{args.model}.txt'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    DATA_PATH = resolve_path(args.data_path)

    TRAIN_DATASET = ModelNetDataLoader(root=str(DATA_PATH), npoint=args.num_point, split='train',
                                                     normal_channel=args.normal,
                                                     shape_names_file=args.shape_names_file,
                                                     train_file=args.train_file,
                                                     test_file=args.test_file)
    TEST_DATASET = ModelNetDataLoader(root=str(DATA_PATH), npoint=args.num_point, split='test',
                                                    normal_channel=args.normal,
                                                    shape_names_file=args.shape_names_file,
                                                    train_file=args.train_file,
                                                    test_file=args.test_file)
    trainDataLoader = torch.utils.data.DataLoader(
        TRAIN_DATASET, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True,        # 加速数据传输到GPU
        drop_last=True          # 丢弃最后不完整的batch，保持batch size一致
    )
    testDataLoader = torch.utils.data.DataLoader(
        TEST_DATASET, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True         # 加速数据传输到GPU
    )

    '''MODEL LOADING'''
    # 从数据加载器中动态获取类别数量
    num_class = len(TRAIN_DATASET.classes)
    log_string('Number of classes: %d' % num_class)
    MODEL = importlib.import_module(args.model)
    model_src = ROOT_DIR / 'models' / f'{args.model}.py'
    pointnet_util_src = ROOT_DIR / 'models' / 'pointnet_util.py'
    try:
        shutil.copy(model_src, experiment_dir)
        shutil.copy(pointnet_util_src, experiment_dir)
    except FileNotFoundError as err:
        log_string(f'Warning: failed to copy model files: {err}')

    classifier = MODEL.get_model(num_class,normal_channel=args.normal).cuda()
    criterion = MODEL.get_loss().cuda()

    checkpoint_path = checkpoints_dir / 'best_model.pth'
    try:
        checkpoint = torch.load(str(checkpoint_path))
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0


    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    mean_correct = []

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch,args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        scheduler.step()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, target = data
            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
            points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
            points = torch.Tensor(points)
            target = target[:, 0]

            points = points.transpose(2, 1)
            points, target = points.cuda(non_blocking=True), target.cuda(non_blocking=True)  # 非阻塞传输，加速
            optimizer.zero_grad()

            classifier = classifier.train()
            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long(), trans_feat)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1

        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)


        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f'% (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = checkpoints_dir / 'best_model.pth'
                log_string('Saving at %s'% savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, str(savepath))
            global_epoch += 1

    logger.info('End of training...')

if __name__ == '__main__':
    args = parse_args()
    main(args)
