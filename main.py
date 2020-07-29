import argparse
import sys
import os
import shutil
import time
import numpy as np
from random import sample
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from deepKNet.data import deepKNetDataset, get_train_val_test_loader
from deepKNet.model2D import LeNet5, ResNet, BasicBlock
from deepKNet.model3D import PointNetCls

parser = argparse.ArgumentParser(description='deepKNet model')
parser.add_argument('--task', choices=['regression', 'classification'],
                    default='classification')
parser.add_argument('--algo', default='PointNetCls', type=str, metavar='NETWORK')
parser.add_argument('--dim', default=3, type=int, metavar='FEATURE DIMENSION')
parser.add_argument('--target', default='MIT', metavar='TARGET_PROPERTY')
parser.add_argument('--root', default='./data_gen/', metavar='DATA_DIR')
parser.add_argument('--run_name', default='run1', metavar='RUNID')
parser.add_argument('--gpu_id', default=0, type=int, metavar='GPUID')
# hyper parameter tuning
parser.add_argument('--padding', default='zero', type=str, metavar='POINT PADDING')
parser.add_argument('--cutoff', default=6000, type=int, metavar='NPOINT CUTOFF')
parser.add_argument('--data_aug', action='store_true')
parser.add_argument('--stn', action='store_true')
parser.add_argument('--disable_normalization', action='store_true')
parser.add_argument('--epochs', default=60, type=int, metavar='N')
parser.add_argument('--batch_size', default=64, type=int, metavar='N')
parser.add_argument('--optim', default='Adam', type=str, metavar='OPTIM')
parser.add_argument('--lr', default=0.001, type=float, metavar='LR')
parser.add_argument('--lr_milestones', default=[30, 50], nargs='+', type=int)
parser.add_argument('--dropout', default=0.3, type=float, metavar='DROPOUT')
# default params
parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
parser.add_argument('--wd', '--weight_decay', default=0, type=float,
                    metavar='W', help='weigh decay (default: 0)',
                    dest='weight_decay')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--train_ratio', default=0.7, type=float, metavar='n/N')
parser.add_argument('--val_ratio', default=0.15, type=float, metavar='n/N')
parser.add_argument('--test_ratio', default=0.15, type=float, metavar='n/N')
n_threads = torch.get_num_threads()
parser.add_argument('--num_threads', default=n_threads, type=int, metavar='N_thread')
parser.add_argument('--num_data_workers', default=4, type=int, metavar='N')
parser.add_argument('--print_freq', default=10, type=int, metavar='N')
parser.add_argument('--test_freq', default=10, type=int, metavar='N')
parser.add_argument('--resume', default='', type=str, metavar='PATH')
parser.add_argument('--disable_cuda', action='store_true')

# parse args
args = parser.parse_args()
args.cuda = torch.cuda.is_available() and not args.disable_cuda
cuda_device = torch.device('cuda:{}'.format(args.gpu_id)) if args.cuda else None
if args.num_threads != n_threads:
    torch.set_num_threads(args.num_threads)
# print out args
print('User defined variables:', flush=True)
for key, val in vars(args).items():
    print('  => {:17s}: {}'.format(key, val), flush=True)

if args.task == 'classification':
    best_performance = 0.
else:
    best_performance = 1E10

def main():
    global args, best_performance, cuda_device

    # load data
    data_root = os.path.join(args.root, 'data_pointnet') if args.dim == 3 \
                else os.path.join(args.root, 'data_multiview')
    dataset = deepKNetDataset(root=data_root, target=args.target, 
                              cutoff=args.cutoff, padding=args.padding,
                              data_aug=args.data_aug)
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset, batch_size=args.batch_size, 
        train_ratio=args.train_ratio, val_ratio=args.val_ratio, 
        test_ratio=args.test_ratio, pin_memory=args.cuda, 
        num_data_workers=args.num_data_workers)

    # obtain target value normalizer
    normalizer = Normalizer(torch.zeros(2))
    normalizer.load_state_dict({'mean': 0., 'std': 1.})
    if args.task == 'regression' and not args.disable_normalization:
        sample_target = [dataset[i][-1] for i in 
                         sample(range(len(dataset)), 1000)]
        normalizer = Normalizer(sample_target)

    # build model
    if args.algo == 'PointNetCls' and args.dim == 3:
        model = PointNetCls(k=4, dp=args.dropout,
                            classification=args.task=='classification',
                            stn=args.stn)
    elif args.algo == 'LeNet5' and args.dim == 2:
        model = LeNet5()
    elif args.algo == 'ResNet' and args.dim == 2:
        model = ResNet(BasicBlock, [3, 4, 6, 3])
    else:
        raise NameError('Specified algorithm not implemented yet..')
    # number of trainable model parameters
    trainable_params = sum(p.numel() for p in model.parameters() 
                           if p.requires_grad)
    print('Number of trainable model parameters: {:d}' \
           .format(trainable_params), flush=True)

    if args.cuda:
        print('running on GPU:{}..'.format(args.gpu_id), flush=True)
        model = model.cuda(device=cuda_device)
    else:
        print('running on CPU..', flush=True)

    # define loss function 
    if args.task == 'classification':
        criterion = nn.NLLLoss()
    else:
        criterion = nn.MSELoss()

    # optimization algo
    if args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, 
                               weight_decay=args.weight_decay)
    elif args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    else:
        raise NameError('Only Adam or SGD is allowed as --optim')

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume), flush=True)
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            best_auc = checkpoint['best_auc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']), flush=True)
        else:
            print("=> no checkpoint found at '{}', existing.." \
                   .format(args.resume), flush=True)
            sys.exit(1)

    # TensorBoard writer
    summary_root = './runs/'
    summary_file = summary_root + args.run_name
    if not os.path.exists(summary_root):
        os.mkdir(summary_root)
    if os.path.exists(summary_file):
        print('run file already exists, use a different --run-name')
        sys.exit(1)
    writer = SummaryWriter(summary_file)

    # learning-rate scheduler
    scheduler = MultiStepLR(optimizer=optimizer, milestones=args.lr_milestones,
                            gamma=0.1, last_epoch=-1)
    
    for epoch in range(args.start_epoch, args.start_epoch+args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, normalizer, writer)

        # evaluate on validation set
        performance = validate(val_loader, model, criterion, epoch, normalizer, writer)

        scheduler.step()

        # remember best auc and save checkpoint
        if args.task == 'classification':
            is_best = performance > best_performance
            best_performance = max(performance, best_performance)
        else:
            is_best = performance < best_performance
            best_performance = min(performance, best_performance)

        # save checkpoint
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_performance': best_performance,
            'optimizer': optimizer.state_dict(),
        }, is_best)

        if (epoch-args.start_epoch) % args.test_freq == 0 \
            and (epoch-args.start_epoch) != 0:
            # test best model
            print('---------Evaluate Model on Test Set---------------', flush=True)
            best_model = load_best_model()
            print('best validation performance: {}'.format(best_model['best_performance']))
            model.load_state_dict(best_model['state_dict'])
            validate(test_loader, model, criterion, epoch, normalizer, writer, test_mode=True)

    # test best model
    print('---------Evaluate Model on Test Set---------------', flush=True)
    best_model = load_best_model()
    print('best validation performance: {}'.format(best_model['best_performance']))
    model.load_state_dict(best_model['state_dict'])
    validate(test_loader, model, criterion, epoch, normalizer, writer, test_mode=True)


def train(train_loader, model, criterion, optimizer, epoch, normalizer, writer):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':4.2f')
    losses = AverageMeter('Loss', ':6.3f')
    if args.task == 'classification':
        accuracies = AverageMeter('Accu', ':6.3f')
        precisions = AverageMeter('Prec', ':6.3f')
        recalls = AverageMeter('Rec', ':6.3f')
        fscores = AverageMeter('Fsc', ':6.3f')
        auc_scores = AverageMeter('AUC', ':6.3f')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, accuracies, precisions, recalls, fscores, auc_scores],
            prefix="Epoch: [{}]".format(epoch)
        )
    else:
        maes = AverageMeter()
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, maes],
            prefix="Epoch: [{}]".format(epoch)
        )

    # switch to training mode
    model.train()

    end = time.time()
    running_loss = 0.0
    for idx, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        image, target = data
        # normalize target
        if args.task == 'classification':
            target_normed = target.view(-1).long()
        else:
            target_normed = normalizer.norm(target)

        if args.cuda:
            image = image.cuda(device=cuda_device)
            target_normed = target_normed.cuda(device=cuda_device)

        # compute output
        output = model(image)
        loss = criterion(output, target_normed)

        # measure accuracy and record loss
        if args.task == 'classification':
            accuracy, precision, recall, fscore, auc_score =\
                class_eval(output, target)
            losses.update(loss.item(), target.size(0))
            accuracies.update(accuracy.item(), target.size(0))
            precisions.update(precision.item(), target.size(0))
            recalls.update(recall.item(), target.size(0))
            fscores.update(fscore.item(), target.size(0))
            auc_scores.update(auc_score.item(), target.size(0))
        else:
            mae = compute_mae(normalizer.denorm(output), target)
            losses.update(loss.item(), target.size(0))
            maes.update(mae.item(), target.size(0))

        # compute gradient and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print progress and write to TensorBoard
        running_loss += loss.item()
        if idx % args.print_freq == 0 and idx != 0:
            progress.display(idx)
            writer.add_scalar('training loss',
                            running_loss / args.print_freq,
                            epoch * len(train_loader) + idx)
            running_loss = 0.0


def validate(val_loader, model, criterion, epoch, writer, test_mode=False):
    batch_time = AverageMeter('Time', ':4.2f')
    losses = AverageMeter('Loss', ':6.3f')
    if args.task == 'classification':
        accuracies = AverageMeter('Accu', ':6.3f')
        precisions = AverageMeter('Prec', ':6.3f')
        recalls = AverageMeter('Rec', ':6.3f')
        fscores = AverageMeter('Fsc', ':6.3f')
        auc_scores = AverageMeter('AUC', ':6.3f')
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, accuracies, precisions, recalls, fscores, auc_scores],
            prefix='Validate: ' if not test_mode else 'Test: '
        )
    else:
        maes = AverageMeter()
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, maes],
            prefix='Validate: ' if not test_mode else 'Test: '
        )
    if test_mode:
        test_targets = []
        test_preds = []
    
    # switch to evaluation mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        running_loss = 0.0
        for idx, data in enumerate(val_loader):
            image, target = data
            if args.task == 'classification':
                target_normed = target.view(-1).long()
            else:
                target_normed = normalizer.norm(target)

            if args.cuda:
                image = image.cuda(device=cuda_device)
                target_normed = target_normed.cuda(device=cuda_device)

            # compute output
            output = model(image)
            loss = criterion(output, target_normed)
        
            # measure accuracy and record loss
            if args.task == 'classification':
                accuracy, precision, recall, fscore, auc_score =\
                    class_eval(output, target)
                losses.update(loss.item(), target.size(0))
                accuracies.update(accuracy.item(), target.size(0))
                precisions.update(precision.item(), target.size(0))
                recalls.update(recall.item(), target.size(0))
                fscores.update(fscore.item(), target.size(0))
                auc_scores.update(auc_score.item(), target.size(0))
                if test_mode:
                    test_pred = torch.exp(output)
                    test_target = target
                    assert test_pred.shape[1] == 2
                    test_preds += test_pred[:, 1].tolist()
                    test_targets += test_target.view(-1).tolist()
            else:
                mae = compute_mae(normalizer.denorm(output), target)
                losses.update(loss.item(), target.size(0))
                maes.update(mae.item(), target.size(0))
                if test_mode:
                    test_pred = normalizer.denorm(output)
                    test_target = target
                    test_preds += test_pred.view(-1).tolist()
                    test_targets += test_target.view(-1).tolist()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print progress and  write to TensorBoard
            running_loss += loss.item()
            if idx % args.print_freq == 0 and idx != 0 and not test_mode:
                progress.display(idx)
                writer.add_scalar('validation loss',
                                running_loss / args.print_freq,
                                epoch * len(val_loader) + idx)
                running_loss = 0.0
    
    if args.task == 'classification':
        print(' * AUC {auc.avg:.3f}'.format(auc=auc_scores), flush=True)
        return auc_scores.avg
    else:
        print(' * MAE {maes.avg:.3f}'.format(maes=maes), flush=True)
        return maes.avg


class Normalizer(object):
    """Normalize a Tensor and restore it later. """
    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def save_checkpoint(state, is_best):
    check_root = './checkpoints/'
    if not os.path.exists(check_root):
        os.mkdir(check_root)
    filename = check_root + args.run_name + '_checkpoint.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, check_root+args.run_name+'_model_best.pth.tar')


def load_best_model():
    check_root = './checkpoints/'
    if not os.path.exists(check_root):
        print('{} dir does not exist, exiting...', flush=True)
        sys.exit(1)
    filename = check_root + args.run_name + '_model_best.pth.tar'
    if not os.path.isfile(filename):
        print('checkpoint {} not found, exiting...', flush=True)
        sys.exit(1)
    return torch.load(filename)


def compute_mae(prediction, target):
    target = target.detach().cpu()
    prediction = prediction.detach().cpu()
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target):
    prediction = np.exp(prediction.detach().cpu().numpy())
    target = target.detach().cpu().numpy()
    # TODO change threshold
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary', warn_for=tuple())
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.cnt = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries), flush=True)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == "__main__":
    main()


