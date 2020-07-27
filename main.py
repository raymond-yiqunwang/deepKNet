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
## dataset and target property
parser.add_argument('--root', default='./data_gen/', metavar='DATA_DIR',
                    help='path to data root directory')
parser.add_argument('--target', default='MIT', metavar='TARGET_PROPERTY')
## training-relevant params
parser.add_argument('--dim', default=3, type=int, metavar='FEATURE DIMENSION',
                    help='select 2D multi-view CNN or 3D pointnet model')
parser.add_argument('--algo', default='PointNetCls', type=str, metavar='NETWORK')
parser.add_argument('--optim', default='Adam', type=str, metavar='OPTIM',
                    help='torch.optim (Adam or SGD), (default: Adam)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of epochs to run (default: 100)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch start number (useful on restarts)')
parser.add_argument('--batch-size', default=128, type=int, metavar='N',
                    help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', dest='lr',
                    help='initial learning rate (default: 0.001)')
parser.add_argument('--lr-milestones', default=[30, 60], nargs='+', type=int,
                    help='learning rate decay milestones (default: [30, 60])')
parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                    metavar='W', help='weigh decay (default: 0)',
                    dest='weight_decay')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for SGD optimizer')
parser.add_argument('--train-ratio', default=0.7, type=float, metavar='n/N',
                    help='fraction of data for training')
parser.add_argument('--val-ratio', default=0.15, type=float, metavar='n/N',
                    help='fraction of data for validation')
parser.add_argument('--test-ratio', default=0.15, type=float, metavar='n/N',
                    help='fraction of data for test')
## misc
n_threads = torch.get_num_threads()
parser.add_argument('--num_threads', default=n_threads, type=int, metavar='N_thread',
                    help='number of threads used for parallelizing CPU operations')
parser.add_argument('--num_data_workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--print-freq', default=10, type=int, metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: None)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='disable CUDA (default: False)')
parser.add_argument('--gpu-id', default=0, type=int, metavar='GPUID',
                    help='GPU ID (default: 0)')
parser.add_argument('--run-name', default='run1', metavar='RUNID')

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

best_auc = 0.

def main():
    global args, best_auc, cuda_device

    # load data
    data_root = os.path.join(args.root, 'data_pointnet') if args.dim == 3 \
                else os.path.join(args.root, 'data_multiview')
    dataset = deepKNetDataset(root=data_root, target=args.target)
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset, batch_size=args.batch_size, 
        train_ratio=args.train_ratio, val_ratio=args.val_ratio, 
        test_ratio=args.test_ratio, pin_memory=args.cuda, 
        num_data_workers=args.num_data_workers)

    # build model
    if args.algo == 'PointNetCls' and args.dim == 3:
        model = PointNetCls(k=4, dp=0.3)
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
    criterion = nn.NLLLoss()

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
        train(train_loader, model, criterion, optimizer, epoch, writer)

        # evaluate on validation set
        auc = validate(val_loader, model, criterion, epoch, writer)

        scheduler.step()

        # remember best auc and save checkpoint
        is_best = auc > best_auc
        best_auc = max(auc, best_auc)

        # save checkpoint
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_auc': best_auc,
            'optimizer': optimizer.state_dict(),
        }, is_best)

    # test best model
    print('---------Evaluate Model on Test Set---------------', flush=True)
    best_model = load_best_model()
    model.load_state_dict(best_model['state_dict'])
    validate(test_loader, model, criterion, epoch, writer, test_mode=True)


def train(train_loader, model, criterion, optimizer, epoch, writer):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':4.2f')
    losses = AverageMeter('Loss', ':6.3f')
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

    # switch to training mode
    model.train()

    end = time.time()
    running_loss = 0.0
    for idx, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        image, target = data
        target = target.view(-1).long()

        if args.cuda:
            image = image.cuda(device=cuda_device)
            target = target.cuda(device=cuda_device)

        # compute output
        output = model(image)
        loss = criterion(output, target)

        # measure accuracy and record loss
        accuracy, precision, recall, fscore, auc_score =\
            class_eval(output, target)
        losses.update(loss.item(), target.size(0))
        accuracies.update(accuracy.item(), target.size(0))
        precisions.update(precision.item(), target.size(0))
        recalls.update(recall.item(), target.size(0))
        fscores.update(fscore.item(), target.size(0))
        auc_scores.update(auc_score.item(), target.size(0))

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
            target = target.view(-1).long()

            if args.cuda:
                image = image.cuda(device=cuda_device)
                target = target.cuda(device=cuda_device)

            # compute output
            output = model(image)
            loss = criterion(output, target)
        
            # measure accuracy and record loss
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
    
    print(' * AUC {auc.avg:.3f}'.format(auc=auc_scores), flush=True)
    return auc_scores.avg


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


