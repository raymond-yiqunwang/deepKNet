import argparse
import os
import shutil
import time
from random import sample
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from deepKNet.data import deepKNetDataset
from deepKNet.data import get_train_val_test_loader
from deepKNet.model import deepKNet

parser = argparse.ArgumentParser(description='DeepKNet model')
parser.add_argument('--root', default='./data/', metavar='DATA_ROOT',
                    help='path to root directory')
parser.add_argument('--target', default='band_gap', metavar='TARGET_PROPERTY',
                    help='path to root directory')
parser.add_argument('-j', '--num_workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch start number (useful on restarts)')
parser.add_argument('--lr-milestones', default=[50, 100], type=int, metavar='[N]',
                    help='learning rate decay milestones (default: [50, 100])')
parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N',
                    help='mini-batch size (default=16)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weigh decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: None)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--disable-cuda', action='store_true',
                    help='disable CUDA')
parser.add_argument('--train-size', default=0.8, type=float, metavar='n/N',
                    help='fraction of training data')
parser.add_argument('--val-size', default=0.2, type=float, metavar='n/N',
                    help='fraction of validation data')

# define global variables
args = parser.parse_args()
args.cuda = torch.cuda.is_available() and not args.disable_cuda
best_mae = 1e8

def main():
    global args, best_mae


    # load data
    dataset = deepKNetDataset(root=args.root, target=args.target)
    train_loader, val_loader = get_train_val_test_loader(
        dataset=dataset, batch_size=args.batch_size, 
        train_size=args.train_size, val_size=args.val_size, 
        num_workers=args.num_workers, pin_memory=args.cuda
    )

    # normalizer
    sample_target = torch.tensor([dataset[i][1].item() for i in \
                                 sample(range(len(dataset)), 500)])
    normalizer = Normalizer(sample_target)

    # build model
    model = deepKNet()
    if args.cuda: model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            best_mae = checkpoint['best_mae']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            normalizer.load_state_dict(checkpoint['normalizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # evaluation only
    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    # TensorBoard writer
    summary_file = 'runs/test1'
    if os.path.exists(summary_file):
        shutil.rmtree(summary_file)
    writer = SummaryWriter('runs/test1')

    # learning-rate scheduler
    scheduler = MultiStepLR(optimizer=optimizer, milestones=args.lr_milestones,
                            gamma=0.1, last_epoch=-1)
    
    # training
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, writer, normalizer)

        # evaluate on validation set
        mae = validate(val_loader, model, criterion, epoch, writer, normalizer)

        # remember best mae and save checkpoint
        is_best = mae < best_mae
        best_mea = min(mae, best_mae)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_mae': best_mae,
            'optimizer': optimizer.state_dict(),
            'normalizer': normalizer.state_dict()
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, writer, normalizer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':4e')
    maes = AverageMeter('MAE', ':6.3f')
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

        # features and target
        point_cloud, target = data
        if args.cuda:
            point_cloud = point_cloud.cuda()
            target = target.cuda()

        # normalize target
        target_normed = normalizer.norm(target)

        # compute output
        output = model(point_cloud)
        loss = criterion(output, target_normed)

        # measure accuracy and record loss
        mae = compute_mae(target, normalizer.denorm(output))
        losses.update(loss.item(), target.size(0))
        maes.update(mae, target.size(0))
        
        # compute gradient and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print progress
        if idx % args.print_freq == 0:
            progress.display(idx)
        
        # write to TensorBoard
        running_loss += loss.item()
        if (idx+1) % args.print_freq == 0:
            writer.add_scalar('training loss',
                            running_loss / args.print_freq,
                            epoch * len(train_loader) + idx)
            running_loss = 0.0


def validate(val_loader, model, criterion, epoch, writer, normalizer):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    maes = AverageMeter('MAE', ':6.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, maes],
        prefix='Validate: '
    )
    
    # switch to evaluation mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        running_loss = 0.0
        for idx, data in enumerate(val_loader):
            # features and target
            point_cloud, target = data
            if args.cuda:
                point_cloud = point_cloud.cuda()
                target = target.cuda()

            # normalize target
            target_normed = normalizer.norm(target)

            # compute output
            output = model(point_cloud)
            loss = criterion(output, target_normed)

            # measure accuracy and record loss
            mae = compute_mae(target, normalizer.denorm(output))
            losses.update(loss.item(), target.size(0))
            maes.update(mae, target.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print progress
            if idx % args.print_freq == 0:
                progress.display(idx)
        
            # write to TensorBoard
            running_loss += loss.item()
            if (idx+1) % args.print_freq == 0:
                writer.add_scalar('validation loss',
                                running_loss / args.print_freq,
                                epoch * len(val_loader) + idx)
                running_loss = 0.0
    
    return maes.avg


def save_checkpoint(state, is_best):
    check_root = './checkpoints/'
    filename = check_root + 'checkpoint.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, check_root+'model_best.pth.tar')


class Normalizer(object):
    """Normalize a Tensor and restore it later"""
    def __init__(self, tensor):
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std
    
    def denorm(self, normed_tensor):
        return normed_tensor + self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}
    
    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def compute_mae(y_pred, y_true):
    return torch.mean(torch.abs(y_pred - y_true))


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
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == "__main__":
    main()


