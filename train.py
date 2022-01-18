import argparse
import sys
import os
import logging
import shutil
import numpy as np
import warnings
warnings.simplefilter("ignore", UserWarning)
from sklearn import metrics
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from models.data_PointNet import get_PointNet_train_valid_test_loader
from models.PointNet import PointNet

parser = argparse.ArgumentParser(description='Learning the Crystal Structure Genome')
# task-specific parameters
parser.add_argument('--data-root', type=str)
parser.add_argument('--target', choices=['band_gap', 'e_above_hull', 'bulk_modulus', 'shear_modulus'])
parser.add_argument('--model', type=str, choices=['FCNet', 'PointNet'])
parser.add_argument('--run-name', type=str, default='TEST_run1')
parser.add_argument('--hdfs-dir', type=str, default=None)
parser.add_argument('--num-data-workers', type=int, default=8)
parser.add_argument('--rand-seed', type=int, default=123)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--max-Miller', type=int, default=3)
parser.add_argument('--diffraction', type=str, choices=['XRD', 'ND'])
parser.add_argument('--cell-type', type=str, choices=['primitive', 'conventional'])
# FCNet-specific hyper-parameters
parser.add_argument('--fcnet-permute_hkl', type=lambda x: eval(x), default=False)
parser.add_argument('--fcnet-randomize_hkl', type=lambda x: eval(x), default=False)
parser.add_argument('--fcnet-fc-dims', type=int, nargs='+')
# PointNet-specific hyper-parameters
parser.add_argument('--pointnet-conv-filters', type=int, nargs='+')
parser.add_argument('--pointnet-fc-dims', type=int, nargs='+')
parser.add_argument('--pointnet-randomly-scale-intensity', type=lambda x: eval(x), default=False)
parser.add_argument('--pointnet-systematic-absence', type=lambda x: eval(x), default=False)
# general hyper-parameters
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001)
# default parameters
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--print-freq', type=int, default=200)
parser.add_argument('--restore-path', type=str, default='')

# parse args
args = parser.parse_args(sys.argv[1:])
args.device = torch.device('cuda:0') if torch.cuda.is_available() \
                                     else torch.device('cpu')
args.start_epoch = 0                                     
print('User defined variables:', flush=True)
for key, val in vars(args).items():
    print('  => {:17s}: {}'.format(key, val), flush=True)

handler = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%y/%m/%d %H:%M:%S', handlers=[handler])

best_performance = 0

# random seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

def main():
    global args, best_performance

    set_seed(args.rand_seed)

    if args.model == 'FCNet':
        # dataloader
        train_loader, valid_loader, test_loader = get_FCNet_train_valid_test_loader(
            root=args.data_root,
            target=args.target,
            max_Miller=args.max_Miller,
            diffraction=args.diffraction,
            cell_type=args.cell_type,
            permute_hkl=args.fcnet_permute_hkl,
            randomize_hkl=args.fcnet_randomize_hkl,
            batch_size=args.batch_size,
            num_data_workers=args.num_data_workers)
        # construct model
        model = FCNet(max_Miller=args.max_Miller,
                      fc_dims=args.fcnet_fc_dims,
                      dropout=args.dropout)
    elif args.model == 'PointNet':
        # dataloader
        train_loader, valid_loader, test_loader = get_PointNet_train_valid_test_loader(
            root=args.data_root,
            target=args.target,
            max_Miller=args.max_Miller,
            diffraction=args.diffraction,
            cell_type=args.cell_type,
            randomly_scale_intensity=args.pointnet_randomly_scale_intensity,
            systematic_absence=args.pointnet_systematic_absence,
            batch_size=args.batch_size,
            num_data_workers=args.num_data_workers)
        # construct model
        model = PointNet(conv_filters=args.pointnet_conv_filters,
                         fc_dims=args.pointnet_fc_dims,
                         dropout=args.dropout)
    else:
        raise NotImplementedError

    # send model to device
    if torch.cuda.is_available():
        print('running on GPU:\n')
    else:
        print('running on CPU\n')
    model = model.to(args.device)

    # show number of trainable model parameters
    trainable_params = sum(p.numel() for p in model.parameters() 
                           if p.requires_grad)
    print('Number of trainable model parameters: {:d}'.format(trainable_params))

    # define loss function 
    criterion = torch.nn.NLLLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # HDFS 
    if args.hdfs_dir is not None:
        os.system(f'hdfs dfs -mkdir -p {args.hdfs_dir}')

    # optionally resume from a checkpoint
    if args.restore_path != '':
        assert os.path.isfile(args.restore_path)
        print("=> loading checkpoint '{}'".format(args.restore_path), flush=True)
        checkpoint = torch.load(args.restore_path, map_location=torch.device('cpu'))
        args.start_epoch = checkpoint['epoch'] + 1
        best_performance = checkpoint['best_performance']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.restore_path, checkpoint['epoch']), flush=True)

    # learning-rate scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer,
                                            T_0=args.epochs,
                                            eta_min=1E-8)
    
    print('\nStart training..', flush=True)
    for epoch in range(args.start_epoch, args.start_epoch+args.epochs):
        lr = scheduler.get_last_lr()
        logging.info('Epoch: {}, LR: {:.6f}'.format(epoch, lr[0]))

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        performance = validate(valid_loader, model, criterion)
        
        scheduler.step()

        # check performance
        is_best = performance > best_performance
        best_performance = max(performance, best_performance)

        # save checkpoint
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_performance': best_performance,
            'optimizer': optimizer.state_dict(),
        }, is_best, args)

    # test best model
    print('---------Evaluate Model on Test Set---------------', flush=True)
    best_model = load_best_model()
    print('best validation performance: {:.3f}'.format(best_model['best_performance']))
    model.load_state_dict(best_model['state_dict'])
    validate(test_loader, model, criterion, test_mode=True)


def train(train_loader, model, criterion, optimizer, epoch):
    # init average meters
    losses = AverageMeter('Loss', ':6.3f')
    accuracies = AverageMeter('Accu', ':6.3f')
    precisions = AverageMeter('Prec', ':6.3f')
    recalls = AverageMeter('Rec', ':6.3f')
    fscores = AverageMeter('Fsc', ':6.3f')
    auc_scores = AverageMeter('AUC', ':6.3f')
    ave_precisions = AverageMeter('AP', ':6.3f')
    report = [losses, accuracies, precisions, 
              recalls, fscores, ave_precisions, auc_scores]
    # progress meter
    progress = ProgressMeter(
        len(train_loader),
        report,
        prefix="Epoch: [{}]".format(epoch)
    )

    # switch to training mode
    model.train()

    for idx, (data, target) in enumerate(train_loader):
        # send data to device
        data = data.to(args.device)
        target = target.to(args.device)
        
        # compute model output
        output = model(data)

        # compute loss
        loss = criterion(output, target.squeeze(dim=-1))

        # performance metrics
        accuracy, precision, recall, fscore, auc_score, ave_precision =\
            class_eval(output.detach().cpu().numpy(), target.cpu().numpy())
        losses.update(loss.cpu().item(), target.size(0))
        accuracies.update(accuracy, target.size(0))
        precisions.update(precision, target.size(0))
        recalls.update(recall, target.size(0))
        fscores.update(fscore, target.size(0))
        auc_scores.update(auc_score, target.size(0))
        ave_precisions.update(ave_precision, target.size(0))

        # compute gradient and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (idx+1) % args.print_freq == 0:
            progress.display(idx+1)


def validate(valid_loader, model, criterion, test_mode=False):
    # init average meters
    losses = AverageMeter('Loss', ':6.3f')
    accuracies = AverageMeter('Accu', ':6.3f')
    precisions = AverageMeter('Prec', ':6.3f')
    recalls = AverageMeter('Rec', ':6.3f')
    fscores = AverageMeter('Fsc', ':6.3f')
    auc_scores = AverageMeter('AUC', ':6.3f')
    ave_precisions = AverageMeter('AP', ':6.3f')
    report = [losses, accuracies, precisions, 
              recalls, fscores, ave_precisions, auc_scores]
    # progress meter
    progress = ProgressMeter(
        len(valid_loader),
        report,
        prefix='Validate: ' if not test_mode else 'Test: '
    )
    
    # switch to evaluation mode
    model.eval()

    with torch.no_grad():
        for idx, (data, target) in enumerate(valid_loader):

            # send data to device
            data = data.to(args.device)
            target = target.to(args.device)
        
            # compute model output
            output = model(data)

            # compute loss
            loss = criterion(output, target.squeeze(dim=-1))
        
            # performance metrics
            accuracy, precision, recall, fscore, auc_score, ave_precision =\
                class_eval(output.detach().cpu().numpy(), target.cpu().numpy())
            losses.update(loss.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))
            ave_precisions.update(ave_precision, target.size(0))
        progress.display(idx+1)
    return auc_scores.avg


def save_checkpoint(state, is_best, args):
    check_root = args.run_name + '_checkpoints/'
    if not os.path.exists(check_root):
        os.mkdir(check_root)
    filename = check_root + 'model_last.pt'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, check_root + 'model_best.pt')
    # save to HDFS
    if args.hdfs_dir is not None:
        os.system(f'hdfs dfs -put -f {check_root} {args.hdfs_dir}')


def load_best_model():
    filename = args.run_name + '_checkpoints/' + 'model_best.pt'
    if not os.path.isfile(filename):
        print('checkpoint {} not found, exiting...', flush=True)
        sys.exit(1)
    return torch.load(filename)


def class_eval(prediction, target_label):
    pred_label = np.argmax(np.exp(prediction), axis=1)
    assert(prediction.shape[1] == 2)
    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
        target_label, pred_label, average='binary', warn_for=tuple())
    try:
        auc_score = metrics.roc_auc_score(target_label, prediction[:,1])
    except:
        auc_score = float('-inf')
    accuracy = metrics.accuracy_score(target_label, pred_label)
    ave_precision = metrics.average_precision_score(target_label, prediction[:,1])
    
    return accuracy, precision, recall, fscore, auc_score, ave_precision


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.avg = 0.
        self.sum = 0.
        self.cnt = 0.

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
    
    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('  '.join(entries), flush=True)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == "__main__":
    main()


