import argparse
import pandas as pd
import sys
import os
import shutil
import time
import numpy as np
from random import sample
from sklearn import metrics
import torch
from torch.optim.lr_scheduler import MultiStepLR
from tensorboardX import SummaryWriter
from deepKNet.data import get_train_valid_test_loader
from deepKNet.model3D_open import PointNet

parser = argparse.ArgumentParser(description='deepKNet model')
parser.add_argument('--root', metavar='DATA_DIR')
parser.add_argument('--modelpath', metavar='MODEL_PATH')
parser.add_argument('--target', metavar='TARGET_PROPERTY')
parser.add_argument('--nclass', type=int)
parser.add_argument('--threshold', default=0.5, type=float)
parser.add_argument('--run_name', default='runx', metavar='RUNID')
parser.add_argument('--gpu_id', default=0, type=int, metavar='GPUID')
# hyper parameter tuning
parser.add_argument('--npoint', type=int, metavar='NPOINT CUTOFF')
parser.add_argument('--point_dim', default=4, type=int, metavar='NPOINT DIM')
parser.add_argument('--data_aug', default='True', type=str)
parser.add_argument('--rot_range', type=float, nargs='+')
parser.add_argument('--random_intensity', type=str)
parser.add_argument('--systematic_absence', type=str)
parser.add_argument('--conv_dims', type=int, nargs='+')
parser.add_argument('--nbert', default=4, type=int)
parser.add_argument('--fc_dims', type=int, nargs='+')
parser.add_argument('--pool', default='CLS', type=str)
parser.add_argument('--batch_size', default=64, type=int, metavar='N')
parser.add_argument('--stn', action='store_true')
# default params
n_threads = torch.get_num_threads()
parser.add_argument('--num_threads', default=n_threads, type=int, metavar='N_thread')
parser.add_argument('--num_data_workers', default=4, type=int, metavar='N')
parser.add_argument('--print_freq', default=1, type=int, metavar='N')
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

def main():
    global args, cuda_device

    # get data loader
    _, _, test_loader = get_train_valid_test_loader(
        root=args.root,
        target=args.target,
        npoint=args.npoint, 
        point_dim=args.point_dim,
        data_aug=args.data_aug=='True',
        rot_range=args.rot_range,
        random_intensity=args.random_intensity=='True',
        systematic_absence=args.systematic_absence=='True',
        batch_size=args.batch_size,
        num_data_workers=args.num_data_workers,
        pin_memory=args.cuda)

    # build model
    model = PointNet(nclass=args.nclass,
                     conv_dims=args.conv_dims,
                     nbert=args.nbert,
                     fc_dims=args.fc_dims,
                     pool=args.pool,
                     dropout=0.0,
                     stn=args.stn)
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
    criterion = torch.nn.NLLLoss()
    if args.cuda:
        criterion = criterion.cuda(device=cuda_device)

    # load checkpoint
    if os.path.isfile(args.modelpath):
        print("=> loading checkpoint '{}'".format(args.modelpath), flush=True)
        checkpoint = torch.load(args.modelpath, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}', existing..".format(args.modelpath))
        sys.exit(1)

    validate(test_loader, model, criterion, args.nclass)



def validate(valid_loader, model, criterion, nclass):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':4.2f')
    losses = AverageMeter('Loss', ':6.3f')
    accuracies = AverageMeter('Accu', ':6.3f')
    precisions = AverageMeter('Prec', ':6.3f')
    recalls = AverageMeter('Rec', ':6.3f')
    fscores = AverageMeter('Fsc', ':6.3f')
    auc_scores = AverageMeter('AUC', ':6.3f')
    ave_precisions = AverageMeter('AP', ':6.3f')
    if nclass == 2:
        report = [batch_time, data_time, losses, accuracies, precisions, 
                  recalls, fscores, ave_precisions, auc_scores]
    else:
        report = [batch_time, data_time, losses, accuracies]
    progress = ProgressMeter(
        len(valid_loader),
        report,
        prefix='Test:'
    )
    
    # switch to evaluation mode
    model.eval()

    misclassified_ids = []
    material_ids_all = []
    true_labels = []
    true_label_scores = []
    mispred_labels = []
    mispred_label_scores = []
    with torch.no_grad():
        end = time.time()
        running_loss = 0.0
        for idx, data in enumerate(valid_loader):
            image, target, material_ids = data
            material_ids_all += list(material_ids)
            
            # optionally skip the last batch
            if target.size(0) < 8: continue
            
            target = target.view(-1)

            if args.cuda:
                image = image.cuda(device=cuda_device)
                target = target.cuda(device=cuda_device)

            # compute output
            output = model(image)
            loss = criterion(output, target)
        
            # measure accuracy and record loss
            accuracy, precision, recall, fscore, auc_score, ave_precision, fpr, tpr, thresholds =\
                class_eval(output, target, args.threshold)
            losses.update(loss.item(), target.size(0))
            accuracies.update(accuracy.item(), target.size(0))
            precisions.update(precision.item(), target.size(0))
            recalls.update(recall.item(), target.size(0))
            fscores.update(fscore.item(), target.size(0))
            auc_scores.update(auc_score.item(), target.size(0))
            ave_precisions.update(ave_precision.item(), target.size(0))
#            ROC_curve = pd.DataFrame([fpr, tpr]).transpose()
#            ROC_curve.to_csv("ROC_curve{}.csv".format(str(idx)), index=False, header=['fpr', 'tpr'])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print progress and  write to TensorBoard
            running_loss += loss.item()
            if (idx+1) % args.print_freq == 0:
                progress.display(idx+1)
                running_loss = 0.0

            # print misclassified
            pred_scores = np.exp(output.cpu().numpy())
            test_preds = np.argmax(pred_scores, axis=1)
            test_targets = target.cpu().numpy()
            for key, val in enumerate(test_preds==test_targets):
                if not val:
                    misclassified_ids.append(material_ids[key])
                    true_label = test_targets[key]
                    true_labels.append(true_label)
                    true_label_scores.append(pred_scores[key,true_label])
                    mispred_label = test_preds[key]
                    mispred_labels.append(mispred_label)
                    mispred_label_scores.append(pred_scores[key,mispred_label])
    
    with open('material_ids_all.txt', 'w') as f:
        for mat_id in material_ids_all:
            f.write(mat_id)
            f.write('\n')
    print('misclass number:', len(mispred_labels))
    misclass_out = pd.DataFrame([misclassified_ids, true_labels, true_label_scores, \
                                                    mispred_labels, mispred_label_scores])
    misclass_out = misclass_out.transpose()
    all_pred = pd.DataFrame([material_ids, test_preds, pred_scores, test_targets])
    all_pred = all_pred.transpose()
    all_pred_header = ['id', 'pred_label', 'pred_score', 'true_label']
    all_pred.to_csv('all_predict.csv', header=all_pred_header, index=False)
    header_out = ['id', 'true', 'true_score', 'pred', 'pred_score']
    misclass_out.to_csv('misclass.csv', header=header_out, index=False)

    if nclass == 2:
        print(' * AUC {auc.avg:.3f}'.format(auc=auc_scores), flush=True)
        return auc_scores.avg
    else:
        print(' * ACCU {accu.avg:.3f}'.format(accu=accuracies), flush=True)
        return accuracies.avg


def class_eval(prediction, target, threshold):
    prediction = np.exp(prediction.detach().cpu().numpy())
    pred_label = np.argmax(prediction, axis=1)
#    pred_label = prediction[:,1] > threshold
    target = target.detach().cpu().numpy()
    target_label = np.squeeze(target)
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary', warn_for=tuple())
        try:
            auc_score = metrics.roc_auc_score(target_label, prediction[:,1])
        except:
            auc_score = np.float64(-1E8)
        accuracy = metrics.accuracy_score(target_label, pred_label)
        ave_precision = metrics.average_precision_score(target_label, prediction[:,1])
        fpr, tpr, thresholds = metrics.roc_curve(target_label, prediction[:,1])
    else:
        correct = np.equal(pred_label, target_label).sum()
        precision, recall = np.float64(0.0), np.float64(0.0)
        fscore, auc_score = np.float64(0.0), np.float64(0.0)
        accuracy = np.float64(correct/float(target_label.size))
        ave_precision = np.float64(0.0)
        fpr, tpr, thresholds = [], [], []
    return accuracy, precision, recall, fscore, auc_score, ave_precision, fpr, tpr, thresholds


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
        print('  '.join(entries), flush=True)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == "__main__":
    main()


