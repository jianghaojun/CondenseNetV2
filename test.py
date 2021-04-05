import os
import PIL
import time
import argparse
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from PIL import Image
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from model import *
from utils import measure_model, ProgressMeter, AverageMeter, accuracy

parser = argparse.ArgumentParser(description='PyTorch Condensed Convolutional Networks')
parser.add_argument('--data_url', metavar='DIR', default='~/data',
                    help='path to dataset')
parser.add_argument('--dataset', metavar='DATASET', default='imagenet', choices=['cifar10', 'cifar100', 'imagenet'],
                    help='dataset')
parser.add_argument('--model', default='condensenetv2.cdnv2_a', type=str, metavar='M',
                    help='model to train the dataset')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=1024, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--train_url', type=str, metavar='PATH', default='test',
                    help='path to save result and checkpoint (default: results/savedir)')
parser.add_argument('--evaluate_from', default=None, type=str, metavar='PATH',
                    help='path to saved checkpoint (default: none)')
parser.add_argument('--print_freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')


def main():
    args = parser.parse_args()

    assert args.dataset == 'imagenet'
    args.num_classes = 1000
    args.IMAGE_SIZE = 224

    if args.train_url and not os.path.exists(args.train_url):
        os.makedirs(args.train_url)

    ### Create Model
    model = eval(args.model)(args)

    assert args.evaluate_from is not None, "Please give the checkpoint path of the model which is used to be " \
                                          "evaluated!"

    print("=> Load model from '{}'".format(args.evaluate_from))

    state_dict = torch.load(args.evaluate_from)['state_dict']
    print('Loading pretrained parameter from state_dict...')

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module'):
            name = k[7:]
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model = nn.DataParallel(model).cuda()
    print("=> Load checkpoint done!")

    ### Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    cudnn.benchmark = True

    ### Data loading
    valdir = args.data_url + 'val/'

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256, interpolation=PIL.Image.BILINEAR if args.model in ['cdnv2_c', 'converted_cdnv2_c'] else PIL.Image.BICUBIC),
                ### Documentation for transforms.Resize
                # If size is a sequence like (h, w), output size will be matched to this.
                # If size is an int, smaller edge of the image will be matched to this number.
                # i.e, if height > width, then image will be rescaled to (size * height / width, size)
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    val_acc_top1, val_acc_top5, valid_loss = [], [], []
    val_acc1, val_acc5, val_loss = validate(val_loader, model, criterion, args)
    val_acc_top1.append(val_acc1)
    val_acc_top5.append(val_acc5)
    valid_loss.append(val_loss)
    df = pd.DataFrame({'val_acc_top1': val_acc_top1, 'val_acc_top5': val_acc_top5, 'valid_loss': valid_loss})
    if args.train_url:
        log_file = os.path.join(args.train_url + 'log.txt')
        with open(log_file, "w") as f:
            df.to_csv(f)

    n_flops, n_params = measure_model(model, args.IMAGE_SIZE, args.IMAGE_SIZE)
    print('FLOPs: %.2fM, Params: %.2fM' % (n_flops / 1e6, n_params / 1e6))
    if args.train_url:
        log_file = os.path.join(args.train_url + 'measure_model.txt')
        with open(log_file, "w") as f:
            f.write(str(n_flops / 1e6))
            f.write(str(n_params / 1e6))

        f.close()
    return


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    ### Switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            ### Compute output
            output = model(images)
            loss = criterion(output, target)

            ### Measure accuracy and record loss
            acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            ### Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


if __name__ == '__main__':
    main()
