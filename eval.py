import os
import time
import argparse

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from folder import ImageFolder
from model import init_encoder_model
from train_poin import train_label_emb
from utils import AverageMeter
from poincare import dist_p, dist_matrix


def main_worker(args):
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    if args.pretrained:
        print('=> using pre-trained model')
        model = init_encoder_model(args.embed_size, args.pretrained)
    else:
        print('=> Pretrained model not specified.')
        return

    model = model.to(args.device)

    e_weights = torch.FloatTensor(train_label_emb())
    label_model = nn.Embedding.from_pretrained(e_weights)
    label_model = label_model.to(args.device)

    # Data loading code
    print("Initializing Datasets and Dataloaders...")

    transformer = transforms.Compose([
            transforms.Resize(args.input_size),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # Create training and validation datasets
    image_datasets = ImageFolder(os.path.join(args.data, 'test'), transformer)
    # Create training and validation dataloaders
    dataloaders= torch.utils.data.DataLoader(image_datasets, batch_size=args.batch_size, shuffle=True, num_workers=4)

    print('Evaluating')
    validate(dataloaders, model, label_model, args)
    print('Finished')


def validate(val_loader, model, label_model, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.to(args.device)
            target = target.to(args.device)

            # compute output
            output = model(input)
            loss = dist_p(output, label_model(target)).mean()

            # measure accuracy and record loss
            preds = dist_matrix(output, label_model.weight[0:172])
            acc1, acc5 = accuracy(preds.to(args.device), target, topk=(1, 5), largest=False)
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def accuracy(output, target, topk=(1,), largest=True):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, largest, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def main():
    parser = argparse.ArgumentParser(description='Train Hierarchy model')
    parser.add_argument('-data', help='Dataset dir', type=str, default='data')
    parser.add_argument('-embed_size', help='Embedding size', type=int, default=100)
    parser.add_argument('-input_size', help='Input size', type=int, default=224)
    parser.add_argument('-num_classes', help='Number of classes', type=int, default=172)
    parser.add_argument('-batch_size', help='Batch size', type=int, default=32)
    parser.add_argument('-gpu', help='gpu', type=int, default=0)
    parser.add_argument('-print_freq', help='Print freq', type=int, default=1000)
    parser.add_argument('-pretrained', help='Pretrained', type=str, default=None)

    parser.add_argument('-nodesize', help='Node size', type=int, default=192)
    args = parser.parse_args()

    main_worker(args)


if __name__ == '__main__':
    main()