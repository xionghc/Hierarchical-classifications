import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils import save_checkpoint, accuracy, adjust_learning_rate, AverageMeter
from folder import ImageFolder
from model import init_encoder_model
from poincare import dist_p, dist_matrix
from train_poin import train_label_emb

best_acc1 = 0

def main_worker(args):
    global best_acc1

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    if args.pretrained:
        print('=> using pre-trained model')
        model = init_encoder_model(args.embed_size, args.pretrained)
    else:
        print('=> creating model')
        model = init_encoder_model(args.embed_size, args.pretrained)
    model = model.to(args.device)

    # print("Params to learn:")
    model_params = []
    for name, param in dict(model.named_parameters()).items():
        if name.find("bias") > -1:
            print('Model Layer {} will be trained @ {}'.format(name, args.lr*2))
            model_params.append({'params': param, 'lr': args.lr*2, 'weight_decay': 0})
        else:
            print('Model Layer {} will be trained @ {}'.format(name, args.lr))
            model_params.append({'params': param, 'lr': args.lr, 'weight_decay': args.weight_decay})

    e_weights = torch.FloatTensor(train_label_emb())
    label_model = nn.Embedding.from_pretrained(e_weights)
    label_model = label_model.to(args.device)
    
    optimizer = optim.SGD(model_params, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(args.device)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    # Data loading code
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(args.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(args.input_size),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {x: ImageFolder(os.path.join(args.data, x), data_transforms[x]) for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=4)
        for x in['train', 'val']}

    if args.evaluate:
        validate(dataloaders_dict['val'], model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(dataloaders_dict['train'], model, label_model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(dataloaders_dict['val'], model, label_model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
            }, is_best)


def pairwise_pc_distance(matrix_1, matrix_2, targets):
    # matrix_1 is N-by-Dim
    # matrix_2 is C-by-Dim
    # return N-by-1 

    eps = 1e-19
    # Pairwise Euclidean Distance
    N, M = matrix_1.size(0), matrix_2.size(0)
    norms_1 = torch.sum(matrix_1**2, dim=1, keepdim=True) # matrix_1's l2-norm, shape (N, 1)
    norms_2 = torch.sum(matrix_2**2, dim=1, keepdim=True) # matrix_2's l2-norm, shape (M, 1)
    
    norms_1_sta = norms_1.expand(N, M); # expand norms_1 to (N, M)
    norms_2_sta = norms_2.transpose(0, 1).expand(N, M) # expand norms_2 to (N, M)

    #print("1 max {}".format(torch.max(norms_1_sta)))
    #print("1 min {}".format(torch.min(norms_1_sta)))

    euc_dist = norms_1_sta + norms_2_sta - 2 * matrix_1.mm(matrix_2.t())
    # Denorm
    denorm_1_sta = torch.clamp( 1 - norms_1_sta, eps, 1 - eps) # (N, M)
    denorm_2_sta = torch.clamp( 1 - norms_2_sta, eps, 1 - eps) # (N, M)

    denorm = denorm_1_sta * denorm_2_sta # (N, M)

    # Euclidean / Denorm
    dist_stage = torch.div(euc_dist , denorm + eps) # (N, M)
    dist_stage = 1 + 2 * dist_stage

    dist_stage = torch.log(dist_stage + torch.sqrt(dist_stage ** 2 - 1))

    scores = 1 / ( dist_stage + eps)
    #l2norm = torch.norm(scores, p=2, dim=1, keepdim=True).detach() + eps
    #scores  = scores / l2norm

    dist_stage = dist_stage.gather(1, targets.unsqueeze(1)) # Select corresponding nodes (N, 1)

    loss_dist = torch.mean(dist_stage)
    return loss_dist


def norm_pc_distance(matrix_1, matrix_2, targets):
    eps = 1e-19
    N, M = matrix_1.size(0), matrix_2.size(0)
    norms_1 = torch.sum(matrix_1**2, dim=1, keepdim=True) # matrix_1's l2-norm, shape (N, 1)
    # To coord-origin dist
    euc_dist_1 = norms_1
    denorm_1   = torch.clamp(1 - norms_1, eps, 1 - eps)
    dist_stage_1 = torch.div(euc_dist_1, denorm_1 + eps)
    dist_stage_1 = 1 + 2 * dist_stage_1
    dist_stage_1 = torch.log(dist_stage_1 + torch.sqrt(dist_stage_1 ** 2 - 1))

    norms_2 = torch.sum(matrix_2**2, dim=1, keepdim=True) # matrix_2's l2-norm, shape (M, 1)
    norms_2_sta = norms_2.transpose(0, 1).expand(N, M) # expand norms_2 to (N, M)
    euc_dist_2 = norms_2_sta
    denorm_2   = torch.clamp(1 - norms_2_sta, eps, 1 - eps)
    dist_stage_2 = torch.div(euc_dist_2, denorm_2)
    dist_stage_2 = 1 + 2 * dist_stage_2 # N-by-M
    dist_stage_2 = torch.log(dist_stage_2 + torch.sqrt(dist_stage_2 ** 2 - 1))
    # select by index
    dist_stage_2 = dist_stage_2.gather(1, targets.unsqueeze(1)) # Select corresponding nodes (N, 1)
    loss_norm = torch.abs(dist_stage_1 - dist_stage_2)
    loss_norm = torch.mean(loss_norm)
    return loss_norm


def train(train_loader, model, label_model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_norm = AverageMeter()
    losses_dist = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    label_norm = label_model.weight.norm(dim=1)
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.to(args.device)
        target = target.to(args.device)

        # compute output
        output = model(input)

        #loss = dist_p(output, label_model(target)).mean()
        # measure accuracy and record loss
        #preds = dist_matrix(output, label_model.weight[:172]).to(args.device)
        #acc1, acc5 = accuracy(preds, target, topk=(1, 5), largest=False)
        label_weight = label_model.weight[:172]
        loss_dist = pairwise_pc_distance(output, label_weight, target)
        loss_norm = norm_pc_distance(output, label_weight, target)
        loss = loss_norm + loss_dist
        
        preds = dist_matrix(output, label_model.weight[:172]).to(args.device)
        acc1, acc5 = accuracy(preds, target, topk=(1, 5), largest=False)
        losses.update(loss.item(), input.size(0))
        losses_norm.update(loss_norm.item(), input.size(0))
        losses_dist.update(loss_dist.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss_norm {ln.val:.4f} ({ln.avg:.4f})\t'
                  'Loss_dist {ld.val:.4f} ({ld.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, ln=losses_norm, ld=losses_dist,
                   loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, label_model, criterion, args):
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
            #loss = dist_p(output, label_model(target)).mean()

            # measure accuracy and record loss
            #preds = dist_matrix(output, label_model.weight[0:172])
            #acc1, acc5 = accuracy(preds.to(args.device), target, topk=(1, 5), largest=False)
            label_weight = label_model.weight[:172]
            loss_dist = pairwise_pc_distance(output, label_weight, target)
            loss_norm = norm_pc_distance(output, label_weight, target)
            loss = 2 *loss_norm + loss_dist
        
            preds = dist_matrix(output, label_model.weight[:172]).to(args.device)
            acc1, acc5 = accuracy(preds, target, topk=(1, 5), largest=False)
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def main():
    parser = argparse.ArgumentParser(description='Train Hierarchy model')
    parser.add_argument('-data', help='Dataset dir', type=str, default='data')
    parser.add_argument('-embed_size', help='Embedding size', type=int, default=100)
    parser.add_argument('--start-epoch', help='manual epoch number (useful on restarts)', type=int, default=0)
    parser.add_argument('-epochs', help='Number of epochs', type=int, default=50)
    parser.add_argument('-input_size', help='Input size', type=int, default=224)
    parser.add_argument('-num_classes', help='Number of classes', type=int, default=172)
    parser.add_argument('-batch_size', help='Batch size', type=int, default=32)
    parser.add_argument('-gpu', help='gpu', type=int, default=0)
    parser.add_argument('-lr', help='Learning rate', type=float, default=0.001)
    parser.add_argument('-momentum', help='momentum', type=float, default=0.9)
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-print_freq', help='Print freq', type=int, default=50)
    parser.add_argument('-evaluate', type=bool, default=False)
    parser.add_argument('-pretrained', help='Pretrained', type=str, default=None)
    parser.add_argument('-resume', help='Resume', type=str, default=None)

    parser.add_argument('-nodesize', help='Node size', type=int, default=192)
    args = parser.parse_args()

    main_worker(args)


if __name__ == '__main__':
    main()
