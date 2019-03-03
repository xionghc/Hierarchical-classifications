import argparse
import torch
from train_resnet import train_resnet
# from train import train
from train_emb import train


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Hierarchy model')
    parser.add_argument('-dset', help='Dataset dir', type=str, default='data')
    parser.add_argument('-epochs', help='Number of epochs', type=int, default=25)
    parser.add_argument('-input_size', help='Input size', type=int, default=224)
    parser.add_argument('-num_classes', help='Number of classes', type=int, default=172)
    parser.add_argument('-batch_size', help='Batch size', type=int, default=8)
    parser.add_argument('-gpu', help='gpu', type=str, default=None)

    parser.add_argument('-dim', help='Embedding dimension', type=int, default=256)
    parser.add_argument('-nodesize', help='Node size', type=int, default=192)
    parser.add_argument('-lr', help='Learning rate', type=float)
    parser.add_argument('-sample_size', help='Sample size', type=int, default=100000)
    parser.add_argument('-print_freq', help='Print freq', type=int, default=25)
    args = parser.parse_args()

    # Parsing gpu id.
    if args.gpu is None:
        args.device_ids = list(range(torch.cuda.device_count()))
    else:
        args.device_ids = [int(i) for i in args.gpu.split(',')]

    # train_resnet(args, 172, feature_extract=False, use_pretrained=True)
    train(args, 172, feature_extract=False, use_pretrained=True)
