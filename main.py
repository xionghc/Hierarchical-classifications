import argparse
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Hierarchy model')
    parser.add_argument('-data', help='Dataset dir', type=str, default='data')
    parser.add_argument('-epochs', help='Number of epochs', type=int, default=50)
    parser.add_argument('-input_size', help='Input size', type=int, default=224)
    parser.add_argument('-num_classes', help='Number of classes', type=int, default=172)
    parser.add_argument('-batch_size', help='Batch size', type=int, default=32)
    parser.add_argument('-gpu', help='gpu', type=int, default=0)
    parser.add_argument('-lr', help='Learning rate', type=float)
    parser.add_argument('-print_freq', help='Print freq', type=int, default=25)
    parser.add_argument('-resume', help='Resume', type=bool, default=False)

    parser.add_argument('-nodesize', help='Node size', type=int, default=192)
    parser.add_argument('-sample_size', help='Sample size', type=int, default=100000)
    args = parser.parse_args()


