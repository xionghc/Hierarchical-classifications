import argparse
import torch
from train import train

def evaluate_once(model, u, v):
    u = model(torch.tensor(u))
    v = model(torch.tensor(v))

    sim = torch.dot(u, v)
    print('Similarity = %f' % sim.data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Hierarchy model')
    parser.add_argument('-graph', help='File of edge list', type=str, default='graph/hierarchy.edgelist')
    parser.add_argument('-nodesize', help='Node size', type=int, default=192)
    parser.add_argument('-dim', help='Embedding dimension', type=int, default=128)
    parser.add_argument('-epochs', help='Number of epochs', type=int, default=50)
    parser.add_argument('-batchsize', help='Batch size', type=int, default=50)
    parser.add_argument('-lr', help='Learning rate', type=float)
    parser.add_argument('-sample_size', help='Sample size', type=int, default=10000)
    parser.add_argument('-print_freq', help='Print freq', type=int, default=25)
    opt = parser.parse_args()

    # model = HierarchyModel(opt.nodesize, opt.dim)
    # train_hierarchy_model(model, opt.graph, opt.epochs, opt.sample_size)
    train(172, opt, feature_extract=False, use_pretrained=True)

