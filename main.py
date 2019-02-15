import argparse
import torch
import torch.nn as nn
import torch.optim as optim


from model import HierarchyModel
from utils import load_data


def train(model, graph_file, epoch, sample_size):
    # load data
    traindata = load_data(graph_file, sample_size)

    # loss
    criterion = nn.MarginRankingLoss(margin=0.2)
    cos = nn.CosineSimilarity(dim=-1)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for _ in range(epoch):
        for s, u, v, t in traindata:
            # forward
            s_emb = model(s)
            u_emb = model(u)
            v_emb = model(v)

            model.zero_grad()

            us_sim = cos(u_emb, s_emb)
            vs_sim = cos(v_emb, s_emb)

            loss = criterion(us_sim, vs_sim, t)

            # back ward & optimize
            loss.backward()
            optimizer.step()
    print('Done')


def evaluate_once(u, v):
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
    parser.add_argument('-sample_size', help='Sample size', type=int, default=1000)
    opt = parser.parse_args()

    model = HierarchyModel(opt.nodesize, opt.dim)
    train(model, opt.graph, opt.epochs, opt.sample_size)
