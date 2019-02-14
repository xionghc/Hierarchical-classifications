import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F


from model import SementicModel
from utils import load_data


def train(model):
    # parameters
    epoch = 10
    graph_file = 'graph/karate.edgelist'
    sample_size = 10000

    # load data
    traindata = load_data(graph_file, sample_size)

    # loss
    criterion = nn.MarginRankingLoss(margin=0.2)
    cos = nn.CosineSimilarity(dim=-1)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for _ in range(epoch):
        for batch_id, (s, u, v, t) in enumerate(traindata):
            # forward
            s_emb = model(s)
            u_emb = model(u)
            v_emb = model(v)

            # TODO: should get data in torch.tensor, but not int.  Otherwise can not backward.
            model.zero_grad()

            us_sim = cos(u_emb, s_emb)
            vs_sim = cos(v_emb, s_emb)

            print(us_sim)

            loss = criterion(us_sim, vs_sim, t)

            if batch_id % 1000:
                # print(batch_id)
                print(loss.data)
                # print(model.embed(torch.tensor([1])))

            # back ward & optimize
            loss.backward()
            optimizer.step()
    print('Done')

    while True:
        u = int(input())
        v = int(input())
        uuu = model(torch.tensor(u))
        vvv = model(torch.tensor(v))

        sim = torch.dot(uuu, vvv)

        print('similarity = %f' % sim.data)


if __name__ == '__main__':
    model = SementicModel(35, 128)
    train(model)
