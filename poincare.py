import pmath

import torch as th
import torch.nn as nn


def acosh(x):
    return th.log(x + th.sqrt(x**2-1))


def dist_p(u,v):
    z  = 2*th.norm(u-v,2,1)**2
    uu = 1. + th.div(z,((1-th.norm(u,2,1)**2)*(1-th.norm(v,2,1)**2)))
    return acosh(uu)


def dist_row(vector_1, vector_all):
    m = vector_all.size(0)
    return dist_p(vector_1.clone().unsqueeze(0).repeat(m, 1), vector_all)


def dist_matrix(vectors1, vectors2):
    w, h = vectors1.size(0), vectors2.size(0)
    rets = th.zeros(w, h, dtype=th.float)
    for i in range(w):
        rets[i, :] = dist_row(vectors1[i], vectors2)
    return rets


class ToPoincare(nn.Module):
    r"""
    Module which maps points in n-dim Euclidean space
    to n-dim Poincare ball
    """
    def __init__(self, c, train_c=False, train_x=False, ball_dim=None):
        super(ToPoincare, self).__init__()
        if train_x:
            if ball_dim is None:
                raise ValueError("if train_x=True, ball_dim has to be integer, got {}".format(ball_dim))
            self.xp = nn.Parameter(torch.zeros((ball_dim,)))
        else:
            self.register_parameter('xp', None)

        if train_c:
            self.c = nn.Parameter(torch.Tensor([c,]))
        else:
            self.c = c

        self.train_x = train_x

    def forward(self, x):
        if self.train_x:
            xp = pmath.project(pmath.expmap0(self.xp, c=self.c), c=self.c)
            return pmath.project(pmath.expmap(xp, x, c=self.c), c=self.c)
        return pmath.project(pmath.expmap0(x, c=self.c), c=self.c)

    def extra_repr(self):
        return 'c={}, train_x={}'.format(self.c, self.train_x)
