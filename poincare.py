import torch as th


def acosh(x):
    return th.log(x + th.sqrt(x**2-1))


def dist_p(u,v):
    z  = 2*th.norm(u-v,2,1)**2
    uu = 1. + th.div(z,((1-th.norm(u,2,1)**2)*(1-th.norm(v,2,1)**2)))
    return acosh(uu)


def dist_row(self, vector_1, vector_all):
    m = vector_all.size(0)
    return self.dist_p(vector_1.clone().unsqueeze(0).repeat(m, 1), vector_all)


def dist_matrix(self, vectors1, vectors2):
    w, h = vectors1.size(0), vectors2.size(0)
    rets = th.zeros(w, h, dtype=th.double)
    for i in range(w):
        rets[i, :] = self.dist_row(i)
    return rets


def test_dist_p():
    a = th.tensor([[1.,1]])
    b = th.tensor([[1.,2]])

    print(dist_p(a, b))
    print('*' * 20)


if __name__ == '__main__':
    test_dist_p()
