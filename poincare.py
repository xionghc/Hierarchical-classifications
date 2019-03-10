import torch as th


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


def test_dist_p():
    a = th.randn((32,  100))
    b = th.randn((172, 100))
    dist_mat = dist_matrix(a, b)
 
    print(dist_mat.size())
    print(dist_mat)

    # print(dist_p(a, b))
    print('*' * 20)
    if dist_mat[10][3].item() == dist_p(a[10, :].unsqueeze(0), b[3, :].unsqueeze(0)).item():
        print('test pass')
    else:
        print('test failed')

if __name__ == '__main__':
    test_dist_p()
