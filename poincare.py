import torch as th


def acosh(x):
    return th.log(x + th.sqrt(x**2-1))


def dist_p(u,v):
    z  = 2*th.norm(u-v,2,1)**2
    uu = 1. + th.div(z,((1-th.norm(u,2,1)**2)*(1-th.norm(v,2,1)**2)))
    return acosh(uu)

# def vector_distance_batch(vector_1, vector_all):
#     euclidean_dists = th.norm(vector_1 - vector_all, dim=1)
#     norm = th.norm(vector_1)
#     all_norms = th.norm(vector_all, dim=1)
#     return acosh(
#         1 + 2 * (
#             (euclidean_dists ** 2) / ((1 - norm ** 2) * (1 - all_norms ** 2))
#         )
#     )

def test_dist_p():
    a = th.tensor([[1.,1]])
    b = th.tensor([[1.,2]])

    print(dist_p(a, b))
    print('*' * 20)


if __name__ == '__main__':
    test_dist_p()