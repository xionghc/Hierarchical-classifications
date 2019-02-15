import networkx as nx
import numpy.random as random
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def read_graph(fin, weighted=False, directed=False):
    """ Reads the input network in networkx. """
    if weighted:
        G = nx.read_edgelist(fin, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(fin, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not directed:
        G = G.to_undirected()
    return G


# TODO: the return value is null, to be fixed.
def calcul_length(graph, s_nodes, n_nodes):
    assert (len(s_nodes) == len(n_nodes))

    lengths = np.array([])

    for s, n in zip(s_nodes, n_nodes):
        lengths = np.append(lengths, nx.shortest_path_length(graph, s, n))
    return lengths


def gen_data(graph, window_size, sample_size, low=0):
    node_size = graph.number_of_nodes()
    limit = node_size

    nodes = [random.randint(low, limit, sample_size) for _ in range(window_size)]

    u_len = calcul_length(graph, nodes[0], nodes[1])
    v_len = calcul_length(graph, nodes[0], nodes[2])
    t = [-1 if u > v else -1 for u, v in zip(u_len, v_len)]
    return nodes[0], nodes[1], nodes[2], t


def load_data(graph_file, sample_size):
    graph = read_graph(graph_file)
    s, u, v, t = gen_data(graph, 3, sample_size)

    dataset = TensorDataset(torch.tensor(s), torch.tensor(u), torch.tensor(v), torch.tensor(t, dtype=torch.float))
    dataloader = DataLoader(dataset, batch_size=5)
    return dataloader


def draw_graph(graph):
    import matplotlib.pyplot as plt
    nx.draw(graph, with_labels=True, font_weight='bold')
    plt.show()


def test():
    # parameters
    graph_file = 'graph/hierarchy.edgelist'
    window_size = 3
    sample_size = 100
    # end parameters

    g = read_graph(graph_file)
    gen_data(g, window_size, sample_size)
    draw_graph(g)


if __name__ == '__main__':
    test()
