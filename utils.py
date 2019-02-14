import networkx as nx
import numpy.random as random
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def read_graph(inputfile, weighted=False, directed=False):
    """ Reads the input network in networkx. """
    if weighted:
        G = nx.read_edgelist(inputfile, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(inputfile, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not directed:
        G = G.to_undirected()
    return G


def gen_random_samples(limit, sample_size):
    return random.randint(1, limit, size=sample_size)


# TODO: the return value is null, to be fixed.
def calcul_length(graph, s_nodes, n_nodes):
    assert (len(s_nodes) == len(n_nodes))

    lengths = np.array([])

    for s, n in zip(s_nodes, n_nodes):
        lengths = np.append(lengths, nx.shortest_path_length(graph, s, n))
    return lengths


def gen_data(graph_file, sample_size):
    Graph = read_graph(graph_file)
    node_size = Graph.number_of_nodes()
    limit = node_size+1

    s_nodes = gen_random_samples(limit, sample_size)
    u_nodes = gen_random_samples(limit, sample_size)
    v_nodes = gen_random_samples(limit, sample_size)

    u_len = calcul_length(Graph, s_nodes, u_nodes)
    v_len = calcul_length(Graph, s_nodes, v_nodes)
    diff = u_len - v_len

    t = [-1 if i > 0 else 1 for i in diff]

    return s_nodes, u_nodes, v_nodes, t


def load_data(graph_file, sample_size):
    s, u, v, t = gen_data(graph_file, sample_size)

    dataset = TensorDataset(torch.tensor(s), torch.tensor(u), torch.tensor(v), torch.tensor(t, dtype=torch.float))
    dataloader = DataLoader(dataset, batch_size=5)
    return dataloader


def draw_graph(graph):
    import matplotlib.pyplot as plt
    nx.draw(graph, with_labels=True, font_weight='bold')
    plt.show()


def test():
    # parameters
    graph_file = 'graph/karate.edgelist'
    sample_size = 100
    # end parameters

    gen_data(graph_file, sample_size)
    g = read_graph(graph_file)
    draw_graph(g)


if __name__ == '__main__':
    test()
