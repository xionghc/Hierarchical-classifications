from gensim.models.poincare import PoincareModel
import torch
import gensim
import numpy as np


def load_edges_list(file_path):
    relations = []
    with open(file_path) as f:
        for line in f:
            p0, p1 = line.strip().split(' ')
            relations.append((p0, p1))

    return relations


def train(file_path, dim=100, epochs=200):
    relations = load_edges_list(file_path)
    model = PoincareModel(relations, size=dim, negative=2)
    model.train(epochs, print_every=10)
    print('Done')
    return model


def re_rank_embeddings(rank_list, model):
    emb = np.ndarray((192, 100), dtype=np.float)
    for idx, item in enumerate(rank_list):
        emb[idx] = model.kv.word_vec(item)
    return emb


def train_label_emb():
    namelist = []
    with open('./data/namelist.txt') as f:
        namelist = [line.strip() for line in f]

    model = train('./data/food.csv', dim=100, epochs=200)
    embeddings = re_rank_embeddings(namelist, model)
    return embeddings


if __name__ == '__main__':
    print(train_label_emb())
