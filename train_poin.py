from gensim.models.poincare import PoincareModel
from gensim.models.poincare import ReconstructionEvaluation


def load_edges_list(file_path):
    relations = []
    with open(file_path) as f:
        for line in f:
            p0, p1 = line.strip().split(' ')
            relations.append((p0, p1))

    return relations


relations = load_edges_list('food.csv')
nodes = []

for pair in relations:
    nodes.append(pair[0])
    nodes.append(pair[1])
nodes = set(nodes)
print(len(nodes))
model = PoincareModel(relations, size=100, negative=2)
model.train(epochs=50)
print('Done')

print('Evaluate')
