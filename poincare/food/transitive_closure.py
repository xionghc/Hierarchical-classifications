import pandas

edges = set()
with open('./food_hierarchy.txt') as f:
    for line in f:
        id_0, id_1 = line.strip().split(' ')
        edges.add((id_0, id_1))

foods = pandas.DataFrame(list(edges), columns=['id1', 'id2'])
foods['weights'] = 1

foods.to_csv('food_closure.csv', index=False)