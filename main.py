from torch_geometric.datasets import Planetoid

import os.path as osp

from Memory import Memory
from GraphProcessor import GraphProcessor
from GraphEncoder import GraphEncoder

path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/Planetoid')
graph = Planetoid(path, name="Cora")

gp = GraphProcessor(graph)
ge = GraphEncoder(gp)
memory = Memory(ge)

node_id = 0  # selected node id
k = 2

memory.make_text_embedding()
memory.make_features_embedding()
memory.make_node_embedding()
top_k_indices = memory.extract_top_k_similar_nodes(node_id, k)

print('Selected node:')
print(ge.text_encode_node(node_id))
print(ge.node_features(node_id))
print('Top {} similar nodes:'.format(k))
# top_k_nodes = memory.node_embedding[top_k_indices]
for idx in top_k_indices:
    print(ge.text_encode_node(idx))
    print(ge.node_features(idx))

print("end")
