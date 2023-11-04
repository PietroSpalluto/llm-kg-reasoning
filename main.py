import torch

from torch_geometric.datasets import Planetoid

import os.path as osp

from MakeEmbedding import MakeEmbedding
from GraphProcessor import GraphProcessor
from GraphEncoder import GraphEncoder

path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/Planetoid')
graph = Planetoid(path, name="Cora")

gp = GraphProcessor(graph)
ge = GraphEncoder(gp)

text_encoded_node = ge.text_encode_node(0)

string = "Hi, I am ugly"

me = MakeEmbedding()
embedding = me.get_embedding(string)

print("end")
