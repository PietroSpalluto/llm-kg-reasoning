import torch


class GraphProcessor:
    def __init__(self, graph):
        self.graph = graph
        self.num_nodes = graph.data.x.shape[0]

    def get_graph(self):
        return self.graph

    def get_nodes_features(self) -> torch.Tensor:
        return self.graph.data.x

    def get_nodes_classes(self) -> torch.Tensor:
        return self.graph.data.y
