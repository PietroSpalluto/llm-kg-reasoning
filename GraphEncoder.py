from GraphProcessor import GraphProcessor

import torch


class GraphEncoder:
    def __init__(self, graph_processor: GraphProcessor):
        self.graph_processor = graph_processor
        self.graph = self.graph_processor.get_graph()

    def get_node(self, i):
        node_name = "node {}".format(i)
        node_features = self.graph_processor.get_nodes_features()[i]
        node_class = self.graph_processor.get_nodes_classes()[i]

        return node_name, node_features, node_class

    def text_encode_node(self, i):
        node_name, node_features, node_class = self.get_node(i)
        node_neighbors = self.sample_n_hops_neighbors(i, 2)
        text_encoded_node = ('node name: {}, node class: {}, node attributes: {}, node neighbors: {}'
                             .format(node_name,
                                     node_class,
                                     ', '.join(str(e) for e in node_features.tolist()),
                                     ', '.join(str(e) for e in node_neighbors.tolist())))

        return text_encoded_node

    def sample_n_hops_neighbors(self, node, n):
        """
        Samples node in the n-hops neighborhood of a given node
        :param node: the index of the selected node
        :param n: number of hops
        :return: list containing neighbors with no repetitions
        """
        from torch_geometric.loader import NeighborLoader
        # mask with True in the position of the selected node
        mask = torch.Tensor([False] * self.graph_processor.num_nodes)
        mask[node] = True
        mask = mask.type("torch.BoolTensor")

        loader = NeighborLoader(
            self.graph.data,
            num_neighbors=[30] * n,  # number of samples to select and number of hops
            # Use a batch size of 128 for sampling training nodes
            batch_size=128,
            input_nodes=mask,
        )

        return next(iter(loader)).n_id
