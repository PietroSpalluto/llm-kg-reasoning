from GraphProcessor import GraphProcessor
from MakeEmbedding import MakeEmbedding

import torch


class GraphEncoder:
    def __init__(self, graph_processor: GraphProcessor):
        self.graph_processor = graph_processor
        self.graph = self.graph_processor.get_graph()

    def get_node(self, i: int) -> tuple:
        """
        Gets node information given the index of that node
        :param i: index of the node
        :return: node information
        """
        node_name = "node {}".format(i)
        node_features = self.graph_processor.get_nodes_features()[i]
        node_class = self.graph_processor.get_nodes_classes()[i]

        return node_name, node_features, node_class

    def text_encode_node(self, i: int) -> str:
        """
        Use an encoding function to encode node information into a text
        :param i: index of the node
        :return: text encoded node
        """
        node_name, _, node_class = self.get_node(i)
        node_neighbors = self.sample_k_hops_neighbors(i, 2)
        text_encoded_node = ('node name: {}, node class: {}, node neighbors: {}'
                             .format(node_name,
                                     node_class,
                                     node_neighbors.tolist()))

        return text_encoded_node

    def node_text_embedding(self, i: int) -> torch.Tensor:
        """
        Computes the text embedding of a given node
        :param i: index of the node
        :return: embedding of the node in a 1-dimensional tensor of length N*M where N is the number of words
                 and M is the embedding size
        """
        text_encoded_node = self.text_encode_node(i)

        me = MakeEmbedding()
        text_embedding = me.get_embedding(text_encoded_node)

        return torch.reshape(text_embedding, (-1,))

    def node_features(self, i: int) -> torch.Tensor:
        """
        Gets the features of the node generated in the GraphProcessor
        :param i: index of the node
        :return: tensor of node features
        """
        _, node_features, _ = self.get_node(i)

        return node_features

    def sample_k_hops_neighbors(self, node: int, k: int, n=30) -> torch.Tensor:
        """
        Samples node in the k-hops neighborhood of a given node
        :param node: the index of the selected node
        :param k: number of hops
        :param n: maximum number of nodes to sample
        :return: list containing neighbors with no repetitions
        """
        from torch_geometric.loader import NeighborLoader
        # mask with True in the position of the selected node
        mask = torch.Tensor([False] * self.graph_processor.num_nodes)
        mask[node] = True
        mask = mask.type("torch.BoolTensor")

        loader = NeighborLoader(
            self.graph.data,
            num_neighbors=[n] * k,  # number of samples to select and number of hops
            # Use a batch size of 128 for sampling training nodes
            batch_size=128,
            input_nodes=mask,
        )

        return next(iter(loader)).n_id
