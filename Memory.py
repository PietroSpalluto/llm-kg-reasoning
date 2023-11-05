from GraphEncoder import GraphEncoder

import torch
from torch.nn import ConstantPad1d, CosineSimilarity


class Memory:
    def __init__(self, graph_encoder: GraphEncoder):
        self.graph_encoder = graph_encoder

        self.text_embedding = []
        self.features_embedding = []
        self.node_embedding = []

    def make_text_embedding(self):
        """
        Makes the text embedding for each node and appends it to a list.
        The encoding is dona in GraphEncoder and can be custom.
        """
        # for node in range(self.graph_encoder.graph_processor.num_nodes):
        for node in range(5):
            self.text_embedding.append(self.graph_encoder.node_text_embedding(node))

        self.pad_text_embedding()

    def make_features_embedding(self):
        """
        Appends the features of a node to a list. The features are generated in GraphEncoder and
        can be customized.
        """
        # for node in range(self.graph_encoder.graph_processor.num_nodes):
        for node in range(5):
            self.features_embedding.append(self.graph_encoder.node_features(node))

    def make_node_embedding(self):
        """
        Concatenates the text embedding and the node features in a single tensor.
        """
        for text_emb, node_emb in zip(self.text_embedding, self.features_embedding):
            # merge textual embedding end node features
            node_embedding = torch.cat([text_emb, node_emb])

            self.node_embedding.append(node_embedding)

        # convert a list of tensors into a 2-dimensional tensor
        self.node_embedding = torch.stack(self.node_embedding)

    def pad_text_embedding(self):
        """
        Pads the text embeddings to make them all the same length
        """
        max_len = max([len(emb) for emb in self.text_embedding])
        for i in range(len(self.text_embedding)):
            diff = max_len - len(self.text_embedding[i])
            pad = ConstantPad1d((0, diff), 0)
            self.text_embedding[i] = pad(self.text_embedding[i])

    def extract_top_k_similar_nodes(self, i: int, k: int) -> list:
        """
        Computes the cosine similarity between the selected node and all the other nodes in the graph
        and return the k more similar
        :param i: selected node
        :param k: number of node to return
        :return: array of nodes that are more similar to the selected one
        """
        selected_node = self.node_embedding[i]
        similarity = CosineSimilarity(dim=0)
        similarities = []
        for emb in self.node_embedding:
            similarities.append(similarity(selected_node, emb).item())

        # remove the first index since it is the selected node itself
        top_k_indices = torch.tensor(similarities).topk(k+1).indices[1:].tolist()

        return top_k_indices
