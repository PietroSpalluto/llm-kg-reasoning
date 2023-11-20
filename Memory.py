from MakeTextEmbedding import MakeTextEmbedding
from MakeGraphEmbedding import MakeGraphEmbedding

import torch
from torch.nn import ConstantPad1d, CosineSimilarity


class Memory:
    """
    Memory contains tensors containing embeddings of textual description and graph embeddings
    """
    def __init__(self, make_text_emb: MakeTextEmbedding, make_graph_emb: MakeGraphEmbedding):
        """
        :param make_text_emb: MakeTextEmbedding object to embed the node description
        :param make_graph_emb: MakeGraphEmbedding object to embed the node
        """
        self.make_text_emb = make_text_emb
        self.make_graph_emb = make_graph_emb

        self.text_embedding = []  # embedding of the textual description
        self.graph_embedding = []  # embedding of the nodes
        self.node_embedding = []  # concatenation of textual embedding and node embedding

    def make_text_embeddings(self):
        """
        Add to Memory a 2-dimensional torch Tensor object containing the embeddings for each
        description
        """
        self.text_embedding = self.make_text_emb.compute_nodes_text_embedding()

    def make_graph_embedding(self):
        """
        Add to Memory a 2-dimensional torch Tensor object containing the embeddings for each
        node in the graph
        """
        self.graph_embedding = self.make_graph_emb.compute_graph_embedding()

    def make_node_embedding(self):
        """
        Concatenates the text embedding and the node features in a single tensor.
        """
        for text_emb, node_emb in zip(self.text_embedding, self.graph_embedding):
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

    def extract_top_k_similar_nodes(self, i, k: int, emb_type='node') -> tuple:
        """
        Computes the cosine similarity between the selected node and all the other nodes in the graph
        and return the k more similar
        :param i: selected node
        :param k: number of node to return
        :param emb_type: which embedding to use to compute similarity
                         -'text': text embedding of textual description
                         -'graph': graph embedding of the nodes
                         -'node': concatenation of text and graph embedding
        :return: tuple of two lists containing the top k similarities and the corresponding indices
        """
        selected_node = None
        embeddings = None
        if emb_type == 'node':
            selected_node = self.node_embedding[i]
            embeddings = self.node_embedding
        elif emb_type == 'text':
            selected_node = self.text_embedding[i]
            embeddings = self.text_embedding
        elif emb_type == 'graph':
            selected_node = self.graph_embedding[i]
            embeddings = self.graph_embedding

        similarity = CosineSimilarity(dim=0)
        similarities = []
        for emb in embeddings:
            similarities.append(similarity(selected_node, emb).item())

        # remove the first index since it is the selected node itself
        top_k_sim = torch.Tensor(similarities).topk(k+1)
        top_k_sim_values = top_k_sim.values.tolist()[1:]
        top_k_indices = top_k_sim.indices.tolist()[1:]

        return top_k_sim_values, top_k_indices
