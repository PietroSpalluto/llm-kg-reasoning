import torch


class MakeGraphEmbedding:
    """
    MakeGraphEmbedding computes a vector representation of a node
    """
    def __init__(self):
        # self.graph = torch.load("data/PrimeKG_pyg.pt")
        model_state = torch.load("data/ultra_4g.pth", map_location=torch.device('cpu'))
        # self.model = ...

    def get_embedding(self):
        """
        The features are generated in GraphEncoder and can be customized.
        """
        pass

    def compute_graph_embedding(self):
        pass
