import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


class MakeTextEmbedding:
    """
    MakeTextEmbedding uses a pre-trained model to find a vector representation of a node description
    """
    def __init__(self, model_name):
        """
        :param model_name: name of the LLM used to compute the text embedding
        """
        self.textual_nodes = pd.read_csv("data/textual_nodes.csv")
        self.model = SentenceTransformer(model_name)

    def get_embedding(self, string: str) -> torch.Tensor:
        """
        Gets a description and computes an embedding.
        The textual encoding of the node is done in GraphEncoder and can be custom.
        :param string: natural language description of a node
        :return: a torch Tensor object containing a 1-dimensional embedding
        """
        embedding = self.model.encode(string, convert_to_tensor=True)

        return embedding

    def compute_nodes_text_embedding(self) -> torch.Tensor:
        """
        Computes the text embeddings of a list of nodes
        :return: a 2-dimensional torch Tensor object containing the embedding for each sentence
        """
        text_embeddings = []
        for i in range(len(self.textual_nodes)):
            node_name = self.textual_nodes['name'].iloc[i]
            print('Computing text embedding of {}, {}/{} done'.format(node_name,
                                                                      i, len(self.textual_nodes)))
            node_description = self.textual_nodes['description'].iloc[i]
            text_embedding = self.get_embedding(node_description)

            text_embeddings.append(text_embedding)

        text_embeddings = torch.stack(text_embeddings)
        return text_embeddings
