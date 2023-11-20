import networkx as nx
import pandas as pd

from PrimeKG import PrimeKG


class GraphProcessor:
    """
    GraphProcessor returns a given node information. It supports all types of nodes in PrimeKG,
    but only nodes of type drug and disease have features.
    """
    def __init__(self, prime_kg: PrimeKG):
        """
        :param prime_kg: PrimeKG object containing data about PrimeKG dataset
        """
        self.prime_kg = prime_kg
        self.num_nodes = len(self.prime_kg.graph.nodes)  # number of nodes in the graph

    def get_node_features(self, i, node_type: str) -> tuple:
        """
        Gets node name, type and other features (only for drugs and diseases)
        :param i: real index (not sequential) of the node
        :param node_type: type (essential since some nodes have the same id but different types)
        :return: name, type and features
        """
        # node_index is a sequential id, node_id is the real id of the elements
        node_index = self.prime_kg.nodes[(self.prime_kg.nodes['node_id'] == i) &
                                         (self.prime_kg.nodes['node_type'] == node_type)]['node_index'].values[0]
        node_name = self.prime_kg.nodes[(self.prime_kg.nodes['node_id'] == i) &
                                        (self.prime_kg.nodes['node_type'] == node_type)]['node_name'].values[0]
        node_type = self.prime_kg.nodes[(self.prime_kg.nodes['node_id'] == i) &
                                        (self.prime_kg.nodes['node_type'] == node_type)]['node_type'].values[0]

        # features only have node_index, so we need to find the index given the node_id
        if node_type == 'drug':
            node_features = self.prime_kg.drug_features[self.prime_kg.drug_features['node_index'] == node_index]
        elif node_type == 'disease':
            node_features = self.prime_kg.disease_features[self.prime_kg.disease_features['node_index'] == node_index]
        else:
            node_features = pd.DataFrame([])

        return node_name, node_type, node_features
