import pandas as pd
import pickle

import networkx as nx


class PrimeKG:
    def __init__(self):
        print('Loading data...')
        # NetworkX graph object
        self.graph = pickle.load(open('data/PrimeKG_nx.pkl', 'rb'))
        # dataframe containing nodes information
        self.nodes = pd.read_csv("data/nodes.csv")
        # dataframes containing drugs and diseases features
        self.disease_features = pd.read_csv("data/primekg_disease_feature.tab", sep='\t')
        self.drug_features = pd.read_csv("data/primekg_drug_feature.tab", sep='\t')
        print('Done.')
        # dataframe containing the average node degree for each type
        self.avg_deg = self.avg_degree_type()

        # self.kg = pd.read_csv("data/primekg.tab", dtype={'x_id': 'string', 'y_id': 'string'})

    def avg_degree_type(self) -> pd.Series:
        """
        Computes the average degree for each type of node in the graph
        :return: a pandas Series containing the average degree for each node
        """
        # find the degree for each node and merge this DataFrame with nodes to obtain the type of each node
        nodes_degree = pd.DataFrame([dict(self.graph.degree).keys(), dict(self.graph.degree).values()]).transpose()
        nodes_degree.columns = ['node_index', 'node_degree']
        nodes = (nodes_degree.merge(self.nodes, on='node_index')
                 .drop(['node_id', 'node_name', 'node_source', 'node_index'], axis=1))

        # group nodes by type and compute the average degree
        return nodes.groupby(by='node_type').mean()['node_degree']

    # def get_graph(self) -> nx.Graph:
    #     """
    #     :return: NetworkX Graph object
    #     """
    #     return self.graph
    #
    # def get_nodes(self) -> pd.DataFrame:
    #     """
    #     :return: Pandas DataFrame containing basic information on nodes
    #     """
    #     return self.nodes
    #
    # @staticmethod
    # def get_drug_features() -> pd.DataFrame:
    #     """
    #     :return: Pandas DataFrame containing drug features
    #     """
    #     return pd.read_csv("data/primekg_drug_feature.tab", sep='\t')
    #
    # @staticmethod
    # def get_disease_features() -> pd.DataFrame:
    #     """
    #     :return: Pandas DataFrame containing disease features
    #     """
    #     return pd.read_csv("data/primekg_disease_feature.tab", sep='\t')
    #
    # @staticmethod
    # def get_relations() -> pd.DataFrame:
    #     """
    #     :return: Pandas DataFrame containing detailed information on edges
    #     """
    #     return pd.read_csv("data/primekg.tab", dtype={'x_id': 'string', 'y_id': 'string'})
