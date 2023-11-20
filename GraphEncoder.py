import numpy as np
import pandas as pd

import networkx as nx

from GraphProcessor import GraphProcessor
from TextEncoder import TextEncoder

import torch


class GraphEncoder:
    """
    GraphEncoder can encode node into textual information. It uses node textual features and
    node neighbors that can be extracted from the graph
    """
    def __init__(self, graph_processor: GraphProcessor, k: int, n: int, func: str):
        """
        :param graph_processor:  GraphProcessor object to retrieve node information
        :param k: maximum number of hops to select neighbors
        :param n: number of nodes to sample if the sampling function is not 'all'
        :param func: sampling function to use
                     -'all': return all paths to a target node
                     -'random': return n random paths to a node (if n<1 it is considered as a fraction)
                     -'graph-agent': select the top n nodes for each hop where a node importance is:
                                     degree(N)/D_(avg, type(N)), where N is a neighbor node and
                                     D_(avg, type(N)) is the average degree for a category of nodes
        """
        self.graph_processor = graph_processor

        self.n = n
        self.k = k
        self.func = func

    def get_node(self, i: str, node_type: str) -> tuple:
        """
        Gets node information given the index of that node
        :param i: index of the node
        :param node_type: type of the node (it is necessary since some id are associated to more nodes)
        :return: node information
        """
        node_name, node_type, node_features = self.graph_processor.get_node_features(i, node_type)
        # node_class = self.graph_processor.get_nodes_classes()[i]

        return node_name, node_type, node_features

    def text_encode_node(self, i: str, node_type: str) -> str:
        """
        Use an encoding function to encode node information into a text
        :param i: index of the node
        :param node_type: type of the node (it is necessary since some id are associated to more nodes)
        :return: text encoded node
        """
        node_name, node_type, node_features = self.get_node(i, node_type)
        shortest_paths = self.sample_k_hops_neighbors(i, node_type, node_name)
        text_encoded_node = self.text_encoding(node_name,
                                               node_type,
                                               node_features,
                                               shortest_paths)

        return text_encoded_node

    def node_features(self, i: str, node_type: str) -> torch.Tensor:
        """
        Gets the features of the node generated in the GraphProcessor
        :param i: index of the node
        :param node_type: type of the node
        :return: tensor of node features
        """
        _, _, node_features = self.get_node(i, node_type)

        return node_features

    def sample_k_hops_neighbors(self, node_id, node_type: str, node_name: str) -> list:
        """
        Samples node in the k-hops neighborhood of a given node
        :param node_id: the index of the selected node
        :param node_type: type of the node (it is necessary since some id are associated to more nodes)
        :param node_name: name of the node (same as node type)
        :return: list of strings containing the shortest paths in the form:
                 head IS_TYPE head_type -> ... -> relation -> ... -> tail IS_TYPE tail type
        """

        # mapping from real id to sequential id used in the graph
        # name, id and type are necessary
        node_id = self.graph_processor.prime_kg.nodes[
            (self.graph_processor.prime_kg.nodes['node_id'] == node_id) &
            (self.graph_processor.prime_kg.nodes['node_type'] == node_type) &
            (self.graph_processor.prime_kg.nodes['node_name'] == node_name)]['node_index'].values[0]

        # find the shortest paths from node id to at most its k-hops neighbors
        shortest_paths = nx.single_source_shortest_path(self.graph_processor.prime_kg.graph,
                                                        node_id,
                                                        cutoff=self.k)

        shortest_paths = self.sampling_function(shortest_paths)

        shortest_paths = self.k_hops_paths(shortest_paths)

        return shortest_paths

    def k_hops_paths(self, shortest_paths: dict) -> list:
        """
        Returns a string for each path describing the path itself in the form:
        head IS_TYPE head_type -> ... -> relation -> ... -> tail IS_TYPE tail type
        :param shortest_paths: dict containing the target as key and a list of nodes in the path as value
        :return: a list of strings containing the paths
        """
        relations = []
        for k in list(shortest_paths.keys()):
            shortest_path = shortest_paths[k]
            if len(shortest_path) > 1:  # skip the 0-hops path
                path = ''
                for i in range(len(shortest_path) - 1):
                    source = shortest_path[i]
                    target = shortest_path[i + 1]
                    source_name = self.graph_processor.prime_kg.graph[source][target]['x_name']
                    source_type = self.graph_processor.prime_kg.graph[source][target]['x_type']
                    relation = self.graph_processor.prime_kg.graph[source][target]['relation']
                    display_relation = self.graph_processor.prime_kg.graph[source][target]['display_relation']
                    path = (path + source_name + ' IS_TYPE ' + source_type + ' -> ' +
                            relation + ' ' + display_relation)
                    # we avoid the repetition of the tail except for the last one
                    if i == len(shortest_path) - 2:
                        path = path + ' -> '
                        target_name = self.graph_processor.prime_kg.graph[source][target]['y_name']
                        target_type = self.graph_processor.prime_kg.graph[source][target]['y_type']
                        path = path + target_name + ' IS_TYPE ' + target_type

                    path = path + ' -> '

                path = path[:-4]  # remove last arrow
                relations.append(path)

        return relations

    def sampling_function(self, shortest_paths: dict) -> dict:
        """
        Selected nodes are sampled using a sampling function
        :param shortest_paths: neighbors of a node
        :return: a list of sampled shortest paths
        """
        if self.func == 'all':
            return shortest_paths
        elif self.func == 'random' and self.n < 1:
            n = int(len(shortest_paths) * self.n)
            idx = np.random.randint(0, len(shortest_paths), self.n)
            shortest_paths = {k: shortest_paths[k] for k in
                              [target for i, target in enumerate(shortest_paths) if i in idx]}
        elif self.func == 'random':
            idx = np.random.randint(0, len(shortest_paths), min(len(shortest_paths), self.n))
            shortest_paths = {k: shortest_paths[k] for k in
                              [target for i, target in enumerate(shortest_paths) if i in idx]}
        elif self.func == 'graph_agent':
            hops_dict = {}
            for i in range(self.k):
                hops_dict[i + 1] = {}  # the keys are the number of hops

            for target in shortest_paths:
                # save a node at the corresponding key
                if len(shortest_paths[target]) > 1:  # skip 0-hops
                    hop_number = len(shortest_paths[target]) - 1  # number of hops
                    neighbor_value = self.compute_node_value(target)  # value of the neighbor
                    # if there are less than k_top nodes in the dictionary at a hop_number then add
                    # a new node
                    if len(hops_dict[hop_number].keys()) < self.n:
                        hops_dict[hop_number][target] = neighbor_value
                    # if there are already k_top nodes remove check if the new node has a value higher
                    # the node with the smallest value, if True delete that node and ad the new one
                    else:
                        sort_dict = sorted(hops_dict[hop_number].items(), key=lambda x: x[1])
                        if not neighbor_value < sort_dict[0][1]:
                            del hops_dict[hop_number][sort_dict[0][0]]
                            hops_dict[hop_number][target] = neighbor_value

            # return a dict of selected paths
            selected_paths = {}
            for i in range(self.k):
                selected_paths.update(
                    {target: shortest_paths[target] for target in shortest_paths
                     if target in hops_dict[i + 1].keys()})

            shortest_paths = selected_paths

        return shortest_paths

    def compute_node_value(self, n_index: int) -> float:
        """
        Computes the node value given the node index using the formula:
        degree(n) / D_(avg, type(n))
        :param n_index: node index
        :return: node value
        """
        n_type = self.graph_processor.prime_kg.nodes[self.graph_processor.prime_kg.nodes['node_index']
                                                     == n_index]['node_type'].values[0]
        n_deg = self.graph_processor.prime_kg.graph.degree[n_index]
        avg_deg_type = self.graph_processor.prime_kg.avg_deg[n_type]
        value = n_deg / avg_deg_type
        return value

    def save_textual_nodes(self, id_list: list, name_list: list, node_type: str):
        """
        Saves node id description in a file
        :param id_list: list of nodes to save
        :param name_list: list of node names
        :param node_type: type of nodes to be saved
        """
        nodes = []
        for i, node_id in enumerate(id_list):
            print('encoding {}, {}/{}'.format(name_list[i], i, len(id_list)))
            nodes.append(self.text_encode_node(node_id, node_type))
            print('added {}, {}/{} nodes added'.format(node_id, i, len(id_list)))

        nodes = pd.DataFrame([id_list, name_list, nodes]).transpose()
        nodes.columns = ['id', 'name', 'description']
        nodes.to_csv("data/textual_nodes.csv", index=False)
        print('saved nodes')

    @staticmethod
    def text_encoding(node_name: str, node_type: str, node_features: pd.Series, shortest_paths: list) -> str:
        """
        Computes the natural language description of a node
        :param node_name: node name
        :param node_type: node type
        :param node_features: textual features of the node
        :param shortest_paths: shortest paths between a given node and some targets
        :return: a string containing a natural language description
        """
        te = TextEncoder(node_name, node_type, node_features, shortest_paths)
        text = te.encode()

        return text
