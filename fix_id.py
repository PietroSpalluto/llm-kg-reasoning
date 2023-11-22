import pandas as pd
import networkx as nx
from torch_geometric.utils.convert import from_networkx
import pickle
import torch


def get_mapping():
    mapping = ['interacts with the following protein',
               'has the following protein as carrier (passive transport)',
               'has the following protein as enzyme',
               'has the following protein as target',
               'has the following protein as transporter (active transport)',
               'is a contraindication for the following disease',
               'is indicated for the following disease',
               'is an off-label use for the following disease',
               'has a synergistic interaction with the following drug',
               'is associated with the following effect or phenotype',
               'is in a parent-child relations with the following effect or phenotype',
               'has the absence of the following phenotype',
               'has the presence of the following phenotype',
               'is present in the following disease',
               'is associated with the following disease',
               'is in a parent-child relation with the following disease',
               'has the following side effect',
               'is in a parent-child relation with the following biological process',
               'is in a parent-child relation with the following molecular function',
               'is in a parent-child relation with the following cellular component',
               'interacts with the following molecular function',
               'interacts with the following cellular component',
               'interacts with the following biological process',
               'interacts with the following gene or protein',
               'is an exposure of the following disease',
               'is in a parent-child relation with the following exposure',
               'interacts with the following biological process',
               'interacts with the following molecular function',
               'interacts with the following cellular component',
               'is in a parent-child relation with the following pathway',
               'interacts with the following pathway',
               'is in a parent-child relation with the following anatomical component',
               'has an expression that is present (so it synthesizes a protein) for the following anatomical component',
               'has an expression that is absent (so it does not synthesizes a protein) for the following anatomical component',
               'is a carrier (so a passive transporter) for this drug',
               'is an enzyme for the following drug',
               'is a target of the following drug',
               'is a transporter (so an active transporter) for the following drug',
               'has the following drug as a contraindication', 'has the following drug as a indication',
               'has the following drug as a off-label use',
               'is associated with the following protein or gene',
               'is absent in the following disease',
               'is associated with the following gene or protein',
               'is a side effect of the following drug',
               'interacts with the following gene of protein',
               'interacts with the following gene of protein',
               'interacts with the following gene of protein',
               'interacts with the following exposure',
               'is linked to the following exposure',
               'interacts with the following exposure',
               'interacts with the following exposure',
               'interacts with the following exposure',
               'interacts with the following gene or protein',
               'the following gene do synthesize proteins for the following anatomical part',
               'the following gene do not synthesize proteins for the following anatomical part']

    return mapping


# used to make a sequential_id column that is unique (x_id and y_id have mixed types)
kg = pd.read_csv("data/old_primekg.tab", dtype={'x_id': 'string', 'y_id': 'string'})

# remove space in relation and display relation for easier encoding in TextEncoder
kg['relation'] = kg['relation'].apply(lambda x: x.replace(' ', '_'))
kg['display_relation'] = kg['display_relation'].apply(lambda x: x.replace(' ', '_'))

# make a relations DataFrame to make a mapping from relation to natural language
relations = kg[['x_type', 'relation', 'display_relation', 'y_type']].drop_duplicates()
relations['mapping'] = get_mapping()
relations.to_csv("data/relations_mapping.csv")

nodes = pd.read_csv("data/nodes.csv")

# add to the KG the columns node_index from nodes, some ids are used for more nodes but the
# sequential id in the nodes DataFrame is unique
kg = kg.merge(nodes, left_on=['x_id', 'x_type', 'x_name'],
              right_on=['node_id', 'node_type', 'node_name'],
              how='left')
kg = kg.drop(['node_id', 'node_name', 'node_type'], axis=1)
kg = kg.rename(columns={'node_index': 'x_node_index'})

kg = kg.merge(nodes, left_on=['y_id', 'y_type', 'y_name'],
              right_on=['node_id', 'node_type', 'node_name'],
              how='left')
kg = kg.drop(['node_id', 'node_name', 'node_type'], axis=1)
kg = kg.rename(columns={'node_index': 'y_node_index'})

kg.to_csv("data/primekg.tab", index=False)

# make a graph from the fixed KG
graph_nx = nx.from_pandas_edgelist(kg,
                                   source='x_node_index',
                                   target='y_node_index',
                                   # we encode in the edge all the information about nodes and edge
                                   edge_attr=['x_name', 'x_type', 'relation', 'display_relation', 'y_type', 'y_name'],
                                   create_using=nx.DiGraph)
with open("data/PrimeKG_nx.pkl", "wb") as f:
    pickle.dump(graph_nx, f)

graph_pyg = from_networkx(graph_nx)
torch.save(graph_pyg, "data/PrimeKG_pyg.pt")
