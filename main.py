import numpy as np
import pandas as pd

from PrimeKG import PrimeKG
from GraphProcessor import GraphProcessor
from GraphEncoder import GraphEncoder
from MakeTextEmbedding import MakeTextEmbedding
from MakeGraphEmbedding import MakeGraphEmbedding
from Memory import Memory

import os


def random_disease():
    nodes = pd.read_csv("data/nodes.csv")
    nodes = nodes[nodes['node_type'] == 'disease']
    selected_id = np.random.randint(0, len(nodes))
    node_idx = nodes.iloc[selected_id]['node_id']

    return node_idx, 'disease'


def random_drug():
    nodes = pd.read_csv("data/nodes.csv")
    nodes = nodes[nodes['node_type'] == 'drug']
    selected_id = np.random.randint(0, len(nodes))
    node_idx = nodes.iloc[selected_id]['node_id']

    return node_idx, 'drug'


def get_diseases():
    nodes = pd.read_csv("data/nodes.csv")
    disease = pd.read_csv("data/primekg_disease_feature.tab", sep='\t')
    id_name = disease.merge(nodes, left_on='node_index', right_on='node_index')[['node_id', 'node_name']]
    nodes_idx = id_name['node_id'].to_list()
    nodes_name = id_name['node_name'].to_list()

    return nodes_idx, nodes_name


def get_drugs():
    nodes = pd.read_csv("data/nodes.csv")
    drug = pd.read_csv("data/primekg_drug_feature.tab", sep='\t')
    id_name = drug.merge(nodes, left_on='node_index', right_on='node_index')[['node_id', 'node_name']]
    nodes_idx = id_name['node_id'].to_list()
    nodes_name = id_name['node_name'].to_list()

    return nodes_idx, nodes_name


k = 3
n = 10
sampling_function = 'graph_agent'
if not os.path.exists("data/textual_nodes.csv"):
    primeKG = PrimeKG()
    gp = GraphProcessor(primeKG)
    ge = GraphEncoder(gp, k, n, sampling_function)

    disease_id, disease_name = get_diseases()
    ge.save_textual_nodes(disease_id[:1000], disease_name[:1000], node_type='disease')

    # drug_id, drug_names = get_drugs()
    # ge.save_textual_nodes(drug_id[:6], drug_names[:6], node_type='drug')

    # del drug_id, drug_names, gp, ge, primeKG
    del disease_id, disease_name, gp, ge, primeKG

# sentence transformer models: all-mpnet-base-v2 (SOTA), all-MiniLM-L6-v2
# transformer models: bert-base-uncased, bert-large-uncased
make_text_emb = MakeTextEmbedding("all-mpnet-base-v2")
make_graph_emb = MakeGraphEmbedding()
memory = Memory(make_text_emb, make_graph_emb)
memory.make_text_embeddings()

memory.pad_text_embedding()

selected = 2  # selected node id (not the real id but a sequential id inside the textual dataset)
k = 5  # number of similar node to extract

top_k_sim_values, top_k_indices = memory.extract_top_k_similar_nodes(selected, k, emb_type='text')
# selected node
selected_node_id = make_text_emb.textual_nodes.iloc[selected]['id']
selected_node_name = make_text_emb.textual_nodes.iloc[selected]['name']
selected_node_description = make_text_emb.textual_nodes.iloc[selected]['description']
# top k similar nodes
similar_nodes_id = make_text_emb.textual_nodes.iloc[top_k_indices]['id'].values.tolist()
similar_nodes_name = make_text_emb.textual_nodes.iloc[top_k_indices]['name'].values.tolist()
similar_nodes_descriptions = make_text_emb.textual_nodes.iloc[top_k_indices]['description'].values.tolist()

print("end")
