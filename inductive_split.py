import random

import pandas as pd
import networkx as nx


def inductive_split(graph_df: pd.DataFrame, cols: list, train_split=0.75, val_split=0.11, test_split=0.11) -> tuple:
    """
    Split the graph for inductive training
    :param graph_df: DataFrame containing the edges
    :param cols: list of column names in the form: [head, tail, relation]
    :param train_split: percentage of graph_df to assign to train graph
    :param val_split: percentage of inference graph to assign to validation graph
    :param test_split: percentage of inference graph to assign to test graph
    :return: a tuple containing 4 DataFrames representing 4 graphs (train, inference, validation and test)
    """
    # extract all the entities in the graph
    entities = list(set(pd.concat([graph_df[cols[0]], graph_df[cols[1]]]).to_list()))
    random.shuffle(entities)
    num_train_entities = int(len(entities) * train_split)

    # select train and inference entities
    train_entities = entities[:num_train_entities]
    inf_entities = entities[num_train_entities:]

    # make two disjoint sub-graphs
    training_triples = graph_df[(graph_df[cols[0]].isin(train_entities)) & (graph_df[cols[1]].isin(train_entities))]
    inference_triples = graph_df[(graph_df[cols[0]].isin(inf_entities)) & (graph_df[cols[1]].isin(inf_entities))]

    # check if the set of relations in inference is a subset of the set of relations in training
    if not set(inference_triples[cols[2]]).issubset(set(training_triples[cols[2]])):
        # find the triples in inference that are not in training and remove them from inference_triples
        difference_triples = list(set(inference_triples[cols[2]]).difference(set(training_triples[cols[2]])))
        inference_triples = inference_triples.drop(
            inference_triples[inference_triples[cols[2]].isin(difference_triples)].index)

    # remove disconnected components from the training graph and keep the biggest one
    G = nx.from_pandas_edgelist(training_triples, source=cols[0], target=cols[1], edge_attr=cols[2])
    connected_components = sorted(nx.connected_components(G), key=len, reverse=True)[0]
    G = G.subgraph(connected_components)
    training_triples = nx.to_pandas_edgelist(G, source=cols[0], target=cols[1], edge_key=cols[2])

    # do de same for the inference graph
    G = nx.from_pandas_edgelist(inference_triples, source=cols[0], target=cols[1], edge_attr=cols[2])
    connected_components = sorted(nx.connected_components(G), key=len, reverse=True)[0]
    G = G.subgraph(connected_components)
    inference_triples = nx.to_pandas_edgelist(G, source=cols[0], target=cols[1], edge_key=cols[2])

    # sample triples from the inference graph such that they only contain nodes from the inference graph
    split1 = int(len(inference_triples) * val_split)
    split2 = int(len(inference_triples) * (val_split + test_split))
    validation_triples = inference_triples.iloc[:split1]
    test_triples = inference_triples.iloc[split1 + 1:split2]
    inference_triples = inference_triples.iloc[split2 + 1:]

    return training_triples, inference_triples, validation_triples, test_triples


primeKG = pd.read_csv("data/primekg_no_inv_relations.tab",
                      usecols=['x_node_index', 'relation', 'display_relation', 'y_node_index'])
# merge relation and display_relation
rels = list(primeKG[['relation', 'display_relation']].itertuples(index=False, name=None))
rels = ['{} {}'.format(rel[0], rel[1]) for rel in rels]

primeKG['relation'] = rels
primeKG = primeKG.drop('display_relation', axis=1)
columns = ['x_node_index', 'y_node_index', 'relation']
training_triples, inference_triples, validation_triples, test_triples = inductive_split(primeKG, columns)

training_triples.to_csv("data/PrimeKG_inductive_split/transductive_train.txt", header=None, index=None, sep='\t', mode='a')
inference_triples.to_csv("data/PrimeKG_inductive_split/inference_graph.txt", header=None, index=None, sep='\t', mode='a')
validation_triples.to_csv("data/PrimeKG_inductive_split/inf_valid.txt", header=None, index=None, sep='\t', mode='a')
test_triples.to_csv("data/PrimeKG_inductive_split/inf_test.txt", header=None, index=None, sep='\t', mode='a')
