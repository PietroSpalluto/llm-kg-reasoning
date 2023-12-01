import random

import pandas as pd
import networkx as nx


def stratified_sampling(graph_df: pd.DataFrame, perc: float, key: list) -> pd.DataFrame:
    """
    Performs stratified sampling of the input graph using as key the columns passed
    :param graph_df: graph in a DataFrame object
    :param perc: proportion to sample
    :param key: column(s) to use for the sampling
    :return: a DataFrame object containing the sampled graph
    """
    # graph_df = (graph_df.groupby(key, group_keys=False)
    #             .apply(lambda x: x.sample(max(int(len(x) * perc), 1))))
    graph_df = graph_df.groupby(key, group_keys=False).apply(lambda x: x.sample(frac=perc))
    return graph_df


def inductive_split(graph_df: pd.DataFrame, cols: list, train_split=0.75, val_split=0.11, test_split=0.11) -> tuple:
    """
    Split the graph for inductive training
    :param graph_df: DataFrame containing all the edges
    :param cols: list of column names in the form: [head, tail, relation]
    :param train_split: percentage of graph_df to assign to train graph
    :param val_split: percentage of inference graph to assign to validation graph
    :param test_split: percentage of inference graph to assign to test graph
    :return: a tuple containing 4 DataFrames representing 4 graphs (train, inference, validation and test)
    """
    # extract all the entities in the graph
    entities = list(set(pd.concat([graph_df[cols[0]], graph_df[cols[2]]]).to_list()))
    random.shuffle(entities)
    num_train_entities = int(len(entities) * train_split)

    # select train and inference entities
    train_entities = entities[:num_train_entities]
    inf_entities = entities[num_train_entities:]

    # make two disjoint sub-graphs
    training_triples = graph_df[(graph_df[cols[0]].isin(train_entities))
                                & (graph_df[cols[2]].isin(train_entities))]
    inference_triples = graph_df[(graph_df[cols[0]].isin(inf_entities))
                                 & (graph_df[cols[2]].isin(inf_entities))]

    # check if the set of relations in inference is a subset of the set of relations in training
    if not set(inference_triples[cols[1]]).issubset(set(training_triples[cols[1]])):
        # find the triples in inference that are not in training and remove them from inference_triples
        difference_triples = list(set(inference_triples[cols[1]]).difference(set(training_triples[cols[1]])))
        inference_triples = inference_triples.drop(
            inference_triples[inference_triples[cols[1]].isin(difference_triples)].index)

    # remove disconnected components from the training graph and keep the biggest one
    G = nx.from_pandas_edgelist(training_triples, source=cols[0], target=cols[2], edge_attr=cols[1])
    connected_components = sorted(nx.connected_components(G), key=len, reverse=True)[0]
    G = G.subgraph(connected_components)
    training_triples = nx.to_pandas_edgelist(G, source=cols[0], target=cols[2], edge_key=cols[1])
    training_triples = training_triples.sample(frac=1)  # shuffle triples

    # do de same for the inference graph
    G = nx.from_pandas_edgelist(inference_triples, source=cols[0], target=cols[2], edge_attr=cols[1])
    connected_components = sorted(nx.connected_components(G), key=len, reverse=True)[0]
    G = G.subgraph(connected_components)
    inference_triples = nx.to_pandas_edgelist(G, source=cols[0], target=cols[2], edge_key=cols[1])
    inference_triples = inference_triples.sample(frac=1)  # shuffle triples

    # sample triples from the inference graph such that they only contain nodes from the inference graph
    split1 = int(len(inference_triples) * val_split)
    split2 = int(len(inference_triples) * (val_split + test_split))
    validation_triples = inference_triples.iloc[:split1]
    test_triples = inference_triples.iloc[split1 + 1:split2]
    inference_triples = inference_triples.iloc[split2 + 1:]

    return training_triples[cols], inference_triples[cols], validation_triples[cols], test_triples[cols]


primeKG = pd.read_csv("data/primekg_no_inv_relations.tab",
                      usecols=['x_node_index', 'relation', 'display_relation', 'y_node_index'])
primeKG = primeKG.sample(frac=1)  # shuffle

print('{} unique relations'.format(len(primeKG[['relation', 'display_relation']].drop_duplicates())))

# # Stratified subsampling
# subsample_percent = 0.04
# primeKG = stratified_sampling(primeKG, subsample_percent, key=['relation', 'display_relation'])
# # Random subsampling
# num_triples = int(len(primeKG) * subsample_percent)
# primeKG = primeKG.sample(num_triples)

# merge relation and display_relation
rels = list(primeKG[['relation', 'display_relation']].itertuples(index=False, name=None))
rels = ['{} {}'.format(rel[0], rel[1]) for rel in rels]

primeKG['relation'] = rels
primeKG = primeKG.drop('display_relation', axis=1)

columns = ['x_node_index', 'relation', 'y_node_index']
training_triples, inference_triples, validation_triples, test_triples = inductive_split(primeKG, columns)

print('{} unique triples'.format(len(pd.concat([training_triples, inference_triples, validation_triples, test_triples])[['relation']].drop_duplicates())))
total_nodes = len(pd.concat([training_triples['x_node_index'], training_triples['y_node_index'],
                             inference_triples['x_node_index'], inference_triples['y_node_index'],
                             validation_triples['x_node_index'], validation_triples['y_node_index'],
                             test_triples['x_node_index'], test_triples['y_node_index']]).unique())
print('{} unique nodes'.format(total_nodes))
total_triples = len(training_triples) + len(inference_triples) + len(validation_triples) + len(test_triples)
print('{} triples in total'.format(total_triples))

training_triples.to_csv("data/PrimeKG_inductive_split/transductive_train.txt", header=None, index=None, sep='\t')
inference_triples.to_csv("data/PrimeKG_inductive_split/inference_graph.txt", header=None, index=None, sep='\t')
validation_triples.to_csv("data/PrimeKG_inductive_split/inf_valid.txt", header=None, index=None, sep='\t')
test_triples.to_csv("data/PrimeKG_inductive_split/inf_test.txt", header=None, index=None, sep='\t')
