import pandas as pd
from torch_geometric.data import Data


def process(graph: pd.DataFrame) -> Data:
    pass


primeKG = pd.read_csv("data/primekg_no_inv_relations.tab",
                      usecols=['x_node_index', 'relation', 'display_relation', 'y_node_index'])
rels = list(primeKG[['relation', 'display_relation']].itertuples(index=False, name=None))
rels = ['{} {}'.format(rel[0], rel[1]) for rel in rels]

primeKG['relation'] = rels
primeKG = primeKG.drop('display_relation', axis=1)

# RELATION ENCODING

primeKG = process(primeKG)
