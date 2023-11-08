import os
import numpy as np
import pandas as pd

# NCBI to Reactome id and name
df_ncbi2reactome = pd.read_csv('../data/PrimeKG/reactome/NCBI2Reactome.txt',
                               sep='\t',
                               names=['ncbi_id', 'reactome_id', 'url', 'reactome_name',
                                      'evidence_code', 'species'])
df_ncbi2reactome = df_ncbi2reactome.query('species=="Homo sapiens"')
df_ncbi2reactome = df_ncbi2reactome.drop(['url', 'evidence_code', 'species'], axis=1)
df_ncbi2reactome = df_ncbi2reactome.reset_index().drop('index', axis=1).drop_duplicates()
df_ncbi2reactome.to_csv('../data/reactome/PrimeKG_processed/reactome_ncbi.csv', index=False)

# pathways
df_terms = pd.read_csv('../data/PrimeKG/reactome/ReactomePathways.txt',
                       sep='\t',
                       names=['reactome_id', 'reactome_name', 'species'])
df_terms = df_terms.query('species=="Homo sapiens"')
df_terms = df_terms.reset_index().drop('index', axis=1)
df_terms.to_csv('../data/PrimeKG_processed/reactome/reactome_terms.csv', index=False)

valid_terms = df_terms.get('reactome_id').values

# pathway-pathway (and pathway-protein(?)) interactions
df_rels = pd.read_csv('../data/PrimeKG/reactome/ReactomePathwaysRelation.txt',
                      sep='\t',
                      names=['reactome_id_1', 'reactome_id_2'])
df_rels = df_rels.query('reactome_id_1 in @valid_terms and reactome_id_2 in @valid_terms')
df_rels = df_rels.reset_index().drop('index', axis=1)
df_rels.to_csv('../data/PrimeKG_processed/reactome/reactome_relations.csv', index=False)
