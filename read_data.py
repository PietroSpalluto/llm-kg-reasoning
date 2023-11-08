import pandas as pd

from torch_geometric.data import HeteroData

diseases = pd.read_csv("data/dataverse_files/disease_features.csv")
drugs = pd.read_csv("data/dataverse_files/drug_features.csv")
genes_proteins = pd.read_csv("data/hgnc_complete_set_2023-10-01.tsv", sep='\t')
gene_ontology = pd.read_csv("data/gene_ontology.tsv", sep='\t')
ctd_exposures = pd.read_csv("data/CTD__exposure_20231105134810.csv")
nodes = pd.read_csv("data/dataverse_files/nodes.csv")

# obtain features merging nodes of type gene/protein with data containing gene/protein features
genes_proteins_features = (nodes[nodes['node_type'] == 'gene/protein']
                           .merge(genes_proteins,
                                  left_on="node_name",
                                  right_on="symbol",
                                  how="left"))
genes_proteins_features.drop(["node_id", "node_type", "node_name", "node_source"],
                             axis=1,
                             inplace=True)

# use gene ontology to add features to node of types biological_process, molecular_function
# and cellular_component
biological_processes_features = (nodes[nodes['node_type'] == 'biological_process']
                                 .merge(gene_ontology[gene_ontology['ontology source'] == 'biological_process'],
                                        left_on='node_name',
                                        right_on='term',
                                        how='left'))
biological_processes_features.drop(["node_id", "node_type", "node_name", "node_source"],
                                   axis=1,
                                   inplace=True)

molecular_functions_features = (nodes[nodes['node_type'] == 'molecular_function']
                                .merge(gene_ontology[gene_ontology['ontology source'] == 'molecular_function'],
                                       left_on='node_name',
                                       right_on='term',
                                       how='left'))
molecular_functions_features.drop(["node_id", "node_type", "node_name", "node_source"],
                                  axis=1,
                                  inplace=True)

cellular_components_features = (nodes[nodes['node_type'] == 'cellular_component']
                                .merge(gene_ontology[gene_ontology['ontology source'] == 'cellular_component'],
                                       left_on='node_name',
                                       right_on='term',
                                       how='left'))
cellular_components_features.drop(["node_id", "node_type", "node_name", "node_source"],
                                  axis=1,
                                  inplace=True)

# DOWNLOAD CTD to extract exposure features
# DOWNLOAD Reactome to extract pathway features
# DOWNLOAD UBERON to extract anatomy features

# edges = pd.read_csv("data/dataverse_files/edges.csv")
# kg = pd.read_csv("data/dataverse_files/kg.csv")
# kg = pd.read_csv("data/dataverse_files/kg_giant.csv")
# kg = pd.read_csv("data/dataverse_files/kg_grouped.csv")
# kg = pd.read_csv("data/dataverse_files/kg_grouped_diseases.csv")
# kg = pd.read_csv("data/dataverse_files/kg_grouped_diseases_bert_map.csv")
# kg = pd.read_csv("data/dataverse_files/kg_raw.csv")

# graph = HeteroData()
#
# graph['disease'].x = diseases.to_numpy()
# graph['drug'].x = drugs.to_numpy()

print("end")
