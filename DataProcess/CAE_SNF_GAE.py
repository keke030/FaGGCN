import pandas as pd
import os
import random
import warnings
import torch
import torch.optim as optim
import numpy as np
import snf

# Set seeds and suppress warnings
random.seed(203)
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load omics and survival data
survive = pd.read_csv('data/down_survive.csv', sep=',').set_index('Unnamed: 0')
RNA = pd.read_csv('data/RNA.csv', sep=',').set_index('Unnamed: 0')
CNV = pd.read_csv('data/CNV.csv', sep=',').set_index('Unnamed: 0')
SNV = pd.read_csv('data/SNV.csv', sep=',').set_index('Unnamed: 0')
MIRNA = pd.read_csv('data/MIRNA.csv', sep=',').set_index('Unnamed: 0')
clinical = pd.read_csv('data/down_clinical.csv', sep=',', index_col=0)

# Import modules
from model import myCAE
from model.Survive_select import survive_select
from model.edge_predictor import VGAE, train

# Initialize CAE models
CNV_model = myCAE.CAE(CNV.shape[1])
RNA_model = myCAE.CAE(RNA.shape[1])
MIRNA_model = myCAE.CAE(MIRNA.shape[1])
SNV_model = myCAE.CAE(SNV.shape[1])

# Fit models and extract features
CNV_model.fit(CNV)
RNA_model.fit(RNA)
MIRNA_model.fit(MIRNA)
SNV_model.fit(SNV)
CNV_feature = CNV_model.extract_feature(CNV)
RNA_feature = RNA_model.extract_feature(RNA)
MIRNA_feature = MIRNA_model.extract_feature(MIRNA)
SNV_feature = SNV_model.extract_feature(SNV)

# Concatenate features from all omics
flatten = pd.concat([
    pd.DataFrame(CNV_feature),
    pd.DataFrame(RNA_feature),
    pd.DataFrame(MIRNA_feature),
    pd.DataFrame(SNV_feature)
], axis=1)
flatten.index = survive.index

# Remove zero-only and low-variance features
cleaned_flatten = flatten.loc[:, (flatten != 0).any(axis=0)]
variances = cleaned_flatten.var()
columns_to_drop = variances[variances < 0.05].index
cleaned_flatten_filtered = cleaned_flatten.drop(columns=columns_to_drop)

# Perform survival-based feature selection
SURVIVE_SELECT = survive_select(survive, cleaned_flatten, 0.05)
SURVIVE_SELECT.index = survive.index
SURVIVE_SELECT.to_csv('data/SURVIVE_SELECT.csv', index=True, header=True)

# Save individual CAE features
pd.DataFrame(CNV_feature, index=survive.index).to_csv('data/CNV_CAE100.csv', index=True, header=True)
pd.DataFrame(RNA_feature, index=survive.index).to_csv('data/RNA_CAE100.csv', index=True, header=True)
pd.DataFrame(MIRNA_feature, index=survive.index).to_csv('data/MIRNA_CAE100.csv', index=True, header=True)
pd.DataFrame(SNV_feature, index=survive.index).to_csv('data/SNV_CAE100.csv', index=True, header=True)

# Load saved features
rna_data = pd.read_csv('data/RNA_CAE100.csv', sep=',').set_index('Unnamed: 0')
cnv_data = pd.read_csv('data/CNV_CAE100.csv', sep=',').set_index('Unnamed: 0')
snv_data = pd.read_csv('data/SNV_CAE100.csv', sep=',').set_index('Unnamed: 0')
mirna_data = pd.read_csv('data/MIRNA_CAE100.csv', sep=',').set_index('Unnamed: 0')

# Rename index for consistency
for df in [rna_data, cnv_data, snv_data, mirna_data]:
    df.index.name = 'Sample'

# Build affinity networks using SNF
affinity_nets = snf.make_affinity([
    rna_data.values.astype(float),
    cnv_data.values.astype(float),
    snv_data.values.astype(float),
    mirna_data.values.astype(float)
], metric='euclidean', K=20, mu=0.5)

# Fuse affinity networks into a single matrix
fused_net = snf.snf(affinity_nets, K=20)
fused_df = pd.DataFrame(fused_net, index=rna_data.index, columns=rna_data.index)
fused_df.to_csv('data/SNF400.csv', header=True, index=True)

# Load fused SNF graph and selected features
affinity_df_snf = pd.read_csv('data/SNF400.csv', sep=',', index_col=0)
features_df = pd.read_csv('data/SURVIVE_SELECT.csv', sep=',', index_col=0)
features_df = features_df.reindex(affinity_df_snf.index)

# Prepare data for VGAE
features = torch.FloatTensor(features_df.values)
adjacency_matrix = torch.FloatTensor(affinity_df_snf.values)

# Initialize VGAE model
model = VGAE(input_feats=features.size(1), hidden_dims=32, output_dims=16, use_gae=False)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train VGAE model
final_adjacency_matrix, training_losses = train(model, adjacency_matrix, features, optimizer)

# Save reconstructed adjacency matrix
final_adjacency_matrix_np = final_adjacency_matrix.numpy()
final_adjacency_df = pd.DataFrame(final_adjacency_matrix_np,
                                   index=affinity_df_snf.index,
                                   columns=affinity_df_snf.columns)
final_adjacency_df.to_csv('data/GAE_adjacency.csv', header=True, index=True)

# Binarize the adjacency matrix (threshold = 0.95)
binary_adjacency_df = (final_adjacency_df > 0.95).astype(int)

# Compute number of edges
degree_series = binary_adjacency_df.sum()
n_edges = np.tril(binary_adjacency_df).sum().sum()

# Save binary adjacency matrix
binary_adjacency_df.to_csv('data/GAE0.95.csv')