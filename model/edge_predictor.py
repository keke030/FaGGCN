# model/edge_predictor.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# Define GCN layer
class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, activation=None, dropout=0.0):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = activation
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
    def forward(self, adj, features):
        if self.dropout:
            features = self.dropout(features)
        features = self.linear(features)
        features = torch.spmm(adj, features)
        if self.activation:
            features = self.activation(features)
        return features

# Define VGAE model
class VGAE(nn.Module):
    def __init__(self, input_feats, hidden_dims, output_dims, use_gae=False):
        super(VGAE, self).__init__()
        self.use_gae = use_gae
        self.gcn_base = GCNLayer(input_feats, hidden_dims, activation=F.relu)
        self.gcn_mean = GCNLayer(hidden_dims, output_dims, activation=None)
        self.gcn_logstd = GCNLayer(hidden_dims, output_dims, activation=None)
    def forward(self, adj, features):
        hidden = self.gcn_base(adj, features)
        mean = self.gcn_mean(adj, hidden)
        if self.use_gae:
            Z = mean
        else:
            logstd = self.gcn_logstd(adj, hidden)
            std = torch.exp(logstd)
            gaussian_noise = torch.randn_like(std)
            Z = gaussian_noise * std + mean
        adj_rec = torch.sigmoid(torch.mm(Z, Z.t()))
        return adj_rec, mean, logstd if not self.use_gae else None

# Train the model
def train(model, adj, features, optimizer, epochs=200):
    model.train()
    final_adj_rec = None  # Variable to store the final reconstructed adjacency matrix
    losses = []  # List to store the loss value at each epoch
    for epoch in range(epochs):
        optimizer.zero_grad()
        adj_rec, Z, logstd = model(adj, features)
        loss = F.binary_cross_entropy(adj_rec, adj)
        if not model.use_gae:
            kl_divergence = -0.5 / adj_rec.size(0) * torch.sum(1 + 2 * logstd - Z ** 2 - torch.exp(2 * logstd))
            loss += kl_divergence
        loss.backward()
        optimizer.step()
        # Save the loss of current epoch
        losses.append(loss.item())
        # Save adj_rec at the final epoch
        if epoch == epochs - 1:
            final_adj_rec = adj_rec.data
    return final_adj_rec, losses  # Return the final reconstructed adjacency matrix and the list of losses
