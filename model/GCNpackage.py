# model/GCNpackage.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from scipy.sparse import csr_matrix
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, adjusted_rand_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GCN(nn.Module):
    def __init__(self, n_in, n_hid, n_out, dropout=None):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(n_in, n_hid)
        self.gc2 = GraphConvolution(n_hid, n_out)
        self.dropout = dropout

    def forward(self, input, adj):
        x = self.gc1(input, adj)
        x = F.elu(x)
        if self.dropout:
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1), x

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train(model, optimizer, features, adj, labels, idx_train):
    model.train()
    optimizer.zero_grad()
    output, _ = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    return loss_train.item()

def validate(model, features, adj, labels, idx_val):
    model.eval()
    with torch.no_grad():
        output, _ = model(features, adj)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        preds = output[idx_val].max(1)[1].type_as(labels)
        accuracy_val = accuracy_score(labels[idx_val].cpu(), preds.cpu())
        precision_val, recall_val, f1_val, _ = precision_recall_fscore_support(labels[idx_val].cpu(), preds.cpu(), average='weighted')
        _, _, f1_macro_val, _ = precision_recall_fscore_support(
            labels[idx_val].cpu(), preds.cpu(), average='macro')
        ari_val = adjusted_rand_score(labels[idx_val].cpu(), preds.cpu())
        mcc_val = matthews_corrcoef(labels[idx_val].cpu(), preds.cpu())
    return loss_val.item(), accuracy_val, precision_val, recall_val, f1_val,f1_macro_val, ari_val, mcc_val

def run_gcn(config):
    setup_seed(config['seed'])
    adj_matrix_df = pd.read_csv(config['adj_matrix_path'], sep=',', index_col=0)
    sim_matrix_df = pd.read_csv(config['sim_matrix_path'], sep=',', index_col=0)

    filtered_sim_values = sim_matrix_df.values[adj_matrix_df.values == 1]
    rows, cols = np.where(adj_matrix_df.values == 1)
    sparse_sim_matrix = csr_matrix((filtered_sim_values, (rows, cols)), shape=sim_matrix_df.shape)
    dense_matrix = sparse_sim_matrix.todense()
    dense_df = pd.DataFrame(dense_matrix, index=sim_matrix_df.index, columns=sim_matrix_df.columns)

    affinity_df_snf = dense_df
    features = pd.read_csv(config['features_path'], sep=',', index_col=0)
    clinical = pd.read_csv(config['clinical_path'], sep=',', index_col=0)

    features = features.reindex(affinity_df_snf.index)
    clinical = clinical.reindex(affinity_df_snf.index)

    label_map = config['label_map']
    labels = clinical[config['label_column']].map(label_map).values
    features = features.values
    adj = affinity_df_snf.values

    features = torch.tensor(features, dtype=torch.float32)
    adj = torch.tensor(adj, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    n_in, n_hid, n_out = features.shape[1], features.shape[1], features.shape[1]
    dropout = config['dropout']

    k_folds = config['k_folds']
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=config['seed'])

    fold_performance = []
    accuracies = []
    precision = []
    recall = []
    f1 = []
    ari = []
    mcc = []
    f1_macro = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(labels)), labels)):
        print(f"Starting fold {fold + 1}/{k_folds}")
        idx_train_fold = torch.tensor(train_idx, dtype=torch.long)
        idx_val_fold = torch.tensor(val_idx, dtype=torch.long)
        model = GCN(n_in=n_in, n_hid=n_hid, n_out=n_out, dropout=dropout)
        optimizer = Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        features = features.to(device)
        adj = adj.to(device)
        labels = labels.to(device)
        idx_train_fold = idx_train_fold.to(device)
        idx_val_fold = idx_val_fold.to(device)
        epochs = config['epochs']
        best_val_acc = 0
        epoch_at_best_acc = 0
        best_val_precision = 0
        epoch_at_best_precision = 0
        best_val_recall = 0
        epoch_at_best_recall = 0
        best_val_f1 = 0
        epoch_at_best_f1_macro = 0
        best_val_f1_macro = 0
        epoch_at_best_f1 = 0
        best_val_ari = 0
        epoch_at_best_ari = 0
        best_val_mcc = 0
        epoch_at_best_mcc = 0
        for epoch in range(epochs):
            loss_train = train(model, optimizer, features, adj, labels, idx_train_fold)
            loss_val, accuracy_val, precision_val, recall_val, f1_val, f1_macro_val, ari_val, mcc_val = validate(model, features, adj, labels, idx_val_fold)
            if accuracy_val > best_val_acc:
                best_val_acc = accuracy_val
                epoch_at_best_acc = epoch + 1
            if precision_val > best_val_precision:
                best_val_precision = precision_val
                epoch_at_best_precision = epoch + 1
            if recall_val > best_val_recall:
                best_val_recall = recall_val
                epoch_at_best_recall = epoch + 1
            if f1_val > best_val_f1:
                best_val_f1 = f1_val
                epoch_at_best_f1 = epoch + 1
            if f1_macro_val > best_val_f1_macro:
                best_val_f1_macro = f1_macro_val
                epoch_at_best_f1_macro = epoch + 1
            if ari_val > best_val_ari:
                best_val_ari = ari_val
                epoch_at_best_ari = epoch + 1
            if mcc_val > best_val_mcc:
                best_val_mcc = mcc_val
                epoch_at_best_mcc = epoch + 1
            if (epoch + 1) % 1 == 0:
                print(f"Fold {fold + 1}, Epoch {epoch + 1}: Loss Train {loss_train:.4f}, Loss Val {loss_val:.4f}, Acc Val {accuracy_val:.4f}, "
                      f"pre Val {precision_val:.4f}, recall Val {recall_val:.4f}, F1 Val {f1_val:.4f}, ari Val {ari_val:.4f}, mcc Val {mcc_val:.4f}")
                fold_performance.append({
                    'fold': fold + 1,
                    'epoch': epoch + 1,
                    'loss_train': loss_train,
                    'loss_val': loss_val,
                    'accuracy_val': accuracy_val,
                    'precision_val': precision_val,
                    'recall_val': recall_val,
                    'f1_val': f1_val,
                    'ari_val': ari_val,
                    'mcc_val': mcc_val
                })
        accuracies.append((best_val_acc, epoch_at_best_acc))
        precision.append((best_val_precision, epoch_at_best_precision))
        recall.append((best_val_recall, epoch_at_best_recall))
        f1.append((best_val_f1, epoch_at_best_f1))
        ari.append((best_val_ari, epoch_at_best_ari))
        f1_macro.append((best_val_f1_macro, epoch_at_best_f1_macro))
        mcc.append((best_val_mcc, epoch_at_best_mcc))

    # Calculate mean and standard deviation for all metrics
    # Only operate on the first value of each tuple (i.e., the performance value)
    mean_acc = np.mean([x[0] for x in accuracies])
    std_acc = np.std([x[0] for x in accuracies])

    mean_precision = np.mean([x[0] for x in precision])
    std_precision = np.std([x[0] for x in precision])

    mean_recall = np.mean([x[0] for x in recall])
    std_recall = np.std([x[0] for x in recall])

    mean_f1 = np.mean([x[0] for x in f1])
    std_f1 = np.std([x[0] for x in f1])

    mean_f1_macro = np.mean([x[0] for x in f1_macro])
    std_f1_macro = np.std([x[0] for x in f1_macro])

    mean_ari = np.mean([x[0] for x in ari])
    std_ari = np.std([x[0] for x in ari])

    mean_mcc = np.mean([x[0] for x in mcc])
    std_mcc = np.std([x[0] for x in mcc])

    # Print mean and std of all performance metrics
    print(f"Overall Performance: \n"
          f"Accuracy: {mean_acc:.2f} ± {std_acc:.2f}, \n"
          f"Precision: {mean_precision:.2f} ± {std_precision:.2f}, \n"
          f"Recall: {mean_recall:.2f} ± {std_recall:.2f}, \n"
          f"F1 Score: {mean_f1:.2f} ± {std_f1:.2f}, \n"
          f"Macro F1 Score: {mean_f1_macro:.2f} ± {std_f1_macro:.2f}, \n"
          f"ARI: {mean_ari:.2f} ± {std_ari:.2f}, \n"
          f"MCC: {mean_mcc:.2f} ± {std_mcc:.2f}")

    return fold_performance, accuracies, precision,recall,f1,ari, mcc

# Example configuration for running the model
config = {
        'seed': 658,
        'k_folds': 5,
        'adj_matrix_path': 'data/GAE0.95_ForReplication.csv',
        'sim_matrix_path': 'data/SNF400_ForReplication.csv',
        'features_path': 'data/SURVIVE_SELECT_ForReplication.csv',
        'clinical_path': 'data/down_clinical.csv',
        'dropout': 0.4,
        'lr': 0.05,
        'weight_decay': 0.001,
        'epochs': 500,
        'label_map': {'LumA': 0, 'LumB': 1, 'Basal': 2, 'Her2': 3},
        'label_column': 'PAM50Call_RNAseq'
    }


# To run the model from an external script
if __name__ == "__main__":
    run_gcn(config)

