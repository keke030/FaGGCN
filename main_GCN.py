from model.GCNpackage import run_gcn
import pandas as pd

config = {
        'seed': 658,
        'k_folds': 5,
        'adj_matrix_path': 'data/BRCA/GAE0.95_ForReplication.csv',
        'sim_matrix_path': 'data/BRCA/SNF400_ForReplication.csv',
        'features_path': 'data/BRCA/SURVIVE_SELECT_ForReplication.csv',
        'clinical_path': 'data/BRCA/down_clinical.csv',
        'dropout': 0.4,
        'lr': 0.05,
        'weight_decay': 0.001,
        'epochs': 500,
        'label_map': {'LumA': 0, 'LumB': 1, 'Basal': 2, 'Her2': 3},
        'label_column': 'PAM50Call_RNAseq'
    }
fold_performance, accuracies, precision, recall, f1, ari, mcc = run_gcn(config)
fold_performance_df = pd.DataFrame(fold_performance)

fold_performance_df.to_csv('data/BRCA/fold_performance5.csv', index=False)
performance_metrics_df = pd.DataFrame({
        'accuracies': [x[0] for x in accuracies],
        'precision': [x[0] for x in precision],
        'recall': [x[0] for x in recall],
        'f1': [x[0] for x in f1],
        'ari': [x[0] for x in ari],
        'mcc': [x[0] for x in mcc]
    })
performance_metrics_df.to_csv('data/BRCA/performance_metrics5.csv', index=False)


config = {
        'seed': 567,
        'k_folds': 5,
        'adj_matrix_path': 'data/COAD/GAE0.95.csv',
        'sim_matrix_path': 'data/COAD/SNF400.csv',
        'features_path': 'data/COAD/SURVIVE_SELECT.csv',
        'clinical_path': 'data/COAD/clinical_241.csv',
        'dropout': 0.4,
        'lr': 0.05,
        'weight_decay': 0.001,
        'epochs': 500,
        'label_map': {'8140/3': 0, '8480/3': 1},
        'label_column': 'icd_o_3_histology'
    }
fold_performance, accuracies, precision, recall, f1, ari, mcc = run_gcn(config)
fold_performance_df = pd.DataFrame(fold_performance)

fold_performance_df.to_csv('data/COAD/fold_performance5.csv', index=False)
performance_metrics_df = pd.DataFrame({
        'accuracies': [x[0] for x in accuracies],
        'precision': [x[0] for x in precision],
        'recall': [x[0] for x in recall],
        'f1': [x[0] for x in f1],
        'ari': [x[0] for x in ari],
        'mcc': [x[0] for x in mcc]
    })
performance_metrics_df.to_csv('data/COAD/performance_metrics5.csv', index=False)