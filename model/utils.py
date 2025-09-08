# model/utils.py

import pandas as pd
from sklearn.cluster import KMeans
from lifelines.statistics import multivariate_logrank_test
from sklearn import metrics


class ClusterProcessor:

    def __init__(self, data, sur_data):
        self.data = data
        self.sur_data = sur_data

    def KmeansCluster(self, nclusters):
        """
        Clusters the data using K-means algorithm.

        Parameters:
        - nclusters: the number of clusters to form.

        Returns:
        - An array of cluster labels.
        """
        K_mod = KMeans(n_clusters=nclusters)
        K_mod.fit(self.data)
        clusters = K_mod.predict(self.data)
        return clusters

    def LogRankp(self, nclusters):
        """
        Performs the Log-rank test for clustering quality evaluation.

        Parameters:
        - nclusters: the number of clusters to evaluate.

        Returns:
        - The p-value of the Log-rank test and an array of cluster labels.
        """
        clusters = self.KmeansCluster(nclusters)
        self.sur_data['Type'] = clusters
        pvalue = multivariate_logrank_test(self.sur_data['OS.time'], self.sur_data['Type'], self.sur_data['OS'])
        return pvalue, clusters

    def compute_indexes(self, maxclusters):
        """
        Computes and prints the clustering evaluation indexes for different cluster numbers.

        Parameters:
        - maxclusters: the maximum number of clusters to evaluate.
        """
        for i in range(2, maxclusters+1):
            pvalue, clusters = self.LogRankp(i)
            estimator = KMeans(n_clusters=i)
            estimator.fit(self.data)

            print("Number of clusters: ", i)
            print("Silhouette score: ", metrics.silhouette_score(self.data, estimator.labels_, metric='euclidean'))
            print("P-value: ", pvalue.p_value)


def do_km_plot(survive_data, pvalue, cindex, cancer_type, model_name):
    # import necessary packages
    from lifelines import KaplanMeierFitter
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    # extract relevant data
    values = np.asarray(survive_data['Type'])
    events = np.asarray(survive_data['OS'])
    times = np.asarray(survive_data['OS.time'])

    # set plotting style
    sns.set(style='ticks', context='notebook', font_scale=1.5)

    # create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # customize plot style
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)  # set thickness of x-axis line
    ax.spines['left'].set_linewidth(1.5)  # set thickness of y-axis line

    # fit and plot Kaplan-Meier survival curves for each cluster
    kaplan = KaplanMeierFitter()
    for label in set(values):
        kaplan.fit(times[values == label],
                   event_observed=events[values == label],
                   label='cluster {0}'.format(label))
        kaplan.plot_survival_function(ax=ax, ci_alpha=0)
        ax.legend(loc=1, frameon=False)

    # customize plot labels and title based on whether C-index was calculated or not
    if cindex == None:
        ax.set_xlabel('days', fontsize=20)
        ax.set_ylabel('Survival Probability', fontsize=20)
        ax.set_title('{1} \n Cancer: {0}    p-value.:{2: .1e} '.format(
            cancer_type, model_name, pvalue),
            fontsize=18,
            fontweight='bold')
    else:
        ax.set_xlabel('days', fontsize=20)
        ax.set_title('{1} \n Cancer: {0}  p-value.:{2: .1e}   Cindex: {3: .2f}'.format(
            cancer_type, model_name, pvalue, cindex),
            fontsize=18,
            fontweight='bold')

    # save plot as a .tiff file
    fig.savefig('./' + str(cancer_type) + model_name + '.tiff', dpi=300)


def cindex_test(sur_list, data_list):
    from lifelines import CoxPHFitter
    from sklearn.model_selection import train_test_split
    from lifelines.utils import concordance_index
    c_index_list = []
    cph = CoxPHFitter(penalizer=0.1)
    # The following lines need correct indentation
    x_train, x_test, y_train, y_test = train_test_split(data_list, sur_list, test_size=0.2, random_state=2022)
    sur_train = pd.concat([x_train, y_train], axis=1)
    sur_test = pd.concat([x_test, y_test], axis=1)
    # Drop column
    del sur_test['Type']
    # Fit model
    cph.fit(sur_train, "OS.time", "OS")
    c_index = concordance_index(sur_test['OS.time'], -cph.predict_partial_hazard(sur_test), sur_test['OS'])
    # Append the computed C-index to the list
    c_index_list.append(c_index)
    # Return result
    return c_index_list

def cindex_test1(sur_list, data_list):
    from lifelines import CoxPHFitter
    from sklearn.model_selection import train_test_split
    from lifelines.utils import concordance_index
    c_index_list = []
    cph = CoxPHFitter(penalizer=0.1)
    # The following lines need correct indentation
    x_train, x_test, y_train, y_test = train_test_split(data_list, sur_list, test_size=0.2, random_state=2022)
    sur_train = pd.concat([x_train, y_train], axis=1)
    sur_test = pd.concat([x_test, y_test], axis=1)
    # Fit model
    cph.fit(sur_train, "OS.time", "OS")
    c_index = concordance_index(sur_test['OS.time'], -cph.predict_partial_hazard(sur_test), sur_test['OS'])
    # Append the computed C-index to the list
    c_index_list.append(c_index)
    # Return result
    return c_index_list


def cindex_test_kfold(sur_list, data_list, n_splits=10):
    from lifelines import CoxPHFitter
    from sklearn.model_selection import KFold
    from lifelines.utils import concordance_index
    import numpy as np
    import pandas as pd  # Ensure pandas is correctly imported

    c_index_list = []
    cph = CoxPHFitter(penalizer=0.1)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=2024)

    for train_index, test_index in kf.split(data_list):
        # Split into training and testing sets
        x_train, x_test = data_list.iloc[train_index], data_list.iloc[test_index]
        y_train, y_test = sur_list.iloc[train_index], sur_list.iloc[test_index]

        # Combine train and test data
        sur_train = pd.concat([x_train, y_train], axis=1)
        sur_test = pd.concat([x_test, y_test], axis=1)

        # Drop 'Type' column in test set
        if 'Type' in sur_test.columns:
            del sur_test['Type']

        # Fit Cox model
        cph.fit(sur_train, "OS.time", "OS")

        # Compute C-index
        c_index = concordance_index(sur_test['OS.time'], -cph.predict_partial_hazard(sur_test), sur_test['OS'])
        c_index_list.append(c_index)

        # Print C-index for each fold
        print(f"Fold C-index: {c_index:.4f}")

    # Compute average and standard deviation of C-index
    mean_c_index = np.mean(c_index_list)
    std_c_index = np.std(c_index_list)

    # Print all C-indices and summary
    print(f"\nAll C-indices: {c_index_list}")
    print(f"Average C-index: {mean_c_index:.4f}")
    print(f"Standard Deviation of C-index: {std_c_index:.4f}")

    return c_index_list, mean_c_index, std_c_index
