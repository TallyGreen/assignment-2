import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, mixture
from sklearn.datasets import *
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage as scipy_linkage, fcluster
from itertools import cycle, islice
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage  #

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

"""
Part 4.	
Evaluation of Hierarchical Clustering over Diverse Datasets:
In this task, you will explore hierarchical clustering over different datasets. You will also evaluate different ways to merge clusters and good ways to find the cut-off point for breaking the dendrogram.
"""

# Fill these two functions with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_hierarchical_cluster(dataset, linkage, n_clusters):
    algorithm = cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    dataset= dataset[0]
    dataset= StandardScaler().fit_transform(dataset)
    algorithm.fit(dataset)
    y_pred = algorithm.labels_.astype(int)
    return y_pred

def fit_modified():
    return None


def compute():
    answers = {}

    """
    A.	Repeat parts 1.A and 1.B with hierarchical clustering. That is, write a function called fit_hierarchical_cluster (or something similar) that takes the dataset, the linkage type and the number of clusters, that trains an AgglomerativeClustering sklearn estimator and returns the label predictions. Apply the same standardization as in part 1.B. Use the default distance metric (euclidean) and the default linkage (ward).
    """
    nc = make_circles(n_samples=100, factor=0.5, noise=0.05, random_state=42)
    nm = make_moons(n_samples=100, noise=0.05, random_state=42)
    b = make_blobs(n_samples=100, random_state=42)
    bvv = make_blobs(n_samples=100, cluster_std=[1.0, 2.5, 0.5], random_state=42)
    #Anisotropicly distributed data
    X, y = make_blobs(n_samples=100, random_state=42)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    add = (X_aniso, y)

    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    # keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)
    dct = answers["4A: datasets"] = {"nc": [nc[0], nc[1]],
    "nm": [nm[0], nm[1]],
    "bvv": [bvv[0], bvv[1]],
    "add": [add[0], add[1]],
    "b": [b[0], b[1]],}

    # dct value:  the `fit_hierarchical_cluster` function
    dct = answers["4A: fit_hierarchical_cluster"] = fit_hierarchical_cluster

    """
    B.	Apply your function from 4.A and make a plot similar to 1.C with the four linkage types (single, complete, ward, centroid: rows in the figure), and use 2 clusters for all runs. Compare the results to problem 1, specifically, are there any datasets that are now correctly clustered that k-means could not handle?

    Create a pdf of the plots and return in your report. 
    """
    datasets = {
    "nc": nc,
    "nm": nm,
    "b": b,
    "bvv": bvv,
    "add": add}
    #Run clustering for each dataset with different linkage types
    linkage_types = ['single', 'complete', 'ward', 'average']
    num_clusters = 2

    # Plot
    fig, axs = plt.subplots(len(datasets), len(linkage_types), figsize=(15, 15))

    for i, (name, dataset) in enumerate(datasets.items()):
        for j, linkage in enumerate(linkage_types):
            # Perform hierarchical clustering
            y_pred = fit_hierarchical_cluster(dataset, linkage=linkage, n_clusters=num_clusters)

            # Plot the data with cluster assignments
            axs[i, j].scatter(dataset[0][:, 0], dataset[0][:, 1], c=y_pred, cmap='viridis', s=50, alpha=0.5)
            axs[i, j].set_title(f"{name.capitalize()} Dataset - {linkage.capitalize()} Linkage")

    plt.tight_layout()
    plt.show()

    # dct value: list of dataset abbreviations (see 1.C)
    dct = answers["4B: cluster successes"] = ["nc","nm"]

    """
    C.	There are essentially two main ways to find the cut-off point for breaking the diagram: specifying the number of clusters and specifying a maximum distance. The latter is challenging to optimize for without knowing and/or directly visualizing the dendrogram, however, sometimes simple heuristics can work well. The main idea is that since the merging of big clusters usually happens when distances increase, we can assume that a large distance change between clusters means that they should stay distinct. Modify the function from part 1.A to calculate a cut-off distance before classification. Specifically, estimate the cut-off distance as the maximum rate of change of the distance between successive cluster merges (you can use the scipy.hierarchy.linkage function to calculate the linkage matrix with distances). Apply this technique to all the datasets and make a plot similar to part 4.B.
    
    Create a pdf of the plots and return in your report. 
    """

    datasets = {
        "nc": nc,
        "nm": nm,
        "b": b,
        "bvv": bvv,
        "add": add
    }

    linkage_types = ['single', 'complete', 'ward', 'average']

    def find_optimal_cutoff(Z):
        distances = Z[:, 2]
        distance_diff = np.diff(distances)
        max_diff_idx = np.argmax(distance_diff)
        optimal_cutoff = distances[max_diff_idx]
        return optimal_cutoff

    def fit_hier_cluster_auto_cutoff(data, method):
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data[0])
        Z = scipy_linkage(data_scaled, method=method)
        cutoff = find_optimal_cutoff(Z)
        predicted_labels = fcluster(Z, cutoff, criterion='distance') - 1
        return predicted_labels

    auto_cutoff_predictions = []
    for method in linkage_types:
        predictions = [fit_hier_cluster_auto_cutoff(dataset, method) for dataset in datasets.values()]
        auto_cutoff_predictions.append(predictions)

    def plot_all_linkage_clusters(datasets, cluster_predictions, linkage_methods, title):
        num_linkages = len(linkage_methods)
        num_datasets = len(datasets)
        fig, axs = plt.subplots(num_linkages, num_datasets, figsize=(20, 16), sharex=True, sharey=True)
        fig.suptitle(title, fontsize=20)

        dataset_names = ['Noisy Circles', 'Noisy Moons', 'Varied', 'Anisotropic', 'Blobs']

        for i, linkage in enumerate(linkage_methods):
            for j, (dataset, pred_labels) in enumerate(zip(datasets.values(), cluster_predictions[i])):
                axs[i, j].scatter(dataset[0][:, 0], dataset[0][:, 1], c=pred_labels, s=10, cmap='viridis')
                if i == num_linkages - 1:
                    axs[i, j].set_xlabel(dataset_names[j])
                if j == 0:
                    axs[i, j].set_ylabel(linkage)
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    plot_all_linkage_clusters(datasets, auto_cutoff_predictions, linkage_types, 'Hierarchical Clustering')



    def fit_modified(data, method):
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data[0])
        Z = scipy_linkage(data_scaled, method=method)
        cutoff = find_optimal_cutoff(Z)
        predicted_labels = fcluster(Z, cutoff, criterion='distance') - 1
        return predicted_labels
    
    # dct is the function described above in 4.C
    dct = answers["4A: modified function"] = fit_modified

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part4.pkl", "wb") as f:
        pickle.dump(answers, f)
