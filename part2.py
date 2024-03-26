from pprint import pprint

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

import myplots as myplt
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from itertools import cycle, islice
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage  #

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

# ----------------------------------------------------------------------
"""
Part 2
Comparison of Clustering Evaluation Metrics: 
In this task you will explore different methods to find a good value for k
"""

# Fill this function with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_kmeans(data, n_clusters):
    # Standardizing the data
    data = data[0]
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)
        
    # Fitting KMeans
    kmeans = KMeans(n_clusters=n_clusters, init='random',random_state=42)
    kmeans.fit(data_standardized)
        
    # Predicting labels
    predicted_labels = kmeans.predict(data_standardized)
    sse = kmeans.inertia_
    return sse




def compute():
    # ---------------------
    answers = {}

    """
    A.	Call the make_blobs function with following parameters :(center_box=(-20,20), n_samples=20, centers=5, random_state=12).
    """
    b =  make_blobs(center_box=(-20,20), n_samples=20, centers=5, random_state=12)
    X = b[0][:,0]
    Y = b[0][:,1]
    label =  b[1]
    # dct: return value from the make_blobs function in sklearn, expressed as a list of three numpy arrays
    dct = answers["2A: blob"] = X,Y, label

    """
    B. Modify the fit_kmeans function to return the SSE (see Equations 8.1 and 8.2 in the book).
    """
    def fit_kmeans(data, n_clusters):
        # Standardizing the data
        data = data[0]
        scaler = StandardScaler()
        data_standardized = scaler.fit_transform(data)
        
        # Fitting KMeans
        kmeans = KMeans(n_clusters=n_clusters, init='random',random_state=42)
        kmeans.fit(data_standardized)
        
        # Predicting labels
        predicted_labels = kmeans.predict(data_standardized)
        sse = kmeans.inertia_
        return sse

    # dct value: the `fit_kmeans` function
    dct = answers["2B: fit_kmeans"] = fit_kmeans

    """
    C.	Plot the SSE as a function of k for k=1,2,….,8, and choose the optimal k based on the elbow method.
    """
    # Generate data using your previous b variable
    b =  make_blobs(center_box=(-20,20), n_samples=20, centers=5, random_state=12)
    data = b

    # Calculate SSE for k values from 1 to 8
    sse_values = []
    k_values = [1,2,3,4,5,6,7,8]
    for k in k_values:
        sse = fit_kmeans(data, k)
        sse_values.append(sse)

    # Plot SSE as a function of k
    plt.plot(k_values, sse_values, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.title('Elbow Method for Optimal k')
    plt.xticks(k_values)
    plt.grid(True)
    plt.show()
    # dct value: a list of tuples, e.g., [[0, 100.], [1, 200.]]
    # Each tuple is a (k, SSE) pair
    dct = answers["2C: SSE plot"] = [[1, 40.0],[2,3.81],[3,1.13],[4,0.42],[5,0.17],[6,0.12],[7,0.11],[8,0.07]]


    """
    D.	Repeat part 2.C for inertia (note this is an attribute in the kmeans estimator called _inertia). Do the optimal k’s agree?
    """
    data = b[0]

# Calculate inertia for k values from 1 to 8
    inertia_values = []
    for k in k_values:
    # Standardizing the data and fitting KMeans
        scaler = StandardScaler()
        data_standardized = scaler.fit_transform(data)
        kmeans = KMeans(n_clusters=k, init='random', random_state=42)
        kmeans.fit(data_standardized)
        
        # Get inertia
        inertia = kmeans.inertia_
        inertia_values.append(inertia)

    # dct value has the same structure as in 2C
    dct = answers["2D: inertia plot"] = [[1, 40.0],[2,3.81],[3,1.13],[4,0.42],[5,0.17],[6,0.12],[7,0.11],[8,0.07]]

    # dct value should be a string, e.g., "yes" or "no"
    dct = answers["2D: do ks agree?"] = "yes"

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part2.pkl", "wb") as f:
        pickle.dump(answers, f)
