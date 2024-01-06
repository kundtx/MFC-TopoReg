import seaborn as sn
from sklearn.cluster import KMeans
import numpy as np
import scipy.sparse as sp
import umap
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


def print_results(c, y, n_clusters=10):
    y_train_to_clustered = np.dstack([y, c])[0]
    clustered_tallies = np.zeros((n_clusters, n_clusters), dtype=int)
    for i in range(0, len(y_train_to_clustered)):
        clustered_tallies[y_train_to_clustered[i][1]][y_train_to_clustered[i][0]] += 1

    cluster_to_num_map = list(map(lambda x: np.argmax(x), clustered_tallies))
    clustered_tallies = sorted(clustered_tallies, key=lambda e: np.argmax(e))

    fig, ax = plt.subplots(1, figsize=(5, 5))
    p = sn.heatmap(clustered_tallies, annot=True, fmt="d", annot_kws={"size": 10}, cmap='coolwarm', ax=ax, square=True,
                   yticklabels=cluster_to_num_map)
    plt.xlabel('Actual')
    plt.ylabel('Cluster')
    p.tick_params(length=0)
    p.xaxis.tick_top()
    p.xaxis.set_label_position('top')
    plt.title('Cluster match count for each number')

    # purity - sum of correct in each class divided by the total number of images
    purity_sums = np.zeros((10, 1))

    for i in range(0, len(y_train_to_clustered[:])):
        if cluster_to_num_map[y_train_to_clustered[i][1]] == y[i]:
            purity_sums[cluster_to_num_map[y_train_to_clustered[i][0]]] += 1

    print('Purity ', np.add.reduce(purity_sums)[0] / len(y))
    plt.show()


def TSNE_visualize(encoded, color="k", centroids=None):
    reducer = TSNE(n_components=2,random_state=33,init="pca",learning_rate="auto")
    embedding = reducer.fit_transform(encoded)
    # centroids = reducer.fit_transform(centroids)
    fig = plt.figure(figsize = (5,5))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=color, s=20)
    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], c="k", marker="*")
    plt.title('TSNE projection of the dataset')
    plt.show()

def umap_visualize(encoded, color="k", centroids=None):
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(encoded)
    centroids = reducer.transform(centroids)
    fig = plt.figure(figsize = (5,5))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=color, s=20)
    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], c="k", marker="*")
    plt.title('UMAP projection of the dataset')
    plt.show()