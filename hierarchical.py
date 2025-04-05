import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles, make_classification
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist

class HierarchicalClustering:
    def __init__(self, dataset_type='blobs', n_samples=300, n_clusters=3, method='ward', seed=42):
        self.dataset_type = dataset_type
        self.n_samples = n_samples
        self.n_clusters = n_clusters
        self.method = method
        self.seed = seed
        self.X, self.y = self.load_data()
        self.labels = None
        self.linkage_matrix = None

    def load_data(self):
        dataset_options = {
            'blobs': make_blobs,
            'moons': make_moons,
            'circles': make_circles,
            'classification': make_classification
        }

        dataset_func = dataset_options[self.dataset_type]

        if self.dataset_type == 'blobs':
            return dataset_func(n_samples=self.n_samples, centers=self.n_clusters, random_state=self.seed)
        elif self.dataset_type == 'moons':
            return dataset_func(n_samples=self.n_samples, noise=0.1, random_state=self.seed)
        elif self.dataset_type == 'circles':
            return dataset_func(n_samples=self.n_samples, noise=0.1, factor=0.5, random_state=self.seed)
        elif self.dataset_type == 'classification':
            return dataset_func(n_samples=self.n_samples, n_features=2, n_informative=2, n_redundant=0, n_classes=self.n_clusters, n_clusters_per_class=1, random_state=self.seed)

    def fit(self):
        distance_matrix = pdist(self.X)
        self.linkage_matrix = linkage(distance_matrix, method=self.method)
        self.labels = fcluster(self.linkage_matrix, self.n_clusters, criterion='maxclust')

    def plot_clusters(self):
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.labels, cmap='viridis', alpha=0.6)
        plt.title(f'Hierarchical: {self.dataset_type.capitalize()}')
        plt.show()

    def plot_dendrogram(self):
        plt.figure(figsize=(10, 5))
        dendrogram(self.linkage_matrix)
        plt.title(f'Dendrogram: {self.dataset_type.capitalize()}')
        plt.xlabel('Samples')
        plt.ylabel('Distance')
        plt.show()

datasets = ['blobs', 'moons', 'circles', 'classification']

for dataset in datasets:
    print(f"Hierarchical Clustering: {dataset.capitalize()}")
    hc = HierarchicalClustering(dataset_type=dataset, n_samples=300, n_clusters=3)
    hc.fit()
    hc.plot_clusters()
    hc.plot_dendrogram()
