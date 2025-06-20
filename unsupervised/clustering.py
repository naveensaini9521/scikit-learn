from sklearn.cluster import SpectralBiclustering
from time import time
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets, manifold
from sklearn.cluster import AgglomerativeClustering

# sc = SpectralBiclustering(3, affinity='precomputed', n_init=100, assign_labels='discretize')

# sc.fit_predict(adjacency_matrix)

# Load digits data
digits = datasets.load_digits()
X, y = digits.data, digits.target
n_samples, n_features = X.shape

np.random.seed(0)

# Visualize the clustering 
def plot_clustering(X_red, labels, title=None):
    x_main, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
    
    plt.figure(figsize=(6,4))
    for digit in digits.target_names:
        plt.scatter(
            *X_red[y == digit].T,
            marker=f"${digit}$",
            s=50,
            c=plt.cm.nipy_spectral(labels[y ==digit]/ 10),
            alpha=0.5,
        )
    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, size=17)
    plt.axis("off")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    
# 2D embedding of the digits dataset
print("Computing embedding")
X_red = manifold.SpectralEmbedding(n_components=2).fit_transform(X)
print("Done")


for linkage in ("ward", "average", "complete", "single"):
    print(f"Fitting {linkage} linkage")
    t0 = time()
    
    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=10)
    clustering.fit(X_red)
    print("%s :\t%.2fs"% (linkage, time() - t0))
    
    plot_clustering(X_red, clustering.labels_, "%s linkage" % linkage)
    
plt.show()