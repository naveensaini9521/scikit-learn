import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.datasets import make_classification
from sklearn.datasets import make_circles, make_moons

X, y = make_blobs(centers=3, cluster_std=0.5, random_state=0)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Three normally-distributed clusters")
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True, sharex=True)
titles = ["Two classes,\none informative feature,\none cluster per class",
          "Two classes,\ntwo informative features,\ntwo clusters per class",
          "Three classes,\ntwo informative features,\none cluster per class"]
params = [
    {"n_informative": 1, "n_clusters_per_class": 1, "n_classes": 2},
    {"n_informative": 2, "n_clusters_per_class": 2, "n_classes": 2},
    {"n_informative": 2, "n_clusters_per_class": 1, "n_classes": 3}
]

for i, param in enumerate(params):
    X, Y = make_classification(n_features=2, n_redundant=0, random_state=1, **param)
    axs[i].scatter(X[:, 0], X[:, 1], c=Y)
    axs[i].set_title(titles[i])
    
plt.tight_layout()
plt.show()


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

X, Y = make_circles(noise=0.1, factor=0.3, random_state=0)
ax1.scatter(X[:, 0], X[:, 1], c=Y)
ax1.set_title("make_circles")

X, Y = make_moons(noise=0.1, random_state=0)
ax2.scatter(X[:, 0], X[:, 1], c=Y)
ax2.set_title("make_moons")

plt.tight_layout()
plt.show()