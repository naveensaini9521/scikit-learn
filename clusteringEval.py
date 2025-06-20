from sklearn import metrics
labels_true = [0, 0, 0, 1, 1, 1]
labels_true = [0, 0, 0, 1, 1, 1]

metrics.rand_score(labels_true, labels_pred)

metrics.adjusted_rand_score(labels_true, labels_pred)

labels_pred = [1, 1, 0, 0, 3, 3]
metrics.rand_score(labels_true, labels_pred)

metrics.adjusted_rand_score(labels_pred, labels_true)

labels_pred = labels_true[:]
metrics.rand_score(labels_true, labels_pred)

labels_true = [0, 0, 0, 0, 0, 1, 1]
labels_pred = [0, 1, 2, 3, 4, 5, 6]

metrics.rand_score(labels_true, labels_pred)

metrics.adjusted_rand_score(labels_true, labels_pred)

metrics.adjusted_rand_score(labels_true, labels_pred)


# Homogeneity

labels_true = [0, 0, 0, 1, 1, 1]
labels_pred = [0, 0, 1, 1, 2, 2]

metrics.homogeneity_score(labels_true, labels_pred)

metrics.completeness_score(labels_true, labels_pred)

metrics.v_measure_score(labels_true, labels_pred)

metrics.v_measure_score(labels_true, labels_pred, beta=0.6)


metrics.homogeneity_completeness_v_measure(labels_true, labels_pred)

# Calinski-Harabasz Index
from sklearn.metrics import pairwise_distances
from sklearn import datasets
X, y = datasets.load_iris(return_X_y=True)

import numpy as np
from sklearn.cluster import KMeans
kmeans_model = KMeans(n_clusters=3, random_state=1).fit(X)
labels = kmeans_model.labels_
metrics.calinski_harabasz_score(X, labels)


# Davies-Bouldin Index

from sklearn import datasets
iris = datasets.load_iris()
X = iris.data

from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
kmeans = KMeans(n_clusters=3, random_state=1).fit(X)
labels = kmeans.labels_
davies_bouldin_score(X, labels)


# Contigency Matrix

from sklearn.metrics.cluster import contingency_matrix
x = ["a", "a", "a", "b", "b", "b"]
y = [0, 0, 1, 1, 2, 2]
contingency_matrix(x, y)


# Pair Confusion Matrix

from sklearn.metrics.cluster import pair_confusion_matrix
pair_confusion_matrix([0, 0, 1, 1], [0, 0, 1, 1])

