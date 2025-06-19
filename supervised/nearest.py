import numpy as np
from sklearn.neighbors import NearestNeighbors, KDTree, BallTree, NearestCentroid, KNeighborsTransformer, KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.datasets import load_iris, make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.manifold import Isomap
import tempfile

# NearestNeighbors 
X = np.array([[-1, -1], [2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)
print("Nearest Neighbors indices:\n",indices)
print("Distances:\n", distances)
print("KNN Graph:\n", nbrs.kneighbors_graph(X).toarray())


# KDTree and BallTree
kdt = KDTree(X, leaf_size=30, metric='euclidean')
print("KDTree nearest neighbors: \n", kdt.query(X, k=2, return_distance=False))

print("KDTree nearest neighbors:\n",KDTree.valid_metrics)
print("BAllTree valid matrics:\n",BallTree.valid_metrics)

y = np.array([1, 1, 1, 2, 2, 2])
clf = NearestCentroid()
clf.fit(X, y)
print(clf.predict([[-0.8, -1]]))


# Nearest Neighbors Transformer

cache_path = tempfile.gettempdir()
X_regr, _ = make_regression(n_samples=50, n_features=25, random_state=0)
estimator = make_pipeline(
    KNeighborsTransformer(mode='distance'),
    Isomap(n_components=3, metric='precomputed'),
    memory=cache_path)

X_embedded = estimator.fit_transform(X)
print(X_embedded.shape)

# NeighborhoodComponentsAnalysis + KNN on iris dataset
X_iris, y_iris = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, stratify=y, test_size=0.7, random_state=42)

# Classification
nca = NeighborhoodComponentsAnalysis(random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
nca_pipe = Pipeline([('nca', nca), ('knn', knn)])
nca_pipe.fit(X_train, y_train)
print("NCA + KNN accuracy on test set:\n ", nca_pipe.score(X_test, y_test))