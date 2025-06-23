from sklearn import ensemble
from sklearn import datasets

clf = ensemble.HistGradientBoostingClassifier()
X, y = datasets.load_iris(return_X_y=True)
clf.fit(X, y)