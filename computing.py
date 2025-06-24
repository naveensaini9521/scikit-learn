import sklearn
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import SGDRegressor
from sklearn import config_context
from joblib import parallel_backend
from sklearn.model_selection import cross_val_score

X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]

# 
with sklearn.config_context(assume_finite=True):
    pass

# Sparsity Ratio Function
def sparsity_ratio(X):
    return 1.0 - np.count_nonzero(X) / float(X.shape[0] * X.shape[1])
print("input sparsity ratio:", sparsity_ratio(X))

# # Limiting Working Memory
with sklearn.config_context(working_memory=128):
    pass

# Model Compression
clf = SGDRegressor(penalty='elasticnet', l1_ratio=0.25, random_state=42)
clf.fit(X_train, y_train).sparsify()
# clf.predict(X_test)
clf.sparsify()

with parallel_backend('threading', n_jobs=2):
    scores = cross_val_score(clf, X, y, cv=5)
    print("Cross validated scores:", scores)