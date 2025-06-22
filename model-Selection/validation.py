import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, metrics
from sklearn.model_selection import cross_val_score

# Load iris dataset
X, y = datasets.load_iris(return_X_y=True)
print(X.shape, y.shape)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
print("Train shape:", X_train.shape, y_train.shape)

# Fit linear SVM
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
print("Test accuracy", clf.score(X_test, y_test))

# Computing Cross-validated metrics
clf = svm.SVC(kernel='linear', C=1, random_state=42)
scores = cross_val_score(clf, X, y, cv=5)
print("Cross-validated accuracy scores", scores)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

# Cross-validated F1 macro score
scores1 = cross_val_score(clf, X, y, cv=5, scoring='f1_macro')
print("F1 macro scores:", scores1)

# The cross_validate function and multiple metric evaluation
from sklearn .model_selection import cross_validate
from sklearn.metrics import recall_score
from sklearn.svm import SVC

clf = svm.SVC(kernel='linear', C=1, random_state=0)
scoring = ['precision_macro', "recall_macro"]
scores = cross_validate(clf, X, y, scoring=scoring, cv=5, return_train_score=False)
print(sorted(scores.keys()))

print("Precision scores",scores['test_recall_macro'])

from sklearn.metrics import make_scorer
scoring = {'prec_macro': 'precision_macro',
           'rec_macro': make_scorer(recall_score, average='macro')}
scores = cross_validate(clf, X, y, scoring=scoring,
                        cv=5, return_train_score=True)

sorted(scores.keys())

scores['train_rec_macro']

# K-fold
from sklearn.model_selection import KFold

X = ["a", "b", "c", "d"]
kf = KFold(n_splits=2)
print("\nK-Fold splits:")
for train, test in kf.split(X):
    print("%s %s" % (train, test))
    

# Repeated K-Fold
from sklearn.model_selection import RepeatedKFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
random_state = 12883823
rkf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=random_state)
for train, test in rkf.split(X):
    print("%s %s" % (train, test))
    

# Leave One Out(LOO)
from sklearn.model_selection import LeaveOneOut
Z = [1, 2, 3, 4]
loo = LeaveOneOut()
for train, test in loo.split(Z):
    print("%s %s" % (train, test))
    

# Leave P Out (LPO)
from sklearn.model_selection import LeavePOut

Z = np.ones(4)
lpo = LeavePOut(p=2)
for train, test in lpo.split(Z):
    print("%s %s" % (train, test))
    

# Random permutations cross-validation 
from sklearn.model_selection import ShuffleSplit
X = np.arange(10)
ss = ShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
for train_index, test_index in ss.split(X):
    print("%s %s" % (train_index, test_index))
    
# Stratified K-fold
from sklearn.model_selection import StratifiedKFold, KFold
X, y = np.ones((50, 1)), np.hstack(([0] * 45, [1] * 5))
skf = StratifiedKFold(n_splits=3)
for train, test in skf.split(X, y):
    print('train - {} | test - {}'.format(
        np.bincount(y[train]), np.bincount(y[test])))
    
kf = KFold(n_splits=3) 
for train, test in kf.split(X, y):
    print('train - {} | test - {}'.format(
        np.bincount(y[train]),np.bincount(y[test])
    ))              

# Group K-fold
from sklearn.model_selection import GroupKFold
X = [0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 8.8, 9, 10]
y = ["a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]
groups = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]

gkf = GroupKFold(n_splits=3)
for train, test in gkf.split(X, y, groups=groups):
    print("%s %s" % (train, test))
    

# StratifiedGroupFold
from sklearn.model_selection import StratifiedGroupKFold
X = list(range(18))
y = [1] * 6 + [0] * 12
groups = [1, 2, 3, 3, 4, 4, 1, 1, 2, 2, 3, 4, 5, 5, 5, 6, 6, 6]
sgkf = StratifiedGroupKFold(n_splits=3)
for train, test in sgkf.split(X, y, groups=groups):
    print("%s %s" % (train, test))
    

#Using cross-validation iterations to split train and test
from sklearn.model_selection import GroupShuffleSplit

X = np.array([0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 0.001])
y = np.array(["a", "b", "b", "b", "c", "c", "c", "a"])
groups = np.array([1, 1, 2, 2, 3, 3, 4, 4])
train_indx, test_indx = next(
    GroupShuffleSplit(random_state=7).split(X, y, groups)
)
X_train, X_test, y_train, y_test = \
    X[train_indx], X[test_indx], y[train_indx], y[test_indx]
    
np.unique(groups[train_indx]), np.unique(groups[test_indx])
