import numpy as np
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV, GroupKFold
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import make_pipeline

n_samples, n_features = 100, 4
rng = np.random.RandomState(42)
X = rng.rand(n_samples, n_features)
y = rng.rand(n_samples, n_features)
my_groups = rng.randint(0, 10, size=n_samples)
my_weights = rng.rand(n_samples)
my_other_weights = rng.rand(n_samples)


weighted_acc = make_scorer(accuracy_score).set_score_request(sample_weight=True)
lr = LogisticRegressionCV(
    cv=GroupKFold(),
    scoring=weighted_acc
).set_fit_request(sample_weight=True)
cv_results = cross_validate(
    lr,
    X,
    y,
    params={"sample_weight": my_weights, "groups": my_groups},
    cv=GroupKFold(),
    scoring=weighted_acc,
)


# API Interface
param_grid = {"C": [0.1, 1]}
lr = LogisticRegression().set_fit_request(sample_weight=True)
try:
    GridSearchCV(
        estimator=lr, param_grid=param_grid
    ).fit(X, y, sample_weight=my_weights)
except ValueError as e:
    print(e)
    
