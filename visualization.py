from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

X, y  = load_iris(return_X_y=True)
y = y == 2 # make binary
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.8, random_state=42
)
clf = LogisticRegression(random_state=42, C=.01)
clf.fit(X_train, y_train)

y_pred = clf.predict_proba(X_test)[:, 1]
clf_disp = RocCurveDisplay.from_estimator(clf, X_test, y_test)


rfc = RandomForestClassifier(n_estimators=10, random_state=42)
rfc.fit(X_train, y_train)

ax = plt.gca()
rfc_disp = RocCurveDisplay.from_estimator(
    rfc, X_test, y_test, ax=ax, curve_kwargs={"alpha": 0.8}
)
clf_disp.plot(ax=ax, curve_kwargs={"alpha": 0.8})

# Add grid and title
plt.grid(True)
plt.title("ROC Curve Comparison", fontsize=14)
plt.legend()
plt.show()