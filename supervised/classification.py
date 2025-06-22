from sklearn.linear_model import SGDClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# SGD Classifier with pipeline
est = make_pipeline(StandardScaler(), SGDClassifier(loss="log_loss", max_iter=1000))
est.fit(X_train, y_train)

y_pred = est.predict(X_test)
print("Predictions:", y_pred)

# Report
print(classificaton_report(y_test, y_pred))

X = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
clf.fit(X, y)


print("Prediction for [2., 2.]:", clf.predict([[2., 2.]]))
print("Weights (coef_):", clf.coef_)
print("Intercept:", clf.intercept_)
print("Decision function:", clf.decision_function([[2., 2.]]))

clf = SGDClassifier(loss="log_loss", max_iter=5).fit(X, y)
print("log probabilities:", clf.predict_log_proba([[1., 1.]]))


##
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#
est = make_pipeline(StandardScaler(), SGDClassifier())
est.fit(X_train)
est.predict(X_test)
