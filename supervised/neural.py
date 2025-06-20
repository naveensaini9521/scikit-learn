from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

X = [[0., 0.], [1., 1.]]
y = [[0, 1], [1, 1]]

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X, y)

clf.predict([[2., 2.], [-1., -2.]])

[coef.shape for coef in clf.coefs_]


clf.predict_proba([[2., 2.], [1., 2.]])

clf.predict([[1., 2.]])
clf.predict([[0., 0.]])

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)