from sklearn import svm
X = [[0, 0], [2, 2]]
y = [0.5, 2.5]
regr = svm.SVR()
regr.fit(X, y)

regr.predict([[1, 1]])


# 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
clf = make_pipeline(StandardScaler(), SVC())

clf = make_pipeline(StandardScaler(), SVC())

# Kernel Functions

linear_svc = svm.SVC(kernel='linear')
linear_svc._sparse_kernels

rbf_svc = svm.SVC(kernel='rbf')
rbf_svc.kernel