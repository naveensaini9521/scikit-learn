from sklearn import linear_model

reg = linear_model.LinearRegression()
reg.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
#LinearRegression()
print(reg.coef_)
# array([0.5, 0.5])
print(reg.intercept_)
# 0.0


# Regression

reg = linear_model.Ridge(alpha=.5)
reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])

# Ridge(alpha=0.5)

print(reg.coef_)
# array([0.34545455, 0.34545455])

print(reg.intercept_)
# np.float64(0.13636)


# Setting the regularization parameter
import numpy as np

reg = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))
reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])

# RidgeCV(alphas=arrays([1.e-06, 1.e-05, 1.e-04, 1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02, 1.e+03, 1.e+04, 1.e+05, 1.e+06]))

print(reg.alpha_)
# np.float64(0.01)

#Lasso
reg = linear_model.Lasso(alpha=0.1)
reg.fit([[0, 0], [1, 1]], [0, 1])
# Lasso(alpha=0.1)
print(reg.predict([[1, 1]]))
# array([0.8])

# LARS Lasso
reg = linear_model.LassoLars(alpha=.1)
reg.fit([[0, 0], [1, 1]], [0, 1])
#LassoLars(alpha=.1)
print(reg.coef_)
#array([0.6, 0.])


# Bayesian Ridge Regression
X = [[0., 0.], [1., 1.], [2., 2.], [3., 3.]]
Y = [0., 1., 2., 3.]
reg = linear_model.BayesianRidge()
reg.fit(X, Y)
# BayesianRidge()

reg.predict([[1, 0.]])  
# array([0.50000013])

print(reg.coef_)
# array([0.49999993, 0.49999993])

# LARS Lasso
reg = linear_model.LassoLars(alpha=.1)
reg.fit([[0, 0], [1, 1]], [0, 1])
# LassoLars(alpha=0.1)
reg.coef_

# Baysen Ridge Regression

X = [[0., 0.], [1., 1.], [2., 2.], [3., 3.]]
Y = [0., 1., 2., 3.]
reg = linear_model.BayesianRidge()
reg.fit(X, Y)
#BayesianRidge