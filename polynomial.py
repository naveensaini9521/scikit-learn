from sklearn.preprocessing import PolynomialFeatures
import numpy as np

X = np.arange(6).reshape(3, 2)
# X

"""
array ([[0, 1],
[2, 3],
[4, 5]]) 
"""

poly = PolynomialFeatures(degree=2)
poly.fit_transform(X)


from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np

model = Pipeline([('poly', PolynomialFeatures(degree=3)),
                  ('linear', LinearRegression(fit_intercept=False))])

# fit to an order-3 polynomial data

x = np.arange(5)
y = 3 -2 * x + x ** 2 - x ** 3
model = model.fit(x[:, np.newaxis], y)
model.named_steps['linear'].coef_
# array([ 3., -2.,  1.,  -1.])

