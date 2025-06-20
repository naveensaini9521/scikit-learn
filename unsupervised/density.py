from sklearn.neighbors import KernelDensity
import numpy as np

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
kde = KernelDensity(kernel="gaussian", bandwidth=0.2).fit(X)
kde.score_samples(X)

log_density = kde.score_samples(X)
print(log_density)