import numpy as np

data = np.arrange(100).reshape(10, 10)
rows = np.array([0, 2, 3])[:, np.newaxis]
columns = np.array([1, 2])
data[rows, columns]

