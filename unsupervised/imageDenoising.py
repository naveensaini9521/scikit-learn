import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X, y = fetch_openml(data_id=41082, as_frame=False, return_X_y=True)
X = MinMaxScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=0, train_size=1000, test_size=100
)

rng = np.random.RandomState(0)
noise = rng.normal(scale=0.25, size=X_test.shape)

noise = rng.normal(scale=0.25, size=X_train.shape)
X_test_noisy = X_test + noise

X_train_noisy = X_train + noise


import matplotlib.pyplot as plt

def plot_digits(X, title):
    
    fig, axs = plt.subplots(nrows=10, ncols=10, figsize=(8, 8))
    
    for img, ax in zip(X, axs.ravel()):
        ax.imshow(img.reshape((16, 16)), cmap="Greys")
        ax.axis("off")
    fig.subtitle(title, fontsize=24)
    
plot_digits(X_test, "uncorrupted test images")
plot_digits(
    X_test_noisy, f"Noisy test images\nMSE: {np.mean((X_test - X_test_noisy) ** 2):.2f}"
)
plt.show()