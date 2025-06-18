import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

n_neighbour = 3
random_state = 0

# Load Digits dataset
X, y = datasets.load_digits(return_X_y=True)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, stratify=y, random_state=random_state
)

dim = len(X[0])
n_classes = len(np.unique(y))

# Reduce dimesion to 2 with PCA
pca = make_pipeline(StandardScaler(), PCA(n_components=2, random_state=random_state))

# Reduce dimension to 2 with LDA
lda = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(n_components=2))

# Reduce dimension to 2 with NCA
nca = make_pipeline(
    StandardScaler(),
    NeighborhoodComponentsAnalysis(n_components=2, random_state=random_state)
)

# Use a nearest neighbour classifier to evaluate the methods
knn = KNeighborsClassifier(n_neighbors=n_neighbour)

# Make a list of the methods to be compared
dim_reduction_methods = [("PCA", pca), ("LDA", lda), ("NCA", nca)]

# plt.figure()
for i, (name, model) in enumerate(dim_reduction_methods):
    plt.figure(figsize=(6, 5))
    
    #plt.subplot(1, 3, i + 1, aspect=1)
    
    # Fit the method's model
    model.fit(X_train, y_train)
    
    # Fit a nearest neighbour classifier on the embedded training set
    knn.fit(model.transform(X_train), y_train)
    
    # Compute the nearest neighbour accuracy using the fitted model
    acc_knn = knn.score(model.transform(X_train), y_train)
    
    # Embed the data set in 2 dimensions using the fitted model
    X_embedded = model.transform(X)
     
    # Plot the projected points and show the evaluation score
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, s=30, cmap="Set1", edgecolor='k')
    plt.title("{}, KNN (k={})\nTest accuracy = {:.2f}".format(name, n_neighbour, acc_knn))
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    
plt.tight_layout()
plt.show()