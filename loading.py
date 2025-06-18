from sklearn.datasets import load_iris
data = load_iris()
X, y = data.data, data.target

# Splitting Data into Training and Testing

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Data Transformation 
# a) Standardization

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)
X_train_standardized = scaler.transform(X_train)
X_train_standardized = scaler.transform(X_test)

# b) Normaliziation
from sklearn.preprocessing import Normalizer
scaler = Normalizer().fit(X_train)
X_train_normalized = scaler.transform(X_train)
X_test_normalized = scaler.transform(X_test)


# c) Binarization

from sklearn.preprocessing import Binarizer
binarizer = Binarizer(threshold=1.0).fit(X)
X_binarized = binarizer.transform(X)

# Encoding Non-Numrical Data
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Handling Missing Values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Creating Polynomial Features 
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)