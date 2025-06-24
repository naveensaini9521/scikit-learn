from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

random_state = 42
X, y = make_regression(random_state=random_state, n_features=1, noise=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=random_state)

scaler = StandardScaler()
X_train_transformed = scaler.fit_transform(X_train)
model = LinearRegression().fit(X_train_transformed, y_train)
print(mean_squared_error(y_test, model.predict(X_test)))

X_test_transformed = scaler.transform(X_test)
print(mean_squared_error(y_test, model.predict(X_test_transformed)))

model = make_pipeline(StandardScaler(), LinearRegression())
model.fit(X_train, y_train)

mean_squared_error(y_test, model.predict(X_test))