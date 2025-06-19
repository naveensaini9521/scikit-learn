# Evaluating Model Performance

# a) Metrices for Classification Models

from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))

# Classification Report

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
