import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# Load model
model = joblib.load('model.joblib')

# Load and preprocess test data
data = fetch_california_housing()
X = data.data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Predict with first 5 samples
preds = model.predict(X_scaled[:5])
print(f"Sample predictions: {preds}")
