import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import os

# Load original sklearn model
model = joblib.load('model.joblib')
coef = model.coef_.astype(np.float32)
intercept = np.array([model.intercept_], dtype=np.float32)

# Save unquantized parameters
unquant_params = {
    'weight': coef,
    'bias': intercept
}
joblib.dump(unquant_params, 'unquant_params.joblib')

# Manual quantization
def quantize(tensor, num_bits=8):
    qmin, qmax = 0, 2 ** num_bits - 1
    min_val, max_val = tensor.min(), tensor.max()
    scale = (max_val - min_val) / (qmax - qmin) if max_val != min_val else 1.0
    zero_point = qmin - np.round(min_val / scale)
    q_tensor = np.round(tensor / scale + zero_point).clip(qmin, qmax)
    dq_tensor = (q_tensor - zero_point) * scale
    return dq_tensor.astype(np.float32)

# Quantize weight and bias
quant_weight = quantize(coef)
quant_bias = quantize(intercept)

# Save quantized parameters
quant_params = {
    'weight': quant_weight,
    'bias': quant_bias
}
joblib.dump(quant_params, 'quant_params.joblib')

# Load test data
data = fetch_california_housing()
X, y = data.data, data.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define single-layer PyTorch model
class QuantModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

# Create model and set quantized weights
model = QuantModel(X.shape[1])
with torch.no_grad():
    model.linear.weight = nn.Parameter(torch.tensor(quant_weight.reshape(1, -1)))
    model.linear.bias = nn.Parameter(torch.tensor(quant_bias))

# Run inference
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_pred = model(X_tensor).detach().numpy().flatten()

# Evaluate
r2 = r2_score(y, y_pred)
print(f"Quantized PyTorch R² Score: {r2:.4f}")

# Compare file sizes
uq_size = os.path.getsize("unquant_params.joblib") / 1024
q_size = os.path.getsize("quant_params.joblib") / 1024
print(f"Model Size (unquantized): {uq_size:.2f} KB")
print(f"Model Size (quantized):   {q_size:.2f} KB")
