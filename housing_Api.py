from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import joblib

# Create Flask app
app = Flask(__name__)

# ---- MODEL CLASS ----
class HousingNN(nn.Module):
    def __init__(self, input_dim):
        super(HousingNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.out(x)
        return x

# ---- Load Model ----
model = HousingNN(input_dim=12)  # Make sure this matches your training
model.load_state_dict(torch.load('housing_nn.pt'))
model.eval()

# ---- Load scalers ----
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')

# ---- Prediction Route ----
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)

    features_scaled = scaler_X.transform(features)
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

    with torch.no_grad():
        prediction_scaled = model(features_tensor).numpy().flatten()[0]

    prediction = scaler_y.inverse_transform([[prediction_scaled]])[0][0]

    return jsonify({'predicted_house_price': float(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
