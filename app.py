from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import IsolationForest
import pickle
import os

app = Flask(__name__)

# Load your model
model_path = 'models/transaction_anomaly_model.pkl'
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Relevant features used during training
relevant_features = ['Transaction_Amount', 'Average_Transaction_Amount', 'Frequency_of_Transactions']

@app.route('/')
def home():
    return "Flask API is running. Use the '/predict' endpoint to make predictions."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not all(feature in data for feature in relevant_features):
            return jsonify({"error": "Missing one or more required features in the request data."}), 400

        user_df = pd.DataFrame([data], columns=relevant_features)
        user_anomaly_pred = model.predict(user_df)
        user_anomaly_pred_binary = 1 if user_anomaly_pred == -1 else 0
        
        if user_anomaly_pred_binary == 1:
            return jsonify({"result": "Anomaly detected: This transaction is flagged as an anomaly."})
        else:
            return jsonify({"result": "No anomaly detected: This transaction is normal."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

