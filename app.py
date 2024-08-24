from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import IsolationForest
import pickle

app = Flask(__name__)

# Load your model
with open('C:/Users/aprav/Downloads/transaction_anomaly_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Relevant features used during training
relevant_features = ['Transaction_Amount', 'Average_Transaction_Amount', 'Frequency_of_Transactions']

@app.route('/')
def home():
    return "Flask API is running. Use the '/predict' endpoint to make predictions."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_df = pd.DataFrame([data], columns=relevant_features)
    user_anomaly_pred = model.predict(user_df)
    user_anomaly_pred_binary = 1 if user_anomaly_pred == -1 else 0
    
    if user_anomaly_pred_binary == 1:
        return jsonify({"result": "Anomaly detected: This transaction is flagged as an anomaly."})
    else:
        return jsonify({"result": "No anomaly detected: This transaction is normal."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
