from fastapi import FastAPI
import tensorflow as tf
import numpy as np
import joblib

app = FastAPI()

# Load model & scaler
model = tf.keras.models.load_model("../../models/traffic_prediction/checkpoints/lstm_traffic_model.h5")
scaler = joblib.load("../../models/traffic_prediction/checkpoints/scaler.pkl")

@app.post("/predict_traffic")
def predict_traffic(data: dict):
    input_data = np.array(data["traffic_data"]).reshape(1, -1, 1)
    input_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_scaled)
    return {"predicted_traffic_volume": prediction.tolist()}
