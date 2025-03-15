import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

DATA_PATH = "/Users/yashcomputers/Desktop/SSI_Hackathon/AI-Traffic-Disease-Prediction/data/traffic/traffic_data.csv"
CHECKPOINT_DIR = os.path.abspath("../../checkpoints") 
SCALER_PATH = os.path.join(CHECKPOINT_DIR, "scaler.pkl")
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "lstm_traffic_model.h5")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


df = pd.read_csv(DATA_PATH)

scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[['traffic_volume']])

joblib.dump(scaler, SCALER_PATH)  
print(f"✅ Scaler saved to {SCALER_PATH}")

X, y = [], []
seq_length = 10

for i in range(len(df_scaled) - seq_length):
    X.append(df_scaled[i:i+seq_length])
    y.append(df_scaled[i+seq_length])

X, y = np.array(X), np.array(y)

model = Sequential([
    LSTM(32, return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.1),
    LSTM(16),
    Dense(1)
    # LSTM(64, return_sequences=True, input_shape=(seq_length, 1)),
    # Dropout(0.2),
    # LSTM(32),
    # Dense(1)
])

model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=10)

model.save(MODEL_PATH)
print(f"✅ Traffic Model saved to {MODEL_PATH}")
