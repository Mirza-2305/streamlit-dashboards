import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

# Load Dataset
@st.cache_data
def load_data():
    file_path = "C:/Users/SMRC/Desktop/crude_oil_data.csv"  # Update with actual file path
    df = pd.read_csv(file_path)

    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')

    # Convert numeric columns
    df['Change %'] = df['Change %'].str.replace('%', '').astype(float)
    cols_to_convert = ['Price', 'Open', 'High', 'Low']
    df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')

    # Convert 'Vol.' column to numeric, forcing errors to NaN
    df['Vol.'] = pd.to_numeric(df['Vol.'].str.replace(',', '').str.replace('M', '').str.replace('K', ''), errors='coerce')

    # Fill missing values in 'Vol.' column with its mean
    df['Vol.'].fillna(df['Vol.'].mean(), inplace=True)
    
    return df

df = load_data()

# Streamlit UI
st.title("📈 LSTM-Based Crude Oil Price Prediction App")
st.markdown("This app predicts **future crude oil prices** using **LSTM deep learning**.")

# Show Data
st.subheader("🔍 Dataset Overview")
st.write(df.tail())

# Plot Price Trends
st.subheader("📊 Crude Oil Price Trends")
fig = px.line(df, x="Date", y="Price", title="Oil Price Over Time")
st.plotly_chart(fig)

# Normalize Data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df[['Price']])

# Create Sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 30  # Use last 30 days to predict next day
X, y = create_sequences(scaled_data, seq_length)

# Split Data
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Build LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

# Predict Future Prices
def predict_future_prices(days):
    last_sequence = X[-1]  # Last available sequence
    future_predictions = []
    future_dates = [df['Date'].max() + timedelta(days=i) for i in range(1, days+1)]

    for _ in range(days):
        pred_scaled = model.predict(np.expand_dims(last_sequence, axis=0))[0]
        pred = scaler.inverse_transform(pred_scaled.reshape(-1,1))[0][0]
        future_predictions.append(pred)
        
        # Update last_sequence for next step
        last_sequence = np.append(last_sequence[1:], pred_scaled).reshape(seq_length, 1)

    # Convert 'Date' column to string to avoid TypeError
    return pd.DataFrame({"Date": future_dates, "Predicted Price": future_predictions}).astype({"Date": str})


# User Input for Prediction
st.subheader("⏳ Predict Future Prices")
days = st.slider("Select number of future days to predict:", 1, 30, 7)

if st.button("🔮 Predict"):
    predictions_df = predict_future_prices(days)
    st.write(predictions_df)

    # Plot Predictions
    fig_future = px.line(predictions_df, x="Date", y="Predicted Price", title="Future Crude Oil Price Forecast")
    st.plotly_chart(fig_future)

st.markdown("#### 🚀 Built with Python, TensorFlow, and Streamlit")
