import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from api_key import API_KEY
import os

# Disable oneDNN optimization warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Streamlit App Config
st.set_page_config(page_title="💰 Bitcoin Price Predictor", layout="wide")
st.title("💰 Bitcoin Price Prediction App")

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        model = load_model("bitcoin_price_lstm_model.h5")
        scaler = joblib.load("bitcoin_price_scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"❌ Error loading model or scaler: {e}")
        st.stop()

model, scaler = load_model_and_scaler()

# Sidebar settings
st.sidebar.header("Settings")
days_to_predict = st.sidebar.number_input(
    "Enter number of future days to predict",
    min_value=1,
    max_value=365,
    value=60
)

# ✅ Add Enter / Predict Button
start_prediction = st.sidebar.button("🔮 Predict Bitcoin Prices")

# Fetch Bitcoin data from CoinGecko
@st.cache_data(ttl=3600)
def get_bitcoin_data(days=90):
    try:
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        params = {'vs_currency': 'usd', 'days': str(days)}
        headers = {'x-cg-demo-api-key': API_KEY}
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        st.error(f"❌ Failed to fetch Bitcoin data: {e}")
        st.stop()

st.write("📊 Fetching recent Bitcoin data...")
df = get_bitcoin_data(90)
st.line_chart(df['price'], use_container_width=True)

# Run prediction only after pressing Enter
if start_prediction:
    # Prepare data for prediction
    scaled_data = scaler.transform(df[['price']])
    input_data = scaled_data[-60:].reshape(1, 60, 1)

    # Predict future prices
    future_scaled = []
    for _ in range(days_to_predict):
        next_pred = model.predict(input_data, verbose=0)[0, 0]
        future_scaled.append(next_pred)
        new_input = np.append(input_data[0][1:], [[next_pred]], axis=0)
        input_data = new_input.reshape(1, 60, 1)

    # Inverse scale predictions
    predicted_prices = scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1))

    # Create future date range
    last_date = df.index[-1]
    future_dates = [last_date + timedelta(days=i + 1) for i in range(days_to_predict)]

    # Combine predictions
    pred_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Price (USD)": predicted_prices.flatten()
    }).set_index("Date")

    # Plot results
    st.subheader(f"📈 Bitcoin Price Forecast for Next {days_to_predict} Days")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df['price'], label="Historical Price", color='blue')
    ax.plot(pred_df.index, pred_df["Predicted Price (USD)"], label="Predicted Price", linestyle="--", color='orange')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    st.pyplot(fig)

    # Show table
    st.dataframe(pred_df.tail(10))

    st.success(f"✅ Prediction completed for next {days_to_predict} days.")
else:
    st.info("👈 Set the number of days and click **'🔮 Predict Bitcoin Prices'** to start forecasting.")
