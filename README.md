# Bitcoin-Price-Prediction-using-Machine-Learning-and-LSTM

Tools Used: Python, TensorFlow, Scikit-learn, Streamlit, Pandas, NumPy, Matplotlib, XGBoost
Dataset: Bitcoin Historical Price Data (CoinGecko API)

# Project Overview

Bitcoin is one of the most volatile assets in the world. Predicting its price movement is challenging due to market uncertainty and rapid fluctuations.
This project applies Machine Learning and Deep Learning (LSTM) models to predict future Bitcoin prices using historical data and technical indicators.

The goal is to assist traders and investors in making data-driven investment decisions by forecasting Bitcoin’s short- and long-term price trends.

# Objective

Build a time-series forecasting model that predicts Bitcoin price movements based on historical data and financial indicators such as:

RSI (Relative Strength Index)

SMA (Simple Moving Average)

EMA (Exponential Moving Average)

Bollinger Bands

Deploy the trained model as a Streamlit web application for real-time prediction and visualization.

# Dataset Description

Source: CoinGecko API & Historical Bitcoin Dataset

Data Range: Multi-year Bitcoin trading data

Features:

Date

Open, High, Low, Close

Volume

Derived Indicators: SMA, EMA, RSI, Bollinger Bands

Split: 80% Training | 20% Testing

Data Type: Time Series

# Data Preprocessing

To prepare the dataset for model training:

Converted Date column to DateTimeIndex

Filled missing values using forward-fill

Normalized all features using MinMaxScaler

Added new features:

SMA_30 and EMA_30 for trend analysis

RSI_14 for momentum detection

Upper_Band and Lower_Band for volatility analysis

Maintained chronological order to prevent data leakage

# Exploratory Data Analysis (EDA)

Bitcoin prices show long-term upward trends with high short-term volatility

RSI and EMA correlate strongly with price momentum

Bollinger Bands widen during volatility and narrow during stability

Volume surges often precede major price changes

# Key Visual Insights:

Bitcoin Price Trend over time

Bollinger Bands with Upper & Lower limits

RSI over time (Overbought & Oversold signals)

Correlation Heatmap between features

# Models Implemented & Comparison

Multiple machine learning and deep learning models were trained and compared.

Model	Type	RMSE	R² Score
Linear Regression	Statistical	523.7	0.78
Random Forest	Ensemble	430.5	0.86
XGBoost	Gradient Boosting	368.1	0.91
LSTM	Deep Learning	176.0	0.94 

# LSTM achieved the lowest RMSE (176.0) and highest accuracy (R² = 0.94), making it the best performer.

# LSTM Model Architecture

Input: 60-day time window (past 60 days’ prices)

LSTM Layer: 50 neurons, activation = tanh

Dropout: 0.2 (prevents overfitting)

Dense Layer: 1 neuron (predicts next-day closing price)

Optimizer: Adam

Loss Function: Mean Squared Error (MSE)

Epochs: 50

Batch Size: 32

The model captures temporal dependencies and market patterns effectively, outperforming all traditional models.

# Model Evaluation Metrics
Metric	Value
Mean Absolute Error (MAE)	125.4
Mean Squared Error (MSE)	31012.7
Root Mean Squared Error (RMSE)	176.0
R² Score	0.94

Interpretation:
A high R² and low RMSE confirm that the LSTM model accurately captures market movements and predicts future price trends with minimal error.

# Visualization Insights

Bitcoin Price Trend: Long-term growth with periodic corrections.

RSI Chart: Shows overbought (>70) and oversold (<30) conditions.

Bollinger Bands: Highlight price volatility zones.

Predicted vs Actual Prices: Close alignment between true and predicted values.

Model Comparison Graphs: LSTM outperforms all others visually and numerically.

# Streamlit Web Application

An interactive Streamlit app was built for real-time Bitcoin price forecasting.

# Features:

Fetches live Bitcoin data via CoinGecko API

User inputs desired prediction days

Displays forecasted trends with charts

Visual indicators:

🔴 High Volatility

🟢 Stable Market

Downloadable CSV for future predictions

# Files:

bitcoin_price_lstm_model.h5 – Trained LSTM model

bitcoin_price_scaler.pkl – Scaler used for normalization

api_key.py – CoinGecko API configuration

To launch the app:

streamlit run app.py

# Technical Stack
Category	Tools
Language	Python 3.10+
Frameworks	TensorFlow / Keras
Libraries	Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, XGBoost
APIs	CoinGecko, Binance (for live data)
Deployment	Streamlit
Visualization	Plotly, Matplotlib

# Results Summary

LSTM model outperformed all others with the best accuracy.

Achieved R² = 0.94 and RMSE = 176.0.

Real-time app provides visual and numerical Bitcoin price forecasts.

Captures market reversals, trend shifts, and volatility patterns accurately.

# Future Enhancements

Integrate Transformer-based architectures (GPT/TimeGPT) for improved forecasting.

Add Sentiment Analysis from social media and news data.

Implement live trading alerts via Binance or Coinbase API.

Develop a comprehensive trading dashboard in Streamlit.

Expand to multi-cryptocurrency forecasting.

 Repository Structure
 Bitcoin-Price-Prediction/
│
├──  bitcoin.csv                         # Dataset
├──  Bitcoin_New_LSTM_final.ipynb         # Model training notebook
├──  Bitcoin_online_data.ipynb            # Live data integration notebook
├──  Bitcoin_Price_Prediction_Final_Presentation_With_Models.pptx
├──  Bitcoin Price Prediction using Machine Learning in Python.pdf
├──  app.py                               # Streamlit app script
├──  requirements.txt                     # Dependencies
├──  model/
│   ├── bitcoin_price_lstm_model.h5
│   └── bitcoin_price_scaler.pkl
└──  README.md                            # Project documentation

 Installation & Usage

Clone the repository

git clone https://github.com/yourusername/Bitcoin-Price-Prediction.git
cd Bitcoin-Price-Prediction


Install dependencies

pip install -r requirements.txt


Run the Streamlit app

streamlit run app.py


Open in browser:
http://localhost:8501

🏁 Conclusion

This project demonstrates how Machine Learning and Deep Learning can effectively model financial data for cryptocurrency forecasting.



https://github.com/user-attachments/assets/108733ae-b684-43b0-a080-097b45aeeb72

Training video more than 10MB. 
The LSTM-based prediction system provides reliable insights for traders to identify entry and exit points, helping to make data-driven investment decisions.

“AI-driven trading analytics can revolutionize financial forecasting by blending data science and market intuition.”
