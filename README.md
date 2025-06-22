# 📈 Stock Analysis and Forecasting Web App

A **Streamlit** web application for comprehensive stock market analysis and future price prediction. This tool allows users to visualize stock trends, analyze technical indicators, and forecast stock prices using advanced time series models.

---

## 🚀 Features

### 🧭 Dashboard

* Real-time stock data (powered by Yahoo Finance)
* Interactive price charts with historical data
* Key stock metrics and company information

### 🔍 Stock Analysis

* Technical indicators:

  * Simple Moving Average (SMA)
  * Bollinger Bands
  * Relative Strength Index (RSI)
  * Moving Average Convergence Divergence (MACD)
* Statistical summary
* Buy/Sell trading signals

### 🔮 Stock Predictor

* Time-series forecasting using:

  * **ARIMA**
  * **SARIMA**
  * **Prophet**
  * **LSTM**
* Model performance comparison
* Residual and error analysis

---

## 📁 Project Structure

```
stock_analysis_and_forecasting/
├── main.py                     # Entry point for Streamlit app
├── pages/
│   ├── stock_analysis.py       # Stock analysis UI & logic
│   └── stock_predictor.py      # Forecasting interface & models
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## ⚠️ Notes

* **Data Source:** Yahoo Finance (via yfinance library)
* **Disclaimer:** Forecasts are for **educational and informational purposes only**
* **Internet Required:** Application fetches real-time and historical data online
* **Resource Usage:** LSTM model may require more memory and computation time

---

## 🛠️ Installation & Run

```bash
# Clone the repository
git clone https://github.com/yourusername/stock_analysis_and_forecasting.git
cd stock_analysis_and_forecasting

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run main.py
```

---

## 📬 Feedback & Contributions

Feel free to open an issue or submit a pull request to improve the app. Contributions are welcome!

---

Let me know if you'd like to include demo screenshots, deployment instructions (e.g., for Streamlit Cloud), or license information.
