import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Stock Analysis", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">📈 Stock Technical Analysis</h1>', unsafe_allow_html=True)

# Navigation
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🏠 Dashboard", use_container_width=True):
        st.switch_page("main.py")
with col2:
    if st.button("📈 Stock Analysis", use_container_width=True, type="primary"):
        st.rerun()
with col3:
    if st.button("🔮 Stock Predictor", use_container_width=True):
        st.switch_page("pages/stock_predictor.py")

st.markdown("---")

# Sidebar inputs
st.sidebar.header("📊 Analysis Parameters")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=datetime(2022, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.today())

# Technical indicators options
st.sidebar.subheader("Technical Indicators")
show_sma = st.sidebar.checkbox("Simple Moving Averages", value=True)
show_bollinger = st.sidebar.checkbox("Bollinger Bands", value=True)
show_rsi = st.sidebar.checkbox("RSI", value=True)
show_macd = st.sidebar.checkbox("MACD", value=True)

@st.cache_data
def load_stock_data(ticker, start, end):
    """Load stock data from Yahoo Finance"""
    try:
        df = yf.download(ticker, start=start, end=end)
        if df.empty:
            return None
        
        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [' '.join(col).strip() for col in df.columns.values]
        
        # Find Close column
        close_col = None
        for col in df.columns:
            if 'Close' in str(col):
                close_col = col
                break
        
        if close_col is None:
            return None
        
        # Rename columns for easier access
        df = df.rename(columns={
            close_col: 'Close',
            [col for col in df.columns if 'Open' in str(col)][0]: 'Open',
            [col for col in df.columns if 'High' in str(col)][0]: 'High',
            [col for col in df.columns if 'Low' in str(col)][0]: 'Low',
            [col for col in df.columns if 'Volume' in str(col)][0]: 'Volume'
        })
        
        return df.dropna()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def calculate_technical_indicators(df):
    """Calculate various technical indicators"""
    # Simple Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    return df

# Load and process data
df = load_stock_data(ticker, start_date, end_date)

if df is not None and not df.empty:
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    
    # Basic stock information
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
    
    with col1:
        st.metric("Current Price", f"${current_price:.2f}", f"{price_change:.2f} ({price_change_pct:.2f}%)")
    
    with col2:
        st.metric("52W High", f"${df['High'].max():.2f}")
    
    with col3:
        st.metric("52W Low", f"${df['Low'].min():.2f}")
    
    with col4:
        avg_volume = df['Volume'].mean()
        st.metric("Avg Volume", f"{avg_volume:,.0f}")
    
    # Main price chart with technical indicators
    st.subheader(f"📊 {ticker.upper()} Price Chart with Technical Indicators")
    
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'{ticker.upper()} Price & Volume', 'RSI', 'MACD', 'Volume'),
        row_heights=[0.5, 0.2, 0.2, 0.1]
    )
    
    # Price chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Price"
    ), row=1, col=1)
    
    # Moving averages
    if show_sma:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='orange')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], name='SMA 200', line=dict(color='red')), row=1, col=1)
    
    # Bollinger Bands
    if show_bollinger:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', line=dict(color='gray', dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', line=dict(color='gray', dash='dash')), row=1, col=1)
    
    # RSI
    if show_rsi:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    if show_macd:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', line=dict(color='red')), row=3, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Histogram'], name='Histogram', marker_color='gray'), row=3, col=1)
    
    # Volume
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='lightblue'), row=4, col=1)
    
    fig.update_layout(height=800, title_text=f"{ticker.upper()} Technical Analysis")
    fig.update_xaxes(rangeslider_visible=False)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical Analysis
    st.subheader("📊 Statistical Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price statistics
        st.write("**Price Statistics**")
        price_stats = df['Close'].describe()
        st.dataframe(price_stats)
    
    with col2:
        # Returns analysis
        st.write("**Returns Analysis**")
        df['Daily_Return'] = df['Close'].pct_change()
        returns_stats = df['Daily_Return'].describe()
        st.dataframe(returns_stats)
    
    # Stationarity Test
    st.subheader("📈 Stationarity Test (ADF)")
    adf_result = adfuller(df['Close'].dropna())
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("ADF Statistic", f"{adf_result[0]:.4f}")
    with col2:
        st.metric("p-value", f"{adf_result[1]:.6f}")
    with col3:
        is_stationary = "✅ Stationary" if adf_result[1] < 0.05 else "❌ Non-Stationary"
        st.metric("Result", is_stationary)
    
    # Rolling Statistics
    st.subheader("📉 Rolling Statistics")
    
    window = st.slider("Rolling Window (days)", 10, 100, 30)
    
    rolling_mean = df['Close'].rolling(window=window).mean()
    rolling_std = df['Close'].rolling(window=window).std()
    
    fig_rolling = go.Figure()
    fig_rolling.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price', line=dict(color='blue')))
    fig_rolling.add_trace(go.Scatter(x=df.index, y=rolling_mean, name=f'Rolling Mean ({window}d)', line=dict(color='red')))
    fig_rolling.add_trace(go.Scatter(x=df.index, y=rolling_std, name=f'Rolling Std ({window}d)', line=dict(color='green')))
    
    fig_rolling.update_layout(
        title=f"Rolling Statistics (Window: {window} days)",
        xaxis_title="Date",
        yaxis_title="Price",
        height=400
    )
    
    st.plotly_chart(fig_rolling, use_container_width=True)
    
    # Correlation Analysis
    st.subheader("🔗 Price and Volume Correlation")
    correlation = df['Close'].corr(df['Volume'])
    st.metric("Price-Volume Correlation", f"{correlation:.4f}")
    
    # Recent performance table
    st.subheader("📋 Recent Performance (Last 20 Days)")
    recent_data = df.tail(20)[['Open', 'High', 'Low', 'Close', 'Volume']].round(2)
    recent_data['Daily_Change'] = recent_data['Close'].diff().round(2)
    recent_data['Daily_Change_%'] = (recent_data['Close'].pct_change() * 100).round(2)
    
    st.dataframe(recent_data, use_container_width=True)
    
    # Trading signals
    st.subheader("🚦 Basic Trading Signals")
    
    # Simple signals based on moving averages
    last_price = df['Close'].iloc[-1]
    last_sma_20 = df['SMA_20'].iloc[-1]
    last_sma_50 = df['SMA_50'].iloc[-1]
    last_rsi = df['RSI'].iloc[-1]
    
    signals = []
    
    if last_price > last_sma_20 > last_sma_50:
        signals.append("🟢 Bullish: Price above both SMA 20 and SMA 50")
    elif last_price < last_sma_20 < last_sma_50:
        signals.append("🔴 Bearish: Price below both SMA 20 and SMA 50")
    else:
        signals.append("🟡 Neutral: Mixed signals from moving averages")
    
    if last_rsi > 70:
        signals.append("🔴 Overbought: RSI above 70")
    elif last_rsi < 30:
        signals.append("🟢 Oversold: RSI below 30")
    else:
        signals.append("🟡 Neutral: RSI in normal range")
    
    for signal in signals:
        st.write(signal)

else:
    st.error(f"Unable to load data for ticker '{ticker}'. Please check the ticker symbol and try again.")
