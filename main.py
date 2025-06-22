import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    [data-testid="stSidebarNav"] > ul {
    display: none;
    }
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b3;
        margin-bottom: 2rem;
    }
    .metric-card {
    background-color: #ffffff;  /* Brighter background */
    padding: 1rem;
    border-radius: 0.5rem;
    text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05); /* subtle shadow */
    }

    .metric-card h2 {
        color: #111111;  /* Darker text for readability */
    }
    .metric-card h3 {
        color: #444444;  /* Darker subtitle text */
    }
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">📊 Stock Analysis Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown('<div class="sidebar-header">Navigation</div>', unsafe_allow_html=True)

# Navigation buttons
if st.sidebar.button("🏠 Dashboard", use_container_width=True):
    st.rerun()

if st.sidebar.button("📈 Stock Analysis", use_container_width=True):
    st.switch_page("pages/stock_analysis.py")

if st.sidebar.button("🔮 Stock Predictor", use_container_width=True):
    st.switch_page("pages/stock_predictor.py")

st.sidebar.markdown("---")

# Stock input section
st.sidebar.markdown('<div class="sidebar-header">Quick Stock Lookup</div>', unsafe_allow_html=True)
ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL", placeholder="e.g., AAPL, GOOGL, TSLA")

@st.cache_data
def get_stock_data(symbol, period="1y"):
    """Fetch stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        info = stock.info
        return data, info
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None, None

# Main content
if ticker:
    data, info = get_stock_data(ticker.upper())
    
    if data is not None and not data.empty:
        # Stock info section
        col1, col2, col3, col4 = st.columns(4)
        
        current_price = data['Close'].iloc[-1]
        prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Current Price</h3>
                <h2>${current_price:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            color = "green" if change >= 0 else "red"
            arrow = "↗️" if change >= 0 else "↘️"
            st.markdown(f"""
            <div class="metric-card">
                <h3>Change</h3>
                <h2 style="color: {color}">{arrow} ${change:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            color = "green" if change_pct >= 0 else "red"
            st.markdown(f"""
            <div class="metric-card">
                <h3>Change %</h3>
                <h2 style="color: {color}">{change_pct:.2f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            volume = data['Volume'].iloc[-1]
            st.markdown(f"""
            <div class="metric-card">
                <h3>Volume</h3>
                <h2>{volume:,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Stock chart
        st.subheader(f"📈 {ticker.upper()} Stock Price (1 Year)")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.update_layout(
            title=f"{ticker.upper()} Stock Price Movement",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=400,
            showlegend=True,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent performance table
        st.subheader("📊 Recent Performance (Last 10 Days)")
        recent_data = data.tail(10)[['Open', 'High', 'Low', 'Close', 'Volume']].round(2)
        recent_data['Change'] = recent_data['Close'].diff().round(2)
        recent_data['Change %'] = (recent_data['Close'].pct_change() * 100).round(2)
        
        st.dataframe(recent_data, use_container_width=True)
        
        # Additional stock information
        if info:
            st.subheader("ℹ️ Company Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Company Name:** {info.get('longName', 'N/A')}")
                st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                st.write(f"**Market Cap:** ${info.get('marketCap', 0):,}" if info.get('marketCap') else "Market Cap: N/A")
            
            with col2:
                st.write(f"**52 Week High:** ${info.get('fiftyTwoWeekHigh', 'N/A')}")
                st.write(f"**52 Week Low:** ${info.get('fiftyTwoWeekLow', 'N/A')}")
                st.write(f"**P/E Ratio:** {info.get('trailingPE', 'N/A')}")
                st.write(f"**Dividend Yield:** {info.get('dividendYield', 'N/A')}")
    
    else:
        st.error(f"Unable to fetch data for ticker '{ticker}'. Please check if the ticker symbol is correct.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <h3>Welcome to Stock Analysis Dashboard</h3>
    <p>Navigate using the sidebar to access detailed stock analysis and prediction tools.</p>
    <p><strong>📈 Stock Analysis:</strong> Comprehensive technical analysis with charts and indicators</p>
    <p><strong>🔮 Stock Predictor:</strong> Advanced forecasting using ARIMA, SARIMA, Prophet, and LSTM models</p>
</div>
""", unsafe_allow_html=True)