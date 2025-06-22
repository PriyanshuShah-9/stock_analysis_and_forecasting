import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Stock Predictor", layout="wide")

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

st.markdown('<h1 class="main-header">🔮 Stock Price Prediction Models</h1>', unsafe_allow_html=True)

# Navigation
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🏠 Dashboard", use_container_width=True):
        st.switch_page("main.py")
with col2:
    if st.button("📈 Stock Analysis", use_container_width=True):
        st.switch_page("pages/stock_analysis.py")
with col3:
    if st.button("🔮 Stock Predictor", use_container_width=True, type="primary"):
        st.rerun()

st.markdown("---")

# Sidebar inputs
st.sidebar.header("🔧 Prediction Parameters")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.today())
forecast_days = st.sidebar.slider("Forecast Period (days)", 30, 180, 60)

# Model selection
st.sidebar.subheader("📊 Models to Run")
run_arima = st.sidebar.checkbox("ARIMA", value=True)
run_sarima = st.sidebar.checkbox("SARIMA", value=True)
run_prophet = st.sidebar.checkbox("Prophet", value=True)
run_lstm = st.sidebar.checkbox("LSTM", value=True)

@st.cache_data
def load_data(ticker, start, end):
    """Load stock data from Yahoo Finance"""
    try:
        df = yf.download(ticker, start=start, end=end)
        if df.empty:
            return pd.DataFrame()
        
        # Handle MultiIndex columns properly
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [' '.join(col).strip() for col in df.columns.values]
        
        # Find Close column
        close_col = None
        for col in df.columns:
            if 'Close' in str(col):
                close_col = col
                break
        
        if close_col is None:
            return pd.DataFrame()
        
        # Create clean dataframe with Close column
        result_df = pd.DataFrame({'Close': df[close_col]})
        return result_df.dropna()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def find_best_arima_params(data, max_p=3, max_d=2, max_q=3):
    """Find optimal ARIMA parameters using AIC"""
    best_aic = float('inf')
    best_params = None
    
    # Start with common parameter combinations
    param_combinations = [
        (1, 1, 1), (1, 1, 0), (0, 1, 1), (2, 1, 1), (1, 1, 2),
        (2, 1, 2), (1, 0, 1), (2, 0, 2), (0, 1, 0), (1, 0, 0)
    ]
    
    for p, d, q in param_combinations:
        try:
            model = ARIMA(data, order=(p, d, q))
            fitted = model.fit()
            if fitted.aic < best_aic and not np.isnan(fitted.aic):
                best_aic = fitted.aic
                best_params = (p, d, q)
        except:
            continue
    
    return best_params if best_params else (1, 1, 1)

def find_best_sarima_params(data, seasonal_period=7):
    """Find optimal SARIMA parameters using AIC"""
    best_aic = float('inf')
    best_params = None
    best_seasonal = None
    
    param_combinations = [(1, 1, 1), (1, 1, 0), (0, 1, 1), (2, 1, 1)]
    seasonal_params = [(0, 1, 1, seasonal_period), (1, 1, 1, seasonal_period)]
    
    for p, d, q in param_combinations:
        for seasonal in seasonal_params:
            try:
                model = SARIMAX(data, order=(p, d, q), seasonal_order=seasonal)
                fitted = model.fit(disp=False, maxiter=50)
                if fitted.aic < best_aic and not np.isnan(fitted.aic):
                    best_aic = fitted.aic
                    best_params = (p, d, q)
                    best_seasonal = seasonal
            except:
                continue
    
    return (best_params if best_params else (1, 1, 1), 
            best_seasonal if best_seasonal else (1, 1, 1, 7))

def evaluate_model(true, pred):
    """Calculate RMSE and MAE"""
    min_len = min(len(true), len(pred))
    true = true[:min_len]
    pred = pred[:min_len]
    
    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    return rmse, mae

# Load and validate data
df_raw = load_data(ticker, start_date, end_date)

if df_raw.empty:
    st.error(f"No data found for ticker '{ticker}'. Please check the ticker symbol and date range.")
    st.stop()

if len(df_raw) < 100:
    st.error(f"Insufficient data for analysis. Found {len(df_raw)} rows, need at least 100.")
    st.stop()

# Display basic info
st.subheader(f"📊 {ticker.upper()} Data Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Data Points", len(df_raw))
with col2:
    st.metric("Date Range", f"{df_raw.index[0].strftime('%Y-%m-%d')} to {df_raw.index[-1].strftime('%Y-%m-%d')}")
with col3:
    st.metric("Current Price", f"${df_raw['Close'].iloc[-1]:.2f}")
with col4:
    price_change = df_raw['Close'].iloc[-1] - df_raw['Close'].iloc[0]
    st.metric("Total Change", f"${price_change:.2f}")

# Stationarity Test
st.subheader("📈 Stationarity Test (ADF)")
adf_result = adfuller(df_raw['Close'].dropna())

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("ADF Statistic", f"{adf_result[0]:.4f}")
with col2:
    st.metric("p-value", f"{adf_result[1]:.6f}")
with col3:
    is_stationary = "✅ Stationary" if adf_result[1] < 0.05 else "❌ Non-Stationary"
    st.metric("Status", is_stationary)

# Rolling Statistics Chart
st.subheader("📉 Rolling Mean and Standard Deviation")
rolling_mean = df_raw['Close'].rolling(30).mean()
rolling_std = df_raw['Close'].rolling(30).std()

fig_rolling = go.Figure()
fig_rolling.add_trace(go.Scatter(x=df_raw.index, y=df_raw['Close'], name='Original', line=dict(color='blue')))
fig_rolling.add_trace(go.Scatter(x=df_raw.index, y=rolling_mean, name='Rolling Mean (30d)', line=dict(color='orange')))
fig_rolling.add_trace(go.Scatter(x=df_raw.index, y=rolling_std, name='Rolling Std (30d)', line=dict(color='green')))

fig_rolling.update_layout(
    title='Rolling Statistics',
    xaxis_title='Date',
    yaxis_title='Price ($)',
    height=400
)

st.plotly_chart(fig_rolling, use_container_width=True)

# Prepare data for modeling
train_size = len(df_raw) - 30
train_data = df_raw['Close'][:train_size]
test_data = df_raw['Close'][train_size:]

# Create forecast index
forecast_index = pd.date_range(df_raw.index[-1] + timedelta(days=1), periods=forecast_days, freq='D')

# Initialize results storage
forecasts = {}
accuracies = {}
residuals_data = {}

# Model execution
if run_arima:
    st.subheader("🔮 ARIMA Model")
    with st.spinner("Training ARIMA model..."):
        try:
            # Find optimal parameters
            arima_params = find_best_arima_params(train_data)
            st.write(f"**Optimal ARIMA Order:** {arima_params}")
            
            # Fit model on training data
            arima_model = ARIMA(train_data, order=arima_params)
            arima_fitted = arima_model.fit()
            
            # Predict on test data
            arima_pred = arima_fitted.forecast(steps=len(test_data))
            
            # Fit on full data and forecast future
            full_arima_model = ARIMA(df_raw['Close'], order=arima_params)
            full_arima_fitted = full_arima_model.fit()
            arima_forecast = full_arima_fitted.forecast(steps=forecast_days)
            
            # Store results
            forecasts['ARIMA'] = pd.Series(arima_forecast, index=forecast_index)
            rmse, mae = evaluate_model(test_data.values, arima_pred.values)
            accuracies['ARIMA'] = {'RMSE': rmse, 'MAE': mae}
            residuals_data['ARIMA'] = full_arima_fitted.resid
            
            # Display forecast chart
            fig_arima = go.Figure()
            fig_arima.add_trace(go.Scatter(x=df_raw.index, y=df_raw['Close'], name='Historical', line=dict(color='blue')))
            fig_arima.add_trace(go.Scatter(x=forecast_index, y=arima_forecast, name='ARIMA Forecast', line=dict(color='red')))
            fig_arima.update_layout(title='ARIMA Forecast', xaxis_title='Date', yaxis_title='Price ($)', height=400)
            st.plotly_chart(fig_arima, use_container_width=True)
            
            st.success(f"✅ ARIMA completed - RMSE: {rmse:.2f}, MAE: {mae:.2f}")
            
        except Exception as e:
            st.error(f"❌ ARIMA model failed: {str(e)}")

if run_sarima:
    st.subheader("🔮 SARIMA Model")
    with st.spinner("Training SARIMA model..."):
        try:
            # Find optimal parameters
            sarima_params, seasonal_params = find_best_sarima_params(train_data)
            st.write(f"**Optimal SARIMA Order:** {sarima_params}")
            st.write(f"**Seasonal Order:** {seasonal_params}")
            
            # Fit model on training data
            sarima_model = SARIMAX(train_data, order=sarima_params, seasonal_order=seasonal_params)
            sarima_fitted = sarima_model.fit(disp=False, maxiter=50)
            
            # Predict on test data
            sarima_pred = sarima_fitted.forecast(steps=len(test_data))
            
            # Fit on full data and forecast future
            full_sarima_model = SARIMAX(df_raw['Close'], order=sarima_params, seasonal_order=seasonal_params)
            full_sarima_fitted = full_sarima_model.fit(disp=False, maxiter=50)
            sarima_forecast = full_sarima_fitted.forecast(steps=forecast_days)
            
            # Store results
            forecasts['SARIMA'] = pd.Series(sarima_forecast, index=forecast_index)
            rmse, mae = evaluate_model(test_data.values, sarima_pred.values)
            accuracies['SARIMA'] = {'RMSE': rmse, 'MAE': mae}
            residuals_data['SARIMA'] = full_sarima_fitted.resid
            
            # Display forecast chart
            fig_sarima = go.Figure()
            fig_sarima.add_trace(go.Scatter(x=df_raw.index, y=df_raw['Close'], name='Historical', line=dict(color='blue')))
            fig_sarima.add_trace(go.Scatter(x=forecast_index, y=sarima_forecast, name='SARIMA Forecast', line=dict(color='orange')))
            fig_sarima.update_layout(title='SARIMA Forecast', xaxis_title='Date', yaxis_title='Price ($)', height=400)
            st.plotly_chart(fig_sarima, use_container_width=True)
            
            st.success(f"✅ SARIMA completed - RMSE: {rmse:.2f}, MAE: {mae:.2f}")
            
        except Exception as e:
            st.error(f"❌ SARIMA model failed: {str(e)}")

if run_prophet:
    st.subheader("🔮 Prophet Model")
    with st.spinner("Training Prophet model..."):
        try:
            # Prepare data for Prophet
            prophet_data = df_raw.reset_index()
            prophet_data.columns = ['ds', 'y']
            
            # Split data
            prophet_train = prophet_data.iloc[:train_size]
            prophet_test = prophet_data.iloc[train_size:]
            
            # Fit Prophet model
            prophet_model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
            prophet_model.fit(prophet_train)
            
            # Make future dataframe
            future_dates = prophet_model.make_future_dataframe(periods=forecast_days)
            prophet_forecast_full = prophet_model.predict(future_dates)
            
            # Extract forecast values
            prophet_forecast = prophet_forecast_full.iloc[-forecast_days:]['yhat'].values
            
            # Store results
            forecasts['Prophet'] = pd.Series(prophet_forecast, index=forecast_index)
            
            # Calculate accuracy on test set
            prophet_test_pred = prophet_forecast_full.iloc[train_size:len(prophet_data)]['yhat']
            if len(prophet_test_pred) > 0:
                rmse, mae = evaluate_model(prophet_test['y'].values, prophet_test_pred.values)
                accuracies['Prophet'] = {'RMSE': rmse, 'MAE': mae}
            
            # Calculate residuals
            prophet_fitted_values = prophet_forecast_full.iloc[:len(prophet_data)]['yhat']
            prophet_residuals = prophet_data['y'] - prophet_fitted_values
            prophet_residuals.index = prophet_data['ds']
            residuals_data['Prophet'] = prophet_residuals
            
            # Display forecast chart
            fig_prophet = go.Figure()
            fig_prophet.add_trace(go.Scatter(x=df_raw.index, y=df_raw['Close'], name='Historical', line=dict(color='blue')))
            fig_prophet.add_trace(go.Scatter(x=forecast_index, y=prophet_forecast, name='Prophet Forecast', line=dict(color='green')))
            fig_prophet.update_layout(title='Prophet Forecast', xaxis_title='Date', yaxis_title='Price ($)', height=400)
            st.plotly_chart(fig_prophet, use_container_width=True)
            
            if 'Prophet' in accuracies:
                st.success(f"✅ Prophet completed - RMSE: {accuracies['Prophet']['RMSE']:.2f}, MAE: {accuracies['Prophet']['MAE']:.2f}")
            else:
                st.success("✅ Prophet completed")
            
        except Exception as e:
            st.error(f"❌ Prophet model failed: {str(e)}")

if run_lstm:
    st.subheader("🔮 LSTM Model")
    with st.spinner("Training LSTM model..."):
        try:
            # Prepare data for LSTM
            data = df_raw['Close'].values.reshape(-1, 1)
            data_min, data_max = data.min(), data.max()
            scaled_data = (data - data_min) / (data_max - data_min)
            
            # Create sequences
            sequence_length = 60
            X, y = [], []
            for i in range(sequence_length, len(scaled_data) - 30):
                X.append(scaled_data[i-sequence_length:i, 0])
                y.append(scaled_data[i, 0])
            
            X = np.array(X)
            y = np.array(y)
            
            # Reshape for LSTM
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            # Build and train LSTM model
            lstm_model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
                LSTM(50, return_sequences=False),
                Dense(25),
                Dense(1)
            ])
            
            lstm_model.compile(optimizer='adam', loss='mean_squared_error')
            lstm_model.fit(X, y, batch_size=32, epochs=10, verbose=0)
            
            # Generate predictions
            last_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
            lstm_predictions = []
            
            for _ in range(forecast_days):
                pred = lstm_model.predict(last_sequence, verbose=0)[0, 0]
                lstm_predictions.append(pred)
                
                # Update sequence for next prediction
                last_sequence = np.roll(last_sequence, -1, axis=1)
                last_sequence[0, -1, 0] = pred
            
            # Inverse transform predictions
            lstm_predictions = np.array(lstm_predictions)
            lstm_forecast = lstm_predictions * (data_max - data_min) + data_min
            
            # Store results
            forecasts['LSTM'] = pd.Series(lstm_forecast.flatten(), index=forecast_index)
            
            # Calculate accuracy on test set
            test_X = []
            for i in range(len(scaled_data) - 30, len(scaled_data)):
                if i >= sequence_length:
                    test_X.append(scaled_data[i-sequence_length:i, 0])
            
            if len(test_X) > 0:
                test_X = np.array(test_X).reshape((len(test_X), sequence_length, 1))
                lstm_test_pred = lstm_model.predict(test_X, verbose=0)
                lstm_test_pred = lstm_test_pred.flatten() * (data_max - data_min) + data_min
                
                actual_test = data[-len(lstm_test_pred):].flatten()
                rmse, mae = evaluate_model(actual_test, lstm_test_pred)
                accuracies['LSTM'] = {'RMSE': rmse, 'MAE': mae}
            
            # Calculate residuals
            train_predictions = lstm_model.predict(X, verbose=0)
            train_predictions = train_predictions.flatten() * (data_max - data_min) + data_min
            actual_train = data[sequence_length:len(scaled_data)-30].flatten()
            lstm_residuals = actual_train - train_predictions
            residual_dates = df_raw.index[sequence_length:len(scaled_data)-30]
            residuals_data['LSTM'] = pd.Series(lstm_residuals, index=residual_dates)
            
            # Display forecast chart
            fig_lstm = go.Figure()
            fig_lstm.add_trace(go.Scatter(x=df_raw.index, y=df_raw['Close'], name='Historical', line=dict(color='blue')))
            fig_lstm.add_trace(go.Scatter(x=forecast_index, y=lstm_forecast.flatten(), name='LSTM Forecast', line=dict(color='purple')))
            fig_lstm.update_layout(title='LSTM Forecast', xaxis_title='Date', yaxis_title='Price ($)', height=400)
            st.plotly_chart(fig_lstm, use_container_width=True)
            
            if 'LSTM' in accuracies:
                st.success(f"✅ LSTM completed - RMSE: {accuracies['LSTM']['RMSE']:.2f}, MAE: {accuracies['LSTM']['MAE']:.2f}")
            else:
                st.success("✅ LSTM completed")
            
        except Exception as e:
            st.error(f"❌ LSTM model failed: {str(e)}")

# Combined forecast comparison
if forecasts:
    st.subheader("📊 Combined Forecast Comparison")
    
    # Create combined forecast chart
    fig_combined = go.Figure()
    fig_combined.add_trace(go.Scatter(x=df_raw.index, y=df_raw['Close'], name='Historical Data', line=dict(color='blue', width=2)))
    
    colors = ['red', 'orange', 'green', 'purple']
    for i, (model_name, forecast_data) in enumerate(forecasts.items()):
        fig_combined.add_trace(go.Scatter(
            x=forecast_data.index, 
            y=forecast_data.values, 
            name=f'{model_name} Forecast',
            line=dict(color=colors[i % len(colors)], width=2)
        ))
    
    fig_combined.update_layout(
        title='All Model Forecasts Comparison',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_combined, use_container_width=True)
    
    # Forecast statistics table
    st.subheader("📋 Forecast Statistics")
    forecast_df = pd.DataFrame(forecasts)
    if not forecast_df.empty:
        st.dataframe(forecast_df.describe().round(2))

# Model accuracy comparison
if accuracies:
    st.subheader("🎯 Model Accuracy Comparison")
    
    accuracy_df = pd.DataFrame(accuracies).T
    st.dataframe(accuracy_df.round(2))
    
    # Best model determination
    if 'RMSE' in accuracy_df.columns:
        best_model = accuracy_df['RMSE'].idxmin()
        best_rmse = accuracy_df.loc[best_model, 'RMSE']
        st.success(f"🏆 **Best performing model:** {best_model} (RMSE: {best_rmse:.2f})")

# Residuals analysis
if residuals_data:
    st.subheader("📉 Residuals Analysis")
    
    # Create residuals comparison chart
    fig_residuals = make_subplots(
        rows=len(residuals_data), 
        cols=1,
        subplot_titles=[f'{model} Residuals' for model in residuals_data.keys()],
        vertical_spacing=0.05
    )
    
    for i, (model_name, residuals) in enumerate(residuals_data.items(), 1):
        fig_residuals.add_trace(
            go.Scatter(x=residuals.index, y=residuals.values, name=f'{model_name} Residuals', line=dict(width=1)),
            row=i, col=1
        )
        fig_residuals.add_hline(y=0, line_dash="dash", line_color="black", row=i, col=1)
    
    fig_residuals.update_layout(height=200 * len(residuals_data), showlegend=False)
    fig_residuals.update_xaxes(title_text="Date")
    fig_residuals.update_yaxes(title_text="Residuals")
    
    st.plotly_chart(fig_residuals, use_container_width=True)

# Model information
st.subheader("ℹ️ Model Information")
st.markdown("""
**ARIMA (AutoRegressive Integrated Moving Average):** 
- Uses optimal parameters found through AIC minimization
- Good for stationary time series data
- Captures linear relationships and trends

**SARIMA (Seasonal ARIMA):** 
- Includes seasonal components for better handling of periodic patterns
- Extension of ARIMA with seasonal parameters
- Better for data with clear seasonal patterns

**Prophet:** 
- Facebook's time series forecasting tool
- Automatic seasonality detection (daily, weekly, yearly)
- Robust to missing data and outliers
- Good for business time series with strong seasonal effects

**LSTM (Long Short-Term Memory):** 
- Neural network for capturing long-term dependencies
- Can learn complex non-linear patterns
- Requires more data for training
- Good for capturing complex temporal relationships
""")

st.markdown("---")
st.markdown("**Note:** All models are trained on historical data and forecasts are predictions. Past performance does not guarantee future results. Please consult with financial advisors before making investment decisions.")