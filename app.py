import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# --- Configuration ---
st.set_page_config(
    page_title="StockPro: AI-Powered Analytics",
    page_icon="hmC",
    layout="wide"
)

# --- CSS Styling (Pro Look) ---
st.markdown("""
<style>
    .metric-container {
        background-color: #262730;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
    .stMetric {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data.reset_index(inplace=True)
    return data

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def train_model(data):
    # Prepare data for ML (Predicting 'Close' based on Date integer)
    df_ml = data[['Date', 'Close']].copy()
    df_ml['Date_Ordinal'] = pd.to_datetime(df_ml['Date']).map(pd.Timestamp.toordinal)
    
    X = df_ml[['Date_Ordinal']]
    y = df_ml['Close']
    
    # Train Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model

# --- Sidebar Controls ---
st.sidebar.header("🔍 Stock Configuration")
ticker = st.sidebar.text_input("Enter Ticker Symbol", "AAPL").upper()
years = st.sidebar.slider("Historical Data (Years)", 1, 5, 2)
forecast_days = st.sidebar.slider("Forecast Days", 7, 90, 30)

# Dates
start_date = date.today() - timedelta(days=years*365)
end_date = date.today()

# --- Main Logic ---
st.title(f"📈 StockPro: {ticker} Analysis")

if ticker:
    try:
        # Load Data
        data = load_data(ticker, start_date, end_date)
        
        if data.empty:
            st.warning("No data found. Please check the ticker symbol.")
            st.stop()
            
       # --- Corrected Data Extraction ---
        # We use float() to ensure we have a raw number, not a pandas Series
        current_price = float(data['Close'].iloc[-1])
        prev_price = float(data['Close'].iloc[-2])
        
        delta = current_price - prev_price
        delta_percent = (delta / prev_price) * 100
        
        # --- Dashboard Metrics ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", f"${current_price:.2f}", f"{delta:.2f} ({delta_percent:.2f}%)")
        col2.metric("High (24h)", f"${float(data['High'].iloc[-1]):.2f}")
        col3.metric("Low (24h)", f"${float(data['Low'].iloc[-1]):.2f}")
        col4.metric("Volume", f"{int(data['Volume'].iloc[-1]):,}")
        col4.metric("Volume", f"{data['Volume'].iloc[-1]:,}")

        # --- Tabs for Content ---
        tab1, tab2, tab3 = st.tabs(["📊 Technical Charts", "🔮 AI Prediction", "💾 Raw Data"])

        with tab1:
            st.subheader("Interactive Technical Analysis")
            
            # Candlestick Chart
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=data['Date'],
                open=data['Open'], high=data['High'],
                low=data['Low'], close=data['Close'],
                name='Market Data'
            ))
            
            # Moving Averages Checkboxes
            col_ma1, col_ma2 = st.columns(2)
            if col_ma1.checkbox("Show 50-Day MA", value=True):
                ma50 = data['Close'].rolling(window=50).mean()
                fig.add_trace(go.Scatter(x=data['Date'], y=ma50, mode='lines', name='50-Day MA', line=dict(color='orange')))
            
            if col_ma2.checkbox("Show 200-Day MA"):
                ma200 = data['Close'].rolling(window=200).mean()
                fig.add_trace(go.Scatter(x=data['Date'], y=ma200, mode='lines', name='200-Day MA', line=dict(color='green')))
            
            fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
            # RSI Indicator
            st.subheader("Relative Strength Index (RSI)")
            data['RSI'] = calculate_rsi(data)
            fig_rsi = go.Figure(go.Scatter(x=data['Date'], y=data['RSI'], name='RSI', line=dict(color='purple')))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
            fig_rsi.update_layout(height=300, template="plotly_dark", yaxis_title="RSI Score")
            st.plotly_chart(fig_rsi, use_container_width=True)

        with tab2:
            st.subheader("AI Trend Forecast")
            st.write("This model uses a **Random Forest Regressor** to estimate future price trends based on historical movement.")
            
            if st.button("Run Prediction Model"):
                with st.spinner("Training AI Model..."):
                    model = train_model(data)
                    
                    # Create Future Dates
                    last_date = data['Date'].iloc[-1]
                    future_dates = [last_date + timedelta(days=x) for x in range(1, forecast_days + 1)]
                    future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
                    
                    # Predict
                    predictions = model.predict(future_ordinals)
                    
                    # Plot Forecast
                    fig_pred = go.Figure()
                    
                    # Historical
                    fig_pred.add_trace(go.Scatter(
                        x=data['Date'], y=data['Close'],
                        mode='lines', name='Historical Data',
                        line=dict(color='blue')
                    ))
                    
                    # Predicted
                    fig_pred.add_trace(go.Scatter(
                        x=future_dates, y=predictions,
                        mode='lines+markers', name='AI Forecast',
                        line=dict(color='red', dash='dot')
                    ))
                    
                    fig_pred.update_layout(
                        title=f"{ticker} Price Prediction (Next {forecast_days} Days)",
                        xaxis_title="Date",
                        yaxis_title="Price (USD)",
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                    # Display Raw Forecast Data
                    df_forecast = pd.DataFrame({"Date": future_dates, "Predicted Price": predictions})
                    st.dataframe(df_forecast)

        with tab3:
            st.subheader("Raw Data Inspector")
            st.dataframe(data.sort_values(by='Date', ascending=False))
            
            # CSV Download
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Dataset as CSV",
                data=csv,
                file_name=f'{ticker}_stock_data.csv',
                mime='text/csv',
            )

    except Exception as e:
        st.error(f"Error fetching data: {e}")
else:
    st.info("Enter a stock ticker in the sidebar to begin.")
