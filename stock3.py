import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas_ta as ta
import os

# Set page config
st.set_page_config(page_title="Yahoo Finance Stock Analysis", layout="wide", 
                  page_icon="📊", initial_sidebar_state="expanded")

# Set up session for API calls
def get_session():
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.1)
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

# Fetch stock data with short cache for live updates
@st.cache_data(ttl=15, show_spinner=False)  # Cache expires every 15 seconds
def get_stock_data(ticker, interval='1m', period='7d'):
    try:
        session = get_session()
        data = yf.download(ticker, period=period, interval=interval, session=session)
        if len(data) > 0:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            return data.dropna()  # Drop any NaN values
        st.error(f"No data returned for {ticker}. Check ticker or market hours (9:30 AM - 4:00 PM EST for intraday data).")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

# Fetch tickers (cached for 24 hours)
@st.cache_data(ttl=86400)
def get_tickers_cached(_session):
    try:
        if os.path.exists("yfinance_tickers.csv"):
            file_time = os.path.getmtime("yfinance_tickers.csv")
            if time.time() - file_time < 86400:
                df = pd.read_csv("yfinance_tickers.csv")
                return df["Symbol"].tolist()
        
        common_tickers = [
            "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "TSLA", "NVDA", 
            "AMD", "INTC", "CSCO", "ORCL", "IBM", "ADBE", "CRM", "NFLX",
            "JPM", "BAC", "WFC", "C", "GS", "V", "MA", "PG", "JNJ", "UNH",
            "CVX", "XOM", "WMT", "HD", "MCD", "KO", "PEP", "DIS", "VZ", "T",
            "BABA", "TCEHY", "TSM", "TM", "BHP", "BP", "NVO", "SAP", "RY", "TD",
            "ROKU", "SNAP", "TWTR", "SQ", "SHOP", "UBER", "LYFT", "PINS", "ZM", "DOCU",
            "ETSY", "PTON", "BYND", "DKNG", "CHWY", "PLTR", "Z", "ABNB", "DASH", "COIN",
            "SPY", "QQQ", "DIA", "IWM", "EEM", "GLD", "SLV", "USO", "VTI", "VOO",
            "VEA", "VWO", "BND", "TLT", "HYG", "LQD", "XLK", "XLF", "XLE", "XLV",
            "BRK-B", "LLY", "AVGO", "COST", "ABBV", "MRK", "PFE", "TMO", "BMY", "RTX",
            "COP", "AMGN", "HON", "CAT", "DE", "SBUX", "GE", "BA", "QCOM",
            "F", "GM", "DOW", "MMM", "MO", "KHC", "CL", "KMB", "GIS"
        ]
        
        common_tickers = sorted(list(set(common_tickers)))
        df = pd.DataFrame({"Symbol": common_tickers})
        df.to_csv("yfinance_tickers.csv", index=False)
        return common_tickers
        
    except Exception as e:
        st.error(f"Error fetching tickers: {str(e)}")
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT"]

# Calculate MACD and RSI using pandas_ta with error handling
def calculate_indicators(data):
    if data.empty or len(data) < 26:  # Minimum 26 periods for MACD
        return data
    try:
        data.ta.macd(fast=7, slow=25, signal=9, append=True)
        data.ta.rsi(length=6, append=True)
        return data.dropna()  # Drop any NaN values after calculation
    except Exception as e:
        st.warning(f"Failed to calculate indicators due to insufficient or invalid data: {e}")
        return data

# Create Interactive Chart
def create_interactive_chart(data, ticker, interval):
    if data.empty or len(data) < 2:
        st.error("Insufficient data to create chart.")
        return go.Figure()

    has_indicators = all(col in data.columns for col in ['MACD_7_25_9', 'MACDs_7_25_9', 'RSI_6'])

    fig = make_subplots(rows=3, cols=1, 
                        specs=[[{"type": "candlestick"}],
                               [{"type": "bar"}],
                               [{"type": "scatter"}]],
                        vertical_spacing=0.15,  # Increased spacing for gap
                        subplot_titles=('Candlestick', 'Volume', 'Indicators'))

    # Candlestick Chart (Row 1)
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], 
                                 low=data['Low'], close=data['Close'], 
                                 increasing_line_color='#00FF00',  # Bright green
                                 decreasing_line_color='#FF4040',  # Vibrant red
                                 name="Price"),
                  row=1, col=1)

    # Volume Chart (Row 2)
    volume_colors = []
    for i in range(len(data)):
        open_price = float(data['Open'].iloc[i])
        close_price = float(data['Close'].iloc[i])
        volume_colors.append('#00CED1' if open_price <= close_price  # Neon teal
                             else '#FF7F50')  # Coral
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], marker=dict(color=volume_colors), 
                         opacity=0.7, name='Volume'),
                  row=2, col=1)

    # Indicators Chart (Row 3)
    if has_indicators:
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD_7_25_9'], mode='lines', 
                                 line=dict(color='#1E90FF', width=2.5),  # Bold blue
                                 name='MACD'),
                      row=3, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['MACDs_7_25_9'], mode='lines', 
                                 line=dict(color='#FF4500', width=2.5),  # Bold orange-red
                                 name='Signal'),
                      row=3, col=1)
        fig.add_trace(go.Bar(x=data.index, y=data['MACDh_7_25_9'], marker=dict(color='#A9A9A9'), 
                             name='Histogram'),
                      row=3, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['RSI_6'], mode='lines', 
                                 line=dict(color='#FFFF00', width=2.5),  # Bright yellow
                                 name='RSI'),
                      row=3, col=1)
        fig.add_shape(type='line', x0=data.index[0], y0=70, x1=data.index[-1], y1=70, 
                      xref='x', yref='y3', line=dict(color='#FF0000', width=2, dash='dash'), 
                      row=3, col=1)
        fig.add_shape(type='line', x0=data.index[0], y0=30, x1=data.index[-1], y1=30, 
                      xref='x', yref='y3', line=dict(color='#00FF00', width=2, dash='dash'), 
                      row=3, col=1)
        fig.add_annotation(x=data.index[0], y=70, xref='x', yref='y3', text='OVERBOUGHT 70', 
                           showarrow=False, font=dict(size=12, color='#FF0000', family='Arial Black'), 
                           row=3, col=1)
        fig.add_annotation(x=data.index[0], y=30, xref='x', yref='y3', text='OVERSOLD 30', 
                           showarrow=False, font=dict(size=12, color='#00FF00', family='Arial Black'), 
                           row=3, col=1)

    fig.update_layout(
        title=f"<b><span style='color:#FFD700; font-size:24px'>{ticker} - {interval.upper()} Data</span></b>",
        template='plotly_dark',
        paper_bgcolor='rgba(0, 0, 20, 0.9)',
        plot_bgcolor='rgba(0, 0, 30, 0.7)',
        xaxis_rangeslider_visible=False,
        height=900,
        margin=dict(l=20, r=20, t=80, b=20),
        hovermode='x unified',
        showlegend=True,
        xaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.2)', 
                   zeroline=False, type='date', tickfont=dict(size=12, color='#FFFFFF')),
        xaxis2=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.2)', 
                    zeroline=False, type='date', tickfont=dict(size=12, color='#FFFFFF')),
        xaxis3=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.2)', 
                    zeroline=False, type='date', tickfont=dict(size=12, color='#FFFFFF')),
        yaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.2)', zeroline=False, side='right',
                   tickfont=dict(size=12, color='#FFFFFF')),
        yaxis2=dict(showgrid=False, zeroline=False, side='right',
                    tickfont=dict(size=12, color='#FFFFFF')),
        yaxis3=dict(showgrid=False, zeroline=False, side='right', domain=[0, 0.25],
                    tickfont=dict(size=12, color='#FFFFFF')),
        legend=dict(x=1.1, y=1, xanchor='left', yanchor='top', font=dict(size=12, color='#FFD700'))
    )

    fig.update_traces(line=dict(width=2.5), selector=dict(type='scatter'))
    fig.update_traces(marker=dict(line=dict(width=1, color='rgba(255,255,255,0.3)')), selector=dict(type='bar'))

    return fig

def main():
    st.title("📊 Yahoo Finance Stock Analysis")
    
    # Sidebar with stock selection, interval, and time range
    st.sidebar.header("Stock Selection")
    if st.sidebar.button("Refresh Yahoo Finance Tickers"):
        if os.path.exists("yfinance_tickers.csv"):
            os.remove("yfinance_tickers.csv")
        st.success("Ticker list will be refreshed!")
        st.experimental_rerun()

    tickers = get_tickers_cached(None)
    ticker_df = pd.DataFrame({"Symbol": tickers})
    csv_data = ticker_df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="Download Ticker List",
        data=csv_data,
        file_name="yahoo_finance_tickers.csv",
        mime="text/csv"
    )

    ticker_search = st.sidebar.text_input("Search Tickers:")
    if ticker_search:
        filtered_tickers = [t for t in tickers if ticker_search.upper() in t]
        display_tickers = filtered_tickers[:100] if len(filtered_tickers) > 100 else filtered_tickers
        if len(filtered_tickers) > 100:
            st.sidebar.info(f"Found {len(filtered_tickers)} matches. Showing first 100.")
    else:
        popular = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT"]
        other_tickers = [t for t in tickers if t not in popular]
        display_tickers = popular + other_tickers[:90]

    selected_ticker = st.sidebar.selectbox("Select Stock", display_tickers)
    
    # Interval and Time Range Filters
    st.sidebar.header("Chart Settings")
    interval = st.sidebar.selectbox("Interval", ['1m', '15m', '1d'], 
                                   help="Select interval: 1m (1-minute), 15m (15-minute), 1d (1-day).")
    time_range = st.sidebar.selectbox("Time Range", ['1h', '1d', '1w', '1mo', '1y'], 
                                     help="Select time range for the chart.")

    # Validate interval and time range compatibility
    period = time_range
    if time_range == '1h':
        if interval != '1m':
            st.sidebar.warning("1-hour range requires 1-minute interval. Setting interval to 1m.")
            interval = '1m'
        period = '1d'  # yfinance doesn't support '1h', so fetch 1d and filter
    elif time_range == '1d':
        if interval == '1d':
            st.sidebar.warning("1-day range requires a smaller interval (1m or 15m). Setting interval to 15m.")
            interval = '15m'
        period = '2d'  # Extend to 2 days to ensure enough data for 1d range
    elif time_range == '1w':
        if interval == '1m':
            period = '7d'
        elif interval == '15m':
            period = '7d'
        else:  # interval == '1d'
            period = '7d'
    elif time_range == '1mo':
        if interval == '1m':
            st.sidebar.warning("1-month range with 1m interval is limited to 7 days. Setting time range to 1w.")
            time_range = '1w'
            period = '7d'
        elif interval == '15m':
            period = '1mo'
        else:  # interval == '1d'
            period = '1mo'
    elif time_range == '1y':
        if interval in ['1m', '15m']:
            st.sidebar.warning("1-year range requires 1d interval. Setting interval to 1d.")
            interval = '1d'
        period = '1y'

    # KPI Placeholder (to avoid duplicates)
    kpi_placeholder = st.empty()

    # Placeholders for chart and counter
    chart_placeholder = st.empty()
    counter_placeholder = st.empty()

    # Update function
    def update_chart():
        with st.spinner(f"Loading {selected_ticker} live data..."):
            data = get_stock_data(selected_ticker, interval, period)
            if not data.empty and len(data) > 1:
                # Filter data based on time range (for 1h)
                if time_range == '1h':
                    end_time = data.index[-1]
                    start_time = end_time - pd.Timedelta(hours=1)
                    data = data.loc[start_time:end_time]

                data = calculate_indicators(data)
                current_price = float(data['Close'].iloc[-1])
                price_change = current_price - float(data['Open'].iloc[0])
                price_change_pct = (price_change / float(data['Open'].iloc[0])) * 100 if float(data['Open'].iloc[0]) != 0 else 0
                high_24h = float(data['High'].max())
                low_24h = float(data['Low'].min())

                # Update KPIs using placeholder with corrected styling
                with kpi_placeholder.container():
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Price", f"${current_price:.2f}")
                    col2.metric("Change", f"${price_change:.2f} ({price_change_pct:.2f}%)")
                    col3.metric("High", f"${high_24h:.2f}")
                    col4.metric("Low", f"${low_24h:.2f}")
                    st.markdown(f"<p style='color:#FFFFFF; font-size:16px'>Last Update: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>", 
                                unsafe_allow_html=True)

                    # Add colored labels using markdown below each metric
                    col1.markdown("<p style='color:#FFD700; font-size:12px'>Price</p>", unsafe_allow_html=True)
                    col2.markdown("<p style='color:#FF4500; font-size:12px'>Change</p>", unsafe_allow_html=True)
                    col3.markdown("<p style='color:#00FF00; font-size:12px'>High</p>", unsafe_allow_html=True)
                    col4.markdown("<p style='color:#1E90FF; font-size:12px'>Low</p>", unsafe_allow_html=True)

                # Update Chart
                fig = create_interactive_chart(data, selected_ticker, interval)
                chart_placeholder.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"No live data for {selected_ticker}. Check ticker or market hours (9:30 AM - 4:00 PM EST for intraday data).")

    # Initial update
    last_refresh = time.time()
    update_chart()
    
    # Live refresh (fixed 15-second interval)
    st.write("🔄 <span style='color:#FFD700'>Auto-refreshing every 15 seconds...</span>", unsafe_allow_html=True)
    while True:
        current_time = time.time()
        elapsed = current_time - last_refresh
        remaining = max(0, 15 - elapsed)
        counter_placeholder.write(f"<span style='color:#00FF00'>Next Update in: {int(remaining)} seconds</span>", 
                                 unsafe_allow_html=True)
        time.sleep(1)
        if elapsed >= 15:
            update_chart()
            last_refresh = time.time()

if __name__ == "__main__":
    main()