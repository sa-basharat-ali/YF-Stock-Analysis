import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os
import copy
from scipy.interpolate import interp1d
import random
import json
import functools  # For partial function application
import logging
import io
import sys

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add debug information at startup
logger.info("Starting application...")
logger.info(f"Python version: {sys.version}")
logger.info(f"Pandas version: {pd.__version__}")
logger.info(f"NumPy version: {np.__version__}")
logger.info(f"Plotly version: {plotly.__version__}")
logger.info(f"YFinance version: {yf.__version__ if hasattr(yf, '__version__') else 'unknown'}")

# Set page config with error capture
try:
    st.set_page_config(
        page_title="Yahoo Finance Stock Analysis",
        layout="wide",
        page_icon="📊",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/sa-basharat-ali/YF-Stock-Analysis/issues',
            'Report a bug': 'https://github.com/sa-basharat-ali/YF-Stock-Analysis/issues',
            'About': 'Stock analysis tool using Yahoo Finance data'
        }
    )
    logger.info("Page config set successfully")
except Exception as e:
    logger.error(f"Error setting page config: {e}")

# Add error handling for session state
try:
    if 'data_buffer' not in st.session_state:
        st.session_state.data_buffer = None
        logger.info("Initialized data_buffer in session state")
    
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
        logger.info("Initialized current_index in session state")
    
    if 'animation_speed' not in st.session_state:
        st.session_state.animation_speed = 0.5
        logger.info("Initialized animation_speed in session state")
    
    if 'is_animating' not in st.session_state:
        st.session_state.is_animating = False
        logger.info("Initialized is_animating in session state")
    
    if 'buffer_loaded' not in st.session_state:
        st.session_state.buffer_loaded = False
        logger.info("Initialized buffer_loaded in session state")
except Exception as e:
    logger.error(f"Error initializing session state: {e}")
    st.error("Error initializing application state. Please refresh the page.")

# Set up improved session for API calls
def get_session():
    session = requests.Session()
    
    # More robust retry configuration with exponential backoff
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    
    adapter = HTTPAdapter(max_retries=retries, pool_maxsize=10)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    # Add custom headers to mimic a browser
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
    })
    
    # Add longer timeout (30 seconds)
    session.request = functools.partial(session.request, timeout=30)
    
    return session

# Custom implementation of technical indicators
def calculate_macd(data, fast=7, slow=25, signal=9):
    """Calculate MACD using pandas EMA"""
    try:
        ema_short = data['Close'].ewm(span=fast, adjust=False).mean()
        ema_long = data['Close'].ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_short - ema_long
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'MACD_7_25_9': macd_line,
            'MACDs_7_25_9': signal_line,
            'MACDh_7_25_9': histogram
        })
    except Exception as e:
        logger.error(f"Error calculating MACD: {e}")
        return pd.DataFrame()

def calculate_rsi(data, periods=6):
    """Calculate RSI using pandas"""
    try:
        delta = data['Close'].diff()
        gains = delta.copy()
        losses = delta.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        avg_gains = gains.ewm(alpha=1/periods, adjust=False).mean()
        avg_losses = losses.ewm(alpha=1/periods, adjust=False).mean()
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return pd.Series(rsi, name='RSI_6')
    except Exception as e:
        logger.error(f"Error calculating RSI: {e}")
        return pd.Series(dtype=float)

def calculate_indicators(data):
    """Calculate technical indicators"""
    if data.empty:
        logger.info("Empty data received in calculate_indicators")
        return data
    
    try:
        result_data = data.copy()
        
        # Remove any existing indicator columns to prevent duplicates
        indicator_cols = ['MACD_7_25_9', 'MACDs_7_25_9', 'MACDh_7_25_9', 'RSI_6']
        for col in indicator_cols:
            if col in result_data.columns:
                result_data = result_data.drop(columns=[col])
        
        # Adjust MACD periods based on data length
        if len(result_data) < 26:
            fast, slow, signal = 5, 13, 5  # Use shorter periods for smaller datasets
        else:
            fast, slow, signal = 7, 25, 9  # Default periods
        
        exp1 = result_data['Close'].ewm(span=fast, adjust=False).mean()
        exp2 = result_data['Close'].ewm(span=slow, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        result_data['MACD_7_25_9'] = macd_line
        result_data['MACDs_7_25_9'] = signal_line
        result_data['MACDh_7_25_9'] = histogram
        
        # Adjust RSI period based on data length
        rsi_period = min(6, len(result_data) - 1)
        delta = result_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        result_data['RSI_6'] = 100 - (100 / (1 + rs))
        
        result_data = result_data.fillna(method='ffill')
        result_data = result_data.fillna(method='bfill')
        
        logger.info(f"Indicators calculated successfully. Shape: {result_data.shape}")
        return result_data
    
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return data

# Enhanced fetch function with fallbacks
@st.cache_data(ttl=300, show_spinner=False)  # 5 minutes cache
def get_stock_data_with_buffer(ticker, interval='1m', period='1d'):
    """Fetch stock data with improved caching"""
    logger.info(f"Fetching data for {ticker}, {interval}, {period}")
    
    try:
        session = get_session()
        
        # Define valid interval-period combinations
        valid_combinations = {
            '1m': ['1d', '7d'],
            '2m': ['1d', '7d', '60d'],
            '5m': ['1d', '7d', '60d'],
            '15m': ['1d', '7d', '60d'],
            '30m': ['1d', '7d', '60d'],
            '60m': ['1d', '7d', '60d'],
            '90m': ['1d', '7d', '60d', '1mo'],
            '1h': ['1d', '7d', '60d', '1mo'],
            '1d': ['7d', '60d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'],
            '5d': ['1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'],
            '1wk': ['1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'],
            '1mo': ['3mo', '6mo', '1y', '2y', '5y', 'max'],
            '3mo': ['6mo', '1y', '2y', '5y', 'max']
        }

        # Validate and adjust interval-period combination
        if interval in valid_combinations:
            if period not in valid_combinations[interval]:
                suggested_period = valid_combinations[interval][0]
                logger.warning(f"{interval} interval works best with these periods: {valid_combinations[interval]}. Using {suggested_period}.")
                period = suggested_period
        
        # Download data with retries
        for attempt in range(3):
            try:
                data = yf.download(
                    ticker,
                    period=period,
                    interval=interval,
                    session=session,
                    prepost=True,
                    progress=False,
                    threads=False
                )
                
                if len(data) > 0:
                    logger.info(f"Successfully downloaded {len(data)} rows of data")
                    break
                
                logger.warning(f"Empty data received on attempt {attempt + 1}, retrying...")
                time.sleep(2)
            except Exception as e:
                logger.error(f"Error downloading data on attempt {attempt + 1}: {e}")
                if attempt < 2:
                    time.sleep(2)
                else:
                    return pd.DataFrame()
        
        if len(data) == 0:
            logger.warning(f"No data available for {ticker}")
            return pd.DataFrame()
        
        # Clean up column names if multi-index
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Validate required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_cols):
            logger.error(f"Missing columns. Available columns: {data.columns.tolist()}")
            return pd.DataFrame()
        
        # Keep only required columns and remove any NaN values
        data = data[required_cols].copy()
        data = data.dropna()
        
        if len(data) < 2:
            logger.warning("Insufficient data points")
            return pd.DataFrame()
        
        # Calculate indicators only once
        data = calculate_indicators(data)
        
        # Log data information
        logger.info(f"Final data shape: {data.shape}")
        logger.info(f"Time range: {data.index[0]} to {data.index[-1]}")
        logger.info(f"Columns: {data.columns.tolist()}")
        
        return data
    
    except Exception as e:
        logger.error(f"Error in get_stock_data_with_buffer: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60, show_spinner=False)
def get_current_quote(ticker):
    """Get current quote with proper serialization"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.fast_info
        if info is not None:
            # Convert FastInfo object to dictionary
            quote_data = {
                'last_price': getattr(info, 'last_price', None),
                'volume': getattr(info, 'volume', None),
                'timezone': getattr(info, 'timezone', None)
            }
            return quote_data
        return None
    except Exception as e:
        logger.error(f"Error in get_current_quote: {e}")
        return None

@st.cache_data(ttl=86400, max_entries=1)
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
        logger.error(f"Error fetching tickers: {e}")
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT"]

def update_kpis(data, kpi_placeholder):
    if data is None or data.empty:
        return
    
    try:
        current_price = float(data['Close'].iloc[-1])
        price_change = current_price - float(data['Open'].iloc[0])
        price_change_pct = (price_change / float(data['Open'].iloc[0])) * 100 if float(data['Open'].iloc[0]) != 0 else 0
        high_24h = float(data['High'].max())
        low_24h = float(data['Low'].min())
        
        with kpi_placeholder.container():
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Price", f"${current_price:.2f}")
            col2.metric("Change", f"${price_change:.2f} ({price_change_pct:.2f}%)")
            col3.metric("High", f"${high_24h:.2f}")
            col4.metric("Low", f"${low_24h:.2f}")
            st.markdown(f"<p style='color:#FFFFFF; font-size:16px'>Last Update: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>", 
                        unsafe_allow_html=True)
            
            col1.markdown("<p style='color:#FFD700; font-size:12px'>Price</p>", unsafe_allow_html=True)
            col2.markdown("<p style='color:#FF4500; font-size:12px'>Change</p>", unsafe_allow_html=True)
            col3.markdown("<p style='color:#00FF00; font-size:12px'>High</p>", unsafe_allow_html=True)
            col4.markdown("<p style='color:#1E90FF; font-size:12px'>Low</p>", unsafe_allow_html=True)
            
    except Exception as e:
        logger.error(f"Error updating KPIs: {e}")

def update_quick_kpis(current_price, kpi_placeholder):
    if current_price is None:
        return
    
    try:
        data_buffer = st.session_state.data_buffer
        if data_buffer is None or data_buffer.empty:
            return
        
        first_price = float(data_buffer['Open'].iloc[0])
        price_change = current_price - first_price
        price_change_pct = (price_change / first_price) * 100 if first_price != 0 else 0
        high_24h = float(data_buffer['High'].max())
        low_24h = float(data_buffer['Low'].min())
        
        with kpi_placeholder.container():
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Price", f"${current_price:.2f}")
            col2.metric("Change", f"${price_change:.2f} ({price_change_pct:.2f}%)")
            col3.metric("High", f"${high_24h:.2f}")
            col4.metric("Low", f"${low_24h:.2f}")
            st.markdown(f"<p style='color:#FFFFFF; font-size:16px'>Last Update: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>", 
                        unsafe_allow_html=True)
            
            col1.markdown("<p style='color:#FFD700; font-size:12px'>Price</p>", unsafe_allow_html=True)
            col2.markdown("<p style='color:#FF4500; font-size:12px'>Change</p>", unsafe_allow_html=True)
            col3.markdown("<p style='color:#00FF00; font-size:12px'>High</p>", unsafe_allow_html=True)
            col4.markdown("<p style='color:#1E90FF; font-size:12px'>Low</p>", unsafe_allow_html=True)
            
    except Exception as e:
        logger.error(f"Error updating quick KPIs: {e}")

def safe_filter_data(data_buffer, time_range):
    if data_buffer is None or data_buffer.empty:
        return pd.DataFrame()
    
    try:
        if len(data_buffer) < 2:
            return data_buffer
        
        end_time = data_buffer.index[-1]
        
        if time_range == '1h':
            start_time = end_time - pd.Timedelta(hours=1)
        elif time_range == '1d':
            start_time = end_time - pd.Timedelta(days=1)
        elif time_range == '1w':
            start_time = end_time - pd.Timedelta(weeks=1)
        else:
            return data_buffer
        
        filtered_data = data_buffer[data_buffer.index >= start_time].copy()
        
        if filtered_data.empty or len(filtered_data) < 2:
            st.warning("Insufficient data for selected time range. Using all available data.")
            return data_buffer
        
        time_diff = pd.Series(filtered_data.index).diff()
        interval = (filtered_data.index[1] - filtered_data.index[0]).total_seconds() / 60
        max_gap = time_diff.max().total_seconds() / 60
        
        if max_gap > interval * 5:
            st.warning(f"Data contains gaps (max gap: {max_gap:.1f} minutes). Some periods might be missing.")
        
        filtered_data = calculate_indicators(filtered_data)
        
        return filtered_data
    
    except Exception as e:
        logger.error(f"Error in safe_filter_data: {e}")
        return data_buffer

def validate_data_buffer(data_buffer, ticker, interval):
    if data_buffer is None or data_buffer.empty:
        st.warning(f"No data available for {ticker}. Please check market hours or try a different ticker.")
        return False
    
    if len(data_buffer) < 2:
        st.warning(f"Insufficient data points for {ticker}. Please try a different interval or time range.")
        return False
    
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in data_buffer.columns]
    if missing_columns:
        return False
    
    if data_buffer.isnull().any().any():
        data_buffer = data_buffer.ffill().bfill()
    
    return True

def generate_simulated_ticks(data_buffer, num_ticks=5, start_idx=None, volatility=0.0005):
    if data_buffer is None or data_buffer.empty or len(data_buffer) < 2:
        return []
    
    try:
        if start_idx is None:
            start_idx = len(data_buffer) - 2
        
        start_idx = min(max(0, start_idx), len(data_buffer) - 2)
        
        current_candle = data_buffer.iloc[start_idx]
        next_candle = data_buffer.iloc[start_idx + 1]
        
        start_time = pd.Timestamp(data_buffer.index[start_idx])
        end_time = pd.Timestamp(data_buffer.index[start_idx + 1])
        time_delta = (end_time - start_time) / (num_ticks + 1)
        
        ticks = []
        prev_price = float(current_candle['Close'])
        base_volume = float(next_candle['Volume']) / num_ticks
        
        price_range = max(
            abs(float(next_candle['High']) - float(next_candle['Low'])),
            abs(float(next_candle['Close']) - float(current_candle['Close'])),
            0.0001
        )
        
        target_price = float(next_candle['Close'])
        price_diff = target_price - prev_price
        
        for i in range(num_ticks):
            try:
                tick_time = start_time + time_delta * (i + 1)
                progress = (i + 1) / (num_ticks + 1)
                price_movement = price_diff * progress
                noise = np.random.normal(0, volatility * price_range)
                new_price = prev_price + price_movement * 0.7 + noise
                
                new_price = max(min(new_price, float(next_candle['High'])), float(next_candle['Low']))
                vol_factor = 1 + np.random.normal(0, 0.2)
                simulated_volume = max(1, int(base_volume * vol_factor))
                
                tick = {
                    'timestamp': tick_time,
                    'price': new_price,
                    'volume': simulated_volume
                }
                
                ticks.append(tick)
                prev_price = new_price
                
            except Exception as e:
                logger.error(f"Error generating tick {i}: {e}")
                continue
        
        return ticks
    
    except Exception as e:
        logger.error(f"Error in generate_simulated_ticks: {e}")
        return []

def update_candle_with_tick(candle_data, tick):
    if candle_data is None or tick is None:
        return None
    
    try:
        updated = candle_data.copy()
        price = float(tick['price'])
        volume = int(tick['volume'])
        
        updated['Close'] = price
        updated['High'] = max(float(updated['High']), price)
        updated['Low'] = min(float(updated['Low']), price)
        current_volume = int(float(updated['Volume']))
        updated['Volume'] = current_volume + volume
        
        return updated
    
    except Exception as e:
        logger.error(f"Error in update_candle_with_tick: {e}")
        return None

def create_animation_frame(data_buffer, current_index, tick_index=None, tick=None):
    if data_buffer is None or data_buffer.empty:
        return pd.DataFrame()
    
    try:
        current_index = min(max(0, current_index), len(data_buffer) - 1)
        frame = data_buffer.iloc[:current_index+1].copy()
        
        if tick is not None and len(frame) > 0:
            try:
                last_candle = frame.iloc[-1].copy()
                updated_candle = update_candle_with_tick(last_candle, tick)
                if updated_candle is not None:
                    frame.iloc[-1] = updated_candle
            except Exception as e:
                logger.error(f"Error updating candle with tick: {e}")
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in frame.columns for col in required_columns):
            logger.error("Missing required columns in frame")
            return pd.DataFrame()
        
        try:
            macd_data = calculate_macd(frame)
            if not macd_data.empty:
                for col in macd_data.columns:
                    frame[col] = macd_data[col]
            
            rsi_data = calculate_rsi(frame)
            if not rsi_data.empty:
                frame['RSI_6'] = rsi_data
            
            frame = frame.ffill().bfill()
        except Exception as e:
            logger.error(f"Error calculating indicators in animation frame: {e}")
            pass
        
        return frame
    
    except Exception as e:
        logger.error(f"Error in create_animation_frame: {e}")
        return pd.DataFrame()

def create_interactive_chart(data, ticker, interval, is_animation_frame=False, last_price=None, last_tick_time=None):
    if data is None or data.empty or len(data) < 2:
        if not is_animation_frame:
            st.warning("Waiting for sufficient data to create chart...")
        return go.Figure()

    fig = make_subplots(rows=3, cols=1,
                       specs=[[{"type": "candlestick"}],
                             [{"type": "bar"}],
                             [{"type": "scatter"}]],
                       vertical_spacing=0.12,
                       row_heights=[0.5, 0.2, 0.3],
                       subplot_titles=('Price', 'Volume', 'Indicators'))

    # Add candlestick chart
    candlestick = go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        increasing=dict(line=dict(color='#00FF00', width=1), fillcolor='#00FF00'),
        decreasing=dict(line=dict(color='#FF4040', width=1), fillcolor='#FF4040'),
        name="Price"
    )
    fig.add_trace(candlestick, row=1, col=1)

    # Add last price annotation
    if is_animation_frame and len(data) > 0:
        current_price = last_price if last_price is not None else data['Close'].iloc[-1]
        fig.add_annotation(
            x=last_tick_time or data.index[-1],
            y=current_price,
            text=f"${current_price:.2f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#FFFFFF",
            ax=50,
            ay=0,
            bordercolor="#FFFFFF",
            borderwidth=1,
            borderpad=4,
            bgcolor="#000000",
            opacity=0.8,
            font=dict(size=12, color="#FFFFFF"),
            row=1, col=1
        )

    # Add volume bars
    colors = ['#00CED1' if row['Close'] >= row['Open'] else '#FF7F50'
             for _, row in data.iterrows()]
    volume = go.Bar(
        x=data.index,
        y=data['Volume'],
        marker_color=colors,
        marker_line_width=0,
        name='Volume'
    )
    fig.add_trace(volume, row=2, col=1)

    # Add indicators
    if all(col in data.columns for col in ['MACD_7_25_9', 'MACDs_7_25_9', 'RSI_6']):
        # MACD
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MACD_7_25_9'],
            line=dict(color='#1E90FF', width=2),
            name='MACD'
        ), row=3, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MACDs_7_25_9'],
            line=dict(color='#FF4500', width=2),
            name='Signal'
        ), row=3, col=1)
        
        # MACD Histogram
        colors = ['#00CED1' if val >= 0 else '#FF7F50' for val in data['MACDh_7_25_9']]
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['MACDh_7_25_9'],
            marker_color=colors,
            name='Histogram'
        ), row=3, col=1)

        # RSI
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['RSI_6'],
            line=dict(color='#FFFF00', width=2),
            name='RSI'
        ), row=3, col=1)

        # Add RSI levels
        fig.add_hline(y=70, line=dict(color='#FF0000', width=1, dash='dash'), row=3, col=1)
        fig.add_hline(y=30, line=dict(color='#00FF00', width=1, dash='dash'), row=3, col=1)

    # Update layout
    title_text = f"{ticker} - {interval.upper()} Data{' 🔴 LIVE' if is_animation_frame else ''}"
    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.5,
            y=0.98,
            xanchor='center',
            yanchor='top',
            font=dict(size=20, color='#FFD700')
        ),
        template='plotly_dark',
        paper_bgcolor='rgba(0, 0, 20, 0.9)',
        plot_bgcolor='rgba(0, 0, 30, 0.7)',
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True,
        legend=dict(
            x=1.0,
            y=1.0,
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='#FFFFFF',
            font=dict(color='#FFFFFF')
        ),
        margin=dict(l=50, r=50, t=80, b=50),
        uirevision=True  # This helps maintain zoom level during updates
    )

    # Update axes
    for i in range(1, 4):
        axis_props = dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='rgba(128, 128, 128, 0.5)',
        )
        
        # X-axis configuration
        fig.update_xaxes(
            **axis_props,
            row=i, col=1,
            rangeslider_visible=False,
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # Hide weekends
                dict(bounds=[20, 9], pattern="hour") if interval not in ['1d', '5d', '1wk', '1mo', '3mo'] else None
            ]
        )
        
        # Y-axis configuration
        fig.update_yaxes(
            **axis_props,
            row=i, col=1,
            side='right' if i == 3 else 'left',
            tickfont=dict(size=10),
            title_font=dict(size=12)
        )

    return fig

def main():
    try:
        st.title("📊 Yahoo Finance Stock Analysis")
        
        # Create placeholders
        kpi_placeholder = st.empty()
        chart_placeholder = st.empty()
        status_container = st.container()
        animation_status = status_container.empty()
        
        # Sidebar setup
        st.sidebar.header("Stock Selection")
        tickers = get_tickers_cached(None)
        if not tickers:
            st.warning("Could not load tickers. Using default popular tickers.")
            tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT"]
        
        ticker_search = st.sidebar.text_input("Search Tickers:")
        if ticker_search:
            filtered_tickers = [t for t in tickers if ticker_search.upper() in t]
            display_tickers = filtered_tickers[:100] if len(filtered_tickers) > 100 else filtered_tickers
        else:
            popular = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT"]
            other_tickers = [t for t in tickers if t not in popular]
            display_tickers = popular + other_tickers[:90]
        
        selected_ticker = st.sidebar.selectbox("Select Stock", display_tickers)
        interval = st.sidebar.selectbox("Interval", ['1m', '15m', '1d'])
        time_range = st.sidebar.selectbox("Time Range", ['1h', '1d', '1w'])
        
        # Animation settings
        st.sidebar.header("Animation Settings")
        enable_animation = st.sidebar.checkbox("Enable Live Animation", value=True, key="enable_animation")
        animation_speed = st.sidebar.slider("Animation Speed", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
        volatility = st.sidebar.slider("Volatility", min_value=0.0001, max_value=0.001, value=0.0003, step=0.0001)
        
        st.session_state.animation_speed = animation_speed
        
        # Add a stop button for animation
        if enable_animation:
            if st.sidebar.button("Stop Animation"):
                st.session_state.enable_animation = False
                st.rerun()
        
        # Load data buffer
        need_new_data = (
            'data_buffer' not in st.session_state or
            st.session_state.data_buffer is None or
            'current_ticker' not in st.session_state or
            st.session_state.current_ticker != selected_ticker or
            'current_interval' not in st.session_state or
            st.session_state.current_interval != interval
        )
        
        if need_new_data:
            with st.spinner(f"Loading {selected_ticker} data..."):
                data_buffer = get_stock_data_with_buffer(selected_ticker, interval)
                
                if data_buffer is None or data_buffer.empty:
                    st.error(f"No data available for {selected_ticker}. Please try a different ticker or interval.")
                    return
                
                st.session_state.data_buffer = data_buffer
                st.session_state.current_ticker = selected_ticker
                st.session_state.current_interval = interval
                st.session_state.current_index = min(20, max(0, len(data_buffer) - 1))
                st.session_state.last_update = time.time()
                
                # Update KPIs
                update_kpis(data_buffer, kpi_placeholder)
        
        # Filter data based on time range
        filtered_data = safe_filter_data(st.session_state.data_buffer, time_range)
        if filtered_data.empty:
            st.warning("No data available for the selected time range.")
            return
        
        # Update session state with filtered data
        st.session_state.data_buffer = filtered_data
        
        # Display chart
        if not enable_animation:
            fig = create_interactive_chart(st.session_state.data_buffer, selected_ticker, interval)
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            animation_status.markdown("Animation disabled")
        else:
            animation_status.markdown("🔴 Live simulation running...")
            chart_container = chart_placeholder.empty()
            
            current_time = time.time()
            if 'last_update' not in st.session_state or (current_time - st.session_state.last_update) >= animation_speed:
                current_index = st.session_state.current_index
                
                # Generate simulated ticks for smooth animation
                ticks = generate_simulated_ticks(st.session_state.data_buffer, num_ticks=5, start_idx=current_index, volatility=volatility)
                
                if ticks:
                    for tick in ticks:
                        frame = create_animation_frame(st.session_state.data_buffer, current_index, tick=tick)
                        if frame is not None and not frame.empty:
                            fig = create_interactive_chart(frame, selected_ticker, interval, is_animation_frame=True, last_price=tick['price'], last_tick_time=tick['timestamp'])
                            chart_container.plotly_chart(fig, use_container_width=True)
                            st.session_state.last_update = time.time()
                            time.sleep(animation_speed / len(ticks))
                            st.rerun()
                
                current_index += 1
                if current_index >= len(st.session_state.data_buffer) - 1:
                    current_index = 0
                st.session_state.current_index = current_index
                st.session_state.last_update = current_time
                
                st.rerun()
    
    except Exception as e:
        logger.error(f"Main function error: {e}")
        st.error("An unexpected error occurred. Please refresh the page.")

if __name__ == "__main__":
    try:
        os.makedirs('/tmp/streamlit_cache', exist_ok=True)
        main()
    except Exception as e:
        st.error("Application error. Please refresh the page.")
        st.error(f"Details: {str(e)}")