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
import os
import copy
from scipy.interpolate import interp1d
import random
import json
import functools  # For partial function application
import logging
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add numpy compatibility layer - use nan instead of NaN
np.NaN = np.nan  # This is the correct way to handle NaN in newer NumPy versions

# Set page config
st.set_page_config(page_title="Yahoo Finance Stock Analysis", layout="wide", 
                  page_icon="📊", initial_sidebar_state="expanded")

# Create session state variables to store data and animation state
if 'data_buffer' not in st.session_state:
    st.session_state.data_buffer = None
    
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
    
if 'animation_speed' not in st.session_state:
    st.session_state.animation_speed = 0.5  # seconds between frames
    
if 'is_animating' not in st.session_state:
    st.session_state.is_animating = False
    
if 'buffer_loaded' not in st.session_state:
    st.session_state.buffer_loaded = False

# Set up improved session for API calls
def get_session():
    session = requests.Session()
    
    # More robust retry configuration with exponential backoff
    retries = Retry(
        total=5,  # More retry attempts
        backoff_factor=0.5,  # Longer backoff between retries
        status_forcelist=[429, 500, 502, 503, 504],  # Retry on these status codes
        allowed_methods=["HEAD", "GET", "OPTIONS"]  # Allow retries for these methods
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

# Simple file-based cache for fallback
def save_to_cache(ticker, interval, period, data):
    """Save data to a cache file"""
    try:
        cache_dir = '.streamlit/cache'
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_file = f"{cache_dir}/{ticker}_{interval}_{period}.csv"
        data.to_csv(cache_file)
        logger.info(f"Saved data to cache: {cache_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving cache: {e}")
        return False

def load_from_cache(ticker, interval, period):
    """Load data from cache file"""
    try:
        cache_file = f".streamlit/cache/{ticker}_{interval}_{period}.csv"
        if os.path.exists(cache_file):
            # Check if file is not too old (24 hours)
            file_age = time.time() - os.path.getmtime(cache_file)
            if file_age < 86400:  # 24 hours in seconds
                data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                logger.info(f"Loaded from cache: {cache_file}")
                return data
    except Exception as e:
        logger.error(f"Error loading from cache: {e}")
    return None

# Custom implementation of technical indicators
def calculate_macd(data, fast=7, slow=25, signal=9):
    """Calculate MACD using pandas EMA"""
    try:
        # Calculate the short and long EMAs
        ema_short = data['Close'].ewm(span=fast, adjust=False).mean()
        ema_long = data['Close'].ewm(span=slow, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = ema_short - ema_long
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        
        # Calculate histogram
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
        # Calculate price changes
        delta = data['Close'].diff()
        
        # Get gains and losses
        gains = delta.copy()
        losses = delta.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        # Calculate average gains and losses
        avg_gains = gains.ewm(alpha=1/periods, adjust=False).mean()
        avg_losses = losses.ewm(alpha=1/periods, adjust=False).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return pd.Series(rsi, name='RSI_6')
    except Exception as e:
        logger.error(f"Error calculating RSI: {e}")
        return pd.Series(dtype=float)

# Replace the calculate_indicators function
def calculate_indicators(data):
    if data.empty:
        return data
    
    # Check if we have enough data points for calculation
    if len(data) < 26:  # Minimum 26 periods for MACD
        return data
    
    try:
        # Make a copy to avoid modifying the original dataframe
        result_data = data.copy()
        
        # Ensure data types are correct
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in result_data.columns:
                result_data[col] = pd.to_numeric(result_data[col], errors='coerce')
        
        # Drop any NaN values before calculation
        result_data = result_data.dropna()
        
        if len(result_data) >= 26:
            # Calculate MACD
            macd_data = calculate_macd(result_data)
            if not macd_data.empty:
                result_data = pd.concat([result_data, macd_data], axis=1)
            
            # Calculate RSI
            rsi_data = calculate_rsi(result_data)
            if not rsi_data.empty:
                result_data['RSI_6'] = rsi_data
            
            # Forward fill first
            result_data = result_data.ffill()
            # Then backward fill any remaining NaNs
            result_data = result_data.bfill()
            
        return result_data
            
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return data

# Enhanced fetch function with fallbacks
@st.cache_data(ttl=60, show_spinner=False)
def get_stock_data_with_buffer(ticker, interval='1m', period='1d'):
    """Fetch a day's worth of data to use as a buffer for animations with robust error handling"""
    logger.info(f"Fetching data for {ticker}, interval={interval}, period={period}")
    
    # Adjust period based on interval for better data availability
    if interval == '1m':
        buffer_period = '5d'  # Fetch more historical data to ensure availability
    elif interval == '15m':
        buffer_period = '7d'
    elif interval == '1d':
        buffer_period = '60d'
    else:
        buffer_period = '5d'
    
    # Try different methods to get data
    data = None
    
    # Method 1: Try direct yfinance download
    for attempt in range(3):
        try:
            session = get_session()
            logger.info(f"Attempt {attempt+1} using yf.download")
            
            data = yf.download(
                ticker, 
                period=buffer_period, 
                interval=interval, 
                session=session, 
                prepost=True,
                progress=False,
                threads=False  # Disable multithreading which can cause issues
            )
            
            if len(data) > 0:
                logger.info(f"Success with yf.download: {len(data)} rows")
                break
                
            logger.warning(f"Empty data from download method, retrying...")
            time.sleep(2 ** attempt)  # Exponential backoff
        except Exception as e:
            logger.error(f"Error with yf.download: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff
    
    # Method 2: If download fails, try using Ticker.history()
    if data is None or len(data) == 0:
        for attempt in range(3):
            try:
                logger.info(f"Attempt {attempt+1} using Ticker.history()")
                session = get_session()
                
                stock = yf.Ticker(ticker, session=session)
                data = stock.history(period=buffer_period, interval=interval)
                
                if len(data) > 0:
                    logger.info(f"Success with Ticker.history: {len(data)} rows")
                    break
                    
                logger.warning(f"Empty data from Ticker.history, retrying...")
                time.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"Error with Ticker.history: {e}")
                time.sleep(2 ** attempt)
    
    # Method 3: Try to load from cache if API methods fail
    if data is None or len(data) == 0:
        logger.info("Trying to load from cache...")
        data = load_from_cache(ticker, interval, period)
        if data is not None and len(data) > 0:
            st.info("Using cached data. Market data may not be current.")
    
    # Process and clean the data if we got something
    if data is not None and len(data) > 0:
        # Clean up column names if we have a MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Make sure we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Filter to only required columns if they exist
        existing_cols = [col for col in required_cols if col in data.columns]
        if len(existing_cols) == len(required_cols):
            data = data[required_cols].copy()
        else:
            missing = [col for col in required_cols if col not in data.columns]
            logger.error(f"Missing columns in data: {missing}")
            st.warning(f"Data is missing required columns: {missing}")
            return pd.DataFrame()
        
        # Clean the data
        data = data.ffill().bfill()
        
        # Ensure we have minimum required data
        if len(data) < 2:
            st.warning(f"Insufficient data points for {ticker}. Try a different interval or time range.")
            return pd.DataFrame()
        
        # Calculate technical indicators if enough data
        if len(data) >= 26:
            try:
                data = calculate_indicators(data)
            except Exception as e:
                logger.error(f"Error calculating indicators: {e}")
        
        # Save successful result to cache for future fallback
        save_to_cache(ticker, interval, period, data)
        
        return data
    
    # If all methods failed, show appropriate message
    st.warning(f"Unable to fetch data for {ticker}. Please try:")
    st.markdown("""
    - Refreshing the page
    - Changing the time interval
    - Selecting a different ticker
    """)
    return pd.DataFrame()

# Get current quote (just the latest data)
@st.cache_data(ttl=30, show_spinner=False, max_entries=50)
def get_current_quote(ticker):
    """Get current stock quote with retries"""
    try:
        session = get_session()
        
        # Exponential backoff for retries
        for attempt in range(3):
            try:
                stock = yf.Ticker(ticker, session=session)
                quote = stock.fast_info
                if quote is not None:
                    return quote
                time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
            except Exception as e:
                if attempt == 2:  # Last attempt failed
                    logger.error(f"Failed to fetch quote after 3 attempts: {e}")
                    return None
                time.sleep(2 ** attempt)
                
        logger.warning(f"No quote data available for {ticker}")
        return None
        
    except Exception as e:
        logger.error(f"Error in get_current_quote: {e}")
        return None

# Fetch tickers (cached for 24 hours)
@st.cache_data(ttl=86400, max_entries=1)  # Cache for 24 hours, only keep latest
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

# Function to update KPIs with full data
def update_kpis(data, kpi_placeholder):
    """Update KPIs with full historical data"""
    if data is None or data.empty:
        return
    
    try:
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
            
    except Exception as e:
        logger.error(f"Error updating KPIs: {e}")

# Function to update KPIs with quick price update
def update_quick_kpis(current_price, kpi_placeholder):
    """Update KPIs with just the current price"""
    if current_price is None:
        return
    
    try:
        # Get the current data buffer
        data_buffer = st.session_state.data_buffer
        if data_buffer is None or data_buffer.empty:
            return
        
        # Calculate price change from the first price in the buffer
        first_price = float(data_buffer['Open'].iloc[0])
        price_change = current_price - first_price
        price_change_pct = (price_change / first_price) * 100 if first_price != 0 else 0
        
        # Get high and low from the buffer
        high_24h = float(data_buffer['High'].max())
        low_24h = float(data_buffer['Low'].min())
        
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
            
    except Exception as e:
        logger.error(f"Error updating quick KPIs: {e}")

# Enhanced error handling for data filtering
def safe_filter_data(data_buffer, time_range):
    """Safely filter data based on time range with error handling"""
    if data_buffer is None or data_buffer.empty:
        return pd.DataFrame()
    
    try:
        # Ensure we have enough data points
        if len(data_buffer) < 2:
            return data_buffer
        
        # Get the time range of available data
        data_start = data_buffer.index[0]
        data_end = data_buffer.index[-1]
        total_minutes = (data_end - data_start).total_seconds() / 60
        
        # Check if we have enough data for the requested time range
        if time_range == '1h' and total_minutes < 60:
            st.warning("Less than 1 hour of data available. Using all available data.")
            return data_buffer
        elif time_range == '1d' and total_minutes < 1440:
            st.warning("Less than 1 day of data available. Using all available data.")
            return data_buffer
        elif time_range == '1w' and total_minutes < 10080:
            st.warning("Less than 1 week of data available. Using all available data.")
            return data_buffer
            
        # Calculate the start time based on the end time
        end_time = data_end
        
        try:
            if time_range == '1h':
                start_time = end_time - pd.Timedelta(hours=1)
            elif time_range == '1d':
                start_time = end_time - pd.Timedelta(days=1)
            elif time_range == '1w':
                start_time = end_time - pd.Timedelta(days=7)
            else:
                return data_buffer
                
            # Ensure start_time is not before the earliest available data
            start_time = max(start_time, data_start)
            
            # Filter data
            filtered_data = data_buffer.loc[start_time:end_time].copy()
            
            # Validate filtered data
            if filtered_data.empty or len(filtered_data) < 2:
                return data_buffer
                
            # Check for gaps in data
            time_diff = pd.Series(filtered_data.index).diff()
            if time_diff.max().total_seconds() > 300:  # Gap larger than 5 minutes
                st.warning("Data may contain gaps. Some periods might be missing.")
            
            return filtered_data
            
        except pd.errors.OutOfBoundsDatetime:
            return data_buffer
            
    except Exception:
        return data_buffer

# Enhanced data validation
def validate_data_buffer(data_buffer, ticker, interval):
    """Validate data buffer and provide appropriate warnings"""
    if data_buffer is None or data_buffer.empty:
        st.warning(f"No data available for {ticker}. Please check market hours (9:30 AM - 4:00 PM EST) or try a different ticker.")
        return False
    
    # Check for minimum required data points
    if len(data_buffer) < 2:
        st.warning(f"Insufficient data points for {ticker}. Please try a different interval or time range.")
        return False
    
    # Check for required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in data_buffer.columns]
    if missing_columns:
        return False
    
    # Check for valid values
    if data_buffer.isnull().any().any():
        # Handle silently
        data_buffer = data_buffer.ffill().bfill()
    
    return True

# Function to create live-like tick data
def generate_simulated_ticks(data_buffer, num_ticks=5, start_idx=None, volatility=0.0005):
    """Generate simulated tick data between candles to create fluid movement"""
    if data_buffer is None or data_buffer.empty or len(data_buffer) < 2:
        return []
    
    try:
        if start_idx is None:
            # Default to second-to-last candle
            start_idx = len(data_buffer) - 2
        
        # Ensure start_idx is valid and we have enough data
        start_idx = min(max(0, start_idx), len(data_buffer) - 2)
        
        # Get two consecutive candles
        current_candle = data_buffer.iloc[start_idx]
        next_candle = data_buffer.iloc[start_idx + 1]
        
        # Extract timestamps
        start_time = pd.Timestamp(data_buffer.index[start_idx])
        end_time = pd.Timestamp(data_buffer.index[start_idx + 1])
        time_delta = (end_time - start_time) / (num_ticks + 1)
        
        # Generate ticks
        ticks = []
        prev_price = float(current_candle['Close'])
        base_volume = float(next_candle['Volume']) / num_ticks
        
        # Calculate price range for volatility
        price_range = max(
            abs(float(next_candle['High']) - float(next_candle['Low'])),
            abs(float(next_candle['Close']) - float(current_candle['Close'])),
            0.0001
        )
        
        # Pre-calculate the target price
        target_price = float(next_candle['Close'])
        price_diff = target_price - prev_price
        
        for i in range(num_ticks):
            try:
                # Calculate time for this tick
                tick_time = start_time + time_delta * (i + 1)
                
                # Progress factor (0 to 1)
                progress = (i + 1) / (num_ticks + 1)
                
                # Calculate price with momentum and controlled volatility
                price_movement = price_diff * progress
                noise = np.random.normal(0, volatility * price_range)
                new_price = prev_price + price_movement * 0.7 + noise
                
                # Constrain price within candle bounds
                new_price = max(min(new_price, float(next_candle['High'])), float(next_candle['Low']))
                
                # Generate realistic volume
                vol_factor = 1 + np.random.normal(0, 0.2)  # 20% standard deviation
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

# Function to update a candle with tick data in real-time
def update_candle_with_tick(candle_data, tick):
    """Update the current candle with a new tick"""
    if candle_data is None or tick is None:
        return None
    
    try:
        # Create a copy of the candle data
        updated = candle_data.copy()
        
        # Ensure numeric types for all values
        price = float(tick['price'])
        volume = int(tick['volume'])
        
        # Update the current price
        updated['Close'] = price
        
        # Update High if new price is higher
        updated['High'] = max(float(updated['High']), price)
        
        # Update Low if new price is lower
        updated['Low'] = min(float(updated['Low']), price)
        
        # Update Volume with proper type casting
        current_volume = int(float(updated['Volume']))  # Handle potential string or float volumes
        updated['Volume'] = current_volume + volume
        
        return updated
        
    except Exception as e:
        logger.error(f"Error in update_candle_with_tick: {e}")
        return None

# Function to create a live-updating data frame for animation
def create_animation_frame(data_buffer, current_index, tick_index=None, tick=None):
    """Create a new data frame showing current state for animation"""
    if data_buffer is None or data_buffer.empty:
        return pd.DataFrame()
    
    try:
        # Make sure current_index is valid
        current_index = min(max(0, current_index), len(data_buffer) - 1)
        
        # Create a copy of the buffer up to current_index + 1
        frame = data_buffer.iloc[:current_index+1].copy()
        
        # If we have a tick, update the last candle
        if tick is not None and len(frame) > 0:
            try:
                last_candle = frame.iloc[-1].copy()
                updated_candle = update_candle_with_tick(last_candle, tick)
                if updated_candle is not None:
                    frame.iloc[-1] = updated_candle
            except Exception as e:
                logger.error(f"Error updating candle with tick: {e}")
        
        # Ensure we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in frame.columns for col in required_columns):
            logger.error("Missing required columns in frame")
            return pd.DataFrame()
            
        # Calculate indicators when needed
        if len(frame) >= 26:  # Only calculate if we have enough data points
            try:
                # Calculate MACD
                macd_data = calculate_macd(frame)
                if not macd_data.empty:
                    for col in macd_data.columns:
                        frame[col] = macd_data[col]
                
                # Calculate RSI
                rsi_data = calculate_rsi(frame)
                if not rsi_data.empty:
                    frame['RSI_6'] = rsi_data
                
                # Fill any NaN values
                frame = frame.ffill().bfill()
            except Exception as e:
                logger.error(f"Error calculating indicators in animation frame: {e}")
                # Don't return empty frame, continue with price data only
                pass
        
        return frame
        
    except Exception as e:
        logger.error(f"Error in create_animation_frame: {e}")
        return pd.DataFrame()

# Create Interactive Chart
def create_interactive_chart(data, ticker, interval, is_animation_frame=False, last_price=None, last_tick_time=None):
    if data is None or data.empty or len(data) < 2:
        # Only show error in non-animation mode to avoid flooding
        if not is_animation_frame:
            st.warning("Waiting for sufficient data to create chart...")
            logger.error("No data available for chart creation")
        return go.Figure()

    # Add debug logging
    logger.info(f"Creating chart for {ticker} with {len(data)} data points")
    logger.info(f"Data columns: {data.columns.tolist()}")
    logger.info(f"First timestamp: {data.index[0]}, Last timestamp: {data.index[-1]}")

    has_indicators = all(col in data.columns for col in ['MACD_7_25_9', 'MACDs_7_25_9', 'RSI_6'])
    logger.info(f"Has indicators: {has_indicators}")

    # Create figure with subplots
    try:
        fig = make_subplots(rows=3, cols=1, 
                           specs=[[{"type": "candlestick"}],
                                 [{"type": "bar"}],
                                 [{"type": "scatter"}]],
                           vertical_spacing=0.15,
                           subplot_titles=('Candlestick', 'Volume', 'Indicators'))

        # Candlestick Chart (Row 1)
        candlestick = go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            increasing_line_color='#00FF00',
            decreasing_line_color='#FF4040',
            name="Price"
        )
        fig.add_trace(candlestick, row=1, col=1)

        # Volume Chart (Row 2)
        colors = ['#00CED1' if row['Close'] >= row['Open'] else '#FF7F50' for _, row in data.iterrows()]
        volume = go.Bar(
            x=data.index,
            y=data['Volume'],
            marker_color=colors,
            opacity=0.7,
            name='Volume'
        )
        fig.add_trace(volume, row=2, col=1)

        # Indicators (Row 3)
        if has_indicators:
            # MACD
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['MACD_7_25_9'],
                mode='lines',
                line=dict(color='#1E90FF', width=2),
                name='MACD'
            ), row=3, col=1)
            
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['MACDs_7_25_9'],
                mode='lines',
                line=dict(color='#FF4500', width=2),
                name='Signal'
            ), row=3, col=1)
            
            fig.add_trace(go.Bar(
                x=data.index,
                y=data['MACDh_7_25_9'],
                marker_color='#A9A9A9',
                name='Histogram'
            ), row=3, col=1)

            # RSI
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['RSI_6'],
                mode='lines',
                line=dict(color='#FFFF00', width=2),
                name='RSI'
            ), row=3, col=1)

        # Update layout
        fig.update_layout(
            title=dict(
                text=f"{ticker} - {interval.upper()} Data",
                x=0.5,
                font=dict(size=24, color='#FFD700')
            ),
            template='plotly_dark',
            paper_bgcolor='rgba(0, 0, 20, 0.9)',
            plot_bgcolor='rgba(0, 0, 30, 0.7)',
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True,
            hovermode='x unified'
        )

        # Update axes
        for i in range(1, 4):
            fig.update_xaxes(
                gridcolor='rgba(255, 255, 255, 0.2)',
                zeroline=False,
                row=i,
                col=1
            )
            fig.update_yaxes(
                gridcolor='rgba(255, 255, 255, 0.2)',
                zeroline=False,
                row=i,
                col=1
            )

        return fig

    except Exception as e:
        logger.error(f"Error creating interactive chart: {e}")
        return go.Figure()

# Enhanced error handling for main function
def main():
    try:
        st.title("📊 Yahoo Finance Stock Analysis")
        
        # Add debug info at the start
        logger.info("Starting main function")
        
        # Create placeholders with error handling
        try:
            kpi_placeholder = st.empty()
            chart_placeholder = st.empty()
            status_container = st.container()
            animation_status = status_container.empty()
            logger.info("UI components created successfully")
        except Exception as e:
            logger.error(f"Error creating UI components: {e}")
            st.error("Error creating UI components. Please refresh the page.")
            return

        # Sidebar setup
        st.sidebar.header("Stock Selection")
        
        # Safe ticker loading
        try:
            tickers = get_tickers_cached(None)
            if not tickers:
                st.warning("Could not load tickers. Using default popular tickers.")
                tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT"]
        except Exception as e:
            st.warning("Error loading tickers. Using default popular tickers.")
            tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT"]
        
        # Safe ticker search
        ticker_search = st.sidebar.text_input("Search Tickers:")
        if ticker_search:
            filtered_tickers = [t for t in tickers if ticker_search.upper() in t]
            display_tickers = filtered_tickers[:100] if len(filtered_tickers) > 100 else filtered_tickers
        else:
            popular = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT"]
            other_tickers = [t for t in tickers if t not in popular]
            display_tickers = popular + other_tickers[:90]

        selected_ticker = st.sidebar.selectbox("Select Stock", display_tickers)
        
        # Interval and Time Range Filters
        st.sidebar.header("Chart Settings")
        interval = st.sidebar.selectbox("Interval", ['1m', '15m', '1d'])
        time_range = st.sidebar.selectbox("Time Range", ['1h', '1d', '1w'])

        # Animation settings
        st.sidebar.header("Animation Settings")
        enable_animation = st.sidebar.checkbox("Enable Live Animation", value=True)
        animation_speed = st.sidebar.slider("Animation Speed", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
        volatility = st.sidebar.slider("Volatility", min_value=0.0001, max_value=0.001, value=0.0003, step=0.0001)
        
        st.session_state.animation_speed = animation_speed

        # Load data buffer with enhanced error handling
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
                try:
                    logger.info(f"Fetching data for {selected_ticker} with interval {interval}")
                    data_buffer = get_stock_data_with_buffer(selected_ticker, interval)
                    
                    if data_buffer is None or data_buffer.empty:
                        st.error(f"No data available for {selected_ticker}. Please try a different ticker or interval.")
                        return
                    
                    logger.info(f"Received data with shape: {data_buffer.shape}")
                    logger.info(f"Data columns: {data_buffer.columns.tolist()}")
                    
                    # Calculate indicators
                    data_buffer = calculate_indicators(data_buffer)
                    
                    # Store in session state
                    st.session_state.data_buffer = data_buffer
                    st.session_state.current_ticker = selected_ticker
                    st.session_state.current_interval = interval
                    st.session_state.current_index = min(20, max(0, len(data_buffer) - 1))
                    
                    # Update KPIs
                    update_kpis(data_buffer, kpi_placeholder)
                    
                except Exception as e:
                    logger.error(f"Error loading data: {e}")
                    st.error(f"Error loading data for {selected_ticker}. Please try again.")
                    return

        # Display initial chart
        try:
            if not enable_animation:
                fig = create_interactive_chart(st.session_state.data_buffer, selected_ticker, interval)
                chart_placeholder.plotly_chart(fig, use_container_width=True)
                animation_status.markdown("Animation disabled")
            else:
                animation_status.markdown("🔴 Live simulation running...")
                
                while enable_animation:
                    current_index = st.session_state.current_index
                    frame = create_animation_frame(st.session_state.data_buffer, current_index)
                    
                    if frame is not None and not frame.empty:
                        fig = create_interactive_chart(frame, selected_ticker, interval, is_animation_frame=True)
                        chart_placeholder.plotly_chart(fig, use_container_width=True)
                    
                    time.sleep(animation_speed)
                    
                    # Update index
                    current_index += 1
                    if current_index >= len(st.session_state.data_buffer) - 1:
                        current_index = 0
                    st.session_state.current_index = current_index
                    
        except Exception as e:
            logger.error(f"Error displaying chart: {e}")
            st.error("Error displaying chart. Please try refreshing the page.")
            
    except Exception as e:
        logger.error(f"Main function error: {e}")
        st.error("An unexpected error occurred. Please refresh the page.")

if __name__ == "__main__":
    try:
        # Create cache directory if it doesn't exist
        os.makedirs('.streamlit/cache', exist_ok=True)
        
        # Import sys for diagnostic info
        import sys
        
        # Diagnostic info
        st.sidebar.markdown("### Diagnostic Info")
        expander = st.sidebar.expander("Show Diagnostic Info")
        with expander:
            st.write(f"Python version: {sys.version}")
            st.write(f"pandas version: {pd.__version__}")
            st.write(f"numpy version: {np.__version__}")
            st.write(f"yfinance version: {yf.__version__ if hasattr(yf, '__version__') else 'unknown'}")
            st.write(f"Current time: {datetime.datetime.now()}")
        
        main()
    except Exception as e:
        st.error("Application error. Please refresh the page.")
        st.error(f"Details: {str(e)}")