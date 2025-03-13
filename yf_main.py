import streamlit as st
import pandas as pd
import numpy as np
from fix_imports import NaN  # Add compatibility for older pandas-ta
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
import copy
from scipy.interpolate import interp1d
import random

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

# Set up session for API calls
def get_session():
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.1)
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

# Fetch stock data with buffer
@st.cache_data(ttl=60*60, show_spinner=False)  # Cache for 1 hour
def get_stock_data_with_buffer(ticker, interval='1m', period='1d'):
    """Fetch a day's worth of data to use as a buffer for animations"""
    try:
        session = get_session()
        
        # Check current time (EST)
        est_tz = datetime.timezone(datetime.timedelta(hours=-5))
        current_time = datetime.datetime.now(est_tz)
        is_market_hours = (
            current_time.weekday() < 5 and  # Monday to Friday
            datetime.time(9, 30) <= current_time.time() <= datetime.time(16, 0)
        )
        
        # Adjust period based on interval and market hours
        if interval == '1m':
            if not is_market_hours:
                st.warning("Market is currently closed. Using latest available data.")
            buffer_period = '1d'
        elif interval == '15m':
            buffer_period = '7d'
        elif interval == '1d':
            buffer_period = '60d'
        else:
            buffer_period = '1d'
            
        # Fetch data
        data = yf.download(ticker, period=buffer_period, interval=interval, session=session)
        
        if len(data) > 0:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            
            # Handle data cleaning
            data = data.dropna()  # Remove any NaN values
            
            # Ensure we have minimum required data
            if len(data) < 2:
                st.warning(f"Insufficient data points for {ticker}. Try a different interval or time range.")
                return pd.DataFrame()
            
            # Calculate technical indicators if enough data
            if len(data) >= 26:
                try:
                    data.ta.macd(fast=7, slow=25, signal=9, append=True)
                    data.ta.rsi(length=6, append=True)
                except Exception as e:
                    print(f"Error calculating indicators: {e}")
            
            return data
            
        st.warning(f"No data returned for {ticker}. This could be due to:")
        st.markdown("""
        - Market hours (9:30 AM - 4:00 PM EST for US stocks)
        - Invalid ticker symbol
        - Recent IPO or delisting
        - Trading halts or suspensions
        """)
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

# Get current quote (just the latest data)
@st.cache_data(ttl=15, show_spinner=False)  # Cache expires every 15 seconds
def get_current_quote(ticker):
    try:
        session = get_session()
        data = yf.download(ticker, period='1d', interval='1m', session=session)
        if len(data) > 0:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            return data.iloc[-1]
        return None
    except Exception as e:
        return None

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

# Calculate MACD and RSI using pandas_ta with improved error handling
def calculate_indicators(data):
    if data.empty:
        return data
    
    # Check if we have enough data points for calculation
    if len(data) < 26:  # Minimum 26 periods for MACD
        # Not enough data, return original without attempting calculation
        return data
    
    try:
        # Make a copy to avoid modifying the original dataframe
        result_data = data.copy()
        
        # Ensure data types are correct
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in result_data.columns:
                # Convert to numeric, coerce errors to NaN
                result_data[col] = pd.to_numeric(result_data[col], errors='coerce')
        
        # If we have NaN values after conversion, drop or fill them
        if result_data.isna().any().any():
            # Option 1: Drop rows with any NaN values
            result_data = result_data.dropna()
            
            # If we no longer have enough data after dropping NaNs, return original
            if len(result_data) < 26:
                return data
        
        # Calculate indicators only if we have valid data
        if not result_data.empty:
            # Calculate MACD
            result_data.ta.macd(fast=7, slow=25, signal=9, append=True)
            
            # Calculate RSI
            result_data.ta.rsi(length=6, append=True)
            
            # Clean up any NaN values that might have been introduced during calculation
            result_data = result_data.fillna(method='ffill').fillna(method='bfill')
            
            return result_data
        else:
            return data
            
    except Exception as e:
        # Log the error but don't show warning during animation
        if not st.session_state.get('is_animating', False):
            st.warning(f"Failed to calculate indicators: {e}")
        
        # Return original data if calculation fails
        return data

# Function to create live-like tick data
def generate_simulated_ticks(data_buffer, num_ticks=5, start_idx=None, volatility=0.0005):
    """Generate simulated tick data between candles to create fluid movement"""
    if data_buffer is None or data_buffer.empty or len(data_buffer) < 2:
        return []
    
    try:
        if start_idx is None:
            # Default to second-to-last candle
            start_idx = len(data_buffer) - 2
        
        # Ensure start_idx is valid
        start_idx = min(max(0, start_idx), len(data_buffer) - 2)
        
        # Get two consecutive candles
        current_candle = data_buffer.iloc[start_idx]
        next_candle = data_buffer.iloc[start_idx + 1]
        
        # Extract timestamps and ensure they're timezone-naive
        start_time = pd.Timestamp(data_buffer.index[start_idx]).replace(tzinfo=None)
        end_time = pd.Timestamp(data_buffer.index[start_idx + 1]).replace(tzinfo=None)
        time_delta = (end_time - start_time) / (num_ticks + 1)
        
        # Generate ticks
        ticks = []
        prev_price = float(current_candle['Close'])  # Ensure numeric type
        base_volume = float(next_candle['Volume']) / num_ticks  # Ensure numeric type
        
        # Calculate price range for volatility
        price_range = max(
            abs(float(next_candle['High']) - float(next_candle['Low'])),
            abs(float(next_candle['Close']) - float(current_candle['Close'])),
            0.0001
        )
        
        for i in range(num_ticks):
            try:
                # Calculate time for this tick
                tick_time = start_time + time_delta * (i + 1)
                
                # Progress factor (0 to 1)
                progress = (i + 1) / (num_ticks + 1)
                
                # Calculate target price
                target_price = float(current_candle['Close']) + progress * (float(next_candle['Close']) - float(current_candle['Close']))
                
                # Add random noise proportional to volatility and price range
                noise = np.random.normal(0, volatility * price_range)
                
                # New price with momentum and noise
                new_price = prev_price + (target_price - prev_price) * 0.7 + noise
                
                # Constrain price within reasonable bounds
                new_price = max(min(new_price, float(next_candle['High']) * 1.001), float(next_candle['Low']) * 0.999)
                
                # Generate simulated volume with proper type casting
                simulated_volume = max(1, int(base_volume * (1 + random.uniform(-0.25, 0.25))))
                
                # Create simulated tick
                tick = {
                    'timestamp': tick_time,
                    'price': new_price,
                    'volume': simulated_volume
                }
                
                ticks.append(tick)
                prev_price = new_price
                
            except Exception as e:
                print(f"Error generating tick {i}: {e}")
                continue
        
        return ticks
        
    except Exception as e:
        print(f"Error in generate_simulated_ticks: {e}")
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
        print(f"Error in update_candle_with_tick: {e}")
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
            last_candle = frame.iloc[-1].copy()
            updated_candle = update_candle_with_tick(last_candle, tick)
            if updated_candle is not None:
                frame.iloc[-1] = updated_candle
        
        # Calculate indicators when needed
        indicator_cols = ['MACD_7_25_9', 'MACDs_7_25_9', 'MACDh_7_25_9', 'RSI_6']
        has_indicators = all(col in data_buffer.columns for col in indicator_cols)
        
        if has_indicators and len(frame) >= 26:
            # Copy existing indicators from the buffer
            for col in indicator_cols:
                if col in data_buffer.columns:
                    # Ensure index alignment
                    common_index = frame.index.intersection(data_buffer.index)
                    frame.loc[common_index, col] = data_buffer.loc[common_index, col]
        
        return frame
        
    except Exception as e:
        print(f"Error in create_animation_frame: {e}")
        return pd.DataFrame()

# Create Interactive Chart
def create_interactive_chart(data, ticker, interval, is_animation_frame=False, last_price=None, last_tick_time=None):
    if data is None or data.empty or len(data) < 2:
        # Only show error in non-animation mode to avoid flooding
        if not is_animation_frame:
            st.warning("Waiting for sufficient data to create chart...")
        return go.Figure()

    has_indicators = all(col in data.columns for col in ['MACD_7_25_9', 'MACDs_7_25_9', 'RSI_6'])

    # Create figure with subplots
    try:
        fig = make_subplots(rows=3, cols=1, 
                            specs=[[{"type": "candlestick"}],
                                   [{"type": "bar"}],
                                   [{"type": "scatter"}]],
                            vertical_spacing=0.15,
                            subplot_titles=('Candlestick', 'Volume', 'Indicators'))
        
        # Candlestick Chart (Row 1)
        fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], 
                                     low=data['Low'], close=data['Close'], 
                                     increasing_line_color='#00FF00',  # Bright green
                                     decreasing_line_color='#FF4040',  # Vibrant red
                                     name="Price"),
                      row=1, col=1)
        
        # Add the current price line if we're animating
        if is_animation_frame and last_price is not None and last_tick_time is not None:
            # Add a price line that extends to the right
            fig.add_trace(go.Scatter(
                x=[data.index[-1], last_tick_time], 
                y=[data['Close'].iloc[-1], last_price],
                mode='lines',
                line=dict(color='#FFFFFF', width=1, dash='dot'),
                showlegend=False
            ), row=1, col=1)
            
            # Add the last price as a marker
            fig.add_trace(go.Scatter(
                x=[last_tick_time],
                y=[last_price],
                mode='markers',
                marker=dict(
                    color='#FFFFFF',
                    size=8,
                    line=dict(color='#000000', width=1)
                ),
                showlegend=False
            ), row=1, col=1)
            
            # Add price label
            fig.add_annotation(
                x=last_tick_time,
                y=last_price,
                text=f"${last_price:.2f}",
                showarrow=True,
                arrowhead=0,
                arrowcolor="#FFFFFF",
                arrowsize=0.3,
                arrowwidth=1,
                ax=40,
                ay=0,
                font=dict(color="#FFFFFF", size=12),
                bgcolor="#000000",
                bordercolor="#FFFFFF",
                borderwidth=1,
                borderpad=4,
                opacity=0.8
            )

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

        animation_text = " 🔴 LIVE" if is_animation_frame else ""
        fig.update_layout(
            title=f"<b><span style='color:#FFD700; font-size:24px'>{ticker} - {interval.upper()} Data{animation_text}</span></b>",
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

        # Add animation marker
        if is_animation_frame:
            # Add a pulsing circle to indicate animation is in progress
            fig.add_annotation(
                x=0.5,
                y=1.05,
                xref="paper",
                yref="paper",
                text="🔴 LIVE SIMULATION",
                showarrow=False,
                font=dict(size=14, color="#FF4040"),
                bgcolor="rgba(255, 0, 0, 0.2)",
                bordercolor="#FF4040",
                borderwidth=2,
                borderpad=4,
                opacity=0.8
            )

        return fig
    except Exception as e:
        if not is_animation_frame:
            st.warning("Chart creation temporarily unavailable. Retrying...")
        return go.Figure()

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
        print(f"Error updating KPIs: {e}")

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
        print(f"Error updating quick KPIs: {e}")

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
        data_buffer = data_buffer.fillna(method='ffill').fillna(method='bfill')
    
    return True

# Enhanced error handling for main function
def main():
    try:
        st.title("📊 Yahoo Finance Stock Analysis")
        
        # Sidebar with stock selection, interval, and time range
        st.sidebar.header("Stock Selection")
        
        # Error handling for ticker refresh
        try:
            if st.sidebar.button("Refresh Yahoo Finance Tickers"):
                if os.path.exists("yfinance_tickers.csv"):
                    try:
                        os.remove("yfinance_tickers.csv")
                        st.success("Ticker list will be refreshed!")
                    except Exception as e:
                        st.warning(f"Could not remove existing ticker file. Using cached data. Details: {str(e)}")
                st.experimental_rerun()
        except Exception as e:
            st.warning("Error refreshing tickers. Using default ticker list.")
            
        # Safe ticker loading
        try:
            tickers = get_tickers_cached(None)
            if not tickers:
                st.warning("Could not load tickers. Using default popular tickers.")
                tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT"]
        except Exception as e:
            st.warning("Error loading tickers. Using default popular tickers.")
            tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT"]
        
        ticker_df = pd.DataFrame({"Symbol": tickers})
        csv_data = ticker_df.to_csv(index=False).encode('utf-8')
        
        try:
            st.sidebar.download_button(
                label="Download Ticker List",
                data=csv_data,
                file_name="yahoo_finance_tickers.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.sidebar.warning("Could not create download button for ticker list.")

        # Safe ticker search
        ticker_search = st.sidebar.text_input("Search Tickers:")
        if ticker_search:
            try:
                filtered_tickers = [t for t in tickers if ticker_search.upper() in t]
                display_tickers = filtered_tickers[:100] if len(filtered_tickers) > 100 else filtered_tickers
                if len(filtered_tickers) > 100:
                    st.sidebar.info(f"Found {len(filtered_tickers)} matches. Showing first 100.")
                if not filtered_tickers:
                    st.sidebar.warning("No matches found. Showing all tickers.")
                    display_tickers = tickers[:100]
            except Exception as e:
                st.sidebar.warning("Error in ticker search. Showing default tickers.")
                display_tickers = tickers[:100]
        else:
            try:
                popular = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT"]
                other_tickers = [t for t in tickers if t not in popular]
                display_tickers = popular + other_tickers[:90]
            except Exception as e:
                st.sidebar.warning("Error creating ticker list. Using popular tickers only.")
                display_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT"]

        selected_ticker = st.sidebar.selectbox("Select Stock", display_tickers)
        
        # Interval and Time Range Filters with validation
        st.sidebar.header("Chart Settings")
        interval = st.sidebar.selectbox("Interval", ['1m', '15m', '1d'], 
                                       help="Select interval: 1m (1-minute), 15m (15-minute), 1d (1-day).")
        time_range = st.sidebar.selectbox("Time Range", ['1h', '1d', '1w', '1mo', '1y'], 
                                         help="Select time range for the chart.")

        # Animation settings with validation
        st.sidebar.header("Animation Settings")
        enable_animation = st.sidebar.checkbox("Enable Live Animation", value=True, 
                                             help="Enable animations that simulate live price movements")
        animation_speed = st.sidebar.slider("Animation Speed", min_value=0.1, max_value=2.0, value=0.5, step=0.1,
                                          help="Speed of price animation (lower is faster)")
        volatility = st.sidebar.slider("Volatility", min_value=0.0001, max_value=0.001, value=0.0003, step=0.0001,
                                      format="%.4f", help="Controls the randomness of price movements")
        
        # Set animation speed in session state
        st.session_state.animation_speed = animation_speed

        # Validate and adjust interval and time range compatibility
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

        # Create placeholders with error handling
        try:
            kpi_placeholder = st.empty()
            chart_placeholder = st.empty()
            status_container = st.container()
            animation_status = status_container.empty()
        except Exception as e:
            st.error("Error creating UI components. Please refresh the page.")
            return
        
        # Load data buffer with enhanced error handling
        need_new_data = (
            st.session_state.data_buffer is None or 
            not st.session_state.buffer_loaded or 
            'current_ticker' not in st.session_state or 
            st.session_state.current_ticker != selected_ticker or
            'current_interval' not in st.session_state or
            st.session_state.current_interval != interval
        )
        
        if need_new_data:
            with st.spinner(f"Loading {selected_ticker} data buffer..."):
                try:
                    data_buffer = get_stock_data_with_buffer(selected_ticker, interval)
                    
                    # Validate the data buffer
                    if not validate_data_buffer(data_buffer, selected_ticker, interval):
                        return
                    
                    # Safely filter data based on time range
                    data_buffer = safe_filter_data(data_buffer, time_range)
                    
                    # Calculate indicators with error handling
                    try:
                        data_buffer = calculate_indicators(data_buffer)
                    except Exception as e:
                        st.warning("Error calculating technical indicators. Chart will show price data only.")
                    
                    # Store buffer in session state
                    st.session_state.data_buffer = data_buffer
                    st.session_state.buffer_loaded = True
                    st.session_state.current_ticker = selected_ticker
                    st.session_state.current_interval = interval
                    st.session_state.current_index = min(20, max(0, len(data_buffer) - 1))
                    
                    # Update KPIs with error handling
                    try:
                        update_kpis(data_buffer, kpi_placeholder)
                    except Exception as e:
                        st.warning("Error updating KPIs. Some metrics may not be displayed correctly.")
                        
                except Exception as e:
                    st.error(f"Error loading data for {selected_ticker}. Please try again or select a different ticker.")
                    return
        
        # Animation handling with error recovery
        if enable_animation and st.session_state.data_buffer is not None and not st.session_state.data_buffer.empty:
            st.session_state.is_animating = True
            
            try:
                data_buffer = st.session_state.data_buffer
                current_index = st.session_state.current_index
                
                animation_status.markdown("<p style='color:#00FF00; font-size:14px'>🔴 Live simulation running...</p>", 
                                         unsafe_allow_html=True)
                
                # Initial display
                try:
                    frame = create_animation_frame(data_buffer, current_index)
                    fig = create_interactive_chart(frame, selected_ticker, interval, is_animation_frame=True)
                    chart_placeholder.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning("Error displaying initial chart. Retrying...")
                    time.sleep(1)
                
                # Animation loop with error recovery
                while enable_animation and st.session_state.is_animating:
                    try:
                        ticks = generate_simulated_ticks(data_buffer, num_ticks=10, 
                                                       start_idx=current_index, volatility=volatility)
                        
                        for i, tick in enumerate(ticks):
                            try:
                                frame = create_animation_frame(data_buffer, current_index, i, tick)
                                fig = create_interactive_chart(
                                    frame, selected_ticker, interval, 
                                    is_animation_frame=True,
                                    last_price=tick['price'],
                                    last_tick_time=tick['timestamp']
                                )
                                chart_placeholder.plotly_chart(fig, use_container_width=True)
                                
                                try:
                                    update_quick_kpis(tick['price'], kpi_placeholder)
                                except Exception as e:
                                    print(f"Error updating quick KPIs: {e}")
                                
                                time.sleep(st.session_state.animation_speed / len(ticks))
                                
                            except Exception as e:
                                print(f"Error in tick animation: {e}")
                                continue
                        
                        # Safe index update
                        current_index += 1
                        if current_index >= len(data_buffer) - 1:
                            current_index = max(0, len(data_buffer) - 20)
                        
                        st.session_state.current_index = current_index
                        
                        # Safe data refresh
                        if current_index % 5 == 0:
                            try:
                                latest_quote = get_current_quote(selected_ticker)
                                if latest_quote is not None:
                                    update_quick_kpis(latest_quote['Close'], kpi_placeholder)
                            except Exception as e:
                                print(f"Error fetching latest quote: {e}")
                        
                    except Exception as e:
                        print(f"Error in animation loop: {e}")
                        time.sleep(1)  # Brief pause before retrying
                        continue
                    
            except Exception as e:
                st.warning("Animation encountered an error. Attempting to recover...")
                time.sleep(1)
            finally:
                st.session_state.is_animating = False
        else:
            # Non-animated display with error handling
            if st.session_state.data_buffer is not None and not st.session_state.data_buffer.empty:
                try:
                    fig = create_interactive_chart(st.session_state.data_buffer, selected_ticker, interval)
                    chart_placeholder.plotly_chart(fig, use_container_width=True)
                    
                    update_kpis(st.session_state.data_buffer, kpi_placeholder)
                    
                    animation_status.markdown("<p style='color:#FFD700; font-size:14px'>Animation disabled. Enable in sidebar.</p>", 
                                             unsafe_allow_html=True)
                except Exception as e:
                    st.error("Error displaying chart. Please try refreshing the page.")
            else:
                st.warning("No data available. Please try refreshing or selecting a different ticker.")
                
    except Exception as e:
        st.error("An unexpected error occurred. Please refresh the page and try again.")
        print(f"Main function error: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("Application error. Please refresh the page.")
        print(f"Application error: {e}")