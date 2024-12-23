from cachetools import TTLCache
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import ccxt

# Initialize cache with a TTL of 300 seconds (5 minutes)
cache = TTLCache(maxsize=50, ttl=300)

def fetch_live_data(symbol, exchange, timeframe='5m', limit=1000):
    """
    Fetch live OHLCV data for a specified timeframe from the exchange.
    """
    try:
        # Define timeframe in milliseconds
        timeframe_ms = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '1h': 60 * 60 * 1000,
        }.get(timeframe, 5 * 60 * 1000)

        current_time = int(time.time() * 1000)
        since = current_time - (timeframe_ms * limit)

        # Fetch data from the exchange
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        data = pd.DataFrame(ohlcv, columns=columns)

        # Convert timestamp to datetime
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        data.set_index('timestamp', inplace=True)

        # Filter out invalid candles
        valid_data = data[
            (data['volume'] >= 0) & 
            ~(data['open'] == data['high']) & 
            ~(data['high'] == data['low']) & 
            ~(data['low'] == data['close'])
        ]

        if valid_data.empty:
            print("[WARNING] No valid market data fetched.")
            return pd.DataFrame()

        # Log skipped candles
        skipped_candles = len(data) - len(valid_data)
        if skipped_candles > 0:
            print(f"[INFO] Skipped {skipped_candles} invalid candles.")

        return valid_data

    except ccxt.NetworkError as e:
        print(f"[ERROR] Network error while fetching live data: {e}")
    except ccxt.ExchangeError as e:
        print(f"[ERROR] Exchange error while fetching live data: {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error while fetching live data: {e}")
    
    return pd.DataFrame()

def fetch_live_data_with_cache(symbol, exchange, timeframe='5m', limit=1000):
    """
    Fetch live OHLCV data with caching.
    """
    try:
        # Create a descriptive cache key
        cache_key = f"{symbol}_{timeframe}_{limit}"
        
        # Check if data is already cached
        if cache_key in cache:
            print("[DEBUG] Returning cached data.")
            return cache[cache_key]
        
        # Fetch new data if not in cache
        print("[INFO] Fetching new market data...")
        data = fetch_live_data(symbol, exchange, timeframe=timeframe, limit=limit)
        
        # Validate and cache the data
        if not data.empty:
            cache[cache_key] = data
            print(f"[DEBUG] Cached {len(data)} candles for {timeframe}")
            return data
        else:
            print(f"[WARNING] No valid market data fetched for {timeframe}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"[ERROR] Cache operation failed: {e}")
        return pd.DataFrame()

def preprocess_and_resample_data(data, timeframe):
    """
    Preprocess and resample OHLCV data for trading timeframes.
    """
    try:
        # Convert timeframe format
        tf_map = {
            '5m': '5T',
            '15m': '15T',
            '1h': '60T',
            '1d': '1D'
        }
        resample_timeframe = tf_map.get(timeframe, timeframe)

        # Convert to DataFrame and validate
        df = pd.DataFrame(data)
        if df.empty:
            raise ValueError("Empty data received")

        # Ensure proper datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

        # Validate required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("Missing required OHLCV columns")

        # Resample to required timeframe
        resampled = df.resample(resample_timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })

        # Handle missing data
        resampled.ffill(inplace=True)
        
        # Sort index for consistency
        resampled.sort_index(inplace=True)

        # Validate output
        nan_values = resampled.isna().sum()
        if nan_values.any():
            print(f"[DEBUG] NaN values in {timeframe} timeframe:")
            print(nan_values)

        print(f"[DEBUG] Processed {len(resampled)} candles for {timeframe} timeframe.")
        
        return resampled.dropna()

    except Exception as e:
        print(f"[ERROR] Failed to preprocess and resample data: {e}")
        return pd.DataFrame()
