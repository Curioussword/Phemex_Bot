from cachetools import TTLCache
import time
import pandas as pd
import numpy as np
import ccxt

# Initialize cache with a TTL of 300 seconds (5 minutes)
cache = TTLCache(maxsize=50, ttl=300)

def fetch_live_data(symbol, exchange, timeframe='5m', limit=500):
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

        # Filter out invalid candles (e.g., volume == 0 or identical OHLC values)
        valid_data = data[
            (data['volume'] > 0) & 
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

def fetch_live_data_with_cache(symbol, exchange, timeframe='5m', limit=500):
    """
    Fetch live OHLCV data with caching.
    """
    # Create a descriptive cache key
    cache_key = f"{symbol}_{timeframe}_{limit}"
    
    # Check if data is already cached
    if cache_key in cache:
        print("[DEBUG] Returning cached data.")
        return cache[cache_key]
    
    # Fetch new data if not in cache
    print("[INFO] Fetching new market data...")
    data = fetch_live_data(symbol, exchange, timeframe=timeframe, limit=limit)

    # Cache the valid data
    if not data.empty:
        cache[cache_key] = data
    
    return data

def preprocess_data(data):
    """
    Preprocess market data by normalizing features.
    """
    if data.empty:
        print("[WARNING] No valid market data to preprocess.")
        return None

    scaler = StandardScaler()
    
    try:
        # Normalize numerical columns (excluding timestamp)
        numerical_columns = ['open', 'high', 'low', 'close', 'volume']
        scaled_data = scaler.fit_transform(data[numerical_columns])
        
        # Replace original columns with normalized values
        normalized_data = pd.DataFrame(scaled_data, columns=numerical_columns)
        normalized_data['timestamp'] = data['timestamp'].values
        
        return normalized_data

    except Exception as e:
        print(f"[ERROR] Failed to preprocess market data: {e}")
    
    return None

# Example usage
if __name__ == "__main__":
    exchange = ccxt.binance()  # Replace with your actual exchange instance
    symbol = "BTC/USDT"

    # Fetch and preprocess market data with caching
    raw_data = fetch_live_data_with_cache(symbol, exchange)
    
    if not raw_data.empty:
        processed_data = preprocess_data(raw_data)
        
        if processed_data is not None:
            print("[INFO] Preprocessed market data:")
            print(processed_data.head())
