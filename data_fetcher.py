from cachetools import TTLCache
import time
import pandas as pd

# Initialize cache with a TTL of 300 seconds (5 minutes)
cache = TTLCache(maxsize=10, ttl=300)

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

        # Fetch data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        data = pd.DataFrame(ohlcv, columns=columns)
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        return data

    except Exception as e:
        print(f"[ERROR] Failed to fetch live data: {e}")
        return pd.DataFrame()

def fetch_live_data_with_cache(symbol, exchange, timeframe='5m', limit=500):
    """
    Fetch live OHLCV data with caching.
    """
    cache_key = f"{symbol}_{timeframe}"
    
    # Check if data is already cached
    if cache_key in cache:
        print("[DEBUG] Returning cached data.")
        return cache[cache_key]
    
    # Fetch new data if not in cache
    data = fetch_live_data(symbol, exchange, timeframe=timeframe, limit=limit)
    cache[cache_key] = data
    return data
