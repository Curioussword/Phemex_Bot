from cachetools import TTLCache
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
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
        data.set_index('timestamp', inplace=True)

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

def preprocess_and_resample_data(df, fetched_timeframe):
    """
    Preprocess and resample OHLCV data for specified and derived timeframes.
    Handles missing data, reindexes to complete time range, and aggregates data.
    """
    try:
        # Ensure the DataFrame has a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

        # Complete time range from start to end at 1-minute intervals if fetched timeframe is 1m
        if fetched_timeframe == '1m':
            full_time_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='1T')
            df = df.reindex(full_time_range)

        # Forward fill to handle missing data after reindexing
        df.ffill(inplace=True)

        # Aggregate the data as before
        df = df.resample(fetched_timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })

        # Sort the DataFrame by the index (just in case)
        df.sort_index(inplace=True)

        # Define the derived time frames and corresponding new column names
        derived_time_frames = ['5T', '15T', '30T', '60T', '240T', '1440T']  # in minutes
        new_columns = ['5_min', '15_min', '30_min', '1_hour', '4_hour', '1_day']
        columns_to_resample = ['open', 'close', 'high', 'low', 'volume']

        # Initialize a MultiIndex for the new columns
        tuples = [(col, sub_col) for col in new_columns for sub_col in columns_to_resample]
        multi_index = pd.MultiIndex.from_tuples(tuples)
        resampled_df = pd.DataFrame(index=df.index, columns=multi_index)

        # Assigning resampled values to new columns and sub-columns
        for time_frame, col in zip(derived_time_frames, new_columns):
            df_resampled = df.resample(time_frame).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })

            # Forward fill the resampled data to avoid NaN values
            df_resampled.ffill(inplace=True)

            # Assign resampled values to sub-columns
            for sub_col in columns_to_resample:
                resampled_df[(col, sub_col)] = df_resampled[sub_col]

        # Combine the original df with the resampled_df
        df_combined = pd.concat([df, resampled_df], axis=1)

        # Reorder the columns to match the expected order
        column_order = ['open', 'close', 'high', 'low', 'volume'] + list(resampled_df.columns)
        df_combined = df_combined[column_order]

        # Forward fill again to handle any NaN values after reindexing
        df_combined.ffill(inplace=True)

        # Back fill to complete any remaining missing data
        df_combined.bfill(inplace=True)

        # Check for NaN values (for debugging purposes)
        nan_values = df_combined.isna().sum()
        print("[DEBUG] NaN values per column:")
        print(nan_values)

        return df_combined

    except Exception as e:
        print(f"[ERROR] Failed to preprocess and resample data: {e}")
        return pd.DataFrame()

            
