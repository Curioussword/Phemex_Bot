import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import time
import psycopg2

def calculate_ema(data, period):
    return data.ewm(span=period, adjust=False).mean()

def store_data_to_db(data):
    """Store OHLCV and EMA data into TimescaleDB."""
    conn = psycopg2.connect("dbname=trading_data user=postgres password=kaye0315! host=89.117.50.60 port=45081")
    cursor = conn.cursor()

    for _, row in data.iterrows():
        cursor.execute('''
            INSERT INTO trading_data (timestamp, open, high, low, close, volume, ema_20, ema_200)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (timestamp) DO NOTHING;
        ''', (row['timestamp'], row['open'], row['high'], row['low'], row['close'], row['volume'], row['EMA_20'], row['EMA_200']))

    conn.commit()
    cursor.close()
    conn.close()

def test_fetch_ohlcv():
    try:
        # Initialize exchange
        exchange = ccxt.phemex({
            'enableRateLimit': True,
        })
        
        # Fetch 15-minute candles for BTC/USD
        symbol = 'BTCUSD'  # Adjust this if needed
        timeframe = '1m'
        limit = 500  # Fetch last 500 candles to have enough data for 200 EMA

        # Fetch OHLCV data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Convert timestamp to readable format
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Calculate EMAs
        df['EMA_20'] = calculate_ema(df['close'], 20)
        df['EMA_200'] = calculate_ema(df['close'], 200)

        # Store the fetched data into the database
        store_data_to_db(df)

        # Print last 5 candles with EMAs
        for _, row in df.tail(5).iterrows():
            print(f"\nCandle at {row['timestamp']}:")
            print(f"Open:   {row['open']}")
            print(f"High:   {row['high']}")
            print(f"Low:    {row['low']}")
            print(f"Close:  {row['close']}")
            print(f"Volume: {row['volume']}")
            print(f"EMA_20: {row['EMA_20']:.2f}")
            print(f"EMA_200: {row['EMA_200']:.2f}")

    except Exception as e:
        print(f"[ERROR] Failed to fetch data: {e}")

if __name__ == "__main__":
    test_fetch_ohlcv()
