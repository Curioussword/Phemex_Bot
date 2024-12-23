import ccxt
import pandas as pd
from datetime import datetime

def test_fetch_ohlcv():
    try:
        # Initialize exchange
        exchange = ccxt.phemex({
            'enableRateLimit': True,
        })
        
        # Fetch 1-minute candles for BTC/USD
        symbol = 'BTCUSD'
        timeframe = '1m'
        limit = 5  # Fetch last 5 candles
        
        # Fetch OHLCV data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Convert timestamp to readable format
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Print each candle
        for _, row in df.iterrows():
            print(f"\nCandle at {row['timestamp']}:")
            print(f"Open:   {row['open']}")
            print(f"High:   {row['high']}")
            print(f"Low:    {row['low']}")
            print(f"Close:  {row['close']}")
            print(f"Volume: {row['volume']}")
            
    except Exception as e:
        print(f"[ERROR] Failed to fetch data: {e}")

if __name__ == "__main__":
    test_fetch_ohlcv()
