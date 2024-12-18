import time
import pandas as pd

def fetch_live_data(symbol, interval, exchange, limit=500):
    """
    Fetch live OHLCV data from the exchange.
    
    Args:
        symbol (str): Trading pair symbol (e.g., 'BTC/USD').
        interval (str): Timeframe for candles (e.g., '1m', '5m').
        exchange (ccxt.Exchange): Exchange object initialized with API keys.
        limit (int): Number of candles to fetch.

    Returns:
        pd.DataFrame: DataFrame containing OHLCV data.
    """
    try:
        if exchange is None:
            raise ValueError("[ERROR] Exchange object is not provided.")
        
        # Define timeframe in milliseconds
        timeframe_ms = {'1m': 60 * 1000, '5m': 5 * 60 * 1000}
        if interval not in timeframe_ms:
            raise ValueError(f"[ERROR] Unsupported interval: {interval}")
        
        current_time = int(time.time() * 1000)
        since = current_time - (timeframe_ms[interval] * limit)
        
        # Fetch data in chunks
        all_ohlcv = []
        fetch_limit = min(1000, limit)  # Most exchanges limit to 1000 per request
        
        while len(all_ohlcv) < limit:
            ohlcv = exchange.fetch_ohlcv(
                symbol,
                timeframe=interval,
                since=since,
                limit=fetch_limit
            )
            
            if not ohlcv:
                print("[WARNING] No more data available.")
                break
            
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1  # Avoid overlapping candles
            
            # Respect rate limits
            time.sleep(exchange.rateLimit / 1000)
            
            if len(ohlcv) < fetch_limit:
                print("[INFO] Fetched fewer rows than expected; stopping early.")
                break
        
        # Convert to DataFrame
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        data = pd.DataFrame(all_ohlcv[:limit], columns=columns)
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        
        print(f"[DEBUG] Fetched {len(data)} {interval} candles.")
        return data

    except Exception as e:
        import traceback
        print(f"[ERROR] Failed to fetch live data: {e}")
        traceback.print_exc()
        return pd.DataFrame()
