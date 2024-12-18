import pandas as pd
import numpy as np
from collections import deque

import pandas as pd
import numpy as np
from collections import deque

class TimeframeData:
    def __init__(self, window_size=250, atr_period=14):
        self.window_size = window_size
        self.atr_period = atr_period
        self.data = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        self.ema_20 = None
        self.ema_200 = None
        self.vwap = None
        self.atr = None
        self.volatility = None

        # Separate data structures for each timeframe
        self.cumulative_pv = {'1m': 0, '5m': 0}
        self.cumulative_volume = {'1m': 0, '5m': 0}
        self.true_ranges = {'1m': deque(maxlen=atr_period), '5m': deque(maxlen=atr_period)}
        self.log_returns = {'1m': deque(maxlen=atr_period), '5m': deque(maxlen=atr_period)}

    def add_candle(self, timestamp, open, high, low, close, volume, timeframe):
        new_candle = pd.DataFrame({
            'timestamp': [timestamp],
            'open': [open],
            'high': [high],
            'low': [low],
            'close': [close],
            'volume': [volume]
        })

        # Maintain window size for DataFrame
        self.data = pd.concat([self.data, new_candle], ignore_index=True)
        if len(self.data) > self.window_size:
            self.data = self.data.iloc[-self.window_size:]

        # Calculate indicators based on the latest candle
        self.calculate_indicators(close, volume, timeframe)
        

    def calculate_indicators(self, close, volume, timeframe):
        if len(self.data) >= 2:
            # Update only the latest EMA values
            self.ema_20 = self.update_ema(close, 20, self.ema_20)
            self.ema_200 = self.update_ema(close, 200, self.ema_200)

            # Update VWAP with cumulative values
            self.cumulative_pv[timeframe] += close * volume
            self.cumulative_volume[timeframe] += volume
            self.vwap = (
                self.cumulative_pv[timeframe] / self.cumulative_volume[timeframe]
                if self.cumulative_volume[timeframe] > 0 else close
            )

        # Update ATR
        high_low = self.data['high'].iloc[-1] - self.data['low'].iloc[-1]
        high_close = abs(self.data['high'].iloc[-1] - self.data['close'].iloc[-2])
        low_close = abs(self.data['low'].iloc[-1] - self.data['close'].iloc[-2])
        true_range = max(high_low, high_close, low_close)
        self.true_ranges[timeframe].append(true_range)
        
        if len(self.true_ranges[timeframe]) > 0:
            self.atr = sum(self.true_ranges[timeframe]) / len(self.true_ranges[timeframe])

        # Update volatility
        try:
            log_return = np.log(close / self.data['close'].iloc[-2])
            self.log_returns[timeframe].append(log_return)
            if len(self.log_returns[timeframe]) > 0:
                self.volatility = np.std(list(self.log_returns[timeframe])) * np.sqrt(252)
            else:
                self.volatility = 0.0
        except (ValueError, ZeroDivisionError):
            self.volatility = 0.0



    def update_ema(self, new_price, period, prev_ema):
        if prev_ema is None:
            return new_price
        k = 2 / (period + 1)
        return (new_price * k) + (prev_ema * (1 - k))

class PhemexBot:
    def __init__(self):
        self.data_5m = TimeframeData()

    def add_5m_candle(self, timestamp, open, high, low, close, volume):
        self.data_5m.add_candle(timestamp, open, high, low, close, volume, '5m')

    def get_indicators(self):
        data = self.data_5m

        # Ensure we have enough data for calculations
        if len(data.data) < 200:  # Minimum required for EMA200
            print(f"[WARNING] Insufficient data for calculations")
            return None
        
        indicators = {
            'EMA_20': data.ema_20,
            'EMA_200': data.ema_200,
            'VWAP': data.vwap,
            'ATR': data.atr,
            'Volatility': data.volatility
        }
    
        # Validate indicator values
        if any(v is None or not isinstance(v, (int, float)) for v in indicators.values()):
            print(f"[WARNING] Invalid indicator values")
            return None
        
        print(f"[DEBUG] Indicators validated successfully")
        return indicators




