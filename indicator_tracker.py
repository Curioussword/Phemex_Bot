import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import deque
from indicator_calculator import IndicatorCalculator

class TimeframeData:
    def __init__(self, window_size=250, atr_period=14):
        self.window_size = window_size
        self.atr_period = atr_period
        self.data = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        self.required_length = 20
        self.scaler = StandardScaler()

    def add_candle(self, timeframe, timestamp, open_, high_, low_, close_, volume):
        try:
            # Enhanced validation
            if not all(isinstance(x, (int, float)) for x in [open_, high_, low_, close_, volume]):
                print(f"[DEBUG] Invalid data types: open={type(open_)}, close={type(close_)}, volume={type(volume)}")
                raise ValueError("All values must be numeric")
            
            # Convert string values if needed
            close_ = float(close_)
            volume = float(volume)
            
            if close_ <= 0:
                print(f"[DEBUG] Invalid close price: {close_}")
                raise ValueError(f"Close price must be positive, got {close_}")
            
            if volume <= 0:
                print(f"[DEBUG] Invalid volume: {volume}")
                raise ValueError(f"Volume must be positive, got {volume}")

            # Create new candle data
            new_candle = pd.DataFrame({
                'timestamp': [pd.to_datetime(timestamp)],
                'open': [float(open_)],
                'high': [float(high_)],
                'low': [float(low_)],
                'close': [close_],
                'volume': [volume]
            })

            # Update dataframe with window size management
            if self.data is None:
                self.data = new_candle
            else:
                self.data = pd.concat([self.data, new_candle], ignore_index=True).tail(self.window_size)

            # Calculate indicators if we have enough data
            if len(self.data) >= self.window_size:
                self._update_indicators()
                return True
            return False

        except Exception as e:
            print(f"[ERROR] Failed to add candle: {e}")
            return False

    def _update_indicators(self):
        try:
            if len(self.data) < max(self.window_size, 200):
                print(f"[WARNING] Insufficient data for indicators: {len(self.data)}/{self.window_size}")
                return False

            close_series = self.data['close'].astype(float)
            
            self.ema_20 = IndicatorCalculator.calculate_ema(close_series, 20)
            self.ema_200 = IndicatorCalculator.calculate_ema(close_series, 200)
            self.vwap = IndicatorCalculator.calculate_vwap(self.data[['high', 'low', 'close', 'volume']])
            self.atr = IndicatorCalculator.calculate_atr(self.data[['high', 'low', 'close']], self.atr_period)
            self.volatility = IndicatorCalculator.calculate_volatility(close_series, period=20)

            if any(pd.isna([
                self.ema_20.iloc[-1],
                self.ema_200.iloc[-1],
                self.vwap.iloc[-1],
                self.atr.iloc[-1],
                self.volatility.iloc[-1]
            ])):
                print("[WARNING] NaN values detected in indicators")
                return False
            return True

        except Exception as e:
            print(f"[ERROR] Indicator calculation failed: {str(e)}")
            return False

    def prepare_features(self):
        if len(self.data) < self.window_size:
            return None

        features = np.array([
            self.ema_20.iloc[-1],
            self.ema_200.iloc[-1],
            self.vwap.iloc[-1],
            self.atr.iloc[-1],
            self.volatility.iloc[-1],
            self.data['close'].iloc[-1]
        ]).reshape(1, -1)

        return self.scaler.transform(features)

class PhemexBot:
    def __init__(self):
        self.timeframes = {
            '5m': TimeframeData(),
            '15m': TimeframeData(),
            '1h': TimeframeData(),
        }

    def add_candle(self, timeframe, timestamp, open_, high_, low_, close_, volume):
        if timeframe in self.timeframes:
            print(f"Adding candle to {timeframe} timeframe.")
            self.timeframes[timeframe].add_candle(timeframe, timestamp, open_, high_, low_, close_, volume)

    def get_combined_features(self):
        combined_features = []
        for tf_name, tf_data in self.timeframes.items():
            features = tf_data.prepare_features()
            if features is None:
                print(f"[WARNING] Insufficient data for {tf_name} timeframe.")
                return None
            combined_features.append(features)
        
        if not combined_features:
            return None
            
        return np.concatenate(combined_features, axis=1).reshape(1, 1, -1)
