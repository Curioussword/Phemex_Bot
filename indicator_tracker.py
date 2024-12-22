import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from collections import deque
from indicator_calculator import IndicatorCalculator  # Import static calculations


class TimeframeData:
    def __init__(self, window_size=250, atr_period=14):
        self.window_size = window_size
        self.atr_period = atr_period
        self.data = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])        
        self.required_length = 20
        # Initialize StandardScaler for feature normalization
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
        
            # Update dataframe
            if self.data is None:
                self.data = new_candle
            else:
                self.data = pd.concat([self.data, new_candle], ignore_index=True)
            
            return True

        except Exception as e:
            print(f"[ERROR] Failed to add candle: {e}")
            return False


    def _update_indicators(self):
        """Calculate technical indicators with proper error handling and validation."""
        try:
            # Validate data length
            if len(self.data) < max(self.window_size, 200):  # 200 for EMA200
                print(f"[WARNING] Insufficient data for indicators: {len(self.data)}/{self.window_size}")
                return False

            # Get clean price data
            close_series = self.data['close'].astype(float)

            # Calculate all indicators
            self.ema_20 = IndicatorCalculator.calculate_ema(close_series, 20)
            self.ema_200 = IndicatorCalculator.calculate_ema(close_series, 200)

            # VWAP calculation
            self.vwap = IndicatorCalculator.calculate_vwap(
                self.data[['high', 'low', 'close', 'volume']]
            )

            # ATR calculation
            self.atr = IndicatorCalculator.calculate_atr(
                self.data[['high', 'low', 'close']],
                self.atr_period
            )

            # Volatility calculation (20-period standard deviation)
            self.volatility = IndicatorCalculator.calculate_volatility(
                close_series,
                period=20
            )

            # Validate calculated indicators
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

    def calculate_indicators(self, close, volume):
        """Calculate EMA, VWAP, ATR, and volatility incrementally."""
        if len(self.data) >= 2:  # Ensure there are at least two candles for calculations
            self.ema_20 = self.update_ema(close, 20, self.ema_20)
            self.ema_200 = self.update_ema(close, 200, self.ema_200)
            self.update_vwap(close, volume)

            if len(self.true_ranges) < self.atr_period:
                self.atr = IndicatorCalculator.calculate_atr(self.data[-self.atr_period:], self.atr_period).iloc[-1]
            else:
                self.update_atr()

            if len(self.log_returns) < self.atr_period:
                self.volatility = IndicatorCalculator.calculate_volatility(self.data[-self.atr_period:], self.atr_period).iloc[-1]
            else:
                self.update_volatility(close)

    def update_vwap(self, close, volume):
        """Incrementally update VWAP."""
        self.cumulative_pv += close * volume
        self.cumulative_volume += volume
        self.vwap = self.cumulative_pv / self.cumulative_volume if self.cumulative_volume > 0 else close

    def update_atr(self):
        """Incrementally update ATR using the latest true range."""
        high_low = self.data['high'].iloc[-1] - self.data['low'].iloc[-1]
        high_close = abs(self.data['high'].iloc[-1] - self.data['close'].iloc[-2])
        low_close = abs(self.data['low'].iloc[-1] - self.data['close'].iloc[-2])
        true_range = max(high_low, high_close, low_close)

        self.true_ranges.append(true_range)
        self.atr = sum(self.true_ranges) / len(self.true_ranges)

    def update_ema(self, new_price, period, prev_ema):
        """Update EMA with the latest price."""
        if prev_ema is None:
            return new_price
        k = 2 / (period + 1)  # Smoothing factor
        return (new_price * k) + (prev_ema * (1 - k))

    def update_volatility(self, close):
        """Incrementally update volatility using log returns."""
        try:
            log_return = np.log(close / self.data['close'].iloc[-2])
            self.log_returns.append(log_return)
            self.volatility = np.std(list(self.log_returns)) * np.sqrt(252) if len(self.log_returns) > 0 else 0.0
        except Exception as e:
            print(f"[WARNING] Error calculating volatility: {e}")
            self.volatility = 0.0

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
        """Initialize the bot with multiple timeframes."""
        self.timeframes = {
            '5m': TimeframeData(),
            '15m': TimeframeData(),
            '1h': TimeframeData(),
        }

    def add_candle(self, timeframe, timestamp, open_, high_, low_, close_, volume):
        """Add a new candle to the specified timeframe."""
        if timeframe in self.timeframes:
            print(f"Adding candle to {timeframe} timeframe.")
            self.timeframes[timeframe].add_candle(timestamp, open_, high_, low_, close_, volume)

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
