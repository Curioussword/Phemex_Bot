import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from collections import deque
from indicator_calculator import IndicatorCalculator  # Import static calculations


class TimeframeData:
    def __init__(self, window_size=250, atr_period=14):
        """
        Initialize the data structure for a specific timeframe.
        """
        self.window_size = window_size
        self.atr_period = atr_period
        self.data = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        self.ema_20 = None
        self.ema_200 = None
        self.vwap = None
        self.atr = None
        self.volatility = None

        # Data structures for incremental updates
        self.cumulative_pv = 0  # Cumulative price * volume for VWAP
        self.cumulative_volume = 0  # Cumulative volume for VWAP
        self.true_ranges = deque(maxlen=atr_period)  # True ranges for ATR calculation
        self.log_returns = deque(maxlen=atr_period)  # Log returns for volatility calculation

        # Initialize StandardScaler for feature normalization
        self.scaler = StandardScaler()

    def add_candle(self, timestamp, open_, high_, low_, close_, volume):
        """
        Add a new candle and update indicators.
        """
        if any(pd.isna([open_, high_, low_, close_, volume])) or volume <= 0 or close_ <= 0:
            print(f"[WARNING] Invalid candle data: {timestamp}, {open_}, {high_}, {low_}, {close_}, {volume}. Skipping update.")
            return

        new_candle = pd.DataFrame({
            'timestamp': [timestamp],
            'open': [open_],
            'high': [high_],
            'low': [low_],
            'close': [close_],
            'volume': [volume]
        })

        # Append the new candle and maintain window size
        self.data = pd.concat([self.data, new_candle], ignore_index=True)
        if len(self.data) > self.window_size:
            self.data = self.data.iloc[-self.window_size:]

        # Update indicators based on the latest candle
        self.calculate_indicators(close_, volume)

    def calculate_indicators(self, close, volume):
        """
        Calculate EMA, VWAP, ATR, and volatility incrementally.
        """
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
        """
        Incrementally update VWAP.
        """
        self.cumulative_pv += close * volume
        self.cumulative_volume += volume
        self.vwap = self.cumulative_pv / self.cumulative_volume if self.cumulative_volume > 0 else close

    def update_atr(self):
        """
        Incrementally update ATR using the latest true range.
        """
        high_low = self.data['high'].iloc[-1] - self.data['low'].iloc[-1]
        high_close = abs(self.data['high'].iloc[-1] - self.data['close'].iloc[-2])
        low_close = abs(self.data['low'].iloc[-1] - self.data['close'].iloc[-2])
        true_range = max(high_low, high_close, low_close)

        self.true_ranges.append(true_range)
        self.atr = sum(self.true_ranges) / len(self.true_ranges)

    def update_ema(self, new_price, period, prev_ema):
        """
        Update EMA with the latest price.
        """
        if prev_ema is None:
            return new_price
        k = 2 / (period + 1)  # Smoothing factor
        return (new_price * k) + (prev_ema * (1 - k))

    def update_volatility(self, close):
        """
        Incrementally update volatility using log returns.
        """
        try:
            log_return = np.log(close / self.data['close'].iloc[-2])
            self.log_returns.append(log_return)
            self.volatility = np.std(list(self.log_returns)) * np.sqrt(252) if len(self.log_returns) > 0 else 0.0
        except Exception as e:
            print(f"[WARNING] Error calculating volatility: {e}")
            self.volatility = 0.0

    def prepare_features(self):
        """
        Prepare normalized features for the DQN agent.
        """
        if len(self.data) < 200:  # Ensure sufficient data for EMA_200
            return None

        features = {
            'EMA_20': self.ema_20,
            'EMA_200': self.ema_200,
            'VWAP': self.vwap,
            'ATR': self.atr,
            'Volatility': self.volatility,
            'Current_Price': self.data['close'].iloc[-1],
        }

        feature_values = np.array(list(features.values())).reshape(1, -1)

        # Fit scaler only once when sufficient data is available
        if not hasattr(self.scaler, "fitted") or not self.scaler.fitted:
            self.scaler.fit(feature_values)
            self.scaler.fitted = True

        normalized_features = self.scaler.transform(feature_values)  # Normalize features
        return normalized_features.reshape(1, 1, -1)


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
            self.timeframes[timeframe].add_candle(timestamp, open_, high_, low_, close_, volume)

    def get_combined_features(self):
        """
        Combine features from all timeframes into a single state vector.
        """
        combined_features = []

        for tf_name, tf_data in self.timeframes.items():
            features = tf_data.prepare_features()
            if features is None:
                print(f"[WARNING] Missing features for {tf_name} timeframe.")
                return None
            combined_features.extend(features.flatten())

        return np.array(combined_features).reshape(1, 1, -1)
