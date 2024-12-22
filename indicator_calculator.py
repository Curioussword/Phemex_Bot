import numpy as np
import pandas as pd
import time

class IndicatorCalculator:
    @staticmethod
    def calculate_ema(data, period):
        return data.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_vwap(data):
        cumulative_pv = (data['close'] * data['volume']).cumsum()
        cumulative_volume = data['volume'].cumsum()
        return cumulative_pv / cumulative_volume

    @staticmethod
    def calculate_atr(data, period):
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift())
        low_close = abs(data['low'] - data['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

    @staticmethod
    def calculate_volatility(data, period):
        log_returns = np.log(data['close'] / data['close'].shift())
        return log_returns.rolling(window=period).std() * np.sqrt(252)
