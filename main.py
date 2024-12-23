import time
from datetime import datetime
import numpy as np
from tqdm import tqdm
import pandas as pd

from utils import load_config
from data_fetcher import fetch_live_data_with_cache, preprocess_and_resample_data
from exchange_setup import initialize_exchange
from indicator_tracker import PhemexBot
from state_manager import StateManager
from dqn_agent import DQNLSTMAgent  # LSTM-based DQN Agent
from bot import execute_trade, calculate_tp_sl  # Trade execution utilities
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU 1


class TradingSystem:
    def __init__(self):
        # Enable TensorFlow GPU memory growth
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("[INFO] Enabled GPU memory growth.")
            except RuntimeError as e:
                print(f"[ERROR] Failed to set GPU memory growth: {e}")

        print("[INFO] Loading configuration...")
        self.config = load_config()

        print("[INFO] Initializing exchange...")
        self.exchange = initialize_exchange()

        print("[INFO] Initializing state manager...")
        self.state_manager = StateManager(self.exchange)

        print("[INFO] Initializing PhemexBot...")
        self.bot = PhemexBot()

        print("[INFO] Initializing DQN Agent...")
        state_size = 5  # Example: EMA_20, EMA_200, ATR, current_price, position size
        action_size = 3  # Actions: [Hold=0, Buy=1, Sell=2]
        self.agent = DQNLSTMAgent(state_size=state_size, action_size=action_size)

        # Trade log and caching
        self.trade_log = []
        self.last_trade = None

    def get_current_price(self):
        """Fetch the current market price."""
        try:
            ticker = self.exchange.fetch_ticker(self.config['trade_parameters']['symbol'])
            return ticker['last']
        except Exception as e:
            print(f"[ERROR] Failed to fetch current price: {e}")
            return None

    def update_market_data(self):
	    """
	    Fetch, preprocess, and cache market data for multiple timeframes.
	    """
	    try:
	        base_timeframes = ['5m', '15m', '1h']  # Removed 1m as it's not used for indicators
	        for timeframe in base_timeframes:
	            print(f"[DEBUG] Fetching new {timeframe} data...")
	            raw_data = fetch_live_data_with_cache(
	                self.config['trade_parameters']['symbol'],
	                exchange=self.exchange,
	                timeframe=timeframe,
	                limit=1000
	            )

	            if not raw_data.empty:
	                processed_data = preprocess_and_resample_data(raw_data, timeframe)
	                if processed_data is not None and not processed_data.empty:
	                    print(f"[DEBUG] Processed {len(processed_data)} candles for {timeframe} timeframe.")
	                    for _, row in processed_data.iterrows():
	                        self.bot.add_candle(
	                            timeframe=timeframe,
	                            timestamp=row.name,
	                            open_=row['open'],
	                            high_=row['high'],
	                            low_=row['low'],
	                            close_=row['close'],
	                            volume=row['volume']
	                        )
	                else:
	                    print(f"[WARNING] No valid processed data available for {timeframe}.")
	            else:
	                print(f"[WARNING] No raw market data fetched for {timeframe}.")
                
	        return True

	    except Exception as e:
	        print(f"[ERROR] Market data update failed: {e}")
	        return False



    def prepare_state(self):
        required_timeframes = ['5m', '15m']
        min_candles = 20  # Minimum candles needed for feature calculation

        for tf in required_timeframes:
            if tf not in self.bot.timeframes or len(self.bot.timeframes[tf].data) < min_candles:
                print(f"[WARNING] Insufficient data for {tf} timeframe. Need at least {min_candles} candles.")
                return None

        try:
            features = self.bot.get_combined_features()
            return features
        except Exception as e:
            print(f"[ERROR] State preparation failed: {e}")
            return None

    def run(self):
        """Main trading loop with real-time learning."""
        episodes = 1000
        batch_size = 32

        for episode in tqdm(range(episodes), desc="Episodes"):
            try:
                self.update_market_data()
                state = self.prepare_state()
                if state is None:
                    print("[WARNING] Insufficient state data. Waiting for more candles...")
                    time.sleep(60)
                    continue

                action = self.agent.act(state)
                current_price = self.get_current_price()
                if current_price is None:
                    print("[WARNING] Failed to fetch current price. Retrying...")
                    time.sleep(60)
                    continue

                if action == 1:  # Buy Signal
                    execute_trade(
                        "BUY",
                        self.config['trade_parameters']['order_amount'],
                        self.config,
                        current_price,
                        self.exchange,
                    )
                elif action == 2:  # Sell Signal
                    execute_trade(
                        "SELL",
                        self.config['trade_parameters']['order_amount'],
                        self.config,
                        current_price,
                        self.exchange,
                    )
                else:
                    print("[INFO] Hold signal detected.")

                unrealized_pnl = self.state_manager.get_unrealized_pnl(self.config['trade_parameters']['symbol'])
                transaction_costs = self.config['trade_parameters'].get('transaction_costs', 0)
                reward = unrealized_pnl - transaction_costs

                next_state = self.prepare_state()
                done = False
                self.agent.remember(state, action, reward, next_state, done)

                if len(self.agent.memory) > batch_size:
                    self.agent.replay(batch_size)

                time.sleep(60)

            except KeyboardInterrupt:
                print("\n[INFO] Gracefully shutting down trading bot...")
                break

            except Exception as e:
                print(f"[ERROR] An unexpected error occurred: {e}")


if __name__ == "__main__":
    trading_system = TradingSystem()
    trading_system.run()
