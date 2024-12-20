import time
from datetime import datetime
import numpy as np
from tqdm import tqdm

from utils import load_config
from data_fetcher import fetch_live_data_with_cache
from exchange_setup import initialize_exchange
from indicator_tracker import PhemexBot
from state_manager import StateManager
from dqn_agent import DQNLSTMAgent  # LSTM-based DQN Agent
from bot import execute_trade, calculate_tp_sl  # Trade execution utilities


class TradingSystem:
    def __init__(self):
        # Load configuration
        self.config = load_config()

        # Initialize exchange and components
        self.exchange = initialize_exchange()
        self.state_manager = StateManager(self.exchange)
        self.bot = PhemexBot()

        # Initialize LSTM-Based DQN Agent
        state_size = 6  # Example: EMA_20, EMA_200, ATR, VWAP, volatility, current_price
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
        """Fetch and cache 5-minute market data."""
        try:
            print("[DEBUG] Fetching new 5m data...")
            data = fetch_live_data_with_cache(
                self.config['trade_parameters']['symbol'],
                exchange=self.exchange,
                limit=1000
            )
            if not data.empty:
                print(f"[DEBUG] Fetched {len(data)} candles for 5m timeframe.")
                for _, row in data.iterrows():
                    self.bot.add_candle('5m', row['timestamp'], row['open'], row['high'], row['low'], row['close'], row['volume'])
            else:
                print("[WARNING] No data fetched for 5m timeframe.")
        except Exception as e:
            print(f"[ERROR] Failed to update market data: {e}")

    def prepare_state(self):
        """Prepare normalized state vector from PhemexBot."""
        state = self.bot.get_combined_features()
        if state is None:
            print("[WARNING] Insufficient data for state preparation.")
            return None
        return state

    def run(self):
        """Main trading loop with real-time learning."""
        episodes = 1000
        batch_size = 32

        for episode in tqdm(range(episodes), desc="Episodes"):
            try:
                # Update market data and indicators
                self.update_market_data()

                # Prepare state for the DQN agent
                state = self.prepare_state()
                if state is None:
                    time.sleep(60)
                    continue

                # Use DQN agent to decide action: Hold=0, Buy=1, Sell=2
                action = self.agent.act(state)

                # Fetch the current price for trade execution
                current_price = self.get_current_price()
                if current_price is None:
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

                # Calculate reward based on PnL and transaction costs (example logic)
                unrealized_pnl = self.state_manager.get_unrealized_pnl(self.config['trade_parameters']['symbol'])
                transaction_costs = self.config['trade_parameters'].get('transaction_costs', 0)
                reward = unrealized_pnl - transaction_costs

                # Store experience in replay memory and train periodically
                next_state = self.prepare_state()
                done = False  # Define when an episode ends (e.g., end of trading session)
                self.agent.remember(state, action, reward, next_state, done)

                if len(self.agent.memory) > batch_size:
                    self.agent.replay(batch_size)

                time.sleep(60)  # Wait before next cycle

            except KeyboardInterrupt:
                print("\n[INFO] Gracefully shutting down trading bot...")
                break

            except Exception as e:
                print(f"[ERROR] An unexpected error occurred: {e}")


if __name__ == "__main__":
    trading_system = TradingSystem()
    trading_system.run()
