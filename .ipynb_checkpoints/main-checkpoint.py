import time
from datetime import datetime

from utils import load_config
from data_fetcher import fetch_live_data
from exchange_setup import initialize_exchange
from indicator_tracker import PhemexBot
from state_manager import StateManager
from dqn_agent import DQNAgent  # New DQN agent

class TradingSystem:
    def __init__(self):
        # Load configuration
        self.config = load_config()

        # Initialize exchange and components
        self.exchange = initialize_exchange()
        self.state_manager = StateManager(self.exchange)
        self.bot = PhemexBot()

        # Initialize DQN Agent
        state_size = 5  # Example: EMA_20, EMA_200, ATR, current_price, position size
        action_size = 3  # Actions: [Hold=0, Buy=1, Sell=2]
        self.agent = DQNAgent(state_size=state_size, action_size=action_size)

        # Trade log and caching
        self.trade_log = []
        self.last_trade = None
        self.cached_data = {
            '5m': {'data': None, 'last_update': 0}
        }

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
        current_time = time.time()
        cache_ttl = 300  # Cache TTL for 5-minute data

        if (self.cached_data['5m']['data'] is None or 
            current_time - self.cached_data['5m']['last_update'] > cache_ttl):
            print("[DEBUG] Fetching new 5m data...")
            data = fetch_live_data(
                self.config['trade_parameters']['symbol'],
                interval='5m',
                exchange=self.exchange,
                limit=1000
            )
            if not data.empty:
                print(f"[DEBUG] Fetched {len(data)} candles for 5m timeframe.")
                self.cached_data['5m']['data'] = data
                self.cached_data['5m']['last_update'] = current_time

                # Update indicators in PhemexBot
                for _, row in data.iterrows():
                    self.bot.add_5m_candle(
                        row['timestamp'], row['open'], row['high'], row['low'], row['close'], row['volume']
                    )
            else:
                print("[WARNING] No data fetched for 5m timeframe.")

    def run(self):
        """Main trading loop with real-time learning."""
        print("[INFO] Trading system initialized")
        episodes = 1000  # Number of episodes for training

        for episode in range(episodes):
            print(f"Starting Episode {episode + 1}/{episodes}")

        try:
            # Update market data
            self.update_market_data()

            # Fetch indicators from PhemexBot
            indicators = self.bot.get_indicators('5m')
            if not indicators:
                print("[WARNING] Insufficient indicators for decision-making. Skipping...")
                time.sleep(60)
                continue

            # Fetch current price and position details
            current_price = self.get_current_price()
            if current_price is None:
                time.sleep(60)
                continue

            total_size, position_details = self.state_manager.get_positions_details(
                self.exchange, 
                self.config['trade_parameters']['symbol']
            )
            entry_price = position_details[0]['entry_price'] if position_details else None

            # Prepare state for DQN agent
            state = [
                indicators['EMA_20'],
                indicators['EMA_200'],
                indicators['ATR'],
                current_price,
                total_size  # Position size from StateManager
            ]
            state = np.array(state).reshape(1, -1)

            # Use DQN agent to decide action: Hold=0, Buy=1, Sell=2
            action = self.agent.act(state)

            # Execute trade based on action
            if action == 1:  # Buy Signal
                print("[INFO] Buy signal detected.")
                execute_buy(
                    amount=self.config['trade_parameters']['order_amount'],
                    symbol=self.config['trade_parameters']['symbol'],
                    price=current_price,
                    exchange=self.exchange
                )

            elif action == 2:  # Sell Signal
                print("[INFO] Sell signal detected.")
                execute_sell(
                    amount=self.config['trade_parameters']['order_amount'],
                    symbol=self.config['trade_parameters']['symbol'],
                    price=current_price,
                    exchange=self.exchange
                )

            else:
                print("[INFO] Hold signal detected. Monitoring...")

            # Fetch updated market data and calculate new indicators for next state
            next_indicators = self.bot.get_indicators('5m')
            if not next_indicators:
                print("[WARNING] Insufficient next indicators. Ending episode...")
                break

            next_state = np.array([
                next_indicators['EMA_20'],
                next_indicators['EMA_200'],
                next_indicators['ATR'],
                current_price,
                total_size
            ]).reshape(1, -1)

            # Calculate reward using StateManager details
            reward = self.calculate_reward(current_price=current_price,
                                           entry_price=entry_price,
                                           position_size=total_size)

            done = False  # Set to True if a specific condition ends this episode

            # Store experience in memory and train the agent with replay memory
            self.agent.remember(state, action, reward, next_state, done)

            if len(self.agent.memory) > 32:  # Train only when enough experiences are stored
                self.agent.replay(batch_size=32)

        except KeyboardInterrupt:
            print("\n[INFO] Gracefully shutting down trading bot...")
            break

        except Exception as e:
            print(f"[ERROR] An unexpected error occurred: {e}")



if __name__ == "__main__":
    trading_system = TradingSystem()
    trading_system.run()
