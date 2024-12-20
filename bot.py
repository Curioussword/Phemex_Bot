import ccxt
import pandas as pd
import time
from datetime import datetime
from utils import load_config
from exchange_setup import initialize_exchange
from data_fetcher import fetch_live_data
from indicator_tracker import PhemexBot
from state_manager import StateManager, get_positions_details

# Load configuration
config = load_config()
phemex_futures = initialize_exchange()
state_manager = StateManager(phemex_futures)
MAX_SIZE = config['trade_parameters']['max_size']

# Set leverage
phemex_futures.set_leverage(20, 'BTC/USD:BTC')


def calculate_tp_sl(order_type, current_price, atr, ema_20, ema_200):
    """
    Dynamically calculate Take Profit and Stop Loss prices using ATR and EMA levels.
    """
    tp_multiplier = 1.5
    sl_multiplier = 1.5

    if order_type == "BUY":
        stop_loss_price = max(ema_20, current_price - (atr * sl_multiplier))
        take_profit_price = current_price + (atr * tp_multiplier)
    elif order_type == "SELL":
        stop_loss_price = min(ema_20, current_price + (atr * sl_multiplier))
        take_profit_price = current_price - (atr * tp_multiplier)

    return round(take_profit_price, 2), round(stop_loss_price, 2)


def execute_trade(order_type, amount, config, current_price, exchange):
    """
    Execute a trade with retry and error handling.
    """
    try:
        symbol = config['trade_parameters']['symbol']

        # Fetch positions details
        total_size, position_details = get_positions_details(exchange, symbol)

        # Check if adding this trade exceeds MAX_SIZE
        if total_size + amount > MAX_SIZE:
            print(f"[WARNING] Max size exceeded. Current total: {total_size}, Attempted: {amount}, Max: {MAX_SIZE}")
            return

        # Fetch EMA levels and ATR for dynamic TP/SL
        ema_20, ema_200 = fetch_live_data(symbol)[:2]  # Ensure this returns required data
        atr = fetch_live_data(symbol)[2]  # Ensure ATR is fetched

        if ema_20 is None or ema_200 is None or atr is None:
            print("[ERROR] Failed to fetch necessary data for TP/SL calculation. Aborting trade.")
            return

        # Dynamically calculate TP and SL levels
        take_profit_price, stop_loss_price = calculate_tp_sl(order_type, current_price, atr, ema_20, ema_200)

        # Place the limit order
        limit_price = round(current_price + (atr * (-1 if order_type == "SELL" else 1)), 5)
        print(f"[DEBUG] Dynamic Limit Price for {order_type}: {limit_price}")

        order = exchange.create_order(
            symbol=symbol,
            type="limit",
            side=order_type.lower(),
            amount=amount,
            price=limit_price,
            params={"ordType": "Limit"}
        )
        print("[INFO] Limit order placed.")

        # Place Stop Loss and Take Profit orders
        tp_sl_side = "BUY" if order_type == "SELL" else "SELL"
        exchange.create_order(
            symbol=symbol,
            type="stop",
            side=tp_sl_side.lower(),
            amount=amount,
            price=stop_loss_price,
            params={"ordType": "Stop", "stopPx": stop_loss_price}
        )
        print(f"[INFO] Stop Loss order placed at {stop_loss_price}.")

        exchange.create_order(
            symbol=symbol,
            type="limit",
            side=tp_sl_side.lower(),
            amount=amount,
            price=take_profit_price,
            params={"ordType": "LimitIfTouched", "stopPx": take_profit_price}
        )
        print(f"[INFO] Take Profit order placed at {take_profit_price}.")

        # Log the trade details
        log_trade(order_type, amount, current_price)

    except Exception as main_error:
        print(f"[ERROR] Critical error in execute_trade: {main_error}")


def log_trade(order_type, amount, price):
    """
    Log trade details.
    """
    global last_trade

    trade_data = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "order_type": order_type,
        "amount": amount,
        "price": price,
    }

    if last_trade and last_trade["order_type"] != order_type:
        if order_type == "SELL":
            trade_data["PnL"] = (price - last_trade["price"]) * amount
        elif order_type == "BUY":
            trade_data["PnL"] = (last_trade["price"] - price) * amount

    last_trade = trade_data


def update_replay_memory(agent, state, action, reward, next_state):
    """
    Update replay memory with experience tuple.
    """
    done = False  # Define when an episode ends (e.g., end of trading session)
    agent.remember(state, action, reward, next_state, done)


# Example integration point for LSTM-based DQN Agent
def integrate_with_agent(agent):
    """
    Example function to integrate bot.py with LSTM-based DQN Agent.
    """
    state = PhemexBot().get_combined_features()
    action = agent.act(state)

    if action == 1:  # Buy Signal
        execute_trade("BUY", config['trade_parameters']['order_amount'], config)
    elif action == 2:  # Sell Signal
        execute_trade("SELL", config['trade_parameters']['order_amount'], config)
