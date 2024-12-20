import ccxt

import pandas as pd

import time

from datetime import datetime

from utils import load_config

from exchange_setup import initialize_exchange

from data_fetcher import fetch_live_data

from indicator_tracker import PhemexBot

from state_manager import StateManager, get_positions_details

from main import get_current_price



# Load configuration

config = load_config()

phemex_futures = initialize_exchange()

state_manager = StateManager(phemex_futures)

MAX_SIZE = config['trade_parameters']['max_size']


# Set leverage

phemex_futures.set_leverage(20, 'BTC/USD:BTC')


def calculate_tp_sl(order_type, current_price, atr, ema_20, ema_200):

#    Dynamically calculate Take Profit and Stop Loss prices using ATR and EMA levels.
#    Args:
#        order_type (str): "BUY" or "SELL"
#        current_price (float): Current market price
#        atr (float): Average True Range for volatility adjustment
#        ema_20 (float): 20 EMA level
#        ema_200 (float): 200 EMA level
#    Returns:
#        tuple: (take_profit_price, stop_loss_price)

 # Set ATR multipliers for risk/reward adjustment

    tp_multiplier = 1.5

    sl_multiplier = 1.5



    if order_type == "BUY":

        # Use the higher of 20 EMA or ATR-adjusted SL as the stop-loss level

        stop_loss_price = max(ema_20, current_price - (atr * sl_multiplier))

        take_profit_price = current_price + (atr * tp_multiplier)



    elif order_type == "SELL":

        # Use the lower of 20 EMA or ATR-adjusted SL as the stop-loss level

        stop_loss_price = min(ema_20, current_price + (atr * sl_multiplier))

        take_profit_price = current_price - (atr * tp_multiplier)



    # Ensure TP/SL values are rounded for precision

    return round(take_profit_price, 2), round(stop_loss_price, 2)




def execute_trade(order_type, amount, config, current_price, exchange):

    """Execute a trade with retry and error handling."""

    try:

        symbol = config['trade_parameters']['symbol']



        # Fetch positions details

        total_size, position_details = get_positions_details(exchange, symbol)



        # Check if adding this trade exceeds MAX_SIZE

        if total_size + amount > MAX_SIZE:

            print(f"[WARNING] Max size exceeded. Current total: {total_size}, Attempted: {amount}, Max: {MAX_SIZE}")

            return



        # Fetch EMA levels and ATR for dynamic TP/SL

        ema_20, ema_200, _ = fetch_live_data(symbol)  # Ensure this returns required data

        atr = calculate_atr(symbol)  # Calculate ATR



        if ema_20 is None or ema_200 is None or atr is None:

            print("[ERROR] Failed to fetch necessary data for TP/SL calculation. Aborting trade.")

            return



        # Dynamically calculate TP and SL

        take_profit_price, stop_loss_price = calculate_tp_sl(order_type, current_price, atr, ema_20, ema_200)



        # Set limit price for the order

        # Example with ATR

        atr = calculate_atr(data)  # Ensure ATR is fetched

        limit_price = round(current_price + (atr * (-1 if order_type == "SELL" else 1)), 5)

        print(f"[DEBUG] Dynamic Limit Price for {order_type}: {limit_price}")



        # Log TP/SL and limit price details

        print(f"[DEBUG] Setting Take Profit at {take_profit_price}")

        print(f"[DEBUG] Setting Stop Loss at {stop_loss_price}")

        print(f"[DEBUG] Setting Limit Price at {limit_price}")



        # Place the limit order

        try:

            order = exchange.create_order(

                symbol=symbol,

                type="limit",

                side=order_type.lower(),

                amount=amount,

                price=limit_price,

                params={"ordType": "Limit"}

            )

            print("[INFO] Limit order placed. Waiting for execution...")

            time.sleep(5)



            # Check order status and handle it

            order_status = exchange.fetch_order(order['id'], symbol)

            if order_status.get('status') != 'closed':

                print("[INFO] Limit order not filled. Canceling order...")

                exchange.cancel_order(order['id'], symbol)

                print("[INFO] Limit order canceled. No fallback order placed.")



        except Exception as order_error:

            print(f"[ERROR] Failed to place or manage the limit order: {order_error}")

            return



        # Place Stop Loss and Take Profit orders

        tp_sl_side = "BUY" if order_type == "SELL" else "SELL"

        try:

            # Stop Loss

            exchange.create_order(

                symbol=symbol,

                type="stop",

                side=tp_sl_side.lower(),

                amount=amount,

                price=stop_loss_price,

                params={"ordType": "Stop", "stopPx": stop_loss_price}

            )

            print(f"[INFO] Stop Loss order placed at {stop_loss_price}.")



            # Take Profit

            exchange.create_order(

                symbol=symbol,

                type="limit",

                side=tp_sl_side.lower(),

                amount=amount,

                price=take_profit_price,

                params={"ordType": "LimitIfTouched", "stopPx": take_profit_price}

            )

            print(f"[INFO] Take Profit order placed at {take_profit_price}.")

        except Exception as tp_sl_error:

            print(f"[ERROR] Failed to place TP/SL orders: {tp_sl_error}")



        # Log the trade details

        log_trade(order_type, amount, current_price, order)



    except Exception as main_error:

        print(f"[ERROR] Critical error in execute_trade: {main_error}")





def log_trade(order_type, amount, price, order=None):

    """Log trade details."""

    global last_trade  



    trade_data = {

        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "order_type": order_type,

        "amount": amount,

        "price": price,

        "status": "executed" if order else "simulated",

        "PnL": 0

    }



    if last_trade and last_trade["order_type"] != order_type:

        if order_type == "SELL":

            trade_data["PnL"] = (price - last_trade["price"]) * amount

        elif order_type == "BUY":

            trade_data["PnL"] = (last_trade["price"] - price) * amount



        print(f"[TRADE] {order_type} at {price} with amount {amount}. PnL: {trade_data['PnL']:.2f}")



    last_trade = trade_data

    trade_log.append(trade_data)



    cumulative_PnL = sum(trade["PnL"] for trade in trade_log)

    print(f"[SUMMARY] Cumulative PnL: {cumulative_PnL:.2f}")



    if order:

        print("\n=== Live Order Details ===")

        print(f"Order Type: {order_type}")

        print(f"Status: {order['info'].get('status', 'N/A')}")

        print(f"Order ID: {order['id']}")

        print(f"Symbol: {order['symbol']}")

        print(f"Price: {order.get('price', 'N/A')}")

        print(f"Amount: {order['amount']}")

        print(f"Average Executed Price: {order.get('average', 'N/A')}")

        print(f"Filled: {order['filled']}")

        print(f"Remaining: {order['remaining']}")

        print(f"Order Status: {order['status']}")

        print("=== End of Order Details ===\n")