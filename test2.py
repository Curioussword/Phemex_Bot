import time
import pandas as pd
import ccxt
from utils import load_config
from data_fetcher import fetch_live_data
from exchange_setup import initialize_exchange
from indicator_tracker import PhemexBot
from state_manager import StateManager
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def get_current_price(exchange, symbol):
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker['last']
    except Exception as e:
        logging.error(f"Failed to fetch current price: {e}")
        return None

def determine_volatility(data):
    """Evaluate volatility based on price changes."""
    try:
        recent_close_prices = data["close"]
        if len(recent_close_prices) < 5:
            return 5  # Default to 5 minutes if insufficient data

        volatility_measure = abs(recent_close_prices.iloc[-1] - recent_close_prices.iloc[-5]) / recent_close_prices.iloc[-5]
        return 1 if volatility_measure > 0.002 else 5  # Switch intervals based on threshold
    except Exception as e:
        logging.error(f"Error in determine_volatility: {e}")
        return 5

def evaluate_market_conditions(indicators, config, current_price):
    """Evaluate market conditions and execute trades."""
    try:
        vwap_1m = indicators['1m']['vwap']
        ema_20_1m = indicators['1m']['ema_20']
        ema_200_1m = indicators['1m']['ema_200']
        
        vwap_5m = indicators['5m']['vwap']
        ema_20_5m = indicators['5m']['ema_20']
        ema_200_5m = indicators['5m']['ema_200']

        logging.info(f"Current Price: {current_price}")
        logging.info(f"[1M] VWAP: {vwap_1m}, EMA 20: {ema_20_1m}, EMA 200: {ema_200_1m}")
        logging.info(f"[5M] VWAP: {vwap_5m}, EMA 20: {ema_20_5m}, EMA 200: {ema_200_5m}")

        threshold = config['trade_parameters']['neutral_zone_threshold']
        
        if abs(ema_20_1m - ema_200_1m) < threshold and abs(ema_20_5m - ema_200_5m) < threshold:
            logging.info("Market is within the neutral zone. No trades executed.")
            return
        
        # Trading conditions logic...
        
    except Exception as e:
        logging.error(f"An error occurred in evaluate_market_conditions: {e}")

def main():
    config = load_config()
    phemex_futures = initialize_exchange()
    state_manager = StateManager(phemex_futures)
    bot = PhemexBot()

    # Set leverage
    phemex_futures.set_leverage(20, 'BTC/USD:BTC')

    while True:
        try:
            limit = 2000  # Required periods for indicators

            # Fetch live data for both timeframes
            data_1m = fetch_live_data(config['trade_parameters']['symbol'], interval="1m", exchange=phemex_futures, limit=limit)
            data_5m = fetch_live_data(config['trade_parameters']['symbol'], interval="5m", exchange=phemex_futures, limit=limit)

            if data_1m.empty or data_5m.empty:
                logging.error("No data fetched. Retrying...")
                time.sleep(60)
                continue

            # Update bot with new candle data
            for _, row in data_1m.iterrows():
                bot.add_1m_candle(row['timestamp'], row['open'], row['high'], row['low'], row['close'], row['volume'])

            for _, row in data_5m.iterrows():
                bot.add_5m_candle(row['timestamp'], row['open'], row['high'], row['low'], row['close'], row['volume'])

            # Get current price
            current_price = get_current_price(phemex_futures, config['trade_parameters']['symbol'])
            if current_price is None:
                logging.error("Failed to get current price. Retrying...")
                time.sleep(60)
                continue

            # Determine market volatility to adjust timeframe dynamically
            current_interval = determine_volatility(data_1m)
            timeframe = '1m' if current_interval == 1 else '5m'
            indicators = bot.get_indicators(timeframe)

            # Check positions and evaluate market conditions
            state_manager.check_and_display_positions(config['trade_parameters']['symbol'], current_price)
            
            evaluate_market_conditions(indicators, config, current_price)

            time.sleep(60)  # Wait before next iteration

        except KeyboardInterrupt:
            logging.info("Exiting trading bot...")
            break

        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
