import time
import json
from enum import Enum
import numpy as np
import pandas as pd




# Define shared constant
POSITION_CHECK_INTERVAL = 180  # Interval to check positions (in seconds)


# Bot States
class BotState(Enum):
    SEARCHING = 1
    TRADING = 2


class StateManager:
    def __init__(self, exchange):
        self.exchange = exchange
        self.current_state = BotState.SEARCHING
        self.last_position_check = 0
        self.position_check_interval = POSITION_CHECK_INTERVAL

    def get_account_positions(self, currency='BTC'):
        """Fetch account positions for a given currency."""
        try:
            params = {'currency': currency}
            response = self.exchange.privateGetAccountsAccountPositions(params=params)

            if isinstance(response, str):
                response = json.loads(response)

            positions = response.get('data', {}).get('positions', [])
            return positions if isinstance(positions, list) else []
        except Exception as e:
            print(f"[ERROR] Error fetching account positions: {e}")
            return []

    def get_open_positions(self, symbol):
        """Extract open positions for a given symbol."""
        positions = self.get_account_positions()
        return [
            p for p in positions
            if p.get('symbol') == symbol and abs(float(p.get('size', 0))) > 0
        ]

    def display_open_positions(self, symbol, current_price):
        """Display open positions for a given symbol along with PnL and leverage."""
        positions = self.get_open_positions(symbol)
        print("\nOpen Positions:")

        if not positions:
            print("No open positions found.")
            return 0

        for position in positions:
            size = float(position.get('size', 0))
            entry_price = float(position.get('avgEntryPrice', 0))
            leverage = float(position.get('leverage', 0))

            # Calculate Realized PnL
            realized_pnl = float(position.get('cumClosedPnlEv', 0)) / 1e8

            # Calculate Unrealized PnL
            unrealized_pnl = 0.0
            if size != 0 and entry_price > 0:  # Ensure valid position data
                if size > 0:  # Long position
                    unrealized_pnl = (current_price - entry_price) * size
                elif size < 0:  # Short position
                    unrealized_pnl = (entry_price - current_price) * abs(size)

            # Display position details
            print(f"Symbol: {position.get('symbol', 'N/A')}, "
                  f"Side: {'Long' if size > 0 else 'Short'}, "
                  f"Size: {abs(size)}, "
                  f"Entry Price: {entry_price:.2f}, "
                  f"Realized PnL: {realized_pnl:.2f}, "
                  f"Unrealized PnL: {unrealized_pnl:.2f}, "
                  f"Leverage: {leverage:.2f}")

        return len(positions)

    def check_and_display_positions(self, symbol, current_price):
        """Check and display open positions at specified intervals."""
        current_time = time.time()
    
        if current_time - self.last_position_check >= self.position_check_interval:
            try:
                # Get position details
                total_size, positions = self.get_positions_details(symbol)
            
                # Display positions with proper error handling
                open_position_count = self.display_open_positions(symbol, current_price)
            
                # Update bot state based on position count and risk metrics
                if open_position_count >= 5:
                    if total_size > self.max_position_size:
                        print("[WARNING] Total position size exceeds maximum allowed")
                        return False
                
                    self.current_state = BotState.TRADING
                    print(f"[INFO] Monitoring {open_position_count} active positions")
                else:
                    self.current_state = BotState.SEARCHING
                    print(f"[INFO] Searching for entries ({open_position_count}/5 positions open)")
            
                self.last_position_check = current_time
                return True
            
            except Exception as e:
                print(f"[ERROR] Position check failed: {e}")
                return False


    def get_positions_details(self, symbol):
        """Fetch total size and detailed open positions."""
        account_positions = self.get_account_positions()
        total_size = sum(abs(float(p.get('size', 0))) for p in account_positions)
        position_details = [
            {
                'id': p.get('id', 'N/A'),
                'symbol': p.get('symbol', 'N/A'),
                'side': 'Long' if float(p.get('size', 0)) > 0 else 'Short',
                'size': abs(float(p.get('size', 0))),
                'entry_price': float(p.get('avgEntryPrice', 0))
            }
            for p in account_positions if p.get('symbol') == symbol and abs(float(p.get('size', 0))) > 0
        ]
        return total_size, position_details


def get_positions_details(exchange, symbol):
    """Fetch total size and detailed open positions."""
    state_manager = StateManager(exchange)
    account_positions = state_manager.get_account_positions()
    total_size = sum(abs(float(p.get('size', 0))) for p in account_positions)
    position_details = [
        {
            'id': p.get('id', 'N/A'),
            'symbol': p.get('symbol', 'N/A'),
            'side': 'Long' if float(p.get('size', 0)) > 0 else 'Short',
            'size': abs(float(p.get('size', 0))),
        }
        for p in account_positions if p.get('symbol') == symbol and abs(float(p.get('size', 0))) > 0
    ]
    return total_size, position_details
