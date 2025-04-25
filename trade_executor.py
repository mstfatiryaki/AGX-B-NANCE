from random import choice
import random
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SentientTrader.AI - Trade Executor V2
-------------------------------------
This module reads strategy decisions, executes trades in real-time
using Binance API, and logs trade data for performance analysis.
"""

import os
import sys
import json
import time
import uuid
import hmac
import hashlib
import logging
import datetime
import argparse
import requests
import subprocess
import threading
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from urllib.parse import urlencode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("trade_executor.log")]
)
logger = logging.getLogger("TradeExecutor")

# Try to import colorama for terminal colors
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORS = {
        "blue": Fore.BLUE,
        "green": Fore.GREEN,
        "red": Fore.RED,
        "yellow": Fore.YELLOW,
        "magenta": Fore.MAGENTA,
        "cyan": Fore.CYAN,
        "white": Fore.WHITE,
        "bright": Style.BRIGHT,
        "reset": Style.RESET_ALL
    }
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORS = {
        "blue": "", "green": "", "red": "", "yellow": "", 
        "magenta": "", "cyan": "", "white": "", "bright": "", "reset": ""
    }
    COLORAMA_AVAILABLE = False
    logger.warning("Colorama library not found. Terminal colors will be disabled.")

# Try to import alert system
try:
    import alert_system
    ALERT_SYSTEM_AVAILABLE = True
except ImportError:
    ALERT_SYSTEM_AVAILABLE = False
    logger.warning("Alert system not found. Alerts will be disabled.")

# Constants and configuration
REAL_MODE = False  # Set to True for real trading, False for mock trading
CURRENT_TIME = "2025-04-21 19:15:19"  # UTC
CURRENT_USER = "mstfatiryaki"

# File paths
STRATEGY_DECISION_FILE = "strategy_decision.json"
EXECUTED_TRADES_LOG = "executed_trades_log.json"

# Strategy modules
STRATEGY_MODULES = {
    "long_strategy.py": "LONG",
    "short_strategy.py": "SHORT",
    "sniper_strategy.py": "SNIPER"
}

# Trading parameters
DEFAULT_TRADE_CONFIG = {
    "max_trades_per_execution": 10,
    "max_trades_per_coin": 3,
    "trade_size_percentage": 2.0,  # % of available balance per trade
    "use_market_orders": True,     # Use market orders (vs limit orders)
    "auto_take_profit": True,      # Set automatic take profit
    "auto_stop_loss": True,        # Set automatic stop loss
    "tp_percentage": 3.0,          # Default take profit percentage
    "sl_percentage": 1.5,          # Default stop loss percentage
    "max_slippage": 0.5,           # Max acceptable slippage %
    "max_retries": 3,              # Max retry attempts for failed trades
    "cooldown_seconds": 5          # Cooldown between retries
}

# Binance API configuration (to be loaded from .env)
BINANCE_CONFIG = {
    "api_key": "",
    "api_secret": "",
    "base_url": "https://api.binance.com",
    "testnet_url": "https://testnet.binance.vision"
}

class TradeExecutor:
    """SentientTrader.AI trade execution system"""
    
    def __init__(self, real_mode: bool = REAL_MODE, simulation_mode: bool = False):
        """
        Initialize the trade executor
        
        Args:
            real_mode (bool): Whether to execute real trades (vs mock)
            simulation_mode (bool): Whether running in simulation mode
        """
        self.real_mode = real_mode
        self.simulation_mode = simulation_mode
        self.strategy_decision = {}
        self.executed_trades = []
        self.new_trades = []
        self.active_trades = []
        self.trade_config = DEFAULT_TRADE_CONFIG.copy()
        self.available_balance = 0.0
        
        # Load configuration
        self._load_config()
        
        # Initialize executed trades log
        self._load_executed_trades()
        
        # Track subprocess status
        self.subprocesses = []
        
        # Load existing trades
        self._load_executed_trades()
        
        # Initialize Binance API
        self._init_binance_api()
        
        logger.info(f"Trade Executor initialized (REAL_MODE: {self.real_mode})")
    
    def _load_config(self) -> None:
        """Load configuration from .env file"""
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            # Load API keys
            BINANCE_CONFIG["api_key"] = os.getenv("BINANCE_API_KEY", "")
            BINANCE_CONFIG["api_secret"] = os.getenv("BINANCE_API_SECRET", "")
            
            # Load any other configuration values
            if os.getenv("TRADE_SIZE_PERCENTAGE"):
                self.trade_config["trade_size_percentage"] = float(os.getenv("TRADE_SIZE_PERCENTAGE"))
            
            if os.getenv("TP_PERCENTAGE"):
                self.trade_config["tp_percentage"] = float(os.getenv("TP_PERCENTAGE"))
            
            if os.getenv("SL_PERCENTAGE"):
                self.trade_config["sl_percentage"] = float(os.getenv("SL_PERCENTAGE"))
            
            logger.info("Configuration loaded from .env file")
            
        except ImportError:
            logger.warning("Python-dotenv not installed. Using default configuration.")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
    
    def _init_binance_api(self) -> None:
        """Initialize Binance API with credentials"""
        if not BINANCE_CONFIG["api_key"] or not BINANCE_CONFIG["api_secret"]:
            logger.warning("Binance API credentials not found. Mock mode will be used regardless of REAL_MODE setting.")
            self.real_mode = False
            
            # Create a warning alert
            if ALERT_SYSTEM_AVAILABLE:
                alert_system.warning(
                    "Missing Binance API credentials. Using mock mode.",
                    {"module": "trade_executor", "action": "init"},
                    "trade_executor"
                )
    
    def _load_executed_trades(self) -> None:
        """Load existing executed trades from log file"""
        try:
            if os.path.exists(EXECUTED_TRADES_LOG):
                with open(EXECUTED_TRADES_LOG, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                if isinstance(data, dict) and "trades" in data:
                    self.executed_trades = data["trades"]
                    logger.info(f"Loaded {len(self.executed_trades)} trades from log file")
                else:
                    logger.warning(f"Invalid format in {EXECUTED_TRADES_LOG}, creating new log")
                    self._save_executed_trades()
            else:
                logger.info(f"{EXECUTED_TRADES_LOG} not found, creating new log")
                self._save_executed_trades()
        except Exception as e:
            logger.error(f"Error loading executed trades: {str(e)}")
            self._save_executed_trades()
    
    def _save_executed_trades(self) -> None:
        """Save executed trades to log file"""
        try:
            data = {
                "last_update": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user": CURRENT_USER,
                "trades": self.executed_trades
            }
            
            with open(EXECUTED_TRADES_LOG, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            
            logger.info(f"Saved {len(self.executed_trades)} trades to log file")
        except Exception as e:
            logger.error(f"Error saving executed trades: {str(e)}")
    
    def load_strategy_decision(self) -> bool:
        """
        Load strategy decision from JSON file
        
        Returns:
            bool: Success status
        """
        try:
            if not os.path.exists(STRATEGY_DECISION_FILE):
                logger.error(f"{STRATEGY_DECISION_FILE} not found")
                return False
            
            with open(STRATEGY_DECISION_FILE, "r", encoding="utf-8") as f:
                self.strategy_decision = json.load(f)
            
            # Validate strategy decision
            if not isinstance(self.strategy_decision, dict) or "strategy_decision" not in self.strategy_decision:
                logger.error(f"Invalid format in {STRATEGY_DECISION_FILE}")
                return False
            
            logger.info(f"Strategy decision loaded successfully")
            return True
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in {STRATEGY_DECISION_FILE}")
            return False
        except Exception as e:
            logger.error(f"Error loading strategy decision: {str(e)}")
            return False
    
    def get_current_balance(self) -> float:
        """
        Get current available balance from capital manager
        
        Returns:
            float: Available balance in USDT
        """
        try:
            # Try to get balance from capital_manager.py (in a real implementation)
            # For now, we'll simulate with a mock balance
            
            # In a real implementation, you'd call capital_manager.py to get the actual balance
            # Example: subprocess.run(['python', 'capital_manager.py', '--get-balance'])
            
            if self.real_mode:
                # Try to get actual balance from Binance
                account_info = self._binance_get_account()
                if account_info and "balances" in account_info:
                    for asset in account_info["balances"]:
                        if asset["asset"] == "USDT":
                            return float(asset["free"])
            
            # Mock balance for testing
            mock_balance = 10000.0
            logger.info(f"Using mock balance: ${mock_balance}")
            return mock_balance
            
        except Exception as e:
            logger.error(f"Error getting balance: {str(e)}")
            # Default to a safe balance
            return 5000.0
    
    def execute_strategy(self) -> bool:
        """
        Execute the selected strategy for target coins
        
        Returns:
            bool: Success status
        """
        try:
            strategy_data = self.strategy_decision.get("strategy_decision", {}).get("strategy", {})
            
            if not strategy_data:
                logger.error("No strategy data found in decision file")
                return False
            
            strategy_name = strategy_data.get("name")
            target_coins = strategy_data.get("target_coins", [])
            
            if not strategy_name or not target_coins:
                logger.error("Missing strategy name or target coins")
                return False
            
            # Check if strategy module exists
            if not os.path.exists(strategy_name):
                logger.error(f"Strategy module {strategy_name} not found")
                return False
            
            logger.info(f"Executing strategy {strategy_name} for coins: {', '.join(target_coins)}")
            print(f"{COLORS['bright']}{COLORS['cyan']}Executing Strategy: {strategy_name}{COLORS['reset']}")
            print(f"{COLORS['cyan']}Target Coins: {', '.join(target_coins)}{COLORS['reset']}")
            
            self.available_balance = self.get_current_balance()
            print(f"{COLORS['yellow']}Available Balance: ${self.available_balance:.2f}{COLORS['reset']}")
            
            # Define how much to allocate per coin
            max_coins = min(len(target_coins), self.trade_config["max_trades_per_execution"])
            allocation_per_coin = (self.available_balance * (self.trade_config["trade_size_percentage"] / 100)) / max_coins
            
            # Track successful executions
            successful_executions = 0
            
            # Execute the strategy for each target coin
            for coin in target_coins[:max_coins]:
                # Prepare environment variables for the subprocess
                env = os.environ.copy()
                env["COIN"] = coin
                env["ALLOCATION"] = str(allocation_per_coin)
                env["MODE"] = "REAL" if self.real_mode else "MOCK"
                
                # Run the strategy as a subprocess
                try:
                    print(f"{COLORS['cyan']}▶ Processing {coin}...{COLORS['reset']}")
                    
                    # Execute strategy with subprocess
                    cmd = [sys.executable, strategy_name, "--coin", coin, "--allocation", str(allocation_per_coin)]
                    if self.simulation_mode:
                        cmd.append("--simulation")
                    
                    result = subprocess.run(
                        cmd,
                        env=env,
                        capture_output=True,
                        text=True,
                        check=False
                    )
                    
                    # Check subprocess result
                    if result.returncode != 0:
                        logger.error(f"Strategy execution failed for {coin}: {result.stderr}")
                        print(f"{COLORS['red']}✗ Strategy failed for {coin}{COLORS['reset']}")
                        continue
                    
                    logger.info(f"Strategy executed successfully for {coin}")
                    print(f"{COLORS['green']}✓ Strategy executed for {coin}{COLORS['reset']}")
                    successful_executions += 1
                    
                    # Process any trades created by the strategy
                    self._process_new_trades()
                    
                except Exception as e:
                    logger.error(f"Error executing strategy for {coin}: {str(e)}")
                    print(f"{COLORS['red']}✗ Error processing {coin}: {str(e)}{COLORS['reset']}")
            
            logger.info(f"Strategy execution completed - {successful_executions}/{len(target_coins[:max_coins])} successful")
            return successful_executions > 0
            
        except Exception as e:
            logger.error(f"Error executing strategy: {str(e)}")
            return False
    
    def _process_new_trades(self) -> None:
        """Process new trades that were added to the executed trades log"""
        try:
            # Load the latest trades to get any new ones added by the strategy
            current_trade_ids = {t["trade_id"] for t in self.executed_trades if "trade_id" in t}
            
            with open(EXECUTED_TRADES_LOG, "r", encoding="utf-8") as f:
                updated_trades_data = json.load(f)
            
            if isinstance(updated_trades_data, dict) and "trades" in updated_trades_data:
                updated_trades = updated_trades_data["trades"]
                
                # Find new trades
                new_trades = []
                for trade in updated_trades:
                    if "trade_id" in trade and trade["trade_id"] not in current_trade_ids:
                        new_trades.append(trade)
                        
                        # Update our record to include this trade
                        self.executed_trades.append(trade)
                
                if new_trades:
                    logger.info(f"Found {len(new_trades)} new trades")
                    self.new_trades.extend(new_trades)
                    
                    # Process each new trade
                    for trade in new_trades:
                        self._handle_trade(trade)
        
        except Exception as e:
            logger.error(f"Error processing new trades: {str(e)}")
    
    def _handle_trade(self, trade: Dict[str, Any]) -> None:
        """
        Handle a new trade
        
        Args:
            trade (Dict[str, Any]): Trade data
        """
        try:
            # Log the trade
            symbol = trade.get("symbol", "UNKNOWN")
            operation = trade.get("operation", "UNKNOWN")
            entry_price = trade.get("entry_price", 0)
            
            print(f"{COLORS['bright']}New Trade: {symbol} {operation} at ${entry_price}{COLORS['reset']}")
            
            # If in real mode, execute the actual trade
            if self.real_mode:
                self._execute_real_trade(trade)
            else:
                self._execute_mock_trade(trade)
                
        except Exception as e:
            logger.error(f"Error handling trade: {str(e)}")
    
    def _execute_real_trade(self, trade: Dict[str, Any]) -> None:
        """
        Execute a real trade using Binance API
        
        Args:
            trade (Dict[str, Any]): Trade data
        """
        if not self.real_mode:
            return
        
        symbol = trade.get("symbol", "")
        operation = trade.get("operation", "")
        allocation = trade.get("investment_usd", 0)
        
        if not symbol or not operation or allocation <= 0:
            logger.error(f"Invalid trade parameters: {trade}")
            return
        
        # Prepare trading pair
        trading_pair = f"{symbol}USDT"
        
        try:
            # Get current price
            ticker_data = self._binance_get_ticker(trading_pair)
            if not ticker_data or "price" not in ticker_data:
                logger.error(f"Could not get price for {trading_pair}")
                return
            
            current_price = float(ticker_data["price"])
            
            # Calculate quantity
            quantity = allocation / current_price
            
            # Round quantity to appropriate precision (this is simplified)
            quantity = round(quantity, 5)
            
            # Execute order
            side = "BUY" if operation in ["LONG", "SNIPER"] else "SELL"
            
            # Create the order
            if self.trade_config["use_market_orders"]:
                order_result = self._binance_create_market_order(trading_pair, side, quantity)
            else:
                order_result = self._binance_create_limit_order(trading_pair, side, quantity, current_price)
            
            if not order_result:
                logger.error(f"Failed to create order for {trading_pair}")
                return
            
            logger.info(f"Order executed: {trading_pair} {side} {quantity} at ~${current_price}")
            
            # Update trade with real execution data
            trade["actual_entry_price"] = current_price
            trade["quantity"] = quantity
            trade["order_id"] = order_result.get("orderId", "")
            trade["status"] = "OPEN"
            trade["entry_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Update the trade in executed_trades
            for i, t in enumerate(self.executed_trades):
                if t.get("trade_id") == trade.get("trade_id"):
                    self.executed_trades[i] = trade
                    break
            
            # Save updated trades
            self._save_executed_trades()
            
        except Exception as e:
            logger.error(f"Error executing real trade: {str(e)}")
            
            # Send an alert
            if ALERT_SYSTEM_AVAILABLE:
                alert_system.error(
                    f"Failed to execute trade: {symbol} {operation}",
                    {"error": str(e), "symbol": symbol, "operation": operation},
                    "trade_executor"
                )
    
    def _execute_mock_trade(self, trade: Dict[str, Any]) -> None:
        """
        Simulate trade execution (mock mode)
        
        Args:
            trade (Dict[str, Any]): Trade data
        """
        symbol = trade.get("symbol", "")
        operation = trade.get("operation", "")
        allocation = trade.get("investment_usd", 0)
        
        if not symbol or not operation or allocation <= 0:
            logger.error(f"Invalid mock trade parameters: {trade}")
            return
        
        # Prepare trading pair
        trading_pair = f"{symbol}USDT"
        
        try:
            # Get current price or use the entry price from the trade
            current_price = trade.get("entry_price", 0)
            
            if current_price <= 0:
                # Try to get actual price from Binance
                ticker_data = self._binance_get_ticker(trading_pair)
                if ticker_data and "price" in ticker_data:
                    current_price = float(ticker_data["price"])
                else:
                    # Generate a mock price
                    current_price = self._generate_mock_price(symbol)
            
            # Calculate quantity
            quantity = allocation / current_price
            
            # Round quantity to appropriate precision
            quantity = round(quantity, 5)
            
            logger.info(f"Mock order executed: {trading_pair} {operation} {quantity} at ${current_price}")
            
            # Update trade with execution data
            trade["actual_entry_price"] = current_price
            trade["quantity"] = quantity
            trade["order_id"] = f"mock-{int(time.time())}"
            trade["status"] = "OPEN"
            trade["entry_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Also set up a mock exit for the trade after a random delay
            threading.Timer(
                random.randint(20, 60),  # Random delay between 20-60 seconds
                self._simulate_trade_exit,
                args=[trade]
            ).start()
            
            # Update the trade in executed_trades
            for i, t in enumerate(self.executed_trades):
                if t.get("trade_id") == trade.get("trade_id"):
                    self.executed_trades[i] = trade
                    break
            
            # Save updated trades
            self._save_executed_trades()
            
        except Exception as e:
            logger.error(f"Error executing mock trade: {str(e)}")
    
    def _simulate_trade_exit(self, trade: Dict[str, Any]) -> None:
        """
        Simulate a trade exit (for mock mode)
        
        Args:
            trade (Dict[str, Any]): Trade data
        """
        try:
            symbol = trade.get("symbol", "")
            operation = trade.get("operation", "")
            entry_price = trade.get("actual_entry_price", trade.get("entry_price", 0))
            quantity = trade.get("quantity", 0)
            
            # Generate a random price movement (0.5% to 3% in either direction)
            price_change_pct = random.uniform(-3.0, 3.0) / 100
            
            # For LONG trades, positive change is profit; for SHORT trades, negative change is profit
            if operation == "LONG":
                exit_price = entry_price * (1 + price_change_pct)
            else:  # SHORT or SNIPER
                exit_price = entry_price * (1 - price_change_pct)
            
            # Calculate profit/loss
            if operation == "LONG":
                profit_loss = (exit_price - entry_price) * quantity
            else:  # SHORT or SNIPER
                profit_loss = (entry_price - exit_price) * quantity
            
            # Update trade with exit data
            for i, t in enumerate(self.executed_trades):
                if t.get("trade_id") == trade.get("trade_id"):
                    self.executed_trades[i]["status"] = "CLOSED"
                    self.executed_trades[i]["exit_price"] = exit_price
                    self.executed_trades[i]["exit_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.executed_trades[i]["profit_loss"] = profit_loss
                    self.executed_trades[i]["roi"] = (profit_loss / (entry_price * quantity)) * 100
                    break
            
            # Save updated trades
            self._save_executed_trades()
            
            # Log the exit
            profit_str = "PROFIT" if profit_loss > 0 else "LOSS"
            logger.info(f"Mock trade exited: {symbol} {operation} at ${exit_price} - {profit_str}: ${profit_loss:.2f}")
            
            # Print to console with color
            color = COLORS["green"] if profit_loss > 0 else COLORS["red"]
            print(f"{color}Trade Closed: {symbol} {operation} - ${profit_loss:.2f} ({(profit_loss / (entry_price * quantity)) * 100:.2f}%){COLORS['reset']}")
            
        except Exception as e:
            logger.error(f"Error simulating trade exit: {str(e)}")
    
    def _generate_mock_price(self, symbol: str) -> float:
        """
        Generate a mock price for a symbol
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            float: Mock price
        """
        # Base prices for common cryptocurrencies
        base_prices = {
            "BTC": 65000.0,
            "ETH": 3500.0,
            "BNB": 600.0,
            "SOL": 150.0,
            "ADA": 0.45,
            "XRP": 0.55,
            "DOT": 7.5,
            "DOGE": 0.08,
            "AVAX": 35.0,
            "MATIC": 0.65,
            "LINK": 18.0
        }
        
        # Get base price for the symbol or use default
        base_price = base_prices.get(symbol, 100.0)
        
        # Add small random variation (+/- 1%)
        variation = random.uniform(-0.01, 0.01)
        mock_price = base_price * (1 + variation)
        
        return mock_price
    
    def run_post_trade_analysis(self) -> None:
        """Run post-trade analysis modules"""
        if not self.new_trades:
            logger.info("No new trades to analyze")
            return
        
        logger.info("Running post-trade analysis")
        print(f"\n{COLORS['cyan']}Running Post-Trade Analysis...{COLORS['reset']}")
        
        try:
            # Run risk manager
            if os.path.exists("risk_manager.py"):
                print(f"{COLORS['yellow']}▶ Running Risk Manager...{COLORS['reset']}")
                subprocess.run([sys.executable, "risk_manager.py"], check=False)
            
            # Run performance tracker
            if os.path.exists("performance_tracker.py"):
                print(f"{COLORS['yellow']}▶ Running Performance Tracker...{COLORS['reset']}")
                subprocess.run([sys.executable, "performance_tracker.py"], check=False)
            
            # Run learning engine
            if os.path.exists("learning_engine.py"):
                print(f"{COLORS['yellow']}▶ Running Learning Engine...{COLORS['reset']}")
                subprocess.run([sys.executable, "learning_engine.py"], check=False)
            
            print(f"{COLORS['green']}✓ Post-Trade Analysis Complete{COLORS['reset']}")
            
        except Exception as e:
            logger.error(f"Error running post-trade analysis: {str(e)}")
            print(f"{COLORS['red']}✗ Error in Post-Trade Analysis: {str(e)}{COLORS['reset']}")
    
    def _binance_get_timestamp(self) -> int:
        """
        Get current timestamp for Binance API
        
        Returns:
            int: Current timestamp in milliseconds
        """
        return int(time.time() * 1000)
    
    def _binance_generate_signature(self, params: Dict[str, Any]) -> str:
        """
        Generate signature for Binance API
        
        Args:
            params (Dict[str, Any]): Request parameters
            
        Returns:
            str: HMAC SHA256 signature
        """
        query_string = urlencode(params)
        signature = hmac.new(
            BINANCE_CONFIG["api_secret"].encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _binance_api_request(self, endpoint: str, method: str = "GET", 
                           params: Optional[Dict[str, Any]] = None, 
                           signed: bool = False) -> Optional[Dict[str, Any]]:
        """
        Make a request to Binance API
        
        Args:
            endpoint (str): API endpoint
            method (str, optional): HTTP method
            params (Dict[str, Any], optional): Request parameters
            signed (bool, optional): Whether the request needs signature
            
        Returns:
            Optional[Dict[str, Any]]: Response data or None on error
        """
        if not BINANCE_CONFIG["api_key"]:
            logger.error("Binance API key not configured")
            return None
        
        params = params or {}
        headers = {"X-MBX-APIKEY": BINANCE_CONFIG["api_key"]}
        url = f"{BINANCE_CONFIG['base_url']}{endpoint}"
        
        if signed:
            params["timestamp"] = self._binance_get_timestamp()
            params["signature"] = self._binance_generate_signature(params)
        
        try:
            if method == "GET":
                response = requests.get(url, headers=headers, params=params)
            elif method == "POST":
                response = requests.post(url, headers=headers, params=params)
            elif method == "DELETE":
                response = requests.delete(url, headers=headers, params=params)
            else:
                logger.error(f"Unsupported HTTP method: {method}")
                return None
            
            if response.status_code != 200:
                logger.error(f"Binance API error: {response.status_code} - {response.text}")
                return None
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Binance API request error: {str(e)}")
            return None
    
    def _binance_get_account(self) -> Optional[Dict[str, Any]]:
        """
        Get account information from Binance
        
        Returns:
            Optional[Dict[str, Any]]: Account data or None on error
        """
        return self._binance_api_request("/api/v3/account", signed=True)
    
    def _binance_get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get ticker price for a symbol
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            Optional[Dict[str, Any]]: Ticker data or None on error
        """
        return self._binance_api_request("/api/v3/ticker/price", params={"symbol": symbol})
    
    def _binance_create_market_order(self, symbol: str, side: str, quantity: float) -> Optional[Dict[str, Any]]:
        """
        Create a market order
        
        Args:
            symbol (str): Trading symbol
            side (str): Order side (BUY or SELL)
            quantity (float): Order quantity
            
        Returns:
            Optional[Dict[str, Any]]: Order data or None on error
        """
        params = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": quantity
        }
        
        return self._binance_api_request("/api/v3/order", method="POST", params=params, signed=True)
    
    def _binance_create_limit_order(self, symbol: str, side: str, quantity: float, price: float) -> Optional[Dict[str, Any]]:
        """
        Create a limit order
        
        Args:
            symbol (str): Trading symbol
            side (str): Order side (BUY or SELL)
            quantity (float): Order quantity
            price (float): Order price
            
        Returns:
            Optional[Dict[str, Any]]: Order data or None on error
        """
        params = {
            "symbol": symbol,
            "side": side,
            "type": "LIMIT",
            "timeInForce": "GTC",
            "quantity": quantity,
            "price": price
        }
        
        return self._binance_api_request("/api/v3/order", method="POST", params=params, signed=True)
    
    def display_summary(self) -> None:
        """Display execution summary in terminal"""
        print(f"\n{COLORS['bright']}{COLORS['cyan']}{'=' * 60}{COLORS['reset']}")
        print(f"{COLORS['bright']}{COLORS['cyan']}📊 TRADE EXECUTION SUMMARY{COLORS['reset']}")
        print(f"{COLORS['bright']}{COLORS['cyan']}{'=' * 60}{COLORS['reset']}")
        
        # Strategy info
        strategy_data = self.strategy_decision.get("strategy_decision", {}).get("strategy", {})
        strategy_name = strategy_data.get("name", "N/A")
        
        print(f"{COLORS['white']}Strategy: {COLORS['bright']}{strategy_name}{COLORS['reset']}")
        print(f"{COLORS['white']}Mode: {COLORS['yellow']}{'REAL' if self.real_mode else 'MOCK'}{COLORS['reset']}")
        print(f"{COLORS['white']}Time: {COLORS['yellow']}{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{COLORS['reset']}")
        
        # Trade summary
        if self.new_trades:
            print(f"\n{COLORS['bright']}New Trades: {len(self.new_trades)}{COLORS['reset']}")
            
            for i, trade in enumerate(self.new_trades[:5], 1):
                symbol = trade.get("symbol", "UNKNOWN")
                operation = trade.get("operation", "UNKNOWN")
                entry_price = trade.get("actual_entry_price", trade.get("entry_price", 0))
                allocation = trade.get("investment_usd", 0)
                
                print(f"{i}. {COLORS['bright']}{symbol}{COLORS['reset']} - {operation} at ${entry_price:.2f} (${allocation:.2f})")
            
            if len(self.new_trades) > 5:
                print(f"...and {len(self.new_trades) - 5} more trades")
        else:
            print(f"\n{COLORS['yellow']}No new trades executed{COLORS['reset']}")
        
        # Overall stats
        print(f"\n{COLORS['white']}Total Trades: {len(self.executed_trades)}{COLORS['reset']}")
        
        open_trades = [t for t in self.executed_trades if t.get("status") == "OPEN"]
        closed_trades = [t for t in self.executed_trades if t.get("status") == "CLOSED"]
        
        print(f"{COLORS['white']}Open Positions: {len(open_trades)}{COLORS['reset']}")
        print(f"{COLORS['white']}Closed Positions: {len(closed_trades)}{COLORS['reset']}")
        
        # Profit/Loss Summary
        if closed_trades:
            total_pnl = sum(t.get("profit_loss", 0) for t in closed_trades)
            winning_trades = [t for t in closed_trades if t.get("profit_loss", 0) > 0]
            win_rate = len(winning_trades) / len(closed_trades) * 100 if closed_trades else 0
            
            pnl_color = COLORS["green"] if total_pnl >= 0 else COLORS["red"]
            
            print(f"\n{COLORS['white']}Total P&L: {pnl_color}${total_pnl:.2f}{COLORS['reset']}")
            print(f"{COLORS['white']}Win Rate: {len(winning_trades)}/{len(closed_trades)} ({win_rate:.1f}%){COLORS['reset']}")
        
        print(f"\n{COLORS['bright']}{COLORS['cyan']}{'=' * 60}{COLORS['reset']}")
    
    def run(self) -> bool:
        """
        Run the trade execution process
        
        Returns:
            bool: Success status
        """
        print(f"{COLORS['bright']}{COLORS['cyan']}SentientTrader.AI - Trade Executor V2{COLORS['reset']}")
        print(f"{COLORS['cyan']}Mode: {'REAL' if self.real_mode else 'MOCK'} | {'SIMULATION' if self.simulation_mode else 'LIVE'}{COLORS['reset']}")
        
        try:
            # Reset new trades list
            self.new_trades = []
            
            # Step 1: Load strategy decision
            if not self.load_strategy_decision():
                print(f"{COLORS['red']}✗ Failed to load strategy decision{COLORS['reset']}")
                return False
            
            # Step 2: Execute the strategy
            if not self.execute_strategy():
                print(f"{COLORS['red']}✗ Strategy execution failed{COLORS['reset']}")
                return False
            
            # Step 3: Run post-trade analysis if any new trades
            if self.new_trades:
                self.run_post_trade_analysis()
            
            # Step 4: Display execution summary
            self.display_summary()
            
            return True
            
        except Exception as e:
            logger.error(f"Error in trade execution: {str(e)}")
            traceback.print_exc()
            
            if ALERT_SYSTEM_AVAILABLE:
                alert_system.error(
                    "Trade execution error",
                    {"error": str(e)},
                    "trade_executor"
                )
            
            print(f"{COLORS['red']}✗ Trade execution error: {str(e)}{COLORS['reset']}")
            return False

# For use in random number generation
import random

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="SentientTrader.AI Trade Executor")
    parser.add_argument("--real", action="store_true", help="Use real trading mode")
    parser.add_argument("--simulation", action="store_true", help="Run in simulation mode")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Initialize and run the trade executor
    executor = TradeExecutor(real_mode=args.real, simulation_mode=args.simulation)
    executor.run()
