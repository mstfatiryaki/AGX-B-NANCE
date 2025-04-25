from random import choice
import random
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SentientTrader.AI - Short Strategy V2
-------------------------------------
This module implements a short trading strategy that enters positions
on overbought conditions with decreasing volume and negative sentiment.
"""

import os
import sys
import json
import time
import uuid
import random
import logging
import datetime
import argparse
import threading
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("short_strategy.log")]
)
logger = logging.getLogger("ShortStrategy")

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

# Try to import capital manager
try:
    import capital_manager
    CAPITAL_MANAGER_AVAILABLE = True
except ImportError:
    CAPITAL_MANAGER_AVAILABLE = False
    logger.warning("Capital manager not found. Mock capital allocation will be used.")

# Try to import memory core
try:
    import memory_core
    MEMORY_CORE_AVAILABLE = True
except ImportError:
    MEMORY_CORE_AVAILABLE = False
    logger.warning("Memory core not found. Strategy history will not be stored.")

# Try to import performance tracker
try:
    import performance_tracker
    PERFORMANCE_TRACKER_AVAILABLE = True
except ImportError:
    PERFORMANCE_TRACKER_AVAILABLE = False
    logger.warning("Performance tracker not found. Performance will not be tracked.")

# Try to import learning engine
try:
    import learning_engine
    LEARNING_ENGINE_AVAILABLE = True
except ImportError:
    LEARNING_ENGINE_AVAILABLE = False
    logger.warning("Learning engine not found. Strategy improvements will not be learned.")

# Try to import transaction logger
try:
    import transaction_logger
    TRANSACTION_LOGGER_AVAILABLE = True
except ImportError:
    TRANSACTION_LOGGER_AVAILABLE = False
    logger.warning("Transaction logger not found. Transactions will not be logged.")

# Constants and configuration
REAL_MODE = False  # Set to True for real trading
SIMULATION_MODE = True  # Set to True for simulation mode (historical data)
CURRENT_TIME = "2025-04-21 21:11:26"  # UTC
CURRENT_USER = "mstfatiryaki"

# File paths
STRATEGY_DECISION_FILE = "strategy_decision.json"
COLLECTOR_DATA_FILE = "collector_data.json"
COINGECKO_DATA_FILE = "coingecko_data.json"
WALLET_STATE_FILE = "wallet_state.json"
EXECUTED_TRADES_LOG = "executed_trades_log.json"

# Trading parameters
TRADING_PARAMS = {
    "risk_per_trade": 0.1,  # Maximum 10% of available balance per trade
    "leverage": 1,  # No leverage for short strategy
    "stop_loss_percentage": 0.05,  # 5% stop loss (above entry for shorts)
    "take_profit_percentage": 0.15,  # 15% take profit (below entry for shorts)
    "max_rsi": 70,  # Overbought condition
    "max_volume_decrease": 0.8,  # 20% volume decrease (0.8 multiplier)
    "entry_timeout": 600,  # 10 minutes to enter a position
    "max_sentiment_score": 0.0,  # Maximum sentiment score to consider (negative is preferable)
    "confidence_threshold": 0.7,  # Minimum strategy confidence to execute
}

class ShortStrategy:
    """Implements a short trading strategy for cryptocurrency markets"""
    
    def __init__(self, real_mode: bool = REAL_MODE, simulation_mode: bool = SIMULATION_MODE):
        """
        Initialize the short strategy
        
        Args:
            real_mode (bool): Whether to execute real trades
            simulation_mode (bool): Whether to run in simulation mode
        """
        self.real_mode = real_mode
        self.simulation_mode = simulation_mode
        self.strategy_decision = {}
        self.market_data = {}
        self.sentiment_data = {}
        self.technical_indicators = {}
        self.executed_trades = []
        
        # Trade tracking
        self.current_trades = {}
        self.pending_trades = []
        
        # Set timestamp
        self.timestamp = CURRENT_TIME
        
        # Last check time for market data
        self.last_market_check = 0
        
        logger.info(f"Short Strategy initialized (REAL_MODE: {self.real_mode}, SIMULATION_MODE: {self.simulation_mode})")
    
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
            if not self._validate_strategy_decision():
                logger.error(f"Invalid format in {STRATEGY_DECISION_FILE}")
                return False
            
            # Check if this strategy should be executed
            strategy_name = self.strategy_decision.get("strategy_decision", {}).get("strategy", {}).get("name", "")
            if strategy_name != "short_strategy.py" and not self.simulation_mode:
                logger.info(f"Current strategy is {strategy_name}, not short_strategy.py. Exiting.")
                return False
            
            logger.info(f"Strategy decision loaded successfully")
            return True
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in {STRATEGY_DECISION_FILE}")
            return False
        except Exception as e:
            logger.error(f"Error loading strategy decision: {str(e)}")
            return False
    
    def _validate_strategy_decision(self) -> bool:
        """
        Validate strategy decision format
        
        Returns:
            bool: Whether strategy decision format is valid
        """
        if not isinstance(self.strategy_decision, dict):
            return False
        
        if "strategy_decision" not in self.strategy_decision:
            return False
        
        strategy_data = self.strategy_decision["strategy_decision"]
        
        if "strategy" not in strategy_data or "target_coins" not in strategy_data:
            return False
        
        return True
    
    def load_market_data(self) -> bool:
        """
        Load market data from collector files
        
        Returns:
            bool: Success status
        """
        try:
            # Try to load Binance data first
            binance_data_loaded = self._load_binance_data()
            
            # Try to load CoinGecko data as fallback
            coingecko_data_loaded = self._load_coingecko_data()
            
            if not binance_data_loaded and not coingecko_data_loaded:
                logger.error("Failed to load market data from any source")
                return False
            
            # Calculate additional technical indicators
            self._calculate_technical_indicators()
            
            self.last_market_check = time.time()
            logger.info("Market data loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading market data: {str(e)}")
            return False
    
    def _load_binance_data(self) -> bool:
        """
        Load market data from Binance collector
        
        Returns:
            bool: Success status
        """
        try:
            if not os.path.exists(COLLECTOR_DATA_FILE):
                logger.warning(f"{COLLECTOR_DATA_FILE} not found")
                return False
            
            with open(COLLECTOR_DATA_FILE, "r", encoding="utf-8") as f:
                collector_data = json.load(f)
            
            # Extract ticker data
            if "data" not in collector_data:
                logger.warning("No market data found in collector data")
                return False
            
            # Process ticker data
            ticker_data = collector_data["data"]
            
            # Create market data dictionary
            for ticker in ticker_data:
                symbol = ticker.get("symbol", "")
                
                # Skip non-USDT pairs if we're mainly trading against USDT
                if not symbol.endswith("USDT"):
                    continue
                
                # Extract coin symbol without USDT
                coin = symbol[:-4]
                
                # Store market data
                self.market_data[coin] = {
                    "symbol": symbol,
                    "price": float(ticker.get("lastPrice", 0)),
                    "price_change_24h": float(ticker.get("priceChangePercent", 0)),
                    "volume": float(ticker.get("volume", 0)),
                    "quote_volume": float(ticker.get("quoteVolume", 0)),
                    "high_24h": float(ticker.get("highPrice", 0)),
                    "low_24h": float(ticker.get("lowPrice", 0)),
                    "source": "binance",
                    "timestamp": collector_data.get("metadata", {}).get("timestamp", self.timestamp)
                }
            
            logger.info(f"Loaded market data for {len(self.market_data)} coins from Binance")
            return len(self.market_data) > 0
            
        except Exception as e:
            logger.error(f"Error loading Binance data: {str(e)}")
            return False
    
    def _load_coingecko_data(self) -> bool:
        """
        Load market data from CoinGecko collector
        
        Returns:
            bool: Success status
        """
        try:
            if not os.path.exists(COINGECKO_DATA_FILE):
                logger.warning(f"{COINGECKO_DATA_FILE} not found")
                return False
            
            with open(COINGECKO_DATA_FILE, "r", encoding="utf-8") as f:
                coingecko_data = json.load(f)
            
            # Extract coin data
            if "data" not in coingecko_data:
                logger.warning("No market data found in CoinGecko data")
                return False
            
            # Process coin data
            coin_data = coingecko_data["data"]
            
            # Create market data dictionary if not already populated by Binance
            for coin_info in coin_data:
                symbol = coin_info.get("symbol", "").upper()
                
                # Skip if already loaded from Binance
                if symbol in self.market_data:
                    continue
                
                # Store market data
                self.market_data[symbol] = {
                    "symbol": f"{symbol}USDT",  # Assuming USDT pair
                    "price": float(coin_info.get("current_price", 0)),
                    "price_change_24h": float(coin_info.get("price_change_percentage_24h", 0)),
                    "volume": float(coin_info.get("total_volume", 0)),
                    "market_cap": float(coin_info.get("market_cap", 0)),
                    "high_24h": float(coin_info.get("high_24h", 0)) if "high_24h" in coin_info else 0,
                    "low_24h": float(coin_info.get("low_24h", 0)) if "low_24h" in coin_info else 0,
                    "sentiment_score": float(coin_info.get("sentiment_votes_up_percentage", 50)) / 100,
                    "source": "coingecko",
                    "timestamp": coingecko_data.get("metadata", {}).get("timestamp", self.timestamp)
                }
            
            logger.info(f"Loaded market data for {len(self.market_data)} coins from CoinGecko")
            return len(self.market_data) > 0
            
        except Exception as e:
            logger.error(f"Error loading CoinGecko data: {str(e)}")
            return False
    
    def _calculate_technical_indicators(self) -> None:
        """Calculate technical indicators for market analysis"""
        try:
            for coin, data in self.market_data.items():
                price = data.get("price", 0)
                high = data.get("high_24h", 0)
                low = data.get("low_24h", 0)
                volume = data.get("volume", 0)
                
                # Skip coins with missing data
                if price == 0 or high == 0 or low == 0:
                    continue
                
                # Simple "RSI" approximation based on price position in 24h range
                # This is not a true RSI but a simplification for this example
                if high == low:
                    rsi = 50  # Neutral if no range
                else:
                    # Calculate where current price is in the high-low range (0-100%)
                    price_position = (price - low) / (high - low)
                    # For RSI: higher values = more overbought
                    rsi = price_position * 100
                
                # Volume decrease indicator (mock - in a real system, this would use historical data)
                volume_decrease = random.uniform(0.7, 1.1)  # Mock value for simulation
                
                # Store indicators
                self.technical_indicators[coin] = {
                    "rsi": rsi,
                    "volume_decrease": volume_decrease,
                    "price_to_ma_ratio": random.uniform(0.9, 1.1),  # Mock moving average ratio
                    "bollinger_band_position": random.uniform(-1.0, 1.0),  # Mock bollinger position
                    "volatility": (high - low) / price if price > 0 else 0  # 24h volatility
                }
            
            logger.info(f"Calculated technical indicators for {len(self.technical_indicators)} coins")
        
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
    
    def _load_sentiment_data(self) -> bool:
        """
        Load sentiment data if available
        
        Returns:
            bool: Success status
        """
        try:
            sentiment_file = "sentiment_summary.json"
            if not os.path.exists(sentiment_file):
                logger.info(f"{sentiment_file} not found, proceeding without sentiment data")
                return False
            
            with open(sentiment_file, "r", encoding="utf-8") as f:
                sentiment_data = json.load(f)
            
            # Extract coin sentiment
            if "coin_sentiment" in sentiment_data:
                for coin, data in sentiment_data["coin_sentiment"].items():
                    self.sentiment_data[coin] = {
                        "sentiment_score": data.get("sentiment_score", 0.0),
                        "confidence": data.get("confidence", 0.5),
                    }
            
            logger.info(f"Loaded sentiment data for {len(self.sentiment_data)} coins")
            return True
            
        except Exception as e:
            logger.error(f"Error loading sentiment data: {str(e)}")
            return False
    
    def check_entry_conditions(self, coin: str) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Check if entry conditions are met for a short position
        
        Args:
            coin (str): Coin to check
            
        Returns:
            Tuple[bool, float, Dict[str, Any]]: (Entry decision, confidence score, details)
        """
        try:
            # Get market data for the coin
            if coin not in self.market_data:
                logger.warning(f"No market data available for {coin}")
                return False, 0.0, {"error": "No market data"}
            
            market_data = self.market_data[coin]
            
            # Get technical indicators
            if coin not in self.technical_indicators:
                logger.warning(f"No technical indicators available for {coin}")
                return False, 0.0, {"error": "No technical indicators"}
            
            indicators = self.technical_indicators[coin]
            
            # Get sentiment data if available
            sentiment_score = 0.0
            sentiment_confidence = 0.5
            if coin in self.sentiment_data:
                sentiment_data = self.sentiment_data[coin]
                sentiment_score = sentiment_data.get("sentiment_score", 0.0)
                sentiment_confidence = sentiment_data.get("confidence", 0.5)
            
            # Check conditions
            conditions_met = {}
            
            # 1. Check RSI (overbought condition)
            rsi = indicators.get("rsi", 50)
            rsi_condition = rsi >= TRADING_PARAMS["max_rsi"]
            conditions_met["rsi_overbought"] = rsi_condition
            
            # 2. Check volume decrease
            volume_decrease = indicators.get("volume_decrease", 1.0)
            volume_condition = volume_decrease <= TRADING_PARAMS["max_volume_decrease"]
            conditions_met["volume_decreasing"] = volume_condition
            
            # 3. Check sentiment (negative is better for shorts)
            sentiment_condition = sentiment_score <= TRADING_PARAMS["max_sentiment_score"]
            conditions_met["negative_sentiment"] = sentiment_condition
            
            # 4. Price is in downtrend or turning point (simplified check)
            price_change = market_data.get("price_change_24h", 0)
            price_to_ma = indicators.get("price_to_ma_ratio", 1.0)
            downtrend_condition = price_change < 5.0 and price_to_ma <= 1.05
            conditions_met["potential_downtrend"] = downtrend_condition
            
            # Calculate overall confidence score (weighted average of conditions)
            weights = {
                "rsi_overbought": 0.35,
                "volume_decreasing": 0.25,
                "negative_sentiment": 0.2,
                "potential_downtrend": 0.2
            }
            
            confidence_score = sum(
                weights[condition] * (1.0 if met else 0.0)
                for condition, met in conditions_met.items()
            )
            
            # Adjust by sentiment confidence if available
            confidence_score = confidence_score * (0.8 + (sentiment_confidence * 0.2))
            
            # Entry decision
            entry_decision = confidence_score >= TRADING_PARAMS["confidence_threshold"]
            
            # Prepare details
            details = {
                "conditions": conditions_met,
                "indicators": {
                    "rsi": rsi,
                    "volume_decrease": volume_decrease,
                    "sentiment_score": sentiment_score,
                    "price_change_24h": price_change,
                    "price_to_ma_ratio": price_to_ma
                },
                "market_data": {
                    "price": market_data.get("price", 0),
                    "volume": market_data.get("volume", 0)
                },
                "timestamp": self.timestamp
            }
            
            return entry_decision, confidence_score, details
            
        except Exception as e:
            logger.error(f"Error checking entry conditions for {coin}: {str(e)}")
            return False, 0.0, {"error": str(e)}
    
    def calculate_position_size(self, coin: str, available_balance: float) -> float:
        """
        Calculate position size based on risk parameters
        
        Args:
            coin (str): Coin to trade
            available_balance (float): Available balance
            
        Returns:
            float: Position size in USDT
        """
        # Default to maximum 10% of available balance per trade
        max_position = available_balance * TRADING_PARAMS["risk_per_trade"]
        
        # Adjust position size based on market conditions
        if coin in self.technical_indicators and coin in self.market_data:
            indicators = self.technical_indicators[coin]
            market_data = self.market_data[coin]
            
            # Adjust for volatility (reduce size for more volatile coins)
            volatility = indicators.get("volatility", 0.05)
            
            # Reduce position for higher volatility
            if volatility > 0.1:  # More than 10% daily range
                max_position *= (1.0 - (volatility - 0.1))
            
            # Adjust for sentiment (increase size for negative sentiment)
            if coin in self.sentiment_data:
                sentiment_score = self.sentiment_data[coin].get("sentiment_score", 0)
                if sentiment_score < -0.3:
                    # More negative sentiment, increase position
                    max_position *= (1.0 + (abs(sentiment_score) * 0.5))
            
            # Adjust based on RSI (more overbought = slightly larger position)
            rsi = indicators.get("rsi", 50)
            if rsi > 80:  # Extremely overbought
                max_position *= 1.1
        
        # Ensure minimum position size (to avoid dust)
        min_position = 10.0  # Minimum $10 trade
        
        return max(min_position, min(max_position, available_balance * 0.95))
    
    def calculate_entry_exit_levels(self, coin: str) -> Dict[str, float]:
        """
        Calculate entry, stop-loss, and take-profit levels for a short position
        
        Args:
            coin (str): Coin to trade
            
        Returns:
            Dict[str, float]: Entry, stop-loss, and take-profit levels
        """
        # Default levels
        levels = {
            "entry_price": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0
        }
        
        if coin not in self.market_data:
            return levels
        
        # Get current price
        current_price = self.market_data[coin].get("price", 0)
        
        if current_price <= 0:
            return levels
        
        # Set entry price at current price (market order)
        entry_price = current_price
        
        # Calculate stop loss level (default 5% above entry for shorts)
        stop_loss = entry_price * (1 + TRADING_PARAMS["stop_loss_percentage"])
        
        # Calculate take profit level (default 15% below entry for shorts)
        take_profit = entry_price * (1 - TRADING_PARAMS["take_profit_percentage"])
        
        # Adjust based on technical indicators (if available)
        if coin in self.technical_indicators:
            indicators = self.technical_indicators[coin]
            
            # Adjust stop loss based on volatility
            volatility = indicators.get("volatility", 0.05)
            if volatility > 0.15:  # Highly volatile
                # Wider stop-loss for volatile coins
                stop_loss = entry_price * (1 + (TRADING_PARAMS["stop_loss_percentage"] * 1.2))
            
            # Adjust take profit based on RSI
            rsi = indicators.get("rsi", 50)
            if rsi > 80:
                # More aggressive take-profit for extremely overbought coins
                take_profit = entry_price * (1 - (TRADING_PARAMS["take_profit_percentage"] * 1.2))
        
        levels["entry_price"] = entry_price
        levels["stop_loss"] = stop_loss
        levels["take_profit"] = take_profit
        
        return levels
    
    def execute_trade(self, coin: str, levels: Dict[str, float], 
                      allocation: float, details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a short trade for a coin
        
        Args:
            coin (str): Coin to trade
            levels (Dict[str, float]): Entry, stop-loss, and take-profit levels
            allocation (float): Amount to allocate to this trade
            details (Dict[str, Any]): Trade details
            
        Returns:
            Dict[str, Any]: Trade result
        """
        # Generate trade ID
        trade_id = f"SHORT_{int(time.time())}_{coin}_{uuid.uuid4().hex[:8]}"
        
        # Create trade record
        trade = {
            "trade_id": trade_id,
            "coin": coin,
            "symbol": self.market_data[coin]["symbol"] if coin in self.market_data else f"{coin}USDT",
            "strategy": "short_strategy",
            "operation": "SHORT",
            "entry_price": levels["entry_price"],
            "stop_loss": levels["stop_loss"],
            "take_profit": levels["take_profit"],
            "investment_usd": allocation,
            "quantity": allocation / levels["entry_price"] if levels["entry_price"] > 0 else 0,
            "leverage": TRADING_PARAMS["leverage"],
            "status": "PENDING",
            "entry_time": self.timestamp,
            "update_time": self.timestamp,
            "exit_time": None,
            "exit_price": None,
            "pnl": 0.0,
            "roi": 0.0,
            "details": details,
            "real_mode": self.real_mode
        }
        
        # In real mode, allocate funds via capital manager
        if not self.simulation_mode and CAPITAL_MANAGER_AVAILABLE:
            try:
                allocation_result = capital_manager.allocate_funds(coin, allocation, trade_id)
                
                if not allocation_result.get("success", False):
                    logger.error(f"Failed to allocate funds for {coin}: {allocation_result.get('error', 'Unknown error')}")
                    
                    if ALERT_SYSTEM_AVAILABLE:
                        alert_system.error(
                            f"Fund allocation failed for {coin} short trade",
                            {
                                "coin": coin,
                                "allocation": allocation,
                                "error": allocation_result.get("error", "Unknown error")
                            },
                            "short_strategy"
                        )
                    
                    trade["status"] = "FAILED"
                    trade["details"]["failure_reason"] = "Fund allocation failed"
                    return trade
                
                logger.info(f"Allocated ${allocation:.2f} for {coin} short trade")
                
            except Exception as e:
                logger.error(f"Error allocating funds: {str(e)}")
                
                if ALERT_SYSTEM_AVAILABLE:
                    alert_system.error(
                        f"Fund allocation error for {coin} short trade",
                        {
                            "coin": coin,
                            "allocation": allocation,
                            "error": str(e)
                        },
                        "short_strategy"
                    )
                
                trade["status"] = "FAILED"
                trade["details"]["failure_reason"] = f"Fund allocation error: {str(e)}"
                return trade
        
        # In simulation mode or if capital manager not available, mock allocation
        else:
            logger.info(f"[SIMULATION] Allocated ${allocation:.2f} for {coin} short trade")
        
        # At this point, funds are allocated, transition to OPEN status
        trade["status"] = "OPEN"
        
        # Add to executed trades log
        self._add_trade_to_log(trade)
        
        # Store in current trades
        self.current_trades[trade_id] = trade
        
        # Log trade
        logger.info(f"Executed SHORT trade for {coin} at ${levels['entry_price']:.6f}")
        logger.info(f"Stop-loss: ${levels['stop_loss']:.6f}, Take-profit: ${levels['take_profit']:.6f}")
        
        # Send notification via alert system
        if ALERT_SYSTEM_AVAILABLE:
            alert_system.info(
                f"New SHORT position opened for {coin}",
                {
                    "trade_id": trade_id,
                    "coin": coin,
                    "entry_price": levels['entry_price'],
                    "stop_loss": levels['stop_loss'],
                    "take_profit": levels['take_profit'],
                    "allocation": allocation,
                    "leverage": TRADING_PARAMS["leverage"]
                },
                "short_strategy"
            )
        
        # Store trade in memory core
        if MEMORY_CORE_AVAILABLE:
            memory_core.add_memory_record(
                source="short_strategy",
                category="trade_execution",
                data=trade,
                tags=["trade", "short", coin, "entry"]
            )
        
        # Log transaction
        if TRANSACTION_LOGGER_AVAILABLE:
            transaction_logger.log_transaction({
                "type": "TRADE_OPEN",
                "coin": coin,
                "symbol": trade["symbol"],
                "amount": allocation,
                "price": levels["entry_price"],
                "source_module": "short_strategy",
                "trade_id": trade_id,
                "operation": "SHORT"
            })
        
        return trade
    
    def close_trade(self, trade: Dict[str, Any], exit_price: float, exit_reason: str) -> Dict[str, Any]:
        """
        Close a short trade
        
        Args:
            trade (Dict[str, Any]): Trade to close
            exit_price (float): Exit price
            exit_reason (str): Reason for exit
            
        Returns:
            Dict[str, Any]: Updated trade
        """
        try:
            # Extract trade details
            trade_id = trade["trade_id"]
            coin = trade["coin"]
            entry_price = trade["entry_price"]
            investment = trade["investment_usd"]
            quantity = trade["quantity"]
            
            # Calculate profit/loss (for shorts, profit when exit_price < entry_price)
            pnl = quantity * (entry_price - exit_price)
            roi = (pnl / investment) * 100 if investment > 0 else 0
            
            # Update trade
            trade["status"] = "CLOSED"
            trade["exit_time"] = self.timestamp
            trade["exit_price"] = exit_price
            trade["pnl"] = pnl
            trade["roi"] = roi
            trade["details"]["exit_reason"] = exit_reason
            trade["update_time"] = self.timestamp
            
            # In real mode, release funds via capital manager
            if not self.simulation_mode and CAPITAL_MANAGER_AVAILABLE:
                try:
                    release_result = capital_manager.release_funds(coin, investment, pnl, trade_id)
                    
                    if not release_result.get("success", False):
                        logger.error(f"Failed to release funds for {coin}: {release_result.get('error', 'Unknown error')}")
                        
                        if ALERT_SYSTEM_AVAILABLE:
                            alert_system.error(
                                f"Fund release failed for {coin} short trade",
                                {
                                    "coin": coin,
                                    "investment": investment,
                                    "pnl": pnl,
                                    "error": release_result.get("error", "Unknown error")
                                },
                                "short_strategy"
                            )
                        
                        trade["details"]["release_failure"] = "Fund release failed"
                        
                    else:
                        logger.info(f"Released ${investment:.2f} with PnL ${pnl:.2f} for {coin}")
                    
                except Exception as e:
                    logger.error(f"Error releasing funds: {str(e)}")
                    
                    if ALERT_SYSTEM_AVAILABLE:
                        alert_system.error(
                            f"Fund release error for {coin} short trade",
                            {
                                "coin": coin,
                                "investment": investment,
                                "pnl": pnl,
                                "error": str(e)
                            },
                            "short_strategy"
                        )
                    
                    trade["details"]["release_error"] = f"Fund release error: {str(e)}"
            
            # In simulation mode or if capital manager not available, mock release
            else:
                logger.info(f"[SIMULATION] Released ${investment:.2f} with PnL ${pnl:.2f} for {coin}")
            
            # Update log
            self._update_trade_in_log(trade)
            
            # Remove from current trades
            if trade_id in self.current_trades:
                del self.current_trades[trade_id]
            
            # Log trade
            log_color = COLORS["green"] if pnl > 0 else COLORS["red"]
            logger.info(f"Closed SHORT trade for {coin} at ${exit_price:.6f}")
            logger.info(f"PnL: {log_color}${pnl:.2f} ({roi:.2f}%){COLORS['reset']}")
            
            # Send notification via alert system
            if ALERT_SYSTEM_AVAILABLE:
                alert_level = "info" if pnl >= 0 else "warning"
                alert_func = alert_system.info if pnl >= 0 else alert_system.warning
                
                alert_func(
                    f"SHORT position closed for {coin} with {pnl:.2f} USD {pnl >= 0 and 'profit' or 'loss'}",
                    {
                        "trade_id": trade_id,
                        "coin": coin,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl": pnl,
                        "roi": roi,
                        "exit_reason": exit_reason
                    },
                    "short_strategy"
                )
            
            # Store trade in memory core
            if MEMORY_CORE_AVAILABLE:
                memory_core.add_memory_record(
                    source="short_strategy",
                    category="trade_execution",
                    data=trade,
                    tags=["trade", "short", coin, "exit", exit_reason]
                )
            
            # Update performance tracker
            if PERFORMANCE_TRACKER_AVAILABLE:
                try:
                    # This is a simplified call - the actual interface may vary
                    performance_tracker.record_trade(trade)
                except Exception as pe:
                    logger.warning(f"Error recording trade in performance tracker: {pe}")
            
            # Record in learning engine
            if LEARNING_ENGINE_AVAILABLE:
                try:
                    # This is a simplified call - the actual interface may vary
                    learning_engine.process_trade_result(trade)
                except Exception as le:
                    logger.warning(f"Error processing trade in learning engine: {le}")
            
            # Log transaction
            if TRANSACTION_LOGGER_AVAILABLE:
                transaction_logger.log_transaction({
                    "type": "TRADE_CLOSE",
                    "coin": coin,
                    "symbol": trade["symbol"],
                    "amount": investment,
                    "price": exit_price,
                    "source_module": "short_strategy",
                    "trade_id": trade_id,
                    "operation": "SHORT",
                    "pnl": pnl,
                    "roi": roi,
                    "exit_reason": exit_reason
                })
            
            return trade
            
        except Exception as e:
            logger.error(f"Error closing trade: {str(e)}")
            
            if ALERT_SYSTEM_AVAILABLE:
                alert_system.error(
                    f"Error closing trade for {trade.get('coin', 'unknown')}",
                    {
                        "trade_id": trade.get("trade_id", "unknown"),
                        "error": str(e)
                    },
                    "short_strategy"
                )
            
            # Force close the trade with an error
            trade["status"] = "ERROR"
            trade["details"]["close_error"] = str(e)
            trade["exit_time"] = self.timestamp
            trade["update_time"] = self.timestamp
            
            # Update log
            self._update_trade_in_log(trade)
            
            return trade
    
    def _add_trade_to_log(self, trade: Dict[str, Any]) -> None:
        """
        Add a trade to the executed trades log
        
        Args:
            trade (Dict[str, Any]): Trade to add
        """
        try:
            # Load existing log
            trades_log = []
            
            if os.path.exists(EXECUTED_TRADES_LOG):
                try:
                    with open(EXECUTED_TRADES_LOG, "r", encoding="utf-8") as f:
                        log_data = json.load(f)
                        trades_log = log_data
                except (json.JSONDecodeError, FileNotFoundError):
                    logger.warning(f"Could not load {EXECUTED_TRADES_LOG}, creating new log")
                    trades_log = []
            
            # Add new trade
            trades_log.append(trade)
            
            # Save updated log
            with open(EXECUTED_TRADES_LOG, "w", encoding="utf-8") as f:
                json.dump(trades_log, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"Error adding trade to log: {str(e)}")
    
    def _update_trade_in_log(self, trade: Dict[str, Any]) -> None:
        """
        Update a trade in the executed trades log
        
        Args:
            trade (Dict[str, Any]): Trade to update
        """
        try:
            # Load existing log
            trades_log = []
            
            if os.path.exists(EXECUTED_TRADES_LOG):
                try:
                    with open(EXECUTED_TRADES_LOG, "r", encoding="utf-8") as f:
                        log_data = json.load(f)
                        trades_log = log_data
                except (json.JSONDecodeError, FileNotFoundError):
                    logger.warning(f"Could not load {EXECUTED_TRADES_LOG}, creating new log")
                    trades_log = []
            
            # Find and update trade
            trade_id = trade["trade_id"]
            updated = False
            
            for i, existing_trade in enumerate(trades_log):
                if existing_trade.get("trade_id") == trade_id:
                    trades_log[i] = trade
                    updated = True
                    break
            
            # If not found, add as new
            if not updated:
                trades_log.append(trade)
            
            # Save updated log
            with open(EXECUTED_TRADES_LOG, "w", encoding="utf-8") as f:
                json.dump(trades_log, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"Error updating trade in log: {str(e)}")
    
    def update_open_trades(self) -> None:
        """Update open trades with current market prices"""
        # Get current market data if it's stale
        if time.time() - self.last_market_check > 300:  # 5 minutes
            self.load_market_data()
        
        # Process each open trade
        for trade_id, trade in list(self.current_trades.items()):
            try:
                coin = trade["coin"]
                
                # Skip trades that aren't open
                if trade["status"] != "OPEN":
                    continue
                
                # Get current price
                if coin not in self.market_data:
                    logger.warning(f"No market data for {coin}, skipping trade update")
                    continue
                
                current_price = self.market_data[coin]["price"]
                
                # Check if stop loss or take profit hit
                entry_price = trade["entry_price"]
                stop_loss = trade["stop_loss"]
                take_profit = trade["take_profit"]
                
                # Update PnL (unrealized)
                quantity = trade["quantity"]
                # For shorts, profit when price goes down
                unrealized_pnl = quantity * (entry_price - current_price)
                unrealized_roi = (unrealized_pnl / trade["investment_usd"]) * 100 if trade["investment_usd"] > 0 else 0
                
                # Update trade with current metrics
                trade["current_price"] = current_price
                trade["unrealized_pnl"] = unrealized_pnl
                trade["unrealized_roi"] = unrealized_roi
                trade["update_time"] = self.timestamp
                
                # Check stop loss (for shorts, stop loss is when price goes up too much)
                if current_price >= stop_loss:
                    self.close_trade(trade, current_price, "stop_loss")
                    continue
                
                # Check take profit (for shorts, take profit is when price goes down enough)
                if current_price <= take_profit:
                    self.close_trade(trade, current_price, "take_profit")
                    continue
                
            except Exception as e:
                logger.error(f"Error updating trade {trade_id}: {str(e)}")
    
    def execute_strategy(self) -> bool:
        """
        Execute the short strategy for selected coins
        
        Returns:
            bool: Success status
        """
        try:
            # Check if a valid strategy decision is loaded
            if not self.strategy_decision or "strategy_decision" not in self.strategy_decision:
                logger.error("No valid strategy decision loaded")
                return False
            
            # Get target coins from strategy decision
            strategy_data = self.strategy_decision["strategy_decision"]["strategy"]
            target_coins = strategy_data.get("target_coins", [])
            
            if not target_coins:
                logger.warning("No target coins in strategy decision")
                return False
            
            logger.info(f"Executing short strategy for {len(target_coins)} coins")
            
            # Update any existing open trades
            self.update_open_trades()
            
            # Get available balance
            if CAPITAL_MANAGER_AVAILABLE:
                available_balance = capital_manager.get_available_balance()
            else:
                # Mock available balance if capital manager not available
                available_balance = 10000.0  # Default $10,000 for simulation
            
            logger.info(f"Available balance: ${available_balance:.2f}")
            
            # Calculate maximum allocation for new trades
            # Ensure we don't use more than the risk parameter allows
            max_total_allocation = available_balance * TRADING_PARAMS["risk_per_trade"] * 3  # Allow up to 3 trades
            
            # Keep track of allocated amount
            total_allocated = 0.0
            executed_trades = []
            
            # Process each target coin
            for coin in target_coins:
                # Skip if we've allocated too much already
                if total_allocated >= max_total_allocation:
                    logger.info(f"Reached maximum allocation (${total_allocated:.2f}), skipping remaining coins")
                    break
                
                # Normalize coin symbol
                coin = coin.upper()
                
                # Check entry conditions
                entry_decision, confidence, details = self.check_entry_conditions(coin)
                
                # If entry conditions met, execute trade
                if entry_decision:
                    # Calculate position size
                    remaining_allocation = max_total_allocation - total_allocated
                    allocation = self.calculate_position_size(coin, remaining_allocation)
                    
                    # Calculate entry, stop-loss, and take-profit levels
                    levels = self.calculate_entry_exit_levels(coin)
                    
                    # Skip if levels couldn't be determined
                    if levels["entry_price"] <= 0:
                        logger.warning(f"Could not determine entry levels for {coin}, skipping")
                        continue
                    
                    # Execute trade
                    trade = self.execute_trade(coin, levels, allocation, details)
                    
                    # Track allocation
                    if trade["status"] == "OPEN":
                        total_allocated += allocation
                        executed_trades.append(trade)
                    
                    # Add confidence to trade details
                    trade["details"]["confidence"] = confidence
                
                else:
                    logger.info(f"Entry conditions not met for {coin} (confidence: {confidence:.2f})")
            
            # Log summary
            if executed_trades:
                logger.info(f"Executed {len(executed_trades)} short trades, allocated ${total_allocated:.2f}")
                
                # Terminal summary
                print(f"\n{COLORS['bright']}{COLORS['red']}SHORT Strategy Execution Summary{COLORS['reset']}")
                print(f"{COLORS['red']}{'=' * 50}{COLORS['reset']}")
                print(f"Executed {len(executed_trades)} new trades")
                print(f"Total allocated: ${total_allocated:.2f}")
                
                for trade in executed_trades:
                    coin = trade["coin"]
                    entry = trade["entry_price"]
                    sl = trade["stop_loss"]
                    tp = trade["take_profit"]
                    allocation = trade["investment_usd"]
                    
                    sl_pct = ((sl / entry) - 1) * 100
                    tp_pct = (1 - (tp / entry)) * 100
                    
                    print(f"\n{COLORS['bright']}{coin} SHORT @ ${entry:.6f}{COLORS['reset']}")
                    print(f"Stop-Loss: ${sl:.6f} (+{sl_pct:.2f}%)")
                    print(f"Take-Profit: ${tp:.6f} ({tp_pct:.2f}%)")
                    print(f"Allocation: ${allocation:.2f}")
                
                print(f"{COLORS['red']}{'=' * 50}{COLORS['reset']}")
            else:
                logger.info("No short trades executed")
                print(f"\n{COLORS['yellow']}No short trades executed. Entry conditions not met for any target coins.{COLORS['reset']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing strategy: {str(e)}")
            
            if ALERT_SYSTEM_AVAILABLE:
                alert_system.error(
                    "Error executing short strategy",
                    {"error": str(e)},
                    "short_strategy"
                )
            
            return False
    
    def run(self) -> bool:
        """
        Run the complete short strategy pipeline
        
        Returns:
            bool: Success status
        """
        try:
            # Load strategy decision
            if not self.load_strategy_decision():
                if not self.simulation_mode:
                    logger.info("Strategy decision not applicable to short strategy, exiting")
                    return False
                else:
                    logger.info("Simulation mode, proceeding despite missing strategy decision")
            
            # Load market data
            if not self.load_market_data():
                logger.error("Failed to load market data, can't proceed")
                return False
            
            # Load sentiment data if available
            self._load_sentiment_data()
            
            # Execute strategy
            if not self.execute_strategy():
                logger.error("Failed to execute strategy")
                return False
            
            logger.info("Strategy execution completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error running strategy: {str(e)}")
            
            if ALERT_SYSTEM_AVAILABLE:
                alert_system.error(
                    "Error running short strategy",
                    {"error": str(e)},
                    "short_strategy"
                )
            
            return False
    
    def simulate_trades(self, coins: List[str]) -> bool:
        """
        Simulate trades for a list of coins (for testing purposes)
        
        Args:
            coins (List[str]): List of coins to simulate trades for
            
        Returns:
            bool: Success status
        """
        try:
            # Force simulation mode for test
            old_mode = self.simulation_mode
            self.simulation_mode = True
            
            # Create strategy decision for test
            self.strategy_decision = {
                "strategy_decision": {
                    "strategy": {
                        "name": "short_strategy.py",
                        "target_coins": coins
                    }
                }
            }
            
            # Load market data
            if not self.load_market_data():
                self.simulation_mode = old_mode
                logger.error("Failed to load market data for simulation")
                return False
            
            # Load sentiment data if available
            self._load_sentiment_data()
            
            # Execute strategy
            result = self.execute_strategy()
            
            # Restore original mode
            self.simulation_mode = old_mode
            
            return result
            
        except Exception as e:
            logger.error(f"Error simulating trades: {str(e)}")
            return False

# Simple function to get available balance for testing
def get_available_balance() -> float:
    """Get available balance for trading"""
    if CAPITAL_MANAGER_AVAILABLE:
        return capital_manager.get_available_balance()
    else:
        # Mock balance for testing
        return 10000.0

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="SentientTrader.AI Short Strategy")
    parser.add_argument("--real", action="store_true", help="Use real trading mode")
    parser.add_argument("--simulation", action="store_true", help="Run in simulation mode")
    parser.add_argument("--coin", type=str, help="Target coin for the trade")
    parser.add_argument("--allocation", type=float, help="Allocation amount in USD")
    return parser.parse_args()

def main() -> int:
    """Main function for direct execution"""
    args = parse_arguments()
    
    # Set modes based on args
    real_mode = args.real
    simulation_mode = args.simulation or not args.real
    
    try:
        # Initialize strategy
        strategy = ShortStrategy(real_mode=real_mode, simulation_mode=simulation_mode)
        
        # If specific coin provided, run for that coin only
        if args.coin:
            coin = args.coin.upper()
            allocation = args.allocation or 100.0
            
            print(f"{COLORS['cyan']}Running short strategy for {coin} with ${allocation:.2f} allocation{COLORS['reset']}")
            
            # Load market data
            if not strategy.load_market_data():
                print(f"{COLORS['red']}Failed to load market data{COLORS['reset']}")
                return 1
            
            # Load sentiment data
            strategy._load_sentiment_data()
            
            # Check entry conditions
            entry_decision, confidence, details = strategy.check_entry_conditions(coin)
            
            if entry_decision:
                print(f"{COLORS['green']}Entry conditions met for {coin} (confidence: {confidence:.2f}){COLORS['reset']}")
                
                # Calculate levels
                levels = strategy.calculate_entry_exit_levels(coin)
                
                # Execute trade
                trade = strategy.execute_trade(coin, levels, allocation, details)
                
                if trade["status"] == "OPEN":
                    print(f"{COLORS['green']}Successfully opened SHORT position for {coin}{COLORS['reset']}")
                else:
                    print(f"{COLORS['red']}Failed to open position: {trade.get('details', {}).get('failure_reason', 'Unknown error')}{COLORS['reset']}")
            else:
                print(f"{COLORS['yellow']}Entry conditions not met for {coin} (confidence: {confidence:.2f}){COLORS['reset']}")
                
                # Show details
                print("\nCondition details:")
                for condition, met in details.get("conditions", {}).items():
                    color = COLORS["green"] if met else COLORS["red"]
                    print(f"  {condition}: {color}{met}{COLORS['reset']}")
                
                # Show indicators
                print("\nIndicator values:")
                for indicator, value in details.get("indicators", {}).items():
                    print(f"  {indicator}: {value}")
            
            return 0
            
        # Otherwise run full strategy process
        else:
            # If simulation mode, run with sample coins
            if simulation_mode:
                print(f"{COLORS['cyan']}Running short strategy simulation{COLORS['reset']}")
                sample_coins = ["BTC", "ETH", "SOL", "LINK", "ADA"]
                strategy.simulate_trades(sample_coins[:3])  # Use first 3 coins
                
            # Otherwise run normal process
            else:
                print(f"{COLORS['cyan']}Running short strategy{COLORS['reset']}")
                strategy.run()
            
            return 0
        
    except KeyboardInterrupt:
        print(f"\n{COLORS['yellow']}Process interrupted by user{COLORS['reset']}")
        return 130
    except Exception as e:
        print(f"{COLORS['red']}Unexpected error: {str(e)}{COLORS['reset']}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
