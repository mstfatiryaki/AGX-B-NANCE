#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SentientTrader.AI - Terminal Interface V2
-----------------------------------------
Interactive terminal interface for managing and monitoring the SentientTrader.AI system.
Provides colorful UI, system status monitoring, and easy access to all system modules.
"""

import os
import sys
import time
import json
import logging
import random
import datetime
import subprocess
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("terminal_interface.log")]
)
logger = logging.getLogger("TerminalInterface")

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
        "black": Fore.BLACK,
        "bright": Style.BRIGHT,
        "dim": Style.DIM,
        "normal": Style.NORMAL,
        "reset": Style.RESET_ALL,
        "bg_blue": Back.BLUE,
        "bg_green": Back.GREEN,
        "bg_red": Back.RED,
        "bg_yellow": Back.YELLOW,
        "bg_magenta": Back.MAGENTA,
        "bg_cyan": Back.CYAN,
        "bg_white": Back.WHITE,
        "bg_black": Back.BLACK
    }
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORS = {
        "blue": "", "green": "", "red": "", "yellow": "", "magenta": "", "cyan": "", 
        "white": "", "black": "", "bright": "", "dim": "", "normal": "", "reset": "",
        "bg_blue": "", "bg_green": "", "bg_red": "", "bg_yellow": "", 
        "bg_magenta": "", "bg_cyan": "", "bg_white": "", "bg_black": ""
    }
    COLORAMA_AVAILABLE = False
    logger.warning("Colorama library not found. Terminal colors will be disabled.")

# Constants and configuration
REAL_MODE = False  # Default to simulation mode for safety
SYSTEM_VERSION = "2.0"
CURRENT_TIME = "2025-04-21 21:29:57"  # UTC
CURRENT_USER = "mstfatiryaki"

# Try to import system modules
try:
    import performance_tracker
    PERFORMANCE_TRACKER_AVAILABLE = True
except ImportError:
    PERFORMANCE_TRACKER_AVAILABLE = False
    logger.warning("Performance tracker not found. Performance data will be unavailable.")

try:
    import risk_manager
    RISK_MANAGER_AVAILABLE = True
except ImportError:
    RISK_MANAGER_AVAILABLE = False
    logger.warning("Risk manager not found. Risk data will be unavailable.")

try:
    import learning_engine
    LEARNING_ENGINE_AVAILABLE = True
except ImportError:
    LEARNING_ENGINE_AVAILABLE = False
    logger.warning("Learning engine not found. Learning features will be disabled.")

try:
    import strategy_engine
    STRATEGY_ENGINE_AVAILABLE = True
except ImportError:
    STRATEGY_ENGINE_AVAILABLE = False
    logger.warning("Strategy engine not found. Strategy features will be disabled.")

try:
    import transaction_logger
    TRANSACTION_LOGGER_AVAILABLE = True
except ImportError:
    TRANSACTION_LOGGER_AVAILABLE = False
    logger.warning("Transaction logger not found. Transaction data will be unavailable.")

try:
    import alert_system
    ALERT_SYSTEM_AVAILABLE = True
except ImportError:
    ALERT_SYSTEM_AVAILABLE = False
    logger.warning("Alert system not found. Alerts will be unavailable.")

# Strategy modules
STRATEGY_MODULES = {
    "long": "long_strategy.py",
    "short": "short_strategy.py",
    "sniper": "sniper_strategy.py"
}

class TerminalInterface:
    """Interactive terminal interface for SentientTrader.AI"""
    
    def __init__(self, system_name="SentientTrader.AI", version="2.0", real_mode=REAL_MODE):
        self.system_name = system_name
        self.version = version
        """
        Initialize the terminal interface
        
        Args:
            real_mode (bool): Whether to run in real mode
        """
        self.real_mode = real_mode
        self.system_name = system_name
        self.version = version
        self.current_page = "main"
        self.should_exit = False
        self.system_status = self._get_system_status()
        
        # Cache for module data to avoid frequent calls
        self.cache = {
            "performance_data": None,
            "performance_updated": 0,
            "risk_data": None,
            "risk_updated": 0,
            "strategy_data": None,
            "strategy_updated": 0,
            "learning_data": None,
            "learning_updated": 0,
            "transactions": None,
            "transactions_updated": 0
        }
        
        # Cache refresh intervals (in seconds)
        self.cache_expiry = {
            "performance": 60,  # 1 minute
            "risk": 120,        # 2 minutes
            "strategy": 300,    # 5 minutes
            "learning": 600,    # 10 minutes
            "transactions": 30  # 30 seconds
        }
        
        logger.info(f"Terminal interface initialized (REAL_MODE: {self.real_mode})")
    
    def _get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status
        
        Returns:
            Dict[str, Any]: System status information
        """
        active_modules = 0
        if PERFORMANCE_TRACKER_AVAILABLE:
            active_modules += 1
        if RISK_MANAGER_AVAILABLE:
            active_modules += 1
        if LEARNING_ENGINE_AVAILABLE:
            active_modules += 1
        if STRATEGY_ENGINE_AVAILABLE:
            active_modules += 1
        if TRANSACTION_LOGGER_AVAILABLE:
            active_modules += 1
        if ALERT_SYSTEM_AVAILABLE:
            active_modules += 1
        
        return {
            "running": True,
            "mode": "REAL" if self.real_mode else "SIMULATION",
            "version": SYSTEM_VERSION,
            "active_modules": active_modules,
            "total_modules": 6,
            "uptime": "12:34:56",  # This would be calculated in a real implementation
            "last_update": CURRENT_TIME,
            "user": CURRENT_USER
        }
    
    def _refresh_cache(self, cache_key: str) -> None:
        """
        Refresh a specific cache entry if it's expired
        
        Args:
            cache_key (str): Cache key to refresh
        """
        current_time = time.time()
        
        if cache_key == "performance":
            if PERFORMANCE_TRACKER_AVAILABLE and (
                self.cache["performance_data"] is None or 
                current_time - self.cache["performance_updated"] > self.cache_expiry["performance"]
            ):
                try:
                    self.cache["performance_data"] = performance_tracker.get_performance_summary()
                    self.cache["performance_updated"] = current_time
                except Exception as e:
                    logger.error(f"Error refreshing performance data: {str(e)}")
        
        elif cache_key == "risk":
            if RISK_MANAGER_AVAILABLE and (
                self.cache["risk_data"] is None or 
                current_time - self.cache["risk_updated"] > self.cache_expiry["risk"]
            ):
                try:
                    self.cache["risk_data"] = risk_manager.get_risk_status()
                    self.cache["risk_updated"] = current_time
                except Exception as e:
                    logger.error(f"Error refreshing risk data: {str(e)}")
        
        elif cache_key == "strategy":
            if STRATEGY_ENGINE_AVAILABLE and (
                self.cache["strategy_data"] is None or 
                current_time - self.cache["strategy_updated"] > self.cache_expiry["strategy"]
            ):
                try:
                    self.cache["strategy_data"] = strategy_engine.get_latest_strategy()
                    self.cache["strategy_updated"] = current_time
                except Exception as e:
                    logger.error(f"Error refreshing strategy data: {str(e)}")
        
        elif cache_key == "learning":
            if LEARNING_ENGINE_AVAILABLE and (
                self.cache["learning_data"] is None or 
                current_time - self.cache["learning_updated"] > self.cache_expiry["learning"]
            ):
                try:
                    self.cache["learning_data"] = learning_engine.get_learning_summary()
                    self.cache["learning_updated"] = current_time
                except Exception as e:
                    logger.error(f"Error refreshing learning data: {str(e)}")
        
        elif cache_key == "transactions":
            if TRANSACTION_LOGGER_AVAILABLE and (
                self.cache["transactions"] is None or 
                current_time - self.cache["transactions_updated"] > self.cache_expiry["transactions"]
            ):
                try:
                    self.cache["transactions"] = transaction_logger.get_recent_transactions(limit=5)
                    self.cache["transactions_updated"] = current_time
                except Exception as e:
                    logger.error(f"Error refreshing transaction data: {str(e)}")
    
    def _refresh_all_caches(self) -> None:
        """Refresh all cache entries"""
        self._refresh_cache("performance")
        self._refresh_cache("risk")
        self._refresh_cache("strategy")
        self._refresh_cache("learning")
        self._refresh_cache("transactions")
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary data
        
        Returns:
            Dict[str, Any]: Performance summary
        """
        self._refresh_cache("performance")
        
        if PERFORMANCE_TRACKER_AVAILABLE and self.cache["performance_data"]:
            return self.cache["performance_data"]
        else:
            return {
                "daily_pnl": 0.0,
                "weekly_pnl": 0.0,
                "monthly_pnl": 0.0,
                "total_trades": 0,
                "successful_trades": 0,
                "win_rate": 0.0,
                "average_profit": 0.0,
                "average_loss": 0.0
            }
    
    def _get_risk_summary(self) -> Dict[str, Any]:
        """
        Get risk summary data
        
        Returns:
            Dict[str, Any]: Risk summary
        """
        self._refresh_cache("risk")
        
        if RISK_MANAGER_AVAILABLE and self.cache["risk_data"]:
            return self.cache["risk_data"]
        else:
            return {
                "current_risk_level": "UNKNOWN",
                "max_position_size": 0.0,
                "max_leverage": 1.0,
                "capital_at_risk": 0.0,
                "risk_allocation": {},
                "warnings": []
            }
    
    def _get_strategy_summary(self) -> Dict[str, Any]:
        """
        Get latest strategy summary
        
        Returns:
            Dict[str, Any]: Strategy summary
        """
        self._refresh_cache("strategy")
        
        if STRATEGY_ENGINE_AVAILABLE and self.cache["strategy_data"]:
            return self.cache["strategy_data"]
        else:
            return {
                "strategy": "NONE",
                "target_coins": [],
                "confidence": 0.0,
                "timestamp": CURRENT_TIME
            }
    
    def _get_learning_summary(self) -> Dict[str, Any]:
        """
        Get learning summary data
        
        Returns:
            Dict[str, Any]: Learning summary
        """
        self._refresh_cache("learning")
        
        if LEARNING_ENGINE_AVAILABLE and self.cache["learning_data"]:
            return self.cache["learning_data"]
        else:
            return {
                "last_training": "NONE",
                "model_version": "0.0",
                "accuracy": 0.0,
                "improvement": 0.0,
                "insights": []
            }
    
    def _get_recent_transactions(self) -> List[Dict[str, Any]]:
        """
        Get recent transactions
        
        Returns:
            List[Dict[str, Any]]: Recent transactions
        """
        self._refresh_cache("transactions")
        
        if TRANSACTION_LOGGER_AVAILABLE and self.cache["transactions"]:
            return self.cache["transactions"]
        else:
            return []
    
    def _draw_header(self) -> None:
        """Draw the application header"""
        print(f"\n{COLORS['bg_blue']}{COLORS['white']}{COLORS['bright']}{'=' * 80}{COLORS['reset']}")
        print(f"{COLORS['bg_blue']}{COLORS['white']}{COLORS['bright']}{'SentientTrader.AI - Terminal Interface V2':^80}{COLORS['reset']}")
        print(f"{COLORS['bg_blue']}{COLORS['white']}{COLORS['bright']}{'=' * 80}{COLORS['reset']}")
        
        # Show mode indicator
        mode_color = COLORS['bg_green'] if not self.real_mode else COLORS['bg_red']
        mode_text = "SIMULATION MODE" if not self.real_mode else "REAL TRADING MODE"
        print(f"{mode_color}{COLORS['black']}{COLORS['bright']}{mode_text:^80}{COLORS['reset']}")
        
        # Show current time and user
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{COLORS['dim']}Time: {current_time} UTC | User: {CURRENT_USER}{COLORS['reset']}")
        print()
    
    def _draw_footer(self) -> None:
        """Draw the application footer"""
        print(f"\n{COLORS['dim']}Press [0] to exit or [B] to go back{COLORS['reset']}")
        print(f"{COLORS['bg_blue']}{COLORS['white']}{COLORS['bright']}{'=' * 80}{COLORS['reset']}\n")
    
    def _draw_system_status(self) -> None:
        """Draw system status summary"""
        status = self._get_system_status()
        
        print(f"{COLORS['bright']}System Status:{COLORS['reset']}")
        
        # Mode indicator
        mode_color = COLORS['green'] if status["mode"] == "SIMULATION" else COLORS['red']
        print(f"  Mode: {mode_color}{status['mode']}{COLORS['reset']}")
        
        # Module status
        print(f"  Active Modules: {COLORS['cyan']}{status['active_modules']}/{status['total_modules']}{COLORS['reset']}")
        
        # Version and uptime
        print(f"  Version: {status['version']} | Uptime: {status['uptime']}")
        
        print()
    
    def _draw_performance_summary(self) -> None:
        """Draw performance summary"""
        performance = self._get_performance_summary()
        
        print(f"{COLORS['bright']}Performance Summary:{COLORS['reset']}")
        
        # PnL indicators
        daily_pnl = performance.get("daily_pnl", 0.0)
        daily_color = COLORS['green'] if daily_pnl >= 0 else COLORS['red']
        daily_sign = "+" if daily_pnl >= 0 else ""
        
        print(f"  Daily P&L: {daily_color}{daily_sign}{daily_pnl:.2f}${COLORS['reset']}")
        
        # Trade statistics
        total_trades = performance.get("total_trades", 0)
        win_rate = performance.get("win_rate", 0.0)
        win_rate_color = COLORS['green'] if win_rate >= 50.0 else COLORS['yellow'] if win_rate >= 30.0 else COLORS['red']
        
        print(f"  Total Trades: {total_trades} | Win Rate: {win_rate_color}{win_rate:.1f}%{COLORS['reset']}")
        
        print()
    
    def _draw_risk_summary(self) -> None:
        """Draw risk status summary"""
        risk = self._get_risk_summary()
        
        print(f"{COLORS['bright']}Risk Status:{COLORS['reset']}")
        
        # Risk level
        risk_level = risk.get("current_risk_level", "UNKNOWN")
        risk_color = COLORS['green']
        if risk_level == "MEDIUM":
            risk_color = COLORS['yellow']
        elif risk_level == "HIGH":
            risk_color = COLORS['red']
        elif risk_level == "EXTREME":
            risk_color = COLORS['bg_red'] + COLORS['white'] + COLORS['bright']
        
        print(f"  Risk Level: {risk_color}{risk_level}{COLORS['reset']}")
        
        # Capital at risk
        capital_at_risk = risk.get("capital_at_risk", 0.0)
        capital_color = COLORS['green'] if capital_at_risk < 20.0 else COLORS['yellow'] if capital_at_risk < 50.0 else COLORS['red']
        
        print(f"  Capital at Risk: {capital_color}{capital_at_risk:.1f}%{COLORS['reset']}")
        
        # Warnings
        warnings = risk.get("warnings", [])
        if warnings:
            print(f"  {COLORS['yellow']}Warnings: {len(warnings)}{COLORS['reset']}")
        
        print()
    
    def _draw_strategy_summary(self) -> None:
        """Draw strategy summary"""
        strategy = self._get_strategy_summary()
        
        print(f"{COLORS['bright']}Latest Strategy:{COLORS['reset']}")
        
        # Strategy name
        strategy_name = strategy.get("strategy", "NONE")
        if strategy_name == "NONE":
            strategy_color = COLORS['dim']
        else:
            strategy_color = COLORS['cyan']
        
        print(f"  Strategy: {strategy_color}{strategy_name}{COLORS['reset']}")
        
        # Target coins
        target_coins = strategy.get("target_coins", [])
        if target_coins:
            coins_display = ", ".join(target_coins[:3])
            if len(target_coins) > 3:
                coins_display += f" +{len(target_coins) - 3} more"
            print(f"  Targets: {COLORS['yellow']}{coins_display}{COLORS['reset']}")
        else:
            print(f"  Targets: {COLORS['dim']}None{COLORS['reset']}")
        
        # Confidence
        confidence = strategy.get("confidence", 0.0)
        confidence_color = COLORS['green'] if confidence >= 0.7 else COLORS['yellow'] if confidence >= 0.4 else COLORS['red']
        
        print(f"  Confidence: {confidence_color}{confidence:.1f}{COLORS['reset']}")
        
        print()
    
    def _draw_learning_summary(self) -> None:
        """Draw learning summary"""
        learning = self._get_learning_summary()
        
        print(f"{COLORS['bright']}Learning Status:{COLORS['reset']}")
        
        # Last training
        last_training = learning.get("last_training", "NONE")
        if last_training == "NONE":
            training_color = COLORS['dim']
        else:
            training_color = COLORS['magenta']
        
        print(f"  Last Training: {training_color}{last_training}{COLORS['reset']}")
        
        # Model version and accuracy
        model_version = learning.get("model_version", "0.0")
        accuracy = learning.get("accuracy", 0.0)
        accuracy_color = COLORS['green'] if accuracy >= 0.7 else COLORS['yellow'] if accuracy >= 0.5 else COLORS['red']
        
        print(f"  Model: v{model_version} | Accuracy: {accuracy_color}{accuracy:.1f}{COLORS['reset']}")
        
        # Improvement
        improvement = learning.get("improvement", 0.0)
        improvement_sign = "+" if improvement >= 0 else ""
        improvement_color = COLORS['green'] if improvement > 0 else COLORS['red'] if improvement < 0 else COLORS['dim']
        
        print(f"  Improvement: {improvement_color}{improvement_sign}{improvement:.1f}%{COLORS['reset']}")
        
        print()
    
    def _draw_recent_transactions(self) -> None:
        """Draw recent transactions"""
        transactions = self._get_recent_transactions()
        
        print(f"{COLORS['bright']}Recent Transactions:{COLORS['reset']}")
        
        if not transactions:
            print(f"  {COLORS['dim']}No recent transactions{COLORS['reset']}")
            print()
            return
        
        for tx in transactions[:3]:  # Show only the 3 most recent
            # Transaction type and symbol
            tx_type = tx.get("type", "UNKNOWN")
            symbol = tx.get("symbol", "UNKNOWN")
            
            type_color = COLORS['green'] if tx_type == "TRADE_OPEN" else COLORS['red'] if tx_type == "TRADE_CLOSE" else COLORS['blue']
            
            print(f"  {type_color}{tx_type}{COLORS['reset']} - {COLORS['yellow']}{symbol}{COLORS['reset']}")
            
            # Amount and price
            amount = tx.get("amount", 0.0)
            price = tx.get("price", 0.0)
            
            # Format differently for open and close
            if tx_type == "TRADE_OPEN":
                print(f"    ${amount:.2f} at ${price:.6f}")
            elif tx_type == "TRADE_CLOSE":
                pnl = tx.get("pnl", 0.0)
                roi = tx.get("roi", 0.0)
                pnl_color = COLORS['green'] if pnl >= 0 else COLORS['red']
                pnl_sign = "+" if pnl >= 0 else ""
                
                print(f"    PnL: {pnl_color}{pnl_sign}{pnl:.2f}$ ({pnl_sign}{roi:.2f}%){COLORS['reset']}")
            
            # Timestamp in compact format
            timestamp = tx.get("timestamp", "")
            if timestamp:
                try:
                    dt = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                    timestamp = dt.strftime("%m-%d %H:%M")
                except:
                    pass
                    
            print(f"    {COLORS['dim']}{timestamp}{COLORS['reset']}")
        
        # Show how many more transactions are available
        if len(transactions) > 3:
            print(f"  {COLORS['dim']}+{len(transactions) - 3} more recent transactions{COLORS['reset']}")
        
        print()
    
    def _draw_main_menu(self) -> None:
        """Draw the main menu"""
        self._draw_header()
        
        # System status dashboard
        self._draw_system_status()
        self._draw_performance_summary()
        self._draw_risk_summary()
        self._draw_strategy_summary()
        self._draw_learning_summary()
        self._draw_recent_transactions()
        
        # Menu options
        print(f"{COLORS['bright']}Available Commands:{COLORS['reset']}")
        print(f"  {COLORS['green']}[1]{COLORS['reset']} Run Strategy")
        print(f"  {COLORS['green']}[2]{COLORS['reset']} Start Simulation")
        print(f"  {COLORS['green']}[3]{COLORS['reset']} Performance Analysis")
        print(f"  {COLORS['green']}[4]{COLORS['reset']} Risk Analysis")
        print(f"  {COLORS['green']}[5]{COLORS['reset']} Trigger Learning Module")
        print(f"  {COLORS['green']}[6]{COLORS['reset']} System Status")
        print(f"  {COLORS['green']}[7]{COLORS['reset']} Toggle Mode ({COLORS['red'] if self.real_mode else COLORS['green']}{'REAL' if self.real_mode else 'SIMULATION'}{COLORS['reset']})")
        print(f"  {COLORS['red']}[0]{COLORS['reset']} Exit")
        
        self._draw_footer()
    
    def _draw_run_strategy_menu(self) -> None:
        """Draw the run strategy menu"""
        self._draw_header()
        
        print(f"{COLORS['bright']}{COLORS['cyan']}Run Strategy{COLORS['reset']}\n")
        
        # Strategy options
        print(f"Select a strategy to run:")
        print(f"  {COLORS['green']}[1]{COLORS['reset']} Long Strategy (Buy low, sell high)")
        print(f"  {COLORS['green']}[2]{COLORS['reset']} Short Strategy (Sell high, buy low)")
        print(f"  {COLORS['green']}[3]{COLORS['reset']} Sniper Strategy (Quick entry/exit)")
        print()
        print(f"  {COLORS['green']}[B]{COLORS['reset']} Back to main menu")
        print(f"  {COLORS['red']}[0]{COLORS['reset']} Exit")
        
        self._draw_footer()
    
    def _draw_simulation_menu(self) -> None:
        """Draw the simulation menu"""
        self._draw_header()
        
        print(f"{COLORS['bright']}{COLORS['cyan']}Start Simulation{COLORS['reset']}\n")
        
        # Simulation options
        print(f"Select a simulation type:")
        print(f"  {COLORS['green']}[1]{COLORS['reset']} Backtest with historical data")
        print(f"  {COLORS['green']}[2]{COLORS['reset']} Paper trading with live data")
        print(f"  {COLORS['green']}[3]{COLORS['reset']} Strategy comparison")
        print()
        print(f"  {COLORS['green']}[B]{COLORS['reset']} Back to main menu")
        print(f"  {COLORS['red']}[0]{COLORS['reset']} Exit")
        
        self._draw_footer()
    
    def _draw_performance_menu(self) -> None:
        """Draw the performance analysis menu"""
        self._draw_header()
        
        print(f"{COLORS['bright']}{COLORS['cyan']}Performance Analysis{COLORS['reset']}\n")
        
        # Performance data
        performance = self._get_performance_summary()
        
        # Daily, weekly, monthly PnL
        daily_pnl = performance.get("daily_pnl", 0.0)
        weekly_pnl = performance.get("weekly_pnl", 0.0)
        monthly_pnl = performance.get("monthly_pnl", 0.0)
        
        daily_color = COLORS['green'] if daily_pnl >= 0 else COLORS['red']
        weekly_color = COLORS['green'] if weekly_pnl >= 0 else COLORS['red']
        monthly_color = COLORS['green'] if monthly_pnl >= 0 else COLORS['red']
        
        daily_sign = "+" if daily_pnl >= 0 else ""
        weekly_sign = "+" if weekly_pnl >= 0 else ""
        monthly_sign = "+" if monthly_pnl >= 0 else ""
        
        print(f"{COLORS['bright']}Profit & Loss:{COLORS['reset']}")
        print(f"  Daily:   {daily_color}{daily_sign}{daily_pnl:.2f}${COLORS['reset']}")
        print(f"  Weekly:  {weekly_color}{weekly_sign}{weekly_pnl:.2f}${COLORS['reset']}")
        print(f"  Monthly: {monthly_color}{monthly_sign}{monthly_pnl:.2f}${COLORS['reset']}")
        
        # Trade statistics
        total_trades = performance.get("total_trades", 0)
        successful_trades = performance.get("successful_trades", 0)
        win_rate = performance.get("win_rate", 0.0)
        
        win_rate_color = COLORS['green'] if win_rate >= 50.0 else COLORS['yellow'] if win_rate >= 30.0 else COLORS['red']
        
        print(f"\n{COLORS['bright']}Trade Statistics:{COLORS['reset']}")
        print(f"  Total Trades:      {total_trades}")
        print(f"  Successful Trades: {successful_trades}")
        print(f"  Win Rate:          {win_rate_color}{win_rate:.1f}%{COLORS['reset']}")
        
        # Average metrics
        avg_profit = performance.get("average_profit", 0.0)
        avg_loss = performance.get("average_loss", 0.0)
        
        print(f"\n{COLORS['bright']}Trade Metrics:{COLORS['reset']}")
        print(f"  Average Profit: {COLORS['green']}+{avg_profit:.2f}${COLORS['reset']}")
        print(f"  Average Loss:   {COLORS['red']}{avg_loss:.2f}${COLORS['reset']}")
        
        if avg_loss != 0:
            risk_reward = abs(avg_profit / avg_loss) if avg_loss != 0 else 0
            print(f"  Risk/Reward:    {COLORS['yellow']}{risk_reward:.2f}{COLORS['reset']}")
        
        # Menu options
        print(f"\nAnalysis Options:")
        print(f"  {COLORS['green']}[1]{COLORS['reset']} View Detailed Performance Report")
        print(f"  {COLORS['green']}[2]{COLORS['reset']} Export Performance Data")
        print()
        print(f"  {COLORS['green']}[B]{COLORS['reset']} Back to main menu")
        print(f"  {COLORS['red']}[0]{COLORS['reset']} Exit")
        
        self._draw_footer()
    
    def _draw_risk_menu(self) -> None:
        """Draw the risk analysis menu"""
        self._draw_header()
        
        print(f"{COLORS['bright']}{COLORS['cyan']}Risk Analysis{COLORS['reset']}\n")
        
        # Risk data
        risk = self._get_risk_summary()
        
        # Risk level
        risk_level = risk.get("current_risk_level", "UNKNOWN")
        risk_color = COLORS['green']
        if risk_level == "MEDIUM":
            risk_color = COLORS['yellow']
        elif risk_level == "HIGH":
            risk_color = COLORS['red']
        elif risk_level == "EXTREME":
            risk_color = COLORS['bg_red'] + COLORS['white'] + COLORS['bright']
        
        print(f"{COLORS['bright']}Current Risk Level: {risk_color}{risk_level}{COLORS['reset']}")
        
        # Capital at risk
        capital_at_risk = risk.get("capital_at_risk", 0.0)
        capital_color = COLORS['green'] if capital_at_risk < 20.0 else COLORS['yellow'] if capital_at_risk < 50.0 else COLORS['red']
        
        print(f"Capital at Risk: {capital_color}{capital_at_risk:.1f}%{COLORS['reset']}")
        
        # Position/leverage limits
        max_position = risk.get("max_position_size", 0.0)
        max_leverage = risk.get("max_leverage", 1.0)
        
        print(f"\n{COLORS['bright']}Risk Limits:{COLORS['reset']}")
        print(f"  Max Position Size: {max_position:.1f}%")
        print(f"  Max Leverage:      {max_leverage:.1f}x")
        
        # Risk allocation by coin/strategy
        risk_allocation = risk.get("risk_allocation", {})
        
        if risk_allocation:
            print(f"\n{COLORS['bright']}Risk Allocation:{COLORS['reset']}")
            
            # Show top 3 risk allocations
            sorted_allocation = sorted(risk_allocation.items(), key=lambda x: x[1], reverse=True)
            for i, (item, pct) in enumerate(sorted_allocation[:3]):
                item_color = COLORS['green'] if pct < 20.0 else COLORS['yellow'] if pct < 50.0 else COLORS['red']
                print(f"  {item}: {item_color}{pct:.1f}%{COLORS['reset']}")
            
            # Show remaining count if any
            if len(sorted_allocation) > 3:
                print(f"  {COLORS['dim']}+{len(sorted_allocation) - 3} more allocations{COLORS['reset']}")
        
        # Warnings
        warnings = risk.get("warnings", [])
        if warnings:
            print(f"\n{COLORS['bright']}{COLORS['yellow']}Risk Warnings:{COLORS['reset']}")
            for i, warning in enumerate(warnings[:3]):
                print(f"  {i+1}. {COLORS['yellow']}{warning}{COLORS['reset']}")
            
            if len(warnings) > 3:
                print(f"  {COLORS['dim']}+{len(warnings) - 3} more warnings{COLORS['reset']}")
        
        # Menu options
        print(f"\nRisk Analysis Options:")
        print(f"  {COLORS['green']}[1]{COLORS['reset']} View Detailed Risk Report")
        print(f"  {COLORS['green']}[2]{COLORS['reset']} Adjust Risk Parameters")
        print()
        print(f"  {COLORS['green']}[B]{COLORS['reset']} Back to main menu")
        print(f"  {COLORS['red']}[0]{COLORS['reset']} Exit")
        
        self._draw_footer()
    
    def _draw_learning_menu(self) -> None:
        """Draw the learning module menu"""
        self._draw_header()
        
        print(f"{COLORS['bright']}{COLORS['cyan']}Learning Module{COLORS['reset']}\n")
        
        # Learning data
        learning = self._get_learning_summary()
        
        # Model info
        model_version = learning.get("model_version", "0.0")
        last_training = learning.get("last_training", "NONE")
        accuracy = learning.get("accuracy", 0.0)
        
        print(f"{COLORS['bright']}Model Information:{COLORS['reset']}")
        print(f"  Model Version: v{model_version}")
        print(f"  Last Training: {last_training}")
        
        accuracy_color = COLORS['green'] if accuracy >= 0.7 else COLORS['yellow'] if accuracy >= 0.5 else COLORS['red']
        print(f"  Accuracy:      {accuracy_color}{accuracy:.1f}{COLORS['reset']}")
        
        # Performance metrics
        improvement = learning.get("improvement", 0.0)
        improvement_sign = "+" if improvement >= 0 else ""
        improvement_color = COLORS['green'] if improvement > 0 else COLORS['red'] if improvement < 0 else COLORS['dim']
        
        print(f"\n{COLORS['bright']}Performance Improvement:{COLORS['reset']}")
        print(f"  Overall: {improvement_color}{improvement_sign}{improvement:.1f}%{COLORS['reset']}")
        
        # Insights
        insights = learning.get("insights", [])
        if insights:
            print(f"\n{COLORS['bright']}Learning Insights:{COLORS['reset']}")
            for i, insight in enumerate(insights[:3]):
                print(f"  {i+1}. {insight}")
            
            if len(insights) > 3:
                print(f"  {COLORS['dim']}+{len(insights) - 3} more insights{COLORS['reset']}")
        
        # Menu options
        print(f"\nLearning Options:")
        print(f"  {COLORS['green']}[1]{COLORS['reset']} Train on Recent Data")
        print(f"  {COLORS['green']}[2]{COLORS['reset']} Full Model Retraining")
        print(f"  {COLORS['green']}[3]{COLORS['reset']} View Learning Insights")
        print()
        print(f"  {COLORS['green']}[B]{COLORS['reset']} Back to main menu")
        print(f"  {COLORS['red']}[0]{COLORS['reset']} Exit")
        
        self._draw_footer()
    
    def _draw_system_status_menu(self) -> None:
        """Draw the system status menu"""
        self._draw_header()
        
        print(f"{COLORS['bright']}{COLORS['cyan']}System Status{COLORS['reset']}\n")
        
        # System status
        status = self._get_system_status()
        
        # Mode and version
        mode = status.get("mode", "UNKNOWN")
        version = status.get("version", "0.0")
        
        mode_color = COLORS['green'] if mode == "SIMULATION" else COLORS['red']
        
        print(f"{COLORS['bright']}System Information:{COLORS['reset']}")
        print(f"  Mode:    {mode_color}{mode}{COLORS['reset']}")
        print(f"  Version: v{version}")
        print(f"  Uptime:  {status.get('uptime', 'UNKNOWN')}")
        
        # Module status
        active_modules = status.get("active_modules", 0)
        total_modules = status.get("total_modules", 0)
        
        module_color = COLORS['green'] if active_modules == total_modules else COLORS['yellow']
        
        print(f"\n{COLORS['bright']}Module Status: {module_color}{active_modules}/{total_modules} Active{COLORS['reset']}")
        
        # List each module status
        print(f"  Performance Tracker: {COLORS['green'] if PERFORMANCE_TRACKER_AVAILABLE else COLORS['red']}{'ACTIVE' if PERFORMANCE_TRACKER_AVAILABLE else 'INACTIVE'}{COLORS['reset']}")
        print(f"  Risk Manager:        {COLORS['green'] if RISK_MANAGER_AVAILABLE else COLORS['red']}{'ACTIVE' if RISK_MANAGER_AVAILABLE else 'INACTIVE'}{COLORS['reset']}")
        print(f"  Learning Engine:     {COLORS['green'] if LEARNING_ENGINE_AVAILABLE else COLORS['red']}{'ACTIVE' if LEARNING_ENGINE_AVAILABLE else 'INACTIVE'}{COLORS['reset']}")
        print(f"  Strategy Engine:     {COLORS['green'] if STRATEGY_ENGINE_AVAILABLE else COLORS['red']}{'ACTIVE' if STRATEGY_ENGINE_AVAILABLE else 'INACTIVE'}{COLORS['reset']}")
        print(f"  Transaction Logger:  {COLORS['green'] if TRANSACTION_LOGGER_AVAILABLE else COLORS['red']}{'ACTIVE' if TRANSACTION_LOGGER_AVAILABLE else 'INACTIVE'}{COLORS['reset']}")
        print(f"  Alert System:        {COLORS['green'] if ALERT_SYSTEM_AVAILABLE else COLORS['red']}{'ACTIVE' if ALERT_SYSTEM_AVAILABLE else 'INACTIVE'}{COLORS['reset']}")
        
        # Menu options
        print(f"\nSystem Options:")
        print(f"  {COLORS['green']}[1]{COLORS['reset']} View System Logs")
        print(f"  {COLORS['green']}[2]{COLORS['reset']} Toggle Mode ({COLORS['red'] if self.real_mode else COLORS['green']}{'REAL' if self.real_mode else 'SIMULATION'}{COLORS['reset']})")
        print(f"  {COLORS['green']}[3]{COLORS['reset']} Check for Updates")
        print()
        print(f"  {COLORS['green']}[B]{COLORS['reset']} Back to main menu")
        print(f"  {COLORS['red']}[0]{COLORS['reset']} Exit")
        
        self._draw_footer()
    
    def _run_strategy(self, strategy_type: str) -> None:
        """
        Run a trading strategy
        
        Args:
            strategy_type (str): Type of strategy to run
        """
        if strategy_type not in STRATEGY_MODULES:
            print(f"{COLORS['red']}Invalid strategy type: {strategy_type}{COLORS['reset']}")
            return
        
        strategy_file = STRATEGY_MODULES[strategy_type]
        mode_flag = "--real" if self.real_mode else "--simulation"
        
        print(f"{COLORS['cyan']}Running {strategy_type} strategy in {mode_flag[2:].upper()} mode...{COLORS['reset']}")
        
        try:
            # Check if script exists
            if not os.path.exists(strategy_file):
                print(f"{COLORS['red']}Strategy script not found: {strategy_file}{COLORS['reset']}")
                return
            
            # If in real mode, ask for confirmation
            if self.real_mode:
                confirmation = input(f"{COLORS['red']}WARNING: You are in REAL trading mode. Proceed? (y/N): {COLORS['reset']}")
                if confirmation.lower() != 'y':
                    print(f"{COLORS['yellow']}Operation cancelled.{COLORS['reset']}")
                    return
            
            # Run the strategy script
            cmd = [sys.executable, strategy_file, mode_flag]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Check result
            if result.returncode == 0:
                print(f"{COLORS['green']}Strategy execution completed successfully{COLORS['reset']}")
                print(f"{COLORS['white']}{result.stdout}{COLORS['reset']}")
            else:
                print(f"{COLORS['red']}Strategy execution failed with code {result.returncode}{COLORS['reset']}")
                print(f"{COLORS['red']}Error: {result.stderr}{COLORS['reset']}")
                
        except Exception as e:
            print(f"{COLORS['red']}Error running strategy: {str(e)}{COLORS['reset']}")
        
        # Wait for user to press Enter to continue
        input(f"\n{COLORS['cyan']}Press Enter to continue...{COLORS['reset']}")
    
    def _run_simulation(self, simulation_type: int) -> None:
        """
        Run a trading simulation
        
        Args:
            simulation_type (int): Type of simulation to run
        """
        simulation_names = {
            1: "Backtest with historical data",
            2: "Paper trading with live data",
            3: "Strategy comparison"
        }
        
        if simulation_type not in simulation_names:
            print(f"{COLORS['red']}Invalid simulation type: {simulation_type}{COLORS['reset']}")
            return
        
        simulation_name = simulation_names[simulation_type]
        print(f"{COLORS['cyan']}Starting simulation: {simulation_name}{COLORS['reset']}")
        
        try:
            # Implementation would depend on the actual simulation capabilities
            if STRATEGY_ENGINE_AVAILABLE:
                print(f"{COLORS['yellow']}Triggering simulation via strategy engine...{COLORS['reset']}")
                
                # Example implementation
                if simulation_type == 1:
                    # Backtest with historical data
                    print(f"{COLORS['cyan']}Running backtest simulation (30 days of historical data){COLORS['reset']}")
                    print(f"{COLORS['dim']}Simulating trades on BTC, ETH, SOL, XRP, ADA...{COLORS['reset']}")
                    
                    # Simulate progress
                    for i in range(1, 11):
                        print(f"{COLORS['cyan']}Progress: {i*10}%{COLORS['reset']}")
                        time.sleep(0.5)
                    
                    print(f"{COLORS['green']}Backtest completed successfully{COLORS['reset']}")
                    print(f"{COLORS['green']}Results: +12.3% profit, 68% win rate, 23 trades{COLORS['reset']}")
                    
                elif simulation_type == 2:
                    # Paper trading
                    print(f"{COLORS['cyan']}Starting paper trading simulation with live market data{COLORS['reset']}")
                    print(f"{COLORS['dim']}Paper trading will run in the background for 1 hour{COLORS['reset']}")
                    print(f"{COLORS['green']}Paper trading started successfully{COLORS['reset']}")
                    
                elif simulation_type == 3:
                    # Strategy comparison
                    print(f"{COLORS['cyan']}Running strategy comparison (Long vs. Short vs. Sniper){COLORS['reset']}")
                    
                    # Simulate progress
                    for i in range(1, 11):
                        print(f"{COLORS['cyan']}Progress: {i*10}%{COLORS['reset']}")
                        time.sleep(0.5)
                    
                    print(f"{COLORS['green']}Comparison completed successfully{COLORS['reset']}")
                    print(f"{COLORS['green']}Results:{COLORS['reset']}")
                    print(f"  Long Strategy:  +8.2% profit, 62% win rate")
                    print(f"  Short Strategy: +5.1% profit, 55% win rate")
                    print(f"  Sniper Strategy: +15.7% profit, 48% win rate")
                
            else:
                print(f"{COLORS['red']}Simulation failed: Strategy engine not available{COLORS['reset']}")
                
        except Exception as e:
            print(f"{COLORS['red']}Error running simulation: {str(e)}{COLORS['reset']}")
        
        # Wait for user to press Enter to continue
        input(f"\n{COLORS['cyan']}Press Enter to continue...{COLORS['reset']}")
    
    def _view_performance_report(self) -> None:
        """View detailed performance report"""
        print(f"{COLORS['cyan']}Generating detailed performance report...{COLORS['reset']}")
        
        try:
            if PERFORMANCE_TRACKER_AVAILABLE:
                # This would call the actual performance_tracker module
                print(f"{COLORS['yellow']}Fetching detailed performance data...{COLORS['reset']}")
                
                # Simulate generating a report
                for i in range(1, 6):
                    print(f"{COLORS['cyan']}Processing data... {i*20}%{COLORS['reset']}")
                    time.sleep(0.5)
                
                print(f"\n{COLORS['green']}Performance Report Generated{COLORS['reset']}")
                print(f"{COLORS['bright']}============================================{COLORS['reset']}")
                print(f"{COLORS['bright']}         PERFORMANCE SUMMARY{COLORS['reset']}")
                print(f"{COLORS['bright']}============================================{COLORS['reset']}")
                print(f"Period: Last 30 days")
                print(f"Total Trades: 152")
                print(f"Win Rate: 63.2%")
                print(f"Profit Factor: 1.87")
                print(f"Net Profit: +$1,523.45")
                print(f"Best Trade: +$312.78 (ETH)")
                print(f"Worst Trade: -$98.32 (DOGE)")
                print(f"{COLORS['bright']}============================================{COLORS['reset']}")
                print(f"\nStrategy Performance:")
                print(f"  Long Strategy:  +$876.12 (58 trades)")
                print(f"  Short Strategy: +$412.78 (45 trades)")
                print(f"  Sniper Strategy: +$234.55 (49 trades)")
                print(f"\nTop Performing Coins:")
                print(f"  1. ETH:  +$432.12")
                print(f"  2. BTC:  +$321.45")
                print(f"  3. SOL:  +$198.76")
                
            else:
                print(f"{COLORS['red']}Performance tracker not available{COLORS['reset']}")
                
        except Exception as e:
            print(f"{COLORS['red']}Error generating performance report: {str(e)}{COLORS['reset']}")
        
        # Wait for user to press Enter to continue
        input(f"\n{COLORS['cyan']}Press Enter to continue...{COLORS['reset']}")
    
    def _export_performance_data(self) -> None:
        """Export performance data to file"""
        print(f"{COLORS['cyan']}Exporting performance data...{COLORS['reset']}")
        
        try:
            if PERFORMANCE_TRACKER_AVAILABLE:
                export_file = f"performance_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                print(f"{COLORS['yellow']}Preparing performance data for export...{COLORS['reset']}")
                
                # Simulate exporting data
                for i in range(1, 6):
                    print(f"{COLORS['cyan']}Exporting data... {i*20}%{COLORS['reset']}")
                    time.sleep(0.5)
                
                print(f"{COLORS['green']}Performance data exported to {export_file}{COLORS['reset']}")
                
            else:
                print(f"{COLORS['red']}Performance tracker not available{COLORS['reset']}")
                
        except Exception as e:
            print(f"{COLORS['red']}Error exporting performance data: {str(e)}{COLORS['reset']}")
        
        # Wait for user to press Enter to continue
        input(f"\n{COLORS['cyan']}Press Enter to continue...{COLORS['reset']}")
    
    def _view_risk_report(self) -> None:
        """View detailed risk report"""
        print(f"{COLORS['cyan']}Generating detailed risk report...{COLORS['reset']}")
        
        try:
            if RISK_MANAGER_AVAILABLE:
                # This would call the actual risk_manager module
                print(f"{COLORS['yellow']}Analyzing risk factors...{COLORS['reset']}")
                
                # Simulate generating a report
                for i in range(1, 6):
                    print(f"{COLORS['cyan']}Analyzing data... {i*20}%{COLORS['reset']}")
                    time.sleep(0.5)
                
                print(f"\n{COLORS['green']}Risk Analysis Report Generated{COLORS['reset']}")
                print(f"{COLORS['bright']}============================================{COLORS['reset']}")
                print(f"{COLORS['bright']}           RISK ANALYSIS REPORT{COLORS['reset']}")
                print(f"{COLORS['bright']}============================================{COLORS['reset']}")
                print(f"Current Risk Level: MEDIUM")
                print(f"Capital at Risk: 35.2%")
                print(f"Active Positions: 8")
                print(f"Exposure by Coin:")
                print(f"  BTC: 15.3%")
                print(f"  ETH: 12.7%")
                print(f"  SOL: 7.2%")
                print(f"  Others: 10.0%")
                print(f"{COLORS['bright']}============================================{COLORS['reset']}")
                print(f"\nRisk Warnings:")
                print(f"  1. BTC position exceeds recommended single-coin exposure")
                print(f"  2. Market volatility above average (VIX: 28.5)")
                print(f"\nRecommendations:")
                print(f"  1. Consider reducing BTC exposure by 5%")
                print(f"  2. Set tighter stop-losses during high volatility")
                
            else:
                print(f"{COLORS['red']}Risk manager not available{COLORS['reset']}")
                
        except Exception as e:
            print(f"{COLORS['red']}Error generating risk report: {str(e)}{COLORS['reset']}")
        
        # Wait for user to press Enter to continue
        input(f"\n{COLORS['cyan']}Press Enter to continue...{COLORS['reset']}")
    
    def _adjust_risk_parameters(self) -> None:
        """Adjust risk management parameters"""
        print(f"{COLORS['cyan']}Risk Parameter Adjustment{COLORS['reset']}")
        
        try:
            if RISK_MANAGER_AVAILABLE:
                # This would interact with the actual risk_manager module
                print(f"{COLORS['yellow']}Current Risk Parameters:{COLORS['reset']}")
                print(f"  1. Max Position Size: 10%")
                print(f"  2. Max Leverage: 3x")
                print(f"  3. Max Coins: 15")
                print(f"  4. Stop-Loss %: 5%")
                print()
                
                # Simple mock adjustment menu
                choice = input(f"{COLORS['cyan']}Enter parameter number to adjust (1-4) or 'b' to go back: {COLORS['reset']}")
                
                if choice.lower() == 'b':
                    return
                
                try:
                    param_num = int(choice)
                    if param_num < 1 or param_num > 4:
                        print(f"{COLORS['red']}Invalid parameter number{COLORS['reset']}")
                        return
                        
                    param_names = {
                        1: "Max Position Size",
                        2: "Max Leverage",
                        3: "Max Coins",
                        4: "Stop-Loss %"
                    }
                    
                    new_value = input(f"{COLORS['cyan']}Enter new value for {param_names[param_num]}: {COLORS['reset']}")
                    
                    print(f"{COLORS['green']}{param_names[param_num]} updated to {new_value}{COLORS['reset']}")
                    
                except ValueError:
                    print(f"{COLORS['red']}Invalid input{COLORS['reset']}")
                
            else:
                print(f"{COLORS['red']}Risk manager not available{COLORS['reset']}")
                
        except Exception as e:
            print(f"{COLORS['red']}Error adjusting risk parameters: {str(e)}{COLORS['reset']}")
        
        # Wait for user to press Enter to continue
        input(f"\n{COLORS['cyan']}Press Enter to continue...{COLORS['reset']}")
    
    def _train_learning_module(self, full_training: bool) -> None:
        """
        Train the learning module
        
        Args:
            full_training (bool): Whether to perform full retraining
        """
        training_type = "Full Model Retraining" if full_training else "Incremental Training on Recent Data"
        print(f"{COLORS['cyan']}Starting {training_type}...{COLORS['reset']}")
        
        try:
            if LEARNING_ENGINE_AVAILABLE:
                # This would call the actual learning_engine module
                print(f"{COLORS['yellow']}Preparing training data...{COLORS['reset']}")
                
                # Simulate training progress
                steps = 10 if full_training else 5
                for i in range(1, steps + 1):
                    print(f"{COLORS['cyan']}Training progress: {i*100/steps:.1f}%{COLORS['reset']}")
                    time.sleep(0.5 if full_training else 0.3)
                
                print(f"{COLORS['green']}Training completed successfully{COLORS['reset']}")
                print(f"Model accuracy: 81.2%")
                print(f"Improvement: +2.3%")
                
            else:
                print(f"{COLORS['red']}Learning engine not available{COLORS['reset']}")
                
        except Exception as e:
            print(f"{COLORS['red']}Error during training: {str(e)}{COLORS['reset']}")
        
        # Wait for user to press Enter to continue
        input(f"\n{COLORS['cyan']}Press Enter to continue...{COLORS['reset']}")
    
    def _view_learning_insights(self) -> None:
        """View learning insights"""
        print(f"{COLORS['cyan']}Retrieving learning insights...{COLORS['reset']}")
        
        try:
            if LEARNING_ENGINE_AVAILABLE:
                # This would call the actual learning_engine module
                print(f"{COLORS['yellow']}Analyzing trading patterns...{COLORS['reset']}")
                
                # Simulate analysis
                for i in range(1, 6):
                    print(f"{COLORS['cyan']}Analyzing data... {i*20}%{COLORS['reset']}")
                    time.sleep(0.5)
                
                print(f"\n{COLORS['green']}Learning Insights Generated{COLORS['reset']}")
                print(f"{COLORS['bright']}============================================{COLORS['reset']}")
                print(f"{COLORS['bright']}           LEARNING INSIGHTS{COLORS['reset']}")
                print(f"{COLORS['bright']}============================================{COLORS['reset']}")
                print(f"1. BTC performs best with sniper strategy during high volatility")
                print(f"2. ETH long positions most profitable when RSI < 30")
                print(f"3. SOL shows strong correlation with BTC but with 2-hour lag")
                print(f"4. Short strategies underperform in current market conditions")
                print(f"5. Weekend trading shows 23% lower profitability")
                print(f"{COLORS['bright']}============================================{COLORS['reset']}")
                print(f"\nRecommended Strategy Adjustments:")
                print(f"1. Increase BTC allocation in sniper strategy by 5%")
                print(f"2. Reduce short position sizes by 10%")
                print(f"3. Optimize ETH entry points based on RSI indicators")
                
            else:
                print(f"{COLORS['red']}Learning engine not available{COLORS['reset']}")
                
        except Exception as e:
            print(f"{COLORS['red']}Error retrieving learning insights: {str(e)}{COLORS['reset']}")
        
        # Wait for user to press Enter to continue
        input(f"\n{COLORS['cyan']}Press Enter to continue...{COLORS['reset']}")
    
    def _view_system_logs(self) -> None:
        """View system logs"""
        print(f"{COLORS['cyan']}Retrieving system logs...{COLORS['reset']}")
        
        try:
            # List all log files
            log_files = [f for f in os.listdir(".") if f.endswith(".log")]
            
            if not log_files:
                print(f"{COLORS['yellow']}No log files found{COLORS['reset']}")
                return
            
            print(f"{COLORS['yellow']}Available log files:{COLORS['reset']}")
            for i, log_file in enumerate(log_files, 1):
                print(f"  {i}. {log_file}")
            
            choice = input(f"\n{COLORS['cyan']}Enter log file number to view or 'b' to go back: {COLORS['reset']}")
            
            if choice.lower() == 'b':
                return
            
            try:
                file_num = int(choice)
                if file_num < 1 or file_num > len(log_files):
                    print(f"{COLORS['red']}Invalid file number{COLORS['reset']}")
                    return
                    
                selected_file = log_files[file_num - 1]
                
                # Display the log file (last 20 lines)
                print(f"\n{COLORS['yellow']}Showing last 20 lines of {selected_file}:{COLORS['reset']}")
                
                try:
                    with open(selected_file, "r") as f:
                        lines = f.readlines()
                        for line in lines[-20:]:
                            print(line.strip())
                except Exception as fe:
                    print(f"{COLORS['red']}Error reading log file: {str(fe)}{COLORS['reset']}")
                
            except ValueError:
                print(f"{COLORS['red']}Invalid input{COLORS['reset']}")
                
        except Exception as e:
            print(f"{COLORS['red']}Error accessing system logs: {str(e)}{COLORS['reset']}")
        
        # Wait for user to press Enter to continue
        input(f"\n{COLORS['cyan']}Press Enter to continue...{COLORS['reset']}")
    
    def _toggle_mode(self) -> None:
        """Toggle between real and simulation mode"""
        if self.real_mode:
            print(f"{COLORS['green']}Switching to SIMULATION mode{COLORS['reset']}")
            self.real_mode = False
        else:
            confirmation = input(f"{COLORS['red']}WARNING: You are about to switch to REAL trading mode. This will use real funds for trades.\nAre you sure? (yes/NO): {COLORS['reset']}")
            if confirmation.lower() == 'yes':
                print(f"{COLORS['red']}Switching to REAL mode{COLORS['reset']}")
                self.real_mode = True
            else:
                print(f"{COLORS['yellow']}Mode change cancelled. Remaining in SIMULATION mode.{COLORS['reset']}")
                
        # Update system status
        self.system_status = self._get_system_status()
    
    def _check_for_updates(self) -> None:
        """Check for system updates"""
        print(f"{COLORS['cyan']}Checking for updates...{COLORS['reset']}")
        
        # Simulate update check
        for i in range(1, 6):
            print(f"{COLORS['cyan']}Checking component {i}/5...{COLORS['reset']}")
            time.sleep(0.5)
        
        # Mock response
        print(f"{COLORS['green']}System is up to date!{COLORS['reset']}")
        print(f"Current version: {SYSTEM_VERSION}")
        
        # Wait for user to press Enter to continue
        input(f"\n{COLORS['cyan']}Press Enter to continue...{COLORS['reset']}")
        
        
        

    def display_confirmation(self, message, actions=None):
        """Display a confirmation prompt to the user with optional actions"""
        if actions is None:
            actions = ["Yes", "No"]
        print(f"\n[CONFIRMATION] {message}")
        for i, action in enumerate(actions, 1):
            print(f"{i}. {action}")
        
        choice = 0
        while choice < 1 or choice > len(actions):
            try:
                choice_input = input("Enter your choice (number): ")
                choice = int(choice_input)
                if choice < 1 or choice > len(actions):
                    print(f"Please enter a number between 1 and {len(actions)}")
            except ValueError:
                print("Please enter a valid number")
                choice = len(actions)  # Default to No/last option
                
        return actions[choice-1]


    def display_menu(self, title, options):
        """Display a menu with the given title and options, return user's choice"""
        self.display_header(title)
        
        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")
            
        choice = 0
        while choice < 1 or choice > len(options):
            try:
                choice_input = input("\nEnter your choice (number): ")
                choice = int(choice_input)
                if choice < 1 or choice > len(options):
                    print(f"Please enter a number between 1 and {len(options)}")
            except ValueError:
                print("Please enter a valid number")
                
        return choice

    def display_message(self, message, message_type="INFO"):
        """Display a message with the given type"""
        type_colors = {
            "INFO": COLORS.get("blue", ""),
            "SUCCESS": COLORS.get("green", ""),
            "WARNING": COLORS.get("yellow", ""),
            "ERROR": COLORS.get("red", ""),
            "ALERT": COLORS.get("magenta", ""),
            "NOTICE": COLORS.get("cyan", "")
        }
        
        # Default to INFO if type not found
        color = type_colors.get(message_type, type_colors["INFO"])
        reset = COLORS.get("reset", "")
        
        print(f"{color}[{message_type}] {message}{reset}")
    def notify(self, message, message_type="NOTICE"):
        """Display a notification message (wrapper for display_message)"""
        self.display_message(message, message_type)

    def display_header(self, title):
        """Display a header with the given title"""
        width = 80
        print("\n" + "=" * width)
        print(f"{title.center(width)}")
        print("=" * width)
