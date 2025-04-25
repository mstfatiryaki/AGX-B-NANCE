from random import choice
import random
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SentientTrader.AI - Capital Manager V2
--------------------------------------
This module manages the system's trading capital,
allocates and releases funds for trades, and tracks PnL.
"""

import os
import sys
import json
import uuid
import time
import logging
import datetime
import argparse
from typing import Dict, List, Any, Optional, Union, Tuple
from decimal import Decimal, ROUND_DOWN
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("capital_manager.log")]
)
logger = logging.getLogger("CapitalManager")

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
REAL_MODE = False  # Set to True for real balance management, False for simulation
CURRENT_TIME = "2025-04-21 19:21:24"  # UTC
CURRENT_USER = "mstfatiryaki"

# File paths
WALLET_STATE_FILE = "wallet_state.json"
WALLET_LOG_FILE = "wallet_log.jsonl"

# Default wallet state
DEFAULT_WALLET_STATE = {
    "total_balance": 10000.0,   # Default starting balance in USDT
    "free_balance": 10000.0,    # Available for trading
    "locked_balance": 0.0,      # Currently in use for trades
    "allocated_funds": {},      # Funds allocated to specific coins
    "last_update": CURRENT_TIME,
    "created_at": CURRENT_TIME,
    "user": CURRENT_USER
}

# Alert thresholds
ALERT_THRESHOLDS = {
    "low_balance_warning": 1000.0,   # Warning when free balance drops below
    "low_balance_critical": 500.0,   # Critical when free balance drops below
    "high_allocation_warning": 30.0,  # Warning when allocation exceeds % of total
    "high_allocation_critical": 50.0, # Critical when allocation exceeds % of total
    "negative_balance_warning": True  # Alert on negative balance
}

class CapitalManager:
    """Manages the capital and fund allocation for SentientTrader.AI"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, real_mode: bool = REAL_MODE):
        """Singleton pattern to ensure only one CapitalManager instance"""
        if cls._instance is None:
            cls._instance = super(CapitalManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, real_mode: bool = REAL_MODE):
        """
        Initialize the capital manager
        
        Args:
            real_mode (bool): Whether to use real balance management
        """
        # Only initialize once due to singleton pattern
        if CapitalManager._initialized:
            return
            
        self.real_mode = real_mode
        self.wallet_state = DEFAULT_WALLET_STATE.copy()
        self.transactions = []
        self.transaction_lock = False  # Simple transaction lock
        
        # Ensure wallet log directory exists
        log_dir = os.path.dirname(WALLET_LOG_FILE)
        if log_dir and not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir)
            except Exception as e:
                logger.error(f"Failed to create log directory: {e}")
        
        # Load existing wallet state
        self.load_wallet_state()
        
        CapitalManager._initialized = True
        logger.info(f"Capital Manager initialized (REAL_MODE: {self.real_mode})")
    
    def load_wallet_state(self) -> bool:
        """
        Load wallet state from JSON file
        
        Returns:
            bool: Success status
        """
        try:
            if os.path.exists(WALLET_STATE_FILE):
                with open(WALLET_STATE_FILE, "r", encoding="utf-8") as f:
                    loaded_state = json.load(f)
                
                # Validate and update wallet state
                if self._validate_wallet_state(loaded_state):
                    self.wallet_state = loaded_state
                    logger.info("Wallet state loaded successfully")
                    
                    # Log current state after loading
                    self._log_wallet_action("wallet_loaded", {
                        "total_balance": self.wallet_state["total_balance"],
                        "free_balance": self.wallet_state["free_balance"],
                        "locked_balance": self.wallet_state["locked_balance"]
                    })
                    
                    return True
                else:
                    logger.warning("Invalid wallet state format, using defaults")
            else:
                logger.info(f"{WALLET_STATE_FILE} not found, creating new wallet")
            
            # Initialize with default wallet
            self.wallet_state = DEFAULT_WALLET_STATE.copy()
            self.save_wallet_state()
            
            return True
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in {WALLET_STATE_FILE}")
            return False
        except Exception as e:
            logger.error(f"Error loading wallet state: {str(e)}")
            return False
    
    def save_wallet_state(self) -> bool:
        """
        Save wallet state to JSON file
        
        Returns:
            bool: Success status
        """
        try:
            # Update timestamp
            self.wallet_state["last_update"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(WALLET_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(self.wallet_state, f, indent=4, ensure_ascii=False)
            
            logger.info("Wallet state saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving wallet state: {str(e)}")
            return False
    
    def _validate_wallet_state(self, wallet_state: Dict[str, Any]) -> bool:
        """
        Validate wallet state format
        
        Args:
            wallet_state (Dict[str, Any]): Wallet state to validate
            
        Returns:
            bool: Validation result
        """
        required_keys = ["total_balance", "free_balance", "locked_balance", "allocated_funds"]
        
        if not isinstance(wallet_state, dict):
            return False
        
        # Check if all required keys exist
        if not all(key in wallet_state for key in required_keys):
            return False
        
        # Check if balance values are numeric
        if not all(isinstance(wallet_state[key], (int, float)) for key in ["total_balance", "free_balance", "locked_balance"]):
            return False
        
        # Check if allocated_funds is a dictionary
        if not isinstance(wallet_state["allocated_funds"], dict):
            return False
        
        return True
    
    def get_total_balance(self) -> float:
        """
        Get total balance
        
        Returns:
            float: Total balance in USDT
        """
        return self.wallet_state["total_balance"]
    
    def get_free_balance(self) -> float:
        """
        Get free (available) balance
        
        Returns:
            float: Free balance in USDT
        """
        return self.wallet_state["free_balance"]
    
    def get_locked_balance(self) -> float:
        """
        Get locked balance (in trades)
        
        Returns:
            float: Locked balance in USDT
        """
        return self.wallet_state["locked_balance"]
    
    def get_allocated_funds(self) -> Dict[str, float]:
        """
        Get all allocated funds by coin
        
        Returns:
            Dict[str, float]: Allocated funds by coin
        """
        return self.wallet_state["allocated_funds"]
    
    def get_coin_allocation(self, coin: str) -> float:
        """
        Get allocation for a specific coin
        
        Args:
            coin (str): Coin symbol
            
        Returns:
            float: Allocated amount for the coin
        """
        return self.wallet_state["allocated_funds"].get(coin, 0.0)
    
    def allocate_funds(self, coin: str, amount: float, transaction_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Allocate funds for a trade
        
        Args:
            coin (str): Coin symbol
            amount (float): Amount to allocate in USDT
            transaction_id (Optional[str]): Transaction identifier
            
        Returns:
            Dict[str, Any]: Allocation result
        """
        # Input validation
        if not coin or not isinstance(coin, str):
            error_msg = "Invalid coin symbol"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        
        coin = coin.upper()  # Normalize coin symbol
        
        try:
            amount = float(amount)
        except (ValueError, TypeError):
            error_msg = f"Invalid amount: {amount}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        
        if amount <= 0:
            error_msg = "Amount must be positive"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        
        # Transaction lock for thread safety
        if self.transaction_lock:
            error_msg = "Another transaction is in progress"
            logger.warning(error_msg)
            return {"success": False, "error": error_msg}
        
        # Create transaction ID if not provided
        if not transaction_id:
            transaction_id = f"alloc_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        try:
            self.transaction_lock = True
            
            # Check if we have enough free balance
            if amount > self.wallet_state["free_balance"]:
                error_msg = f"Insufficient free balance: {self.wallet_state['free_balance']} < {amount}"
                logger.error(error_msg)
                
                if ALERT_SYSTEM_AVAILABLE:
                    alert_system.warning(
                        "Insufficient funds for allocation",
                        {
                            "coin": coin,
                            "requested": amount,
                            "available": self.wallet_state["free_balance"]
                        },
                        "capital_manager"
                    )
                
                return {"success": False, "error": error_msg}
            
            # Update balances
            self.wallet_state["free_balance"] -= amount
            self.wallet_state["locked_balance"] += amount
            
            # Update allocated funds for the coin
            if coin in self.wallet_state["allocated_funds"]:
                self.wallet_state["allocated_funds"][coin] += amount
            else:
                self.wallet_state["allocated_funds"][coin] = amount
            
            # Check allocation thresholds
            total_balance = self.wallet_state["total_balance"]
            coin_allocation = self.wallet_state["allocated_funds"][coin]
            allocation_percentage = (coin_allocation / total_balance) * 100 if total_balance > 0 else 0
            
            # Save updated wallet state
            self.save_wallet_state()
            
            # Log the allocation
            allocation_data = {
                "transaction_id": transaction_id,
                "coin": coin,
                "amount": amount,
                "free_balance_after": self.wallet_state["free_balance"],
                "locked_balance_after": self.wallet_state["locked_balance"],
                "coin_allocation_after": coin_allocation,
                "allocation_percentage": allocation_percentage
            }
            
            self._log_wallet_action("allocation", allocation_data)
            
            # Check if we need to send allocation warnings
            self._check_allocation_warnings(coin, allocation_percentage)
            
            logger.info(f"Allocated {amount} USDT for {coin}")
            return {
                "success": True,
                "transaction_id": transaction_id,
                "coin": coin,
                "amount": amount,
                "free_balance_after": self.wallet_state["free_balance"],
                "total_allocation": coin_allocation
            }
            
        except Exception as e:
            error_msg = f"Error allocating funds: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        finally:
            self.transaction_lock = False
    
    def release_funds(self, coin: str, amount: float, pnl: float = 0.0, 
                    transaction_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Release funds from a trade and apply PnL
        
        Args:
            coin (str): Coin symbol
            amount (float): Original allocated amount in USDT
            pnl (float): Profit/Loss amount in USDT
            transaction_id (Optional[str]): Transaction identifier
            
        Returns:
            Dict[str, Any]: Release result
        """
        # Input validation
        if not coin or not isinstance(coin, str):
            error_msg = "Invalid coin symbol"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        
        coin = coin.upper()  # Normalize coin symbol
        
        try:
            amount = float(amount)
            pnl = float(pnl)
        except (ValueError, TypeError):
            error_msg = f"Invalid amount or PnL: {amount}, {pnl}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        
        if amount <= 0:
            error_msg = "Amount must be positive"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        
        # Transaction lock for thread safety
        if self.transaction_lock:
            error_msg = "Another transaction is in progress"
            logger.warning(error_msg)
            return {"success": False, "error": error_msg}
        
        # Create transaction ID if not provided
        if not transaction_id:
            transaction_id = f"rel_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        try:
            self.transaction_lock = True
            
            # Check if we have enough allocation for this coin
            coin_allocation = self.wallet_state["allocated_funds"].get(coin, 0.0)
            
            if amount > coin_allocation:
                # This is a potential error condition, but we'll handle it gracefully
                logger.warning(f"Release amount ({amount}) exceeds allocation for {coin} ({coin_allocation})")
                amount = coin_allocation  # Adjust to max available
            
            # Update balances
            release_amount = amount + pnl  # Original amount plus profit (or minus loss)
            
            self.wallet_state["free_balance"] += release_amount
            self.wallet_state["locked_balance"] -= amount
            
            # Ensure locked balance doesn't go negative
            if self.wallet_state["locked_balance"] < 0:
                adjustment = abs(self.wallet_state["locked_balance"])
                logger.warning(f"Negative locked balance detected, adjusting by {adjustment}")
                self.wallet_state["locked_balance"] = 0
            
            # Update total balance with PnL
            self.wallet_state["total_balance"] += pnl
            
            # Update allocated funds for the coin
            if coin in self.wallet_state["allocated_funds"]:
                self.wallet_state["allocated_funds"][coin] -= amount
                
                # Ensure coin allocation doesn't go negative
                if self.wallet_state["allocated_funds"][coin] <= 0:
                    self.wallet_state["allocated_funds"].pop(coin)
            
            # Save updated wallet state
            self.save_wallet_state()
            
            # Log the release
            release_data = {
                "transaction_id": transaction_id,
                "coin": coin,
                "original_amount": amount,
                "pnl": pnl,
                "release_amount": release_amount,
                "free_balance_after": self.wallet_state["free_balance"],
                "locked_balance_after": self.wallet_state["locked_balance"],
                "total_balance_after": self.wallet_state["total_balance"]
            }
            
            self._log_wallet_action("release", release_data)
            
            # Check for balance warnings
            self._check_balance_warnings()
            
            # PnL message with appropriate color
            pnl_msg = f"+{pnl:.2f}" if pnl >= 0 else f"{pnl:.2f}"
            pnl_color = COLORS["green"] if pnl >= 0 else COLORS["red"]
            
            logger.info(f"Released {amount} USDT for {coin} with PnL: {pnl_msg}")
            print(f"{COLORS['cyan']}Released {amount} USDT for {coin} with PnL: {pnl_color}{pnl_msg}{COLORS['reset']}")
            
            return {
                "success": True,
                "transaction_id": transaction_id,
                "coin": coin,
                "amount": amount,
                "pnl": pnl,
                "free_balance_after": self.wallet_state["free_balance"],
                "total_balance_after": self.wallet_state["total_balance"]
            }
            
        except Exception as e:
            error_msg = f"Error releasing funds: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        finally:
            self.transaction_lock = False
    
    def adjust_balance(self, amount: float, reason: str, 
                      transaction_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Adjust total and free balance (for deposits, withdrawals, etc.)
        
        Args:
            amount (float): Amount to adjust (positive or negative)
            reason (str): Reason for adjustment
            transaction_id (Optional[str]): Transaction identifier
            
        Returns:
            Dict[str, Any]: Adjustment result
        """
        # Input validation
        try:
            amount = float(amount)
        except (ValueError, TypeError):
            error_msg = f"Invalid amount: {amount}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        
        if not reason:
            reason = "Manual adjustment"
        
        # Transaction lock for thread safety
        if self.transaction_lock:
            error_msg = "Another transaction is in progress"
            logger.warning(error_msg)
            return {"success": False, "error": error_msg}
        
        # Create transaction ID if not provided
        if not transaction_id:
            transaction_id = f"adj_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        try:
            self.transaction_lock = True
            
            # Check if we have enough free balance for negative adjustments
            if amount < 0 and abs(amount) > self.wallet_state["free_balance"]:
                error_msg = f"Insufficient free balance for adjustment: {self.wallet_state['free_balance']} < {abs(amount)}"
                logger.error(error_msg)
                
                if ALERT_SYSTEM_AVAILABLE:
                    alert_system.warning(
                        "Insufficient funds for balance adjustment",
                        {
                            "requested": amount,
                            "available": self.wallet_state["free_balance"],
                            "reason": reason
                        },
                        "capital_manager"
                    )
                
                return {"success": False, "error": error_msg}
            
            # Update balances
            self.wallet_state["total_balance"] += amount
            self.wallet_state["free_balance"] += amount
            
            # Save updated wallet state
            self.save_wallet_state()
            
            # Log the adjustment
            adjustment_data = {
                "transaction_id": transaction_id,
                "amount": amount,
                "reason": reason,
                "free_balance_after": self.wallet_state["free_balance"],
                "total_balance_after": self.wallet_state["total_balance"]
            }
            
            self._log_wallet_action("adjustment", adjustment_data)
            
            # Check for balance warnings
            self._check_balance_warnings()
            
            # Adjustment message with appropriate color
            adj_msg = f"+{amount:.2f}" if amount >= 0 else f"{amount:.2f}"
            adj_color = COLORS["green"] if amount >= 0 else COLORS["red"]
            
            logger.info(f"Balance adjusted by {adj_msg} USDT ({reason})")
            print(f"{COLORS['cyan']}Balance adjusted by {adj_color}{adj_msg}{COLORS['reset']} USDT ({reason})")
            
            return {
                "success": True,
                "transaction_id": transaction_id,
                "amount": amount,
                "reason": reason,
                "free_balance_after": self.wallet_state["free_balance"],
                "total_balance_after": self.wallet_state["total_balance"]
            }
            
        except Exception as e:
            error_msg = f"Error adjusting balance: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        finally:
            self.transaction_lock = False
    
    def get_available_balance(self) -> float:
        """
        Get available balance for trading
        
        Returns:
            float: Available balance in USDT
        """
        return self.wallet_state["free_balance"]
    
    def get_wallet_summary(self) -> Dict[str, Any]:
        """
        Get wallet summary
        
        Returns:
            Dict[str, Any]: Wallet summary
        """
        try:
            # Calculate metrics
            total_balance = self.wallet_state["total_balance"]
            free_balance = self.wallet_state["free_balance"]
            locked_balance = self.wallet_state["locked_balance"]
            
            allocated_funds = self.wallet_state["allocated_funds"]
            total_allocated = sum(allocated_funds.values())
            
            # Calculate allocation percentages
            allocation_percentages = {}
            for coin, amount in allocated_funds.items():
                percentage = (amount / total_balance) * 100 if total_balance > 0 else 0
                allocation_percentages[coin] = percentage
            
            # Calculate utilization percentage
            utilization = (locked_balance / total_balance) * 100 if total_balance > 0 else 0
            
            # Display wallet summary
            self._display_wallet_summary()
            
            return {
                "total_balance": total_balance,
                "free_balance": free_balance,
                "locked_balance": locked_balance,
                "utilization_percentage": utilization,
                "allocated_funds": allocated_funds,
                "allocation_percentages": allocation_percentages,
                "last_update": self.wallet_state.get("last_update", "")
            }
            
        except Exception as e:
            logger.error(f"Error getting wallet summary: {str(e)}")
            return {
                "total_balance": 0.0,
                "free_balance": 0.0,
                "locked_balance": 0.0,
                "error": str(e)
            }
    
    def _display_wallet_summary(self) -> None:
        """Display wallet summary in terminal"""
        try:
            total_balance = self.wallet_state["total_balance"]
            free_balance = self.wallet_state["free_balance"]
            locked_balance = self.wallet_state["locked_balance"]
            
            allocated_funds = self.wallet_state["allocated_funds"]
            last_update = self.wallet_state.get("last_update", "")
            
            # Calculate utilization percentage
            utilization = (locked_balance / total_balance) * 100 if total_balance > 0 else 0
            
            # Display header
            print(f"\n{COLORS['bright']}{COLORS['cyan']}{'=' * 50}{COLORS['reset']}")
            print(f"{COLORS['bright']}{COLORS['cyan']}💰 WALLET SUMMARY{COLORS['reset']}")
            print(f"{COLORS['bright']}{COLORS['cyan']}{'=' * 50}{COLORS['reset']}")
            
            # Basic wallet info
            print(f"{COLORS['bright']}Total Balance: {COLORS['green']}${total_balance:.2f}{COLORS['reset']}")
            print(f"{COLORS['white']}Free Balance: {COLORS['green']}${free_balance:.2f}{COLORS['reset']}")
            print(f"{COLORS['white']}Locked Balance: {COLORS['yellow']}${locked_balance:.2f}{COLORS['reset']}")
            print(f"{COLORS['white']}Utilization: {COLORS['yellow']}{utilization:.1f}%{COLORS['reset']}")
            
            # Allocated funds
            if allocated_funds:
                print(f"\n{COLORS['bright']}Allocated Funds:{COLORS['reset']}")
                
                # Sort by allocation amount
                sorted_allocations = sorted(
                    allocated_funds.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                for coin, amount in sorted_allocations:
                    # Calculate percentage
                    percentage = (amount / total_balance) * 100 if total_balance > 0 else 0
                    
                    # Determine color based on percentage
                    perc_color = COLORS["green"]
                    if percentage > ALERT_THRESHOLDS["high_allocation_critical"]:
                        perc_color = COLORS["red"]
                    elif percentage > ALERT_THRESHOLDS["high_allocation_warning"]:
                        perc_color = COLORS["yellow"]
                    
                    print(f"  {COLORS['bright']}{coin}: {COLORS['white']}${amount:.2f} ({perc_color}{percentage:.1f}%{COLORS['reset']})")
            else:
                print(f"\n{COLORS['bright']}No Allocated Funds{COLORS['reset']}")
            
            # Last update
            print(f"\n{COLORS['white']}Last Update: {last_update}{COLORS['reset']}")
            print(f"{COLORS['white']}Mode: {'REAL' if self.real_mode else 'MOCK'}{COLORS['reset']}")
            
            print(f"{COLORS['bright']}{COLORS['cyan']}{'=' * 50}{COLORS['reset']}\n")
            
        except Exception as e:
            logger.error(f"Error displaying wallet summary: {str(e)}")
    
    def _log_wallet_action(self, action_type: str, data: Dict[str, Any]) -> None:
        """
        Log wallet action to JSONL file
        
        Args:
            action_type (str): Type of action
            data (Dict[str, Any]): Action data
        """
        try:
            log_entry = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "action": action_type,
                "data": data
            }
            
            with open(WALLET_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                
        except Exception as e:
            logger.error(f"Error logging wallet action: {str(e)}")
    
    def _check_balance_warnings(self) -> None:
        """Check balance levels and send alerts if necessary"""
        if not ALERT_SYSTEM_AVAILABLE:
            return
        
        free_balance = self.wallet_state["free_balance"]
        total_balance = self.wallet_state["total_balance"]
        
        # Check for low free balance
        if free_balance < ALERT_THRESHOLDS["low_balance_critical"]:
            alert_system.error(
                "Critical low balance",
                {
                    "free_balance": free_balance,
                    "total_balance": total_balance,
                    "threshold": ALERT_THRESHOLDS["low_balance_critical"]
                },
                "capital_manager"
            )
        elif free_balance < ALERT_THRESHOLDS["low_balance_warning"]:
            alert_system.warning(
                "Low balance warning",
                {
                    "free_balance": free_balance,
                    "total_balance": total_balance,
                    "threshold": ALERT_THRESHOLDS["low_balance_warning"]
                },
                "capital_manager"
            )
        
        # Check for negative balance
        if ALERT_THRESHOLDS["negative_balance_warning"] and total_balance < 0:
            alert_system.critical(
                "Negative total balance detected",
                {
                    "total_balance": total_balance,
                    "free_balance": free_balance
                },
                "capital_manager"
            )
    
    def _check_allocation_warnings(self, coin: str, allocation_percentage: float) -> None:
        """
        Check allocation levels and send alerts if necessary
        
        Args:
            coin (str): Coin symbol
            allocation_percentage (float): Allocation percentage
        """
        if not ALERT_SYSTEM_AVAILABLE:
            return
        
        # Check for high allocation
        if allocation_percentage > ALERT_THRESHOLDS["high_allocation_critical"]:
            alert_system.error(
                f"Critical high allocation for {coin}",
                {
                    "coin": coin,
                    "allocation_percentage": allocation_percentage,
                    "threshold": ALERT_THRESHOLDS["high_allocation_critical"]
                },
                "capital_manager"
            )
        elif allocation_percentage > ALERT_THRESHOLDS["high_allocation_warning"]:
            alert_system.warning(
                f"High allocation warning for {coin}",
                {
                    "coin": coin,
                    "allocation_percentage": allocation_percentage,
                    "threshold": ALERT_THRESHOLDS["high_allocation_warning"]
                },
                "capital_manager"
            )

# Singleton instance
_capital_manager = None

def init(real_mode: bool = REAL_MODE) -> None:
    """
    Initialize the capital manager
    
    Args:
        real_mode (bool): Whether to use real balance management
    """
    global _capital_manager
    _capital_manager = CapitalManager(real_mode=real_mode)

def get_available_balance() -> float:
    """
    Get available balance for trading
    
    Returns:
        float: Available balance in USDT
    """
    global _capital_manager
    if _capital_manager is None:
        init()
    return _capital_manager.get_available_balance()

def get_total_balance() -> float:
    """
    Get total balance
    
    Returns:
        float: Total balance in USDT
    """
    global _capital_manager
    if _capital_manager is None:
        init()
    return _capital_manager.get_total_balance()

def get_locked_balance() -> float:
    """
    Get locked balance (in trades)
    
    Returns:
        float: Locked balance in USDT
    """
    global _capital_manager
    if _capital_manager is None:
        init()
    return _capital_manager.get_locked_balance()

def allocate_funds(coin: str, amount: float, transaction_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Allocate funds for a trade
    
    Args:
        coin (str): Coin symbol
        amount (float): Amount to allocate in USDT
        transaction_id (Optional[str]): Transaction identifier
        
    Returns:
        Dict[str, Any]: Allocation result
    """
    global _capital_manager
    if _capital_manager is None:
        init()
    return _capital_manager.allocate_funds(coin, amount, transaction_id)

def release_funds(coin: str, amount: float, pnl: float = 0.0, 
                transaction_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Release funds from a trade and apply PnL
    
    Args:
        coin (str): Coin symbol
        amount (float): Original allocated amount in USDT
        pnl (float): Profit/Loss amount in USDT
        transaction_id (Optional[str]): Transaction identifier
        
    Returns:
        Dict[str, Any]: Release result
    """
    global _capital_manager
    if _capital_manager is None:
        init()
    return _capital_manager.release_funds(coin, amount, pnl, transaction_id)

def adjust_balance(amount: float, reason: str, 
                  transaction_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Adjust total and free balance (for deposits, withdrawals, etc.)
    
    Args:
        amount (float): Amount to adjust (positive or negative)
        reason (str): Reason for adjustment
        transaction_id (Optional[str]): Transaction identifier
        
    Returns:
        Dict[str, Any]: Adjustment result
    """
    global _capital_manager
    if _capital_manager is None:
        init()
    return _capital_manager.adjust_balance(amount, reason, transaction_id)

def get_wallet_summary() -> Dict[str, Any]:
    """
    Get wallet summary
    
    Returns:
        Dict[str, Any]: Wallet summary
    """
    global _capital_manager
    if _capital_manager is None:
        init()
    return _capital_manager.get_wallet_summary()

def load_wallet_state() -> bool:
    """
    Load wallet state from JSON file
    
    Returns:
        bool: Success status
    """
    global _capital_manager
    if _capital_manager is None:
        init()
    return _capital_manager.load_wallet_state()

def save_wallet_state() -> bool:
    """
    Save wallet state to JSON file
    
    Returns:
        bool: Success status
    """
    global _capital_manager
    if _capital_manager is None:
        init()
    return _capital_manager.save_wallet_state()

# Initialize the capital manager when this module is imported
init()

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="SentientTrader.AI Capital Manager")
    parser.add_argument("--real", action="store_true", help="Use real balance management")
    parser.add_argument("--get-balance", action="store_true", help="Get available balance")
    parser.add_argument("--allocate", nargs=2, metavar=("COIN", "AMOUNT"), help="Allocate funds for a coin")
    parser.add_argument("--release", nargs=3, metavar=("COIN", "AMOUNT", "PNL"), help="Release funds with PnL")
    parser.add_argument("--adjust", nargs=2, metavar=("AMOUNT", "REASON"), help="Adjust balance")
    parser.add_argument("--summary", action="store_true", help="Display wallet summary")
    parser.add_argument("--deposit", type=float, help="Add funds to wallet")
    parser.add_argument("--withdraw", type=float, help="Withdraw funds from wallet")
    return parser.parse_args()

def run_cli() -> int:
    """Run command line interface"""
    args = parse_arguments()
    
    # Initialize with real mode if specified
    if args.real:
        init(real_mode=True)
    
    # Display wallet summary by default
    display_summary = True
    
    # Process commands
    if args.get_balance:
        balance = get_available_balance()
        print(f"{COLORS['bright']}Available Balance: {COLORS['green']}${balance:.2f}{COLORS['reset']}")
    
    elif args.allocate:
        coin, amount = args.allocate
        result = allocate_funds(coin, float(amount))
        
        if result["success"]:
            print(f"{COLORS['green']}✅ Successfully allocated ${amount} for {coin}{COLORS['reset']}")
        else:
            print(f"{COLORS['red']}❌ Failed to allocate funds: {result.get('error', 'Unknown error')}{COLORS['reset']}")
    
    elif args.release:
        coin, amount, pnl = args.release
        result = release_funds(coin, float(amount), float(pnl))
        
        if result["success"]:
            pnl_float = float(pnl)
            pnl_color = COLORS["green"] if pnl_float >= 0 else COLORS["red"]
            pnl_str = f"+{pnl_float:.2f}" if pnl_float >= 0 else f"{pnl_float:.2f}"
            
            print(f"{COLORS['green']}✅ Successfully released ${amount} for {coin} with PnL: {pnl_color}{pnl_str}{COLORS['reset']}")
        else:
            print(f"{COLORS['red']}❌ Failed to release funds: {result.get('error', 'Unknown error')}{COLORS['reset']}")
    
    elif args.adjust:
        amount, reason = args.adjust
        result = adjust_balance(float(amount), reason)
        
        if result["success"]:
            amount_float = float(amount)
            amount_color = COLORS["green"] if amount_float >= 0 else COLORS["red"]
            amount_str = f"+{amount_float:.2f}" if amount_float >= 0 else f"{amount_float:.2f}"
            
            print(f"{COLORS['green']}✅ Successfully adjusted balance by {amount_color}{amount_str}{COLORS['reset']} ({reason})")
        else:
            print(f"{COLORS['red']}❌ Failed to adjust balance: {result.get('error', 'Unknown error')}{COLORS['reset']}")
    
    elif args.deposit:
        result = adjust_balance(args.deposit, "Deposit")
        
        if result["success"]:
            print(f"{COLORS['green']}✅ Successfully deposited ${args.deposit:.2f}{COLORS['reset']}")
        else:
            print(f"{COLORS['red']}❌ Failed to deposit: {result.get('error', 'Unknown error')}{COLORS['reset']}")
    
    elif args.withdraw:
        result = adjust_balance(-args.withdraw, "Withdrawal")
        
        if result["success"]:
            print(f"{COLORS['green']}✅ Successfully withdrew ${args.withdraw:.2f}{COLORS['reset']}")
        else:
            print(f"{COLORS['red']}❌ Failed to withdraw: {result.get('error', 'Unknown error')}{COLORS['reset']}")
    
    # Display summary if requested or if no other command was specified
    if args.summary or display_summary:
        get_wallet_summary()
    
    return 0

if __name__ == "__main__":
    sys.exit(run_cli())
