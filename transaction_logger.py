from random import choice
from random import choice
import random
import random
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SentientTrader.AI - Transaction Logger V2
-----------------------------------------
This module logs all financial transactions in the system,
organized by coin and date in JSONL format, and provides
analysis tools for transaction history.
"""

import os
import sys
import csv
import json
import time
import uuid
import logging
import datetime
import argparse
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from decimal import Decimal, ROUND_DOWN
from threading import Lock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("transaction_logger.log")]
)
logger = logging.getLogger("TransactionLogger")

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
REAL_MODE = False  # Set to True for real transactions, False for simulation
CURRENT_TIME = "2025-04-21 19:27:33"  # UTC
CURRENT_USER = "mstfatiryaki"

# File paths
LOGS_DIRECTORY = "logs"
TRANSACTION_HISTORY_FILE = "transaction_history.jsonl"
EXPORTS_DIRECTORY = "exports"

# Transaction types
TRANSACTION_TYPES = {
    "TRADE_OPEN": "New trade opened",
    "TRADE_CLOSE": "Trade closed",
    "ALLOCATION": "Funds allocated",
    "RELEASE": "Funds released",
    "DEPOSIT": "Deposit to account",
    "WITHDRAWAL": "Withdrawal from account",
    "ADJUSTMENT": "Manual balance adjustment",
    "FEE": "Transaction fee",
    "INTEREST": "Interest earned/paid",
    "TRANSFER": "Internal transfer"
}

# Anomaly detection thresholds
ANOMALY_THRESHOLDS = {
    "large_trade_multiplier": 5.0,  # Multiple of average trade size
    "unusual_time_window_start": 0,  # Hour (0-23)
    "unusual_time_window_end": 5,   # Hour (0-23)
    "max_daily_transaction_count": 50,  # Max transactions per day
    "negative_balance_threshold": -0.01,  # Negative balance threshold (to handle rounding)
    "max_trade_amount_usdt": 10000.0  # Maximum single trade amount
}

class TransactionLogger:
    """Manages logging of all financial transactions in the system"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, real_mode: bool = REAL_MODE):
        """Singleton pattern to ensure only one TransactionLogger instance"""
        if cls._instance is None:
            cls._instance = super(TransactionLogger, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, real_mode: bool = REAL_MODE):
        """
        Initialize the transaction logger
        
        Args:
            real_mode (bool): Whether logging real or simulated transactions
        """
        # Only initialize once due to singleton pattern
        if TransactionLogger._initialized:
            return
            
        self.real_mode = real_mode
        self.log_file_locks = {}  # Locks for each log file
        self.history_file_lock = Lock()  # Lock for the main history file
        
        # Transaction cache for anomaly detection
        self.transaction_cache = {
            "recent_transactions": [],  # Last N transactions for analysis
            "daily_counts": {},         # Daily transaction counts
            "coin_averages": {}         # Average trade size by coin
        }
        
        # Ensure log directories exist
        self._ensure_directories()
        
        TransactionLogger._initialized = True
        logger.info(f"Transaction Logger initialized (REAL_MODE: {self.real_mode})")
    
    def _ensure_directories(self) -> None:
        """Ensure log and export directories exist"""
        try:
            # Create logs directory if it doesn't exist
            if not os.path.exists(LOGS_DIRECTORY):
                os.makedirs(LOGS_DIRECTORY)
                logger.info(f"Created logs directory: {LOGS_DIRECTORY}")
            
            # Create exports directory if it doesn't exist
            if not os.path.exists(EXPORTS_DIRECTORY):
                os.makedirs(EXPORTS_DIRECTORY)
                logger.info(f"Created exports directory: {EXPORTS_DIRECTORY}")
                
        except Exception as e:
            logger.error(f"Error creating directories: {str(e)}")
    
    def log_transaction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Log a new transaction
        
        Args:
            data (Dict[str, Any]): Transaction data
            
        Returns:
            Dict[str, Any]: Processed transaction with added metadata
        """
        try:
            # Validate transaction data
            if not self._validate_transaction_data(data):
                error_msg = "Invalid transaction data"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
            
            # Ensure required fields
            transaction = data.copy()  # Create a copy to avoid modifying original
            
            # Add metadata if not present
            if "timestamp" not in transaction:
                transaction["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            if "transaction_id" not in transaction:
                transaction["transaction_id"] = f"tx_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            
            if "user" not in transaction:
                transaction["user"] = CURRENT_USER
            
            if "real_mode" not in transaction:
                transaction["real_mode"] = self.real_mode
            
            # Extract coin and date for file naming
            coin = transaction.get("coin", transaction.get("symbol", "UNKNOWN")).upper()
            transaction_date = datetime.datetime.strptime(
                transaction["timestamp"], 
                "%Y-%m-%d %H:%M:%S"
            ).strftime("%Y-%m-%d")
            
            # Log to coin-specific file
            coin_log_file = os.path.join(LOGS_DIRECTORY, f"{coin}_{transaction_date}.log")
            self._append_to_log_file(coin_log_file, transaction)
            
            # Log to main history file
            self._append_to_history_file(transaction)
            
            # Update transaction cache for anomaly detection
            self._update_transaction_cache(transaction)
            
            # Check for anomalies
            anomalies = self._check_transaction_anomalies(transaction)
            if anomalies:
                transaction["anomalies"] = anomalies
                self._handle_anomalies(transaction, anomalies)
            
            logger.info(f"Transaction logged: {transaction['transaction_id']} | {coin} | {transaction.get('type', 'UNKNOWN')}")
            return {"success": True, "transaction": transaction}
            
        except Exception as e:
            error_msg = f"Error logging transaction: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def _validate_transaction_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate transaction data
        
        Args:
            data (Dict[str, Any]): Transaction data
            
        Returns:
            bool: Validation result
        """
        # Check if data is a dictionary
        if not isinstance(data, dict):
            return False
        
        # Required fields
        required_fields = ["type", "amount"]
        
        # Check required fields
        if not all(field in data for field in required_fields):
            return False
        
        # Check if transaction type is valid
        if data.get("type") not in TRANSACTION_TYPES:
            return False
        
        # Check if amount is numeric
        try:
            float(data.get("amount", 0))
        except (ValueError, TypeError):
            return False
        
        # Either coin or symbol must be present
        if "coin" not in data and "symbol" not in data:
            return False
        
        return True
    
    def _get_file_lock(self, file_path: str) -> Lock:
        """
        Get or create a lock for a specific file
        
        Args:
            file_path (str): Path to file
            
        Returns:
            Lock: Thread lock for file
        """
        if file_path not in self.log_file_locks:
            self.log_file_locks[file_path] = Lock()
        return self.log_file_locks[file_path]
    
    def _append_to_log_file(self, file_path: str, data: Dict[str, Any]) -> None:
        """
        Append transaction to log file with thread safety
        
        Args:
            file_path (str): Path to log file
            data (Dict[str, Any]): Transaction data
        """
        lock = self._get_file_lock(file_path)
        
        try:
            with lock:
                # Ensure directory exists
                directory = os.path.dirname(file_path)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory)
                
                # Append to file
                with open(file_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
                    
        except Exception as e:
            logger.error(f"Error appending to log file {file_path}: {str(e)}")
    
    def _append_to_history_file(self, data: Dict[str, Any]) -> None:
        """
        Append transaction to main history file with thread safety
        
        Args:
            data (Dict[str, Any]): Transaction data
        """
        try:
            with self.history_file_lock:
                with open(TRANSACTION_HISTORY_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
                    
        except Exception as e:
            logger.error(f"Error appending to history file: {str(e)}")
    
    def _update_transaction_cache(self, transaction: Dict[str, Any]) -> None:
        """
        Update transaction cache for anomaly detection
        
        Args:
            transaction (Dict[str, Any]): Transaction data
        """
        try:
            # Add to recent transactions (keep last 100)
            self.transaction_cache["recent_transactions"].append(transaction)
            if len(self.transaction_cache["recent_transactions"]) > 100:
                self.transaction_cache["recent_transactions"].pop(0)
            
            # Update daily counts
            transaction_date = transaction["timestamp"].split(" ")[0]
            if transaction_date not in self.transaction_cache["daily_counts"]:
                self.transaction_cache["daily_counts"][transaction_date] = 0
            self.transaction_cache["daily_counts"][transaction_date] += 1
            
            # Update coin averages for trade transactions
            if transaction["type"] in ["TRADE_OPEN", "TRADE_CLOSE"]:
                coin = transaction.get("coin", transaction.get("symbol", "UNKNOWN")).upper()
                amount = float(transaction["amount"])
                
                if coin not in self.transaction_cache["coin_averages"]:
                    self.transaction_cache["coin_averages"][coin] = {
                        "sum": amount,
                        "count": 1,
                        "average": amount
                    }
                else:
                    coin_data = self.transaction_cache["coin_averages"][coin]
                    coin_data["sum"] += amount
                    coin_data["count"] += 1
                    coin_data["average"] = coin_data["sum"] / coin_data["count"]
                    
        except Exception as e:
            logger.error(f"Error updating transaction cache: {str(e)}")
    
    def _check_transaction_anomalies(self, transaction: Dict[str, Any]) -> List[str]:
        """
        Check transaction for anomalies
        
        Args:
            transaction (Dict[str, Any]): Transaction data
            
        Returns:
            List[str]: Detected anomalies
        """
        anomalies = []
        
        try:
            # Extract basic data
            coin = transaction.get("coin", transaction.get("symbol", "UNKNOWN")).upper()
            amount = float(transaction["amount"])
            transaction_type = transaction["type"]
            timestamp = transaction["timestamp"]
            
            # 1. Check for unusually large transactions
            if coin in self.transaction_cache["coin_averages"]:
                avg_amount = self.transaction_cache["coin_averages"][coin]["average"]
                large_threshold = avg_amount * ANOMALY_THRESHOLDS["large_trade_multiplier"]
                
                if amount > large_threshold and amount > 100:  # Min $100 to avoid flagging small trades
                    anomalies.append(f"large_transaction:{amount:.2f}>{large_threshold:.2f}")
            
            # 2. Check for transactions at unusual hours
            try:
                tx_hour = int(timestamp.split(" ")[1].split(":")[0])
                if ANOMALY_THRESHOLDS["unusual_time_window_start"] <= tx_hour <= ANOMALY_THRESHOLDS["unusual_time_window_end"]:
                    anomalies.append(f"unusual_hour:{tx_hour}")
            except:
                pass
            
            # 3. Check for high transaction frequency
            transaction_date = timestamp.split(" ")[0]
            if transaction_date in self.transaction_cache["daily_counts"]:
                daily_count = self.transaction_cache["daily_counts"][transaction_date]
                if daily_count > ANOMALY_THRESHOLDS["max_daily_transaction_count"]:
                    anomalies.append(f"high_frequency:{daily_count}")
            
            # 4. Check for very large trades
            if amount > ANOMALY_THRESHOLDS["max_trade_amount_usdt"] and transaction_type in ["TRADE_OPEN", "TRADE_CLOSE"]:
                anomalies.append(f"excessive_amount:{amount:.2f}")
            
            # 5. Check for negative balance after transaction
            if "balance_after" in transaction and transaction["balance_after"] < ANOMALY_THRESHOLDS["negative_balance_threshold"]:
                anomalies.append(f"negative_balance:{transaction['balance_after']:.2f}")
                
        except Exception as e:
            logger.error(f"Error checking for anomalies: {str(e)}")
        
        return anomalies
    
    def _handle_anomalies(self, transaction: Dict[str, Any], anomalies: List[str]) -> None:
        """
        Handle detected anomalies
        
        Args:
            transaction (Dict[str, Any]): Transaction data
            anomalies (List[str]): Detected anomalies
        """
        if not ALERT_SYSTEM_AVAILABLE or not anomalies:
            return
        
        try:
            # Extract basic info
            coin = transaction.get("coin", transaction.get("symbol", "UNKNOWN")).upper()
            transaction_type = transaction["type"]
            amount = float(transaction["amount"])
            
            # Determine alert level based on anomalies
            critical_anomalies = ["negative_balance", "excessive_amount"]
            has_critical = any(any(a.startswith(ca) for ca in critical_anomalies) for a in anomalies)
            
            alert_level = "error" if has_critical else "warning"
            alert_function = alert_system.error if has_critical else alert_system.warning
            
            # Format anomaly messages
            anomaly_messages = []
            
            for anomaly in anomalies:
                if anomaly.startswith("large_transaction:"):
                    _, value = anomaly.split(":")
                    anomaly_messages.append(f"Unusually large transaction (${value})")
                
                elif anomaly.startswith("unusual_hour:"):
                    _, hour = anomaly.split(":")
                    anomaly_messages.append(f"Transaction at unusual hour ({hour}:00)")
                
                elif anomaly.startswith("high_frequency:"):
                    _, count = anomaly.split(":")
                    anomaly_messages.append(f"High transaction frequency ({count} today)")
                
                elif anomaly.startswith("excessive_amount:"):
                    _, value = anomaly.split(":")
                    anomaly_messages.append(f"Excessive transaction amount (${value})")
                
                elif anomaly.startswith("negative_balance:"):
                    _, value = anomaly.split(":")
                    anomaly_messages.append(f"Negative balance after transaction (${value})")
                
                else:
                    anomaly_messages.append(f"Unknown anomaly: {anomaly}")
            
            # Send alert
            alert_message = f"Transaction anomalies detected for {coin} {transaction_type}"
            alert_data = {
                "transaction_id": transaction["transaction_id"],
                "coin": coin,
                "type": transaction_type,
                "amount": amount,
                "anomalies": anomaly_messages
            }
            
            alert_function(alert_message, alert_data, "transaction_logger")
            
        except Exception as e:
            logger.error(f"Error handling anomalies: {str(e)}")
    
    def load_logs(self, coin: Optional[str] = None, date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load transaction logs for specific coin and/or date
        
        Args:
            coin (Optional[str]): Coin symbol (or None for all coins)
            date (Optional[str]): Date in YYYY-MM-DD format (or None for all dates)
            
        Returns:
            List[Dict[str, Any]]: List of transactions
        """
        try:
            # If both coin and date are specified, load specific log file
            if coin and date:
                coin = coin.upper()
                log_file = os.path.join(LOGS_DIRECTORY, f"{coin}_{date}.log")
                
                if os.path.exists(log_file):
                    return self._load_log_file(log_file)
                else:
                    logger.info(f"No log file found for {coin} on {date}")
                    return []
            
            # If only date is specified, load all coin logs for that date
            elif date:
                transactions = []
                date_pattern = f"_{date}.log"
                
                for filename in os.listdir(LOGS_DIRECTORY):
                    if date_pattern in filename:
                        log_file = os.path.join(LOGS_DIRECTORY, filename)
                        transactions.extend(self._load_log_file(log_file))
                
                return transactions
            
            # If only coin is specified, load all logs for that coin
            elif coin:
                coin = coin.upper()
                transactions = []
                coin_pattern = f"{coin}_"
                
                for filename in os.listdir(LOGS_DIRECTORY):
                    if filename.startswith(coin_pattern):
                        log_file = os.path.join(LOGS_DIRECTORY, filename)
                        transactions.extend(self._load_log_file(log_file))
                
                return transactions
            
            # If neither is specified, load from main history file
            else:
                if os.path.exists(TRANSACTION_HISTORY_FILE):
                    return self._load_log_file(TRANSACTION_HISTORY_FILE)
                else:
                    logger.info(f"No transaction history file found")
                    return []
                
        except Exception as e:
            logger.error(f"Error loading logs: {str(e)}")
            return []
    
    def _load_log_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load transactions from a log file
        
        Args:
            file_path (str): Path to log file
            
        Returns:
            List[Dict[str, Any]]: List of transactions
        """
        transactions = []
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            transaction = json.loads(line)
                            transactions.append(transaction)
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON in {file_path}: {line}")
            
            return transactions
            
        except Exception as e:
            logger.error(f"Error loading log file {file_path}: {str(e)}")
            return []
    
    def summarize_logs(self, coin: Optional[str] = None, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a summary of logs
        
        Args:
            coin (Optional[str]): Coin symbol (or None for all coins)
            date (Optional[str]): Date in YYYY-MM-DD format (or None for all dates)
            
        Returns:
            Dict[str, Any]: Log summary
        """
        try:
            # Load relevant logs
            transactions = self.load_logs(coin, date)
            
            if not transactions:
                summary_type = []
                if coin:
                    summary_type.append(f"coin={coin}")
                if date:
                    summary_type.append(f"date={date}")
                
                summary_desc = " and ".join(summary_type) if summary_type else "all transactions"
                logger.info(f"No transactions found for {summary_desc}")
                
                return {
                    "total_transactions": 0,
                    "message": f"No transactions found for {summary_desc}"
                }
            
            # Initialize summary
            summary = {
                "total_transactions": len(transactions),
                "transaction_types": {},
                "coins": {},
                "dates": {},
                "total_volume": 0.0,
                "largest_transaction": {
                    "amount": 0.0,
                    "transaction_id": "",
                    "type": "",
                    "timestamp": ""
                },
                "time_distribution": {
                    "by_hour": {},
                    "by_day_of_week": {}
                }
            }
            
            # Process each transaction
            for tx in transactions:
                # Skip invalid transactions
                if not isinstance(tx, dict):
                    continue
                
                # Extract data
                tx_type = tx.get("type", "UNKNOWN")
                tx_coin = tx.get("coin", tx.get("symbol", "UNKNOWN")).upper()
                tx_amount = float(tx.get("amount", 0))
                tx_timestamp = tx.get("timestamp", "")
                
                # Update transaction types count
                if tx_type not in summary["transaction_types"]:
                    summary["transaction_types"][tx_type] = 0
                summary["transaction_types"][tx_type] += 1
                
                # Update coins stats
                if tx_coin not in summary["coins"]:
                    summary["coins"][tx_coin] = {
                        "count": 0,
                        "volume": 0.0
                    }
                summary["coins"][tx_coin]["count"] += 1
                summary["coins"][tx_coin]["volume"] += tx_amount
                
                # Update dates stats
                if tx_timestamp:
                    tx_date = tx_timestamp.split(" ")[0]
                    if tx_date not in summary["dates"]:
                        summary["dates"][tx_date] = 0
                    summary["dates"][tx_date] += 1
                
                # Update total volume
                summary["total_volume"] += tx_amount
                
                # Check for largest transaction
                if tx_amount > summary["largest_transaction"]["amount"]:
                    summary["largest_transaction"] = {
                        "amount": tx_amount,
                        "transaction_id": tx.get("transaction_id", ""),
                        "type": tx_type,
                        "timestamp": tx_timestamp,
                        "coin": tx_coin
                    }
                
                # Update time distribution
                if tx_timestamp:
                    try:
                        dt = datetime.datetime.strptime(tx_timestamp, "%Y-%m-%d %H:%M:%S")
                        
                        # By hour
                        hour = dt.hour
                        if hour not in summary["time_distribution"]["by_hour"]:
                            summary["time_distribution"]["by_hour"][hour] = 0
                        summary["time_distribution"]["by_hour"][hour] += 1
                        
                        # By day of week
                        day_of_week = dt.strftime("%A")
                        if day_of_week not in summary["time_distribution"]["by_day_of_week"]:
                            summary["time_distribution"]["by_day_of_week"][day_of_week] = 0
                        summary["time_distribution"]["by_day_of_week"][day_of_week] += 1
                    except:
                        pass
            
            # Calculate average transaction volume
            summary["average_transaction"] = summary["total_volume"] / len(transactions) if transactions else 0
            
            # Sort coins by volume
            sorted_coins = sorted(
                [(coin, data["volume"]) for coin, data in summary["coins"].items()],
                key=lambda x: x[1],
                reverse=True
            )
            summary["top_coins_by_volume"] = [{"coin": coin, "volume": volume} for coin, volume in sorted_coins[:5]]
            
            # Sort transaction days by count
            sorted_dates = sorted(
                [(date, count) for date, count in summary["dates"].items()],
                key=lambda x: x[1],
                reverse=True
            )
            summary["top_active_days"] = [{"date": date, "count": count} for date, count in sorted_dates[:5]]
            
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing logs: {str(e)}")
            return {"error": str(e)}
    
    def detect_anomalies(self) -> Dict[str, Any]:
        """
        Detect anomalies in transaction history
        
        Returns:
            Dict[str, Any]: Detected anomalies
        """
        try:
            # Load all transactions
            transactions = self.load_logs()
            
            if not transactions:
                logger.info("No transactions found for anomaly detection")
                return {"anomalies_found": False, "message": "No transactions found"}
            
            # Initialize anomaly report
            anomaly_report = {
                "anomalies_found": False,
                "total_transactions": len(transactions),
                "anomalies": {
                    "large_transactions": [],
                    "unusual_timing": [],
                    "high_frequency_days": [],
                    "excessive_amounts": [],
                    "negative_balances": []
                }
            }
            
            # Process transactions by date and coin
            by_date = {}
            by_coin = {}
            
            for tx in transactions:
                # Skip invalid transactions
                if not isinstance(tx, dict):
                    continue
                
                # Extract data
                tx_type = tx.get("type", "UNKNOWN")
                tx_coin = tx.get("coin", tx.get("symbol", "UNKNOWN")).upper()
                tx_amount = float(tx.get("amount", 0))
                tx_timestamp = tx.get("timestamp", "")
                tx_id = tx.get("transaction_id", "")
                
                # Group by date
                if tx_timestamp:
                    tx_date = tx_timestamp.split(" ")[0]
                    if tx_date not in by_date:
                        by_date[tx_date] = []
                    by_date[tx_date].append(tx)
                
                # Group by coin
                if tx_coin not in by_coin:
                    by_coin[tx_coin] = []
                by_coin[tx_coin].append(tx)
                
                # Check for negative balance
                if "balance_after" in tx and tx["balance_after"] < ANOMALY_THRESHOLDS["negative_balance_threshold"]:
                    anomaly_report["anomalies"]["negative_balances"].append({
                        "transaction_id": tx_id,
                        "timestamp": tx_timestamp,
                        "type": tx_type,
                        "coin": tx_coin,
                        "balance": tx["balance_after"]
                    })
                    anomaly_report["anomalies_found"] = True
                
                # Check for excessive amounts
                if tx_amount > ANOMALY_THRESHOLDS["max_trade_amount_usdt"] and tx_type in ["TRADE_OPEN", "TRADE_CLOSE"]:
                    anomaly_report["anomalies"]["excessive_amounts"].append({
                        "transaction_id": tx_id,
                        "timestamp": tx_timestamp,
                        "type": tx_type,
                        "coin": tx_coin,
                        "amount": tx_amount
                    })
                    anomaly_report["anomalies_found"] = True
                
                # Check for unusual timing
                if tx_timestamp:
                    try:
                        tx_hour = int(tx_timestamp.split(" ")[1].split(":")[0])
                        if ANOMALY_THRESHOLDS["unusual_time_window_start"] <= tx_hour <= ANOMALY_THRESHOLDS["unusual_time_window_end"]:
                            anomaly_report["anomalies"]["unusual_timing"].append({
                                "transaction_id": tx_id,
                                "timestamp": tx_timestamp,
                                "type": tx_type,
                                "coin": tx_coin,
                                "hour": tx_hour
                            })
                            anomaly_report["anomalies_found"] = True
                    except:
                        pass
            
            # Check for high frequency days
            for date, date_txs in by_date.items():
                if len(date_txs) > ANOMALY_THRESHOLDS["max_daily_transaction_count"]:
                    anomaly_report["anomalies"]["high_frequency_days"].append({
                        "date": date,
                        "count": len(date_txs)
                    })
                    anomaly_report["anomalies_found"] = True
            
            # Calculate average transaction size by coin and check for large transactions
            for coin, coin_txs in by_coin.items():
                # Only consider coins with enough transactions
                if len(coin_txs) >= 5:
                    trade_txs = [tx for tx in coin_txs if tx.get("type") in ["TRADE_OPEN", "TRADE_CLOSE"]]
                    if trade_txs:
                        amounts = [float(tx.get("amount", 0)) for tx in trade_txs]
                        avg_amount = sum(amounts) / len(amounts)
                        
                        # Check for unusually large transactions
                        large_threshold = avg_amount * ANOMALY_THRESHOLDS["large_trade_multiplier"]
                        large_txs = [tx for tx in trade_txs if float(tx.get("amount", 0)) > large_threshold and float(tx.get("amount", 0)) > 100]
                        
                        for tx in large_txs:
                            anomaly_report["anomalies"]["large_transactions"].append({
                                "transaction_id": tx.get("transaction_id", ""),
                                "timestamp": tx.get("timestamp", ""),
                                "type": tx.get("type", ""),
                                "coin": coin,
                                "amount": float(tx.get("amount", 0)),
                                "average": avg_amount,
                                "threshold": large_threshold
                            })
                            anomaly_report["anomalies_found"] = True
            
            return anomaly_report
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return {"error": str(e)}
    
    def export_logs_to_csv(self, coin: Optional[str] = None, date: Optional[str] = None) -> str:
        """
        Export logs to CSV format
        
        Args:
            coin (Optional[str]): Coin symbol (or None for all coins)
            date (Optional[str]): Date in YYYY-MM-DD format (or None for all dates)
            
        Returns:
            str: Path to exported CSV file
        """
        try:
            # Load relevant logs
            transactions = self.load_logs(coin, date)
            
            if not transactions:
                summary_type = []
                if coin:
                    summary_type.append(f"coin={coin}")
                if date:
                    summary_type.append(f"date={date}")
                
                summary_desc = " and ".join(summary_type) if summary_type else "all transactions"
                logger.info(f"No transactions found for {summary_desc}")
                return ""
            
            # Create export filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_parts = ["transactions"]
            
            if coin:
                filename_parts.append(coin.upper())
            
            if date:
                filename_parts.append(date)
            
            filename_parts.append(timestamp)
            export_filename = os.path.join(EXPORTS_DIRECTORY, f"{'_'.join(filename_parts)}.csv")
            
            # Define fields to export
            fields = [
                "transaction_id", "timestamp", "type", "coin", "amount", 
                "user", "real_mode", "source_module", "balance_after"
            ]
            
            # Ensure exports directory exists
            if not os.path.exists(EXPORTS_DIRECTORY):
                os.makedirs(EXPORTS_DIRECTORY)
            
            # Write to CSV
            with open(export_filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                
                for tx in transactions:
                    # Ensure coin field exists
                    if "coin" not in tx and "symbol" in tx:
                        tx["coin"] = tx["symbol"]
                    
                    writer.writerow(tx)
            
            logger.info(f"Exported {len(transactions)} transactions to {export_filename}")
            return export_filename
            
        except Exception as e:
            logger.error(f"Error exporting logs to CSV: {str(e)}")
            return ""
    
    def display_summary(self, summary: Dict[str, Any]) -> None:
        """
        Display transaction summary in terminal
        
        Args:
            summary (Dict[str, Any]): Transaction summary
        """
        try:
            # Display header
            print(f"\n{COLORS['bright']}{COLORS['cyan']}{'=' * 60}{COLORS['reset']}")
            print(f"{COLORS['bright']}{COLORS['cyan']}📊 TRANSACTION SUMMARY{COLORS['reset']}")
            print(f"{COLORS['bright']}{COLORS['cyan']}{'=' * 60}{COLORS['reset']}")
            
            # Basic summary
            total_tx = summary.get("total_transactions", 0)
            total_volume = summary.get("total_volume", 0)
            avg_tx = summary.get("average_transaction", 0)
            
            print(f"{COLORS['bright']}Total Transactions: {COLORS['green']}{total_tx}{COLORS['reset']}")
            print(f"{COLORS['white']}Total Volume: {COLORS['green']}${total_volume:.2f}{COLORS['reset']}")
            print(f"{COLORS['white']}Average Transaction: {COLORS['green']}${avg_tx:.2f}{COLORS['reset']}")
            
            # Largest transaction
            largest = summary.get("largest_transaction", {})
            if largest:
                print(f"\n{COLORS['bright']}Largest Transaction:{COLORS['reset']}")
                print(f"  {COLORS['white']}Amount: {COLORS['green']}${largest.get('amount', 0):.2f}{COLORS['reset']}")
                print(f"  {COLORS['white']}Coin: {largest.get('coin', 'UNKNOWN')}{COLORS['reset']}")
                print(f"  {COLORS['white']}Type: {largest.get('type', 'UNKNOWN')}{COLORS['reset']}")
                print(f"  {COLORS['white']}Time: {largest.get('timestamp', '')}{COLORS['reset']}")
            
            # Top coins by volume
            top_coins = summary.get("top_coins_by_volume", [])
            if top_coins:
                print(f"\n{COLORS['bright']}Top Coins by Volume:{COLORS['reset']}")
                for i, item in enumerate(top_coins, 1):
                    print(f"  {i}. {COLORS['bright']}{item.get('coin')}{COLORS['reset']}: ${item.get('volume', 0):.2f}")
            
            # Transaction types
            tx_types = summary.get("transaction_types", {})
            if tx_types:
                print(f"\n{COLORS['bright']}Transaction Types:{COLORS['reset']}")
                for tx_type, count in sorted(tx_types.items(), key=lambda x: x[1], reverse=True):
                    tx_desc = TRANSACTION_TYPES.get(tx_type, tx_type)
                    print(f"  {COLORS['white']}{tx_desc}: {count}{COLORS['reset']}")
            
            # Most active days
            active_days = summary.get("top_active_days", [])
            if active_days:
                print(f"\n{COLORS['bright']}Most Active Days:{COLORS['reset']}")
                for i, item in enumerate(active_days, 1):
                    print(f"  {i}. {COLORS['white']}{item.get('date')}: {item.get('count')} transactions{COLORS['reset']}")
            
            # Time distribution
            time_dist = summary.get("time_distribution", {})
            if "by_hour" in time_dist and time_dist["by_hour"]:
                # Find the most active hour
                most_active_hour, most_active_count = max(time_dist["by_hour"].items(), key=lambda x: x[1])
                
                print(f"\n{COLORS['bright']}Most Active Hour: {COLORS['white']}{most_active_hour}:00 ({most_active_count} transactions){COLORS['reset']}")
            
            if "by_day_of_week" in time_dist and time_dist["by_day_of_week"]:
                # Find the most active day of week
                most_active_day, most_active_count = max(time_dist["by_day_of_week"].items(), key=lambda x: x[1])
                
                print(f"{COLORS['bright']}Most Active Day: {COLORS['white']}{most_active_day} ({most_active_count} transactions){COLORS['reset']}")
            
            print(f"{COLORS['bright']}{COLORS['cyan']}{'=' * 60}{COLORS['reset']}\n")
            
        except Exception as e:
            logger.error(f"Error displaying summary: {str(e)}")
    
    def display_anomalies(self, anomaly_report: Dict[str, Any]) -> None:
        """
        Display anomaly report in terminal
        
        Args:
            anomaly_report (Dict[str, Any]): Anomaly report
        """
        try:
            # Display header
            print(f"\n{COLORS['bright']}{COLORS['yellow']}{'=' * 60}{COLORS['reset']}")
            print(f"{COLORS['bright']}{COLORS['yellow']}⚠️ TRANSACTION ANOMALIES{COLORS['reset']}")
            print(f"{COLORS['bright']}{COLORS['yellow']}{'=' * 60}{COLORS['reset']}")
            
            # Check if anomalies were found
            if not anomaly_report.get("anomalies_found", False):
                print(f"{COLORS['green']}✓ No anomalies detected in {anomaly_report.get('total_transactions', 0)} transactions{COLORS['reset']}")
                print(f"{COLORS['bright']}{COLORS['yellow']}{'=' * 60}{COLORS['reset']}\n")
                return
            
            # Display anomalies
            anomalies = anomaly_report.get("anomalies", {})
            
            # 1. Negative balances (critical)
            negative_balances = anomalies.get("negative_balances", [])
            if negative_balances:
                print(f"{COLORS['bright']}{COLORS['red']}🚨 Negative Balances Detected:{COLORS['reset']}")
                for item in negative_balances:
                    print(f"  {COLORS['red']}• Balance: ${item.get('balance', 0):.2f} | {item.get('coin')} | {item.get('timestamp', '')}{COLORS['reset']}")
            
            # 2. Excessive amounts
            excessive = anomalies.get("excessive_amounts", [])
            if excessive:
                print(f"\n{COLORS['bright']}{COLORS['red']}💰 Excessive Transaction Amounts:{COLORS['reset']}")
                for item in excessive:
                    print(f"  {COLORS['red']}• ${item.get('amount', 0):.2f} | {item.get('coin')} | {item.get('timestamp', '')}{COLORS['reset']}")
            
            # 3. Large transactions
            large_txs = anomalies.get("large_transactions", [])
            if large_txs:
                print(f"\n{COLORS['bright']}{COLORS['yellow']}📈 Unusually Large Transactions:{COLORS['reset']}")
                for item in large_txs:
                    print(f"  {COLORS['yellow']}• ${item.get('amount', 0):.2f} | {item.get('coin')} | {item.get('timestamp', '')}{COLORS['reset']}")
                    print(f"    (Avg: ${item.get('average', 0):.2f}, Threshold: ${item.get('threshold', 0):.2f})")
            
            # 4. Unusual timing
            unusual_timing = anomalies.get("unusual_timing", [])
            if unusual_timing:
                print(f"\n{COLORS['bright']}{COLORS['yellow']}🕒 Unusual Transaction Timing:{COLORS['reset']}")
                for item in unusual_timing:
                    print(f"  {COLORS['yellow']}• {item.get('hour', 0)}:00 | {item.get('coin')} | {item.get('timestamp', '')}{COLORS['reset']}")
            
            # 5. High frequency days
            high_freq = anomalies.get("high_frequency_days", [])
            if high_freq:
                print(f"\n{COLORS['bright']}{COLORS['yellow']}🔄 High Transaction Frequency Days:{COLORS['reset']}")
                for item in high_freq:
                    print(f"  {COLORS['yellow']}• {item.get('date', '')} | {item.get('count', 0)} transactions{COLORS['reset']}")
            
            print(f"{COLORS['bright']}{COLORS['yellow']}{'=' * 60}{COLORS['reset']}\n")
            
        except Exception as e:
            logger.error(f"Error displaying anomalies: {str(e)}")

# Singleton instance
_transaction_logger = None

def init(real_mode: bool = REAL_MODE) -> None:
    """
    Initialize the transaction logger
    
    Args:
        real_mode (bool): Whether logging real or simulated transactions
    """
    global _transaction_logger
    _transaction_logger = TransactionLogger(real_mode=real_mode)

def log_transaction(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Log a new transaction
    
    Args:
        data (Dict[str, Any]): Transaction data
        
    Returns:
        Dict[str, Any]: Processed transaction with added metadata
    """
    global _transaction_logger
    if _transaction_logger is None:
        init()
    return _transaction_logger.log_transaction(data)

def load_logs(coin: Optional[str] = None, date: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load transaction logs
    
    Args:
        coin (Optional[str]): Coin symbol (or None for all coins)
        date (Optional[str]): Date in YYYY-MM-DD format (or None for all dates)
        
    Returns:
        List[Dict[str, Any]]: List of transactions
    """
    global _transaction_logger
    if _transaction_logger is None:
        init()
    return _transaction_logger.load_logs(coin, date)

def summarize_logs(coin: Optional[str] = None, date: Optional[str] = None) -> Dict[str, Any]:
    """
    Create summary of logs
    
    Args:
        coin (Optional[str]): Coin symbol (or None for all coins)
        date (Optional[str]): Date in YYYY-MM-DD format (or None for all dates)
        
    Returns:
        Dict[str, Any]: Log summary
    """
    global _transaction_logger
    if _transaction_logger is None:
        init()
    return _transaction_logger.summarize_logs(coin, date)

def detect_anomalies() -> Dict[str, Any]:
    """
    Detect anomalies in transaction history
    
    Returns:
        Dict[str, Any]: Detected anomalies
    """
    global _transaction_logger
    if _transaction_logger is None:
        init()
    return _transaction_logger.detect_anomalies()

def export_logs_to_csv(coin: Optional[str] = None, date: Optional[str] = None) -> str:
    """
    Export logs to CSV
    
    Args:
        coin (Optional[str]): Coin symbol (or None for all coins)
        date (Optional[str]): Date in YYYY-MM-DD format (or None for all dates)
        
    Returns:
        str: Path to exported CSV file
    """
    global _transaction_logger
    if _transaction_logger is None:
        init()
    return _transaction_logger.export_logs_to_csv(coin, date)

def display_summary(coin: Optional[str] = None, date: Optional[str] = None) -> None:
    """
    Display transaction summary in terminal
    
    Args:
        coin (Optional[str]): Coin symbol (or None for all coins)
        date (Optional[str]): Date in YYYY-MM-DD format (or None for all dates)
    """
    global _transaction_logger
    if _transaction_logger is None:
        init()
    summary = _transaction_logger.summarize_logs(coin, date)
    _transaction_logger.display_summary(summary)

def display_anomalies() -> None:
    """Display anomaly report in terminal"""
    global _transaction_logger
    if _transaction_logger is None:
        init()
    anomaly_report = _transaction_logger.detect_anomalies()
    _transaction_logger.display_anomalies(anomaly_report)

# Initialize the transaction logger when this module is imported
init()

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="SentientTrader.AI Transaction Logger")
    parser.add_argument("--real", action="store_true", help="Use real transaction mode")
    parser.add_argument("--log", action="store_true", help="Log a sample transaction")
    parser.add_argument("--summary", action="store_true", help="Display transaction summary")
    parser.add_argument("--coin", type=str, help="Filter by coin")
    parser.add_argument("--date", type=str, help="Filter by date (YYYY-MM-DD)")
    parser.add_argument("--anomalies", action="store_true", help="Detect and display anomalies")
    parser.add_argument("--export", action="store_true", help="Export logs to CSV")
    return parser.parse_args()

def create_sample_transaction() -> Dict[str, Any]:
    """Create a sample transaction for testing"""
    import random
    
    coins = ["BTC", "ETH", "SOL", "LINK", "ADA", "DOT", "AVAX"]
    tx_types = list(TRANSACTION_TYPES.keys())
    
    coin = random.choice(coins)
    tx_type = random.choice(tx_types)
    
    # Generate amount based on transaction type
    if tx_type in ["TRADE_OPEN", "TRADE_CLOSE"]:
        amount = round(random.uniform(100, 1000), 2)
    elif tx_type in ["ALLOCATION", "RELEASE"]:
        amount = round(random.uniform(500, 2000), 2)
    elif tx_type in ["DEPOSIT", "WITHDRAWAL"]:
        amount = round(random.uniform(1000, 5000), 2)
    else:
        amount = round(random.uniform(10, 100), 2)
    
    # Create transaction data
    transaction = {
        "type": tx_type,
        "coin": coin,
        "amount": amount,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user": CURRENT_USER,
        "source_module": random.choice(["trade_executor", "capital_manager", "risk_manager"]),
        "balance_after": round(random.uniform(5000, 15000), 2)
    }
    
    return transaction

def main() -> int:
    """Run from command line"""
    args = parse_arguments()
    
    # Initialize with real mode if specified
    if args.real:
        init(real_mode=True)
    
    # Process commands
    if args.log:
        # Create and log a sample transaction
        sample_tx = create_sample_transaction()
        result = log_transaction(sample_tx)
        
        if result["success"]:
            print(f"{COLORS['green']}✅ Transaction logged successfully{COLORS['reset']}")
            print(f"Transaction ID: {result['transaction']['transaction_id']}")
            print(f"Coin: {result['transaction']['coin']}")
            print(f"Type: {result['transaction']['type']}")
            print(f"Amount: ${result['transaction']['amount']}")
        else:
            print(f"{COLORS['red']}❌ Failed to log transaction: {result.get('error', 'Unknown error')}{COLORS['reset']}")
    
    if args.summary:
        # Display transaction summary
        display_summary(args.coin, args.date)
    
    if args.anomalies:
        # Detect and display anomalies
        display_anomalies()
    
    if args.export:
        # Export logs to CSV
        csv_path = export_logs_to_csv(args.coin, args.date)
        
        if csv_path:
            print(f"{COLORS['green']}✅ Logs exported to: {csv_path}{COLORS['reset']}")
        else:
            print(f"{COLORS['red']}❌ Failed to export logs{COLORS['reset']}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
