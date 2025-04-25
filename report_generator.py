from random import choice
import random
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SentientTrader.AI - Report Generator V2
---------------------------------------
Generates detailed performance reports and analytics based on trading history,
simulation results, and learning insights.
"""

import os
import sys
import json
import time
import logging
import argparse
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import defaultdict, Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("report_generator.log")]
)
logger = logging.getLogger("ReportGenerator")

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
        "dim": Style.DIM,
        "reset": Style.RESET_ALL
    }
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORS = {
        "blue": "", "green": "", "red": "", "yellow": "", 
        "magenta": "", "cyan": "", "white": "", "bright": "", "dim": "", "reset": ""
    }
    COLORAMA_AVAILABLE = False
    logger.warning("Colorama library not found. Terminal colors will be disabled.")

# Try to import optional matplotlib for advanced charts
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.info("Matplotlib not found. Advanced charts will be disabled.")

# Try to import PDF export libraries
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    PDF_EXPORT_AVAILABLE = True
except ImportError:
    PDF_EXPORT_AVAILABLE = False
    logger.info("ReportLab not found. PDF export will be disabled.")

# Try to import alert system
try:
    import alert_system
    ALERT_SYSTEM_AVAILABLE = True
except ImportError:
    ALERT_SYSTEM_AVAILABLE = False
    logger.warning("Alert system not found. Alerts will be disabled.")

# Try to import memory core
try:
    import memory_core
    MEMORY_CORE_AVAILABLE = True
except ImportError:
    MEMORY_CORE_AVAILABLE = False
    logger.warning("Memory core not found. Report history will not be stored.")

# Constants and configuration
CURRENT_TIME = "2025-04-21 21:53:55"  # UTC
CURRENT_USER = "mstfatiryaki"

# File paths
EXECUTED_TRADES_LOG = "executed_trades_log.json"
SIMULATION_REPORT = "simulation_report.json"
LEARNING_SUMMARY = "learning_summary.json"
PERFORMANCE_REPORT = "performance_report.json"
RISK_REPORT = "risk_report.json"

# Output file paths
DEFAULT_REPORT_DIR = "reports"
DEFAULT_REPORT_PREFIX = "trading_report"

class ReportGenerator:
    """Generates detailed reports based on trading data and system performance"""
    
    def __init__(self, time_range: str = "all", report_type: str = "full", 
                 strategy_filter: Optional[str] = None, coin_filter: Optional[str] = None):
        """
        Initialize the report generator
        
        Args:
            time_range (str): Time range for the report (day, week, month, year, all)
            report_type (str): Type of report (summary, detailed, full)
            strategy_filter (Optional[str]): Filter by strategy (long, short, sniper)
            coin_filter (Optional[str]): Filter by coin (BTC, ETH, etc.)
        """
        self.time_range = time_range
        self.report_type = report_type
        self.strategy_filter = strategy_filter
        self.coin_filter = coin_filter.upper() if coin_filter else None
        
        # Initialize data containers
        self.trades_data = []
        self.simulation_data = {}
        self.learning_data = {}
        self.performance_data = {}
        self.risk_data = {}
        
        # Set timestamp
        self.timestamp = CURRENT_TIME
        
        # Data for analysis
        self.filtered_trades = []
        self.analysis_results = {}
        
        # Report status
        self.data_loaded = False
        self.analysis_complete = False
        
        logger.info(f"Report Generator initialized (time_range: {time_range}, report_type: {report_type})")
    
    def load_data(self) -> bool:
        """
        Load all required data from files
        
        Returns:
            bool: Whether all data was loaded successfully
        """
        trades_loaded = self._load_trades_data()
        simulation_loaded = self._load_simulation_data()
        learning_loaded = self._load_learning_data()
        performance_loaded = self._load_performance_data()
        risk_loaded = self._load_risk_data()
        
        # We need at least trades data to generate a report
        self.data_loaded = trades_loaded
        
        if not self.data_loaded:
            logger.error("Failed to load trades data. Cannot generate report.")
            return False
        
        # Filter trades based on time range, strategy, and coin
        self._filter_trades()
        
        logger.info(f"Data loaded (trades: {trades_loaded}, simulation: {simulation_loaded}, "
                   f"learning: {learning_loaded}, performance: {performance_loaded}, risk: {risk_loaded})")
        
        return self.data_loaded
    
    def _load_trades_data(self) -> bool:
        """
        Load trades data from executed_trades_log.json
        
        Returns:
            bool: Whether trades data was loaded successfully
        """
        try:
            if not os.path.exists(EXECUTED_TRADES_LOG):
                logger.warning(f"{EXECUTED_TRADES_LOG} not found")
                # Try to create sample data for testing
                self._create_sample_trades_data()
                return len(self.trades_data) > 0
            
            with open(EXECUTED_TRADES_LOG, "r", encoding="utf-8") as f:
                self.trades_data = json.load(f)
            
            logger.info(f"Loaded {len(self.trades_data)} trades from {EXECUTED_TRADES_LOG}")
            return len(self.trades_data) > 0
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in {EXECUTED_TRADES_LOG}")
            return False
        except Exception as e:
            logger.error(f"Error loading trades data: {str(e)}")
            return False
    
    def _create_sample_trades_data(self) -> None:
        """Create sample trades data for testing when actual data is missing"""
        logger.info("Creating sample trades data for testing")
        
        # Create some sample trades
        strategies = ["long_strategy", "short_strategy", "sniper_strategy"]
        coins = ["BTC", "ETH", "SOL", "DOGE", "ADA", "XRP"]
        operations = ["LONG", "SHORT"]
        statuses = ["CLOSED", "OPEN", "FAILED"]
        
        # Generate sample data with more trades
        sample_trades = []
        for i in range(100):  # Generate 100 sample trades
            coin = coins[i % len(coins)]
            strategy = strategies[i % len(strategies)]
            operation = operations[i % len(operations)]
            status = statuses[0] if i < 80 else statuses[i % len(statuses)]  # Most trades are closed
            
            # Calculate entry and exit times
            now = datetime.datetime.strptime(CURRENT_TIME, "%Y-%m-%d %H:%M:%S")
            days_ago = i % 30  # Spread over last 30 days
            entry_time = (now - datetime.timedelta(days=days_ago, hours=i%24)).strftime("%Y-%m-%d %H:%M:%S")
            exit_time = (now - datetime.timedelta(days=days_ago, hours=(i%24)-2)).strftime("%Y-%m-%d %H:%M:%S") if status == "CLOSED" else None
            
            # Generate realistic prices
            base_price = 100.0 if coin == "ETH" else 10000.0 if coin == "BTC" else 1.0
            entry_price = base_price * (0.95 + (i % 10) / 100.0)
            
            # For closed trades, calculate exit price, PnL, ROI
            exit_price = None
            pnl = None
            roi = None
            
            if status == "CLOSED":
                # Determine if profitable based on index
                profitable = (i % 3) != 0  # 2/3 of trades are profitable
                
                if operation == "LONG":
                    price_change = 0.05 if profitable else -0.03
                    exit_price = entry_price * (1 + price_change)
                    pnl = (exit_price - entry_price) * 0.1 * entry_price  # Simplified calculation
                else:  # SHORT
                    price_change = -0.05 if profitable else 0.03
                    exit_price = entry_price * (1 + price_change)
                    pnl = (entry_price - exit_price) * 0.1 * entry_price  # Simplified calculation
                
                roi = (pnl / (0.1 * entry_price)) * 100 if pnl is not None else None
            
            # Create the trade record
            trade = {
                "trade_id": f"SAMPLE_{strategy.split('_')[0].upper()}_{i}",
                "coin": coin,
                "symbol": f"{coin}USDT",
                "strategy": strategy,
                "operation": operation,
                "entry_price": entry_price,
                "stop_loss": entry_price * (0.95 if operation == "LONG" else 1.05),
                "take_profit": entry_price * (1.1 if operation == "LONG" else 0.9),
                "investment_usd": 0.1 * entry_price,
                "quantity": 0.1,
                "leverage": 1,
                "status": status,
                "entry_time": entry_time,
                "update_time": exit_time if exit_time else entry_time,
                "exit_time": exit_time,
                "exit_price": exit_price,
                "pnl": pnl,
                "roi": roi,
                "details": {
                    "conditions": {},
                    "indicators": {},
                    "exit_reason": "take_profit" if (profitable and status == "CLOSED") else 
                                "stop_loss" if (not profitable and status == "CLOSED") else None
                },
                "real_mode": False
            }
            
            sample_trades.append(trade)
        
        self.trades_data = sample_trades
        logger.info(f"Created {len(sample_trades)} sample trades")
    
    def _load_simulation_data(self) -> bool:
        """
        Load simulation data from simulation_report.json
        
        Returns:
            bool: Whether simulation data was loaded successfully
        """
        try:
            if not os.path.exists(SIMULATION_REPORT):
                logger.warning(f"{SIMULATION_REPORT} not found")
                return False
            
            with open(SIMULATION_REPORT, "r", encoding="utf-8") as f:
                self.simulation_data = json.load(f)
            
            logger.info(f"Loaded simulation data from {SIMULATION_REPORT}")
            return True
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in {SIMULATION_REPORT}")
            return False
        except Exception as e:
            logger.error(f"Error loading simulation data: {str(e)}")
            return False
    
    def _load_learning_data(self) -> bool:
        """
        Load learning data from learning_summary.json
        
        Returns:
            bool: Whether learning data was loaded successfully
        """
        try:
            if not os.path.exists(LEARNING_SUMMARY):
                logger.warning(f"{LEARNING_SUMMARY} not found")
                return False
            
            with open(LEARNING_SUMMARY, "r", encoding="utf-8") as f:
                self.learning_data = json.load(f)
            
            logger.info(f"Loaded learning data from {LEARNING_SUMMARY}")
            return True
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in {LEARNING_SUMMARY}")
            return False
        except Exception as e:
            logger.error(f"Error loading learning data: {str(e)}")
            return False
    
    def _load_performance_data(self) -> bool:
        """
        Load performance data from performance_report.json
        
        Returns:
            bool: Whether performance data was loaded successfully
        """
        try:
            if not os.path.exists(PERFORMANCE_REPORT):
                logger.warning(f"{PERFORMANCE_REPORT} not found")
                return False
            
            with open(PERFORMANCE_REPORT, "r", encoding="utf-8") as f:
                self.performance_data = json.load(f)
            
            logger.info(f"Loaded performance data from {PERFORMANCE_REPORT}")
            return True
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in {PERFORMANCE_REPORT}")
            return False
        except Exception as e:
            logger.error(f"Error loading performance data: {str(e)}")
            return False
    
    def _load_risk_data(self) -> bool:
        """
        Load risk data from risk_report.json
        
        Returns:
            bool: Whether risk data was loaded successfully
        """
        try:
            if not os.path.exists(RISK_REPORT):
                logger.warning(f"{RISK_REPORT} not found")
                return False
            
            with open(RISK_REPORT, "r", encoding="utf-8") as f:
                self.risk_data = json.load(f)
            
            logger.info(f"Loaded risk data from {RISK_REPORT}")
            return True
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in {RISK_REPORT}")
            return False
        except Exception as e:
            logger.error(f"Error loading risk data: {str(e)}")
            return False
    
    def _filter_trades(self) -> None:
        """Filter trades based on time range, strategy, and coin"""
        if not self.trades_data:
            self.filtered_trades = []
            return
        
        # Parse the current timestamp
        now = datetime.datetime.strptime(self.timestamp, "%Y-%m-%d %H:%M:%S")
        
        # Define time ranges
        time_ranges = {
            "day": now - datetime.timedelta(days=1),
            "week": now - datetime.timedelta(days=7),
            "month": now - datetime.timedelta(days=30),
            "year": now - datetime.timedelta(days=365)
        }
        
        # Filter trades based on time range
        if self.time_range in time_ranges:
            start_time = time_ranges[self.time_range]
            self.filtered_trades = [
                trade for trade in self.trades_data 
                if "entry_time" in trade and 
                datetime.datetime.strptime(trade["entry_time"], "%Y-%m-%d %H:%M:%S") >= start_time
            ]
        else:
            # Default to all trades
            self.filtered_trades = self.trades_data.copy()
        
        # Filter by strategy if needed
        if self.strategy_filter:
            strategy_name = f"{self.strategy_filter}_strategy"
            self.filtered_trades = [
                trade for trade in self.filtered_trades 
                if "strategy" in trade and trade["strategy"] == strategy_name
            ]
        
        # Filter by coin if needed
        if self.coin_filter:
            self.filtered_trades = [
                trade for trade in self.filtered_trades 
                if "coin" in trade and trade["coin"].upper() == self.coin_filter
            ]
        
        logger.info(f"Filtered to {len(self.filtered_trades)} trades")
    
    def analyze_data(self) -> bool:
        """
        Perform comprehensive analysis of the filtered trades
        
        Returns:
            bool: Whether analysis was completed successfully
        """
        if not self.data_loaded or not self.filtered_trades:
            logger.error("No data loaded or no trades available for analysis")
            return False
        
        try:
            # Reset analysis results
            self.analysis_results = {}
            
            # Analyze overall performance
            self._analyze_overall_performance()
            
            # Analyze by strategy
            self._analyze_by_strategy()
            
            # Analyze by coin
            self._analyze_by_coin()
            
            # Find best and worst trades
            self._analyze_best_worst_trades()
            
            # Temporal analysis
            self._analyze_temporal_performance()
            
            # Anomaly detection
            self._detect_anomalies()
            
            # Risk-reward analysis
            self._analyze_risk_reward()
            
            # Strategy recommendations
            self._analyze_recommendations()
            
            self.analysis_complete = True
            logger.info("Data analysis completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error analyzing data: {str(e)}")
            return False
    
    def _analyze_overall_performance(self) -> None:
        """Analyze overall trading performance"""
        # Initialize performance metrics
        performance = {
            "total_trades": len(self.filtered_trades),
            "closed_trades": 0,
            "open_trades": 0,
            "failed_trades": 0,
            "successful_trades": 0,
            "unsuccessful_trades": 0,
            "total_pnl": 0.0,
            "total_investment": 0.0,
            "roi": 0.0,
            "win_rate": 0.0,
            "average_profit": 0.0,
            "average_loss": 0.0,
            "profit_factor": 0.0,
            "largest_profit": 0.0,
            "largest_loss": 0.0,
            "average_holding_time": 0.0
        }
        
        # Collect profit/loss data
        profits = []
        losses = []
        total_profit = 0.0
        total_loss = 0.0
        holding_times = []
        
        # Process each trade
        for trade in self.filtered_trades:
            # Count by status
            status = trade.get("status", "UNKNOWN")
            if status == "CLOSED":
                performance["closed_trades"] += 1
                
                pnl = trade.get("pnl", 0.0)
                if pnl is not None:
                    performance["total_pnl"] += pnl
                    
                    if pnl > 0:
                        performance["successful_trades"] += 1
                        profits.append(pnl)
                        total_profit += pnl
                        
                        if pnl > performance["largest_profit"]:
                            performance["largest_profit"] = pnl
                    else:
                        performance["unsuccessful_trades"] += 1
                        losses.append(pnl)
                        total_loss += pnl
                        
                        if pnl < performance["largest_loss"]:
                            performance["largest_loss"] = pnl
                
                # Calculate holding time
                if "entry_time" in trade and "exit_time" in trade and trade["exit_time"]:
                    try:
                        entry_time = datetime.datetime.strptime(trade["entry_time"], "%Y-%m-%d %H:%M:%S")
                        exit_time = datetime.datetime.strptime(trade["exit_time"], "%Y-%m-%d %H:%M:%S")
                        holding_time = (exit_time - entry_time).total_seconds() / 3600  # in hours
                        holding_times.append(holding_time)
                    except:
                        pass
                
            elif status == "OPEN":
                performance["open_trades"] += 1
            elif status == "FAILED":
                performance["failed_trades"] += 1
            
            # Track total investment
            investment = trade.get("investment_usd", 0.0)
            if investment is not None:
                performance["total_investment"] += investment
        
        # Calculate derived metrics
        if performance["closed_trades"] > 0:
            performance["win_rate"] = (performance["successful_trades"] / performance["closed_trades"]) * 100
            
        if performance["total_investment"] > 0:
            performance["roi"] = (performance["total_pnl"] / performance["total_investment"]) * 100
            
        if profits:
            performance["average_profit"] = sum(profits) / len(profits)
            
        if losses:
            performance["average_loss"] = sum(losses) / len(losses)
            
        if total_loss < 0 and total_profit > 0:
            performance["profit_factor"] = total_profit / abs(total_loss)
            
        if holding_times:
            performance["average_holding_time"] = sum(holding_times) / len(holding_times)
        
        # Store results
        self.analysis_results["overall_performance"] = performance
    
    def _analyze_by_strategy(self) -> None:
        """Analyze performance by trading strategy"""
        # Group trades by strategy
        strategies = defaultdict(list)
        
        for trade in self.filtered_trades:
            strategy = trade.get("strategy", "unknown")
            strategies[strategy].append(trade)
        
        # Initialize strategy performance
        strategy_performance = {}
        
        # Analyze each strategy
        for strategy, trades in strategies.items():
            closed_trades = [t for t in trades if t.get("status") == "CLOSED"]
            successful_trades = [t for t in closed_trades if t.get("pnl", 0) > 0]
            
            total_pnl = sum(t.get("pnl", 0) for t in closed_trades if t.get("pnl") is not None)
            total_investment = sum(t.get("investment_usd", 0) for t in closed_trades if t.get("investment_usd") is not None)
            
            win_rate = (len(successful_trades) / len(closed_trades)) * 100 if closed_trades else 0
            roi = (total_pnl / total_investment) * 100 if total_investment > 0 else 0
            
            avg_holding_time = 0
            holding_times = []
            
            for trade in closed_trades:
                if "entry_time" in trade and "exit_time" in trade and trade["exit_time"]:
                    try:
                        entry_time = datetime.datetime.strptime(trade["entry_time"], "%Y-%m-%d %H:%M:%S")
                        exit_time = datetime.datetime.strptime(trade["exit_time"], "%Y-%m-%d %H:%M:%S")
                        holding_time = (exit_time - entry_time).total_seconds() / 3600  # in hours
                        holding_times.append(holding_time)
                    except:
                        pass
            
            if holding_times:
                avg_holding_time = sum(holding_times) / len(holding_times)
            
            # Store strategy performance
            strategy_performance[strategy] = {
                "total_trades": len(trades),
                "closed_trades": len(closed_trades),
                "successful_trades": len(successful_trades),
                "total_pnl": total_pnl,
                "total_investment": total_investment,
                "win_rate": win_rate,
                "roi": roi,
                "avg_holding_time": avg_holding_time
            }
        
        # Store results
        self.analysis_results["strategy_performance"] = strategy_performance
    
    def _analyze_by_coin(self) -> None:
        """Analyze performance by coin"""
        # Group trades by coin
        coins = defaultdict(list)
        
        for trade in self.filtered_trades:
            coin = trade.get("coin", "unknown").upper()
            coins[coin].append(trade)
        
        # Initialize coin performance
        coin_performance = {}
        
        # Analyze each coin
        for coin, trades in coins.items():
            closed_trades = [t for t in trades if t.get("status") == "CLOSED"]
            successful_trades = [t for t in closed_trades if t.get("pnl", 0) > 0]
            
            total_pnl = sum(t.get("pnl", 0) for t in closed_trades if t.get("pnl") is not None)
            total_investment = sum(t.get("investment_usd", 0) for t in closed_trades if t.get("investment_usd") is not None)
            
            win_rate = (len(successful_trades) / len(closed_trades)) * 100 if closed_trades else 0
            roi = (total_pnl / total_investment) * 100 if total_investment > 0 else 0
            
            # Group by strategy for this coin
            strategy_breakdown = defaultdict(int)
            for trade in trades:
                strategy = trade.get("strategy", "unknown")
                strategy_breakdown[strategy] += 1
            
            # Store coin performance
            coin_performance[coin] = {
                "total_trades": len(trades),
                "closed_trades": len(closed_trades),
                "successful_trades": len(successful_trades),
                "total_pnl": total_pnl,
                "total_investment": total_investment,
                "win_rate": win_rate,
                "roi": roi,
                "strategy_breakdown": dict(strategy_breakdown)
            }
        
        # Store results
        self.analysis_results["coin_performance"] = coin_performance
    
    def _analyze_best_worst_trades(self) -> None:
        """Find and analyze best and worst trades"""
        # Filter closed trades with valid PnL
        closed_trades = [
            t for t in self.filtered_trades 
            if t.get("status") == "CLOSED" and t.get("pnl") is not None
        ]
        
        # Sort by PnL
        sorted_by_pnl = sorted(closed_trades, key=lambda t: t.get("pnl", 0), reverse=True)
        sorted_by_roi = sorted(closed_trades, key=lambda t: t.get("roi", 0), reverse=True)
        
        # Get top and bottom trades
        best_worst = {
            "best_trades_pnl": sorted_by_pnl[:5] if len(sorted_by_pnl) >= 5 else sorted_by_pnl,
            "worst_trades_pnl": sorted_by_pnl[-5:] if len(sorted_by_pnl) >= 5 else sorted_by_pnl[::-1],
            "best_trades_roi": sorted_by_roi[:5] if len(sorted_by_roi) >= 5 else sorted_by_roi,
            "worst_trades_roi": sorted_by_roi[-5:] if len(sorted_by_roi) >= 5 else sorted_by_roi[::-1]
        }
        
        # Store results
        self.analysis_results["best_worst_trades"] = best_worst
    
    def _analyze_temporal_performance(self) -> None:
        """Analyze performance over time (daily, weekly, hourly)"""
        # Filter closed trades with valid timestamps
        valid_trades = [
            t for t in self.filtered_trades 
            if t.get("status") == "CLOSED" and "entry_time" in t and "pnl" in t and t["pnl"] is not None
        ]
        
        if not valid_trades:
            self.analysis_results["temporal_performance"] = {}
            return
        
        # Initialize temporal data
        temporal = {
            "daily": defaultdict(lambda: {"pnl": 0.0, "trades": 0}),
            "weekly": defaultdict(lambda: {"pnl": 0.0, "trades": 0}),
            "hourly": defaultdict(lambda: {"pnl": 0.0, "trades": 0})
        }
        
        # Process each trade
        for trade in valid_trades:
            try:
                entry_time = datetime.datetime.strptime(trade["entry_time"], "%Y-%m-%d %H:%M:%S")
                
                # Daily performance (by date)
                day_key = entry_time.strftime("%Y-%m-%d")
                temporal["daily"][day_key]["pnl"] += trade.get("pnl", 0)
                temporal["daily"][day_key]["trades"] += 1
                
                # Weekly performance (by week number)
                week_key = entry_time.strftime("%Y-W%U")
                temporal["weekly"][week_key]["pnl"] += trade.get("pnl", 0)
                temporal["weekly"][week_key]["trades"] += 1
                
                # Hourly performance (by hour of day)
                hour_key = entry_time.strftime("%H:00")
                temporal["hourly"][hour_key]["pnl"] += trade.get("pnl", 0)
                temporal["hourly"][hour_key]["trades"] += 1
                
            except:
                continue
        
        # Convert defaultdicts to regular dicts for JSON serialization
        temporal["daily"] = dict(temporal["daily"])
        temporal["weekly"] = dict(temporal["weekly"])
        temporal["hourly"] = dict(temporal["hourly"])
        
        # Find best and worst periods
        if temporal["daily"]:
            best_day = max(temporal["daily"].items(), key=lambda x: x[1]["pnl"])
            worst_day = min(temporal["daily"].items(), key=lambda x: x[1]["pnl"])
            temporal["best_day"] = {"date": best_day[0], "pnl": best_day[1]["pnl"], "trades": best_day[1]["trades"]}
            temporal["worst_day"] = {"date": worst_day[0], "pnl": worst_day[1]["pnl"], "trades": worst_day[1]["trades"]}
        
        if temporal["hourly"]:
            best_hour = max(temporal["hourly"].items(), key=lambda x: x[1]["pnl"])
            temporal["best_hour"] = {"hour": best_hour[0], "pnl": best_hour[1]["pnl"], "trades": best_hour[1]["trades"]}
        
        # Store results
        self.analysis_results["temporal_performance"] = temporal
    
    def _detect_anomalies(self) -> None:
        """Detect anomalies and interesting patterns in trading data"""
        anomalies = []
        
        # Filter closed trades
        closed_trades = [t for t in self.filtered_trades if t.get("status") == "CLOSED"]
        
        if not closed_trades:
            self.analysis_results["anomalies"] = anomalies
            return
        
        # 1. Check for unusually large trades
        avg_investment = sum(t.get("investment_usd", 0) for t in closed_trades) / len(closed_trades)
        large_trades = [
            t for t in closed_trades 
            if t.get("investment_usd", 0) > avg_investment * 3  # 3x average
        ]
        
        if large_trades:
            anomalies.append({
                "type": "large_trades",
                "description": f"Found {len(large_trades)} unusually large trades (>3x average investment)",
                "trades": large_trades[:3]  # Include top 3 as examples
            })
        
        # 2. Check for trades with extreme ROI (positive or negative)
        roi_values = [t.get("roi", 0) for t in closed_trades if t.get("roi") is not None]
        if roi_values:
            avg_roi = sum(roi_values) / len(roi_values)
            std_roi = (sum((r - avg_roi) ** 2 for r in roi_values) / len(roi_values)) ** 0.5  # stddev
            
            extreme_roi_trades = [
                t for t in closed_trades 
                if t.get("roi") is not None and abs(t.get("roi", 0) - avg_roi) > 2 * std_roi  # 2 stddev away
            ]
            
            if extreme_roi_trades:
                anomalies.append({
                    "type": "extreme_roi",
                    "description": f"Found {len(extreme_roi_trades)} trades with extreme ROI (>2 standard deviations)",
                    "trades": sorted(extreme_roi_trades, key=lambda t: abs(t.get("roi", 0)), reverse=True)[:3]
                })
        
        # 3. Check for unusual holding times
        holding_times = []
        for trade in closed_trades:
            if "entry_time" in trade and "exit_time" in trade and trade["exit_time"]:
                try:
                    entry_time = datetime.datetime.strptime(trade["entry_time"], "%Y-%m-%d %H:%M:%S")
                    exit_time = datetime.datetime.strptime(trade["exit_time"], "%Y-%m-%d %H:%M:%S")
                    holding_time = (exit_time - entry_time).total_seconds() / 3600  # in hours
                    holding_times.append((trade, holding_time))
                except:
                    pass
        
        if holding_times:
            # Sort by holding time
            holding_times.sort(key=lambda x: x[1])
            
            # Very short trades (< 1 hour)
            very_short_trades = [t for t, h in holding_times if h < 1]
            if very_short_trades:
                anomalies.append({
                    "type": "very_short_trades",
                    "description": f"Found {len(very_short_trades)} very short trades (<1 hour)",
                    "trades": very_short_trades[:3]
                })
            
            # Very long trades (> 7 days)
            very_long_trades = [t for t, h in holding_times if h > 168]  # 7*24 = 168 hours
            if very_long_trades:
                anomalies.append({
                    "type": "very_long_trades",
                    "description": f"Found {len(very_long_trades)} very long trades (>7 days)",
                    "trades": very_long_trades[:3]
                })
        
        # 4. Check for unusual coin performance
        if "coin_performance" in self.analysis_results:
            coin_perf = self.analysis_results["coin_performance"]
            
            # Coins with 100% win rate (but at least 5 trades)
            perfect_coins = [
                (coin, data) for coin, data in coin_perf.items()
                if data["win_rate"] == 100 and data["closed_trades"] >= 5
            ]
            
            if perfect_coins:
                anomalies.append({
                    "type": "perfect_win_rate",
                    "description": f"Found {len(perfect_coins)} coins with 100% win rate (min 5 trades)",
                    "coins": [c[0] for c in perfect_coins]
                })
            
            # Coins with 0% win rate (but at least 5 trades)
            zero_win_coins = [
                (coin, data) for coin, data in coin_perf.items()
                if data["win_rate"] == 0 and data["closed_trades"] >= 5
            ]
            
            if zero_win_coins:
                anomalies.append({
                    "type": "zero_win_rate",
                    "description": f"Found {len(zero_win_coins)} coins with 0% win rate (min 5 trades)",
                    "coins": [c[0] for c in zero_win_coins]
                })
        
        # Store results
        self.analysis_results["anomalies"] = anomalies
    
    def _analyze_risk_reward(self) -> None:
        """Analyze risk-reward metrics"""
        # Filter closed trades
        closed_trades = [t for t in self.filtered_trades if t.get("status") == "CLOSED"]
        
        if not closed_trades:
            self.analysis_results["risk_reward"] = {}
            return
        
        # Initialize risk-reward metrics
        risk_reward = {
            "avg_profit_loss_ratio": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_trades": [],
            "sharpe_ratio": 0.0,
            "volatility": 0.0,
            "risk_per_trade": 0.0
        }
        
        # Calculate profit-loss ratio
        if "overall_performance" in self.analysis_results:
            perf = self.analysis_results["overall_performance"]
            avg_profit = perf.get("average_profit", 0)
            avg_loss = perf.get("average_loss", 0)
            
            if avg_loss != 0:
                risk_reward["avg_profit_loss_ratio"] = abs(avg_profit / avg_loss)
        
        # Calculate maximum drawdown
        # Sort trades by entry time
        sorted_trades = sorted(closed_trades, key=lambda t: t.get("entry_time", ""))
        
        # Calculate cumulative PnL
        cumulative_pnl = 0
        max_pnl = 0
        max_drawdown = 0
        drawdown_start_idx = 0
        drawdown_end_idx = 0
        current_drawdown_start_idx = 0
        
        for i, trade in enumerate(sorted_trades):
            pnl = trade.get("pnl", 0)
            if pnl is not None:
                cumulative_pnl += pnl
                
                if cumulative_pnl > max_pnl:
                    max_pnl = cumulative_pnl
                    current_drawdown_start_idx = i
                
                drawdown = max_pnl - cumulative_pnl
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                    drawdown_start_idx = current_drawdown_start_idx
                    drawdown_end_idx = i
        
        risk_reward["max_drawdown"] = max_drawdown
        
        # Get the trades that caused the max drawdown
        if max_drawdown > 0 and drawdown_start_idx <= drawdown_end_idx:
            risk_reward["max_drawdown_trades"] = sorted_trades[drawdown_start_idx:drawdown_end_idx+1]
        
        # Calculate volatility (standard deviation of returns)
        returns = [t.get("roi", 0) for t in closed_trades if t.get("roi") is not None]
        if returns:
            avg_return = sum(returns) / len(returns)
            variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
            risk_reward["volatility"] = variance ** 0.5
            
            # Calculate Sharpe Ratio (assuming risk-free rate = 0)
            if risk_reward["volatility"] > 0:
                risk_reward["sharpe_ratio"] = avg_return / risk_reward["volatility"]
        
        # Calculate average risk per trade (based on stop loss)
        risk_percentages = []
        for trade in closed_trades:
            if trade.get("operation") == "LONG" and "entry_price" in trade and "stop_loss" in trade:
                entry = trade["entry_price"]
                stop = trade["stop_loss"]
                if entry > 0:
                    risk_pct = (entry - stop) / entry
                    risk_percentages.append(risk_pct)
            elif trade.get("operation") == "SHORT" and "entry_price" in trade and "stop_loss" in trade:
                entry = trade["entry_price"]
                stop = trade["stop_loss"]
                if entry > 0:
                    risk_pct = (stop - entry) / entry
                    risk_percentages.append(risk_pct)
        
        if risk_percentages:
            risk_reward["risk_per_trade"] = sum(risk_percentages) / len(risk_percentages)
        
        # Store results
        self.analysis_results["risk_reward"] = risk_reward
    
    def _analyze_recommendations(self) -> None:
        """Analyze and extract recommendations from learning data"""
        recommendations = []
        
        # Add recommendations from learning engine if available
        if self.learning_data:
            if "recommendations" in self.learning_data:
                for rec in self.learning_data["recommendations"]:
                    recommendations.append({
                        "source": "learning_engine",
                        "recommendation": rec
                    })
            elif "insights" in self.learning_data:
                for insight in self.learning_data["insights"]:
                    recommendations.append({
                        "source": "learning_engine",
                        "recommendation": insight
                    })
        
        # Generate recommendations based on analysis results
        if "strategy_performance" in self.analysis_results:
            strategy_perf = self.analysis_results["strategy_performance"]
            
            # Find the best performing strategy by ROI
            if strategy_perf:
                best_strategy = max(strategy_perf.items(), key=lambda x: x[1]["roi"])
                worst_strategy = min(strategy_perf.items(), key=lambda x: x[1]["roi"])
                
                if best_strategy[1]["roi"] > 0:
                    recommendations.append({
                        "source": "report_analysis",
                        "recommendation": f"Consider increasing allocation to {best_strategy[0]} strategy (ROI: {best_strategy[1]['roi']:.1f}%)"
                    })
                
                if worst_strategy[1]["roi"] < 0:
                    recommendations.append({
                        "source": "report_analysis",
                        "recommendation": f"Review or reduce allocation to {worst_strategy[0]} strategy (ROI: {worst_strategy[1]['roi']:.1f}%)"
                    })
        
        # Recommendations based on coin performance
        if "coin_performance" in self.analysis_results:
            coin_perf = self.analysis_results["coin_performance"]
            
            # Filter coins with at least 5 trades
            active_coins = {
                coin: data for coin, data in coin_perf.items() 
                if data["closed_trades"] >= 5
            }
            
            if active_coins:
                best_coin = max(active_coins.items(), key=lambda x: x[1]["roi"])
                worst_coin = min(active_coins.items(), key=lambda x: x[1]["roi"])
                
                if best_coin[1]["roi"] > 10:  # Only recommend if ROI is significant
                    recommendations.append({
                        "source": "report_analysis",
                        "recommendation": f"{best_coin[0]} shows strong performance (ROI: {best_coin[1]['roi']:.1f}%) with {best_coin[1]['closed_trades']} trades"
                    })
                
                if worst_coin[1]["roi"] < -10:  # Only warn if ROI is significantly negative
                    recommendations.append({
                        "source": "report_analysis",
                        "recommendation": f"Consider reviewing trading strategy for {worst_coin[0]} (ROI: {worst_coin[1]['roi']:.1f}%)"
                    })
        
        # Recommendations based on temporal analysis
        if "temporal_performance" in self.analysis_results:
            temporal = self.analysis_results["temporal_performance"]
            
            if "best_hour" in temporal:
                best_hour = temporal["best_hour"]
                recommendations.append({
                    "source": "report_analysis",
                    "recommendation": f"Most profitable trading hour: {best_hour['hour']} (PnL: ${best_hour['pnl']:.2f})"
                })
            
            if "hourly" in temporal:
                hourly_data = temporal["hourly"]
                worst_hours = []
                
                for hour, data in hourly_data.items():
                    if data["pnl"] < 0 and data["trades"] >= 3:  # At least 3 trades
                        worst_hours.append((hour, data["pnl"]))
                
                if worst_hours:
                    worst_hours.sort(key=lambda x: x[1])
                    worst_hour = worst_hours[0]
                    recommendations.append({
                        "source": "report_analysis",
                        "recommendation": f"Consider avoiding trades during {worst_hour[0]} hour (PnL: ${worst_hour[1]:.2f})"
                    })
        
        # Recommendations based on risk-reward analysis
        if "risk_reward" in self.analysis_results:
            risk_reward = self.analysis_results["risk_reward"]
            
            profit_loss_ratio = risk_reward.get("avg_profit_loss_ratio", 0)
            if profit_loss_ratio < 1.5:
                recommendations.append({
                    "source": "report_analysis",
                    "recommendation": f"Profit/Loss ratio ({profit_loss_ratio:.2f}) is below target 1.5. Consider adjusting take-profit levels."
                })
            
            sharpe = risk_reward.get("sharpe_ratio", 0)
            if sharpe < 1.0:
                recommendations.append({
                    "source": "report_analysis",
                    "recommendation": f"Sharpe ratio ({sharpe:.2f}) is below 1.0. Consider reducing trade frequency or improving strategy."
                })
        
        # Store results
        self.analysis_results["recommendations"] = recommendations
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a complete report based on analysis results
        
        Returns:
            Dict[str, Any]: Complete report
        """
        if not self.data_loaded or not self.analysis_complete:
            logger.error("Cannot generate report: data not loaded or analysis not complete")
            return {}
        
        # Generate report timestamp
        report_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create report structure
        report = {
            "report_id": f"REPORT_{int(time.time())}",
            "timestamp": report_timestamp,
            "time_range": self.time_range,
            "report_type": self.report_type,
            "strategy_filter": self.strategy_filter,
            "coin_filter": self.coin_filter,
            "data_sources": {
                "trades_data": len(self.trades_data),
                "filtered_trades": len(self.filtered_trades),
                "simulation_data": bool(self.simulation_data),
                "learning_data": bool(self.learning_data),
                "performance_data": bool(self.performance_data),
                "risk_data": bool(self.risk_data)
            },
            "analysis_results": self.analysis_results
        }
        
        return report
    
    def display_report(self, report: Dict[str, Any]) -> None:
        """
        Display report in the terminal
        
        Args:
            report (Dict[str, Any]): Report to display
        """
        if not report:
            print(f"{COLORS['red']}No report data to display{COLORS['reset']}")
            return
        
        # Display header
        self._display_report_header(report)
        
        # Display overall performance
        if "overall_performance" in report["analysis_results"]:
            self._display_overall_performance(report["analysis_results"]["overall_performance"])
        
        # Display strategy performance
        if "strategy_performance" in report["analysis_results"]:
            self._display_strategy_performance(report["analysis_results"]["strategy_performance"])
        
        # Display coin performance
        if "coin_performance" in report["analysis_results"]:
            self._display_coin_performance(report["analysis_results"]["coin_performance"])
        
        # Display best/worst trades
        if "best_worst_trades" in report["analysis_results"]:
            self._display_best_worst_trades(report["analysis_results"]["best_worst_trades"])
        
        # Display temporal performance
        if "temporal_performance" in report["analysis_results"]:
            self._display_temporal_performance(report["analysis_results"]["temporal_performance"])
        
        # Display anomalies
        if "anomalies" in report["analysis_results"]:
            self._display_anomalies(report["analysis_results"]["anomalies"])
        
        # Display risk-reward analysis
        if "risk_reward" in report["analysis_results"]:
            self._display_risk_reward(report["analysis_results"]["risk_reward"])
        
        # Display recommendations
        if "recommendations" in report["analysis_results"]:
            self._display_recommendations(report["analysis_results"]["recommendations"])
        
        # Display footer
        self._display_report_footer(report)
    
    def _display_report_header(self, report: Dict[str, Any]) -> None:
        """Display report header"""
        print(f"\n{COLORS['bright']}{COLORS['cyan']}{'=' * 80}{COLORS['reset']}")
        print(f"{COLORS['bright']}{COLORS['cyan']}{'SENTIENTTRADER.AI - TRADING PERFORMANCE REPORT':^80}{COLORS['reset']}")
        print(f"{COLORS['bright']}{COLORS['cyan']}{'=' * 80}{COLORS['reset']}")
        
        # Report metadata
        time_range = report["time_range"].upper()
        report_type = report["report_type"].upper()
        
        print(f"{COLORS['bright']}Report ID:    {COLORS['reset']}{report['report_id']}")
        print(f"{COLORS['bright']}Generated:    {COLORS['reset']}{report['timestamp']}")
        print(f"{COLORS['bright']}Time Range:   {COLORS['reset']}{time_range}")
        print(f"{COLORS['bright']}Report Type:  {COLORS['reset']}{report_type}")
        
        if report["strategy_filter"]:
            print(f"{COLORS['bright']}Strategy:     {COLORS['reset']}{report['strategy_filter']}")
        
        if report["coin_filter"]:
            print(f"{COLORS['bright']}Coin:         {COLORS['reset']}{report['coin_filter']}")
        
        # Data sources
        print(f"\n{COLORS['bright']}Data Sources:{COLORS['reset']}")
        print(f"  Total Trades:     {report['data_sources']['trades_data']}")
        print(f"  Filtered Trades:  {report['data_sources']['filtered_trades']}")
        
        # Additional data sources
        sources = []
        if report['data_sources']['simulation_data']:
            sources.append("Simulation Data")
        if report['data_sources']['learning_data']:
            sources.append("Learning Data")
        if report['data_sources']['performance_data']:
            sources.append("Performance Data")
        if report['data_sources']['risk_data']:
            sources.append("Risk Data")
            
        if sources:
            print(f"  Additional Sources: {', '.join(sources)}")
        
        print(f"{COLORS['cyan']}{'=' * 80}{COLORS['reset']}\n")
    
    def _display_overall_performance(self, performance: Dict[str, Any]) -> None:
        """Display overall performance metrics"""
        print(f"{COLORS['bright']}{COLORS['blue']}OVERALL PERFORMANCE{COLORS['reset']}")
        print(f"{COLORS['blue']}{'-' * 80}{COLORS['reset']}")
        
        # Trade counts
        total_trades = performance["total_trades"]
        closed_trades = performance["closed_trades"]
        open_trades = performance["open_trades"]
        failed_trades = performance["failed_trades"]
        
        print(f"{COLORS['bright']}Trade Counts:{COLORS['reset']}")
        print(f"  Total Trades:     {total_trades}")
        print(f"  Closed Trades:    {closed_trades}")
        print(f"  Open Trades:      {open_trades}")
        print(f"  Failed Trades:    {failed_trades}")
        
        # Performance metrics
        successful = performance["successful_trades"]
        unsuccessful = performance["unsuccessful_trades"]
        win_rate = performance["win_rate"]
        
        win_rate_color = COLORS["green"] if win_rate >= 50 else COLORS["yellow"] if win_rate >= 40 else COLORS["red"]
        
        print(f"\n{COLORS['bright']}Performance Metrics:{COLORS['reset']}")
        print(f"  Successful Trades:   {successful}")
        print(f"  Unsuccessful Trades: {unsuccessful}")
        print(f"  Win Rate:            {win_rate_color}{win_rate:.1f}%{COLORS['reset']}")
        
        # Financial metrics
        total_pnl = performance["total_pnl"]
        roi = performance["roi"]
        
        pnl_color = COLORS["green"] if total_pnl >= 0 else COLORS["red"]
        roi_color = COLORS["green"] if roi >= 0 else COLORS["red"]
        
        pnl_sign = "+" if total_pnl >= 0 else ""
        roi_sign = "+" if roi >= 0 else ""
        
        print(f"\n{COLORS['bright']}Financial Metrics:{COLORS['reset']}")
        print(f"  Total PnL:           {pnl_color}{pnl_sign}${total_pnl:.2f}{COLORS['reset']}")
        print(f"  ROI:                 {roi_color}{roi_sign}{roi:.2f}%{COLORS['reset']}")
        print(f"  Total Investment:    ${performance['total_investment']:.2f}")
        
        # Profit/Loss metrics
        avg_profit = performance["average_profit"]
        avg_loss = performance["average_loss"]
        largest_profit = performance["largest_profit"]
        largest_loss = performance["largest_loss"]
        
        profit_factor = performance["profit_factor"]
        profit_factor_color = COLORS["green"] if profit_factor >= 1.5 else COLORS["yellow"] if profit_factor >= 1 else COLORS["red"]
        
        print(f"\n{COLORS['bright']}Profit/Loss Metrics:{COLORS['reset']}")
        print(f"  Average Profit:      {COLORS['green']}+${avg_profit:.2f}{COLORS['reset']}")
        print(f"  Average Loss:        {COLORS['red']}${avg_loss:.2f}{COLORS['reset']}")
        print(f"  Largest Profit:      {COLORS['green']}+${largest_profit:.2f}{COLORS['reset']}")
        print(f"  Largest Loss:        {COLORS['red']}${largest_loss:.2f}{COLORS['reset']}")
        print(f"  Profit Factor:       {profit_factor_color}{profit_factor:.2f}{COLORS['reset']}")
        
        # Timing metrics
        avg_holding = performance["average_holding_time"]
        
        print(f"\n{COLORS['bright']}Timing Metrics:{COLORS['reset']}")
        print(f"  Avg Holding Time:    {avg_holding:.1f} hours ({avg_holding/24:.1f} days)")
        
        print()
    
    def _display_strategy_performance(self, strategy_perf: Dict[str, Dict[str, Any]]) -> None:
        """Display performance by strategy"""
        print(f"{COLORS['bright']}{COLORS['blue']}STRATEGY PERFORMANCE{COLORS['reset']}")
        print(f"{COLORS['blue']}{'-' * 80}{COLORS['reset']}")
        
        if not strategy_perf:
            print(f"{COLORS['yellow']}No strategy performance data available{COLORS['reset']}\n")
            return
        
        # Sort strategies by PnL
        sorted_strategies = sorted(strategy_perf.items(), key=lambda x: x[1]["total_pnl"], reverse=True)
        
        for strategy_name, data in sorted_strategies:
            # Clean up strategy name for display
            display_name = strategy_name.replace("_strategy", "").upper()
            
            # Get metrics
            total_trades = data["total_trades"]
            closed_trades = data["closed_trades"]
            success_trades = data["successful_trades"]
            total_pnl = data["total_pnl"]
            roi = data["roi"]
            win_rate = data["win_rate"]
            
            # Colors
            pnl_color = COLORS["green"] if total_pnl >= 0 else COLORS["red"]
            roi_color = COLORS["green"] if roi >= 0 else COLORS["red"]
            win_rate_color = COLORS["green"] if win_rate >= 50 else COLORS["yellow"] if win_rate >= 40 else COLORS["red"]
            
            pnl_sign = "+" if total_pnl >= 0 else ""
            roi_sign = "+" if roi >= 0 else ""
            
            print(f"{COLORS['bright']}{display_name} Strategy:{COLORS['reset']}")
            print(f"  Total Trades:     {total_trades} ({closed_trades} closed)")
            print(f"  Win Rate:         {win_rate_color}{win_rate:.1f}%{COLORS['reset']} ({success_trades}/{closed_trades})")
            print(f"  Total PnL:        {pnl_color}{pnl_sign}${total_pnl:.2f}{COLORS['reset']}")
            print(f"  ROI:              {roi_color}{roi_sign}{roi:.2f}%{COLORS['reset']}")
            
            if "avg_holding_time" in data:
                holding_time = data["avg_holding_time"]
                print(f"  Avg Holding Time: {holding_time:.1f} hours")
            
            print()
    
    def _display_coin_performance(self, coin_perf: Dict[str, Dict[str, Any]]) -> None:
        """Display performance by coin"""
        print(f"{COLORS['bright']}{COLORS['blue']}COIN PERFORMANCE{COLORS['reset']}")
        print(f"{COLORS['blue']}{'-' * 80}{COLORS['reset']}")
        
        if not coin_perf:
            print(f"{COLORS['yellow']}No coin performance data available{COLORS['reset']}\n")
            return
        
        # Sort coins by PnL
        sorted_coins = sorted(coin_perf.items(), key=lambda x: x[1]["total_pnl"], reverse=True)
        
        # Display top 5 coins
        print(f"{COLORS['bright']}Top Performing Coins:{COLORS['reset']}")
        
        for i, (coin, data) in enumerate(sorted_coins[:5], 1):
            # Get metrics
            total_trades = data["total_trades"]
            win_rate = data["win_rate"]
            total_pnl = data["total_pnl"]
            roi = data["roi"]
            
            # Colors
            pnl_color = COLORS["green"] if total_pnl >= 0 else COLORS["red"]
            roi_color = COLORS["green"] if roi >= 0 else COLORS["red"]
            win_rate_color = COLORS["green"] if win_rate >= 50 else COLORS["yellow"] if win_rate >= 40 else COLORS["red"]
            
            pnl_sign = "+" if total_pnl >= 0 else ""
            roi_sign = "+" if roi >= 0 else ""
            
            print(f"  {i}. {COLORS['bright']}{coin}{COLORS['reset']} - {total_trades} trades, "
                  f"{win_rate_color}{win_rate:.1f}%{COLORS['reset']} win rate, "
                  f"{pnl_color}{pnl_sign}${total_pnl:.2f}{COLORS['reset']} ({roi_color}{roi_sign}{roi:.2f}%{COLORS['reset']})")
        
        # Display bottom 3 coins (if more than 8 coins)
        if len(sorted_coins) > 8:
            print(f"\n{COLORS['bright']}Worst Performing Coins:{COLORS['reset']}")
            
            for i, (coin, data) in enumerate(sorted_coins[-3:], 1):
                # Get metrics
                total_trades = data["total_trades"]
                win_rate = data["win_rate"]
                total_pnl = data["total_pnl"]
                roi = data["roi"]
                
                # Colors
                pnl_color = COLORS["green"] if total_pnl >= 0 else COLORS["red"]
                roi_color = COLORS["green"] if roi >= 0 else COLORS["red"]
                win_rate_color = COLORS["green"] if win_rate >= 50 else COLORS["yellow"] if win_rate >= 40 else COLORS["red"]
                
                pnl_sign = "+" if total_pnl >= 0 else ""
                roi_sign = "+" if roi >= 0 else ""
                
                print(f"  {i}. {COLORS['bright']}{coin}{COLORS['reset']} - {total_trades} trades, "
                      f"{win_rate_color}{win_rate:.1f}%{COLORS['reset']} win rate, "
                      f"{pnl_color}{pnl_sign}${total_pnl:.2f}{COLORS['reset']} ({roi_color}{roi_sign}{roi:.2f}%{COLORS['reset']})")
        
        print()
    
    def _display_best_worst_trades(self, best_worst: Dict[str, List[Dict[str, Any]]]) -> None:
                """Display best and worst trades"""
        print(f"{COLORS['bright']}{COLORS['blue']}BEST & WORST TRADES{COLORS['reset']}")
        print(f"{COLORS['blue']}{'-' * 80}{COLORS['reset']}")
        
        if not best_worst:
            print(f"{COLORS['yellow']}No best/worst trades data available{COLORS['reset']}\n")
            return
        
        # Best trades by PnL
        best_pnl = best_worst.get("best_trades_pnl", [])
        if best_pnl:
            print(f"{COLORS['bright']}Best Trades by PnL:{COLORS['reset']}")
            
            for i, trade in enumerate(best_pnl[:3], 1):
                # Get trade details
                coin = trade.get("coin", "UNKNOWN")
                operation = trade.get("operation", "UNKNOWN")
                entry_price = trade.get("entry_price", 0)
                exit_price = trade.get("exit_price", 0)
                pnl = trade.get("pnl", 0)
                roi = trade.get("roi", 0)
                
                # Format dates
                entry_time = trade.get("entry_time", "")
                exit_time = trade.get("exit_time", "")
                
                if entry_time:
                    try:
                        entry_dt = datetime.datetime.strptime(entry_time, "%Y-%m-%d %H:%M:%S")
                        entry_time = entry_dt.strftime("%m-%d %H:%M")
                    except:
                        pass
                        
                if exit_time:
                    try:
                        exit_dt = datetime.datetime.strptime(exit_time, "%Y-%m-%d %H:%M:%S")
                        exit_time = exit_dt.strftime("%m-%d %H:%M")
                    except:
                        pass
                
                # Colors
                op_color = COLORS["green"] if operation == "LONG" else COLORS["red"]
                
                print(f"  {i}. {COLORS['bright']}{coin}{COLORS['reset']} - "
                      f"{op_color}{operation}{COLORS['reset']} - "
                      f"{COLORS['green']}+${pnl:.2f}{COLORS['reset']} "
                      f"({COLORS['green']}+{roi:.2f}%{COLORS['reset']})")
                print(f"     Entry: ${entry_price:.6f} ({entry_time}) → Exit: ${exit_price:.6f} ({exit_time})")
        
        # Worst trades by PnL
        worst_pnl = best_worst.get("worst_trades_pnl", [])
        if worst_pnl:
            print(f"\n{COLORS['bright']}Worst Trades by PnL:{COLORS['reset']}")
            
            for i, trade in enumerate(worst_pnl[:3], 1):
                # Get trade details
                coin = trade.get("coin", "UNKNOWN")
                operation = trade.get("operation", "UNKNOWN")
                entry_price = trade.get("entry_price", 0)
                exit_price = trade.get("exit_price", 0)
                pnl = trade.get("pnl", 0)
                roi = trade.get("roi", 0)
                
                # Format dates
                entry_time = trade.get("entry_time", "")
                exit_time = trade.get("exit_time", "")
                
                if entry_time:
                    try:
                        entry_dt = datetime.datetime.strptime(entry_time, "%Y-%m-%d %H:%M:%S")
                        entry_time = entry_dt.strftime("%m-%d %H:%M")
                    except:
                        pass
                        
                if exit_time:
                    try:
                        exit_dt = datetime.datetime.strptime(exit_time, "%Y-%m-%d %H:%M:%S")
                        exit_time = exit_dt.strftime("%m-%d %H:%M")
                    except:
                        pass
                
                # Colors
                op_color = COLORS["green"] if operation == "LONG" else COLORS["red"]
                
                print(f"  {i}. {COLORS['bright']}{coin}{COLORS['reset']} - "
                      f"{op_color}{operation}{COLORS['reset']} - "
                      f"{COLORS['red']}${pnl:.2f}{COLORS['reset']} "
                      f"({COLORS['red']}{roi:.2f}%{COLORS['reset']})")
                print(f"     Entry: ${entry_price:.6f} ({entry_time}) → Exit: ${exit_price:.6f} ({exit_time})")
        
        print()
    
    def _display_temporal_performance(self, temporal: Dict[str, Any]) -> None:
        """Display temporal performance analysis"""
        print(f"{COLORS['bright']}{COLORS['blue']}TEMPORAL PERFORMANCE{COLORS['reset']}")
        print(f"{COLORS['blue']}{'-' * 80}{COLORS['reset']}")
        
        if not temporal:
            print(f"{COLORS['yellow']}No temporal performance data available{COLORS['reset']}\n")
            return
        
        # Best/worst day
        if "best_day" in temporal:
            best_day = temporal["best_day"]
            best_date = best_day["date"]
            best_pnl = best_day["pnl"]
            best_trades = best_day["trades"]
            
            print(f"{COLORS['bright']}Best Trading Day:{COLORS['reset']} {best_date} - "
                  f"{COLORS['green']}+${best_pnl:.2f}{COLORS['reset']} from {best_trades} trades")
        
        if "worst_day" in temporal:
            worst_day = temporal["worst_day"]
            worst_date = worst_day["date"]
            worst_pnl = worst_day["pnl"]
            worst_trades = worst_day["trades"]
            
            print(f"{COLORS['bright']}Worst Trading Day:{COLORS['reset']} {worst_date} - "
                  f"{COLORS['red']}${worst_pnl:.2f}{COLORS['reset']} from {worst_trades} trades")
        
        # Best hour
        if "best_hour" in temporal:
            best_hour = temporal["best_hour"]
            hour = best_hour["hour"]
            pnl = best_hour["pnl"]
            trades = best_hour["trades"]
            
            print(f"{COLORS['bright']}Best Trading Hour:{COLORS['reset']} {hour} - "
                  f"{COLORS['green']}+${pnl:.2f}{COLORS['reset']} from {trades} trades")
        
        # Generate simple ASCII chart for hourly performance
        if "hourly" in temporal and temporal["hourly"]:
            print(f"\n{COLORS['bright']}Hourly Performance:{COLORS['reset']}")
            
            # Create sorted list of hours
            hourly_data = temporal["hourly"]
            sorted_hours = sorted([
                (hour, data["pnl"])
                for hour, data in hourly_data.items()
                if data["trades"] >= 2  # At least 2 trades for significance
            ], key=lambda x: int(x[0].split(":")[0]))  # Sort by hour
            
            if sorted_hours:
                # Find max PnL for scaling
                max_pnl = max(abs(pnl) for _, pnl in sorted_hours)
                scale_factor = 20 / max_pnl if max_pnl > 0 else 1
                
                for hour, pnl in sorted_hours:
                    # Calculate bar length
                    bar_length = int(abs(pnl) * scale_factor)
                    bar_length = max(1, min(bar_length, 40))  # Limit to 1-40 chars
                    
                    # Create the bar
                    if pnl >= 0:
                        bar = f"{COLORS['green']}{'█' * bar_length}{COLORS['reset']}"
                    else:
                        bar = f"{COLORS['red']}{'█' * bar_length}{COLORS['reset']}"
                    
                    # Format PnL
                    pnl_sign = "+" if pnl >= 0 else ""
                    pnl_color = COLORS["green"] if pnl >= 0 else COLORS["red"]
                    
                    print(f"  {hour}: {bar} {pnl_color}{pnl_sign}${pnl:.2f}{COLORS['reset']}")
        
        print()
    
    def _display_anomalies(self, anomalies: List[Dict[str, Any]]) -> None:
        """Display detected anomalies"""
        print(f"{COLORS['bright']}{COLORS['blue']}DETECTED ANOMALIES{COLORS['reset']}")
        print(f"{COLORS['blue']}{'-' * 80}{COLORS['reset']}")
        
        if not anomalies:
            print(f"{COLORS['yellow']}No anomalies detected{COLORS['reset']}\n")
            return
        
        for i, anomaly in enumerate(anomalies, 1):
            anomaly_type = anomaly.get("type", "unknown").replace("_", " ").title()
            description = anomaly.get("description", "No description")
            
            print(f"{i}. {COLORS['bright']}{anomaly_type}:{COLORS['reset']} {COLORS['yellow']}{description}{COLORS['reset']}")
        
        print()
    
    def _display_risk_reward(self, risk_reward: Dict[str, Any]) -> None:
        """Display risk-reward analysis"""
        print(f"{COLORS['bright']}{COLORS['blue']}RISK-REWARD ANALYSIS{COLORS['reset']}")
        print(f"{COLORS['blue']}{'-' * 80}{COLORS['reset']}")
        
        if not risk_reward:
            print(f"{COLORS['yellow']}No risk-reward data available{COLORS['reset']}\n")
            return
        
        # Profit-loss ratio
        ratio = risk_reward.get("avg_profit_loss_ratio", 0)
        ratio_color = COLORS["green"] if ratio >= 1.5 else COLORS["yellow"] if ratio >= 1 else COLORS["red"]
        
        print(f"{COLORS['bright']}Avg Profit/Loss Ratio:{COLORS['reset']} {ratio_color}{ratio:.2f}{COLORS['reset']} (target > 1.5)")
        
        # Maximum drawdown
        drawdown = risk_reward.get("max_drawdown", 0)
        print(f"{COLORS['bright']}Maximum Drawdown:{COLORS['reset']} {COLORS['red']}${drawdown:.2f}{COLORS['reset']}")
        
        # Risk per trade
        risk_per_trade = risk_reward.get("risk_per_trade", 0) * 100  # Convert to percentage
        print(f"{COLORS['bright']}Average Risk Per Trade:{COLORS['reset']} {risk_per_trade:.2f}%")
        
        # Sharpe ratio
        sharpe = risk_reward.get("sharpe_ratio", 0)
        sharpe_color = COLORS["green"] if sharpe >= 1 else COLORS["yellow"] if sharpe >= 0.5 else COLORS["red"]
        
        print(f"{COLORS['bright']}Sharpe Ratio:{COLORS['reset']} {sharpe_color}{sharpe:.2f}{COLORS['reset']} (target > 1.0)")
        
        # Volatility
        volatility = risk_reward.get("volatility", 0)
        print(f"{COLORS['bright']}Return Volatility:{COLORS['reset']} {volatility:.2f}%")
        
        print()
    
    def _display_recommendations(self, recommendations: List[Dict[str, Any]]) -> None:
        """Display recommendations"""
        print(f"{COLORS['bright']}{COLORS['blue']}RECOMMENDATIONS{COLORS['reset']}")
        print(f"{COLORS['blue']}{'-' * 80}{COLORS['reset']}")
        
        if not recommendations:
            print(f"{COLORS['yellow']}No recommendations available{COLORS['reset']}\n")
            return
        
        for i, rec in enumerate(recommendations, 1):
            source = rec.get("source", "system").replace("_", " ").title()
            recommendation = rec.get("recommendation", "")
            
            print(f"{i}. {COLORS['bright']}[{source}]{COLORS['reset']} {COLORS['green']}{recommendation}{COLORS['reset']}")
        
        print()
    
    def _display_report_footer(self, report: Dict[str, Any]) -> None:
        """Display report footer"""
        print(f"{COLORS['cyan']}{'=' * 80}{COLORS['reset']}")
        
        # Overall performance summary if available
        if "overall_performance" in report["analysis_results"]:
            perf = report["analysis_results"]["overall_performance"]
            
            total_trades = perf.get("total_trades", 0)
            win_rate = perf.get("win_rate", 0)
            total_pnl = perf.get("total_pnl", 0)
            roi = perf.get("roi", 0)
            
            win_rate_color = COLORS["green"] if win_rate >= 50 else COLORS["yellow"] if win_rate >= 40 else COLORS["red"]
            pnl_color = COLORS["green"] if total_pnl >= 0 else COLORS["red"]
            roi_color = COLORS["green"] if roi >= 0 else COLORS["red"]
            
            pnl_sign = "+" if total_pnl >= 0 else ""
            roi_sign = "+" if roi >= 0 else ""
            
            print(f"{COLORS['bright']}SUMMARY: {COLORS['reset']}"
                  f"{total_trades} trades, "
                  f"{win_rate_color}{win_rate:.1f}% win rate{COLORS['reset']}, "
                  f"{pnl_color}{pnl_sign}${total_pnl:.2f} PnL{COLORS['reset']} "
                  f"({roi_color}{roi_sign}{roi:.2f}% ROI{COLORS['reset']})")
        
        print(f"{COLORS['dim']}Generated by SentientTrader.AI Report Generator v2.0{COLORS['reset']}")
        print(f"{COLORS['cyan']}{'=' * 80}{COLORS['reset']}\n")
    
    def export_to_csv(self, report: Dict[str, Any], filename: Optional[str] = None) -> bool:
        """
        Export report data to CSV file
        
        Args:
            report (Dict[str, Any]): Report to export
            filename (Optional[str]): Custom filename, defaults to auto-generated
            
        Returns:
            bool: Whether export was successful
        """
        try:
            import csv
            
            # Create output directory if it doesn't exist
            os.makedirs(DEFAULT_REPORT_DIR, exist_ok=True)
            
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{DEFAULT_REPORT_PREFIX}_{timestamp}.csv"
                
            filepath = os.path.join(DEFAULT_REPORT_DIR, filename)
            
            # Extract trades data
            if "filtered_trades" in self.__dict__ and self.filtered_trades:
                trades = self.filtered_trades
            else:
                logger.warning("No filtered trades data available for CSV export")
                return False
            
            # Write to CSV
            with open(filepath, 'w', newline='') as f:
                # Determine fields from first trade
                if trades:
                    fields = list(trades[0].keys())
                else:
                    fields = ["trade_id", "coin", "strategy", "operation", "entry_price", "exit_price", 
                             "pnl", "roi", "status", "entry_time", "exit_time"]
                
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                
                for trade in trades:
                    writer.writerow(trade)
            
            logger.info(f"Report exported to CSV: {filepath}")
            print(f"{COLORS['green']}Report exported to CSV: {filepath}{COLORS['reset']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {str(e)}")
            print(f"{COLORS['red']}Error exporting to CSV: {str(e)}{COLORS['reset']}")
            return False
    
    def export_to_pdf(self, report: Dict[str, Any], filename: Optional[str] = None) -> bool:
        """
        Export report data to PDF file
        
        Args:
            report (Dict[str, Any]): Report to export
            filename (Optional[str]): Custom filename, defaults to auto-generated
            
        Returns:
            bool: Whether export was successful
        """
        if not PDF_EXPORT_AVAILABLE:
            logger.error("PDF export not available (ReportLab library not installed)")
            print(f"{COLORS['red']}PDF export not available. Install ReportLab library.{COLORS['reset']}")
            return False
        
        try:
            # Create output directory if it doesn't exist
            os.makedirs(DEFAULT_REPORT_DIR, exist_ok=True)
            
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{DEFAULT_REPORT_PREFIX}_{timestamp}.pdf"
                
            filepath = os.path.join(DEFAULT_REPORT_DIR, filename)
            
            # Create PDF
            c = canvas.Canvas(filepath, pagesize=letter)
            width, height = letter
            
            # Set up initial position
            y_position = height - 50
            line_height = 15
            
            # Helper function to add text
            def add_text(text, position, fontsize=12, bold=False):
                nonlocal y_position
                font = "Helvetica-Bold" if bold else "Helvetica"
                c.setFont(font, fontsize)
                c.drawString(position, y_position, text)
                y_position -= line_height
            
            # Add title
            c.setFont("Helvetica-Bold", 18)
            c.drawString(50, y_position, "SentientTrader.AI Trading Report")
            y_position -= 30
            
            # Add metadata
            add_text(f"Report Generated: {report['timestamp']}", 50, 10)
            add_text(f"Time Range: {report['time_range'].upper()}", 50, 10)
            if report["strategy_filter"]:
                add_text(f"Strategy: {report['strategy_filter']}", 50, 10)
            if report["coin_filter"]:
                add_text(f"Coin: {report['coin_filter']}", 50, 10)
            y_position -= 10
            
            # Add overall performance if available
            if "overall_performance" in report["analysis_results"]:
                perf = report["analysis_results"]["overall_performance"]
                
                add_text("OVERALL PERFORMANCE", 50, 14, True)
                y_position -= 5
                
                add_text(f"Total Trades: {perf.get('total_trades', 0)}", 60, 10)
                add_text(f"Win Rate: {perf.get('win_rate', 0):.1f}%", 60, 10)
                add_text(f"Total PnL: ${perf.get('total_pnl', 0):.2f}", 60, 10)
                add_text(f"ROI: {perf.get('roi', 0):.2f}%", 60, 10)
                y_position -= 10
            
            # Strategy performance
            if "strategy_performance" in report["analysis_results"]:
                strategy_perf = report["analysis_results"]["strategy_performance"]
                
                if strategy_perf:
                    add_text("STRATEGY PERFORMANCE", 50, 14, True)
                    y_position -= 5
                    
                    for strategy, data in strategy_perf.items():
                        display_name = strategy.replace("_strategy", "").upper()
                        add_text(f"{display_name}: {data.get('closed_trades', 0)} trades, "
                               f"{data.get('win_rate', 0):.1f}% win rate, ${data.get('total_pnl', 0):.2f} PnL", 60, 10)
                    
                    y_position -= 10
            
            # Check if we need a new page
            if y_position < 100:
                c.showPage()
                y_position = height - 50
            
            # Recommendations
            if "recommendations" in report["analysis_results"]:
                recommendations = report["analysis_results"]["recommendations"]
                
                if recommendations:
                    add_text("RECOMMENDATIONS", 50, 14, True)
                    y_position -= 5
                    
                    for i, rec in enumerate(recommendations, 1):
                        recommendation = rec.get("recommendation", "")
                        rec_text = f"{i}. {recommendation}"
                        
                        # Handle long recommendations
                        if len(rec_text) > 80:
                            # Split into multiple lines
                            words = rec_text.split()
                            lines = []
                            current_line = words[0]
                            
                            for word in words[1:]:
                                if len(current_line + " " + word) <= 80:
                                    current_line += " " + word
                                else:
                                    lines.append(current_line)
                                    current_line = word
                            
                            if current_line:
                                lines.append(current_line)
                            
                            # Add each line
                            for j, line in enumerate(lines):
                                indent = 60 if j > 0 else 60
                                add_text(line, indent, 10)
                        else:
                            add_text(rec_text, 60, 10)
                    
                    y_position -= 10
            
            # Save PDF
            c.save()
            
            logger.info(f"Report exported to PDF: {filepath}")
            print(f"{COLORS['green']}Report exported to PDF: {filepath}{COLORS['reset']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to PDF: {str(e)}")
            print(f"{COLORS['red']}Error exporting to PDF: {str(e)}{COLORS['reset']}")
            return False
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete report generation pipeline
        
        Returns:
            Dict[str, Any]: Generated report
        """
        try:
            # Load data
            if not self.load_data():
                logger.error("Failed to load data, can't generate report")
                return {}
            
            # Analyze data
            if not self.analyze_data():
                logger.error("Failed to analyze data, can't generate report")
                return {}
            
            # Generate report
            report = self.generate_report()
            
            # Store report in memory core if available
            if report and MEMORY_CORE_AVAILABLE:
                memory_core.add_memory_record(
                    source="report_generator",
                    category="trading_report",
                    data=report,
                    tags=["report", self.time_range, self.report_type]
                )
                logger.info("Report stored in memory core")
            
            # Alert if any critical issues found
            if ALERT_SYSTEM_AVAILABLE and report:
                if "anomalies" in report["analysis_results"] and report["analysis_results"]["anomalies"]:
                    anomalies = report["analysis_results"]["anomalies"]
                    alert_system.info(
                        f"Report generated with {len(anomalies)} detected anomalies",
                        {
                            "report_id": report["report_id"],
                            "time_range": report["time_range"],
                            "anomalies": [a["type"] for a in anomalies]
                        },
                        "report_generator"
                    )
            
            logger.info("Report generation completed successfully")
            return report
            
        except Exception as e:
            logger.error(f"Error running report generation: {str(e)}")
            
            if ALERT_SYSTEM_AVAILABLE:
                alert_system.error(
                    "Error generating trading report",
                    {"error": str(e)},
                    "report_generator"
                )
            
            return {}

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="SentientTrader.AI Report Generator")
    parser.add_argument("--time-range", choices=["day", "week", "month", "year", "all"], default="all",
                        help="Time range for the report")
    parser.add_argument("--report-type", choices=["summary", "detailed", "full"], default="full",
                        help="Type of report to generate")
    parser.add_argument("--strategy", help="Filter by strategy (long, short, sniper)")
    parser.add_argument("--coin", help="Filter by coin")
    parser.add_argument("--summary", action="store_true", help="Generate a summary report")
    parser.add_argument("--detailed", action="store_true", help="Generate a detailed report")
    parser.add_argument("--export-csv", action="store_true", help="Export report data to CSV")
    parser.add_argument("--export-pdf", action="store_true", help="Export report data to PDF")
    parser.add_argument("--output", help="Output filename for export")
    return parser.parse_args()

def main() -> int:
    """Main function for direct execution"""
    args = parse_arguments()
    
    # Determine report type
    report_type = args.report_type
    if args.summary:
        report_type = "summary"
    elif args.detailed:
        report_type = "detailed"
    
    try:
        # Initialize and run report generator
        generator = ReportGenerator(
            time_range=args.time_range,
            report_type=report_type,
            strategy_filter=args.strategy,
            coin_filter=args.coin
        )
        
        # Generate report
        print(f"{COLORS['cyan']}Generating trading report...{COLORS['reset']}")
        report = generator.run()
        
        if not report:
            print(f"{COLORS['red']}Failed to generate report{COLORS['reset']}")
            return 1
        
        # Display report
        generator.display_report(report)
        
        # Export if requested
        if args.export_csv:
            generator.export_to_csv(report, args.output)
        
        if args.export_pdf:
            generator.export_to_pdf(report, args.output)
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n{COLORS['yellow']}Process interrupted by user{COLORS['reset']}")
        return 130
    except Exception as e:
        print(f"{COLORS['red']}Unexpected error: {str(e)}{COLORS['reset']}")
        logger.exception("Unexpected error in main")
        return 1

if __name__ == "__main__":
    sys.exit(main())
        
