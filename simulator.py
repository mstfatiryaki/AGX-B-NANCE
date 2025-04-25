from random import choice
from random import choice
import random
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SentientTrader.AI - Trading Simulator v2
----------------------------------------
A comprehensive simulation environment that integrates GPT-4o and DeepSeek
for cryptocurrency trading strategy testing. This module uses real market data
with simulated capital to test trading strategies in a controlled environment.

Created by: SentientTrader.AI Team
Date: 2025-04-22
Version: 2.0.0
"""

import os
import sys
import json
import time
import random
import logging
import argparse
import datetime
import subprocess
import threading
import queue
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
import pandas as pd
import numpy as np

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

# Try importing system modules
try:
    from modules.utils import alert_system
    ALERT_SYSTEM_AVAILABLE = True
except ImportError:
    ALERT_SYSTEM_AVAILABLE = False
    logging.warning("alert_system module not available, using standard logging")

try:
    from modules.core import memory_core
    MEMORY_CORE_AVAILABLE = True
except ImportError:
    MEMORY_CORE_AVAILABLE = False
    logging.warning("memory_core module not available, memory features disabled")

try:
    from modules.utils import transaction_logger
    TRANSACTION_LOGGER_AVAILABLE = True
except ImportError:
    TRANSACTION_LOGGER_AVAILABLE = False
    logging.warning("transaction_logger module not available, using standard logging")

try:
    from modules.utils import terminal_interface
    TERMINAL_INTERFACE_AVAILABLE = True
except ImportError:
    TERMINAL_INTERFACE_AVAILABLE = False
    logging.warning("terminal_interface module not available, using standard output")

try:
    from modules.analysis import performance_tracker
    PERFORMANCE_TRACKER_AVAILABLE = True
except ImportError:
    PERFORMANCE_TRACKER_AVAILABLE = False
    logging.warning("performance_tracker module not available, performance metrics disabled")

try:
    from modules.ai import thinking_controller
    THINKING_CONTROLLER_AVAILABLE = True
except ImportError:
    THINKING_CONTROLLER_AVAILABLE = False
    logging.warning("thinking_controller module not available, AI decision analysis disabled")

try:
    from modules.ai import deep_thinking_process
    DEEP_THINKING_AVAILABLE = True
except ImportError:
    DEEP_THINKING_AVAILABLE = False
    logging.warning("deep_thinking_process module not available, DeepSeek thinking disabled")

try:
    from modules.learning import learning_engine
    LEARNING_ENGINE_AVAILABLE = True
except ImportError:
    LEARNING_ENGINE_AVAILABLE = False
    logging.warning("learning_engine module not available, learning features disabled")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, "logs", "simulator.log"))
    ]
)

logger = logging.getLogger("Simulator")

# Constants
VERSION = "2.0.0"
STRATEGY_DECISION_FILE = os.path.join(project_root, "data", "strategy_decision.json")
COINGECKO_DATA_FILE = os.path.join(project_root, "data", "coingecko_data.json")
BINANCE_DATA_FILE = os.path.join(project_root, "data", "binance_data.json")
SENTIMENT_DATA_FILE = os.path.join(project_root, "data", "sentiment_summary.json")
SIMULATION_REPORT_FILE = os.path.join(project_root, "data", "simulation_report.json")
SIMULATION_LOGS_DIR = os.path.join(project_root, "logs", "simulations")
THINKING_LOGS_DIR = os.path.join(project_root, "logs", "thinking")
DEFAULT_INITIAL_CAPITAL = 10000.0  # USD
DEFAULT_TRANSACTION_FEE = 0.0010  # 0.1% fee
DEFAULT_SLIPPAGE = 0.0005  # 0.05% slippage
MIN_SIMULATIONS = 10
MAX_SIMULATIONS = 20
THINKING_ANIMATION_FRAMES = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]


class SimulationMode(Enum):
    """Enum for different simulation modes."""
    BACKTEST = "backtest"  # Historical data testing
    PAPER = "paper"        # Real-time with fake money
    COMBINED = "combined"  # Mix of backtest and paper


class StrategyType(Enum):
    """Enum for different trading strategy types."""
    LONG = "long"             # Long position strategy
    SHORT = "short"           # Short position strategy
    SNIPER = "sniper"         # Quick entry/exit strategy
    SWING = "swing"           # Longer term trades
    SCALP = "scalp"           # Very short term trades
    GRID = "grid"             # Grid-based strategy
    DCA = "dca"               # Dollar cost averaging
    UNKNOWN = "unknown"       # Unrecognized strategy


class ThinkingLevel(Enum):
    """Enum for DeepSeek thinking levels."""
    MINIMAL = 1   # Basic analysis
    LOW = 2       # Low detail thinking
    MODERATE = 3  # Standard thinking depth
    HIGH = 4      # Detailed analysis
    MAXIMUM = 5   # Exhaustive analysis


class Simulator:
    """
    Trading simulator that integrates with GPT-4o and DeepSeek for decision making
    and uses real market data with simulated capital for strategy testing.
    """
    
    def __init__(self, 
                 real_mode: bool = False,
                 silent_mode: bool = False,
                 thinking_level: int = 3,
                 initial_capital: float = DEFAULT_INITIAL_CAPITAL,
                 transaction_fee: float = DEFAULT_TRANSACTION_FEE,
                 slippage: float = DEFAULT_SLIPPAGE,
                 simulation_mode: SimulationMode = SimulationMode.COMBINED):
        """
        Initialize the simulator.
        
        Args:
            real_mode (bool): Whether the system is using real money (always False in simulator)
            silent_mode (bool): Whether to minimize console output
            thinking_level (int): Level of DeepSeek thinking depth (1-5)
            initial_capital (float): Initial capital in USD
            transaction_fee (float): Transaction fee as a decimal (e.g., 0.001 for 0.1%)
            slippage (float): Slippage as a decimal (e.g., 0.0005 for 0.05%)
            simulation_mode (SimulationMode): Mode of simulation
        """
        # Force real_mode to False since this is a simulator
        self.real_mode = False
        if real_mode:
            logger.warning("Real mode requested but this is a simulator. Setting to simulation mode.")
            
        self.silent_mode = silent_mode
        self.thinking_level = ThinkingLevel(thinking_level) if 1 <= thinking_level <= 5 else ThinkingLevel.MODERATE
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.transaction_fee = transaction_fee
        self.slippage = slippage
        self.simulation_mode = simulation_mode
        self.strategy_decisions = {}
        self.market_data = {}
        self.sentiment_data = {}
        self.simulation_results = []
        self.trading_history = []
        self.thinking_history = []
        self.start_time = None
        self.end_time = None
        self.thinking_queue = queue.Queue()
        
        # Initialize system modules if available
        self._init_system_modules()
        
        # Create necessary directories
        os.makedirs(SIMULATION_LOGS_DIR, exist_ok=True)
        os.makedirs(THINKING_LOGS_DIR, exist_ok=True)
        
        # Load data files
        self.load_data_files()
        
        logger.info(f"Simulator v2 initialized (thinking_level: {self.thinking_level.name}, " 
                   f"capital: ${self.initial_capital:.2f}, mode: {self.simulation_mode.value})")
        
        if not self.silent_mode:
            self._print_to_terminal(f"🚀 Simulator v2 initialized with {self.thinking_level.name} thinking level")
    
    def _init_system_modules(self):
        """Initialize system modules if available."""
        # Initialize alert system
        if ALERT_SYSTEM_AVAILABLE:
            self.alert = alert_system.AlertSystem(
                module_name="simulator",
                real_mode=self.real_mode,
                log_to_file=True
            )
        else:
            self.alert = logger
            
        # Initialize memory core
        if MEMORY_CORE_AVAILABLE:
            self.memory = memory_core.MemoryCore(real_mode=self.real_mode)
        else:
            self.memory = None
            
        # Initialize transaction logger
        if TRANSACTION_LOGGER_AVAILABLE:
            self.transaction_logger = transaction_logger.TransactionLogger(
                module_name="simulator",
                real_mode=self.real_mode
            )
        else:
            self.transaction_logger = None
            
        # Initialize terminal interface
        if TERMINAL_INTERFACE_AVAILABLE:
            self.terminal = terminal_interface.TerminalInterface()
        else:
            self.terminal = None
            
        # Initialize performance tracker
        if PERFORMANCE_TRACKER_AVAILABLE:
            self.performance_tracker = performance_tracker.PerformanceTracker(
                real_mode=self.real_mode
            )
        else:
            self.performance_tracker = None
            
        # Initialize thinking controller
        if THINKING_CONTROLLER_AVAILABLE:
            self.thinking_controller = thinking_controller.ThinkingController(
                real_mode=self.real_mode,
                silent_mode=self.silent_mode
            )
            self.thinking_controller.start_monitoring()
        else:
            self.thinking_controller = None
            
        # Initialize deep thinking process
        if DEEP_THINKING_AVAILABLE:
            self.deep_thinking = deep_thinking_process.DeepThinkingProcess(
                real_mode=self.real_mode,
                silent_mode=self.silent_mode,
                thinking_level=self.thinking_level.value
            )
        else:
            self.deep_thinking = None
            
        # Initialize learning engine
        if LEARNING_ENGINE_AVAILABLE:
            self.learning_engine = learning_engine.LearningEngine(
                real_mode=self.real_mode
            )
        else:
            self.learning_engine = None
    
    def load_data_files(self):
        """Load necessary data files for simulation."""
        try:
            # Load strategy decisions
            if os.path.exists(STRATEGY_DECISION_FILE):
                with open(STRATEGY_DECISION_FILE, 'r') as f:
                    self.strategy_decisions = json.load(f)
                logger.info(f"Loaded {len(self.strategy_decisions)} strategy decisions")
            else:
                logger.warning(f"Strategy decision file not found: {STRATEGY_DECISION_FILE}")
                self.strategy_decisions = {}
            
            # Load market data from CoinGecko
            if os.path.exists(COINGECKO_DATA_FILE):
                with open(COINGECKO_DATA_FILE, 'r') as f:
                    coingecko_data = json.load(f)
                    self.market_data["coingecko"] = coingecko_data
                logger.info(f"Loaded CoinGecko data for {len(coingecko_data)} coins")
            else:
                logger.warning(f"CoinGecko data file not found: {COINGECKO_DATA_FILE}")
                self.market_data["coingecko"] = {}
            
            # Load market data from Binance
            if os.path.exists(BINANCE_DATA_FILE):
                with open(BINANCE_DATA_FILE, 'r') as f:
                    binance_data = json.load(f)
                    self.market_data["binance"] = binance_data
                logger.info(f"Loaded Binance data")
            else:
                logger.warning(f"Binance data file not found: {BINANCE_DATA_FILE}")
                self.market_data["binance"] = {}
            
            # Load sentiment data
            if os.path.exists(SENTIMENT_DATA_FILE):
                with open(SENTIMENT_DATA_FILE, 'r') as f:
                    self.sentiment_data = json.load(f)
                logger.info(f"Loaded sentiment data")
            else:
                logger.warning(f"Sentiment data file not found: {SENTIMENT_DATA_FILE}")
                self.sentiment_data = {}
                
        except Exception as e:
            logger.error(f"Error loading data files: {str(e)}")
            raise
    
    def run_simulation(self, num_simulations: int = None) -> Dict[str, Any]:
        """
        Run the trading simulation.
        
        Args:
            num_simulations (int): Number of simulation iterations to run
            
        Returns:
            Dict[str, Any]: Simulation results
        """
        self.start_time = datetime.datetime.now()
        
        # Determine number of simulations
        if num_simulations is None:
            # Random number between MIN and MAX
            num_simulations = random.randint(MIN_SIMULATIONS, MAX_SIMULATIONS)
        
        if not self.silent_mode:
            self._print_to_terminal(f"🔄 Starting simulation with {num_simulations} iterations")
        
        logger.info(f"Starting simulation with {num_simulations} iterations")
        
        # Reset simulation state
        self.current_capital = self.initial_capital
        self.simulation_results = []
        self.trading_history = []
        self.thinking_history = []
        
        # Start DeepSeek thinking in background thread
        thinking_thread = threading.Thread(target=self._background_thinking_worker)
        thinking_thread.daemon = True
        thinking_thread.start()
        
        # Process each strategy decision
        for coin, decision in self.strategy_decisions.items():
            if not isinstance(decision, dict):
                logger.warning(f"Invalid decision format for {coin}: {decision}")
                continue
                
            try:
                # Extract strategy information
                strategy_type = self._determine_strategy_type(decision)
                
                if not self.silent_mode:
                    self._print_to_terminal(
                        f"⚙️ Processing {coin} with {strategy_type.value} strategy"
                    )
                
                # Perform DeepSeek thinking before trading
                gpt_decision = self._prepare_gpt_decision(coin, decision, strategy_type)
                
                # Queue the thinking task
                self.thinking_queue.put({
                    'type': 'pre_trade',
                    'coin': coin,
                    'gpt_decision': gpt_decision,
                    'strategy_type': strategy_type.value,
                    'market_data': self._get_coin_market_data(coin),
                    'sentiment_data': self._get_coin_sentiment_data(coin)
                })
                
                # Display thinking animation
                if not self.silent_mode:
                    self._display_thinking_animation(f"DeepSeek analyzing {coin}")
                
                # Run the appropriate strategy based on the type
                num_trades = max(1, num_simulations // len(self.strategy_decisions))
                strategy_results = self._execute_strategy(
                    coin, 
                    decision, 
                    strategy_type, 
                    num_trades
                )
                
                # Perform DeepSeek thinking after trading
                self.thinking_queue.put({
                    'type': 'post_trade',
                    'coin': coin,
                    'strategy_type': strategy_type.value,
                    'strategy_results': strategy_results,
                    'market_data': self._get_coin_market_data(coin),
                    'sentiment_data': self._get_coin_sentiment_data(coin)
                })
                
                # Display thinking animation
                if not self.silent_mode:
                    self._display_thinking_animation(f"DeepSeek evaluating {coin} results")
                
                # Add to simulation results
                self.simulation_results.append({
                    'coin': coin,
                    'strategy_type': strategy_type.value,
                    'results': strategy_results,
                    'timestamp': datetime.datetime.now().isoformat()
                })
                
                # Update performance metrics
                self._update_performance_metrics(coin, strategy_results)
                
            except Exception as e:
                logger.error(f"Error simulating {coin}: {str(e)}")
                if ALERT_SYSTEM_AVAILABLE and hasattr(self.alert, 'error'):
                    self.alert.error(
                        f"Simulation error for {coin}: {str(e)}",
                        module="simulator",
                        category="simulation_error"
                    )
        
        # Signal the thinking thread to stop
        self.thinking_queue.put(None)
        thinking_thread.join(timeout=10.0)
        
        self.end_time = datetime.datetime.now()
        simulation_duration = (self.end_time - self.start_time).total_seconds()
        
        # Generate simulation report
        report = self._generate_simulation_report(simulation_duration)
        
        # Save simulation report
        self._save_simulation_report(report)
        
        # Send simulation results to learning engine
        self._send_to_learning_engine(report)
        
        if not self.silent_mode:
            self._print_simulation_summary(report)
        
        return report
    
    def _background_thinking_worker(self):
        """Worker for processing thinking tasks in background."""
        logger.info("Started background thinking worker")
        
        while True:
            try:
                # Get thinking task
                task = self.thinking_queue.get()
                
                # Check for termination signal
                if task is None:
                    logger.info("Stopping background thinking worker")
                    break
                
                # Process thinking task
                thinking_result = self._process_thinking_task(task)
                
                # Add to thinking history
                if thinking_result:
                    self.thinking_history.append(thinking_result)
                    
                    # Save thinking result to file
                    self._save_thinking_result(thinking_result)
                    
                    # Send to thinking controller if available
                    if (self.thinking_controller is not None and 
                        'gpt_decision' in task and 
                        'deepseek_decision' in thinking_result):
                        
                        self.thinking_controller.add_decision_pair(
                            task['gpt_decision'],
                            thinking_result['deepseek_decision']
                        )
                
                # Mark task as done
                self.thinking_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in thinking worker: {str(e)}")
                continue
    
    def _process_thinking_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a thinking task.
        
        Args:
            task (Dict): Thinking task
            
        Returns:
            Dict: Thinking result
        """
        if not DEEP_THINKING_AVAILABLE or self.deep_thinking is None:
            logger.warning("Deep thinking module not available, skipping thinking task")
            return None
        
        task_type = task.get('type', '')
        coin = task.get('coin', 'UNKNOWN')
        
        logger.info(f"Processing {task_type} thinking task for {coin}")
        
        try:
            if task_type == 'pre_trade':
                # Pre-trade thinking
                thinking_result = self.deep_thinking.process_pre_trade(
                    coin=coin,
                    gpt_decision=task.get('gpt_decision', {}),
                    strategy_type=task.get('strategy_type', ''),
                    market_data=task.get('market_data', {}),
                    sentiment_data=task.get('sentiment_data', {}),
                    thinking_level=self.thinking_level.value
                )
                
            elif task_type == 'post_trade':
                # Post-trade thinking
                thinking_result = self.deep_thinking.process_post_trade(
                    coin=coin,
                    strategy_type=task.get('strategy_type', ''),
                    strategy_results=task.get('strategy_results', {}),
                    market_data=task.get('market_data', {}),
                    sentiment_data=task.get('sentiment_data', {}),
                    thinking_level=self.thinking_level.value
                )
                
            else:
                logger.warning(f"Unknown thinking task type: {task_type}")
                return None
            
            # Add metadata to result
            thinking_result.update({
                'timestamp': datetime.datetime.now().isoformat(),
                'task_type': task_type,
                'coin': coin,
                'thinking_level': self.thinking_level.value
            })
            
            # Store in memory if available
            if MEMORY_CORE_AVAILABLE and self.memory:
                self.memory.store(
                    data=thinking_result,
                    category="deepseek_thinking",
                    subcategory=task_type,
                    source="simulator",
                    real_mode=self.real_mode
                )
            
            return thinking_result
            
        except Exception as e:
            logger.error(f"Error in thinking process: {str(e)}")
            return None
    
    def _determine_strategy_type(self, decision: Dict[str, Any]) -> StrategyType:
        """
        Determine the strategy type from the decision.
        
        Args:
            decision (Dict): Strategy decision
            
        Returns:
            StrategyType: Determined strategy type
        """
        strategy_str = decision.get('strategy', '').lower()
        
        if 'long' in strategy_str:
            return StrategyType.LONG
        elif 'short' in strategy_str:
            return StrategyType.SHORT
        elif 'sniper' in strategy_str:
            return StrategyType.SNIPER
        elif 'swing' in strategy_str:
            return StrategyType.SWING
        elif 'scalp' in strategy_str:
            return StrategyType.SCALP
        elif 'grid' in strategy_str:
            return StrategyType.GRID
        elif 'dca' in strategy_str:
            return StrategyType.DCA
        else:
            return StrategyType.UNKNOWN
    
    def _prepare_gpt_decision(self, 
                             coin: str, 
                             decision: Dict[str, Any], 
                             strategy_type: StrategyType) -> Dict[str, Any]:
        """
        Prepare GPT decision object for DeepSeek analysis.
        
        Args:
            coin (str): Coin symbol
            decision (Dict): Strategy decision
            strategy_type (StrategyType): Strategy type
            
        Returns:
            Dict: GPT decision object
        """
        # Extract information from decision
        entry_price = decision.get('entry_price', 0.0)
        target_price = decision.get('target_price', 0.0)
        stop_loss = decision.get('stop_loss', 0.0)
        
        # Calculate position size based on strategy type
        position_size = self._calculate_position_size(strategy_type, coin)
        
        # Calculate risk ratio
        risk_ratio = 0.0
        if entry_price > 0 and stop_loss > 0:
            if strategy_type == StrategyType.LONG:
                risk_ratio = abs(entry_price - stop_loss) / entry_price
            elif strategy_type == StrategyType.SHORT:
                risk_ratio = abs(stop_loss - entry_price) / entry_price
        
        # Determine direction based on strategy type
        if strategy_type in [StrategyType.LONG, StrategyType.SWING, StrategyType.DCA]:
            direction = "long"
        elif strategy_type in [StrategyType.SHORT]:
            direction = "short"
        elif strategy_type in [StrategyType.SNIPER, StrategyType.SCALP]:
            direction = "sniper"
        else:
            direction = "unknown"
        
        # Create GPT decision object
        gpt_decision = {
            'action': 'buy' if direction == "long" else 'sell' if direction == "short" else 'snipe',
            'symbol': coin,
            'direction': direction,
            'size': position_size,
            'stop_loss': stop_loss,
            'take_profit': target_price,
            'timeframe': decision.get('timeframe', '1h'),
            'urgency': decision.get('urgency', 5),
            'type': 'entry',
            'reasoning': decision.get('reasoning', ''),
            'confidence': decision.get('confidence', 0.7),
            'risk_ratio': risk_ratio,
            'strategy_type': strategy_type.value,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        return gpt_decision
    
    def _execute_strategy(self, 
                         coin: str, 
                         decision: Dict[str, Any], 
                         strategy_type: StrategyType, 
                         num_trades: int) -> Dict[str, Any]:
        """
        Execute a trading strategy.
        
        Args:
            coin (str): Coin symbol
            decision (Dict): Strategy decision
            strategy_type (StrategyType): Strategy type
            num_trades (int): Number of trades to simulate
            
        Returns:
            Dict: Strategy execution results
        """
        logger.info(f"Executing {strategy_type.value} strategy for {coin} with {num_trades} trades")
        
        # Get coin market data
        market_data = self._get_coin_market_data(coin)
        if not market_data:
            logger.warning(f"No market data available for {coin}")
            return {
                'success': False,
                'error': 'No market data available',
                'trades': []
            }
        
        # Initialize strategy parameters
        entry_price = decision.get('entry_price', 0.0)
        if entry_price <= 0:
            entry_price = self._get_current_price(coin)
            
        target_price = decision.get('target_price', 0.0)
        stop_loss = decision.get('stop_loss', 0.0)
        
        # Calculate values if not provided
        if target_price <= 0 or stop_loss <= 0:
            target_price, stop_loss = self._calculate_targets(
                entry_price, 
                strategy_type,
                decision
            )
        
        # Initialize results
        trades = []
        total_profit_loss = 0.0
        win_count = 0
        loss_count = 0
        
        # Execute trades
        for i in range(num_trades):
            # Simulate a single trade
            trade_result = self._simulate_trade(
                coin, 
                entry_price, 
                target_price, 
                stop_loss, 
                strategy_type,
                i
            )
            
            # Add to trade history
            trade_id = f"{coin}_{strategy_type.value}_{int(time.time())}_{i}"
            trade_result['id'] = trade_id
            trades.append(trade_result)
            self.trading_history.append(trade_result)
            
            # Update statistics
            total_profit_loss += trade_result['pnl_amount']
            if trade_result['pnl_amount'] > 0:
                win_count += 1
            else:
                loss_count += 1
            
            # Log transaction if available
            if TRANSACTION_LOGGER_AVAILABLE and self.transaction_logger:
                self.transaction_logger.log_transaction(
                    transaction_type=trade_result['result'].lower(),
                    coin=coin,
                    amount=trade_result['position_size'],
                    price=trade_result['exit_price'],
                    pnl=trade_result['pnl_amount'],
                    strategy=strategy_type.value,
                    real_mode=self.real_mode
                )
            
            # Update current capital
            self.current_capital += trade_result['pnl_amount']
            
            # Introduce delay between trades
            time.sleep(0.1)
        
        # Calculate statistics
        win_rate = win_count / num_trades if num_trades > 0 else 0
        avg_profit_loss = total_profit_loss / num_trades if num_trades > 0 else 0
        
        # Prepare strategy results
        strategy_results = {
            'success': True,
            'coin': coin,
            'strategy_type': strategy_type.value,
            'entry_price': entry_price,
            'target_price': target_price,
            'stop_loss': stop_loss,
            'num_trades': num_trades,
            'total_profit_loss': total_profit_loss,
            'win_count': win_count,
            'loss_count': loss_count,
            'win_rate': win_rate,
            'avg_profit_loss': avg_profit_loss,
            'trades': trades,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        return strategy_results
    
    def _simulate_trade(self, 
                       coin: str, 
                       entry_price: float, 
                       target_price: float, 
                       stop_loss: float, 
                       strategy_type: StrategyType,
                       trade_index: int) -> Dict[str, Any]:
        """
        Simulate a single trade.
        
        Args:
            coin (str): Coin symbol
            entry_price (float): Entry price
            target_price (float): Target price
            stop_loss (float): Stop loss price
            strategy_type (StrategyType): Strategy type
            trade_index (int): Trade index
            
        Returns:
            Dict: Trade simulation result
        """
        # Calculate position size
        position_size = self._calculate_position_size(strategy_type, coin)
        position_value = position_size * entry_price
        
        # Calculate leverage based on strategy
        leverage = self._calculate_leverage(strategy_type)
        
        # Determine trade direction
        is_long = strategy_type in [StrategyType.LONG, StrategyType.SWING, StrategyType.DCA, StrategyType.SNIPER]
        
        # Prepare trade parameters
        entry_time = datetime.datetime.now() - datetime.timedelta(minutes=random.randint(10, 60))
        holding_duration = self._calculate_holding_duration(strategy_type)
        exit_time = entry_time + datetime.timedelta(minutes=holding_duration)
        
        # Simulate market movement
        exit_price = self._simulate_price_movement(
            entry_price, 
            target_price, 
            stop_loss, 
            is_long,
            strategy_type,
            trade_index
        )
        
        # Calculate fees
        entry_fee = position_value * self.transaction_fee
        exit_fee = position_value * (exit_price / entry_price) * self.transaction_fee
        total_fees = entry_fee + exit_fee
        
        # Calculate profit/loss
        if is_long:
            pnl_percentage = (exit_price - entry_price) / entry_price
        else:
            pnl_percentage = (entry_price - exit_price) / entry_price
            
        # Apply leverage
        pnl_percentage = pnl_percentage * leverage
        
        # Apply fees
        pnl_percentage = pnl_percentage - (total_fees / position_value)
        
        # Calculate absolute PnL
        pnl_amount = position_value * pnl_percentage
        
        # Determine result
        if exit_price == target_price:
            result = "TARGET_HIT"
        elif exit_price == stop_loss:
            result = "STOP_LOSS"
        else:
            result = "MARKET_EXIT"
        
        # Create trade result
        trade_result = {
            'entry_time': entry_time.isoformat(),
            'exit_time': exit_time.isoformat(),
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size': position_size,
            'position_value': position_value,
            'leverage': leverage,
            'is_long': is_long,
            'pnl_percentage': pnl_percentage,
            'pnl_amount': pnl_amount,
            'fees': total_fees,
            'result': result,
            'holding_duration_minutes': holding_duration,
            'strategy_type': strategy_type.value,
            'coin': coin,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        return trade_result
    
    def _calculate_targets(self, 
                          entry_price: float, 
                          strategy_type: StrategyType,
                          decision: Dict[str, Any]) -> Tuple[float, float]:
        """
        Calculate target price and stop loss if not provided.
        
        Args:
            entry_price (float): Entry price
            strategy_type (StrategyType): Strategy type
            decision (Dict): Strategy decision
            
        Returns:
            Tuple[float, float]: Target price and stop loss
        """
        # Default risk-reward ratios based on strategy type
        risk_reward_ratios = {
            StrategyType.LONG: 2.0,
            StrategyType.SHORT: 2.0,
            StrategyType.SNIPER: 3.0,
            StrategyType.SWING: 2.5,
            StrategyType.SCALP: 1.5,
            StrategyType.GRID: 1.0,
            StrategyType.DCA: 1.5,
            StrategyType.UNKNOWN: 2.0
        }
        
        # Default stop percentages based on strategy type
        stop_percentages = {
            StrategyType.LONG: 0.05,
            StrategyType.SHORT: 0.05,
            StrategyType.SNIPER: 0.03,
            StrategyType.SWING: 0.10,
            StrategyType.SCALP: 0.02,
            StrategyType.GRID: 0.08,
            StrategyType.DCA: 0.15,
            StrategyType.UNKNOWN: 0.05
        }
        
        # Get risk-reward ratio and stop percentage for this strategy
        risk_reward_ratio = risk_reward_ratios.get(strategy_type, 2.0)
        stop_percentage = stop_percentages.get(strategy_type, 0.05)
        
        # Adjust based on volatility if available
        volatility = decision.get('volatility', 0.0)
        if volatility > 0:
            stop_percentage = min(0.2, stop_percentage * (1 + volatility))
        
        # Calculate target and stop
        if strategy_type in [StrategyType.LONG, StrategyType.SWING, StrategyType.DCA, StrategyType.SNIPER]:
            # Long strategies
            stop_loss = entry_price * (1 - stop_percentage)
            price_change = entry_price * stop_percentage * risk_reward_ratio
            target_price = entry_price + price_change
        else:
            # Short strategies
            stop_loss = entry_price * (1 + stop_percentage)
            price_change = entry_price * stop_percentage * risk_reward_ratio
            target_price = entry_price - price_change
            
        return target_price, stop_loss
    
    def _calculate_position_size(self, strategy_type: StrategyType, coin: str) -> float:
        """
        Calculate position size based on strategy type and risk parameters.
        
        Args:
            strategy_type (StrategyType): Strategy type
            coin (str): Coin symbol
            
        Returns:
            float: Position size as a percentage of capital (0.0-1.0)
        """
        # Base position sizes for different strategies
        base_sizes = {
            StrategyType.LONG: 0.3,    # 30% of capital
            StrategyType.SHORT: 0.25,  # 25% of capital
            StrategyType.SNIPER: 0.4,  # 40% of capital
            StrategyType.SWING: 0.2,   # 20% of capital
            StrategyType.SCALP: 0.5,   # 50% of capital
            StrategyType.GRID: 0.15,   # 15% of capital
            StrategyType.DCA: 0.1,     # 10% of capital
            StrategyType.UNKNOWN: 0.1  # 10% of capital
        }
        
        # Get base size for this strategy
        base_size = base_sizes.get(strategy_type, 0.1)
        
        # Adjust based on sentiment if available
        sentiment_score = self._get_coin_sentiment_score(coin)
        if sentiment_score is not None:
            if strategy_type in [StrategyType.LONG, StrategyType.SWING, StrategyType.DCA, StrategyType.SNIPER]:
                # For long strategies, increase size with positive sentiment
                sentiment_adjustment = (sentiment_score - 0.5) * 0.4  # -0.2 to +0.2
            else:
                # For short strategies, increase size with negative sentiment
                sentiment_adjustment = (0.5 - sentiment_score) * 0.4  # -0.2 to +0.2
                
            base_size = max(0.05, min(0.7, base_size + sentiment_adjustment))
        
        # Add some randomness (±10%)
        random_factor = 1.0 + random.uniform(-0.1, 0.1)
        size = base_size * random_factor
        
        # Ensure size is within reasonable bounds
        size = max(0.05, min(0.7, size))
        
        return size
    
    def _calculate_leverage(self, strategy_type: StrategyType) -> float:
        """
        Calculate leverage based on strategy type.
        
        Args:
            strategy_type (StrategyType): Strategy type
            
        Returns:
            float: Leverage multiplier
        """
        # Base leverage for different strategies
        base_leverage = {
            StrategyType.LONG: 1.5,
            StrategyType.SHORT: 2.0,
            StrategyType.SNIPER: 3.0,
            StrategyType.SWING: 1.0,
            StrategyType.SCALP: 5.0,
            StrategyType.GRID: 1.0,
            StrategyType.DCA: 1.0,
            StrategyType.UNKNOWN: 1.0
        }
        
        # Get base leverage for this strategy
        leverage = base_leverage.get(strategy_type, 1.0)
        
        # Add some randomness (±20%)
        random_factor = 1.0 + random.uniform(-0.2, 0.2)
        leverage = leverage * random_factor
        
        # Round to one decimal place
        leverage = round(leverage, 1)
        
        # Ensure leverage is within reasonable bounds
        leverage = max(1.0, min(10.0, leverage))
        
        return leverage
    
    def _calculate_holding_duration(self, strategy_type: StrategyType) -> int:
        """
        Calculate holding duration in minutes based on strategy type.
        
        Args:
            strategy_type (StrategyType): Strategy type
            
        Returns:
            int: Holding duration in minutes
        """
        # Base durations for different strategies (in minutes)
        base_durations = {
            StrategyType.LONG: 60 * 24,     # 1 day
            StrategyType.SHORT: 60 * 12,    # 12 hours
            StrategyType.SNIPER: 60,        # 1 hour
            StrategyType.SWING: 60 * 72,    # 3 days
            StrategyType.SCALP: 15,         # 15 minutes
            StrategyType.GRID: 60 * 48,     # 2 days
            StrategyType.DCA: 60 * 24 * 7,  # 1 week
            StrategyType.UNKNOWN: 60 * 24   # 1 day
        }
        
        # Get base duration for this strategy
        base_duration = base_durations.get(strategy_type, 60 * 24)
        
        # Add some randomness (±30%)
        random_factor = 1.0 + random.uniform(-0.3, 0.3)
        duration = int(base_duration * random_factor)
        
        # Ensure duration is positive
        duration = max(1, duration)
        
        return duration
    
    def _simulate_price_movement(self, 
                                entry_price: float, 
                                target_price: float, 
                                stop_loss: float, 
                                is_long: bool,
                                strategy_type: StrategyType,
                                trade_index: int) -> float:
        """
        Simulate price movement to determine exit price.
        
        Args:
            entry_price (float): Entry price
            target_price (float): Target price
            stop_loss (float): Stop loss price
            is_long (bool): Whether this is a long position
            strategy_type (StrategyType): Strategy type
            trade_index (int): Trade index for deterministic randomness
            
        Returns:
            float: Exit price
        """
        # Win rates for different strategies
        win_rates = {
            StrategyType.LONG: 0.55,
            StrategyType.SHORT: 0.52,
            StrategyType.SNIPER: 0.65,
            StrategyType.SWING: 0.58,
            StrategyType.SCALP: 0.70,
            StrategyType.GRID: 0.48,
            StrategyType.DCA: 0.62,
            StrategyType.UNKNOWN: 0.50
        }
        
        # Get win rate for this strategy
        win_rate = win_rates.get(strategy_type, 0.5)
        
        # Use trade index for deterministic randomness
        random.seed(int(time.time()) + trade_index)
        outcome = random.random()
        
        if outcome < win_rate:
            # Win - hit target
            exit_price = target_price
        elif outcome < win_rate + ((1 - win_rate) * 0.7):
            # Loss - hit stop loss
            exit_price = stop_loss
        else:
            # Random exit between entry and target/stop
            if is_long:
                price_range = [
                    entry_price, 
                    entry_price + (target_price - entry_price) * 0.3,
                    entry_price - (entry_price - stop_loss) * 0.7
                ]
            else:
                price_range = [
                    entry_price,
                    entry_price - (entry_price - target_price) * 0.3,
                    entry_price + (stop_loss - entry_price) * 0.7
                ]
                
            exit_price = random.choice(price_range)
        
        return exit_price
    
    def _get_current_price(self, coin: str) -> float:
        """
        Get current price for a coin.
        
        Args:
            coin (str): Coin symbol
            
        Returns:
            float: Current price
        """
        # Try to get price from market data
        market_data = self._get_coin_market_data(coin)
        
        if market_data:
            return market_data.get('current_price', 0.0)
        
        # Fallback to default price
        return 0.0
    
    def _get_coin_market_data(self, coin: str) -> Dict[str, Any]:
        """
        Get market data for a coin.
        
        Args:
            coin (str): Coin symbol
            
        Returns:
            Dict: Market data for the coin
        """
        # Try to get data from CoinGecko
        coingecko_data = self.market_data.get("coingecko", {})
        if coin in coingecko_data:
            return coingecko_data[coin]
        
        # Try to get data from Binance
        binance_data = self.market_data.get("binance", {})
        if "symbols" in binance_data:
            for symbol in binance_data["symbols"]:
                if symbol.get("baseAsset", "").upper() == coin.upper():
                    return {
                        "current_price": float(symbol.get("price", 0.0)),
                        "market_cap": 0.0,
                        "volume": float(symbol.get("volume", 0.0)),
                        "price_change_24h": float(symbol.get("priceChange", 0.0)),
                        "price_change_percentage_24h": float(symbol.get("priceChangePercent", 0.0))
                    }
        
        # No data found
        return {}
    
    def _get_coin_sentiment_data(self, coin: str) -> Dict[str, Any]:
        """
        Get sentiment data for a coin.
        
        Args:
            coin (str): Coin symbol
            
        Returns:
            Dict: Sentiment data for the coin
        """
        # Check if we have sentiment data for this coin
        if coin in self.sentiment_data:
            return self.sentiment_data[coin]
        
        # No data found
        return {}
    
    def _get_coin_sentiment_score(self, coin: str) -> Optional[float]:
        """
        Get sentiment score for a coin.
        
        Args:
            coin (str): Coin symbol
            
        Returns:
            Optional[float]: Sentiment score (0.0-1.0) or None if not available
        """
        sentiment_data = self._get_coin_sentiment_data(coin)
        
        if "sentiment_score" in sentiment_data:
            return float(sentiment_data["sentiment_score"])
        
        if "sentiment" in sentiment_data:
            sentiment = sentiment_data["sentiment"]
            if isinstance(sentiment, (int, float)):
                return float(sentiment)
            elif isinstance(sentiment, str):
                sentiment_mapping = {
                    "very_bullish": 0.9,
                    "bullish": 0.7,
                    "neutral": 0.5,
                    "bearish": 0.3,
                    "very_bearish": 0.1
                }
                return sentiment_mapping.get(sentiment.lower(), 0.5)
        
        return None
    
    def _update_performance_metrics(self, coin: str, strategy_results: Dict[str, Any]) -> None:
        """
        Update performance metrics for a strategy.
        
        Args:
            coin (str): Coin symbol
            strategy_results (Dict): Strategy execution results
        """
        if not PERFORMANCE_TRACKER_AVAILABLE or not self.performance_tracker:
            return
        
        try:
            # Extract performance data
            strategy_type = strategy_results.get('strategy_type', 'unknown')
            total_pnl = strategy_results.get('total_profit_loss', 0.0)
            win_rate = strategy_results.get('win_rate', 0.0)
            num_trades = strategy_results.get('num_trades', 0)
            
            # Update performance tracker
            self.performance_tracker.update_strategy_performance(
                coin=coin,
                strategy=strategy_type,
                pnl=total_pnl,
                win_rate=win_rate,
                trade_count=num_trades,
                timestamp=datetime.datetime.now().isoformat()
            )
            
            logger.debug(f"Updated performance metrics for {coin} {strategy_type}")
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")
    
    def _generate_simulation_report(self, duration: float) -> Dict[str, Any]:
        """
        Generate a comprehensive simulation report.
        
        Args:
            duration (float): Simulation duration in seconds
            
        Returns:
            Dict: Simulation report
        """
        # Calculate overall statistics
        total_pnl = 0.0
        total_trades = 0
        win_count = 0
        loss_count = 0
        
        for result in self.simulation_results:
            total_pnl += result.get('results', {}).get('total_profit_loss', 0.0)
            total_trades += result.get('results', {}).get('num_trades', 0)
            win_count += result.get('results', {}).get('win_count', 0)
            loss_count += result.get('results', {}).get('loss_count', 0)
        
        # Calculate win rate
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        # Calculate ROI
        roi = total_pnl / self.initial_capital if self.initial_capital > 0 else 0
        
        # Prepare strategy performances
        strategy_performances = {}
        for result in self.simulation_results:
            strategy_type = result.get('strategy_type', 'unknown')
            coin = result.get('coin', 'unknown')
            key = f"{coin}_{strategy_type}"
            
            strategy_performances[key] = {
                'coin': coin,
                'strategy_type': strategy_type,
                'total_pnl': result.get('results', {}).get('total_profit_loss', 0.0),
                'win_rate': result.get('results', {}).get('win_rate', 0.0),
                'num_trades': result.get('results', {}).get('num_trades', 0)
            }
        
        # Prepare report
        report = {
            'timestamp': datetime.datetime.now().isoformat(),
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': duration,
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
            'total_pnl': total_pnl,
            'roi': roi,
            'total_trades': total_trades,
            'win_count': win_count,
            'loss_count': loss_count,
            'win_rate': win_rate,
            'strategy_performances': strategy_performances,
            'thinking_level': self.thinking_level.value,
            'simulation_mode': self.simulation_mode.value,
            'version': VERSION
        }
        
        return report
    
    def _save_simulation_report(self, report: Dict[str, Any]) -> None:
        """
        Save simulation report to file.
        
        Args:
            report (Dict): Simulation report
        """
        try:
            # Save to simulation report file
            with open(SIMULATION_REPORT_FILE, 'w') as f:
                json.dump(report, f, indent=2)
                
            logger.info(f"Saved simulation report to {SIMULATION_REPORT_FILE}")
            
            # Save detailed logs for each coin
            for result in self.simulation_results:
                coin = result.get('coin', 'unknown')
                strategy_type = result.get('strategy_type', 'unknown')
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                
                log_file = os.path.join(
                    SIMULATION_LOGS_DIR, 
                    f"{coin}_{strategy_type}_{timestamp}.json"
                )
                
                with open(log_file, 'w') as f:
                    json.dump(result, f, indent=2)
                    
                logger.debug(f"Saved detailed log for {coin} to {log_file}")
                
        except Exception as e:
            logger.error(f"Error saving simulation report: {str(e)}")
    
    def _save_thinking_result(self, thinking_result: Dict[str, Any]) -> None:
        """
        Save thinking result to file.
        
        Args:
            thinking_result (Dict): Thinking result
        """
        try:
            # Create timestamp and file name
            coin = thinking_result.get('coin', 'unknown')
            task_type = thinking_result.get('task_type', 'unknown')
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            thinking_file = os.path.join(
                THINKING_LOGS_DIR,
                f"{coin}_{task_type}_{timestamp}.json"
            )
            
            # Save to file
            with open(thinking_file, 'w') as f:
                json.dump(thinking_result, f, indent=2)
                
            logger.debug(f"Saved thinking result for {coin} to {thinking_file}")
            
        except Exception as e:
            logger.error(f"Error saving thinking result: {str(e)}")
    
    def _send_to_learning_engine(self, report: Dict[str, Any]) -> None:
        """
        Send simulation results to learning engine.
        
        Args:
            report (Dict): Simulation report
        """
        if not LEARNING_ENGINE_AVAILABLE or not self.learning_engine:
            return
        
        try:
            # Extract relevant data
            training_data = {
                'timestamp': report.get('timestamp'),
                'strategies': [],
                'overall_performance': {
                    'win_rate': report.get('win_rate', 0.0),
                    'roi': report.get('roi', 0.0),
                    'total_trades': report.get('total_trades', 0)
                }
            }
            
            # Add strategy data
            for strategy_key, performance in report.get('strategy_performances', {}).items():
                training_data['strategies'].append({
                    'coin': performance.get('coin'),
                    'strategy_type': performance.get('strategy_type'),
                    'win_rate': performance.get('win_rate', 0.0),
                    'pnl': performance.get('total_pnl', 0.0),
                    'num_trades': performance.get('num_trades', 0)
                })
            
            # Send to learning engine
            self.learning_engine.add_simulation_results(training_data)
            
            logger.info("Sent simulation results to learning engine")
            
        except Exception as e:
            logger.error(f"Error sending to learning engine: {str(e)}")
    
    def _print_to_terminal(self, message: str) -> None:
        """
        Print a message to the terminal.
        
        Args:
            message (str): Message to print
        """
        if self.silent_mode:
            return
            
        if TERMINAL_INTERFACE_AVAILABLE and self.terminal:
            self.terminal.print(message, module="simulator")
        else:
            print(message)
    
    def _display_thinking_animation(self, message: str, duration: float = 2.0) -> None:
        """
        Display a thinking animation in the terminal.
        
        Args:
            message (str): Message to display
            duration (float): Duration in seconds
        """
        if self.silent_mode:
            return
            
        start_time = time.time()
        i = 0
        
        try:
            while time.time() - start_time < duration:
                frame = THINKING_ANIMATION_FRAMES[i % len(THINKING_ANIMATION_FRAMES)]
                print(f"\r{frame} {message}...", end="", flush=True)
                time.sleep(0.1)
                i += 1
                
            # Clear the line
            print("\r" + " " * (len(message) + 15) + "\r", end="", flush=True)
            
        except Exception:
            # In case of any error (like KeyboardInterrupt), ensure we clear the line
            print("\r" + " " * (len(message) + 15) + "\r", end="", flush=True)
    
    def _print_simulation_summary(self, report: Dict[str, Any]) -> None:
        """
        Print a summary of the simulation results.
        
        Args:
            report (Dict): Simulation report
        """
        if self.silent_mode:
            return
            
        print("\n" + "=" * 50)
        print(f"  SIMULATION SUMMARY")
        print("=" * 50)
        
        # Basic information
        duration = report.get('duration_seconds', 0)
        duration_str = f"{duration:.1f} seconds"
        
        print(f"Duration: {duration_str}")
        print(f"Initial Capital: ${report.get('initial_capital', 0):.2f}")
        print(f"Final Capital: ${report.get('final_capital', 0):.2f}")
        print(f"Total P&L: ${report.get('total_pnl', 0):.2f} ({report.get('roi', 0) * 100:.2f}%)")
        print(f"Total Trades: {report.get('total_trades', 0)}")
        print(f"Win Rate: {report.get('win_rate', 0) * 100:.2f}%")
        print(f"Win/Loss: {report.get('win_count', 0)}/{report.get('loss_count', 0)}")
        
        # Strategy performances
        print("\nStrategy Performances:")
        print("-" * 50)
        print(f"{'Coin':<8} {'Strategy':<10} {'PnL':>10} {'Win Rate':>10} {'Trades':>8}")
        print("-" * 50)
        
        for strategy_key, performance in report.get('strategy_performances', {}).items():
            coin = performance.get('coin', '')
            strategy_type = performance.get('strategy_type', '')
            pnl = performance.get('total_pnl', 0.0)
            win_rate = performance.get('win_rate', 0.0)
            num_trades = performance.get('num_trades', 0)
            
            print(f"{coin:<8} {strategy_type:<10} ${pnl:>9.2f} {win_rate*100:>9.2f}% {num_trades:>8}")
            
        print("=" * 50)
        print(f"Report saved to: {SIMULATION_REPORT_FILE}")
        print("=" * 50 + "\n")


def main():
    """Main function to run the simulator."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='SentientTrader.AI Simulator v2')
    parser.add_argument('--real', action='store_true', help='Run in real mode (ignored in simulator)')
    parser.add_argument('--silent', action='store_true', help='Run in silent mode (minimal output)')
    parser.add_argument('--thinking-level', type=int, default=3, choices=range(1, 6),
                       help='DeepSeek thinking level (1-5, default: 3)')
    parser.add_argument('--simulations', type=int, default=None,
                       help='Number of simulations to run (default: random between 10-20)')
    parser.add_argument('--capital', type=float, default=DEFAULT_INITIAL_CAPITAL,
                       help=f'Initial capital in USD (default: {DEFAULT_INITIAL_CAPITAL})')
    args = parser.parse_args()
    
    # Print header
    print("\n" + "=" * 60)
    print("  SentientTrader.AI - Simulator v2")
    print("=" * 60)
    print(f"  Thinking Level: {args.thinking_level}")
    print(f"  Initial Capital: ${args.capital:.2f}")
    print(f"  Silent Mode: {'Yes' if args.silent else 'No'}")
    print("=" * 60 + "\n")
    
    # Initialize simulator
    simulator = Simulator(
        real_mode=False,  # Always false in simulator
        silent_mode=args.silent,
        thinking_level=args.thinking_level,
        initial_capital=args.capital
    )
    
    try:
        # Run simulation
        report = simulator.run_simulation(args.simulations)
        
        # Exit with success code
        sys.exit(0)
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nError during simulation: {str(e)}")
        sys.exit(2)


if __name__ == "__main__":
    main()
