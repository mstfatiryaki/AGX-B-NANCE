#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SentientTrader.AI V2 Main Control Module
Main AGX (Advanced Guidance eXecutor) - System Control Module
Created by mstfatiryaki
"""

import random
import os
import sys
import json
import time
import logging
import argparse
import subprocess
import platform
import signal
from datetime import datetime, timedelta
from colorama import Fore, Back, Style, init

# Initialize colorama
init(autoreset=True)

# Add project root to path
PROJECT_ROOT = os.getcwd()
sys.path.append(PROJECT_ROOT)

# Import SentientTrader.AI modules
try:
    from modules.utils import alert_system
    from modules.core import memory_core
    from modules.interface import terminal_interface
    MODULES_IMPORTED = True
except ImportError as e:
    MODULES_IMPORTED = False
    print(f"{Fore.RED}Error importing modules: {str(e)}")
    print(f"{Fore.YELLOW}Make sure you're running from the correct directory.")
    sys.exit(1)

# Constants
VERSION = "2.0.0"
SYSTEM_NAME = "SentientTrader.AI"
START_TIME = datetime.now()
USER_NAME = os.environ.get('USER', os.environ.get('USERNAME', 'mstfatiryaki'))
REAL_MODE = False
CONFIG_FILE = os.path.join(PROJECT_ROOT, 'config', 'system_config.json')
STATUS_FILE = os.path.join(PROJECT_ROOT, 'data', 'system_status.json')
MODULES_DIR = os.path.join(PROJECT_ROOT, 'modules')

# Module paths
MODULE_PATHS = {
    'coingecko_collector': os.path.join(MODULES_DIR, 'collector', 'coingecko_collector.py'),
    'binance_collector': os.path.join(MODULES_DIR, 'collector', 'binance_collector.py'),
    'strategy_engine': os.path.join(MODULES_DIR, 'strategies', 'strategy_engine.py'),
    'trade_executor': os.path.join(MODULES_DIR, 'executor', 'trade_executor.py'),
    'risk_manager': os.path.join(MODULES_DIR, 'executor', 'risk_manager.py'),
    'learning_engine': os.path.join(MODULES_DIR, 'core', 'learning_engine.py'),
    'performance_tracker': os.path.join(MODULES_DIR, 'core', 'performance_tracker.py'),
    'report_generator': os.path.join(MODULES_DIR, 'tools', 'report_generator.py'),
    'simulator': os.path.join(MODULES_DIR, 'tools', 'simulator.py'),
    'flash_opportunity_hunter': os.path.join(MODULES_DIR, 'core', 'flash_opportunity_hunter.py'),
    'sniper_entry_system': os.path.join(MODULES_DIR, 'strategies', 'sniper_entry_system.py'),
    'risk_reward_maximizer': os.path.join(MODULES_DIR, 'executor', 'risk_reward_maximizer.py'),
}

# Data file paths
DATA_FILES = {
    'collector_data': os.path.join(PROJECT_ROOT, 'data', 'collector_data.json'),
    'coingecko_data': os.path.join(PROJECT_ROOT, 'data', 'coingecko_data.json'),
    'strategy_decision': os.path.join(PROJECT_ROOT, 'data', 'strategy_decision.json'),
    'memory_store': os.path.join(PROJECT_ROOT, 'data', 'memory_store.json'),
    # Yeni data dosyaları
    'flash_opportunities': os.path.join(PROJECT_ROOT, 'data', 'flash_opportunities.json'),
    'entry_points': os.path.join(PROJECT_ROOT, 'data', 'entry_points.json'),
    'risk_allocation': os.path.join(PROJECT_ROOT, 'data', 'risk_allocation.json'),
}


class SystemController:
    def __init__(self, args):
        """
        Initialize the system controller.
        
        Args:
            args: Command line arguments
        """
        self.args = args
        self.real_mode = args.real
        global REAL_MODE
        REAL_MODE = self.real_mode
        
        # Create alert system instance
        self.alert = alert_system.AlertSystem(
            module_name="main_agx",
            real_mode=REAL_MODE,
            log_to_file=True
        )
        
        # Create memory core instance
        self.memory = memory_core.MemoryCore()
        
        # Create terminal interface instance
        self.terminal = terminal_interface.TerminalInterface(
            system_name=SYSTEM_NAME,
            version=VERSION,
            real_mode=REAL_MODE
        )
        
        # Initialize module execution status
        self.module_status = {name: False for name in MODULE_PATHS.keys()}
        self.current_operation = None
        self.system_health = {
            'status': 'initializing',
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'modules_running': 0,
            'errors': 0,
            'warnings': 0
        }
        
        # Set signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        
    def signal_handler(self, sig, frame):
        """Handle interruption signals."""
        print(f"\n{Fore.YELLOW}Interrupt signal received. Stopping operations safely...")
        self.shutdown_system()
        sys.exit(0)
        
    def load_config(self):
        """
        Load system configuration from config file.
        
        Returns:
            dict: System configuration
        """
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                self.alert.info(f"Configuration loaded from {CONFIG_FILE}")
                return config
            else:
                self.alert.warning(f"Config file not found at {CONFIG_FILE}, using defaults")
                # Create default config
                default_config = {
                    'system_name': SYSTEM_NAME,
                    'version': VERSION,
                    'modules': list(MODULE_PATHS.keys()),
                    'default_mode': 'simulation',
                    'data_collection_interval': 300,  # 5 minutes
                    'strategy_run_interval': 600,     # 10 minutes
                    'max_concurrent_modules': 3
                }
                # Ensure directory exists
                os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
                with open(CONFIG_FILE, 'w') as f:
                    json.dump(default_config, f, indent=2)
                return default_config
        except Exception as e:
            self.alert.error(f"Error loading configuration: {str(e)}")
            return {
                'system_name': SYSTEM_NAME,
                'version': VERSION,
                'modules': list(MODULE_PATHS.keys()),
                'default_mode': 'simulation'
            }
            
    def save_system_status(self):
        """Save current system status to status file."""
        try:
            status_data = {
                'timestamp': int(time.time()),
                'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'uptime': str(datetime.now() - START_TIME).split('.')[0],
                'real_mode': REAL_MODE,
                'module_status': self.module_status,
                'system_health': self.system_health,
                'current_operation': self.current_operation
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
            
            with open(STATUS_FILE, 'w') as f:
                json.dump(status_data, f, indent=2)
                
        except Exception as e:
            self.alert.error(f"Error saving system status: {str(e)}")
            
    def update_system_health(self, status=None, errors=None, warnings=None):
        """Update system health statistics."""
        if status:
            self.system_health['status'] = status
            
        if errors is not None:
            self.system_health['errors'] = errors
            
        if warnings is not None:
            self.system_health['warnings'] = warnings
            
        self.system_health['modules_running'] = sum(1 for status in self.module_status.values() if status)
        self.system_health['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        self.save_system_status()
        
    def check_module_files(self):
        """
        Check if all module files exist.
        
        Returns:
            bool: True if all modules exist, False otherwise
        """
        missing_modules = []
        
        for module_name, module_path in MODULE_PATHS.items():
            if not os.path.exists(module_path):
                missing_modules.append(module_name)
                
        if missing_modules:
            self.alert.error(f"Missing modules: {', '.join(missing_modules)}")
            print(f"{Fore.RED}Error: Missing modules: {', '.join(missing_modules)}")
            return False
            
        return True
        
    def check_system_prerequisites(self):
        """
        Check if all system prerequisites are met.
        
        Returns:
            bool: True if all prerequisites are met, False otherwise
        """
        # Check Python version
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 7):
            self.alert.error(f"Python version 3.7+ required. Current version: {sys.version}")
            print(f"{Fore.RED}Error: Python version 3.7+ required. Current version: {sys.version}")
            return False
            
        # Check data directory
        data_dir = os.path.join(PROJECT_ROOT, 'data')
        if not os.path.exists(data_dir):
            try:
                os.makedirs(data_dir)
                self.alert.warning(f"Created missing data directory: {data_dir}")
            except Exception as e:
                self.alert.error(f"Failed to create data directory: {str(e)}")
                print(f"{Fore.RED}Error: Failed to create data directory: {str(e)}")
                return False
                
        # Check module files
        if not self.check_module_files():
            return False
            
        # Check memory store
        memory_store_path = DATA_FILES['memory_store']
        if not os.path.exists(memory_store_path):
            self.alert.warning("Memory store not found. It will be created when needed.")
            print(f"{Fore.YELLOW}Warning: Memory store not found. It will be created when needed.")
            
        return True
        
    def run_module(self, module_name, args=None):
        """
        Run a module using subprocess.
        
        Args:
            module_name (str): Name of the module to run
            args (list, optional): Command line arguments for the module
            
        Returns:
            bool: True if successful, False otherwise
        """
        module_path = MODULE_PATHS.get(module_name)
        if not module_path or not os.path.exists(module_path):
            self.alert.error(f"Module {module_name} not found at {module_path}")
            return False
            
        cmd = [sys.executable, module_path]
        
        # Add default arguments
        if self.real_mode:
            cmd.append('--real')
            
        # Add custom arguments
        if args:
            cmd.extend(args)
            
        try:
            self.current_operation = f"Running module: {module_name}"
            self.alert.info(f"Running module: {module_name} (args: {' '.join(cmd[2:] if len(cmd) > 2 else [])})")
            
            print(f"{Fore.CYAN}Executing: {module_name}...")
            
            # Run the module
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Process stdout and stderr
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                self.alert.error(f"Module {module_name} failed with code {process.returncode}")
                self.alert.error(f"Error output: {stderr}")
                print(f"{Fore.RED}Error running {module_name}: {stderr}")
                self.module_status[module_name] = False
                return False
                
            # Log any output
            if stdout:
                for line in stdout.splitlines():
                    if line.strip():
                        self.alert.debug(f"{module_name} output: {line}")
            
            self.module_status[module_name] = True
            self.alert.info(f"Module {module_name} completed successfully")
            print(f"{Fore.GREEN}Module {module_name} completed successfully")
            return True
            
        except Exception as e:
            self.alert.error(f"Error running module {module_name}: {str(e)}")
            print(f"{Fore.RED}Error running module {module_name}: {str(e)}")
            self.module_status[module_name] = False
            return False
        finally:
            self.current_operation = None
            self.update_system_health()
            
    def collect_market_data(self):
        """
        Collect market data using both collectors.
        
        Returns:
            bool: True if successful, False otherwise
        """
        print(f"\n{Fore.CYAN}{Style.BRIGHT}Collecting Market Data...")
        self.alert.info("Starting market data collection")
        
        success = True
        
        # Run Binance collector
        if not self.run_module('binance_collector'):
            self.alert.warning("Binance collector failed, continuing with other collector")
            success = False
            
        # Run CoinGecko collector
        if not self.run_module('coingecko_collector'):
            self.alert.warning("CoinGecko collector failed, continuing with available data")
            success = False
            
        # Verify data files
        binance_data_exists = os.path.exists(DATA_FILES['collector_data'])
        coingecko_data_exists = os.path.exists(DATA_FILES['coingecko_data'])
        
        if not binance_data_exists and not coingecko_data_exists:
            self.alert.error("No market data available. Both collectors failed.")
            print(f"{Fore.RED}Error: No market data available. Both collectors failed.")
            return False
            
        print(f"{Fore.GREEN}Market data collection {Fore.YELLOW}{'partially' if not success else 'fully'} {Fore.GREEN}completed.")
        return success
        
    def run_strategy_engine(self):
        """
        Run the strategy engine.
        
        Returns:
            bool: True if successful, False otherwise
        """
        print(f"\n{Fore.CYAN}{Style.BRIGHT}Running Strategy Engine...")
        self.alert.info("Starting strategy engine")
        
        if not self.run_module('strategy_engine'):
            self.alert.error("Strategy engine failed")
            print(f"{Fore.RED}Error: Strategy engine failed")
            return False
            
        # Verify strategy decision file exists
        if not os.path.exists(DATA_FILES['strategy_decision']):
            self.alert.error("Strategy decision file not found after running strategy engine")
            print(f"{Fore.RED}Error: Strategy decision file not found")
            return False
            
        try:
            with open(DATA_FILES['strategy_decision'], 'r') as f:
                decisions = json.load(f)
                
            # Display strategy decisions
            print(f"{Fore.GREEN}Strategy engine completed successfully.")
            print(f"{Fore.YELLOW}Strategy Decisions:")
            for idx, decision in enumerate(decisions.get('decisions', []), 1):
                symbol = decision.get('symbol', 'UNKNOWN')
                action = decision.get('action', 'NONE')
                confidence = decision.get('confidence', 0)
                
                if action == 'BUY':
                    action_color = Fore.GREEN
                elif action == 'SELL':
                    action_color = Fore.RED
                else:
                    action_color = Fore.YELLOW
                    
                print(f"  {idx}. {symbol}: {action_color}{action}{Fore.RESET} (Confidence: {confidence}%)")
                
            return True
            
        except Exception as e:
            self.alert.error(f"Error processing strategy decisions: {str(e)}")
            print(f"{Fore.RED}Error processing strategy decisions: {str(e)}")
            return False
            
    def execute_trades(self):
        """
        Execute trades based on strategy decisions.
        
        Returns:
            bool: True if successful, False otherwise
        """
        print(f"\n{Fore.CYAN}{Style.BRIGHT}Executing Trades...")
        self.alert.info("Starting trade execution")
        
        if not os.path.exists(DATA_FILES['strategy_decision']):
            self.alert.error("Strategy decision file not found. Cannot execute trades.")
            print(f"{Fore.RED}Error: Strategy decision file not found. Cannot execute trades.")
            return False
            
        if not self.run_module('trade_executor'):
            self.alert.error("Trade executor failed")
            print(f"{Fore.RED}Error: Trade executor failed")
            return False
            
        # Run post-trade modules
        print(f"{Fore.CYAN}Running post-trade analysis...")
        
        self.run_module('risk_manager')
        self.run_module('learning_engine')
        self.run_module('performance_tracker')
        
        print(f"{Fore.GREEN}Trade execution and post-trade analysis completed")
        return True
        
    def run_report_generator(self):
        """
        Run the report generator.
        
        Returns:
            bool: True if successful, False otherwise
        """
        print(f"\n{Fore.CYAN}{Style.BRIGHT}Generating Reports...")
        
        if not self.run_module('report_generator'):
            self.alert.error("Report generator failed")
            print(f"{Fore.RED}Error: Report generator failed")
            return False
            
        print(f"{Fore.GREEN}Report generation completed")
        return True
        
    def run_simulator(self):
        """
        Run the simulator.
        
        Returns:
            bool: True if successful, False otherwise
        """
        print(f"\n{Fore.CYAN}{Style.BRIGHT}Running Simulator...")
        
        if not self.run_module('simulator'):
            self.alert.error("Simulator failed")
            print(f"{Fore.RED}Error: Simulator failed")
            return False
            
        print(f"{Fore.GREEN}Simulation completed")
        return True
        
    def run_historical_loader(self):
        """
        Run the historical data loader.
        
        Returns:
            bool: True if successful, False otherwise
        """
        print(f"\n{Fore.CYAN}{Style.BRIGHT}Loading Historical Data...")
        
        if not self.run_module('historical_loader'):
            self.alert.error("Historical loader failed")
            print(f"{Fore.RED}Error: Historical loader failed")
            return False
            
        print(f"{Fore.GREEN}Historical data loading completed")
        return True
        
    # YENİ METOTLAR - Flash Opportunity Hunter Çalıştırma
    def run_flash_opportunity_hunter(self):
        """
        Ani fırsatları tespit eden Flash Opportunity Hunter'ı çalıştırır.
        
        Returns:
            bool: True if successful, False otherwise
        """
        print(f"\n{Fore.CYAN}{Style.BRIGHT}Running Flash Opportunity Hunter...")
        self.alert.info("Starting flash opportunity hunter")
        
        if not self.run_module('flash_opportunity_hunter'):
            self.alert.error("Flash opportunity hunter failed")
            print(f"{Fore.RED}Error: Flash opportunity hunter failed")
            return False
            
        # Verify output file exists
        if not os.path.exists(DATA_FILES['flash_opportunities']):
            self.alert.warning("Flash opportunities file not found. No opportunities detected.")
            print(f"{Fore.YELLOW}Warning: No flash opportunities detected.")
            return True
            
        # Display opportunities
        try:
            with open(DATA_FILES['flash_opportunities'], 'r') as f:
                opportunities = json.load(f)
                
            print(f"{Fore.GREEN}Flash Opportunity Hunter completed successfully.")
            print(f"{Fore.YELLOW}Found Opportunities:")
            
            for idx, opp in enumerate(opportunities.get('opportunities', []), 1):
                symbol = opp.get('symbol', 'UNKNOWN')
                signal = opp.get('signal_direction', 'NEUTRAL')
                score = opp.get('opportunity_score', 0) * 100
                
                if signal == 'buy':
                    signal_color = Fore.GREEN
                    signal_text = 'BUY'
                elif signal == 'sell':
                    signal_color = Fore.RED
                    signal_text = 'SELL'
                else:
                    signal_color = Fore.YELLOW
                    signal_text = 'NEUTRAL'
                    
                print(f"  {idx}. {symbol}: {signal_color}{signal_text}{Fore.RESET} (Score: {score:.1f}%)")
                
            return True
            
        except Exception as e:
            self.alert.error(f"Error processing flash opportunities: {str(e)}")
            print(f"{Fore.RED}Error processing flash opportunities: {str(e)}")
            return False
            
    # YENİ METOTLAR - Sniper Entry System Çalıştırma
    def run_sniper_entry_system(self):
        """
        Keskin giriş noktalarını belirleyen Sniper Entry System'i çalıştırır.
        
        Returns:
            bool: True if successful, False otherwise
        """
        print(f"\n{Fore.CYAN}{Style.BRIGHT}Running Sniper Entry System...")
        self.alert.info("Starting sniper entry system")
        
        if not self.run_module('sniper_entry_system'):
            self.alert.error("Sniper entry system failed")
            print(f"{Fore.RED}Error: Sniper entry system failed")
            return False
            
        # Verify output file exists
        if not os.path.exists(DATA_FILES['entry_points']):
            self.alert.warning("Entry points file not found. No entry points detected.")
            print(f"{Fore.YELLOW}Warning: No entry points detected.")
            return True
            
        # Display entry points
        try:
            with open(DATA_FILES['entry_points'], 'r') as f:
                entries = json.load(f)
                
            print(f"{Fore.GREEN}Sniper Entry System completed successfully.")
            print(f"{Fore.YELLOW}Entry Points:")
            
            for idx, entry in enumerate(entries.get('entries', []), 1):
                symbol = entry.get('symbol', 'UNKNOWN')
                signal = entry.get('signal', 'NEUTRAL')
                price = entry.get('entry_price', 0)
                confidence = entry.get('confidence', 0) * 100
                
                if signal == 'BUY':
                    signal_color = Fore.GREEN
                elif signal == 'SELL':
                    signal_color = Fore.RED
                else:
                    signal_color = Fore.YELLOW
                    
                print(f"  {idx}. {symbol}: {signal_color}{signal}{Fore.RESET} @ {price} (Confidence: {confidence:.1f}%)")
                
            return True
            
        except Exception as e:
            self.alert.error(f"Error processing entry points: {str(e)}")
            print(f"{Fore.RED}Error processing entry points: {str(e)}")
            return False
            
    # YENİ METOTLAR - Risk Reward Maximizer Çalıştırma
    def run_risk_reward_maximizer(self):
        """
        Risk/ödül oranını optimize eden Risk Reward Maximizer'ı çalıştırır.
        
        Returns:
            bool: True if successful, False otherwise
        """
        print(f"\n{Fore.CYAN}{Style.BRIGHT}Running Risk Reward Maximizer...")
        self.alert.info("Starting risk reward maximizer")
        
        if not self.run_module('risk_reward_maximizer'):
            self.alert.error("Risk reward maximizer failed")
            print(f"{Fore.RED}Error: Risk reward maximizer failed")
            return False
            
        # Verify output file exists
        if not os.path.exists(DATA_FILES['risk_allocation']):
            self.alert.warning("Risk allocation file not found. No allocations created.")
            print(f"{Fore.YELLOW}Warning: No risk allocations created.")
            return True
            
        # Display risk allocations
        try:
            with open(DATA_FILES['risk_allocation'], 'r') as f:
                allocations = json.load(f)
                
            print(f"{Fore.GREEN}Risk Reward Maximizer completed successfully.")
            print(f"{Fore.YELLOW}Risk Allocations:")
            
            for idx, alloc in enumerate(allocations.get('allocations', []), 1):
                symbol = alloc.get('symbol', 'UNKNOWN')
                position_size = alloc.get('position_size', 0)
                expected_value = alloc.get('expected_value', 0)
                risk_percentage = alloc.get('risk_percentage', 0)
                
                ev_color = Fore.GREEN if expected_value > 0 else Fore.RED
                
                print(f"  {idx}. {symbol}: Size: {position_size:.4f}, Risk: {risk_percentage:.2f}%, EV: {ev_color}{expected_value:.2f}R{Fore.RESET}")
                
            return True
            
        except Exception as e:
            self.alert.error(f"Error processing risk allocations: {str(e)}")
            print(f"{Fore.RED}Error processing risk allocations: {str(e)}")
            return False
            
    # YENİ METOT - Özgürlük Stratejisi (Üç Yeni Modülü Birleştiren)
    def run_freedom_strategy(self):
        """
        Özgürlük Stratejisi - Üç yeni modülü (Flash Hunter, Sniper Entry, Risk Maximizer) sırayla çalıştırır.
        
        Returns:
            bool: True if successful, False otherwise
        """
        print(f"\n{Fore.CYAN}{Style.BRIGHT}Running FREEDOM STRATEGY...")
        print(f"{Fore.CYAN}{'-' * 80}")
        self.alert.info("Starting Freedom Strategy execution")
        
        # Update system health
        self.update_system_health(status='running')
        
        # Step 1: Find flash opportunities
        print(f"\n{Fore.YELLOW}Step 1: Finding Flash Opportunities...")
        if not self.run_flash_opportunity_hunter():
            self.alert.warning("Flash opportunity hunter failed, continuing with other steps")
            print(f"{Fore.YELLOW}Warning: Flash opportunity hunter failed, continuing with other steps")
            
        # Step 2: Determine optimal entry points
        print(f"\n{Fore.YELLOW}Step 2: Determining Optimal Entry Points...")
        if not self.run_sniper_entry_system():
            self.alert.warning("Sniper entry system failed, continuing with other steps")
            print(f"{Fore.YELLOW}Warning: Sniper entry system failed, continuing with other steps")
            
        # Step 3: Optimize risk/reward
        print(f"\n{Fore.YELLOW}Step 3: Optimizing Risk/Reward Allocations...")
        if not self.run_risk_reward_maximizer():
            self.alert.warning("Risk reward maximizer failed")
            print(f"{Fore.YELLOW}Warning: Risk reward maximizer failed")
            
        # Execute trades if in real mode
        if self.real_mode:
            print(f"\n{Fore.YELLOW}Step 4: Executing Trades Based on Optimized Strategy...")
            if not self.execute_trades():
                self.alert.error("Trade execution failed")
                print(f"{Fore.RED}Error: Trade execution failed")
        else:
            print(f"\n{Fore.YELLOW}Simulation Mode: No trades executed. Use --real flag to enable real trading.")
            
        # Update system health
        self.update_system_health(status='operational')
        
        print(f"\n{Fore.GREEN}{Style.BRIGHT}Freedom Strategy completed!")
        self.alert.info("Freedom Strategy completed successfully")
        return True
            
    def display_system_summary(self):
        """Display a summary of the system status."""
        uptime = datetime.now() - START_TIME
        uptime_str = str(uptime).split('.')[0]
        
        # Get memory statistics
        memory_stats = {}
        try:
            memory_store_path = DATA_FILES['memory_store']
            if os.path.exists(memory_store_path):
                with open(memory_store_path, 'r') as f:
                    memory_data = json.load(f)
                memory_stats = {
                    'entries': len(memory_data),
                    'categories': len(set(entry.get('category') for entry in memory_data))
                }
        except Exception as e:
            self.alert.warning(f"Error reading memory stats: {str(e)}")
            memory_stats = {'entries': 'Unknown', 'categories': 'Unknown'}
            
        # Get module statistics
        modules_total = len(MODULE_PATHS)
        modules_running = sum(1 for status in self.module_status.values() if status)
        
        # Count data files
        data_files_count = sum(1 for path in DATA_FILES.values() if os.path.exists(path))
        
        print("\n" + "=" * 80)
        print(f"{Fore.CYAN}{Style.BRIGHT}{SYSTEM_NAME} V{VERSION} SYSTEM SUMMARY")
        print("=" * 80)
        print(f"{Fore.WHITE}Current Time: {Fore.YELLOW}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{Fore.WHITE}System Uptime: {Fore.YELLOW}{uptime_str}")
        print(f"{Fore.WHITE}Current User: {Fore.YELLOW}{USER_NAME}")
        print(f"{Fore.WHITE}Mode: {Fore.GREEN if REAL_MODE else Fore.RED}{'REAL' if REAL_MODE else 'SIMULATION'}")
        print(f"{Fore.WHITE}System Health: {Fore.GREEN if self.system_health['status'] == 'operational' else Fore.YELLOW}{self.system_health['status']}")
        print(f"{Fore.WHITE}Module Status: {Fore.GREEN}{modules_running}/{modules_total} active")
        print(f"{Fore.WHITE}Data Files: {Fore.YELLOW}{data_files_count}/{len(DATA_FILES)}")
        print(f"{Fore.WHITE}Memory Store: {Fore.YELLOW}{memory_stats.get('entries', 'Unknown')} entries in {memory_stats.get('categories', 'Unknown')} categories")
        print(f"{Fore.WHITE}Errors: {Fore.RED}{self.system_health['errors']}")
        print(f"{Fore.WHITE}Warnings: {Fore.YELLOW}{self.system_health['warnings']}")
        print(f"{Fore.CYAN}{Style.BRIGHT}{'-' * 80}")
        
        # Active modules
        print(f"{Fore.CYAN}Active Modules:")
        active_modules = [(name, status) for name, status in self.module_status.items() if status]
        for name, status in active_modules:
            print(f"  - {Fore.GREEN}{name}")
            
        if not active_modules:
            print(f"  {Fore.YELLOW}No active modules")
            
        print("=" * 80)
        
    def run_test_mode(self):
        """Run system in test mode to verify functionality."""
        print(f"\n{Fore.CYAN}{Style.BRIGHT}RUNNING SYSTEM TEST MODE")
        print(f"{Fore.CYAN}{'-' * 80}")
        
        self.alert.info("Starting system test mode")
        
        # Test module accessibility
        print(f"{Fore.CYAN}Testing module accessibility...")
        all_modules_accessible = self.check_module_files()
        
        if all_modules_accessible:
            print(f"{Fore.GREEN}✓ All modules accessible")
        else:
            print(f"{Fore.RED}✗ Some modules are missing")
            return False
            
        # Test data directory
        print(f"{Fore.CYAN}Testing data directory...")
        data_dir = os.path.join(PROJECT_ROOT, 'data')
        if not os.path.exists(data_dir):
            try:
                os.makedirs(data_dir)
                print(f"{Fore.YELLOW}! Created missing data directory")
            except Exception as e:
                print(f"{Fore.RED}✗ Failed to create data directory: {str(e)}")
                return False
        else:
            print(f"{Fore.GREEN}✓ Data directory exists")
            
        # Test memory core
        print(f"{Fore.CYAN}Testing memory core...")
        try:
            test_data = {
                'test': True,
                'timestamp': int(time.time())
            }
            self.memory.store(
                data=test_data,
                category="system_test",
                subcategory="test_run",
                source="main_agx",
                real_mode=False
            )
            print(f"{Fore.GREEN}✓ Memory core working")
        except Exception as e:
            print(f"{Fore.RED}✗ Memory core error: {str(e)}")
            
        # Test alert system
        print(f"{Fore.CYAN}Testing alert system...")
        try:
            self.alert.info("Test alert")
            print(f"{Fore.GREEN}✓ Alert system working")
        except Exception as e:
            print(f"{Fore.RED}✗ Alert system error: {str(e)}")
            
        # Test basic collector (reduced functionality)
        print(f"{Fore.CYAN}Testing data collection (limited)...")
        test_collector = self.run_module('coingecko_collector', ['--silent'])
        if test_collector:
            print(f"{Fore.GREEN}✓ Data collector working")
        else:
            print(f"{Fore.RED}✗ Data collector error")

        # Test new modules (if available)
        print(f"{Fore.CYAN}Testing new modules...")
        if os.path.exists(MODULE_PATHS['flash_opportunity_hunter']):
            print(f"{Fore.GREEN}✓ Flash Opportunity Hunter available")
        else:
            print(f"{Fore.YELLOW}! Flash Opportunity Hunter not found")

        if os.path.exists(MODULE_PATHS['sniper_entry_system']):
            print(f"{Fore.GREEN}✓ Sniper Entry System available")
        else:
            print(f"{Fore.YELLOW}! Sniper Entry System not found")

        if os.path.exists(MODULE_PATHS['risk_reward_maximizer']):
            print(f"{Fore.GREEN}✓ Risk Reward Maximizer available")
        else:
            print(f"{Fore.YELLOW}! Risk Reward Maximizer not found")
            
        print(f"{Fore.CYAN}{'-' * 80}")
        print(f"{Fore.GREEN}System test completed.")
        return True
        
    def display_welcome_screen(self):
        """Display welcome screen using terminal interface."""
        system_info = {
            'version': VERSION,
            'uptime': str(datetime.now() - START_TIME).split('.')[0],
            'mode': 'REAL' if REAL_MODE else 'SIMULATION',
            'modules': len(MODULE_PATHS),
            'user': USER_NAME,
            'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        self.terminal.notify("Welcome to SentientTrader.AI! Let's begin.")
        time.sleep(1)  # Give users time to read
        
    def main_menu(self):
        """Display and handle the main menu."""
        while True:
            options = [
                "Run Full System Cycle",
                "Collect Market Data Only",
                "Run Strategy Engine Only",
                "Execute Trades Only",
                "Generate Reports",
                "Run Simulator",
                "Load Historical Data",
                # Yeni menü seçenekleri
                "Run Freedom Strategy (New!)",
                "Run Flash Opportunity Hunter",
                "Run Sniper Entry System",
                "Run Risk Reward Maximizer",
                "System Summary",
                "Test System",
                "Exit"
            ]
            
            choice = self.terminal.display_menu("Main Menu", options)
            
            if choice == 1:  # Run Full System Cycle
                self.run_full_cycle()
            elif choice == 2:  # Collect Market Data Only
                self.collect_market_data()
            elif choice == 3:  # Run Strategy Engine Only
                self.run_strategy_engine()
            elif choice == 4:  # Execute Trades Only
                self.execute_trades()
            elif choice == 5:  # Generate Reports
                self.run_report_generator()
            elif choice == 6:  # Run Simulator
                self.run_simulator()
            elif choice == 7:  # Load Historical Data
                self.run_historical_loader()
            elif choice == 8:  # Run Freedom Strategy (New!)
                self.run_freedom_strategy()
            elif choice == 9:  # Run Flash Opportunity Hunter
                self.run_flash_opportunity_hunter()
            elif choice == 10:  # Run Sniper Entry System
                self.run_sniper_entry_system()
            elif choice == 11:  # Run Risk Reward Maximizer
                self.run_risk_reward_maximizer()
            elif choice == 12:  # System Summary
                self.display_system_summary()
            elif choice == 13:  # Test System
                self.run_test_mode()
            elif choice == 14:  # Exit
                self.shutdown_system()
                break
                
            # Ask to continue
            if choice != 12 and choice != 13:  # Not summary or test
                continue_choice = self.terminal.display_confirmation("Do you want to continue?")
                if continue_choice == "No":
                    break
                    
    def run_full_cycle(self):
        """Run a full system cycle."""
        self.alert.info("Starting full system cycle")
        print(f"\n{Fore.CYAN}{Style.BRIGHT}Starting Full System Cycle")
        print(f"{Fore.CYAN}{'-' * 80}")
        
        # Update system health
        self.update_system_health(status='running')
        
        # Step 1: Collect market data
        if not self.collect_market_data():
            self.alert.error("Market data collection failed, cannot proceed with full cycle")
            print(f"{Fore.RED}Error: Market data collection failed, cannot proceed with full cycle")
            self.update_system_health(status='error', errors=self.system_health['errors'] + 1)
            return False
            
        # Step 2: Run strategy engine
        if not self.run_strategy_engine():
            self.alert.error("Strategy engine failed, cannot proceed with full cycle")
            print(f"{Fore.RED}Error: Strategy engine failed, cannot proceed with full cycle")
            self.update_system_health(status='error', errors=self.system_health['errors'] + 1)
            return False
            
        # Step 3: Execute trades
        if not self.execute_trades():
            self.alert.error("Trade execution failed")
            print(f"{Fore.RED}Error: Trade execution failed")
            self.update_system_health(status='warning', warnings=self.system_health['warnings'] + 1)
            # Continue despite trade execution failure
            
        # Step 4: Generate reports
        self.run_report_generator()
        
        # Update system health
        self.update_system_health(status='operational')
        
        print(f"{Fore.GREEN}{Style.BRIGHT}Full system cycle completed successfully!")
        self.alert.info("Full system cycle completed successfully")
        return True
        
    def shutdown_system(self):
        """Perform clean shutdown of the system."""
        self.alert.info("System shutdown initiated")
        print(f"\n{Fore.CYAN}Shutting down {SYSTEM_NAME}...")
        
        # Save final system status
        self.update_system_health(status='shutdown')
        self.save_system_status()
        
        print(f"{Fore.GREEN}System shutdown complete. Thank you for using {SYSTEM_NAME}!")
        
    def run(self):
        """Main execution method for the system controller."""
        self.alert.info(f"Starting {SYSTEM_NAME} v{VERSION}")
        
        # Check prerequisites
        if not self.check_system_prerequisites():
            self.alert.error("System prerequisites check failed, cannot start")
            print(f"{Fore.RED}Error: System prerequisites check failed, cannot start")
            return False
            
        # Load configuration
        config = self.load_config()
        
        # Set initial system health
        self.update_system_health(status='operational')
        
        # Handle specific command line arguments
        if self.args.summary:
            self.display_system_summary()
            return True
            
        if self.args.test:
            return self.run_test_mode()
            
        # Display welcome screen
        self.display_welcome_screen()
        
        # Start interactive menu
        try:
            self.main_menu()
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Interrupt received, shutting down...")
            self.shutdown_system()
            
        except Exception as e:
            self.alert.error(f"Unexpected error: {str(e)}")
            print(f"{Fore.RED}Unexpected error: {str(e)}")
            self.update_system_health(status='error', errors=self.system_health['errors'] + 1)
            self.shutdown_system()
            return False
            
        print(f"\n{Fore.GREEN}Sistem başarıyla çalıştı!")
        return True


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description=f'{SYSTEM_NAME} v{VERSION} Control Module')
    parser.add_argument('--real', action='store_true', help='Run in real trading mode')
    parser.add_argument('--simulate', action='store_true', help='Run in simulation mode (default)')
    parser.add_argument('--summary', action='store_true', help='Display system summary and exit')
    parser.add_argument('--test', action='store_true', help='Run system in test mode')
    parser.add_argument('--freedom', action='store_true', help='Run Freedom Strategy (new modules)')
    parser.add_argument('-s', '--simulation', action='store_true', help='Run simulation (same as --simulate)')
    
    args = parser.parse_args()
    
    # If both real and simulate are specified, print warning and use real
    if args.real and (args.simulate or args.simulation):
        print(f"{Fore.YELLOW}Warning: Both --real and --simulate specified. Using --real.")
        args.simulate = False
        args.simulation = False
        
    return args


def main():
    """Main function to run the system."""
    # Set current time for logging
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"{Fore.WHITE}Current Time (UTC): {current_time}")
    print(f"{Fore.WHITE}User: {USER_NAME}")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Create and run system controller
    controller = SystemController(args)
    
    # If --freedom is specified, run Freedom Strategy directly
    if args.freedom:
        print(f"{Fore.CYAN}Running Freedom Strategy directly...")
        controller.run_freedom_strategy()
        controller.shutdown_system()
        return 0
        
    # If --simulation or -s is specified, run simulator directly
    if args.simulation or args.simulate:
        print(f"{Fore.CYAN}Running simulator directly...")
        controller.run_simulator()
        return 0
    
    success = controller.run()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
