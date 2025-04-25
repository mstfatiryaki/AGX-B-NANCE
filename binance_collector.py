from random import choice
import random
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json
import time
import logging
import os
import sys
import argparse
from datetime import datetime
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Import SentientTrader.AI modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from modules.core import memory_core
    from modules.utils import alert_system
    MODULES_IMPORTED = True
except ImportError:
    MODULES_IMPORTED = False
    print(f"{Fore.YELLOW}Warning: Running in standalone mode. SentientTrader.AI modules not available.")

# Constants
BINANCE_API_BASE_URL = 'https://api.binance.com'
TICKER_24HR_ENDPOINT = '/api/v3/ticker/24hr'
TICKER_PRICE_ENDPOINT = '/api/v3/ticker/price'
TICKER_BOOK_ENDPOINT = '/api/v3/ticker/bookTicker'
EXCHANGE_INFO_ENDPOINT = '/api/v3/exchangeInfo'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, 'collector_data.json')
MAX_RETRIES = 5
RETRY_DELAY = 2  # seconds
RATE_LIMIT_PAUSE = 60  # seconds
REAL_MODE = False  # Default to simulation mode


class BinanceCollector:
    def __init__(self, args=None):
        """
        Initialize the Binance data collector with given arguments.
        
        Args:
            args: Command line arguments
        """
        self.args = args if args else self.parse_arguments()
        self.silent = self.args.silent
        self.save_data = not self.args.no_save
        self.real_mode = self.args.real
        
        global REAL_MODE
        REAL_MODE = self.real_mode
        
        # Setup alert system
        if MODULES_IMPORTED:
            self.alert = alert_system.AlertSystem(module_name="binance_collector", 
                                                  real_mode=REAL_MODE,
                                                  log_to_file=True)
        else:
            self.setup_basic_logging()
            self.alert = logging
            
        self.alert.info(f"Binance Collector V2 initialized (REAL_MODE: {REAL_MODE})")
        
    def setup_basic_logging(self):
        """
        Setup basic logging when alert_system module is not available
        """
        log_file = os.path.join(BASE_DIR, 'binance_collector.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ]
        )
    
    def parse_arguments(self):
        """
        Parse command line arguments
        
        Returns:
            argparse.Namespace: Parsed arguments
        """
        parser = argparse.ArgumentParser(description='SentientTrader.AI Binance Data Collector')
        parser.add_argument('--silent', action='store_true', help='Suppress terminal output')
        parser.add_argument('--no-save', action='store_true', help='Do not save to file')
        parser.add_argument('--real', action='store_true', help='Run in real mode (not simulation)')
        return parser.parse_args()
    
    def get_binance_data(self):
        """
        Fetches real-time data for all USDT spot coins from Binance API.
        
        Returns:
            list: Collection of coin data dictionaries
        """
        if not self.silent:
            print(f"{Fore.CYAN}Starting data collection from Binance API...")
        
        self.alert.info("Starting data collection from Binance API")
        
        try:
            # Get exchange info to identify valid trading pairs
            exchange_info = self.make_api_request(f"{BINANCE_API_BASE_URL}{EXCHANGE_INFO_ENDPOINT}")
            if not exchange_info:
                self.alert.error("Failed to get exchange info")
                return []
                
            # Filter for valid USDT spot trading pairs
            valid_symbols = [symbol['symbol'] for symbol in exchange_info.get('symbols', []) 
                           if symbol.get('quoteAsset') == 'USDT' and 
                           symbol.get('status') == 'TRADING' and
                           symbol.get('isSpotTradingAllowed', False)]
            
            if not valid_symbols:
                self.alert.error("No valid USDT trading pairs found")
                return []
                
            self.alert.info(f"Found {len(valid_symbols)} valid USDT trading pairs")
            
            # Get all 24hr ticker data
            ticker_24hr_url = f"{BINANCE_API_BASE_URL}{TICKER_24HR_ENDPOINT}"
            ticker_24hr_response = self.make_api_request(ticker_24hr_url)
            
            if not ticker_24hr_response:
                self.alert.error("Failed to get 24hr ticker data")
                return []
            
            # Get latest prices
            price_url = f"{BINANCE_API_BASE_URL}{TICKER_PRICE_ENDPOINT}"
            price_response = self.make_api_request(price_url)
            
            if not price_response:
                self.alert.error("Failed to get latest price data")
                return []
            
            # Get order book data (bid/ask)
            book_url = f"{BINANCE_API_BASE_URL}{TICKER_BOOK_ENDPOINT}"
            book_response = self.make_api_request(book_url)
            
            if not book_response:
                self.alert.error("Failed to get order book data")
                return []
            
            # Process and combine the data
            processed_data = self.process_coin_data(valid_symbols, ticker_24hr_response, 
                                                  price_response, book_response)
            
            self.alert.info(f"Successfully collected data for {len(processed_data)} coins")
            return processed_data
            
        except Exception as e:
            self.alert.error(f"Error in get_binance_data: {str(e)}")
            return []
    
    def make_api_request(self, url, params=None):
        """
        Makes a request to the Binance API with retry logic and rate limit handling.
        
        Args:
            url (str): The API endpoint URL
            params (dict, optional): Query parameters
            
        Returns:
            dict or list: The API response data, or None if failed
        """
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(url, params=params, timeout=30)
                
                # Check for rate limit
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', RATE_LIMIT_PAUSE))
                    self.alert.warning(f"Rate limit hit. Waiting {retry_after} seconds...")
                    time.sleep(retry_after)
                    continue
                    
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                self.alert.warning(f"API request failed (attempt {attempt+1}/{MAX_RETRIES}): {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    sleep_time = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                    time.sleep(sleep_time)
                else:
                    self.alert.error(f"API request failed after {MAX_RETRIES} attempts")
                    return None
    
    def process_coin_data(self, valid_symbols, ticker_24hr_data, price_data, book_data):
        """
        Processes and combines data from different API responses.
        
        Args:
            valid_symbols (list): List of valid trading pairs
            ticker_24hr_data (list): 24hr ticker data
            price_data (list): Latest price data
            book_data (list): Order book data
            
        Returns:
            list: Processed coin data
        """
        # Filter for USDT spot pairs only
        spot_coins = [item for item in ticker_24hr_data 
                     if item['symbol'].endswith('USDT') and item['symbol'] in valid_symbols]
        
        # Create price and book lookups for faster access
        price_lookup = {item['symbol']: item['price'] for item in price_data}
        book_lookup = {item['symbol']: {'bidPrice': item['bidPrice'], 'askPrice': item['askPrice']} 
                      for item in book_data}
        
        processed_data = []
        for coin in spot_coins:
            symbol = coin['symbol']
            
            # Skip if we don't have all required data for this symbol
            if symbol not in price_lookup or symbol not in book_lookup:
                self.alert.warning(f"Missing data for {symbol}, skipping")
                continue
                
            try:
                coin_data = {
                    'symbol': symbol,
                    'lastPrice': float(coin['lastPrice']),
                    'priceChangePercent': float(coin['priceChangePercent']),
                    'volume': float(coin['volume']),
                    'bidPrice': float(book_lookup[symbol]['bidPrice']),
                    'askPrice': float(book_lookup[symbol]['askPrice']),
                    'timestamp': int(time.time() * 1000),
                    'datetime': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
                }
                processed_data.append(coin_data)
            except (KeyError, ValueError) as e:
                self.alert.warning(f"Error processing data for {symbol}: {str(e)}")
                continue
        
        return processed_data
    
    def save_to_json(self, data):
        """
        Saves collected data to a JSON file.
        
        Args:
            data (list): List of coin data dictionaries
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not data:
            self.alert.error("No data to save")
            return False
            
        if not self.save_data:
            self.alert.info("File saving disabled (--no-save flag)")
            return True
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
            
            # Create a metadata wrapper
            output_data = {
                'timestamp': int(time.time()),
                'datetime': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                'count': len(data),
                'real_mode': REAL_MODE,
                'data': data
            }
            
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(output_data, f, indent=2)
                
            self.alert.info(f"Data successfully saved to {OUTPUT_FILE}")
            return True
            
        except Exception as e:
            self.alert.error(f"Error saving data to JSON: {str(e)}")
            return False
    
    def save_to_memory_core(self, data):
        """
        Saves data to memory_core module.
        
        Args:
            data (list): List of coin data dictionaries
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not MODULES_IMPORTED:
            self.alert.warning("memory_core module not available, skipping memory store")
            return False
            
        try:
            memory = memory_core.MemoryCore()
            
            # Store summary data
            summary = {
                'timestamp': int(time.time()),
                'datetime': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                'count': len(data),
                'collector_version': 'v2'
            }
            
            memory.store(
                data=summary,
                category="market_data",
                subcategory="summary",
                source="binance_collector",
                real_mode=REAL_MODE
            )
            
            # Store individual coin data
            batch_data = []
            for coin in data:
                batch_data.append({
                    'data': coin,
                    'category': "market_data",
                    'subcategory': coin['symbol'],
                    'source': "binance_collector",
                    'real_mode': REAL_MODE
                })
            
            memory.store_batch(batch_data)
            
            self.alert.info(f"Successfully stored {len(data)} coins in memory_core")
            return True
            
        except Exception as e:
            self.alert.error(f"Error saving to memory_core: {str(e)}")
            return False
    
    def display_summary(self, data):
        """
        Displays a colorful summary of the collected data in the terminal.
        
        Args:
            data (list): List of coin data dictionaries
        """
        if self.silent or not data:
            return
            
        # Find best and worst performers
        best_performer = max(data, key=lambda x: float(x['priceChangePercent']))
        worst_performer = min(data, key=lambda x: float(x['priceChangePercent']))
        
        # Find highest volume
        highest_volume = max(data, key=lambda x: float(x['volume']))
        
        # Highest price and lowest price coins
        highest_price = max(data, key=lambda x: float(x['lastPrice']))
        lowest_price = min(data, key=lambda x: float(x['lastPrice']))
        
        # Print summary
        print("\n" + "=" * 80)
        print(f"{Fore.CYAN}{Style.BRIGHT}BINANCE COLLECTOR SUMMARY ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
        print(f"{Fore.CYAN}{Style.BRIGHT}{'=' * 80}")
        print(f"{Fore.WHITE}Total coins collected: {Fore.YELLOW}{len(data)}")
        print(f"{Fore.WHITE}Mode: {Fore.GREEN if REAL_MODE else Fore.RED}{'REAL' if REAL_MODE else 'SIMULATION'}")
        print(f"{Fore.WHITE}Data saved to: {Fore.YELLOW}{OUTPUT_FILE}")
        print(f"{Fore.CYAN}{Style.BRIGHT}{'-' * 80}")
        
        # Performance section
        print(f"{Fore.GREEN}{Style.BRIGHT}BEST PERFORMER: {best_performer['symbol']} ({best_performer['priceChangePercent']}%)")
        print(f"{Fore.RED}{Style.BRIGHT}WORST PERFORMER: {worst_performer['symbol']} ({worst_performer['priceChangePercent']}%)")
        print(f"{Fore.YELLOW}{Style.BRIGHT}HIGHEST VOLUME: {highest_volume['symbol']} ({highest_volume['volume']})")
        print(f"{Fore.MAGENTA}HIGHEST PRICE: {highest_price['symbol']} ({highest_price['lastPrice']})")
        print(f"{Fore.MAGENTA}LOWEST PRICE: {lowest_price['symbol']} ({lowest_price['lastPrice']})")
        
        # Stats breakdown
        pos_change = sum(1 for coin in data if float(coin['priceChangePercent']) > 0)
        neg_change = sum(1 for coin in data if float(coin['priceChangePercent']) < 0)
        no_change = len(data) - pos_change - neg_change
        
        print(f"{Fore.CYAN}{Style.BRIGHT}{'-' * 80}")
        print(f"{Fore.GREEN}Coins with positive change: {pos_change} ({pos_change/len(data)*100:.2f}%)")
        print(f"{Fore.RED}Coins with negative change: {neg_change} ({neg_change/len(data)*100:.2f}%)")
        print(f"{Fore.YELLOW}Coins with no change: {no_change} ({no_change/len(data)*100:.2f}%)")
        print("=" * 80 + "\n")
    
    def run(self):
        """
        Main execution method for the collector.
        
        Returns:
            bool: True if successful, False otherwise
        """
        start_time = time.time()
        
        # Get data from Binance
        coin_data = self.get_binance_data()
        
        if not coin_data:
            self.alert.error("Data collection failed")
            return False
            
        # Save data to file if enabled
        if self.save_data:
            self.save_to_json(coin_data)
            
        # Save to memory_core if available
        if MODULES_IMPORTED:
            self.save_to_memory_core(coin_data)
            
        # Display summary in terminal
        self.display_summary(coin_data)
        
        execution_time = time.time() - start_time
        self.alert.info(f"Data collection completed for {len(coin_data)} coins in {execution_time:.2f} seconds")
        
        return True


def main():
    """
    Main function to execute the data collection workflow.
    """
    collector = BinanceCollector()
    collector.run()


if __name__ == "__main__":
    main()
