from random import choice
import random
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SentientTrader.AI - News Scraper V2
-----------------------------------
Real-time crypto news collector from multiple trusted sources.
Provides data for sentiment analysis and market insights.
"""

import os
import sys
import json
import time
import random
import logging
import argparse
import datetime
import re
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import defaultdict, Counter
import urllib.parse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("news_scraper.log")]
)
logger = logging.getLogger("NewsScraper")

# Try to import required libraries
try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    logger.critical("Required library 'requests' not found. Please install: pip install requests")
    sys.exit(1)

try:
    import feedparser
except ImportError:
    logger.critical("Required library 'feedparser' not found. Please install: pip install feedparser")
    sys.exit(1)

try:
    from bs4 import BeautifulSoup
except ImportError:
    logger.critical("Required library 'beautifulsoup4' not found. Please install: pip install beautifulsoup4")
    sys.exit(1)

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

# Try to import optional system integrations
try:
    import alert_system
    ALERT_SYSTEM_AVAILABLE = True
except ImportError:
    ALERT_SYSTEM_AVAILABLE = False
    logger.warning("Alert system not found. Alerts will be disabled.")

try:
    import memory_core
    MEMORY_CORE_AVAILABLE = True
except ImportError:
    MEMORY_CORE_AVAILABLE = False
    logger.warning("Memory core not found. News data will not be stored in memory.")

# Constants and configuration
USER_AGENT = "SentientTrader/2.0 NewsScraper (Compatible; Python)"
MAX_RETRIES = 5
BACKOFF_FACTOR = 0.5
TIMEOUT = 30
NEWS_DATA_FILE = "news_data.json"
DEFAULT_LIMIT = 50
CURRENT_TIME = "2025-04-21 22:06:36"  # UTC timestamp

# List of supported news sources
NEWS_SOURCES = {
    "cointelegraph": {
        "name": "Cointelegraph",
        "url": "https://cointelegraph.com/rss",
        "type": "rss",
        "rate_limit": 5  # seconds between requests
    },
    "binance": {
        "name": "Binance News",
        "url": "https://www.binance.com/en/feed/news",
        "type": "api",
        "api_url": "https://www.binance.com/bapi/composite/v1/public/cms/article/catalog/list/query",
        "rate_limit": 10
    },
    "decrypt": {
        "name": "Decrypt",
        "url": "https://decrypt.co/feed",
        "type": "rss",
        "rate_limit": 5
    },
    "yahoo_finance": {
        "name": "Yahoo Finance",
        "url": "https://finance.yahoo.com/rss/topic/crypto",
        "type": "rss",
        "rate_limit": 8
    },
    "coindesk": {
        "name": "CoinDesk",
        "url": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "type": "rss",
        "rate_limit": 5
    },
    "google_news": {
        "name": "Google News Crypto",
        "url": "https://news.google.com/rss/search?q=cryptocurrency",
        "type": "rss",
        "rate_limit": 15
    }
}

# List of common cryptocurrencies for detection in news
CRYPTOCURRENCIES = {
    "BTC": ["bitcoin", "btc", "xbt"],
    "ETH": ["ethereum", "eth"],
    "BNB": ["binance coin", "bnb"],
    "SOL": ["solana", "sol"],
    "XRP": ["ripple", "xrp"],
    "ADA": ["cardano", "ada"],
    "DOGE": ["dogecoin", "doge"],
    "DOT": ["polkadot", "dot"],
    "AVAX": ["avalanche", "avax"],
    "SHIB": ["shiba inu", "shib"],
    "MATIC": ["polygon", "matic"],
    "LINK": ["chainlink", "link"],
    "UNI": ["uniswap", "uni"],
    "LTC": ["litecoin", "ltc"],
    "ATOM": ["cosmos", "atom"],
    "FTM": ["fantom", "ftm"],
    "NEAR": ["near protocol", "near"],
    "ALGO": ["algorand", "algo"],
    "XLM": ["stellar", "xlm"],
    "TRX": ["tron", "trx"]
}


class NewsScraper:
    """Scrapes crypto news from multiple sources and standardizes the data format"""
    
    def __init__(self, silent: bool = False, limit: int = DEFAULT_LIMIT, 
                 sources: Optional[List[str]] = None, summary_only: bool = False):
        """
        Initialize the news scraper with configuration options
        
        Args:
            silent (bool): Whether to suppress terminal output
            limit (int): Maximum number of news items to fetch per source
            sources (Optional[List[str]]): List of source identifiers to use, or None for all
            summary_only (bool): Whether to show summary-only output
        """
        self.silent = silent
        self.limit = limit
        self.sources = sources if sources else list(NEWS_SOURCES.keys())
        self.summary_only = summary_only
        
        # Use only sources that exist in NEWS_SOURCES
        self.sources = [s for s in self.sources if s in NEWS_SOURCES]
        
        # Session for making requests
        self.session = self._create_session()
        
        # Storage for scraped news
        self.news_data = []
        
        # Store seen URLs to avoid duplicates
        self.seen_urls = set()
        
        # Store statistics
        self.stats = {
            "sources": defaultdict(int),
            "coins": defaultdict(int),
            "total_fetched": 0,
            "total_new": 0,
            "errors": 0
        }
        
        logger.info(f"News Scraper initialized with: limit={limit}, sources={self.sources}")
    
    def _create_session(self) -> requests.Session:
        """
        Create a requests session with retry logic
        
        Returns:
            requests.Session: Configured session object
        """
        session = requests.Session()
        
        # Configure retry mechanism
        retry_strategy = Retry(
            total=MAX_RETRIES,
            backoff_factor=BACKOFF_FACTOR,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set default headers
        session.headers.update({
            "User-Agent": USER_AGENT
        })
        
        return session
    
    def fetch_all_sources(self) -> List[Dict[str, Any]]:
        """
        Fetch news from all configured sources
        
        Returns:
            List[Dict[str, Any]]: Combined list of news items
        """
        all_news = []
        
        for source_id in self.sources:
            if source_id not in NEWS_SOURCES:
                error_msg = f"Unknown source: {source_id}"
                logger.error(error_msg)
                if ALERT_SYSTEM_AVAILABLE:
                    alert_system.warning(f"Skipping unknown news source", 
                                         {"source": source_id}, "news_scraper")
                continue
            
            source_config = NEWS_SOURCES[source_id]
            
            # Print progress if not silent
            if not self.silent:
                print(f"{COLORS['cyan']}Fetching from {source_config['name']}...{COLORS['reset']}")
            
            try:
                # Choose the appropriate method based on source type
                if source_config["type"] == "rss":
                    news = self._fetch_rss(source_id, source_config)
                elif source_config["type"] == "api":
                    news = self._fetch_api(source_id, source_config)
                else:
                    logger.warning(f"Unsupported source type: {source_config['type']} for {source_id}")
                    continue
                
                # Add to statistics
                self.stats["sources"][source_id] = len(news)
                self.stats["total_fetched"] += len(news)
                
                # Add news to the combined list
                all_news.extend(news)
                
                # Sleep to respect rate limits
                time.sleep(source_config["rate_limit"])
                
            except Exception as e:
                error_msg = f"Error fetching from {source_config['name']}: {str(e)}"
                logger.error(error_msg)
                self.stats["errors"] += 1
                
                if ALERT_SYSTEM_AVAILABLE:
                    alert_system.error(f"Failed to fetch news from {source_config['name']}", 
                                      {"source": source_id, "error": str(e)}, "news_scraper")
        
        # Sort by published date (newest first)
        all_news.sort(key=lambda x: x.get("published", ""), reverse=True)
        
        return all_news
    
    def _fetch_rss(self, source_id: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Fetch news from an RSS feed
        
        Args:
            source_id (str): Source identifier
            config (Dict[str, Any]): Source configuration
            
        Returns:
            List[Dict[str, Any]]: List of standardized news items
        """
        news_items = []
        
        try:
            # Parse the RSS feed
            feed = feedparser.parse(config["url"])
            
            # Check if feed parsing was successful
            if feed.get("bozo_exception"):
                error_msg = f"Error parsing RSS feed from {config['name']}: {feed.get('bozo_exception')}"
                logger.error(error_msg)
                if ALERT_SYSTEM_AVAILABLE:
                    alert_system.error(f"RSS parsing error for {config['name']}", 
                                      {"source": source_id, "error": str(feed.get("bozo_exception"))}, 
                                      "news_scraper")
                return news_items
            
            # Process each entry
            for i, entry in enumerate(feed.entries):
                if i >= self.limit:
                    break
                
                # Extract and standardize the data
                news_item = self._standardize_rss_item(entry, source_id, config["name"])
                
                # Check if we've already seen this URL
                if news_item["link"] in self.seen_urls:
                    continue
                
                # Add this URL to seen set
                self.seen_urls.add(news_item["link"])
                
                # Add to news items
                news_items.append(news_item)
                
                # Detect coins mentioned in this item
                self._detect_coins(news_item)
            
            return news_items
            
        except Exception as e:
            logger.error(f"Error in _fetch_rss for {config['name']}: {str(e)}")
            if ALERT_SYSTEM_AVAILABLE:
                alert_system.error(f"Failed to fetch RSS feed for {config['name']}", 
                                  {"source": source_id, "error": str(e)}, "news_scraper")
            raise
    
    def _fetch_api(self, source_id: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Fetch news from an API
        
        Args:
            source_id (str): Source identifier
            config (Dict[str, Any]): Source configuration
            
        Returns:
            List[Dict[str, Any]]: List of standardized news items
        """
        news_items = []
        
        try:
            # Handle different APIs
            if source_id == "binance":
                return self._fetch_binance_news(config)
            else:
                logger.warning(f"Unsupported API source: {source_id}")
                return news_items
                
        except Exception as e:
            logger.error(f"Error in _fetch_api for {config['name']}: {str(e)}")
            if ALERT_SYSTEM_AVAILABLE:
                alert_system.error(f"Failed to fetch API data for {config['name']}", 
                                  {"source": source_id, "error": str(e)}, "news_scraper")
            raise
    
    def _fetch_binance_news(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Fetch news from Binance News API
        
        Args:
            config (Dict[str, Any]): Source configuration
            
        Returns:
            List[Dict[str, Any]]: List of standardized news items
        """
        news_items = []
        
        try:
            # Binance API request payload
            payload = {
                "catalogId": "48",
                "pageNo": 1,
                "pageSize": self.limit,
                "rnd": str(time.time())
            }
            
            # Make the request
            response = self.session.post(config["api_url"], json=payload, timeout=TIMEOUT)
            response.raise_for_status()
            
            # Parse the response
            data = response.json()
            
            if "data" not in data or "list" not in data["data"]:
                logger.warning(f"Unexpected response format from Binance News API")
                return news_items
            
            # Process each article
            for article in data["data"]["list"]:
                # Extract and standardize the data
                news_item = {
                    "title": article.get("title", "No title"),
                    "summary": article.get("summary", ""),
                    "link": f"https://www.binance.com/en/news/flash/{article.get('code', '')}",
                    "published": self._format_binance_date(article.get("publishDate", "")),
                    "source": "Binance News",
                    "timestamp": CURRENT_TIME,
                    "id": hashlib.md5(f"binance_{article.get('code', '')}".encode()).hexdigest()
                }
                
                # Check if we've already seen this URL
                if news_item["link"] in self.seen_urls:
                    continue
                
                # Add this URL to seen set
                self.seen_urls.add(news_item["link"])
                
                # Add to news items
                news_items.append(news_item)
                
                # Detect coins mentioned in this item
                self._detect_coins(news_item)
            
            return news_items
            
        except Exception as e:
            logger.error(f"Error fetching Binance news: {str(e)}")
            raise
    
    def _standardize_rss_item(self, item: Dict[str, Any], source_id: str, source_name: str) -> Dict[str, Any]:
        """
        Standardize an RSS feed item to our news format
        
        Args:
            item (Dict[str, Any]): RSS item
            source_id (str): Source identifier
            source_name (str): Human-readable source name
            
        Returns:
            Dict[str, Any]: Standardized news item
        """
        # Extract the link (different sources might use different fields)
        link = item.get("link", "")
        if not link:
            link = item.get("id", "")
        
        # Extract the summary (different sources might use different fields)
        summary = item.get("summary", "")
        if not summary:
            summary = item.get("description", "")
        
        # Clean up HTML from summary
        if summary:
            summary = BeautifulSoup(summary, "html.parser").get_text().strip()
        
        # Extract and format the published date
        published = self._format_date(item.get("published", ""))
        
        # Create a unique ID
        news_id = hashlib.md5(f"{source_id}_{link}".encode()).hexdigest()
        
        # Create the standardized news item
        news_item = {
            "title": item.get("title", "No title").strip(),
            "summary": summary,
            "link": link,
            "published": published,
            "source": source_name,
            "timestamp": CURRENT_TIME,
            "id": news_id
        }
        
        return news_item
    
    def _format_date(self, date_str: str) -> str:
        """
        Format a date string to ISO 8601 format
        
        Args:
            date_str (str): Original date string
            
        Returns:
            str: Formatted date string in ISO 8601 format
        """
        if not date_str:
            return datetime.datetime.now().isoformat()
        
        try:
            # Parse with feedparser's date parser
            parsed_time = feedparser._parse_date(date_str)
            if parsed_time:
                return time.strftime("%Y-%m-%dT%H:%M:%SZ", parsed_time)
        except:
            pass
        
        try:
            # Try some common formats
            for fmt in ["%a, %d %b %Y %H:%M:%S %z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"]:
                try:
                    dt = datetime.datetime.strptime(date_str, fmt)
                    return dt.isoformat()
                except:
                    continue
        except:
            pass
        
        # Return as-is if parsing fails
        return date_str
    
    def _format_binance_date(self, timestamp_ms: Union[int, str]) -> str:
        """
        Format a Binance timestamp to ISO 8601 format
        
        Args:
            timestamp_ms (Union[int, str]): Timestamp in milliseconds
            
        Returns:
            str: Formatted date string in ISO 8601 format
        """
        try:
            # Convert to integer if it's a string
            if isinstance(timestamp_ms, str):
                timestamp_ms = int(timestamp_ms)
            
            # Convert milliseconds to seconds
            timestamp_s = timestamp_ms / 1000
            
            # Convert to datetime
            dt = datetime.datetime.fromtimestamp(timestamp_s, tz=datetime.timezone.utc)
            
            # Format to ISO 8601
            return dt.isoformat()
            
        except Exception as e:
            logger.debug(f"Error formatting Binance date: {e}")
            return datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
    
    def _detect_coins(self, news_item: Dict[str, Any]) -> None:
        """
        Detect cryptocurrencies mentioned in a news item and update statistics
        
        Args:
            news_item (Dict[str, Any]): News item to analyze
        """
        # Combine title and summary for detection
        text = (news_item.get("title", "") + " " + news_item.get("summary", "")).lower()
        
        # Add detected coins to the news item
        detected_coins = []
        
        for coin, keywords in CRYPTOCURRENCIES.items():
            for keyword in keywords:
                # Check if the keyword is present as a whole word
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, text):
                    detected_coins.append(coin)
                    self.stats["coins"][coin] += 1
                    break  # Found one keyword for this coin, move to next coin
        
        # Add unique detected coins to the news item
        news_item["coins"] = list(set(detected_coins))
    
    def save_to_file(self) -> bool:
        """
        Save fetched news data to JSON file
        
        Returns:
            bool: Whether the save was successful
        """
        try:
            # Load existing data if any
            existing_data = []
            
            if os.path.exists(NEWS_DATA_FILE):
                try:
                    with open(NEWS_DATA_FILE, "r", encoding="utf-8") as f:
                        existing_data = json.load(f)
                        
                        # Extract URLs from existing data
                        for item in existing_data:
                            if "link" in item:
                                self.seen_urls.add(item["link"])
                except Exception as e:
                    logger.error(f"Error reading existing news data: {str(e)}")
                    # Start with empty data if file exists but has invalid JSON
                    existing_data = []
            
            # Add new items (avoid duplicates)
            new_items = []
            for item in self.news_data:
                if item["link"] not in self.seen_urls or item["link"] == "":
                    new_items.append(item)
                    self.seen_urls.add(item["link"])
            
            # Update statistics
            self.stats["total_new"] = len(new_items)
            
            # Combine existing and new data
            all_data = existing_data + new_items
            
            # Sort by published date (newest first)
            all_data.sort(key=lambda x: x.get("published", ""), reverse=True)
            
            # Save combined data
            with open(NEWS_DATA_FILE, "w", encoding="utf-8") as f:
                json.dump(all_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(new_items)} new news items to {NEWS_DATA_FILE}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving news data: {str(e)}")
            
            if ALERT_SYSTEM_AVAILABLE:
                alert_system.error("Failed to save news data", 
                                  {"error": str(e)}, "news_scraper")
            
            return False
    
    def store_in_memory_core(self) -> bool:
        """
        Store news data in memory core
        
        Returns:
            bool: Whether the storage was successful
        """
        if not MEMORY_CORE_AVAILABLE:
            return False
        
        try:
            # Store each news item in memory core
            for item in self.news_data:
                # Add tags based on source and detected coins
                tags = ["news", f"source_{item['source'].lower().replace(' ', '_')}"]
                tags.extend([f"coin_{coin}" for coin in item.get("coins", [])])
                
                memory_core.add_memory_record(
                    source="news_scraper",
                    category="news",
                    data=item,
                    tags=tags
                )
            
            logger.info(f"Stored {len(self.news_data)} news items in memory core")
            return True
            
        except Exception as e:
            logger.error(f"Error storing news in memory core: {str(e)}")
            return False
    
    def create_sentiment_data(self) -> Dict[str, Any]:
        """
        Create data structure for sentiment analyzer
        
        Returns:
            Dict[str, Any]: News data formatted for sentiment analyzer
        """
        sentiment_data = {
            "metadata": {
                "timestamp": CURRENT_TIME,
                "source_count": len(self.stats["sources"]),
                "total_articles": len(self.news_data)
            },
            "articles": []
        }
        
        # Add articles
        for item in self.news_data:
            article = {
                "title": item.get("title", ""),
                "summary": item.get("summary", ""),
                "source": item.get("source", ""),
                "link": item.get("link", ""),
                "published": item.get("published", ""),
                "coins": item.get("coins", []),
                # Placeholder for sentiment - will be filled by sentiment analyzer
                "sentiment": 0.0,
                "timestamp": item.get("timestamp", CURRENT_TIME)
            }
            
            sentiment_data["articles"].append(article)
        
        return sentiment_data
    
    def display_statistics(self) -> None:
        """Display statistics about fetched news"""
        if self.silent:
            return
        
        print(f"\n{COLORS['bright']}{COLORS['cyan']}{'=' * 60}{COLORS['reset']}")
        print(f"{COLORS['bright']}{COLORS['cyan']}News Scraper Results{COLORS['reset']}")
        print(f"{COLORS['bright']}{COLORS['cyan']}{'=' * 60}{COLORS['reset']}")
        
        # Overall statistics
        print(f"{COLORS['bright']}Sources: {COLORS['reset']}{len(self.stats['sources'])}")
        print(f"{COLORS['bright']}Total fetched: {COLORS['reset']}{self.stats['total_fetched']}")
        print(f"{COLORS['bright']}New articles: {COLORS['reset']}{self.stats['total_new']}")
        print(f"{COLORS['bright']}Errors: {COLORS['reset']}{self.stats['errors']}")
        
        # News sources breakdown
        print(f"\n{COLORS['bright']}Articles per source:{COLORS['reset']}")
        for source_id, count in self.stats["sources"].items():
            name = NEWS_SOURCES[source_id]["name"]
            print(f"  {COLORS['cyan']}{name}: {COLORS['reset']}{count}")
        
        # Coin mentions
        if self.stats["coins"]:
            print(f"\n{COLORS['bright']}Top coin mentions:{COLORS['reset']}")
            # Sort by mention count (descending)
            sorted_coins = sorted(self.stats["coins"].items(), key=lambda x: x[1], reverse=True)
            for coin, count in sorted_coins[:10]:  # Show top 10
                print(f"  {COLORS['yellow']}{coin}: {COLORS['reset']}{count} articles")
        
        # Recent news
        if self.news_data:
            print(f"\n{COLORS['bright']}Recent news:{COLORS['reset']}")
            for i, item in enumerate(self.news_data[:5]):  # Show top 5
                title = item.get("title", "No title")
                source = item.get("source", "Unknown")
                published = item.get("published", "")
                
                # Format date for display
                try:
                    dt = datetime.datetime.fromisoformat(published.replace("Z", "+00:00"))
                    published = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    pass
                
                # Show detected coins if any
                coins = ""
                if "coins" in item and item["coins"]:
                    coins = f" [{', '.join(item['coins'])}]"
                
                # Print news item
                print(f"  {i+1}. {COLORS['bright']}{title}{COLORS['reset']}")
                print(f"     {COLORS['dim']}{source} | {published}{COLORS['yellow']}{coins}{COLORS['reset']}")
        
        print(f"{COLORS['bright']}{COLORS['cyan']}{'=' * 60}{COLORS['reset']}\n")
    
    def summarize_news(self) -> None:
        """Display a summary-only view of news"""
        if self.silent:
            return
        
        print(f"\n{COLORS['bright']}{COLORS['cyan']}News Summary{COLORS['reset']}")
        print(f"{COLORS['cyan']}{'=' * 60}{COLORS['reset']}")
        
        # Show only title and source for each news item
        for i, item in enumerate(self.news_data[:self.limit]):
            title = item.get("title", "No title")
            source = item.get("source", "Unknown")
            
            # Print news item
            print(f"{i+1}. {COLORS['bright']}{title}{COLORS['reset']} {COLORS['dim']}({source}){COLORS['reset']}")
        
        print(f"{COLORS['cyan']}{'=' * 60}{COLORS['reset']}\n")
    
    def run(self) -> bool:
        """
        Run the complete news scraping pipeline
        
        Returns:
            bool: Whether the scraping was successful
        """
        try:
            # Fetch news from all sources
            self.news_data = self.fetch_all_sources()
            
            if not self.news_data:
                logger.warning("No news data fetched from any source")
                if not self.silent:
                    print(f"{COLORS['yellow']}No news data fetched from any source{COLORS['reset']}")
                return False
            
            # Save to file
            if not self.save_to_file():
                logger.error("Failed to save news data to file")
                if not self.silent:
                    print(f"{COLORS['red']}Failed to save news data to file{COLORS['reset']}")
            
            # Store in memory core
            if MEMORY_CORE_AVAILABLE:
                self.store_in_memory_core()
            
            # Create sentiment data (Not saving, just for reference)
            sentiment_data = self.create_sentiment_data()
            
            # Display results
            if self.summary_only:
                self.summarize_news()
            else:
                self.display_statistics()
            
            return True
            
        except Exception as e:
            logger.error(f"Error running news scraper: {str(e)}")
            
            if ALERT_SYSTEM_AVAILABLE:
                alert_system.error("Error running news scraper", 
                                  {"error": str(e)}, "news_scraper")
            
            if not self.silent:
                print(f"{COLORS['red']}Error running news scraper: {str(e)}{COLORS['reset']}")
            
            return False


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="SentientTrader.AI News Scraper")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, 
                        help=f"Maximum number of news items to fetch per source (default: {DEFAULT_LIMIT})")
    
    parser.add_argument("--source", action="append", dest="sources",
                        help="Source to fetch news from (can be specified multiple times)")
    
    parser.add_argument("--summary", action="store_true", 
                        help="Show only title and source summary")
    
    parser.add_argument("--silent", action="store_true", 
                        help="Suppress terminal output")
    
    return parser.parse_args()

def main() -> int:
    """Main function for command-line execution"""
    args = parse_arguments()
    
    try:
        # Initialize the news scraper
        scraper = NewsScraper(
            silent=args.silent,
            limit=args.limit,
            sources=args.sources,
            summary_only=args.summary
        )
        
        # Run the scraper
        success = scraper.run()
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print(f"\n{COLORS['yellow']}Process interrupted by user{COLORS['reset']}")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"{COLORS['red']}Unexpected error: {str(e)}{COLORS['reset']}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
