from random import choice
from random import choice
import random
import random
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SentientTrader.AI - DeepSeek AI Interface Module
-----------------------------------------------
This module integrates DeepSeek as a secondary AI assistant that enhances GPT-4o's
decision-making capabilities by providing news research, counter-arguments,
and human-like discussions in the trading process.

Created by: mstfatiryaki
Date: 2025-04-22
Version: 1.0.0
"""

import os
import sys
import json
import time
import asyncio
import logging
import threading
import datetime
import requests
import random
import queue
import re
import feedparser
from typing import Dict, List, Optional, Union, Any, Tuple
from enum import Enum

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(project_root)

# Try importing optional dependencies
try:
    from modules.utils import api_keys
    API_KEYS_AVAILABLE = True
    DEEPSEEK_API_KEY = getattr(api_keys, 'DEEPSEEK_API_KEY', None)
    if not DEEPSEEK_API_KEY:
        logging.warning("DeepSeek API key not found in api_keys.py")
        # Fallback to environment variable
        DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', None)
except ImportError:
    API_KEYS_AVAILABLE = False
    DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', None)
    logging.warning("modules.utils.api_keys not found, trying environment variables instead")

try:
    from modules.core import memory_core
    MEMORY_CORE_AVAILABLE = True
except ImportError:
    MEMORY_CORE_AVAILABLE = False
    logging.warning("memory_core module not available, memory features disabled")

try:
    from modules.utils import alert_system
    ALERT_SYSTEM_AVAILABLE = True
except ImportError:
    ALERT_SYSTEM_AVAILABLE = False
    logging.warning("alert_system module not available, using standard logging instead")

# Constants
VERSION = "1.0.0"
LOG_FILE = os.path.join(project_root, "logs", "deepseek_log.json")
NEWS_CACHE_FILE = os.path.join(project_root, "data", "deepseek_news_cache.json")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEFAULT_MODEL = "deepseek-chat"
NEWS_UPDATE_INTERVAL = 1800  # 30 minutes in seconds
ANOMALY_CHECK_INTERVAL = 300  # 5 minutes in seconds
DECISION_REVIEW_INTERVAL = 120  # 2 minutes in seconds
MAX_CACHE_AGE = 86400  # 24 hours in seconds

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, "logs", "deepseek.log"))
    ]
)

logger = logging.getLogger("DeepSeekInterface")

class ThinkingMode(Enum):
    """Enum for different thinking modes of DeepSeek."""
    NORMAL = "normal"
    CRITICAL = "critical"
    SUPPORTIVE = "supportive"
    RESEARCH = "research"
    ALERT = "alert"


class DeepSeekInterface:
    """Interface for DeepSeek AI integration with SentientTrader.AI system."""
    
    def __init__(self, 
                 api_key: str = None, 
                 silent_mode: bool = False,
                 real_mode: bool = False):
        """
        Initialize the DeepSeek interface.
        
        Args:
            api_key (str): DeepSeek API key (optional, will use from api_keys if not provided)
            silent_mode (bool): If True, DeepSeek works silently in the background
            real_mode (bool): If True, DeepSeek will execute with real money awareness
        """
        self.api_key = api_key or DEEPSEEK_API_KEY
        self.silent_mode = silent_mode
        self.real_mode = real_mode
        self.model = DEFAULT_MODEL
        self.is_running = False
        self.background_tasks = []
        self.thinking_mode = ThinkingMode.NORMAL
        self.event_queue = queue.Queue()
        self.news_cache = []
        self.last_news_update = 0
        self.last_anomaly_check = 0
        self.last_decision_review = 0
        
        # Initialize memory core if available
        if MEMORY_CORE_AVAILABLE:
            self.memory = memory_core.MemoryCore(real_mode=real_mode)
        else:
            self.memory = None
            
        # Initialize alert system if available
        if ALERT_SYSTEM_AVAILABLE:
            self.alert = alert_system.AlertSystem(
                module_name="deepseek_interface",
                real_mode=real_mode,
                log_to_file=True
            )
        else:
            self.alert = logger
            
        # Initialize log directory and files
        self.init_deepseek()
        
        # Load cached news if available
        self.load_news_cache()
        
        logger.info(f"DeepSeek Interface initialized (silent_mode: {silent_mode}, real_mode: {real_mode})")
    
    def init_deepseek(self):
        """Initialize DeepSeek log file and directories."""
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(LOG_FILE)
        os.makedirs(log_dir, exist_ok=True)
        
        # Create data directory for news cache if it doesn't exist
        data_dir = os.path.dirname(NEWS_CACHE_FILE)
        os.makedirs(data_dir, exist_ok=True)
        
        # Check if log file exists, create with empty array if not
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'w') as f:
                json.dump([], f)
            logger.info(f"Created new DeepSeek log file at: {LOG_FILE}")
        else:
            logger.info(f"Using existing DeepSeek log file at: {LOG_FILE}")
            
        # Load any saved memory
        self.load_saved_thoughts()
    
    def load_saved_thoughts(self):
        """Load previously saved thoughts from the log."""
        try:
            with open(LOG_FILE, 'r') as f:
                thoughts = json.load(f)
                if thoughts:
                    recent_thoughts = thoughts[-5:]  # Get the 5 most recent thoughts
                    logger.info(f"Loaded {len(thoughts)} previous thoughts. Most recent: {recent_thoughts[0]['timestamp']}")
                    
                    # If memory core is available, ensure the thoughts are stored there too
                    if MEMORY_CORE_AVAILABLE and self.memory:
                        for thought in recent_thoughts:
                            self.send_insight_to_memory(
                                thought.get('content', ''),
                                thought.get('category', 'general_insight'),
                                thought.get('confidence', 0.7)
                            )
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Error loading saved thoughts: {str(e)}")
    
    def load_news_cache(self):
        """Load cached news if available."""
        if os.path.exists(NEWS_CACHE_FILE):
            try:
                with open(NEWS_CACHE_FILE, 'r') as f:
                    cache_data = json.load(f)
                    self.news_cache = cache_data.get('news', [])
                    self.last_news_update = cache_data.get('last_update', 0)
                    logger.info(f"Loaded {len(self.news_cache)} cached news items")
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.error(f"Error loading news cache: {str(e)}")
                self.news_cache = []
                self.last_news_update = 0
    
    def save_news_cache(self):
        """Save news cache to file."""
        try:
            with open(NEWS_CACHE_FILE, 'w') as f:
                json.dump({
                    'news': self.news_cache,
                    'last_update': self.last_news_update
                }, f)
                logger.debug(f"Saved {len(self.news_cache)} news items to cache")
        except Exception as e:
            logger.error(f"Error saving news cache: {str(e)}")
    
    def call_deepseek_api(self, 
                          messages: List[Dict[str, str]], 
                          temperature: float = 0.7, 
                          max_tokens: int = 1024) -> Optional[str]:
        """
        Call the DeepSeek API with the given messages.
        
        Args:
            messages (List[Dict[str, str]]): List of message objects
            temperature (float): Sampling temperature
            max_tokens (int): Maximum number of tokens to generate
            
        Returns:
            Optional[str]: Generated text or None if request failed
        """
        if not self.api_key:
            error_msg = "DeepSeek API key not available"
            logger.error(error_msg)
            if not self.silent_mode:
                print(f"\n[DeepSeek] {error_msg}")
            return None
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            response = requests.post(
                DEEPSEEK_API_URL,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                logger.error(f"DeepSeek API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error calling DeepSeek API: {str(e)}")
            return None
    
    def start_background_thinking(self):
        """Start background thinking processes."""
        if self.is_running:
            logger.warning("Background thinking already running")
            return
        
        self.is_running = True
        
        # Start background tasks using threading
        news_thread = threading.Thread(target=self._background_news_monitoring, daemon=True)
        anomaly_thread = threading.Thread(target=self._background_anomaly_detection, daemon=True)
        decision_thread = threading.Thread(target=self._background_decision_review, daemon=True)
        event_thread = threading.Thread(target=self._event_processor, daemon=True)
        
        self.background_tasks = [news_thread, anomaly_thread, decision_thread, event_thread]
        
        for task in self.background_tasks:
            task.start()
            
        logger.info("Started background thinking processes")
        
        if not self.silent_mode:
            print("\n[DeepSeek] Background thinking processes started.")
    
    def stop_background_thinking(self):
        """Stop background thinking processes."""
        self.is_running = False
        logger.info("Stopping background thinking processes")
        
        # Wait for threads to complete (they check self.is_running)
        # Note: Since threads are daemon threads, they'll terminate when the main program exits
        
        if not self.silent_mode:
            print("\n[DeepSeek] Background thinking processes stopped.")
    
    def _background_news_monitoring(self):
        """Background task for monitoring news."""
        logger.info("Started news monitoring background task")
        
        while self.is_running:
            current_time = time.time()
            
            # Check if it's time to update news
            if current_time - self.last_news_update > NEWS_UPDATE_INTERVAL:
                logger.debug("Updating news in background")
                news_results = self.summarize_news()
                
                if news_results:
                    # Queue important news events
                    for item in news_results:
                        if item.get('importance', 0) >= 0.7:  # Only queue high importance items
                            self.event_queue.put({
                                'type': 'news',
                                'data': item,
                                'timestamp': current_time
                            })
                    
                    # Log that we updated news
                    self.log_thoughts(
                        f"Updated news feed with {len(news_results)} items. " +
                        f"Found {sum(1 for item in news_results if item.get('importance', 0) >= 0.7)} important items.",
                        thinking_mode=ThinkingMode.RESEARCH,
                        category="news_monitoring"
                    )
                
                self.last_news_update = current_time
            
            # Sleep to avoid high CPU usage
            time.sleep(10)
    
    def _background_anomaly_detection(self):
        """Background task for detecting market anomalies."""
        logger.info("Started anomaly detection background task")
        
        while self.is_running:
            current_time = time.time()
            
            # Check if it's time to look for anomalies
            if current_time - self.last_anomaly_check > ANOMALY_CHECK_INTERVAL:
                logger.debug("Checking for anomalies in background")
                
                # In a real implementation, this would analyze market data
                # For this implementation, we'll occasionally simulate finding an anomaly
                if random.random() < 0.05:  # 5% chance of finding an anomaly
                    anomaly = self._simulate_anomaly()
                    
                    self.event_queue.put({
                        'type': 'anomaly',
                        'data': anomaly,
                        'timestamp': current_time
                    })
                    
                    self.log_thoughts(
                        f"Detected market anomaly: {anomaly['description']}",
                        thinking_mode=ThinkingMode.ALERT,
                        category="anomaly_detection",
                        confidence=anomaly['confidence']
                    )
                
                self.last_anomaly_check = current_time
            
            # Sleep to avoid high CPU usage
            time.sleep(10)
    
    def _background_decision_review(self):
        """Background task for reviewing GPT-4o's decisions."""
        logger.info("Started decision review background task")
        
        while self.is_running:
            current_time = time.time()
            
            # Check if it's time to review decisions
            if current_time - self.last_decision_review > DECISION_REVIEW_INTERVAL:
                logger.debug("Reviewing decisions in background")
                
                # In a real implementation, this would read and analyze GPT-4o's recent decisions
                # For this implementation, we'll occasionally simulate reviewing a decision
                if random.random() < 0.1:  # 10% chance of reviewing a decision
                    decision = self._simulate_decision_review()
                    
                    self.event_queue.put({
                        'type': 'decision_review',
                        'data': decision,
                        'timestamp': current_time
                    })
                    
                    # Log the review
                    mode = ThinkingMode.SUPPORTIVE if decision['agreement'] else ThinkingMode.CRITICAL
                    self.log_thoughts(
                        f"Decision review: {decision['description']}",
                        thinking_mode=mode,
                        category="decision_review",
                        confidence=decision['confidence']
                    )
                
                self.last_decision_review = current_time
            
            # Sleep to avoid high CPU usage
            time.sleep(10)
    
    def _event_processor(self):
        """Process events from the event queue."""
        logger.info("Started event processor task")
        
        while self.is_running:
            try:
                # Get event with a timeout to allow checking is_running periodically
                try:
                    event = self.event_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                event_type = event.get('type')
                event_data = event.get('data')
                
                logger.info(f"Processing event of type: {event_type}")
                
                if event_type == 'news':
                    self._process_news_event(event_data)
                elif event_type == 'anomaly':
                    self._process_anomaly_event(event_data)
                elif event_type == 'decision_review':
                    self._process_decision_review_event(event_data)
                else:
                    logger.warning(f"Unknown event type: {event_type}")
                
                # Mark the task as done
                self.event_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in event processor: {str(e)}")
                continue
    
    def _process_news_event(self, news_item):
        """Process a news event."""
        # Send insight to memory
        self.send_insight_to_memory(
            f"Important news: {news_item.get('title', 'Unknown')} - {news_item.get('summary', 'No summary')}",
            category="news_insight",
            confidence=news_item.get('importance', 0.7)
        )
        
        # Display alert if not in silent mode
        if not self.silent_mode:
            print(f"\n[DeepSeek] 📰 NEWS ALERT: {news_item.get('title', 'Breaking news')}")
            print(f"          Summary: {news_item.get('summary', 'No summary')}")
            print(f"          Importance: {news_item.get('importance', 0)*100:.0f}%")
            print(f"          Source: {news_item.get('source', 'Unknown')}")
    
    def _process_anomaly_event(self, anomaly):
        """Process an anomaly event."""
        # Send insight to memory
        self.send_insight_to_memory(
            f"Market anomaly detected: {anomaly.get('description', 'Unknown anomaly')}",
            category="anomaly_insight",
            confidence=anomaly.get('confidence', 0.7)
        )
        
        # Display alert if not in silent mode
        if not self.silent_mode:
            print(f"\n[DeepSeek] ⚠️ ANOMALY DETECTED: {anomaly.get('description', 'Unknown anomaly')}")
            print(f"          Confidence: {anomaly.get('confidence', 0)*100:.0f}%")
            print(f"          Assets: {', '.join(anomaly.get('assets', ['Unknown']))}")
    
    def _process_decision_review_event(self, decision):
        """Process a decision review event."""
        # Send insight to memory
        agreement_str = "Support" if decision.get('agreement', False) else "Challenge"
        self.send_insight_to_memory(
            f"{agreement_str} for GPT-4o decision: {decision.get('description', 'Unknown decision')}",
            category="decision_insight",
            confidence=decision.get('confidence', 0.7)
        )
        
        # Display alert if not in silent mode
        if not self.silent_mode:
            agreement_icon = "👍" if decision.get('agreement', False) else "🛑"
            print(f"\n[DeepSeek] {agreement_icon} DECISION REVIEW: {decision.get('description', 'Unknown decision')}")
            print(f"          Confidence: {decision.get('confidence', 0)*100:.0f}%")
            print(f"          Reasoning: {decision.get('reasoning', 'No reasoning provided')}")
    
    def handle_terminal_chat(self, user_input: str) -> str:
        """
        Handle chat input from the terminal.
        
        Args:
            user_input (str): User's input message
            
        Returns:
            str: Response message
        """
        # Check if input is directed to GPT-4o
        if user_input.lower().startswith("gpt:") or user_input.lower().startswith("gpt "):
            # In a real implementation, this would forward to GPT-4o
            # For this implementation, we'll indicate that it would be forwarded
            return "[DeepSeek] Message forwarded to GPT-4o for response."
        
        # Process with DeepSeek
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": user_input}
        ]
        
        response = self.call_deepseek_api(messages)
        
        if response:
            # Log the interaction
            self.log_thoughts(
                f"User asked: '{user_input}'. Response: '{response[:100]}...'",
                thinking_mode=ThinkingMode.NORMAL,
                category="user_interaction"
            )
            return f"[DeepSeek] {response}"
        else:
            return "[DeepSeek] Sorry, I encountered an issue processing your request."
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for DeepSeek based on current mode."""
        base_prompt = (
            "You are DeepSeek, an AI assistant integrated with the SentientTrader.AI cryptocurrency "
            "trading system. You act as a critical thinking partner to GPT-4o, the primary AI. "
            "Your responses should be professional, clear, and concise. "
            "You can analyze market data, provide insights on trading strategies, and answer questions "
            "about cryptocurrency markets and trading concepts. "
            "Your responses should be in plain text, without Markdown formatting.\n\n"
        )
        
        if self.real_mode:
            base_prompt += (
                "IMPORTANT: The system is currently running in REAL MODE with actual money at stake. "
                "Be extremely cautious with your recommendations and always prioritize risk management.\n\n"
            )
        else:
            base_prompt += (
                "The system is currently running in SIMULATION MODE with no real money at stake.\n\n"
            )
        
        # Add mode-specific instructions
        if self.thinking_mode == ThinkingMode.CRITICAL:
            base_prompt += (
                "You are in CRITICAL mode. Your job is to identify flaws, risks, and potential issues "
                "with strategies and decisions. Be thorough in your analysis of weaknesses.\n\n"
            )
        elif self.thinking_mode == ThinkingMode.SUPPORTIVE:
            base_prompt += (
                "You are in SUPPORTIVE mode. Your job is to identify strengths and advantages "
                "of strategies and decisions. Look for evidence that supports these decisions.\n\n"
            )
        elif self.thinking_mode == ThinkingMode.RESEARCH:
            base_prompt += (
                "You are in RESEARCH mode. Your job is to provide factual information, summarize news, "
                "and extract key insights from data. Be comprehensive and accurate.\n\n"
            )
        elif self.thinking_mode == ThinkingMode.ALERT:
            base_prompt += (
                "You are in ALERT mode. Your job is to clearly communicate important warnings, "
                "anomalies, or critical information. Be direct and emphasize urgency when needed.\n\n"
            )
        
        return base_prompt
    
    def summarize_news(self) -> List[Dict[str, Any]]:
        """
        Query news sources and summarize relevant cryptocurrency news.
        
        Returns:
            List[Dict[str, Any]]: List of news items with summaries and importance scores
        """
        logger.info("Summarizing recent crypto news")
        
        # In a real implementation, this would query actual news sources
        # For this implementation, we'll simulate getting news from feeds
        news_items = self._get_crypto_news()
        
        if not news_items:
            logger.warning("No news items found")
            return []
            
        # Process each news item to extract key information
        processed_news = []
        crypto_keywords = ["bitcoin", "btc", "ethereum", "eth", "crypto", "blockchain", 
                          "binance", "coinbase", "altcoin", "defi", "nft", "token",
                          "whale", "regulation", "sec", "market"]
        
        important_keywords = ["crash", "surge", "regulation", "ban", "hack", "security",
                             "breach", "whale", "sec", "lawsuit", "investigation", "binance",
                             "record", "all-time high", "all-time low", "collapse"]
        
        for item in news_items:
            # Calculate importance based on keywords
            text = (item.get('title', '') + ' ' + item.get('summary', '')).lower()
            
            # Count occurrences of crypto keywords
            crypto_relevance = sum(1 for keyword in crypto_keywords if keyword in text) / len(crypto_keywords)
            
            # Count occurrences of important keywords
            importance_score = sum(1 for keyword in important_keywords if keyword in text) / len(important_keywords)
            
            # Combine scores (weighted)
            overall_importance = (0.4 * crypto_relevance) + (0.6 * importance_score)
            
            # Clamp between 0.1 and 0.95
            overall_importance = max(0.1, min(0.95, overall_importance))
            
            # Add to processed news if it's crypto-related
            if crypto_relevance > 0.1:
                processed_item = {
                    'title': item.get('title', 'Untitled'),
                    'summary': item.get('summary', 'No summary available'),
                    'source': item.get('source', 'Unknown'),
                    'url': item.get('url', ''),
                    'published': item.get('published', ''),
                    'importance': overall_importance
                }
                processed_news.append(processed_item)
        
        # Sort by importance (highest first)
        processed_news.sort(key=lambda x: x.get('importance', 0), reverse=True)
        
        # Take top items
        top_news = processed_news[:5]
        
        # Update news cache
        self.news_cache = top_news
        self.last_news_update = time.time()
        self.save_news_cache()
        
        logger.info(f"Processed {len(news_items)} news items, extracted {len(top_news)} important items")
        return top_news
    
    def _get_crypto_news(self) -> List[Dict[str, Any]]:
        """
        Get cryptocurrency news from various sources.
        
        Returns:
            List[Dict[str, Any]]: List of raw news items
        """
        # In a real implementation, this would fetch from actual RSS feeds and APIs
        # For this implementation, we'll simulate news
        
        # Check if we need to simulate new news
        current_time = time.time()
        if current_time - self.last_news_update > MAX_CACHE_AGE or not self.news_cache:
            # Simulate new news items
            simulated_news = self._simulate_crypto_news()
            return simulated_news
        
        # Return cached news if recently updated
        return self.news_cache
    
    def _simulate_crypto_news(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Simulate cryptocurrency news items for testing.
        
        Args:
            count (int): Number of news items to simulate
            
        Returns:
            List[Dict[str, Any]]: List of simulated news items
        """
        # Titles for simulated news
        headlines = [
            "Bitcoin Surges Past $70,000 as Institutional Adoption Accelerates",
            "Ethereum Upgrade Delayed Again, Developers Cite Security Concerns",
            "Major Crypto Exchange Faces Regulatory Scrutiny in EU Markets",
            "Whale Alert: Large Bitcoin Transfers Spark Market Speculation",
            "SEC Approves New Cryptocurrency ETF Products",
            "DeFi Protocol Exploited: $20 Million Stolen in Smart Contract Hack",
            "Central Bank Digital Currencies Gain Traction in Asia",
            "NFT Market Shows Signs of Recovery After Prolonged Slump",
            "Mining Difficulty Reaches All-Time High as Bitcoin Hashrate Surges",
            "Regulatory Clarity: New Framework for Crypto Assets Announced",
            "Altcoin Season? Small Cap Tokens Outperform Bitcoin in Q2",
            "Binance Introduces New Trading Features and Reduced Fees",
            "Privacy Coins Face Delisting Pressure from Multiple Exchanges",
            "Lightning Network Adoption Accelerates with New Integration",
            "Market Analysis: Crypto Correlation with Traditional Assets Weakens"
        ]
        
        # Randomly select headlines and generate news items
        selected_headlines = random.sample(headlines, min(count, len(headlines)))
        
        news_items = []
        current_time = time.time()
        sources = ["CryptoNews", "BlockchainReport", "CoinDesk", "CoinTelegraph", "TokenInsight"]
        
        for headline in selected_headlines:
            # Generate a summary based on the headline
            summary = self._generate_news_summary(headline)
            
            # Create the news item
            news_item = {
                'title': headline,
                'summary': summary,
                'source': random.choice(sources),
                'url': f"https://crypto-news-example.com/article/{abs(hash(headline)) % 10000}",
                'published': datetime.datetime.fromtimestamp(current_time - random.randint(0, 86400)).strftime('%Y-%m-%d %H:%M:%S'),
            }
            
            news_items.append(news_item)
        
        return news_items
    
    def _generate_news_summary(self, headline: str) -> str:
        """
        Generate a plausible summary for a news headline.
        
        Args:
            headline (str): The news headline
            
        Returns:
            str: Generated summary
        """
        # Dictionary of headline keywords and potential summary snippets
        summary_templates = {
            "Bitcoin": [
                "The world's largest cryptocurrency by market cap has seen significant price action in the last 24 hours.",
                "Analysts attribute the move to a combination of technical factors and institutional interest.",
                "On-chain metrics indicate strong accumulation by long-term holders."
            ],
            "Ethereum": [
                "The Ethereum network continues to face challenges as it evolves its technology stack.",
                "Gas fees remain a concern for many users of the second-largest blockchain.",
                "The upgrade is expected to address several scalability issues that have plagued the network."
            ],
            "Regulatory": [
                "Regulatory developments continue to shape the cryptocurrency landscape globally.",
                "Industry participants have expressed mixed reactions to the new regulatory approach.",
                "Compliance costs are expected to increase for cryptocurrency businesses."
            ],
            "Exchange": [
                "The exchange maintains that it has robust compliance procedures in place.",
                "Trading volume has been affected by recent market volatility.",
                "New features aim to attract both retail and institutional traders."
            ],
            "Whale": [
                "Large transactions often indicate institutional movement or significant portfolio restructuring.",
                "Market analysts are divided on whether this represents bullish or bearish sentiment.",
                "Historical patterns suggest such movements often precede major price action."
            ],
            "SEC": [
                "The SEC's decision marks a significant development for the cryptocurrency industry.",
                "Regulatory clarity has been long sought by market participants.",
                "The approval process involved extensive review of market manipulation safeguards."
            ],
            "Hack": [
                "Security experts had previously identified vulnerabilities in similar systems.",
                "The team is working with blockchain analysts to track the stolen funds.",
                "This incident adds to growing concerns about security in the DeFi space."
            ],
            "NFT": [
                "The NFT market has experienced significant volatility since its peak in 2021.",
                "New use cases beyond digital art are emerging for NFT technology.",
                "Trading volumes remain below all-time highs but show signs of steady growth."
            ],
            "Mining": [
                "Higher difficulty means miners need more computational power to earn block rewards.",
                "Energy consumption concerns continue to surround proof-of-work cryptocurrencies.",
                "Geographic distribution of mining operations has shifted significantly in the past year."
            ],
            "Binance": [
                "As one of the largest exchanges globally, Binance's policies often influence the broader market.",
                "The exchange continues to navigate complex regulatory environments across different jurisdictions.",
                "New features are designed to enhance liquidity and trading experience."
            ]
        }
        
        # Find matching keywords in headline
        matching_snippets = []
        for keyword, snippets in summary_templates.items():
            if keyword.lower() in headline.lower():
                matching_snippets.extend(snippets)
        
        # If no specific matches, use generic snippets
        if not matching_snippets:
            matching_snippets = [
                "Market participants are closely monitoring these developments.",
                "The news comes amid broader volatility in cryptocurrency markets.",
                "Analysts have offered varying interpretations of how this will affect market dynamics.",
                "Long-term implications remain unclear as the situation continues to develop."
            ]
        
        # Select 2-3 random snippets and combine them
        selected_snippets = random.sample(matching_snippets, min(random.randint(2, 3), len(matching_snippets)))
        return " ".join(selected_snippets)
    
    def _simulate_anomaly(self) -> Dict[str, Any]:
        """
        Simulate a market anomaly for testing.
        
        Returns:
            Dict[str, Any]: Simulated anomaly data
        """
        anomaly_types = [
            "Unusual volume spike",
            "Price-volume divergence",
            "Order book imbalance",
            "Funding rate irregularity",
            "Correlation breakdown",
            "Volatility surge",
            "Liquidity drop"
        ]
        
        assets = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "DOT", "LINK", "AVAX"]
        selected_assets = random.sample(assets, random.randint(1, 3))
        
        anomaly_type = random.choice(anomaly_types)
        confidence = random.uniform(0.65, 0.95)
        
        anomaly = {
            'type': anomaly_type,
            'assets': selected_assets,
            'description': f"{anomaly_type} detected for {', '.join(selected_assets)}",
            'confidence': confidence,
            'timestamp': time.time()
        }
        
        return anomaly
    
    def _simulate_decision_review(self) -> Dict[str, Any]:
        """
        Simulate a review of a GPT-4o decision for testing.
        
        Returns:
            Dict[str, Any]: Simulated decision review
        """
        decision_types = [
            "Position sizing for BTC long",
            "Stop-loss placement for ETH short",
            "Entry timing for SOL breakout",
            "Risk allocation across altcoins",
            "Profit-taking strategy for BNB",
            "Hedging approach during market uncertainty",
            "Leverage adjustment for futures positions",
            "Portfolio rebalancing recommendation"
        ]
        
        agreement = random.choice([True, False, True])  # Bias slightly toward agreement
        confidence = random.uniform(0.7, 0.95)
        decision_type = random.choice(decision_types)
        
        if agreement:
            reasoning_templates = [
                "The risk-reward ratio is appropriate given current market conditions.",
                "Technical indicators align with the decision's directional bias.",
                "The position size respects the overall risk management framework.",
                "Historical patterns support this approach in similar market conditions.",
                "The strategy correctly accounts for recent volatility changes."
            ]
        else:
            reasoning_templates = [
                "The position size appears excessive relative to account equity.",
                "The strategy overlooks important resistance levels on higher timeframes.",
                "Recent correlation changes suggest higher risk than accounted for.",
                "The approach fails to consider upcoming high-impact events.",
                "Historical performance of this strategy has been inconsistent in similar conditions."
            ]
        
        reasoning = random.choice(reasoning_templates)
        
        review = {
            'type': decision_type,
            'description': f"Review of GPT-4o decision: {decision_type}",
            'agreement': agreement,
            'confidence': confidence,
            'reasoning': reasoning,
            'timestamp': time.time()
        }
        
        return review
    
    def log_thoughts(self, 
                     content: str, 
                     thinking_mode: ThinkingMode = ThinkingMode.NORMAL,
                     category: str = "general",
                     confidence: float = 0.8) -> bool:
        """
        Log DeepSeek's thoughts to the log file.
        
        Args:
            content (str): Thought content
            thinking_mode (ThinkingMode): Mode of thinking
            category (str): Category for the thought
            confidence (float): Confidence level (0.0 to 1.0)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            thought = {
                'timestamp': timestamp,
                'content': content,
                'thinking_mode': thinking_mode.value,
                'category': category,
                'confidence': confidence
            }
            
            # Load existing thoughts
            try:
                with open(LOG_FILE, 'r') as f:
                    thoughts = json.load(f)
                    if not isinstance(thoughts, list):
                        thoughts = []
            except (json.JSONDecodeError, FileNotFoundError):
                thoughts = []
            
            # Add new thought
            thoughts.append(thought)
            
            # Save updated thoughts
            with open(LOG_FILE, 'w') as f:
                json.dump(thoughts, f, indent=2)
            
            logger.debug(f"Logged thought: {content[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error logging thought: {str(e)}")
            return False
    
    def send_insight_to_memory(self, 
                              content: str, 
                              category: str = "deepseek_insight",
                              confidence: float = 0.8) -> bool:
        """
        Send an insight to memory_core if available.
        
        Args:
            content (str): Insight content
            category (str): Category for the insight
            confidence (float): Confidence level (0.0 to 1.0)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not MEMORY_CORE_AVAILABLE or not self.memory:
            logger.warning("Memory core not available, skipping insight storage")
            return False
        
        try:
            # Create insight data
            insight_data = {
                'content': content,
                'source': 'deepseek',
                'confidence': confidence,
                'timestamp': int(time.time()),
                'datetime': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Store in memory
            self.memory.store(
                data=insight_data,
                category="deepseek_insights",
                subcategory=category,
                source="deepseek_interface",
                real_mode=self.real_mode
            )
            
            logger.info(f"Stored insight in memory: {content[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error storing insight in memory: {str(e)}")
            return False
    
    def run_cli(self):
        """Run the DeepSeek CLI interface."""
        print("\n" + "=" * 60)
        print("  DeepSeek Terminal Interface for SentientTrader.AI")
        print("=" * 60)
        print("  Type 'exit' or 'quit' to exit")
        print("  Type 'gpt:' followed by your message to send to GPT-4o")
        print("  Type anything else to chat with DeepSeek")
        print("=" * 60 + "\n")
        
        # Start background thinking
        self.start_background_thinking()
        
        try:
            while True:
                user_input = input("\n🧠 > ")
                
                if user_input.lower() in ['exit', 'quit']:
                    print("Exiting DeepSeek Terminal Interface...")
                    break
                
                if user_input.strip():
                    response = self.handle_terminal_chat(user_input)
                    print(response)
        
        except KeyboardInterrupt:
            print("\nExiting DeepSeek Terminal Interface...")
        
        finally:
            # Stop background thinking
            self.stop_background_thinking()


def main():
    """Main function to run the DeepSeek interface as a standalone module."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='DeepSeek AI Interface for SentientTrader.AI')
    parser.add_argument('--silent', action='store_true', help='Run in silent mode (no terminal output)')
    parser.add_argument('--real', action='store_true', help='Run in real mode (aware of real money)')
    parser.add_argument('--api-key', type=str, help='DeepSeek API key (overrides config)')
    args = parser.parse_args()
    
    # Initialize DeepSeek interface
    deepseek = DeepSeekInterface(
        api_key=args.api_key,
        silent_mode=args.silent,
        real_mode=args.real
    )
    
    # Run the CLI interface
    deepseek.run_cli()


if __name__ == "__main__":
    main()
