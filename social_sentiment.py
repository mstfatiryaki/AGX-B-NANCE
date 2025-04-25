from random import choice
from random import choice
import random
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SentientTrader.AI - Social Sentiment Analyzer V2
------------------------------------------------
Analyzes social media posts from Reddit, Twitter/X and other platforms
to generate cryptocurrency sentiment scores.
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
import statistics
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from collections import defaultdict, Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("social_sentiment.log")]
)
logger = logging.getLogger("SocialSentiment")

# Try to import required libraries
try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    logger.critical("Required library 'requests' not found. Please install: pip install requests")
    sys.exit(1)

# Try to import sentiment analysis libraries
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logger.warning("TextBlob not found. Will use fallback sentiment analysis.")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
    vader_analyzer = SentimentIntensityAnalyzer()
except ImportError:
    VADER_AVAILABLE = False
    logger.warning("VADER Sentiment not found. Will use alternative sentiment analysis.")

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
    logger.warning("Memory core not found. Sentiment data will not be stored in memory.")

# Constants and configuration
USER_AGENT = "SentientTrader/2.0 SocialSentimentAnalyzer (Compatible; Python)"
MAX_RETRIES = 5
BACKOFF_FACTOR = 0.5
TIMEOUT = 30
SOCIAL_SENTIMENT_FILE = "social_sentiment_data.json"
DEFAULT_LOOKBACK_HOURS = 24
CURRENT_TIME = "2025-04-21 22:27:10"  # UTC
CURRENT_USER = "mstfatiryaki"

# Mock data file paths for testing
MOCK_DATA_DIR = "mock_data"
MOCK_TWITTER_FILE = os.path.join(MOCK_DATA_DIR, "twitter_posts.json")
MOCK_REDDIT_FILE = os.path.join(MOCK_DATA_DIR, "reddit_posts.json")

# List of supported cryptocurrencies for detection
CRYPTOCURRENCIES = {
    "BTC": ["bitcoin", "btc", "xbt", "#bitcoin", "#btc"],
    "ETH": ["ethereum", "eth", "#ethereum", "#eth"],
    "BNB": ["binance coin", "bnb", "#bnb"],
    "SOL": ["solana", "sol", "#solana", "#sol"],
    "XRP": ["ripple", "xrp", "#ripple", "#xrp"],
    "ADA": ["cardano", "ada", "#cardano", "#ada"],
    "DOGE": ["dogecoin", "doge", "#dogecoin", "#doge"],
    "DOT": ["polkadot", "dot", "#polkadot", "#dot"],
    "AVAX": ["avalanche", "avax", "#avalanche", "#avax"],
    "SHIB": ["shiba inu", "shib", "#shib", "#shibainu"],
    "MATIC": ["polygon", "matic", "#polygon", "#matic"],
    "LINK": ["chainlink", "link", "#chainlink", "#link"],
    "UNI": ["uniswap", "uni", "#uniswap", "#uni"],
    "LTC": ["litecoin", "ltc", "#litecoin", "#ltc"],
    "ATOM": ["cosmos", "atom", "#cosmos", "#atom"],
    "FTM": ["fantom", "ftm", "#fantom", "#ftm"],
    "NEAR": ["near protocol", "near", "#near"],
    "ALGO": ["algorand", "algo", "#algorand", "#algo"],
    "XLM": ["stellar", "xlm", "#stellar", "#xlm"],
    "TRX": ["tron", "trx", "#tron", "#trx"]
}

# Sentiment keywords and emojis for simple matching
POSITIVE_KEYWORDS = [
    "bullish", "moon", "mooning", "to the moon", "hodl", "buy", "buying", "bought", 
    "long", "good", "great", "excellent", "amazing", "wow", "gain", "gains", "profit", 
    "profits", "win", "winning", "winner", "up", "upward", "rise", "rising", "risen", 
    "high", "higher", "highest", "strong", "stronger", "strongest", "strength", "positive", 
    "optimistic", "confidence", "confident", "secure", "success", "successful", 
    "promising", "potential", "opportunity", "undervalued", "hope", "hoping", "excited",
    "exciting", "breakout", "breakthrough", "happy", "joy", "pump", "pumping", "green"
]

NEGATIVE_KEYWORDS = [
    "bearish", "crash", "crashing", "crashed", "dump", "dumping", "dumped", "sell", 
    "selling", "sold", "short", "bad", "terrible", "awful", "horrible", "loss", "losses", 
    "lose", "losing", "loser", "down", "downward", "fall", "falling", "fallen", "low", 
    "lower", "lowest", "weak", "weaker", "weakest", "weakness", "negative", "pessimistic", 
    "doubt", "doubtful", "risky", "risk", "uncertain", "uncertainty", "fear", "scared", 
    "scary", "afraid", "panic", "fail", "failing", "failed", "failure", "problem", 
    "problematic", "issue", "issues", "worrying", "worried", "scam", "fake", "fraud", 
    "worthless", "useless", "dead", "death", "dying", "danger", "dangerous", "red"
]

POSITIVE_EMOJIS = ["🚀", "🌙", "💎", "🙌", "👍", "💪", "🔥", "⬆️", "📈", "✅", "🤑", "💰", "🏆", "😊", "😄", "🎯", "🎊", "🥳"]
NEGATIVE_EMOJIS = ["📉", "⬇️", "👎", "😢", "😭", "😡", "🤬", "❌", "⚠️", "🚨", "💩", "🙈", "💔", "😔", "😞", "😥", "🤕", "🤢"]


class SocialSentimentAnalyzer:
    """Analyzes social media posts to generate cryptocurrency sentiment scores"""
    
    def __init__(self, silent: bool = False, lookback_hours: int = DEFAULT_LOOKBACK_HOURS, 
                 target_coin: Optional[str] = None, demo_mode: bool = False):
        """
        Initialize the social sentiment analyzer
        
        Args:
            silent (bool): Whether to suppress terminal output
            lookback_hours (int): Hours of data to analyze
            target_coin (Optional[str]): Specific coin to analyze, or None for all
            demo_mode (bool): Whether to use mock data instead of API calls
        """
        self.silent = silent
        self.lookback_hours = lookback_hours
        self.target_coin = target_coin.upper() if target_coin else None
        self.demo_mode = demo_mode
        
        # Session for making requests
        self.session = self._create_session()
        
        # Storage for social media posts
        self.twitter_posts = []
        self.reddit_posts = []
        self.all_posts = []
        
        # Analysis results
        self.coin_sentiments = {}
        self.alerts = []
        self.previous_sentiments = {}
        
        # Statistics
        self.stats = {
            "total_posts": 0,
            "posts_with_sentiment": 0,
            "twitter_posts": 0,
            "reddit_posts": 0,
            "unknown_source_posts": 0,
            "coins_mentioned": 0,
            "processing_time": 0.0
        }
        
        logger.info(f"Social Sentiment Analyzer initialized with: "
                  f"lookback_hours={lookback_hours}, target_coin={target_coin}, "
                  f"demo_mode={demo_mode}")
    
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
    
    def load_previous_sentiments(self) -> bool:
        """
        Load previous sentiment data for change detection
        
        Returns:
            bool: Whether previous data was loaded successfully
        """
        if not os.path.exists(SOCIAL_SENTIMENT_FILE):
            logger.info(f"No previous sentiment data found at {SOCIAL_SENTIMENT_FILE}")
            return False
            
        try:
            with open(SOCIAL_SENTIMENT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                if "coin_sentiments" in data:
                    self.previous_sentiments = data["coin_sentiments"]
                    logger.info(f"Loaded previous sentiment data for {len(self.previous_sentiments)} coins")
                    return True
                    
        except Exception as e:
            logger.error(f"Error loading previous sentiment data: {str(e)}")
            
        return False
    
    def fetch_social_data(self) -> bool:
        """
        Fetch social media data from APIs or mock data
        
        Returns:
            bool: Whether data was fetched successfully
        """
        start_time = time.time()
        
        if self.demo_mode:
            # Use mock data
            twitter_success = self._load_mock_twitter_data()
            reddit_success = self._load_mock_reddit_data()
        else:
            # Use real APIs (note: actual API calls would be implemented here)
            twitter_success = self._load_mock_twitter_data()  # Fallback to mock for the demo
            reddit_success = self._load_mock_reddit_data()    # Fallback to mock for the demo
            
            if not twitter_success or not reddit_success:
                logger.warning("Using mock data as fallback since API fetching failed")
        
        # Combine all posts for processing
        self.all_posts = self.twitter_posts + self.reddit_posts
        
        # Update statistics
        self.stats["total_posts"] = len(self.all_posts)
        self.stats["twitter_posts"] = len(self.twitter_posts)
        self.stats["reddit_posts"] = len(self.reddit_posts)
        
        processing_time = time.time() - start_time
        self.stats["processing_time"] = round(processing_time, 2)
        
        # Print progress
        if not self.silent:
            print(f"{COLORS['cyan']}Fetched {self.stats['total_posts']} posts "
                  f"({self.stats['twitter_posts']} Twitter, {self.stats['reddit_posts']} Reddit){COLORS['reset']}")
        
        return len(self.all_posts) > 0
    
    def _load_mock_twitter_data(self) -> bool:
        """
        Load mock Twitter data from file
        
        Returns:
            bool: Whether data was loaded successfully
        """
        try:
            # Create mock directory if it doesn't exist
            if not os.path.exists(MOCK_DATA_DIR):
                os.makedirs(MOCK_DATA_DIR)
            
            # Check if file exists
            if os.path.exists(MOCK_TWITTER_FILE):
                # Load existing mock data
                with open(MOCK_TWITTER_FILE, "r", encoding="utf-8") as f:
                    mock_data = json.load(f)
                    
                logger.info(f"Loaded {len(mock_data)} mock Twitter posts from {MOCK_TWITTER_FILE}")
            else:
                # Generate mock data if file doesn't exist
                mock_data = self._generate_mock_twitter_data()
                
                # Save to file for future use
                with open(MOCK_TWITTER_FILE, "w", encoding="utf-8") as f:
                    json.dump(mock_data, f, indent=2)
                    
                logger.info(f"Generated and saved {len(mock_data)} mock Twitter posts")
            
            # Filter by coin if target coin specified
            if self.target_coin:
                filtered_posts = [post for post in mock_data if self._post_mentions_coin(post, self.target_coin)]
                self.twitter_posts = filtered_posts
                logger.info(f"Filtered to {len(filtered_posts)} Twitter posts mentioning {self.target_coin}")
            else:
                self.twitter_posts = mock_data
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading mock Twitter data: {str(e)}")
            return False
    
    def _load_mock_reddit_data(self) -> bool:
        """
        Load mock Reddit data from file
        
        Returns:
            bool: Whether data was loaded successfully
        """
        try:
            # Create mock directory if it doesn't exist
            if not os.path.exists(MOCK_DATA_DIR):
                os.makedirs(MOCK_DATA_DIR)
            
            # Check if file exists
            if os.path.exists(MOCK_REDDIT_FILE):
                # Load existing mock data
                with open(MOCK_REDDIT_FILE, "r", encoding="utf-8") as f:
                    mock_data = json.load(f)
                    
                logger.info(f"Loaded {len(mock_data)} mock Reddit posts from {MOCK_REDDIT_FILE}")
            else:
                # Generate mock data if file doesn't exist
                mock_data = self._generate_mock_reddit_data()
                
                # Save to file for future use
                with open(MOCK_REDDIT_FILE, "w", encoding="utf-8") as f:
                    json.dump(mock_data, f, indent=2)
                    
                logger.info(f"Generated and saved {len(mock_data)} mock Reddit posts")
            
            # Filter by coin if target coin specified
            if self.target_coin:
                filtered_posts = [post for post in mock_data if self._post_mentions_coin(post, self.target_coin)]
                self.reddit_posts = filtered_posts
                logger.info(f"Filtered to {len(filtered_posts)} Reddit posts mentioning {self.target_coin}")
            else:
                self.reddit_posts = mock_data
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading mock Reddit data: {str(e)}")
            return False
    
    def _generate_mock_twitter_data(self) -> List[Dict[str, Any]]:
        """
        Generate realistic mock Twitter data
        
        Returns:
            List[Dict[str, Any]]: List of mock Twitter posts
        """
        # Templates for tweet texts
        tweet_templates = [
            "Just bought some #{}! Feeling bullish about its prospects 🚀",
            "I think #{} is going to crash soon. Sentiment looks terrible 📉",
            "Waiting for #{} to drop before buying more. Patience is key.",
            "#{} to the moon! 💎🙌",
            "Not sure about #{}, seems like a lot of FUD going around.",
            "Just sold all my #{} at a great profit! Thanks for the ride! 💰",
            "Lost so much on #{} today. Crypto market is brutal sometimes 😭",
            "Technical analysis suggests #{} will break resistance soon!",
            "#{} developers just announced a huge update! Bullish!",
            "Why is #{} dropping so hard? Any news I missed?",
            "Long-term hodler of #{} here. Not worried about daily fluctuations.",
            "Is it time to buy more #{}? Price looks attractive right now.",
            "Exchanges are showing unusual #{} volume. Something's up!",
            "#{} forming a beautiful bull flag pattern. Time to load up?",
            "My stop loss on #{} just triggered. Will re-enter when market stabilizes."
        ]
        
        mock_posts = []
        current_dt = datetime.datetime.strptime(CURRENT_TIME, "%Y-%m-%d %H:%M:%S")
        
        # Generate mock posts for each coin
        for coin, keywords in CRYPTOCURRENCIES.items():
            # Number of posts varies by coin popularity
            num_posts = random.randint(5, 20)
            
            for _ in range(num_posts):
                # Choose a random tweet template
                template = random.choice(tweet_templates)
                tweet_text = template.format(coin)
                
                # Create a random timestamp within lookback period
                hours_ago = random.randint(0, self.lookback_hours)
                minutes_ago = random.randint(0, 59)
                timestamp = current_dt - datetime.timedelta(hours=hours_ago, minutes=minutes_ago)
                timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                
                # Create a unique post ID
                post_id = hashlib.md5(f"{timestamp_str}_{tweet_text}".encode()).hexdigest()
                
                # Create the post object
                post = {
                    "id": post_id,
                    "content": tweet_text,
                    "created_at": timestamp_str,
                    "source": "twitter",
                    "user": f"crypto_trader_{random.randint(100, 999)}",
                    "coins_mentioned": [coin],
                    "likes": random.randint(0, 100),
                    "retweets": random.randint(0, 30)
                }
                
                mock_posts.append(post)
        
        # Sort by timestamp (newest first)
        mock_posts.sort(key=lambda x: x["created_at"], reverse=True)
        
        return mock_posts
    
    def _generate_mock_reddit_data(self) -> List[Dict[str, Any]]:
        """
        Generate realistic mock Reddit data
        
        Returns:
            List[Dict[str, Any]]: List of mock Reddit posts
        """
        # Templates for Reddit post titles
        post_templates = [
            "What's your price prediction for {} by the end of the year?",
            "Just invested 50% of my portfolio in {}. Thoughts?",
            "Why {} is undervalued right now - Technical Analysis",
            "Beware of {} scams going around! Stay safe everyone.",
            "{} vs competitors - A detailed comparison",
            "Has anyone noticed unusual {} price action lately?",
            "My {} investment journey - From $100 to $10,000",
            "Breaking: Major exchange to delist {}?",
            "{} team announces partnership with Big Tech company",
            "Is {} a good long term hold in this bear market?",
            "Just got liquidated trading {} with leverage. A word of warning.",
            "Found this interesting insight about {} technology",
            "{} is forming a classic pattern on the daily chart",
            "How to stake {} for passive income - Guide",
            "The case for {} as digital gold"
        ]
        
        # Templates for Reddit post content
        content_templates = [
            "I've been researching {} for a while now and I think it's really promising. The technology is solid and the team is experienced. What do you guys think?",
            
            "I'm really bullish on {} right now. The recent price action looks very promising and I think we could see a major breakout soon. Not financial advice though!",
            
            "I don't understand why people are so excited about {}. The tokenomics seem problematic and there are better alternatives out there. Change my mind.",
            
            "Been holding {} for over a year now. Through all the ups and downs. Diamond hands baby! 💎🙌",
            
            "Just saw some concerning news about {}. Apparently there might be regulatory issues coming. Has anyone else heard about this?",
            
            "Technical analysis for {}: Looking at the 4H chart, we can see strong support at the current level. RSI is oversold, suggesting a potential reversal soon.",
            
            "I think {} is getting way too much hype right now. The fundamentals don't support the current valuation. Be careful out there.",
            
            "What's your DCA strategy for {}? I'm buying small amounts every week regardless of price. Long term this will be huge.",
            
            "Comparing {} to other similar projects, I think it has the strongest team and roadmap. Very excited to see where this goes in the next bull run.",
            
            "I lost a significant amount on {} recently. The market is so manipulated. Whales dumping on retail as usual."
        ]
        
        subreddits = [
            "r/CryptoCurrency", "r/CryptoMarkets", "r/Bitcoin", "r/Ethereum",
            "r/altcoin", "r/SatoshiStreetBets", "r/CryptoMoonShots"
        ]
        
        mock_posts = []
        current_dt = datetime.datetime.strptime(CURRENT_TIME, "%Y-%m-%d %H:%M:%S")
        
        # Generate mock posts for each coin
        for coin, keywords in CRYPTOCURRENCIES.items():
            # Number of posts varies by coin popularity
            num_posts = random.randint(3, 15)
            
            for _ in range(num_posts):
                # Choose random templates
                post_title = random.choice(post_templates).format(coin)
                post_content = random.choice(content_templates).format(coin)
                
                # Create a random timestamp within lookback period
                hours_ago = random.randint(0, self.lookback_hours)
                minutes_ago = random.randint(0, 59)
                timestamp = current_dt - datetime.timedelta(hours=hours_ago, minutes=minutes_ago)
                timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                
                # Create a unique post ID
                post_id = hashlib.md5(f"{timestamp_str}_{post_title}".encode()).hexdigest()
                
                # Create the post object
                post = {
                    "id": post_id,
                    "title": post_title,
                    "content": post_content,
                    "created_at": timestamp_str,
                    "source": "reddit",
                    "user": f"redditor_{random.randint(100, 999)}",
                    "subreddit": random.choice(subreddits),
                    "coins_mentioned": [coin],
                    "upvotes": random.randint(1, 500),
                    "comments": random.randint(0, 50)
                }
                
                mock_posts.append(post)
        
        # Sort by timestamp (newest first)
        mock_posts.sort(key=lambda x: x["created_at"], reverse=True)
        
        return mock_posts
    
    def _post_mentions_coin(self, post: Dict[str, Any], coin: str) -> bool:
        """
        Check if a post mentions a specific coin
        
        Args:
            post (Dict[str, Any]): Post data
            coin (str): Coin symbol
            
        Returns:
            bool: Whether the post mentions the coin
        """
        # Check if coin is already mentioned in coins_mentioned field
        if "coins_mentioned" in post and isinstance(post["coins_mentioned"], list):
            if coin in post["coins_mentioned"]:
                return True
        
        # Check title if available
        if "title" in post and post["title"]:
            title = post["title"].lower()
            # Check for exact coin symbol or keywords
            if coin.lower() in title or any(keyword in title for keyword in CRYPTOCURRENCIES.get(coin, [])):
                return True
        
        # Check content if available
        if "content" in post and post["content"]:
            content = post["content"].lower()
            # Check for exact coin symbol or keywords
            if coin.lower() in content or any(keyword in content for keyword in CRYPTOCURRENCIES.get(coin, [])):
                return True
        
        return False
    
    def analyze_sentiments(self) -> bool:
        """
        Analyze sentiments from all collected posts
        
        Returns:
            bool: Whether sentiment analysis was successful
        """
        if not self.all_posts:
            logger.warning("No posts available for sentiment analysis")
            return False
            
        start_time = time.time()
        
        # Detect mentioned coins if not already detected
        self._detect_coins_in_posts()
        
        # Analyze sentiment of each post
        for post in self.all_posts:
            sentiment_score = self._analyze_post_sentiment(post)
            post["sentiment_score"] = sentiment_score
            
            if sentiment_score is not None:
                self.stats["posts_with_sentiment"] += 1
        
        # Analyze by coin
        self._analyze_posts_by_coin()
        
        # Check for significant sentiment changes
        self._detect_sentiment_changes()
        
        # Update statistics
        processing_time = time.time() - start_time
        self.stats["processing_time"] += round(processing_time, 2)
        
        logger.info(f"Analyzed sentiments for {len(self.all_posts)} posts in {processing_time:.2f} seconds")
        
        # Return success if we have any sentiments
        return self.stats["posts_with_sentiment"] > 0
    
    def _detect_coins_in_posts(self) -> None:
        """Detect cryptocurrency mentions in posts"""
        for post in self.all_posts:
            # Skip if coins already detected
            if "coins_mentioned" in post and post["coins_mentioned"]:
                continue
                
            mentioned_coins = []
            
            # Combine title and content for detection
            post_text = ""
            if "title" in post and post["title"]:
                post_text += post["title"].lower() + " "
            if "content" in post and post["content"]:
                post_text += post["content"].lower()
            
            # Detect coins
            for coin, keywords in CRYPTOCURRENCIES.items():
                for keyword in keywords:
                    # Check if the keyword is present as a whole word
                    pattern = r'\b' + re.escape(keyword) + r'\b'
                    if re.search(pattern, post_text):
                        mentioned_coins.append(coin)
                        break  # Found one keyword for this coin, move to next coin
            
            # Store unique detected coins
            post["coins_mentioned"] = list(set(mentioned_coins))
    
    def _analyze_post_sentiment(self, post: Dict[str, Any]) -> Optional[float]:
        """
        Analyze sentiment of a single post
        
        Args:
            post (Dict[str, Any]): Post data
            
        Returns:
            Optional[float]: Sentiment score from -1.0 to 1.0, or None if analysis failed
        """
        # Skip if already analyzed
        if "sentiment_score" in post:
            return post["sentiment_score"]
            
        # Get text to analyze
        text = ""
        if "title" in post and post["title"]:
            text += post["title"] + " "
        if "content" in post and post["content"]:
            text += post["content"]
            
        if not text:
            return None
            
        try:
            # Try multiple sentiment analysis methods and average them
            sentiment_scores = []
            
            # 1. TextBlob analysis if available
            if TEXTBLOB_AVAILABLE:
                textblob_score = TextBlob(text).sentiment.polarity
                sentiment_scores.append(textblob_score)
            
            # 2. VADER sentiment if available
            if VADER_AVAILABLE:
                vader_score = vader_analyzer.polarity_scores(text)["compound"]
                sentiment_scores.append(vader_score)
            
            # 3. Keyword-based sentiment as fallback or additional method
            keyword_score = self._analyze_keywords(text)
            sentiment_scores.append(keyword_score)
            
            # Calculate average sentiment (if any methods worked)
            if sentiment_scores:
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                # Bound within -1.0 to 1.0
                bounded_sentiment = max(-1.0, min(1.0, avg_sentiment))
                return round(bounded_sentiment, 2)
            else:
                return None
                
        except Exception as e:
            logger.debug(f"Error in sentiment analysis: {str(e)}")
            return None
    
    def _analyze_keywords(self, text: str) -> float:
        """
        Simple keyword-based sentiment analysis
        
        Args:
            text (str): Text to analyze
            
        Returns:
            float: Sentiment score between -1.0 and 1.0
        """
        text = text.lower()
        
        # Count positive and negative matches
        positive_count = sum(1 for keyword in POSITIVE_KEYWORDS if keyword in text)
        negative_count = sum(1 for keyword in NEGATIVE_KEYWORDS if keyword in text)
        
        # Count emojis
        positive_emoji_count = sum(text.count(emoji) for emoji in POSITIVE_EMOJIS)
        negative_emoji_count = sum(text.count(emoji) for emoji in NEGATIVE_EMOJIS)
        
        # Add emoji counts (with higher weight)
        positive_count += positive_emoji_count * 2
        negative_count += negative_emoji_count * 2
        
        # Calculate sentiment score
        total_matches = positive_count + negative_count
        if total_matches == 0:
            return 0.0
        
        # Calculate weighted score between -1.0 and 1.0
        sentiment_score = (positive_count - negative_count) / (positive_count + negative_count)
        
        # Scale by number of matches (more matches = stronger signal)
        confidence_factor = min(1.0, total_matches / 10.0)  # Cap at 1.0
        return sentiment_score * confidence_factor
    
    def _analyze_posts_by_coin(self) -> None:
        """Analyze posts grouped by cryptocurrency"""
        # Group posts by coin
        coin_posts = defaultdict(list)
        
        for post in self.all_posts:
            if "coins_mentioned" in post and post["coins_mentioned"]:
                for coin in post["coins_mentioned"]:
                    coin_posts[coin].append(post)
                    
        # Update statistics
        self.stats["coins_mentioned"] = len(coin_posts)
        
        # Skip if specific coin requested and not in data
        if self.target_coin and self.target_coin not in coin_posts:
            logger.warning(f"Target coin {self.target_coin} not found in any posts")
            return
            
        # Filter by target coin if specified
        if self.target_coin:
            coin_posts = {self.target_coin: coin_posts[self.target_coin]} if self.target_coin in coin_posts else {}
        
        # Calculate sentiment stats for each coin
        current_time = CURRENT_TIME
        for coin, posts in coin_posts.items():
            # Calculate average sentiment
            sentiments = [p.get("sentiment_score", 0) for p in posts if "sentiment_score" in p]
            
            if not sentiments:
                continue
            
            avg_sentiment = statistics.mean(sentiments) if sentiments else 0
            
            # Find last mentioned time
            last_mentioned = None
            for post in sorted(posts, key=lambda p: p.get("created_at", ""), reverse=True):
                if "created_at" in post and post["created_at"]:
                    last_mentioned = post["created_at"]
                    break
            
            # Default to current time if no timestamp found
            if not last_mentioned:
                last_mentioned = current_time
            
            # Store coin sentiment
            self.coin_sentiments[coin] = {
                "mentions": len(posts),
                "avg_sentiment": round(avg_sentiment, 2),
                "last_mentioned": last_mentioned,
                "positive_count": sum(1 for s in sentiments if s > 0.1),
                "negative_count": sum(1 for s in sentiments if s < -0.1),
                "neutral_count": sum(1 for s in sentiments if -0.1 <= s <= 0.1),
                "sources": {
                    "twitter": sum(1 for p in posts if p.get("source") == "twitter"),
                    "reddit": sum(1 for p in posts if p.get("source") == "reddit"),
                    "other": sum(1 for p in posts if p.get("source") not in ["twitter", "reddit"])
                },
                "sentiment_trend": self._calculate_sentiment_trend(coin, posts)
            }
            
            # Check if sentiment is extremely positive or negative
            if avg_sentiment >= 0.5:
                self.coin_sentiments[coin]["status"] = "very_positive"
            elif avg_sentiment >= 0.2:
                self.coin_sentiments[coin]["status"] = "positive"
            elif avg_sentiment <= -0.5:
                self.coin_sentiments[coin]["status"] = "very_negative"
            elif avg_sentiment <= -0.2:
                self.coin_sentiments[coin]["status"] = "negative"
            else:
                self.coin_sentiments[coin]["status"] = "neutral"
    
    def _calculate_sentiment_trend(self, coin: str, posts: List[Dict[str, Any]]) -> str:
        """
        Calculate sentiment trend (improving, declining, stable)
        
        Args:
            coin (str): Coin symbol
            posts (List[Dict[str, Any]]): List of posts mentioning the coin
            
        Returns:
            str: Trend description
        """
        # Need enough posts with timestamps for trend analysis
        valid_posts = [p for p in posts if "created_at" in p and "sentiment_score" in p]
        if len(valid_posts) < 3:
            return "insufficient_data"
            
        # Sort by timestamp
        sorted_posts = sorted(valid_posts, key=lambda p: p["created_at"])
        
        # Divide posts into time periods
        num_periods = min(3, len(sorted_posts) // 3)  # Use max 3 periods
        if num_periods < 2:
            return "insufficient_data"
            
        period_size = len(sorted_posts) // num_periods
        periods = [sorted_posts[i*period_size:(i+1)*period_size] for i in range(num_periods)]
        
        # Calculate average sentiment for each period
        period_sentiments = []
        for period in periods:
            sentiments = [p.get("sentiment_score", 0) for p in period if "sentiment_score" in p]
            if sentiments:
                avg_sentiment = statistics.mean(sentiments)
                period_sentiments.append(avg_sentiment)
        
        # Calculate trend from earliest to latest
        if len(period_sentiments) < 2:
            return "stable"
            
        first_sentiment = period_sentiments[0]
        last_sentiment = period_sentiments[-1]
        
        # Determine trend
        diff = last_sentiment - first_sentiment
        if diff >= 0.2:
            return "improving"
        elif diff <= -0.2:
            return "declining"
        else:
            return "stable"
    
    def _detect_sentiment_changes(self) -> None:
        """Detect significant sentiment changes for alerts"""
        # Skip if no previous sentiments
        if not self.previous_sentiments:
            return
            
        for coin, sentiment in self.coin_sentiments.items():
            # Skip if coin wasn't in previous data
            if coin not in self.previous_sentiments:
                continue
                
            prev_sentiment = self.previous_sentiments[coin]
            
            # Check for significant sentiment changes
            prev_score = prev_sentiment.get("avg_sentiment", 0)
            current_score = sentiment.get("avg_sentiment", 0)
            
            # Calculate change
            change = current_score - prev_score
            
            # Alert on significant changes
            if abs(change) >= 0.3:
                alert_type = "sentiment_spike" if change > 0 else "sentiment_drop"
                
                alert = {
                    "coin": coin,
                    "type": alert_type,
                    "change": f"{'+' if change > 0 else ''}{change:.2f}",
                    "previous": prev_score,
                    "current": current_score,
                    "message": f"{coin} sentiment {alert_type.split('_')[1]} detected on social media"
                }
                
                self.alerts.append(alert)
                
                # Send alert notification if available
                if ALERT_SYSTEM_AVAILABLE:
                    alert_level = "info" if alert_type == "sentiment_spike" else "warning"
                    alert_func = alert_system.info if alert_type == "sentiment_spike" else alert_system.warning
                    
                    alert_func(
                        f"{coin} sentiment {alert_type.split('_')[1]} detected",
                        {
                            "coin": coin,
                            "change": f"{'+' if change > 0 else ''}{change:.2f}",
                            "previous": prev_score,
                            "current": current_score
                        },
                        "social_sentiment"
                    )
    
    def generate_output_data(self) -> Dict[str, Any]:
        """
        Generate structured data for output file
        
        Returns:
            Dict[str, Any]: Structured sentiment data
        """
        # Get current timestamp
        timestamp = CURRENT_TIME
        
        # Create output structure
        output_data = {
            "timestamp": timestamp,
            "source": "SentientTrader.AI Social Sentiment V2",
            "analysis_params": {
                "lookback_hours": self.lookback_hours,
                "target_coin": self.target_coin,
                "demo_mode": self.demo_mode
            },
            "stats": self.stats,
            "coin_sentiments": self.coin_sentiments,
            "alerts": self.alerts
        }
        
        return output_data
    
    def save_to_file(self, output_data: Dict[str, Any]) -> bool:
        """
        Save sentiment data to JSON file
        
        Args:
            output_data (Dict[str, Any]): Data to save
            
        Returns:
            bool: Whether the save was successful
        """
        try:
            with open(SOCIAL_SENTIMENT_FILE, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Saved social sentiment data to {SOCIAL_SENTIMENT_FILE}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving social sentiment data: {str(e)}")
            
            if ALERT_SYSTEM_AVAILABLE:
                alert_system.error(
                    "Failed to save social sentiment data",
                    {"error": str(e)},
                    "social_sentiment"
                )
            
            return False
    
    def store_in_memory(self, output_data: Dict[str, Any]) -> bool:
        """
        Store sentiment data in memory core
        
        Args:
            output_data (Dict[str, Any]): Data to store
            
        Returns:
            bool: Whether the storage was successful
        """
        if not MEMORY_CORE_AVAILABLE:
            return False
            
        try:
            # Store overall sentiment summary
            memory_core.add_memory_record(
                source="social_sentiment",
                category="sentiment_summary",
                data={
                    "timestamp": output_data["timestamp"],
                    "coin_count": len(output_data["coin_sentiments"]),
                    "total_posts": output_data["stats"]["total_posts"],
                    "alerts": output_data["alerts"]
                },
                tags=["sentiment", "social", "summary"]
            )
            
            # Store individual coin sentiments
            for coin, sentiment in output_data["coin_sentiments"].items():
                memory_core.add_memory_record(
                    source="social_sentiment",
                    category="coin_sentiment",
                    data={
                        "coin": coin,
                        "timestamp": output_data["timestamp"],
                        "sentiment": sentiment
                    },
                    tags=["sentiment", "social", f"coin_{coin.lower()}", sentiment["status"]]
                )
            
            logger.info(f"Stored sentiment data for {len(output_data['coin_sentiments'])} coins in memory core")
            return True
            
        except Exception as e:
            logger.error(f"Error storing sentiment in memory core: {str(e)}")
            return False
    
    def display_results(self) -> None:
        """Display sentiment analysis results in the terminal"""
        if self.silent:
            return
            
        print(f"\n{COLORS['bright']}{COLORS['magenta']}{'=' * 60}{COLORS['reset']}")
        print(f"{COLORS['bright']}{COLORS['magenta']}Social Media Sentiment Analysis{COLORS['reset']}")
        print(f"{COLORS['bright']}{COLORS['magenta']}{'=' * 60}{COLORS['reset']}")
        
        # Display analysis parameters
        target_str = f"for {COLORS['yellow']}{self.target_coin}{COLORS['reset']}" if self.target_coin else "for all coins"
        print(f"Analysis {target_str} over the past {self.lookback_hours} hours")
        print(f"Data mode: {COLORS['cyan']}{'Demo' if self.demo_mode else 'Real'}{COLORS['reset']}")
        
        # Display statistics
        print(f"\n{COLORS['bright']}Data Sources:{COLORS['reset']}")
        print(f"  Twitter: {self.stats['twitter_posts']} posts")
        print(f"  Reddit: {self.stats['reddit_posts']} posts")
        print(f"  Total: {self.stats['total_posts']} posts across {self.stats['coins_mentioned']} coins")
        
        # Display coin sentiments
        if self.coin_sentiments:
            print(f"\n{COLORS['bright']}Coin Sentiment Summary:{COLORS['reset']}")
            
            # Sort by mentions (most discussed first)
            sorted_coins = sorted(self.coin_sentiments.items(), key=lambda x: x[1]["mentions"], reverse=True)
            
            for coin, data in sorted_coins:
                sentiment_value = data["avg_sentiment"]
                
                # Determine color based on sentiment
                if sentiment_value >= 0.3:
                    sentiment_color = COLORS["green"]
                elif sentiment_value >= 0.1:
                    sentiment_color = COLORS["cyan"]
                elif sentiment_value <= -0.3:
                    sentiment_color = COLORS["red"]
                elif sentiment_value <= -0.1:
                    sentiment_color = COLORS["yellow"]
                else:
                    sentiment_color = COLORS["white"]
                
                # Format sentiment value
                sentiment_str = f"{sentiment_value:+.2f}" if sentiment_value != 0 else "0.00"
                
                # Get counts
                mentions = data["mentions"]
                pos_count = data.get("positive_count", 0)
                neg_count = data.get("negative_count", 0)
                
                # Display trend if available
                trend = data.get("sentiment_trend", "")
                trend_str = ""
                if trend == "improving":
                    trend_str = f" {COLORS['green']}↗{COLORS['reset']}"
                elif trend == "declining":
                    trend_str = f" {COLORS['red']}↘{COLORS['reset']}"
                
                # Print coin sentiment summary
                print(f"  {COLORS['bright']}{coin}{COLORS['reset']}: "
                      f"{sentiment_color}{sentiment_str}{COLORS['reset']}{trend_str} "
                      f"({mentions} mentions, {pos_count}+ / {neg_count}-)")
        
        # Display alerts if any
        if self.alerts:
            print(f"\n{COLORS['bright']}Alerts:{COLORS['reset']}")
            
            for alert in self.alerts:
                coin = alert["coin"]
                change = alert["change"]
                message = alert["message"]
                
                # Determine color based on alert type
                if "spike" in alert["type"]:
                    alert_color = COLORS["green"]
                else:
                    alert_color = COLORS["red"]
                
                print(f"  {COLORS['yellow']}{coin}{COLORS['reset']}: "
                      f"{alert_color}{change}{COLORS['reset']} - {message}")
        
        # Show most discussed coin in detail
        if self.coin_sentiments and not self.target_coin:
            most_discussed = max(self.coin_sentiments.items(), key=lambda x: x[1]["mentions"])
            coin, data = most_discussed
            
            print(f"\n{COLORS['bright']}Most Discussed: {COLORS['yellow']}{coin}{COLORS['reset']}")
            print(f"  Mentions: {data['mentions']} ({data['sources']['twitter']} Twitter, {data['sources']['reddit']} Reddit)")
            print(f"  Sentiment: {data['avg_sentiment']:+.2f} ({data['positive_count']}+ / {data['negative_count']}- / {data['neutral_count']}=)")
            
            # Show sentiment trend
            trend = data.get("sentiment_trend", "")
            if trend == "improving":
                print(f"  Trend: {COLORS['green']}Improving{COLORS['reset']}")
            elif trend == "declining":
                print(f"  Trend: {COLORS['red']}Declining{COLORS['reset']}")
            elif trend == "stable":
                print(f"  Trend: {COLORS['white']}Stable{COLORS['reset']}")
        
        # Show footer
        print(f"{COLORS['magenta']}{'=' * 60}{COLORS['reset']}")
        print(f"{COLORS['dim']}Analysis completed in {self.stats['processing_time']:.2f} seconds{COLORS['reset']}\n")
    
    def run(self) -> bool:
        """
        Run the complete social sentiment analysis pipeline
        
        Returns:
            bool: Whether the analysis was successful
        """
        try:
            # Load previous sentiment data for change detection
            self.load_previous_sentiments()
            
            # Fetch social media data
            if not self.fetch_social_data():
                logger.error("Failed to fetch social media data")
                if not self.silent:
                    print(f"{COLORS['red']}Failed to fetch social media data{COLORS['reset']}")
                return False
            
            # Analyze sentiments
            if not self.analyze_sentiments():
                logger.error("Failed to analyze sentiments")
                if not self.silent:
                    print(f"{COLORS['red']}Failed to analyze sentiments{COLORS['reset']}")
                return False
            
            # Generate output data
            output_data = self.generate_output_data()
            
            # Save to file
            if not self.save_to_file(output_data):
                logger.error("Failed to save output data to file")
                if not self.silent:
                    print(f"{COLORS['red']}Failed to save output data to file{COLORS['reset']}")
            
            # Store in memory if available
            if MEMORY_CORE_AVAILABLE:
                self.store_in_memory(output_data)
            
            # Display results
            self.display_results()
            
            logger.info("Social sentiment analysis completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error running social sentiment analysis: {str(e)}")
            
            if ALERT_SYSTEM_AVAILABLE:
                alert_system.error(
                    "Error in social sentiment analysis",
                    {"error": str(e)},
                    "social_sentiment"
                )
            
            if not self.silent:
                print(f"{COLORS['red']}Error: {str(e)}{COLORS['reset']}")
            
            return False


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="SentientTrader.AI Social Sentiment Analyzer")
    parser.add_argument("--hours", type=int, default=DEFAULT_LOOKBACK_HOURS,
                        help=f"Hours of data to analyze (default: {DEFAULT_LOOKBACK_HOURS})")
    parser.add_argument("--coin", type=str,
                        help="Specific coin to analyze (e.g., BTC)")
    parser.add_argument("--summary", action="store_true",
                        help="Display only a summary of results")
    parser.add_argument("--silent", action="store_true",
                        help="Suppress terminal output")
    parser.add_argument("--demo", action="store_true",
                        help="Use demo mode with generated data")
    return parser.parse_args()

def main() -> int:
    """Main function for command-line execution"""
    args = parse_arguments()
    
    try:
        # Initialize the analyzer
        analyzer = SocialSentimentAnalyzer(
            silent=args.silent,
            lookback_hours=args.hours,
            target_coin=args.coin,
            demo_mode=args.demo
        )
        
        # Run the analyzer
        success = analyzer.run()
        
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
