from random import choice
import random
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SentientTrader.AI - Sentiment Analyzer V2
-----------------------------------------
Analyzes news and social media sentiment data to produce sentiment scores
for cryptocurrencies and detect market sentiment trends.
"""

import os
import sys
import json
import math
import time
import logging
import datetime
import argparse
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("sentiment_analyzer.log")]
)
logger = logging.getLogger("SentimentAnalyzer")

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

# Try to import memory core for storing sentiment history
try:
    import memory_core
    MEMORY_CORE_AVAILABLE = True
except ImportError:
    MEMORY_CORE_AVAILABLE = False
    logger.warning("Memory core not found. Sentiment history will not be stored.")

# Try to import learning engine
try:
    import learning_engine
    LEARNING_ENGINE_AVAILABLE = True
except ImportError:
    LEARNING_ENGINE_AVAILABLE = False
    logger.warning("Learning engine not found. Learning feedback will not be provided.")

# Constants and configuration
CURRENT_TIME = "2025-04-21 20:08:06"  # UTC
CURRENT_USER = "mstfatiryaki"

# File paths
NEWS_DATA_FILE = "news_data.json"
SOCIAL_SENTIMENT_FILE = "social_sentiment_data.json"
SENTIMENT_SUMMARY_FILE = "sentiment_summary.json"
LEARNING_FEEDBACK_FILE = "learning_feedback.json"

# Sentiment thresholds
SENTIMENT_THRESHOLDS = {
    "extreme_positive": 0.8,     # Very bullish sentiment
    "positive": 0.3,             # Bullish sentiment
    "neutral_high": 0.1,         # Slightly bullish
    "neutral": 0.0,              # Neutral sentiment
    "neutral_low": -0.1,         # Slightly bearish
    "negative": -0.3,            # Bearish sentiment
    "extreme_negative": -0.8,    # Very bearish sentiment
    "abnormal_change": 0.4,      # Abnormal sentiment change threshold
    "hype_detection": 0.7,       # Sentiment level to trigger hype alert
    "panic_detection": -0.7,     # Sentiment level to trigger panic alert
    "minimum_mentions": 50       # Minimum mentions to consider sentiment reliable
}

# Sentiment weights
SENTIMENT_WEIGHTS = {
    "news": 0.6,                # Weight for news sentiment
    "social": 0.4,               # Weight for social media sentiment
    "twitter": 0.5,              # Weight for Twitter within social
    "reddit": 0.3,               # Weight for Reddit within social
    "telegram": 0.2,             # Weight for Telegram within social
    "recency_factor": 1.5        # Factor to weight recent data more heavily
}

class SentimentAnalyzer:
    """Analyzes sentiment data from news and social media sources"""
    
    def __init__(self):
        """Initialize the sentiment analyzer"""
        self.news_data = {}
        self.social_data = {}
        self.sentiment_scores = {}
        self.market_sentiment = {
            "overall_score": 0.0,
            "classification": "Neutral",
            "trend": "Stable",
            "confidence": 0.0
        }
        self.top_coins = []
        self.previous_sentiment = {}  # For tracking sentiment changes
        self.abnormal_changes = []
        
        # Set timestamp
        self.timestamp = CURRENT_TIME
        
        # Track if data is loaded
        self.data_loaded = False
        
        # Load previous sentiment summary if available
        self._load_previous_sentiment()
        
        logger.info("Sentiment Analyzer initialized")
    
    def _load_previous_sentiment(self) -> None:
        """Load previous sentiment data for change detection"""
        try:
            if os.path.exists(SENTIMENT_SUMMARY_FILE):
                with open(SENTIMENT_SUMMARY_FILE, "r", encoding="utf-8") as f:
                    previous_data = json.load(f)
                
                if "coin_sentiment" in previous_data:
                    self.previous_sentiment = {
                        coin: data.get("sentiment_score", 0.0) 
                        for coin, data in previous_data["coin_sentiment"].items()
                    }
                
                logger.info(f"Loaded previous sentiment data for {len(self.previous_sentiment)} coins")
        except Exception as e:
            logger.warning(f"Could not load previous sentiment data: {e}")
    
    def load_data(self) -> bool:
        """
        Load news and social sentiment data
        
        Returns:
            bool: Whether data was loaded successfully
        """
        news_loaded = self._load_news_data()
        social_loaded = self._load_social_data()
        
        self.data_loaded = news_loaded or social_loaded
        
        if not self.data_loaded:
            logger.error("Failed to load both news and social data")
            return False
        
        return self.data_loaded
    
    def _load_news_data(self) -> bool:
        """
        Load news sentiment data
        
        Returns:
            bool: Whether news data was loaded successfully
        """
        try:
            if not os.path.exists(NEWS_DATA_FILE):
                logger.warning(f"{NEWS_DATA_FILE} not found")
                return False
            
            with open(NEWS_DATA_FILE, "r", encoding="utf-8") as f:
                self.news_data = json.load(f)
            
            # Validate news data format
            if not self._validate_news_data():
                logger.error(f"Invalid format in {NEWS_DATA_FILE}")
                return False
            
            logger.info(f"News data loaded successfully")
            return True
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in {NEWS_DATA_FILE}")
            return False
        except Exception as e:
            logger.error(f"Error loading news data: {str(e)}")
            return False
    
    def _validate_news_data(self) -> bool:
        """
        Validate news data format
        
        Returns:
            bool: Whether news data format is valid
        """
        if not isinstance(self.news_data, dict):
            return False
        
        if "articles" not in self.news_data or not isinstance(self.news_data["articles"], list):
            return False
        
        # Check if we have at least some articles
        if len(self.news_data["articles"]) == 0:
            logger.warning("News data contains no articles")
            return True  # Empty but valid
        
        # Check first article for required fields
        first_article = self.news_data["articles"][0]
        required_fields = ["title", "sentiment", "coins", "timestamp"]
        
        if not all(field in first_article for field in required_fields):
            return False
        
        return True
    
    def _load_social_data(self) -> bool:
        """
        Load social media sentiment data
        
        Returns:
            bool: Whether social data was loaded successfully
        """
        try:
            if not os.path.exists(SOCIAL_SENTIMENT_FILE):
                logger.warning(f"{SOCIAL_SENTIMENT_FILE} not found")
                return False
            
            with open(SOCIAL_SENTIMENT_FILE, "r", encoding="utf-8") as f:
                self.social_data = json.load(f)
            
            # Validate social data format
            if not self._validate_social_data():
                logger.error(f"Invalid format in {SOCIAL_SENTIMENT_FILE}")
                return False
            
            logger.info(f"Social sentiment data loaded successfully")
            return True
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in {SOCIAL_SENTIMENT_FILE}")
            return False
        except Exception as e:
            logger.error(f"Error loading social data: {str(e)}")
            return False
    
    def _validate_social_data(self) -> bool:
        """
        Validate social media data format
        
        Returns:
            bool: Whether social data format is valid
        """
        if not isinstance(self.social_data, dict):
            return False
        
        # Check for platforms
        platforms = ["twitter", "reddit", "telegram"]
        if not any(platform in self.social_data for platform in platforms):
            return False
        
        # Check each platform's format
        for platform in platforms:
            if platform in self.social_data:
                if not isinstance(self.social_data[platform], list):
                    return False
                
                # If there's data, check format of first item
                if self.social_data[platform]:
                    first_item = self.social_data[platform][0]
                    required_fields = ["coin", "sentiment", "mentions", "timestamp"]
                    
                    if not all(field in first_item for field in required_fields):
                        return False
        
        return True
    
    def analyze_sentiment(self) -> bool:
        """
        Analyze sentiment data and calculate scores
        
        Returns:
            bool: Whether analysis was successful
        """
        if not self.data_loaded:
            logger.error("Cannot analyze sentiment: No data loaded")
            return False
        
        try:
            # Reset sentiment scores
            self.sentiment_scores = {}
            
            # Process news sentiment
            self._analyze_news_sentiment()
            
            # Process social media sentiment
            self._analyze_social_sentiment()
            
            # Calculate final sentiment scores by combining news and social data
            self._calculate_combined_sentiment()
            
            # Calculate overall market sentiment
            self._calculate_market_sentiment()
            
            # Find top coins by mentions and sentiment
            self._find_top_coins()
            
            # Detect abnormal sentiment changes
            self._detect_sentiment_changes()
            
            # Check for extreme sentiment cases
            self._detect_extreme_sentiment()
            
            logger.info("Sentiment analysis completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return False
    
    def _analyze_news_sentiment(self) -> None:
        """Analyze news data and extract sentiment by coin"""
        if "articles" not in self.news_data:
            logger.warning("No articles found in news data")
            return
        
        logger.info(f"Analyzing {len(self.news_data['articles'])} news articles")
        
        # Initialize data structures
        coin_sentiments = defaultdict(list)
        coin_mentions = defaultdict(int)
        
        # Process each article
        for article in self.news_data["articles"]:
            # Skip articles without necessary data
            if not all(key in article for key in ["sentiment", "coins", "timestamp"]):
                continue
            
            # Extract data
            sentiment = float(article["sentiment"])
            coins = article["coins"]
            timestamp = article["timestamp"]
            
            # Apply recency weighting
            recency_weight = self._calculate_recency_weight(timestamp)
            weighted_sentiment = sentiment * recency_weight
            
            # Add to each mentioned coin
            for coin in coins:
                coin_sentiments[coin].append(weighted_sentiment)
                coin_mentions[coin] += 1
        
        # Calculate average sentiment for each coin
        for coin, sentiments in coin_sentiments.items():
            if sentiments:
                avg_sentiment = sum(sentiments) / len(sentiments)
                mentions = coin_mentions[coin]
                
                # Store news sentiment
                if coin not in self.sentiment_scores:
                    self.sentiment_scores[coin] = {
                        "news_sentiment": avg_sentiment,
                        "news_mentions": mentions,
                        "social_sentiment": 0.0,
                        "social_mentions": 0,
                        "combined_sentiment": 0.0,
                        "total_mentions": mentions
                    }
                else:
                    self.sentiment_scores[coin]["news_sentiment"] = avg_sentiment
                    self.sentiment_scores[coin]["news_mentions"] = mentions
                    self.sentiment_scores[coin]["total_mentions"] += mentions
    
    def _analyze_social_sentiment(self) -> None:
        """Analyze social media data and extract sentiment by coin"""
        # Define platforms and their weights
        platforms = {
            "twitter": SENTIMENT_WEIGHTS["twitter"],
            "reddit": SENTIMENT_WEIGHTS["reddit"],
            "telegram": SENTIMENT_WEIGHTS["telegram"]
        }
        
        # Check available platforms
        available_platforms = [p for p in platforms if p in self.social_data]
        
        if not available_platforms:
            logger.warning("No social media data available")
            return
        
        # Normalize platform weights based on available data
        total_weight = sum(platforms[p] for p in available_platforms)
        normalized_weights = {p: platforms[p]/total_weight for p in available_platforms}
        
        # Process each platform
        coin_sentiments = defaultdict(float)
        coin_mentions = defaultdict(int)
        
        for platform, weight in normalized_weights.items():
            if platform not in self.social_data:
                continue
                
            platform_items = self.social_data[platform]
            logger.info(f"Analyzing {len(platform_items)} {platform} entries")
            
            for item in platform_items:
                # Skip items without necessary data
                if not all(key in item for key in ["coin", "sentiment", "mentions", "timestamp"]):
                    continue
                
                # Extract data
                coin = item["coin"]
                sentiment = float(item["sentiment"])
                mentions = int(item["mentions"])
                timestamp = item["timestamp"]
                
                # Apply recency weighting
                recency_weight = self._calculate_recency_weight(timestamp)
                weighted_sentiment = sentiment * recency_weight * weight * mentions
                
                # Add weighted sentiment to coin
                coin_sentiments[coin] += weighted_sentiment
                coin_mentions[coin] += mentions
        
        # Calculate normalized sentiment for each coin
        for coin, weighted_sum in coin_sentiments.items():
            mentions = coin_mentions[coin]
            if mentions > 0:
                # Normalize by mentions to get average sentiment
                avg_sentiment = weighted_sum / mentions
                
                # Store social sentiment
                if coin not in self.sentiment_scores:
                    self.sentiment_scores[coin] = {
                        "news_sentiment": 0.0,
                        "news_mentions": 0,
                        "social_sentiment": avg_sentiment,
                        "social_mentions": mentions,
                        "combined_sentiment": 0.0,
                        "total_mentions": mentions
                    }
                else:
                    self.sentiment_scores[coin]["social_sentiment"] = avg_sentiment
                    self.sentiment_scores[coin]["social_mentions"] = mentions
                    self.sentiment_scores[coin]["total_mentions"] += mentions
    
    def _calculate_combined_sentiment(self) -> None:
        """Calculate combined sentiment scores from news and social data"""
        for coin, data in self.sentiment_scores.items():
            news_sentiment = data["news_sentiment"]
            social_sentiment = data["social_sentiment"]
            news_mentions = data["news_mentions"]
            social_mentions = data["social_mentions"]
            
            # Skip coins with insufficient mentions
            if data["total_mentions"] < SENTIMENT_THRESHOLDS["minimum_mentions"]:
                # Use available data with lower confidence
                if news_mentions > 0 and social_mentions > 0:
                    # If both sources have data, use weighted average
                    news_weight = min(1.0, news_mentions / SENTIMENT_THRESHOLDS["minimum_mentions"])
                    social_weight = min(1.0, social_mentions / SENTIMENT_THRESHOLDS["minimum_mentions"])
                    
                    total_weight = news_weight + social_weight
                    combined = ((news_sentiment * news_weight) + (social_sentiment * social_weight)) / total_weight
                    confidence = total_weight / 2  # Max confidence is 1.0
                    
                elif news_mentions > 0:
                    # Only news data
                    combined = news_sentiment
                    confidence = min(1.0, news_mentions / SENTIMENT_THRESHOLDS["minimum_mentions"])
                    
                elif social_mentions > 0:
                    # Only social data
                    combined = social_sentiment
                    confidence = min(1.0, social_mentions / SENTIMENT_THRESHOLDS["minimum_mentions"])
                    
                else:
                    # No mentions at all
                    combined = 0.0
                    confidence = 0.0
                    
            else:
                # Calculate weighted average based on mentions
                if news_mentions > 0 and social_mentions > 0:
                    # Use sentiment weights if both sources have data
                    news_weight = SENTIMENT_WEIGHTS["news"]
                    social_weight = SENTIMENT_WEIGHTS["social"]
                    combined = (news_sentiment * news_weight) + (social_sentiment * social_weight)
                    confidence = 1.0
                    
                elif news_mentions > 0:
                    # Only news data
                    combined = news_sentiment
                    confidence = min(1.0, news_mentions / SENTIMENT_THRESHOLDS["minimum_mentions"])
                    
                elif social_mentions > 0:
                    # Only social data
                    combined = social_sentiment
                    confidence = min(1.0, social_mentions / SENTIMENT_THRESHOLDS["minimum_mentions"])
                    
                else:
                    # No mentions at all (shouldn't happen in this branch)
                    combined = 0.0
                    confidence = 0.0
            
            # Ensure the score is in range [-1.0, 1.0]
            combined = max(-1.0, min(1.0, combined))
            
            # Update sentiment data
            self.sentiment_scores[coin]["combined_sentiment"] = combined
            self.sentiment_scores[coin]["confidence"] = confidence
            self.sentiment_scores[coin]["sentiment_score"] = combined  # For easier access
            
            # Add classification based on score
            self.sentiment_scores[coin]["classification"] = self._classify_sentiment(combined)
    
    def _calculate_market_sentiment(self) -> None:
        """Calculate overall market sentiment based on individual coin sentiment"""
        # Reset market sentiment
        self.market_sentiment = {
            "overall_score": 0.0,
            "classification": "Neutral",
            "trend": "Stable",
            "confidence": 0.0,
            "sample_size": 0
        }
        
        if not self.sentiment_scores:
            logger.warning("No sentiment data available for market sentiment calculation")
            return
        
        # Get top coins by market cap/volume (use total mentions as proxy)
        coin_weights = {}
        total_mentions = sum(data["total_mentions"] for data in self.sentiment_scores.values())
        
        if total_mentions == 0:
            logger.warning("No mentions found in sentiment data")
            return
        
        # Calculate weighted score based on coin mentions
        weighted_sum = 0.0
        total_weight = 0.0
        confidence_sum = 0.0
        
        for coin, data in self.sentiment_scores.items():
            if data["total_mentions"] > 0:
                # Weight by percentage of total mentions
                weight = data["total_mentions"] / total_mentions
                confidence = data["confidence"] if "confidence" in data else 0.5
                
                weighted_sum += data["sentiment_score"] * weight * confidence
                total_weight += weight
                confidence_sum += confidence * weight
        
        # Calculate overall market sentiment
        if total_weight > 0:
            overall_score = weighted_sum / total_weight
            confidence = confidence_sum / total_weight
        else:
            overall_score = 0.0
            confidence = 0.0
        
        # Determine trend by comparing with previous overall sentiment
        trend = "Stable"
        if hasattr(self, 'previous_market_sentiment'):
            prev_score = getattr(self, 'previous_market_sentiment', {}).get("overall_score", 0.0)
            score_diff = overall_score - prev_score
            
            if score_diff > 0.1:
                trend = "Improving"
            elif score_diff < -0.1:
                trend = "Deteriorating"
        
        # Store current market sentiment as previous for next run
        self.previous_market_sentiment = {
            "overall_score": overall_score,
            "timestamp": self.timestamp
        }
        
        # Update market sentiment
        self.market_sentiment = {
            "overall_score": overall_score,
            "classification": self._classify_sentiment(overall_score),
            "trend": trend,
            "confidence": confidence,
            "sample_size": len(self.sentiment_scores),
            "total_mentions": total_mentions,
            "timestamp": self.timestamp
        }
        
        logger.info(f"Market sentiment: {self.market_sentiment['classification']} (Score: {overall_score:.2f}, Trend: {trend})")
    
    def _find_top_coins(self) -> None:
        """Find top coins by mentions and sentiment"""
        if not self.sentiment_scores:
            self.top_coins = []
            return
        
        # Sort coins by total mentions
        mentions_sorted = sorted(
            [(coin, data["total_mentions"]) for coin, data in self.sentiment_scores.items()],
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Get top 10 most mentioned coins
        top_by_mentions = mentions_sorted[:10]
        
        # Sort coins by most positive sentiment (with minimum mentions)
        positive_sorted = sorted(
            [(coin, data["sentiment_score"]) for coin, data in self.sentiment_scores.items() 
             if data["total_mentions"] >= SENTIMENT_THRESHOLDS["minimum_mentions"]],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Get top 5 most positive coins
        top_positive = positive_sorted[:5]
        
        # Sort coins by most negative sentiment (with minimum mentions)
        negative_sorted = sorted(
            [(coin, data["sentiment_score"]) for coin, data in self.sentiment_scores.items()
             if data["total_mentions"] >= SENTIMENT_THRESHOLDS["minimum_mentions"]],
            key=lambda x: x[1]
        )
        
        # Get top 5 most negative coins
        top_negative = negative_sorted[:5]
        
        # Create top coins summary
        self.top_coins = {
            "by_mentions": [{"coin": coin, "mentions": mentions} for coin, mentions in top_by_mentions],
            "most_positive": [{"coin": coin, "sentiment": score} for coin, score in top_positive],
            "most_negative": [{"coin": coin, "sentiment": score} for coin, score in top_negative]
        }
    
    def _detect_sentiment_changes(self) -> None:
        """Detect abnormal changes in sentiment"""
        self.abnormal_changes = []
        
        # Compare current sentiment with previous
        for coin, data in self.sentiment_scores.items():
            if coin in self.previous_sentiment:
                previous_score = self.previous_sentiment[coin]
                current_score = data["sentiment_score"]
                
                # Calculate change
                change = current_score - previous_score
                
                # Check if change exceeds threshold
                if abs(change) >= SENTIMENT_THRESHOLDS["abnormal_change"]:
                    # Only include if there are sufficient mentions
                    if data["total_mentions"] >= SENTIMENT_THRESHOLDS["minimum_mentions"]:
                        self.abnormal_changes.append({
                            "coin": coin,
                            "previous_score": previous_score,
                            "current_score": current_score,
                            "change": change,
                            "direction": "up" if change > 0 else "down",
                            "mentions": data["total_mentions"]
                        })
        
        # Sort by magnitude of change
        self.abnormal_changes.sort(key=lambda x: abs(x["change"]), reverse=True)
        
        # Log abnormal changes
        if self.abnormal_changes:
            logger.info(f"Detected {len(self.abnormal_changes)} coins with abnormal sentiment changes")
            for change in self.abnormal_changes[:3]:  # Log top 3
                direction = "↑" if change["direction"] == "up" else "↓"
                logger.info(f"{change['coin']}: {change['previous_score']:.2f} → {change['current_score']:.2f} ({direction} {abs(change['change']):.2f})")
    
    def _detect_extreme_sentiment(self) -> None:
        """Detect extreme sentiment cases and send alerts"""
        if not ALERT_SYSTEM_AVAILABLE:
            return
        
        # Check for coins with extreme sentiment
        for coin, data in self.sentiment_scores.items():
            score = data["sentiment_score"]
            mentions = data["total_mentions"]
            
            # Only consider coins with sufficient mentions
            if mentions < SENTIMENT_THRESHOLDS["minimum_mentions"]:
                continue
            
            # Check for extreme positive sentiment (hype)
            if score >= SENTIMENT_THRESHOLDS["hype_detection"]:
                alert_system.warning(
                    f"Extreme positive sentiment detected for {coin}",
                    {
                        "coin": coin,
                        "sentiment_score": score,
                        "mentions": mentions,
                        "type": "hype_alert"
                    },
                    "sentiment_analyzer"
                )
                logger.warning(f"HYPE ALERT: {coin} sentiment at {score:.2f} with {mentions} mentions")
            
            # Check for extreme negative sentiment (panic)
            elif score <= SENTIMENT_THRESHOLDS["panic_detection"]:
                alert_system.warning(
                    f"Extreme negative sentiment detected for {coin}",
                    {
                        "coin": coin,
                        "sentiment_score": score,
                        "mentions": mentions,
                        "type": "panic_alert"
                    },
                    "sentiment_analyzer"
                )
                logger.warning(f"PANIC ALERT: {coin} sentiment at {score:.2f} with {mentions} mentions")
        
        # Check for abnormal changes
        for change in self.abnormal_changes:
            if abs(change["change"]) >= SENTIMENT_THRESHOLDS["abnormal_change"] * 1.5:  # More significant threshold
                direction = "positive" if change["direction"] == "up" else "negative"
                alert_system.info(
                    f"Significant {direction} sentiment shift for {change['coin']}",
                    {
                        "coin": change["coin"],
                        "previous_score": change["previous_score"],
                        "current_score": change["current_score"],
                        "change": change["change"],
                        "mentions": change["mentions"],
                        "type": "sentiment_shift"
                    },
                    "sentiment_analyzer"
                )
        
        # Check for extreme market sentiment
        market_score = self.market_sentiment["overall_score"]
        
        if market_score >= SENTIMENT_THRESHOLDS["extreme_positive"]:
            alert_system.warning(
                "Extreme bullish market sentiment detected",
                {
                    "market_sentiment": market_score,
                    "classification": self.market_sentiment["classification"],
                    "type": "market_euphoria"
                },
                "sentiment_analyzer"
            )
            logger.warning(f"MARKET EUPHORIA ALERT: Market sentiment at {market_score:.2f}")
            
        elif market_score <= SENTIMENT_THRESHOLDS["extreme_negative"]:
            alert_system.warning(
                "Extreme bearish market sentiment detected",
                {
                    "market_sentiment": market_score,
                    "classification": self.market_sentiment["classification"],
                    "type": "market_panic"
                },
                "sentiment_analyzer"
            )
            logger.warning(f"MARKET PANIC ALERT: Market sentiment at {market_score:.2f}")
    
    def _calculate_recency_weight(self, timestamp: str) -> float:
        """
        Calculate recency weight based on timestamp
        
        Args:
            timestamp (str): Timestamp string
            
        Returns:
            float: Recency weight factor
        """
        try:
            # Parse timestamps
            current_time = datetime.datetime.strptime(self.timestamp, "%Y-%m-%d %H:%M:%S")
            item_time = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            
            # Calculate time difference in hours
            time_diff = (current_time - item_time).total_seconds() / 3600
            
            # Apply decay function (more recent = higher weight)
            # Exponential decay with half-life of 24 hours
            if time_diff <= 0:
                return SENTIMENT_WEIGHTS["recency_factor"]  # Current or future (shouldn't happen)
            elif time_diff <= 6:
                return SENTIMENT_WEIGHTS["recency_factor"]  # Very recent (last 6 hours)
            elif time_diff <= 24:
                return 1.0  # Recent (last 24 hours)
            elif time_diff <= 72:
                return 0.8  # Somewhat recent (1-3 days)
            else:
                return 0.5  # Old data
                
        except Exception as e:
            logger.debug(f"Error calculating recency weight: {e}")
            return 1.0  # Default weight
    
    def _classify_sentiment(self, score: float) -> str:
        """
        Classify sentiment score into categories
        
        Args:
            score (float): Sentiment score
            
        Returns:
            str: Sentiment classification
        """
        if score >= SENTIMENT_THRESHOLDS["extreme_positive"]:
            return "Very Bullish"
        elif score >= SENTIMENT_THRESHOLDS["positive"]:
            return "Bullish"
        elif score >= SENTIMENT_THRESHOLDS["neutral_high"]:
            return "Slightly Bullish"
        elif score > SENTIMENT_THRESHOLDS["neutral_low"]:
            return "Neutral"
        elif score > SENTIMENT_THRESHOLDS["negative"]:
            return "Slightly Bearish"
        elif score > SENTIMENT_THRESHOLDS["extreme_negative"]:
            return "Bearish"
        else:
            return "Very Bearish"
    
    def generate_sentiment_summary(self) -> Dict[str, Any]:
        """
        Generate a complete sentiment summary
        
        Returns:
            Dict[str, Any]: Sentiment summary
        """
        # Generate summary timestamp
        summary_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create summary
        summary = {
            "timestamp": summary_timestamp,
            "market_sentiment": self.market_sentiment,
            "coin_sentiment": {}
        }
        
        # Add sentiment data for each coin
        for coin, data in self.sentiment_scores.items():
            summary["coin_sentiment"][coin] = {
                "sentiment_score": data["sentiment_score"],
                "classification": data["classification"] if "classification" in data else self._classify_sentiment(data["sentiment_score"]),
                "confidence": data["confidence"] if "confidence" in data else 0.5,
                "news_sentiment": data["news_sentiment"],
                "social_sentiment": data["social_sentiment"],
                "total_mentions": data["total_mentions"]
            }
        
        # Add top coins
        summary["top_coins"] = self.top_coins
        
        # Add sentiment changes
        summary["sentiment_changes"] = self.abnormal_changes
        
        # Add source statistics
        summary["source_stats"] = {
            "news_count": len(self.news_data.get("articles", [])),
            "social_platforms": {
                platform: len(data) for platform, data in self.social_data.items() if isinstance(data, list)
            }
        }
        
        return summary
    
    def save_sentiment_summary(self) -> bool:
        """
        Save sentiment summary to file
        
        Returns:
            bool: Whether the save was successful
        """
        try:
            summary = self.generate_sentiment_summary()
            
            with open(SENTIMENT_SUMMARY_FILE, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Sentiment summary saved to {SENTIMENT_SUMMARY_FILE}")
            
            # Also save to memory core if available
            if MEMORY_CORE_AVAILABLE:
                memory_core.add_memory_record(
                    source="sentiment_analyzer",
                    category="sentiment_analysis",
                    data=summary,
                    tags=["sentiment", "market_analysis"]
                )
                logger.info("Sentiment data added to memory core")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving sentiment summary: {str(e)}")
            return False
    
    def generate_learning_feedback(self) -> Dict[str, Any]:
        """
        Generate learning feedback for the learning engine
        
        Returns:
            Dict[str, Any]: Learning feedback
        """
        feedback = {
            "timestamp": self.timestamp,
            "market_sentiment": {
                "score": self.market_sentiment["overall_score"],
                "classification": self.market_sentiment["classification"],
                "trend": self.market_sentiment["trend"]
            },
            "coins": {}
        }
        
        # Add data for coins with sufficient mentions
        for coin, data in self.sentiment_scores.items():
            if data["total_mentions"] >= SENTIMENT_THRESHOLDS["minimum_mentions"]:
                feedback["coins"][coin] = {
                    "sentiment_score": data["sentiment_score"],
                    "confidence": data["confidence"] if "confidence" in data else 0.5,
                    "mentions": data["total_mentions"],
                    "classification": data["classification"] if "classification" in data else self._classify_sentiment(data["sentiment_score"])
                }
        
        # Add sentiment change information
        if self.abnormal_changes:
            feedback["sentiment_changes"] = [
                {
                    "coin": change["coin"],
                    "change": change["change"],
                    "direction": change["direction"]
                }
                for change in self.abnormal_changes
            ]
        
        return feedback
    
    def save_learning_feedback(self) -> bool:
        """
        Save learning feedback to file
        
        Returns:
            bool: Whether the save was successful
        """
        try:
            feedback = self.generate_learning_feedback()
            
            with open(LEARNING_FEEDBACK_FILE, "w", encoding="utf-8") as f:
                json.dump(feedback, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Learning feedback saved to {LEARNING_FEEDBACK_FILE}")
            
            # Push to learning engine if available
            if LEARNING_ENGINE_AVAILABLE:
                try:
                    # This requires learning_engine to have a function for processing sentiment data
                    # The exact interface might vary
                    learning_engine.process_sentiment_data(feedback)
                    logger.info("Sentiment data sent to learning engine")
                except Exception as le:
                    logger.warning(f"Failed to send data to learning engine: {le}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving learning feedback: {str(e)}")
            return False
    
    def display_sentiment_summary(self) -> None:
        """Display sentiment summary in terminal"""
        try:
            # Display header
            print(f"\n{COLORS['bright']}{COLORS['cyan']}{'=' * 60}{COLORS['reset']}")
            print(f"{COLORS['bright']}{COLORS['cyan']}📊 SENTIMENT ANALYSIS SUMMARY{COLORS['reset']}")
            print(f"{COLORS['bright']}{COLORS['cyan']}{'=' * 60}{COLORS['reset']}")
            
            # Market sentiment
            market_score = self.market_sentiment["overall_score"]
            market_class = self.market_sentiment["classification"]
            market_trend = self.market_sentiment["trend"]
            
            # Determine color based on sentiment
            if market_score >= SENTIMENT_THRESHOLDS["positive"]:
                sentiment_color = COLORS["green"]
            elif market_score <= SENTIMENT_THRESHOLDS["negative"]:
                sentiment_color = COLORS["red"]
            else:
                sentiment_color = COLORS["yellow"]
            
            print(f"{COLORS['bright']}Market Sentiment: {sentiment_color}{market_class}{COLORS['reset']}")
            print(f"Score: {sentiment_color}{market_score:.2f}{COLORS['reset']} | Trend: {market_trend}")
            print(f"Sample Size: {self.market_sentiment.get('sample_size', 0)} coins, {self.market_sentiment.get('total_mentions', 0)} mentions")
            
            # Top coins by mentions
            if self.top_coins and "by_mentions" in self.top_coins:
                print(f"\n{COLORS['bright']}Most Discussed Coins:{COLORS['reset']}")
                for i, item in enumerate(self.top_coins["by_mentions"][:5], 1):
                    coin = item["coin"]
                    mentions = item["mentions"]
                    # Get sentiment data for this coin
                    sentiment = 0.0
                    sentiment_class = "Neutral"
                    if coin in self.sentiment_scores:
                        sentiment = self.sentiment_scores[coin]["sentiment_score"]
                        sentiment_class = self.sentiment_scores[coin].get("classification", self._classify_sentiment(sentiment))
                    
                    # Determine color based on sentiment
                    if sentiment >= SENTIMENT_THRESHOLDS["positive"]:
                        coin_color = COLORS["green"]
                    elif sentiment <= SENTIMENT_THRESHOLDS["negative"]:
                        coin_color = COLORS["red"]
                    else:
                        coin_color = COLORS["white"]
                    
                    print(f"  {i}. {COLORS['bright']}{coin}{COLORS['reset']} - {mentions} mentions | Sentiment: {coin_color}{sentiment_class} ({sentiment:.2f}){COLORS['reset']}")
            
            # Most positive coins
            if self.top_coins and "most_positive" in self.top_coins and self.top_coins["most_positive"]:
                print(f"\n{COLORS['bright']}Most Bullish Coins:{COLORS['reset']}")
                for i, item in enumerate(self.top_coins["most_positive"], 1):
                    coin = item["coin"]
                    score = item["sentiment"]
                    mentions = self.sentiment_scores.get(coin, {}).get("total_mentions", 0)
                    print(f"  {i}. {COLORS['bright']}{coin}{COLORS['reset']} - {COLORS['green']}Score: {score:.2f}{COLORS['reset']} ({mentions} mentions)")
            
            # Most negative coins
            if self.top_coins and "most_negative" in self.top_coins and self.top_coins["most_negative"]:
                print(f"\n{COLORS['bright']}Most Bearish Coins:{COLORS['reset']}")
                for i, item in enumerate(self.top_coins["most_negative"], 1):
                    coin = item["coin"]
                    score = item["sentiment"]
                    mentions = self.sentiment_scores.get(coin, {}).get("total_mentions", 0)
                    print(f"  {i}. {COLORS['bright']}{coin}{COLORS['reset']} - {COLORS['red']}Score: {score:.2f}{COLORS['reset']} ({mentions} mentions)")
            
            # Significant sentiment changes
            if self.abnormal_changes:
                print(f"\n{COLORS['bright']}Significant Sentiment Changes:{COLORS['reset']}")
                for i, change in enumerate(self.abnormal_changes[:5], 1):
                    coin = change["coin"]
                    prev = change["previous_score"]
                    curr = change["current_score"]
                    direction = "↑" if change["direction"] == "up" else "↓"
                    change_val = abs(change["change"])
                    
                    # Determine color based on direction
                    change_color = COLORS["green"] if change["direction"] == "up" else COLORS["red"]
                    
                    print(f"  {i}. {COLORS['bright']}{coin}{COLORS['reset']} - {prev:.2f} → {curr:.2f} {change_color}({direction} {change_val:.2f}){COLORS['reset']}")
            
            # Source statistics
            print(f"\n{COLORS['bright']}Source Statistics:{COLORS['reset']}")
            print(f"  News Articles: {len(self.news_data.get('articles', []))}")
            
            for platform, items in self.social_data.items():
                if isinstance(items, list):
                    print(f"  {platform.capitalize()}: {len(items)} records")
            
            print(f"{COLORS['bright']}{COLORS['cyan']}{'=' * 60}{COLORS['reset']}\n")
            
        except Exception as e:
            logger.error(f"Error displaying sentiment summary: {str(e)}")
    
    def run(self) -> bool:
        """
        Run the complete sentiment analysis pipeline
        
        Returns:
            bool: Whether the analysis was successful
        """
        try:
            # Load news and social data
            if not self.load_data():
                logger.error("Failed to load sentiment data")
                print(f"{COLORS['yellow']}⚠️ Failed to load sentiment data. Check if data files exist and are valid.{COLORS['reset']}")
                return False
            
            # Analyze sentiment
            if not self.analyze_sentiment():
                logger.error("Failed to analyze sentiment")
                print(f"{COLORS['red']}❌ Failed to analyze sentiment data.{COLORS['reset']}")
                return False
            
            # Save results
            if not self.save_sentiment_summary():
                logger.warning("Failed to save sentiment summary")
                print(f"{COLORS['yellow']}⚠️ Failed to save sentiment summary, but analysis completed.{COLORS['reset']}")
            
            # Save learning feedback
            if not self.save_learning_feedback():
                logger.warning("Failed to save learning feedback")
                print(f"{COLORS['yellow']}⚠️ Failed to save learning feedback, but analysis completed.{COLORS['reset']}")
            
            # Display summary
            self.display_sentiment_summary()
            
            return True
            
        except Exception as e:
            logger.error(f"Error running sentiment analysis: {str(e)}")
            print(f"{COLORS['red']}❌ Error running sentiment analysis: {str(e)}{COLORS['reset']}")
            return False

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="SentientTrader.AI Sentiment Analyzer")
    parser.add_argument("--coin", type=str, help="Display sentiment for specific coin")
    parser.add_argument("--market", action="store_true", help="Display only market sentiment")
    parser.add_argument("--changes", action="store_true", help="Display sentiment changes")
    parser.add_argument("--summary", action="store_true", help="Display full sentiment summary")
    parser.add_argument("--no-display", action="store_true", help="Don't display results in terminal")
    return parser.parse_args()

def main() -> int:
    """Run from command line"""
    args = parse_arguments()
    
    try:
        # Initialize and run sentiment analyzer
        analyzer = SentimentAnalyzer()
        
        # Run sentiment analysis
        success = analyzer.run()
        
        if not success:
            print(f"{COLORS['red']}Failed to complete sentiment analysis{COLORS['reset']}")
            return 1
        
        # Handle command line options
        if not args.no_display:
            if args.coin:
                coin = args.coin.upper()
                sentiment = analyzer.sentiment_scores.get(coin, {})
                
                if sentiment:
                    score = sentiment.get("sentiment_score", 0.0)
                    classification = sentiment.get("classification", "Unknown")
                    mentions = sentiment.get("total_mentions", 0)
                    
                    # Determine color based on sentiment
                    if score >= SENTIMENT_THRESHOLDS["positive"]:
                        color = COLORS["green"]
                    elif score <= SENTIMENT_THRESHOLDS["negative"]:
                        color = COLORS["red"]
                    else:
                        color = COLORS["yellow"]
                    
                    print(f"\n{COLORS['bright']}Sentiment for {coin}:{COLORS['reset']}")
                    print(f"Classification: {color}{classification}{COLORS['reset']}")
                    print(f"Score: {color}{score:.2f}{COLORS['reset']}")
                    print(f"Mentions: {mentions}")
                    
                    if "news_sentiment" in sentiment and "social_sentiment" in sentiment:
                        print(f"News Sentiment: {sentiment['news_sentiment']:.2f}")
                        print(f"Social Sentiment: {sentiment['social_sentiment']:.2f}")
                else:
                    print(f"{COLORS['yellow']}No sentiment data found for {coin}{COLORS['reset']}")
                    
            elif args.market:
                market = analyzer.market_sentiment
                score = market.get("overall_score", 0.0)
                classification = market.get("classification", "Unknown")
                trend = market.get("trend", "Stable")
                
                # Determine color based on sentiment
                if score >= SENTIMENT_THRESHOLDS["positive"]:
                    color = COLORS["green"]
                elif score <= SENTIMENT_THRESHOLDS["negative"]:
                    color = COLORS["red"]
                else:
                    color = COLORS["yellow"]
                
                print(f"\n{COLORS['bright']}Market Sentiment:{COLORS['reset']}")
                print(f"Classification: {color}{classification}{COLORS['reset']}")
                print(f"Score: {color}{score:.2f}{COLORS['reset']}")
                print(f"Trend: {trend}")
                
            elif args.changes:
                changes = analyzer.abnormal_changes
                
                if changes:
                    print(f"\n{COLORS['bright']}Significant Sentiment Changes:{COLORS['reset']}")
                    for i, change in enumerate(changes, 1):
                        coin = change["coin"]
                        prev = change["previous_score"]
                        curr = change["current_score"]
                        direction = "↑" if change["direction"] == "up" else "↓"
                        change_val = abs(change["change"])
                        
                        # Determine color based on direction
                        change_color = COLORS["green"] if change["direction"] == "up" else COLORS["red"]
                        
                        print(f"{i}. {COLORS['bright']}{coin}{COLORS['reset']} - {prev:.2f} → {curr:.2f} {change_color}({direction} {change_val:.2f}){COLORS['reset']}")
                else:
                    print(f"{COLORS['yellow']}No significant sentiment changes detected{COLORS['reset']}")
                    
            elif args.summary:
                analyzer.display_sentiment_summary()
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n{COLORS['yellow']}Process interrupted by user.{COLORS['reset']}")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        print(f"\n{COLORS['red']}An unexpected error occurred: {str(e)}{COLORS['reset']}")
        logger.exception("Unexpected error in main")
        return 1

if __name__ == "__main__":
    sys.exit(main())
