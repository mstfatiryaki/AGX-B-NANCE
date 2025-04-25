from random import choice
from random import choice
import random
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SentientTrader.AI - Deep Thinking Process
-----------------------------------------
This module represents DeepSeek's cognitive analysis system that evaluates trading decisions,
generates alternative scenarios, assesses risks, and provides critical thinking against
GPT-4o's recommendations to optimize the system's intelligence.

Created by: SentientTrader.AI Team
Date: 2025-04-23
Version: 1.0.0
"""

import os
import sys
import json
import time
import logging
import random
import argparse
import datetime
import statistics
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from colorama import Fore, Style, init as colorama_init

# Initialize colorama for cross-platform colored terminal output
colorama_init(autoreset=True)

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
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
    from modules.ai import thinking_controller
    THINKING_CONTROLLER_AVAILABLE = True
except ImportError:
    THINKING_CONTROLLER_AVAILABLE = False
    logging.warning("thinking_controller module not available, analysis sharing disabled")

try:
    from modules.data import sentiment_analyzer
    SENTIMENT_ANALYZER_AVAILABLE = True
except ImportError:
    SENTIMENT_ANALYZER_AVAILABLE = False
    logging.warning("sentiment_analyzer module not available, using cached sentiment data")

try:
    from modules.data import coingecko_collector
    COINGECKO_AVAILABLE = True
except ImportError:
    COINGECKO_AVAILABLE = False
    logging.warning("coingecko_collector module not available, using cached market data")

try:
    from modules.data import binance_collector
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    logging.warning("binance_collector module not available, using cached market data")

try:
    from modules.utils import report_generator
    REPORT_GENERATOR_AVAILABLE = True
except ImportError:
    REPORT_GENERATOR_AVAILABLE = False
    logging.warning("report_generator module not available, reports disabled")

# Configure logging
LOG_DIR = os.path.join(project_root, "logs")
THINKING_LOGS_DIR = os.path.join(LOG_DIR, "thinking")

# Create log directories if they don't exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(THINKING_LOGS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "deep_thinking.log"))
    ]
)

logger = logging.getLogger("DeepThinking")

# Constants
VERSION = "1.0.0"
DEFAULT_THINKING_LEVEL = 3
DATA_DIR = os.path.join(project_root, "data")
SENTIMENT_DATA_FILE = os.path.join(DATA_DIR, "sentiment_summary.json")
COINGECKO_DATA_FILE = os.path.join(DATA_DIR, "coingecko_data.json")
BINANCE_DATA_FILE = os.path.join(DATA_DIR, "binance_data.json")
NEWS_DATA_FILE = os.path.join(DATA_DIR, "news_data.json")


class ThinkingLevel(Enum):
    """Enum for different thinking depth levels."""
    MINIMAL = 1   # Basic sentiment and price check
    LOW = 2       # Add basic technical analysis
    MODERATE = 3  # Add risk assessment and simple scenarios
    HIGH = 4      # Add detailed scenario analysis and market context
    MAXIMUM = 5   # Add complex correlations, sentiment deep dive, and exhaustive scenarios


class ScenarioType(Enum):
    """Enum for different market scenario types."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"
    MANIPULATION = "manipulation"
    BLACK_SWAN = "black_swan"


class ThoughtCategory(Enum):
    """Enum for different thought categories."""
    SENTIMENT = "sentiment"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    RISK = "risk"
    OPPORTUNITY = "opportunity"
    CORRELATION = "correlation"
    NEWS = "news"
    PATTERN = "pattern"
    CONTRARIAN = "contrarian"
    RECOMMENDATION = "recommendation"


class DeepThinkingProcess:
    """
    DeepSeek's cognitive analysis system that evaluates trading decisions,
    generates alternative scenarios, and provides critical thinking.
    """
    
    def __init__(self, 
                 real_mode: bool = False, 
                 silent_mode: bool = False,
                 thinking_level: int = DEFAULT_THINKING_LEVEL):
        """
        Initialize the deep thinking process.
        
        Args:
            real_mode (bool): Whether the system is using real money
            silent_mode (bool): Whether to minimize console output
            thinking_level (int): Thinking depth level (1-5)
        """
        self.real_mode = real_mode
        self.silent_mode = silent_mode
        self.thinking_level = self._validate_thinking_level(thinking_level)
        self.sentiment_data = {}
        self.market_data = {}
        self.news_data = {}
        self.cached_analyses = {}
        
        # Initialize required modules
        self._init_modules()
        
        # Load cached data files
        self._load_data_files()
        
        logger.info(f"Deep Thinking Process initialized (thinking_level: {self.thinking_level.name}, real_mode: {real_mode})")
        
        if not silent_mode:
            self._print_init_message()
    
    def _validate_thinking_level(self, level: int) -> ThinkingLevel:
        """
        Validate and convert numeric thinking level to enum.
        
        Args:
            level (int): Numeric thinking level (1-5)
            
        Returns:
            ThinkingLevel: Thinking level enum
        """
        try:
            # Ensure level is within bounds
            bounded_level = max(1, min(5, level))
            return ThinkingLevel(bounded_level)
        except ValueError:
            logger.warning(f"Invalid thinking level: {level}, using MODERATE (3)")
            return ThinkingLevel.MODERATE
    
    def _init_modules(self):
        """Initialize required system modules if available."""
        # Initialize alert system if available
        if ALERT_SYSTEM_AVAILABLE:
            self.alert = alert_system.AlertSystem(
                module_name="deep_thinking",
                real_mode=self.real_mode,
                log_to_file=True
            )
        else:
            self.alert = logger
            
        # Initialize memory core if available
        if MEMORY_CORE_AVAILABLE:
            self.memory = memory_core.MemoryCore(real_mode=self.real_mode)
        else:
            self.memory = None
            
        # Initialize thinking controller if available
        if THINKING_CONTROLLER_AVAILABLE:
            self.thinking_controller = thinking_controller.ThinkingController(
                real_mode=self.real_mode,
                silent_mode=self.silent_mode
            )
        else:
            self.thinking_controller = None
    
    def _load_data_files(self):
        """Load cached data files if direct module imports are not available."""
        try:
            # Load sentiment data
            if os.path.exists(SENTIMENT_DATA_FILE):
                with open(SENTIMENT_DATA_FILE, 'r') as f:
                    self.sentiment_data = json.load(f)
                logger.debug(f"Loaded sentiment data from {SENTIMENT_DATA_FILE}")
            
            # Load CoinGecko market data
            if os.path.exists(COINGECKO_DATA_FILE):
                with open(COINGECKO_DATA_FILE, 'r') as f:
                    self.market_data["coingecko"] = json.load(f)
                logger.debug(f"Loaded CoinGecko data from {COINGECKO_DATA_FILE}")
            
            # Load Binance market data
            if os.path.exists(BINANCE_DATA_FILE):
                with open(BINANCE_DATA_FILE, 'r') as f:
                    self.market_data["binance"] = json.load(f)
                logger.debug(f"Loaded Binance data from {BINANCE_DATA_FILE}")
            
            # Load news data if available
            if os.path.exists(NEWS_DATA_FILE):
                with open(NEWS_DATA_FILE, 'r') as f:
                    self.news_data = json.load(f)
                logger.debug(f"Loaded news data from {NEWS_DATA_FILE}")
                
        except Exception as e:
            logger.error(f"Error loading data files: {str(e)}")
    
    def _print_init_message(self):
        """Print initialization message with thinking level details."""
        level_descriptions = {
            ThinkingLevel.MINIMAL: "Basic sentiment and price check",
            ThinkingLevel.LOW: "Basic technical analysis included",
            ThinkingLevel.MODERATE: "Risk assessment and simple scenarios",
            ThinkingLevel.HIGH: "Detailed scenario analysis and market context",
            ThinkingLevel.MAXIMUM: "Complex correlations and exhaustive scenarios"
        }
        
        description = level_descriptions.get(self.thinking_level, "Custom thinking level")
        
        print(f"\n{Fore.CYAN}DeepSeek Deep Thinking Process {VERSION}{Style.RESET_ALL}")
        print(f"Thinking Level: {Fore.YELLOW}{self.thinking_level.name}{Style.RESET_ALL} ({self.thinking_level.value}/5)")
        print(f"Description: {Fore.WHITE}{description}{Style.RESET_ALL}")
        print(f"Mode: {Fore.RED if self.real_mode else Fore.GREEN}{'REAL' if self.real_mode else 'SIMULATION'}{Style.RESET_ALL}")
        print(f"Log Directory: {THINKING_LOGS_DIR}\n")
    
    def process_pre_trade(self, 
                         coin: str, 
                         gpt_decision: Dict[str, Any], 
                         strategy_type: str,
                         market_data: Dict[str, Any] = None, 
                         sentiment_data: Dict[str, Any] = None,
                         thinking_level: int = None) -> Dict[str, Any]:
        """
        Process thinking before trade execution (pre-trade analysis).
        
        Args:
            coin (str): Coin symbol
            gpt_decision (Dict): Decision from GPT-4o
            strategy_type (str): Strategy type (e.g., "long", "short", "sniper")
            market_data (Dict): Market data for the coin (optional)
            sentiment_data (Dict): Sentiment data for the coin (optional)
            thinking_level (int): Override default thinking level (optional)
            
        Returns:
            Dict: Deep thinking analysis results
        """
        start_time = time.time()
        
        if thinking_level is not None:
            current_thinking_level = self._validate_thinking_level(thinking_level)
        else:
            current_thinking_level = self.thinking_level
        
        logger.info(f"Starting pre-trade thinking for {coin} with level {current_thinking_level.name}")
        
        # Ensure we have market data
        if market_data is None:
            market_data = self._get_market_data(coin)
        
        # Ensure we have sentiment data
        if sentiment_data is None:
            sentiment_data = self._get_sentiment_data(coin)
        
        # Initialize thinking results structure
        thinking_results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "coin": coin,
            "strategy_type": strategy_type,
            "thinking_level": current_thinking_level.value,
            "thinking_type": "pre_trade",
            "gpt_decision": gpt_decision,
            "thoughts": [],
            "scenarios": [],
            "risk_assessment": {},
            "deepseek_decision": {},
            "confidence_score": 0.0,
            "agreement": False,
            "override_suggestion": False
        }
        
        # Generate thoughts based on thinking level
        self._generate_thoughts(
            thinking_results, 
            coin, 
            market_data, 
            sentiment_data, 
            current_thinking_level
        )
        
        # Generate scenarios if thinking level is moderate or higher
        if current_thinking_level.value >= ThinkingLevel.MODERATE.value:
            self._generate_scenarios(
                thinking_results,
                coin,
                market_data,
                sentiment_data,
                gpt_decision,
                current_thinking_level
            )
        
        # Assess risk if thinking level is low or higher
        if current_thinking_level.value >= ThinkingLevel.LOW.value:
            self._assess_risk(
                thinking_results,
                coin,
                market_data,
                sentiment_data,
                gpt_decision,
                current_thinking_level
            )
        
        # Generate DeepSeek's own decision
        self._generate_deepseek_decision(
            thinking_results,
            gpt_decision,
            coin,
            market_data,
            sentiment_data,
            current_thinking_level
        )
        
        # Calculate overall confidence score
        confidence_score = self._calculate_confidence_score(
            thinking_results,
            gpt_decision,
            coin,
            current_thinking_level
        )
        thinking_results["confidence_score"] = confidence_score
        
        # Determine if DeepSeek agrees with GPT-4o
        deepseek_decision = thinking_results["deepseek_decision"]
        gpt_action = gpt_decision.get("action", "").lower() if gpt_decision else ""
        gpt_direction = gpt_decision.get("direction", "").lower() if gpt_decision else ""
        deepseek_action = deepseek_decision.get("action", "").lower()
        deepseek_direction = deepseek_decision.get("direction", "").lower()
        
        agreement = (
            gpt_action == deepseek_action and
            gpt_direction == deepseek_direction
        )
        thinking_results["agreement"] = agreement
        
        # Determine if DeepSeek suggests overriding GPT-4o's decision
        override_suggestion = (
            not agreement and
            thinking_results.get("confidence_score", 0) > 0.7 and
            thinking_results.get("risk_assessment", {}).get("total_risk_score", 0) < 0.6
        )
        thinking_results["override_suggestion"] = override_suggestion
        
        # Add execution time
        execution_time = time.time() - start_time
        thinking_results["execution_time_seconds"] = execution_time
        
        # Log the thinking process
        self._log_thinking_process(thinking_results, "pre_trade")
        
        # Store in memory if available
        if MEMORY_CORE_AVAILABLE and self.memory:
            self.memory.store(
                data=thinking_results,
                category="deepseek_thinking",
                subcategory="pre_trade",
                source="deep_thinking_process",
                real_mode=self.real_mode
            )
        
        logger.info(f"Completed pre-trade thinking for {coin} in {execution_time:.2f} seconds (agreement: {agreement})")
        
        return thinking_results
    
    def process_post_trade(self, 
                          coin: str, 
                          strategy_type: str, 
                          strategy_results: Dict[str, Any],
                          market_data: Dict[str, Any] = None, 
                          sentiment_data: Dict[str, Any] = None,
                          thinking_level: int = None) -> Dict[str, Any]:
        """
        Process thinking after trade execution (post-trade analysis).
        
        Args:
            coin (str): Coin symbol
            strategy_type (str): Strategy type (e.g., "long", "short", "sniper")
            strategy_results (Dict): Results of the strategy execution
            market_data (Dict): Market data for the coin (optional)
            sentiment_data (Dict): Sentiment data for the coin (optional)
            thinking_level (int): Override default thinking level (optional)
            
        Returns:
            Dict: Deep thinking analysis results
        """
        start_time = time.time()
        
        if thinking_level is not None:
            current_thinking_level = self._validate_thinking_level(thinking_level)
        else:
            current_thinking_level = self.thinking_level
        
        logger.info(f"Starting post-trade thinking for {coin} with level {current_thinking_level.name}")
        
        # Ensure we have market data
        if market_data is None:
            market_data = self._get_market_data(coin)
        
        # Ensure we have sentiment data
        if sentiment_data is None:
            sentiment_data = self._get_sentiment_data(coin)
        
        # Initialize thinking results structure
        thinking_results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "coin": coin,
            "strategy_type": strategy_type,
            "thinking_level": current_thinking_level.value,
            "thinking_type": "post_trade",
            "strategy_results": strategy_results,
            "thoughts": [],
            "performance_analysis": {},
            "learning_insights": [],
            "improvement_suggestions": [],
            "future_outlook": {},
            "confidence_in_future": 0.0
        }
        
        # Analyze strategy performance
        self._analyze_performance(
            thinking_results,
            coin,
            strategy_results,
            current_thinking_level
        )
        
        # Generate post-trade thoughts
        self._generate_post_trade_thoughts(
            thinking_results,
            coin,
            strategy_results,
            market_data,
            sentiment_data,
            current_thinking_level
        )
        
        # Generate learning insights if thinking level is moderate or higher
        if current_thinking_level.value >= ThinkingLevel.MODERATE.value:
            self._generate_learning_insights(
                thinking_results,
                coin,
                strategy_results,
                current_thinking_level
            )
        
        # Generate improvement suggestions if thinking level is low or higher
        if current_thinking_level.value >= ThinkingLevel.LOW.value:
            self._generate_improvement_suggestions(
                thinking_results,
                coin,
                strategy_results,
                market_data,
                sentiment_data,
                current_thinking_level
            )
        
        # Generate future outlook if thinking level is high or maximum
        if current_thinking_level.value >= ThinkingLevel.HIGH.value:
            self._generate_future_outlook(
                thinking_results,
                coin,
                strategy_results,
                market_data,
                sentiment_data,
                current_thinking_level
            )
        
        # Add execution time
        execution_time = time.time() - start_time
        thinking_results["execution_time_seconds"] = execution_time
        
        # Log the thinking process
        self._log_thinking_process(thinking_results, "post_trade")
        
        # Store in memory if available
        if MEMORY_CORE_AVAILABLE and self.memory:
            self.memory.store(
                data=thinking_results,
                category="deepseek_thinking",
                subcategory="post_trade",
                source="deep_thinking_process",
                real_mode=self.real_mode
            )
        
        logger.info(f"Completed post-trade thinking for {coin} in {execution_time:.2f} seconds")
        
        return thinking_results
    
    def _generate_thoughts(self, 
                          thinking_results: Dict[str, Any], 
                          coin: str, 
                          market_data: Dict[str, Any], 
                          sentiment_data: Dict[str, Any],
                          thinking_level: ThinkingLevel):
        """
        Generate thoughts based on market and sentiment data.
        
        Args:
            thinking_results (Dict): Thinking results structure to populate
            coin (str): Coin symbol
            market_data (Dict): Market data for the coin
            sentiment_data (Dict): Sentiment data for the coin
            thinking_level (ThinkingLevel): Thinking level
        """
        thoughts = []
        
        # Always generate these basic thoughts (all thinking levels)
        
        # Sentiment thoughts
        sentiment_score = self._extract_sentiment_score(sentiment_data)
        if sentiment_score is not None:
            sentiment_thought = {
                "category": ThoughtCategory.SENTIMENT.value,
                "thought": self._generate_sentiment_thought(coin, sentiment_score),
                "sentiment_score": sentiment_score,
                "confidence": min(0.9, 0.5 + abs(sentiment_score - 0.5))
            }
            thoughts.append(sentiment_thought)
        
        # Price trend thought
        price_trend = self._analyze_price_trend(market_data)
        if price_trend:
            price_thought = {
                "category": ThoughtCategory.TECHNICAL.value,
                "thought": f"{coin} is in a {price_trend['trend']} trend with {price_trend['strength']} strength.",
                "trend": price_trend["trend"],
                "strength": price_trend["strength"],
                "confidence": price_trend["confidence"]
            }
            thoughts.append(price_thought)
        
        # For Low thinking level and above
        if thinking_level.value >= ThinkingLevel.LOW.value:
            # Volume analysis
            volume_analysis = self._analyze_volume(market_data)
            if volume_analysis:
                volume_thought = {
                    "category": ThoughtCategory.TECHNICAL.value,
                    "thought": f"Trading volume is {volume_analysis['state']} which {volume_analysis['implication']}.",
                    "volume_state": volume_analysis["state"],
                    "implication": volume_analysis["implication"],
                    "confidence": volume_analysis["confidence"]
                }
                thoughts.append(volume_thought)
            
            # Basic support/resistance
            levels = self._identify_support_resistance(market_data)
            if levels and levels.get("nearest_support") and levels.get("nearest_resistance"):
                sr_thought = {
                    "category": ThoughtCategory.TECHNICAL.value,
                    "thought": f"Key levels: Support at {levels['nearest_support']}, Resistance at {levels['nearest_resistance']}.",
                    "support": levels["nearest_support"],
                    "resistance": levels["nearest_resistance"],
                    "confidence": levels["confidence"]
                }
                thoughts.append(sr_thought)
        
        # For Moderate thinking level and above
        if thinking_level.value >= ThinkingLevel.MODERATE.value:
            # Market cycle analysis
            cycle = self._analyze_market_cycle(market_data)
            if cycle:
                cycle_thought = {
                    "category": ThoughtCategory.FUNDAMENTAL.value,
                    "thought": f"{coin} appears to be in the {cycle['phase']} phase of its market cycle.",
                    "cycle_phase": cycle["phase"],
                    "confidence": cycle["confidence"]
                }
                thoughts.append(cycle_thought)
            
            # News impact
            news_impact = self._analyze_news_impact(coin)
            if news_impact:
                news_thought = {
                    "category": ThoughtCategory.NEWS.value,
                    "thought": f"Recent news sentiment is {news_impact['sentiment']} with {news_impact['importance']} importance.",
                    "news_sentiment": news_impact["sentiment"],
                    "importance": news_impact["importance"],
                    "confidence": news_impact["confidence"]
                }
                thoughts.append(news_thought)
        
        # For High thinking level and above
        if thinking_level.value >= ThinkingLevel.HIGH.value:
            # Correlation with BTC/ETH
            correlation = self._analyze_correlations(coin, market_data)
            if correlation:
                corr_thought = {
                    "category": ThoughtCategory.CORRELATION.value,
                    "thought": f"{coin} shows {correlation['strength']} correlation with {correlation['asset']} ({correlation['coefficient']:.2f}).",
                    "correlated_asset": correlation["asset"],
                    "coefficient": correlation["coefficient"],
                    "confidence": correlation["confidence"]
                }
                thoughts.append(corr_thought)
            
            # Pattern recognition
            pattern = self._identify_patterns(market_data)
            if pattern:
                pattern_thought = {
                    "category": ThoughtCategory.PATTERN.value,
                    "thought": f"Detected a potential {pattern['pattern']} pattern forming, which typically indicates {pattern['indication']}.",
                    "pattern_type": pattern["pattern"],
                    "indication": pattern["indication"],
                    "confidence": pattern["confidence"]
                }
                thoughts.append(pattern_thought)
        
        # For Maximum thinking level only
        if thinking_level.value >= ThinkingLevel.MAXIMUM.value:
            # Contrarian indicators
            contrarian = self._analyze_contrarian_indicators(coin, market_data, sentiment_data)
            if contrarian:
                contrarian_thought = {
                    "category": ThoughtCategory.CONTRARIAN.value,
                    "thought": f"Contrarian indicator: {contrarian['indicator']} suggests {contrarian['implication']}.",
                    "indicator": contrarian["indicator"],
                    "implication": contrarian["implication"],
                    "confidence": contrarian["confidence"]
                }
                thoughts.append(contrarian_thought)
            
            # Liquidity analysis
            liquidity = self._analyze_liquidity(coin, market_data)
            if liquidity:
                liquidity_thought = {
                    "category": ThoughtCategory.TECHNICAL.value,
                    "thought": f"Market liquidity is {liquidity['state']}, which {liquidity['implication']}.",
                    "liquidity_state": liquidity["state"],
                    "implication": liquidity["implication"],
                    "confidence": liquidity["confidence"]
                }
                thoughts.append(liquidity_thought)
            
            # Whale activity
            whale_activity = self._analyze_whale_activity(coin)
            if whale_activity:
                whale_thought = {
                    "category": ThoughtCategory.FUNDAMENTAL.value,
                    "thought": f"Whale activity: {whale_activity['description']}",
                    "activity_type": whale_activity["type"],
                    "significance": whale_activity["significance"],
                    "confidence": whale_activity["confidence"]
                }
                thoughts.append(whale_thought)
        
        # Add thoughts to thinking results
        thinking_results["thoughts"] = thoughts
    
    def _generate_scenarios(self, 
                           thinking_results: Dict[str, Any], 
                           coin: str, 
                           market_data: Dict[str, Any], 
                           sentiment_data: Dict[str, Any],
                           gpt_decision: Dict[str, Any],
                           thinking_level: ThinkingLevel):
        """
        Generate potential market scenarios.
        
        Args:
            thinking_results (Dict): Thinking results structure to populate
            coin (str): Coin symbol
            market_data (Dict): Market data for the coin
            sentiment_data (Dict): Sentiment data for the coin
            gpt_decision (Dict): Decision from GPT-4o
            thinking_level (ThinkingLevel): Thinking level
        """
        scenarios = []
        
        # Basic scenarios for all levels (Moderate and above)
        
        # Bullish scenario
        bullish_scenario = {
            "type": ScenarioType.BULLISH.value,
            "description": f"{coin} continues its upward momentum due to positive market sentiment and increasing adoption.",
            "probability": self._calculate_scenario_probability(coin, ScenarioType.BULLISH, market_data, sentiment_data),
            "impact": "Positive for long positions, negative for shorts",
            "signals_to_watch": ["Increasing volume on green candles", "Positive news catalysts", "Higher highs formation"]
        }
        scenarios.append(bullish_scenario)
        
        # Bearish scenario
        bearish_scenario = {
            "type": ScenarioType.BEARISH.value,
            "description": f"{coin} faces selling pressure due to market uncertainty and profit-taking.",
            "probability": self._calculate_scenario_probability(coin, ScenarioType.BEARISH, market_data, sentiment_data),
            "impact": "Negative for long positions, positive for shorts",
            "signals_to_watch": ["Increasing volume on red candles", "Breaking below support levels", "Lower highs formation"]
        }
        scenarios.append(bearish_scenario)
        
        # Sideways scenario
        sideways_scenario = {
            "type": ScenarioType.SIDEWAYS.value,
            "description": f"{coin} enters a consolidation phase with reduced volatility.",
            "probability": self._calculate_scenario_probability(coin, ScenarioType.SIDEWAYS, market_data, sentiment_data),
            "impact": "Suboptimal for directional positions, potential for range trading",
            "signals_to_watch": ["Decreasing volume", "Price constriction", "Failure to break support/resistance"]
        }
        scenarios.append(sideways_scenario)
        
        # Additional scenarios for High thinking level and above
        if thinking_level.value >= ThinkingLevel.HIGH.value:
            # Breakout scenario
            breakout_scenario = {
                "type": ScenarioType.BREAKOUT.value,
                "description": f"{coin} breaks above key resistance with increasing volume, triggering a significant move upward.",
                "probability": self._calculate_scenario_probability(coin, ScenarioType.BREAKOUT, market_data, sentiment_data),
                "impact": "Very positive for long positions, catastrophic for shorts",
                "signals_to_watch": ["Volume spike", "Resistance test", "Bullish candlestick patterns", "Funding rate changes"]
            }
            scenarios.append(breakout_scenario)
            
            # Breakdown scenario
            breakdown_scenario = {
                "type": ScenarioType.BREAKDOWN.value,
                "description": f"{coin} breaks below key support, accelerating selling pressure.",
                "probability": self._calculate_scenario_probability(coin, ScenarioType.BREAKDOWN, market_data, sentiment_data),
                "impact": "Very negative for long positions, very positive for shorts",
                "signals_to_watch": ["Volume spike on selling", "Support level test", "Bearish candlestick patterns"]
            }
            scenarios.append(breakdown_scenario)
        
        # Complex scenarios for Maximum thinking level only
        if thinking_level.value >= ThinkingLevel.MAXIMUM.value:
            # Manipulation scenario
            manipulation_scenario = {
                "type": ScenarioType.MANIPULATION.value,
                "description": f"{coin} experiences artificial price movements due to large holders' activity.",
                "probability": self._calculate_scenario_probability(coin, ScenarioType.MANIPULATION, market_data, sentiment_data),
                "impact": "Unpredictable; can lead to stop hunts and liquidation cascades",
                "signals_to_watch": ["Sudden large transactions", "Unusual order book patterns", "Divergence from sector trends"]
            }
            scenarios.append(manipulation_scenario)
            
            # Black swan scenario
            black_swan_scenario = {
                "type": ScenarioType.BLACK_SWAN.value,
                "description": f"Unexpected major news or event causes extreme volatility in {coin}.",
                "probability": self._calculate_scenario_probability(coin, ScenarioType.BLACK_SWAN, market_data, sentiment_data),
                "impact": "Extreme volatility; could be dramatically positive or negative",
                "signals_to_watch": ["Unusual silence from project team", "Regulatory announcements", "Major partnership rumors"]
            }
            scenarios.append(black_swan_scenario)
        
        # Sort scenarios by probability (highest first)
        scenarios.sort(key=lambda x: x["probability"], reverse=True)
        
        # Add scenarios to thinking results
        thinking_results["scenarios"] = scenarios
    
    def _assess_risk(self, 
                    thinking_results: Dict[str, Any], 
                    coin: str, 
                    market_data: Dict[str, Any], 
                    sentiment_data: Dict[str, Any],
                    gpt_decision: Dict[str, Any],
                    thinking_level: ThinkingLevel):
        """
        Assess various risk factors for the trading decision.
        
        Args:
            thinking_results (Dict): Thinking results structure to populate
            coin (str): Coin symbol
            market_data (Dict): Market data for the coin
            sentiment_data (Dict): Sentiment data for the coin
            gpt_decision (Dict): Decision from GPT-4o
            thinking_level (ThinkingLevel): Thinking level
        """
        risk_assessment = {}
        
        # Extract decision parameters
        entry_price = gpt_decision.get("entry_price", 0) or market_data.get("current_price", 0)
        stop_loss = gpt_decision.get("stop_loss", 0)
        take_profit = gpt_decision.get("take_profit", 0) or gpt_decision.get("target_price", 0)
        position_size = gpt_decision.get("size", 0)
        leverage = gpt_decision.get("leverage", 1)
        is_long = gpt_decision.get("direction", "").lower() in ["long", "buy"]
        
        # Calculate basic risk metrics
        
        # Volatility risk (0-1)
        volatility = self._calculate_volatility(market_data)
        volatility_risk = min(1.0, volatility * 5)  # Normalize to 0-1
        risk_assessment["volatility_risk"] = {
            "score": volatility_risk,
            "value": volatility,
            "assessment": self._get_risk_assessment_text(volatility_risk),
            "explanation": f"Recent price volatility is {volatility:.2%} which indicates {self._get_risk_assessment_text(volatility_risk)} risk."
        }
        
        # Stop loss risk (0-1)
        stop_loss_risk = 1.0
        if stop_loss > 0 and entry_price > 0:
            if is_long:
                stop_distance = (entry_price - stop_loss) / entry_price
            else:
                stop_distance = (stop_loss - entry_price) / entry_price
                
            # Normalize: closer to 0 is riskier (tight stop)
            # 1% stop distance → high risk (0.8)
            # 10% stop distance → low risk (0.2)
            stop_loss_risk = max(0.0, min(1.0, 1.0 - (stop_distance * 10)))
        
        risk_assessment["stop_loss_risk"] = {
            "score": stop_loss_risk,
            "assessment": self._get_risk_assessment_text(stop_loss_risk),
            "explanation": f"Stop loss placement indicates {self._get_risk_assessment_text(stop_loss_risk)} risk."
        }
        
        # Risk-reward ratio
        risk_reward = 1.0
        if stop_loss > 0 and take_profit > 0 and entry_price > 0:
            if is_long:
                risk = entry_price - stop_loss
                reward = take_profit - entry_price
            else:
                risk = stop_loss - entry_price
                reward = entry_price - take_profit
                
            if risk > 0:
                risk_reward = reward / risk
        
        # Risk based on risk-reward ratio (lower ratio = higher risk)
        rr_risk = max(0.0, min(1.0, 1.0 - (risk_reward / 5)))
        
        risk_assessment["risk_reward_risk"] = {
            "score": rr_risk,
            "ratio": risk_reward,
            "assessment": self._get_risk_assessment_text(rr_risk),
            "explanation": f"Risk-reward ratio of {risk_reward:.2f} indicates {self._get_risk_assessment_text(rr_risk)} risk."
        }
        
        # Position size risk
        position_risk = min(1.0, position_size * 2)  # 0.5 position size -> 1.0 risk
        risk_assessment["position_size_risk"] = {
            "score": position_risk,
            "size": position_size,
            "assessment": self._get_risk_assessment_text(position_risk),
            "explanation": f"Position size of {position_size:.2%} of capital indicates {self._get_risk_assessment_text(position_risk)} risk."
        }
        
        # Leverage risk
        leverage_risk = min(1.0, (leverage - 1) / 9)  # 1x -> 0.0, 10x -> 1.0
        risk_assessment["leverage_risk"] = {
            "score": leverage_risk,
            "leverage": leverage,
            "assessment": self._get_risk_assessment_text(leverage_risk),
            "explanation": f"{leverage}x leverage indicates {self._get_risk_assessment_text(leverage_risk)} risk."
        }
        
        # Additional risk factors for higher thinking levels
        
        # Market sentiment risk (level: Low+)
        if thinking_level.value >= ThinkingLevel.LOW.value:
            sentiment_score = self._extract_sentiment_score(sentiment_data)
            if sentiment_score is not None:
                # For long positions: low sentiment is high risk
                # For short positions: high sentiment is high risk
                if is_long:
                    sentiment_risk = 1.0 - sentiment_score
                else:
                    sentiment_risk = sentiment_score
                    
                risk_assessment["sentiment_risk"] = {
                    "score": sentiment_risk,
                    "sentiment": sentiment_score,
                    "assessment": self._get_risk_assessment_text(sentiment_risk),
                    "explanation": f"Current sentiment ({sentiment_score:.2f}) indicates {self._get_risk_assessment_text(sentiment_risk)} risk for {gpt_decision.get('direction', 'unknown')} position."
                }
        
        # Liquidity risk (level: High+)
        if thinking_level.value >= ThinkingLevel.HIGH.value:
            liquidity_analysis = self._analyze_liquidity(coin, market_data)
            if liquidity_analysis:
                liquidity_mapping = {
                    "very high": 0.1,
                    "high": 0.2,
                    "moderate": 0.5,
                    "low": 0.8,
                    "very low": 1.0
                }
                liquidity_risk = liquidity_mapping.get(liquidity_analysis.get("state", "moderate"), 0.5)
                
                risk_assessment["liquidity_risk"] = {
                    "score": liquidity_risk,
                    "liquidity": liquidity_analysis.get("state"),
                    "assessment": self._get_risk_assessment_text(liquidity_risk),
                    "explanation": f"{liquidity_analysis.get('state').capitalize()} market liquidity indicates {self._get_risk_assessment_text(liquidity_risk)} risk."
                }
        
        # Timing risk (level: Moderate+)
        if thinking_level.value >= ThinkingLevel.MODERATE.value:
            price_trend = self._analyze_price_trend(market_data)
            if price_trend:
                # For long positions: counter-trend is higher risk
                # For short positions: counter-trend is higher risk
                trend_direction = price_trend.get("trend", "neutral")
                
                if is_long and trend_direction == "downward":
                    timing_risk = 0.8  # High risk (counter-trend)
                elif not is_long and trend_direction == "upward":
                    timing_risk = 0.8  # High risk (counter-trend)
                elif trend_direction == "neutral":
                    timing_risk = 0.5  # Moderate risk
                else:
                    timing_risk = 0.2  # Low risk (with trend)
                
                risk_assessment["timing_risk"] = {
                    "score": timing_risk,
                    "trend": trend_direction,
                    "assessment": self._get_risk_assessment_text(timing_risk),
                    "explanation": f"Trading {gpt_decision.get('direction', 'unknown')} in a {trend_direction} trend indicates {self._get_risk_assessment_text(timing_risk)} risk."
                }
        
        # News event risk (level: Maximum)
        if thinking_level.value >= ThinkingLevel.MAXIMUM.value:
            news_impact = self._analyze_news_impact(coin)
            if news_impact:
                news_importance = news_impact.get("importance", "low")
                news_sentiment = news_impact.get("sentiment", "neutral")
                
                # Map importance to risk values
                importance_mapping = {
                    "very high": 0.9,
                    "high": 0.7,
                    "moderate": 0.5,
                    "low": 0.3,
                    "very low": 0.1
                }
                
                # Baseline risk based on importance
                news_event_risk = importance_mapping.get(news_importance, 0.5)
                
                # Adjust based on position direction and news sentiment
                if is_long and news_sentiment in ["bearish", "very bearish"]:
                    news_event_risk += 0.2  # Higher risk for longs with bearish news
                elif not is_long and news_sentiment in ["bullish", "very bullish"]:
                    news_event_risk += 0.2  # Higher risk for shorts with bullish news
                
                news_event_risk = min(1.0, news_event_risk)  # Cap at 1.0
                
                risk_assessment["news_event_risk"] = {
                    "score": news_event_risk,
                    "importance": news_importance,
                    "sentiment": news_sentiment,
                    "assessment": self._get_risk_assessment_text(news_event_risk),
                    "explanation": f"{news_importance.capitalize()} importance {news_sentiment} news indicates {self._get_risk_assessment_text(news_event_risk)} risk."
                }
        
        # Calculate total risk score (weighted average)
        risk_weights = {
            "volatility_risk": 0.15,
            "stop_loss_risk": 0.20,
            "risk_reward_risk": 0.20,
            "position_size_risk": 0.15,
            "leverage_risk": 0.15,
            "sentiment_risk": 0.05,
            "liquidity_risk": 0.03,
            "timing_risk": 0.05,
            "news_event_risk": 0.02
        }
        
        total_score = 0
        total_weight = 0
        
        for risk_factor, weight in risk_weights.items():
            if risk_factor in risk_assessment:
                score = risk_assessment[risk_factor]["score"]
                total_score += score * weight
                total_weight += weight
        
        # Calculate weighted average
        if total_weight > 0:
            total_risk_score = total_score / total_weight
        else:
            total_risk_score = 0.5  # Default moderate risk
        
        # Add total risk assessment
        risk_assessment["total_risk_score"] = total_risk_score
        risk_assessment["total_risk_assessment"] = self._get_risk_assessment_text(total_risk_score)
        risk_assessment["total_risk_explanation"] = f"Overall risk assessment: {self._get_risk_assessment_text(total_risk_score)} risk."
        
        # Add risk assessment to thinking results
        thinking_results["risk_assessment"] = risk_assessment
    
    def _generate_deepseek_decision(self, 
                                   thinking_results: Dict[str, Any], 
                                   gpt_decision: Dict[str, Any], 
                                   coin: str, 
                                   market_data: Dict[str, Any], 
                                   sentiment_data: Dict[str, Any],
                                   thinking_level: ThinkingLevel):
        """
        Generate DeepSeek's own decision based on analysis.
        
        Args:
            thinking_results (Dict): Thinking results structure to populate
            gpt_decision (Dict): Decision from GPT-4o
            coin (str): Coin symbol
            market_data (Dict): Market data for the coin
            sentiment_data (Dict): Sentiment data for the coin
            thinking_level (ThinkingLevel): Thinking level
        """
        if not gpt_decision:
            # If no GPT decision provided, just create a placeholder
            thinking_results["deepseek_decision"] = {
                "action": "unknown",
                "direction": "unknown",
                "confidence": 0.0,
                "reasoning": "No GPT decision provided for analysis"
            }
            return
        
        # Extract risk assessment
        risk_assessment = thinking_results.get("risk_assessment", {})
        total_risk_score = risk_assessment.get("total_risk_score", 0.5)
        
        # Extract GPT decision details
        gpt_action = gpt_decision.get("action", "").lower()
        gpt_direction = gpt_decision.get("direction", "").lower()
        gpt_symbol = gpt_decision.get("symbol", coin).upper()
        gpt_size = gpt_decision.get("size", 0.0)
        gpt_stop_loss = gpt_decision.get("stop_loss", 0.0)
        gpt_take_profit = gpt_decision.get("take_profit", 0.0) or gpt_decision.get("target_price", 0.0)
        gpt_leverage = gpt_decision.get("leverage", 1.0)
        gpt_reasoning = gpt_decision.get("reasoning", "")
        
        # Extract key thoughts
        thoughts = thinking_results.get("thoughts", [])
        sentiment_thoughts = [t for t in thoughts if t.get("category") == ThoughtCategory.SENTIMENT.value]
        technical_thoughts = [t for t in thoughts if t.get("category") == ThoughtCategory.TECHNICAL.value]
        
        # Extract scenarios
        scenarios = thinking_results.get("scenarios", [])
        top_scenario = scenarios[0] if scenarios else None
        
        # Initialize DeepSeek's decision (default to GPT's decision)
        deepseek_decision = {
            "action": gpt_action,
            "symbol": gpt_symbol,
            "direction": gpt_direction,
            "size": gpt_size,
            "stop_loss": gpt_stop_loss,
            "take_profit": gpt_take_profit,
            "leverage": gpt_leverage,
            "reasoning": "",
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Determine if DeepSeek should agree with GPT or modify the decision
        modify_decision = False
        agreement_level = "full"  # full, partial, none
        
        # Decision override logic based on analysis
        
        # 1. Check if risk is too high
        if total_risk_score > 0.8:  # Very high risk
            modify_decision = True
            agreement_level = "none"
            
            # Change action to avoid or reduce position
            if gpt_action in ["buy", "long"]:
                deepseek_decision["action"] = "avoid"
                deepseek_decision["direction"] = "neutral"
                deepseek_decision["size"] = 0.0
            elif gpt_size > 0.2:  # If position size is large
                deepseek_decision["size"] = gpt_size * 0.5  # Cut size in half
                agreement_level = "partial"
        
        # 2. Check sentiment vs. direction alignment
        if sentiment_thoughts:
            sentiment_score = sentiment_thoughts[0].get("sentiment_score", 0.5)
            if gpt_direction == "long" and sentiment_score < 0.3:  # Very bearish
                modify_decision = True
                if agreement_level == "full":
                    agreement_level = "partial"
                
                # Either reduce size or consider opposite direction
                if sentiment_score < 0.2:  # Extremely bearish
                    deepseek_decision["action"] = "sell"
                    deepseek_decision["direction"] = "short"
                    agreement_level = "none"
                else:
                    deepseek_decision["size"] = gpt_size * 0.7  # Reduce size
            
            elif gpt_direction == "short" and sentiment_score > 0.7:  # Very bullish
                modify_decision = True
                if agreement_level == "full":
                    agreement_level = "partial"
                
                # Either reduce size or consider opposite direction
                if sentiment_score > 0.8:  # Extremely bullish
                    deepseek_decision["action"] = "buy"
                    deepseek_decision["direction"] = "long"
                    agreement_level = "none"
                else:
                    deepseek_decision["size"] = gpt_size * 0.7  # Reduce size
        
        # 3. Check technical trend vs. direction alignment
        if technical_thoughts:
            for thought in technical_thoughts:
                if "trend" in thought:
                    trend = thought.get("trend", "neutral")
                    if gpt_direction == "long" and trend == "downward":
                        modify_decision = True
                        if agreement_level == "full":
                            agreement_level = "partial"
                        
                        # Adjust to "sniper" for counter-trend trade or reduce size
                        deepseek_decision["direction"] = "sniper"
                        deepseek_decision["size"] = gpt_size * 0.6  # Reduce size
                    
                    elif gpt_direction == "short" and trend == "upward":
                        modify_decision = True
                        if agreement_level == "full":
                            agreement_level = "partial"
                        
                        # Adjust to "sniper" for counter-trend trade or reduce size
                        deepseek_decision["direction"] = "sniper"
                        deepseek_decision["size"] = gpt_size * 0.6  # Reduce size
        
        # 4. Adjust for most likely scenario
        if top_scenario and top_scenario.get("probability", 0) > 0.6:
            scenario_type = top_scenario.get("type", "")
            
            if scenario_type == ScenarioType.BULLISH.value and gpt_direction == "short":
                modify_decision = True
                if agreement_level in ["full", "partial"]:
                    agreement_level = "none"
                deepseek_decision["action"] = "buy"
                deepseek_decision["direction"] = "long"
            
            elif scenario_type == ScenarioType.BEARISH.value and gpt_direction == "long":
                modify_decision = True
                if agreement_level in ["full", "partial"]:
                    agreement_level = "none"
                deepseek_decision["action"] = "sell"
                deepseek_decision["direction"] = "short"
            
            elif scenario_type == ScenarioType.SIDEWAYS.value:
                # For sideways markets, prefer range trading strategies
                if gpt_direction in ["long", "short"]:
                    modify_decision = True
                    if agreement_level == "full":
                        agreement_level = "partial"
                    
                    deepseek_decision["direction"] = "sniper"  # Change to sniper
                    deepseek_decision["size"] = gpt_size * 0.8  # Slightly reduce size
                    
                    # Tighten take profit
                    if deepseek_decision["take_profit"] > 0:
                        current_price = market_data.get("current_price", 0)
                        if current_price > 0:
                            # Reduce distance to take profit by 40%
                            if gpt_direction == "long":
                                new_tp_distance = (deepseek_decision["take_profit"] - current_price) * 0.6
                                deepseek_decision["take_profit"] = current_price + new_tp_distance
                            else:  # short
                                new_tp_distance = (current_price - deepseek_decision["take_profit"]) * 0.6
                                deepseek_decision["take_profit"] = current_price - new_tp_distance
        
        # 5. Final risk-based adjustment to size and leverage
        if thinking_level.value >= ThinkingLevel.HIGH.value:
            # Adjust position size based on risk
            if total_risk_score > 0.6:  # High risk
                deepseek_decision["size"] = min(deepseek_decision["size"], 0.3)  # Cap size at 30%
                modify_decision = True
                if agreement_level == "full":
                    agreement_level = "partial"
            
            # Adjust leverage based on risk
            if total_risk_score > 0.5 and deepseek_decision["leverage"] > 2:
                deepseek_decision["leverage"] = max(1, deepseek_decision["leverage"] * 0.7)  # Reduce leverage
                modify_decision = True
                if agreement_level == "full":
                    agreement_level = "partial"
        
        # Generate reasoning
        reasoning_parts = []
        
        if modify_decision:
            if agreement_level == "none":
                reasoning_parts.append(f"DeepSeek disagrees with GPT-4o's {gpt_direction} recommendation for {coin}.")
            elif agreement_level == "partial":
                reasoning_parts.append(f"DeepSeek partially agrees with GPT-4o's {gpt_direction} direction but recommends adjustments.")
        else:
            reasoning_parts.append(f"DeepSeek agrees with GPT-4o's {gpt_direction} recommendation for {coin}.")
        
        # Add key thoughts to reasoning
        for thought in thoughts[:3]:  # Include top 3 thoughts
            reasoning_parts.append(thought.get("thought", ""))
        
        # Add risk assessment to reasoning
        reasoning_parts.append(risk_assessment.get("total_risk_explanation", ""))
        
        # Add top scenario to reasoning
        if top_scenario:
            reasoning_parts.append(f"Most likely scenario: {top_scenario.get('description', '')}")
        
        # Add specific modifications to reasoning
        if deepseek_decision["action"] != gpt_action or deepseek_decision["direction"] != gpt_direction:
            reasoning_parts.append(f"Changed strategy from '{gpt_action} {gpt_direction}' to '{deepseek_decision['action']} {deepseek_decision['direction']}'.")
        
        if deepseek_decision["size"] != gpt_size:
            reasoning_parts.append(f"Adjusted position size from {gpt_size:.2%} to {deepseek_decision['size']:.2%}.")
        
        if deepseek_decision["leverage"] != gpt_leverage:
            reasoning_parts.append(f"Adjusted leverage from {gpt_leverage}x to {deepseek_decision['leverage']}x.")
        
        if deepseek_decision["stop_loss"] != gpt_stop_loss and gpt_stop_loss > 0:
            reasoning_parts.append(f"Modified stop loss from {gpt_stop_loss} to {deepseek_decision['stop_loss']}.")
        
        if deepseek_decision["take_profit"] != gpt_take_profit and gpt_take_profit > 0:
            reasoning_parts.append(f"Modified take profit from {gpt_take_profit} to {deepseek_decision['take_profit']}.")
        
        # Add confidence based on agreement and risk
        if agreement_level == "full":
            confidence = 0.9 - (total_risk_score * 0.2)  # High confidence, slightly reduced by risk
        elif agreement_level == "partial":
            confidence = 0.7 - (total_risk_score * 0.2)  # Moderate confidence
        else:  # none
            confidence = 0.6 - (total_risk_score * 0.1)  # Lower confidence when totally disagreeing
        
        deepseek_decision["confidence"] = max(0.4, min(0.95, confidence))  # Clamp between 0.4 and 0.95
        deepseek_decision["reasoning"] = " ".join(reasoning_parts)
        deepseek_decision["agreement_level"] = agreement_level
        
        # Add to thinking results
        thinking_results["deepseek_decision"] = deepseek_decision
    
    def _calculate_confidence_score(self, 
                                   thinking_results: Dict[str, Any], 
                                   gpt_decision: Dict[str, Any], 
                                   coin: str, 
                                   thinking_level: ThinkingLevel) -> float:
        """
        Calculate the overall confidence score for DeepSeek's analysis.
        
        Args:
            thinking_results (Dict): Thinking results structure
            gpt_decision (Dict): Decision from GPT-4o
            coin (str): Coin symbol
            thinking_level (ThinkingLevel): Thinking level
            
        Returns:
            float: Confidence score (0.0-1.0)
        """
        # Start with base confidence
        base_confidence = 0.7
        
        # Extract relevant parts
        thoughts = thinking_results.get("thoughts", [])
        risk_assessment = thinking_results.get("risk_assessment", {})
        deepseek_decision = thinking_results.get("deepseek_decision", {})
        scenarios = thinking_results.get("scenarios", [])
        
        # Adjust based on thought confidences
        thought_confidences = [thought.get("confidence", 0.5) for thought in thoughts]
        if thought_confidences:
            avg_thought_confidence = sum(thought_confidences) / len(thought_confidences)
            confidence_adjustment = (avg_thought_confidence - 0.5) * 0.5  # Scale adjustment
            base_confidence += confidence_adjustment
        
        # Adjust based on risk assessment
        total_risk_score = risk_assessment.get("total_risk_score", 0.5)
        risk_adjustment = (0.5 - total_risk_score) * 0.4  # High risk reduces confidence
        base_confidence += risk_adjustment
        
        # Adjust based on agreement with GPT-4o
        agreement_level = deepseek_decision.get("agreement_level", "full")
        if agreement_level == "full":
            agreement_adjustment = 0.1
        elif agreement_level == "partial":
            agreement_adjustment = 0.0
        else:  # none
            agreement_adjustment = -0.1
        base_confidence += agreement_adjustment
        
        # Adjust based on scenario clarity
        if scenarios:
            top_scenario_probability = scenarios[0].get("probability", 0.5)
            scenario_adjustment = (top_scenario_probability - 0.5) * 0.4  # Clear scenarios increase confidence
            base_confidence += scenario_adjustment
        
        # Adjust based on thinking level (deeper thinking = slightly higher confidence)
        level_adjustment = (thinking_level.value - 3) * 0.05  # Level 3 is neutral
        base_confidence += level_adjustment
        
        # Clamp final confidence
        final_confidence = max(0.2, min(0.95, base_confidence))
        
        return final_confidence
    
    def _analyze_performance(self, 
                            thinking_results: Dict[str, Any], 
                            coin: str, 
                            strategy_results: Dict[str, Any], 
                            thinking_level: ThinkingLevel):
        """
        Analyze the performance of a strategy execution.
        
        Args:
            thinking_results (Dict): Thinking results structure to populate
            coin (str): Coin symbol
            strategy_results (Dict): Results of the strategy execution
            thinking_level (ThinkingLevel): Thinking level
        """
        performance_analysis = {}
        
        # Extract key metrics
        total_profit_loss = strategy_results.get("total_profit_loss", 0.0)
        win_rate = strategy_results.get("win_rate", 0.0)
        win_count = strategy_results.get("win_count", 0)
        loss_count = strategy_results.get("loss_count", 0)
        trades = strategy_results.get("trades", [])
        
        # Profitability analysis
        if total_profit_loss > 0:
            profitability_status = "profitable"
        elif total_profit_loss < 0:
            profitability_status = "unprofitable"
        else:
            profitability_status = "breakeven"
            
        performance_analysis["profitability"] = {
            "status": profitability_status,
            "total_pnl": total_profit_loss,
            "average_pnl": total_profit_loss / len(trades) if trades else 0,
            "assessment": f"The strategy was {profitability_status} with a total P&L of {total_profit_loss:.2f}."
        }
        
        # Win rate analysis
        if win_rate >= 0.7:
            win_rate_assessment = "excellent"
        elif win_rate >= 0.5:
            win_rate_assessment = "good"
        elif win_rate >= 0.3:
            win_rate_assessment = "fair"
        else:
            win_rate_assessment = "poor"
            
        performance_analysis["win_rate"] = {
            "rate": win_rate,
            "wins": win_count,
            "losses": loss_count,
            "assessment": win_rate_assessment,
            "explanation": f"Win rate of {win_rate:.2%} is {win_rate_assessment} ({win_count} wins, {loss_count} losses)."
        }
        
        # Trade distribution analysis
        if trades:
            # Analyze trade outcomes
            outcomes = {}
            for trade in trades:
                result = trade.get("result", "UNKNOWN")
                if result in outcomes:
                    outcomes[result] += 1
                else:
                    outcomes[result] = 1
                    
            performance_analysis["outcome_distribution"] = {
                "outcomes": outcomes,
                "explanation": f"Trade outcomes: {', '.join([f'{outcome}: {count}' for outcome, count in outcomes.items()])}"
            }
            
            # Analyze profitability by trade size
            if thinking_level.value >= ThinkingLevel.MODERATE.value:
                trades_level.value >= ThinkingLevel.MODERATE.value
                    trades_by_size = {}
                    for trade in trades:
                        size = trade.get("position_size", 0)
                        size_category = f"{int(size * 100)}%"
                        pnl = trade.get("pnl_amount", 0)
                        
                        if size_category not in trades_by_size:
                            trades_by_size[size_category] = {
                                "count": 0,
                                "total_pnl": 0,
                                "wins": 0,
                                "losses": 0
                            }
                        
                        trades_by_size[size_category]["count"] += 1
                        trades_by_size[size_category]["total_pnl"] += pnl
                        
                        if pnl > 0:
                            trades_by_size[size_category]["wins"] += 1
                        else:
                            trades_by_size[size_category]["losses"] += 1
                    
                    # Calculate average PnL by size
                    for size, data in trades_by_size.items():
                        data["avg_pnl"] = data["total_pnl"] / data["count"] if data["count"] > 0 else 0
                        data["win_rate"] = data["wins"] / data["count"] if data["count"] > 0 else 0
                    
                    performance_analysis["size_analysis"] = {
                        "trades_by_size": trades_by_size,
                        "explanation": "Analysis of trade performance by position size."
                    }
            
            # Time-based analysis (for higher thinking levels)
            if thinking_level.value >= ThinkingLevel.HIGH.value and len(trades) >= 3:
                # Group trades by time periods
                try:
                    # Convert time strings to datetime objects
                    for trade in trades:
                        trade["entry_time_obj"] = datetime.datetime.fromisoformat(trade.get("entry_time", ""))
                        trade["exit_time_obj"] = datetime.datetime.fromisoformat(trade.get("exit_time", ""))
                    
                    # Sort trades by entry time
                    sorted_trades = sorted(trades, key=lambda x: x.get("entry_time_obj", datetime.datetime.min))
                    
                    # Split into early, middle, late periods
                    num_periods = 3
                    trades_per_period = len(sorted_trades) // num_periods
                    
                    periods = []
                    for i in range(num_periods):
                        start_idx = i * trades_per_period
                        end_idx = start_idx + trades_per_period if i < num_periods - 1 else len(sorted_trades)
                        
                        period_trades = sorted_trades[start_idx:end_idx]
                        period_pnl = sum(trade.get("pnl_amount", 0) for trade in period_trades)
                        period_win_count = sum(1 for trade in period_trades if trade.get("pnl_amount", 0) > 0)
                        
                        periods.append({
                            "period": i + 1,
                            "num_trades": len(period_trades),
                            "total_pnl": period_pnl,
                            "win_count": period_win_count,
                            "win_rate": period_win_count / len(period_trades) if period_trades else 0
                        })
                    
                    # Assess trend
                    early_period = periods[0]
                    late_period = periods[-1]
                    
                    if early_period["win_rate"] < late_period["win_rate"]:
                        trend = "improving"
                    elif early_period["win_rate"] > late_period["win_rate"]:
                        trend = "deteriorating"
                    else:
                        trend = "stable"
                    
                    performance_analysis["time_analysis"] = {
                        "periods": periods,
                        "trend": trend,
                        "explanation": f"Performance trend over time: {trend}"
                    }
                except Exception as e:
                    logger.error(f"Error in time-based analysis: {str(e)}")
        
        # Add overall assessment
        if total_profit_loss > 0 and win_rate >= 0.5:
            overall_assessment = "positive"
        elif total_profit_loss > 0 or win_rate >= 0.5:
            overall_assessment = "mixed"
        else:
            overall_assessment = "negative"
            
        performance_analysis["overall_assessment"] = {
            "rating": overall_assessment,
            "explanation": f"Overall performance assessment: {overall_assessment}"
        }
        
        # Add performance analysis to thinking results
        thinking_results["performance_analysis"] = performance_analysis
    
    def _generate_post_trade_thoughts(self, 
                                     thinking_results: Dict[str, Any], 
                                     coin: str, 
                                     strategy_results: Dict[str, Any], 
                                     market_data: Dict[str, Any], 
                                     sentiment_data: Dict[str, Any],
                                     thinking_level: ThinkingLevel):
        """
        Generate thoughts after trade execution.
        
        Args:
            thinking_results (Dict): Thinking results structure to populate
            coin (str): Coin symbol
            strategy_results (Dict): Results of the strategy execution
            market_data (Dict): Market data for the coin
            sentiment_data (Dict): Sentiment data for the coin
            thinking_level (ThinkingLevel): Thinking level
        """
        thoughts = []
        
        # Extract key metrics
        total_profit_loss = strategy_results.get("total_profit_loss", 0.0)
        win_rate = strategy_results.get("win_rate", 0.0)
        strategy_type = strategy_results.get("strategy_type", "unknown")
        
        # Basic post-trade thoughts (for all thinking levels)
        
        # Profitability thought
        profitability_thought = {
            "category": ThoughtCategory.RECOMMENDATION.value,
            "thought": f"The {strategy_type} strategy for {coin} resulted in {total_profit_loss:.2f} total P&L with {win_rate:.2%} win rate.",
            "confidence": 0.95  # High confidence as this is factual
        }
        thoughts.append(profitability_thought)
        
        # Market alignment thought
        price_trend = self._analyze_price_trend(market_data)
        if price_trend:
            trend = price_trend.get("trend", "neutral")
            sentiment_score = self._extract_sentiment_score(sentiment_data)
            
            if strategy_type in ["long", "swing"] and trend == "upward" and sentiment_score and sentiment_score > 0.6:
                alignment = "well aligned"
                alignment_confidence = 0.8
            elif strategy_type in ["short"] and trend == "downward" and sentiment_score and sentiment_score < 0.4:
                alignment = "well aligned"
                alignment_confidence = 0.8
            elif strategy_type in ["sniper", "scalp"]:
                alignment = "suitable for short-term volatility"
                alignment_confidence = 0.7
            else:
                alignment = "not optimally aligned"
                alignment_confidence = 0.6
                
            alignment_thought = {
                "category": ThoughtCategory.TECHNICAL.value,
                "thought": f"The {strategy_type} strategy was {alignment} with the {trend} market trend and sentiment.",
                "confidence": alignment_confidence
            }
            thoughts.append(alignment_thought)
        
        # For Low thinking level and above
        if thinking_level.value >= ThinkingLevel.LOW.value:
            # Risk-reward analysis
            if "trades" in strategy_results and strategy_results["trades"]:
                trades = strategy_results["trades"]
                
                # Calculate average win and loss
                wins = [t.get("pnl_amount", 0) for t in trades if t.get("pnl_amount", 0) > 0]
                losses = [t.get("pnl_amount", 0) for t in trades if t.get("pnl_amount", 0) < 0]
                
                avg_win = sum(wins) / len(wins) if wins else 0
                avg_loss = sum(losses) / len(losses) if losses else 0
                
                if avg_loss != 0:
                    risk_reward_ratio = abs(avg_win / avg_loss) if avg_loss else 0
                    
                    if risk_reward_ratio >= 2.0:
                        rr_assessment = "excellent"
                        rr_confidence = 0.85
                    elif risk_reward_ratio >= 1.5:
                        rr_assessment = "good"
                        rr_confidence = 0.75
                    elif risk_reward_ratio >= 1.0:
                        rr_assessment = "acceptable"
                        rr_confidence = 0.65
                    else:
                        rr_assessment = "poor"
                        rr_confidence = 0.7
                    
                    rr_thought = {
                        "category": ThoughtCategory.RISK.value,
                        "thought": f"Risk-reward ratio of {risk_reward_ratio:.2f} was {rr_assessment} (avg win: {avg_win:.2f}, avg loss: {avg_loss:.2f}).",
                        "confidence": rr_confidence
                    }
                    thoughts.append(rr_thought)
        
        # For Moderate thinking level and above
        if thinking_level.value >= ThinkingLevel.MODERATE.value:
            # Market condition suitability
            volatility = self._calculate_volatility(market_data)
            volume_analysis = self._analyze_volume(market_data)
            
            suitability = ""
            suit_confidence = 0.6
            
            if strategy_type in ["long", "swing"]:
                if trend == "upward" and volatility < 0.03 and volume_analysis and volume_analysis.get("state") == "increasing":
                    suitability = "highly suitable"
                    suit_confidence = 0.85
                elif trend == "upward" or (volatility < 0.05 and volume_analysis and volume_analysis.get("state") != "decreasing"):
                    suitability = "suitable"
                    suit_confidence = 0.7
                else:
                    suitability = "somewhat suitable"
                    suit_confidence = 0.6
            elif strategy_type in ["short"]:
                if trend == "downward" and volatility < 0.03 and volume_analysis and volume_analysis.get("state") == "increasing":
                    suitability = "highly suitable"
                    suit_confidence = 0.85
                elif trend == "downward" or (volatility < 0.05 and volume_analysis and volume_analysis.get("state") != "decreasing"):
                    suitability = "suitable"
                    suit_confidence = 0.7
                else:
                    suitability = "somewhat suitable"
                    suit_confidence = 0.6
            elif strategy_type in ["sniper", "scalp"]:
                if volatility > 0.05:
                    suitability = "highly suitable"
                    suit_confidence = 0.85
                elif volatility > 0.03:
                    suitability = "suitable"
                    suit_confidence = 0.7
                else:
                    suitability = "less suitable"
                    suit_confidence = 0.65
            
            if suitability:
                suitability_thought = {
                    "category": ThoughtCategory.RECOMMENDATION.value,
                    "thought": f"The market conditions were {suitability} for the {strategy_type} strategy.",
                    "confidence": suit_confidence
                }
                thoughts.append(suitability_thought)
        
        # For High thinking level and above
        if thinking_level.value >= ThinkingLevel.HIGH.value:
            # Predict future suitability
            news_impact = self._analyze_news_impact(coin)
            sentiment_momentum = self._analyze_sentiment_momentum(coin, sentiment_data)
            
            future_suitability = ""
            future_confidence = 0.5
            
            if strategy_type in ["long", "swing"]:
                if (trend == "upward" and 
                    news_impact and news_impact.get("sentiment") in ["bullish", "very bullish"] and 
                    sentiment_momentum and sentiment_momentum.get("direction") == "improving"):
                    future_suitability = "increasingly suitable"
                    future_confidence = 0.7
                elif (trend == "downward" or 
                      (news_impact and news_impact.get("sentiment") in ["bearish", "very bearish"]) or 
                      (sentiment_momentum and sentiment_momentum.get("direction") == "deteriorating")):
                    future_suitability = "decreasing in suitability"
                    future_confidence = 0.65
                else:
                    future_suitability = "likely to maintain similar suitability"
                    future_confidence = 0.55
            elif strategy_type in ["short"]:
                if (trend == "downward" and 
                    news_impact and news_impact.get("sentiment") in ["bearish", "very bearish"] and 
                    sentiment_momentum and sentiment_momentum.get("direction") == "deteriorating"):
                    future_suitability = "increasingly suitable"
                    future_confidence = 0.7
                elif (trend == "upward" or 
                      (news_impact and news_impact.get("sentiment") in ["bullish", "very bullish"]) or 
                      (sentiment_momentum and sentiment_momentum.get("direction") == "improving")):
                    future_suitability = "decreasing in suitability"
                    future_confidence = 0.65
                else:
                    future_suitability = "likely to maintain similar suitability"
                    future_confidence = 0.55
            
            if future_suitability:
                future_thought = {
                    "category": ThoughtCategory.RECOMMENDATION.value,
                    "thought": f"This strategy is {future_suitability} for {coin} in the near term.",
                    "confidence": future_confidence
                }
                thoughts.append(future_thought)
        
        # Add thoughts to thinking results
        thinking_results["thoughts"] = thoughts
    
    def _generate_learning_insights(self, 
                                   thinking_results: Dict[str, Any], 
                                   coin: str, 
                                   strategy_results: Dict[str, Any],
                                   thinking_level: ThinkingLevel):
        """
        Generate learning insights based on strategy performance.
        
        Args:
            thinking_results (Dict): Thinking results structure to populate
            coin (str): Coin symbol
            strategy_results (Dict): Results of the strategy execution
            thinking_level (ThinkingLevel): Thinking level
        """
        insights = []
        
        # Extract key data
        performance = thinking_results.get("performance_analysis", {})
        win_rate = strategy_results.get("win_rate", 0.0)
        total_profit_loss = strategy_results.get("total_profit_loss", 0.0)
        strategy_type = strategy_results.get("strategy_type", "unknown")
        trades = strategy_results.get("trades", [])
        
        # Basic insights for all thinking levels (Moderate+)
        
        # Overall effectiveness insight
        if total_profit_loss > 0 and win_rate >= 0.5:
            effectiveness = "effective"
        elif total_profit_loss > 0:
            effectiveness = "profitable despite low win rate"
        elif win_rate >= 0.5:
            effectiveness = "high win rate but overall unprofitable"
        else:
            effectiveness = "not effective"
            
        effectiveness_insight = {
            "category": "effectiveness",
            "insight": f"The {strategy_type} strategy for {coin} was {effectiveness}.",
            "importance": "high"
        }
        insights.append(effectiveness_insight)
        
        # Insights based on performance analysis
        outcome_distribution = performance.get("outcome_distribution", {}).get("outcomes", {})
        if outcome_distribution:
            # Analyze stop loss hits
            stop_loss_count = outcome_distribution.get("STOP_LOSS", 0)
            total_trades = len(trades)
            
            if total_trades > 0:
                stop_loss_rate = stop_loss_count / total_trades
                
                if stop_loss_rate > 0.4:
                    sl_insight = {
                        "category": "risk_management",
                        "insight": f"High stop loss hit rate ({stop_loss_rate:.2%}) suggests too tight stops or poor entry timing.",
                        "importance": "high"
                    }
                    insights.append(sl_insight)
            
            # Analyze target hits
            target_count = outcome_distribution.get("TARGET_HIT", 0)
            
            if total_trades > 0:
                target_rate = target_count / total_trades
                
                if target_rate < 0.3 and win_rate > 0.4:
                    tp_insight = {
                        "category": "trade_management",
                        "insight": f"Low target hit rate ({target_rate:.2%}) with decent win rate suggests targets may be too ambitious.",
                        "importance": "medium"
                    }
                    insights.append(tp_insight)
        
        # Size analysis insights (for Higher thinking levels)
        if thinking_level.value >= ThinkingLevel.HIGH.value:
            size_analysis = performance.get("size_analysis", {}).get("trades_by_size", {})
            if size_analysis:
                # Find best performing size
                best_size = None
                best_profit = float('-inf')
                
                for size, data in size_analysis.items():
                    if data["count"] >= 3 and data["total_pnl"] > best_profit:
                        best_size = size
                        best_profit = data["total_pnl"]
                
                if best_size:
                    size_insight = {
                        "category": "position_sizing",
                        "insight": f"Position size of {best_size} performed best with total P&L of {best_profit:.2f}.",
                        "importance": "high"
                    }
                    insights.append(size_insight)
        
        # Time-based insights
        time_analysis = performance.get("time_analysis", {})
        if time_analysis:
            trend = time_analysis.get("trend")
            
            if trend == "improving":
                time_insight = {
                    "category": "adaptability",
                    "insight": "Strategy performance improved over time, suggesting good adaptability or improving market conditions.",
                    "importance": "medium"
                }
                insights.append(time_insight)
            elif trend == "deteriorating":
                time_insight = {
                    "category": "adaptability",
                    "insight": "Strategy performance deteriorated over time, suggesting changing market conditions or strategy fatigue.",
                    "importance": "high"
                }
                insights.append(time_insight)
        
        # Generate maximum insights for Maximum thinking level
        if thinking_level.value >= ThinkingLevel.MAXIMUM.value:
            # Advanced volatility impact analysis
            if trades and len(trades) >= 5:
                try:
                    # Group trades by approximate volatility levels
                    for trade in trades:
                        # In a real implementation, we would calculate actual volatility for the trade period
                        # Here we're simulating with a heuristic based on price movement
                        entry_price = trade.get("entry_price", 0)
                        exit_price = trade.get("exit_price", 0)
                        if entry_price > 0:
                            price_change_pct = abs(exit_price - entry_price) / entry_price
                            
                            # Assign volatility category
                            if price_change_pct < 0.01:
                                trade["volatility_level"] = "very_low"
                            elif price_change_pct < 0.03:
                                trade["volatility_level"] = "low"
                            elif price_change_pct < 0.05:
                                trade["volatility_level"] = "medium"
                            elif price_change_pct < 0.1:
                                trade["volatility_level"] = "high"
                            else:
                                trade["volatility_level"] = "very_high"
                    
                    # Group by volatility
                    volatility_performance = {}
                    for trade in trades:
                        vol_level = trade.get("volatility_level", "unknown")
                        pnl = trade.get("pnl_amount", 0)
                        
                        if vol_level not in volatility_performance:
                            volatility_performance[vol_level] = {
                                "count": 0,
                                "total_pnl": 0,
                                "wins": 0,
                                "losses": 0
                            }
                        
                        volatility_performance[vol_level]["count"] += 1
                        volatility_performance[vol_level]["total_pnl"] += pnl
                        
                        if pnl > 0:
                            volatility_performance[vol_level]["wins"] += 1
                        else:
                            volatility_performance[vol_level]["losses"] += 1
                    
                    # Find optimal volatility
                    best_vol = None
                    best_win_rate = 0
                    
                    for vol, data in volatility_performance.items():
                        if data["count"] >= 2:  # Need at least 2 trades for meaningful data
                            win_rate = data["wins"] / data["count"] if data["count"] > 0 else 0
                            
                            if win_rate > best_win_rate:
                                best_win_rate = win_rate
                                best_vol = vol
                    
                    if best_vol:
                        vol_insight = {
                            "category": "market_conditions",
                            "insight": f"Strategy performs best in {best_vol.replace('_', ' ')} volatility conditions with {best_win_rate:.2%} win rate.",
                            "importance": "high"
                        }
                        insights.append(vol_insight)
                        
                except Exception as e:
                    logger.error(f"Error in volatility analysis: {str(e)}")
        
        # Add insights to thinking results
        thinking_results["learning_insights"] = insights
    
    def _generate_improvement_suggestions(self, 
                                         thinking_results: Dict[str, Any], 
                                         coin: str, 
                                         strategy_results: Dict[str, Any], 
                                         market_data: Dict[str, Any], 
                                         sentiment_data: Dict[str, Any],
                                         thinking_level: ThinkingLevel):
        """
        Generate suggestions for improving the strategy.
        
        Args:
            thinking_results (Dict): Thinking results structure to populate
            coin (str): Coin symbol
            strategy_results (Dict): Results of the strategy execution
            market_data (Dict): Market data for the coin
            sentiment_data (Dict): Sentiment data for the coin
            thinking_level (ThinkingLevel): Thinking level
        """
        suggestions = []
        
        # Extract key data
        performance = thinking_results.get("performance_analysis", {})
        insights = thinking_results.get("learning_insights", [])
        win_rate = strategy_results.get("win_rate", 0.0)
        total_profit_loss = strategy_results.get("total_profit_loss", 0.0)
        strategy_type = strategy_results.get("strategy_type", "unknown")
        
        # Generate basic suggestions based on performance
        if total_profit_loss <= 0:
            suggestions.append({
                "category": "strategy",
                "suggestion": f"Consider changing from {strategy_type} to {self._suggest_alternative_strategy(strategy_type, market_data)} for {coin} in current market conditions.",
                "priority": "high"
            })
        
        if win_rate < 0.4:
            suggestions.append({
                "category": "entry",
                "suggestion": "Improve entry criteria to increase win rate, potentially using tighter confirmation signals.",
                "priority": "high"
            })
        
        # Generate suggestions based on insights
        for insight in insights:
            if insight.get("category") == "risk_management" and "stop loss" in insight.get("insight", "").lower():
                suggestions.append({
                    "category": "stops",
                    "suggestion": "Widen stop loss placements or implement trailing stops to avoid premature exits.",
                    "priority": "medium"
                })
            
            if insight.get("category") == "trade_management" and "target" in insight.get("insight", "").lower():
                suggestions.append({
                    "category": "exits",
                    "suggestion": "Set more conservative profit targets or implement partial take-profits at multiple levels.",
                    "priority": "medium"
                })
            
            if insight.get("category") == "position_sizing" and "performed best" in insight.get("insight", "").lower():
                # Extract the size that performed best
                best_size = insight.get("insight", "").split("Position size of ")[1].split(" performed")[0]
                
                suggestions.append({
                    "category": "position_sizing",
                    "suggestion": f"Standardize position sizing to {best_size} which has shown optimal performance.",
                    "priority": "high"
                })
        
        # Suggestions based on market analysis
        price_trend = self._analyze_price_trend(market_data)
        if price_trend:
            trend = price_trend.get("trend", "neutral")
            
            if strategy_type == "long" and trend == "downward":
                suggestions.append({
                    "category": "timing",
                    "suggestion": "Wait for trend reversal confirmation before entering long positions, or switch to a short strategy.",
                    "priority": "high"
                })
            elif strategy_type == "short" and trend == "upward":
                suggestions.append({
                    "category": "timing",
                    "suggestion": "Wait for trend reversal confirmation before entering short positions, or switch to a long strategy.",
                    "priority": "high"
                })
        
        # Add more detailed suggestions for higher thinking levels
        if thinking_level.value >= ThinkingLevel.HIGH.value:
            # Volatility-based suggestions
            volatility = self._calculate_volatility(market_data)
            
            if volatility > 0.05 and strategy_type in ["long", "short"]:
                suggestions.append({
                    "category": "strategy_adjustment",
                    "suggestion": "In high volatility, consider switching to scalping or implementing wider stops with smaller position sizes.",
                    "priority": "medium"
                })
            elif volatility < 0.02 and strategy_type in ["scalp", "sniper"]:
                suggestions.append({
                    "category": "strategy_adjustment",
                    "suggestion": "In low volatility, consider switching to swing trading or range trading strategies.",
                    "priority": "medium"
                })
            
            # Correlation-based suggestions
            correlation = self._analyze_correlations(coin, market_data)
            if correlation and correlation.get("coefficient", 0) > 0.8:
                suggestions.append({
                    "category": "risk_management",
                    "suggestion": f"Due to high correlation with {correlation.get('asset', 'BTC')}, consider hedging or diversifying to reduce systematic risk.",
                    "priority": "medium"
                })
        
        # Add maximum suggestions for Maximum thinking level
        if thinking_level.value >= ThinkingLevel.MAXIMUM.value:
            # Sentiment-based suggestions
            sentiment_score = self._extract_sentiment_score(sentiment_data)
            sentiment_momentum = self._analyze_sentiment_momentum(coin, sentiment_data)
            
            if sentiment_score is not None and sentiment_momentum:
                if strategy_type == "long" and sentiment_score < 0.4 and sentiment_momentum.get("direction") == "deteriorating":
                    suggestions.append({
                        "category": "sentiment_alignment",
                        "suggestion": "Given deteriorating sentiment, consider exiting long positions and waiting for sentiment reversal.",
                        "priority": "high"
                    })
                elif strategy_type == "short" and sentiment_score > 0.6 and sentiment_momentum.get("direction") == "improving":
                    suggestions.append({
                        "category": "sentiment_alignment",
                        "suggestion": "Given improving sentiment, consider exiting short positions and waiting for sentiment reversal.",
                        "priority": "high"
                    })
        
        # Sort suggestions by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 3))
        
        # Add suggestions to thinking results
        thinking_results["improvement_suggestions"] = suggestions
    
    def _generate_future_outlook(self, 
                                thinking_results: Dict[str, Any], 
                                coin: str, 
                                strategy_results: Dict[str, Any], 
                                market_data: Dict[str, Any], 
                                sentiment_data: Dict[str, Any],
                                thinking_level: ThinkingLevel):
        """
        Generate future outlook for the coin and strategy.
        
        Args:
            thinking_results (Dict): Thinking results structure to populate
            coin (str): Coin symbol
            strategy_results (Dict): Results of the strategy execution
            market_data (Dict): Market data for the coin
            sentiment_data (Dict): Sentiment data for the coin
            thinking_level (ThinkingLevel): Thinking level
        """
        future_outlook = {}
        
        # Extract key data
        strategy_type = strategy_results.get("strategy_type", "unknown")
        price_trend = self._analyze_price_trend(market_data)
        sentiment_score = self._extract_sentiment_score(sentiment_data)
        sentiment_momentum = self._analyze_sentiment_momentum(coin, sentiment_data)
        news_impact = self._analyze_news_impact(coin)
        
        # Determine price direction outlook
        price_direction = "sideways"  # Default
        direction_confidence = 0.5  # Default
        
        if price_trend:
            trend = price_trend.get("trend", "neutral")
            strength = price_trend.get("strength", "weak")
            
            # Start with trend-based direction
            if trend == "upward":
                price_direction = "bullish"
                direction_confidence = 0.6
            elif trend == "downward":
                price_direction = "bearish"
                direction_confidence = 0.6
            else:
                price_direction = "sideways"
                direction_confidence = 0.7
            
            # Adjust confidence based on strength
            if strength == "strong":
                direction_confidence += 0.1
            elif strength == "weak":
                direction_confidence -= 0.1
        
        # Adjust based on sentiment
        if sentiment_score is not None:
            if sentiment_score > 0.7 and price_direction != "bullish":
                # Strong bullish sentiment contradicting price trend
                direction_confidence -= 0.1
            elif sentiment_score > 0.7 and price_direction == "bullish":
                # Strong bullish sentiment confirming price trend
                direction_confidence += 0.1
            elif sentiment_score < 0.3 and price_direction != "bearish":
                # Strong bearish sentiment contradicting price trend
                direction_confidence -= 0.1
            elif sentiment_score < 0.3 and price_direction == "bearish":
                # Strong bearish sentiment confirming price trend
                direction_confidence += 0.1
        
        # Adjust based on news
        if news_impact:
            news_sentiment = news_impact.get("sentiment", "neutral")
            news_importance = news_impact.get("importance", "low")
            
            if news_importance in ["high", "very high"]:
                if news_sentiment in ["bullish", "very bullish"] and price_direction != "bullish":
                    direction_confidence -= 0.15
                elif news_sentiment in ["bullish", "very bullish"] and price_direction == "bullish":
                    direction_confidence += 0.15
                elif news_sentiment in ["bearish", "very bearish"] and price_direction != "bearish":
                    direction_confidence -= 0.15
                elif news_sentiment in ["bearish", "very bearish"] and price_direction == "bearish":
                    direction_confidence += 0.15
        
        # Add price direction to outlook
        future_outlook["price_direction"] = {
            "outlook": price_direction,
            "confidence": direction_confidence,
            "explanation": f"Price is expected to move in a {price_direction} direction in the near term."
        }
        
        # Determine volatility outlook
        volatility = self._calculate_volatility(market_data)
        volatility_outlook = "stable"  # Default
        vol_confidence = 0.5  # Default
        
        # Current volatility assessment
        if volatility > 0.05:
            current_vol = "high"
        elif volatility > 0.025:
            current_vol = "moderate"
        else:
            current_vol = "low"
        
        # Predict future volatility based on news and sentiment momentum
        vol_change = 0  # -1 (decreasing), 0 (stable), 1 (increasing)
        
        if news_impact and news_impact.get("importance") in ["high", "very high"]:
            vol_change = 1  # High importance news tends to increase volatility
            vol_confidence = 0.7
        elif sentiment_momentum and sentiment_momentum.get("direction") in ["rapidly improving", "rapidly deteriorating"]:
            vol_change = 1  # Rapid sentiment changes tend to increase volatility
            vol_confidence = 0.65
        elif sentiment_momentum and sentiment_momentum.get("direction") in ["stable"]:
            vol_change = -1 if current_vol != "low" else 0  # Stable sentiment tends to decrease volatility
            vol_confidence = 0.6
        
        # Determine future volatility outlook
        if current_vol == "high" and vol_change <= 0:
            volatility_outlook = "decreasing"
        elif current_vol == "low" and vol_change >= 0:
            volatility_outlook = "increasing"
        elif vol_change > 0:
            volatility_outlook = "increasing"
        elif vol_change < 0:
            volatility_outlook = "decreasing"
        else:
            volatility_outlook = "stable"
            vol_confidence = 0.6
        
        # Add volatility to outlook
        future_outlook["volatility"] = {
            "current": current_vol,
            "outlook": volatility_outlook,
            "confidence": vol_confidence,
            "explanation": f"Volatility is currently {current_vol} and is expected to be {volatility_outlook} in the near term."
        }
        
        # Determine strategy effectiveness outlook
        effectiveness_outlook = "viable"  # Default
        effectiveness_confidence = 0.5  # Default
        
        # Check if current strategy aligns with future outlook
        if strategy_type in ["long", "swing"] and price_direction == "bullish":
            effectiveness_outlook = "highly effective"
            effectiveness_confidence = 0.7
        elif strategy_type in ["short"] and price_direction == "bearish":
            effectiveness_outlook = "highly effective"
            effectiveness_confidence = 0.7
        elif strategy_type in ["long", "swing"] and price_direction == "bearish":
            effectiveness_outlook = "ineffective"
            effectiveness_confidence = 0.7
        elif strategy_type in ["short"] and price_direction == "bullish":
            effectiveness_outlook = "ineffective"
            effectiveness_confidence = 0.7
        elif strategy_type in ["sniper", "scalp"] and volatility_outlook == "increasing":
            effectiveness_outlook = "increasingly effective"
            effectiveness_confidence = 0.65
        elif strategy_type in ["sniper", "scalp"] and volatility_outlook == "decreasing":
            effectiveness_outlook = "decreasingly effective"
            effectiveness_confidence = 0.65
        
        # Add strategy effectiveness to outlook
        future_outlook["strategy_effectiveness"] = {
            "outlook": effectiveness_outlook,
            "confidence": effectiveness_confidence,
            "explanation": f"The {strategy_type} strategy is expected to be {effectiveness_outlook} in the near term."
        }
        
        # Calculate overall confidence in future outlook
        future_confidence = (
            direction_confidence * 0.4 +
            vol_confidence * 0.3 +
            effectiveness_confidence * 0.3
        )
        
        # Add to thinking results
        thinking_results["future_outlook"] = future_outlook
        thinking_results["confidence_in_future"] = future_confidence
    
    def _get_risk_assessment_text(self, risk_score: float) -> str:
        """
        Convert a risk score to a textual assessment.
        
        Args:
            risk_score (float): Risk score (0.0-1.0)
            
        Returns:
            str: Risk assessment text
        """
        if risk_score < 0.2:
            return "very low"
        elif risk_score < 0.4:
            return "low"
        elif risk_score < 0.6:
            return "moderate"
        elif risk_score < 0.8:
            return "high"
        else:
            return "very high"
    
    def _generate_sentiment_thought(self, coin: str, sentiment_score: float) -> str:
        """
        Generate a thought about sentiment based on the score.
        
        Args:
            coin (str): Coin symbol
            sentiment_score (float): Sentiment score (0.0-1.0)
            
        Returns:
            str: Sentiment thought
        """
        if sentiment_score > 0.8:
            return f"{coin} currently has very bullish market sentiment."
        elif sentiment_score > 0.6:
            return f"{coin} currently has bullish market sentiment."
        elif sentiment_score > 0.4:
            return f"{coin} currently has neutral market sentiment."
        elif sentiment_score > 0.2:
            return f"{coin} currently has bearish market sentiment."
        else:
            return f"{coin} currently has very bearish market sentiment."
    
    def _extract_sentiment_score(self, sentiment_data: Dict[str, Any]) -> Optional[float]:
        """
        Extract sentiment score from sentiment data.
        
        Args:
            sentiment_data (Dict): Sentiment data
            
        Returns:
            Optional[float]: Sentiment score (0.0-1.0) or None if not available
        """
        if not sentiment_data:
            return None
        
        # Try to extract sentiment score directly
        if "sentiment_score" in sentiment_data:
            score = sentiment_data["sentiment_score"]
            if isinstance(score, (int, float)):
                return float(score)
        
        # Try to extract from sentiment string
        if "sentiment" in sentiment_data:
            sentiment = sentiment_data["sentiment"]
            if isinstance(sentiment, str):
                sentiment_mapping = {
                    "very_bullish": 0.9,
                    "bullish": 0.7,
                    "neutral": 0.5,
                    "bearish": 0.3,
                    "very_bearish": 0.1
                }
                
                # Try direct mapping first
                if sentiment.lower() in sentiment_mapping:
                    return sentiment_mapping[sentiment.lower()]
                
                # Try partial matching
                for key, value in sentiment_mapping.items():
                    if key in sentiment.lower():
                        return value
        
        return None
    
    def _analyze_price_trend(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze price trend based on market data.
        
        Args:
            market_data (Dict): Market data
            
        Returns:
            Dict: Price trend analysis
        """
        if not market_data:
            return {}
        
        # Check if we have price change data
        if "price_change_percentage_24h" in market_data:
            change_24h = market_data["price_change_percentage_24h"]
            
            # Determine trend direction and strength
            if change_24h > 5:
                trend = "upward"
                strength = "strong"
                confidence = 0.8
            elif change_24h > 2:
                trend = "upward"
                strength = "moderate"
                confidence = 0.7
            elif change_24h > 0.5:
                trend = "upward"
                strength = "weak"
                confidence = 0.6
            elif change_24h > -0.5:
                trend = "neutral"
                strength = "stable"
                confidence = 0.7
            elif change_24h > -2:
                trend = "downward"
                strength = "weak"
                confidence = 0.6
            elif change_24h > -5:
                trend = "downward"
                strength = "moderate"
                confidence = 0.7
            else:
                trend = "downward"
                strength = "strong"
                confidence = 0.8
            
            return {
                "trend": trend,
                "strength": strength,
                "change_24h": change_24h,
                "confidence": confidence
            }
        
        # If no price change data, try to infer from price history if available
        if "price_history" in market_data and isinstance(market_data["price_history"], list) and len(market_data["price_history"]) > 1:
            prices = market_data["price_history"]
            
            # Simple trend analysis based on last few prices
            if prices[-1] > prices[0] * 1.05:
                trend = "upward"
                strength = "strong"
                confidence = 0.7
            elif prices[-1] > prices[0] * 1.02:
                trend = "upward"
                strength = "moderate"
                confidence = 0.6
            elif prices[-1] > prices[0]:
                trend = "upward"
                strength = "weak"
                confidence = 0.5
            elif prices[-1] > prices[0] * 0.98:
                trend = "neutral"
                strength = "stable"
                confidence = 0.6
            elif prices[-1] > prices[0] * 0.95:
                trend = "downward"
                strength = "moderate"
                confidence = 0.6
            else:
                trend = "downward"
                strength = "strong"
                confidence = 0.7
            
            return {
                "trend": trend,
                "strength": strength,
                "confidence": confidence
            }
        
        # Default response if no data available
        return {
            "trend": "neutral",
            "strength": "unknown",
            "confidence": 0.3
        }
    
    def _analyze_volume(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze trading volume based on market data.
        
        Args:
            market_data (Dict): Market data
            
        Returns:
            Dict: Volume analysis
        """
        if not market_data or "volume" not in market_data:
            return {}
        
        volume = market_data.get("volume", 0)
        avg_volume = market_data.get("average_volume", volume)
        
        if avg_volume > 0:
            volume_ratio = volume / avg_volume
            
            if volume_ratio > 2:
                state = "significantly increasing"
                implication = "suggests strong market interest or potential trend change"
                confidence = 0.8
            elif volume_ratio > 1.3:
                state = "increasing"
                implication = "indicates growing market interest"
                confidence = 0.7
            elif volume_ratio > 0.8:
                state = "stable"
                implication = "suggests normal market activity"
                confidence = 0.6
            elif volume_ratio > 0.5:
                state = "decreasing"
                implication = "may indicate diminishing interest"
                confidence = 0.7
            else:
                state = "significantly decreasing"
                implication = "suggests potential market exhaustion or disinterest"
                confidence = 0.8
                
            return {
                "state": state,
                "volume": volume,
                "average_volume": avg_volume,
                "volume_ratio": volume_ratio,
                "implication": implication,
                "confidence": confidence
            }
        
        return {}
    
    def _identify_support_resistance(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify support and resistance levels.
        
        Args:
            market_data (Dict): Market data
            
        Returns:
            Dict: Support and resistance levels
        """
        # This is a simplified implementation
        # In a real system, this would use price history and advanced algorithms
        
        if not market_data or "current_price" not in market_data:
            return {}
        
        current_price = market_data.get("current_price", 0)
        
        if current_price <= 0:
            return {}
        
        # Simulate levels based on current price
        # In a real implementation, these would be calculated from actual price history
        nearest_support = current_price * 0.95
        nearest_resistance = current_price * 1.05
        
        return {
            "nearest_support": round(nearest_support, 2),
            "nearest_resistance": round(nearest_resistance, 2),
            "confidence": 0.6  # Moderate confidence due to simplified calculation
        }
    
    def _analyze_market_cycle(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market cycle phase.
        
        Args:
            market_data (Dict): Market data
            
        Returns:
            Dict: Market cycle analysis
        """
        # This is a simplified implementation
        # In a real system, this would use multiple indicators and historical patterns
        
        if not market_data:
            return {}
        
        # Try to use price change percentages for different timeframes if available
        change_24h = market_data.get("price_change_percentage_24h", 0)
        change_7d = market_data.get("price_change_percentage_7d", change_24h)
        change_30d = market_data.get("price_change_percentage_30d", change_7d)
        
        # Simple heuristic for cycle detection
        if change_30d > 30 and change_7d > 10 and change_24h > 0:
            phase = "late uptrend or euphoria"
            confidence = 0.7
        elif change_30d > 20 and change_7d > 5:
            phase = "uptrend"
            confidence = 0.6
        elif change_30d < -30 and change_7d < -10 and change_24h < 0:
            phase = "downtrend or capitulation"
            confidence = 0.7
        elif change_30d < -20 and change_7d < -5:
            phase = "downtrend"
            confidence = 0.6
        elif abs(change_30d) < 10 and abs(change_7d) < 5:
            phase = "accumulation or distribution"
            confidence = 0.6
        elif change_30d < 0 and change_7d > 5:
            phase = "early recovery"
            confidence = 0.5
        else:
            phase = "unclear"
            confidence = 0.4
        
        return {
            "phase": phase,
            "confidence": confidence
        }
    
    def _analyze_news_impact(self, coin: str) -> Dict[str, Any]:
        """
        Analyze news impact for a coin.
        
        Args:
            coin (str): Coin symbol
            
        Returns:
            Dict: News impact analysis
        """
        # Check if we have news data for this coin
        if not self.news_data or coin not in self.news_data:
            return {}
        
        news_items = self.news_data.get(coin, [])
        
        if not news_items:
            return {}
        
        # Analyze recent news (simplified implementation)
        # In a real system, this would use NLP and sentiment analysis on actual news
        
        # Calculate aggregate sentiment
        sentiment_scores = []
        importance_scores = []
        
        for item in news_items:
            sentiment = item.get("sentiment", "neutral")
            importance = item.get("importance", "low")
            
            # Convert to numeric scores
            sentiment_mapping = {
                "very_bullish": 0.9,
                "bullish": 0.7,
                "neutral": 0.5,
                "bearish": 0.3,
                "very_bearish": 0.1
            }
            
            importance_mapping = {
                "very_high": 0.9,
                "high": 0.7,
                "moderate": 0.5,
                "low": 0.3,
                "very_low": 0.1
            }
            
            if isinstance(sentiment, str):
                s_score = sentiment_mapping.get(sentiment.lower(), 0.5)
            else:
                s_score = 0.5
                
            if isinstance(importance, str):
                i_score = importance_mapping.get(importance.lower(), 0.3)
            else:
                i_score = 0.3
                
            sentiment_scores.append(s_score)
            importance_scores.append(i_score)
        
        if not sentiment_scores:
            return {}
        
        # Calculate weighted average sentiment
        weighted_sentiments = [s * i for s, i in zip(sentiment_scores, importance_scores)]
        if sum(importance_scores) > 0:
            avg_sentiment = sum(weighted_sentiments) / sum(importance_scores)
        else:
            avg_sentiment = 0.5
        
        # Calculate max importance
        max_importance = max(importance_scores) if importance_scores else 0.3
        
        # Convert back to text
        if avg_sentiment > 0.75:
            sentiment_text = "very bullish"
        elif avg_sentiment > 0.6:
            sentiment_text = "bullish"
        elif avg_sentiment > 0.4:
            sentiment_text = "neutral"
        elif avg_sentiment > 0.25:
            sentiment_text = "bearish"
        else:
            sentiment_text = "very bearish"
            
        if max_importance > 0.75:
            importance_text = "very high"
        elif max_importance > 0.6:
            importance_text = "high"
        elif max_importance > 0.4:
            importance_text = "moderate"
        elif max_importance > 0.25:
            importance_text = "low"
        else:
            importance_text = "very low"
        
        return {
            "sentiment": sentiment_text,
            "importance": importance_text,
            "sentiment_score": avg_sentiment,
            "importance_score": max_importance,
            "news_count": len(news_items),
            "confidence": min(0.9, 0.5 + (len(news_items) / 20))  # More news items = higher confidence
        }
    
    def _analyze_correlations(self, coin: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze correlations with major assets.
        
        Args:
            coin (str): Coin symbol
            market_data (Dict): Market data
            
        Returns:
            Dict: Correlation analysis
        """
        # In a real implementation, this would calculate actual correlation coefficients
        # Here we're providing a simplified version
        
        if coin.upper() in ["BTC", "BITCOIN"]:
            return {
                "asset": "ETH",
                "coefficient": 0.85,
                "strength": "strong positive",
                "confidence": 0.85
            }
        elif coin.upper() in ["ETH", "ETHEREUM"]:
            return {
                "asset": "BTC",
                "coefficient": 0.85,
                "strength": "strong positive",
                "confidence": 0.85
            }
        elif coin.upper() in ["XRP", "RIPPLE"]:
            return {
                "asset": "BTC",
                "coefficient": 0.72,
                "strength": "moderate positive",
                "confidence": 0.7
            }
        else:
            # For other coins, default to BTC correlation
            # In real implementation, this would be calculated from price data
            return {
                "asset": "BTC",
                "coefficient": 0.67,  # Most alts have strong BTC correlation
                "strength": "moderate positive",
                "confidence": 0.6
            }
    
    def _identify_patterns(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify technical patterns in price data.
        
        Args:
            market_data (Dict): Market data
            
        Returns:
            Dict: Pattern identification
        """
        # This is a simplified implementation
        # In a real system, this would use technical analysis libraries
        
        # Randomly select a pattern for demonstration purposes
        # In a real implementation, actual pattern detection algorithms would be used
        patterns = [
            {
                "pattern": "double bottom",
                "indication": "potential bullish reversal",
                "confidence": 0.65
            },
            {
                "pattern": "head and shoulders",
                "indication": "potential bearish reversal",
                "confidence": 0.7
            },
            {
                "pattern": "bull flag",
                "indication": "potential continuation of uptrend",
                "confidence": 0.6
            },
            {
                "pattern": "descending triangle",
                "indication": "potential breakdown",
                "confidence": 0.6
            },
            {
                "pattern": "ascending triangle",
                "indication": "potential breakout",
                "confidence": 0.6
            }
        ]
        
        # For demonstration, return a random pattern or empty dict
        # In a real implementation, actual pattern detection would be performed
        if random.random() < 0.3:  # 30% chance of detecting a pattern
            return random.choice(patterns)
        else:
            return {}
    
    def _analyze_contrarian_indicators(self, 
                                      coin: str, 
                                      market_data: Dict[str, Any], 
                                      sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze contrarian indicators.
        
        Args:
            coin (str): Coin symbol
            market_data (Dict): Market data
            sentiment_data (Dict): Sentiment data
            
        Returns:
            Dict: Contrarian indicator analysis
        """
        # Check for extreme sentiment as contrarian indicator
        sentiment_score = self._extract_sentiment_score(sentiment_data)
        price_change_24h = market_data.get("price_change_percentage_24h", 0)
        
        if sentiment_score is not None:
            if sentiment_score > 0.85 and price_change_24h > 10:
                return {
                    "indicator": "extreme bullish sentiment",
                    "implication": "potential short-term top or pullback as market may be overheated",
                    "confidence": 0.7
                }
            elif sentiment_score < 0.15 and price_change_24h < -10:
                return {
                    "indicator": "extreme bearish sentiment",
                    "implication": "potential short-term bottom or bounce as market may be oversold",
                    "confidence": 0.7
                }
        
        # Volume divergence as contrarian indicator
        volume = market_data.get("volume", 0)
        avg_volume = market_data.get("average_volume", volume)
        
        if volume > 0 and avg_volume > 0:
            volume_ratio = volume / avg_volume
            
            if volume_ratio > 3 and price_change_24h > 15:
                return {
                    "indicator": "volume climax",
                    "implication": "potential exhaustion of buying pressure and upcoming reversal",
                    "confidence": 0.65
                }
            elif volume_ratio < 0.3 and abs(price_change_24h) < 1:
                return {
                    "indicator": "extremely low volume during consolidation",
                    "implication": "potential upcoming volatility or breakout",
                    "confidence": 0.6
                }
        
        # No strong contrarian indicator found
        return {}
    
    def _analyze_liquidity(self, coin: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market liquidity.
        
        Args:
            coin (str): Coin symbol
            market_data (Dict): Market data
            
        Returns:
            Dict: Liquidity analysis
        """
        # Simplified liquidity analysis based on volume and market cap
        volume = market_data.get("volume", 0)
        market_cap = market_data.get("market_cap", 0)
        
        if volume > 0 and market_cap > 0:
            # Volume to market cap ratio as a simple liquidity measure
            liquidity_ratio = volume / market_cap
            
            if liquidity_ratio > 0.2:
                state = "very high"
                implication = "allows for easy entry and exit with minimal slippage"
            elif liquidity_ratio > 0.1:
                state = "high"
                implication = "supports significant positions with manageable slippage"
            elif liquidity_ratio > 0.05:
                state = "moderate"
                implication = "suitable for medium-sized positions but may experience some slippage"
            elif liquidity_ratio > 0.02:
                state = "low"
                implication = "may experience significant slippage for larger positions"
            else:
                state = "very low"
                implication = "poses high risk of slippage and potential liquidity traps"
                
            return {
                "state": state,
                "liquidity_ratio": liquidity_ratio,
                "implication": implication,
                "confidence": 0.7
            }
        
        # If market cap or volume is missing, make an educated guess based on coin rank or popularity
        if coin.upper() in ["BTC", "ETH"]:
            return {
                "state": "very high",
                "implication": "allows for easy entry and exit with minimal slippage",
                "confidence": 0.8
            }
        elif coin.upper() in ["BNB", "XRP", "SOL", "ADA", "DOT"]:
            return {
                "state": "high",
                "implication": "supports significant positions with manageable slippage",
                "confidence": 0.7
            }
        else:
            return {
                "state": "moderate",
                "implication": "suitable for medium-sized positions but may experience some slippage",
                "confidence": 0.5
            }
    
    def _analyze_whale_activity(self, coin: str) -> Dict[str, Any]:
        """
        Analyze whale activity for a coin.
        
        Args:
            coin (str): Coin symbol
            
        Returns:
            Dict: Whale activity analysis
        """
        # This is a simplified implementation
        # In a real system, this would use data from blockchain analytics providers
        
        # For demonstration purposes only
        # In a real implementation, actual whale transaction data would be analyzed
        activities = [
            {
                "type": "accumulation",
                "description": "Large holders have been accumulating in the past 24 hours",
                "significance": "bullish",
                "confidence": 0.7
            },
            {
                "type": "distribution",
                "description": "Large holders have been distributing in the past 24 hours",
                "significance": "bearish",
                "confidence": 0.7
            },
            {
                "type": "transfer",
                "description": "Significant movement between wallets but no exchange deposits",
                "significance": "neutral",
                "confidence": 0.6
            },
            {
                "type": "exchange_inflow",
                "description": "Increased inflow to exchanges suggesting potential selling pressure",
                "significance": "slightly bearish",
                "confidence": 0.65
            },
            {
                "type": "exchange_outflow",
                "description": "Increased outflow from exchanges suggesting accumulation",
                "significance": "slightly bullish",
                "confidence": 0.65
            }
        ]
        
        # For demonstration, return a random activity or empty dict
        # In a real implementation, actual whale activity would be analyzed
        if random.random() < 0.3:  # 30% chance of detecting whale activity
            return random.choice(activities)
        else:
            return {}
    
    def _calculate_volatility(self, market_data: Dict[str, Any]) -> float:
        """
        Calculate volatility from market data.
        
        Args:
            market_data (Dict): Market data
            
        Returns:
            float: Volatility as a decimal (e.g., 0.05 for 5%)
        """
        # If we have high-low data, use that for the volatility calculation
        if "high_24h" in market_data and "low_24h" in market_data:
            high = market_data["high_24h"]
            low = market_data["low_24h"]
            
            if high > 0:
                volatility = (high - low) / high
                return volatility
        
        # If we have price history, calculate standard deviation
        if "price_history" in market_data and isinstance(market_data["price_history"], list) and len(market_data["price_history"]) > 1:
            prices = market_data["price_history"]
            
            try:
                # Calculate volatility as normalized standard deviation
                if len(prices) > 1:
                    std = statistics.stdev(prices)
                    mean = statistics.mean(prices)
                    
                    if mean > 0:
                        return std / mean
            except Exception:
                pass
        
        # If we have price change percentage, use that as a rough approximation
        if "price_change_percentage_24h" in market_data:
            return abs(market_data["price_change_percentage_24h"]) / 100
        
        # Default volatility estimate
        return 0.03  # 3% default volatility
    
    def _analyze_sentiment_momentum(self, coin: str, sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze sentiment momentum.
        
        Args:
            coin (str): Coin symbol
            sentiment_data (Dict): Sentiment data
            
        Returns:
            Dict: Sentiment momentum analysis
        """
        # In a real implementation, this would analyze sentiment trend over time
        # Here we're using a simplified approach
        
        current_sentiment = self._extract_sentiment_score(sentiment_data)
        if current_sentiment is None:
            return {}
        
        # Check if we have previous sentiment data
        prev_sentiment = sentiment_data.get("previous_sentiment", current_sentiment)
        
        # Calculate change
        sentiment_change = current_sentiment - prev_sentiment
        
        # Determine direction and magnitude
        if sentiment_change > 0.2:
            direction = "rapidly improving"
            magnitude = "strong"
        elif sentiment_change > 0.05:
            direction = "improving"
            magnitude = "moderate"
        elif sentiment_change > -0.05:
            direction = "stable"
            magnitude = "minimal"
        elif sentiment_change > -0.2:
            direction = "deteriorating"
            magnitude = "moderate"
        else:
            direction = "rapidly deteriorating"
            magnitude = "strong"
        
        return {
            "current": current_sentiment,
            "previous": prev_sentiment,
            "change": sentiment_change,
            "direction": direction,
            "magnitude": magnitude,
            "confidence": 0.6  # Moderate confidence due to simplified calculation
        }
    
    def _suggest_alternative_strategy(self, current_strategy: str, market_data: Dict[str, Any]) -> str:
        """
        Suggest an alternative strategy based on market conditions.
        
        Args:
            current_strategy (str): Current strategy
            market_data (Dict): Market data
            
        Returns:
            str: Suggested alternative strategy
        """
        price_trend = self._analyze_price_trend(market_data)
        volatility = self._calculate_volatility(market_data)
        
        trend = price_trend.get("trend", "neutral") if price_trend else "neutral"
        strength = price_trend.get("strength", "weak") if price_trend else "weak"
        
        # Suggest based on market conditions
        if trend == "upward" and strength in ["moderate", "strong"]:
            if current_strategy == "short":
                return "long"
            elif current_strategy == "sniper":
                return "swing"
            else:
                return "long"
        elif trend == "downward" and strength in ["moderate", "strong"]:
            if current_strategy == "long":
                return "short"
            elif current_strategy == "swing":
                return "sniper"
            else:
                return "short"
        elif trend == "neutral" or strength == "weak":
            if volatility > 0.05:  # High volatility
                return "sniper"
            else:  # Low volatility
                return "grid"
        
        # Default alternatives by strategy type
        strategy_alternatives = {
            "long": "sniper",
            "short": "sniper",
            "sniper": "grid",
            "swing": "long",
            "scalp": "sniper",
            "grid": "dca",
            "dca": "grid"
        }
        
        return strategy_alternatives.get(current_strategy.lower(), "sniper")
    
    def _get_market_data(self, coin: str) -> Dict[str, Any]:
        """
        Get market data for a coin.
        
        Args:
            coin (str): Coin symbol
            
        Returns:
            Dict: Market data
        """
        # Try to get from CoinGecko data
        if "coingecko" in self.market_data and coin in self.market_data["coingecko"]:
            return self.market_data["coingecko"][coin]
        
        # Try to get from Binance data
        if "binance" in self.market_data and "symbols" in self.market_data["binance"]:
            for symbol in self.market_data["binance"]["symbols"]:
                if symbol.get("baseAsset", "").upper() == coin.upper():
                    # Convert to standard format
                    return {
                        "current_price": float(symbol.get("price", 0.0)),
                        "market_cap": 0.0,
                        "volume": float(symbol.get("volume", 0.0)),
                        "price_change_percentage_24h": float(symbol.get("priceChangePercent", 0.0))
                    }
        
        # Try to fetch fresh data if available modules exist
        if COINGECKO_AVAILABLE:
            try:
                data = coingecko_collector.get_coin_data(coin)
                if data:
                    return data
            except Exception as e:
                logger.error(f"Error fetching CoinGecko data: {str(e)}")
        
        if BINANCE_AVAILABLE:
            try:
                data = binance_collector.get_symbol_data(f"{coin}USDT")
                if data:
                    return data
            except Exception as e:
                logger.error(f"Error fetching Binance data: {str(e)}")
        
        # Return empty dict if no data found
        return {}
    
    def _get_sentiment_data(self, coin: str) -> Dict[str, Any]:
        """
        Get sentiment data for a coin.
        
        Args:
            coin (str): Coin symbol
            
        Returns:
            Dict: Sentiment data
        """
        # Try to get from cached sentiment data
        if coin in self.sentiment_data:
            return self.sentiment_data[coin]
        
        # Try to fetch fresh data if sentiment analyzer is available
        if SENTIMENT_ANALYZER_AVAILABLE:
            try:
                data = sentiment_analyzer.get_sentiment(coin)
                if data:
                    return data
            except Exception as e:
                logger.error(f"Error fetching sentiment data: {str(e)}")
        
        # Return empty dict if no data found
        return {}
    
    def _calculate_scenario_probability(self, 
                                       coin: str, 
                                       scenario_type: ScenarioType, 
                                       market_data: Dict[str, Any], 
                                       sentiment_data: Dict[str, Any]) -> float:
        """
        Calculate probability for a market scenario.
        
        Args:
            coin (str): Coin symbol
            scenario_type (ScenarioType): Type of scenario
            market_data (Dict): Market data
            sentiment_data (Dict): Sentiment data
            
        Returns:
            float: Probability (0.0-1.0)
        """
        # Extract key factors
        price_trend = self._analyze_price_trend(market_data)
        trend = price_trend.get("trend", "neutral") if price_trend else "neutral"
        strength = price_trend.get("strength", "weak") if price_trend else "weak"
        
        sentiment_score = self._extract_sentiment_score(sentiment_data)
        volatility = self._calculate_volatility(market_data)
        
        # Base probabilities
        base_probs = {
            ScenarioType.BULLISH: 0.3,
            ScenarioType.BEARISH: 0.3,
            ScenarioType.SIDEWAYS: 0.3,
            ScenarioType.BREAKOUT: 0.1,
            ScenarioType.BREAKDOWN: 0.1,
            ScenarioType.MANIPULATION: 0.05,
            ScenarioType.BLACK_SWAN: 0.02
        }
        
        # Get base probability
        probability = base_probs.get(scenario_type, 0.1)
        
        # Adjust based on trend
        if scenario_type == ScenarioType.BULLISH:
            if trend == "upward":
                probability += 0.2
                if strength == "strong":
                    probability += 0.1
            elif trend == "downward":
                probability -= 0.2
                if strength == "strong":
                    probability -= 0.05
        
        elif scenario_type == ScenarioType.BEARISH:
            if trend == "downward":
                probability += 0.2
                if strength == "strong":
                    probability += 0.1
            elif trend == "upward":
                probability -= 0.2
                if strength == "strong":
                    probability -= 0.05
        
        elif scenario_type == ScenarioType.SIDEWAYS:
            if trend == "neutral":
                probability += 0.3
            else:
                probability -= 0.1
                if strength == "strong":
                    probability -= 0.1
        
        elif scenario_type == ScenarioType.BREAKOUT:
            if trend == "upward":
                probability += 0.15
                if strength == "strong":
                    probability += 0.1
            # More likely with high volatility
            probability += volatility
        
        elif scenario_type == ScenarioType.BREAKDOWN:
            if trend == "downward":
                probability += 0.15
                if strength == "strong":
                    probability += 0.1
            # More likely with high volatility
            probability += volatility
        
        # Adjust based on sentiment
        if sentiment_score is not None:
            if scenario_type == ScenarioType.BULLISH:
                probability += (sentiment_score - 0.5) * 0.3
            elif scenario_type == ScenarioType.BEARISH:
                probability += (0.5 - sentiment_score) * 0.3
        
        # Ensure probability is within valid range
        probability = max(0.01, min(0.99, probability))
        
        return probability
    
    def _log_thinking_process(self, thinking_results: Dict[str, Any], thinking_type: str) -> None:
        """
        Log the thinking process to file.
        
        Args:
            thinking_results (Dict): Thinking results to log
            thinking_type (str): Type of thinking process
        """
        try:
            # Create log filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            coin = thinking_results.get("coin", "unknown")
            log_file = os.path.join(THINKING_LOGS_DIR, f"{coin}_{thinking_type}_{timestamp}.json")
            
            # Save to file
            with open(log_file, 'w') as f:
                json.dump(thinking_results, f, indent=2)
                
            logger.info(f"Saved thinking results to {log_file}")
            
            # Generate report if available
            if REPORT_GENERATOR_AVAILABLE:
                try:
                    report_generator.generate_thinking_report(thinking_results, thinking_type)
                except Exception as e:
                    logger.error(f"Error generating thinking report: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error logging thinking process: {str(e)}")


def main():
    """Main function to run the deep thinking process standalone."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='DeepSeek Deep Thinking Process')
    parser.add_argument('--real', action='store_true', help='Run in real mode (with real money)')
    parser.add_argument('--silent', action='store_true', help='Run in silent mode (minimal output)')
    parser.add_argument('--thinking-level', type=int, default=DEFAULT_THINKING_LEVEL, choices=range(1, 6),
                       help=f'Thinking depth level (1-5, default: {DEFAULT_THINKING_LEVEL})')
    parser.add_argument('--coin', type=str, help='Coin to analyze')
    args = parser.parse_args()
    
    # Initialize deep thinking process
    thinking_process = DeepThinkingProcess(
        real_mode=args.real,
        silent_mode=args.silent,
        thinking_level=args.thinking_level
    )
    
    # If coin is provided, do a sample analysis
    if args.coin:
        coin = args.coin.upper()
        print(f"\n{Fore.CYAN}Running sample analysis for {coin}...{Style.RESET_ALL}")
        
        # Mock GPT decision for testing
        gpt_decision = {
            "action": "buy",
            "symbol": coin,
            "direction": "long",
            "size": 0.3,
            "stop_loss": 0.0,  # Will be calculated
            "take_profit": 0.0,  # Will be calculated
            "timeframe": "4h",
            "urgency": 7,
            "type": "entry",
            "reasoning": f"Analysis suggests {coin} is showing bullish momentum with increasing volume.",
            "confidence": 0.75
        }
        
        # Run pre-trade thinking
        print(f"{Fore.YELLOW}Starting pre-trade thinking process...{Style.RESET_ALL}")
        pre_trade_results = thinking_process.process_pre_trade(
            coin=coin,
            gpt_decision=gpt_decision,
            strategy_type="long"
        )
        
        # Display key results
        if pre_trade_results:
            deepseek_decision = pre_trade_results.get("deepseek_decision", {})
            agreement = pre_trade_results.get("agreement", False)
            confidence = pre_trade_results.get("confidence_score", 0.0)
            
            print(f"\n{Fore.GREEN}Pre-trade thinking complete!{Style.RESET_ALL}")
            print(f"Agreement with GPT-4o: {Fore.GREEN if agreement else Fore.RED}{agreement}{Style.RESET_ALL}")
            print(f"Confidence score: {Fore.YELLOW}{confidence:.2f}{Style.RESET_ALL}")
            print(f"\nDeepSeek Decision:")
            print(f"  Action: {Fore.CYAN}{deepseek_decision.get('action', 'unknown')}{Style.RESET_ALL}")
            print(f"  Direction: {Fore.CYAN}{deepseek_decision.get('direction', 'unknown')}{Style.RESET_ALL}")
            print(f"  Size: {Fore.CYAN}{deepseek_decision.get('size', 0):.2f}{Style.RESET_ALL}")
            print(f"  Confidence: {Fore.YELLOW}{deepseek_decision.get('confidence', 0):.2f}{Style.RESET_ALL}")
            
            # Show thoughts
            print(f"\n{Fore.MAGENTA}Top Thoughts:{Style.RESET_ALL}")
            for thought in pre_trade_results.get("thoughts", [])[:3]:
                print(f"  {Fore.WHITE}{thought.get('thought', '')}{Style.RESET_ALL}")
            
            # Show risk assessment
            risk = pre_trade_results.get("risk_assessment", {}).get("total_risk_assessment", "unknown")
            print(f"\nRisk Assessment: {Fore.YELLOW}{risk}{Style.RESET_ALL}")
            
            # Show top scenario
            scenarios = pre_trade_results.get("scenarios", [])
            if scenarios:
                top_scenario = scenarios[0]
                print(f"\nMost Likely Scenario ({top_scenario.get('probability', 0)*100:.1f}%):")
                print(f"  {Fore.WHITE}{top_scenario.get('description', '')}{Style.RESET_ALL}")
    
    else:
        print(f"\n{Fore.YELLOW}No coin specified. Run with --coin SYMBOL to analyze a specific coin.{Style.RESET_ALL}")
        print(f"Example: python deep_thinking_process.py --thinking-level 4 --coin BTC")
    
    print(f"\n{Fore.CYAN}Deep thinking process completed.{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
# Syntax hatası düzeltildi - 2025-04-25 00:17:34
