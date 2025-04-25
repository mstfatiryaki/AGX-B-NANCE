from random import choice
import random
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SentientTrader.AI - Thinking Controller Module
----------------------------------------------
A meta-controller that manages, analyzes and resolves decision conflicts
between GPT-4o (primary AI) and DeepSeek (secondary AI) in the trading system.
It provides oversight, decision comparison, conflict resolution, and memory management.

Created by: mstfatiryaki
Date: 2025-04-22
Version: 1.0.0
"""

import os
import sys
import json
import time
import threading
import logging
import datetime
import queue
import random
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(project_root)

# Try importing optional dependencies
try:
    from modules.utils import alert_system
    ALERT_SYSTEM_AVAILABLE = True
except ImportError:
    ALERT_SYSTEM_AVAILABLE = False
    logging.warning("alert_system module not available, using standard logging instead")

try:
    from modules.core import memory_core
    MEMORY_CORE_AVAILABLE = True
except ImportError:
    MEMORY_CORE_AVAILABLE = False
    logging.warning("memory_core module not available, memory features disabled")

try:
    from modules.ai import deepseek_interface
    DEEPSEEK_INTERFACE_AVAILABLE = True
except ImportError:
    DEEPSEEK_INTERFACE_AVAILABLE = False
    logging.warning("deepseek_interface module not available, dual-AI features limited")

try:
    from modules.interface import terminal_interface
    TERMINAL_INTERFACE_AVAILABLE = True
except ImportError:
    TERMINAL_INTERFACE_AVAILABLE = False
    logging.warning("terminal_interface module not available, using standard output")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, "logs", "thinking_controller.log"))
    ]
)

logger = logging.getLogger("ThinkingController")

# Constants
VERSION = "1.0.0"
LOG_FILE = os.path.join(project_root, "logs", "thinking_controller_decisions.json")
MONITOR_INTERVAL = 5  # seconds

class ThinkingMode(Enum):
    """Enum for different thinking modes of the controller."""
    BALANCED = "balanced"       # Equal weight to both AIs
    CRITICAL = "critical"       # More weight to contradicting opinions
    SUPPORTIVE = "supportive"   # More weight to agreeing opinions
    ANALYTICAL = "analytical"   # More weight to data-backed decisions
    OVERRIDE_ONLY = "override_only"  # Only intervene on critical disagreements


class DecisionType(Enum):
    """Enum for types of trading decisions."""
    ENTRY = "entry"             # New position entry
    EXIT = "exit"               # Position exit
    ADJUSTMENT = "adjustment"   # Position size/stop adjustment
    ANALYSIS = "analysis"       # Market analysis without action
    ALERT = "alert"             # Price/event alert
    STRATEGY = "strategy"       # Strategy selection


class DecisionOutcome(Enum):
    """Enum for the outcome of a decision comparison."""
    AGREEMENT = "agreement"           # Both AIs agree
    MINOR_DISAGREEMENT = "minor_disagreement"  # Small differences in approach
    MAJOR_DISAGREEMENT = "major_disagreement"  # Significant strategy differences
    CRITICAL_CONFLICT = "critical_conflict"    # Completely opposing views
    GPT_OVERRIDE = "gpt_override"     # GPT decision was forced
    DEEPSEEK_OVERRIDE = "deepseek_override"  # DeepSeek decision was forced
    HUMAN_INTERVENTION = "human_intervention"  # Human made the decision


class ThinkingController:
    """
    Meta-controller that manages decisions between GPT-4o and DeepSeek,
    analyzes conflicts, and provides oversight for the trading system.
    """
    
    def __init__(self, 
                 real_mode: bool = False,
                 silent_mode: bool = False,
                 thinking_mode: ThinkingMode = ThinkingMode.BALANCED):
        """
        Initialize the thinking controller.
        
        Args:
            real_mode (bool): Whether the system is using real money
            silent_mode (bool): Whether to minimize console output
            thinking_mode (ThinkingMode): The default thinking mode
        """
        self.real_mode = real_mode
        self.silent_mode = silent_mode
        self.thinking_mode = thinking_mode
        self.is_running = False
        self.decision_queue = queue.Queue()
        self.background_thread = None
        self.last_decisions = {}  # Keep track of recent decisions
        self.decision_history = []  # History of decision comparisons
        
        # Initialize alert system if available
        if ALERT_SYSTEM_AVAILABLE:
            self.alert = alert_system.AlertSystem(
                module_name="thinking_controller",
                real_mode=real_mode,
                log_to_file=True
            )
        else:
            self.alert = logger
            
        # Initialize memory core if available
        if MEMORY_CORE_AVAILABLE:
            self.memory = memory_core.MemoryCore(real_mode=real_mode)
        else:
            self.memory = None
        
        # Initialize terminal interface if available
        if TERMINAL_INTERFACE_AVAILABLE:
            self.terminal = terminal_interface.TerminalInterface()
        else:
            self.terminal = None
            
        # Initialize DeepSeek interface if available
        if DEEPSEEK_INTERFACE_AVAILABLE:
            self.deepseek = deepseek_interface.DeepSeekInterface(
                silent_mode=silent_mode,
                real_mode=real_mode
            )
        else:
            self.deepseek = None
        
        # Initialize log directory and files
        self._init_logs()
        
        logger.info(f"Thinking Controller initialized (mode: {thinking_mode.value}, real_mode: {real_mode})")
    
    def _init_logs(self):
        """Initialize log files and directories."""
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(LOG_FILE)
        os.makedirs(log_dir, exist_ok=True)
        
        # Check if log file exists, create with empty array if not
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'w') as f:
                json.dump([], f)
            logger.info(f"Created new decision log file at: {LOG_FILE}")
        else:
            logger.info(f"Using existing decision log file at: {LOG_FILE}")
    
    def start_monitoring(self):
        """Start the background monitoring thread."""
        if self.is_running:
            logger.warning("Monitoring already running")
            return False
        
        self.is_running = True
        self.background_thread = threading.Thread(
            target=self._background_monitor,
            daemon=True
        )
        self.background_thread.start()
        
        logger.info("Started background decision monitoring")
        if not self.silent_mode:
            self._print_to_terminal("🧠 Thinking Controller started monitoring AI decisions")
        
        return True
    
    def stop_monitoring(self):
        """Stop the background monitoring thread."""
        if not self.is_running:
            logger.warning("Monitoring not running")
            return False
        
        self.is_running = False
        if self.background_thread:
            self.background_thread.join(timeout=2.0)
            
        logger.info("Stopped background decision monitoring")
        if not self.silent_mode:
            self._print_to_terminal("🧠 Thinking Controller stopped monitoring")
        
        return True
    
    def set_thinking_mode(self, mode: ThinkingMode):
        """
        Set the thinking mode for the controller.
        
        Args:
            mode (ThinkingMode): The thinking mode to set
        """
        self.thinking_mode = mode
        logger.info(f"Thinking mode set to: {mode.value}")
        if not self.silent_mode:
            self._print_to_terminal(f"🧠 Thinking mode changed to: {mode.value}")
        
        # If DeepSeek is available, notify it of the change
        if self.deepseek:
            # This would change DeepSeek's thinking mode to match
            # if the deepseek_interface supports this feature
            pass
    
    def compare_decisions(self, 
                         gpt_decision: Dict[str, Any], 
                         deepseek_decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare decisions from GPT-4o and DeepSeek and determine the best course of action.
        
        Args:
            gpt_decision (Dict): Decision from GPT-4o
            deepseek_decision (Dict): Decision from DeepSeek
            
        Returns:
            Dict: Analysis results including confidence scores and recommended action
        """
        logger.info(f"Comparing decisions: {gpt_decision.get('action', 'unknown')} vs {deepseek_decision.get('action', 'unknown')}")
        
        # Basic validation
        if not gpt_decision or not deepseek_decision:
            logger.error("Invalid decision data provided")
            return {
                "outcome": DecisionOutcome.HUMAN_INTERVENTION.value,
                "error": "Invalid decision data provided",
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        # Extract key decision components
        try:
            gpt_action = gpt_decision.get('action', '').lower()
            deepseek_action = deepseek_decision.get('action', '').lower()
            
            gpt_symbol = gpt_decision.get('symbol', '').upper()
            deepseek_symbol = deepseek_decision.get('symbol', '').upper()
            
            gpt_direction = gpt_decision.get('direction', '').lower()
            deepseek_direction = deepseek_decision.get('direction', '').lower()
            
            gpt_size = gpt_decision.get('size', 0)
            deepseek_size = deepseek_decision.get('size', 0)
            
            gpt_reason = gpt_decision.get('reasoning', '')
            deepseek_reason = deepseek_decision.get('reasoning', '')
            
            decision_type = gpt_decision.get('type', DecisionType.ANALYSIS.value)
        except Exception as e:
            logger.error(f"Error extracting decision components: {str(e)}")
            return {
                "outcome": DecisionOutcome.HUMAN_INTERVENTION.value,
                "error": f"Error parsing decision data: {str(e)}",
                "timestamp": datetime.datetime.now().isoformat()
            }
            
        # Analyze differences
        differences = self._analyze_differences(
            gpt_decision, 
            deepseek_decision
        )
        
        # Calculate confidence scores
        confidence_scores = self._calculate_confidence_scores(
            gpt_decision, 
            deepseek_decision, 
            differences
        )
        
        # Determine the outcome
        outcome = self._determine_outcome(differences, confidence_scores)
        
        # Create decision summary
        decision_summary = self._create_decision_summary(
            gpt_decision,
            deepseek_decision,
            differences,
            confidence_scores,
            outcome
        )
        
        # Save to memory and log
        self._save_to_memory(decision_summary)
        self._log_decision(decision_summary)
        
        # Alert user if necessary
        if outcome in [
            DecisionOutcome.MAJOR_DISAGREEMENT.value,
            DecisionOutcome.CRITICAL_CONFLICT.value
        ]:
            self._alert_user(decision_summary)
        
        return decision_summary
    
    def _analyze_differences(self, 
                            gpt_decision: Dict[str, Any], 
                            deepseek_decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the differences between GPT-4o and DeepSeek decisions.
        
        Args:
            gpt_decision (Dict): Decision from GPT-4o
            deepseek_decision (Dict): Decision from DeepSeek
            
        Returns:
            Dict: Analysis of differences
        """
        differences = {}
        
        # Compare actions (e.g., buy, sell, hold)
        if gpt_decision.get('action', '') != deepseek_decision.get('action', ''):
            differences['action'] = {
                'gpt': gpt_decision.get('action', ''),
                'deepseek': deepseek_decision.get('action', ''),
                'severity': 'high'
            }
        
        # Compare symbols
        if gpt_decision.get('symbol', '') != deepseek_decision.get('symbol', ''):
            differences['symbol'] = {
                'gpt': gpt_decision.get('symbol', ''),
                'deepseek': deepseek_decision.get('symbol', ''),
                'severity': 'high'
            }
        
        # Compare direction
        if gpt_decision.get('direction', '') != deepseek_decision.get('direction', ''):
            differences['direction'] = {
                'gpt': gpt_decision.get('direction', ''),
                'deepseek': deepseek_decision.get('direction', ''),
                'severity': 'high'
            }
        
        # Compare size/amount
        gpt_size = gpt_decision.get('size', 0)
        deepseek_size = deepseek_decision.get('size', 0)
        
        if isinstance(gpt_size, (int, float)) and isinstance(deepseek_size, (int, float)):
            size_diff_pct = 0
            if max(gpt_size, deepseek_size) > 0:
                size_diff_pct = abs(gpt_size - deepseek_size) / max(gpt_size, deepseek_size)
                
            if size_diff_pct > 0.2:  # 20% difference threshold
                differences['size'] = {
                    'gpt': gpt_size,
                    'deepseek': deepseek_size,
                    'diff_percentage': round(size_diff_pct * 100, 2),
                    'severity': 'medium' if size_diff_pct < 0.5 else 'high'
                }
        
        # Compare stop loss
        gpt_stop = gpt_decision.get('stop_loss', 0)
        deepseek_stop = deepseek_decision.get('stop_loss', 0)
        
        if gpt_stop and deepseek_stop:
            stop_diff_pct = 0
            if max(gpt_stop, deepseek_stop) > 0:
                stop_diff_pct = abs(gpt_stop - deepseek_stop) / max(gpt_stop, deepseek_stop)
                
            if stop_diff_pct > 0.1:  # 10% difference threshold
                differences['stop_loss'] = {
                    'gpt': gpt_stop,
                    'deepseek': deepseek_stop,
                    'diff_percentage': round(stop_diff_pct * 100, 2),
                    'severity': 'medium' if stop_diff_pct < 0.3 else 'high'
                }
        
        # Compare take profit
        gpt_tp = gpt_decision.get('take_profit', 0)
        deepseek_tp = deepseek_decision.get('take_profit', 0)
        
        if gpt_tp and deepseek_tp:
            tp_diff_pct = 0
            if max(gpt_tp, deepseek_tp) > 0:
                tp_diff_pct = abs(gpt_tp - deepseek_tp) / max(gpt_tp, deepseek_tp)
                
            if tp_diff_pct > 0.15:  # 15% difference threshold
                differences['take_profit'] = {
                    'gpt': gpt_tp,
                    'deepseek': deepseek_tp,
                    'diff_percentage': round(tp_diff_pct * 100, 2),
                    'severity': 'medium' if tp_diff_pct < 0.4 else 'high'
                }
        
        # Compare timeframe
        if gpt_decision.get('timeframe', '') != deepseek_decision.get('timeframe', ''):
            differences['timeframe'] = {
                'gpt': gpt_decision.get('timeframe', ''),
                'deepseek': deepseek_decision.get('timeframe', ''),
                'severity': 'medium'
            }
        
        # Compare urgency
        gpt_urgency = gpt_decision.get('urgency', 0)
        deepseek_urgency = deepseek_decision.get('urgency', 0)
        
        if abs(gpt_urgency - deepseek_urgency) > 2:  # Scale of 1-10
            differences['urgency'] = {
                'gpt': gpt_urgency,
                'deepseek': deepseek_urgency,
                'diff': abs(gpt_urgency - deepseek_urgency),
                'severity': 'medium' if abs(gpt_urgency - deepseek_urgency) < 5 else 'high'
            }
        
        # Count high severity differences
        high_severity_count = sum(1 for diff in differences.values() if diff.get('severity') == 'high')
        
        # Add analysis summary
        differences['summary'] = {
            'total_differences': len(differences),
            'high_severity_count': high_severity_count,
            'medium_severity_count': sum(1 for diff in differences.values() if diff.get('severity') == 'medium'),
            'overall_severity': 'high' if high_severity_count > 0 else 'medium' if differences else 'low'
        }
        
        return differences
    
    def _calculate_confidence_scores(self, 
                                    gpt_decision: Dict[str, Any], 
                                    deepseek_decision: Dict[str, Any],
                                    differences: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate confidence scores for both AI decisions.
        
        Args:
            gpt_decision (Dict): Decision from GPT-4o
            deepseek_decision (Dict): Decision from DeepSeek
            differences (Dict): Analysis of differences
            
        Returns:
            Dict: Confidence scores for each AI and the final recommendation
        """
        # Base confidence scores
        gpt_base = 0.7  # GPT-4o has slightly higher base confidence
        deepseek_base = 0.6
        
        # Adjust based on thinking mode
        if self.thinking_mode == ThinkingMode.BALANCED:
            # No adjustment needed
            pass
        elif self.thinking_mode == ThinkingMode.CRITICAL:
            # Boost the dissenting opinion
            if differences.get('summary', {}).get('total_differences', 0) > 0:
                deepseek_base += 0.1
        elif self.thinking_mode == ThinkingMode.SUPPORTIVE:
            # Boost the agreeing opinion
            if differences.get('summary', {}).get('total_differences', 0) == 0:
                gpt_base += 0.1
        elif self.thinking_mode == ThinkingMode.ANALYTICAL:
            # Boost based on data references
            gpt_data_refs = len(str(gpt_decision.get('data_references', '')))
            deepseek_data_refs = len(str(deepseek_decision.get('data_references', '')))
            
            if gpt_data_refs > deepseek_data_refs:
                gpt_base += 0.1
            elif deepseek_data_refs > gpt_data_refs:
                deepseek_base += 0.1
        elif self.thinking_mode == ThinkingMode.OVERRIDE_ONLY:
            # Higher threshold for intervention
            gpt_base += 0.15
        
        # Adjust based on decision details
        
        # 1. Confidence boost for explicit confidence values
        gpt_explicit_confidence = gpt_decision.get('confidence', 0)
        deepseek_explicit_confidence = deepseek_decision.get('confidence', 0)
        
        if isinstance(gpt_explicit_confidence, (int, float)) and 0 <= gpt_explicit_confidence <= 1:
            gpt_base = (gpt_base + gpt_explicit_confidence) / 2
        
        if isinstance(deepseek_explicit_confidence, (int, float)) and 0 <= deepseek_explicit_confidence <= 1:
            deepseek_base = (deepseek_base + deepseek_explicit_confidence) / 2
        
        # 2. Reasoning length and quality
        gpt_reason = str(gpt_decision.get('reasoning', ''))
        deepseek_reason = str(deepseek_decision.get('reasoning', ''))
        
        # Simple heuristic: longer reasoning with more data points is better
        # In a real system, you'd use NLP for reasoning quality
        gpt_reason_score = min(0.1, len(gpt_reason) / 5000)
        deepseek_reason_score = min(0.1, len(deepseek_reason) / 5000)
        
        gpt_base += gpt_reason_score
        deepseek_base += deepseek_reason_score
        
        # 3. Penalties for high severity differences
        if differences.get('summary', {}).get('high_severity_count', 0) > 0:
            # Both get a slight penalty when in serious disagreement
            gpt_base -= 0.05
            deepseek_base -= 0.05
        
        # 4. Adjust based on historical accuracy (simplified version)
        # In a real system, you'd use actual historical accuracy data
        gpt_historical_accuracy = 0.75
        deepseek_historical_accuracy = 0.70
        
        gpt_base = (gpt_base * 0.8) + (gpt_historical_accuracy * 0.2)
        deepseek_base = (deepseek_base * 0.8) + (deepseek_historical_accuracy * 0.2)
        
        # Calculate final confidence
        gpt_confidence = round(min(0.95, max(0.3, gpt_base)), 2)
        deepseek_confidence = round(min(0.95, max(0.3, deepseek_base)), 2)
        
        # Calculate overall recommendation
        if gpt_confidence > deepseek_confidence:
            recommended = "gpt"
            recommendation_confidence = gpt_confidence
        else:
            recommended = "deepseek"
            recommendation_confidence = deepseek_confidence
        
        # Calculate agreement level (0.0-1.0)
        if differences.get('summary', {}).get('total_differences', 0) == 0:
            agreement_level = 1.0
        else:
            severity_score = {
                'high': 1.0,
                'medium': 0.5,
                'low': 0.2
            }
            
            total_severity = sum(
                severity_score.get(diff.get('severity', 'low'), 0.2) 
                for diff in differences.values() 
                if isinstance(diff, dict) and 'severity' in diff
            )
            
            agreement_level = max(0.0, 1.0 - (total_severity / 5.0))  # Normalize
            agreement_level = round(agreement_level, 2)
        
        return {
            'gpt': gpt_confidence,
            'deepseek': deepseek_confidence,
            'recommended': recommended,
            'recommendation_confidence': recommendation_confidence,
            'agreement_level': agreement_level
        }
    
    def _determine_outcome(self, 
                          differences: Dict[str, Any], 
                          confidence_scores: Dict[str, float]) -> str:
        """
        Determine the outcome of the decision comparison.
        
        Args:
            differences (Dict): Analysis of differences
            confidence_scores (Dict): Confidence scores
            
        Returns:
            str: Decision outcome
        """
        # Get variables for cleaner code
        agreement_level = confidence_scores.get('agreement_level', 0)
        total_differences = differences.get('summary', {}).get('total_differences', 0)
        high_severity_count = differences.get('summary', {}).get('high_severity_count', 0)
        
        # Determine outcome based on agreement level and severity
        if agreement_level > 0.9:
            return DecisionOutcome.AGREEMENT.value
        elif agreement_level > 0.7:
            return DecisionOutcome.MINOR_DISAGREEMENT.value
        elif agreement_level > 0.4:
            return DecisionOutcome.MAJOR_DISAGREEMENT.value
        else:
            return DecisionOutcome.CRITICAL_CONFLICT.value
    
    def _create_decision_summary(self,
                                gpt_decision: Dict[str, Any],
                                deepseek_decision: Dict[str, Any],
                                differences: Dict[str, Any],
                                confidence_scores: Dict[str, float],
                                outcome: str) -> Dict[str, Any]:
        """
        Create a comprehensive summary of the decision comparison.
        
        Args:
            gpt_decision (Dict): Decision from GPT-4o
            deepseek_decision (Dict): Decision from DeepSeek
            differences (Dict): Analysis of differences
            confidence_scores (Dict): Confidence scores
            outcome (str): Decision outcome
            
        Returns:
            Dict: Decision summary
        """
        # Create a user-friendly explanation
        explanation = self._generate_explanation(
            gpt_decision, 
            deepseek_decision, 
            differences, 
            confidence_scores
        )
        
        # Create a concise summary for terminal display
        terminal_summary = self._generate_terminal_summary(
            gpt_decision,
            deepseek_decision,
            outcome,
            confidence_scores
        )
        
        # Define the recommended action
        recommended_ai = confidence_scores.get('recommended', 'gpt')
        recommended_decision = gpt_decision if recommended_ai == 'gpt' else deepseek_decision
        
        decision_summary = {
            'timestamp': datetime.datetime.now().isoformat(),
            'outcome': outcome,
            'thinking_mode': self.thinking_mode.value,
            'real_mode': self.real_mode,
            'gpt_decision': gpt_decision,
            'deepseek_decision': deepseek_decision,
            'differences': differences,
            'confidence_scores': confidence_scores,
            'explanation': explanation,
            'terminal_summary': terminal_summary,
            'recommended_action': {
                'ai': recommended_ai,
                'confidence': confidence_scores.get('recommendation_confidence', 0),
                'decision': recommended_decision
            }
        }
        
        return decision_summary
    
    def _generate_explanation(self,
                             gpt_decision: Dict[str, Any],
                             deepseek_decision: Dict[str, Any],
                             differences: Dict[str, Any],
                             confidence_scores: Dict[str, float]) -> str:
        """
        Generate a human-readable explanation of the decision comparison.
        
        Args:
            gpt_decision (Dict): Decision from GPT-4o
            deepseek_decision (Dict): Decision from DeepSeek
            differences (Dict): Analysis of differences
            confidence_scores (Dict): Confidence scores
            
        Returns:
            str: Human-readable explanation
        """
        explanation = []
        
        # Basic decision information
        action_str = f"GPT-4o suggests {gpt_decision.get('action', '').upper()} {gpt_decision.get('symbol', '')} "
        action_str += f"({gpt_decision.get('direction', '').upper()}), "
        action_str += f"while DeepSeek suggests {deepseek_decision.get('action', '').upper()} {deepseek_decision.get('symbol', '')} "
        action_str += f"({deepseek_decision.get('direction', '').upper()})."
        explanation.append(action_str)
        
        # Differences
        if differences.get('summary', {}).get('total_differences', 0) > 0:
            explanation.append(f"\nThey disagree on {differences.get('summary', {}).get('total_differences', 0)} aspects:")
            
            for key, diff in differences.items():
                if key != 'summary' and isinstance(diff, dict):
                    if key == 'action':
                        explanation.append(f"- Trading action: GPT-4o suggests {diff.get('gpt', '')}, while DeepSeek suggests {diff.get('deepseek', '')}")
                    elif key == 'symbol':
                        explanation.append(f"- Trading symbol: GPT-4o selected {diff.get('gpt', '')}, while DeepSeek selected {diff.get('deepseek', '')}")
                    elif key == 'direction':
                        explanation.append(f"- Position direction: GPT-4o chose {diff.get('gpt', '')}, while DeepSeek chose {diff.get('deepseek', '')}")
                    elif key == 'size':
                        explanation.append(f"- Position size: GPT-4o recommends {diff.get('gpt', '')}, while DeepSeek recommends {diff.get('deepseek', '')} (difference: {diff.get('diff_percentage', 0)}%)")
                    elif key == 'stop_loss':
                        explanation.append(f"- Stop loss: GPT-4o sets at {diff.get('gpt', '')}, while DeepSeek sets at {diff.get('deepseek', '')} (difference: {diff.get('diff_percentage', 0)}%)")
                    elif key == 'take_profit':
                        explanation.append(f"- Take profit: GPT-4o targets {diff.get('gpt', '')}, while DeepSeek targets {diff.get('deepseek', '')} (difference: {diff.get('diff_percentage', 0)}%)")
                    elif key == 'timeframe':
                        explanation.append(f"- Timeframe: GPT-4o uses {diff.get('gpt', '')}, while DeepSeek uses {diff.get('deepseek', '')}")
                    elif key == 'urgency':
                        explanation.append(f"- Urgency: GPT-4o rates {diff.get('gpt', '')}/10, while DeepSeek rates {diff.get('deepseek', '')}/10 (difference: {diff.get('diff', 0)} points)")
        else:
            explanation.append("\nThey are in complete agreement on all aspects of this decision.")
        
        # Confidence scores
        explanation.append(f"\nConfidence scores:")
        explanation.append(f"- GPT-4o: {confidence_scores.get('gpt', 0) * 100:.1f}%")
        explanation.append(f"- DeepSeek: {confidence_scores.get('deepseek', 0) * 100:.1f}%")
        explanation.append(f"- Agreement level: {confidence_scores.get('agreement_level', 0) * 100:.1f}%")
        
        # Recommendation
        explanation.append(f"\nRecommendation: Follow {confidence_scores.get('recommended', 'GPT-4o').upper()}'s decision with {confidence_scores.get('recommendation_confidence', 0) * 100:.1f}% confidence.")
        
        return "\n".join(explanation)
    
    def _generate_terminal_summary(self,
                                  gpt_decision: Dict[str, Any],
                                  deepseek_decision: Dict[str, Any],
                                  outcome: str,
                                  confidence_scores: Dict[str, float]) -> str:
        """
        Generate a concise summary for terminal display.
        
        Args:
            gpt_decision (Dict): Decision from GPT-4o
            deepseek_decision (Dict): Decision from DeepSeek
            outcome (str): Decision outcome
            confidence_scores (Dict): Confidence scores
            
        Returns:
            str: Concise terminal summary
        """
        # Symbol and basic action
        symbol = gpt_decision.get('symbol', deepseek_decision.get('symbol', 'UNKNOWN'))
        
        # Icons for different outcomes
        icons = {
            DecisionOutcome.AGREEMENT.value: "✅",
            DecisionOutcome.MINOR_DISAGREEMENT.value: "⚠️",
            DecisionOutcome.MAJOR_DISAGREEMENT.value: "🔶",
            DecisionOutcome.CRITICAL_CONFLICT.value: "🔴",
            DecisionOutcome.GPT_OVERRIDE.value: "🧠",
            DecisionOutcome.DEEPSEEK_OVERRIDE.value: "🔍",
            DecisionOutcome.HUMAN_INTERVENTION.value: "👤"
        }
        
        icon = icons.get(outcome, "❓")
        
        # Create the summary based on outcome
        if outcome == DecisionOutcome.AGREEMENT.value:
            summary = f"{icon} [{symbol}] Both AIs agree: {gpt_decision.get('action', '').upper()} {gpt_decision.get('direction', '').upper()}"
            
        elif outcome in [DecisionOutcome.MINOR_DISAGREEMENT.value, DecisionOutcome.MAJOR_DISAGREEMENT.value]:
            gpt_action = f"{gpt_decision.get('action', '').upper()} {gpt_decision.get('direction', '').upper()}"
            deepseek_action = f"{deepseek_decision.get('action', '').upper()} {deepseek_decision.get('direction', '').upper()}"
            
            summary = f"{icon} [{symbol}] Disagreement: GPT: {gpt_action} ({confidence_scores.get('gpt', 0)*100:.0f}%) vs DeepSeek: {deepseek_action} ({confidence_scores.get('deepseek', 0)*100:.0f}%)"
            
        elif outcome == DecisionOutcome.CRITICAL_CONFLICT.value:
            gpt_action = f"{gpt_decision.get('action', '').upper()} {gpt_decision.get('direction', '').upper()}"
            deepseek_action = f"{deepseek_decision.get('action', '').upper()} {deepseek_decision.get('direction', '').upper()}"
            
            summary = f"{icon} [{symbol}] CRITICAL CONFLICT: GPT: {gpt_action} vs DeepSeek: {deepseek_action} - HUMAN REVIEW NEEDED"
            
        elif outcome == DecisionOutcome.GPT_OVERRIDE.value:
            summary = f"{icon} [{symbol}] OVERRIDE: Using GPT-4o decision: {gpt_decision.get('action', '').upper()} {gpt_decision.get('direction', '').upper()}"
            
        elif outcome == DecisionOutcome.DEEPSEEK_OVERRIDE.value:
            summary = f"{icon} [{symbol}] OVERRIDE: Using DeepSeek decision: {deepseek_decision.get('action', '').upper()} {deepseek_decision.get('direction', '').upper()}"
            
        else:
            summary = f"{icon} [{symbol}] Undetermined outcome - human intervention required"
        
        # Add recommended action
        if confidence_scores.get('recommended'):
            recommended = confidence_scores.get('recommended', '').upper()
            conf = confidence_scores.get('recommendation_confidence', 0) * 100
            summary += f" → Recommendation: Follow {recommended} ({conf:.0f}%)"
        
        return summary
    
    def _save_to_memory(self, decision_summary: Dict[str, Any]) -> bool:
        """
        Save decision summary to memory core.
        
        Args:
            decision_summary (Dict): Decision summary
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not MEMORY_CORE_AVAILABLE or not self.memory:
            logger.warning("Memory core not available, skipping memory storage")
            return False
        
        try:
            # Store the decision in memory
            self.memory.store(
                data=decision_summary,
                category="ai_decisions",
                subcategory=decision_summary.get('outcome', 'unknown'),
                source="thinking_controller",
                real_mode=self.real_mode
            )
            
            logger.info(f"Decision stored in memory: {decision_summary.get('terminal_summary', '')}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing decision in memory: {str(e)}")
            return False
    
    def _log_decision(self, decision_summary: Dict[str, Any]) -> bool:
        """
        Log decision to the decision log file.
        
        Args:
            decision_summary (Dict): Decision summary
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load existing decisions
            try:
                with open(LOG_FILE, 'r') as f:
                    decisions = json.load(f)
                    if not isinstance(decisions, list):
                        decisions = []
            except (json.JSONDecodeError, FileNotFoundError):
                decisions = []
            
            # Add new decision
            decisions.append(decision_summary)
            
            # Save updated decisions
            with open(LOG_FILE, 'w') as f:
                json.dump(decisions, f, indent=2)
            
            logger.debug(f"Logged decision: {decision_summary.get('terminal_summary', '')}")
            return True
            
        except Exception as e:
            logger.error(f"Error logging decision: {str(e)}")
            return False
    
    def _alert_user(self, decision_summary: Dict[str, Any]) -> None:
        """
        Alert the user about a decision conflict.
        
        Args:
            decision_summary (Dict): Decision summary
        """
        # Skip if in silent mode
        if self.silent_mode:
            return
        
        # Get the terminal summary
        summary = decision_summary.get('terminal_summary', 'Decision conflict detected')
        
        # Alert based on outcome
        outcome = decision_summary.get('outcome', '')
        
        if outcome == DecisionOutcome.MAJOR_DISAGREEMENT.value:
            if ALERT_SYSTEM_AVAILABLE and hasattr(self.alert, 'warning'):
                self.alert.warning(summary, module="thinking_controller", category="decision_conflict")
            else:
                logger.warning(summary)
                
            self._print_to_terminal(summary)
            
        elif outcome == DecisionOutcome.CRITICAL_CONFLICT.value:
            if ALERT_SYSTEM_AVAILABLE and hasattr(self.alert, 'error'):
                self.alert.error(summary, module="thinking_controller", category="critical_conflict")
            else:
                logger.error(summary)
                
            self._print_to_terminal(f"🚨 {summary}")
            
            # In critical conflicts, also print detailed explanation
            explanation = decision_summary.get('explanation', '')
            self._print_to_terminal(f"\nDetailed explanation:\n{explanation}")
    
    def _print_to_terminal(self, message: str) -> None:
        """
        Print a message to the terminal.
        
        Args:
            message (str): Message to print
        """
        if self.silent_mode:
            return
            
        if self.terminal and hasattr(self.terminal, 'print'):
            self.terminal.print(message, module="thinking_controller")
        else:
            print(message)
    
    def _background_monitor(self) -> None:
        """Background monitoring loop for processing decisions."""
        logger.info("Started background decision monitoring")
        
        while self.is_running:
            try:
                # Process any decisions in the queue
                while not self.decision_queue.empty():
                    decision_pair = self.decision_queue.get_nowait()
                    
                    if isinstance(decision_pair, dict) and 'gpt' in decision_pair and 'deepseek' in decision_pair:
                        self.compare_decisions(decision_pair['gpt'], decision_pair['deepseek'])
                    
                    self.decision_queue.task_done()
                
                # Sleep to avoid high CPU usage
                time.sleep(MONITOR_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error in background monitor: {str(e)}")
                time.sleep(MONITOR_INTERVAL)
    
    def add_decision_pair(self, gpt_decision: Dict[str, Any], deepseek_decision: Dict[str, Any]) -> bool:
        """
        Add a decision pair to the comparison queue.
        
        Args:
            gpt_decision (Dict): Decision from GPT-4o
            deepseek_decision (Dict): Decision from DeepSeek
            
        Returns:
            bool: True if successfully added to queue
        """
        try:
            self.decision_queue.put({
                'gpt': gpt_decision,
                'deepseek': deepseek_decision
            })
            
            logger.info(f"Added decision pair to queue: {gpt_decision.get('action', '')} vs {deepseek_decision.get('action', '')}")
            return True
        except Exception as e:
            logger.error(f"Error adding decision pair to queue: {str(e)}")
            return False
    
    def handle_user_intervention(self, command: str) -> Dict[str, Any]:
        """
        Handle user intervention commands.
        
        Args:
            command (str): User command
            
        Returns:
            Dict: Response to user command
        """
        logger.info(f"Processing user intervention: {command}")
        
        command = command.strip().lower()
        
        # Check for GPT override command
        if "force gpt" in command or "use gpt" in command or "gpt override" in command:
            # In a real implementation, this would force GPT's decision
            response = {
                'success': True,
                'message': "GPT-4o decision will be used for the current operation.",
                'action': DecisionOutcome.GPT_OVERRIDE.value
            }
            
            # Log the human intervention
            self._log_human_intervention(command, response)
            
            return response
        
        # Check for DeepSeek override command
        elif "force deepseek" in command or "use deepseek" in command or "deepseek override" in command:
            # In a real implementation, this would force DeepSeek's decision
            response = {
                'success': True,
                'message': "DeepSeek decision will be used for the current operation.",
                'action': DecisionOutcome.DEEPSEEK_OVERRIDE.value
            }
            
            # Log the human intervention
            self._log_human_intervention(command, response)
            
            return response
        
        # Check for mode change commands
        elif "set mode" in command or "change mode" in command:
            for mode in ThinkingMode:
                if mode.value in command:
                    self.set_thinking_mode(mode)
                    return {
                        'success': True,
                        'message': f"Thinking mode changed to: {mode.value}",
                        'action': 'mode_change',
                        'mode': mode.value
                    }
            
            # Mode not recognized
            return {
                'success': False,
                'message': "Thinking mode not recognized. Available modes: balanced, critical, supportive, analytical, override_only",
                'action': 'error'
            }
        
        # Unknown command
        return {
            'success': False,
            'message': "Command not recognized. Try 'force gpt', 'force deepseek', or 'set mode balanced'",
            'action': 'error'
        }
    
    def _log_human_intervention(self, command: str, response: Dict[str, Any]) -> None:
        """
        Log human intervention to memory and logs.
        
        Args:
            command (str): User command
            response (Dict): Response to user command
        """
        intervention = {
            'timestamp': datetime.datetime.now().isoformat(),
            'command': command,
            'response': response,
            'outcome': DecisionOutcome.HUMAN_INTERVENTION.value
        }
        
        # Log to file
        self._log_decision(intervention)
        
        # Save to memory
        if MEMORY_CORE_AVAILABLE and self.memory:
            self.memory.store(
                data=intervention,
                category="human_interventions",
                subcategory=response.get('action', 'unknown'),
                source="thinking_controller",
                real_mode=self.real_mode
            )


# Test scenario function
def run_test_scenario(use_real_mode: bool = False, silent_mode: bool = False):
    """
    Run a test scenario to demonstrate the thinking controller.
    
    Args:
        use_real_mode (bool): Whether to use real mode
        silent_mode (bool): Whether to use silent mode
    """
    print("\n" + "=" * 70)
    print("  THINKING CONTROLLER TEST SCENARIO")
    print("=" * 70)
    print(f"  Mode: {'REAL' if use_real_mode else 'SIMULATION'}, Silent: {'Yes' if silent_mode else 'No'}\n")
    
    # Initialize the thinking controller
    controller = ThinkingController(
        real_mode=use_real_mode,
        silent_mode=silent_mode,
        thinking_mode=ThinkingMode.BALANCED
    )
    
    # Start monitoring
    controller.start_monitoring()
    
    # Sample decisions
    gpt_decision = {
        'action': 'buy',
        'symbol': 'ETH',
        'direction': 'long',
        'size': 0.5,  # 50% of available capital
        'stop_loss': 1820.50,
        'take_profit': 2150.75,
        'timeframe': '4h',
        'urgency': 7,
        'type': DecisionType.ENTRY.value,
        'reasoning': "ETH is showing a strong bullish pattern on the 4h chart with increasing volume. The price has broken above the 50 EMA and is likely to continue upward. RSI is at 65, indicating momentum without being overbought. This looks like a solid opportunity for a swing trade with a good risk-reward ratio of 1:3.",
        'confidence': 0.78,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    deepseek_decision = {
        'action': 'buy',
        'symbol': 'ETH',
        'direction': 'sniper',  # Different approach - quick in and out
        'size': 0.3,  # 30% of available capital - more conservative
        'stop_loss': 1805.25,  # Tighter stop loss
        'take_profit': 2050.00,  # Lower but quicker profit target
        'timeframe': '1h',  # Shorter timeframe
        'urgency': 9,  # Higher urgency
        'type': DecisionType.ENTRY.value,
        'reasoning': "ETH shows a short-term reversal opportunity based on the 1h chart. Significant whale buying detected in the last 30 minutes, with $25M inflows to exchanges. Volume spike of 3x average detected. This is likely a short-term opportunity requiring quick action. Recent news about ETH network upgrades is bullish. Recommend a 'sniper' entry with quick take profit.",
        'confidence': 0.82,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    print("Sample Decision Scenario: ETH Trading Strategy")
    print("---------------------------------------------")
    print("GPT-4o suggests: Standard LONG position on ETH")
    print("DeepSeek suggests: SNIPER approach for ETH")
    print("\nComparing decisions...")
    
    # Compare the decisions
    result = controller.compare_decisions(gpt_decision, deepseek_decision)
    
    print("\nThinking Controller Analysis Complete")
    print("-------------------------------------")
    print(f"Outcome: {result.get('outcome', 'Unknown')}")
    print(f"Terminal Summary: {result.get('terminal_summary', 'None')}")
    print("\nDetailed Explanation:")
    print(result.get('explanation', 'No explanation available'))
    
    # Show recommended action
    recommended = result.get('recommended_action', {})
    print("\nRecommended Action:")
    print(f"AI: {recommended.get('ai', 'unknown').upper()}")
    print(f"Confidence: {recommended.get('confidence', 0) * 100:.1f}%")
    
    # Test user intervention
    print("\nTesting User Intervention...")
    intervention = controller.handle_user_intervention("force deepseek override")
    print(f"User Command: 'force deepseek override'")
    print(f"Response: {intervention.get('message', 'No response')}")
    
    # Stop monitoring
    controller.stop_monitoring()
    
    print("\nTest scenario completed!")
    print("=" * 70 + "\n")


# Main function
def main():
    """Main function to run the thinking controller standalone."""
    import argparse
    parser = argparse.ArgumentParser(description='Thinking Controller for SentientTrader.AI')
    parser.add_argument('--real', action='store_true', help='Run in real mode (aware of real money)')
    parser.add_argument('--silent', action='store_true', help='Run in silent mode (minimal output)')
    parser.add_argument('--test', action='store_true', help='Run test scenario')
    args = parser.parse_args()
    
    if args.test:
        run_test_scenario(use_real_mode=args.real, silent_mode=args.silent)
        return
    
    # Initialize the thinking controller
    controller = ThinkingController(
        real_mode=args.real,
        silent_mode=args.silent
    )
    
    # Start monitoring
    controller.start_monitoring()
    
    print("\n" + "=" * 60)
    print("  Thinking Controller for SentientTrader.AI")
    print("=" * 60)
    print("  Type 'exit' or 'quit' to exit")
    print("  Type 'help' for available commands")
    print("=" * 60 + "\n")
    
    try:
        while True:
            user_input = input("\n🧠 > ")
            
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting Thinking Controller...")
                break
            
            if user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("  force gpt - Override with GPT-4o's decision")
                print("  force deepseek - Override with DeepSeek's decision")
                print("  set mode [balanced|critical|supportive|analytical|override_only]")
                print("  help - Show this help message")
                print("  exit/quit - Exit the program")
                continue
            
            if user_input.strip():
                response = controller.handle_user_intervention(user_input)
                print(f"Response: {response.get('message', 'No response')}")
    
    except KeyboardInterrupt:
        print("\nExiting Thinking Controller...")
    
    finally:
        # Stop monitoring
        controller.stop_monitoring()


if __name__ == "__main__":
    main()
