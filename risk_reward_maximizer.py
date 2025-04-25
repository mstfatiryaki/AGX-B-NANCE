#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SentientTrader.AI Risk Reward Maximizer
Optimizes risk/reward allocations for entry points
"""

import os
import sys
import json
import time
import logging
import argparse
import random
from datetime import datetime

# Module setup
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def optimize_risk_allocations(entry_points, total_capital=10000, max_risk_per_trade=0.02):
    """
    Optimize risk/reward allocations for entry points
    
    Args:
        entry_points (list): Entry points
        total_capital (float): Total capital available
        max_risk_per_trade (float): Maximum risk per trade (percent)
        
    Returns:
        list: Risk allocations
    """
    allocations = []
    
    # Total number of valid entry points
    valid_entries = [e for e in entry_points if e.get("confidence", 0) > 0.5]
    
    if not valid_entries:
        return allocations
    
    # Sort by confidence (highest first)
    valid_entries.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    
    # Calculate allocations
    total_allocated = 0
    
    for entry in valid_entries:
        symbol = entry.get("symbol", "")
        signal = entry.get("signal", "")
        entry_price = entry.get("entry_price", 0)
        stop_loss = entry.get("stop_loss", 0)
        
        if entry_price <= 0 or stop_loss <= 0:
            continue
        
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss)
        
        # Amount to risk (adjusted by confidence)
        confidence = entry.get("confidence", 0.7)
        risk_amount = total_capital * max_risk_per_trade * confidence
        
        # Calculate position size
        position_size = risk_amount / risk_per_unit
        
        # Calculate potential return (R multiple)
        take_profit = entry.get("take_profit", entry_price)
        potential_profit = abs(take_profit - entry_price) * position_size
        potential_loss = risk_amount
        
        # Expected value
        win_probability = confidence
        expected_value = (win_probability * potential_profit) - ((1 - win_probability) * potential_loss)
        expected_value_r = expected_value / potential_loss if potential_loss > 0 else 0
        
        # Add allocation
        allocations.append({
            "symbol": symbol,
            "signal": signal,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "position_size": position_size,
            "capital_allocated": position_size * entry_price,
            "risk_amount": risk_amount,
            "risk_percentage": max_risk_per_trade * 100 * confidence,
            "expected_value": expected_value_r,
            "confidence": confidence,
            "timeframe": entry.get("timeframe", "1h")
        })
        
        total_allocated += position_size * entry_price
        
        # Stop if we've allocated too much capital
        if total_allocated > total_capital * 0.8:
            break
    
    return allocations

# Main execution
if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Risk Reward Maximizer")
    parser.add_argument("--silent", action="store_true", help="Run in silent mode")
    parser.add_argument("--real", action="store_true", help="Run in real mode")
    parser.add_argument("--capital", type=float, default=10000, help="Total capital")
    parser.add_argument("--max-risk", type=float, default=0.02, help="Maximum risk per trade (0.02 = 2%)")
    args = parser.parse_args()
    
    # Paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    data_dir = os.path.join(project_root, "data")
    entry_points_file = os.path.join(data_dir, "entry_points.json")
    risk_allocation_file = os.path.join(data_dir, "risk_allocation.json")
    
    # Ensure data directory exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    try:
        # Check if entry points data exists
        if not os.path.exists(entry_points_file):
            logger.error(f"Entry points file not found at {entry_points_file}")
            print(f"Error: Entry points file not found. Run the Sniper Entry System first.")
            
            # Create dummy entry points
            entry_points = [{
                "symbol": "BTC",
                "signal": "BUY",
                "entry_price": 69000,
                "stop_loss": 67500,
                "take_profit": 72000,
                "confidence": 0.85,
                "timeframe": "1h"
            }, {
                "symbol": "ETH",
                "signal": "BUY",
                "entry_price": 4200,
                "stop_loss": 4050,
                "take_profit": 4500,
                "confidence": 0.8,
                "timeframe": "4h"
            }]
        else:
            # Load entry points data
            with open(entry_points_file, 'r') as f:
                entry_points_data = json.load(f)
            
            entry_points = entry_points_data.get("entries", [])
        
        if not entry_points:
            logger.warning("No entry points found in the file")
            print("Warning: No entry points found. Creating sample entries.")
            entry_points = [{
                "symbol": "BTC",
                "signal": "BUY",
                "entry_price": 69000,
                "stop_loss": 67500,
                "take_profit": 72000,
                "confidence": 0.85,
                "timeframe": "1h"
            }, {
                "symbol": "ETH",
                "signal": "BUY",
                "entry_price": 4200,
                "stop_loss": 4050,
                "take_profit": 4500,
                "confidence": 0.8,
                "timeframe": "4h"
            }]
        
        # Optimize risk allocations
        logger.info("Optimizing risk/reward allocations")
        allocations = optimize_risk_allocations(entry_points, args.capital, args.max_risk)
        
        # Save risk allocations
        output_data = {
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "risk_reward_maximizer",
            "allocations": allocations,
            "total_allocations": len(allocations),
            "total_capital": args.capital,
            "max_risk_per_trade": args.max_risk
        }
        
        with open(risk_allocation_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        # Log and print results
        total_allocated = sum(a.get("capital_allocated", 0) for a in allocations)
        logger.info(f"Created {len(allocations)} risk allocations, total capital: {total_allocated:.2f}")
        
        print(f"Risk Reward Maximizer completed successfully!")
        print(f"Created {len(allocations)} optimized risk allocations")
        print(f"Total capital allocated: ${total_allocated:.2f} / ${args.capital:.2f}")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Error in Risk Reward Maximizer: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)
