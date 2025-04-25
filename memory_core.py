from random import choice
from random import choice
import random
import random
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SentientTrader.AI - Memory Core V2
----------------------------------
This module provides the memory management and retention capabilities
for SentientTrader.AI, allowing the system to store, query and learn
from historical decisions, market data, and trading outcomes.
"""

import os
import sys
import json
import time
import uuid
import logging
import datetime
import argparse
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
from threading import Lock
from collections import defaultdict
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("memory_core.log")]
)
logger = logging.getLogger("MemoryCore")

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

# Constants and configuration
REAL_MODE = False  # Set to True for real mode, False for simulation
CURRENT_TIME = "2025-04-21 19:36:02"  # UTC
CURRENT_USER = "mstfatiryaki"

# File paths
MEMORY_STORE_FILE = "memory_store.json"
MEMORY_BACKUP_DIR = "memory_backups"

# Memory categories
MEMORY_CATEGORIES = [
    "strategy_decision",  # Strategy decisions
    "trade_execution",    # Trade execution details
    "market_data",        # Market data snapshots
    "sentiment_analysis", # Sentiment analysis results
    "performance_metric", # Performance metrics
    "system_event",       # System events
    "user_input",         # User inputs/actions
    "external_signal",    # External signals
    "learning_outcome"    # Learning outcomes
]

# Memory sources
MEMORY_SOURCES = [
    "strategy_engine",
    "trade_executor",
    "market_analyzer",
    "sentiment_analyzer",
    "performance_tracker",
    "risk_manager",
    "learning_engine",
    "capital_manager",
    "transaction_logger",
    "user_interface",
    "external_api"
]

# Time frames for summarization
TIME_FRAMES = {
    "1h": {"unit": "hours", "value": 1},
    "4h": {"unit": "hours", "value": 4},
    "12h": {"unit": "hours", "value": 12},
    "1d": {"unit": "days", "value": 1},
    "3d": {"unit": "days", "value": 3},
    "1w": {"unit": "weeks", "value": 1},
    "2w": {"unit": "weeks", "value": 2},
    "1m": {"unit": "months", "value": 1},
    "3m": {"unit": "months", "value": 3},
    "6m": {"unit": "months", "value": 6},
    "1y": {"unit": "years", "value": 1}
}

class MemoryCore:
    """Memory management system for SentientTrader.AI"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, real_mode: bool = REAL_MODE):
        """Singleton pattern to ensure only one MemoryCore instance"""
        if cls._instance is None:
            cls._instance = super(MemoryCore, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, real_mode: bool = REAL_MODE):
        """
        Initialize the memory core
        
        Args:
            real_mode (bool): Whether running in real or simulation mode
        """
        # Only initialize once due to singleton pattern
        if MemoryCore._initialized:
            return
            
        self.real_mode = real_mode
        self.memory_data = {
            "metadata": {
                "version": "2.0",
                "last_update": CURRENT_TIME,
                "created_at": CURRENT_TIME,
                "user": CURRENT_USER,
                "real_mode": real_mode,
                "records_count": 0
            },
            "records": []
        }
        
        # Thread safety
        self.memory_lock = Lock()
        self.file_lock = Lock()
        
        # Cache for fast access
        self.record_index = {}  # id -> record index
        self.tag_index = defaultdict(list)  # tag -> [record indices]
        self.category_index = defaultdict(list)  # category -> [record indices]
        self.source_index = defaultdict(list)  # source -> [record indices]
        self.time_index = {}  # timestamp -> record index
        
        # Ensure backup directory exists
        os.makedirs(MEMORY_BACKUP_DIR, exist_ok=True)
        
        # Load existing memory
        self.load_memory()
        
        MemoryCore._initialized = True
        logger.info(f"Memory Core initialized (REAL_MODE: {self.real_mode})")
    
    def add_memory_record(self, source: str, category: str, data: Dict[str, Any], 
                         tags: List[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Add a new memory record
        
        Args:
            source (str): Source of the memory
            category (str): Category of the memory
            data (Dict[str, Any]): Memory data
            tags (List[str], optional): Tags for the memory
            **kwargs: Additional fields for the record
            
        Returns:
            Dict[str, Any]: Added record or error
        """
        try:
            # Validate inputs
            if not self._validate_input(source, category, data):
                error_msg = "Invalid input data for memory record"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
            
            # Format tags
            tags = tags or []
            # Add source and category as implicit tags if not already present
            if source not in tags:
                tags.append(source)
            if category not in tags:
                tags.append(category)
                
            # Normalize tags: lowercase, replace spaces with underscores
            tags = [self._normalize_tag(tag) for tag in tags]
            
            # Create record
            record = {
                "id": kwargs.get("id", f"mem_{int(time.time())}_{uuid.uuid4().hex[:8]}"),
                "timestamp": kwargs.get("timestamp", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                "source": source,
                "category": category,
                "tags": tags,
                "data": data,
                "real_mode": kwargs.get("real_mode", self.real_mode)
            }
            
            # Add custom fields from kwargs
            for key, value in kwargs.items():
                if key not in record and key not in ["id", "timestamp", "real_mode"]:
                    record[key] = value
            
            # Thread-safe addition to memory
            with self.memory_lock:
                # Add record
                self.memory_data["records"].append(record)
                
                # Update record count
                self.memory_data["metadata"]["records_count"] = len(self.memory_data["records"])
                self.memory_data["metadata"]["last_update"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Update indices
                record_index = len(self.memory_data["records"]) - 1
                self.record_index[record["id"]] = record_index
                
                # Update tag index
                for tag in tags:
                    self.tag_index[tag].append(record_index)
                
                # Update category and source indices
                self.category_index[category].append(record_index)
                self.source_index[source].append(record_index)
                
                # Update time index
                self.time_index[record["timestamp"]] = record_index
            
            # Save memory (optional: could be scheduled to save periodically instead)
            self.save_memory()
            
            logger.info(f"Added memory record: {record['id']} from {source} in category {category}")
            return {"success": True, "record": record}
            
        except Exception as e:
            error_msg = f"Error adding memory record: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def query_memory(self, category: Optional[str] = None, source: Optional[str] = None, 
                    tags: List[str] = None, time_range: Optional[Tuple[str, str]] = None,
                    coin: Optional[str] = None, strategy: Optional[str] = None,
                    limit: int = 100, custom_filter: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """
        Query memory records with flexible filters
        
        Args:
            category (Optional[str]): Filter by category
            source (Optional[str]): Filter by source
            tags (List[str], optional): Filter by tags (AND logic)
            time_range (Optional[Tuple[str, str]]): Filter by time range (start, end)
            coin (Optional[str]): Filter by coin
            strategy (Optional[str]): Filter by strategy
            limit (int): Limit number of results
            custom_filter (Optional[Callable]): Custom filter function
            
        Returns:
            List[Dict[str, Any]]: Matching records
        """
        try:
            tags = tags or []
            
            # Normalize search parameters
            if coin:
                tags.append(coin.upper())
            if strategy:
                tags.append(self._normalize_tag(strategy))
            
            # Thread-safe read
            with self.memory_lock:
                # Start with all record indices
                indices = set(range(len(self.memory_data["records"])))
                
                # Filter by category
                if category:
                    category_indices = set(self.category_index.get(category, []))
                    indices = indices.intersection(category_indices)
                
                # Filter by source
                if source:
                    source_indices = set(self.source_index.get(source, []))
                    indices = indices.intersection(source_indices)
                
                # Filter by tags (AND logic - must have all tags)
                for tag in tags:
                    normalized_tag = self._normalize_tag(tag)
                    tag_indices = set(self.tag_index.get(normalized_tag, []))
                    indices = indices.intersection(tag_indices)
                
                # Filter by time range
                if time_range:
                    start_time, end_time = time_range
                    
                    # Convert to datetime objects for comparison
                    try:
                        start_dt = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S") if start_time else None
                        end_dt = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S") if end_time else None
                        
                        # Filter records by timestamp
                        time_indices = set()
                        for idx in indices:
                            record = self.memory_data["records"][idx]
                            record_dt = datetime.datetime.strptime(record["timestamp"], "%Y-%m-%d %H:%M:%S")
                            
                            if start_dt and end_dt:
                                if start_dt <= record_dt <= end_dt:
                                    time_indices.add(idx)
                            elif start_dt:
                                if start_dt <= record_dt:
                                    time_indices.add(idx)
                            elif end_dt:
                                if record_dt <= end_dt:
                                    time_indices.add(idx)
                        
                        indices = time_indices
                    except Exception as e:
                        logger.error(f"Error parsing time range: {e}")
                
                # Get records from indices
                records = [self.memory_data["records"][idx] for idx in indices]
                
                # Apply custom filter if provided
                if custom_filter and callable(custom_filter):
                    records = [record for record in records if custom_filter(record)]
                
                # Sort by timestamp (newest first)
                records.sort(key=lambda x: x["timestamp"], reverse=True)
                
                # Apply limit
                if limit > 0:
                    records = records[:limit]
                
                return records
                
        except Exception as e:
            logger.error(f"Error querying memory: {str(e)}")
            return []
    
    def summarize_memory(self, timeframe: str = "1d", category: Optional[str] = None,
                        source: Optional[str] = None, tags: List[str] = None) -> Dict[str, Any]:
        """
        Create a summary of memory records for a given timeframe
        
        Args:
            timeframe (str): Timeframe for summarization (e.g. "1h", "1d", "1w")
            category (Optional[str]): Filter by category
            source (Optional[str]): Filter by source
            tags (List[str], optional): Filter by tags
            
        Returns:
            Dict[str, Any]: Memory summary
        """
        try:
            # Validate timeframe
            if timeframe not in TIME_FRAMES:
                logger.error(f"Invalid timeframe: {timeframe}")
                return {"success": False, "error": f"Invalid timeframe: {timeframe}"}
            
            # Calculate time range
            end_time = datetime.datetime.now()
            time_unit = TIME_FRAMES[timeframe]["unit"]
            time_value = TIME_FRAMES[timeframe]["value"]
            
            if time_unit == "hours":
                start_time = end_time - datetime.timedelta(hours=time_value)
            elif time_unit == "days":
                start_time = end_time - datetime.timedelta(days=time_value)
            elif time_unit == "weeks":
                start_time = end_time - datetime.timedelta(weeks=time_value)
            elif time_unit == "months":
                start_time = end_time - datetime.timedelta(days=time_value * 30)  # Approximation
            elif time_unit == "years":
                start_time = end_time - datetime.timedelta(days=time_value * 365)  # Approximation
            else:
                start_time = end_time - datetime.timedelta(days=1)  # Default to 1 day
            
            # Format time range
            time_range = (
                start_time.strftime("%Y-%m-%d %H:%M:%S"),
                end_time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
            # Query records
            records = self.query_memory(
                category=category,
                source=source,
                tags=tags,
                time_range=time_range,
                limit=0  # No limit for summary
            )
            
            if not records:
                logger.info(f"No records found for timeframe {timeframe}")
                return {
                    "success": True,
                    "timeframe": timeframe,
                    "start_time": time_range[0],
                    "end_time": time_range[1],
                    "record_count": 0,
                    "message": "No records found in specified timeframe"
                }
            
            # Initialize summary
            summary = {
                "success": True,
                "timeframe": timeframe,
                "start_time": time_range[0],
                "end_time": time_range[1],
                "record_count": len(records),
                "categories": {},
                "sources": {},
                "tags": {},
                "time_distribution": {
                    "hourly": defaultdict(int),
                    "daily": defaultdict(int)
                }
            }
            
            # Process records
            for record in records:
                # Count by category
                record_category = record.get("category", "unknown")
                if record_category not in summary["categories"]:
                    summary["categories"][record_category] = 0
                summary["categories"][record_category] += 1
                
                # Count by source
                record_source = record.get("source", "unknown")
                if record_source not in summary["sources"]:
                    summary["sources"][record_source] = 0
                summary["sources"][record_source] += 1
                
                # Count by tags
                for tag in record.get("tags", []):
                    if tag not in summary["tags"]:
                        summary["tags"][tag] = 0
                    summary["tags"][tag] += 1
                
                # Time distribution
                try:
                    timestamp = record.get("timestamp", "")
                    if timestamp:
                        dt = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                        
                        # Hourly distribution
                        hour = dt.hour
                        summary["time_distribution"]["hourly"][hour] += 1
                        
                        # Daily distribution
                        day = dt.strftime("%Y-%m-%d")
                        summary["time_distribution"]["daily"][day] += 1
                except:
                    pass
            
            # Sort tags by frequency
            summary["top_tags"] = sorted(
                [{"tag": tag, "count": count} for tag, count in summary["tags"].items()],
                key=lambda x: x["count"],
                reverse=True
            )[:10]  # Top 10 tags
            
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing memory: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def delete_memory(self, condition_func: Callable) -> Dict[str, Any]:
        """
        Delete memory records matching a condition
        
        Args:
            condition_func (Callable): Function that takes a record and returns True if it should be deleted
            
        Returns:
            Dict[str, Any]: Deletion result
        """
        try:
            if not callable(condition_func):
                logger.error("Invalid condition function for memory deletion")
                return {"success": False, "error": "Invalid condition function"}
            
            deleted_records = []
            
            # Thread-safe modification
            with self.memory_lock:
                # Find records to delete
                to_delete = []
                for idx, record in enumerate(self.memory_data["records"]):
                    if condition_func(record):
                        to_delete.append(idx)
                        deleted_records.append(record)
                
                if not to_delete:
                    logger.info("No records matched deletion condition")
                    return {"success": True, "deleted_count": 0, "message": "No records matched deletion condition"}
                
                # Delete in reverse order to maintain correct indices
                for idx in sorted(to_delete, reverse=True):
                    deleted_record = self.memory_data["records"].pop(idx)
                    
                    # Remove from indices
                    record_id = deleted_record["id"]
                    if record_id in self.record_index:
                        del self.record_index[record_id]
                    
                    # Remove from tag index
                    for tag in deleted_record.get("tags", []):
                        if tag in self.tag_index and idx in self.tag_index[tag]:
                            self.tag_index[tag].remove(idx)
                    
                    # Remove from category index
                    category = deleted_record.get("category", "")
                    if category in self.category_index and idx in self.category_index[category]:
                        self.category_index[category].remove(idx)
                    
                    # Remove from source index
                    source = deleted_record.get("source", "")
                    if source in self.source_index and idx in self.source_index[source]:
                        self.source_index[source].remove(idx)
                    
                    # Remove from time index
                    timestamp = deleted_record.get("timestamp", "")
                    if timestamp in self.time_index and self.time_index[timestamp] == idx:
                        del self.time_index[timestamp]
                
                # Rebuild indices after deletion
                self._rebuild_indices()
                
                # Update metadata
                self.memory_data["metadata"]["records_count"] = len(self.memory_data["records"])
                self.memory_data["metadata"]["last_update"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Save changes
            self.save_memory()
            
            logger.info(f"Deleted {len(deleted_records)} memory records")
            return {
                "success": True,
                "deleted_count": len(deleted_records),
                "deleted_records": deleted_records
            }
            
        except Exception as e:
            logger.error(f"Error deleting memory records: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def load_external(self, file_path: str) -> Dict[str, Any]:
        """
        Load records from external file
        
        Args:
            file_path (str): Path to external file
            
        Returns:
            Dict[str, Any]: Load result
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"External file not found: {file_path}")
                return {"success": False, "error": f"File not found: {file_path}"}
            
            with open(file_path, "r", encoding="utf-8") as f:
                external_data = json.load(f)
            
            # Check if valid format (records array or single record)
            records_to_add = []
            
            if isinstance(external_data, dict):
                if "records" in external_data and isinstance(external_data["records"], list):
                    # Memory store format
                    records_to_add = external_data["records"]
                elif "source" in external_data and "category" in external_data:
                    # Single record format
                    records_to_add = [external_data]
            elif isinstance(external_data, list):
                # Array of records
                records_to_add = external_data
            
            if not records_to_add:
                logger.error(f"Invalid data format in {file_path}")
                return {"success": False, "error": f"Invalid data format in {file_path}"}
            
            # Add each record
            added_count = 0
            for record in records_to_add:
                if not isinstance(record, dict) or "source" not in record or "category" not in record:
                    logger.warning(f"Skipping invalid record: {record}")
                    continue
                
                # Extract fields
                source = record.get("source", "external")
                category = record.get("category", "unknown")
                data = record.get("data", {})
                tags = record.get("tags", [])
                
                # Add record
                result = self.add_memory_record(
                    source=source,
                    category=category,
                    data=data,
                    tags=tags,
                    timestamp=record.get("timestamp"),
                    id=record.get("id")
                )
                
                if result["success"]:
                    added_count += 1
            
            logger.info(f"Loaded {added_count} records from {file_path}")
            return {
                "success": True,
                "added_count": added_count,
                "total_records": len(records_to_add)
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path}: {e}")
            return {"success": False, "error": f"Invalid JSON: {e}"}
        except Exception as e:
            logger.error(f"Error loading external file: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def save_memory(self) -> bool:
        """
        Save memory to file
        
        Returns:
            bool: Success status
        """
        try:
            with self.file_lock:
                # Create backup of existing file
                if os.path.exists(MEMORY_STORE_FILE):
                    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                    backup_file = os.path.join(MEMORY_BACKUP_DIR, f"memory_store_backup_{timestamp}.json")
                    
                    try:
                        import shutil
                        shutil.copy2(MEMORY_STORE_FILE, backup_file)
                        logger.info(f"Created backup at {backup_file}")
                    except Exception as e:
                        logger.warning(f"Failed to create backup: {e}")
                
                # Update metadata
                with self.memory_lock:
                    self.memory_data["metadata"]["last_update"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Save to file
                    with open(MEMORY_STORE_FILE, "w", encoding="utf-8") as f:
                        json.dump(self.memory_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Memory saved to {MEMORY_STORE_FILE}")
                return True
                
        except Exception as e:
            logger.error(f"Error saving memory: {str(e)}")
            
            # Send critical alert
            if ALERT_SYSTEM_AVAILABLE:
                alert_system.critical(
                    "Failed to save memory store",
                    {"error": str(e), "records_count": self.memory_data["metadata"]["records_count"]},
                    "memory_core"
                )
            
            return False
    
    def load_memory(self) -> bool:
        """
        Load memory from file
        
        Returns:
            bool: Success status
        """
        try:
            if not os.path.exists(MEMORY_STORE_FILE):
                logger.info(f"{MEMORY_STORE_FILE} not found, initializing new memory store")
                return True
            
            with self.file_lock:
                with open(MEMORY_STORE_FILE, "r", encoding="utf-8") as f:
                    loaded_data = json.load(f)
                
                # Validate format
                if not self._validate_memory_format(loaded_data):
                    logger.error(f"Invalid memory format in {MEMORY_STORE_FILE}")
                    
                    # Try to recover or create backup
                    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                    corrupt_file = os.path.join(MEMORY_BACKUP_DIR, f"memory_store_corrupt_{timestamp}.json")
                    
                    try:
                        import shutil
                        shutil.copy2(MEMORY_STORE_FILE, corrupt_file)
                        logger.info(f"Saved corrupt memory file to {corrupt_file}")
                    except Exception as e:
                        logger.warning(f"Failed to backup corrupt memory: {e}")
                    
                    # Send critical alert
                    if ALERT_SYSTEM_AVAILABLE:
                        alert_system.critical(
                            "Corrupted memory store detected",
                            {"file": MEMORY_STORE_FILE, "backup": corrupt_file},
                            "memory_core"
                        )
                    
                    return False
                
                # Update memory data
                with self.memory_lock:
                    self.memory_data = loaded_data
                    
                    # Rebuild indices
                    self._rebuild_indices()
                
                record_count = self.memory_data["metadata"]["records_count"]
                logger.info(f"Loaded {record_count} memory records from {MEMORY_STORE_FILE}")
                return True
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {MEMORY_STORE_FILE}: {e}")
            
            if ALERT_SYSTEM_AVAILABLE:
                alert_system.critical(
                    "Invalid JSON in memory store",
                    {"error": str(e), "file": MEMORY_STORE_FILE},
                    "memory_core"
                )
            
            return False
        except Exception as e:
            logger.error(f"Error loading memory: {str(e)}")
            
            if ALERT_SYSTEM_AVAILABLE:
                alert_system.critical(
                    "Failed to load memory store",
                    {"error": str(e), "file": MEMORY_STORE_FILE},
                    "memory_core"
                )
            
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get memory usage statistics
        
        Returns:
            Dict[str, Any]: Memory statistics
        """
        with self.memory_lock:
            record_count = len(self.memory_data.get("records", []))
            
            # Category counts
            category_counts = {}
            for category, indices in self.category_index.items():
                category_counts[category] = len(indices)
            
            # Source counts
            source_counts = {}
            for source, indices in self.source_index.items():
                source_counts[source] = len(indices)
            
            # Tag counts
            tag_counts = {}
            for tag, indices in self.tag_index.items():
                tag_counts[tag] = len(indices)
            
            # Time range
            time_range = {"start": None, "end": None}
            if record_count > 0:
                timestamps = [record.get("timestamp", "") for record in self.memory_data.get("records", [])]
                valid_timestamps = [ts for ts in timestamps if ts]
                if valid_timestamps:
                    time_range["start"] = min(valid_timestamps)
                    time_range["end"] = max(valid_timestamps)
            
            # File size
            file_size = 0
            if os.path.exists(MEMORY_STORE_FILE):
                file_size = os.path.getsize(MEMORY_STORE_FILE)
            
            return {
                "record_count": record_count,
                "category_counts": category_counts,
                "source_counts": source_counts,
                "tag_counts": tag_counts,
                "time_range": time_range,
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2) if file_size > 0 else 0,
                "last_update": self.memory_data.get("metadata", {}).get("last_update", ""),
                "created_at": self.memory_data.get("metadata", {}).get("created_at", ""),
                "real_mode": self.memory_data.get("metadata", {}).get("real_mode", self.real_mode)
            }
    
    def _rebuild_indices(self) -> None:
        """Rebuild all indices after bulk changes"""
        # Clear existing indices
        self.record_index = {}
        self.tag_index = defaultdict(list)
        self.category_index = defaultdict(list)
        self.source_index = defaultdict(list)
        self.time_index = {}
        
        # Rebuild indices
        for idx, record in enumerate(self.memory_data["records"]):
            record_id = record.get("id", "")
            if record_id:
                self.record_index[record_id] = idx
            
            for tag in record.get("tags", []):
                self.tag_index[tag].append(idx)
            
            category = record.get("category", "")
            if category:
                self.category_index[category].append(idx)
            
            source = record.get("source", "")
            if source:
                self.source_index[source].append(idx)
            
            timestamp = record.get("timestamp", "")
            if timestamp:
                self.time_index[timestamp] = idx
    
    def _validate_memory_format(self, data: Dict[str, Any]) -> bool:
        """
        Validate memory data format
        
        Args:
            data (Dict[str, Any]): Memory data to validate
            
        Returns:
            bool: Validation result
        """
        # Check if it's a dictionary with required keys
        if not isinstance(data, dict):
            return False
        
        if "metadata" not in data or "records" not in data:
            return False
        
        # Check if records is a list
        if not isinstance(data["records"], list):
            return False
        
        # Check metadata
        metadata = data["metadata"]
        if not isinstance(metadata, dict):
            return False
        
        # Basic validation passed
        return True
    
    def _validate_input(self, source: str, category: str, data: Any) -> bool:
        """
        Validate input for memory record
        
        Args:
            source (str): Source of the memory
            category (str): Category of the memory
            data (Any): Memory data
            
        Returns:
            bool: Validation result
        """
        # Check source
        if not source or not isinstance(source, str):
            return False
        
        # Check category
        if not category or not isinstance(category, str):
            return False
        
        # Check data
        if data is None:
            return False
        
        return True
    
    def _normalize_tag(self, tag: str) -> str:
        """
        Normalize tag format
        
        Args:
            tag (str): Tag to normalize
            
        Returns:
            str: Normalized tag
        """
        if not tag:
            return ""
        
        # Convert to string if not already
        tag = str(tag)
        
        # Convert to lowercase
        tag = tag.lower()
        
        # Replace spaces with underscores
        tag = re.sub(r'\s+', '_', tag)
        
        # Remove special characters
        tag = re.sub(r'[^\w_]', '', tag)
        
        return tag
    
    def display_summary(self, summary: Dict[str, Any]) -> None:
        """
        Display memory summary in terminal
        
        Args:
            summary (Dict[str, Any]): Memory summary
        """
        try:
            # Display header
            print(f"\n{COLORS['bright']}{COLORS['cyan']}{'=' * 60}{COLORS['reset']}")
            print(f"{COLORS['bright']}{COLORS['cyan']}🧠 MEMORY SUMMARY{COLORS['reset']}")
            print(f"{COLORS['bright']}{COLORS['cyan']}{'=' * 60}{COLORS['reset']}")
            
            # Basic info
            print(f"{COLORS['bright']}Timeframe: {COLORS['green']}{summary.get('timeframe', 'all')}{COLORS['reset']}")
            print(f"{COLORS['white']}Start Time: {summary.get('start_time', 'N/A')}{COLORS['reset']}")
            print(f"{COLORS['white']}End Time: {summary.get('end_time', 'N/A')}{COLORS['reset']}")
            print(f"{COLORS['bright']}Records: {COLORS['green']}{summary.get('record_count', 0)}{COLORS['reset']}")
            
            # Categories
            categories = summary.get("categories", {})
            if categories:
                print(f"\n{COLORS['bright']}Categories:{COLORS['reset']}")
                for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {COLORS['white']}{category}: {count}{COLORS['reset']}")
            
            # Sources
            sources = summary.get("sources", {})
            if sources:
                print(f"\n{COLORS['bright']}Sources:{COLORS['reset']}")
                for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {COLORS['white']}{source}: {count}{COLORS['reset']}")
            
            # Top tags
            top_tags = summary.get("top_tags", [])
            if top_tags:
                print(f"\n{COLORS['bright']}Top Tags:{COLORS['reset']}")
                for i, tag_info in enumerate(top_tags[:10], 1):
                    tag_name = tag_info.get("tag", "")
                    tag_count = tag_info.get("count", 0)
                    print(f"  {i}. {COLORS['white']}{tag_name}: {tag_count}{COLORS['reset']}")
            
            # Time distribution
            time_dist = summary.get("time_distribution", {})
            if time_dist and "daily" in time_dist and time_dist["daily"]:
                print(f"\n{COLORS['bright']}Most Active Days:{COLORS['reset']}")
                sorted_days = sorted(time_dist["daily"].items(), key=lambda x: x[1], reverse=True)
                for day, count in sorted_days[:5]:
                    print(f"  {COLORS['white']}{day}: {count} records{COLORS['reset']}")
            
            if time_dist and "hourly" in time_dist and time_dist["hourly"]:
                print(f"\n{COLORS['bright']}Hourly Distribution:{COLORS['reset']}")
                
                # Find most active hour
                most_active_hour, most_active_count = max(time_dist["hourly"].items(), key=lambda x: x[1])
                
                print(f"  Most Active Hour: {COLORS['yellow']}{most_active_hour}:00 ({most_active_count} records){COLORS['reset']}")
            
            print(f"{COLORS['bright']}{COLORS['cyan']}{'=' * 60}{COLORS['reset']}\n")
            
        except Exception as e:
            logger.error(f"Error displaying summary: {str(e)}")
    
    def display_statistics(self) -> None:
        """Display memory statistics in terminal"""
        try:
            stats = self.get_statistics()
            
            # Display header
            print(f"\n{COLORS['bright']}{COLORS['cyan']}{'=' * 60}{COLORS['reset']}")
            print(f"{COLORS['bright']}{COLORS['cyan']}🧠 MEMORY STATISTICS{COLORS['reset']}")
            print(f"{COLORS['bright']}{COLORS['cyan']}{'=' * 60}{COLORS['reset']}")
            
            # Basic stats
            print(f"{COLORS['bright']}Total Records: {COLORS['green']}{stats.get('record_count', 0)}{COLORS['reset']}")
            print(f"{COLORS['white']}File Size: {stats.get('file_size_mb', 0)} MB{COLORS['reset']}")
            print(f"{COLORS['white']}Mode: {'REAL' if stats.get('real_mode', False) else 'SIMULATION'}{COLORS['reset']}")
            
            # Time range
            time_range = stats.get("time_range", {})
            if time_range.get("start") and time_range.get("end"):
                print(f"{COLORS['white']}Time Range: {time_range.get('start')} to {time_range.get('end')}{COLORS['reset']}")
            
            # Top categories
            category_counts = stats.get("category_counts", {})
            if category_counts:
                print(f"\n{COLORS['bright']}Categories:{COLORS['reset']}")
                for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {COLORS['white']}{category}: {count}{COLORS['reset']}")
            
            # Top sources
            source_counts = stats.get("source_counts", {})
            if source_counts:
                print(f"\n{COLORS['bright']}Sources:{COLORS['reset']}")
                for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {COLORS['white']}{source}: {count}{COLORS['reset']}")
            
            # Top tags (limit to top 10)
            tag_counts = stats.get("tag_counts", {})
            if tag_counts:
                print(f"\n{COLORS['bright']}Top Tags:{COLORS['reset']}")
                for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                    print(f"  {COLORS['white']}{tag}: {count}{COLORS['reset']}")
            
            # Creation info
            print(f"\n{COLORS['white']}Created: {stats.get('created_at', 'N/A')}{COLORS['reset']}")
            print(f"{COLORS['white']}Last Update: {stats.get('last_update', 'N/A')}{COLORS['reset']}")
            
            print(f"{COLORS['bright']}{COLORS['cyan']}{'=' * 60}{COLORS['reset']}\n")
            
        except Exception as e:
            logger.error(f"Error displaying statistics: {str(e)}")


    def store(self, data, category, subcategory=None, source=None, real_mode=None):
        """
        Compatibility method for older code using store() instead of add_memory_record()
        
        Args:
            data (dict): Data to store
            category (str): Data category
            subcategory (str, optional): Data subcategory
            source (str, optional): Data source
            real_mode (bool, optional): Whether data was generated in real mode
            
        Returns:
            str: ID of stored data
        """
        tags = []
        if subcategory:
            tags.append(subcategory)
            
        result = self.add_memory_record(
            source=source or "compatibility_layer",
            category=category,
            data=data,
            tags=tags,
            real_mode=real_mode
        )
        
        if result["success"] and "record" in result:
            return result["record"]["id"]
        return None
    
    # Singleton instance
_memory_core = None

def init(real_mode: bool = REAL_MODE) -> None:
    """
    Initialize the memory core
    
    Args:
        real_mode (bool): Whether running in real or simulation mode
    """
    global _memory_core
    _memory_core = MemoryCore(real_mode=real_mode)

def add_memory_record(source: str, category: str, data: Dict[str, Any], 
                     tags: List[str] = None, **kwargs) -> Dict[str, Any]:
    """
    Add a new memory record
    
    Args:
        source (str): Source of the memory
        category (str): Category of the memory
        data (Dict[str, Any]): Memory data
        tags (List[str], optional): Tags for the memory
        **kwargs: Additional fields for the record
        
    Returns:
        Dict[str, Any]: Added record or error
    """
    global _memory_core
    if _memory_core is None:
        init()
    return _memory_core.add_memory_record(source, category, data, tags, **kwargs)

def query_memory(category: Optional[str] = None, source: Optional[str] = None, 
                tags: List[str] = None, time_range: Optional[Tuple[str, str]] = None,
                coin: Optional[str] = None, strategy: Optional[str] = None,
                limit: int = 100, custom_filter: Optional[Callable] = None) -> List[Dict[str, Any]]:
    """
    Query memory records with flexible filters
    
    Args:
        category (Optional[str]): Filter by category
        source (Optional[str]): Filter by source
        tags (List[str], optional): Filter by tags (AND logic)
        time_range (Optional[Tuple[str, str]]): Filter by time range (start, end)
        coin (Optional[str]): Filter by coin
        strategy (Optional[str]): Filter by strategy
        limit (int): Limit number of results
        custom_filter (Optional[Callable]): Custom filter function
        
    Returns:
        List[Dict[str, Any]]: Matching records
    """
    global _memory_core
    if _memory_core is None:
        init()
    return _memory_core.query_memory(category, source, tags, time_range, coin, strategy, limit, custom_filter)

def summarize_memory(timeframe: str = "1d", category: Optional[str] = None,
                    source: Optional[str] = None, tags: List[str] = None) -> Dict[str, Any]:
    """
    Create a summary of memory records for a given timeframe
    
    Args:
        timeframe (str): Timeframe for summarization (e.g. "1h", "1d", "1w")
        category (Optional[str]): Filter by category
        source (Optional[str]): Filter by source
        tags (List[str], optional): Filter by tags
        
    Returns:
        Dict[str, Any]: Memory summary
    """
    global _memory_core
    if _memory_core is None:
        init()
    return _memory_core.summarize_memory(timeframe, category, source, tags)

def delete_memory(condition_func: Callable) -> Dict[str, Any]:
    """
    Delete memory records matching a condition
    
    Args:
        condition_func (Callable): Function that takes a record and returns True if it should be deleted
        
    Returns:
        Dict[str, Any]: Deletion result
    """
    global _memory_core
    if _memory_core is None:
        init()
    return _memory_core.delete_memory(condition_func)

def load_external(file_path: str) -> Dict[str, Any]:
    """
    Load records from external file
    
    Args:
        file_path (str): Path to external file
        
    Returns:
        Dict[str, Any]: Load result
    """
    global _memory_core
    if _memory_core is None:
        init()
    return _memory_core.load_external(file_path)

def save_memory() -> bool:
    """
    Save memory to file
    
    Returns:
        bool: Success status
    """
    global _memory_core
    if _memory_core is None:
        init()
    return _memory_core.save_memory()

def load_memory() -> bool:
    """
    Load memory from file
    
    Returns:
        bool: Success status
    """
    global _memory_core
    if _memory_core is None:
        init()
    return _memory_core.load_memory()

def get_statistics() -> Dict[str, Any]:
    """
    Get memory usage statistics
    
    Returns:
        Dict[str, Any]: Memory statistics
    """
    global _memory_core
    if _memory_core is None:
        init()
    return _memory_core.get_statistics()

def display_summary(timeframe: str = "1d") -> None:
    """
    Display memory summary in terminal
    
    Args:
        timeframe (str): Timeframe for summarization
    """
    global _memory_core
    if _memory_core is None:
        init()
    summary = _memory_core.summarize_memory(timeframe)
    _memory_core.display_summary(summary)

def display_statistics() -> None:
    """Display memory statistics in terminal"""
    global _memory_core
    if _memory_core is None:
        init()
    _memory_core.display_statistics()

# Initialize the memory core when this module is imported
init()

def create_sample_record() -> Dict[str, Any]:
    """Create a sample memory record for testing"""
    import random
    
    # Random sample data
    sources = MEMORY_SOURCES
    categories = MEMORY_CATEGORIES
    
    coins = ["BTC", "ETH", "SOL", "LINK", "ADA", "DOT", "AVAX"]
    strategies = ["long_strategy", "short_strategy", "sniper_strategy"]
    
    # Select random values
    source = random.choice(sources)
    category = random.choice(categories)
    coin = random.choice(coins)
    strategy = random.choice(strategies)
    
    # Create sample tags
    tags = [coin, strategy]
    
    # Create sample data based on category
    data = {}
    
    if category == "strategy_decision":
        data = {
            "coin": coin,
            "strategy": strategy,
            "confidence": round(random.uniform(0.5, 0.95), 2),
            "indicators": {
                "rsi": round(random.uniform(20, 80), 2),
                "macd": round(random.uniform(-10, 10), 2),
                "signal": round(random.uniform(-5, 5), 2)
            },
            "action": random.choice(["BUY", "SELL", "HOLD"])
        }
    elif category == "trade_execution":
        data = {
            "coin": coin,
            "strategy": strategy,
            "operation": random.choice(["LONG", "SHORT"]),
            "entry_price": round(random.uniform(1000, 50000), 2),
            "stop_loss": round(random.uniform(950, 49000), 2),
            "take_profit": round(random.uniform(1050, 55000), 2),
            "amount": round(random.uniform(0.01, 1.0), 6),
            "leverage": random.choice([1, 2, 3, 5, 10])
        }
    elif category == "market_data":
        data = {
            "coin": coin,
            "price": round(random.uniform(1000, 50000), 2),
            "volume_24h": round(random.uniform(1000000, 50000000), 2),
            "change_24h": round(random.uniform(-10, 10), 2),
            "market_cap": round(random.uniform(1000000000, 1000000000000), 2)
        }
    elif category == "sentiment_analysis":
        data = {
            "coin": coin,
            "sentiment_score": round(random.uniform(-1, 1), 2),
            "sources": ["twitter", "reddit", "news"],
            "mentions": random.randint(100, 10000),
            "positive_percentage": round(random.uniform(20, 80), 2)
        }
    else:
        data = {
            "coin": coin,
            "strategy": strategy,
            "value": round(random.uniform(0, 100), 2),
            "note": f"Sample data for {category}"
        }
    
    # Add sample record
    return add_memory_record(source, category, data, tags)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="SentientTrader.AI Memory Core")
    parser.add_argument("--real", action="store_true", help="Use real mode")
    parser.add_argument("--add", type=int, default=0, help="Add sample records")
    parser.add_argument("--summary", action="store_true", help="Display memory summary")
    parser.add_argument("--stats", action="store_true", help="Display memory statistics")
    parser.add_argument("--timeframe", type=str, default="1d", help="Timeframe for summary")
    parser.add_argument("--query", action="store_true", help="Query memory records")
    parser.add_argument("--category", type=str, help="Filter by category")
    parser.add_argument("--source", type=str, help="Filter by source")
    parser.add_argument("--coin", type=str, help="Filter by coin")
    parser.add_argument("--strategy", type=str, help="Filter by strategy")
    parser.add_argument("--export", type=str, help="Export memory to JSON file")
    return parser.parse_args()

def main() -> int:
    """Run from command line"""
    args = parse_arguments()
    
    # Initialize with real mode if specified
    if args.real:
        init(real_mode=True)
    
    # Add sample records
    if args.add > 0:
        print(f"{COLORS['cyan']}Adding {args.add} sample records...{COLORS['reset']}")
        for i in range(args.add):
            result = create_sample_record()
            if result["success"]:
                record = result["record"]
                print(f"{COLORS['green']}✓ Added record: {record['id']} - {record['category']} from {record['source']}{COLORS['reset']}")
        
        # Save memory
        save_memory()
    
    # Display statistics
    if args.stats:
        display_statistics()
    
    # Display summary
    if args.summary:
        display_summary(args.timeframe)
    
    # Query memory
    if args.query:
        query_params = {}
        
        if args.category:
            query_params["category"] = args.category
        
        if args.source:
            query_params["source"] = args.source
        
        if args.coin:
            query_params["coin"] = args.coin
        
        if args.strategy:
            query_params["strategy"] = args.strategy
        
        records = query_memory(**query_params)
        
        print(f"{COLORS['cyan']}Query returned {len(records)} records:{COLORS['reset']}")
        
        for i, record in enumerate(records[:10], 1):
            print(f"{i}. {COLORS['bright']}{record['id']}{COLORS['reset']} - {record['category']} ({record['timestamp']})")
            source = record.get("source", "")
            tags = ", ".join(record.get("tags", []))
            print(f"   Source: {source} | Tags: {tags}")
        
        if len(records) > 10:
            print(f"... and {len(records) - 10} more records")
    
    # Export memory
    if args.export:
        try:
            # Get all memory data
            with _memory_core.memory_lock:
                data_to_export = _memory_core.memory_data
            
            # Save to file
            with open(args.export, "w", encoding="utf-8") as f:
                json.dump(data_to_export, f, indent=2, ensure_ascii=False)
            
            print(f"{COLORS['green']}✓ Memory exported to {args.export}{COLORS['reset']}")
        except Exception as e:
            print(f"{COLORS['red']}✗ Export failed: {e}{COLORS['reset']}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
