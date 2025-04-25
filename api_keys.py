from random import choice
import random
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SentientTrader.AI - API Keys Manager
------------------------------------
Central module for securely managing API keys for various services used throughout the system.
This module loads API keys from the .env file and provides a standardized interface for accessing them.

Created by: mstfatiryaki
Date: 2025-04-23
Version: 1.0.0
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

# Try to import dotenv for .env file loading
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("Warning: dotenv package not found. Install with: pip install python-dotenv")

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
sys.path.append(project_root)

# Try importing AlertSystem if available
try:
    from modules.utils.alert_system import AlertSystem
    ALERT_SYSTEM_AVAILABLE = True
except ImportError:
    ALERT_SYSTEM_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger("APIKeysManager")

# Constants
ENV_FILE_PATH = os.path.join(project_root, '.env')
API_KEYS_CACHE_FILE = os.path.join(project_root, '.api_keys_cache')
DEFAULT_ALERT_LEVEL = "warning"  # Options: info, warning, error

class APIKeysManager:
    """
    Manager for API keys used throughout the SentientTrader.AI system.
    Loads keys from .env file and provides methods for accessing them.
    """
    
    def __init__(self, env_path: str = ENV_FILE_PATH, silent: bool = False):
        """
        Initialize the API Keys Manager.
        
        Args:
            env_path (str): Path to the .env file
            silent (bool): Whether to suppress warnings
        """
        self.env_path = env_path
        self.silent = silent
        self.loaded = False
        self.alert_system = None
        self.api_keys = {}
        
        # Initialize alert system if available
        if ALERT_SYSTEM_AVAILABLE:
            try:
                self.alert_system = AlertSystem(module_name="api_keys", real_mode=False)
            except Exception as e:
                logger.warning(f"Failed to initialize AlertSystem: {e}")
        
        # Load API keys
        self._load_api_keys()
    
    def _load_api_keys(self) -> bool:
        """
        Load API keys from .env file.
        
        Returns:
            bool: True if keys were loaded successfully, False otherwise
        """
        # Initialize return structure
        self.api_keys = {
            "binance": {"api_key": None, "secret_key": None},
            "coingecko": {"api_key": None},
            "openai": {"api_key": None},
            "deepseek": {"api_key": None}
        }
        
        # Load .env file if dotenv is available
        if DOTENV_AVAILABLE:
            # Check if .env file exists
            if os.path.exists(self.env_path):
                try:
                    load_dotenv(self.env_path)
                    logger.info(f"Loaded environment variables from {self.env_path}")
                except Exception as e:
                    self._alert(f"Error loading .env file: {e}", level="error")
                    return False
            else:
                self._alert(f".env file not found at {self.env_path}", level="warning")
        else:
            self._alert("python-dotenv package not installed. Using environment variables directly.", level="warning")
        
        # Load Binance API keys
        self.api_keys["binance"]["api_key"] = self._get_env_var("BINANCE_API_KEY")
        self.api_keys["binance"]["secret_key"] = self._get_env_var("BINANCE_SECRET_KEY")
        
        # Load CoinGecko API key (optional)
        self.api_keys["coingecko"]["api_key"] = self._get_env_var("COINGECKO_API_KEY", required=False)
        
        # Load OpenAI API key
        self.api_keys["openai"]["api_key"] = self._get_env_var("OPENAI_API_KEY")
        
        # Load DeepSeek API key
        self.api_keys["deepseek"]["api_key"] = self._get_env_var("DEEPSEEK_API_KEY")
        
        # Mark as loaded
        self.loaded = True
        
        # Expose API keys as module-level variables for easy import
        self._expose_api_keys()
        
        return True
    
    def _get_env_var(self, var_name: str, required: bool = True) -> Optional[str]:
        """
        Get environment variable and handle missing values.
        
        Args:
            var_name (str): Environment variable name
            required (bool): Whether the variable is required
            
        Returns:
            Optional[str]: Environment variable value or None if not found
        """
        value = os.getenv(var_name)
        
        if not value and required:
            self._alert(f"Required API key not found: {var_name}", level="warning")
        elif not value and not required:
            logger.info(f"Optional API key not found: {var_name}")
        
        return value
    
    def _alert(self, message: str, level: str = DEFAULT_ALERT_LEVEL) -> None:
        """
        Display alert via AlertSystem or fallback to logger.
        
        Args:
            message (str): Alert message
            level (str): Alert level (info, warning, error)
        """
        if self.silent:
            return
        
        if self.alert_system is not None:
            # Use AlertSystem if available
            if level == "info":
                self.alert_system.info(message, module="api_keys", category="api_key_manager")
            elif level == "warning":
                self.alert_system.warning(f"[api_keys] {message}")
            elif level == "error":
                self.alert_system.error(message, module="api_keys", category="api_key_manager")
        else:
            # Fallback to logger
            if level == "info":
                logger.info(message)
            elif level == "warning":
                logger.warning(message)
                print(f"\033[93mWarning: {message}\033[0m")  # Yellow text
            elif level == "error":
                logger.error(message)
                print(f"\033[91mError: {message}\033[0m")  # Red text
    
    def _expose_api_keys(self) -> None:
        """Expose API keys as module-level variables for easy import."""
        # Expose Binance keys
        global BINANCE_API_KEY, BINANCE_SECRET_KEY
        BINANCE_API_KEY = self.api_keys["binance"]["api_key"]
        BINANCE_SECRET_KEY = self.api_keys["binance"]["secret_key"]
        
        # Expose CoinGecko key
        global COINGECKO_API_KEY
        COINGECKO_API_KEY = self.api_keys["coingecko"]["api_key"]
        
        # Expose OpenAI key
        global OPENAI_API_KEY
        OPENAI_API_KEY = self.api_keys["openai"]["api_key"]
        
        # Expose DeepSeek key
        global DEEPSEEK_API_KEY
        DEEPSEEK_API_KEY = self.api_keys["deepseek"]["api_key"]
    
    def get_api_keys(self) -> Dict[str, Dict[str, Optional[str]]]:
        """
        Get all API keys.
        
        Returns:
            Dict: Dictionary containing all API keys
        """
        return self.api_keys
    
    def get_binance_keys(self) -> Dict[str, Optional[str]]:
        """
        Get Binance API keys.
        
        Returns:
            Dict: Binance API and secret keys
        """
        return self.api_keys["binance"]
    
    def get_coingecko_key(self) -> Optional[str]:
        """
        Get CoinGecko API key.
        
        Returns:
            Optional[str]: CoinGecko API key or None if not set
        """
        return self.api_keys["coingecko"]["api_key"]
    
    def get_openai_key(self) -> Optional[str]:
        """
        Get OpenAI API key.
        
        Returns:
            Optional[str]: OpenAI API key or None if not set
        """
        return self.api_keys["openai"]["api_key"]
    
    def get_deepseek_key(self) -> Optional[str]:
        """
        Get DeepSeek API key.
        
        Returns:
            Optional[str]: DeepSeek API key or None if not set
        """
        return self.api_keys["deepseek"]["api_key"]
    
    def is_binance_configured(self) -> bool:
        """
        Check if Binance API is properly configured.
        
        Returns:
            bool: True if both API key and secret key are set
        """
        return (self.api_keys["binance"]["api_key"] is not None and 
                self.api_keys["binance"]["secret_key"] is not None)
    
    def is_coingecko_configured(self) -> bool:
        """
        Check if CoinGecko API is configured (optional).
        
        Returns:
            bool: True if API key is set
        """
        return self.api_keys["coingecko"]["api_key"] is not None
    
    def is_openai_configured(self) -> bool:
        """
        Check if OpenAI API is configured.
        
        Returns:
            bool: True if API key is set
        """
        return self.api_keys["openai"]["api_key"] is not None
    
    def is_deepseek_configured(self) -> bool:
        """
        Check if DeepSeek API is configured.
        
        Returns:
            bool: True if API key is set
        """
        return self.api_keys["deepseek"]["api_key"] is not None
    
    def get_missing_keys(self) -> Dict[str, Any]:
        """
        Get a list of missing (required) API keys.
        
        Returns:
            Dict: Dictionary of missing API keys by platform
        """
        missing = {}
        
        if not self.is_binance_configured():
            missing["binance"] = {
                "api_key": self.api_keys["binance"]["api_key"] is None,
                "secret_key": self.api_keys["binance"]["secret_key"] is None
            }
        
        if not self.is_openai_configured():
            missing["openai"] = {"api_key": True}
        
        if not self.is_deepseek_configured():
            missing["deepseek"] = {"api_key": True}
        
        return missing
    
    def print_status(self) -> None:
        """Print the status of all API keys to the console."""
        print("\n=== API Keys Status ===")
        
        # Binance
        print("\n🔑 Binance API:")
        if self.is_binance_configured():
            api_key = self.api_keys["binance"]["api_key"]
            masked_api = f"{api_key[:5]}...{api_key[-5:]}" if api_key else "None"
            secret_key = self.api_keys["binance"]["secret_key"]
            masked_secret = f"{secret_key[:3]}...{secret_key[-3:]}" if secret_key else "None"
            print(f"  API Key: {masked_api} ✅")
            print(f"  Secret Key: {masked_secret} ✅")
        else:
            print(f"  API Key: {'None' if self.api_keys['binance']['api_key'] is None else '✅'}")
            print(f"  Secret Key: {'None' if self.api_keys['binance']['secret_key'] is None else '✅'}")
        
        # CoinGecko
        print("\n🔑 CoinGecko API:")
        if self.is_coingecko_configured():
            api_key = self.api_keys["coingecko"]["api_key"]
            masked_api = f"{api_key[:5]}...{api_key[-5:]}" if api_key else "None"
            print(f"  API Key: {masked_api} ✅ (Optional)")
        else:
            print("  API Key: None (Optional)")
        
        # OpenAI
        print("\n🔑 OpenAI API:")
        if self.is_openai_configured():
            api_key = self.api_keys["openai"]["api_key"]
            masked_api = f"{api_key[:5]}...{api_key[-5:]}" if api_key else "None"
            print(f"  API Key: {masked_api} ✅")
        else:
            print("  API Key: None ❌")
        
        # DeepSeek
        print("\n🔑 DeepSeek API:")
        if self.is_deepseek_configured():
            api_key = self.api_keys["deepseek"]["api_key"]
            masked_api = f"{api_key[:5]}...{api_key[-5:]}" if api_key else "None"
            print(f"  API Key: {masked_api} ✅")
        else:
            print("  API Key: None ❌")
        
        # Summary
        missing = self.get_missing_keys()
        if missing:
            print("\n⚠️ Missing required API keys:")
            for platform, keys in missing.items():
                for key, is_missing in keys.items():
                    if is_missing:
                        print(f"  - {platform}.{key}")
        else:
            print("\n✅ All required API keys are configured.")
        
        print("\n=======================\n")


# Initialize API Keys Manager and expose keys
_api_manager = APIKeysManager()

# Expose the api_keys dictionary for easy access
api_keys = _api_manager.get_api_keys()

# Expose global functions for easy access
def get_api_keys() -> Dict[str, Dict[str, Optional[str]]]:
    """Get all API keys."""
    return _api_manager.get_api_keys()

def get_binance_keys() -> Dict[str, Optional[str]]:
    """Get Binance API keys."""
    return _api_manager.get_binance_keys()

def get_coingecko_key() -> Optional[str]:
    """Get CoinGecko API key."""
    return _api_manager.get_coingecko_key()

def get_openai_key() -> Optional[str]:
    """Get OpenAI API key."""
    return _api_manager.get_openai_key()

def get_deepseek_key() -> Optional[str]:
    """Get DeepSeek API key."""
    return _api_manager.get_deepseek_key()

def is_binance_configured() -> bool:
    """Check if Binance API is properly configured."""
    return _api_manager.is_binance_configured()

def is_coingecko_configured() -> bool:
    """Check if CoinGecko API is configured."""
    return _api_manager.is_coingecko_configured()

def is_openai_configured() -> bool:
    """Check if OpenAI API is configured."""
    return _api_manager.is_openai_configured()

def is_deepseek_configured() -> bool:
    """Check if DeepSeek API is configured."""
    return _api_manager.is_deepseek_configured()

def get_missing_keys() -> Dict[str, Any]:
    """Get missing required API keys."""
    return _api_manager.get_missing_keys()

def print_status() -> None:
    """Print API keys status."""
    _api_manager.print_status()


def create_sample_env_file(path: str = ENV_FILE_PATH) -> bool:
    """
    Create a sample .env file with placeholders for API keys.
    
    Args:
        path (str): Path where to create the .env file
        
    Returns:
        bool: True if file was created successfully, False otherwise
    """
    if os.path.exists(path):
        print(f"Warning: .env file already exists at {path}")
        return False
    
    try:
        with open(path, 'w') as f:
            f.write("""# SentientTrader.AI - API Keys Configuration
# Replace the placeholders with your actual API keys

# Binance API Keys
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET_KEY=your_binance_secret_key_here

# CoinGecko API Key (optional)
COINGECKO_API_KEY=your_coingecko_api_key_here

# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# DeepSeek API Key
DEEPSEEK_API_KEY=your_deepseek_api_key_here
""")
        print(f"Created sample .env file at {path}")
        return True
    except Exception as e:
        print(f"Error creating sample .env file: {e}")
        return False


# Example usage
if __name__ == "__main__":
    print("\n====== SentientTrader.AI - API Keys Manager ======\n")
    
    # Print current API keys status
    print("Current API keys status:")
    print_status()
    
    # Example: Check if specific APIs are configured
    print("\nAPI Configuration Status:")
    print(f"Binance API configured: {is_binance_configured()}")
    print(f"CoinGecko API configured: {is_coingecko_configured()}")
    print(f"OpenAI API configured: {is_openai_configured()}")
    print(f"DeepSeek API configured: {is_deepseek_configured()}")
    
    # Example: Get missing keys
    missing_keys = get_missing_keys()
    if missing_keys:
        print("\nMissing API keys:")
        for platform, keys in missing_keys.items():
            print(f"- {platform}: {keys}")
    
    # Example: Create sample .env file if it doesn't exist
    if not os.path.exists(ENV_FILE_PATH):
        print("\nDo you want to create a sample .env file? (y/n)")
        if input().lower() == 'y':
            create_sample_env_file()
    
    print("\n====== End of API Keys Manager Demo ======")
