from random import choice
import random
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json
import time
import os
import logging
from datetime import datetime, timezone

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ChainInspector")

# Known exchanges and their addresses (simplified - in production would need a more comprehensive list)
KNOWN_EXCHANGES = {
    "binance": ["0x28c6c06298d514db089934071355e5743bf21d60", "0x21a31ee1afc51d94c2efccaa2092ad1028285549"],
    "kucoin": ["0x2b5634c42055806a59e9107ed44d43c426e58258", "0xf16e9b0d03470827a95cdfd0cb8a8a3b46969b91"],
    "okx": ["0x6cc5f688a315f3dc28a7781717a9a798a59fda7b", "0x5041ed759dd4afc3a72b8192c143f72f4724081a"],
    "coinbase": ["0x71660c4005ba85c37ccec55d0c4493e66fe775d3", "0x503828976d22510aad0201ac7ec88293211d23da"],
    "kraken": ["0x2910543af39aba0cd09dbb2d50200b3e800a63d2", "0x0a869d79a7052c7f1b55a8ebabbea3420f0d1e13"],
    "ftx": ["0x2faf487a4414fe77e2327f0bf4ae2a264a776ad2", "0xc098b2a3aa256d2140208c3de6543aaef5cd3a94"],
    "huobi": ["0x5c985e89dde482efe97ea9f1950ad149eb73829b", "0xab5c66752a9e8167967685f1450532fb96d5d24f"],
    "bitfinex": ["0x77134cbc06cb00b66f4c7e623d5fdbf6777635ec", "0x1151314c646ce4e0efd76d1af4760ae66a9fe30f"],
    "gateio": ["0x0d0707963952f2fba59dd06f2b425ace40b492fe", "0x7793cd85c11a924478d358d49b05b37e91b5810f"]
}

def is_exchange_address(address):
    """
    Check if an address belongs to a known exchange
    
    Args:
        address (str): The blockchain address to check
        
    Returns:
        tuple: (bool, str) - Whether it's an exchange and which exchange it is
    """
    address = address.lower()
    for exchange, addresses in KNOWN_EXCHANGES.items():
        if address in addresses:
            return True, exchange
    return False, None

def get_large_transfers():
    """
    Fetches large transfers (> $500,000) from blockchain data
    
    Returns:
        list: List of dictionaries containing transfer data
    """
    logger.info("Starting to fetch large blockchain transfers")
    
    # API endpoint (note: real Whale Alert API requires a key)
    # In a real implementation, you would use your API key
    base_url = "https://api.whale-alert.io/v1/transactions"
    
    # Current time and 24 hours ago in Unix timestamp
    now = int(time.time())
    start_time = now - (24 * 60 * 60)  # 24 hours ago
    
    params = {
        "api_key": "YOUR_WHALE_ALERT_API_KEY",  # Replace with actual key in production
        "start": start_time,
        "end": now,
        "min_value": 500000,  # $500,000 minimum value
        "limit": 100
    }
    
    headers = {
        "Accept": "application/json",
        "User-Agent": "SentientTrader.AI-Chain-Inspector/1.0"
    }
    
    transfers = []
    
    try:
        # In a real scenario, you'd use this endpoint with your API key
        # response = requests.get(base_url, params=params, headers=headers, timeout=30)
        # response.raise_for_status()
        # data = response.json()
        
        # Since we can't actually call the API without a key, for demonstration purposes
        # I'll create sample data that follows Whale Alert's response structure
        # In a real implementation, you'd use the commented code above
        
        data = simulate_whale_alert_response()
        
        if "transactions" in data and data["transactions"]:
            for tx in data["transactions"]:
                try:
                    # Extract required information
                    timestamp = tx.get("timestamp", 0)
                    symbol = tx.get("symbol", "").upper()
                    amount = tx.get("amount", 0)
                    amount_usd = tx.get("amount_usd", 0)
                    
                    # Skip if below threshold
                    if amount_usd < 500000:
                        continue
                    
                    from_address = tx.get("from", {}).get("address", "")
                    to_address = tx.get("to", {}).get("address", "")
                    
                    # Check if sender or receiver is an exchange
                    from_is_exchange, from_exchange = is_exchange_address(from_address)
                    to_is_exchange, to_exchange = is_exchange_address(to_address)
                    
                    # Determine transaction type
                    tx_type = "Unknown"
                    if from_is_exchange and not to_is_exchange:
                        tx_type = "Withdraw"
                    elif not from_is_exchange and to_is_exchange:
                        tx_type = "Deposit"
                    elif from_is_exchange and to_is_exchange:
                        tx_type = "Exchange-to-Exchange"
                    else:
                        tx_type = "Wallet-to-Wallet"
                    
                    # Create transfer record
                    transfer = {
                        "timestamp": timestamp,
                        "datetime_utc": datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                        "coin": symbol,
                        "amount": amount,
                        "amount_usd": amount_usd,
                        "from_address": from_address,
                        "to_address": to_address,
                        "to_exchange": to_is_exchange,
                        "exchange_name": to_exchange if to_is_exchange else (from_exchange if from_is_exchange else None),
                        "transaction_type": tx_type,
                        "blockchain": tx.get("blockchain", ""),
                        "transaction_hash": tx.get("hash", "")
                    }
                    
                    transfers.append(transfer)
                    
                except Exception as e:
                    logger.warning(f"Error processing transaction: {e}")
            
            logger.info(f"Successfully processed {len(transfers)} large transfers")
        else:
            logger.warning("No transactions found in API response")
    
    except requests.exceptions.RequestException as e:
        logger.error(f"API request error: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    
    return transfers

def simulate_whale_alert_response():
    """
    Simulates a response from the Whale Alert API for testing purposes
    
    Returns:
        dict: Simulated API response
    """
    return {
        "result": "success",
        "count": 5,
        "transactions": [
            {
                "blockchain": "bitcoin",
                "symbol": "BTC",
                "transaction_type": "transfer",
                "hash": "1a0c7897fa8717d9e99cfd61580c517b25945d4ae42f2ac31f2e23d83b629f17",
                "timestamp": int(time.time()) - 3600,
                "amount": 150.5,
                "amount_usd": 4515000,
                "from": {
                    "address": "1MnYBhQrA7nMV3ERaEmgLXSHQ3jcnQMEof",
                    "owner": "unknown",
                    "owner_type": "unknown"
                },
                "to": {
                    "address": "0x503828976d22510aad0201ac7ec88293211d23da",
                    "owner": "coinbase",
                    "owner_type": "exchange"
                }
            },
            {
                "blockchain": "ethereum",
                "symbol": "ETH",
                "transaction_type": "transfer",
                "hash": "0x72b0c2e92b04a0c37aadf898946b261e63e559a072c50a880b711e33d3cf7bbe",
                "timestamp": int(time.time()) - 7200,
                "amount": 850.75,
                "amount_usd": 1701500,
                "from": {
                    "address": "0x28c6c06298d514db089934071355e5743bf21d60",
                    "owner": "binance",
                    "owner_type": "exchange"
                },
                "to": {
                    "address": "0x3e7fc44e25c07be3d67c241e6e59cb838df7c389",
                    "owner": "unknown",
                    "owner_type": "unknown"
                }
            },
            {
                "blockchain": "tron",
                "symbol": "USDT",
                "transaction_type": "transfer",
                "hash": "53c4fef6a7551b1b4b3afa4dad28267f9a49e3adcd3482b866813699c4e0bfb3",
                "timestamp": int(time.time()) - 14400,
                "amount": 2500000,
                "amount_usd": 2500000,
                "from": {
                    "address": "TNaRAoLUyYEV2uF7GUrzSjRQTU8v5ZJ5VR",
                    "owner": "unknown",
                    "owner_type": "unknown"
                },
                "to": {
                    "address": "0x5041ed759dd4afc3a72b8192c143f72f4724081a",
                    "owner": "okx",
                    "owner_type": "exchange"
                }
            },
            {
                "blockchain": "ethereum",
                "symbol": "USDC",
                "transaction_type": "transfer",
                "hash": "0x9be0fc387a8cf4ebe54d5c2e8b2be90b8d1a2fdab27df874f64894236d648dcc",
                "timestamp": int(time.time()) - 21600,
                "amount": 1250000,
                "amount_usd": 1250000,
                "from": {
                    "address": "0x0a869d79a7052c7f1b55a8ebabbea3420f0d1e13",
                    "owner": "kraken",
                    "owner_type": "exchange"
                },
                "to": {
                    "address": "0x7ac049b7d78bb79eee84b9b9ef52c8a9a591f5a3",
                    "owner": "unknown",
                    "owner_type": "unknown"
                }
            },
            {
                "blockchain": "ripple",
                "symbol": "XRP",
                "transaction_type": "transfer",
                "hash": "B6C39477895C7DBDBB9DBF7D0EC33C99F1B7B074E702C045A4E6F93ADE5CC374",
                "timestamp": int(time.time()) - 28800,
                "amount": 5000000,
                "amount_usd": 750000,
                "from": {
                    "address": "rDbWJ9C7uExThZYAwV8m6LsZ5YSX3sa6US",
                    "owner": "unknown",
                    "owner_type": "unknown"
                },
                "to": {
                    "address": "0xf16e9b0d03470827a95cdfd0cb8a8a3b46969b91",
                    "owner": "kucoin",
                    "owner_type": "exchange"
                }
            }
        ]
    }

def save_to_json(data):
    """
    Saves collected data to JSON file
    
    Args:
        data (list): List of dictionaries containing transfer data
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not data:
        logger.warning("No data to save")
        return False
    
    try:
        with open("whale_transfers.json", "w") as f:
            json.dump(data, f, indent=2)
        
        file_size = os.path.getsize("whale_transfers.json") / 1024  # Size in KB
        logger.info(f"Successfully saved data to whale_transfers.json ({file_size:.2f} KB)")
        return True
    
    except IOError as e:
        logger.error(f"File I/O error: {e}")
        return False
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        return False

def main():
    """Main function to run the whale transaction monitoring process"""
    logger.info("Starting blockchain whale transfer monitoring")
    
    try:
        # Get large transfers data
        transfers = get_large_transfers()
        
        # Save data to JSON file
        if transfers:
            save_to_json(transfers)
        else:
            logger.warning("No large transfers detected")
    
    except Exception as e:
        logger.error(f"Error in main process: {e}")

if __name__ == "__main__":
    main()
