from random import choice
import random
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SentientTrader.AI - Strategy Engine
-----------------------------------
Bu modül, çeşitli veri kaynaklarını analiz ederek en uygun strateji ve
hedef coinleri belirler. Gerçek zamanlı piyasa verileri, balina hareketleri,
duygu analizi ve öğrenme çıktılarını kullanarak karar verir.
"""

import os
import sys
import json
import logging
import datetime
import time
import math
import argparse
from typing import Dict, List, Any, Optional, Tuple, Union, Set

# Loglama ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("strategy_engine.log")]
)
logger = logging.getLogger("StrategyEngine")

# Renkli terminal çıktısı için
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

# Sabitler
CURRENT_TIME = "2025-04-21 18:23:35"  # UTC
CURRENT_USER = "mstfatiryaki"

# Girdi dosyaları
COLLECTOR_DATA_FILE = "collector_data.json"
COINGECKO_DATA_FILE = "coingecko_data.json"
WHALE_TRANSFERS_FILE = "whale_transfers.json"
SENTIMENT_SUMMARY_FILE = "sentiment_summary.json"
MEMORY_STORE_FILE = "memory_store.json"
LEARNING_SUMMARY_FILE = "learning_summary.json"

# Çıktı dosyası
STRATEGY_DECISION_FILE = "strategy_decision.json"

# Strateji modülleri
STRATEGY_MODULES = {
    "long_strategy.py": {
        "name": "long_strategy.py",
        "description": "Yükseliş trendlerinde uzun pozisyon stratejisi",
        "conditions": ["Bullish", "Accumulation", "Recovery"],
        "min_score": 70
    },
    "short_strategy.py": {
        "name": "short_strategy.py",
        "description": "Düşüş trendlerinde kısa pozisyon stratejisi",
        "conditions": ["Bearish", "Distribution", "Correction"],
        "min_score": 70
    },
    "sniper_strategy.py": {
        "name": "sniper_strategy.py",
        "description": "Volatil piyasalarda hızlı giriş-çıkış stratejisi",
        "conditions": ["Volatile", "Breakout", "Reversal"],
        "min_score": 75
    }
}

# İşlem zaman aralıkları (UTC)
TRADING_HOURS = {
    "start": 0,  # 00:00 UTC
    "end": 23,   # 23:00 UTC
    # Özel zaman kısıtlamaları eklenebilir
    "blackout_periods": [
        # Örnek: Cuma 22:00 - Pazartesi 02:00 arası işlem yapma (hafta sonu)
        # {"day_start": 4, "hour_start": 22, "day_end": 0, "hour_end": 2}
    ]
}

# Skor ağırlıkları
SCORE_WEIGHTS = {
    "technical": 0.30,  # Teknik analiz
    "fundamental": 0.15,  # Temel analiz
    "sentiment": 0.15,   # Duygu analizi
    "whale": 0.20,       # Balina hareketleri
    "historical": 0.10,  # Geçmiş performans
    "learning": 0.10     # Öğrenme çıktısı
}

class StrategyEngine:
    """SentientTrader.AI sisteminde strateji belirleme motoru"""
    
    def __init__(self, simulation_mode: bool = False):
        """
        Strateji motoru değişkenlerini başlat
        
        Args:
            simulation_mode (bool, optional): Simülasyon modu
        """
        self.simulation_mode = simulation_mode
        
        # Veri depoları
        self.collector_data = {}  # Binance verileri
        self.coingecko_data = {}  # CoinGecko verileri
        self.whale_transfers = {}  # Balina transferleri
        self.sentiment_data = {}  # Duygu analizi verileri 
        self.memory_data = {}     # Hafıza verileri
        self.learning_data = {}   # Öğrenme verileri
        
        # Analiz sonuçları
        self.analyzed_coins = []      # Analiz edilen coinler
        self.market_conditions = {}   # Piyasa koşulları
        self.coin_scores = {}         # Coin skorları
        self.target_coins = []        # Seçilen hedef coinler
        self.selected_strategy = ""   # Seçilen strateji
        self.decision_factors = []    # Karar faktörleri
        
        # Konfigürasyon
        self.min_coin_score = 60     # Minimum coin skoru
        self.max_target_coins = 5     # Maksimum hedef coin sayısı
        self.exclude_weak_coins = True  # Zayıf coinleri hariç tut
    
    def load_collector_data(self) -> bool:
        """
        Binance veri toplayıcısından veri yükle
        
        Returns:
            bool: Başarılı mı?
        """
        try:
            if not os.path.exists(COLLECTOR_DATA_FILE):
                logger.error(f"{COLLECTOR_DATA_FILE} dosyası bulunamadı!")
                return False
            
            with open(COLLECTOR_DATA_FILE, "r", encoding="utf-8") as f:
                self.collector_data = json.load(f)
            
            # Veri doğrulaması
            if not isinstance(self.collector_data, dict) or "symbols" not in self.collector_data:
                logger.error(f"{COLLECTOR_DATA_FILE} geçerli veri formatında değil!")
                return False
            
            logger.info(f"{COLLECTOR_DATA_FILE} yüklendi: {len(self.collector_data.get('symbols', []))} coin verisi")
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"{COLLECTOR_DATA_FILE} dosyası JSON hatası: {e}")
            return False
        except Exception as e:
            logger.error(f"{COLLECTOR_DATA_FILE} yüklenirken hata: {e}")
            return False
    
    def load_coingecko_data(self) -> bool:
        """
        CoinGecko verilerini yükle
        
        Returns:
            bool: Başarılı mı?
        """
        try:
            if not os.path.exists(COINGECKO_DATA_FILE):
                logger.error(f"{COINGECKO_DATA_FILE} dosyası bulunamadı!")
                return False
            
            with open(COINGECKO_DATA_FILE, "r", encoding="utf-8") as f:
                self.coingecko_data = json.load(f)
            
            # Veri doğrulaması
            if not isinstance(self.coingecko_data, dict) or "coins" not in self.coingecko_data:
                logger.error(f"{COINGECKO_DATA_FILE} geçerli veri formatında değil!")
                return False
            
            logger.info(f"{COINGECKO_DATA_FILE} yüklendi: {len(self.coingecko_data.get('coins', []))} coin verisi")
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"{COINGECKO_DATA_FILE} dosyası JSON hatası: {e}")
            return False
        except Exception as e:
            logger.error(f"{COINGECKO_DATA_FILE} yüklenirken hata: {e}")
            return False
    
    def load_whale_transfers(self) -> bool:
        """
        Balina transferlerini yükle
        
        Returns:
            bool: Başarılı mı?
        """
        try:
            if not os.path.exists(WHALE_TRANSFERS_FILE):
                logger.error(f"{WHALE_TRANSFERS_FILE} dosyası bulunamadı!")
                return False
            
            with open(WHALE_TRANSFERS_FILE, "r", encoding="utf-8") as f:
                self.whale_transfers = json.load(f)
            
            # Veri doğrulaması
            if not isinstance(self.whale_transfers, dict) or "transfers" not in self.whale_transfers:
                logger.error(f"{WHALE_TRANSFERS_FILE} geçerli veri formatında değil!")
                return False
            
            logger.info(f"{WHALE_TRANSFERS_FILE} yüklendi: {len(self.whale_transfers.get('transfers', []))} transfer verisi")
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"{WHALE_TRANSFERS_FILE} dosyası JSON hatası: {e}")
            return False
        except Exception as e:
            logger.error(f"{WHALE_TRANSFERS_FILE} yüklenirken hata: {e}")
            return False
    
    def load_sentiment_data(self) -> bool:
        """
        Duygu analizi verilerini yükle
        
        Returns:
            bool: Başarılı mı?
        """
        try:
            if not os.path.exists(SENTIMENT_SUMMARY_FILE):
                logger.error(f"{SENTIMENT_SUMMARY_FILE} dosyası bulunamadı!")
                return False
            
            with open(SENTIMENT_SUMMARY_FILE, "r", encoding="utf-8") as f:
                self.sentiment_data = json.load(f)
            
            # Veri doğrulaması
            if not isinstance(self.sentiment_data, dict):
                logger.error(f"{SENTIMENT_SUMMARY_FILE} geçerli veri formatında değil!")
                return False
            
            logger.info(f"{SENTIMENT_SUMMARY_FILE} yüklendi")
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"{SENTIMENT_SUMMARY_FILE} dosyası JSON hatası: {e}")
            return False
        except Exception as e:
            logger.error(f"{SENTIMENT_SUMMARY_FILE} yüklenirken hata: {e}")
            return False
    
    def load_memory_data(self) -> bool:
        """
        Hafıza verilerini yükle
        
        Returns:
            bool: Başarılı mı?
        """
        try:
            if not os.path.exists(MEMORY_STORE_FILE):
                logger.warning(f"{MEMORY_STORE_FILE} dosyası bulunamadı!")
                return False
            
            with open(MEMORY_STORE_FILE, "r", encoding="utf-8") as f:
                self.memory_data = json.load(f)
            
            # Veri doğrulaması
            if not isinstance(self.memory_data, dict) or "records" not in self.memory_data:
                logger.warning(f"{MEMORY_STORE_FILE} geçerli veri formatında değil!")
                return False
            
            logger.info(f"{MEMORY_STORE_FILE} yüklendi: {len(self.memory_data.get('records', {}))} kayıt")
            return True
            
        except json.JSONDecodeError as e:
            logger.warning(f"{MEMORY_STORE_FILE} dosyası JSON hatası: {e}")
            return False
        except Exception as e:
            logger.warning(f"{MEMORY_STORE_FILE} yüklenirken hata: {e}")
            return False
    
    def load_learning_data(self) -> bool:
        """
        Öğrenme verilerini yükle
        
        Returns:
            bool: Başarılı mı?
        """
        try:
            if not os.path.exists(LEARNING_SUMMARY_FILE):
                logger.warning(f"{LEARNING_SUMMARY_FILE} dosyası bulunamadı!")
                return False
            
            with open(LEARNING_SUMMARY_FILE, "r", encoding="utf-8") as f:
                self.learning_data = json.load(f)
            
            # Veri doğrulaması
            if not isinstance(self.learning_data, dict):
                logger.warning(f"{LEARNING_SUMMARY_FILE} geçerli veri formatında değil!")
                return False
            
            logger.info(f"{LEARNING_SUMMARY_FILE} yüklendi")
            return True
            
        except json.JSONDecodeError as e:
            logger.warning(f"{LEARNING_SUMMARY_FILE} dosyası JSON hatası: {e}")
            return False
        except Exception as e:
            logger.warning(f"{LEARNING_SUMMARY_FILE} yüklenirken hata: {e}")
            return False
    
    def load_all_data(self) -> bool:
        """
        Tüm veri kaynaklarını yükle
        
        Returns:
            bool: Başarılı mı?
        """
        # Gerekli veri kaynaklarını yükle
        collector_loaded = self.load_collector_data()
        coingecko_loaded = self.load_coingecko_data()
        whale_loaded = self.load_whale_transfers()
        sentiment_loaded = self.load_sentiment_data()
        
        # Hafıza ve öğrenme verileri opsiyonel olabilir
        self.load_memory_data()
        self.load_learning_data()
        
        # Kritik veri kaynakları kontrol et
        if not collector_loaded:
            logger.error("Binance verileri yüklenemedi! Strateji belirlenemiyor.")
            return False
        
        if not coingecko_loaded:
            logger.warning("CoinGecko verileri yüklenemedi! Temel analiz eksik olacak.")
            
        if not whale_loaded:
            logger.warning("Balina transferleri yüklenemedi! Balina analizi eksik olacak.")
            
        if not sentiment_loaded:
            logger.warning("Duygu analizi verileri yüklenemedi! Duygu analizi eksik olacak.")
        
        # Tüm kritik veriler yüklendiyse başarılı
        return collector_loaded
    
    def analyze_market_conditions(self) -> Dict[str, Any]:
        """
        Genel piyasa koşullarını analiz et
        
        Returns:
            Dict[str, Any]: Piyasa koşulları değerlendirmesi
        """
        logger.info("Piyasa koşulları analiz ediliyor...")
        
        # Varsayılan değerler
        market_conditions = {
            "trend": "Neutral",          # Bullish, Bearish, Neutral
            "volatility": "Medium",      # High, Medium, Low
            "sentiment": "Neutral",      # Bullish, Bearish, Neutral
            "whale_activity": "Normal",  # High, Normal, Low
            "risk_level": "Medium",      # High, Medium, Low
            "confidence": 50             # 0-100 arası
        }
        
        # 1. Sentiment verilerinden genel piyasa duyarlılığını belirle
        if self.sentiment_data:
            if "overall" in self.sentiment_data:
                overall_sentiment = self.sentiment_data["overall"]
                market_conditions["sentiment"] = overall_sentiment.get("sentiment", "Neutral")
            
            # Fear & Greed indeksi varsa ekle
            if "overall" in self.sentiment_data and "fear_greed_index" in self.sentiment_data["overall"]:
                fg_index = self.sentiment_data["overall"]["fear_greed_index"]
                market_conditions["fear_greed_index"] = fg_index
                
                # Fear & Greed indeksi değerine göre risk seviyesi belirle
                if fg_index > 75:
                    market_conditions["risk_level"] = "High"  # Aşırı açgözlülük, yüksek risk
                elif fg_index < 25:
                    market_conditions["risk_level"] = "High"  # Aşırı korku, yüksek risk
                elif 25 <= fg_index < 45:
                    market_conditions["risk_level"] = "Medium"  # Korku, orta risk
                elif 55 <= fg_index < 75:
                    market_conditions["risk_level"] = "Medium"  # Açgözlülük, orta risk
                else:
                    market_conditions["risk_level"] = "Low"  # Dengeli duygu, düşük risk
        
        # 2. Collector verilerinden piyasa trendi belirle
        if "market_summary" in self.collector_data:
            market_summary = self.collector_data["market_summary"]
            
            # BTC dominance ve piyasa değerine göre trend belirle
            if "btc_dominance" in market_summary and "trend" in market_summary:
                market_conditions["trend"] = market_summary.get("trend", "Neutral")
            
            # Volatilite değeri varsa ekle
            if "volatility" in market_summary:
                market_conditions["volatility"] = market_summary.get("volatility", "Medium")
        
        # 3. Balina transferlerinden aktivite seviyesini belirle
        if "summary" in self.whale_transfers:
            summary = self.whale_transfers["summary"]
            
            if "activity_level" in summary:
                market_conditions["whale_activity"] = summary.get("activity_level", "Normal")
        
        # 4. Öğrenme verilerinden ek bilgiler
        if "market_correlation" in self.learning_data:
            market_corr = self.learning_data["market_correlation"]
            
            if "market_conditions" in market_corr:
                # Öğrenme verilerine göre güven düzeyini artır
                market_conditions["confidence"] += 10
        
        logger.info(f"Piyasa koşulları: {market_conditions['trend']} trend, {market_conditions['volatility']} volatilite, {market_conditions['sentiment']} duyarlılık")
        return market_conditions
    
    def collect_available_coins(self) -> List[str]:
        """
        Analiz için mevcut coinleri belirle
        
        Returns:
            List[str]: Analiz edilecek coinlerin listesi
        """
        available_coins = set()
        
        # Binance verilerinden coinleri ekle
        if "symbols" in self.collector_data:
            for symbol_data in self.collector_data["symbols"]:
                symbol = symbol_data.get("symbol", "")
                if symbol.endswith("USDT"):
                    coin = symbol.replace("USDT", "")
                    available_coins.add(coin)
        
        # CoinGecko verilerinden coinleri ekle
        if "coins" in self.coingecko_data:
            for coin_data in self.coingecko_data["coins"]:
                symbol = coin_data.get("symbol", "").upper()
                if symbol:
                    available_coins.add(symbol)
        
        # Sentiment verilerinden coinleri ekle
        if "coin_sentiment" in self.sentiment_data:
            for coin in self.sentiment_data["coin_sentiment"].keys():
                available_coins.add(coin)
        
        logger.info(f"{len(available_coins)} coin analiz için uygun bulundu")
        return list(available_coins)
    
    def calculate_technical_score(self, coin: str) -> float:
        """
        Teknik analiz skorunu hesapla
        
        Args:
            coin (str): Coin sembolü
            
        Returns:
            float: Teknik analiz skoru (0-100)
        """
        base_score = 50  # Nötr başlangıç
        
        try:
            # Coin için Binance verilerini bul
            coin_data = None
            symbol = f"{coin}USDT"
            
            if "symbols" in self.collector_data:
                for symbol_data in self.collector_data["symbols"]:
                    if symbol_data.get("symbol", "") == symbol:
                        coin_data = symbol_data
                        break
            
            if not coin_data:
                logger.warning(f"{coin} için teknik veri bulunamadı!")
                return base_score
            
            # Temel göstergelere bak
            indicators = coin_data.get("indicators", {})
            
            # RSI değeri (Aşırı alım/satım durumları)
            rsi = indicators.get("rsi", 50)
            if rsi > 70:  # Aşırı alım (düşüş potansiyeli)
                base_score -= 10
            elif rsi < 30:  # Aşırı satım (yükseliş potansiyeli)
                base_score += 10
            
            # Hareketli ortalamalar
            ma_status = indicators.get("ma_status", "Neutral")
            if ma_status == "Bullish":
                base_score += 15
            elif ma_status == "Bearish":
                base_score -= 15
            
            # Bollinger Bantları
            bb_status = indicators.get("bb_status", "Neutral")
            if bb_status == "Upper Breakout":
                base_score += 10
            elif bb_status == "Lower Breakout":
                base_score += 5
            elif bb_status == "Upper Touch":
                base_score -= 5
            elif bb_status == "Lower Touch":
                base_score += 5
            
            # Hacim analizi
            volume_change = indicators.get("volume_change_24h", 0)
            if volume_change > 50:  # Büyük hacim artışı
                base_score += 10
            elif volume_change < -50:  # Büyük hacim düşüşü
                base_score -= 5
            
            # Fiyat değişimlerine bak
            price_change_24h = coin_data.get("price_change_24h", 0)
            
            if price_change_24h > 10:  # %10'dan fazla artış
                base_score += 5
            elif price_change_24h < -10:  # %10'dan fazla düşüş
                base_score -= 5
            
            # Mum formasyonları
            candle_pattern = indicators.get("candle_pattern", "None")
            if candle_pattern in ["Hammer", "Bullish Engulfing", "Morning Star"]:
                base_score += 10
            elif candle_pattern in ["Shooting Star", "Bearish Engulfing", "Evening Star"]:
                base_score -= 10
            
            # Skorun 0-100 aralığında olmasını sağla
            base_score = max(0, min(100, base_score))
            
            return base_score
            
        except Exception as e:
            logger.error(f"{coin} için teknik skor hesaplanırken hata: {e}")
            return base_score
    
    def calculate_fundamental_score(self, coin: str) -> float:
        """
        Temel analiz skorunu hesapla
        
        Args:
            coin (str): Coin sembolü
            
        Returns:
            float: Temel analiz skoru (0-100)
        """
        base_score = 50  # Nötr başlangıç
        
        try:
            # Coin için CoinGecko verilerini bul
            coin_data = None
            
            if "coins" in self.coingecko_data:
                for coin_item in self.coingecko_data["coins"]:
                    if coin_item.get("symbol", "").upper() == coin:
                        coin_data = coin_item
                        break
            
            if not coin_data:
                logger.warning(f"{coin} için temel analiz verisi bulunamadı!")
                return base_score
            
            # Piyasa değeri (market cap)
            market_cap = coin_data.get("market_cap", 0)
            if market_cap > 10000000000:  # >10B USD
                base_score += 15
            elif market_cap > 1000000000:  # >1B USD
                base_score += 10
            elif market_cap < 100000000:  # <100M USD
                base_score -= 10
            
            # Hacim / Piyasa değeri oranı
            volume = coin_data.get("volume_24h", 0)
            if market_cap > 0:
                volume_to_mcap = volume / market_cap
                if volume_to_mcap > 0.3:  # Yüksek likidite
                    base_score += 10
                elif volume_to_mcap < 0.05:  # Düşük likidite
                    base_score -= 10
            
            # Geliştirici aktivitesi
            dev_score = coin_data.get("developer_score", 0)
            if dev_score > 80:
                base_score += 10
            elif dev_score < 30:
                base_score -= 10
            
            # Topluluk puanı
            community_score = coin_data.get("community_score", 0)
            if community_score > 80:
                base_score += 5
            elif community_score < 30:
                base_score -= 5
            
            # Skor türlerini göster
            coingecko_score = coin_data.get("coingecko_score", 0)
            if coingecko_score > 80:
                base_score += 5
            elif coingecko_score < 30:
                base_score -= 5
            
            # Supply bilgileri
            max_supply = coin_data.get("max_supply", 0)
            circulating_supply = coin_data.get("circulating_supply", 0)
            
            if max_supply and circulating_supply:
                supply_ratio = circulating_supply / max_supply
                if supply_ratio > 0.9:  # Neredeyse tüm tokenler dolaşımda
                    base_score += 5
                elif supply_ratio < 0.3:  # Az sayıda token dolaşımda
                    base_score -= 5
            
            # Skorun 0-100 aralığında olmasını sağla
            base_score = max(0, min(100, base_score))
            
            return base_score
            
        except Exception as e:
            logger.error(f"{coin} için temel skor hesaplanırken hata: {e}")
            return base_score
    
    def calculate_sentiment_score(self, coin: str) -> float:
        """
        Duygu analizi skorunu hesapla
        
        Args:
            coin (str): Coin sembolü
            
        Returns:
            float: Duygu analizi skoru (0-100)
        """
        base_score = 50  # Nötr başlangıç
        
        try:
            # Genel duyarlılık
            overall_sentiment = "Neutral"
            if "overall" in self.sentiment_data:
                overall_sentiment = self.sentiment_data["overall"].get("sentiment", "Neutral")
            
            # Genel duyarlılığa göre +/- 5 puan
            if overall_sentiment == "Bullish":
                base_score += 5
            elif overall_sentiment == "Bearish":
                base_score -= 5
            
            # Coin özel duyarlılık
            if "coin_sentiment" in self.sentiment_data and coin in self.sentiment_data["coin_sentiment"]:
                coin_sentiment = self.sentiment_data["coin_sentiment"][coin]
                
                # Coin duyarlılığı
                coin_mood = coin_sentiment.get("sentiment", "Neutral")
                
                if coin_mood == "Bullish":
                    base_score += 15
                elif coin_mood == "Somewhat Bullish":
                    base_score += 10
                elif coin_mood == "Bearish":
                    base_score -= 15
                elif coin_mood == "Somewhat Bearish":
                    base_score -= 10
                
                # Duyarlılık gücü/şiddet
                intensity = coin_sentiment.get("intensity", 50)
                if intensity > 70:  # Yüksek şiddet
                    if coin_mood in ["Bullish", "Somewhat Bullish"]:
                        base_score += 10
                    elif coin_mood in ["Bearish", "Somewhat Bearish"]:
                        base_score -= 10
                
                # Sosyal medya trendslamdeki değişim
                trend_change = coin_sentiment.get("trend_change", 0)
                if trend_change > 50:  # Büyük pozitif değişim
                    base_score += 10
                elif trend_change < -50:  # Büyük negatif değişim
                    base_score -= 10
            
            # Sosyal medya duyarlılığı
            if "social_media" in self.sentiment_data:
                # Twitter duyarlılığı
                if "Twitter" in self.sentiment_data["social_media"]:
                    twitter = self.sentiment_data["social_media"]["Twitter"]
                    if twitter.get("mentioned_coins", {}).get(coin, {}).get("sentiment") == "Bullish":
                        base_score += 5
                    elif twitter.get("mentioned_coins", {}).get(coin, {}).get("sentiment") == "Bearish":
                        base_score -= 5
                
                # Reddit duyarlılığı
                if "Reddit" in self.sentiment_data["social_media"]:
                    reddit = self.sentiment_data["social_media"]["Reddit"]
                    if reddit.get("mentioned_coins", {}).get(coin, {}).get("sentiment") == "Bullish":
                        base_score += 5
                    elif reddit.get("mentioned_coins", {}).get(coin, {}).get("sentiment") == "Bearish":
                        base_score -= 5
            
            # Skorun 0-100 aralığında olmasını sağla
            base_score = max(0, min(100, base_score))
            
            return base_score
            
        except Exception as e:
            logger.error(f"{coin} için duyarlılık skoru hesaplanırken hata: {e}")
            return base_score
    
    def calculate_whale_score(self, coin: str) -> float:
        """
        Balina aktivitesi skorunu hesapla
        
        Args:
            coin (str): Coin sembolü
            
        Returns:
            float: Balina aktivitesi skoru (0-100)
        """
        base_score = 50  # Nötr başlangıç
        
        try:
            # Balina transferlerinde coin'i ara
            if "transfers" not in self.whale_transfers:
                logger.warning(f"{coin} için balina verisi bulunamadı!")
                return base_score
            
            transfers = self.whale_transfers["transfers"]
            coin_transfers = []
            
            # Son 24 saatteki transferleri filtrele
            current_time = datetime.datetime.strptime(CURRENT_TIME, "%Y-%m-%d %H:%M:%S")
            time_threshold = current_time - datetime.timedelta(hours=24)
            
            for transfer in transfers:
                if transfer.get("symbol", "").upper() == coin:
                    transfer_time = datetime.datetime.strptime(transfer.get("timestamp", "2000-01-01 00:00:00"), "%Y-%m-%d %H:%M:%S")
                    if transfer_time >= time_threshold:
                        coin_transfers.append(transfer)
            
            if not coin_transfers:
                logger.debug(f"{coin} için son 24 saatte balina transferi bulunamadı!")
                return base_score
            
            # Alım/satım dengesini hesapla
            inflow_count = 0
            outflow_count = 0
            inflow_volume = 0
            outflow_volume = 0
            
            for transfer in coin_transfers:
                transaction_type = transfer.get("transaction_type", "")
                amount_usd = transfer.get("amount_usd", 0)
                
                if transaction_type in ["Exchange-to-Wallet", "Unknown-to-Wallet"]:
                    inflow_count += 1
                    inflow_volume += amount_usd
                elif transaction_type in ["Wallet-to-Exchange", "Wallet-to-Unknown"]:
                    outflow_count += 1
                    outflow_volume += amount_usd
            
            # Alım-satım dengesine göre skoru ayarla
            total_transfers = inflow_count + outflow_count
            if total_transfers > 0:
                inflow_ratio = inflow_count / total_transfers
                
                if inflow_ratio > 0.7:  # %70'den fazla alım
                    base_score += 20
                elif inflow_ratio > 0.6:  # %60-70 arası alım
                    base_score += 15
                elif inflow_ratio < 0.3:  # %30'dan az alım (çok satış)
                    base_score -= 20
                elif inflow_ratio < 0.4:  # %30-40 arası alım
                    base_score -= 15
            
            # Toplam hacim miktarına göre önem faktörü
            total_volume = inflow_volume + outflow_volume
            if total_volume > 10000000:  # >10M USD
                if inflow_volume > outflow_volume:
                    base_score += 10
                else:
                    base_score -= 10
            
            # Büyük tek işlemler var mı?
            big_transfers = [t for t in coin_transfers if t.get("amount_usd", 0) > 5000000]  # >5M USD
            if big_transfers:
                big_inflows = [t for t in big_transfers if t.get("transaction_type") in ["Exchange-to-Wallet", "Unknown-to-Wallet"]]
                big_outflows = [t for t in big_transfers if t.get("transaction_type") in ["Wallet-to-Exchange", "Wallet-to-Unknown"]]
                
                if len(big_inflows) > len(big_outflows):
                    base_score += 15
                elif len(big_outflows) > len(big_inflows):
                    base_score -= 15
            
            # Skorun 0-100 aralığında olmasını sağla
            base_score = max(0, min(100, base_score))
            
            return base_score
            
        except Exception as e:
            logger.error(f"{coin} için balina skoru hesaplanırken hata: {e}")
            return base_score
    
    def calculate_historical_score(self, coin: str) -> float:
        """
        Geçmiş performans skorunu hesapla
        
        Args:
            coin (str): Coin sembolü
            
        Returns:
            float: Geçmiş performans skoru (0-100)
        """
        base_score = 50  # Nötr başlangıç
        
        try:
            if not self.memory_data or "records" not in self.memory_data:
                return base_score
            
            records = self.memory_data["records"]
            coin_decisions = []
            coin_trades = []
            
            # Karar ve işlem kayıtlarını topla
            for record_id, record in records.items():
                data = record.get("data", {})
                
                if record.get("category") == "decision":
                    strategy_data = data.get("strategy", {})
                    if coin in strategy_data.get("target_coins", []):
                        coin_decisions.append(record)
                
                elif record.get("category") == "trade":
                    if data.get("symbol", "") == coin:
                        coin_trades.append(record)
            
            if not coin_decisions and not coin_trades:
                return base_score
            
            # Geçmiş işlemlerdeki başarı oranını hesapla
            successful_trades = 0
            total_trades = len(coin_trades)
            
            for trade in coin_trades:
                data = trade.get("data", {})
                if data.get("result", "") == "profit":
                    successful_trades += 1
            
            # Başarı oranına göre skoru ayarla
            if total_trades > 0:
                success_rate = (successful_trades / total_trades) * 100
                
                if success_rate > 70:  # %70'den fazla başarı
                    base_score += 20
                elif success_rate > 60:  # %60-70 arası başarı
                    base_score += 15
                elif success_rate > 50:  # %50-60 arası başarı
                    base_score += 10
                elif success_rate < 30:  # %30'dan az başarı
                    base_score -= 20
                elif success_rate < 40:  # %30-40 arası başarı
                    base_score -= 15
            
            # En son işlemin sonucuna göre ek puan
            if coin_trades:
                last_trade = max(coin_trades, key=lambda x: x.get("timestamp", ""))
                last_result = last_trade.get("data", {}).get("result", "")
                
                if last_result == "profit":
                    base_score += 5
                elif last_result == "loss":
                    base_score -= 5
            
            # Skorun 0-100 aralığında olmasını sağla
            base_score = max(0, min(100, base_score))
            
            return base_score
            
        except Exception as e:
            logger.error(f"{coin} için geçmiş skor hesaplanırken hata: {e}")
            return base_score
    
    def calculate_learning_score(self, coin: str) -> float:
        """
        Öğrenme bazlı skoru hesapla
        
        Args:
            coin (str): Coin sembolü
            
        Returns:
            float: Öğrenme bazlı skor (0-100)
        """
        base_score = 50  # Nötr başlangıç
        
        try:
            if not self.learning_data:
                return base_score
            
            # Coin performans verilerini kontrol et
            if "coin_performance" in self.learning_data:
                coin_performance = self.learning_data["coin_performance"]
                
                if "coin_performance" in coin_performance and coin in coin_performance["coin_performance"]:
                    # Coin özel performansını al
                    perf = coin_performance["coin_performance"][coin]
                    
                    # ROI değerine göre skoru ayarla
                    roi = perf.get("roi", 0)
                    if roi > 20:  # %20'den fazla ROI
                        base_score += 20
                    elif roi > 10:  # %10-20 arası ROI
                        base_score += 15
                    elif roi > 5:  # %5-10 arası ROI
                        base_score += 10
                    elif roi < -10:  # %-10'dan az ROI
                        base_score -= 20
                    elif roi < -5:  # %-5 ile %-10 arası ROI
                        base_score -= 15
                    
                    # Başarı oranına göre skoru ayarla
                    win_rate = perf.get("win_rate", 0)
                    if win_rate > 70:  # %70'den fazla başarı
                        base_score += 10
                    elif win_rate < 30:  # %30'dan az başarı
                        base_score -= 10
                
                # En iyi coinler arasında mı?
                best_coins = coin_performance.get("best_coins", [])
                if coin in best_coins:
                    base_score += 15
                
                # En kötü coinler arasında mı?
                worst_coins = coin_performance.get("worst_coins", [])
                if coin in worst_coins:
                    base_score -= 15
            
            # Piyasa koşulları ile coin performansı korelasyonu
            if "market_correlation" in self.learning_data and "coin_market_matrix" in self.learning_data["market_correlation"]:
                coin_market = self.learning_data["market_correlation"]["coin_market_matrix"]
                
                if coin in coin_market:
                    # Mevcut piyasa koşullarıyla coin performansını karşılaştır
                    current_market = self.market_conditions["trend"]  # Bullish, Bearish, Neutral
                    
                    if current_market in coin_market[coin]:
                        win_rate_in_condition = coin_market[coin][current_market].get("win_rate", 0)
                        
                        if win_rate_in_condition > 70:  # Bu koşullarda %70+ başarı
                            base_score += 20
                        elif win_rate_in_condition > 60:  # Bu koşullarda %60-70 başarı
                            base_score += 15
                        elif win_rate_in_condition < 30:  # Bu koşullarda %30- başarı
                            base_score -= 20
                        elif win_rate_in_condition < 40:  # Bu koşullarda %30-40 başarı
                            base_score -= 15
            
            # Skorun 0-100 aralığında olmasını sağla
            base_score = max(0, min(100, base_score))
            
            return base_score
            
        except Exception as e:
            logger.error(f"{coin} için öğrenme skoru hesaplanırken hata: {e}")
            return base_score
    
    def calculate_coin_scores(self) -> Dict[str, Dict[str, Any]]:
        """
        Tüm coinler için toplam skor hesapla
        
        Returns:
            Dict[str, Dict[str, Any]]: Coin skorları ve detayları
        """
        logger.info("Coin skorları hesaplanıyor...")
        
        coin_scores = {}
        
        # Tüm mevcut coinleri analiz et
        for coin in self.analyzed_coins:
            try:
                # Alt skorları hesapla
                technical_score = self.calculate_technical_score(coin)
                fundamental_score = self.calculate_fundamental_score(coin)
                sentiment_score = self.calculate_sentiment_score(coin)
                whale_score = self.calculate_whale_score(coin)
                historical_score = self.calculate_historical_score(coin)
                learning_score = self.calculate_learning_score(coin)
                
                # Ağırlıklı toplam skoru hesapla
                total_score = (
                    technical_score * SCORE_WEIGHTS["technical"] +
                    fundamental_score * SCORE_WEIGHTS["fundamental"] +
                    sentiment_score * SCORE_WEIGHTS["sentiment"] +
                    whale_score * SCORE_WEIGHTS["whale"] +
                    historical_score * SCORE_WEIGHTS["historical"] +
                    learning_score * SCORE_WEIGHTS["learning"]
                )
                
                # Skor detaylarını kaydet
                coin_scores[coin] = {
                    "total_score": round(total_score, 2),
                    "details": {
                        "technical": round(technical_score, 2),
                        "fundamental": round(fundamental_score, 2),
                        "sentiment": round(sentiment_score, 2),
                        "whale": round(whale_score, 2),
                        "historical": round(historical_score, 2),
                        "learning": round(learning_score, 2)
                    }
                }
                
            except Exception as e:
                logger.error(f"{coin} için skor hesaplanırken hata: {e}")
        
        logger.info(f"{len(coin_scores)} coin için skorlar hesaplandı")
        return coin_scores
    
    def select_target_coins(self) -> List[str]:
        """
        En iyi skorlu hedef coinleri seç
        
        Returns:
            List[str]: Seçilen hedef coinler
        """
        try:
            # Coinleri skorlarına göre sırala
            sorted_coins = sorted(
                [(coin, data["total_score"]) for coin, data in self.coin_scores.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            # En yüksek skorlu coinleri seç
            top_coins = []
            min_score = self.min_coin_score
            
            for coin, score in sorted_coins:
                if score >= min_score and len(top_coins) < self.max_target_coins:
                    top_coins.append(coin)
            
            logger.info(f"{len(top_coins)} coin hedef olarak seçildi: {', '.join(top_coins)}")
            return top_coins
            
        except Exception as e:
            logger.error(f"Hedef coin seçilirken hata: {e}")
            return []
    
    def determine_best_strategy(self) -> str:
        """
        En uygun stratejiyi belirle
        
        Returns:
            str: Seçilen strateji modülü
        """
        try:
            # Piyasa koşullarını kullan
            market_trend = self.market_conditions.get("trend", "Neutral")
            market_volatility = self.market_conditions.get("volatility", "Medium")
            market_sentiment = self.market_conditions.get("sentiment", "Neutral")
            
            # Strateji puanları
            strategy_scores = {
                "long_strategy.py": 0,
                "short_strategy.py": 0,
                "sniper_strategy.py": 0
            }
            
            # Trend bazlı puanlama
            if market_trend == "Bullish":
                strategy_scores["long_strategy.py"] += 30
                strategy_scores["short_strategy.py"] -= 20
            elif market_trend == "Bearish":
                strategy_scores["short_strategy.py"] += 30
                strategy_scores["long_strategy.py"] -= 20
            elif market_trend == "Neutral":
                strategy_scores["sniper_strategy.py"] += 10
            
            # Volatilite bazlı puanlama
            if market_volatility == "High":
                strategy_scores["sniper_strategy.py"] += 30
            elif market_volatility == "Medium":
                strategy_scores["long_strategy.py"] += 10
                strategy_scores["short_strategy.py"] += 10
            elif market_volatility == "Low":
                strategy_scores["sniper_strategy.py"] -= 10
            
            # Duyarlılık bazlı puanlama
            if market_sentiment == "Bullish":
                strategy_scores["long_strategy.py"] += 20
                strategy_scores["short_strategy.py"] -= 10
            elif market_sentiment == "Bearish":
                strategy_scores["short_strategy.py"] += 20
                strategy_scores["long_strategy.py"] -= 10
            
            # Öğrenme verilerinden strateji tavsiyeleri
            if "strategy_recommendations" in self.learning_data:
                for strategy, recs in self.learning_data["strategy_recommendations"].items():
                    if "performance_summary" in recs:
                        perf_summary = recs["performance_summary"]
                        
                        # Basit bir şekilde başarı oranını çıkar
                        if "Başarı oranı: %" in perf_summary:
                            success_rate_str = perf_summary.split("Başarı oranı: %")[1].split(",")[0]
                            try:
                                success_rate = float(success_rate_str)
                                if success_rate > 60:
                                    strategy_scores[strategy] += 15
                                elif success_rate < 40:
                                    strategy_scores[strategy] -= 15
                            except ValueError:
                                pass
            
            # Hedef coinlerin en uygun olduğu stratejileri bul
            coin_strategies = {}
            for coin in self.target_coins:
                best_strat = "long_strategy.py"  # Varsayılan
                max_score = 0
                
                coin_details = self.coin_scores.get(coin, {}).get("details", {})
                
                # Teknik ve duyarlılık değerlerine göre en uygun stratejiyi seç
                technical = coin_details.get("technical", 50)
                sentiment = coin_details.get("sentiment", 50)
                whale = coin_details.get("whale", 50)
                
                # Long stratejisi için uygunluk
                long_score = (technical + sentiment + whale) / 3
                if long_score > 60:
                    coin_strategies[coin] = "long_strategy.py"
                    strategy_scores["long_strategy.py"] += 5
                # Short stratejisi için uygunluk
                elif long_score < 40:
                    coin_strategies[coin] = "short_strategy.py"
                    strategy_scores["short_strategy.py"] += 5
                # Sniper stratejisi için uygunluk
                else:
                    coin_strategies[coin] = "sniper_strategy.py"
                    strategy_scores["sniper_strategy.py"] += 5
            
            # En yüksek skorlu stratejiyi seç
            chosen_strategy = max(strategy_scores.items(), key=lambda x: x[1])[0]
            
            # Besleme faktörlerini kaydet
            self.decision_factors.append(f"Piyasa Trendi: {market_trend}")
            self.decision_factors.append(f"Piyasa Volatilitesi: {market_volatility}")
            self.decision_factors.append(f"Piyasa Duyarlılığı: {market_sentiment}")
            self.decision_factors.append(f"Strateji Skorları: LONG={strategy_scores['long_strategy.py']}, SHORT={strategy_scores['short_strategy.py']}, SNIPER={strategy_scores['sniper_strategy.py']}")
            
            logger.info(f"En uygun strateji olarak {chosen_strategy} seçildi")
            return chosen_strategy
            
        except Exception as e:
            logger.error(f"Strateji belirlenirken hata: {e}")
            return "long_strategy.py"  # Hata durumunda varsayılan
    
    def is_trading_time(self) -> bool:
        """
        Şu an işlem yapılabilecek bir zaman mı kontrol et
        
        Returns:
            bool: İşlem zamanı uygun mu?
        """
        try:
            # Şu anki zaman
            current_time = datetime.datetime.strptime(CURRENT_TIME, "%Y-%m-%d %H:%M:%S")
            current_hour = current_time.hour
            current_day = current_time.weekday()  # 0=Pazartesi, 6=Pazar
            
            # İşlem saatleri aralığında mı kontrol et
            if not (TRADING_HOURS["start"] <= current_hour <= TRADING_HOURS["end"]):
                logger.info(f"İşlem saatleri dışında: {current_hour}")
                return False
            
            # Özel kısıtlı saatlerde mi kontrol et
            for blackout in TRADING_HOURS["blackout_periods"]:
                day_start = blackout.get("day_start", 0)
                hour_start = blackout.get("hour_start", 0)
                day_end = blackout.get("day_end", 0)
                hour_end = blackout.get("hour_end", 0)
                
                # Gün ve saat kontrolü
                if day_start <= current_day <= day_end:
                    if day_start == current_day and current_hour >= hour_start:
                        logger.info(f"Kısıtlı zaman aralığında: {current_day} {current_hour}")
                        return False
                    elif day_end == current_day and current_hour <= hour_end:
                        logger.info(f"Kısıtlı zaman aralığında: {current_day} {current_hour}")
                        return False
                    elif day_start < current_day < day_end:
                        logger.info(f"Kısıtlı zaman aralığında: {current_day} {current_hour}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"İşlem zamanı kontrolünde hata: {e}")
            return True  # Hata durumunda izin ver
    
    def save_strategy_decision(self) -> bool:
        """
        Strateji kararını JSON dosyasına kaydet
        
        Returns:
            bool: Başarılı mı?
        """
        try:
            # Karar verisi oluştur
            decision_data = {
                "strategy_decision": {
                    "timestamp": CURRENT_TIME,
                    "user": CURRENT_USER,
                    "strategy": {
                        "name": self.selected_strategy,
                        "target_coins": self.target_coins,
                        "simulation": self.simulation_mode
                    },
                    "market_conditions": self.market_conditions,
                    "decision_factors": self.decision_factors,
                    "coin_scores": {coin: self.coin_scores[coin]["total_score"] for coin in self.target_coins}
                }
            }
            
            # Dosyaya yaz
            with open(STRATEGY_DECISION_FILE, "w", encoding="utf-8") as f:
                json.dump(decision_data, f, ensure_ascii=False, indent=4)
                
            logger.info(f"Strateji kararı {STRATEGY_DECISION_FILE} dosyasına kaydedildi")
            return True
            
        except Exception as e:
            logger.error(f"Strateji kararı kaydedilirken hata: {e}")
            return False
    
    def display_results(self) -> None:
        """
        Sonuçları terminalde göster
        """
        try:
            # Terminal ekranını temizle
            if os.name == 'nt':
                os.system('cls')
            else:
                os.system('clear')
            
            # Banner
            print(f"{COLORS['bright']}{COLORS['cyan']}======= SENTİENTTRADER.AI - STRATEJİ MOTORU ======={COLORS['reset']}")
            print(f"{COLORS['yellow']}Tarih: {CURRENT_TIME} | Kullanıcı: {CURRENT_USER}{COLORS['reset']}")
            print(f"{COLORS['yellow']}{'=' * 50}{COLORS['reset']}")
            
            # Piyasa durumu
            print(f"\n{COLORS['bright']}📊 PİYASA DURUMU:{COLORS['reset']}")
            
            trend_color = COLORS["green"] if self.market_conditions["trend"] == "Bullish" else COLORS["red"] if self.market_conditions["trend"] == "Bearish" else COLORS["yellow"]
            print(f"  Trend: {trend_color}{self.market_conditions['trend']}{COLORS['reset']}")
            
            volatility_color = COLORS["red"] if self.market_conditions["volatility"] == "High" else COLORS["yellow"] if self.market_conditions["volatility"] == "Medium" else COLORS["green"]
            print(f"  Volatilite: {volatility_color}{self.market_conditions['volatility']}{COLORS['reset']}")
            
            sentiment_color = COLORS["green"] if self.market_conditions["sentiment"] == "Bullish" else COLORS["red"] if self.market_conditions["sentiment"] == "Bearish" else COLORS["yellow"]
            print(f"  Duyarlılık: {sentiment_color}{self.market_conditions['sentiment']}{COLORS['reset']}")
            
            if "fear_greed_index" in self.market_conditions:
                fg_index = self.market_conditions["fear_greed_index"]
                fg_color = COLORS["green"] if fg_index > 70 else COLORS["yellow"] if fg_index > 40 else COLORS["red"]
                print(f"  Fear & Greed İndeksi: {fg_color}{fg_index}{COLORS['reset']}")
            
            # Seçilen strateji
            print(f"\n{COLORS['bright']}🔍 SEÇİLEN STRATEJİ:{COLORS['reset']}")
            strategy_desc = STRATEGY_MODULES.get(self.selected_strategy, {}).get("description", "")
            print(f"  {COLORS['green']}{self.selected_strategy}{COLORS['reset']} - {strategy_desc}")
            
            # Seçilen coinler ve skorları
            print(f"\n{COLORS['bright']}💰 HEDEF COİNLER:{COLORS['reset']}")
            for i, coin in enumerate(self.target_coins, 1):
                score = self.coin_scores.get(coin, {}).get("total_score", 0)
                score_color = COLORS["green"] if score >= 70 else COLORS["yellow"] if score >= 50 else COLORS["red"]
                print(f"  {i}. {COLORS['bright']}{coin}{COLORS['reset']} - Skor: {score_color}{score:.2f}/100{COLORS['reset']}")
                
                # Alt skorları göster
                details = self.coin_scores.get(coin, {}).get("details", {})
                if details:
                    print(f"     Teknik: {details.get('technical', 0):.1f} | Temel: {details.get('fundamental', 0):.1f} | Duyarlılık: {details.get('sentiment', 0):.1f} | Balina: {details.get('whale', 0):.1f}")
            
            # Karar faktörleri
            print(f"\n{COLORS['bright']}🧠 KARAR FAKTÖRLERİ:{COLORS['reset']}")
            for factor in self.decision_factors:
                print(f"  • {factor}")
            
            # Çalışma modu
            mode = f"{COLORS['blue']}SİMÜLASYON MODU{COLORS['reset']}" if self.simulation_mode else f"{COLORS['red']}GERÇEK İŞLEM MODU{COLORS['reset']}"
            print(f"\n{COLORS['bright']}⚙️ ÇALIŞMA MODU: {mode}")
            
            print(f"\n{COLORS['yellow']}{'=' * 50}{COLORS['reset']}")
            print(f"{COLORS['green']}Strateji kararı {STRATEGY_DECISION_FILE} dosyasına kaydedildi{COLORS['reset']}")
            print(f"{COLORS['yellow']}{'=' * 50}{COLORS['reset']}")
            
        except Exception as e:
            logger.error(f"Sonuçlar gösterilirken hata: {e}")
            print(f"Hata: {e}")
    
    def run(self) -> bool:
        """
        Strateji belirleme sürecini çalıştır
        
        Returns:
            bool: Başarılı mı?
        """
        try:
                        # Tüm veri kaynaklarını yükle
            if not self.load_all_data():
                logger.error("Veri yüklenemedi, strateji belirlenemiyor!")
                return False
            
            # İşlem zamanını kontrol et
            if not self.is_trading_time():
                logger.warning("Şu anda işlem için uygun zaman değil, strateji belirlenmeyecek.")
                print(f"{COLORS['yellow']}⚠️ UYARI: İşlem için uygun zaman değil. Strateji belirlenmedi.{COLORS['reset']}")
                return False
            
            # Piyasa koşullarını analiz et
            self.market_conditions = self.analyze_market_conditions()
            
            # Mevcut coinleri belirle
            self.analyzed_coins = self.collect_available_coins()
            
            if not self.analyzed_coins:
                logger.error("Analiz edilecek coin bulunamadı!")
                return False
            
            # Coin skorlarını hesapla
            self.coin_scores = self.calculate_coin_scores()
            
            if not self.coin_scores:
                logger.error("Coin skorları hesaplanamadı!")
                return False
            
            # Hedef coinleri seç
            self.target_coins = self.select_target_coins()
            
            if not self.target_coins:
                logger.warning("Kriterlere uygun hedef coin bulunamadı!")
                print(f"{COLORS['yellow']}⚠️ UYARI: Kriterlere uygun hedef coin bulunamadı.{COLORS['reset']}")
                return False
            
            # En uygun stratejiyi belirle
            self.selected_strategy = self.determine_best_strategy()
            
            # Strateji kararını kaydet
            if not self.save_strategy_decision():
                logger.error("Strateji kararı kaydedilemedi!")
                return False
            
            # Sonuçları göster
            self.display_results()
            
            return True
            
        except Exception as e:
            logger.error(f"Strateji belirleme sürecinde hata: {e}")
            print(f"{COLORS['red']}❌ HATA: {e}{COLORS['reset']}")
            return False

def parse_arguments() -> argparse.Namespace:
    """
    Komut satırı argümanlarını işle
    
    Returns:
        argparse.Namespace: İşlenmiş argümanlar
    """
    parser = argparse.ArgumentParser(description="SentientTrader.AI Strategy Engine")
    parser.add_argument("--simulation", action="store_true", help="Simülasyon modunda çalış")
    parser.add_argument("--min-score", type=int, default=60, help="Minimum coin skoru (0-100)")
    parser.add_argument("--max-coins", type=int, default=5, help="Maksimum hedef coin sayısı")
    parser.add_argument("--exclude-weak", action="store_true", help="Zayıf coinleri hariç tut")
    return parser.parse_args()

def display_banner() -> None:
    """Başlık banner'ını göster"""
    banner = f"""
{COLORS["bright"]}{COLORS["cyan"]}
███████╗████████╗██████╗  █████╗ ████████╗███████╗ ██████╗ ██╗   ██╗
██╔════╝╚══██╔══╝██╔══██╗██╔══██╗╚══██╔══╝██╔════╝██╔════╝ ╚██╗ ██╔╝
███████╗   ██║   ██████╔╝███████║   ██║   █████╗  ██║  ███╗ ╚████╔╝ 
╚════██║   ██║   ██╔══██╗██╔══██║   ██║   ██╔══╝  ██║   ██║  ╚██╔╝  
███████║   ██║   ██║  ██║██║  ██║   ██║   ███████╗╚██████╔╝   ██║   
╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚══════╝ ╚═════╝    ╚═╝   
{COLORS["reset"]}{COLORS["magenta"]}                                         ENGINE V2.0{COLORS["reset"]}

{COLORS["yellow"]}📅 {CURRENT_TIME} UTC | 👤 {CURRENT_USER}{COLORS["reset"]}
{COLORS["green"]}{'=' * 66}{COLORS["reset"]}
"""
    print(banner)

def main() -> int:
    """
    Ana çalıştırma fonksiyonu
    
    Returns:
        int: Çıkış kodu (0: Başarılı, 1: Başarısız)
    """
    display_banner()
    
    try:
        # Argümanları işle
        args = parse_arguments()
        
        # Strateji motoru oluştur
        engine = StrategyEngine(simulation_mode=args.simulation)
        
        # Özel konfigürasyonları ayarla
        engine.min_coin_score = args.min_score
        engine.max_target_coins = args.max_coins
        engine.exclude_weak_coins = args.exclude_weak
        
        # Çalışma modunu göster
        mode = "SİMÜLASYON" if args.simulation else "GERÇEK İŞLEM"
        logger.info(f"Strategy Engine {mode} modunda başlatılıyor...")
        print(f"{COLORS['cyan']}ℹ️ Strateji Motoru {mode} modunda başlatılıyor...{COLORS['reset']}")
        
        # Strateji motoru çalıştır
        success = engine.run()
        
        if success:
            logger.info("Strateji motoru başarıyla çalıştı.")
            return 0
        else:
            logger.error("Strateji motoru çalıştırılamadı veya strateji belirlenemedi!")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n{COLORS['yellow']}⚠️ Kullanıcı tarafından durduruldu!{COLORS['reset']}")
        return 130  # SIGINT için standart çıkış kodu
    except Exception as e:
        logger.critical(f"Beklenmeyen hata: {e}")
        print(f"\n{COLORS['red']}❌ Beklenmeyen hata: {e}{COLORS['reset']}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
