from random import choice
import random
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SentientTrader.AI - Learning Engine
-----------------------------------
Bu modül, geçmiş ticaret stratejilerini ve sonuçları analiz ederek
hangi stratejilerin ve coinlerin daha iyi performans gösterdiğini belirler.
Öğrenme sonuçlarını kaydeder ve strateji motoruna geri besleme sağlar.
"""

import os
import sys
import json
import logging
import datetime
import argparse
import math
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from collections import defaultdict

# Loglama ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("learning_engine.log")]
)
logger = logging.getLogger("LearningEngine")

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
CURRENT_TIME = "2025-04-21 18:33:48"  # UTC
CURRENT_USER = "mstfatiryaki"

# Dosya yolları
EXECUTED_TRADES_LOG = "executed_trades_log.json"
SENTIMENT_SUMMARY_FILE = "sentiment_summary.json"
STRATEGY_DECISION_FILE = "strategy_decision.json"
PERFORMANCE_REPORT_FILE = "performance_report.json"

# Çıktı dosyaları
LEARNING_SUMMARY_FILE = "learning_summary.json"
LEARNING_FEEDBACK_FILE = "learning_feedback.json"

# Strateji listesi
STRATEGIES = ["long_strategy.py", "short_strategy.py", "sniper_strategy.py"]

# Piyasa koşulları
MARKET_CONDITIONS = ["Bullish", "Bearish", "Neutral", "Volatile", "Stable"]

class LearningEngine:
    """SentientTrader.AI için öğrenme ve analiz motoru"""
    
    def __init__(self, simulation_mode: bool = False):
        """
        Öğrenme motoru değişkenlerini başlat
        
        Args:
            simulation_mode (bool, optional): Simülasyon modu aktif mi?
        """
        self.simulation_mode = simulation_mode
        
        # Veri depoları
        self.trades_data = {}             # İşlem verileri
        self.sentiment_data = {}          # Duygu analizi verileri
        self.strategy_decisions = {}      # Strateji kararları
        self.performance_data = {}        # Performans verileri
        
        # Analiz sonuçları
        self.strategy_performance = {}    # Strateji performans analizi
        self.coin_performance = {}        # Coin performans analizi
        self.market_correlation = {}      # Piyasa korelasyon analizi
        self.learning_insights = []       # Öğrenme sonuçları
        self.recommendations = {}         # Tavsiyeler
        
        # Ek analiz sonuçları
        self.common_mistakes = []         # Genel hatalar
        self.success_patterns = []        # Başarı desenleri
        self.time_based_patterns = {}     # Zamana dayalı desenler
        
        # Min-max değerler
        self.min_trades_for_analysis = 5  # Analiz için minimum işlem sayısı
        self.analysis_timeframe = 30      # Gün olarak analiz süresi
    
    def load_executed_trades(self) -> bool:
        """
        Tamamlanmış işlem kayıtlarını yükle
        
        Returns:
            bool: Başarılı mı?
        """
        try:
            if not os.path.exists(EXECUTED_TRADES_LOG):
                logger.warning(f"{EXECUTED_TRADES_LOG} dosyası bulunamadı!")
                return False
            
            with open(EXECUTED_TRADES_LOG, "r", encoding="utf-8") as f:
                self.trades_data = json.load(f)
            
            # Yapı kontrolü
            if not isinstance(self.trades_data, dict) or "trades" not in self.trades_data:
                logger.warning(f"{EXECUTED_TRADES_LOG} beklenen formatta değil!")
                return False
            
            trades_count = len(self.trades_data.get("trades", []))
            logger.info(f"{EXECUTED_TRADES_LOG} yüklendi: {trades_count} işlem kaydı")
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"{EXECUTED_TRADES_LOG} dosyası JSON hatası: {e}")
            return False
        except Exception as e:
            logger.error(f"{EXECUTED_TRADES_LOG} yüklenirken hata: {e}")
            return False
    
    def load_sentiment_data(self) -> bool:
        """
        Duygu analizi verilerini yükle
        
        Returns:
            bool: Başarılı mı?
        """
        try:
            if not os.path.exists(SENTIMENT_SUMMARY_FILE):
                logger.warning(f"{SENTIMENT_SUMMARY_FILE} dosyası bulunamadı!")
                return False
            
            with open(SENTIMENT_SUMMARY_FILE, "r", encoding="utf-8") as f:
                self.sentiment_data = json.load(f)
            
            logger.info(f"{SENTIMENT_SUMMARY_FILE} yüklendi")
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"{SENTIMENT_SUMMARY_FILE} dosyası JSON hatası: {e}")
            return False
        except Exception as e:
            logger.error(f"{SENTIMENT_SUMMARY_FILE} yüklenirken hata: {e}")
            return False
    
    def load_strategy_decisions(self) -> bool:
        """
        Strateji kararlarını yükle
        
        Returns:
            bool: Başarılı mı?
        """
        try:
            if not os.path.exists(STRATEGY_DECISION_FILE):
                logger.warning(f"{STRATEGY_DECISION_FILE} dosyası bulunamadı!")
                return False
            
            with open(STRATEGY_DECISION_FILE, "r", encoding="utf-8") as f:
                self.strategy_decisions = json.load(f)
            
            logger.info(f"{STRATEGY_DECISION_FILE} yüklendi")
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"{STRATEGY_DECISION_FILE} dosyası JSON hatası: {e}")
            return False
        except Exception as e:
            logger.error(f"{STRATEGY_DECISION_FILE} yüklenirken hata: {e}")
            return False
    
    def load_performance_data(self) -> bool:
        """
        Performans verilerini yükle
        
        Returns:
            bool: Başarılı mı?
        """
        try:
            if not os.path.exists(PERFORMANCE_REPORT_FILE):
                logger.warning(f"{PERFORMANCE_REPORT_FILE} dosyası bulunamadı!")
                return False
            
            with open(PERFORMANCE_REPORT_FILE, "r", encoding="utf-8") as f:
                self.performance_data = json.load(f)
            
            logger.info(f"{PERFORMANCE_REPORT_FILE} yüklendi")
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"{PERFORMANCE_REPORT_FILE} dosyası JSON hatası: {e}")
            return False
        except Exception as e:
            logger.error(f"{PERFORMANCE_REPORT_FILE} yüklenirken hata: {e}")
            return False
    
    def load_all_data(self) -> bool:
        """
        Tüm veri kaynaklarını yükle
        
        Returns:
            bool: En az bir kaynak başarıyla yüklendiyse True
        """
        # İşlem kaydı yükle (zorunlu)
        trades_loaded = self.load_executed_trades()
        
        if not trades_loaded:
            logger.error("İşlem kayıtları yüklenemedi! Öğrenme analizi yapılamaz.")
            return False
        
        # Diğer kaynakları yükle (opsiyonel)
        sentiment_loaded = self.load_sentiment_data()
        strategy_loaded = self.load_strategy_decisions()
        performance_loaded = self.load_performance_data()
        
        if not sentiment_loaded:
            logger.warning("Duyarlılık verileri yüklenemedi! Duyarlılık analizi yapılamayacak.")
            
        if not strategy_loaded:
            logger.warning("Strateji kararları yüklenemedi! Strateji analizi kısıtlı olacak.")
            
        if not performance_loaded:
            logger.warning("Performans verileri yüklenemedi! Performans analizi kısıtlı olacak.")
        
        return True
    
    def analyze_strategies(self) -> Dict[str, Any]:
        """
        Stratejilerin performansını analiz et
        
        Returns:
            Dict[str, Any]: Strateji performans analizi
        """
        logger.info("Strateji performansı analiz ediliyor...")
        
        strategy_performance = {}
        
        # Ticaret verilerini kontrol et
        if "trades" not in self.trades_data:
            logger.warning("İşlem verileri eksik, strateji analizi yapılamıyor!")
            return strategy_performance
        
        trades = self.trades_data["trades"]
        
        # Her strateji için analiz
        for strategy in STRATEGIES:
            # Strateji ile yapılan işlemleri filtrele
            strategy_trades = [trade for trade in trades if trade.get("strategy") == strategy]
            
            # Yeterli işlem var mı kontrol et
            if len(strategy_trades) < self.min_trades_for_analysis:
                logger.info(f"{strategy} için yeterli işlem yok ({len(strategy_trades)}), analiz atlanıyor.")
                continue
            
            # Tamamlanmış işlemleri filtrele (profit_loss değeri olan)
            completed_trades = [trade for trade in strategy_trades if "profit_loss" in trade and trade.get("status") == "CLOSED"]
            
            # Başarılı işlemleri say
            profitable_trades = [trade for trade in completed_trades if trade.get("profit_loss", 0) > 0]
            
            # Temel metrikleri hesapla
            total_trades = len(completed_trades)
            win_count = len(profitable_trades)
            loss_count = total_trades - win_count
            
            # Başarı oranı
            win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
            
            # Toplam ve ortalama kâr/zarar
            total_profit = sum(trade.get("profit_loss", 0) for trade in profitable_trades)
            total_loss = sum(abs(trade.get("profit_loss", 0)) for trade in completed_trades if trade.get("profit_loss", 0) <= 0)
            
            avg_profit = total_profit / win_count if win_count > 0 else 0
            avg_loss = total_loss / loss_count if loss_count > 0 else 0
            
            # Toplam yatırım ve ROI hesapla
            total_investment = sum(trade.get("investment_usd", 0) for trade in completed_trades)
            net_profit_loss = total_profit - total_loss
            roi = (net_profit_loss / total_investment * 100) if total_investment > 0 else 0
            
            # Risk/ödül oranı
            risk_reward_ratio = avg_profit / avg_loss if avg_loss > 0 else 0
            
            # Sonuçları kaydet
            strategy_performance[strategy] = {
                "total_trades": total_trades,
                "win_count": win_count,
                "loss_count": loss_count,
                "win_rate": round(win_rate, 2),
                "total_profit": round(total_profit, 2),
                "total_loss": round(total_loss, 2),
                "avg_profit": round(avg_profit, 2),
                "avg_loss": round(avg_loss, 2),
                "net_profit_loss": round(net_profit_loss, 2),
                "roi": round(roi, 2),
                "risk_reward_ratio": round(risk_reward_ratio, 2),
                "total_investment": round(total_investment, 2)
            }
            
            logger.info(f"{strategy} analizi tamamlandı: %{win_rate:.2f} başarı oranı, ${net_profit_loss:.2f} net kâr/zarar")
        
        return strategy_performance
    
    def analyze_coins(self) -> Dict[str, Any]:
        """
        Coin bazlı performansı analiz et
        
        Returns:
            Dict[str, Any]: Coin performans analizi
        """
        logger.info("Coin performansı analiz ediliyor...")
        
        coin_performance = {
            "coin_performance": {},       # Coin bazlı detaylı performans
            "best_coins": [],             # En iyi performans gösteren coinler
            "worst_coins": [],            # En kötü performans gösteren coinler
            "most_traded_coins": []       # En çok işlem yapılan coinler
        }
        
        # Ticaret verilerini kontrol et
        if "trades" not in self.trades_data:
            logger.warning("İşlem verileri eksik, coin analizi yapılamıyor!")
            return coin_performance
        
        trades = self.trades_data["trades"]
        
        # Coin'lere göre işlemleri grupla
        coin_trades = defaultdict(list)
        
        for trade in trades:
            symbol = trade.get("symbol", "UNKNOWN")
            if symbol != "UNKNOWN":
                coin_trades[symbol].append(trade)
        
        # Her coin için analiz
        for coin, coin_trade_list in coin_trades.items():
            # Yeterli işlem var mı kontrol et
            if len(coin_trade_list) < self.min_trades_for_analysis:
                logger.info(f"{coin} için yeterli işlem yok ({len(coin_trade_list)}), analiz atlanıyor.")
                continue
            
            # Tamamlanmış işlemleri filtrele
            completed_trades = [trade for trade in coin_trade_list if "profit_loss" in trade and trade.get("status") == "CLOSED"]
            
            # Başarılı işlemleri say
            profitable_trades = [trade for trade in completed_trades if trade.get("profit_loss", 0) > 0]
            
            # Temel metrikleri hesapla
            total_trades = len(completed_trades)
            win_count = len(profitable_trades)
            loss_count = total_trades - win_count
            
            # Başarı oranı
            win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
            
            # Toplam ve ortalama kâr/zarar
            total_profit = sum(trade.get("profit_loss", 0) for trade in profitable_trades)
            total_loss = sum(abs(trade.get("profit_loss", 0)) for trade in completed_trades if trade.get("profit_loss", 0) <= 0)
            
            avg_profit = total_profit / win_count if win_count > 0 else 0
            avg_loss = total_loss / loss_count if loss_count > 0 else 0
            
            # Toplam yatırım ve ROI hesapla
            total_investment = sum(trade.get("investment_usd", 0) for trade in completed_trades)
            net_profit_loss = total_profit - total_loss
            roi = (net_profit_loss / total_investment * 100) if total_investment > 0 else 0
            
            # Strateji bazlı başarı oranları
            strategy_success = {}
            for strategy in STRATEGIES:
                strategy_coin_trades = [t for t in completed_trades if t.get("strategy") == strategy]
                if strategy_coin_trades:
                    strategy_win_count = len([t for t in strategy_coin_trades if t.get("profit_loss", 0) > 0])
                    strategy_win_rate = (strategy_win_count / len(strategy_coin_trades) * 100)
                    strategy_success[strategy] = {
                        "trades": len(strategy_coin_trades),
                        "win_rate": round(strategy_win_rate, 2)
                    }
            
            # Sonuçları kaydet
            coin_performance["coin_performance"][coin] = {
                "total_trades": total_trades,
                "win_count": win_count,
                "loss_count": loss_count,
                "win_rate": round(win_rate, 2),
                "total_profit": round(total_profit, 2),
                "total_loss": round(total_loss, 2),
                "avg_profit": round(avg_profit, 2),
                "avg_loss": round(avg_loss, 2),
                "net_profit_loss": round(net_profit_loss, 2),
                "roi": round(roi, 2),
                "total_investment": round(total_investment, 2),
                "strategy_performance": strategy_success
            }
            
            logger.info(f"{coin} analizi tamamlandı: %{win_rate:.2f} başarı oranı, ${net_profit_loss:.2f} net kâr/zarar")
        
        # En iyi ve en kötü coinleri belirle
        if coin_performance["coin_performance"]:
            # ROI'ye göre sırala
            coin_roi_list = [(coin, data["roi"]) for coin, data in coin_performance["coin_performance"].items()]
            
            # En iyi 5 coin
            best_coins = sorted(coin_roi_list, key=lambda x: x[1], reverse=True)[:5]
            coin_performance["best_coins"] = [coin for coin, _ in best_coins]
            
            # En kötü 5 coin
            worst_coins = sorted(coin_roi_list, key=lambda x: x[1])[:5]
            coin_performance["worst_coins"] = [coin for coin, _ in worst_coins]
            
            # En çok işlem yapılan 5 coin
            trade_volume = [(coin, data["total_trades"]) for coin, data in coin_performance["coin_performance"].items()]
            most_traded = sorted(trade_volume, key=lambda x: x[1], reverse=True)[:5]
            coin_performance["most_traded_coins"] = [coin for coin, _ in most_traded]
        
        return coin_performance
    
    def analyze_market_correlation(self) -> Dict[str, Any]:
        """
        Piyasa koşulları ve stratejiler/coinler arasındaki ilişkiyi analiz et
        
        Returns:
            Dict[str, Any]: Piyasa korelasyon analizi
        """
        logger.info("Piyasa korelasyonu analiz ediliyor...")
        
        market_correlation = {
            "market_conditions": {},       # Piyasa koşulları analizi
            "strategy_market_matrix": {},  # Strateji-piyasa koşulu matrisi
            "coin_market_matrix": {}       # Coin-piyasa koşulu matrisi
        }
        
        # Veri yeterliliğini kontrol et
        if "trades" not in self.trades_data or not self.sentiment_data or not self.strategy_decisions:
            logger.warning("Piyasa korelasyonu için yeterli veri yok, analiz kısıtlı olacak.")
            return market_correlation
        
        trades = self.trades_data["trades"]
        
        # Koşul bazlı işlemleri grupla
        condition_trades = defaultdict(list)
        
        # İşlem tarihlerinde piyasa koşulunu belirle
        for trade in trades:
            if "status" not in trade or trade["status"] != "CLOSED" or "profit_loss" not in trade:
                continue
                
            trade_timestamp = trade.get("timestamp", "")
            market_condition = self._get_market_condition_at_time(trade_timestamp)
            
            if market_condition:
                condition_trades[market_condition].append(trade)
        
        # Her piyasa koşulu için başarı oranı analizi
        for condition, condition_trade_list in condition_trades.items():
            if len(condition_trade_list) < self.min_trades_for_analysis:
                continue
            
            # Başarılı işlemleri say
            profitable_trades = [t for t in condition_trade_list if t.get("profit_loss", 0) > 0]
            
            # Başarı oranı ve ortalama kâr/zarar
            total_trades = len(condition_trade_list)
            win_count = len(profitable_trades)
            win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
            
            avg_profit_loss = sum(t.get("profit_loss", 0) for t in condition_trade_list) / total_trades if total_trades > 0 else 0
            
            # Piyasa koşulu analizini kaydet
            market_correlation["market_conditions"][condition] = {
                "total_trades": total_trades,
                "win_count": win_count,
                "win_rate": round(win_rate, 2),
                "avg_profit_loss": round(avg_profit_loss, 2)
            }
            
            logger.info(f"{condition} piyasa koşulunda %{win_rate:.2f} başarı oranı ({total_trades} işlem)")
        
        # Strateji-piyasa koşulu matrisi oluştur
        for strategy in STRATEGIES:
            market_correlation["strategy_market_matrix"][strategy] = {}
            
            for condition in condition_trades.keys():
                # Bu strateji + koşul kombinasyonundaki işlemleri filtrele
                strategy_condition_trades = [t for t in condition_trades[condition] if t.get("strategy") == strategy]
                
                if len(strategy_condition_trades) < 3:  # Minimum 3 işlem olsun
                    continue
                
                # Başarı oranını hesapla
                win_count = len([t for t in strategy_condition_trades if t.get("profit_loss", 0) > 0])
                total = len(strategy_condition_trades)
                win_rate = (win_count / total * 100) if total > 0 else 0
                
                # Matrisi güncelle
                market_correlation["strategy_market_matrix"][strategy][condition] = {
                    "win_rate": round(win_rate, 2),
                    "total_trades": total
                }
        
        # Coin-piyasa koşulu matrisi oluştur
        coin_list = set(trade.get("symbol", "") for trade in trades if trade.get("symbol"))
        
        for coin in coin_list:
            market_correlation["coin_market_matrix"][coin] = {}
            
            for condition in condition_trades.keys():
                # Bu coin + koşul kombinasyonundaki işlemleri filtrele
                coin_condition_trades = [t for t in condition_trades[condition] if t.get("symbol") == coin]
                
                if len(coin_condition_trades) < 3:  # Minimum 3 işlem olsun
                    continue
                
                # Başarı oranını hesapla
                win_count = len([t for t in coin_condition_trades if t.get("profit_loss", 0) > 0])
                total = len(coin_condition_trades)
                win_rate = (win_count / total * 100) if total > 0 else 0
                
                # Matrisi güncelle
                market_correlation["coin_market_matrix"][coin][condition] = {
                    "win_rate": round(win_rate, 2),
                    "total_trades": total
                }
        
        return market_correlation
    
    def _get_market_condition_at_time(self, timestamp: str) -> Optional[str]:
        """
        Belirli bir zamandaki piyasa koşulunu belirle
        
        Args:
            timestamp (str): İşlem zamanı
            
        Returns:
            Optional[str]: Piyasa koşulu veya None
        """
        if not timestamp or not self.strategy_decisions or "strategy_decision" not in self.strategy_decisions:
            return None
        
        # Varsayılan olarak "overall" sentiment verilerinden al
        if self.sentiment_data and "overall" in self.sentiment_data:
            return self.sentiment_data["overall"].get("sentiment", "Neutral")
        
        # Strateji kararından al
        if "market_conditions" in self.strategy_decisions["strategy_decision"]:
            return self.strategy_decisions["strategy_decision"]["market_conditions"].get("trend", "Neutral")
        
        return "Neutral"  # Varsayılan
    
    def analyze_common_mistakes(self) -> List[Tuple[str, int]]:
        """
        Sık yapılan hataları analiz et
        
        Returns:
            List[Tuple[str, int]]: Hata ve sıklık
        """
        logger.info("Genel hatalar analiz ediliyor...")
        
        mistakes = [
            ("Yüksek volatilite dönemlerinde aşırı işlem", 0),
            ("TP hedeflerinin çok yakın belirlenmesi", 0),
            ("SL hedeflerinin çok uzak belirlenmesi", 0),
            ("Aşırı büyük pozisyon boyutu", 0),
            ("Aşırı düşük pozisyon boyutu", 0),
            ("Trend dönüş noktasında giriş yapma", 0),
            ("RSI aşırı alım/satım bölgelerine dikkat etmeme", 0),
            ("Piyasa koşullarına uymayan strateji seçimi", 0),
            ("Yüksek korelasyonlu paralel işlemler", 0),
            ("Kısa profit hedefi için kâr alma", 0)
        ]
        
        # İşlem verilerini kontrol et
        if "trades" not in self.trades_data:
            logger.warning("İşlem verileri eksik, hata analizi yapılamıyor!")
            return mistakes
        
        trades = self.trades_data["trades"]
        
        # Kapatılmış işlemleri filtrele
        closed_trades = [t for t in trades if t.get("status") == "CLOSED" and "profit_loss" in t]
        
        # Zararda biten işlemleri analiz et
        loss_trades = [t for t in closed_trades if t.get("profit_loss", 0) < 0]
        
        for trade in loss_trades:
            # İndikatörleri kontrol et
            indicators = trade.get("indicators", {})
            
            # Hata 1: Yüksek volatilite dönemlerinde aşırı işlem
            if indicators.get("volatility", "Medium") == "High":
                mistakes[0] = (mistakes[0][0], mistakes[0][1] + 1)
            
            # Hata 2 & 3: TP/SL hedeflerinin uygunsuz belirlenmesi
            if "entry_price" in trade and "take_profit_price" in trade and "stop_loss_price" in trade:
                entry = float(trade.get("entry_price", 0))
                tp = float(trade.get("take_profit_price", 0))
                sl = float(trade.get("stop_loss_price", 0))
                
                # Long pozisyon için
                if trade.get("operation") == "LONG":
                    # TP çok yakın
                    if 0 < (tp - entry) / entry < 0.01:  # %1'den az
                        mistakes[1] = (mistakes[1][0], mistakes[1][1] + 1)
                    
                    # SL çok uzak
                    if 0 < (entry - sl) / entry > 0.05:  # %5'ten fazla
                        mistakes[2] = (mistakes[2][0], mistakes[2][1] + 1)
                
                # Short pozisyon için
                elif trade.get("operation") == "SHORT":
                    # TP çok yakın
                    if 0 < (entry - tp) / entry < 0.01:  # %1'den az
                        mistakes[1] = (mistakes[1][0], mistakes[1][1] + 1)
                    
                    # SL çok uzak
                    if 0 < (sl - entry) / entry > 0.05:  # %5'ten fazla
                        mistakes[2] = (mistakes[2][0], mistakes[2][1] + 1)
            
            # Hata 4 & 5: Pozisyon boyutu
            investment = trade.get("investment_usd", 0)
            if investment > 2000:  # Çok büyük pozisyon
                mistakes[3] = (mistakes[3][0], mistakes[3][1] + 1)
            elif investment < 100:  # Çok küçük pozisyon
                mistakes[4] = (mistakes[4][0], mistakes[4][1] + 1)
            
            # Hata 6: Trend dönüş noktasında giriş
            if indicators.get("trend_change", False):
                mistakes[5] = (mistakes[5][0], mistakes[5][1] + 1)
            
            # Hata 7: RSI aşırı alım/satım
            rsi = indicators.get("rsi", 50)
            if (trade.get("operation") == "LONG" and rsi > 70) or (trade.get("operation") == "SHORT" and rsi < 30):
                mistakes[6] = (mistakes[6][0], mistakes[6][1] + 1)
            
            # Hata 8: Piyasa koşullarına uymayan strateji
            market_trend = self._get_market_condition_at_time(trade.get("timestamp", ""))
            if (market_trend == "Bullish" and trade.get("strategy") == "short_strategy.py") or \
               (market_trend == "Bearish" and trade.get("strategy") == "long_strategy.py"):
                mistakes[7] = (mistakes[7][0], mistakes[7][1] + 1)
        
        # Hata 9: Yüksek korelasyonlu paralel işlemler
        # Son 1 saat içinde aynı yönde birden fazla işlem
        for i, trade1 in enumerate(closed_trades):
            trade1_time = datetime.datetime.strptime(trade1.get("timestamp", "2000-01-01 00:00:00"), "%Y-%m-%d %H:%M:%S")
            
            for trade2 in closed_trades[i+1:]:
                trade2_time = datetime.datetime.strptime(trade2.get("timestamp", "2000-01-01 00:00:00"), "%Y-%m-%d %H:%M:%S")
                
                # 1 saat içinde mi?
                if abs((trade1_time - trade2_time).total_seconds()) <= 3600:
                    # Aynı yönde mi?
                    if trade1.get("operation") == trade2.get("operation"):
                        mistakes[8] = (mistakes[8][0], mistakes[8][1] + 1)
                        break
        
        # Hata 10: Kısa profit hedefi için kâr alma
        for trade in closed_trades:
            if trade.get("profit_loss", 0) > 0 and trade.get("profit_loss", 0) < 5:  # $5'dan az kâr
                mistakes[9] = (mistakes[9][0], mistakes[9][1] + 1)
        
        # Hataları sıklığa göre sırala
        sorted_mistakes = sorted(mistakes, key=lambda x: x[1], reverse=True)
        
        # Sadece sıklığı 0'dan büyük olanları döndür
        return [m for m in sorted_mistakes if m[1] > 0]
    
    def analyze_success_patterns(self) -> List[str]:
        """
        Başarılı işlemlerdeki ortak desenleri analiz et
        
        Returns:
            List[str]: Başarı desenleri
        """
        logger.info("Başarı desenleri analiz ediliyor...")
        
        patterns = []
        
        # İşlem verilerini kontrol et
        if "trades" not in self.trades_data:
            logger.warning("İşlem verileri eksik, başarı deseni analizi yapılamıyor!")
            return patterns
        
        trades = self.trades_data["trades"]
        
        # Kârlı işlemleri filtrele
        profitable_trades = [t for t in trades if t.get("status") == "CLOSED" and t.get("profit_loss", 0) > 0]
        
        if len(profitable_trades) < self.min_trades_for_analysis:
            logger.warning(f"Başarı deseni analizi için yeterli kârlı işlem yok ({len(profitable_trades)})")
            return patterns
        
        # En başarılı stratejileri bul
        strategy_profits = defaultdict(int)
        for trade in profitable_trades:
            strategy_profits[trade.get("strategy", "unknown")] += trade.get("profit_loss", 0)
        
        best_strategy = max(strategy_profits.items(), key=lambda x: x[1])[0] if strategy_profits else None
        
        if best_strategy:
            patterns.append(f"En iyi performans gösteren strateji: {best_strategy}")
        
        # En başarılı coin'leri bul
        coin_profits = defaultdict(int)
        for trade in profitable_trades:
            coin_profits[trade.get("symbol", "unknown")] += trade.get("profit_loss", 0)
        
        best_coins = sorted(coin_profits.items(), key=lambda x: x[1], reverse=True)[:3]
        
        if best_coins:
            coin_list = ", ".join([coin for coin, _ in best_coins])
            patterns.append(f"En iyi performans gösteren coinler: {coin_list}")
        
        # Piyasa koşulları analizi
        market_condition_trades = defaultdict(list)
        for trade in profitable_trades:
            condition = self._get_market_condition_at_time(trade.get("timestamp", ""))
            if condition:
                market_condition_trades[condition].append(trade)
        
        best_condition = max(market_condition_trades.items(), key=lambda x: len(x[1]))[0] if market_condition_trades else None
        
        if best_condition:
            patterns.append(f"En başarılı piyasa koşulu: {best_condition}")
        
        # Giriş zamanlaması analizi
        hour_profits = defaultdict(float)
        weekday_profits = defaultdict(float)
        
        for trade in profitable_trades:
            if "timestamp" in trade:
                try:
                    trade_time = datetime.datetime.strptime(trade["timestamp"], "%Y-%m-%d %H:%M:%S")
                    hour_profits[trade_time.hour] += trade.get("profit_loss", 0)
                    weekday_profits[trade_time.weekday()] += trade.get("profit_loss", 0)
                except:
                    pass
        
        best_hour = max(hour_profits.items(), key=lambda x: x[1])[0] if hour_profits else None
        best_weekday = max(weekday_profits.items(), key=lambda x: x[1])[0] if weekday_profits else None
        
        weekday_names = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
        
        if best_hour is not None:
            patterns.append(f"En başarılı işlem saati: {best_hour}:00")
        
        if best_weekday is not None:
            patterns.append(f"En başarılı işlem günü: {weekday_names[best_weekday]}")
        
        # TP/SL oranları analizi
        tp_sl_ratios = []
        
        for trade in profitable_trades:
            if all(k in trade for k in ["entry_price", "take_profit_price", "stop_loss_price"]):
                entry = float(trade["entry_price"])
                tp = float(trade["take_profit_price"])
                sl = float(trade["stop_loss_price"])
                
                # Long pozisyon için
                if trade.get("operation") == "LONG":
                    tp_distance = (tp - entry) / entry
                    sl_distance = (entry - sl) / entry
                # Short pozisyon için
                elif trade.get("operation") == "SHORT":
                    tp_distance = (entry - tp) / entry
                    sl_distance = (sl - entry) / entry
                else:
                    continue
                
                if sl_distance > 0:  # 0'a bölme hatasını önle
                    tp_sl_ratio = tp_distance / sl_distance
                    tp_sl_ratios.append(tp_sl_ratio)
        
        if tp_sl_ratios:
            avg_ratio = sum(tp_sl_ratios) / len(tp_sl_ratios)
            patterns.append(f"Optimal TP/SL oranı: {avg_ratio:.2f}")
        
        # Pozisyon boyutu analizi
        position_sizes = [trade.get("investment_usd", 0) for trade in profitable_trades if "investment_usd" in trade]
        
        if position_sizes:
            avg_size = sum(position_sizes) / len(position_sizes)
            patterns.append(f"Optimal pozisyon boyutu: ${avg_size:.2f}")
        
        return patterns
    
    def generate_insights(self) -> List[str]:
        """
        Öğrenme analizi sonucunda genel içgörüler oluştur
        
        Returns:
            List[str]: İçgörü listesi
        """
        logger.info("Öğrenme içgörüleri oluşturuluyor...")
        
        insights = []
        
        # Strateji analizi
        if self.strategy_performance:
            # En başarılı strateji
            best_strategy = max(self.strategy_performance.items(), key=lambda x: x[1].get("win_rate", 0))
            best_strat_name = best_strategy[0]
            best_strat_data = best_strategy[1]
            
            insights.append(f"{best_strat_name} en yüksek başarı oranına sahip (%{best_strat_data.get('win_rate', 0)}) ve ${best_strat_data.get('net_profit_loss', 0):.2f} net kâr üretti.")
            
            # En yüksek ROI
            best_roi_strategy = max(self.strategy_performance.items(), key=lambda x: x[1].get("roi", 0))
            best_roi_name = best_roi_strategy[0]
            best_roi_data = best_roi_strategy[1]
            
            if best_roi_name != best_strat_name:
                insights.append(f"{best_roi_name} en yüksek ROI'ye sahip (%{best_roi_data.get('roi', 0):.2f}) ve yatırımın en iyi geri dönüşünü sağladı.")
            
            # Risk/ödül oranları
            best_rr_strategy = max(self.strategy_performance.items(), key=lambda x: x[1].get("risk_reward_ratio", 0))
            best_rr_name = best_rr_strategy[0]
            best_rr_data = best_rr_strategy[1]
            
            insights.append(f"{best_rr_name} en iyi risk/ödül oranına sahip ({best_rr_data.get('risk_reward_ratio', 0):.2f}) ve her $1 risk için ${best_rr_data.get('risk_reward_ratio', 0):.2f} kazanç sağladı.")
        
        # Coin analizi
        if self.coin_performance and "coin_performance" in self.coin_performance and "best_coins" in self.coin_performance:
            best_coins = self.coin_performance["best_coins"]
            
            if best_coins:
                insights.append(f"En iyi performans gösteren coinler: {', '.join(best_coins[:3])}. Bu coinlerde ağırlıklı işlem yapılmalı.")
            
            worst_coins = self.coin_performance.get("worst_coins", [])
            
            if worst_coins:
                insights.append(f"En kötü performans gösteren coinler: {', '.join(worst_coins[:3])}. Bu coinlerde daha dikkatli olunmalı veya kaçınılmalı.")
        
        # Piyasa korelasyonu
        if self.market_correlation and "market_conditions" in self.market_correlation:
            market_conditions = self.market_correlation["market_conditions"]
            
            best_condition = max(market_conditions.items(), key=lambda x: x[1].get("win_rate", 0)) if market_conditions else None
            
            if best_condition:
                cond_name, cond_data = best_condition
                insights.append(f"{cond_name} piyasa koşullarında en yüksek başarı oranı elde edildi (%{cond_data.get('win_rate', 0):.2f}). Bu koşullarda daha agresif işlem yapılabilir.")
            
            # Strateji-piyasa matrisi
            if "strategy_market_matrix" in self.market_correlation:
                strategy_market = self.market_correlation["strategy_market_matrix"]
                
                # Her strateji için en iyi piyasa koşulu
                for strategy, conditions in strategy_market.items():
                    if conditions:
                        best_cond = max(conditions.items(), key=lambda x: x[1].get("win_rate", 0))
                        cond_name, cond_data = best_cond
                        
                        insights.append(f"{strategy} stratejisi {cond_name} piyasa koşullarında en iyi sonuçları veriyor (%{cond_data.get('win_rate', 0):.2f} başarı oranı).")
        
        # Sık yapılan hatalar
        if self.common_mistakes:
            top_mistake = self.common_mistakes[0] if self.common_mistakes else None
            
            if top_mistake:
                insights.append(f"En sık yapılan hata: {top_mistake[0]} ({top_mistake[1]} kez). Bu hatadan kaçınılması başarı oranını artırabilir.")
        
        # Başarı desenleri
        if self.success_patterns:
            for pattern in self.success_patterns[:3]:
                insights.append(pattern)
        
        # Zamana bağlı desenler
        if self.time_based_patterns:
            best_hour = self.time_based_patterns.get("best_hour")
            worst_hour = self.time_based_patterns.get("worst_hour")
            
            if best_hour is not None and worst_hour is not None:
                insights.append(f"En başarılı işlem saati {best_hour}:00, en başarısız işlem saati {worst_hour}:00. İşlem zamanlaması buna göre ayarlanabilir.")
        
        # Eksik boyutları kontrol et ve daha genel öneriler ekle
        if len(insights) < 5:
            insights.append("Daha yüksek başarı için işlem hacminin artırılması ve daha fazla veri toplanması gerekmektedir.")
            insights.append("Zarar durdurma (stop-loss) seviyelerinin sıkılaştırılması toplam riski azaltabilir.")
            insights.append("Daha uzun pozisyon tutma süreleri ortalama kârı artırabilir.")
        
        return insights
    
    def generate_strategy_recommendations(self) -> Dict[str, Any]:
        """
        Strateji bazlı öneriler oluştur
        
        Returns:
            Dict[str, Any]: Strateji önerileri
        """
        logger.info("Strateji önerileri oluşturuluyor...")
        
        recommendations = {}
        
        # Strateji performansı yoksa öneriler oluşturulamaz
        if not self.strategy_performance:
            logger.warning("Strateji performansı verisi yok, öneriler oluşturulamıyor!")
            return recommendations
        
        # Her strateji için öneriler oluştur
        for strategy, perf in self.strategy_performance.items():
            # Performans özeti
            win_rate = perf.get("win_rate", 0)
            roi = perf.get("roi", 0)
            risk_reward = perf.get("risk_reward_ratio", 0)
            
            perf_summary = f"Başarı oranı: %{win_rate}, ROI: %{roi:.2f}, Risk/Ödül: {risk_reward:.2f}"
            
            # Öneriler listesi
            strategy_recs = []
            
            # Geliştirme önerileri
            if win_rate < 50:
                strategy_recs.append(f"Başarı oranını artırmak için giriş kriterlerini sıkılaştırın veya sinyal doğrulama ekleyin.")
            
            if roi < 10:
                strategy_recs.append(f"ROI'yi artırmak için kâr hedeflerini optimize edin.")
            
            if risk_reward < 1.5:
                strategy_recs.append(f"Risk/ödül oranını artırmak için SL seviyelerini sıkılaştırın veya TP seviyelerini genişletin.")
            
            # Piyasa koşullarına göre öneriler
            if self.market_correlation and "strategy_market_matrix" in self.market_correlation:
                strategy_market = self.market_correlation["strategy_market_matrix"].get(strategy, {})
                
                if strategy_market:
                    # En iyi ve en kötü koşullar
                    best_condition = max(strategy_market.items(), key=lambda x: x[1].get("win_rate", 0)) if strategy_market else None
                    worst_condition = min(strategy_market.items(), key=lambda x: x[1].get("win_rate", 0)) if strategy_market else None
                    
                    if best_condition:
                        best_cond_name, best_cond_data = best_condition
                        strategy_recs.append(f"{best_cond_name} piyasa koşullarında daha aktif işlem yapılabilir (%{best_cond_data.get('win_rate', 0):.2f} başarı oranı).")
                    
                    if worst_condition:
                        worst_cond_name, worst_cond_data = worst_condition
                        strategy_recs.append(f"{worst_cond_name} piyasa koşullarında dikkatli olunmalı veya işlem yapılmamalı (%{worst_cond_data.get('win_rate', 0):.2f} başarı oranı).")
            
            # Başarılı coinler
            if self.coin_performance and "coin_performance" in self.coin_performance:
                strategy_coins = []
                
                for coin, coin_data in self.coin_performance["coin_performance"].items():
                    if "strategy_performance" in coin_data and strategy in coin_data["strategy_performance"]:
                        strat_perf = coin_data["strategy_performance"][strategy]
                        if strat_perf.get("win_rate", 0) > 60:  # %60'tan fazla başarı
                            strategy_coins.append((coin, strat_perf.get("win_rate", 0)))
                
                # En iyi 3 coin
                best_strategy_coins = sorted(strategy_coins, key=lambda x: x[1], reverse=True)[:3]
                
                if best_strategy_coins:
                    coin_list = ", ".join([f"{coin} (%{win_rate:.2f})" for coin, win_rate in best_strategy_coins])
                    strategy_recs.append(f"Bu strateji için en uygun coinler: {coin_list}")
            
            # Tüm öneri sonuçlarını kaydet
            recommendations[strategy] = {
                "performance_summary": perf_summary,
                "recommendations": strategy_recs
            }
        
        return recommendations
    
    def generate_time_based_patterns(self) -> Dict[str, Any]:
        """
        Zamana bağlı başarı desenlerini analiz et
        
        Returns:
            Dict[str, Any]: Zamana bağlı desenler
        """
        logger.info("Zamana bağlı desenler analiz ediliyor...")
        
        time_patterns = {
            "hourly_performance": {},
            "daily_performance": {},
            "best_hour": None,
            "worst_hour": None,
            "best_day": None,
            "worst_day": None
        }
        
        # İşlem verilerini kontrol et
        if "trades" not in self.trades_data:
            logger.warning("İşlem verileri eksik, zaman analizi yapılamıyor!")
            return time_patterns
        
        trades = self.trades_data["trades"]
        
        # Tamamlanmış işlemleri filtrele
        completed_trades = [t for t in trades if t.get("status") == "CLOSED" and "profit_loss" in t]
        
        if len(completed_trades) < self.min_trades_for_analysis:
            logger.warning(f"Zaman analizi için yeterli işlem yok ({len(completed_trades)})")
            return time_patterns
        
        # Saatlik ve günlük performans
        hourly_trades = defaultdict(list)
        daily_trades = defaultdict(list)
        
        for trade in completed_trades:
            if "timestamp" in trade:
                try:
                    trade_time = datetime.datetime.strptime(trade["timestamp"], "%Y-%m-%d %H:%M:%S")
                    hour = trade_time.hour
                    day = trade_time.weekday()  # 0 = Pazartesi, 6 = Pazar
                    
                    hourly_trades[hour].append(trade)
                    daily_trades[day].append(trade)
                except:
                    continue
        
        # Saatlik performans analizi
        for hour, hour_trades in hourly_trades.items():
            if len(hour_trades) < 3:  # En az 3 işlem olsun
                continue
            
            win_count = len([t for t in hour_trades if t.get("profit_loss", 0) > 0])
            total = len(hour_trades)
            win_rate = (win_count / total * 100) if total > 0 else 0
            
            avg_profit = sum(t.get("profit_loss", 0) for t in hour_trades) / total if total > 0 else 0
            
            time_patterns["hourly_performance"][hour] = {
                "win_rate": round(win_rate, 2),
                "avg_profit": round(avg_profit, 2),
                "total_trades": total
            }
        
        # Günlük performans analizi
        weekday_names = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
        
        for day, day_trades in daily_trades.items():
            if len(day_trades) < 3:  # En az 3 işlem olsun
                continue
            
            win_count = len([t for t in day_trades if t.get("profit_loss", 0) > 0])
            total = len(day_trades)
            win_rate = (win_count / total * 100) if total > 0 else 0
            
            avg_profit = sum(t.get("profit_loss", 0) for t in day_trades) / total if total > 0 else 0
            
            time_patterns["daily_performance"][weekday_names[day]] = {
                "win_rate": round(win_rate, 2),
                "avg_profit": round(avg_profit, 2),
                "total_trades": total
            }
        
        # En iyi ve en kötü saatler
        if time_patterns["hourly_performance"]:
            best_hour = max(time_patterns["hourly_performance"].items(), key=lambda x: x[1]["win_rate"])
            worst_hour = min(time_patterns["hourly_performance"].items(), key=lambda x: x[1]["win_rate"])
            
            time_patterns["best_hour"] = int(best_hour[0])
            time_patterns["worst_hour"] = int(worst_hour[0])
        
        # En iyi ve en kötü günler
        if time_patterns["daily_performance"]:
            best_day = max(time_patterns["daily_performance"].items(), key=lambda x: x[1]["win_rate"])
            worst_day = min(time_patterns["daily_performance"].items(), key=lambda x: x[1]["win_rate"])
            
            time_patterns["best_day"] = best_day[0]
            time_patterns["worst_day"] = worst_day[0]
        
        return time_patterns
    
    def create_learning_summary(self) -> Dict[str, Any]:
        """
        Öğrenme sonuçlarının özet raporunu oluştur
        
        Returns:
            Dict[str, Any]: Öğrenme özeti
        """
        logger.info("Öğrenme özeti oluşturuluyor...")
        
        summary = {
            "timestamp": CURRENT_TIME,
            "user": CURRENT_USER,
            "analysis_range": "30 days",  # Varsayılan analiz süresi
            "data_sources": {
                "trades_log": bool(self.trades_data),
                "sentiment_data": bool(self.sentiment_data),
                "strategy_decisions": bool(self.strategy_decisions),
                "performance_data": bool(self.performance_data)
            },
            "strategy_performance": self.strategy_performance,
            "coin_performance": self.coin_performance,
            "market_correlation": self.market_correlation,
            "time_patterns": self.time_based_patterns,
            "common_mistakes": self.common_mistakes,
            "success_patterns": self.success_patterns,
            "general_insights": self.learning_insights,
            "strategy_recommendations": self.recommendations
        }
        
        return summary
    
    def create_learning_feedback(self) -> Dict[str, Any]:
        """
        Strateji motoruna geri besleme sağlayacak özet oluştur
        
        Returns:
            Dict[str, Any]: Strateji geri beslemesi
        """
        logger.info("Strateji geri beslemesi oluşturuluyor...")
        
        feedback = {
            "timestamp": CURRENT_TIME,
            "user": CURRENT_USER,
            "priority_strategies": [],
            "priority_coins": [],
            "avoid_coins": [],
            "market_insights": [],
            "recommended_changes": []
        }
        
        # Öncelikli stratejiler
        if self.strategy_performance:
            # Win rate ve ROI kombinasyonu ile önceliklendir
            strategy_scores = {}
            
            for strategy, perf in self.strategy_performance.items():
                win_rate = perf.get("win_rate", 0)
                roi = perf.get("roi", 0)
                
                # Win rate ve ROI birleşiminden skor oluştur
                score = (win_rate * 0.7) + (roi * 0.3)  # %70 win rate, %30 ROI ağırlığı
                strategy_scores[strategy] = score
            
            # En yüksek skorlu stratejileri seç
            sorted_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
            feedback["priority_strategies"] = [s[0] for s in sorted_strategies]
        
        # Öncelikli ve kaçınılması gereken coinler
        if self.coin_performance and "coin_performance" in self.coin_performance:
            # ROI bazlı sıralama
            coin_roi = {}
            
            for coin, perf in self.coin_performance["coin_performance"].items():
                roi = perf.get("roi", 0)
                coin_roi[coin] = roi
            
            # En iyi ve en kötü coinleri belirle
            sorted_coins = sorted(coin_roi.items(), key=lambda x: x[1], reverse=True)
            
            # En iyi 5 coin
            feedback["priority_coins"] = [c[0] for c in sorted_coins[:5]]
            
            # En kötü 3 coin
            worst_coins = [c[0] for c in sorted_coins[-3:] if c[1] < 0]  # Sadece zararda olanları ekle
            feedback["avoid_coins"] = worst_coins
        
        # Piyasa içgörüleri
        if self.market_correlation and "market_conditions" in self.market_correlation:
            market_conditions = self.market_correlation["market_conditions"]
            
            for condition, data in market_conditions.items():
                win_rate = data.get("win_rate", 0)
                insight = f"{condition} piyasasında başarı oranı: %{win_rate:.2f}"
                feedback["market_insights"].append(insight)
        
        # Önerilen değişiklikler (genel içgörülerden)
        if self.learning_insights:
            for insight in self.learning_insights:
                if "artırmak için" in insight or "azaltmak için" in insight or "optimize" in insight:
                    feedback["recommended_changes"].append(insight)
        
        # Sık yapılan hatalardan öneriler
        if self.common_mistakes:
            for mistake, count in self.common_mistakes[:3]:
                change = f"'{mistake}' hatasını düzeltmek için strateji parametrelerini ayarlayın."
                feedback["recommended_changes"].append(change)
        
        return feedback
    
    def save_learning_summary(self) -> bool:
        """
        Öğrenme özetini dosyaya kaydet
        
        Returns:
            bool: Başarılı mı?
        """
        try:
            with open(LEARNING_SUMMARY_FILE, "w", encoding="utf-8") as f:
                json.dump(self.create_learning_summary(), f, ensure_ascii=False, indent=4)
            
            logger.info(f"Öğrenme özeti {LEARNING_SUMMARY_FILE} dosyasına kaydedildi")
            return True
            
        except Exception as e:
            logger.error(f"Öğrenme özeti kaydedilirken hata: {e}")
            return False
    
    def save_learning_feedback(self) -> bool:
        """
        Strateji geri beslemesini dosyaya kaydet
        
        Returns:
            bool: Başarılı mı?
        """
        try:
            with open(LEARNING_FEEDBACK_FILE, "w", encoding="utf-8") as f:
                json.dump(self.create_learning_feedback(), f, ensure_ascii=False, indent=4)
            
            logger.info(f"Strateji geri beslemesi {LEARNING_FEEDBACK_FILE} dosyasına kaydedildi")
            return True
            
        except Exception as e:
            logger.error(f"Strateji geri beslemesi kaydedilirken hata: {e}")
            return False
    
    def analyze_all(self) -> bool:
        """
        Tüm analiz süreçlerini sırayla çalıştır
        
        Returns:
            bool: Başarılı mı?
        """
        try:
            # Strateji performansı analizi
            self.strategy_performance = self.analyze_strategies()
            
            # Coin performansı analizi
            self.coin_performance = self.analyze_coins()
            
            # Piyasa korelasyonu analizi
            self.market_correlation = self.analyze_market_correlation()
            
            # Zamana bağlı desenler
            self.time_based_patterns = self.generate_time_based_patterns()
            
            # Sık yapılan hatalar
            self.common_mistakes = self.analyze_common_mistakes()
            
            # Başarılı desenler
            self.success_patterns = self.analyze_success_patterns()
            
            # Genel içgörüler
            self.learning_insights = self.generate_insights()
            
            # Strateji önerileri
            self.recommendations = self.generate_strategy_recommendations()
            
            return True
            
        except Exception as e:
            logger.error(f"Analiz sürecinde hata: {e}")
            return False
    
    def run(self) -> bool:
        """
        Tüm öğrenme motorunu çalıştır
        
        Returns:
            bool: Başarılı mı?
        """
        try:
            print(f"{COLORS['cyan']}📊 SentientTrader.AI - Öğrenme Motoru başlatılıyor...{COLORS['reset']}")
            logger.info("Öğrenme motoru başlatılıyor...")
            
            # Veri kaynaklarını yükle
            data_loaded = self.load_all_data()
            
            if not data_loaded:
                print(f"{COLORS['red']}❌ Veri yüklenemedi! Analiz yapılamıyor.{COLORS['reset']}")
                return False
            
            # Analizleri çalıştır
            print(f"{COLORS['cyan']}🧠 Veri analizi başlatılıyor...{COLORS['reset']}")
            analysis_success = self.analyze_all()
            
            if not analysis_success:
                print(f"{COLORS['yellow']}⚠️ Analiz sürecinde bazı hatalar oluştu.{COLORS['reset']}")
            
            # Sonuçları kaydet
            print(f"{COLORS['cyan']}💾 Öğrenme sonuçları kaydediliyor...{COLORS['reset']}")
            
            summary_saved = self.save_learning_summary()
            feedback_saved = self.save_learning_feedback()
            
            if summary_saved and feedback_saved:
                print(f"{COLORS['green']}✅ Öğrenme sonuçları başarıyla kaydedildi.{COLORS['reset']}")
            else:
                print(f"{COLORS['yellow']}⚠️ Öğrenme sonuçları kaydedilirken bazı hatalar oluştu.{COLORS['reset']}")
            
            # Özet göster
            self.display_summary()
            
            return True
            
        except Exception as e:
            logger.error(f"Öğrenme motoru çalıştırılırken hata: {e}")
            print(f"{COLORS['red']}❌ Hata: {e}{COLORS['reset']}")
            return False
    
    def display_summary(self) -> None:
        """
        Öğrenme sonuçlarının özetini terminalde göster
        """
        try:
            # Banner
            print(f"\n{COLORS['bright']}{COLORS['cyan']}{'=' * 60}{COLORS['reset']}")
            print(f"{COLORS['bright']}{COLORS['cyan']}🧠 SENTİENTTRADER.AI - ÖĞRENME MOTORU SONUÇLARI{COLORS['reset']}")
            print(f"{COLORS['bright']}{COLORS['cyan']}{'=' * 60}{COLORS['reset']}")
            
            print(f"\n{COLORS['yellow']}📅 Tarih: {CURRENT_TIME} | 👤 Kullanıcı: {CURRENT_USER}{COLORS['reset']}")
            
            # Strateji performansı
            print(f"\n{COLORS['bright']}📈 STRATEJİ PERFORMANSI:{COLORS['reset']}")
            
            if self.strategy_performance:
                for strategy, perf in self.strategy_performance.items():
                    win_rate = perf.get("win_rate", 0)
                    roi = perf.get("roi", 0)
                    net_profit = perf.get("net_profit_loss", 0)
                    
                    # Renk belirle
                    win_color = COLORS["green"] if win_rate >= 50 else COLORS["red"]
                    roi_color = COLORS["green"] if roi >= 0 else COLORS["red"]
                    profit_color = COLORS["green"] if net_profit >= 0 else COLORS["red"]
                    
                    print(f"  {COLORS['bright']}{strategy}:{COLORS['reset']}")
                    print(f"    Başarı Oranı: {win_color}%{win_rate:.2f}{COLORS['reset']} | ROI: {roi_color}%{roi:.2f}{COLORS['reset']} | Net Kâr/Zarar: {profit_color}${net_profit:.2f}{COLORS['reset']}")
            else:
                print(f"  {COLORS['yellow']}Strateji performans verisi mevcut değil{COLORS['reset']}")
            
            # En iyi coin'ler
            print(f"\n{COLORS['bright']}💰 EN İYİ COİNLER:{COLORS['reset']}")
            
            if self.coin_performance and "best_coins" in self.coin_performance and self.coin_performance["best_coins"]:
                best_coins = self.coin_performance["best_coins"]
                for i, coin in enumerate(best_coins[:5], 1):
                    # Coin detaylarını al
                    if "coin_performance" in self.coin_performance and coin in self.coin_performance["coin_performance"]:
                        coin_data = self.coin_performance["coin_performance"][coin]
                        win_rate = coin_data.get("win_rate", 0)
                        roi = coin_data.get("roi", 0)
                        
                        print(f"  {i}. {COLORS['bright']}{coin}{COLORS['reset']} - Başarı: {COLORS['green']}%{win_rate:.2f}{COLORS['reset']} | ROI: {COLORS['green']}%{roi:.2f}{COLORS['reset']}")
                    else:
                        print(f"  {i}. {COLORS['bright']}{coin}{COLORS['reset']}")
            else:
                print(f"  {COLORS['yellow']}Coin performans verisi mevcut değil{COLORS['reset']}")
            
            # İçgörüler
            print(f"\n{COLORS['bright']}💡 ÖĞRENME İÇGÖRÜLERİ:{COLORS['reset']}")
            
            if self.learning_insights:
                for i, insight in enumerate(self.learning_insights[:5], 1):
                    print(f"  {i}. {insight}")
                
                if len(self.learning_insights) > 5:
                    print(f"  ... ve {len(self.learning_insights) - 5} içgörü daha")
            else:
                print(f"  {COLORS['yellow']}İçgörü oluşturulamadı{COLORS['reset']}")
            
            # Hatalar
            print(f"\n{COLORS['bright']}🚫 SIK YAPILAN HATALAR:{COLORS['reset']}")
            
            if self.common_mistakes:
                for i, (mistake, count) in enumerate(self.common_mistakes[:3], 1):
                    print(f"  {i}. {mistake} ({count} kez)")
            else:
                print(f"  {COLORS['yellow']}Hata analizi yapılamadı{COLORS['reset']}")
            
            # Piyasa korelasyonu
            print(f"\n{COLORS['bright']}🌍 PİYASA KORELASYONU:{COLORS['reset']}")
            
            if self.market_correlation and "market_conditions" in self.market_correlation:
                market_conditions = self.market_correlation["market_conditions"]
                
                for condition, data in market_conditions.items():
                    win_rate = data.get("win_rate", 0)
                    win_color = COLORS["green"] if win_rate >= 50 else COLORS["red"]
                    
                    print(f"  {condition}: Başarı Oranı {win_color}%{win_rate:.2f}{COLORS['reset']}")
            else:
                print(f"  {COLORS['yellow']}Piyasa korelasyon verisi mevcut değil{COLORS['reset']}")
            
            # Kaydedilen dosyalar
            print(f"\n{COLORS['bright']}📝 KAYDEDILEN DOSYALAR:{COLORS['reset']}")
            print(f"  📊 Öğrenme Özeti: {COLORS['green']}{LEARNING_SUMMARY_FILE}{COLORS['reset']}")
            print(f"  🔄 Strateji Geri Beslemesi: {COLORS['green']}{LEARNING_FEEDBACK_FILE}{COLORS['reset']}")
            
            print(f"\n{COLORS['bright']}{COLORS['cyan']}{'=' * 60}{COLORS['reset']}")
            print(f"{COLORS['green']}✅ Öğrenme analizi tamamlandı!{COLORS['reset']}")
            print(f"{COLORS['bright']}{COLORS['cyan']}{'=' * 60}{COLORS['reset']}\n")
            
        except Exception as e:
            logger.error(f"Sonuç özeti gösterilirken hata: {e}")
            print(f"{COLORS['red']}❌ Sonuç özeti gösterilirken hata: {e}{COLORS['reset']}")

def parse_arguments() -> argparse.Namespace:
    """
    Komut satırı argümanlarını işle
    
    Returns:
        argparse.Namespace: İşlenmiş argümanlar
    """
    parser = argparse.ArgumentParser(description="SentientTrader.AI Learning Engine")
    parser.add_argument("--simulation", action="store_true", help="Simülasyon modunda çalış")
    parser.add_argument("--min-trades", type=int, default=5, help="Analiz için minimum işlem sayısı")
    parser.add_argument("--timeframe", type=int, default=30, help="Analiz süresi (gün)")
    return parser.parse_args()

def display_banner() -> None:
    """Başlık banner'ını göster"""
    banner = f"""
{COLORS["bright"]}{COLORS["cyan"]}
██╗     ███████╗ █████╗ ██████╗ ███╗   ██╗██╗███╗   ██╗ ██████╗ 
██║     ██╔════╝██╔══██╗██╔══██╗████╗  ██║██║████╗  ██║██╔════╝ 
██║     █████╗  ███████║██████╔╝██╔██╗ ██║██║██╔██╗ ██║██║  ███╗
██║     ██╔══╝  ██╔══██║██╔══██╗██║╚██╗██║██║██║╚██╗██║██║   ██║
███████╗███████╗██║  ██║██║  ██║██║ ╚████║██║██║ ╚████║╚██████╔╝
╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝ ╚═════╝ 
{COLORS["reset"]}{COLORS["magenta"]}                                    ENGINE V2.0{COLORS["reset"]}

{COLORS["yellow"]}📅 2025-04-21 23:15:16 | 👤 mstfatiryaki{COLORS["reset"]}
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
        
        # Öğrenme motoru oluştur
        engine = LearningEngine(simulation_mode=args.simulation)
        
        # Özel konfigürasyonları ayarla
        engine.min_trades_for_analysis = args.min_trades
        engine.analysis_timeframe = args.timeframe
        
        # Çalışma modunu göster
        mode = "SİMÜLASYON" if args.simulation else "GERÇEK İŞLEM"
        logger.info(f"Learning Engine {mode} modunda başlatılıyor...")
        print(f"{COLORS['cyan']}ℹ️ Öğrenme Motoru {mode} modunda başlatılıyor...{COLORS['reset']}")
        
        # Öğrenme motoru çalıştır
        success = engine.run()
        
        if success:
            logger.info("Öğrenme motoru başarıyla çalıştı.")
            return 0
        else:
            logger.error("Öğrenme motoru çalıştırılamadı veya analiz tamamlanamadı!")
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
