from random import choice
import random
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SentientTrader.AI - Performance Tracker Module
Analiz ve raporlama aracı - İşlem performansını takip eder ve gelişmiş metrikler üretir.

Author: mstfatiryaki
Created: 2025-04-21
Version: 2.0
License: Proprietary
"""

import os
import sys
import json
import time
import logging
import argparse
import threading
import statistics
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta, timezone
from colorama import init, Fore, Back, Style

# Colorama'yı başlat
init(autoreset=True)

# Sabitler
VERSION = "2.0"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
CURRENT_TIME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
CURRENT_USER = os.environ.get("USER", "mstfatiryaki")

# Dosya yolları
EXECUTED_TRADES_FILE = "executed_trades_log.json"
PERFORMANCE_REPORT_FILE = "performance_report.json"
LEARNING_FEEDBACK_FILE = "performance_feedback.json"
LOG_FILE = "logs/performance_tracker.log"

# Renk tanımları - kolay kullanım için
COLORS = {
    "red": Fore.RED,
    "green": Fore.GREEN,
    "yellow": Fore.YELLOW,
    "blue": Fore.BLUE,
    "magenta": Fore.MAGENTA,
    "cyan": Fore.CYAN,
    "white": Fore.WHITE,
    "bright": Style.BRIGHT,
    "dim": Style.DIM,
    "normal": Style.NORMAL,
    "reset": Style.RESET_ALL
}

# Loglama yapılandırması
try:
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    logging.basicConfig(filename=LOG_FILE, format=LOG_FORMAT, level=logging.DEBUG)
except Exception as e:
    # Log dizini oluşturulamazsa konsola yazdır
    logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG)
    logging.warning(f"Log dosyası oluşturulamadı: {e}")

# Logger
logger = logging.getLogger("PerformanceTracker")

# Thread güvenliği için kilit
file_lock = threading.Lock()

# Global piyasa verileri - historical_loader tarafından doldurulacak
market_data = {}

# Diğer modülleri koşullu olarak import et
try:
    from alert_system import AlertSystem
    ALERT_SYSTEM_AVAILABLE = True
except ImportError:
    ALERT_SYSTEM_AVAILABLE = False
    logger.warning("AlertSystem bulunamadı. Uyarı özellikleri devre dışı.")

try:
    from memory_core import MemoryCore
    MEMORY_CORE_AVAILABLE = True
except ImportError:
    MEMORY_CORE_AVAILABLE = False
    logger.warning("MemoryCore bulunamadı. Hafıza kayıt özellikleri devre dışı.")

# AlertSystem nesnesini oluştur
if ALERT_SYSTEM_AVAILABLE:
    alert_system = AlertSystem()
else:
    # AlertSystem mevcut değilse basit bir yedek oluştur
    class DummyAlertSystem:
        def info(self, message):
            print(f"{COLORS['cyan']}ℹ️ {message}{COLORS['reset']}")
            
        def warning(self, message):
            print(f"{COLORS['yellow']}⚠️ {message}{COLORS['reset']}")
            
        def critical(self, message):
            print(f"{COLORS['red']}🚨 {message}{COLORS['reset']}")
    
    alert_system = DummyAlertSystem()

# MemoryCore nesnesini oluştur
if MEMORY_CORE_AVAILABLE:
    memory_core = MemoryCore()  # Güvenli mod
else:
    # MemoryCore mevcut değilse basit bir yedek oluştur
    class DummyMemoryCore:
        def store_data(self, key, data):
            logger.info(f"MemoryCore mevcut değil. Veri kaydedilemiyor: {key}")
            return False
            
        def retrieve_data(self, key):
            logger.info(f"MemoryCore mevcut değil. Veri alınamıyor: {key}")
            return None
    
    memory_core = DummyMemoryCore()

class PerformanceTracker:
    """
    İşlem performansını analiz eden ve raporlayan temel sınıf
    """
    
    def __init__(self, simulation_mode: bool = False) -> None:
        """
        PerformanceTracker sınıfını başlat
        
        Args:
            simulation_mode: Simülasyon modunda çalış (gerçek işlem yapma)
        """
        self.simulation_mode = simulation_mode
        
        # Analiz için minimum işlem sayısı
        self.min_trades_for_analysis = 3
        
        # Analiz için gün sayısı (son X gün)
        self.analysis_timeframe = 30
        
        # İşlem verisi
        self.trades_data = []
        
        # Analiz sonuçları
        self.summary_stats = {}
        self.strategy_performance = {}
        self.coin_performance = {}
        self.time_performance = {}
        self.advanced_metrics = {}
        
        # Thread güvenliği için kilitler
        self.data_lock = threading.Lock()
        
        logger.info(f"PerformanceTracker başlatıldı (SIMULATION_MODE: {simulation_mode})")
    
    def load_trades_data(self) -> bool:
        """
        İşlem verilerini yükle
        
        Returns:
            bool: Başarılı yüklendi mi?
        """
        try:
            with file_lock:
                if not os.path.exists(EXECUTED_TRADES_FILE):
                    logger.warning(f"{EXECUTED_TRADES_FILE} bulunamadı.")
                    alert_system.warning(f"İşlem log dosyası bulunamadı: {EXECUTED_TRADES_FILE}")
                    return False
                
                with open(EXECUTED_TRADES_FILE, "r", encoding="utf-8") as f:
                    trades = json.load(f)
            
            if not trades:
                logger.warning("İşlem verisi boş.")
                return False
            
            # Son X günlük işlemleri filtrele
            cutoff_date = (datetime.now() - timedelta(days=self.analysis_timeframe)).timestamp()
            recent_trades = [trade for trade in trades if trade.get("close_time", 0) >= cutoff_date]
            
            with self.data_lock:
                self.trades_data = recent_trades
            
            logger.info(f"Toplam {len(trades)} işlemden {len(recent_trades)} tanesi son {self.analysis_timeframe} gün içinde.")
            
            if len(recent_trades) < self.min_trades_for_analysis:
                alert_system.warning(f"Analiz için yeterli işlem yok. Minimum: {self.min_trades_for_analysis}, Bulunan: {len(recent_trades)}")
                return False
                
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON okuma hatası: {e}")
            alert_system.critical(f"İşlem log dosyası bozuk: {EXECUTED_TRADES_FILE}")
            return False
            
        except Exception as e:
            logger.error(f"İşlem verileri yüklenirken hata: {e}")
            alert_system.warning(f"İşlem verileri yüklenirken hata: {e}")
            return False
    
    def analyze_strategy_performance(self) -> Dict[str, Any]:
        """
        Stratejilerin performansını analiz et
        
        Returns:
            Dict[str, Any]: Strateji performans metrikleri
        """
        if not self.trades_data:
            return {}
        
        strategy_metrics = {}
        
        for trade in self.trades_data:
            strategy = trade.get("strategy", "unknown")
            profit_loss = trade.get("net_profit_loss", 0)
            investment = trade.get("investment_amount", 0)
            is_win = profit_loss > 0
            
            # Strateji ilk kez görülüyorsa başlat
            if strategy not in strategy_metrics:
                strategy_metrics[strategy] = {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "net_profit_loss": 0,
                    "total_investment": 0,
                    "win_rate": 0,
                    "roi": 0,
                    "avg_profit_per_trade": 0,
                    "consecutive_wins": 0,
                    "consecutive_losses": 0,
                    "max_consecutive_wins": 0,
                    "max_consecutive_losses": 0,
                    "last_result": None  # Son işlem sonucu (win/loss)
                }
            
            # Metrikleri güncelle
            strategy_metrics[strategy]["total_trades"] += 1
            strategy_metrics[strategy]["net_profit_loss"] += profit_loss
            
            if investment > 0:
                strategy_metrics[strategy]["total_investment"] += investment
            
            if is_win:
                strategy_metrics[strategy]["winning_trades"] += 1
                
                # Ardışık kazanç/kayıp takibi
                if strategy_metrics[strategy]["last_result"] == "win":
                    strategy_metrics[strategy]["consecutive_wins"] += 1
                else:
                    strategy_metrics[strategy]["consecutive_wins"] = 1
                    strategy_metrics[strategy]["consecutive_losses"] = 0
                
                strategy_metrics[strategy]["last_result"] = "win"
                
                # Max ardışık kazanç güncelleme
                if strategy_metrics[strategy]["consecutive_wins"] > strategy_metrics[strategy]["max_consecutive_wins"]:
                    strategy_metrics[strategy]["max_consecutive_wins"] = strategy_metrics[strategy]["consecutive_wins"]
            else:
                strategy_metrics[strategy]["losing_trades"] += 1
                
                # Ardışık kazanç/kayıp takibi
                if strategy_metrics[strategy]["last_result"] == "loss":
                    strategy_metrics[strategy]["consecutive_losses"] += 1
                else:
                    strategy_metrics[strategy]["consecutive_losses"] = 1
                    strategy_metrics[strategy]["consecutive_wins"] = 0
                
                strategy_metrics[strategy]["last_result"] = "loss"
                
                # Max ardışık kayıp güncelleme
                if strategy_metrics[strategy]["consecutive_losses"] > strategy_metrics[strategy]["max_consecutive_losses"]:
                    strategy_metrics[strategy]["max_consecutive_losses"] = strategy_metrics[strategy]["consecutive_losses"]
        
        # Son hesaplamalar
        for strategy, metrics in strategy_metrics.items():
            total_trades = metrics["total_trades"]
            winning_trades = metrics["winning_trades"]
            total_investment = metrics["total_investment"]
            net_profit_loss = metrics["net_profit_loss"]
            
            # Başarı oranı
            if total_trades > 0:
                metrics["win_rate"] = (winning_trades / total_trades) * 100
                metrics["avg_profit_per_trade"] = net_profit_loss / total_trades
            
            # ROI
            if total_investment > 0:
                metrics["roi"] = (net_profit_loss / total_investment) * 100
        
        return strategy_metrics
    
    def analyze_coin_performance(self) -> Dict[str, Any]:
        """
        Coinlerin performansını analiz et
        
        Returns:
            Dict[str, Any]: Coin performans metrikleri
        """
        if not self.trades_data:
            return {}
        
        coin_metrics = {}
        
        for trade in self.trades_data:
            coin = trade.get("coin", "unknown")
            profit_loss = trade.get("net_profit_loss", 0)
            investment = trade.get("investment_amount", 0)
            is_win = profit_loss > 0
            
            # Coin ilk kez görülüyorsa başlat
            if coin not in coin_metrics:
                coin_metrics[coin] = {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "net_profit_loss": 0,
                    "total_investment": 0,
                    "win_rate": 0,
                    "roi": 0,
                    "avg_profit_per_trade": 0
                }
            
            # Metrikleri güncelle
            coin_metrics[coin]["total_trades"] += 1
            coin_metrics[coin]["net_profit_loss"] += profit_loss
            
            if investment > 0:
                coin_metrics[coin]["total_investment"] += investment
            
            if is_win:
                coin_metrics[coin]["winning_trades"] += 1
            else:
                coin_metrics[coin]["losing_trades"] += 1
        
        # Son hesaplamalar
        for coin, metrics in coin_metrics.items():
            total_trades = metrics["total_trades"]
            winning_trades = metrics["winning_trades"]
            total_investment = metrics["total_investment"]
            net_profit_loss = metrics["net_profit_loss"]
            
            # Başarı oranı
            if total_trades > 0:
                metrics["win_rate"] = (winning_trades / total_trades) * 100
                metrics["avg_profit_per_trade"] = net_profit_loss / total_trades
            
            # ROI
            if total_investment > 0:
                metrics["roi"] = (net_profit_loss / total_investment) * 100
        
        # En karlı ve en zararlı coinleri bul
        most_profitable_coins = sorted(
            [(coin, metrics["net_profit_loss"]) for coin, metrics in coin_metrics.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        least_profitable_coins = sorted(
            [(coin, metrics["net_profit_loss"]) for coin, metrics in coin_metrics.items()],
            key=lambda x: x[1]
        )
        
        # En yüksek win rate'e sahip coinleri bul
        best_win_rate_coins = sorted(
            [(coin, metrics["win_rate"]) for coin, metrics in coin_metrics.items() if metrics["total_trades"] >= 3],
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            "coin_metrics": coin_metrics,
            "most_profitable_coins": most_profitable_coins,
            "least_profitable_coins": least_profitable_coins,
            "best_win_rate_coins": best_win_rate_coins
        }
    
    def analyze_time_performance(self) -> Dict[str, Any]:
        """
        Zamana göre performansı analiz et
        
        Returns:
            Dict[str, Any]: Zamana göre performans metrikleri
        """
        if not self.trades_data:
            return {}
        
        # Saat dilimleri (4 saatlik dilimler)
        hour_segments = {
            "00:00-04:00": {"trades": 0, "wins": 0, "losses": 0, "profit_loss": 0, "win_rate": 0},
            "04:00-08:00": {"trades": 0, "wins": 0, "losses": 0, "profit_loss": 0, "win_rate": 0},
            "08:00-12:00": {"trades": 0, "wins": 0, "losses": 0, "profit_loss": 0, "win_rate": 0},
            "12:00-16:00": {"trades": 0, "wins": 0, "losses": 0, "profit_loss": 0, "win_rate": 0},
            "16:00-20:00": {"trades": 0, "wins": 0, "losses": 0, "profit_loss": 0, "win_rate": 0},
            "20:00-24:00": {"trades": 0, "wins": 0, "losses": 0, "profit_loss": 0, "win_rate": 0}
        }
        
        # Haftanın günleri
        days_of_week = {
            "Pazartesi": {"trades": 0, "wins": 0, "losses": 0, "profit_loss": 0, "win_rate": 0},
            "Salı": {"trades": 0, "wins": 0, "losses": 0, "profit_loss": 0, "win_rate": 0},
            "Çarşamba": {"trades": 0, "wins": 0, "losses": 0, "profit_loss": 0, "win_rate": 0},
            "Perşembe": {"trades": 0, "wins": 0, "losses": 0, "profit_loss": 0, "win_rate": 0},
            "Cuma": {"trades": 0, "wins": 0, "losses": 0, "profit_loss": 0, "win_rate": 0},
            "Cumartesi": {"trades": 0, "wins": 0, "losses": 0, "profit_loss": 0, "win_rate": 0},
            "Pazar": {"trades": 0, "wins": 0, "losses": 0, "profit_loss": 0, "win_rate": 0}
        }
        
        # Günün bölümleri
        daytime_segments = {
            "Sabah (06:00-12:00)": {"trades": 0, "wins": 0, "losses": 0, "profit_loss": 0, "win_rate": 0},
            "Öğle (12:00-18:00)": {"trades": 0, "wins": 0, "losses": 0, "profit_loss": 0, "win_rate": 0},
            "Akşam (18:00-24:00)": {"trades": 0, "wins": 0, "losses": 0, "profit_loss": 0, "win_rate": 0},
            "Gece (00:00-06:00)": {"trades": 0, "wins": 0, "losses": 0, "profit_loss": 0, "win_rate": 0}
        }
        
        # Gün içi ve haftalık performans
        for trade in self.trades_data:
            # İşlem zamanını datetime nesnesine çevir
            close_time = trade.get("close_time", 0)
            if not close_time:
                continue
                
            dt = datetime.fromtimestamp(close_time)
            profit_loss = trade.get("net_profit_loss", 0)
            is_win = profit_loss > 0
            
            # Saat dilimlerini belirle
            hour = dt.hour
            if 0 <= hour < 4:
                segment = "00:00-04:00"
            elif 4 <= hour < 8:
                segment = "04:00-08:00"
            elif 8 <= hour < 12:
                segment = "08:00-12:00"
            elif 12 <= hour < 16:
                segment = "12:00-16:00"
            elif 16 <= hour < 20:
                segment = "16:00-20:00"
            else:
                segment = "20:00-24:00"
            
            # Saat dilimi istatistiklerini güncelle
            hour_segments[segment]["trades"] += 1
            hour_segments[segment]["profit_loss"] += profit_loss
            if is_win:
                hour_segments[segment]["wins"] += 1
            else:
                hour_segments[segment]["losses"] += 1
            
            # Haftanın günü
            day_names = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
            day_of_week = day_names[dt.weekday()]
            
            # Haftanın günü istatistiklerini güncelle
            days_of_week[day_of_week]["trades"] += 1
            days_of_week[day_of_week]["profit_loss"] += profit_loss
            if is_win:
                days_of_week[day_of_week]["wins"] += 1
            else:
                days_of_week[day_of_week]["losses"] += 1
            
            # Günün bölümleri
            if 6 <= hour < 12:
                daytime = "Sabah (06:00-12:00)"
            elif 12 <= hour < 18:
                daytime = "Öğle (12:00-18:00)"
            elif 18 <= hour < 24:
                daytime = "Akşam (18:00-24:00)"
            else:
                daytime = "Gece (00:00-06:00)"
            
            # Günün bölümü istatistiklerini güncelle
            daytime_segments[daytime]["trades"] += 1
            daytime_segments[daytime]["profit_loss"] += profit_loss
            if is_win:
                daytime_segments[daytime]["wins"] += 1
            else:
                daytime_segments[daytime]["losses"] += 1
        
        # Win rate hesapla - saat dilimleri
        for segment, data in hour_segments.items():
            if data["trades"] > 0:
                data["win_rate"] = (data["wins"] / data["trades"]) * 100
        
        # Win rate hesapla - haftanın günleri
        for day, data in days_of_week.items():
            if data["trades"] > 0:
                data["win_rate"] = (data["wins"] / data["trades"]) * 100
        
        # Win rate hesapla - günün bölümleri
        for segment, data in daytime_segments.items():
            if data["trades"] > 0:
                data["win_rate"] = (data["wins"] / data["trades"]) * 100
        
        return {
            "hour_segments": hour_segments,
            "day_of_week": days_of_week,
            "daytime_segments": daytime_segments
        }
    
    def analyze_advanced_metrics(self) -> Dict[str, Any]:
        """
        Gelişmiş metrikleri analiz et
        
        Returns:
            Dict[str, Any]: Gelişmiş metrikler
        """
        if not self.trades_data:
            return {}
        
        # Sıralı işlemleri al (kapanış zamanına göre)
        sorted_trades = sorted(self.trades_data, key=lambda x: x.get("close_time", 0))
        
        # Drawdown hesaplama
        cumulative_balance = 0
        peak_balance = 0
        drawdowns = []
        current_drawdown = 0
        
        # Volatilite için karlılık değerleri
        profit_percentages = []
        
        # Kazanç / kayıp serileri
        current_win_streak = 0
        current_loss_streak = 0
        longest_win_streak = 0
        longest_loss_streak = 0
        
        # Risk/Ödül analizi
        total_risk = 0
        total_reward = 0
        risk_reward_pairs = []
        
        # Coin korelasyonları için veri
        coin_profit_data = {}
        
        # İşlemleri işle
        for trade in sorted_trades:
            profit_loss = trade.get("net_profit_loss", 0)
            investment = trade.get("investment_amount", 0)
            coin = trade.get("coin", "unknown")
            is_win = profit_loss > 0
            
            # Coin bazlı karlılık verisi
            if coin not in coin_profit_data:
                coin_profit_data[coin] = []
            
            if investment > 0:
                profit_percentage = (profit_loss / investment) * 100
                coin_profit_data[coin].append(profit_percentage)
                profit_percentages.append(profit_percentage)
            
            # Drawdown hesaplama
            cumulative_balance += profit_loss
            
            if cumulative_balance > peak_balance:
                peak_balance = cumulative_balance
                current_drawdown = 0
            else:
                current_drawdown = peak_balance - cumulative_balance
                drawdowns.append(current_drawdown)
            
            # Risk/Ödül hesaplama
            stop_loss = trade.get("stop_loss", 0)
            take_profit = trade.get("take_profit", 0)
            entry_price = trade.get("entry_price", 0)
            
            if entry_price > 0 and stop_loss > 0 and take_profit > 0:
                # Long pozisyon
                if trade.get("position_type", "").lower() == "long":
                    risk = (entry_price - stop_loss) / entry_price
                    reward = (take_profit - entry_price) / entry_price
                # Short pozisyon
                else:
                    risk = (stop_loss - entry_price) / entry_price
                    reward = (entry_price - take_profit) / entry_price
                
                if risk > 0:
                    risk_reward_ratio = reward / risk
                    risk_reward_pairs.append(risk_reward_ratio)
                    
                    total_risk += risk
                    total_reward += reward
            
            # Kazanç / kayıp serileri
            if is_win:
                current_win_streak += 1
                current_loss_streak = 0
                
                if current_win_streak > longest_win_streak:
                    longest_win_streak = current_win_streak
            else:
                current_loss_streak += 1
                current_win_streak = 0
                
                if current_loss_streak > longest_loss_streak:
                    longest_loss_streak = current_loss_streak
        
        # Max drawdown hesaplama
        max_drawdown = max(drawdowns) if drawdowns else 0
        
        # Max drawdown yüzdesi
        if peak_balance > 0:
            max_drawdown_pct = (max_drawdown / peak_balance) * 100
        else:
            max_drawdown_pct = 0
        
        # Volatilite hesaplama (işlem karlılıklarının standart sapması)
        volatility = statistics.stdev(profit_percentages) if len(profit_percentages) > 1 else 0
        
        # Ortalama risk/ödül oranı
        avg_risk_reward_ratio = sum(risk_reward_pairs) / len(risk_reward_pairs) if risk_reward_pairs else 0
        
        # Coin korelasyonları
        correlations = {}
        high_correlations = {}
        
        # En az 5 işlem yapılan coinler arasında korelasyon hesapla
        coins_with_enough_data = [coin for coin, data in coin_profit_data.items() if len(data) >= 5]
        
        for i, coin1 in enumerate(coins_with_enough_data):
            for coin2 in coins_with_enough_data[i+1:]:
                # Ortak uzunluğu bul (daha kısa olan listenin uzunluğu)
                common_length = min(len(coin_profit_data[coin1]), len(coin_profit_data[coin2]))
                
                # Son N işlemi al
                data1 = coin_profit_data[coin1][-common_length:]
                data2 = coin_profit_data[coin2][-common_length:]
                
                # Pearson korelasyon katsayısı hesapla
                try:
                    correlation = self.calculate_correlation(data1, data2)
                    correlations[f"{coin1}-{coin2}"] = correlation
                    
                    # Yüksek korelasyonları kaydet (0.7'den büyük veya -0.7'den küçük)
                    if abs(correlation) > 0.7:
                        high_correlations[f"{coin1}-{coin2}"] = correlation
                except:
                    pass
        
        return {
            "drawdown_analysis": {
                "max_drawdown": max_drawdown,
                "max_drawdown_pct": max_drawdown_pct,
                "current_drawdown": current_drawdown,
            },
            "volatility": volatility,
            "winning_streaks": {
                "longest_win_streak": longest_win_streak,
                "longest_loss_streak": longest_loss_streak,
                "current_win_streak": current_win_streak,
                "current_loss_streak": current_loss_streak
            },
            "risk_reward": {
                "avg_risk_reward_ratio": avg_risk_reward_ratio,
                "total_risk": total_risk,
                "total_reward": total_reward
            },
            "coin_correlations": {
                "correlations": correlations,
                "high_correlations": high_correlations
            }
        }
    
    def calculate_correlation(self, data1: List[float], data2: List[float]) -> float:
        """
        İki veri seti arasındaki Pearson korelasyon katsayısını hesapla
        
        Args:
            data1: İlk veri seti
            data2: İkinci veri seti
            
        Returns:
            float: Korelasyon katsayısı (-1 ile 1 arasında)
        """
        if len(data1) != len(data2) or len(data1) < 2:
            return 0
        
        # Ortalamaları hesapla
        mean1 = sum(data1) / len(data1)
        mean2 = sum(data2) / len(data2)
        
        # Kovaryans ve standart sapmaları hesapla
        covariance = sum((x - mean1) * (y - mean2) for x, y in zip(data1, data2))
        variance1 = sum((x - mean1) ** 2 for x in data1)
        variance2 = sum((y - mean2) ** 2 for y in data2)
        
        # Pearson korelasyon katsayısı
        if variance1 > 0 and variance2 > 0:
            correlation = covariance / ((variance1 * variance2) ** 0.5)
            return correlation
        else:
            return 0
    
    def calculate_overall_performance(self) -> Dict[str, Any]:
        """
        Genel performans metriklerini hesapla
        
        Returns:
            Dict[str, Any]: Genel performans metrikleri
        """
        if not self.trades_data:
            return {}
        
        # Tüm işlemler için metrikler
        total_trades = len(self.trades_data)
        total_profit_loss = sum(trade.get("net_profit_loss", 0) for trade in self.trades_data)
        total_investment = sum(trade.get("investment_amount", 0) for trade in self.trades_data)
        winning_trades = sum(1 for trade in self.trades_data if trade.get("net_profit_loss", 0) > 0)
        losing_trades = total_trades - winning_trades
        
        # Başarı oranı
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Ortalama işlem başı kar
        avg_profit_per_trade = total_profit_loss / total_trades if total_trades > 0 else 0
        
        # ROI
        roi = (total_profit_loss / total_investment) * 100 if total_investment > 0 else 0
        
        # Son 30 günlük veriler
        last_30d_cutoff = (datetime.now() - timedelta(days=30)).timestamp()
        last_30d_trades = [trade for trade in self.trades_data if trade.get("close_time", 0) >= last_30d_cutoff]
        
        last_30d_total = len(last_30d_trades)
        last_30d_profit = sum(trade.get("net_profit_loss", 0) for trade in last_30d_trades)
        last_30d_investment = sum(trade.get("investment_amount", 0) for trade in last_30d_trades)
        last_30d_wins = sum(1 for trade in last_30d_trades if trade.get("net_profit_loss", 0) > 0)
        
        last_30d_win_rate = (last_30d_wins / last_30d_total) * 100 if last_30d_total > 0 else 0
        last_30d_roi = (last_30d_profit / last_30d_investment) * 100 if last_30d_investment > 0 else 0
        
        return {
            "overall": {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": win_rate,
                "net_profit_loss": total_profit_loss,
                "total_investment": total_investment,
                "roi": roi,
                "avg_profit_per_trade": avg_profit_per_trade
            },
            "last_30_days": {
                "total_trades": last_30d_total,
                "winning_trades": last_30d_wins,
                "losing_trades": last_30d_total - last_30d_wins,
                "win_rate": last_30d_win_rate,
                "net_profit_loss": last_30d_profit,
                "total_investment": last_30d_investment,
                "roi": last_30d_roi
            }
        }
    
    def create_performance_report(self) -> Dict[str, Any]:
        """
        Tam performans raporunu oluştur
        
        Returns:
            Dict[str, Any]: Performans raporu
        """
        return {
            "timestamp": time.time(),
            "date": CURRENT_TIME,
            "user": CURRENT_USER,
            "total_trades_analyzed": len(self.trades_data),
            "analysis_timeframe_days": self.analysis_timeframe,
            "summary_stats": self.summary_stats,
            "strategy_performance": self.strategy_performance,
            "coin_performance": self.coin_performance,
            "time_performance": self.time_performance,
            "advanced_metrics": self.advanced_metrics
        }
    
    def create_learning_feedback(self) -> Dict[str, Any]:
        """
        Öğrenme motoruna gönderilecek geri besleme verisi oluştur
        
        Returns:
            Dict[str, Any]: Öğrenme geri beslemesi
        """
        feedback = {
            "timestamp": time.time(),
            "date": CURRENT_TIME,
            "total_trades": len(self.trades_data),
            "overall_win_rate": self.summary_stats.get("overall", {}).get("win_rate", 0),
            "overall_roi": self.summary_stats.get("overall", {}).get("roi", 0),
            "best_strategies": [],
            "worst_strategies": [],
            "best_coins": [],
            "worst_coins": [],
            "best_timeframes": [],
            "suggested_improvements": []
        }
        
        # En iyi 3 strateji
        if self.strategy_performance:
            sorted_strategies = sorted(
                [(strat, data.get("roi", 0)) for strat, data in self.strategy_performance.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            feedback["best_strategies"] = sorted_strategies[:3]
            feedback["worst_strategies"] = sorted_strategies[-3:] if len(sorted_strategies) >= 3 else []
        
        # En iyi 3 coin
        if "coin_metrics" in self.coin_performance:
            sorted_coins = sorted(
                [(coin, data.get("roi", 0)) for coin, data in self.coin_performance["coin_metrics"].items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            feedback["best_coins"] = sorted_coins[:3]
            feedback["worst_coins"] = sorted_coins[-3:] if len(sorted_coins) >= 3 else []
        
        # En iyi zaman dilimleri
        if "daytime_segments" in self.time_performance:
            sorted_times = sorted(
                [(time, data.get("win_rate", 0)) for time, data in self.time_performance["daytime_segments"].items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            feedback["best_timeframes"] = sorted_times[:2]
        
        # Önerilen iyileştirmeler
        suggestions = []
        
        # Düşük win rate stratejiler için öneri
        low_win_strategies = [
            strat for strat, data in self.strategy_performance.items() 
            if data.get("win_rate", 0) < 40 and data.get("total_trades", 0) >= 5
        ]
        
        if low_win_strategies:
            suggestions.append(f"Düşük başarı oranlı stratejileri gözden geçirin: {', '.join(low_win_strategies)}")
        
        # Yüksek volatiliteli coinler için öneri
        if "coin_metrics" in self.coin_performance:
            volatile_coins = []
            for coin, data in self.coin_performance["coin_metrics"].items():
                trades = data.get("total_trades", 0)
                win_rate = data.get("win_rate", 0)
                if trades >= 5 and win_rate < 40:
                    volatile_coins.append(coin)
            
            if volatile_coins:
                suggestions.append(f"Bu coinlerde daha dikkatli olun: {', '.join(volatile_coins)}")
        
        # Max drawdown için öneri
        max_dd_pct = self.advanced_metrics.get("drawdown_analysis", {}).get("max_drawdown_pct", 0)
        if max_dd_pct > 20:
            suggestions.append(f"Yüksek drawdown (%{max_dd_pct:.2f}) tespit edildi. Risk yönetiminizi gözden geçirin.")
        
        # Zaman dilimi bazlı öneriler
        if "daytime_segments" in self.time_performance:
            worst_time = min(
                self.time_performance["daytime_segments"].items(),
                key=lambda x: x[1].get("win_rate", 100) if x[1].get("trades", 0) > 3 else 100
            )
            
            if worst_time[1].get("trades", 0) > 3 and worst_time[1].get("win_rate", 0) < 40:
                suggestions.append(f"{worst_time[0]} zaman diliminde işlem yapmaktan kaçının (başarı oranı: %{worst_time[1].get('win_rate', 0):.2f})")
        
        # Uzun kayıp serisi için öneri
        longest_loss = self.advanced_metrics.get("winning_streaks", {}).get("longest_loss_streak", 0)
        if longest_loss >= 5:
            suggestions.append(f"Ardışık {longest_loss} kayıp işleminiz olmuş. Kayıpları sınırlamak için stop-loss stratejinizi güçlendirin.")
        
        # Yüksek korelasyonlu coinler varsa
        high_correlations = self.advanced_metrics.get("coin_correlations", {}).get("high_correlations", {})
        if high_correlations:
            suggestions.append("Yüksek korelasyonlu coin çiftlerinde aynı anda aynı yönde işlem açmaktan kaçının")
        
        feedback["suggested_improvements"] = suggestions
        
        return feedback
    
    def save_performance_report(self) -> bool:
        """
        Performans raporunu dosyaya kaydet
        
        Returns:
            bool: Başarılı mı?
        """
        try:
            with open(PERFORMANCE_REPORT_FILE, "w", encoding="utf-8") as f:
                json.dump(self.create_performance_report(), f, ensure_ascii=False, indent=4)
            
            logger.info(f"Performans raporu {PERFORMANCE_REPORT_FILE} dosyasına kaydedildi")
            return True
            
        except Exception as e:
            logger.error(f"Performans raporu kaydedilirken hata: {e}")
            return False
    
    def save_learning_feedback(self) -> bool:
        """
        Öğrenme geri bildirimini dosyaya kaydet
        
        Returns:
            bool: Başarılı mı?
        """
        try:
            with open(LEARNING_FEEDBACK_FILE, "w", encoding="utf-8") as f:
                json.dump(self.create_learning_feedback(), f, ensure_ascii=False, indent=4)
            
            logger.info(f"Öğrenme geri bildirimi {LEARNING_FEEDBACK_FILE} dosyasına kaydedildi")
            return True
            
        except Exception as e:
            logger.error(f"Öğrenme geri bildirimi kaydedilirken hata: {e}")
            return False
    
    def analyze_all(self) -> bool:
        """
        Tüm analiz işlemlerini yürüt
        
        Returns:
            bool: Başarılı mı?
        """
        try:
            # Strateji performansı
            self.strategy_performance = self.analyze_strategy_performance()
            
            # Coin performansı
            self.coin_performance = self.analyze_coin_performance()
            
            # Zaman performansı
            self.time_performance = self.analyze_time_performance()
            
            # Gelişmiş metrikler
            self.advanced_metrics = self.analyze_advanced_metrics()
            
            # Genel performans
            self.summary_stats = self.calculate_overall_performance()
            
            return True
            
        except Exception as e:
            logger.error(f"Performans analizi sırasında hata: {e}")
            return False
        
    def run(self) -> bool:
        """
        Tüm performans analizi sürecini çalıştır
        
        Returns:
            bool: Başarılı mı?
        """
        try:
            print(f"{COLORS['cyan']}📊 SentientTrader.AI - Performans Takipçisi başlatılıyor...{COLORS['reset']}")
            logger.info("Performans takipçisi başlatılıyor...")
            
            # İşlem verilerini yükle
            data_loaded = self.load_trades_data()
            
            if not data_loaded:
                print(f"{COLORS['red']}❌ İşlem verileri yüklenemedi! Analiz yapılamıyor.{COLORS['reset']}")
                return False
            
            # Tüm analizleri çalıştır
            print(f"{COLORS['cyan']}🧮 Performans analizi başlatılıyor...{COLORS['reset']}")
            analysis_success = self.analyze_all()
            
            if not analysis_success:
                print(f"{COLORS['yellow']}⚠️ Analiz sürecinde bazı hatalar oluştu.{COLORS['reset']}")
            
            # Sonuçları kaydet
            print(f"{COLORS['cyan']}💾 Performans sonuçları kaydediliyor...{COLORS['reset']}")
            
            report_saved = self.save_performance_report()
            feedback_saved = self.save_learning_feedback()
            
            if report_saved and feedback_saved:
                print(f"{COLORS['green']}✅ Performans sonuçları başarıyla kaydedildi.{COLORS['reset']}")
            else:
                print(f"{COLORS['yellow']}⚠️ Performans sonuçları kaydedilirken bazı hatalar oluştu.{COLORS['reset']}")
            
            # Özet göster
            self.display_summary()
            
            return True
            
        except Exception as e:
            logger.error(f"Performans takipçisi çalıştırılırken hata: {e}")
            print(f"{COLORS['red']}❌ Hata: {e}{COLORS['reset']}")
            return False

    def display_summary(self) -> None:
        """
        Performans analizinin özetini terminalde göster
        """
        try:
            # Banner
            print(f"\n{COLORS['bright']}{COLORS['cyan']}{'=' * 60}{COLORS['reset']}")
            print(f"{COLORS['bright']}{COLORS['cyan']}📊 SENTİENTTRADER.AI - PERFORMANS ANALİZİ SONUÇLARI{COLORS['reset']}")
            print(f"{COLORS['bright']}{COLORS['cyan']}{'=' * 60}{COLORS['reset']}")
            
            print(f"\n{COLORS['yellow']}📅 Tarih: {CURRENT_TIME} | 👤 Kullanıcı: {CURRENT_USER}{COLORS['reset']}")
            
            # Genel performans özeti
            print(f"\n{COLORS['bright']}📈 GENEL PERFORMANS ÖZETİ:{COLORS['reset']}")
            
            if "overall" in self.summary_stats and self.summary_stats["overall"]:
                overall = self.summary_stats["overall"]
                win_rate = overall.get("win_rate", 0)
                total_trades = overall.get("total_trades", 0)
                net_profit = overall.get("net_profit_loss", 0)
                roi = overall.get("roi", 0)
                
                # Renk seçimi
                win_color = COLORS["green"] if win_rate >= 50 else COLORS["red"]
                profit_color = COLORS["green"] if net_profit >= 0 else COLORS["red"]
                roi_color = COLORS["green"] if roi >= 0 else COLORS["red"]
                
                print(f"  Toplam İşlem: {total_trades}")
                print(f"  Başarı Oranı: {win_color}%{win_rate:.2f}{COLORS['reset']}")
                print(f"  Net Kâr/Zarar: {profit_color}${net_profit:.2f}{COLORS['reset']}")
                print(f"  ROI: {roi_color}%{roi:.2f}{COLORS['reset']}")
                
                # Son 30 günlük performans
                if "last_30_days" in self.summary_stats and self.summary_stats["last_30_days"]:
                    last_30d = self.summary_stats["last_30_days"]
                    recent_win_rate = last_30d.get("win_rate", 0)
                    recent_net_profit = last_30d.get("net_profit_loss", 0)
                    
                    recent_win_color = COLORS["green"] if recent_win_rate >= 50 else COLORS["red"]
                    recent_profit_color = COLORS["green"] if recent_net_profit >= 0 else COLORS["red"]
                    
                    print(f"\n  {COLORS['white']}Son 30 Gün:{COLORS['reset']}")
                    print(f"    Başarı Oranı: {recent_win_color}%{recent_win_rate:.2f}{COLORS['reset']}")
                    print(f"    Net Kâr/Zarar: {recent_profit_color}${recent_net_profit:.2f}{COLORS['reset']}")
            else:
                print(f"  {COLORS['yellow']}Genel performans verisi mevcut değil{COLORS['reset']}")
            
            # Strateji performansı
            print(f"\n{COLORS['bright']}🔍 STRATEJİ PERFORMANSI:{COLORS['reset']}")
            
            if self.strategy_performance:
                # Stratejileri ROI'ye göre sırala
                sorted_strategies = sorted(
                    self.strategy_performance.items(),
                    key=lambda x: x[1].get("roi", 0),
                    reverse=True
                )
                
                for strategy, perf in sorted_strategies:
                    win_rate = perf.get("win_rate", 0)
                    roi = perf.get("roi", 0)
                    trades = perf.get("total_trades", 0)
                    
                    # Renk seçimi
                    strategy_color = COLORS["green"] if roi > 0 else COLORS["red"]
                    
                    print(f"  {strategy_color}{strategy}{COLORS['reset']}:")
                    print(f"    Başarı Oranı: %{win_rate:.2f} | ROI: %{roi:.2f} | İşlem: {trades}")
            else:
                print(f"  {COLORS['yellow']}Strateji performans verisi mevcut değil{COLORS['reset']}")
            
            # Coin performansı
            print(f"\n{COLORS['bright']}💰 EN İYİ COİNLER:{COLORS['reset']}")
            
            if "most_profitable_coins" in self.coin_performance and self.coin_performance["most_profitable_coins"]:
                profit_coins = self.coin_performance["most_profitable_coins"]
                
                for i, (coin, profit) in enumerate(profit_coins[:5], 1):
                    print(f"  {i}. {COLORS['bright']}{coin}{COLORS['reset']} - Net Kâr: {COLORS['green']}${profit:.2f}{COLORS['reset']}")
            elif "coin_metrics" in self.coin_performance:
                # En yüksek ROI'li 5 coini göster
                sorted_coins = sorted(
                    [(coin, data.get("roi", 0)) for coin, data in self.coin_performance["coin_metrics"].items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                
                for i, (coin, roi) in enumerate(sorted_coins, 1):
                    print(f"  {i}. {COLORS['bright']}{coin}{COLORS['reset']} - ROI: {COLORS['green']}%{roi:.2f}{COLORS['reset']}")
            else:
                print(f"  {COLORS['yellow']}Coin performans verisi mevcut değil{COLORS['reset']}")
            
            # Zaman bazlı performans
            print(f"\n{COLORS['bright']}⏱️ ZAMAN PERFORMANSI:{COLORS['reset']}")
            
            if "daytime_segments" in self.time_performance and self.time_performance["daytime_segments"]:
                # Win rate'e göre sırala
                sorted_segments = sorted(
                    self.time_performance["daytime_segments"].items(),
                    key=lambda x: x[1].get("win_rate", 0),
                    reverse=True
                )
                
                for segment, data in sorted_segments:
                    win_rate = data.get("win_rate", 0)
                    trades = data.get("trades", 0)
                    
                    segment_color = COLORS["green"] if win_rate >= 50 else COLORS["red"]
                    print(f"  {segment}: Başarı Oranı {segment_color}%{win_rate:.2f}{COLORS['reset']} ({trades} işlem)")
            
                # En iyi gün
                if "day_of_week" in self.time_performance and self.time_performance["day_of_week"]:
                    best_day = max(
                        self.time_performance["day_of_week"].items(),
                        key=lambda x: x[1].get("win_rate", 0)
                    )
                    
                    day_name = best_day[0]
                    day_win_rate = best_day[1].get("win_rate", 0)
                    
                    print(f"\n  En Başarılı Gün: {COLORS['green']}{day_name}{COLORS['reset']} (%{day_win_rate:.2f})")
            else:
                print(f"  {COLORS['yellow']}Zaman performans verisi mevcut değil{COLORS['reset']}")
            
            # Gelişmiş metrikler
            print(f"\n{COLORS['bright']}🔬 GELİŞMİŞ METRİKLER:{COLORS['reset']}")
            
            # Drawdown analizi
            if "drawdown_analysis" in self.advanced_metrics:
                drawdown = self.advanced_metrics["drawdown_analysis"]
                max_dd_pct = drawdown.get("max_drawdown_pct", 0)
                
                dd_color = COLORS["green"] if max_dd_pct < 10 else COLORS["yellow"] if max_dd_pct < 20 else COLORS["red"]
                print(f"  Maksimum Drawdown: {dd_color}%{max_dd_pct:.2f}{COLORS['reset']}")
            
            # Kazanç serileri
            if "winning_streaks" in self.advanced_metrics:
                streaks = self.advanced_metrics["winning_streaks"]
                max_win = streaks.get("longest_win_streak", 0)
                max_loss = streaks.get("longest_loss_streak", 0)
                
                print(f"  En Uzun Kazanç Serisi: {COLORS['green']}{max_win}{COLORS['reset']} işlem")
                print(f"  En Uzun Kayıp Serisi: {COLORS['red']}{max_loss}{COLORS['reset']} işlem")
            
            # Kaydedilen dosyalar
            print(f"\n{COLORS['bright']}📝 KAYDEDILEN DOSYALAR:{COLORS['reset']}")
            print(f"  📊 Performans Raporu: {COLORS['green']}{PERFORMANCE_REPORT_FILE}{COLORS['reset']}")
            print(f"  🔄 Öğrenme Geri Bildirimi: {COLORS['green']}{LEARNING_FEEDBACK_FILE}{COLORS['reset']}")
            
            print(f"\n{COLORS['bright']}{COLORS['cyan']}{'=' * 60}{COLORS['reset']}")
            print(f"{COLORS['green']}✅ Performans analizi tamamlandı!{COLORS['reset']}")
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
    parser = argparse.ArgumentParser(description="SentientTrader.AI Performance Tracker")
    parser.add_argument("--simulation", action="store_true", help="Simülasyon modunda çalış")
    parser.add_argument("--min-trades", type=int, default=3, help="Analiz için minimum işlem sayısı")
    parser.add_argument("--timeframe", type=int, default=30, help="Analiz süresi (gün)")
    return parser.parse_args()

def display_banner() -> None:
    """Başlık banner'ını göster"""
    banner = f"""
{COLORS["bright"]}{COLORS["cyan"]}
██████╗░███████╗██████╗░███████╗░█████╗░██████╗░███╗░░░███╗░█████╗░███╗░░██╗░█████╗░███████╗
██╔══██╗██╔════╝██╔══██╗██╔════╝██╔══██╗██╔══██╗████╗░████║██╔══██╗████╗░██║██╔══██╗██╔════╝
██████╔╝█████╗░░██████╔╝█████╗░░██║░░██║██████╔╝██╔████╔██║███████║██╔██╗██║██║░░╚═╝█████╗░░
██╔═══╝░██╔══╝░░██╔══██╗██╔══╝░░██║░░██║██╔══██╗██║╚██╔╝██║██╔══██║██║╚████║██║░░██╗██╔══╝░░
██║░░░░░███████╗██║░░██║██║░░░░░╚█████╔╝██║░░██║██║░╚═╝░██║██║░░██║██║░╚███║╚█████╔╝███████╗
╚═╝░░░░░╚══════╝╚═╝░░╚═╝╚═╝░░░░░░╚════╝░╚═╝░░╚═╝╚═╝░░░░░╚═╝╚═╝░░╚═╝╚═╝░░╚══╝░╚════╝░╚══════╝

████████╗██████╗░░█████╗░░█████╗░██╗░░██╗███████╗██████╗░
╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██║░██╔╝██╔════╝██╔══██╗
░░░██║░░░██████╔╝███████║██║░░╚═╝█████═╝░█████╗░░██████╔╝
░░░██║░░░██╔══██╗██╔══██║██║░░██╗██╔═██╗░██╔══╝░░██╔══██╗
░░░██║░░░██║░░██║██║░░██║╚█████╔╝██║░╚██╗███████╗██║░░██║
░░░╚═╝░░░╚═╝░░╚═╝╚═╝░░╚═╝░╚════╝░╚═╝░░╚═╝╚══════╝╚═╝░░╚═╝
{COLORS["reset"]}{COLORS["magenta"]}                                             V2.0{COLORS["reset"]}

{COLORS["yellow"]}📅 2025-04-21 23:57:21 | 👤 mstfatiryaki{COLORS["reset"]}
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
        
        # Performans takipçisi oluştur
        tracker = PerformanceTracker(simulation_mode=args.simulation)
        
        # Özel konfigürasyonları ayarla
        tracker.min_trades_for_analysis = args.min_trades
        tracker.analysis_timeframe = args.timeframe
        
        # Çalışma modunu göster
        mode = "SİMÜLASYON" if args.simulation else "GERÇEK İŞLEM"
        logger.info(f"Performance Tracker {mode} modunda başlatılıyor...")
        print(f"{COLORS['cyan']}ℹ️ Performans Takipçisi {mode} modunda başlatılıyor...{COLORS['reset']}")
        
        # Takipçiyi çalıştır
        success = tracker.run()
        
        if success:
            logger.info("Performans takipçisi başarıyla çalıştı.")
            return 0
        else:
            logger.error("Performans takipçisi çalıştırılamadı veya analiz tamamlanamadı!")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n{COLORS['yellow']}⚠️ Kullanıcı tarafından durduruldu!{COLORS['reset']}")
        return 130  # SIGINT için standart çıkış kodu
    except Exception as e:
        logger.critical(f"Beklenmeyen hata: {e}")
        print(f"\n{COLORS['red']}❌ Beklenmeyen hata: {e}{COLORS['reset']}")
        return 1

# Modül seviyesinde fonksiyon - historical_loader.py ile entegrasyon için
def add_market_benchmark(source, coin_data: dict) -> bool:
    """
    Historical Loader'dan gelen piyasa karşılaştırma verisini işle
    
    Args:
        source: Veri kaynağı (çağıran modül)
        coin_data: Coin karşılaştırma verisi (tarihsel fiyat, hacim, vb.)
        
    Returns:
        bool: İşlem başarılı mı?
    """
    try:
        logger.info(f"Market benchmark verisi alındı: {len(coin_data) if isinstance(coin_data, dict) else 'N/A'} coin")
        
        # Veriyi global market_data değişkenine kaydet
        global market_data
        market_data = coin_data
        
        # Bellek çekirdeğine kaydet (eğer mevcutsa)
        if MEMORY_CORE_AVAILABLE:
            memory_core.store_data("market_benchmark", coin_data)
            
        return True
        
    except Exception as e:
        logger.error(f"Market benchmark verisi işlenirken hata: {e}")
        return False


if __name__ == "__main__":
    sys.exit(main())


