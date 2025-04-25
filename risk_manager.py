from random import choice
import random
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SentientTrader.AI - Risk Manager V2
-----------------------------------
Bu modül, gerçekleştirilen işlemlerin risk analizini yapar
ve sistemin risk durumunu değerlendirerek uyarılar üretir.
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
    handlers=[logging.StreamHandler(), logging.FileHandler("risk_manager.log")]
)
logger = logging.getLogger("RiskManager")

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

# Alert sistem entegrasyonu
try:
    from alert_system import send_alert, critical, warning, info
    ALERT_SYSTEM_AVAILABLE = True
except ImportError:
    ALERT_SYSTEM_AVAILABLE = False
    logger.warning("Alert System modülü bulunamadı, uyarılar sadece konsola yazılacak.")

# Sabitler
CURRENT_TIME = "2025-04-21 18:58:55"  # UTC
CURRENT_USER = "mstfatiryaki"

# Dosya yolları
EXECUTED_TRADES_LOG = "executed_trades_log.json"
RISK_REPORT_FILE = "risk_report.json"
RISK_TAGS_FILE = "risk_tags.json"

# Risk eşik değerleri
RISK_THRESHOLDS = {
    "max_position_exposure": 20.0,         # % olarak maksimum pozisyon maruziyeti
    "max_leverage": 10,                    # Maksimum kaldıraç oranı
    "high_risk_leverage": 5,               # Yüksek riskli kaldıraç eşiği
    "min_risk_reward_ratio": 1.0,          # Minimum risk/ödül oranı
    "max_daily_loss_percentage": 5.0,      # Günlük maksimum kayıp yüzdesi
    "max_drawdown_percentage": 20.0,       # Maksimum drawdown yüzdesi
    "portfolio_concentration_limit": 40.0,  # % olarak bir coine maksimum yatırım
    "high_risk_trade_frequency": 20,       # 24 saat içinde maksimum işlem sayısı
    "min_coin_diversity": 3,               # Minimum coin çeşitliliği
    "critical_consecutive_losses": 5       # Kritik ardışık kayıp sayısı
}

class RiskManager:
    """SentientTrader.AI risk analizi ve yönetimi"""
    
    def __init__(self, simulation_mode: bool = False):
        """
        Risk yöneticisi değişkenlerini başlat
        
        Args:
            simulation_mode (bool, optional): Simülasyon modu aktif mi?
        """
        self.simulation_mode = simulation_mode
        
        # Veri depoları
        self.trades_data = {}                   # İşlem verileri
        self.open_positions = []                # Açık pozisyonlar
        self.closed_positions = []              # Kapalı pozisyonlar
        
        # Risk metrikleri
        self.coin_risk_metrics = {}             # Coin bazlı risk metrikleri
        self.portfolio_risk_metrics = {}        # Portföy risk metrikleri
        self.risk_warnings = []                 # Risk uyarıları
        self.critical_risks = []                # Kritik riskler
        
        # Risk örüntüleri
        self.risk_patterns = []                 # Tespit edilen risk örüntüleri
        self.risk_tags = {}                     # Risk etiketleri
        
        # İzlenecek coinler ve stratejiler
        self.monitored_coins = set()            # İzlenen coinler
        self.monitored_strategies = set()       # İzlenen stratejiler
        
        # Konfigürasyon
        self.risk_thresholds = RISK_THRESHOLDS.copy()
        self.current_capital = 10000.0          # Varsayılan sermaye ($)
        self.max_alerts_per_run = 5             # Bir çalıştırmada maksimum kritik uyarı sayısı
        self.alert_count = 0                    # Gönderilen uyarı sayısı
    
    def load_trades_data(self) -> bool:
        """
        İşlem kayıtlarını yükle
        
        Returns:
            bool: Başarılı mı?
        """
        try:
            if not os.path.exists(EXECUTED_TRADES_LOG):
                logger.error(f"{EXECUTED_TRADES_LOG} dosyası bulunamadı!")
                return False
            
            with open(EXECUTED_TRADES_LOG, "r", encoding="utf-8") as f:
                self.trades_data = json.load(f)
            
            # Yapı kontrolü
            if not isinstance(self.trades_data, dict) or "trades" not in self.trades_data:
                logger.error(f"{EXECUTED_TRADES_LOG} beklenen formatta değil!")
                return False
            
            # Açık ve kapalı pozisyonları ayır
            trades = self.trades_data.get("trades", [])
            self.open_positions = [t for t in trades if t.get("status", "") == "OPEN"]
            self.closed_positions = [t for t in trades if t.get("status", "") == "CLOSED"]
            
            # İzlenen coin ve stratejileri güncelle
            self.update_monitored_items()
            
            trades_count = len(trades)
            open_count = len(self.open_positions)
            closed_count = len(self.closed_positions)
            
            logger.info(f"{EXECUTED_TRADES_LOG} yüklendi: {trades_count} işlem kaydı ({open_count} açık, {closed_count} kapalı)")
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"{EXECUTED_TRADES_LOG} dosyası JSON hatası: {e}")
            return False
        except Exception as e:
            logger.error(f"{EXECUTED_TRADES_LOG} yüklenirken hata: {e}")
            return False
    
    def update_monitored_items(self) -> None:
        """İzlenen coin ve stratejileri güncelle"""
        all_trades = self.open_positions + self.closed_positions
        
        # İzlenen coinleri güncelle
        self.monitored_coins = set(trade.get("symbol", "") for trade in all_trades if trade.get("symbol"))
        
        # İzlenen stratejileri güncelle
        self.monitored_strategies = set(trade.get("strategy", "") for trade in all_trades if trade.get("strategy"))
        
        logger.debug(f"İzlenen coinler: {', '.join(self.monitored_coins)}")
        logger.debug(f"İzlenen stratejiler: {', '.join(self.monitored_strategies)}")
    
    def calculate_coin_exposure(self) -> Dict[str, Dict[str, Any]]:
        """
        Coin bazlı pozisyon maruziyetini hesapla
        
        Returns:
            Dict[str, Dict[str, Any]]: Coin bazlı maruziyet verileri
        """
        logger.info("Coin bazlı maruziyet hesaplanıyor...")
        
        coin_exposure = {}
        
        # Toplam yatırım tutarını hesapla
        total_investment = 0.0
        
        for position in self.open_positions:
            investment = position.get("investment_usd", 0)
            leverage = position.get("leverage", 1)
            
            # Kaldıraçlı yatırım tutarını hesapla
            leveraged_investment = investment * leverage
            total_investment += investment  # Toplam yatırıma kaldıraçsız ekle
        
        # Sermaye değeri kontrol et
        if total_investment > 0:
            # Sermayeyi güncelle
            self.current_capital = max(self.current_capital, total_investment * 1.5)  # Güvenli tahmini sermaye
        
        # Coin bazlı maruziyeti hesapla
        for coin in self.monitored_coins:
            # Bu coine ait açık pozisyonları filtrele
            coin_positions = [p for p in self.open_positions if p.get("symbol", "") == coin]
            
            if not coin_positions:
                continue
            
            # Coin için toplam yatırım ve pozisyon boyutu
            coin_investment = sum(p.get("investment_usd", 0) for p in coin_positions)
            
            # Kaldıraçlı pozisyon boyutu
            leveraged_positions = [(p.get("investment_usd", 0), p.get("leverage", 1)) for p in coin_positions]
            leveraged_exposure = sum(inv * lev for inv, lev in leveraged_positions)
            
            # Maruziyet yüzdeleri
            investment_percentage = (coin_investment / self.current_capital * 100) if self.current_capital > 0 else 0
            exposure_percentage = (leveraged_exposure / self.current_capital * 100) if self.current_capital > 0 else 0
            
            # Ortalama kaldıraç
            avg_leverage = sum(p.get("leverage", 1) for p in coin_positions) / len(coin_positions)
            
            # Açık pozisyon sayısı
            open_positions_count = len(coin_positions)
            
            # İşlem yönü dağılımı
            long_positions = [p for p in coin_positions if p.get("operation", "") == "LONG"]
            short_positions = [p for p in coin_positions if p.get("operation", "") == "SHORT"]
            
            # Risk seviyesi belirleme
            risk_level = "LOW"
            
            if exposure_percentage > self.risk_thresholds["max_position_exposure"]:
                risk_level = "HIGH"
            elif exposure_percentage > self.risk_thresholds["max_position_exposure"] / 2:
                risk_level = "MEDIUM"
                
            if avg_leverage > self.risk_thresholds["high_risk_leverage"]:
                risk_level = "HIGH"
            
            # Sonuçları kaydet
            coin_exposure[coin] = {
                "investment_usd": round(coin_investment, 2),
                "leveraged_exposure_usd": round(leveraged_exposure, 2),
                "investment_percentage": round(investment_percentage, 2),
                "exposure_percentage": round(exposure_percentage, 2),
                "avg_leverage": round(avg_leverage, 2),
                "open_positions": open_positions_count,
                "long_positions": len(long_positions),
                "short_positions": len(short_positions),
                "risk_level": risk_level
            }
            
            # Yüksek maruziyeti olan coinleri uyarı listesine ekle
            if exposure_percentage > self.risk_thresholds["max_position_exposure"]:
                self.risk_warnings.append(f"{coin} için yüksek maruziyet riski: %{exposure_percentage:.2f} (limit: %{self.risk_thresholds['max_position_exposure']})")
                
                # Çok yüksek maruziyet için kritik uyarı
                if exposure_percentage > self.risk_thresholds["max_position_exposure"] * 1.5:
                    self.critical_risks.append(f"{coin} için kritik seviyede yüksek maruziyet: %{exposure_percentage:.2f}")
            
            # Yüksek kaldıraç uyarısı
            if avg_leverage > self.risk_thresholds["high_risk_leverage"]:
                self.risk_warnings.append(f"{coin} için yüksek kaldıraç riski: {avg_leverage:.2f}x (limit: {self.risk_thresholds['high_risk_leverage']}x)")
                
                # Çok yüksek kaldıraç için kritik uyarı
                if avg_leverage > self.risk_thresholds["max_leverage"]:
                    self.critical_risks.append(f"{coin} için kritik seviyede yüksek kaldıraç: {avg_leverage:.2f}x")
            
            logger.info(f"{coin} maruziyet analizi: %{exposure_percentage:.2f} ({risk_level})")
        
        return coin_exposure
    
    def analyze_trade_frequency(self) -> Dict[str, Any]:
        """
        İşlem yoğunluğu ve dağılımını analiz et
        
        Returns:
            Dict[str, Any]: İşlem yoğunluğu ve dağılım analizi
        """
        logger.info("İşlem yoğunluğu analiz ediliyor...")
        
        frequency_analysis = {
            "daily_trade_frequency": {},
            "hourly_distribution": {},
            "coin_diversity": {},
            "strategy_distribution": {},
            "overall_metrics": {}
        }
        
        all_trades = self.open_positions + self.closed_positions
        
        if not all_trades:
            logger.warning("İşlem yoğunluğu analizi için yeterli veri yok!")
            return frequency_analysis
        
        # Günlük ve saatlik işlem dağılımını analiz et
        daily_trades = defaultdict(list)
        hourly_trades = defaultdict(int)
        
        for trade in all_trades:
            if "timestamp" in trade:
                try:
                    trade_time = datetime.datetime.strptime(trade["timestamp"], "%Y-%m-%d %H:%M:%S")
                    day_key = trade_time.strftime("%Y-%m-%d")
                    hour_key = trade_time.hour
                    
                    daily_trades[day_key].append(trade)
                    hourly_trades[hour_key] += 1
                except:
                    continue
        
        # Günlük işlem istatistikleri
        for day, day_trades in daily_trades.items():
            # Coin çeşitliliği
            coins_traded = set(t.get("symbol", "") for t in day_trades if t.get("symbol"))
            
            # İşlem sonuçları (kapalı işlemler için)
            closed_day_trades = [t for t in day_trades if t.get("status", "") == "CLOSED" and "profit_loss" in t]
            win_trades = [t for t in closed_day_trades if t.get("profit_loss", 0) > 0]
            
            win_rate = (len(win_trades) / len(closed_day_trades) * 100) if closed_day_trades else 0
            
            # Günlük metrikleri kaydet
            frequency_analysis["daily_trade_frequency"][day] = {
                "total_trades": len(day_trades),
                "unique_coins": len(coins_traded),
                "coins": list(coins_traded),
                "win_rate": round(win_rate, 2) if closed_day_trades else None
            }
            
            # Çok yüksek günlük işlem sayısı için uyarı
            if len(day_trades) > self.risk_thresholds["high_risk_trade_frequency"]:
                self.risk_warnings.append(f"{day} tarihinde yüksek işlem yoğunluğu: {len(day_trades)} işlem (limit: {self.risk_thresholds['high_risk_trade_frequency']})")
        
        # Saatlik dağılım
        for hour in range(24):
            frequency_analysis["hourly_distribution"][hour] = hourly_trades.get(hour, 0)
        
        # Coin çeşitliliği
        coin_distribution = defaultdict(int)
        for trade in all_trades:
            coin = trade.get("symbol", "")
            if coin:
                coin_distribution[coin] += 1
        
        # Her coin için yüzdeleri hesapla
        total_trades = len(all_trades)
        for coin, count in coin_distribution.items():
            percentage = (count / total_trades * 100) if total_trades > 0 else 0
            frequency_analysis["coin_diversity"][coin] = {
                "trade_count": count,
                "percentage": round(percentage, 2)
            }
        
        # Strateji dağılımı
        strategy_distribution = defaultdict(int)
        for trade in all_trades:
            strategy = trade.get("strategy", "")
            if strategy:
                strategy_distribution[strategy] += 1
        
        # Her strateji için yüzdeleri hesapla
        for strategy, count in strategy_distribution.items():
            percentage = (count / total_trades * 100) if total_trades > 0 else 0
            frequency_analysis["strategy_distribution"][strategy] = {
                "trade_count": count,
                "percentage": round(percentage, 2)
            }
        
        # Genel metrikler
        avg_daily_trades = sum(len(trades) for trades in daily_trades.values()) / len(daily_trades) if daily_trades else 0
        unique_coins = len(set(t.get("symbol", "") for t in all_trades if t.get("symbol")))
        unique_strategies = len(set(t.get("strategy", "") for t in all_trades if t.get("strategy")))
        
        frequency_analysis["overall_metrics"] = {
            "total_trades": total_trades,
            "average_daily_trades": round(avg_daily_trades, 2),
            "unique_coins": unique_coins,
            "unique_strategies": unique_strategies,
            "most_active_day": max(daily_trades.items(), key=lambda x: len(x[1]))[0] if daily_trades else None,
            "most_active_hour": max(hourly_trades.items(), key=lambda x: x[1])[0] if hourly_trades else None
        }
        
        # Düşük coin çeşitliliği uyarısı
        if unique_coins < self.risk_thresholds["min_coin_diversity"]:
            self.risk_warnings.append(f"Düşük coin çeşitliliği: {unique_coins} coin (minimum: {self.risk_thresholds['min_coin_diversity']})")
        
        return frequency_analysis
    
    def analyze_profit_loss_metrics(self) -> Dict[str, Any]:
        """
        Kâr/zarar metriklerini analiz et
        
        Returns:
            Dict[str, Any]: Kâr/zarar analizi
        """
        logger.info("Kâr/zarar metrikleri analiz ediliyor...")
        
        pnl_analysis = {
            "largest_profit_trades": [],
            "largest_loss_trades": [],
            "risk_reward_analysis": {},
            "drawdown_analysis": {},
            "consecutive_losses": {},
            "summary": {}
        }
        
        # Sadece kapalı işlemleri analiz et
        closed_trades = self.closed_positions
        
        if not closed_trades:
            logger.warning("Kâr/zarar analizi için kapalı işlem bulunamadı!")
            return pnl_analysis
        
        # Kârlı ve zararlı işlemleri ayır
        profitable_trades = [t for t in closed_trades if t.get("profit_loss", 0) > 0]
        loss_trades = [t for t in closed_trades if t.get("profit_loss", 0) <= 0]
        
        # En büyük kârlı işlemler
        sorted_profit_trades = sorted(profitable_trades, key=lambda x: x.get("profit_loss", 0), reverse=True)
        pnl_analysis["largest_profit_trades"] = sorted_profit_trades[:5]
        
        # En büyük zararlı işlemler
        sorted_loss_trades = sorted(loss_trades, key=lambda x: x.get("profit_loss", 0))
        pnl_analysis["largest_loss_trades"] = sorted_loss_trades[:5]
        
        # Risk/ödül oranı analizi
        risk_reward_ratios = []
        
        for trade in closed_trades:
            if "entry_price" in trade and "take_profit_price" in trade and "stop_loss_price" in trade:
                entry = float(trade.get("entry_price", 0))
                take_profit = float(trade.get("take_profit_price", 0))
                stop_loss = float(trade.get("stop_loss_price", 0))
                
                # Risk ve ödül hesaplama
                if trade.get("operation", "") == "LONG":
                    reward = take_profit - entry
                    risk = entry - stop_loss
                else:  # SHORT
                    reward = entry - take_profit
                    risk = stop_loss - entry
                
                if risk > 0:
                    rr_ratio = reward / risk
                    risk_reward_ratios.append((trade, rr_ratio))
        
        # Düşük risk/ödül oranlı işlemleri tespit et
        low_rr_trades = [(trade, ratio) for trade, ratio in risk_reward_ratios if ratio < self.risk_thresholds["min_risk_reward_ratio"]]
        
        # Düşük risk/ödül oranı için uyarı
        if low_rr_trades:
            self.risk_warnings.append(f"Düşük risk/ödül oranlı {len(low_rr_trades)} işlem tespit edildi")
            
            # Risk/ödül analizi sonuçlarını kaydet
            pnl_analysis["risk_reward_analysis"]["low_rr_trades"] = [(t.get("symbol", ""), t.get("strategy", ""), round(ratio, 2)) for t, ratio in low_rr_trades]
            pnl_analysis["risk_reward_analysis"]["avg_rr_ratio"] = sum(ratio for _, ratio in risk_reward_ratios) / len(risk_reward_ratios) if risk_reward_ratios else 0
        
        # Ardışık kayıpları analiz et
        consecutive_losses = self._analyze_consecutive_losses(closed_trades)
        pnl_analysis["consecutive_losses"] = consecutive_losses
        
        if consecutive_losses["max_consecutive_losses"] >= self.risk_thresholds["critical_consecutive_losses"]:
            self.critical_risks.append(f"Kritik ardışık kayıp serisi: {consecutive_losses['max_consecutive_losses']} işlem")
            
            if "current_consecutive_losses" in consecutive_losses and consecutive_losses["current_consecutive_losses"] >= self.risk_thresholds["critical_consecutive_losses"]:
                self.critical_risks.append(f"GÜNCEL ardışık kayıp serisi: {consecutive_losses['current_consecutive_losses']} işlem - ACİL DURUM!")
        
        # Drawdown analizi
        drawdown = self._analyze_drawdown(closed_trades)
        pnl_analysis["drawdown_analysis"] = drawdown
        
        if drawdown["max_drawdown_percentage"] > self.risk_thresholds["max_drawdown_percentage"]:
            self.critical_risks.append(f"Kritik drawdown seviyesi: %{drawdown['max_drawdown_percentage']:.2f}")
        
        # Özet metrikleri
        total_profit = sum(t.get("profit_loss", 0) for t in profitable_trades)
        total_loss = sum(abs(t.get("profit_loss", 0)) for t in loss_trades)
        net_pnl = total_profit - total_loss
        
        win_rate = (len(profitable_trades) / len(closed_trades) * 100) if closed_trades else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        pnl_analysis["summary"] = {
            "total_trades": len(closed_trades),
            "profitable_trades": len(profitable_trades),
            "loss_trades": len(loss_trades),
            "win_rate": round(win_rate, 2),
            "total_profit": round(total_profit, 2),
            "total_loss": round(total_loss, 2),
            "net_pnl": round(net_pnl, 2),
            "profit_factor": round(profit_factor, 2)
        }
        
        return pnl_analysis
    
    def analyze_portfolio_risk(self) -> Dict[str, Any]:
        """
        Portföy risk analizi yap
        
        Returns:
            Dict[str, Any]: Portföy risk analizi
        """
        logger.info("Portföy risk analizi yapılıyor...")
        
        portfolio_risk = {
            "concentration_risk": {},
            "strategy_risk": {},
            "leverage_risk": {},
            "overall_risk_score": 0,
            "risk_level": "LOW"
        }
        
        # Açık pozisyonları analiz et
        open_positions = self.open_positions
        
        if not open_positions:
            logger.warning("Portföy risk analizi için açık pozisyon bulunamadı!")
            return portfolio_risk
        
        # 1. Konsantrasyon riski (bir coindeki yoğunlaşma)
        coin_investments = defaultdict(float)
        total_investment = 0.0
        
        for position in open_positions:
            coin = position.get("symbol", "")
            investment = position.get("investment_usd", 0)
            
            if coin:
                coin_investments[coin] += investment
                total_investment += investment
        
        # Coin konsantrasyon yüzdeleri
        concentration_percentages = {}
        
        for coin, investment in coin_investments.items():
            percentage = (investment / total_investment * 100) if total_investment > 0 else 0
            concentration_percentages[coin] = round(percentage, 2)
        
        # En yüksek konsantrasyon
        max_concentration = max(concentration_percentages.values()) if concentration_percentages else 0
        max_concentration_coin = max(concentration_percentages.items(), key=lambda x: x[1])[0] if concentration_percentages else None
        
        # Konsantrasyon riski skoru (0-100)
        concentration_risk_score = (max_concentration / self.risk_thresholds["portfolio_concentration_limit"]) * 100
        concentration_risk_score = min(100, concentration_risk_score)
        
        portfolio_risk["concentration_risk"] = {
            "percentages": concentration_percentages,
            "max_concentration": max_concentration,
            "max_concentration_coin": max_concentration_coin,
            "risk_score": round(concentration_risk_score, 2)
        }
        
        # Yüksek konsantrasyon uyarısı
        if max_concentration > self.risk_thresholds["portfolio_concentration_limit"]:
            self.risk_warnings.append(f"{max_concentration_coin} coininde yüksek konsantrasyon riski: %{max_concentration:.2f}")
            
            # Çok yüksek konsantrasyon için kritik uyarı
            if max_concentration > self.risk_thresholds["portfolio_concentration_limit"] * 1.5:
                self.critical_risks.append(f"{max_concentration_coin} coininde kritik konsantrasyon: %{max_concentration:.2f}")
        
        # 2. Strateji riski (tek stratejiye bağımlılık)
        strategy_investments = defaultdict(float)
        
        for position in open_positions:
            strategy = position.get("strategy", "")
            investment = position.get("investment_usd", 0)
            
            if strategy:
                strategy_investments[strategy] += investment
        
        # Strateji yüzdeleri
        strategy_percentages = {}
        
        for strategy, investment in strategy_investments.items():
            percentage = (investment / total_investment * 100) if total_investment > 0 else 0
            strategy_percentages[strategy] = round(percentage, 2)
        
        # En yüksek strateji konsantrasyonu
        max_strategy_concentration = max(strategy_percentages.values()) if strategy_percentages else 0
        max_strategy = max(strategy_percentages.items(), key=lambda x: x[1])[0] if strategy_percentages else None
        
        # Strateji riski skoru (0-100)
        strategy_risk_score = (max_strategy_concentration / self.risk_thresholds["portfolio_concentration_limit"]) * 100
        strategy_risk_score = min(100, strategy_risk_score)
        
        portfolio_risk["strategy_risk"] = {
            "percentages": strategy_percentages,
            "max_concentration": max_strategy_concentration,
            "max_strategy": max_strategy,
            "risk_score": round(strategy_risk_score, 2)
        }
        
        # 3. Kaldıraç riski
        leveraged_positions = [p for p in open_positions if p.get("leverage", 1) > 1]
        high_leverage_positions = [p for p in leveraged_positions if p.get("leverage", 1) > self.risk_thresholds["high_risk_leverage"]]
        
        # Toplam kaldıraçlı yatırım
        total_leveraged_investment = sum(p.get("investment_usd", 0) for p in leveraged_positions)
        high_leverage_investment = sum(p.get("investment_usd", 0) for p in high_leverage_positions)
        
        # Kaldıraçlı yatırım yüzdesi
        leveraged_percentage = (total_leveraged_investment / total_investment * 100) if total_investment > 0 else 0
        high_leverage_percentage = (high_leverage_investment / total_investment * 100) if total_investment > 0 else 0
        
        # Ortalama kaldıraç
        avg_leverage = sum(p.get("leverage", 1) for p in open_positions) / len(open_positions) if open_positions else 1
        
        # Kaldıraç riski skoru (0-100)
        leverage_risk_score = 0
        
        if avg_leverage > 1:
            # Ortalama kaldıraç ve yüksek kaldıraçlı pozisyon yüzdesini baz alarak skor hesapla
            leverage_factor = (avg_leverage / self.risk_thresholds["max_leverage"]) * 70
            high_leverage_factor = (high_leverage_percentage / 100) * 30
            leverage_risk_score = leverage_factor + high_leverage_factor
            leverage_risk_score = min(100, leverage_risk_score)
        
        portfolio_risk["leverage_risk"] = {
            "total_leveraged_positions": len(leveraged_positions),
            "high_leverage_positions": len(high_leverage_positions),
            "leveraged_percentage": round(leveraged_percentage, 2),
            "high_leverage_percentage": round(high_leverage_percentage, 2),
            "avg_leverage": round(avg_leverage, 2),
            "risk_score": round(leverage_risk_score, 2)
        }
        
        # Yüksek kaldıraç uyarısı
        if avg_leverage > self.risk_thresholds["high_risk_leverage"]:
            self.risk_warnings.append(f"Portföyde yüksek ortalama kaldıraç: {avg_leverage:.2f}x")
            
            # Çok yüksek kaldıraç için kritik uyarı
            if avg_leverage > self.risk_thresholds["max_leverage"]:
                self.critical_risks.append(f"Portföyde kritik ortalama kaldıraç: {avg_leverage:.2f}x")
        
        # Genel risk skoru (tüm risk faktörlerinin ağırlıklı ortalaması)
        overall_risk_score = (
            concentration_risk_score * 0.4 +  # %40 konsantrasyon riski
            strategy_risk_score * 0.2 +      # %20 strateji riski
            leverage_risk_score * 0.4        # %40 kaldıraç riski
        )
        
        # Risk seviyesi belirleme
        risk_level = "LOW"
        if overall_risk_score >= 75:
            risk_level = "CRITICAL"
        elif overall_risk_score >= 50:
            risk_level = "HIGH"
        elif overall_risk_score >= 25:
            risk_level = "MEDIUM"
        
        portfolio_risk["overall_risk_score"] = round(overall_risk_score, 2)
        portfolio_risk["risk_level"] = risk_level
        
        # Genel risk seviyesi uyarısı
        if risk_level in ["HIGH", "CRITICAL"]:
            self.risk_warnings.append(f"Portföy genel risk seviyesi: {risk_level} (skor: {overall_risk_score:.2f}/100)")
            
            if risk_level == "CRITICAL":
                self.critical_risks.append(f"Portföy kritik risk seviyesinde: {overall_risk_score:.2f}/100")
        
        return portfolio_risk
    
    def identify_risk_patterns(self) -> Dict[str, Any]:
        """
        Risk örüntülerini tespit et ve etiketle
        
        Returns:
            Dict[str, Any]: Risk örüntüleri ve etiketleri
        """
        logger.info("Risk örüntüleri tespit ediliyor...")
        
        risk_patterns = {
            "trading_patterns": [],
            "risk_correlations": {},
            "tagged_risks": {}
        }
        
        # Tüm işlemleri analiz et
        all_trades = self.open_positions + self.closed_positions
        
        if not all_trades:
            logger.warning("Risk örüntüleri analizi için yeterli veri yok!")
            return risk_patterns
        
        # 1. İşlem örüntülerini tespit et
        patterns = []
        
        # Yüksek hacimli işlemler
        high_volume_trades = [t for t in all_trades if t.get("investment_usd", 0) > 1000]
        if high_volume_trades:
            patterns.append({
                "name": "high_volume_trading",
                "description": "Yüksek hacimli işlemler",
                "count": len(high_volume_trades),
                "details": [{"id": t.get("trade_id", ""), "symbol": t.get("symbol", ""), "amount": t.get("investment_usd", 0)} for t in high_volume_trades[:5]]
            })
        
        # Yüksek kaldıraçlı işlemler
        high_leverage_trades = [t for t in all_trades if t.get("leverage", 1) > self.risk_thresholds["high_risk_leverage"]]
        if high_leverage_trades:
            patterns.append({
                "name": "high_leverage_trading",
                "description": "Yüksek kaldıraçlı işlemler",
                "count": len(high_leverage_trades),
                "details": [{"id": t.get("trade_id", ""), "symbol": t.get("symbol", ""), "leverage": t.get("leverage", 1)} for t in high_leverage_trades[:5]]
            })
        
        # Dar stop-loss işlemleri
        narrow_sl_trades = []
        for trade in all_trades:
            if "entry_price" in trade and "stop_loss_price" in trade:
                entry = float(trade.get("entry_price", 0))
                stop_loss = float(trade.get("stop_loss_price", 0))
                
                # Risk hesaplama
                if trade.get("operation", "") == "LONG":
                    risk_pct = ((entry - stop_loss) / entry) * 100
                else:  # SHORT
                    risk_pct = ((stop_loss - entry) / entry) * 100
                
                if risk_pct < 1.0:  # %1'den az risk
                    narrow_sl_trades.append((trade, risk_pct))
        
        if narrow_sl_trades:
            patterns.append({
                "name": "narrow_stop_loss",
                "description": "Dar stop-loss aralıklı işlemler",
                "count": len(narrow_sl_trades),
                "details": [{"id": t.get("trade_id", ""), "symbol": t.get("symbol", ""), "risk_percentage": round(risk_pct, 2)} for t, risk_pct in narrow_sl_trades[:5]]
            })
        
        # Düşük risk/ödül işlemleri
        low_rr_trades = []
        for trade in all_trades:
            if "entry_price" in trade and "take_profit_price" in trade and "stop_loss_price" in trade:
                entry = float(trade.get("entry_price", 0))
                take_profit = float(trade.get("take_profit_price", 0))
                stop_loss = float(trade.get("stop_loss_price", 0))
                
                # Risk ve ödül hesaplama
                if trade.get("operation", "") == "LONG":
                    reward = take_profit - entry
                    risk = entry - stop_loss
                else:  # SHORT
                    reward = entry - take_profit
                    risk = stop_loss - entry
                
                if risk > 0:
                    rr_ratio = reward / risk
                    if rr_ratio < self.risk_thresholds["min_risk_reward_ratio"]:
                        low_rr_trades.append((trade, rr_ratio))
        
        if low_rr_trades:
            patterns.append({
                "name": "low_risk_reward_ratio",
                "description": "Düşük risk/ödül oranlı işlemler",
                "count": len(low_rr_trades),
                "details": [{"id": t.get("trade_id", ""), "symbol": t.get("symbol", ""), "rr_ratio": round(ratio, 2)} for t, ratio in low_rr_trades[:5]]
            })
        
        # Sık ticaret örüntüsü
        freq_trades = []
        try:
            sorted_trades = sorted(all_trades, key=lambda x: datetime.datetime.strptime(x.get("timestamp", "2000-01-01 00:00:00"), "%Y-%m-%d %H:%M:%S"))
            
            if len(sorted_trades) >= 2:
                for i in range(1, len(sorted_trades)):
                    prev_time = datetime.datetime.strptime(sorted_trades[i-1].get("timestamp", "2000-01-01 00:00:00"), "%Y-%m-%d %H:%M:%S")
                    curr_time = datetime.datetime.strptime(sorted_trades[i].get("timestamp", "2000-01-01 00:00:00"), "%Y-%m-%d %H:%M:%S")
                    
                    time_diff = (curr_time - prev_time).total_seconds() / 60  # Dakika cinsinden
                    
                    if time_diff < 15:  # 15 dakikadan kısa aralık
                        freq_trades.append((sorted_trades[i], time_diff))
        except Exception as e:
            logger.error(f"Sık ticaret analizi hatası: {e}")
            
        if freq_trades:
            patterns.append({
                "name": "frequent_trading",
                "description": "Kısa aralıklarla yapılan işlemler",
                "count": len(freq_trades),
                "details": [{"id": t.get("trade_id", ""), "symbol": t.get("symbol", ""), "minutes_since_last": round(mins, 2)} for t, mins in freq_trades[:5]]
            })
        
        risk_patterns["trading_patterns"] = patterns
        
        # 2. Risk korelasyonlarını hesapla
        correlations = {}
        
        # Strateji-sonuç korelasyonu
        strategy_outcomes = defaultdict(lambda: {"wins": 0, "losses": 0})
        
        for trade in [t for t in all_trades if t.get("status", "") == "CLOSED" and "profit_loss" in t]:
            strategy = trade.get("strategy", "")
            if strategy:
                if trade.get("profit_loss", 0) > 0:
                    strategy_outcomes[strategy]["wins"] += 1
                else:
                    strategy_outcomes[strategy]["losses"] += 1
        
        # Win rate hesapla
        strategy_win_rates = {}
        for strategy, outcomes in strategy_outcomes.items():
            total = outcomes["wins"] + outcomes["losses"]
            win_rate = (outcomes["wins"] / total * 100) if total > 0 else 0
            strategy_win_rates[strategy] = {
                "win_rate": round(win_rate, 2),
                "total_trades": total
            }
        
        correlations["strategy_performance"] = strategy_win_rates
        
        # Coin-kaldıraç korelasyonu
        coin_leverage = defaultdict(list)
        
        for trade in all_trades:
            coin = trade.get("symbol", "")
            leverage = trade.get("leverage", 1)
            
            if coin:
                coin_leverage[coin].append(leverage)
        
        # Ortalama kaldıraç hesapla
        coin_avg_leverage = {}
        for coin, leverages in coin_leverage.items():
            avg_leverage = sum(leverages) / len(leverages) if leverages else 1
            coin_avg_leverage[coin] = round(avg_leverage, 2)
        
        correlations["coin_leverage"] = coin_avg_leverage
        
        risk_patterns["risk_correlations"] = correlations
        
        # 3. Risk etiketleme
        risk_tags = {}
        
        # En riskli coinler
        high_risk_coins = []
        for coin in self.monitored_coins:
            risk_score = 0
            
            # Leveraja göre risk
            if coin in coin_avg_leverage:
                leverage_factor = coin_avg_leverage[coin] / self.risk_thresholds["max_leverage"] * 100
                risk_score += leverage_factor
            
            # Başarı oranına göre risk (düşük başarı = yüksek risk)
            coin_win_rate = 0
            coin_trades = [t for t in all_trades if t.get("symbol", "") == coin and t.get("status", "") == "CLOSED" and "profit_loss" in t]
            if coin_trades:
                wins = len([t for t in coin_trades if t.get("profit_loss", 0) > 0])
                coin_win_rate = (wins / len(coin_trades) * 100)
                
                # Başarı oranı %50'nin altında ise risk puanı ekle
                if coin_win_rate < 50:
                    win_rate_factor = (50 - coin_win_rate) * 2  # %0 başarı = 100 puan, %50 başarı = 0 puan
                    risk_score += win_rate_factor
            
            # Maruziyete göre risk
            if coin in self.coin_risk_metrics and "exposure_percentage" in self.coin_risk_metrics[coin]:
                exposure = self.coin_risk_metrics[coin]["exposure_percentage"]
                exposure_factor = exposure / self.risk_thresholds["max_position_exposure"] * 100
                risk_score += exposure_factor
            
            # Toplam risk skoru normalize et
            risk_score = min(100, risk_score / 3)  # 3 faktör için ortalama
            
            if risk_score >= 50:
                high_risk_coins.append((coin, risk_score))
        
        # En riskli coinleri kaydet
        risk_tags["high_risk_coins"] = [{"coin": coin, "risk_score": round(score, 2)} for coin, score in sorted(high_risk_coins, key=lambda x: x[1], reverse=True)]
        
        # En riskli stratejiler
        high_risk_strategies = []
        for strategy in self.monitored_strategies:
            if strategy in strategy_win_rates:
                win_rate = strategy_win_rates[strategy]["win_rate"]
                
                # Başarı oranı %50'nin altında ise riskli
                if win_rate < 50:
                    risk_score = (50 - win_rate) * 2  # %0 başarı = 100 puan, %50 başarı = 0 puan
                    high_risk_strategies.append((strategy, risk_score))
        
        # En riskli stratejileri kaydet
        risk_tags["high_risk_strategies"] = [{"strategy": strategy, "risk_score": round(score, 2)} for strategy, score in sorted(high_risk_strategies, key=lambda x: x[1], reverse=True)]
        
        # Risk örüntüleri
        risk_tags["risk_patterns"] = [pattern["name"] for pattern in patterns]
        
        risk_patterns["tagged_risks"] = risk_tags
        
        return risk_patterns
    
    def _analyze_consecutive_losses(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Ardışık kayıpları analiz et
        
        Args:
            trades (List[Dict[str, Any]]): İşlem listesi
            
        Returns:
            Dict[str, Any]: Ardışık kayıp analizi
        """
        result = {
            "max_consecutive_losses": 0,
            "current_consecutive_losses": 0
        }
        
        if not trades:
            return result
        
        # Tarihe göre sırala
        try:
            sorted_trades = sorted(trades, key=lambda x: datetime.datetime.strptime(x.get("timestamp", "2000-01-01 00:00:00"), "%Y-%m-%d %H:%M:%S"))
        except:
            sorted_trades = trades
        
        # Ardışık kayıpları tespit et
        current_streak = 0
        max_streak = 0
        
        for trade in sorted_trades:
            if trade.get("profit_loss", 0) <= 0:  # Zararlı işlem
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:  # Kârlı işlem
                current_streak = 0
        
        result["max_consecutive_losses"] = max_streak
        result["current_consecutive_losses"] = current_streak
        
        return result
    
    def _analyze_drawdown(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Drawdown analizi yap
        
        Args:
            trades (List[Dict[str, Any]]): İşlem listesi
            
        Returns:
            Dict[str, Any]: Drawdown analizi
        """
        result = {
            "max_drawdown": 0,
            "max_drawdown_percentage": 0,
            "drawdown_start": None,
            "drawdown_end": None,
            "current_drawdown": 0
        }
        
        if not trades:
            return result
        
        # Tarihe göre sırala
        try:
            sorted_trades = sorted(trades, key=lambda x: datetime.datetime.strptime(x.get("timestamp", "2000-01-01 00:00:00"), "%Y-%m-%d %H:%M:%S"))
        except:
            sorted_trades = trades
        
        # Equity curve oluştur
        equity_curve = []
        initial_balance = 10000  # Varsayılan başlangıç bakiyesi
        balance = initial_balance
        
        for trade in sorted_trades:
            profit_loss = trade.get("profit_loss", 0)
            balance += profit_loss
            equity_curve.append((trade.get("timestamp", ""), balance))
        
        # Drawdown hesapla
        peak = initial_balance
        max_drawdown = 0
        max_drawdown_start = None
        max_drawdown_end = None
        current_drawdown_start = None
        
        for timestamp, balance in equity_curve:
            if balance > peak:
                peak = balance
                current_drawdown_start = None
            else:
                drawdown = peak - balance
                drawdown_pct = (drawdown / peak * 100) if peak > 0 else 0
                
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                    max_drawdown_start = current_drawdown_start or timestamp
                    max_drawdown_end = timestamp
                
                if current_drawdown_start is None:
                    current_drawdown_start = timestamp
        
        # Şu anki drawdown
        current_drawdown = peak - balance if balance < peak else 0
        current_drawdown_pct = (current_drawdown / peak * 100) if peak > 0 else 0
        
        result["max_drawdown"] = round(max_drawdown, 2)
        result["max_drawdown_percentage"] = round((max_drawdown / initial_balance * 100) if initial_balance > 0 else 0, 2)
        result["drawdown_start"] = max_drawdown_start
        result["drawdown_end"] = max_drawdown_end
        result["current_drawdown"] = round(current_drawdown, 2)
        result["current_drawdown_percentage"] = round(current_drawdown_pct, 2)
        
        return result
    
    def send_risk_alerts(self) -> None:
        """
        Tespit edilen risklere göre uyarılar gönder
        """
        if not ALERT_SYSTEM_AVAILABLE:
            logger.warning("Alert System modülü bulunamadı, uyarılar sadece konsola yazılacak.")
            return
        
        if not self.critical_risks:
            return  # Kritik risk yoksa uyarı gönderme
        
        # Maksimum uyarı sayısını kontrol et
        alerts_to_send = min(len(self.critical_risks), self.max_alerts_per_run)
        
        for i in range(alerts_to_send):
            risk_message = self.critical_risks[i]
            
            try:
                # Risk mesajını gönder
                critical(
                    f"SentientTrader.AI Risk Uyarısı: {risk_message}",
                    {
                        "source": "risk_manager.py",
                        "risk_level": "CRITICAL",
                        "timestamp": CURRENT_TIME,
                        "user": CURRENT_USER
                    }
                )
                
                self.alert_count += 1
                logger.info(f"Kritik risk uyarısı gönderildi: {risk_message}")
            except Exception as e:
                logger.error(f"Risk uyarısı gönderilirken hata: {e}")
    
    def create_risk_report(self) -> Dict[str, Any]:
        """
        Risk analiz raporunu oluştur
        
        Returns:
            Dict[str, Any]: Risk raporu
        """
        logger.info("Risk raporu oluşturuluyor...")
        
        report = {
            "timestamp": CURRENT_TIME,
            "user": CURRENT_USER,
            "coin_risk_metrics": self.coin_risk_metrics,
            "portfolio_risk_metrics": self.portfolio_risk_metrics,
            "trade_frequency": self.analyze_trade_frequency(),
            "profit_loss_analysis": self.analyze_profit_loss_metrics(),
            "risk_patterns": self.identify_risk_patterns(),
            "risk_warnings": self.risk_warnings,
            "critical_risks": self.critical_risks,
            "simulation_mode": self.simulation_mode
        }
        
        return report
    
    def create_risk_tags(self) -> Dict[str, Any]:
        """
        Öğrenme motoru için risk etiketleri oluştur
        
        Returns:
            Dict[str, Any]: Risk etiketleri
        """
        logger.info("Risk etiketleri oluşturuluyor...")
        
        tags = {
            "timestamp": CURRENT_TIME,
            "user": CURRENT_USER,
            "high_risk_coins": [],
            "high_risk_strategies": [],
            "risk_patterns": [],
            "portfolio_risk_level": ""
        }
        
        # Risk etiketlerini ekle
        if "risk_patterns" in self.portfolio_risk_metrics and "tagged_risks" in self.portfolio_risk_metrics["risk_patterns"]:
            risk_tags = self.portfolio_risk_metrics["risk_patterns"]["tagged_risks"]
            
            if "high_risk_coins" in risk_tags:
                tags["high_risk_coins"] = risk_tags["high_risk_coins"]
            
            if "high_risk_strategies" in risk_tags:
                tags["high_risk_strategies"] = risk_tags["high_risk_strategies"]
            
            if "risk_patterns" in risk_tags:
                tags["risk_patterns"] = risk_tags["risk_patterns"]
        
        # Portföy risk seviyesi
        if "overall_risk_score" in self.portfolio_risk_metrics and "risk_level" in self.portfolio_risk_metrics:
            tags["portfolio_risk_level"] = self.portfolio_risk_metrics["risk_level"]
            tags["portfolio_risk_score"] = self.portfolio_risk_metrics["overall_risk_score"]
        
        return tags
    
    def save_risk_report(self) -> bool:
        """
        Risk raporunu dosyaya kaydet
        
        Returns:
            bool: Başarılı mı?
        """
        try:
            with open(RISK_REPORT_FILE, "w", encoding="utf-8") as f:
                json.dump(self.create_risk_report(), f, ensure_ascii=False, indent=4)
            
            logger.info(f"Risk raporu {RISK_REPORT_FILE} dosyasına kaydedildi")
            return True
            
        except Exception as e:
            logger.error(f"Risk raporu kaydedilirken hata: {e}")
            return False
    
    def save_risk_tags(self) -> bool:
        """
        Risk etiketlerini dosyaya kaydet
        
        Returns:
            bool: Başarılı mı?
        """
        try:
            with open(RISK_TAGS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.create_risk_tags(), f, ensure_ascii=False, indent=4)
            
            logger.info(f"Risk etiketleri {RISK_TAGS_FILE} dosyasına kaydedildi")
            return True
            
        except Exception as e:
            logger.error(f"Risk etiketleri kaydedilirken hata: {e}")
            return False
    
    def analyze_all(self) -> bool:
        """
        Tüm risk analizlerini çalıştır
        
        Returns:
            bool: Başarılı mı?
        """
        try:
            # Coin maruziyetini hesapla
            self.coin_risk_metrics = self.calculate_coin_exposure()
            
            # Portföy risk metrikleri
            self.portfolio_risk_metrics = self.analyze_portfolio_risk()
            
            return True
            
        except Exception as e:
            logger.error(f"Risk analizi sırasında hata: {e}")
            return False
    
    def run(self) -> bool:
        """
        Tüm risk yönetimi sürecini çalıştır
        
        Returns:
            bool: Başarılı mı?
        """
        try:
            print(f"{COLORS['cyan']}🔍 SentientTrader.AI - Risk Yöneticisi başlatılıyor...{COLORS['reset']}")
            logger.info("Risk yöneticisi başlatılıyor...")
            
            # İşlem verilerini yükle
            data_loaded = self.load_trades_data()
            
            if not data_loaded:
                print(f"{COLORS['red']}❌ İşlem verileri yüklenemedi! Risk analizi yapılamıyor.{COLORS['reset']}")
                return False
            
            # Tüm risk analizlerini çalıştır
            print(f"{COLORS['cyan']}⚠️ Risk analizi başlatılıyor...{COLORS['reset']}")
            analysis_success = self.analyze_all()
            
            if not analysis_success:
                print(f"{COLORS['yellow']}⚠️ Analiz sürecinde bazı hatalar oluştu.{COLORS['reset']}")
            
            # Kritik riskler varsa alert gönder
            if self.critical_risks:
                print(f"{COLORS['red']}🚨 {len(self.critical_risks)} kritik risk tespit edildi!{COLORS['reset']}")
                self.send_risk_alerts()
            
            # Sonuçları kaydet
            print(f"{COLORS['cyan']}💾 Risk analiz sonuçları kaydediliyor...{COLORS['reset']}")
            
            report_saved = self.save_risk_report()
            tags_saved = self.save_risk_tags()
            
            if report_saved and tags_saved:
                print(f"{COLORS['green']}✅ Risk analiz sonuçları başarıyla kaydedildi.{COLORS['reset']}")
            else:
                print(f"{COLORS['yellow']}⚠️ Risk analiz sonuçları kaydedilirken bazı hatalar oluştu.{COLORS['reset']}")
            
            # Özet göster
            self.display_summary()
            
            return True
            
        except Exception as e:
            logger.error(f"Risk yöneticisi çalıştırılırken hata: {e}")
            print(f"{COLORS['red']}❌ Hata: {e}{COLORS['reset']}")
            return False
    
    def display_summary(self) -> None:
        """
        Risk analizinin özetini terminalde göster
        """
        try:
            # Banner
            print(f"\n{COLORS['bright']}{COLORS['cyan']}{'=' * 60}{COLORS['reset']}")
            print(f"{COLORS['bright']}{COLORS['cyan']}⚠️ SENTİENTTRADER.AI - RİSK ANALİZİ SONUÇLARI{COLORS['reset']}")
            print(f"{COLORS['bright']}{COLORS['cyan']}{'=' * 60}{COLORS['reset']}")
            
            print(f"\n{COLORS['yellow']}📅 Tarih: {CURRENT_TIME} | 👤 Kullanıcı: {CURRENT_USER}{COLORS['reset']}")
            
            # Genel risk seviyesi
            print(f"\n{COLORS['bright']}🔍 GENEL RİSK DEĞERLENDİRMESİ:{COLORS['reset']}")
            
            if "risk_level" in self.portfolio_risk_metrics and "overall_risk_score" in self.portfolio_risk_metrics:
                risk_level = self.portfolio_risk_metrics["risk_level"]
                risk_score = self.portfolio_risk_metrics["overall_risk_score"]
                
                # Renk seçimi
                risk_color = COLORS["green"]
                if risk_level == "MEDIUM":
                    risk_color = COLORS["yellow"]
                elif risk_level == "HIGH":
                    risk_color = COLORS["red"]
                elif risk_level == "CRITICAL":
                    risk_color = COLORS["red"] + COLORS["bright"]
                
                print(f"  Portföy Risk Seviyesi: {risk_color}{risk_level}{COLORS['reset']} (Skor: {risk_score}/100)")
                
                # Konsantrasyon riski
                if "concentration_risk" in self.portfolio_risk_metrics:
                    max_coin = self.portfolio_risk_metrics["concentration_risk"].get("max_concentration_coin")
                    max_pct = self.portfolio_risk_metrics["concentration_risk"].get("max_concentration")
                    
                                        conc_color = COLORS["green"]
                    if max_pct > self.risk_thresholds["portfolio_concentration_limit"]:
                        conc_color = COLORS["red"]
                    elif max_pct > self.risk_thresholds["portfolio_concentration_limit"] / 2:
                        conc_color = COLORS["yellow"]
                    
                    print(f"  Konsantrasyon Riski: {conc_color}%{max_pct:.2f} ({max_coin}){COLORS['reset']}")
                
                # Kaldıraç riski
                if "leverage_risk" in self.portfolio_risk_metrics:
                    avg_leverage = self.portfolio_risk_metrics["leverage_risk"].get("avg_leverage", 1)
                    
                    lev_color = COLORS["green"]
                    if avg_leverage > self.risk_thresholds["high_risk_leverage"]:
                        lev_color = COLORS["red"]
                    elif avg_leverage > 1:
                        lev_color = COLORS["yellow"]
                    
                    print(f"  Ortalama Kaldıraç: {lev_color}{avg_leverage:.2f}x{COLORS['reset']}")
            else:
                print(f"  {COLORS['yellow']}Portföy risk analizi verisi mevcut değil{COLORS['reset']}")
            
            # Coin bazlı maruziyet
            print(f"\n{COLORS['bright']}💰 COİN MARUZİYETİ:{COLORS['reset']}")
            
            if self.coin_risk_metrics:
                # Maruziyeti yüzdeye göre sırala
                sorted_coins = sorted(
                    self.coin_risk_metrics.items(),
                    key=lambda x: x[1].get("exposure_percentage", 0),
                    reverse=True
                )
                
                for coin, metrics in sorted_coins[:5]:  # En yüksek 5 coin
                    exposure_pct = metrics.get("exposure_percentage", 0)
                    risk_level = metrics.get("risk_level", "LOW")
                    
                    # Renk seçimi
                    pct_color = COLORS["green"]
                    if risk_level == "HIGH":
                        pct_color = COLORS["red"]
                    elif risk_level == "MEDIUM":
                        pct_color = COLORS["yellow"]
                    
                    print(f"  {COLORS['bright']}{coin}{COLORS['reset']}: {pct_color}%{exposure_pct:.2f}{COLORS['reset']} ({risk_level})")
            else:
                print(f"  {COLORS['yellow']}Coin maruziyet verisi mevcut değil{COLORS['reset']}")
            
            # Kritik riskler
            if self.critical_risks:
                print(f"\n{COLORS['bright']}{COLORS['red']}🚨 KRİTİK RİSKLER:{COLORS['reset']}")
                
                for i, risk in enumerate(self.critical_risks[:3], 1):
                    print(f"  {i}. {COLORS['red']}{risk}{COLORS['reset']}")
                
                if len(self.critical_risks) > 3:
                    print(f"  ... ve {len(self.critical_risks) - 3} kritik risk daha")
            
            # Risk uyarıları
            if self.risk_warnings:
                print(f"\n{COLORS['bright']}{COLORS['yellow']}⚠️ RİSK UYARILARI:{COLORS['reset']}")
                
                for i, warning in enumerate(self.risk_warnings[:5], 1):
                    print(f"  {i}. {COLORS['yellow']}{warning}{COLORS['reset']}")
                
                if len(self.risk_warnings) > 5:
                    print(f"  ... ve {len(self.risk_warnings) - 5} uyarı daha")
            
            # Risk örüntüleri
            if "risk_patterns" in self.portfolio_risk_metrics and "trading_patterns" in self.portfolio_risk_metrics["risk_patterns"]:
                patterns = self.portfolio_risk_metrics["risk_patterns"]["trading_patterns"]
                
                if patterns:
                    print(f"\n{COLORS['bright']}🔄 TESPİT EDİLEN RİSK ÖRÜNTÜLERİ:{COLORS['reset']}")
                    
                    for i, pattern in enumerate(patterns[:3], 1):
                        print(f"  {i}. {pattern['description']}: {pattern['count']} işlem")
            
            # P&L Metrikleri
            if "profit_loss_analysis" in self.portfolio_risk_metrics and "summary" in self.portfolio_risk_metrics["profit_loss_analysis"]:
                pnl_summary = self.portfolio_risk_metrics["profit_loss_analysis"]["summary"]
                
                print(f"\n{COLORS['bright']}📊 KÂR/ZARAR METRİKLERİ:{COLORS['reset']}")
                
                if pnl_summary:
                    win_rate = pnl_summary.get("win_rate", 0)
                    profit_factor = pnl_summary.get("profit_factor", 0)
                    
                    win_color = COLORS["green"] if win_rate >= 50 else COLORS["red"]
                    profit_factor_color = COLORS["green"] if profit_factor >= 1 else COLORS["red"]
                    
                    print(f"  Başarı Oranı: {win_color}%{win_rate:.2f}{COLORS['reset']}")
                    print(f"  Profit Faktörü: {profit_factor_color}{profit_factor:.2f}{COLORS['reset']}")
            
            # Kaydedilen dosyalar
            print(f"\n{COLORS['bright']}📝 KAYDEDILEN DOSYALAR:{COLORS['reset']}")
            print(f"  📊 Risk Raporu: {COLORS['green']}{RISK_REPORT_FILE}{COLORS['reset']}")
            print(f"  🔄 Risk Etiketleri: {COLORS['green']}{RISK_TAGS_FILE}{COLORS['reset']}")
            
            print(f"\n{COLORS['bright']}{COLORS['cyan']}{'=' * 60}{COLORS['reset']}")
            print(f"{COLORS['green']}✅ Risk analizi tamamlandı!{COLORS['reset']}")
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
    parser = argparse.ArgumentParser(description="SentientTrader.AI Risk Manager")
    parser.add_argument("--simulation", action="store_true", help="Simülasyon modunda çalış")
    parser.add_argument("--capital", type=float, default=10000.0, help="Toplam sermaye (USD)")
    parser.add_argument("--max-exposure", type=float, default=RISK_THRESHOLDS["max_position_exposure"], 
                        help="Maksimum pozisyon maruziyeti (%)")
    parser.add_argument("--max-leverage", type=float, default=RISK_THRESHOLDS["max_leverage"], 
                        help="Maksimum kaldıraç oranı")
    return parser.parse_args()

def display_banner() -> None:
    """Başlık banner'ını göster"""
    banner = f"""
{COLORS["bright"]}{COLORS["cyan"]}
██████╗ ██╗███████╗██╗  ██╗    ███╗   ███╗ █████╗ ███╗   ██╗ █████╗  ██████╗ ███████╗██████╗ 
██╔══██╗██║██╔════╝██║ ██╔╝    ████╗ ████║██╔══██╗████╗  ██║██╔══██╗██╔════╝ ██╔════╝██╔══██╗
██████╔╝██║███████╗█████╔╝     ██╔████╔██║███████║██╔██╗ ██║███████║██║  ███╗█████╗  ██████╔╝
██╔══██╗██║╚════██║██╔═██╗     ██║╚██╔╝██║██╔══██║██║╚██╗██║██╔══██║██║   ██║██╔══╝  ██╔══██╗
██║  ██║██║███████║██║  ██╗    ██║ ╚═╝ ██║██║  ██║██║ ╚████║██║  ██║╚██████╔╝███████╗██║  ██║
╚═╝  ╚═╝╚═╝╚══════╝╚═╝  ╚═╝    ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝
{COLORS["reset"]}{COLORS["magenta"]}                                                                     V2.0{COLORS["reset"]}

{COLORS["yellow"]}📅 {CURRENT_TIME} UTC | 👤 {CURRENT_USER}{COLORS["reset"]}
{COLORS["green"]}{'=' * 78}{COLORS["reset"]}
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
        
        # Risk yöneticisi oluştur
        risk_manager = RiskManager(simulation_mode=args.simulation)
        
        # Özel konfigürasyonları ayarla
        risk_manager.current_capital = args.capital
        risk_manager.risk_thresholds["max_position_exposure"] = args.max_exposure
        risk_manager.risk_thresholds["max_leverage"] = args.max_leverage
        
        # Çalışma modunu göster
        mode = "SİMÜLASYON" if args.simulation else "GERÇEK İŞLEM"
        logger.info(f"Risk Manager {mode} modunda başlatılıyor...")
        print(f"{COLORS['cyan']}ℹ️ Risk Yöneticisi {mode} modunda başlatılıyor... (Sermaye: ${args.capital:,.2f}){COLORS['reset']}")
        
        # Risk yöneticisini çalıştır
        success = risk_manager.run()
        
        if success:
            logger.info("Risk yöneticisi başarıyla çalıştı.")
            return 0
        else:
            logger.error("Risk yöneticisi çalıştırılamadı veya analiz tamamlanamadı!")
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
