from random import choice
import random
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple

class FlashOpportunityHunter:
    """
    Piyasadaki ani fırsatları tespit eden modül.
    Volatilite, haber etkisi ve teknik analizi birleştirerek yüksek
    güvenilirlikte işlem fırsatlarını belirler.
    """
    
    def __init__(self, exchange_api, news_api, sentiment_analyzer, market_data_manager):
        self.exchange = exchange_api
        self.news = news_api
        self.sentiment = sentiment_analyzer
        self.market_data = market_data_manager
        self.opportunity_threshold = 0.85
        self.volatility_threshold = 3.0  # Normal volatilitenin 3 katı
        self.min_volume_increase = 2.5   # Normal hacmin 2.5 katı
        self.significant_news_score = 0.7 # 0-1 arasında önem skoru
        self.min_risk_reward = 3.0       # Risk/ödül oranı minimum 1:3
        self.scan_interval = 300         # 5 dakikada bir tara (saniye)
        
    async def find_volatile_assets(self) -> List[Dict[str, Any]]:
        """Yüksek volatilite gösteren varlıkları bulur"""
        all_tickers = await self.exchange.get_all_tickers()
        volatile_assets = []
        
        for ticker in all_tickers:
            # Son 5 dakikalık veri
            ohlcv = await self.market_data.get_ohlcv(ticker['symbol'], '5m', limit=12)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Volatilite hesaplama (ATR benzeri)
            df['tr'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            )
            
            avg_tr = df['tr'].mean()
            current_tr = df['tr'].iloc[-1]
            
            # Hacim analizi
            avg_volume = df['volume'].mean()
            current_volume = df['volume'].iloc[-1]
            
            # Volatilite ve hacim eşiklerini kontrol et
            if (current_tr > avg_tr * self.volatility_threshold and 
                current_volume > avg_volume * self.min_volume_increase):
                
                price_change = abs(df['close'].iloc[-1] - df['open'].iloc[-1]) / df['open'].iloc[-1] * 100
                
                volatile_assets.append({
                    'symbol': ticker['symbol'],
                    'volatility_ratio': current_tr / avg_tr,
                    'volume_ratio': current_volume / avg_volume,
                    'price_change_pct': price_change,
                    'current_price': df['close'].iloc[-1],
                    'timestamp': datetime.now()
                })
        
        # Volatilite oranına göre sırala
        return sorted(volatile_assets, key=lambda x: x['volatility_ratio'], reverse=True)
    
    async def filter_by_news_impact(self, volatile_assets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Haber etkisi altında olan varlıkları filtreler"""
        news_impacted = []
        
        for asset in volatile_assets:
            # Son 2 saate ait haberleri al
            recent_news = await self.news.get_asset_news(
                asset['symbol'].replace('USDT', ''), 
                hours=2
            )
            
            if not recent_news:
                continue
                
            # Haberlerin sentiment analizini yap
            news_sentiment = await self.sentiment.analyze_news(recent_news)
            
            # Önemli haber varsa
            if (abs(news_sentiment['compound_score']) > self.significant_news_score or
                news_sentiment['importance_score'] > self.significant_news_score):
                
                asset['news_impact'] = {
                    'sentiment': news_sentiment['compound_score'],
                    'importance': news_sentiment['importance_score'],
                    'headline': recent_news[0]['title'] if recent_news else '',
                    'source': recent_news[0]['source'] if recent_news else ''
                }
                
                news_impacted.append(asset)
        
        return news_impacted
    
    async def filter_by_technical_strength(self, assets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Teknik analiz sinyalleri güçlü olanları filtreler"""
        strong_signals = []
        
        for asset in assets:
            # Farklı zaman dilimlerinde teknik analiz yap
            signals = {
                '5m': await self.analyze_technicals(asset['symbol'], '5m'),
                '15m': await self.analyze_technicals(asset['symbol'], '15m'),
                '1h': await self.analyze_technicals(asset['symbol'], '1h')
            }
            
            # Zaman dilimleri arası uyum skoru
            timeframe_consensus = self.calculate_timeframe_consensus(signals)
            
            # Güçlü sinyal varsa
            if timeframe_consensus['strength'] > 0.7:
                asset['technical_signals'] = signals
                asset['signal_strength'] = timeframe_consensus['strength']
                asset['signal_direction'] = timeframe_consensus['direction']
                strong_signals.append(asset)
        
        return strong_signals
    
    async def analyze_technicals(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Belirli bir zaman dilimi için teknik analiz yapar"""
        ohlcv = await self.market_data.get_ohlcv(symbol, timeframe, limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).fillna(0)
        loss = -delta.where(delta < 0, 0).fillna(0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        
        # Bollinger Bands
        sma20 = df['close'].rolling(window=20).mean()
        std20 = df['close'].rolling(window=20).std()
        upper_band = sma20 + (std20 * 2)
        lower_band = sma20 - (std20 * 2)
        
        # Sonuçları analiz et
        current_rsi = rsi.iloc[-1]
        current_macd = macd.iloc[-1]
        current_signal = signal.iloc[-1]
        current_close = df['close'].iloc[-1]
        
        # Sinyalleri belirle
        signals = {
            'rsi': {
                'value': current_rsi,
                'signal': 'buy' if current_rsi < 30 else 'sell' if current_rsi > 70 else 'neutral'
            },
            'macd': {
                'value': current_macd,
                'signal': 'buy' if current_macd > current_signal else 'sell'
            },
            'bollinger': {
                'value': {
                    'upper': upper_band.iloc[-1],
                    'middle': sma20.iloc[-1],
                    'lower': lower_band.iloc[-1]
                },
                'signal': 'buy' if current_close < lower_band.iloc[-1] else 
                          'sell' if current_close > upper_band.iloc[-1] else 'neutral'
            }
        }
        
        # Genel sinyal yönü
        buy_signals = sum(1 for s in signals.values() if s['signal'] == 'buy')
        sell_signals = sum(1 for s in signals.values() if s['signal'] == 'sell')
        
        return {
            'timeframe': timeframe,
            'signals': signals,
            'direction': 'buy' if buy_signals > sell_signals else 
                         'sell' if sell_signals > buy_signals else 'neutral',
            'strength': max(buy_signals, sell_signals) / len(signals)
        }
    
    def calculate_timeframe_consensus(self, signals: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Farklı zaman dilimleri arasındaki uyumu hesaplar"""
        directions = [s['direction'] for s in signals.values()]
        strengths = [s['strength'] for s in signals.values()]
        
        # Tüm zaman dilimlerinde aynı yön varsa
        if all(d == directions[0] for d in directions) and directions[0] != 'neutral':
            return {
                'direction': directions[0],
                'strength': sum(strengths) / len(strengths) * 1.5  # Uyum bonusu
            }
        
        # En güçlü yönü bul
        buy_count = directions.count('buy')
        sell_count = directions.count('sell')
        
        if buy_count > sell_count:
            return {
                'direction': 'buy',
                'strength': buy_count / len(directions) * sum(strengths) / len(strengths)
            }
        elif sell_count > buy_count:
            return {
                'direction': 'sell',
                'strength': sell_count / len(directions) * sum(strengths) / len(strengths)
            }
        else:
            return {
                'direction': 'neutral',
                'strength': 0.5
            }
    
    async def filter_by_risk_reward(self, assets: List[Dict[str, Any]], min_ratio: float = 3.0) -> List[Dict[str, Any]]:
        """Risk/ödül oranı uygun olanları filtreler"""
        good_opportunities = []
        
        for asset in assets:
            # Eğer alım sinyali varsa
            if asset['signal_direction'] == 'buy':
                support_levels = await self.identify_support_levels(asset['symbol'])
                resistance_levels = await self.identify_resistance_levels(asset['symbol'])
                
                current_price = asset['current_price']
                nearest_support = max([s for s in support_levels if s < current_price], default=current_price * 0.95)
                nearest_resistance = min([r for r in resistance_levels if r > current_price], default=current_price * 1.1)
                
                # Stop-loss ve hedef belirle
                stop_loss = nearest_support * 0.99  # Support altında biraz marj
                target = nearest_resistance
                
                # Risk/ödül oranı
                risk = (current_price - stop_loss) / current_price
                reward = (target - current_price) / current_price
                ratio = reward / risk if risk > 0 else 0
                
                if ratio >= min_ratio:
                    asset['risk_reward'] = {
                        'ratio': ratio,
                        'current_price': current_price,
                        'stop_loss': stop_loss,
                        'target': target,
                        'risk_percentage': risk * 100,
                        'reward_percentage': reward * 100
                    }
                    good_opportunities.append(asset)
            
            # Eğer satış sinyali varsa (short için)
            elif asset['signal_direction'] == 'sell':
                support_levels = await self.identify_support_levels(asset['symbol'])
                resistance_levels = await self.identify_resistance_levels(asset['symbol'])
                
                current_price = asset['current_price']
                nearest_support = max([s for s in support_levels if s < current_price], default=current_price * 0.9)
                nearest_resistance = min([r for r in resistance_levels if r > current_price], default=current_price * 1.05)
                
                # Stop-loss ve hedef belirle (short için)
                stop_loss = nearest_resistance * 1.01  # Resistance üstünde biraz marj
                target = nearest_support
                
                # Risk/ödül oranı
                risk = (stop_loss - current_price) / current_price
                reward = (current_price - target) / current_price
                ratio = reward / risk if risk > 0 else 0
                
                if ratio >= min_ratio:
                    asset['risk_reward'] = {
                        'ratio': ratio,
                        'current_price': current_price,
                        'stop_loss': stop_loss,
                        'target': target,
                        'risk_percentage': risk * 100,
                        'reward_percentage': reward * 100
                    }
                    good_opportunities.append(asset)
        
        return good_opportunities
    
    async def identify_support_levels(self, symbol: str) -> List[float]:
        """Destek seviyelerini belirler"""
        ohlcv = await self.market_data.get_ohlcv(symbol, '1h', limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Son 100 mum içinde en düşük 5 noktayı bul
        lows = sorted(df['low'].tolist())
        support_candidates = lows[:10]  # En düşük 10 nokta
        
        # Yakın olanları grupla (yüzde 0.5 tolerans)
        grouped_supports = []
        for candidate in support_candidates:
            found_group = False
            for group in grouped_supports:
                if abs(candidate - sum(group) / len(group)) / candidate < 0.005:  # %0.5 tolerans
                    group.append(candidate)
                    found_group = True
                    break
            if not found_group:
                grouped_supports.append([candidate])
        
        # Her grup için ortalama al
        return [sum(group) / len(group) for group in grouped_supports]
    
    async def identify_resistance_levels(self, symbol: str) -> List[float]:
        """Direnç seviyelerini belirler"""
        ohlcv = await self.market_data.get_ohlcv(symbol, '1h', limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Son 100 mum içinde en yüksek 5 noktayı bul
        highs = sorted(df['high'].tolist(), reverse=True)
        resistance_candidates = highs[:10]  # En yüksek 10 nokta
        
        # Yakın olanları grupla (yüzde 0.5 tolerans)
        grouped_resistances = []
        for candidate in resistance_candidates:
            found_group = False
            for group in grouped_resistances:
                if abs(candidate - sum(group) / len(group)) / candidate < 0.005:  # %0.5 tolerans
                    group.append(candidate)
                    found_group = True
                    break
            if not found_group:
                grouped_resistances.append([candidate])
        
        # Her grup için ortalama al
        return [sum(group) / len(group) for group in grouped_resistances]
    
    def rank_opportunities(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fırsatları puanlayarak sıralar"""
        ranked = []
        
        for opp in opportunities:
            # Puanlama kriterleri
            volatility_score = min(opp['volatility_ratio'] / 10, 1) * 0.3  # %30 ağırlık
            volume_score = min(opp['volume_ratio'] / 5, 1) * 0.2  # %20 ağırlık
            news_score = opp.get('news_impact', {}).get('importance', 0) * 0.15  # %15 ağırlık
            technical_score = opp.get('signal_strength', 0) * 0.15  # %15 ağırlık
            risk_reward_score = min(opp.get('risk_reward', {}).get('ratio', 0) / 5, 1) * 0.2  # %20 ağırlık
            
            # Toplam skor
            total_score = volatility_score + volume_score + news_score + technical_score + risk_reward_score
            
            opp['opportunity_score'] = total_score
            ranked.append(opp)
        
        # Skora göre sırala
        return sorted(ranked, key=lambda x: x['opportunity_score'], reverse=True)
    
    async def hunt(self) -> List[Dict[str, Any]]:
        """Ana metod: Tüm aşamaları çalıştırarak en iyi fırsatları bulur"""
        try:
            # 1. Yüksek volatiliteli varlıkları bul
            print("[Flash Hunter] Volatilite taraması başladı...")
            volatile_assets = await self.find_volatile_assets()
            print(f"[Flash Hunter] {len(volatile_assets)} volatil varlık bulundu.")
            
            if not volatile_assets:
                return []
            
            # En yüksek volatiliteli 20 varlığı al
            top_volatile = volatile_assets[:20]
            
            # 2. Haber etkisi altında olanları filtrele
            print("[Flash Hunter] Haber etkisi analiz ediliyor...")
            news_impacted = await self.filter_by_news_impact(top_volatile)
            print(f"[Flash Hunter] {len(news_impacted)} haber etkili varlık bulundu.")
            
            if not news_impacted:
                news_impacted = top_volatile[:10]  # Haber etkisi yoksa en volatil 10'u al
            
            # 3. Teknik analiz yap
            print("[Flash Hunter] Teknik analiz yapılıyor...")
            strong_signals = await self.filter_by_technical_strength(news_impacted)
            print(f"[Flash Hunter] {len(strong_signals)} güçlü teknik sinyalli varlık bulundu.")
            
            if not strong_signals:
                return []
            
            # 4. Risk/ödül oranını kontrol et
            print("[Flash Hunter] Risk/ödül analizi yapılıyor...")
            good_opportunities = await self.filter_by_risk_reward(strong_signals, self.min_risk_reward)
            print(f"[Flash Hunter] {len(good_opportunities)} iyi risk/ödül oranlı fırsat bulundu.")
            
            # 5. Fırsatları puanla ve sırala
            ranked_opportunities = self.rank_opportunities(good_opportunities)
            
            # En iyi 3 fırsatı döndür
            top_opportunities = ranked_opportunities[:3]
            
            # Sonuçları raporla
            for i, opp in enumerate(top_opportunities, 1):
                direction = opp['signal_direction'].upper()
                symbol = opp['symbol']
                score = opp['opportunity_score'] * 100
                risk_reward = opp.get('risk_reward', {}).get('ratio', 0)
                print(f"[Flash Hunter] Fırsat #{i}: {direction} {symbol} (Skor: {score:.1f}%, R/R: {risk_reward:.1f})")
            
            return top_opportunities
            
        except Exception as e:
            print(f"[Flash Hunter] Hata: {str(e)}")
            return []
    
    async def start_hunting(self, interval_seconds=300):
        """Sürekli olarak fırsatları tarar"""
        while True:
            print(f"[Flash Hunter] Fırsat taraması başlıyor... ({datetime.now()})")
            opportunities = await self.hunt()
            
            if opportunities:
                print(f"[Flash Hunter] {len(opportunities)} fırsat bulundu!")
                # Bulunan fırsatları işle veya bildir
            else:
                print("[Flash Hunter] Şu anda uygun fırsat bulunamadı.")
            
            # Belirlenen süre kadar bekle
            print(f"[Flash Hunter] {interval_seconds} saniye sonra tekrar tarama yapılacak.")
            await asyncio.sleep(interval_seconds)
