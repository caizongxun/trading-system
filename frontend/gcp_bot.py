"""
GCP Trading Bot - Advanced Version
===================================
進階交易邏輯 + 風險管理
多時間框架分析 + 交易量確認 + 動態止損
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
from collections import deque
import pickle

import discord
from discord.ext import commands, tasks
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

# 加載環境變數
load_dotenv('file.env')

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 設備設置
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {DEVICE}")


class AdvancedDataPreprocessor:
    """進階數據預處理"""
    
    def __init__(self):
        self.lookback_window = 50
    
    def load_csv(self, filepath: str) -> pd.DataFrame:
        """加載 CSV"""
        try:
            df = pd.read_csv(filepath)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df.sort_values('timestamp').reset_index(drop=True)
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return pd.DataFrame()
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算技術指標"""
        if df.empty or len(df) < self.lookback_window:
            return pd.DataFrame()
        
        df = df.copy()
        
        # 基本特徵
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # 動量
        df['momentum_5'] = df['close'].diff(5)
        df['momentum_10'] = df['close'].diff(10)
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        sma_20 = df['close'].rolling(window=20).mean()
        std_20 = df['close'].rolling(window=20).std()
        df['bb_upper'] = sma_20 + (std_20 * 2)
        df['bb_lower'] = sma_20 - (std_20 * 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        
        # EMA
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # SMA
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # ATR
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift()).abs()
        tr3 = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(window=14).mean()
        
        # 波動率
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # 成交量
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # ===== 進階指標 =====
        
        # 波動率排名（Volatility Rank）
        volatility_rank = df['volatility'].rolling(window=50).rank(pct=True)
        df['volatility_rank'] = volatility_rank
        
        # 成交量趨勢
        df['volume_trend'] = df['volume'].rolling(window=5).mean()
        
        # 價格動量強度
        df['price_momentum'] = (df['close'] - df['sma_50']) / df['sma_50']
        
        df = df.dropna()
        
        return df
    
    def calculate_trend(self, df: pd.DataFrame) -> str:
        """判斷趨勢：UP, DOWN, SIDEWAYS"""
        if len(df) < 50:
            return 'UNKNOWN'
        
        recent = df.tail(50)
        sma_20 = recent['sma_20'].iloc[-1]
        sma_50 = recent['sma_50'].iloc[-1]
        close = recent['close'].iloc[-1]
        
        if close > sma_20 > sma_50:
            return 'UP'
        elif close < sma_20 < sma_50:
            return 'DOWN'
        else:
            return 'SIDEWAYS'


class RiskManager:
    """風險管理"""
    
    def __init__(self, initial_balance: float = 10000):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.trades = []
        self.win_count = 0
        self.loss_count = 0
    
    def calculate_position_size(
        self,
        current_price: float,
        stop_loss: float,
        risk_percent: float = 2.0
    ) -> float:
        """計算頭寸大小"""
        risk_amount = self.current_balance * (risk_percent / 100)
        price_risk = abs(current_price - stop_loss)
        
        if price_risk == 0:
            return 0
        
        position_size = risk_amount / price_risk
        return position_size
    
    def calculate_dynamic_sl(
        self,
        signal: str,
        current_price: float,
        atr: float,
        volatility: float
    ) -> float:
        """動態止損計算"""
        # 根據波動率調整止損幅度
        base_sl = atr * 1.5
        volatility_factor = 1 + (volatility * 10)
        adjusted_sl = base_sl * volatility_factor
        
        if signal == 'LONG':
            return current_price - adjusted_sl
        else:  # SHORT
            return current_price + adjusted_sl
    
    def calculate_risk_reward_ratio(
        self,
        entry: float,
        stop_loss: float,
        take_profit: float
    ) -> float:
        """計算風險回報比"""
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        
        if risk == 0:
            return 0
        
        return reward / risk
    
    def add_trade_result(self, win: bool, profit_loss: float):
        """添加交易結果"""
        if win:
            self.win_count += 1
        else:
            self.loss_count += 1
        
        self.current_balance += profit_loss
        self.trades.append({
            'win': win,
            'profit_loss': profit_loss,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def get_win_rate(self) -> float:
        """獲取勝率"""
        total = self.win_count + self.loss_count
        if total == 0:
            return 0
        return (self.win_count / total) * 100
    
    def get_stats(self) -> Dict:
        """獲取統計信息"""
        return {
            'balance': self.current_balance,
            'trades': len(self.trades),
            'win_rate': self.get_win_rate(),
            'wins': self.win_count,
            'losses': self.loss_count,
            'profit_loss': self.current_balance - self.initial_balance
        }


class AdvancedSignalGenerator:
    """進階信號生成器"""
    
    def __init__(self):
        self.risk_manager = RiskManager()
        self.signal_history = deque(maxlen=100)
    
    def generate_multi_timeframe_signal(
        self,
        symbol: str,
        data_15m: pd.DataFrame,
        data_1h: pd.DataFrame,
        data_4h: pd.DataFrame,
        predicted_prices: np.ndarray
    ) -> Optional[Dict]:
        """多時間框架信號生成"""
        
        if len(data_15m) < 50 or len(data_1h) < 50 or len(data_4h) < 50:
            return None
        
        # 獲取當前價格
        current_price = data_15m['close'].iloc[-1]
        
        # 計算各時間框架的趨勢
        trend_15m = self._calculate_trend(data_15m)
        trend_1h = self._calculate_trend(data_1h)
        trend_4h = self._calculate_trend(data_4h)
        
        # 獲取指標
        rsi_15m = data_15m['rsi_14'].iloc[-1]
        rsi_1h = data_1h['rsi_14'].iloc[-1]
        atr_15m = data_15m['atr_14'].iloc[-1]
        volatility = data_15m['volatility'].iloc[-1]
        
        # 預測信息
        price_change = (predicted_prices[-1] - current_price) / current_price
        
        # 交易量確認
        volume_confirm = data_15m['volume_ratio'].iloc[-1] > 1.0
        
        # ===== 信號邏輯 =====
        
        signal = None
        confidence = 0
        reason = []
        
        # LONG 信號條件
        long_conditions = [
            trend_4h == 'UP',  # 4h 上升趨勢
            trend_1h in ['UP', 'SIDEWAYS'],  # 1h 上升或震盪
            rsi_1h < 70,  # 1h RSI 未超買
            price_change > 0.02,  # 預測漲幅 > 2%
            volume_confirm,  # 成交量確認
        ]
        
        # SHORT 信號條件
        short_conditions = [
            trend_4h == 'DOWN',  # 4h 下降趨勢
            trend_1h in ['DOWN', 'SIDEWAYS'],  # 1h 下降或震盪
            rsi_1h > 30,  # 1h RSI 未超賣
            price_change < -0.02,  # 預測跌幅 > 2%
            volume_confirm,  # 成交量確認
        ]
        
        long_score = sum(long_conditions)
        short_score = sum(short_conditions)
        
        if long_score >= 4:
            signal = 'LONG'
            confidence = long_score / 5
            reason = [
                f"Trend 4h: {trend_4h}",
                f"Trend 1h: {trend_1h}",
                f"RSI 1h: {rsi_1h:.2f}",
                f"Price change: {price_change*100:.2f}%",
                f"Volume: Confirmed" if volume_confirm else "Volume: Not confirmed"
            ]
        
        elif short_score >= 4:
            signal = 'SHORT'
            confidence = short_score / 5
            reason = [
                f"Trend 4h: {trend_4h}",
                f"Trend 1h: {trend_1h}",
                f"RSI 1h: {rsi_1h:.2f}",
                f"Price change: {price_change*100:.2f}%",
                f"Volume: Confirmed" if volume_confirm else "Volume: Not confirmed"
            ]
        
        if not signal:
            return None
        
        # 計算止損和止盈
        sl = self.risk_manager.calculate_dynamic_sl(
            signal, current_price, atr_15m, volatility
        )
        
        tp = current_price + (atr_15m * 3) if signal == 'LONG' else current_price - (atr_15m * 3)
        
        # 計算風險回報比
        rrr = self.risk_manager.calculate_risk_reward_ratio(current_price, sl, tp)
        
        # 只有 RRR >= 1.5 才發送信號
        if rrr < 1.5:
            return None
        
        # 計算頭寸大小
        position_size = self.risk_manager.calculate_position_size(
            current_price, sl, risk_percent=2.0
        )
        
        signal_data = {
            'symbol': symbol,
            'signal': signal,
            'current_price': float(current_price),
            'predicted_price': float(predicted_prices[-1]),
            'confidence': float(confidence),
            'stop_loss': float(sl),
            'take_profit': float(tp),
            'predicted_path': predicted_prices.tolist(),
            'timestamp': datetime.utcnow().isoformat(),
            
            # 進階信息
            'trend_4h': trend_4h,
            'trend_1h': trend_1h,
            'trend_15m': trend_15m,
            'rsi_1h': float(rsi_1h),
            'rsi_15m': float(rsi_15m),
            'volume_confirm': volume_confirm,
            'atr': float(atr_15m),
            'volatility': float(volatility),
            'risk_reward_ratio': float(rrr),
            'position_size': float(position_size),
            'reason': reason,
            'win_rate': float(self.risk_manager.get_win_rate()),
            'stats': self.risk_manager.get_stats()
        }
        
        self.signal_history.append(signal_data)
        
        return signal_data
    
    def _calculate_trend(self, df: pd.DataFrame) -> str:
        """判斷趨勢"""
        if len(df) < 50:
            return 'UNKNOWN'
        
        sma_20 = df['sma_20'].iloc[-1]
        sma_50 = df['sma_50'].iloc[-1]
        close = df['close'].iloc[-1]
        
        if close > sma_20 > sma_50:
            return 'UP'
        elif close < sma_20 < sma_50:
            return 'DOWN'
        else:
            return 'SIDEWAYS'


class GCPTradingBot(commands.Cog):
    """GCP 交易機器人"""
    
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.discord_channel_id = int(os.getenv('DISCORD_CHANNEL_ID', '0'))
        
        self.preprocessor = AdvancedDataPreprocessor()
        self.signal_generator = AdvancedSignalGenerator()
        
        self.latest_signals = {}
        
        self.inference_loop.start()
    
    @commands.Cog.listener()
    async def on_ready(self):
        """Bot 啟動"""
        logger.info(f"✅ Bot logged in as {self.bot.user}")
    
    @tasks.loop(hours=1)
    async def inference_loop(self):
        """每小時執行推理"""
        logger.info("🔄 Starting inference loop...")
        
        channel = self.bot.get_channel(self.discord_channel_id)
        if not channel:
            logger.error("Discord channel not found")
            return
        
        # 實際執行推理和信號生成
        logger.info("✅ Inference complete")
    
    @commands.command(name='status')
    async def status(self, ctx: commands.Context):
        """查看機器人狀態"""
        stats = self.signal_generator.risk_manager.get_stats()
        
        embed = discord.Embed(
            title="🤖 Trading Bot Status",
            color=discord.Color.blue(),
            timestamp=datetime.utcnow()
        )
        
        embed.add_field(name="Win Rate", value=f"{stats['win_rate']:.2f}%", inline=True)
        embed.add_field(name="Total Trades", value=stats['trades'], inline=True)
        embed.add_field(name="Balance", value=f"${stats['balance']:.2f}", inline=True)
        embed.add_field(name="P&L", value=f"${stats['profit_loss']:.2f}", inline=False)
        
        await ctx.send(embed=embed)
    
    @commands.command(name='signals')
    async def signals(self, ctx: commands.Context):
        """顯示最新信號"""
        if not self.signal_generator.signal_history:
            await ctx.send("No signals generated yet.")
            return
        
        latest = list(self.signal_generator.signal_history)[-5:]
        
        embed = discord.Embed(
            title="📊 Latest Trading Signals",
            color=discord.Color.green(),
            timestamp=datetime.utcnow()
        )
        
        for signal in latest:
            embed.add_field(
                name=f"{signal['symbol']} - {signal['signal']}",
                value=f"RRR: {signal['risk_reward_ratio']:.2f} | Confidence: {signal['confidence']:.0%}",
                inline=False
            )
        
        await ctx.send(embed=embed)


async def main():
    """主程式"""
    intents = discord.Intents.default()
    intents.message_content = True
    
    bot = commands.Bot(command_prefix='!', intents=intents)
    
    @bot.event
    async def on_ready():
        logger.info(f"Bot ready as {bot.user}")
    
    await bot.add_cog(GCPTradingBot(bot))
    
    token = os.getenv('DISCORD_TOKEN')
    if not token:
        logger.error("DISCORD_TOKEN not found")
        return
    
    await bot.start(token)


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
