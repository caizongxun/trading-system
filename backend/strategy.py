"""
Trading Strategy Module
=======================
Centralized feature engineering for both training and inference.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class TradingStrategy:
    """計算所有技術指標的類"""

    # 所有特徵列名（共19個）
    INDICATOR_COLS = [
        'rsi_14', 'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
        'ema_12', 'ema_26', 'ema_50', 'sma_20', 'sma_50',
        'atr_14', 'volatility_std_20',
        'returns', 'momentum_10',
        'adx_14', 'di_plus', 'di_minus'
    ]

    @staticmethod
    def get_feature_columns() -> List[str]:
        """返回特徵列名"""
        return TradingStrategy.INDICATOR_COLS.copy()

    @staticmethod
    def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        計算所有技術指標

        輸入：df 必須包含 open, high, low, close, volume 列
        輸出：df + 19 個指標列
        """
        df = df.copy()

        # 檢查必要的列
        required = {'open', 'high', 'low', 'close', 'volume'}
        if not required.issubset(df.columns):
            raise ValueError(f"Missing columns. Need: {required}")

        # ===== RSI (相對強度指數) =====
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # ===== MACD (移動平均收斂發散) =====
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # ===== Bollinger Bands (布林帶) =====
        sma_20 = df['close'].rolling(window=20).mean()
        std_20 = df['close'].rolling(window=20).std()
        df['bb_upper'] = sma_20 + (std_20 * 2)
        df['bb_middle'] = sma_20
        df['bb_lower'] = sma_20 - (std_20 * 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']

        # ===== 指數移動平均線 (EMA) =====
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

        # ===== 簡單移動平均線 (SMA) =====
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()

        # ===== ATR (平均真實波幅) - 波動率指標 =====
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift()).abs()
        tr3 = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(window=14).mean()

        # ===== 波動率 (標準差) =====
        df['volatility_std_20'] = df['close'].rolling(window=20).std()

        # ===== 收益率和動量 =====
        df['returns'] = df['close'].pct_change()
        df['momentum_10'] = df['close'].diff(periods=10)

        # ===== ADX (平均方向指數) - 趨勢強度 =====
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr_sum = tr.rolling(window=14).sum()
        df['di_plus'] = 100 * (plus_dm.rolling(window=14).sum() / tr_sum)
        df['di_minus'] = 100 * (minus_dm.rolling(window=14).sum() / tr_sum)

        di_diff = (df['di_plus'] - df['di_minus']).abs()
        di_sum = df['di_plus'] + df['di_minus']
        df['adx_14'] = (di_diff / di_sum).rolling(window=14).mean() * 100

        return df

    @staticmethod
    def find_peaks_troughs(prices: np.ndarray, window: int = 3) -> Tuple[List[int], List[int]]:
        """
        在預測價格中找到峰值和谷值

        輸入：價格數組（例如預測的下 5 根 K 線）
        輸出：(峰值索引列表, 谷值索引列表)
        """
        peaks = []
        troughs = []

        if len(prices) < window:
            return peaks, troughs

        for i in range(window, len(prices) - window):
            # 檢查局部峰值
            if prices[i] == max(prices[i - window:i + window + 1]):
                peaks.append(i)

            # 檢查局部谷值
            if prices[i] == min(prices[i - window:i + window + 1]):
                troughs.append(i)

        return peaks, troughs

    @staticmethod
    def calculate_signal_levels(
            current_price: float,
            predicted_path: np.ndarray,
            atr: float,
            direction: str = 'LONG'
    ) -> Tuple[float, float]:
        """
        根據 ATR 計算止損和獲利目標

        輸入：
            current_price: 當前價格
            predicted_path: 預測的未來價格陣列
            atr: 平均真實波幅
            direction: 'LONG' 或 'SHORT'

        輸出：(止損, 獲利目標)
        """
        if atr <= 0:
            atr = abs(current_price * 0.02)

        if direction.upper() == 'LONG':
            stop_loss = current_price - (atr * 1.5)
            take_profit = current_price + (atr * 2.5)
        else:  # SHORT
            stop_loss = current_price + (atr * 1.5)
            take_profit = current_price - (atr * 2.5)

        return stop_loss, take_profit

    @staticmethod
    def calculate_confidence(
            current_price: float,
            predicted_path: np.ndarray,
            direction: str = 'LONG'
    ) -> float:
        """
        計算信號的置信度分數 (0.0 到 1.0)
        """
        if len(predicted_path) == 0:
            return 0.0

        # 價格變動百分比
        final_price = predicted_path[-1]
        price_change_pct = abs(final_price - current_price) / current_price

        # 趨勢一致性
        if direction.upper() == 'LONG':
            trend_align = sum(predicted_path[i + 1] > predicted_path[i]
                              for i in range(len(predicted_path) - 1)) / len(predicted_path)
        else:
            trend_align = sum(predicted_path[i + 1] < predicted_path[i]
                              for i in range(len(predicted_path) - 1)) / len(predicted_path)

        # 結合因素
        confidence = (trend_align * 0.6) + (min(price_change_pct, 0.1) / 0.1 * 0.4)
        confidence = max(0.0, min(1.0, confidence))

        return confidence
