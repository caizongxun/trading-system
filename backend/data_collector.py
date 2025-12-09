"""
Data Collector for Trading System
==================================
支持多個資料源：
- Binance API（加密貨幣，無需 API Key）
- yfinance（美股）
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
import time

import pandas as pd
import yfinance as yf
import requests
from dotenv import load_dotenv

# 加載環境變數
load_dotenv('file.env')

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BinanceDataCollector:
    """
    從幣安公開 API 收集加密貨幣資料
    不需要 API Key，完全免費
    """

    def __init__(self):
        """初始化幣安公開 API"""
        self.base_url = 'https://api.binance.com/api/v3/klines'
        self.data_dir = Path('data/crypto')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Crypto data directory: {self.data_dir.absolute()}")

        # 時間框架轉換為幣安格式
        self.timeframe_map = {
            '15m': '15m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d',
        }

    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        days: int = 90
    ) -> pd.DataFrame:
        """
        從幣安公開 API 獲取 K 線資料
        """
        try:
            logger.info(f"  ↓ Fetching {symbol} {interval}...")

            # 計算開始時間
            start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

            # 幣安公開 API 參數
            params = {
                'symbol': symbol,
                'interval': self.timeframe_map[interval],
                'startTime': start_time,
                'limit': 1000
            }

            all_klines = []

            # 分批獲取
            while True:
                response = requests.get(self.base_url, params=params, timeout=10)
                response.raise_for_status()

                klines = response.json()

                if not klines:
                    break

                all_klines.extend(klines)

                # 更新開始時間
                params['startTime'] = klines[-1][0] + 1

                # 避免被限流
                time.sleep(0.05)

                if len(klines) < 1000:
                    break

            if not all_klines:
                logger.warning(f"    ✗ No data for {symbol}")
                return pd.DataFrame()

            # 轉換為 DataFrame
            df = pd.DataFrame(all_klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])

            # 清理資料
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

            # 轉換為浮點數
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df = df.dropna()

            logger.info(f"    ✓ Got {len(df)} candles")
            return df

        except Exception as e:
            logger.error(f"    ✗ Error: {e}")
            return pd.DataFrame()

    def save_csv(self, df: pd.DataFrame, symbol: str, interval: str) -> bool:
        """保存為 CSV"""
        try:
            filepath = self.data_dir / f"{symbol}_{interval}.csv"
            df.to_csv(filepath, index=False)
            logger.info(f"    💾 Saved to {filepath.name}")
            return True
        except Exception as e:
            logger.error(f"    ✗ Save error: {e}")
            return False

    def collect_all(self, pairs: List[str], intervals: List[str]) -> int:
        """收集所有交易對和時間框架的資料，返回成功數"""
        logger.info(f"\n📊 Binance Crypto Data Collection")
        logger.info(f"  Pairs: {', '.join(pairs)}")
        logger.info(f"  Intervals: {', '.join(intervals)}")
        logger.info("-" * 60)

        success_count = 0

        for symbol in pairs:
            logger.info(f"\n  {symbol}:")
            for interval in intervals:
                df = self.fetch_klines(symbol, interval)

                if not df.empty and self.save_csv(df, symbol, interval):
                    success_count += 1

        return success_count


class YFinanceDataCollector:
    """從 yfinance 收集美股資料 - 簡化版本"""

    def __init__(self):
        """初始化"""
        self.data_dir = Path('data/stock')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Stock data directory: {self.data_dir.absolute()}")

    def fetch_stock(
            self,
            symbol: str,
            interval: str,
            days: int = 90
    ) -> pd.DataFrame:
        """
        從 yfinance 獲取股票資料
        """
        try:
            logger.info(f"  ↓ Fetching {symbol} for {days} days with {interval} interval...")

            df = yf.download(
                symbol,
                period=f"{days}d",
                interval=interval,
                auto_adjust=True
            )

            if df is None or df.empty:
                logger.warning(f"    ✗ No data for {symbol}")
                return pd.DataFrame()

            logger.info(f"    ✓ Downloaded {len(df)} rows, processing...")

            # 重置索引（這會把 Date/Datetime 變成一列）
            df = df.reset_index()

            # 用位置索引而不是列名來訪問（完全避免列名問題）
            try:
                # 通常結構是：Date/Datetime, Open, High, Low, Close, Volume, ...
                # 我們直接用第 0-5 列
                result_df = pd.DataFrame()

                # 第 0 列：時間戳
                result_df['timestamp'] = df.iloc[:, 0]

                # 第 1-5 列：OHLCV
                result_df['open'] = df.iloc[:, 1]
                result_df['high'] = df.iloc[:, 2]
                result_df['low'] = df.iloc[:, 3]
                result_df['close'] = df.iloc[:, 4]
                result_df['volume'] = df.iloc[:, 5]

                # 轉換時間戳
                result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])

                # 轉換數值
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    result_df[col] = pd.to_numeric(result_df[col], errors='coerce')

                # 移除 NaN
                result_df = result_df.dropna()

                if result_df.empty:
                    logger.warning(f"    ✗ No valid data for {symbol}")
                    return pd.DataFrame()

                logger.info(f"    ✓ Got {len(result_df)} valid candles")
                return result_df

            except Exception as e:
                logger.error(f"    ✗ Error processing columns: {e}")
                logger.error(f"    DataFrame shape: {df.shape}, columns: {list(df.columns)}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"    ✗ Error fetching {symbol}: {e}")
            return pd.DataFrame()

    def save_csv(self, df: pd.DataFrame, symbol: str, interval: str) -> bool:
        """保存為 CSV"""
        try:
            filepath = self.data_dir / f"{symbol}_{interval}.csv"
            df.to_csv(filepath, index=False)
            logger.info(f"    💾 Saved to {filepath.name}")
            return True
        except Exception as e:
            logger.error(f"    ✗ Save error: {e}")
            return False

    def collect_all(self, symbols: List[str], intervals: List[str]) -> int:
        """收集所有股票和時間框架的資料"""
        logger.info(f"\n📈 yfinance Stock Data Collection")
        logger.info(f"  Symbols: {', '.join(symbols)}")
        logger.info(f"  Intervals: {', '.join(intervals)}")
        logger.info("-" * 60)

        success_count = 0

        for symbol in symbols:
            logger.info(f"\n  Processing {symbol}:")
            for interval in intervals:
                df = self.fetch_stock(symbol, interval)

                if not df.empty and self.save_csv(df, symbol, interval):
                    success_count += 1

        return success_count


def main():
    """主程式"""
    try:
        logger.info("=" * 70)
        logger.info("🚀 Trading Data Collection System")
        logger.info("=" * 70)

        # 硬編碼配置
        crypto_pairs = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT',
                        'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT', 'DOTUSDT', 'LTCUSDT']
        crypto_timeframes = ['15m', '1h', '4h']

        stock_pairs = ['SBUX', 'KO', 'AMZN', 'AAPL', 'TSLA',
                       'NVDA', 'MSFT', 'GOOGL', 'META', 'JPM']
        stock_timeframes = ['1h', '1d']  # 改成只抓 1h 和 1d（15m 沒有歷史資料）

        # 收集加密貨幣資料
        crypto_collector = BinanceDataCollector()
        crypto_success = crypto_collector.collect_all(crypto_pairs, crypto_timeframes)

        # 收集美股資料
        stock_collector = YFinanceDataCollector()
        stock_success = stock_collector.collect_all(stock_pairs, stock_timeframes)

        # 總結
        total = len(crypto_pairs) * len(crypto_timeframes) + len(stock_pairs) * len(stock_timeframes)
        success = crypto_success + stock_success

        logger.info("\n" + "=" * 70)
        logger.info(f"✅ Data Collection Complete!")
        logger.info(f"   Success: {success}/{total}")
        logger.info(f"   Crypto: {crypto_success} files")
        logger.info(f"   Stock:  {stock_success} files")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
