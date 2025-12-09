"""
Data Fetcher
============
定期抓取新數據並更新到 HF
每隔幾分鐘運行一次，確保數據最新
"""

import os
import logging
from pathlib import Path
from datetime import datetime, timedelta
import time

import pandas as pd
import yfinance as yf
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv

# 加載環境變數
load_dotenv('file.env')

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataFetcher:
    """數據獲取和更新"""
    
    def __init__(self):
        self.data_dir = Path('backend/data')
        self.hf_token = os.getenv('HF_TOKEN')
        self.hf_dataset_repo = os.getenv('HF_DATASET_REPO', 'your_username/trading-data')
        
        # 定義交易對
        self.crypto_pairs = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT',
            'DOGEUSDT', 'LTCUSDT', 'SOLUSDT', 'LINKUSDT', 'DOTUSDT',
            # ... 更多加密對
        ]
        
        self.stock_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
            'META', 'TSLA', 'JPM', 'KO', 'SBUX',
            # ... 更多股票
        ]
    
    def fetch_crypto_data(self, pair: str, timeframe: str = '15m') -> pd.DataFrame:
        """獲取加密貨幣數據（從 yfinance）"""
        try:
            logger.info(f"Fetching {pair} {timeframe}...")
            
            # yfinance 不支持 15m，使用替代方案
            # 實際應該用 Binance API 或其他加密交易所 API
            
            # 這裡簡化版本，實際應該調用 Binance API
            symbol = f"{pair.replace('USDT', '')}-USD"
            
            data = yf.download(
                symbol,
                start=datetime.now() - timedelta(days=30),
                progress=False
            )
            
            if data.empty:
                logger.warning(f"No data for {pair}")
                return pd.DataFrame()
            
            # 重新命名列
            data.columns = ['open', 'high', 'low', 'close', 'volume']
            data = data.reset_index()
            data.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
            return data
        
        except Exception as e:
            logger.error(f"Error fetching {pair}: {e}")
            return pd.DataFrame()
    
    def fetch_stock_data(self, symbol: str, timeframe: str = '1h') -> pd.DataFrame:
        """獲取美股數據"""
        try:
            logger.info(f"Fetching {symbol} {timeframe}...")
            
            data = yf.download(
                symbol,
                start=datetime.now() - timedelta(days=30),
                progress=False
            )
            
            if data.empty:
                logger.warning(f"No data for {symbol}")
                return pd.DataFrame()
            
            data.columns = ['open', 'high', 'low', 'close', 'volume']
            data = data.reset_index()
            data.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
            return data
        
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()
    
    def update_local_data(self):
        """更新本地 CSV 數據"""
        
        # 建立目錄
        crypto_dir = self.data_dir / 'crypto'
        stock_dir = self.data_dir / 'stock'
        crypto_dir.mkdir(parents=True, exist_ok=True)
        stock_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Updating local data...")
        
        # 更新加密貨幣數據
        for pair in self.crypto_pairs[:5]:  # 先更新前 5 個
            try:
                df = self.fetch_crypto_data(pair, '15m')
                if not df.empty:
                    csv_path = crypto_dir / f"{pair}_15m.csv"
                    df.to_csv(csv_path, index=False)
                    logger.info(f"✅ Updated {pair}")
            except Exception as e:
                logger.error(f"Error updating {pair}: {e}")
        
        # 更新股票數據
        for symbol in self.stock_symbols[:5]:  # 先更新前 5 個
            try:
                df = self.fetch_stock_data(symbol, '1h')
                if not df.empty:
                    csv_path = stock_dir / f"{symbol}_1h.csv"
                    df.to_csv(csv_path, index=False)
                    logger.info(f"✅ Updated {symbol}")
            except Exception as e:
                logger.error(f"Error updating {symbol}: {e}")
    
    def upload_to_hf(self):
        """上傳數據到 HF"""
        
        if not self.hf_token:
            logger.error("HF_TOKEN not found")
            return False
        
        api = HfApi()
        
        try:
            logger.info(f"Uploading data to {self.hf_dataset_repo}...")
            
            # 建立 repo
            create_repo(
                repo_id=self.hf_dataset_repo,
                repo_type="dataset",
                private=False,
                exist_ok=True,
                token=self.hf_token
            )
            
            # 上傳資料夾
            api.upload_folder(
                folder_path=str(self.data_dir),
                repo_id=self.hf_dataset_repo,
                repo_type="dataset",
                token=self.hf_token,
                commit_message=f"Update data - {datetime.utcnow().isoformat()}"
            )
            
            logger.info(f"✅ Data uploaded to HF")
            return True
        
        except Exception as e:
            logger.error(f"Error uploading to HF: {e}")
            return False
    
    def run_once(self):
        """執行一次更新"""
        logger.info("=" * 70)
        logger.info("🔄 Data Update Cycle")
        logger.info("=" * 70)
        
        self.update_local_data()
        self.upload_to_hf()
        
        logger.info("=" * 70)
        logger.info("✅ Update complete")
        logger.info("=" * 70)
    
    def run_continuous(self, interval_minutes: int = 5):
        """持續運行（每隔 N 分鐘更新一次）"""
        
        logger.info(f"Starting continuous data fetcher (update every {interval_minutes} minutes)")
        
        while True:
            try:
                self.run_once()
                
                # 等待指定時間
                logger.info(f"Next update in {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)
            
            except KeyboardInterrupt:
                logger.info("Data fetcher stopped by user")
                break
            
            except Exception as e:
                logger.error(f"Error in continuous loop: {e}")
                time.sleep(60)


def main():
    """主程式"""
    
    print("🚀 Data Fetcher")
    print("=" * 70)
    print("")
    print("Choose mode:")
    print("1. Run once (update data once)")
    print("2. Run continuously (update every N minutes)")
    print("")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    fetcher = DataFetcher()
    
    if choice == '1':
        fetcher.run_once()
    
    elif choice == '2':
        interval = input("Enter interval in minutes (default 5): ").strip()
        interval = int(interval) if interval.isdigit() else 5
        fetcher.run_continuous(interval)
    
    else:
        print("Invalid choice")


if __name__ == '__main__':
    main()
