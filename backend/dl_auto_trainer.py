"""
Deep Learning Auto Trainer - PyTorch Version
==============================================
使用 LSTM/GRU/TCN 訓練深度學習模型
支持 GPU 和 CPU 自動降級
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List, Optional
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from dotenv import load_dotenv

# 加載環境變數
load_dotenv('file.env')

# 抑制警告
warnings.filterwarnings('ignore')

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ===== 設備檢測和配置 =====
def get_device() -> str:
    """
    智能選擇計算設備
    優先級：NVIDIA GPU > Apple Metal > CPU
    """
    if torch.cuda.is_available():
        device = 'cuda'
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"✅ NVIDIA GPU detected: {gpu_name}")
            logger.info(f"   GPU Memory: {gpu_memory:.2f} GB")
        except:
            logger.info("✅ CUDA GPU available")
        return device
    
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        logger.info("✅ Apple Metal Performance Shaders available")
        return device
    
    else:
        device = 'cpu'
        logger.warning("⚠️  No GPU detected, using CPU (slower, but will work)")
        logger.warning("    Training will be slower. Consider using GPU for faster results.")
        return device


# 初始化設備
DEVICE = get_device()
logger.info(f"Using device: {DEVICE.upper()}")
logger.info("")


class TimeSeriesDataset(Dataset):
    """時序資料集類"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: (n_samples, lookback_window, n_features)
            y: (n_samples, forecast_horizon)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    """LSTM 模型"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 5,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size)
        
        Returns:
            (batch_size, output_size)
        """
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        output = self.fc(last_hidden)
        return output


class GRUModel(nn.Module):
    """GRU 模型（比 LSTM 快）"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 5,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_hidden = gru_out[:, -1, :]
        output = self.fc(last_hidden)
        return output


class TCNModel(nn.Module):
    """TCN 模型（時間卷積網絡，最快）"""
    
    def __init__(
        self,
        input_size: int,
        output_size: int = 5,
        num_channels: List[int] = None,
        kernel_size: int = 5,
        dropout: float = 0.2
    ):
        super().__init__()
        
        if num_channels is None:
            num_channels = [64, 128, 64]
        
        layers = []
        
        # 第一層
        layers.append(nn.Conv1d(
            input_size,
            num_channels[0],
            kernel_size=kernel_size,
            padding=kernel_size // 2
        ))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # 中間層
        for i in range(len(num_channels) - 1):
            layers.append(nn.Conv1d(
                num_channels[i],
                num_channels[i + 1],
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.conv_layers = nn.Sequential(*layers)
        
        # 全連接層
        self.fc = nn.Sequential(
            nn.Linear(num_channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size)
        
        Returns:
            (batch_size, output_size)
        """
        # TCN 需要 (batch, channels, seq_len)
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        x = x[:, :, -1]
        output = self.fc(x)
        return output


class DataPreprocessor:
    """資料預處理"""
    
    def __init__(self, lookback_window: int = 50, forecast_horizon: int = 5):
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
    
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
        """計算技術指標特徵"""
        if df.empty or len(df) < self.lookback_window:
            return pd.DataFrame()
        
        df = df.copy()
        
        # ===== 基本特徵 =====
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # ===== 動量 =====
        df['momentum_5'] = df['close'].diff(5)
        df['momentum_10'] = df['close'].diff(10)
        
        # ===== RSI =====
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # ===== MACD =====
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # ===== Bollinger Bands =====
        sma_20 = df['close'].rolling(window=20).mean()
        std_20 = df['close'].rolling(window=20).std()
        df['bb_upper'] = sma_20 + (std_20 * 2)
        df['bb_lower'] = sma_20 - (std_20 * 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        
        # ===== EMA =====
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # ===== SMA =====
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # ===== ATR =====
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift()).abs()
        tr3 = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(window=14).mean()
        
        # ===== 波動率 =====
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # ===== 成交量 =====
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        df = df.dropna()
        
        return df
    
    def prepare_dl_data(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """為深度學習準備資料"""
        feature_cols = [
            'returns', 'log_returns', 'momentum_5', 'momentum_10',
            'rsi_14', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_lower', 'bb_width',
            'ema_12', 'ema_26', 'ema_50', 'sma_20', 'sma_50',
            'atr_14', 'volatility', 'volume_ratio'
        ]
        
        missing = [col for col in feature_cols if col not in df.columns]
        if missing:
            feature_cols = [col for col in feature_cols if col in df.columns]
        
        X_data = df[feature_cols].values
        y_data = df['close'].values
        
        X_sequences = []
        y_sequences = []
        
        for i in range(len(df) - self.lookback_window - self.forecast_horizon):
            X_sequences.append(X_data[i:i + self.lookback_window])
            y_sequences.append(y_data[i + self.lookback_window:i + self.lookback_window + self.forecast_horizon])
        
        X = np.array(X_sequences)
        y = np.array(y_sequences)
        
        return X, y, feature_cols


class DLTrainer:
    """深度學習訓練器 - 支持 GPU 和 CPU 自動降級"""
    
    def __init__(self, model_type: str = 'lstm'):
        """
        Args:
            model_type: 'lstm', 'gru', 或 'tcn'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.device = DEVICE
        
        # 根據設備調整訓練參數
        self.adjust_for_device()
    
    def adjust_for_device(self):
        """根據設備類型調整訓練參數"""
        if self.device == 'cuda':
            self.default_batch_size = 64
            self.default_epochs = 30
        else:
            self.default_batch_size = 16
            self.default_epochs = 20
    
    def build_model(
        self,
        input_size: int,
        output_size: int = 5
    ) -> nn.Module:
        """建立模型"""
        try:
            if self.model_type == 'lstm':
                self.model = LSTMModel(
                    input_size=input_size,
                    hidden_size=128,
                    num_layers=2,
                    output_size=output_size,
                    dropout=0.2
                )
            elif self.model_type == 'gru':
                self.model = GRUModel(
                    input_size=input_size,
                    hidden_size=128,
                    num_layers=2,
                    output_size=output_size,
                    dropout=0.2
                )
            elif self.model_type == 'tcn':
                self.model = TCNModel(
                    input_size=input_size,
                    output_size=output_size
                )
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            self.model = self.model.to(self.device)
            return self.model
        
        except Exception as e:
            logger.error(f"❌ Error building model: {e}")
            raise
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        validation_split: float = 0.2,
        learning_rate: float = 0.001
    ) -> Dict[str, float]:
        """
        訓練模型，自動適應 GPU/CPU
        
        Args:
            X: (n_samples, lookback, n_features)
            y: (n_samples, forecast_horizon)
            epochs: 訓練 epoch 數（None 則自動選擇）
            batch_size: 批次大小（None 則自動選擇）
            validation_split: 驗證集比例
            learning_rate: 學習率
        
        Returns:
            訓練指標
        """
        try:
            # 使用預設值如果沒有指定
            if epochs is None:
                epochs = self.default_epochs
            if batch_size is None:
                batch_size = self.default_batch_size
            
            # 標準化特徵
            n_samples, lookback, n_features = X.shape
            X_flat = X.reshape(n_samples, -1)
            X_scaled = self.scaler.fit_transform(X_flat)
            X_scaled = X_scaled.reshape(n_samples, lookback, n_features)
            
            # 分割資料
            n_train = int(n_samples * (1 - validation_split))
            X_train, X_val = X_scaled[:n_train], X_scaled[n_train:]
            y_train, y_val = y[:n_train], y[n_train:]
            
            # 建立資料集
            train_dataset = TimeSeriesDataset(X_train, y_train)
            val_dataset = TimeSeriesDataset(X_val, y_val)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # 建立模型
            self.build_model(n_features, y.shape[1])
            
            # 優化器和損失函數
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()
            
            best_val_loss = float('inf')
            patience = 10
            patience_counter = 0
            
            train_losses = []
            val_losses = []
            
            for epoch in range(epochs):
                # 訓練
                self.model.train()
                train_loss = 0
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                train_losses.append(train_loss)
                
                # 驗證
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        
                        outputs = self.model(X_batch)
                        loss = criterion(outputs, y_batch)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                val_losses.append(val_loss)
                
                # 早停
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    break
                
                if (epoch + 1) % max(1, epochs // 5) == 0:
                    logger.info(f"      Epoch {epoch + 1}/{epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")
            
            # 最終評估
            self.model.eval()
            with torch.no_grad():
                X_val_tensor = torch.FloatTensor(X_val).to(self.device)
                y_pred = self.model(X_val_tensor).cpu().numpy()
            
            val_mape = mean_absolute_percentage_error(y_val, y_pred)
            
            metrics = {
                'train_loss': train_losses[-1],
                'val_loss': val_losses[-1],
                'val_mape': val_mape
            }
            
            return metrics
        
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() or 'cuda' in str(e).lower():
                logger.error(f"❌ GPU out of memory or CUDA error")
                logger.warning("⚠️  Falling back to CPU training...")
                torch.cuda.empty_cache()
                
                # 降級到 CPU
                if self.device == 'cuda':
                    self.device = 'cpu'
                    self.adjust_for_device()
                    
                    # 遞迴重試（用 CPU）
                    return self.train(
                        X, y,
                        epochs=self.default_epochs,
                        batch_size=self.default_batch_size,
                        validation_split=validation_split,
                        learning_rate=learning_rate
                    )
            
            logger.error(f"❌ Training error: {e}")
            raise
    
    def save(self, filepath: str):
        """保存模型"""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_type': self.model_type,
                'scaler': self.scaler,
                'device': self.device
            }, filepath)
        except Exception as e:
            logger.error(f"❌ Error saving model: {e}")


class AutoTrainer:
    """自動訓練管道"""
    
    def __init__(self, model_type: str = 'lstm'):
        self.model_type = model_type
        self.data_dir_crypto = Path(__file__).parent / 'data' / 'crypto'
        self.data_dir_stock = Path(__file__).parent / 'data' / 'stock'
        self.models_dir = Path(__file__).parent.parent / 'models'
        self.models_dir.mkdir(exist_ok=True)
        
        logger.info(f"Crypto data dir: {self.data_dir_crypto}")
        logger.info(f"Stock data dir:  {self.data_dir_stock}")
        logger.info(f"Models dir:      {self.models_dir}")
        logger.info("")
        
        self.preprocessor = DataPreprocessor(lookback_window=50, forecast_horizon=5)
    
    def find_data_files(self) -> Dict[str, List[str]]:
        """找到所有資料檔案"""
        crypto_files = list(self.data_dir_crypto.glob('*.csv'))
        stock_files = list(self.data_dir_stock.glob('*.csv'))
        
        return {
            'crypto': sorted([str(f) for f in crypto_files]),
            'stock': sorted([str(f) for f in stock_files])
        }
    
    def train_single_model(self, filepath: str) -> Tuple[bool, Dict]:
        """訓練單個模型"""
        try:
            filename = Path(filepath).stem
            logger.info(f"  Training {filename}...")
            
            # 加載和處理資料
            df = self.preprocessor.load_csv(filepath)
            if df.empty:
                return False, {}
            
            df = self.preprocessor.calculate_features(df)
            if df.empty or len(df) < 100:
                return False, {}
            
            X, y, _ = self.preprocessor.prepare_dl_data(df)
            if X.shape[0] < 10:
                return False, {}
            
            # 訓練
            trainer = DLTrainer(self.model_type)
            metrics = trainer.train(X, y)
            
            # 保存
            model_path = self.models_dir / f"dl_{self.model_type}_{filename}.pt"
            trainer.save(str(model_path))
            
            logger.info(f"    ✓ Val Loss: {metrics['val_loss']:.6f}, MAPE: {metrics['val_mape']:.4f}")
            
            return True, metrics
        
        except Exception as e:
            logger.error(f"Error training {filepath}: {e}")
            return False, {}
    
    def run(self):
        """執行訓練"""
        logger.info("=" * 70)
        logger.info(f"🚀 Deep Learning Auto Trainer - {self.model_type.upper()}")
        logger.info("=" * 70)
        logger.info("")
        
        files = self.find_data_files()
        all_files = files['crypto'] + files['stock']
        
        logger.info(f"Found {len(all_files)} data files")
        logger.info(f"  Crypto: {len(files['crypto'])}")
        logger.info(f"  Stock:  {len(files['stock'])}")
        logger.info("")
        
        success_count = 0
        
        logger.info(f"[1/2] Training Crypto Models...")
        for filepath in files['crypto']:
            success, _ = self.train_single_model(filepath)
            if success:
                success_count += 1
        
        logger.info(f"\n[2/2] Training Stock Models...")
        for filepath in files['stock']:
            success, _ = self.train_single_model(filepath)
            if success:
                success_count += 1
        
        # 總結
        logger.info("\n" + "=" * 70)
        logger.info(f"✅ Training Complete!")
        logger.info(f"   Models trained: {success_count}/{len(all_files)}")
        logger.info(f"   Models saved in: {self.models_dir}")
        logger.info("=" * 70)


def main():
    """主程式"""
    logger.info("")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info("")
    
    # 選擇最快的模型
    if DEVICE == 'cuda':
        model_type = 'lstm'
        logger.info("→ Using LSTM (best accuracy on GPU)")
    else:
        model_type = 'tcn'
        logger.info("→ Using TCN (fastest on CPU)")
    
    logger.info("")
    
    trainer = AutoTrainer(model_type=model_type)
    trainer.run()


if __name__ == '__main__':
    main()
