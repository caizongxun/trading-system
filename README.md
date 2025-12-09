# Automated Trading Signal System

一個完整的機器學習交易信號系統，支持加密貨幣和美股。

## 快速開始

### 1. 環境設定
python -m venv venv
source venv/bin/activate # macOS/Linux
pip install -r requirements.txt

text

### 2. 配置
編輯 `file.env`，填入你的 Discord Token 和 Hugging Face Token

### 3. 運行
訓練模型
python backend/auto_trainer.py

啟動 Bot
python frontend/bot.py

text

## 結構說明

- `backend/` - 資料收集和模型訓練
- `frontend/` - Discord Bot 和推理
- `data/` - 歷史資料（CSV）
- `models/` - 訓練後的模型
- `logs/` - 日誌檔案