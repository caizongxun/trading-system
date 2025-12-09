"""
Flask Dashboard
===============
實時顯示交易信號和預測 K 線圖表
在 GCP VM 上運行，通過 HTTP 訪問
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# 加載環境變數
load_dotenv('file.env')

# 初始化 Flask
app = Flask(__name__)
CORS(app)

# 全局信號存儲
signals_db = {
    'signals': [],
    'last_updated': datetime.utcnow().isoformat()
}


@app.route('/')
def index():
    """主頁"""
    return render_template('index.html')


@app.route('/api/signals', methods=['GET'])
def get_signals():
    """獲取信號列表"""
    return jsonify(signals_db)


@app.route('/api/signals/<symbol>', methods=['GET'])
def get_symbol_signals(symbol):
    """獲取特定交易對的信號"""
    symbol_signals = [s for s in signals_db['signals'] if s['symbol'] == symbol]
    return jsonify({
        'symbol': symbol,
        'signals': symbol_signals,
        'count': len(symbol_signals)
    })


@app.route('/api/signals', methods=['POST'])
def add_signal():
    """添加新信號（來自機器人）"""
    try:
        data = request.get_json()
        
        # 驗證數據
        required = ['symbol', 'signal', 'current_price', 'predicted_path']
        if not all(k in data for k in required):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # 添加時間戳
        data['timestamp'] = datetime.utcnow().isoformat()
        
        # 添加到數據庫
        signals_db['signals'].append(data)
        signals_db['last_updated'] = data['timestamp']
        
        # 限制歷史記錄（只保存最近 500 個）
        if len(signals_db['signals']) > 500:
            signals_db['signals'] = signals_db['signals'][-500:]
        
        return jsonify({
            'status': 'success',
            'signal': data
        }), 201
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """獲取統計信息"""
    signals = signals_db['signals']
    
    if not signals:
        return jsonify({
            'total_signals': 0,
            'long_count': 0,
            'short_count': 0,
            'symbols': []
        })
    
    # 統計
    symbols = list(set([s['symbol'] for s in signals]))
    long_count = len([s for s in signals if s.get('signal') == 'LONG'])
    short_count = len([s for s in signals if s.get('signal') == 'SHORT'])
    
    return jsonify({
        'total_signals': len(signals),
        'long_count': long_count,
        'short_count': short_count,
        'symbols': symbols,
        'last_updated': signals_db['last_updated']
    })


@app.route('/api/chart/<symbol>', methods=['GET'])
def get_chart_data(symbol):
    """獲取圖表數據"""
    symbol_signals = [s for s in signals_db['signals'] if s['symbol'] == symbol]
    
    if not symbol_signals:
        return jsonify({'error': f'No signals for {symbol}'}), 404
    
    # 取最新信號
    latest = symbol_signals[-1]
    
    # 構建圖表數據
    chart_data = {
        'symbol': symbol,
        'current_price': latest.get('current_price', 0),
        'predicted_path': latest.get('predicted_path', []),
        'signal': latest.get('signal', 'N/A'),
        'confidence': latest.get('confidence', 0),
        'stop_loss': latest.get('stop_loss', 0),
        'take_profit': latest.get('take_profit', 0),
        'timestamp': latest.get('timestamp', '')
    }
    
    return jsonify(chart_data)


@app.route('/api/health', methods=['GET'])
def health_check():
    """健康檢查"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.utcnow().isoformat(),
        'signals_count': len(signals_db['signals'])
    })


@app.errorhandler(404)
def not_found(error):
    """404 處理"""
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """500 處理"""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # 開發環境
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )
