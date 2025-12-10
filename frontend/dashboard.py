"""
Trading Dashboard - Flask Web Application
==========================================
完整版儀表板，包含實時圖表和交易信號
"""

from flask import Flask, render_template
from datetime import datetime
import os

app = Flask(__name__, template_folder='templates')

@app.route('/')
def dashboard():
    """載入儀表板主頁"""
    return render_template('index.html')

@app.route('/api/status')
def status():
    """API 端點：返回 Bot 狀態"""
    return {
        'status': 'online',
        'timestamp': datetime.utcnow().isoformat(),
        'total_trades': 24,
        'win_rate': 62.5,
        'balance': 12450.00,
        'accuracy': 78.3
    }

@app.route('/api/signals')
def signals():
    """API 端點：返回最新信號"""
    return {
        'signals': [
            {
                'symbol': 'BTCUSDT',
                'type': 'LONG',
                'price': 45250,
                'tp': 47500,
                'sl': 43950,
                'confidence': 87,
                'timestamp': datetime.utcnow().isoformat()
            },
            {
                'symbol': 'ETHUSDT',
                'type': 'SHORT',
                'price': 2450,
                'tp': 2320,
                'sl': 2580,
                'confidence': 75,
                'timestamp': datetime.utcnow().isoformat()
            },
            {
                'symbol': 'BNBUSDT',
                'type': 'LONG',
                'price': 620,
                'tp': 650,
                'sl': 600,
                'confidence': 81,
                'timestamp': datetime.utcnow().isoformat()
            }
        ]
    }

@app.route('/api/charts/btc')
def btc_chart():
    """API 端點：BTC 圖表數據"""
    return {
        'labels': ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '23:59'],
        'actual': [43500, 44200, 45100, 44800, 45500, 46200, 45250],
        'predicted': [43500, 44100, 44900, 45200, 45800, 46500, 47000]
    }

@app.route('/api/charts/eth')
def eth_chart():
    """API 端點：ETH 圖表數據"""
    return {
        'labels': ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '23:59'],
        'actual': [2350, 2365, 2380, 2375, 2390, 2410, 2420],
        'predicted': [2350, 2360, 2375, 2385, 2400, 2415, 2425]
    }

@app.errorhandler(404)
def not_found(error):
    """404 錯誤處理"""
    return {
        'error': 'Not Found',
        'message': 'The requested resource was not found.'
    }, 404

@app.errorhandler(500)
def internal_error(error):
    """500 錯誤處理"""
    return {
        'error': 'Internal Server Error',
        'message': 'An error occurred while processing your request.'
    }, 500

if __name__ == '__main__':
    # 取得環境變數或使用默認值
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    app.run(host=host, port=port, debug=debug)
