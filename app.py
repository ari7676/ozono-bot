from flask import Flask, jsonify, render_template, Response
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
import json, time, threading
from datetime import datetime
import requests as req

app = Flask(__name__)
CORS(app)

SYMBOLS = {
    'wallstreet': [
        'NVDA','AAPL','MSFT','GOOGL','AMZN','META','TSLA','AVGO',
        'JPM','V','MA','UNH','XOM','WMT','LLY','JNJ','PG','MRK',
        'HD','COST','ABBV','BAC','NFLX','CRM','AMD','INTC','MU','CSCO'
    ],
    'forex': [
        'EURUSD=X','GBPUSD=X','USDJPY=X','AUDUSD=X','USDCAD=X',
        'USDCHF=X','NZDUSD=X','EURGBP=X','EURJPY=X','GBPJPY=X',
        'USDARS=X','USDBRL=X','USDMXN=X','USDCLP=X'
    ],
    'indices': [
        '^GSPC','^DJI','^IXIC','^RUT','^VIX',
        '^FTSE','^GDAXI','^FCHI','^N225','^HSI'
    ],
    'crypto': [
        'BTC-USD','ETH-USD','BNB-USD','SOL-USD','XRP-USD',
        'ADA-USD','DOGE-USD','AVAX-USD','DOT-USD','MATIC-USD'
    ]
}

_cache = {}
_cache_time = {}
CACHE_TTL = 180
_sse_clients = []
_prev_signals = {}

def calc_ema(s, p): return s.ewm(span=p, adjust=False).mean()

def calc_rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    rs = g / l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_macd(s, fast=12, slow=26, sig=9):
    m = calc_ema(s, fast) - calc_ema(s, slow)
    return m, calc_ema(m, sig)

def calc_supertrend(high, low, close, p=10, m=3):
    hl2 = (high + low) / 2
    tr = pd.concat([high-low,(high-close.shift()).abs(),(low-close.shift()).abs()],axis=1).max(axis=1)
    atr = tr.rolling(p).mean()
    upper = hl2 + m * atr
    lower = hl2 - m * atr
    direction = pd.Series(0, index=close.index)
    for i in range(1, len(close)):
        if close.iloc[i] > upper.iloc[i-1]: direction.iloc[i] = 1
        elif close.iloc[i] < lower.iloc[i-1]: direction.iloc[i] = -1
        else: direction.iloc[i] = direction.iloc[i-1]
    st = lower.where(direction==1, upper)
    return st, direction

def get_signal(rsi, macd_v, macd_s, ema9, price, bbp):
    score = 0
    if macd_v > macd_s: score += 1
    else: score -= 1
    if price > ema9: score += 1
    else: score -= 1
    if rsi > 55: score += 1
    elif rsi < 45: score -= 1
    if bbp > 0: score += 1
    else: score -= 1
    if score >= 3: return 'STRONG BUY', round(score/4, 3)
    if score >= 1: return 'BUY', round(score/4, 3)
    if score <= -3: return 'STRONG SELL', round(score/4, 3)
    if score <= -1: return 'SELL', round(score/4, 3)
    return 'NEUTRAL', 0.0

def fetch_symbol(ticker):
    try:
        hist = yf.download(ticker, period='1y', interval='1d', progress=False, auto_adjust=True)
        if hist.empty or len(hist) < 30: return None
        close = hist['Close'].squeeze()
        high  = hist['High'].squeeze()
        low   = hist['Low'].squeeze()
        vol   = hist['Volume'].squeeze()
        price = float(close.iloc[-1])
        prev  = float(close.iloc[-2])
        chg   = round((price-prev)/prev*100, 2)
        rsi   = float(calc_rsi(close).iloc[-1])
        if np.isnan(rsi): rsi = 50.0
        rsi   = round(rsi, 1)
        ml, ms = calc_macd(close)
        mv = round(float(ml.iloc[-1]), 4)
        msv= round(float(ms.iloc[-1]), 4)
        ema9 = round(float(calc_ema(close,9).iloc[-1]), 2)
        st_s, st_d = calc_supertrend(high, low, close)
        st_val = round(float(st_s.iloc[-1]), 2)
        st_up  = int(st_d.iloc[-1]) == 1
        bbp = round(float(high.iloc[-1]-low.iloc[-1])*(rsi/100-0.5)*2, 2)
        w52h = round(float(close.tail(252).max()), 2)
        w52l = round(float(close.tail(252).min()), 2)
        sig, rec = get_signal(rsi, mv, msv, ema9, price, bbp)
        sym = ticker.replace('-USD','').replace('^','')
        return {
            'symbol': sym, 'price': round(price,6 if price<1 else 2),
            'change': chg, 'volume': int(float(vol.iloc[-1])),
            'signal': sig, 'recommend': rec,
            'w52h': w52h, 'w52l': w52l,
            'pe': None, 'beta': None, 'mcap': None,
            'indicators': {
                'macd':       {'val': mv,   'status': 'Bull' if mv>msv else 'Bear'},
                'rsi':        {'val': rsi,  'status': 'Oversold' if rsi<30 else 'Overbought' if rsi>70 else 'Neutral'},
                'ema9':       {'val': ema9, 'status': 'Above' if price>ema9 else 'Below'},
                'supertrend': {'val': st_val, 'status': 'Up' if st_up else 'Down'},
                'bbp':        {'val': bbp,  'status': 'Bull' if bbp>0 else 'Bear'},
            }
        }
    except Exception as e:
        print(f"[fetch] {ticker}: {e}")
        return None

def fetch_market(market):
    now = time.time()
    if market in _cache and (now - _cache_time.get(market,0)) < CACHE_TTL:
        return _cache[market]
    syms = SYMBOLS.get(market, [])
    results = []
    for sym in syms:
        d = fetch_symbol(sym)
        if d: results.append(d)
    _cache[market] = results
    _cache_time[market] = now
    print(f"[cache] {market}: {len(results)} simbolos")
    return results

def monitor_loop():
    while True:
        try:
            for market in ('wallstreet','crypto','forex'):
                for item in fetch_market(market):
                    sym = item['symbol']; sig = item['signal']
                    prev = _prev_signals.get(sym)
                    if prev and prev != sig and sig in ('BUY','SELL','STRONG BUY','STRONG SELL'):
                        alert = {'symbol':sym,'signal':sig,'prev':prev,'price':item['price'],'change':item['change'],'market':market,'time':datetime.now().strftime('%H:%M:%S')}
                        for q in list(_sse_clients): q.append(alert)
                    _prev_signals[sym] = sig
        except Exception as e:
            print(f"[monitor] {e}")
        time.sleep(180)

@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/scan/<market>')
def scan(market): return jsonify(fetch_market(market))

@app.route('/api/fear-greed')
def fear_greed():
    try:
        r = req.get('https://api.alternative.me/fng/?limit=2', timeout=8)
        data = r.json()['data']
        cur = data[0]; prev = data[1] if len(data)>1 else None
        return jsonify({'value':int(cur['value']),'label':cur['value_classification'],'prev':int(prev['value']) if prev else None})
    except:
        return jsonify({'value':None,'label':'N/A','prev':None})

@app.route('/api/summary')
def summary():
    out = {}
    for m in ('wallstreet','crypto','indices','forex'):
        items = fetch_market(m)
        bulls = [i for i in items if 'BUY' in i['signal']]
        bears = [i for i in items if 'SELL' in i['signal']]
        by_chg = sorted(items, key=lambda x: x['change'])
        out[m] = {'total':len(items),'bulls':len(bulls),'bears':len(bears),'neutral':len(items)-len(bulls)-len(bears),'best':by_chg[-1] if by_chg else None,'worst':by_chg[0] if by_chg else None}
    return jsonify(out)

@app.route('/api/alerts/stream')
def alert_stream():
    q = []; _sse_clients.append(q)
    def gen():
        try:
            while True:
                yield f"data: {json.dumps(q.pop(0))}\n\n" if q else ": ping\n\n"
                time.sleep(1)
        finally:
            if q in _sse_clients: _sse_clients.remove(q)
    return Response(gen(), content_type='text/event-stream', headers={'Cache-Control':'no-cache'})

if __name__ == '__main__':
    threading.Thread(target=monitor_loop, daemon=True).start()
    print("Ozono Bot corriendo en http://localhost:5000")
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
