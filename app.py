from flask import Flask, jsonify, render_template, Response
from flask_cors import CORS
import pandas as pd
import numpy as np
import json, time, threading, os
from datetime import datetime
import requests as req

app = Flask(__name__)
CORS(app)

ALPHA_KEY = os.environ.get('alpha_key', '')
BASE_URL = 'https://www.alphavantage.co/query'

SYMBOLS = {
    'wallstreet': ['NVDA','AAPL','MSFT','GOOGL','AMZN','META','TSLA','JPM','V','NFLX'],
    'forex':      ['EUR','GBP','AUD','CAD','JPY'],
    'indices':    ['SPY','QQQ','DIA','IWM','VIX'],
    'crypto':     ['BTC','ETH','BNB','SOL','XRP']
}

_cache = {}
_cache_time = {}
CACHE_TTL = 600
_sse_clients = []
_prev_signals = {}

def calc_ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

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
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(p).mean()
    upper = hl2 + m * atr
    lower = hl2 - m * atr
    direction = pd.Series(0, index=close.index)
    for i in range(1, len(close)):
        if close.iloc[i] > upper.iloc[i-1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < lower.iloc[i-1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]
    st = lower.where(direction == 1, upper)
    return st, direction

def calc_score(close, high, low, rsi, mv, msv, ema9, st_up, vol):
    score = 0
    senales = []
    price = float(close.iloc[-1])
    ema21 = float(calc_ema(close, 21).iloc[-1])
    ema50 = float(calc_ema(close, 50).iloc[-1])

    if price > ema9 > ema21 > ema50:
        score += 25
        senales.append("EMA up")
    elif price < ema9 < ema21 < ema50:
        score += 25
        senales.append("EMA down")

    if mv > msv:
        score += 20
        senales.append("MACD bull")
    else:
        score += 10
        senales.append("MACD bear")

    if 50 < rsi < 70:
        score += 20
        senales.append("RSI bull")
    elif 30 < rsi < 50:
        score += 20
        senales.append("RSI bear")
    elif rsi < 30:
        score += 10
        senales.append("RSI oversold")
    elif rsi > 70:
        score += 10
        senales.append("RSI overbought")

    if st_up:
        score += 20
        senales.append("ST up")
    else:
        score += 20
        senales.append("ST down")

    vol_avg = float(vol.rolling(20).mean().iloc[-1])
    if vol_avg > 0 and float(vol.iloc[-1]) > vol_avg * 1.3:
        score += 15
        senales.append("Vol alto")

    return min(score, 100), senales

def get_signal_from_score(score, st_up):
    if score >= 75:
        return ('STRONG BUY' if st_up else 'STRONG SELL'), round(score/100, 3)
    if score >= 55:
        return ('BUY' if st_up else 'SELL'), round(score/100, 3)
    return 'NEUTRAL', 0.0

def fetch_daily(ticker):
    r = req.get(BASE_URL, params={
        'function': 'TIME_SERIES_DAILY',
        'symbol': ticker,
        'outputsize': 'full',
        'apikey': ALPHA_KEY
    }, timeout=15)
    data = r.json().get('Time Series (Daily)', {})
    if not data:
        return None
    df = pd.DataFrame(data).T
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.rename(columns={
        '1. open': 'open', '2. high': 'high',
        '3. low': 'low', '4. close': 'close', '5. volume': 'volume'
    }).astype(float)
    return df

def fetch_forex(from_sym):
    r = req.get(BASE_URL, params={
        'function': 'FX_DAILY',
        'from_symbol': from_sym,
        'to_symbol': 'USD',
        'outputsize': 'full',
        'apikey': ALPHA_KEY
    }, timeout=15)
    data = r.json().get('Time Series FX (Daily)', {})
    if not data:
        return None
    df = pd.DataFrame(data).T
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.rename(columns={
        '1. open': 'open', '2. high': 'high',
        '3. low': 'low', '4. close': 'close'
    }).astype(float)
    df['volume'] = 1000000
    return df

def fetch_crypto(symbol):
    r = req.get(BASE_URL, params={
        'function': 'DIGITAL_CURRENCY_DAILY',
        'symbol': symbol,
        'market': 'USD',
        'apikey': ALPHA_KEY
    }, timeout=15)
    data = r.json().get('Time Series (Digital Currency Daily)', {})
    if not data:
        return None
    df = pd.DataFrame(data).T
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df['open'] = df['1a. open (USD)'].astype(float)
    df['high'] = df['2a. high (USD)'].astype(float)
    df['low'] = df['3a. low (USD)'].astype(float)
    df['close'] = df['4a. close (USD)'].astype(float)
    df['volume'] = df['5. volume'].astype(float)
    return df[['open', 'high', 'low', 'close', 'volume']]

def process_df(df, ticker):
    if df is None or len(df) < 50:
        return None
    close = df['close']
    high = df['high']
    low = df['low']
    vol = df['volume']
    price = float(close.iloc[-1])
    prev = float(close.iloc[-2])
    chg = round((price - prev) / prev * 100, 2)
    rsi = round(float(calc_rsi(close).iloc[-1]), 1)
    if np.isnan(rsi):
        rsi = 50.0
    ml, ms = calc_macd(close)
    mv = round(float(ml.iloc[-1]), 6)
    msv = round(float(ms.iloc[-1]), 6)
    ema9 = float(calc_ema(close, 9).iloc[-1])
    st_s, st_d = calc_supertrend(high, low, close)
    st_val = round(float(st_s.iloc[-1]), 4)
    st_up = int(st_d.iloc[-1]) == 1
    score, senales = calc_score(close, high, low, rsi, mv, msv, ema9, st_up, vol)
    sig, rec = get_signal_from_score(score, st_up)
    w52h = round(float(close.tail(252).max()), 4)
    w52l = round(float(close.tail(252).min()), 4)
    return {
        'symbol': ticker,
        'price': round(price, 6 if price < 1 else 2),
        'change': chg,
        'volume': int(float(vol.iloc[-1])),
        'signal': sig,
        'recommend': rec,
        'score': score,
        'senales': senales,
        'w52h': w52h,
        'w52l': w52l,
        'pe': None,
        'beta': None,
        'mcap': None,
        'indicators': {
            'macd': {'val': mv, 'status': 'Bull' if mv > msv else 'Bear'},
            'rsi': {'val': rsi, 'status': 'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral'},
            'ema9': {'val': round(ema9, 4), 'status': 'Above' if price > ema9 else 'Below'},
            'supertrend': {'val': st_val, 'status': 'Up' if st_up else 'Down'},
            'bbp': {'val': 0, 'status': 'Bull' if st_up else 'Bear'}
        }
    }

def fetch_symbol(ticker, market):
    try:
        if market == 'forex':
            df = fetch_forex(ticker)
        elif market == 'crypto':
            df = fetch_crypto(ticker)
        else:
            df = fetch_daily(ticker)
        return process_df(df, ticker)
    except Exception as e:
        print(f"[fetch] {ticker}: {e}")
        return None

def fetch_market_background(market):
    syms = SYMBOLS.get(market, [])
    results = []
    for sym in syms:
        d = fetch_symbol(sym, market)
        if d:
            results.append(d)
        time.sleep(13)
    results.sort(key=lambda x: x.get('score', 0), reverse=True)
    _cache[market] = results
    _cache_time[market] = time.time()
    print(f"[cache] {market}: {len(results)} simbolos")

def fetch_market(market):
    now = time.time()
    # Si hay cache válido, devolvelo inmediatamente
    if market in _cache and (now - _cache_time.get(market, 0)) < CACHE_TTL:
        return _cache[market]
    # Si no hay cache, lanzá fetch en background y devolvé lista vacía
    if market not in _cache:
        threading.Thread(target=fetch_market_background, args=(market,), daemon=True).start()
    return _cache.get(market, [])

@app.route('/api/scan/<market>')
def scan(market):
    return jsonify(fetch_market(market))

def monitor_loop():
    time.sleep(30)
    while True:
        try:
            for market in SYMBOLS:
                for item in fetch_market(market):
                    sym = item['symbol']
                    sig = item['signal']
                    prev = _prev_signals.get(sym)
                    if prev and prev != sig and sig in ('BUY', 'SELL', 'STRONG BUY', 'STRONG SELL'):
                        alert = {
                            'symbol': sym,
                            'signal': sig,
                            'prev': prev,
                            'price': item['price'],
                            'change': item['change'],
                            'score': item.get('score', 0),
                            'market': market,
                            'time': datetime.now().strftime('%H:%M:%S')
                        }
                        for q in list(_sse_clients):
                            q.append(alert)
                    _prev_signals[sym] = sig
        except Exception as e:
            print(f"[monitor] {e}")
        time.sleep(600)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/scan/<market>')
def scan(market):
    return jsonify(fetch_market(market))

@app.route('/api/fear-greed')
def fear_greed():
    try:
        r = req.get('https://api.alternative.me/fng/?limit=2', timeout=8)
        data = r.json()['data']
        cur = data[0]
        prev = data[1] if len(data) > 1 else None
        return jsonify({
            'value': int(cur['value']),
            'label': cur['value_classification'],
            'prev': int(prev['value']) if prev else None
        })
    except Exception:
        return jsonify({'value': None, 'label': 'N/A', 'prev': None})

@app.route('/api/summary')
def summary():
    out = {}
    for m in SYMBOLS:
        items = _cache.get(m, [])
        bulls = [i for i in items if 'BUY' in i['signal']]
        bears = [i for i in items if 'SELL' in i['signal']]
        by_chg = sorted(items, key=lambda x: x['change'])
        out[m] = {
            'total': len(items),
            'bulls': len(bulls),
            'bears': len(bears),
            'neutral': len(items) - len(bulls) - len(bears),
            'best': by_chg[-1] if by_chg else None,
            'worst': by_chg[0] if by_chg else None
        }
    return jsonify(out)

@app.route('/api/alerts/stream')
def alert_stream():
    q = []
    _sse_clients.append(q)
    def gen():
        try:
            while True:
                if q:
                    yield f"data: {json.dumps(q.pop(0))}\n\n"
                else:
                    yield ": ping\n\n"
                time.sleep(1)
        finally:
            if q in _sse_clients:
                _sse_clients.remove(q)
    return Response(gen(), content_type='text/event-stream', headers={'Cache-Control': 'no-cache'})

threading.Thread(target=monitor_loop, daemon=True).start()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
