from flask import Flask, jsonify, render_template, Response
from flask_cors import CORS
import pandas as pd
import numpy as np
import json, time, threading, os
from datetime import datetime
import requests as req

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
CORS(app)

TWELVE_KEY       = os.environ.get('TWELVE_KEY', '')
TELEGRAM_TOKEN   = os.environ.get('TELEGRAM_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')
BASE_URL         = 'https://api.twelvedata.com'

# =====================
# SIMBOLOS — sincronizado con Royal Purple Scanners v2
# =====================
SYMBOLS = {
    'wallstreet': ['NVDA','AAPL','MSFT','GOOGL','AMZN','META','TSLA','AVGO',
                   'JPM','V','MA','JNJ','PG','MRK','COST','ABBV','BAC',
                   'NFLX','CRM','AMD','INTC','MU','CSCO','TSM','AMAT',
                   'MCD','VLO','PLTR','IBM','BMRN','ORCL','GEV','AVAV'],
    'indices':    ['SPY','QQQ','DIA','IWM','GLD','SLV','USO','TLT','XLK'],
    'crypto':     ['BTC/USD','ETH/USD','BNB/USD','SOL/USD','XRP/USD',
                   'DOGE/USD','ADA/USD','AVAX/USD']
}

_cache        = {}
_cache_time   = {}
_loading      = set()
CACHE_TTL     = 3600
_sse_clients  = []
_prev_signals = {}

def send_telegram(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        req.post(f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage',
                 json={'chat_id': TELEGRAM_CHAT_ID, 'text': msg, 'parse_mode': 'HTML'},
                 timeout=10)
    except Exception as e:
        print(f'[telegram] {e}')

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
    tr  = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(p).mean()
    upper = hl2 + m * atr
    lower = hl2 - m * atr
    direction = pd.Series(0, index=close.index)
    for i in range(1, len(close)):
        if   close.iloc[i] > upper.iloc[i-1]: direction.iloc[i] =  1
        elif close.iloc[i] < lower.iloc[i-1]: direction.iloc[i] = -1
        else:                                  direction.iloc[i] = direction.iloc[i-1]
    return lower.where(direction == 1, upper), direction

# =====================
# ROYAL PURPLE v2 — con filtro RSI < 75
# Consistente con Royal Purple Scanner v2 de TradingView
# =====================
def calc_royal_purple(close, high, low, fast=20, slow=50, regime=200, atr_mult=3.0, rsi_max=75):
    if len(close) < regime + 10:
        return None, None
    ema_fast   = calc_ema(close, fast)
    ema_slow   = calc_ema(close, slow)
    ema_regime = calc_ema(close, regime)
    atr        = (high - low).rolling(14).mean()
    rsi        = calc_rsi(close)

    price = float(close.iloc[-1])
    ef    = float(ema_fast.iloc[-1])
    es    = float(ema_slow.iloc[-1])
    er    = float(ema_regime.iloc[-1])
    rsi_v = float(rsi.iloc[-1])

    # Condicion base: 3 EMAs alineadas + RSI no sobrecomprado
    bull_state = price > es and price > er and ef > es and rsi_v < rsi_max
    trail      = round(float(close.iloc[-1] - atr.iloc[-1] * atr_mult), 6)

    if bull_state:
        return 'LONG', trail
    else:
        return None, trail

def calc_score(close, high, low, rsi, mv, msv, ema9, st_up, vol):
    score = 0
    tags  = []
    price = float(close.iloc[-1])
    ema21 = float(calc_ema(close, 21).iloc[-1])
    ema50 = float(calc_ema(close, 50).iloc[-1])

    if price > ema9 > ema21 > ema50:   score += 25; tags.append("EMA up")
    elif price < ema9 < ema21 < ema50: score += 25; tags.append("EMA down")

    if mv > msv: score += 20; tags.append("MACD bull")
    else:        score += 10; tags.append("MACD bear")

    if   50 < rsi < 70: score += 20; tags.append(f"RSI {rsi:.0f} bull")
    elif 30 < rsi < 50: score += 20; tags.append(f"RSI {rsi:.0f} bear")
    elif rsi <= 30:     score += 10; tags.append(f"RSI {rsi:.0f} oversold")
    else:               score += 10; tags.append(f"RSI {rsi:.0f} overbought")

    score += 20; tags.append("ST up" if st_up else "ST down")

    vol_avg = float(vol.rolling(20).mean().iloc[-1])
    if vol_avg > 0 and float(vol.iloc[-1]) > vol_avg * 1.3:
        score += 15; tags.append("Vol alto")

    return min(score, 100), tags

def get_signal_from_score(score, st_up):
    if score >= 75: return ('STRONG BUY' if st_up else 'STRONG SELL'), round(score/100, 3)
    if score >= 55: return ('BUY'        if st_up else 'SELL'),         round(score/100, 3)
    return 'NEUTRAL', 0.0

def fetch_twelve(symbol):
    r = req.get(f'{BASE_URL}/time_series', params={
        'symbol':     symbol,
        'interval':   '1day',
        'outputsize': 300,
        'apikey':     TWELVE_KEY
    }, timeout=15)
    data = r.json()
    if 'values' not in data:
        print(f"[twelve] {symbol}: {data.get('message','no values')}")
        return None
    df = pd.DataFrame(data['values'])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()
    for col in ['open','high','low','close','volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'volume' not in df.columns:
        df['volume'] = 1_000_000
    return df

def process_df(df, ticker):
    if df is None or len(df) < 50:
        return None
    close = df['close']
    high  = df['high']
    low   = df['low']
    vol   = df['volume']
    price = float(close.iloc[-1])
    prev  = float(close.iloc[-2])
    chg   = round((price - prev) / prev * 100, 2)
    rsi   = round(float(calc_rsi(close).iloc[-1]), 1)
    if np.isnan(rsi): rsi = 50.0
    ml, ms  = calc_macd(close)
    mv      = round(float(ml.iloc[-1]), 6)
    msv     = round(float(ms.iloc[-1]), 6)
    ema9    = float(calc_ema(close, 9).iloc[-1])
    st_s, st_d = calc_supertrend(high, low, close)
    st_val  = round(float(st_s.iloc[-1]), 4)
    st_up   = int(st_d.iloc[-1]) == 1
    score, tags = calc_score(close, high, low, rsi, mv, msv, ema9, st_up, vol)
    sig, rec    = get_signal_from_score(score, st_up)
    w52h = round(float(close.tail(252).max()), 4)
    w52l = round(float(close.tail(252).min()), 4)
    sym  = ticker.replace('/USD','').replace('/','_')
    rp_signal, rp_trail = calc_royal_purple(close, high, low)
    return {
        'symbol':       sym,
        'price':        round(price, 6 if price < 1 else 2),
        'change':       chg,
        'volume':       int(float(vol.iloc[-1])),
        'signal':       sig,
        'recommend':    rec,
        'score':        score,
        'senales':      tags,
        'w52h':         w52h,
        'w52l':         w52l,
        'royal_purple': rp_signal,
        'rp_trail':     rp_trail,
        'pe': None, 'beta': None, 'mcap': None,
        'indicators': {
            'macd':       {'val': mv,            'status': 'Bull' if mv > msv else 'Bear'},
            'rsi':        {'val': rsi,            'status': 'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral'},
            'ema9':       {'val': round(ema9, 4), 'status': 'Above' if price > ema9 else 'Below'},
            'supertrend': {'val': st_val,         'status': 'Up' if st_up else 'Down'},
            'bbp':        {'val': 0,              'status': 'Bull' if st_up else 'Bear'}
        }
    }

def fetch_symbol(ticker):
    try:
        df = fetch_twelve(ticker)
        return process_df(df, ticker)
    except Exception as e:
        print(f"[fetch] {ticker}: {e}")
        return None

def fetch_market_background(market):
    _loading.add(market)
    syms    = SYMBOLS.get(market, [])
    results = []
    batch_size = 4
    for i in range(0, len(syms), batch_size):
        batch = syms[i:i+batch_size]
        for sym in batch:
            d = fetch_symbol(sym)
            if d: results.append(d)
        if i + batch_size < len(syms):
            time.sleep(15)
    results.sort(key=lambda x: x.get('score', 0), reverse=True)
    _cache[market]      = results
    _cache_time[market] = time.time()
    _loading.discard(market)
    print(f"[cache] {market}: {len(results)} simbolos")

def fetch_market(market):
    now = time.time()
    if market in _cache and (now - _cache_time.get(market, 0)) < CACHE_TTL:
        return _cache[market]
    if market not in _loading:
        threading.Thread(target=fetch_market_background, args=(market,), daemon=True).start()
    return _cache.get(market, [])

def monitor_loop():
    time.sleep(60)
    while True:
        try:
            for market in SYMBOLS:
                for item in _cache.get(market, []):
                    sym  = item['symbol']
                    sig  = item['signal']
                    rp   = item.get('royal_purple')
                    prev = _prev_signals.get(sym, {})
                    prev_sig = prev.get('signal')
                    prev_rp  = prev.get('rp')

                    if prev_sig and prev_sig != sig and sig in ('BUY','SELL','STRONG BUY','STRONG SELL'):
                        alert = {
                            'symbol': sym, 'signal': sig, 'prev': prev_sig,
                            'price':  item['price'], 'change': item['change'],
                            'score':  item.get('score', 0), 'market': market,
                            'time':   datetime.now().strftime('%H:%M:%S')
                        }
                        for q in list(_sse_clients): q.append(alert)

                    # =====================
                    # ALERTA ROYAL PURPLE v2 — incluye RSI en el mensaje
                    # =====================
                    if rp == 'LONG' and prev_rp != 'LONG':
                        trail   = item.get('rp_trail', 0)
                        rsi_val = item.get('indicators', {}).get('rsi', {}).get('val', 0)
                        msg = (
                            f"🟣 <b>ROYAL PURPLE v2 — ENTRADA LONG</b>\n"
                            f"📊 <b>{sym}</b> | {market.upper()}\n"
                            f"💰 Precio: <b>${item['price']}</b>\n"
                            f"📉 RSI: <b>{rsi_val}</b> (filtro &lt;75 activo)\n"
                            f"🛡 Trail Stop: <b>${trail}</b>\n"
                            f"📈 Score: {item.get('score', 0)}/100\n"
                            f"🕐 {datetime.now().strftime('%H:%M:%S')}"
                        )
                        send_telegram(msg)
                        for q in list(_sse_clients): q.append({
                            'symbol': sym, 'signal': 'ROYAL PURPLE LONG', 'prev': '',
                            'price':  item['price'], 'change': item['change'],
                            'score':  item.get('score', 0), 'market': market,
                            'time':   datetime.now().strftime('%H:%M:%S')
                        })

                    _prev_signals[sym] = {'signal': sig, 'rp': rp}
        except Exception as e:
            print(f"[monitor] {e}")
        time.sleep(300)

@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/scan/<market>')
def scan(market): return jsonify(fetch_market(market))

@app.route('/api/refresh/<market>')
def force_refresh(market):
    _cache.pop(market, None)
    _cache_time.pop(market, None)
    threading.Thread(target=fetch_market_background, args=(market,), daemon=True).start()
    return jsonify({'status': 'refreshing', 'market': market})

@app.route('/api/royal-purple')
def royal_purple_activos():
    results = []
    for market, items in _cache.items():
        for item in items:
            if item.get('royal_purple') == 'LONG':
                results.append({**item, 'market': market})
    results.sort(key=lambda x: x.get('score', 0), reverse=True)
    return jsonify(results)

@app.route('/api/fear-greed')
def fear_greed():
    try:
        r = req.get('https://api.alternative.me/fng/?limit=2', timeout=8)
        data = r.json()['data']
        cur  = data[0]; prev = data[1] if len(data) > 1 else None
        return jsonify({
            'value': int(cur['value']),
            'label': cur['value_classification'],
            'prev':  int(prev['value']) if prev else None
        })
    except Exception:
        return jsonify({'value': None, 'label': 'N/A', 'prev': None})

@app.route('/api/summary')
def summary():
    out = {}
    for m in SYMBOLS:
        items  = _cache.get(m, [])
        bulls  = [i for i in items if 'BUY'  in i['signal']]
        bears  = [i for i in items if 'SELL' in i['signal']]
        by_chg = sorted(items, key=lambda x: x['change'])
        out[m] = {
            'total':   len(items), 'bulls': len(bulls), 'bears': len(bears),
            'neutral': len(items) - len(bulls) - len(bears),
            'best':    by_chg[-1] if by_chg else None,
            'worst':   by_chg[0]  if by_chg else None
        }
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
    return Response(gen(), content_type='text/event-stream', headers={'Cache-Control': 'no-cache'})

def startup_loader():
    for market in SYMBOLS:
        fetch_market_background(market)
        time.sleep(5)

threading.Thread(target=startup_loader, daemon=True).start()
threading.Thread(target=monitor_loop, daemon=True).start()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
