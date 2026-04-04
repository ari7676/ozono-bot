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
BASE_URL   = 'https://www.alphavantage.co/query'

SYMBOLS = {
    'wallstreet': ['NVDA','AAPL','MSFT','GOOGL','AMZN','META','TSLA','JPM','V','NFLX'],
    'forex':      ['EUR','GBP','AUD','CAD','JPY'],
    'indices':    ['SPY','QQQ','DIA','IWM','VIX'],
    'crypto':     ['BTC','ETH','BNB','SOL','XRP']
}

_cache      = {}
_cache_time = {}
CACHE_TTL   = 600   # 10 min — ahorra requests
_sse_clients  = []
_prev_signals = {}

# ─── INDICADORES ──────────────────────────────────────────
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
    tr  = pd.concat([high-low,
                     (high-close.shift()).abs(),
                     (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(p).mean()
    upper = hl2 + m * atr
    lower = hl2 - m * atr
    direction = pd.Series(0, index=close.index)
    for i in range(1, len(close)):
        if   close.iloc[i] > upper.iloc[i-1]: direction.iloc[i] =  1
        elif close.iloc[i] < lower.iloc[i-1]: direction.iloc[i] = -1
        else:                                  direction.iloc[i] = direction.iloc[i-1]
    return lower.where(direction==1, upper), direction

def calc_score(close, high, low, rsi, mv, msv, ema9, st_up, vol):
    score = 0; señales = []
    price = float(close.iloc[-1])
    ema21 = float(calc_ema(close, 21).iloc[-1])
    ema50 = float(calc_ema(close, 50).iloc[-1])

    if price > ema9 > ema21 > ema50: score += 25; señales.append("EMA ↑")
    elif price < ema9 < ema21 < ema50: score += 25; señales.append("EMA ↓")

    if mv > msv: score += 20; señales.append("MACD bull")
    else:        score += 10; señales.append("MACD bear")

    if   50 < rsi < 70: score += 20; señales.append(f"RSI {rsi:.0f} bull")
    elif 30 < rsi < 50: score += 20; señales.append(f"RSI {rsi:.0f} bear")
    elif rsi < 30:      score += 10; señales.append(f"RSI {rsi:.0f} oversold")
    elif rsi > 70:      score += 10; señales.append(f"RSI {rsi:.0f} overbought")

    if st_up: score += 20; señales.append("ST ↑")
    else:     score += 20; señales.append("ST ↓")

    vol_avg = float(vol.rolling(20).mean().iloc[-1])
    if vol_avg > 0 and float(vol.iloc[-1]) > vol_avg * 1.3:
        score += 15; señales.append("Vol alto")

    return min(score, 100), señales

def get_signal_from_score(score, st_up):
    if score >= 75: return ('STRONG BUY' if st_up else 'STRONG SELL'), round(score/100, 3)
    if score >= 55: return ('BUY'        if st_up else 'SELL'),         round(score/100, 3)
    return 'NEUTRAL', 0.0

# ─── FETCH ALPHA VANTAGE ──────────────────────────────────
def fetch_daily(ticker):
    """Acciones e índices ETF"""
    r = req.get(BASE_URL, params={
        'function':   'TIME_SERIES_DAILY',
        'symbol':     ticker,
        'outputsize': 'full',
        'apikey':     ALPHA_KEY
    }, timeout=15)
    data = r.json().get('Time Series (Daily)', {})
    if not data: return None
    df = pd.DataFrame(data).T
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.rename(columns={
        '1. open':'open','2. high':'high',
        '3. low':'low', '4. close':'close','5. volume':'volume'
    }).astype(float)
    return df

def fetch_forex(from_sym):
    """Forex — par vs USD"""
    r = req.get(BASE_URL, params={
        'function':   'FX_DAILY',
        'from_symbol': from_sym,
        'to_symbol':  'USD',
        'outputsize': 'full',
        'apikey':     ALPHA_KEY
    }, timeout=15)
    data = r.json().get('Time Series FX (Daily)', {})
    if not data: return None
    df = pd.DataFrame(data).T
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.rename(columns={
        '1. open':'open','2. high':'high',
        '3. low':'low', '4. close':'close'
    }).astype(float)
    df['volume'] = 1000000  # forex no tiene volumen real
    return df

def fetch_crypto(symbol):
    """Crypto vs USD"""
    r = req.get(BASE_URL, params={
        'function': 'DIGITAL_CURRENCY_DAILY',
        'symbol':   symbol,
        'market':   'USD',
        'apikey':   ALPHA_KEY
    }, timeout=15)
    data = r.json().get('Time Series (Digital Currency Daily)', {})
    if not data: return None
    df = pd.DataFrame(data).T
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df['open']   = df['1a. open (USD)'].astype(float)
    df['high']   = df['2a. high (USD)'].astype(float)
    df['low']    = df['3a. low (USD)'].astype(float)
    df['close']  = df['4a. close (USD)'].astype(float)
    df['volume'] = df['5. volume'].astype(float)
    return df[['open','high','low','close','volume']]

def process_df(df, ticker):
    if df is None or len(df) < 50: return None
    close = df['close']
    high  = df['high']
    low   = df['low']
    vol   = df['volume']
    price = float(close.iloc[-1])
    prev  = float(close.iloc[-2])
    chg   = round((price-prev)/prev*100, 2)
    rsi   = round(float(calc_rsi(close).iloc[-1]), 1)
    if np.isnan(rsi): rsi = 50.0
    ml, ms  = calc_macd(close)
    mv      = round(float(ml.iloc[-1]), 6)
    msv     = round(float(ms.iloc[-1]), 6)
    ema9    = float(calc_ema(close, 9).iloc[-1])
    st_s, st_d = calc_supertrend(high, low, close)
    st_val  = round(float(st_s.iloc[-1]), 4)
    st_up   = int(st_d.iloc[-1]) == 1
    score, señales = calc_score(close, high, low, rsi, mv, msv, ema9, st_up, vol)
    sig, rec = get_signal_from_score(score, st_up)
    w52h = round(float(close.tail(252).max()), 4)
    w52l = round(float(close.tail(252).min()), 4)
    return {
        'symbol': ticker, 'price': round(price, 6 if price<1 else 2),
        'change': chg, 'volume': int(float(vol.iloc[-1])),
        'signal': sig, 'recommend': rec,
        'score': score, 'señales': señales,
        'w52h': w52h, 'w52l': w52l,
        'pe': None, 'beta': None, 'mcap': None,
        'indicators': {
            'macd':       {'val': mv,           'status': 'Bull' if mv>msv else 'Bear'},
            'rsi':        {'val': rsi,           'status': 'Oversold' if rsi<30 else 'Neutral' if rsi<70 else 'Overbought'},
            'ema9':       {'val': round(ema9,4), 'status': 'Above' if price>ema9 else 'Below'},
            'supertrend': {'val': st_val,        'status': 'Up' if st_up else 'Down'},
            'bbp':        {'val': 0,             'status': 'Bull' if st_up else 'Bear'},
        }
}
    }
    183            'bbp':        {'val': 0,             'status': 'Bull' if st_up else 'Bear'},
184        }
185    }
