import streamlit as st
import ccxt
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta  # noqa: F401
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests

st.set_page_config(page_title="PRO 퀀트 트레이딩 대시보드", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# 튜닝 상수
# ==========================================
RANGE_N = 120
PIV_K = 5
LONG_EMA = 100
CRASH_DD = 0.20
ACC_CAP_NO_TRIPLE = 55
BEARFLAG_SLOPE = -2.0
BEARFLAG_PENALTY = 15
MTF_VETO_RATIO = 1.0
POC_BINS = 24

UPPER_TF = {"15분": "1시간", "1시간": "4시간", "4시간": "1일", "1일": "주봉", "주봉": "월봉", "월봉": None}
HTF_DELTA = {"1시간": pd.Timedelta(hours=1), "4시간": pd.Timedelta(hours=4), "1일": pd.Timedelta(days=1),
             "주봉": pd.Timedelta(days=7), "월봉": pd.Timedelta(days=31)}

# ==========================================
# 매크로 및 공통 유틸리티
# ==========================================
@st.cache_data(ttl=3600)
def get_krw_exchange_rate():
    try:
        h = yf.Ticker("KRW=X").history(period="5d")
        return float(h['Close'].dropna().iloc[-1])
    except Exception:
        return 1350.0

KRW_RATE = get_krw_exchange_rate()

@st.cache_data(ttl=600)
def get_btc_dominance():
    try:
        res = requests.get('https://api.coingecko.com/api/v3/global', timeout=5).json()
        return res['data']['market_cap_percentage']['btc']
    except Exception:
        return 53.0

@st.cache_data(ttl=900)
def get_macro_indicators():
    macro = {"fng": 50, "fng_label": "Neutral", "dxy_trend": "Neutral", "vix": 20.0, "vix_panic": False}
    try:
        res = requests.get('https://api.alternative.me/fng/', timeout=3).json()
        macro["fng"] = int(res['data'][0]['value'])
        macro["fng_label"] = res['data'][0]['value_classification']
    except Exception:
        pass
    
    try:
        tickers = yf.download(["^VIX", "DX-Y.NYB"], period="15d", progress=False)
        if not tickers.empty:
            if isinstance(tickers.columns, pd.MultiIndex):
                vix_s = tickers['Close']['^VIX'].dropna()
                dxy_s = tickers['Close']['DX-Y.NYB'].dropna()
            else:
                vix_s = tickers['Close'].dropna()
                dxy_s = tickers['Close'].dropna()
                
            macro['vix'] = float(vix_s.iloc[-1])
            macro['vix_panic'] = macro['vix'] >= 35.0
            macro['dxy_trend'] = "Bull" if float(dxy_s.iloc[-1]) > float(dxy_s.iloc[-10]) else "Bear"
    except Exception:
        pass
    return macro

@st.cache_data(ttl=300)
def get_crypto_sentiment(krw_rate):
    sent = {"kimp": 0.0, "funding_rate": 0.0}
    try:
        b_btc = ccxt.binance().fetch_ticker('BTC/USDT')['last']
        u_btc = ccxt.upbit().fetch_ticker('BTC/KRW')['last']
        sent['kimp'] = (u_btc / (b_btc * krw_rate) - 1) * 100
    except Exception:
        pass
    
    try:
        fund = ccxt.binance({'options': {'defaultType': 'future'}}).fetch_funding_rate('BTC/USDT')
        sent['funding_rate'] = float(fund['fundingRate']) * 100
    except Exception:
        pass
    return sent

@st.cache_data(ttl=600)
def get_sector_volumes(sectors_dict):
    all_tickers = [t.split(" ")[0] for ticks in sectors_dict.values() for t in ticks]
    try:
        df = yf.download(all_tickers, period="5d", progress=False)
        is_multi = isinstance(df.columns, pd.MultiIndex)
        vols = {}
        for sec, ticks in sectors_dict.items():
            sec_vol = 0
            for tick_str in ticks:
                tick = tick_str.split(" ")[0]
                try:
                    if is_multi:
                        close = df['Close'][tick].dropna().iloc[-1]
                        vol = df['Volume'][tick].dropna().iloc[-1]
                    else:
                        close = df['Close'].dropna().iloc[-1]
                        vol = df['Volume'].dropna().iloc[-1]
                    sec_vol += float(close * vol)
                except Exception:
                    pass
            vols[sec] = sec_vol
        return vols
    except Exception:
        return {s: 0 for s in sectors_dict.keys()}

def add_indicators(df):
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']
    df['EMA_100'] = c.ewm(span=LONG_EMA, adjust=False, min_periods=LONG_EMA).mean()

    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    df['ATR_PCT'] = tr.rolling(14).mean() / c
    df['OBV'] = (np.sign(c.diff()).fillna(0) * v).cumsum()

    # RSI
    diff = c.diff()
    avg_g = diff.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_l = (-diff).clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_g / avg_l.replace(0, np.nan)
    df['RSI'] = (100 - 100 / (1 + rs)).where(avg_l != 0, 100.0)

    # MACD
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False, min_periods=26).mean()
    df['MACD_HIST'] = macd - macd.ewm(span=9, adjust=False).mean()

    # 볼린저 밴드 + 스퀴즈
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std(ddof=0)
    bb_up, bb_lo = mid + 2 * sd, mid - 2 * sd
    df['BB_MID'] = mid
    df['BBW'] = (bb_up - bb_lo) / mid
    atr20 = tr.rolling(20).mean()
    kc_mid = c.ewm(span=20, adjust=False).mean()
    df['SQUEEZE'] = (bb_up < kc_mid + 1.5 * atr20) & (bb_lo > kc_mid - 1.5 * atr20)

    # 후행성 피벗 (미래 참조 방지)
    K, W = PIV_K, 2 * PIV_K + 1
    is_ph = h.shift(K) == h.rolling(W).max()
    is_pl = l.shift(K) == l.rolling(W).min()
    is_ph = is_ph & ~(is_ph.shift(1, fill_value=False) & (h.shift(K) == h.shift(K + 1)))
    is_pl = is_pl & ~(is_pl.shift(1, fill_value=False) & (l.shift(K) == l.shift(K + 1)))
    df['PIV_H'] = h.shift(K).where(is_ph)
    df['PIV_H_RSI'] = df['RSI'].shift(K).where(is_ph)
    df['PIV_H_MACD'] = df['MACD_HIST'].shift(K).where(is_ph)
    df['PIV_L'] = l.shift(K).where(is_pl)
    df['PIV_L_RSI'] = df['RSI'].shift(K).where(is_pl)
    df['PIV_L_MACD'] = df['MACD_HIST'].shift(K).where(is_pl)
    return df

# ==========================================
# 시장 국면 엔진
# ==========================================
PHASE_ORDER = ["하락", "분배", "매집", "상승"]
PHASE_META = {
    "매집": {"icon": "🔵", "color": "#2196f3"},
    "상승": {"icon": "🟢", "color": "#26a69a"},
    "분배": {"icon": "🟣", "color": "#ab47bc"},
    "하락": {"icon": "🔴", "color": "#ef5350"},
}

def _val(df, col, idx=-1, default=None):
    try:
        v = df[col].iloc[idx]
        return default if pd.isna(v) else float(v)
    except Exception:
        return default

def _fmt_price(x):
    if x >= 1000: return f"{x:,.0f}"
    if x >= 1: return f"{x:,.2f}"
    return f"{x:.6g}"

def _naive(ts):
    ts = pd.to_datetime(ts)
    if getattr(ts.dt, 'tz', None) is not None:
        ts = ts.dt.tz_convert('UTC').dt.tz_localize(None)
    return ts

def _trail_true(arr):
    cnt = 0
    for x in arr[::-1]:
        if x: cnt += 1
        else: break
    return cnt

def _pivot_idx(d, col, within=RANGE_N):
    if col not in d.columns: return np.array([], dtype=int)
    arr = d[col].to_numpy(dtype=float)
    idx = np.flatnonzero(~np.isnan(arr))
    return idx[idx >= len(d) - within]

def _divergence(d, side, within=RANGE_N, max_age=40):
    col = 'PIV_L' if side == 'L' else 'PIV_H'
    if col + '_RSI' not in d.columns or col + '_MACD' not in d.columns:
        return {"rsi": False, "macd": False}
    idx = _pivot_idx(d, col, within)
    if len(idx) < 2: return {"rsi": False, "macd": False}
    a, b = idx[-2], idx[-1]
    px = d[col].to_numpy(dtype=float)
    rsi = d[col + '_RSI'].to_numpy(dtype=float)
    mac = d[col + '_MACD'].to_numpy(dtype=float)
    fresh = (len(d) - 1 - b) <= max_age
    
    if side == 'L':
        price_ok = px[b] < px[a]
        rsi_ok = not (np.isnan(rsi[a]) or np.isnan(rsi[b])) and rsi[b] > rsi[a] + 1.0
        mac_ok = not (np.isnan(mac[a]) or np.isnan(mac[b])) and mac[b] > mac[a]
    else:
        price_ok = px[b] > px[a]
        rsi_ok = not (np.isnan(rsi[a]) or np.isnan(rsi[b])) and rsi[b] < rsi[a] - 1.0
        mac_ok = not (np.isnan(mac[a]) or np.isnan(mac[b])) and mac[b] < mac[a]
    return {"rsi": bool(price_ok and rsi_ok and fresh), "macd": bool(price_ok and mac_ok and fresh)}

def _poc(w, bins=POC_BINS):
    lo, hi = float(w['low'].min()), float(w['high'].max())
    if not hi > lo: return None
    tp = ((w['high'] + w['low'] + w['close']) / 3).to_numpy(dtype=float)
    hist, edges = np.histogram(tp, bins=bins, range=(lo, hi), weights=w['volume'].to_numpy(dtype=float))
    if hist.sum() <= 0: return None
    k = int(hist.argmax())
    return float((edges[k] + edges[k + 1]) / 2)

def _poc_state(w, price, poc, tol):
    if poc is None: return "none"
    r = w.tail(10)
    if price >= poc:
        return "support" if float(r['low'].min()) <= poc + tol else "above"
    return "resist" if float(r['high'].max()) >= poc - tol else "below"

def htf_regime(hsub, label):
    n = len(hsub)
    if n < 60: return {"state": "n/a", "label": label, "span": 0, "approx": False}
    c = hsub['close']
    span = 200 if n >= 220 else max(50, n // 2)
    e = c.ewm(span=span, adjust=False).mean()
    atr_med = float(hsub['ATR_PCT'].dropna().median()) if 'ATR_PCT' in hsub.columns else 0.02
    atr_med = max(atr_med, 1e-4)
    slope = (float(e.iloc[-1]) / float(e.iloc[-11]) - 1) / atr_med
    price = float(c.iloc[-1])
    
    if price < e.iloc[-1] and slope < -0.5: state = "bear"
    elif price > e.iloc[-1] and slope > 0.5: state = "bull"
    else: state = "neutral"
    return {"state": state, "label": label, "span": span, "approx": span < 200}

def prepare_htf(htf, tf_label):
    up = UPPER_TF.get(tf_label)
    if htf is None or len(htf) == 0 or up is None: return None
    h = htf.reset_index(drop=True).copy()
    done = (_naive(h['time']) + HTF_DELTA[up]).to_numpy(dtype='datetime64[ns]')
    return {"df": h, "done": done, "label": up}

def mtf_at(pack, t64):
    if pack is None: return None
    k = int(np.searchsorted(pack['done'], t64, side='right'))
    return htf_regime(pack['df'].iloc[:k], pack['label'])

def analyze_market_cycle(df, mtf=None, macro=None, crypto_sent=None):
    d = df.tail(200)
    n = len(d)
    w = d.tail(RANGE_N)
    price = float(d['close'].iloc[-1])
    hi, lo = float(w['high'].max()), float(w['low'].min())
    rng = max(hi - lo, 1e-12)
    pos = (price - lo) / rng
    dd = 1 - price / hi if hi > 0 else 0.0

    ema20 = _val(d, 'EMA_20', default=price)
    ema50 = _val(d, 'EMA_50', default=price)
    ema50_prev = _val(d, 'EMA_50', -11, default=ema50)
    ema_long = _val(d, 'EMA_100', default=None)
    atr_s = d['ATR_PCT'].dropna() if 'ATR_PCT' in d.columns else pd.Series(dtype=float)
    atr_med = max(float(atr_s.median()) if len(atr_s) else 0.02, 1e-4)
    atr_now = float(atr_s.iloc[-1]) if len(atr_s) else atr_med
    slope = (ema50 / ema50_prev - 1) / atr_med if ema50_prev else 0.0
    comp = atr_now / atr_med < 0.85

    up = price > ema20 > ema50
    down = price < ema20 < ema50
    flat = abs(slope) < 1.0
    crack = price < ema20 and ema20 > ema50
    rebound = ema20 < price < ema50

    sq_arr = d['SQUEEZE'].to_numpy(dtype=bool) if 'SQUEEZE' in d.columns else np.zeros(n, dtype=bool)
    sq_on = bool(sq_arr[-1])
    sq_cnt = _trail_true(sq_arr)
    sq_release = (not sq_on) and bool(sq_arr[-4:-1].any())
    bb_mid = _val(d, 'BB_MID', default=price)
    dir_up = price > bb_mid
    bbw_now = _val(d, 'BBW', default=None)
    bbw_hist = w['BBW'].dropna() if 'BBW' in w.columns else pd.Series(dtype=float)
    bbw_pct = float((bbw_hist <= bbw_now).mean()) if (bbw_now is not None and len(bbw_hist)) else 1.0

    ih, il = _pivot_idx(d, 'PIV_H'), _pivot_idx(d, 'PIV_L')
    enough = len(ih) >= 2 and len(il) >= 2
    if enough:
        ph, pl = d['PIV_H'].to_numpy(dtype=float), d['PIV_L'].to_numpy(dtype=float)
        HH, LH = ph[ih[-1]] > ph[ih[-2]], ph[ih[-1]] < ph[ih[-2]]
        HL, LL = pl[il[-1]] > pl[il[-2]], pl[il[-1]] < pl[il[-2]]
    else:
        HH = LH = HL = LL = False

    vol = d['volume']
    vol_base = float(vol.tail(60).median())
    vol_base = vol_base if vol_base > 0 else 1.0
    vr = float(vol.tail(10).mean()) / vol_base
    dry = vr < 0.85
    r20 = d.tail(20)
    up_v = r20.loc[r20['close'] >= r20['open'], 'volume'].mean()
    dn_v = r20.loc[r20['close'] < r20['open'], 'volume'].mean()
    up_v = 0.0 if pd.isna(up_v) else float(up_v)
    dn_v = 0.0 if pd.isna(dn_v) else float(dn_v)
    ud = up_v / dn_v if dn_v > 0 else (2.0 if up_v > 0 else 1.0)

    r10 = d.tail(10)
    bar_rng = (r10['high'] - r10['low']).replace(0, np.nan)
    low_pos = (r10['low'] - lo) / rng
    high_pos = (r10['high'] - lo) / rng
    upper_wick = (r10['high'] - r10[['open', 'close']].max(axis=1)) / bar_rng
    sell_climax = bool(((r10['volume'] > 2.5 * vol_base) & (r10['close'] < r10['open']) & (low_pos < 0.25)).any())
    dist_climax = bool(((r10['volume'] > 2.0 * vol_base) & (upper_wick > 0.4) & (high_pos > 0.75)).any())
    wick_cnt = int((upper_wick > 0.4).sum())
    wick_cluster = wick_cnt >= 3 and pos >= 0.6

    pr20 = price / float(d['close'].iloc[-21]) - 1 if n > 25 else 0.0
    bull_obv = bear_obv = False
    if n > 25 and 'OBV' in d.columns:
        obv_chg = float(d['OBV'].iloc[-1] - d['OBV'].iloc[-21])
        big = abs(obv_chg) > vol_base * 2
        bull_obv = big and pos <= 0.45 and pr20 <= 0.01 and obv_chg > 0
        bear_obv = big and pos >= 0.6 and pr20 >= -0.01 and obv_chg < 0
    vol_exhaust_up = pos >= 0.7 and pr20 > 0.03 and vr < 0.85

    dv_l, dv_h = _divergence(d, 'L'), _divergence(d, 'H')
    bull_rsi, bull_macd = dv_l['rsi'] and pos <= 0.5, dv_l['macd'] and pos <= 0.5
    bear_rsi, bear_macd = dv_h['rsi'] and pos >= 0.5, dv_h['macd'] and pos >= 0.5

    poc = _poc(w)
    tol = price * max(atr_now, 0.005)
    poc_state = _poc_state(w, price, poc, tol)

    is_crashed = ema_long is not None and price < ema_long and dd >= CRASH_DD
    is_consolidating = sq_on or (comp and bbw_pct <= 0.35)
    triple = {"crashed": bool(is_crashed), "consolidating": bool(is_consolidating), "hl": bool(HL)}

    # ---------- 1. 추세 ----------
    t = {
        "하락": 25 if (down and slope < -1.0) else 15 if down else 8 if (price < ema50 and slope < 0) else 0,
        "매집": 25 if (flat and not up) else 15 if rebound else 0,
        "상승": 25 if (up and slope >= 1.0) else 15 if up else 8 if price > ema20 else 0,
        "분배": 12 if crack else 12 if (flat and price >= ema50) else 0,
    }

    # ---------- 2. 위치 및 POC ----------
    pos_pts = {
        "하락": 17 if pos <= 0.5 else 8 if pos <= 0.7 else 0,
        "매집": 17 if pos <= 0.3 else 8 if pos <= 0.45 else 0,
        "상승": 17 if pos >= 0.45 else 7 if pos >= 0.3 else 0,
        "분배": 17 if pos >= 0.7 else 8 if pos >= 0.55 else 0,
    }
    poc_pts = {
        "하락": {"resist": 8, "below": 6}.get(poc_state, 0),
        "매집": {"support": 8, "above": 5}.get(poc_state, 0),
        "상승": {"support": 8, "above": 6}.get(poc_state, 0),
        "분배": ({"resist": 8, "below": 5}.get(poc_state, 0) if pos >= 0.5 else 0),
    }
    p = {k: pos_pts[k] + poc_pts[k] for k in PHASE_ORDER}

    # ---------- 3. 파동 ----------
    if enough:
        wv = {
            "하락": 25 if (LH and LL) else 12 if LL else 0,
            "매집": 25 if (HL and not HH) else 10 if (not HL and not LL) else 0,
            "상승": 25 if (HH and HL) else 12 if HL else 0,
            "분배": 25 if (LH and not LL) else 10 if (HH and not HL) else 0,
        }
    else:
        wv = {"하락": 5, "매집": 10, "상승": 5, "분배": 10}

    # ---------- 4. 수급 ----------
    exhaust = sell_climax or bull_obv or bull_rsi or bull_macd
    v_up = 25 if (ud > 1.2 and vr >= 0.9) else 12 if ud > 1.0 else 0
    if vol_exhaust_up:
        v_up = min(v_up, 5)
    v = {
        "하락": min(25, (25 if (ud < 0.8 and not exhaust) else 10 if ud < 1.0 else 0)
                    + (6 if (sq_release and not dir_up) else 0)),
        "매집": min(25, (12 if sell_climax else 0) + (8 if bull_obv else 0) + (8 if bull_rsi else 0)
                    + (8 if bull_macd else 0) + (4 if (bull_rsi and bull_macd) else 0)
                    + (6 if (sq_on and sq_cnt >= 3) else 0) + (4 if (sq_release and dir_up) else 0)
                    + (4 if (dry or comp) else 0) + (3 if ud >= 1.0 else 0)),
        "상승": min(25, v_up + (6 if (sq_release and dir_up) else 0)),
        "분배": min(25, (12 if dist_climax else 0) + (8 if wick_cluster else 0) + (8 if bear_obv else 0)
                    + (8 if bear_rsi else 0) + (8 if bear_macd else 0) + (4 if (bear_rsi and bear_macd) else 0)
                    + (10 if vol_exhaust_up else 0) + (8 if (ud < 1.0 and pos >= 0.6) else 0)),
    }

    # ---------- 종합 및 매크로 보정 ----------
    totals = {k: t[k] + p[k] + wv[k] + v[k] for k in PHASE_ORDER}
    adjust = []

    bearflag = down and slope < BEARFLAG_SLOPE and (comp or sq_on or dry)
    if bearflag:
        totals["매집"] = max(0, totals["매집"] - BEARFLAG_PENALTY)
        adjust.append(f"급락 중 베어플래그 횡보 의심 (-{BEARFLAG_PENALTY})")
        
    if not all(triple.values()) and totals["매집"] > ACC_CAP_NO_TRIPLE:
        totals["매집"] = ACC_CAP_NO_TRIPLE
        adjust.append(f"진바닥 3박자 미충족 → 매집 상한 {ACC_CAP_NO_TRIPLE}")
        
    veto = bool(mtf and mtf.get("state") == "bear" and MTF_VETO_RATIO > 0)
    if veto:
        totals["매집"] = int(totals["매집"] * (1 - MTF_VETO_RATIO))
        adjust.append(f"상위 프레임({mtf['label']}) 하락 → 매집 " + ("무효화" if MTF_VETO_RATIO >= 1 else "감점"))

    # 거시 경제 지표(Macro) 보정
    if macro:
        if macro.get('fng', 50) <= 20:
            totals['매집'] += 5
            adjust.append("극단적 공포(F&G) 매수 가점")
        elif macro.get('fng', 50) >= 80:
            totals['분배'] += 5
            adjust.append("극단적 탐욕(F&G) 분배 가점")
            
        if macro.get('vix_panic', False):
            totals['매집'] += 10
            adjust.append("VIX 공포 지수 극단(패닉 셀) 탐색 가점")
            
        if macro.get('dxy_trend', '') == 'Bull':
            totals['매집'] = max(0, totals['매집'] - 5)
            totals['상승'] = max(0, totals['상승'] - 5)
            adjust.append("강달러 장세(위험자산 이탈) 감점")

    # 암호화폐 전용 심리/과열 지표 보정
    if crypto_sent:
        kimp = crypto_sent.get('kimp', 0.0)
        fund_rate = crypto_sent.get('funding_rate', 0.0)
        
        if kimp >= 8.0:
            totals['분배'] += 10
            totals['매집'] = max(0, totals['매집'] - 10)
            adjust.append("김프 과열(로컬 FOMO) 매수 감점 및 분배 가점")
        elif kimp < 0.0:
            totals['매집'] += 5
            adjust.append("역프(대중 투매 종료) 바닥 가점")
            
        if fund_rate < -0.05:
            totals['매집'] += 5
            adjust.append("펀딩비 극단적 숏(스퀴즈 기대) 가점")
        elif fund_rate > 0.05:
            totals['분배'] += 5
            adjust.append("펀딩비 극단적 롱(청산 경계) 분배 가점")

    totals = {k: int(min(100, max(0, vv))) for k, vv in totals.items()}
    phase = max(PHASE_ORDER, key=lambda k: totals[k])
    top = totals[phase]
    late = (phase == "하락" and not veto and not bearflag
            and (sell_climax or bull_obv or bull_rsi or bull_macd or totals["매집"] >= top - 10))

    # ---------- 팩트체크 문구 ----------
    if up: f_ema = "20·50 이평선 위 정배열이며 중기 이평선이 " + ("가파르게 " if slope >= 1.0 else "완만하게 ") + "우상향 중입니다."
    elif down: f_ema = "20·50 이평선 아래 역배열로 " + ("가파르게 하락하며 " if slope < -1.0 else "") + "하방 압력이 이어지고 있습니다."
    elif crack: f_ema = "상승 추세 중 20 이평선이 깨졌습니다. 거래량·윗꼬리 확인이 필요합니다."
    elif flat: f_ema = "이평선이 평평하게 수렴하며 방향성을 탐색하는 횡보 구간입니다."
    elif rebound: f_ema = "50 이평선 아래이지만 20 이평선을 회복하며 반등을 시도 중입니다."
    else: f_ema = "20 이평선 아래로 밀려 있으나 추세는 뚜렷하지 않습니다."

    pct = f"{pos * 100:.0f}%"
    if pos <= 0.2: f_pos = f"최근 {len(w)}봉 범위 하단 {pct} 지점으로 바닥권입니다."
    elif pos <= 0.45: f_pos = f"최근 {len(w)}봉 범위 하단 {pct} 지점으로 저가 구간입니다."
    elif pos < 0.7: f_pos = f"최근 {len(w)}봉 범위 {pct} 지점으로 중간대입니다."
    else: f_pos = f"최근 {len(w)}봉 범위 {pct} 지점으로 고점권입니다."

    if not enough: f_wave = "아직 확정된 고점/저점 파동이 부족해 방향성을 탐색 중입니다."
    elif HH and HL: f_wave = "고점과 저점을 모두 높이는 상승 파동(HH/HL)이 살아 있습니다."
    elif LH and LL: f_wave = "고점과 저점이 모두 낮아지는 하락 파동(LH/LL)이 이어지고 있습니다."
    elif HL and not HH: f_wave = "고점은 못 넘었지만 저점을 높이고 있습니다(저점 상승). 매집 가능성을 시사합니다."
    elif LH and not LL: f_wave = "저점은 지켰지만 고점이 낮아지고 있습니다(고점 하락). 분배 가능성을 시사합니다."
    elif HH and not HL: f_wave = "고점은 높였지만 저점이 따라 오르지 못해 상승 힘이 약해지는 모습입니다."
    else: f_wave = "고점과 저점이 뚜렷한 방향 없이 혼조를 보이고 있습니다."
    
    mk = lambda ok: "✔" if ok else "✘"
    f_wave += (f" [진바닥 3박자] 장기선 이탈 {mk(triple['crashed'])} · 변동성 수축 {mk(triple['consolidating'])}"
               f" · 저점 상승(HL) {mk(triple['hl'])}")

    if sell_climax: f_vol = "저점권에서 투매성 대량 거래량(셀링 클라이맥스)이 나왔습니다. 매도 물량이 소진되는 신호일 수 있습니다."
    elif bull_obv: f_vol = "가격은 눌려 있는데 거래량 흐름(OBV)은 오르는 강세 다이버전스입니다. 조용한 매집 가능성이 있습니다."
    elif dist_climax: f_vol = "고점권에서 대량 거래량에도 윗꼬리가 길어지며 물량이 넘어가는 흔적(분배)이 보입니다."
    elif bear_obv: f_vol = "가격은 버티는데 거래량 흐름(OBV)은 꺾이는 약세 다이버전스입니다."
    elif ud > 1.2: f_vol = "상승봉 거래량이 하락봉보다 우세해 매수 수급이 강합니다."
    elif ud < 0.8: f_vol = "하락봉 거래량이 우세해 매도 압력이 큽니다."
    else: f_vol = "상승·하락 거래량이 비슷하게 균형을 이루고 있습니다."

    return {
        "phase": phase, "score": top, "totals": totals, "late": late, "hl": bool(HL),
        "weak": top < 50, "bars": n, "triple": triple, "veto": veto, "bearflag": bool(bearflag),
        "adjust": adjust, "mtf": mtf,
        "facts": {"EMA": f_ema, "POS": f_pos, "WAVE": f_wave, "VOL": f_vol},
    }

@st.cache_data(ttl=300)
def cached_phase_history(df, htf_pack=None, macro=None, crypto_sent=None, start=60):
    tn = _naive(df['time']).to_numpy(dtype='datetime64[ns]')
    labels = []
    for i in range(start, len(df)):
        labels.append(analyze_market_cycle(df.iloc[:i + 1], mtf_at(htf_pack, tn[i]), macro, crypto_sent)['phase'])
    runs = []
    for k, ph in enumerate(labels):
        if runs and runs[-1][0] == ph: runs[-1][2] = k
        else: runs.append([ph, k, k])
    merged = []
    for r in runs:
        if merged and (r[2] - r[1] + 1 < 3 or merged[-1][0] == r[0]): merged[-1][2] = r[2]
        else: merged.append(r)
    return [(str(df['time'].iloc[start + r[1]]), str(df['time'].iloc[start + r[2]]), r[0]) for r in merged]

def calculate_fibonacci_levels(df, is_commodity=False, window=RANGE_N):
    recent = df.tail(window)
    hi = float(recent['high'].max())
    lo = float(recent['low'].min())
    diff = max(hi - lo, 1e-12)
    ratios = (0.236, 0.382, 0.5) if is_commodity else (0.382, 0.618, 0.786)
    buy = [(f"{i + 1}차({r})", hi - diff * r) for i, r in enumerate(ratios)]
    sell = [("1차(0.764)", hi - diff * 0.236), ("2차(전고점)", hi), ("3차(확장 1.272)", hi + diff * 0.272)]
    return {'high': hi, 'low': lo, 'fib_1': buy[0][1], 'fib_2': buy[1][1], 'fib_3': buy[2][1], 'buy': buy, 'sell': sell}

# ==========================================
# 화면 구성 요소
# ==========================================
def render_phase_legend():
    with st.expander("📌 시장 국면 가이드 (매집 → 상승 → 분배 → 하락)", expanded=True):
        st.markdown("- 🔵 **매집**: `[바닥권 횡보 · 거래량 감소 · 저점 상승]` - 3분할 매수 구간")
        st.markdown("- 🟢 **상승**: `[정배열 · 고점/저점 상승]` - 보유 유지, 눌림목만 매수 (추격 주의)")
        st.markdown("- 🟣 **분배**: `[고점권 거래량 폭증 · 윗꼬리 · 고점 하락]` - 신규 매수 금지, 분할 매도")
        st.markdown("- 🔴 **하락**: `[역배열 · 거래량 동반 하락]` - 매수 보류 (떨어지는 칼날)")
        st.markdown("- 🔎 **진바닥 3박자**: `[장기선 이탈 + 변동성 수축 + 저점 상승]` 동시 충족 시에만 매집 고득점")

def render_fact_box(facts):
    with st.container(border=True):
        st.markdown("##### 📝 4대 팩트체크")
        st.markdown(f"- **[추세 EMA]**: {facts['EMA']}")
        st.markdown(f"- **[위치 POS]**: {facts['POS']}")
        st.markdown(f"- **[파동 WAVE]**: {facts['WAVE']}")
        st.markdown(f"- **[수급 VOL]**: {facts['VOL']}")

def render_trade_guide(info, fibs, curr, rate):
    def fmt(v): return f"₩{v * rate:,.0f}"
    buy_line = " | ".join(f"{lab}: {fmt(p)}" + (" (도달)" if curr <= p else "") for lab, p in fibs['buy'])
    sell_line = " | ".join(f"{lab}: {fmt(p)}" + (" (도달)" if curr >= p else "") for lab, p in fibs['sell'])
    ph = info['phase']

    if ph == "하락":
        if info['veto']: st.error("📉 하락 국면이며 상위 프레임도 하락 추세입니다. 하위 프레임의 반등·매집 신호는 무효로 처리하니 매수를 보류하세요.")
        elif info['late']:
            st.warning("🟠 하락 말기 신호가 보입니다. 바닥 확정 전이므로 3차 타점 부근에서만 소액 탐색을 고려하세요.")
            lab, p = fibs['buy'][2]
            st.markdown(f"**[바닥 탐색 타점]**\n- {lab}: {fmt(p)} | 무효화 기준(전저점): {fmt(fibs['low'])} 이탈 시 재평가")
        else: st.error("📉 하락 국면입니다. 현금 비중을 유지하고 매수를 보류하세요.")
    elif ph == "매집":
        tr = info['triple']
        size = "진바닥 3박자 충족 · 정상 진입" if all(tr.values()) else "3박자 미충족 · 가짜 바닥 주의, 관망 우선"
        st.markdown(f"**[3분할 매수 타점]** (매집 구간 · {size})\n- {buy_line}\n- 무효화 기준(전저점): {fmt(fibs['low'])} 이탈 시 매집 가설 폐기")
    elif ph == "상승":
        st.markdown(f"**[눌림목 매수 타점]** (추격 매수 주의)\n- {buy_line}\n**[3분할 매도 타점]** (분할 매도 준비)\n- {sell_line}")
    else:  # 분배
        st.error("🟣 분배 국면입니다. 신규 매수를 금지하고 보유분은 분할 매도를 검토하세요.")
        st.markdown(f"**[3분할 매도 타점]**\n- {sell_line}")
    st.caption("※ 규칙 기반 참고 지표이며, 투자 판단과 책임은 본인에게 있습니다.")

def build_chart(df, fibs, rate=1.0, history=None):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    if history:
        for x0, x1, ph in history:
            fig.add_vrect(x0=x0, x1=x1, fillcolor=PHASE_META[ph]['color'], opacity=0.12, layer="below", line_width=0, row=1, col=1)
    fig.add_trace(go.Candlestick(x=df['time'], open=df['open'] * rate, high=df['high'] * rate, low=df['low'] * rate, close=df['close'] * rate, name='Price'), row=1, col=1)
    if 'EMA_20' in df.columns: fig.add_trace(go.Scatter(x=df['time'], y=df['EMA_20'] * rate, line=dict(color='orange', width=1), name='EMA 20'), row=1, col=1)
    fig.add_hline(y=fibs['fib_1'] * rate, line_dash="dot", line_color="orange", row=1, col=1)
    fig.add_hline(y=fibs['fib_3'] * rate, line_dash="dot", line_color="red", row=1, col=1)
    colors = ['#26a69a' if r['close'] >= r['open'] else '#ef5350' for _, r in df.iterrows()]
    fig.add_trace(go.Bar(x=df['time'], y=df['volume'], marker_color=colors, name='Volume'), row=2, col=1)
    fig.update_layout(height=600, margin=dict(l=0, r=0, t=0, b=0), xaxis_rangeslider_visible=False)
    return fig

def render_dashboard(df, c1, c2, show_hist, rate_text=1.0, rate_chart=1.0, is_commodity=False,
                     captions=(), tf_label=None, htf=None, macro=None, crypto_sent=None):
    curr = float(df['close'].iloc[-1])
    fibs = calculate_fibonacci_levels(df, is_commodity=is_commodity)
    pack = prepare_htf(htf, tf_label)
    t_last = _naive(df['time']).iloc[-1].to_datetime64().astype('datetime64[ns]')
    info = analyze_market_cycle(df, mtf_at(pack, t_last), macro, crypto_sent)
    meta = PHASE_META[info['phase']]

    with c1:
        st.subheader(f"현재 국면: {meta['icon']} {info['phase']} ({info['score']}/100점 일치)")
        st.caption("국면 점수 — " + " | ".join(f"{k} {info['totals'][k]}" for k in ["매집", "상승", "분배", "하락"]))
        if info['adjust']:
            st.caption("🚨 로직 보정 내역: " + " · ".join(info['adjust']))
        for cap in captions:
            st.caption(cap)
        if info['bars'] < 80:
            st.caption("⚠️ 데이터가 짧아 국면 판별 신뢰도가 낮습니다.")
        render_fact_box(info['facts'])
        render_phase_legend()
        st.divider()
        render_trade_guide(info, fibs, curr, rate_text)

    with c2:
        history = cached_phase_history(df, pack, macro, crypto_sent) if show_hist else None
        st.plotly_chart(build_chart(df, fibs, rate_chart, history), use_container_width=True)

def load_yf_data(ticker, tf_label):
    tf_map = {"15분": "15m", "1시간": "60m", "4시간": "60m", "1일": "1d", "주봉": "1wk", "월봉": "1mo"}
    yf_tf = tf_map[tf_label]
    period = "60d" if yf_tf in ["15m", "60m"] else ("2y" if yf_tf == "1d" else "10y")
    try:
        df = yf.download(ticker, period=period, interval=yf_tf, progress=False)
        if df.empty: return pd.DataFrame(), "yfinance에서 빈 데이터가 반환되었습니다."
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df.columns = [str(c).lower() for c in df.columns]
        df = df.rename(columns={'date': 'time', 'datetime': 'time'})
        if tf_label == "4시간":
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time').resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna().reset_index()
        df = add_indicators(df)
        return df, None
    except Exception as e:
        return pd.DataFrame(), f"{type(e).__name__}: {e}"

# ==========================================
# 사이드바 네비게이션 및 거시 데이터 로드
# ==========================================
st.sidebar.title("🧭 PRO 퀀트 네비게이션")
tabs = ["🪙 암호화폐 (Crypto)", "📈 국내 주식 (KOSPI)", "🦅 미국 주식 (US Stock)", "📊 미국 ETF (US ETF)", "🥇 금·은 원자재 (Commodities)"]
market_choice = st.sidebar.radio("트레이딩 룸 선택", tabs)
TF_OPTIONS = ["15분", "1시간", "4시간", "1일", "주봉", "월봉"]

# 글로벌 매크로 지표 패치
macro_data = get_macro_indicators()
st.sidebar.divider()
st.sidebar.markdown(f"**🌐 매크로 동향**\n"
                    f"- F&G 지수: {macro_data['fng']} ({macro_data['fng_label']})\n"
                    f"- 달러 추세: {macro_data['dxy_trend']}\n"
                    f"- VIX(공포): {macro_data['vix']:.2f}")

# ==========================================
# 1. 🪙 암호화폐 마켓
# ==========================================
if market_choice == "🪙 암호화폐 (Crypto)":
    st.title("🪙 암호화폐 현물 퀀트 봇")

    btc_d = get_btc_dominance()
    crypto_sent = get_crypto_sentiment(KRW_RATE)
    
    c_m1, c_m2, c_m3 = st.columns(3)
    c_m1.metric(label="BTC 도미넌스", value=f"{btc_d:.2f}%")
    c_m2.metric(label="김치 프리미엄", value=f"{crypto_sent['kimp']:.2f}%")
    c_m3.metric(label="선물 펀딩비", value=f"{crypto_sent['funding_rate']:.4f}%")
    
    if btc_d > 55.0: st.warning("⚠️ **[수급 경고]** 도미넌스가 매우 높습니다. 알트코인 수급 이탈(흡성대법)에 주의하세요.")
    elif btc_d < 45.0: st.success("🔥 **[알트 시즌]** 도미넌스가 낮아 알트코인으로 자금이 순환될 확률이 높습니다.")

    crypto_tabs = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "직접 검색"]
    selected_crypto = st.radio("관심 심볼", crypto_tabs, horizontal=True)
    target_symbol = st.text_input("심볼 커스텀 입력", value="DOGE/USDT").upper() if selected_crypto == "직접 검색" else selected_crypto.split(" ")[0]

    @st.cache_data(ttl=60)
    def load_crypto(symbol, tf_label):
        tf_map = {"15분": "15m", "1시간": "1h", "4시간": "4h", "1일": "1d", "주봉": "1w", "월봉": "1M"}
        tf = tf_map[tf_label]
        errors = []
        for name in ["binance", "okx", "bybit", "kraken", "coinbase"]:
            try:
                ex = getattr(ccxt, name)()
                bars = ex.fetch_ohlcv(symbol, timeframe=tf, limit=300)
                if not bars: continue
                df = pd.DataFrame(bars, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
                df['time'] = pd.to_datetime(df['time'], unit='ms')
                df = add_indicators(df)
                return df, name, None
            except Exception as e:
                errors.append(f"{name}: {type(e).__name__}")
        return pd.DataFrame(), None, " / ".join(errors)

    c1, c2 = st.columns([1.3, 2])
    with c2:
        selected_tf = st.radio("프레임 선택", TF_OPTIONS, index=3, horizontal=True, key="c_tf")
        show_hist = st.checkbox("과거 국면 표시 (검증용)", key="c_hist")

    df, source, err = load_crypto(target_symbol, selected_tf)
    if df.empty:
        st.error(f"데이터 로드 실패 → {err}")
    else:
        up_tf = UPPER_TF.get(selected_tf)
        htf_df = load_crypto(target_symbol, up_tf)[0] if up_tf else None
        render_dashboard(df, c1, c2, show_hist, rate_text=KRW_RATE, rate_chart=1.0,
                         captions=(f"데이터 소스: {source}",), tf_label=selected_tf, htf=htf_df, 
                         macro=macro_data, crypto_sent=crypto_sent)

# ==========================================
# 2. 📈 국내 주식 (KOSPI) 마켓
# ==========================================
elif market_choice == "📈 국내 주식 (KOSPI)":
    st.title("📈 KOSPI 주도주 현물 매수 봇")
    kospi_sectors = {
        "반도체": ["005930.KS (삼성전자)", "000660.KS (SK하이닉스)", "042700.KS (한미반도체)", "000990.KS (DB하이텍)", "058470.KQ (리노공업)"],
        "2차전지": ["373220.KS (LG엔솔)", "005490.KS (POSCO홀딩스)", "006400.KS (삼성SDI)", "247540.KQ (에코프로비엠)", "051910.KS (LG화학)"],
        "바이오": ["207940.KS (삼성바이오)", "068270.KS (셀트리온)", "000100.KS (유한양행)", "326030.KS (SK바이오팜)", "128940.KS (한미약품)"],
        "인터넷/게임": ["035420.KS (NAVER)", "035720.KS (카카오)", "259960.KS (크래프톤)", "036570.KS (엔씨소프트)", "251270.KS (넷마블)"],
        "자동차": ["005380.KS (현대차)", "000270.KS (기아)", "012330.KS (현대모비스)", "204320.KS (HL만도)", "018880.KS (한온시스템)"],
        "항공우주/방산": ["012450.KS (한화에어로스페이스)", "047810.KS (한국항공우주)", "079550.KS (LIG넥스원)", "064350.KS (현대로템)", "272210.KS (한화시스템)"]
    }
    vols = get_sector_volumes(kospi_sectors)
    if sum(vols.values()) > 0:
        fig_pie = go.Figure(data=[go.Pie(labels=list(vols.keys()), values=list(vols.values()), hole=.4)])
        fig_pie.update_layout(title="국내 시장 섹터별 자금 유입(거래대금) 현황", height=300, margin=dict(t=30, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    sel_sec = st.selectbox("📂 시장 주도 섹터", list(kospi_sectors.keys()))
    sel_tick = st.radio(f"🏆 {sel_sec} 거래대금 상위 Top 5", kospi_sectors[sel_sec], horizontal=True).split(" ")[0]

    @st.cache_data(ttl=300)
    def load_stock(ticker, tf_label): return load_yf_data(ticker, tf_label)

    c1, c2 = st.columns([1.3, 2])
    with c2:
        selected_tf = st.radio("프레임 선택", TF_OPTIONS, index=3, horizontal=True, key="s_tf")
        show_hist = st.checkbox("과거 국면 표시 (검증용)", key="s_hist")

    df, err = load_stock(sel_tick, selected_tf)
    if df.empty:
        st.error(f"데이터 로드 실패 → {err}")
    else:
        up_tf = UPPER_TF.get(selected_tf)
        htf_df = load_stock(sel_tick, up_tf)[0] if up_tf else None
        render_dashboard(df, c1, c2, show_hist, rate_text=1.0, rate_chart=1.0, tf_label=selected_tf, htf=htf_df, macro=macro_data)

# ==========================================
# 3. 🦅 미국 주식 (US Stock)
# ==========================================
elif market_choice == "🦅 미국 주식 (US Stock)":
    st.title("🦅 미국 주식 주도주 퀀트 봇")
    us_sectors = {
        "테크/소프트웨어": ["AAPL (애플)", "MSFT (마이크로소프트)", "GOOGL (구글)", "META (메타)", "AMZN (아마존)"],
        "반도체/AI": ["NVDA (엔비디아)", "TSM (TSMC)", "AVGO (브로드컴)", "AMD (AMD)", "QCOM (퀄컴)"],
        "금융/핀테크": ["JPM (JP모건)", "V (비자)", "MA (마스터카드)", "BAC (뱅크오브아메리카)", "WFC (웰스파고)"],
        "헬스케어": ["LLY (일라이릴리)", "UNH (유나이티드헬스)", "JNJ (존슨앤존슨)", "ABBV (애브비)", "MRK (머크)"],
        "항공우주/방산": ["LMT (록히드마틴)", "RTX (RTX)", "BA (보잉)", "GD (제너럴다이내믹스)", "NOC (노스롭그루먼)"]
    }
    vols = get_sector_volumes(us_sectors)
    if sum(vols.values()) > 0:
        fig_pie = go.Figure(data=[go.Pie(labels=list(vols.keys()), values=list(vols.values()), hole=.4)])
        fig_pie.update_layout(title="미국 시장 섹터별 자금 유입(거래대금) 현황", height=300, margin=dict(t=30, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    sel_sec = st.selectbox("📂 미국 주도 섹터", list(us_sectors.keys()))
    sel_tick = st.radio(f"🏆 {sel_sec} 거래대금 상위 Top 5", us_sectors[sel_sec], horizontal=True).split(" ")[0]

    @st.cache_data(ttl=300)
    def load_us_stock(ticker, tf_label): return load_yf_data(ticker, tf_label)

    c1, c2 = st.columns([1.3, 2])
    with c2:
        selected_tf = st.radio("프레임 선택", TF_OPTIONS, index=3, horizontal=True, key="us_tf")
        show_hist = st.checkbox("과거 국면 표시 (검증용)", key="us_hist")

    df, err = load_us_stock(sel_tick, selected_tf)
    if df.empty:
        st.error(f"데이터 로드 실패 → {err}")
    else:
        up_tf = UPPER_TF.get(selected_tf)
        htf_df = load_us_stock(sel_tick, up_tf)[0] if up_tf else None
        render_dashboard(df, c1, c2, show_hist, rate_text=KRW_RATE, rate_chart=KRW_RATE,
                         captions=(f"환율 적용: 1 USD = {KRW_RATE:,.0f} KRW",), tf_label=selected_tf, htf=htf_df, macro=macro_data)

# ==========================================
# 4. 📊 미국 ETF (US ETF)
# ==========================================
elif market_choice == "📊 미국 ETF (US ETF)":
    st.title("📊 미국 인기 ETF 퀀트 봇")
    
    us_etfs = {
        "지수 추종 (Index)": ["SPY (S&P 500)", "QQQ (나스닥 100)", "DIA (다우존스)", "IWM (러셀 2000)", "VOO (S&P 500)"],
        "테크/반도체 (Tech)": ["SOXX (미국 반도체)", "XLK (기술주)", "SMH (글로벌 반도체)", "VGT (정보기술)", "QQQM (나스닥 미니)"],
        "레버리지 3X (Leverage)": ["SOXL (반도체 3X)", "TQQQ (나스닥 3X)", "UPRO (S&P 3X)", "FAS (금융 3X)", "TECL (기술주 3X)"],
        "인버스 3X (Inverse)": ["SOXS (반도체 -3X)", "SQQQ (나스닥 -3X)", "SPXU (S&P -3X)", "FAZ (금융 -3X)", "TECS (기술주 -3X)"],
        "고배당/인컴 (Dividend)": ["SCHD (미국 배당성장)", "JEPI (커버드콜)", "VYM (고배당)", "VNQ (부동산 리츠)", "JEPQ (나스닥 커버드콜)"]
    }
    
    vols = get_sector_volumes(us_etfs)
    if sum(vols.values()) > 0:
        fig_pie = go.Figure(data=[go.Pie(labels=list(vols.keys()), values=list(vols.values()), hole=.4)])
        fig_pie.update_layout(title="미국 ETF 테마별 자금 유입(거래대금) 현황", height=300, margin=dict(t=30, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    sel_sec = st.selectbox("📂 ETF 테마 선택", list(us_etfs.keys()))
    sel_tick = st.radio(f"🏆 {sel_sec} 주요 ETF Top 5", us_etfs[sel_sec], horizontal=True).split(" ")[0]

    @st.cache_data(ttl=300)
    def load_us_etf(ticker, tf_label): return load_yf_data(ticker, tf_label)

    c1, c2 = st.columns([1.3, 2])
    with c2:
        selected_tf = st.radio("프레임 선택", TF_OPTIONS, index=3, horizontal=True, key="etf_tf")
        show_hist = st.checkbox("과거 국면 표시 (검증용)", key="etf_hist")

    df, err = load_us_etf(sel_tick, selected_tf)
    if df.empty:
        st.error(f"데이터 로드 실패 → {err}")
    else:
        up_tf = UPPER_TF.get(selected_tf)
        htf_df = load_us_etf(sel_tick, up_tf)[0] if up_tf else None
        render_dashboard(df, c1, c2, show_hist, rate_text=KRW_RATE, rate_chart=KRW_RATE,
                         captions=(f"환율 적용: 1 USD = {KRW_RATE:,.0f} KRW",), tf_label=selected_tf, htf=htf_df, macro=macro_data)

# ==========================================
# 5. 🥇 금·은 원자재 (Commodities)
# ==========================================
elif market_choice == "🥇 금·은 원자재 (Commodities)":
    st.title("🥇 금/은 안전자산 현물 봇")
    com_tick = st.radio("원자재 선택", ["GC=F (금 Gold)", "SI=F (은 Silver)"], horizontal=True).split(" ")[0]

    @st.cache_data(ttl=300)
    def load_commodity(ticker, tf_label): return load_yf_data(ticker, tf_label)

    c1, c2 = st.columns([1.3, 2])
    with c2:
        selected_tf = st.radio("프레임 선택", TF_OPTIONS, index=3, horizontal=True, key="com_tf")
        show_hist = st.checkbox("과거 국면 표시 (검증용)", key="com_hist")

    df, err = load_commodity(com_tick, selected_tf)
    if df.empty:
        st.error(f"데이터 로드 실패 → {err}")
    else:
        up_tf = UPPER_TF.get(selected_tf)
        htf_df = load_commodity(com_tick, up_tf)[0] if up_tf else None
        render_dashboard(df, c1, c2, show_hist, rate_text=KRW_RATE, rate_chart=KRW_RATE, is_commodity=True,
                         captions=(f"환율 적용: 1 USD = {KRW_RATE:,.0f} KRW (온스당)",), tf_label=selected_tf, htf=htf_df, macro=macro_data)
