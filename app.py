import streamlit as st
import ccxt
import yfinance as yf
import pandas as pd
import numpy as np
import math
import pandas_ta as ta  # noqa: F401  (df.ta 접근자 등록용)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from types import SimpleNamespace as NS

st.set_page_config(page_title="PRO 퀀트 트레이딩 대시보드", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# 튜닝 상수 (실제 데이터로 검증하며 여기만 조정하세요)
# ==========================================
RANGE_N = 120              # 지역적 바닥/천장, 볼륨 프로파일, 피보나치 산정 기준 봉 수
PIV_K = 5                  # 피벗 확정 지연: 고점/저점 이후 K봉이 지나야 확정 (미래 참조 방지)
LONG_EMA = 100             # 장기 이평선 (is_crashed 판정용)
CRASH_DD = 0.20            # 범위 고점 대비 이 비율 이상 하락해야 '급락 이탈'로 인정
ACC_CAP_NO_TRIPLE = 55     # 진바닥 3박자 미충족 시 매집 점수 상한 (트리거 확정 전)
BEARFLAG_SLOPE = -2.0      # 역배열 급락 기준 (EMA50 기울기 / 평균 변동성)
BEARFLAG_PENALTY = 15      # 급락 중 횡보 신호가 나오면 매집 점수에서 삭감할 점수
MTF_VETO_RATIO = 1.0       # 상위 프레임 하락 시 매집 점수 삭감 비율 (1.0 = 완전 무효)
POC_BINS = 24              # 볼륨 프로파일 가격 구간 수
VA_PCT = 0.70              # 밸류 에어리어(거래량 누적 비율)

# --- SMC 구조 (Setup & Trigger) ---
CHOCH_BREAK_ATR = 0.3      # IBSL 위로 이 배수(ATR)만큼 종가가 올라서야 '강한 돌파'
CHOCH_SETTLE = 2           # 돌파봉 포함 이 횟수만큼 종가가 돌파선 위에 있어야 '안착'
CHOCH_WINDOW = 12          # 트리거 유효 기간(봉). 이후에는 일반 국면 점수로 판단
CHOCH_CTX_N = 60           # 이 기간 안에 하락 구조(고점 하락 LH)가 있었어야 반전(CHoCH)으로 인정
SETUP_MEM = 60             # 투매(셀링 클라이맥스) 흔적을 셋업 소진 신호로 기억하는 기간(봉)
SETUP_LOOKBACK = 20        # 돌파 직전 이 기간(봉) 안에 바닥 셋업이 성립했으면 '셋업 선행'으로 인정
DIV_MAX_AGE = 40           # 다이버전스·스윕으로 인정할 '마지막 저점 피벗'의 최대 경과 봉수
VOL_BASE_N = 60            # 거래량 기준선(중앙값) 계산 봉수
SSL_WICK_N = 6             # 꼬리로 직전 저점을 쓸어냈는지 확인할 봉수
CHOCH_EXT_ATR = 2.5        # 돌파선에서 이 배수(ATR) 이상 벌어지면 '추격' 경고
TRIGGER_BONUS_ACC = 30     # 트리거 확정 시 매집 가점 (셋업 선행 미확인이면 절반)
TRIGGER_BONUS_UP = 20      # 트리거 확정 시 상승 가점

# --- 매매 계획 (SL/TP, 손익비, 비중) ---
SL_ATR_MULT = 1.5          # 손절 = 진입가 - 1.5 ATR (또는 직전 저점 아래 중 가까운 쪽)
TP1_ATR_MULT = 2.0         # 1차(생존형) 익절 = 진입가 + 2.0 ATR
TP_ATR_MULT = 3.0          # 2차(구조적) 익절의 대체값 = 진입가 + 3.0 ATR (위쪽 저항이 없을 때)
TP3_TRAIL_ATR = 3.0        # 3차(추세 추종) 트레일링 스탑 = 진입 후 고점 - 3.0 ATR
TP_WEIGHTS = (0.4, 0.4, 0.2)   # 기대 손익비(rr_blend) 추정용 가중치 — 화면의 청산 안내는 '1차 50% + 잔여 트레일링' 2단계
HIST_BARS = 60             # '과거 국면 표시'에서 다시 판정할 최근 봉 수 (연산량 제한)
MIN_RISK_ATR = 0.8         # 손절폭 최소값 (너무 타이트한 손절 방지)
RES_MIN_GAP_ATR = 1.0      # 다음 저항으로 인정할 최소 거리 (ATR 배수)
RR_MIN = 1.5               # 이 손익비 미만이면 진입 부적합
RISK_PER_TRADE = 1.0       # 1회 매매 허용 손실 (총자산 대비 %)
MAX_POSITION_PCT = 30.0    # 단일 종목 최대 비중 (총자산 대비 %)

# 한 단계 상위 프레임과, 그 프레임의 봉 하나가 '완성'되는 데 걸리는 시간
UPPER_TF = {"15분": "1시간", "1시간": "4시간", "4시간": "1일", "1일": "주봉", "주봉": "월봉", "월봉": None}
HTF_DELTA = {"1시간": pd.Timedelta(hours=1), "4시간": pd.Timedelta(hours=4), "1일": pd.Timedelta(days=1),
             "주봉": pd.Timedelta(days=7), "월봉": pd.Timedelta(days=31)}

# ==========================================
# 외부 API (모든 호출은 실패해도 앱이 멈추지 않도록 예외 처리)
# ==========================================
def _http_json(url, timeout=5):
    """JSON GET. 실패/지연/비정상 응답이면 None 반환 (예외를 밖으로 내보내지 않음)"""
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

@st.cache_data(ttl=3600)
def get_krw_exchange_rate():
    """USD/KRW 환율. (환율, 출처) 반환. yfinance → open.er-api → 고정값 순으로 시도."""
    try:
        h = yf.Ticker("KRW=X").history(period="5d", timeout=8)
        v = float(h['Close'].dropna().iloc[-1])
        if v > 0:
            return v, "yfinance"
    except Exception:
        pass
    try:
        j = _http_json("https://open.er-api.com/v6/latest/USD", timeout=5)
        v = float(j['rates']['KRW'])
        if v > 0:
            return v, "open.er-api"
    except Exception:
        pass
    return 1350.0, "고정값 (외부 환율 조회 실패)"

KRW_RATE, KRW_SRC = get_krw_exchange_rate()

@st.cache_data(ttl=600)
def get_btc_dominance():
    """BTC 도미넌스(%). 실패 시 None (가짜 기본값을 쓰지 않음)."""
    try:
        j = _http_json('https://api.coingecko.com/api/v3/global', timeout=5)
        v = float(j['data']['market_cap_percentage']['btc'])
        return v if v > 0 else None
    except Exception:
        return None

@st.cache_data(ttl=900)
def get_fear_greed():
    """Alternative.me 공포·탐욕 지수. {'value': int, 'label': str} 또는 None."""
    try:
        j = _http_json('https://api.alternative.me/fng/?limit=1', timeout=5)
        row = j['data'][0]
        return {"value": int(row['value']), "label": str(row.get('value_classification', ''))}
    except Exception:
        return None

def _yf_last_two(ticker):
    """yfinance 최근 종가와 직전 종가. 실패하면 예외를 그대로 올림 (호출하는 쪽에서 N/A 처리)"""
    h = yf.Ticker(ticker).history(period="10d", timeout=8)
    s = h['Close'].dropna()
    if s.empty:
        raise ValueError(f"{ticker}: 빈 데이터")
    last = float(s.iloc[-1])
    prev = float(s.iloc[-2]) if len(s) >= 2 else None
    return last, prev

@st.cache_data(ttl=600)
def get_vix():
    """CBOE 변동성 지수(^VIX). (값, 전일 대비 변화) 반환, 실패 시 ("N/A", None)"""
    try:
        last, prev = _yf_last_two("^VIX")
        return last, (last - prev if prev is not None else None)
    except Exception:
        return "N/A", None

@st.cache_data(ttl=600)
def get_us10y():
    """미국 10년물 국채 금리(^TNX, %). (값, 전일 대비 %p 변화) 반환, 실패 시 ("N/A", None)"""
    try:
        last, prev = _yf_last_two("^TNX")
        if last > 20:                      # 과거 야후 표기(금리 x10)로 들어오는 경우 보정
            last, prev = last / 10, (prev / 10 if prev is not None else None)
        return last, (last - prev if prev is not None else None)
    except Exception:
        return "N/A", None

@st.cache_data(ttl=600)
def get_dxy():
    """달러 인덱스(DX-Y.NYB). (값, 전일 대비 변화) 반환, 실패 시 ("N/A", None)"""
    try:
        last, prev = _yf_last_two("DX-Y.NYB")
        return last, (last - prev if prev is not None else None)
    except Exception:
        return "N/A", None

def _macro_metric(label, data, fmt, delta_fmt, delta_color="normal"):
    """(값, 변화) 튜플을 사이드바 metric으로 출력. 값이 "N/A"면 그대로 표시"""
    val, chg = data
    if val == "N/A" or val is None:
        st.sidebar.metric(label, "N/A")
    else:
        st.sidebar.metric(label, fmt.format(val),
                          delta_fmt.format(chg) if chg is not None else None, delta_color=delta_color)

@st.cache_data(ttl=600)
def get_sector_volumes(sectors_dict):
    all_tickers = [t.split(" ")[0] for ticks in sectors_dict.values() for t in ticks]
    try:
        df = yf.download(all_tickers, period="5d", progress=False, timeout=10)
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

# ==========================================
# 지표 계산 (모든 지표는 '해당 봉까지의 데이터'만 사용 → 미래 참조 없음)
# ==========================================
def add_indicators(df):
    """EMA는 pandas_ta, 나머지는 pandas로 직접 계산 (pandas_ta 버전별 컬럼명 차이로 깨지는 것을 방지)."""
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']
    df['EMA_100'] = c.ewm(span=LONG_EMA, adjust=False, min_periods=LONG_EMA).mean()

    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    df['ATR_ABS'] = atr14
    df['ATR_PCT'] = atr14 / c
    df['OBV'] = (np.sign(c.diff()).fillna(0) * v).cumsum()

    # RSI (Wilder 14)
    diff = c.diff()
    avg_g = diff.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_l = (-diff).clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_g / avg_l.replace(0, np.nan)
    df['RSI'] = (100 - 100 / (1 + rs)).where(avg_l != 0, 100.0)

    # MACD 히스토그램 (12, 26, 9)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False, min_periods=26).mean()
    df['MACD_HIST'] = macd - macd.ewm(span=9, adjust=False).mean()

    # 볼린저 밴드(20, 2σ) 폭 + 켈트너 채널(EMA20 ± 1.5×ATR20) 스퀴즈
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std(ddof=0)
    bb_up, bb_lo = mid + 2 * sd, mid - 2 * sd
    df['BB_MID'] = mid
    df['BBW'] = (bb_up - bb_lo) / mid
    atr20 = tr.rolling(20).mean()
    kc_mid = c.ewm(span=20, adjust=False).mean()
    df['SQUEEZE'] = (bb_up < kc_mid + 1.5 * atr20) & (bb_lo > kc_mid - 1.5 * atr20)

    # 후행성 피벗: (t-K)봉이 t-2K ~ t 구간의 최고/최저면 t 시점에 확정. center=True 미사용.
    K, W = PIV_K, 2 * PIV_K + 1
    is_ph = h.shift(K) == h.rolling(W).max()
    is_pl = l.shift(K) == l.rolling(W).min()
    # 인접한 봉의 고가/저가가 같으면 같은 피벗이 중복 확정되므로, 연속된 동일 값은 첫 번째만 남김
    is_ph = is_ph & ~(is_ph.shift(1, fill_value=False) & (h.shift(K) == h.shift(K + 1)))
    is_pl = is_pl & ~(is_pl.shift(1, fill_value=False) & (l.shift(K) == l.shift(K + 1)))
    df['PIV_H'] = h.shift(K).where(is_ph)
    df['PIV_H_RSI'] = df['RSI'].shift(K).where(is_ph)
    df['PIV_H_MACD'] = df['MACD_HIST'].shift(K).where(is_ph)
    df['PIV_L'] = l.shift(K).where(is_pl)
    df['PIV_L_RSI'] = df['RSI'].shift(K).where(is_pl)
    df['PIV_L_MACD'] = df['MACD_HIST'].shift(K).where(is_pl)

    # SMC 구조: 확정된 최근 피벗 (IBSL = 직전 반등 고점, ISSL = 직전 저점). 모두 ffill이라 미래 참조 없음.
    df['IBSL'] = df['PIV_H'].ffill()
    df['ISSL'] = df['PIV_L'].ffill()
    dh = df['PIV_H'].dropna().diff()
    dl = df['PIV_L'].dropna().diff()
    df['LH_STATE'] = ((dh < 0).astype(float).reindex(df.index).ffill().fillna(0.0) > 0.5)   # 최근 고점이 낮아지는 중(하락 구조)
    df['HL_STATE'] = ((dl > 0).astype(float).reindex(df.index).ffill().fillna(0.0) > 0.5)   # 최근 저점이 높아지는 중

    # Bullish CHoCH 이벤트: 하락 구조(LH) 상태에서, 종가가 '직전 시점의' IBSL을 ATR 0.3배 이상 강하게 상향 돌파
    # 횡보 바닥 안의 잔파동(미세한 HH)이 하락 구조를 지워버리지 않도록, 최근 CHOCH_CTX_N봉 안에 하락 구조(LH)가 있었으면 인정
    down_mem = df['LH_STATE'].astype(float).rolling(CHOCH_CTX_N, min_periods=1).max() > 0.5
    lvl = df['IBSL'].shift(1)
    strong = c > lvl + CHOCH_BREAK_ATR * atr14
    df['BULL_CHOCH'] = strong & ~strong.shift(1, fill_value=False) & down_mem.shift(1, fill_value=False)
    df['BRK_LVL'] = lvl.where(df['BULL_CHOCH'])

    # ==========================================
    # Phase 1 셋업 조건: 모든 봉에 대해 한 번에 벡터 연산 (봉마다 루프를 돌지 않음)
    # rolling/shift만 쓰므로 각 봉은 '그 봉까지의 데이터'만 참조합니다 (미래 참조 없음).
    # ==========================================
    hi_r = h.rolling(RANGE_N, min_periods=20).max()
    lo_r = l.rolling(RANGE_N, min_periods=20).min()
    rng_r = (hi_r - lo_r).replace(0, np.nan)
    df['POS_R'] = (c - lo_r) / rng_r          # 범위 내 위치 (0 = 바닥, 1 = 천장)
    dd_r = 1 - c / hi_r                       # 범위 고점 대비 낙폭

    # (1) 장기선 이탈: 장기 EMA 아래 + 고점 대비 CRASH_DD 이상 하락
    df['CRASHED'] = (c < df['EMA_100']) & (dd_r >= CRASH_DD)

    # (2) 변동성 수축: 스퀴즈 중이거나, ATR이 중앙값 대비 줄고 밴드폭이 하위권
    atr_med_r = df['ATR_PCT'].rolling(200, min_periods=20).median()
    comp_r = (df['ATR_PCT'] / atr_med_r) < 0.85
    bbw_pct_r = df['BBW'].rolling(RANGE_N, min_periods=20).rank(pct=True)
    df['CONSOLIDATING'] = df['SQUEEZE'] | (comp_r & (bbw_pct_r <= 0.35))

    # 직전 두 개의 '확정 저점 피벗'을 각 봉 시점 기준으로 나란히 정렬 (값·RSI·MACD·경과 봉수)
    mask_l = df['PIV_L'].notna()
    def _pair(col):
        s = df[col].where(mask_l)
        return s.ffill(), s[mask_l].shift(1).reindex(df.index).ffill()
    pl_last, pl_prev = _pair('PIV_L')
    plr_last, plr_prev = _pair('PIV_L_RSI')
    plm_last, plm_prev = _pair('PIV_L_MACD')
    idx_s = pd.Series(np.arange(len(df)), index=df.index, dtype=float)
    age_last = idx_s - idx_s.where(mask_l).ffill()
    age_prev = idx_s - idx_s[mask_l].shift(1).reindex(df.index).ffill()
    fresh_l = (age_last <= DIV_MAX_AGE) & (age_prev <= RANGE_N)
    lower_low = pl_last < pl_prev

    # (3) 소진 신호: 투매 흔적 / OBV 강세 / RSI·MACD 강세 다이버전스
    vol_base_r = v.rolling(VOL_BASE_N, min_periods=10).median().replace(0, np.nan)
    climax_bar = (v > 2.5 * vol_base_r) & (c < df['open']) & (((l - lo_r) / rng_r) < 0.25)
    sell_climax_mem = climax_bar.astype(float).rolling(SETUP_MEM, min_periods=1).max() > 0
    pr20 = c / c.shift(20) - 1
    obv_chg = df['OBV'] - df['OBV'].shift(20)
    df['BULL_OBV'] = ((obv_chg.abs() > vol_base_r * 2) & (df['POS_R'] <= 0.45)
                      & (pr20 <= 0.01) & (obv_chg > 0))
    df['BULL_DIV_RSI'] = lower_low & (plr_last > plr_prev + 1.0) & fresh_l & (df['POS_R'] <= 0.5)
    df['BULL_DIV_MACD'] = lower_low & (plm_last > plm_prev) & fresh_l & (df['POS_R'] <= 0.5)
    df['EXHAUSTION'] = sell_climax_mem | df['BULL_OBV'] | df['BULL_DIV_RSI'] | df['BULL_DIV_MACD']

    # (4) SSL 유동성 스윕: 직전 저점을 아래로 쓸어낸 뒤 그 위로 회복 (셋업 보조 항목)
    conf_sweep_r = lower_low & (c > pl_prev) & fresh_l
    wick_sweep_r = (l.rolling(SSL_WICK_N, min_periods=1).min() < df['ISSL']) & (c > df['ISSL'])
    df['SSL_SWEEP'] = conf_sweep_r | wick_sweep_r

    # 셋업 성립(B급) = 장기선 이탈 + 변동성 수축 + 소진 신호
    df['SETUP_OK'] = df['CRASHED'] & df['CONSOLIDATING'] & df['EXHAUSTION']
    # 3중 교차 검증(A급) = 위 조건 + ① SSL 유동성 스윕 + ② 모멘텀 다이버전스 동반
    df['SETUP_DIV'] = df['BULL_DIV_RSI'] | df['BULL_DIV_MACD']
    df['SETUP_A'] = df['SETUP_OK'] & df['SSL_SWEEP'] & df['SETUP_DIV']
    # 차트 매수 마커용: 돌파봉이 마감 확정된 다음 봉에 True (실제로 행동 가능해진 시점)
    df['BULL_CHOCH_CONFIRMED'] = df['BULL_CHOCH'].shift(1).fillna(False).astype(bool)
    return df

# ==========================================
# 시장 국면 엔진: 매집 → 상승 → 분배 → 하락
# ==========================================
# 세력의 의도를 직접 알 수는 없으므로, 가격·거래량에 남는 흔적으로 국면을 추정합니다.
# 점수판(4개 영역 25점씩) + 가짜 바닥 방어 보정 + SMC 2단계 진입(셋업 → 트리거) 게이트로 구성됩니다.
PHASE_ORDER = ["하락", "분배", "매집", "상승"]  # 동점이면 앞쪽(보수적)을 우선
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

def _flag(d, col, idx=-1):
    """미리 계산된 boolean 컬럼을 안전하게 읽기 (컬럼 없음/NaN이면 False)"""
    try:
        v = d[col].iloc[idx]
        return False if pd.isna(v) else bool(v)
    except Exception:
        return False

def _fmt_price(x):
    if x is None: return "N/A"
    if x >= 1000: return f"{x:,.0f}"
    if x >= 1: return f"{x:,.2f}"
    return f"{x:.6g}"

def _naive(ts):
    """시간 열을 tz 없는 UTC 시각으로 통일 (상·하위 프레임 비교용)"""
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
    if col not in d.columns:
        return np.array([], dtype=int)
    arr = d[col].to_numpy(dtype=float)
    idx = np.flatnonzero(~np.isnan(arr))
    return idx[idx >= len(d) - within]

def _divergence(d, side, within=RANGE_N, max_age=DIV_MAX_AGE):
    """약세(side='H') 다이버전스 판정용. 강세(side='L')는 add_indicators의 BULL_DIV_* 컬럼을 사용합니다."""
    col = 'PIV_L' if side == 'L' else 'PIV_H'
    if col + '_RSI' not in d.columns or col + '_MACD' not in d.columns:
        return {"rsi": False, "macd": False}
    idx = _pivot_idx(d, col, within)
    if len(idx) < 2:
        return {"rsi": False, "macd": False}
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

def _volume_profile(w, bins=POC_BINS, va=VA_PCT):
    """최근 구간의 가격대별 거래량(VPVR 근사: 봉의 전형가 hlc3에 거래량 배분) → POC, 밸류 에어리어(VAL~VAH)"""
    lo, hi = float(w['low'].min()), float(w['high'].max())
    if not hi > lo:
        return None
    tp = ((w['high'] + w['low'] + w['close']) / 3).to_numpy(dtype=float)
    vv = w['volume'].to_numpy(dtype=float)
    ok = ~(np.isnan(tp) | np.isnan(vv))
    hist, edges = np.histogram(tp[ok], bins=bins, range=(lo, hi), weights=vv[ok])
    total = float(hist.sum())
    if total <= 0:
        return None
    k = int(hist.argmax())
    a = b = k
    acc = float(hist[k])
    while acc < va * total and (a > 0 or b < bins - 1):
        up = hist[b + 1] if b < bins - 1 else -1.0
        dn = hist[a - 1] if a > 0 else -1.0
        if up >= dn:
            b += 1
            acc += float(hist[b])
        else:
            a -= 1
            acc += float(hist[a])
    centers = (edges[:-1] + edges[1:]) / 2
    return {"poc": float(centers[k]), "val": float(edges[a]), "vah": float(edges[b + 1]),
            "hist": hist, "centers": centers, "total": total}

def _poc_state(w, price, poc, tol):
    if poc is None:
        return "none"
    r = w.tail(10)
    if price >= poc:
        return "support" if float(r['low'].min()) <= poc + tol else "above"
    return "resist" if float(r['high'].max()) >= poc - tol else "below"

def htf_regime(hsub, label):
    """상위 프레임 거시 추세: 장기 EMA(기본 200) 기준 bull / bear / neutral. 데이터가 짧으면 근사 EMA 사용."""
    n = len(hsub)
    if n < 60:
        return {"state": "n/a", "label": label, "span": 0, "approx": False}
    c = hsub['close']
    span = 200 if n >= 220 else max(50, n // 2)
    e = c.ewm(span=span, adjust=False).mean()
    atr_med = 0.02
    if 'ATR_PCT' in hsub.columns:
        m = hsub['ATR_PCT'].dropna().median()
        atr_med = float(m) if not pd.isna(m) else 0.02
    atr_med = max(atr_med, 1e-4)
    slope = (float(e.iloc[-1]) / float(e.iloc[-11]) - 1) / atr_med
    price = float(c.iloc[-1])
    if price < e.iloc[-1] and slope < -0.5:
        state = "bear"
    elif price > e.iloc[-1] and slope > 0.5:
        state = "bull"
    else:
        state = "neutral"
    return {"state": state, "label": label, "span": span, "approx": span < 200}

def prepare_htf(htf, tf_label):
    """상위 프레임 데이터를 '봉이 완성되는 시각' 기준으로 정리 (진행 중인 봉은 쓰지 않음)"""
    up = UPPER_TF.get(tf_label)
    if htf is None or len(htf) == 0 or up is None:
        return None
    h = htf.reset_index(drop=True).copy()
    done = (_naive(h['time']) + HTF_DELTA[up]).to_numpy(dtype='datetime64[ns]')
    return {"df": h, "done": done, "label": up}

def mtf_at(pack, t64):
    if pack is None:
        return None
    k = int(np.searchsorted(pack['done'], t64, side='right'))
    return htf_regime(pack['df'].iloc[:k], pack['label'])

REGIME_SMA = 200          # 일봉 거시 추세 기준선 (단순이평)
REGIME_SLOPE_N = 20       # 기울기 판단 구간 (봉)

def daily_regime(df):
    """
    일봉 데이터 '안에서' 200일 단순이평의 위치·기울기로 거시 추세 판정 (외부 호출 없음).
    스크리너와 상세 분석(1일 프레임)이 같은 함수를 써서 판정이 뒤집히지 않게 함.
      bear    : 종가 < 200일선 이고 200일선이 아직 내려가는 중  → 반등(CHoCH)은 데드캣으로 보고 매수 차단
      bull    : 종가 > 200일선 이고 200일선이 올라가는 중
      neutral : 그 외 (200일선 아래라도 기울기가 멈추면 무릎 매수 후보를 막지 않음)
    """
    c = df['close'].iloc[:-1] if len(df) > 1 else df['close']        # 진행 중인 봉 제외
    base = {"label": "일봉", "span": REGIME_SMA, "approx": False, "kind": "SMA"}
    if len(c) < REGIME_SMA + REGIME_SLOPE_N:
        return {"state": "n/a", **base}
    sma = c.rolling(REGIME_SMA).mean()
    s_now, s_prev, px = float(sma.iloc[-1]), float(sma.iloc[-1 - REGIME_SLOPE_N]), float(c.iloc[-1])
    if px < s_now and s_now < s_prev:
        state = "bear"
    elif px > s_now and s_now > s_prev:
        state = "bull"
    else:
        state = "neutral"
    return {"state": state, **base, "sma": s_now, "gap_pct": px / s_now - 1, "slope_pct": s_now / s_prev - 1}

def _features(d):
    """분석 구간(d)에서 점수·SMC 판정에 필요한 원시 변수를 계산 (d 안의 데이터만 사용)"""
    n = len(d)
    w = d.tail(RANGE_N)
    price = float(d['close'].iloc[-1])
    hi, lo = float(w['high'].max()), float(w['low'].min())
    rng = max(hi - lo, 1e-12)
    pos = (price - lo) / rng  # 0 = 구간 바닥, 1 = 구간 천장
    dd = 1 - price / hi if hi > 0 else 0.0

    # --- 추세 ---
    ema20 = _val(d, 'EMA_20', default=price)
    ema50 = _val(d, 'EMA_50', default=price)
    ema50_prev = _val(d, 'EMA_50', -11, default=ema50)
    ema_long = _val(d, 'EMA_100', default=None)
    atr_s = d['ATR_PCT'].dropna() if 'ATR_PCT' in d.columns else pd.Series(dtype=float)
    atr_med = max(float(atr_s.median()) if len(atr_s) else 0.02, 1e-4)
    atr_now = float(atr_s.iloc[-1]) if len(atr_s) else atr_med
    atr_abs = _val(d, 'ATR_ABS', default=None)
    if atr_abs is None or atr_abs <= 0:
        atr_abs = price * atr_now
    slope = (ema50 / ema50_prev - 1) / atr_med if ema50_prev else 0.0  # 변동성 대비 기울기
    comp = atr_now / atr_med < 0.85  # ATR 수축
    up = price > ema20 > ema50
    down = price < ema20 < ema50
    flat = abs(slope) < 1.0
    crack = price < ema20 and ema20 > ema50
    rebound = ema20 < price < ema50

    # --- 볼린저-켈트너 스퀴즈 ---
    sq_arr = d['SQUEEZE'].to_numpy(dtype=bool) if 'SQUEEZE' in d.columns else np.zeros(n, dtype=bool)
    sq_on = bool(sq_arr[-1])
    sq_cnt = _trail_true(sq_arr)
    sq_release = (not sq_on) and bool(sq_arr[-4:-1].any())
    bb_mid = _val(d, 'BB_MID', default=price)
    dir_up = price > bb_mid
    bbw_now = _val(d, 'BBW', default=None)
    bbw_hist = w['BBW'].dropna() if 'BBW' in w.columns else pd.Series(dtype=float)
    bbw_pct = float((bbw_hist <= bbw_now).mean()) if (bbw_now is not None and len(bbw_hist)) else 1.0

    # --- 확정 피벗 구조 ---
    ih, il = _pivot_idx(d, 'PIV_H'), _pivot_idx(d, 'PIV_L')
    enough = len(ih) >= 2 and len(il) >= 2
    HH = LH = HL = LL = False
    if enough:
        ph, pl = d['PIV_H'].to_numpy(dtype=float), d['PIV_L'].to_numpy(dtype=float)
        HH, LH = bool(ph[ih[-1]] > ph[ih[-2]]), bool(ph[ih[-1]] < ph[ih[-2]])
        HL, LL = bool(pl[il[-1]] > pl[il[-2]]), bool(pl[il[-1]] < pl[il[-2]])

    # --- 거래량 ---
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
    bull_obv = _flag(d, 'BULL_OBV')   # add_indicators에서 미리 계산 (셋업 판정과 동일한 기준을 공유)
    bear_obv = False
    if n > 25 and 'OBV' in d.columns:
        obv_chg = float(d['OBV'].iloc[-1] - d['OBV'].iloc[-21])
        bear_obv = bool(abs(obv_chg) > vol_base * 2 and pos >= 0.6 and pr20 >= -0.01 and obv_chg < 0)
    vol_exhaust_up = pos >= 0.7 and pr20 > 0.03 and vr < 0.85  # 가격은 오르는데 거래량이 마르는 손바꿈 의심

    # --- 모멘텀 다이버전스 (RSI / MACD 히스토그램) ---
    bull_rsi, bull_macd = _flag(d, 'BULL_DIV_RSI'), _flag(d, 'BULL_DIV_MACD')  # 벡터 연산 결과 재사용
    dv_h = _divergence(d, 'H')
    bear_rsi, bear_macd = bool(dv_h['rsi'] and pos >= 0.5), bool(dv_h['macd'] and pos >= 0.5)

    # --- 볼륨 프로파일 (POC / 밸류 에어리어 / 매물대 벽) ---
    vp = _volume_profile(w)
    tol = price * max(atr_now, 0.005)
    if vp:
        poc, vah, val = vp['poc'], vp['vah'], vp['val']
        overhead = float(vp['hist'][vp['centers'] > price].sum() / vp['total'])  # 현재가 위쪽에 쌓인 매물 비중
    else:
        poc = vah = val = None
        overhead = 0.5
    poc_state = _poc_state(w, price, poc, tol)
    va_pos = 'none' if vp is None else ('above' if price > vah else 'below' if price < val else 'inside')
    cl12 = d['close'].tail(12)
    va_break_up = bool(vp and va_pos == 'above' and (cl12.iloc[:-1] <= vah).any())          # 최근 VAH 아래에서 올라온 상방 돌파
    va_hold = bool(vp and va_pos == 'above' and (d['close'].tail(3) > vah).all())            # 3봉 연속 VAH 위 안착
    va_support = bool(vp and va_pos == 'above' and float(d['low'].tail(10).min()) <= vah + tol)  # VAH 리테스트 후 지지
    va_fail = bool(vp and va_pos != 'above' and (cl12 > vah).any())                          # 돌파 실패(되밀림)
    wall_above = bool(vp and overhead >= 0.55 and price < poc)                               # 두꺼운 위쪽 매물벽

    # --- 진바닥 조건 / SSL 스윕: add_indicators의 벡터 연산 결과를 읽어 단일 기준 유지 ---
    is_crashed = _flag(d, 'CRASHED')
    is_consolidating = _flag(d, 'CONSOLIDATING')
    ssl_sweep = _flag(d, 'SSL_SWEEP')
    triple = {"crashed": is_crashed, "consolidating": is_consolidating, "hl": HL}
    return NS(**locals())

def _setup_items_at(d, idx=-1):
    """특정 봉에서의 Phase 1 셋업 항목별 충족 여부 (미리 계산된 컬럼 조회 → O(1))"""
    return {"crashed": _flag(d, 'CRASHED', idx), "consolidating": _flag(d, 'CONSOLIDATING', idx),
            "exhaustion": _flag(d, 'EXHAUSTION', idx), "ssl": _flag(d, 'SSL_SWEEP', idx)}

def _setup_ok_at(d, idx=-1):
    return _flag(d, 'SETUP_OK', idx)

def analyze_market_cycle(df, mtf=None, macro=None, with_facts=True):
    d = df.tail(200)
    F = _features(d)
    n, price, pos = F.n, F.price, F.pos

    # ---------- 1. 추세 (EMA) ----------
    t = {
        "하락": 25 if (F.down and F.slope < -1.0) else 15 if F.down else 8 if (price < F.ema50 and F.slope < 0) else 0,
        "매집": 25 if (F.flat and not F.up) else 15 if F.rebound else 0,
        "상승": 25 if (F.up and F.slope >= 1.0) else 15 if F.up else 8 if price > F.ema20 else 0,
        "분배": 12 if F.crack else 12 if (F.flat and price >= F.ema50) else 0,  # 20EMA 이탈만으로는 분배로 단정하지 않음
    }

    # ---------- 2. 위치 (범위 내 위치 12점 + 볼륨 프로파일 13점) ----------
    pos_pts = {
        "하락": 12 if pos <= 0.5 else 6 if pos <= 0.7 else 0,
        "매집": 12 if pos <= 0.3 else 6 if pos <= 0.45 else 0,
        "상승": 12 if pos >= 0.45 else 5 if pos >= 0.3 else 0,
        "분배": 12 if pos >= 0.7 else 6 if pos >= 0.55 else 0,
    }
    vp_pts = {
        "상승": 13 if (F.va_pos == 'above' and F.va_hold and (F.va_support or F.va_break_up)) else 10 if F.va_pos == 'above'
                else 7 if (F.va_pos == 'inside' and F.poc_state in ('support', 'above')) else 0,
        "매집": 13 if (F.va_break_up and F.va_hold) else 9 if (F.poc_state == 'support' or F.va_support)
                else 6 if (F.va_pos == 'inside' and F.poc_state == 'above') else 3 if F.va_pos == 'inside' else 0,
        "분배": 13 if (F.va_fail or (F.va_pos == 'below' and pos >= 0.5)) else 9 if F.poc_state == 'resist'
                else 5 if (F.va_pos == 'inside' and F.poc_state == 'below') else 0,
        "하락": 13 if (F.va_pos == 'below' and F.wall_above) else 9 if F.va_pos == 'below'
                else 6 if F.poc_state in ('resist', 'below') else 0,
    }
    p = {k: pos_pts[k] + vp_pts[k] for k in PHASE_ORDER}

    # ---------- 3. 파동 (확정 피벗 구조) ----------
    if F.enough:
        wv = {
            "하락": 25 if (F.LH and F.LL) else 12 if F.LL else 0,
            "매집": 25 if (F.HL and not F.HH) else 10 if (not F.HL and not F.LL) else 0,
            "상승": 25 if (F.HH and F.HL) else 12 if F.HL else 0,
            "분배": 25 if (F.LH and not F.LL) else 10 if (F.HH and not F.HL) else 0,
        }
    else:
        wv = {"하락": 5, "매집": 10, "상승": 5, "분배": 10}

    # ---------- 4. 수급 (거래량 + 모멘텀 다이버전스 + 스퀴즈) ----------
    exhaust = F.sell_climax or F.bull_obv or F.bull_rsi or F.bull_macd
    v_up = 25 if (F.ud > 1.2 and F.vr >= 0.9) else 12 if F.ud > 1.0 else 0
    if F.vol_exhaust_up:
        v_up = min(v_up, 5)  # 거래량이 마르는 상승은 '건강한 상승'으로 보지 않음
    v = {
        "하락": min(25, (25 if (F.ud < 0.8 and not exhaust) else 10 if F.ud < 1.0 else 0)
                    + (6 if (F.sq_release and not F.dir_up) else 0)),
        "매집": min(25, (12 if F.sell_climax else 0) + (8 if F.bull_obv else 0) + (8 if F.bull_rsi else 0)
                    + (8 if F.bull_macd else 0) + (4 if (F.bull_rsi and F.bull_macd) else 0)
                    + (6 if (F.sq_on and F.sq_cnt >= 3) else 0) + (4 if (F.sq_release and F.dir_up) else 0)
                    + (4 if (F.dry or F.comp) else 0) + (3 if F.ud >= 1.0 else 0)),
        "상승": min(25, v_up + (6 if (F.sq_release and F.dir_up) else 0)),
        "분배": min(25, (12 if F.dist_climax else 0) + (8 if F.wick_cluster else 0) + (8 if F.bear_obv else 0)
                    + (8 if F.bear_rsi else 0) + (8 if F.bear_macd else 0) + (4 if (F.bear_rsi and F.bear_macd) else 0)
                    + (10 if F.vol_exhaust_up else 0) + (8 if (F.ud < 1.0 and pos >= 0.6) else 0)),
    }
    totals = {k: t[k] + p[k] + wv[k] + v[k] for k in PHASE_ORDER}

    # ---------- SMC: IBSL / ISSL / CHoCH ----------
    # 리페인팅 방지: 진행 중일 수 있는 마지막 봉을 제외한 '확정 마감 봉'만으로 신호를 판정합니다.
    # (가격·손절·익절 등 실행 정보는 아래에서 현재가 price를 그대로 씁니다)
    nc = n - 1 if n > 1 else n
    dc = d.iloc[:nc]
    price_c = float(dc['close'].iloc[-1])
    ibsl = _val(dc, 'IBSL', default=None)
    issl = _val(dc, 'ISSL', default=None)
    atr_abs = F.atr_abs
    atr_c = _val(dc, 'ATR_ABS', default=atr_abs) or atr_abs
    trigger_level = ibsl + CHOCH_BREAK_ATR * atr_c if ibsl is not None else None
    bear_break = bool(issl is not None and price_c < issl - 0.1 * atr_c)  # 직전 저점 종가 이탈 (Bearish CHoCH)

    ev = dc['BULL_CHOCH'].to_numpy(dtype=bool) if 'BULL_CHOCH' in dc.columns else np.zeros(nc, dtype=bool)
    m = min(CHOCH_WINDOW, nc)
    hits = np.flatnonzero(ev[nc - m:])
    choch, setup_grade = None, None
    # 3중 교차 검증 현황: ① SSL 스윕 ② 모멘텀 다이버전스 ③ 구조 반전(CHoCH)
    conf_items = {"ssl": _flag(dc, 'SSL_SWEEP'), "div": _flag(dc, 'SETUP_DIV'), "choch": False}
    if len(hits):
        j = nc - m + int(hits[-1])
        lv = float(dc['BRK_LVL'].iloc[j])
        since = dc['close'].iloc[j:]   # 확정 마감 봉의 종가만 안착 여부로 계산
        choch = {"level": lv, "age": nc - 1 - j, "bars_above": len(since), "held": bool((since > lv).all())}
        # 돌파 직전 SETUP_LOOKBACK봉 안의 셋업 등급을 조회 (컬럼 조회 → 루프 없음)
        win = dc.iloc[max(j - SETUP_LOOKBACK, 0):j + 1]
        if 'SETUP_A' in win.columns and bool(win['SETUP_A'].any()):
            setup_grade = 'A'          # 바닥 조건 + SSL 스윕 + 다이버전스까지 교차 검증
        elif 'SETUP_OK' in win.columns and bool(win['SETUP_OK'].any()):
            setup_grade = 'B'          # 바닥 조건만 충족
        if 'SSL_SWEEP' in win.columns:
            conf_items = {"ssl": bool(win['SSL_SWEEP'].any()), "div": bool(win['SETUP_DIV'].any()), "choch": True}
    setup_ctx = setup_grade is not None
    trigger = bool(choch and choch["held"] and choch["bars_above"] >= CHOCH_SETTLE and not bear_break)
    pending = bool(choch and choch["held"] and choch["bars_above"] < CHOCH_SETTLE and not bear_break)
    recent_choch30 = bool(ev[-30:].any())
    setup_items = _setup_items_at(dc)
    setup_now = _setup_ok_at(dc)
    # 추격 판정은 '지금 사는 가격'(현재가) 기준으로 계산
    extended = bool(trigger and ibsl is not None and (price - choch["level"]) / atr_abs > CHOCH_EXT_ATR)

    veto = bool(mtf and mtf.get("state") == "bear" and MTF_VETO_RATIO > 0)
    if bear_break:
        entry = "WHIPSAW" if recent_choch30 else "DOWN"
    elif (trigger or pending) and veto:
        entry = "BLOCKED"
    elif trigger:
        entry = "TRIGGER" if setup_ctx else "BREAKOUT"  # 바닥 셋업 없이 나온 돌파는 '무릎'이 아닌 단순 추세 돌파로 취급
    elif pending:
        entry = "PENDING"
    elif setup_now:
        entry = "SETUP"
    else:
        entry = "NONE"

    # ---------- 가짜 바닥 방어 보정 → SMC 게이트 ----------
    adjust = []
    bearflag = F.down and F.slope < BEARFLAG_SLOPE and (F.comp or F.sq_on or F.dry)
    if bearflag:
        totals["매집"] = max(0, totals["매집"] - BEARFLAG_PENALTY)
        adjust.append(f"급락 중 횡보(베어플래그) 의심 -{BEARFLAG_PENALTY}")
    if not all(F.triple.values()) and totals["매집"] > ACC_CAP_NO_TRIPLE:
        totals["매집"] = ACC_CAP_NO_TRIPLE
        adjust.append(f"진바닥 3박자 미충족 → 매집 상한 {ACC_CAP_NO_TRIPLE}")
    if veto and mtf.get("kind") == "SMA":
        # 일봉 200일선 모드: 돌파 매수만 차단(BLOCKED)하고, 매집 판정은 관심 종목 관찰용으로 남김
        adjust.append("200일선 아래 · 200일선 하락 중 → 돌파 매수 차단 (매집은 관찰만)")
    elif veto:
        totals["매집"] = int(totals["매집"] * (1 - MTF_VETO_RATIO))
        adjust.append(f"상위 프레임({mtf['label']}) 하락 → 매집 " + ("무효화" if MTF_VETO_RATIO >= 1 else "감점"))
    if bear_break:
        totals["매집"] = 0
        totals["상승"] = 0
        adjust.append("직전 저점(ISSL) 종가 이탈(Bearish CHoCH) → 매수 점수 0")
    if entry in ("TRIGGER", "BREAKOUT"):
        wgt = {'A': 1.0, 'B': 0.65}.get(setup_grade, 0.5)   # 교차 검증 수준에 따라 가점 차등
        bonus_acc = int(round(TRIGGER_BONUS_ACC * wgt))
        bonus_up = int(round(TRIGGER_BONUS_UP * wgt))
        totals["매집"] += bonus_acc
        totals["상승"] += bonus_up
        tag = {'A': "3중 교차 검증(A급 셋업)", 'B': "기본 셋업(B급)"}.get(setup_grade, "바닥 셋업 선행 미확인")
        adjust.append(f"Bullish CHoCH 확정 · {tag} → 매집 +{bonus_acc}, 상승 +{bonus_up}")
    totals = {k: int(min(100, max(0, vv))) for k, vv in totals.items()}

    phase = max(PHASE_ORDER, key=lambda k: totals[k])
    top = totals[phase]

    # ---------- 트레이딩 가이드용 가격대 ----------
    res = []
    if F.vah is not None: res.append(("밸류에어리어 상단(VAH)", F.vah))
    if F.poc is not None: res.append(("매물대 POC", F.poc))
    res.append((f"{len(d.tail(RANGE_N))}봉 고점", F.hi))
    if 'PIV_H' in d.columns:
        for i in _pivot_idx(d, 'PIV_H'):
            res.append(("직전 고점", float(d['PIV_H'].iloc[i])))
    levels = {"price": price, "atr_abs": atr_abs, "ibsl": ibsl, "issl": issl, "trigger": trigger_level,
              "poc": F.poc, "vah": F.vah, "val": F.val, "range_hi": F.hi, "range_lo": F.lo,
              "ema_long": F.ema_long, "resistances": res}

    info = {
        "phase": phase, "score": top, "totals": totals, "hl": bool(F.HL), "weak": top < 50, "bars": n,
        "triple": F.triple, "veto": veto, "bearflag": bool(bearflag), "adjust": adjust, "mtf": mtf,
        "entry": entry, "levels": levels, "extended": extended,
        "smc": {"choch": choch, "trigger": trigger, "pending": pending, "setup_now": setup_now,
                "setup_ctx": setup_ctx, "setup_items": setup_items, "bear_break": bear_break,
                "ibsl": ibsl, "issl": issl, "trigger_level": trigger_level,
                "setup_grade": setup_grade, "conf_items": conf_items},
        "facts": {},
    }
    if not with_facts:
        return info

    # ---------- 팩트체크 문구 ----------
    if F.up:
        f_ema = "20·50 이평선 위 정배열이며 중기 이평선이 " + ("가파르게 " if F.slope >= 1.0 else "완만하게 ") + "우상향 중입니다."
    elif F.down:
        f_ema = "20·50 이평선 아래 역배열로 " + ("이평선이 가파르게 하락하며 " if F.slope < -1.0 else "") + "하방 압력이 이어지고 있습니다."
    elif F.crack:
        f_ema = "상승 추세 중 20 이평선이 깨졌습니다. 단독으로는 분배 확정이 아니며 거래량·윗꼬리 확인이 필요합니다."
    elif F.flat:
        f_ema = "이평선이 평평하게 수렴하며 방향성을 탐색하는 횡보 구간입니다."
    elif F.rebound:
        f_ema = "50 이평선 아래이지만 20 이평선을 회복하며 반등을 시도 중입니다."
    else:
        f_ema = "20 이평선 아래로 밀려 있으나 추세는 뚜렷하지 않습니다."
    if bearflag:
        f_ema += f" 급락 추세 속 일시 횡보(베어플래그) 의심으로 매집 점수 -{BEARFLAG_PENALTY}점을 적용했습니다."
    if mtf is not None:
        lab = mtf['label']
        if mtf.get('kind') == "SMA":       # 일봉 내부 200일선 레짐
            src = f"{mtf['span']}일선"
            gap = f" (현재가 {mtf['gap_pct'] * 100:+.1f}%, 기울기 {'상승' if mtf['slope_pct'] > 0 else '하락'})" if 'gap_pct' in mtf else ""
        else:
            src = f"상위 프레임({lab}) 장기 추세" + (f"({mtf['span']}EMA 근사)" if mtf['approx'] else f"({mtf['span']}EMA)")
            gap = ""
        if mtf['state'] == "n/a":
            f_ema += f" {src} 판정에 필요한 데이터가 부족해 거시 추세 필터는 적용되지 않았습니다."
        elif mtf['state'] == "bear" and mtf.get('kind') == "SMA":
            f_ema += f" {src}{gap} 아래에서 200일선이 아직 내려가는 중이라, 반등 돌파가 나와도 매수를 차단합니다(데드캣 바운스 경계)."
        elif mtf['state'] == "bear":
            f_ema += (f" {src}{gap}가 하락 중이라 매집·반등 신호를 "
                      + ("무효 처리합니다(데드캣 바운스 경계)." if MTF_VETO_RATIO >= 1 else "감점합니다."))
        elif mtf['state'] == "bull":
            f_ema += f" {src}{gap}는 상승입니다."
        else:
            f_ema += f" {src}{gap}는 뚜렷하지 않습니다."

    pct = f"{pos * 100:.0f}%"
    nb = len(F.w)
    if pos <= 0.2:
        f_pos = f"최근 {nb}봉 범위의 하단 {pct} 지점으로 바닥권입니다."
    elif pos <= 0.45:
        f_pos = f"최근 {nb}봉 범위의 하단 {pct} 지점으로 저가 구간입니다."
    elif pos < 0.7:
        f_pos = f"최근 {nb}봉 범위의 {pct} 지점으로 중간대입니다."
    else:
        f_pos = f"최근 {nb}봉 범위의 {pct} 지점으로 고점권입니다."
    if F.vp:
        f_pos += f" 매물대 POC {_fmt_price(F.poc)} · 밸류에어리어 {_fmt_price(F.val)}~{_fmt_price(F.vah)}."
        if F.va_pos == 'above' and F.va_break_up:
            f_pos += " 가격이 밸류에어리어 상단(VAH)을 상방 돌파" + ("해 3봉 안착했습니다." if F.va_hold else "했으나 안착 확인 전입니다.")
        elif F.va_pos == 'above' and F.va_support:
            f_pos += " VAH를 리테스트하며 지지선으로 확인했습니다."
        elif F.va_pos == 'above':
            f_pos += " 가격이 밸류에어리어 위에 있습니다."
        elif F.va_pos == 'inside':
            f_pos += {"support": " POC를 지지선으로 확인했습니다.", "above": " 가격이 POC 위에 있습니다.",
                      "resist": " POC에 눌려 저항을 받고 있습니다.", "below": " 가격이 POC 아래에 있습니다."}.get(F.poc_state, "")
        else:
            f_pos += " 가격이 밸류에어리어 아래에 눌려 있습니다."
        if F.va_fail:
            f_pos += " 최근 VAH 돌파가 실패하고 되밀렸습니다(돌파 실패 주의)."
        f_pos += f" 위쪽 매물 비중 {F.overhead * 100:.0f}%" + (" — 두꺼운 매물벽입니다." if F.wall_above else ".")

    if not F.enough:
        f_wave = "아직 확정된 고점/저점 파동이 부족해 방향성을 탐색 중입니다."
    elif F.HH and F.HL:
        f_wave = "고점과 저점을 모두 높이는 상승 파동(HH/HL)이 살아 있습니다."
    elif F.LH and F.LL:
        f_wave = "고점과 저점이 모두 낮아지는 하락 파동(LH/LL)이 이어지고 있습니다."
    elif F.HL and not F.HH:
        f_wave = "고점은 못 넘었지만 저점을 높이고 있습니다(저점 상승). 매집 가능성을 시사합니다."
    elif F.LH and not F.LL:
        f_wave = "저점은 지켰지만 고점이 낮아지고 있습니다(고점 하락). 분배 가능성을 시사합니다."
    elif F.HH and not F.HL:
        f_wave = "고점은 높였지만 저점이 따라 오르지 못해 상승 힘이 약해지는 모습입니다."
    else:
        f_wave = "고점과 저점이 뚜렷한 방향 없이 혼조를 보이고 있습니다."
    mk = lambda ok: "✔" if ok else "✘"
    f_wave += (f" [진바닥 3박자] 장기선 이탈 {mk(F.triple['crashed'])} · 변동성 수축 {mk(F.triple['consolidating'])}"
               f" · 저점 상승(HL) {mk(F.triple['hl'])}")

    if F.sell_climax:
        f_vol = "저점권에서 투매성 대량 거래량(셀링 클라이맥스)이 나왔습니다. 매도 물량이 소진되는 신호일 수 있습니다."
    elif F.bull_obv:
        f_vol = "가격은 눌려 있는데 거래량 흐름(OBV)은 오르는 강세 다이버전스입니다. 조용한 매집 가능성이 있습니다."
    elif F.dist_climax:
        f_vol = "고점권에서 대량 거래량에도 윗꼬리가 길어지며 물량이 넘어가는 흔적(분배)이 보입니다."
    elif F.bear_obv:
        f_vol = "가격은 버티는데 거래량 흐름(OBV)은 꺾이는 약세 다이버전스입니다."
    elif F.ud > 1.2:
        f_vol = "상승봉 거래량이 하락봉보다 우세해 매수 수급이 강합니다."
    elif F.ud < 0.8:
        f_vol = "하락봉 거래량이 우세해 매도 압력이 큽니다."
    else:
        f_vol = "상승·하락 거래량이 비슷하게 균형을 이루고 있습니다."
    if F.bull_rsi and F.bull_macd:
        f_vol += " RSI·MACD 강세 다이버전스가 중복 포착되어 하락 에너지 고갈 가능성이 큽니다."
    elif F.bull_rsi:
        f_vol += " RSI 강세 다이버전스가 포착됐습니다(MACD는 미확인)."
    elif F.bull_macd:
        f_vol += " MACD 히스토그램 강세 다이버전스가 포착됐습니다(RSI는 미확인)."
    if F.bear_rsi and F.bear_macd:
        f_vol += " RSI·MACD 약세 다이버전스가 중복 포착되어 고점 모멘텀 약화(분배 가능성)가 의심됩니다."
    elif F.bear_rsi:
        f_vol += " RSI 약세 다이버전스가 포착됐습니다."
    elif F.bear_macd:
        f_vol += " MACD 히스토그램 약세 다이버전스가 포착됐습니다."
    if F.wick_cluster:
        f_vol += f" 고점권에서 윗꼬리 캔들이 최근 10봉 중 {F.wick_cnt}개 몰려 있습니다."
    if F.vol_exhaust_up:
        f_vol += " 가격은 오르는데 거래량이 마르고 있어 손바꿈(분배) 의심 구간입니다."
    if F.sq_on:
        f_vol += f" 볼린저 밴드가 켈트너 채널 안으로 수축한 스퀴즈가 {F.sq_cnt}봉째 이어져 돌파 직전 구간입니다."
    elif F.sq_release:
        f_vol += " 스퀴즈가 막 해제되며 " + ("상방" if F.dir_up else "하방") + " 이탈을 시도 중입니다."
    elif F.bbw_pct <= 0.15:
        f_vol += f" 볼린저 밴드폭이 최근 {len(F.bbw_hist)}봉 중 하위 {F.bbw_pct * 100:.0f}%로 좁습니다."
    if F.dry or F.comp:
        f_vol += " 거래량·변동성이 줄어드는 중입니다."

    ib, isl, tl = _fmt_price(ibsl), _fmt_price(issl), _fmt_price(trigger_level)
    gap = f"(현재 종가 대비 {(ibsl / price - 1) * 100:+.1f}%)" if ibsl else ""
    if entry in ("TRIGGER", "BREAKOUT"):
        f_smc = (f"Bullish CHoCH 확정: 하락 구조(LH)였던 시세가 직전 반등 고점(IBSL {_fmt_price(choch['level'])})을 "
                 f"{choch['age']}봉 전 강하게 상향 돌파한 뒤 {choch['bars_above']}봉 연속 안착했습니다. ")
        if setup_grade == 'A':
            f_smc += ("돌파 직전 바닥 셋업에 SSL 유동성 스윕과 모멘텀 다이버전스까지 동반되어 3중 교차 검증을 모두 충족했습니다(A급). "
                      "가장 높은 가점과 비중을 적용합니다.")
        elif setup_grade == 'B':
            miss = [lab for lab, ok in (("SSL 유동성 스윕", conf_items['ssl']), ("모멘텀 다이버전스", conf_items['div'])) if not ok]
            f_smc += (f"바닥 셋업(장기선 이탈·변동성 수축·소진 신호)은 확인됐으나 {' · '.join(miss)}이 빠져 교차 검증이 부족합니다(B급). 비중을 줄여 반영합니다.")
        else:
            f_smc += "다만 돌파 이전 바닥 셋업이 확인되지 않아 '무릎'보다는 단순 추세 돌파에 가깝습니다. 가점과 비중을 줄여 반영합니다."
        if extended:
            f_smc += (f" 다만 돌파선에서 {CHOCH_EXT_ATR} ATR 이상 벌어진 추격 구간이라 신규 진입을 차단했습니다. "
                      "돌파선(IBSL) 부근까지 리테스트를 기다리세요.")
    elif entry == "PENDING":
        f_smc = (f"IBSL({_fmt_price(choch['level'])}) 상향 돌파 직후입니다. 종가 안착이 "
                 f"{max(CHOCH_SETTLE - choch['bars_above'], 1)}회 더 확인되면 Bullish CHoCH로 확정합니다.")
    elif entry == "SETUP":
        it = setup_items
        f_smc = (f"Phase 1 셋업: 장기선 이탈 {mk(it['crashed'])} · 변동성 수축 {mk(it['consolidating'])} · "
                 f"소진 신호(투매/다이버전스) {mk(it['exhaustion'])} 충족. 매수 지시가 아닌 '관심 종목 편입' 상태입니다. "
                 f"SSL(직전 저점 아래 유동성) 스윕 후 회복 " + ("확인됨. " if it['ssl'] else "미확인. ")
                 + f"종가가 {tl} 위에 안착하면 CHoCH 트리거(IBSL {ib} {gap})입니다.")
    elif entry == "WHIPSAW":
        f_smc = (f"직전 저점(ISSL {isl})을 종가 기준으로 하향 이탈했습니다(Bearish CHoCH). "
                 "최근 반등은 가짜 반등(휩쏘)으로 규정하고 매수 점수를 0점으로 삭감했습니다.")
    elif entry == "BLOCKED":
        f_smc = "Bullish CHoCH 신호가 있으나 상위 프레임이 하락 추세라 데드캣 바운스로 간주해 매수 점수를 무효 처리합니다."
    elif entry == "DOWN":
        f_smc = f"하락 구조 진행 중: 종가가 직전 저점(ISSL {isl}) 아래에 있습니다. IBSL {ib} {gap}을 돌파하기 전까지 구조 반전이 아닙니다."
    else:
        f_smc = f"구조 반전 신호 없음. IBSL {ib} {gap} 돌파(Bullish CHoCH)와 ISSL {isl} 이탈(Bearish CHoCH)을 기준으로 관찰하세요."
    if macro and macro.get("fng") is not None:
        fg = macro["fng"]
        if fg <= 25:
            f_smc += f" (참고: 시장 심리 극단적 공포 F&G {fg} — 셋업 보조 정황, 점수에는 미반영)"
        elif fg >= 75:
            f_smc += f" (참고: 시장 심리 극단적 탐욕 F&G {fg} — 과열 주의, 점수에는 미반영)"

    info["facts"] = {"EMA": f_ema, "POS": f_pos, "WAVE": f_wave, "VOL": f_vol, "SMC": f_smc}
    return info

# ==========================================
# 매매 계획: ATR 손절·익절, 기대 손익비, 권장 비중
# ==========================================
def structural_target(L, price, atr):
    """현재가보다 RES_MIN_GAP_ATR(1 ATR) 이상 위에 있는 가장 가까운 저항(VAH·POC·전고점·범위 고점). 없으면 3 ATR 위."""
    cands = sorted((v, lab) for lab, v in (L.get('resistances') or [])
                   if v is not None and np.isfinite(v) and v > price + RES_MIN_GAP_ATR * atr)
    return cands[0] if cands else (price + TP_ATR_MULT * atr, f"ATR {TP_ATR_MULT}배")

def build_trade_plan(info, entry=None, assume_trigger=False):
    """진입가(기본: 현재가) 기준 SL/TP, 다음 저항까지의 손익비, 1% 룰 기반 권장 비중. 계산 불가면 None."""
    L = info['levels']
    price = float(L['price'] if entry is None else entry)
    atr = L['atr_abs']
    if not atr or atr <= 0 or price <= 0:
        return None

    sl_atr = price - SL_ATR_MULT * atr
    issl = L['issl']
    sl_struct = (issl - 0.2 * atr) if (issl is not None and issl < price) else None
    if sl_struct is not None and sl_struct >= sl_atr:
        sl, kind = sl_struct, "구조(직전 저점 아래)"
    else:
        sl, kind = sl_atr, f"ATR {SL_ATR_MULT}배"
    if sl > price - MIN_RISK_ATR * atr:
        sl, kind = price - MIN_RISK_ATR * atr, f"최소 손절폭 {MIN_RISK_ATR} ATR"
    risk = price - sl
    if risk <= 0:
        return None

    # ---------- 3분할 청산 ----------
    # 1차(생존형): 2 ATR. 단, 그 전에 저항이 있으면 저항 아래로 낮춤 (매물벽 위에 익절선을 두지 않음)
    res_up = sorted(v for lab, v in L['resistances'] if v is not None and v > price + 0.2 * atr)
    tp1 = price + TP1_ATR_MULT * atr
    tp1_label = f"ATR {TP1_ATR_MULT}배"
    if res_up and res_up[0] < tp1:
        tp1, tp1_label = res_up[0], "근접 저항"
    # 2차(구조적): 1차보다 위에 있는 가장 가까운 저항(VAH·POC·전고점), 없으면 3 ATR
    cands = sorted((v, lab) for lab, v in L['resistances']
                   if v is not None and v > max(tp1, price + RES_MIN_GAP_ATR * atr) + 1e-12)
    tp2, tp2_label = (cands[0][0], cands[0][1]) if cands else (price + TP_ATR_MULT * atr, f"ATR {TP_ATR_MULT}배")
    # 3차(추세 추종): 가격이 아니라 규칙 — 트레일링 스탑
    tp3_rule = f"진입 후 고점 대비 {TP3_TRAIL_ATR} ATR 하락 시 청산"
    if L.get('ema_long'):
        tp3_rule += f" (또는 {LONG_EMA}EMA {_fmt_price(L['ema_long'])} 종가 이탈 시)"

    # 2단계 청산용 '구조적 익절가': 현재가보다 1 ATR 이상 위에 있는 가장 가까운 매물대 저항 (없으면 3 ATR)
    tp_struct = structural_target(L, price, atr)

    w1, w2, w3 = TP_WEIGHTS
    rr_tp1 = (tp1 - price) / risk
    rr_tp2 = (tp2 - price) / risk
    rr = rr_tp2
    # 분할 청산 기준 기대 손익비 (3차는 최소 2차 수준에서 청산된다고 보수적으로 가정)
    rr_blend = w1 * rr_tp1 + (w2 + w3) * rr_tp2
    risk_pct = risk / price * 100
    risk_cap = min(MAX_POSITION_PCT, RISK_PER_TRADE / risk_pct * 100)  # 1% 룰: 손절 시 총자산의 1% 손실이 되는 비중
    state = "TRIGGER" if assume_trigger else info['entry']
    # 신규 진입 원천 차단 조건 (비중을 줄이는 것이 아니라 0%로 막음)
    blocked = None
    if info.get('extended') and not assume_trigger:
        blocked = f"추격 구간 — 돌파선에서 {CHOCH_EXT_ATR} ATR 이상 벌어졌습니다"
    elif rr_blend < RR_MIN:
        blocked = f"분할 청산 기대 손익비 1:{rr_blend:.1f}이 최소 기준 1:{RR_MIN}에 못 미칩니다"
    scale = 0.0 if blocked else (0.4 if rr_blend < 2 else 0.7 if rr_blend < 3 else 1.0)

    conf, notes = 1.0, []
    mtf = info['mtf']
    if mtf is None or mtf.get('state') in ('n/a', 'neutral'):
        conf *= 0.85
        notes.append("상위 프레임 확신 부족 ×0.85")
    if state == "BREAKOUT":
        conf *= 0.5
        notes.append("바닥 셋업 없는 단순 돌파 ×0.5")
    elif state != "TRIGGER":
        conf *= 0.5
        notes.append("CHoCH 트리거 없는 진입 ×0.5")
    elif info['smc'].get('setup_grade') == 'B' and not assume_trigger:
        conf *= 0.75   # A급(3중 교차 검증)만 최대 비중
        notes.append("3중 교차 검증 미충족(B급 셋업) ×0.75")
    if info['weak']:
        conf *= 0.7
        notes.append("국면 신호 약함 ×0.7")
    position_pct = 0.0 if blocked else round(risk_cap * scale * conf, 1)

    return {"entry": price, "sl": sl, "sl_kind": kind, "sl_pct": (sl / price - 1) * 100,
            "tp1": tp1, "tp1_pct": (tp1 / price - 1) * 100, "tp1_label": tp1_label, "tp1_w": w1,
            "tp2": tp2, "tp2_label": tp2_label, "tp2_pct": (tp2 / price - 1) * 100, "tp2_w": w2,
            "tp3_rule": tp3_rule, "tp3_w": w3,
            "rr": rr, "rr_tp1": rr_tp1, "rr_tp2": rr_tp2, "rr_blend": rr_blend,
            "rr_basis": tp2_label, "risk_pct": risk_pct,
            "tp_struct": tp_struct[0], "tp_struct_label": tp_struct[1], "tp_struct_pct": (tp_struct[0] / price - 1) * 100,
            "rr_struct": (tp_struct[0] - price) / risk,
            "risk_cap": risk_cap, "scale": scale, "conf": conf, "notes": notes,
            "position_pct": position_pct, "blocked": blocked}

@st.cache_data(ttl=300)
def cached_phase_history(df, htf_pack=None, bars=HIST_BARS, min_start=60, daily=False):
    """최근 bars봉만, 각 시점에서 '그 시점까지의 데이터'로 다시 판별한 국면 구간 (검증용 · 연산량 제한)"""
    start = max(len(df) - bars, min_start)
    if start >= len(df):
        return []
    tn = _naive(df['time']).to_numpy(dtype='datetime64[ns]')
    labels = []
    for i in range(start, len(df)):
        sub = df.iloc[:i + 1]
        mtf_i = daily_regime(sub) if daily else mtf_at(htf_pack, tn[i])       # 화면 판정과 같은 거시 필터
        labels.append(analyze_market_cycle(sub, mtf_i, None, False)['phase'])
    runs = []
    for k, ph in enumerate(labels):
        if runs and runs[-1][0] == ph:
            runs[-1][2] = k
        else:
            runs.append([ph, k, k])
    merged = []  # 3봉 미만의 짧은 깜빡임은 직전 국면에 흡수
    for r in runs:
        if merged and (r[2] - r[1] + 1 < 3 or merged[-1][0] == r[0]):
            merged[-1][2] = r[2]
        else:
            merged.append(r)
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
    with st.expander("📌 시장 국면 가이드 (매집 → 상승 → 분배 → 하락)", expanded=False):
        st.markdown("- 🔵 **매집**: `[바닥권 횡보 · 거래량 감소 · 저점 상승]` - 셋업 확인 후 CHoCH 트리거에서만 매수")
        st.markdown("- 🟢 **상승**: `[정배열 · 고점/저점 상승]` - 보유 유지, 눌림목만 매수 (추격 주의)")
        st.markdown("- 🟣 **분배**: `[고점권 거래량 폭증 · 윗꼬리 · 고점 하락]` - 신규 매수 금지, 분할 매도")
        st.markdown("- 🔴 **하락**: `[역배열 · 거래량 동반 하락]` - 매수 보류 (떨어지는 칼날)")
        st.markdown("- 🧲 **Phase 1 셋업**: `[장기선 이탈 + 변동성 수축 + 소진 신호]` → 관심 종목 편입 (매수 아님)")
        st.markdown("- 🎯 **Phase 2 트리거**: `[종가가 직전 반등 고점(IBSL)을 돌파·안착 = Bullish CHoCH]` → 실전 매수 타점")
        st.markdown("- 🅰️ **A급 트리거**: `[SSL 유동성 스윕 + 모멘텀 다이버전스 + CHoCH]` 3중 교차 검증 → 최대 비중")
        st.markdown("- 💰 **청산**: `1차 50%(구조적 저항 · 도달 시 손절선을 본절로)` → `잔여 50%(트레일링 스탑: 고점 − 3 ATR)`")
        st.markdown("- ⚠️ **3박자·매집 점수는 '관심 종목' 기준일 뿐**입니다. 실제 매수는 맨 위 결론이 🟢 매수 실행일 때만 하세요.")

def render_fact_box(facts):
    with st.container(border=True):
        st.markdown("##### 📝 4대 팩트체크 + SMC")
        st.markdown(f"- **[추세 EMA]**: {facts['EMA']}")
        st.markdown(f"- **[위치 POS]**: {facts['POS']}")
        st.markdown(f"- **[파동 WAVE]**: {facts['WAVE']}")
        st.markdown(f"- **[수급 VOL]**: {facts['VOL']}")
        st.markdown(f"- **[구조 SMC]**: {facts['SMC']}")

def _render_plan(plan, rate, title, show_entry=True, show_size=False):
    def fmt(v): return f"₩{v * rate:,.0f}"
    st.markdown(f"**{title}**")
    if show_entry:
        st.markdown(f"- 진입 기준가: {fmt(plan['entry'])}")
        st.markdown(f"- 🛑 권장 손절선(무효화 라인): {fmt(plan['sl'])} ({plan['sl_pct']:+.1f}%) · 기준: {plan['sl_kind']}")
    st.markdown("- 🎯 **2단계 청산** (💼 포지션 관리 탭과 같은 기준)")
    st.markdown(f"    - 1차 (물량 50%, 구조적 익절 · {plan['tp_struct_label']}): "
                f"{fmt(plan['tp_struct'])} ({plan['tp_struct_pct']:+.1f}%) · 손익비 1:{plan['rr_struct']:.1f}")
    st.markdown(f"    - ⚠️ 1차 도달 시 **잔여 물량의 손절선을 진입가({fmt(plan['entry'])})로 상향 조정**해 "
                "무위험(Risk-Free) 트레이딩으로 전환하세요.")
    st.markdown(f"    - 잔여 (물량 50%, 추세 추종): {plan['tp3_rule']} — 손절선은 올리기만 하고 내리지 않습니다.")
    st.markdown(f"- ⚖️ 기대 손익비(참고): 1 : {plan['rr_blend']:.1f} · 손절폭 {plan['risk_pct']:.1f}%")
    if not show_size:
        st.caption("진입 가격·수량·공통 손절가는 맨 위 결론과 🧮 2분할 진입 플랜을 따르세요.")
    elif plan['blocked']:
        st.markdown(f"- 📦 권장 비중: **0% (신규 진입 차단)** — {plan['blocked']}")
    else:
        extra = (" · " + ", ".join(plan['notes'])) if plan['notes'] else ""
        st.markdown(f"- 📦 권장 비중(1차 진입 총량): **총자산의 약 {plan['position_pct']}%** "
                    f"(1회 허용 손실 {RISK_PER_TRADE}% ÷ 손절폭 {plan['risk_pct']:.1f}% = 최대 {plan['risk_cap']:.1f}%, "
                    f"손익비 보정 ×{plan['scale']}{extra})")

def render_trade_guide(info, fibs, curr, rate):
    def fmt(v): return f"₩{v * rate:,.0f}"

    mk = lambda ok: "✔" if ok else "✘"
    sell_line = " | ".join(f"{lab}: {fmt(p)}" + (" (도달)" if curr >= p else "") for lab, p in fibs['sell'])
    entry, ph, smc, L = info['entry'], info['phase'], info['smc'], info['levels']

    if entry in ("TRIGGER", "BREAKOUT"):
        ci = smc.get('conf_items') or {}
        grade = smc.get('setup_grade')
        st.markdown(f"**[3중 교차 검증]** ① SSL 유동성 스윕 {mk(ci.get('ssl'))} · ② 모멘텀 다이버전스 {mk(ci.get('div'))} "
                    f"· ③ 구조 반전 CHoCH {mk(ci.get('choch'))}"
                    + (f" → **{grade}급 셋업**" if grade else " → 셋업 미확인"))
        if info['extended']:
            st.error(f"🚫 추격 매수 금지: 돌파선(IBSL {fmt(smc['choch']['level'])}) 근처 리테스트 대기 — "
                     f"돌파 후 {CHOCH_EXT_ATR} ATR 이상 벌어져 신규 진입 비중을 0%로 차단했습니다.")
        elif entry == "TRIGGER" and grade == 'A':
            st.success("🎯 Phase 2 트리거 (A급): 유동성 스윕 + 다이버전스 + CHoCH가 모두 교차 검증된 최우선 매수 타점입니다.")
        elif entry == "TRIGGER":
            st.success("🎯 Phase 2 트리거 (B급): 바닥 셋업 → Bullish CHoCH 확정. 교차 검증 일부 미충족이라 비중을 줄여 진입하세요.")
        else:
            st.warning("📈 Bullish CHoCH는 확정됐지만 바닥 셋업이 선행되지 않았습니다. 저점 매수가 아닌 추세 돌파 진입이므로 비중을 절반 이하로 줄여 분할 접근하세요.")
        plan = build_trade_plan(info)
        if plan:
            _render_plan(plan, rate, "[청산 계획]", show_entry=False)
        else:
            st.markdown("- ATR을 계산할 수 없어 손절·익절 라인을 산정하지 못했습니다.")
        st.markdown(f"- 리테스트 참고선: IBSL {fmt(smc['choch']['level'])} (종가 이탈 시 신호 실패)")
        if info['phase'] in ("상승",):
            st.markdown("**[매도 참고 가격대 (피보나치)]**")
            st.markdown(f"- {sell_line}")
    elif entry == "PENDING":
        st.info(f"⏳ IBSL 돌파 직후입니다. 종가 안착 확인 전이므로 매수를 보류하세요. (돌파선 {fmt(smc['choch']['level'])})")
    elif entry == "SETUP":
        st.warning("🧲 Phase 1 셋업: 관심 종목에 편입하세요. 아직 매수 지시가 아닙니다.")
        it = smc['setup_items']
        st.markdown(f"- 체크: 장기선 이탈 {mk(it['crashed'])} · 변동성 수축 {mk(it['consolidating'])} · "
                    f"소진 신호 {mk(it['exhaustion'])} · SSL 스윕 후 회복 {mk(it['ssl'])}")
        if smc['trigger_level'] is not None:
            st.markdown(f"- 🎯 트리거 조건: 종가가 **{fmt(smc['trigger_level'])}** (IBSL {fmt(smc['ibsl'])} + {CHOCH_BREAK_ATR} ATR) 위에 {CHOCH_SETTLE}회 안착")
            preview = build_trade_plan(info, entry=smc['trigger_level'], assume_trigger=True)
            if preview:
                _render_plan(preview, rate, "[트리거 발생 시 청산 계획 (미리보기)]", show_entry=False)
        st.markdown(f"- 무효화: 종가가 직전 저점(ISSL {fmt(smc['issl'])}) 아래로 내려가면 셋업 폐기")
    elif entry == "WHIPSAW":
        st.error("🪤 가짜 반등(휩쏘): 직전 저점을 종가 기준으로 이탈했습니다. 매수 점수를 0점으로 삭감했고 매수를 보류하세요.")
    elif entry == "BLOCKED":
        _m = info.get('mtf') or {}
        st.error("🚫 Bullish CHoCH가 나왔지만 " + ("200일선 아래에서 200일선이 아직 내려가는 중입니다."
                 if _m.get('kind') == "SMA" else "상위 프레임이 하락 추세입니다.") + " 데드캣 바운스로 보고 매수를 보류하세요.")
    elif ph == "분배":
        st.error("🟣 분배 국면입니다. 신규 매수를 금지하고 보유분은 분할 매도를 검토하세요.")
        st.markdown("**[매도 참고 가격대 (피보나치)]**")
        st.markdown(f"- {sell_line}")
    elif ph == "상승":
        st.markdown("**[눌림목 참고 계획]** (CHoCH 트리거 없는 기존 상승 추세 · 비중 보수적)")
        plan = build_trade_plan(info)
        if plan:
            _render_plan(plan, rate, "눌림목 참고 계획 (매수 신호 아님)")
        st.markdown("**[매도 참고 가격대 (피보나치)]** (고점권 접근 시 💼 포지션 관리 탭으로 확인)")
        st.markdown(f"- {sell_line}")
    elif ph == "매집":
        st.info("🔵 매집 정황은 있으나 CHoCH 트리거 전입니다. 종가가 IBSL을 돌파·안착할 때까지 매수를 보류하고 관찰하세요.")
        if smc['trigger_level'] is not None:
            st.markdown(f"- 🎯 트리거 조건: 종가가 **{fmt(smc['trigger_level'])}** 위에 {CHOCH_SETTLE}회 안착")
    else:
        st.error("📉 하락 국면입니다. 구조 반전(Bullish CHoCH) 전까지 현금 비중을 유지하고 매수를 보류하세요.")
        if smc['trigger_level'] is not None:
            st.markdown(f"- 반전 확인선: 종가가 **{fmt(smc['trigger_level'])}** 위로 안착하면 구조 반전 후보")

    if info['weak']:
        st.caption("신호가 약해 국면 전환기일 수 있습니다. 단정적인 판단은 피하세요.")
    st.caption("※ 진입 신호는 마감이 확정된 봉만으로 판정합니다(진행 중인 봉 제외 → 장중 신호 깜빡임 방지). "
               "손절·익절 가격은 현재가 기준으로 계산됩니다.")
    st.caption("※ 규칙 기반 참고 지표이며 수익을 보장하지 않습니다. 투자 판단과 책임은 본인에게 있습니다.")

def build_chart(df, fibs, rate=1.0, history=None, info=None, plan=None, extra_lines=None, zones=None):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    if history:
        for x0, x1, ph in history:
            fig.add_vrect(x0=x0, x1=x1, fillcolor=PHASE_META[ph]['color'], opacity=0.12,
                          layer="below", line_width=0, row=1, col=1)
    fig.add_trace(go.Candlestick(x=df['time'], open=df['open'] * rate, high=df['high'] * rate,
                                 low=df['low'] * rate, close=df['close'] * rate, name='Price'), row=1, col=1)
    if 'EMA_20' in df.columns:
        fig.add_trace(go.Scatter(x=df['time'], y=df['EMA_20'] * rate, line=dict(color='orange', width=1), name='EMA 20'), row=1, col=1)
    fig.add_hline(y=fibs['fib_1'] * rate, line_dash="dot", line_color="orange", row=1, col=1)
    fig.add_hline(y=fibs['fib_3'] * rate, line_dash="dot", line_color="red", row=1, col=1)

    # --- SMC 구조선: IBSL(직전 반등 고점) 주황 점선 / ISSL(직전 저점) 붉은 점선 ---
    if info is not None:
        lv = info.get('levels', {})
        if lv.get('ibsl'):
            fig.add_hline(y=lv['ibsl'] * rate, line_dash="dash", line_color="#ffa726", line_width=1.5,
                          annotation_text="IBSL (돌파 기준)", annotation_position="top left", row=1, col=1)
        if lv.get('issl'):
            fig.add_hline(y=lv['issl'] * rate, line_dash="dash", line_color="#ef5350", line_width=1.5,
                          annotation_text="ISSL (무효화)", annotation_position="bottom left", row=1, col=1)

    # --- 매수 마커: CHoCH가 확정된 봉의 저가 아래에 녹색 삼각형 ---
    if 'BULL_CHOCH_CONFIRMED' in df.columns:
        msk = df['BULL_CHOCH_CONFIRMED'].fillna(False).astype(bool)
        if bool(msk.any()):
            fig.add_trace(go.Scatter(x=df.loc[msk, 'time'], y=df.loc[msk, 'low'] * rate * 0.985,
                                     mode='markers', name='CHoCH 확정',
                                     marker=dict(symbol='triangle-up', size=13, color='#26a69a',
                                                 line=dict(width=1, color='#0b5e55'))), row=1, col=1)

    # --- 현재 진입 타점이면 손절·1차 익절 라인 표시 ---
    if plan is not None:
        fig.add_hline(y=plan['sl'] * rate, line_dash="dot", line_color="#ef5350", line_width=2,
                      annotation_text="SL 손절", annotation_position="right", row=1, col=1)
        tp_y = plan.get('tp_struct', plan['tp1'])
        fig.add_hline(y=tp_y * rate, line_dash="dot", line_color="#26a69a", line_width=2,
                      annotation_text="1차 익절(50%)", annotation_position="right", row=1, col=1)
    # --- ICT 수요 구간 (주봉 오더블록·FVG) ---
    for z in (zones or []):
        fig.add_hrect(y0=z['lo'] * rate, y1=z['hi'] * rate, fillcolor="#26a69a", opacity=0.12, line_width=0,
                      annotation_text=z['kind'], annotation_position="top left", row=1, col=1)
    # --- 추가 가로선 (매도 계산기: 평단가·트레일링 스탑 등) ---
    for y, label, color in (extra_lines or []):
        if y is not None and np.isfinite(y):
            fig.add_hline(y=y * rate, line_dash="dashdot", line_color=color, line_width=2,
                          annotation_text=label, annotation_position="right", row=1, col=1)

    colors = ['#26a69a' if r['close'] >= r['open'] else '#ef5350' for _, r in df.iterrows()]
    fig.add_trace(go.Bar(x=df['time'], y=df['volume'], marker_color=colors, name='Volume'), row=2, col=1)
    fig.update_layout(height=600, margin=dict(l=0, r=0, t=0, b=0), xaxis_rangeslider_visible=False)
    return fig

# ==========================================
# 결론 판정 (상세 화면·스크리너 공용): 🔴 절대 매수 금지 / ⏳ 관심 종목·관망 / 🟢 매수 실행
# ==========================================
VERDICT_STYLE = {
    "NO":   dict(title="🔴 절대 매수 금지", bg="#c62828"),
    "WAIT": dict(title="⏳ 관심 종목 / 관망", bg="#ef8f00"),
    "BUY":  dict(title="🟢 매수 실행", bg="#2e7d32"),
}


def action_verdict(info, plan=None):
    """기존 엔진 상태(entry·phase·MTF) → 초보자용 3단 결론. 🟢는 '바닥 셋업 후 돌파 확정(TRIGGER)'일 때만."""
    entry, ph = info.get('entry'), info.get('phase')
    smc = info.get('smc') or {}
    mtf = info.get('mtf') or {}
    trig = smc.get('trigger_level')

    def v(kind, *reasons, trigger=None):
        return dict(kind=kind, reasons=list(reasons), trigger=trigger)

    if entry == "WHIPSAW":
        return v("NO", "가짜 반등(휩쏘): 종가가 직전 저점(ISSL) 아래로 떨어졌습니다.", "다음 바닥 셋업이 나올 때까지 사지 않습니다.")
    if entry == "DOWN":
        return v("NO", "하락 구조 진행 중: 종가가 직전 저점 아래에 있습니다.", "떨어지는 칼날 구간입니다. 현금을 유지하세요.")
    if entry == "BLOCKED":
        return v("NO", "200일선 아래에서, 200일선이 아직 내려가는 중에 나온 반등입니다 (데드캣 경계).",
                 "200일선 하락이 멈추거나 가격이 200일선을 되찾을 때까지 기다립니다.")
    if entry == "TRIGGER":
        if info.get('extended'):
            return v("WAIT", "바닥 셋업 후 돌파는 확정됐지만 이미 많이 올랐습니다 (추격 구간).",
                     "돌파선 근처로 되돌림(리테스트)이 오면 아래 플랜의 2차 가격에서 삽니다.",
                     trigger=(smc.get('choch') or {}).get('level'))
        if plan and plan.get('blocked'):
            return v("WAIT", "바닥 셋업 후 돌파는 확정됐지만 손익비가 기준에 못 미칩니다.", plan['blocked'])
        return v("BUY", "바닥 셋업 → 돌파(Bullish CHoCH) 확정: 무릎 매수 조건을 모두 충족했습니다.",
                 "아래 플랜대로 1차를 지금 사고, 체결 즉시 공통 손절가에 스탑 주문을 걸어 두세요.")
    if entry == "BREAKOUT":
        return v("WAIT", "돌파는 했지만 그 전에 바닥 셋업이 없었습니다 — 무릎이 아니라 단순 추세 돌파입니다.",
                 "이 전략에서는 사지 않고 관망합니다. 다음 바닥 셋업을 기다리세요.")
    if entry == "PENDING":
        return v("WAIT", "돌파 직후입니다. 종가 안착(2봉 연속 돌파선 위)을 확인하는 중입니다.",
                 "안착이 확인되면 🟢 매수 실행으로 바뀝니다. 미리 사지 마세요.", trigger=trig)
    if entry == "SETUP":
        return v("WAIT", "바닥 셋업이 성립했습니다 → 관심 종목에 넣으세요. 아직 매수 신호가 아닙니다.",
                 "종가가 트리거 가격 위로 돌파·안착하면 🟢 매수 실행으로 바뀝니다.", trigger=trig)
    if ph == "하락" or mtf.get('state') == "bear":
        return v("NO", "하락 추세(역배열)입니다.", "구조 반전과 바닥 셋업이 나오기 전까지 현금을 유지합니다.", trigger=trig)
    if ph == "분배":
        return v("NO", "분배(고점권) 국면입니다. 신규 매수 금지.", "보유 중이면 💼 포지션 관리 탭에서 청산 계획을 확인하세요.")
    if ph == "매집":
        return v("WAIT", "매집 정황은 있지만 바닥 셋업·돌파 전입니다.", "관심 종목에 두고 셋업 성립과 트리거 돌파를 기다립니다.", trigger=trig)
    return v("WAIT", "상승 추세 진행 중이지만 새 매수 신호는 없습니다.", "추격하지 말고 다음 바닥 셋업이나 돌파선 리테스트를 기다립니다.", trigger=trig)


# ==========================================
# 2분할 진입 플랜 (1차 50% 돌파 확정 · 2차 50% 돌파선 리테스트 · 공통 손절 1개 · 1% 룰)
# ==========================================
SPLIT_COST = 0.0015            # 편도 수수료+슬리피지 가정 (손실 계산에 포함)
US_QTY_DEC_MAX = 4             # 미국 주식 소수점 매수 자릿수 (이 이하 자릿수는 '주', 그보다 많으면 코인 '개'로 표시)
CRYPTO_QTY_DEC = 8             # 코인 수량 자릿수 (업비트 등 최소 단위)


def _qty_floor(q, qty_dec):
    if qty_dec is None:
        return q
    f = 10 ** qty_dec
    return math.floor(q * f) / f


def entry_plan_2split(info, curr, equity, rate=1.0, qty_dec=0, override=None):
    """
    기본 플랜
      1차 50% : 돌파 확정 가격 (이미 확정이면 현재가, 아직이면 트리거 가격)
      2차 50% : 돌파선(IBSL) 리테스트 가격 — 돌파 후 되돌림에서 지지 확인 시
      손절    : 2차 가격 기준 구조적 손절 (직전 저점 아래 또는 1.5 ATR, 최소 0.8 ATR) — 두 차수 공통
      총액    : 두 차수 모두 체결 후 손절되면 총자산 × (1% × 신뢰도 보정) 만큼만 잃도록 역산, 한 종목 최대 30%
    가격은 원 단위(표시 통화)로 계산해 반환. 계산 불가면 None.
    """
    L, smc = info['levels'], info['smc']
    atr = L.get('atr_abs')
    if not atr or atr <= 0 or not curr:
        return None
    entry = info.get('entry')
    choch_lv = (smc.get('choch') or {}).get('level')
    if entry in ("TRIGGER", "BREAKOUT", "PENDING") and choch_lv:
        p1, base, stage1 = float(curr), float(choch_lv), "now"          # 이미 돌파: 1차는 지금, 2차는 돌파한 선 리테스트
    elif smc.get('trigger_level') and L.get('ibsl'):
        p1, base, stage1 = float(smc['trigger_level']), float(L['ibsl']), "trigger"
    else:
        return None
    p2 = base if base < p1 else p1 - 0.5 * atr
    ref = build_trade_plan(info, entry=p2, assume_trigger=True)
    ref1 = build_trade_plan(info, entry=p1, assume_trigger=(entry != "TRIGGER"))
    if not ref or not ref1:
        return None
    sl = ref['sl']
    if override:
        p1 = override.get('p1', p1 * rate) / rate
        p2 = override.get('p2', p2 * rate) / rate
        sl = override.get('sl', sl * rate) / rate
    conf = max(0.1, float(ref1.get('conf', 1.0)))
    risk_eff = RISK_PER_TRADE * conf
    P1, P2, SL = p1 * rate, p2 * rate, sl * rate
    if not (P1 > SL and P2 > SL):
        return dict(err="매수가가 손절가보다 낮거나 같습니다. 손절가 아래에서는 사지 않습니다.", p1=P1, p2=P2, sl=SL)
    loss = [(p * (1 + SPLIT_COST) - SL * (1 - SPLIT_COST)) / p for p in (P1, P2)]      # 금액 1원당 손절 시 손실 비율
    total = equity * risk_eff / 100 / (0.5 * loss[0] + 0.5 * loss[1])
    capped = total > equity * MAX_POSITION_PCT / 100
    total = min(total, equity * MAX_POSITION_PCT / 100)
    rows, cum_loss = [], 0.0
    for n, (p, lf) in enumerate(zip((P1, P2), loss), start=1):
        q = _qty_floor(total * 0.5 / p, qty_dec)
        amt = q * p
        cum_loss += amt * lf
        rows.append(dict(n=n, price=p, qty=q, amount=amt, cum_loss=cum_loss, cum_loss_pct=cum_loss / equity * 100))
    return dict(err=None, p1=P1, p2=P2, sl=SL, sl_pct=(SL / P1 - 1) * 100, rows=rows, stage1=stage1,
                total=sum(r['amount'] for r in rows), risk_eff=risk_eff, notes=ref1.get('notes', []), capped=capped,
                tp_struct=ref1['tp_struct'] * rate, tp_struct_label=ref1['tp_struct_label'],
                chase=(P2 < (base - 0.5 * atr) * rate))


def _won(v, small=False):
    return "-" if v is None else (f"₩{v:,.2f}" if small else f"₩{v:,.0f}")


def _qty_txt(q, qty_dec):
    if qty_dec == 0:
        return f"{int(q):,}주"
    if qty_dec is None:
        return f"{q:,.4f}"
    if qty_dec <= US_QTY_DEC_MAX:                   # 미국 주식 소수점 (토스 등)
        return f"{q:,.{qty_dec}f}주"
    return f"{q:,.{qty_dec}f}".rstrip("0").rstrip(".") + "개"


def render_action_banner(v, plan2, info, curr, rate, qty_dec, equity):
    """대시보드 최상단: 지금 해야 할 행동 1개 + 숫자 (가격·수량·손절)"""
    sty = VERDICT_STYLE[v['kind']]
    st.markdown(
        f"""<div style="background:{sty['bg']};color:#fff;padding:16px 20px;border-radius:12px;margin:2px 0 10px 0">
<div style="font-size:28px;font-weight:800;line-height:1.25">{sty['title']}</div>
<div style="font-size:15px;opacity:.95;margin-top:6px">{'<br>'.join(v['reasons'])}</div></div>""",
        unsafe_allow_html=True)
    small = curr * rate < 1000
    m1, m2, m3, m4 = st.columns(4)
    ok_plan = plan2 is not None and not plan2.get('err')
    if v['kind'] == "BUY" and ok_plan:
        r1, r2 = plan2['rows']
        m1.metric("① 1차 매수 (지금)", _won(r1['price'], small), _qty_txt(r1['qty'], qty_dec), delta_color="off")
        m2.metric("② 2차 매수 (리테스트)", _won(r2['price'], small), _qty_txt(r2['qty'], qty_dec), delta_color="off")
        m3.metric("③ 공통 손절가", _won(plan2['sl'], small), f"{plan2['sl_pct']:+.1f}%", delta_color="off")
        m4.metric("④ 총 매수 금액", _won(plan2['total']), f"총자산의 {plan2['total'] / equity * 100:.1f}%", delta_color="off")
    elif v['kind'] == "WAIT" and v.get('trigger') and ok_plan:
        m1.metric("① 트리거 가격 (종가 기준)", _won(v['trigger'] * rate, small), f"현재가 대비 {(v['trigger'] / curr - 1) * 100:+.1f}%", delta_color="off")
        m2.metric("② 돌파 시 손절가", _won(plan2['sl'], small), f"{plan2['sl_pct']:+.1f}%", delta_color="off")
        m3.metric("③ 돌파 시 총 매수 금액", _won(plan2['total']), f"총자산의 {plan2['total'] / equity * 100:.1f}%", delta_color="off")
        m4.metric("④ 지금 권장 비중", "0%")
    else:
        mtf = info.get('mtf') or {}
        L = info.get('levels') or {}
        m1.metric("① 지금 권장 비중", "0%")
        m2.metric("② 200일선", _won(mtf['sma'] * rate, small) if mtf.get('sma') else "N/A",
                  f"현재가 대비 {(mtf['sma'] / curr - 1) * 100:+.1f}%" if mtf.get('sma') else None, delta_color="off")
        m3.metric("③ 직전 저점 (ISSL)", _won(L['issl'] * rate, small) if L.get('issl') else "N/A")
        m4.metric("④ 트리거 가격", _won(v['trigger'] * rate, small) if v.get('trigger') else "N/A")
    st.caption("⚠️ 3박자·매집 점수는 **관심 종목 편입 기준**일 뿐입니다. 실제 매수는 이 결론이 **🟢 매수 실행**(바닥 셋업 후 돌파 확정)일 때만 하세요.")


def render_entry_plan(v, plan2, info, curr, rate, qty_dec, equity):
    """🧮 2분할 진입 플랜: 그대로 따라 할 주문표 + 가격 직접 수정"""
    expanded = v['kind'] == "BUY"
    with st.expander("🧮 2분할 진입 플랜 (1차 50% 돌파 확정 · 2차 50% 리테스트 · 공통 손절 · 1% 룰)", expanded=expanded):
        if plan2 is None:
            st.caption("돌파 기준선(IBSL)이나 ATR을 계산할 수 없어 플랜을 만들 수 없습니다.")
            return
        small = curr * rate < 1000
        if st.checkbox("가격 직접 수정", value=False, key=f"ep_edit_{info.get('bars')}"):
            e1, e2, e3 = st.columns(3)
            fmt_in = "%.2f" if small else "%.0f"
            o = dict(p1=e1.number_input("1차 매수가", min_value=0.0, value=float(round(plan2['p1'], 2)), format=fmt_in),
                     p2=e2.number_input("2차 매수가", min_value=0.0, value=float(round(plan2['p2'], 2)), format=fmt_in),
                     sl=e3.number_input("공통 손절가", min_value=0.0, value=float(round(plan2['sl'], 2)), format=fmt_in))
            o = {k: val for k, val in o.items() if isinstance(val, (int, float))}
            plan2 = entry_plan_2split(info, curr, equity, rate, qty_dec, override=o) or plan2
        if plan2.get('err'):
            st.error(plan2['err'])
            return
        r1, r2 = plan2['rows']
        when1 = "지금 (돌파 확정됨)" if plan2['stage1'] == "now" else "종가가 트리거 가격 위로 마감·안착한 다음 날"
        tbl = pd.DataFrame([
            {"차수": "1차 (50%)", "언제": when1, "가격": _won(r1['price'], small), "수량": _qty_txt(r1['qty'], qty_dec),
             "금액": _won(r1['amount']), "여기까지 산 뒤 손절 시 손실": f"{_won(r1['cum_loss'])} ({r1['cum_loss_pct']:.2f}%)"},
            {"차수": "2차 (50%)", "언제": "돌파선까지 되돌림 후 지지 확인 시 (지정가)", "가격": _won(r2['price'], small),
             "수량": _qty_txt(r2['qty'], qty_dec), "금액": _won(r2['amount']),
             "여기까지 산 뒤 손절 시 손실": f"{_won(r2['cum_loss'])} ({r2['cum_loss_pct']:.2f}%)"},
            {"차수": "🛑 손절", "언제": "1차 체결 즉시 스탑(감시) 주문 · 두 차수 공통", "가격": _won(plan2['sl'], small),
             "수량": "전량", "금액": "-", "여기까지 산 뒤 손절 시 손실": "-"},
        ])
        st.dataframe(tbl, hide_index=True)
        st.markdown(f"- 총 매수 금액: **{_won(plan2['total'])}** (총자산의 {plan2['total'] / equity * 100:.1f}%) · "
                    f"적용 리스크 {plan2['risk_eff']:.2f}%"
                    + (f" ({', '.join(plan2['notes'])})" if plan2['notes'] else ""))
        st.markdown(f"- 청산: 1차 50%는 **{_won(plan2['tp_struct'], small)}** ({plan2['tp_struct_label']})에서 익절, "
                    "잔여 50%는 트레일링 스탑(고점 − 3 ATR) — 💼 포지션 관리 탭에서 매일 확인")
        if plan2['capped']:
            st.warning(f"손절폭이 좁아 계산된 금액이 한 종목 최대 비중({MAX_POSITION_PCT:g}%)을 넘어서 {MAX_POSITION_PCT:g}%로 제한했습니다.")
        if plan2['chase']:
            st.warning("⚠️ 2차 가격이 돌파선보다 한참 아래입니다. 이건 리테스트가 아니라 하락 중 물타기입니다. 가격을 다시 확인하세요.")
        if any(r['qty'] == 0 for r in plan2['rows']):
            st.warning("한 차수의 수량이 0입니다 (1주 가격이 배정 금액보다 큼). 이 종목은 1회 진입으로 나누거나 매수하지 않는 편이 맞습니다.")
        if v['kind'] != "BUY":
            st.info("지금은 🟢 매수 실행 상태가 아니므로, 이 표는 **조건이 충족됐을 때의 예약 계획**입니다. 지금 주문하지 마세요.")
        st.caption(f"※ 손실에는 편도 {SPLIT_COST * 100:.2f}%의 수수료·슬리피지를 포함했습니다. 총자산은 사이드바에서 바꿀 수 있습니다.")



# ==========================================
# ICT 분석 (주봉 구간·방향 + 일봉 진입) — 마감 확정 봉만 사용, 미래 참조 없음
#   주봉: 수요 구간(Order Block · FVG), 딜링 레인지(할인/프리미엄 · OTE), 스윙 구조(HH/HL)
#   일봉: ① 유동성 스윕 후 회복 → ② CHoCH 확정 → ③ 돌파선 리테스트  (3단 분할 매수 사다리)
# ==========================================
ICT_W_PIV = 2             # 주봉 스윙 고점·저점 확정에 필요한 좌우 봉 수
ICT_RANGE_W = 52          # 딜링 레인지 기간 (주)
ICT_DISP_ATR = 1.2        # 변위(강한 양봉) 기준: 몸통 ≥ 주봉 ATR × 1.2, 직전 5주 고점 돌파
ICT_OB_LOOK = 3           # 변위 직전 몇 주 안의 마지막 음봉을 오더블록으로 볼지
ICT_MAX_ZONES = 4         # 화면에 보여 줄 수요 구간 최대 개수
ICT_ZONE_TOL_ATR = 1.0    # 수요 구간 '도달'로 볼 거리 (일봉 ATR 배수)
ICT_SWEEP_N = 10          # 1차(스윕 후 회복) 매수가 유효한 기간 (일봉)
ICT_SWEEP_MEM = 40        # CHoCH·리테스트 단계에서 인정할 스윕의 최대 경과 봉수
ICT_LIQ_N = 20            # 스윕 대상 유동성: 직전 20봉 최저가 (또는 현재가 아래 확정 저점)
ICT_FRACTAL = 2           # CHoCH 기준 반등 고점: 좌우 2봉 프랙털
ICT_DISCOUNT_SWEEP = 0.0  # 수요 구간이 없어도 스윕을 인정할 레인지 위치 (0 = 수요 구간 근처 스윕만 인정, 예: 0.15 = 하위 15%)
ICT_RETEST_N = 5          # 3차(리테스트) 매수가 유효한 기간 (일봉)
ICT_RETEST_ATR = 0.5      # 돌파선 ± 이 배수(ATR) 안으로 되돌아오면 리테스트
ICT_STOP_BUF_ATR = 0.25   # 무효화 손절 = 스윕 저점 − 0.25 ATR
ICT_LADDER = (0.3, 0.3, 0.4)   # 1차 스윕 후 회복 / 2차 CHoCH / 3차 리테스트
WEEKLY_DOWN_RISK_MULT = 0.5   # 주봉 하락 추세(주봉 CHoCH 전)일 때 일봉 매수 리스크 배수 (0으로 바꾸면 이 구간 매수 차단)
WEEKLY_SWEEP_RISK_MULT = 0.75 # 주봉 저점 스윕 후 회복(추세 전환 초기)일 때 리스크 배수
W_SWEEP_N = 8                 # 주봉 스윕을 인정할 최근 주 수
ICT_WEAK_FILTER_MULT = 0.75   # 보조 필터 충족이 2개 미만이면 리스크 배수
ICT_SWEEP_EXT_ATR = 1.0       # 1차 신호 종가보다 이 배수(ATR) 이상 오르면 '1차 되돌림 대기'(추격 방지)
GRADE_B_MULT = 0.5            # B급(조건부) 신호의 리스크 배수 (GRADE_ACTIVE일 때만)
GRADE_ACTIVE = False          # 등급으로 매수 여부·비중을 바꿀지. 검증(2019~2026, 43종목)에서 A급이 B급보다 나빠 지금은 참고 표시만
ICT_MIN_STOP_PCT = 3.0        # 손절폭 최소값(%): 검증에서 손절폭 3% 이하 신호는 손절 비율 73~87%로 잡음에 털림
ICT_MIN_STOP_ATR = 1.0        # 손절폭 최소값(ATR 배수) — 둘 중 넓은 쪽 적용
ICT_COOLDOWN_N = 20           # 같은 종목에서 손절(무효화)이 난 뒤 이 봉수 동안 새 매수 신호를 내지 않음 (검증: 손절 직후 재진입 손절률 85% 이상)
GRADE_RULES = dict(pos=0.5, filters=4, space_r=2.0, stop_pct=10.0)   # A급 조건: 레인지 50% 아래 · 보조 필터 4개 · 위 공간 2R · 손절폭 10%


def to_weekly(d):
    """일봉(마감 봉) → 주봉. 마지막 주는 아직 진행 중일 수 있어 제외 (추가 API 호출 없음)"""
    x = d[['time', 'open', 'high', 'low', 'close', 'volume']].copy()
    x['time'] = pd.to_datetime(x['time'])
    w = (x.set_index('time').resample('W-MON', label='left', closed='left')
         .agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
         .dropna(subset=['close']).reset_index())
    return w.iloc[:-1].reset_index(drop=True) if len(w) > 1 else w


def _weekly_zones(w):
    """주봉 수요 구간: ① 오더블록(강한 상승 변위 직전의 마지막 음봉) ② 상승 FVG(3봉 가격 공백). 이후 주봉 종가가 아래로 깨면 무효."""
    if len(w) < 12:
        return [], None
    o, h, l, c = (w[k].to_numpy(dtype=float) for k in ('open', 'high', 'low', 'close'))
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    atr_w = pd.Series(tr).rolling(10, min_periods=5).mean().to_numpy()
    tm = pd.to_datetime(w['time']).to_numpy()
    zones = []
    for i in range(6, len(w)):
        if not np.isfinite(atr_w[i]):
            continue
        if c[i] - o[i] >= ICT_DISP_ATR * atr_w[i] and c[i] > h[i - 5:i].max():
            j = next((k for k in range(i - 1, max(i - 1 - ICT_OB_LOOK, -1), -1) if c[k] < o[k]), None)
            if j is not None:
                zones.append(dict(kind="주봉 오더블록", lo=float(l[j]), hi=float(max(o[j], c[j])), i=i, t=tm[j],
                                  known=tm[i] + np.timedelta64(7, 'D')))
        if i >= 2 and l[i] > h[i - 2] and (l[i] - h[i - 2]) >= 0.3 * atr_w[i] and c[i - 1] > o[i - 1]:
            zones.append(dict(kind="주봉 FVG", lo=float(h[i - 2]), hi=float(l[i]), i=i, t=tm[i - 1],
                              known=tm[i] + np.timedelta64(7, 'D')))
    valid = []
    for z in zones:
        after = c[z['i'] + 1:]
        if not (after < z['lo']).any():          # 주봉 종가가 구간 아래로 마감한 적이 없어야 유효
            valid.append(z)
    valid.sort(key=lambda z: z['hi'], reverse=True)
    merged = []                                   # 겹치는 구간은 하나로
    for z in valid:
        if merged and z['hi'] >= merged[-1]['lo'] and z['lo'] <= merged[-1]['hi']:
            m = merged[-1]
            m.update(lo=min(m['lo'], z['lo']), hi=max(m['hi'], z['hi']), known=min(m['known'], z['known']),
                     kind=m['kind'] if m['kind'] == z['kind'] else "주봉 OB+FVG")
        else:
            merged.append(dict(z))
    return merged, (float(atr_w[-1]) if np.isfinite(atr_w[-1]) else None)


def _weekly_structure(w, k=ICT_W_PIV):
    """주봉 확정 스윙으로 방향 판단: HH+HL = 상승, LH+LL = 하락, 그 외 = 전환·횡보"""
    h, l = w['high'].to_numpy(dtype=float), w['low'].to_numpy(dtype=float)
    n = len(w)
    ph = [i for i in range(k, n - k) if h[i] == h[i - k:i + k + 1].max()]
    pl = [i for i in range(k, n - k) if l[i] == l[i - k:i + k + 1].min()]
    if len(ph) < 2 or len(pl) < 2:
        return dict(bias="판단 불가", hh=None, hl=None)
    hh, hl = h[ph[-1]] > h[ph[-2]], l[pl[-1]] > l[pl[-2]]
    bias = "상승" if (hh and hl) else "하락" if (not hh and not hl) else "전환·횡보"
    return dict(bias=bias, hh=bool(hh), hl=bool(hl), last_ph=float(h[ph[-1]]), last_pl=float(l[pl[-1]]))


W_STATE_META = {
    "W_UP":    dict(label="상승 추세 (주봉 HH·HL)", short="상승", icon="🟢"),
    "W_CHOCH": dict(label="추세 전환 시도 (주봉 CHoCH)", short="전환 시도", icon="🔵"),
    "W_SWEEP": dict(label="하락 추세 · 주봉 저점 스윕 후 회복 (전환 초기 신호)", short="하락·스윕", icon="🟡"),
    "W_DOWN":  dict(label="하락 추세 진행 중 (주봉 CHoCH 전)", short="하락", icon="🔴"),
    "W_RANGE": dict(label="횡보·방향 불분명", short="횡보", icon="⚪"),
    "W_NA":    dict(label="판단 불가 (주봉 데이터 부족)", short="-", icon="⚪"),
}


def weekly_ict(w, px):
    """
    주봉 ICT로 장기 추세 판정 (완성된 주봉만 사용).
      W_UP    : 최근 확정 스윙 고점·저점이 모두 상승 (HH·HL) — 상승 추세
      W_CHOCH : 하락 구조의 마지막 반등 고점(Lower High)을 주봉 종가로 돌파 — 추세 전환 시도
      W_SWEEP : 하락 구조에서 최근 주봉이 직전 스윙 저점을 쓸고 종가 회복 — 전환 초기 신호(아직 CHoCH 전)
      W_DOWN  : 하락 구조 진행 중 (CHoCH 전)
      W_RANGE : 그 외
    level = 주봉 CHoCH 기준선(마지막 Lower High), dist = 현재가 대비 거리
    """
    out = dict(state="W_NA", level=None, dist=None, sweep=None, choch_time=None, mult=1.0)
    n = len(w)
    if n < 16:
        return out
    h, l, c = (w[k].to_numpy(dtype=float) for k in ('high', 'low', 'close'))
    tm = pd.to_datetime(w['time']).to_numpy()
    pc = np.r_[c[0], c[:-1]]
    atr_w = pd.Series(np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))).rolling(10, min_periods=5).mean().to_numpy()
    k = ICT_W_PIV
    fh = [i for i in range(k, n - k) if h[i] == h[i - k:i + k + 1].max()]
    fl = [i for i in range(k, n - k) if l[i] == l[i - k:i + k + 1].min()]
    if len(fh) < 2 or len(fl) < 2:
        return out
    hh, hl = h[fh[-1]] > h[fh[-2]], l[fl[-1]] > l[fl[-2]]
    bos_down = px < l[fl[-1]]                       # 마지막 확정 스윙 저점 아래 → 하락 BOS
    # 마지막 상승 구간(직전 스윙 저점 → 마지막 스윙 고점)의 50% 아래로 되밀림 → 그 상승 추세는 무너진 것으로 봄
    lows_before = [i for i in fl if i < fh[-1]]
    leg_lo = l[lows_before[-1]] if lows_before else l[:fh[-1] + 1].min()
    deep_retrace = px < leg_lo + 0.5 * (h[fh[-1]] - leg_lo)
    down = (not hh) or bos_down or deep_retrace
    # 전환 기준선: 마지막 스윙 고점이 낮아진 고점(LH)이면 그것, 고점에서 쉬지 않고 빠져 LH가 아직 없으면 최근 10주 최고가
    lh_idx = next((fh[j] for j in range(len(fh) - 1, 0, -1) if h[fh[j]] < h[fh[j - 1]] and fh[j] >= fh[-1]), None)
    if lh_idx is not None:
        lv, src, base_i = float(h[lh_idx]), "마지막 반등 고점(LH)", lh_idx
    elif down:
        # 마지막 스윙 고점 이후의 최저 주봉(바닥)을 찾고, 그 직전 10주 안의 최고가를 기준선으로 고정 (반등을 따라 움직이지 않음)
        top = fh[-1]
        bot = top + 1 + int(np.argmin(l[top + 1:])) if top + 1 < n else n - 1
        lo_w = max(top + 1, bot - 10)
        base_i = lo_w + int(np.argmax(h[lo_w:bot])) if bot > lo_w else top
        lv, src = float(h[base_i]), "바닥 직전 반등 고점"
    else:
        lv, src, base_i = None, None, None
    choch = None
    if lv is not None:
        out.update(level=lv, dist=lv / px - 1, level_src=src)
        j = next((i for i in range(base_i + 1, n) if c[i] > lv), None)
        if j is not None:
            choch = j
            out['choch_time'] = tm[j]
    # 주봉 스윕: 최근 W_SWEEP_N주 안에 직전 확정 스윙 저점 − 0.25 ATR 아래를 찍고, 2주 안에 그 위로 종가 회복
    sweep = None
    for t in range(n - 1, max(n - 1 - W_SWEEP_N, k), -1):
        prior = [l[i] for i in fl if i + k < t]
        if not prior or not np.isfinite(atr_w[t]):
            continue
        lv = prior[-1]
        if l[t] < lv - 0.25 * atr_w[t]:
            r = next((r for r in range(t, min(t + 2, n)) if c[r] > lv), None)
            if r is not None:
                sweep = dict(time=tm[t], level=float(lv), low=float(l[t]))
                break
    out['sweep'] = sweep
    if hh and hl and not bos_down and not deep_retrace:
        state = "W_UP"
    elif choch is not None and c[-1] > out['level'] * 0.97:
        state = "W_CHOCH"
    elif down:
        state = "W_SWEEP" if sweep else "W_DOWN"
    else:
        state = "W_RANGE"
    out['state'] = state
    out['mult'] = WEEKLY_DOWN_RISK_MULT if state == "W_DOWN" else WEEKLY_SWEEP_RISK_MULT if state == "W_SWEEP" else 1.0
    return out


def ict_analysis(df, info):
    """
    df   : add_indicators 결과 (마지막 봉은 진행 중일 수 있음 → 판정은 마감 봉만)
    info : analyze_market_cycle 결과 (기존 CHoCH·셋업·레벨 재사용)
    반환 : 단계(stage)와 근거, 사다리 계산용 가격
      INVALID   스윕 저점 아래 종가 → 셋업 실패
      RETEST    CHoCH 후 돌파선 되돌림 지지 → 3차
      CHOCH     스윕(또는 바닥 셋업) 후 CHoCH 확정 → 2차
      CHOCH_EXT CHoCH 후 이미 많이 오름 → 3차 리테스트 대기
      POST      CHoCH 후 2차 시점 지남 → 3차 리테스트 대기
      PENDING   CHoCH 돌파 직후 안착 확인 중
      SWEEP     수요 구간·할인 구간에서 직전 저점을 쓸고 종가 회복 → 1차
      IN_ZONE   주봉 수요 구간 도달, 스윕 대기
      NO_ZONE   ICT 매수 맥락 없음
    """
    d = closed_bars(df)                # 진행 중인 봉만 제외 (장 마감 후·백테스트는 마지막 봉까지 사용)
    n = len(d)
    out = dict(stage="NO_ZONE", zones=[], weekly={}, range=None, sweep=None, choch=None, retest=None,
               filters={}, invalid=None, ok=False)
    if n < 60 or 'ISSL' not in d.columns:
        return out
    lo, hi, cl = (d[k].to_numpy(dtype=float) for k in ('low', 'high', 'close'))
    vol = d['volume'].to_numpy(dtype=float)
    issl = d['ISSL'].to_numpy(dtype=float)
    atr = d['ATR_ABS'].to_numpy(dtype=float) if 'ATR_ABS' in d.columns else np.full(n, np.nan)
    tm = pd.to_datetime(d['time']).to_numpy()
    atr_now = float(atr[-1]) if np.isfinite(atr[-1]) else float(np.nanmedian(atr[-20:]))
    px = float(cl[-1])

    # ---------- 주봉: 구간 · 레인지 · 방향 ----------
    w = to_weekly(d)
    zones, _atr_w = _weekly_zones(w)
    wk = w.tail(ICT_RANGE_W)
    rng = None
    if len(wk) >= 8:
        r_hi, r_lo = float(wk['high'].max()), float(wk['low'].min())
        if r_hi > r_lo:
            R = r_hi - r_lo
            rng = dict(hi=r_hi, lo=r_lo, eq=r_lo + R / 2, pos=(px - r_lo) / R,
                       ote_hi=r_hi - 0.62 * R, ote_lo=r_hi - 0.79 * R)
    out.update(weekly=_weekly_structure(w) if len(w) >= 8 else dict(bias="판단 불가"), range=rng, wict=weekly_ict(w, px))
    below = [z for z in zones if z['lo'] <= px + ICT_ZONE_TOL_ATR * atr_now]       # 현재가 아래(또는 걸친) 구간
    out['zones'] = sorted(below, key=lambda z: z['hi'], reverse=True)[:ICT_MAX_ZONES]
    near = [z for z in out['zones'] if z['lo'] - ICT_ZONE_TOL_ATR * atr_now <= px <= z['hi'] + ICT_ZONE_TOL_ATR * atr_now]

    def in_context(t, close_back):
        """봉 t가 (그때 이미 알려진) 수요 구간을 건드렸는가: 저가가 구간 상단 근처 아래로 내려왔고, 회복 종가가 구간 하단 근처 위
        (구간을 꼬리로 뚫고 내려갔다가 구간 안·위로 돌아온 '구간 스윕'도 포함)"""
        a = atr[t] if np.isfinite(atr[t]) else atr_now
        for z in zones:
            if z['known'] <= tm[t] and lo[t] <= z['hi'] + ICT_ZONE_TOL_ATR * a and close_back >= z['lo'] - ICT_ZONE_TOL_ATR * a:
                return z['kind']
        if ICT_DISCOUNT_SWEEP and rng and (lo[t] - rng['lo']) / (rng['hi'] - rng['lo']) < ICT_DISCOUNT_SWEEP:
            return "주봉 깊은 할인 구간"
        return None

    # ---------- 일봉 ①: 유동성 스윕 후 회복 (가장 최근 것) ----------
    # 유동성 = 그 봉 직전까지 '확정된' 스윙 저점 (좌우 2봉이 더 높았던 저점, 최근 20봉 안) 또는 직전 종가 아래의 확정 저점(ISSL)
    #   · 저점 아래로 0.25 ATR 이상 찌르고, 3봉 안에 그 저점 + 0.1 ATR 위에서 종가 회복해야 스윕으로 인정 (하락 중 잔파동 배제)
    kf = ICT_FRACTAL
    is_fl = np.zeros(n, dtype=bool)
    for i in range(kf, n - kf):
        is_fl[i] = lo[i] == lo[i - kf:i + kf + 1].min()
    sweeps = []                      # 최근 → 과거 순서로 찾은 스윕들 (손절 이력 확인용)
    t = n - 1
    t_end = max(n - 1 - ICT_SWEEP_MEM - ICT_COOLDOWN_N, ICT_LIQ_N)
    while t > t_end:
        a = atr[t] if np.isfinite(atr[t]) else atr_now
        cands = [float(lo[i]) for i in range(t - ICT_LIQ_N, t - kf) if is_fl[i]]          # t 시점에 확정된 프랙털 저점만
        if np.isfinite(issl[t - 1]) and issl[t - 1] < cl[t - 1]:
            cands.append(float(issl[t - 1]))
        hit = sorted(c_ for c_ in cands if lo[t] < c_ - 0.25 * a and c_ < cl[t - 1])     # 직전 종가가 저점 위에 있었어야 '위에서 아래로 쓸고 회복'
        lvl, rec = None, None
        for c_ in hit:                                   # 쓸린 저점 중 가장 깊은 것부터: 그 위로 3봉 안에 종가 회복하면 스윕
            rec = next((r for r in range(t, min(t + 3, n)) if cl[r] > c_ + 0.1 * a), None)
            if rec is not None:
                lvl = c_
                break
        ctx = in_context(t, float(cl[rec])) if rec is not None else None
        if ctx is None:
            t -= 1
            continue
        s_low = float(lo[t:rec + 1].min())
        rc = float(cl[rec])
        stop = min(s_low - ICT_STOP_BUF_ATR * a, rc - ICT_MIN_STOP_ATR * a, rc * (1 - ICT_MIN_STOP_PCT / 100))   # 최소 손절폭 보장
        after = cl[rec + 1:]
        k_inv = rec + 1 + int(np.argmax(after < stop)) if (after < stop).any() else None
        sweeps.append(dict(idx=t, rec=rec, level=float(lvl), low=s_low, stop=stop, ctx=ctx,
                           time=tm[t], rec_close=rc, age=n - 1 - rec, inv=k_inv,
                           vol_spike=bool(vol[t] >= 1.5 * np.nanmedian(vol[max(0, t - 20):t])) if t > 5 else False))
        t -= 1
    sweep = next((x for x in sweeps if x['idx'] >= n - 1 - ICT_SWEEP_MEM), None) if sweeps else None
    if sweep is not None:
        sweep = dict(sweep)
        if sweep['inv'] is not None:                                      # 스윕 저점(손절가) 아래 종가 → 무효화
            k = sweep['inv']
            out['invalid'] = dict(time=tm[k], age=n - 1 - k, stop=sweep['stop'])
            sweep = None
    # 손절(무효화) 이력: 지금 진행 중인 셋업 이전에 ICT_COOLDOWN_N봉 안에서 손절이 났는가
    cur_idx = sweep['idx'] if sweep else n
    inv_ages = [n - 1 - x['inv'] for x in sweeps if x['inv'] is not None and x['inv'] <= cur_idx]
    out['last_stop'] = min(inv_ages) if inv_ages else None
    out['sweep'] = sweep

    # ---------- 일봉 ②: CHoCH (스윕 이후, 또는 기존 엔진의 바닥 셋업 트리거) ----------
    smc = info.get('smc') or {}
    choch = None
    if sweep is not None:
        s0, k = sweep['idx'], ICT_FRACTAL
        fh = [i for i in range(max(k, s0 - 30), s0 - k + 1) if hi[i] == hi[i - k:i + k + 1].max()]     # 스윕 시점에 확정된 반등 고점
        lv = float(hi[fh[-1]]) if fh else float(hi[max(0, s0 - 10):s0].max())
        a = atr[s0] if np.isfinite(atr[s0]) else atr_now
        brk = [j for j in range(sweep['rec'], n) if cl[j] > lv + 0.1 * a]
        if brk:
            j = brk[0]
            since = cl[j:]
            choch = dict(idx=j, level=lv, age=n - 1 - j, bars_above=len(since), held=bool((since > lv).all()),
                         time=tm[j], src="스윕 후 CHoCH")
        sweep['choch_level'] = lv
    if choch is None and info.get('entry') in ("TRIGGER", "PENDING") and smc.get('choch') and smc.get('setup_ctx'):
        c0 = smc['choch']
        j = n - 1 - c0['age']
        choch = dict(idx=j, level=float(c0['level']), age=c0['age'], bars_above=c0['bars_above'], held=c0['held'],
                     time=tm[j] if 0 <= j < n else None, src="바닥 셋업 후 CHoCH")
    if choch is not None and not choch['held']:
        choch = None                                                          # 돌파선 아래로 되밀리면 CHoCH 무효
    out['choch'] = choch
    if sweep is not None and choch is None and px <= sweep['level']:
        out['sweep_failing'] = sweep
        sweep = None
        out['sweep'] = None

    # ---------- 일봉 ③: 돌파선 리테스트 ----------
    retest = None
    if choch and choch['bars_above'] >= CHOCH_SETTLE:
        lv, a = choch['level'], atr_now
        for t in range(n - 1, max(choch['idx'] + 1, n - 1 - ICT_RETEST_N), -1):
            if lo[t] <= lv + ICT_RETEST_ATR * a and cl[t] >= lv - ICT_RETEST_ATR * a:
                retest = dict(idx=t, time=tm[t], level=lv, age=n - 1 - t)
                break
        if retest and px < lv - ICT_RETEST_ATR * a:
            retest = None
    out['retest'] = retest

    # ---------- 보조 필터 (점수 합산이 아니라 '확인용' 체크) ----------
    win = d.tail(20)
    anyc = lambda col: bool(col in win.columns and win[col].fillna(False).astype(bool).any())
    out['filters'] = {
        "RSI 강세 다이버전스": anyc('BULL_DIV_RSI'),
        "MACD 강세 다이버전스": anyc('BULL_DIV_MACD'),
        "OBV 강세 (가격 대비 거래량 유입)": anyc('BULL_OBV'),
        "스윕 시 거래량 급증 (투매 흔적)": bool(sweep and sweep['vol_spike']),
        "일봉 저점 상승 (HL)": bool('HL_STATE' in d.columns and bool(d['HL_STATE'].iloc[-1])),
        "변동성 수축 (횡보 다지기)": anyc('CONSOLIDATING'),
        "주봉 할인 구간 (레인지 50% 아래)": bool(rng and rng['pos'] < 0.5),
    }

    # ---------- 단계 결정 ----------
    ext = bool(choch and (px - choch['level']) > CHOCH_EXT_ATR * atr_now)
    if out['invalid'] and out['invalid']['age'] <= ICT_SWEEP_N and not choch and px < out['invalid']['stop']:
        stage = "INVALID"
    elif choch and choch['bars_above'] < CHOCH_SETTLE:
        stage = "PENDING"
    elif choch and retest:
        stage = "RETEST"
    elif choch and ext:
        stage = "CHOCH_EXT"
    elif choch and choch['age'] <= CHOCH_WINDOW:
        stage = "CHOCH"
    elif choch:
        stage = "POST"
    elif sweep and sweep['age'] <= ICT_SWEEP_N and px > sweep['level']:
        stage = "SWEEP_EXT" if px > sweep['rec_close'] + ICT_SWEEP_EXT_ATR * atr_now else "SWEEP"
    elif out.get('sweep_failing') and (n - 1 - out['sweep_failing']['rec']) <= ICT_SWEEP_N and px > out['sweep_failing']['stop']:
        stage = "RECLAIM_FAIL"                                  # 스윕 후 회복했다가 다시 기준선 아래로 마감 → 재회복 대기
    elif near:
        stage = "IN_ZONE"
    else:
        stage = "NO_ZONE"
    if stage in ("SWEEP", "SWEEP_EXT", "CHOCH", "RETEST", "PENDING") and out.get('last_stop') is not None \
            and out['last_stop'] <= ICT_COOLDOWN_N and not (out['invalid'] and stage == "INVALID"):
        out['cooldown'] = dict(age=out['last_stop'], left=ICT_COOLDOWN_N - out['last_stop'], was=stage)
        stage = "COOLDOWN"
    liq_c = [float(lo[n - ICT_LIQ_N:n].min())] + ([float(issl[-1])] if np.isfinite(issl[-1]) and issl[-1] < px else [])
    out.update(stage=stage, near=near, ext=ext, px=px, atr=atr_now, ok=True, liq=max(liq_c))
    return out


ICT_STAGE_LABEL = {
    "SWEEP": "🟢 1차: 스윕 후 회복", "CHOCH": "🟢 2차: CHoCH 확정", "RETEST": "🟢 3차: 리테스트",
    "PENDING": "⏳ CHoCH 안착 대기", "CHOCH_EXT": "⏳ 3차 리테스트 대기", "POST": "⏳ 3차 리테스트 대기",
    "SWEEP_EXT": "⏳ 1차 되돌림 대기", "RECLAIM_FAIL": "🪤 회복 실패 · 재회복 대기", "COOLDOWN": "🧊 손절 후 휴식",
    "IN_ZONE": "🎯 수요 구간 도달", "INVALID": "🪤 무효화(스윕 실패)",
}


def _ict_stop(ict, L):
    """사다리 공통 손절가와 근거 (종목 통화)"""
    a, px = ict['atr'], ict['px']
    s, sf = ict.get('sweep'), ict.get('sweep_failing')
    if s:
        return s['stop'], "스윕 저점 − 0.25 ATR"
    if ict.get('stage') == "RECLAIM_FAIL" and sf:
        return sf['stop'], "직전 스윕 저점 − 0.25 ATR"
    if ict.get('near'):
        z = ict['near'][0]
        return z['lo'] - 0.5 * a, f"{z['kind']} 하단 − 0.5 ATR (예상)"
    if L.get('issl') and L['issl'] < px:
        return L['issl'] - ICT_STOP_BUF_ATR * a, "직전 저점 − 0.25 ATR (예상)"
    return px - 2 * a, "현재가 − 2 ATR (예상)"


# ---------- 신호 등급 (A·B·C): 구조가 맞는지에 더해 위치·확인·여유 공간까지 함께 봄 ----------
GRADE_NAMES = dict(pos="위치", seq="순서", wk="주봉", conf="확인", space="공간", stop="손절", hold="유지", flow="흐름")
GRADE_CORE = ("pos", "seq", "wk")


def ict_checks(info, ict):
    """A급 8개 조건 중 섹터 흐름을 뺀 7개. [(키, 통과 여부(None=판단 불가), 설명)]"""
    L = info.get('levels') or {}
    px, a = ict['px'], ict['atr']
    s = ict.get('sweep')
    rg = ict.get('range')
    wst = (ict.get('wict') or {}).get('state', "W_NA")
    stop, _src = _ict_stop(ict, L)
    out = []
    if rg:
        base = s['low'] if s else px
        pos = (base - rg['lo']) / (rg['hi'] - rg['lo'])
        in_zone = bool(s and s.get('ctx')) or bool(ict.get('near'))
        out.append(("pos", bool(pos < GRADE_RULES['pos'] and in_zone),
                    f"{'스윕 저점' if s else '현재가'} 레인지 {pos * 100:.0f}%" + (" · 수요 구간" if in_zone else " · 수요 구간 밖")))
    else:
        out.append(("pos", None, "레인지 계산 불가"))
    out.append(("seq", bool(s), "1차 스윕부터 시작" if s else "스윕 없이 시작 (1차 없음)"))
    out.append(("wk", wst in ("W_UP", "W_CHOCH", "W_SWEEP"), W_STATE_META.get(wst, W_STATE_META["W_NA"])['short']))
    n_ok = sum(bool(x) for x in (ict.get('filters') or {}).values())
    out.append(("conf", n_ok >= GRADE_RULES['filters'], f"보조 필터 {n_ok}/7"))
    risk = px - stop
    lv = [v for _lab, v in (L.get('resistances') or []) if v is not None and np.isfinite(v)]
    lv += [x for x in (L.get('ibsl'), (ict.get('wict') or {}).get('level'), (rg or {}).get('hi')) if x]
    above = [v for v in lv if v > px + 0.2 * a]
    if risk <= 0:
        out.append(("space", False, "손절가가 현재가 위"))
    elif not above:
        out.append(("space", True, "위쪽 저항 없음"))
    else:
        r_mult = (min(above) - px) / risk
        out.append(("space", r_mult >= GRADE_RULES['space_r'], f"첫 저항까지 손절폭의 {r_mult:.1f}배"))
    sp = risk / px * 100 if px else 0
    out.append(("stop", 0 < sp <= GRADE_RULES['stop_pct'], f"손절폭 {sp:.1f}%"))
    stg = ict.get('stage')
    if stg == "SWEEP":
        out.append(("hold", s['age'] >= 1, "다음 날 회복선 유지 확인" if s['age'] >= 1 else "오늘 막 회복 (내일 유지 확인 필요)"))
    elif stg in ("CHOCH", "RETEST"):
        out.append(("hold", True, "안착·지지 확인"))
    else:
        out.append(("hold", None, "-"))
    return out


def grade_from(checks):
    """체크 결과 → (등급, 빠진 조건 이름들). A: 핵심 3개(위치·순서·주봉) 통과 + 나머지 실패 1개 이하
    C: 핵심 2개 이상 실패 또는 전체 4개 이상 실패 · 그 외 B (판단 불가 항목은 계산에서 뺌)"""
    fails = [k for k, ok, _ in checks if ok is False]
    core_f = [k for k in fails if k in GRADE_CORE]
    if not core_f and len(fails) <= 1:
        g = "A"
    elif len(core_f) >= 2 or len(fails) >= 4:
        g = "C"
    else:
        g = "B"
    return g, [GRADE_NAMES[k] for k in fails]


def _disp_fmt(rate, curr):
    """결론 문장의 가격을 화면 통화(원)로 표시"""
    if not rate or rate == 1.0:
        return _fmt_price
    small = curr * rate < 1000
    return lambda x: _won(x * rate, small)


def ict_verdict(info, ict, plan_trig=None, fmt=None, flow_ok=None):
    """ICT 단계를 우선으로 한 결론 + 신호 등급. ICT 맥락이 없으면 기존 판정을 쓰되 🟢는 내지 않음 (ICT가 메인).
    fmt: 가격 표시 함수(화면 통화) · flow_ok: 섹터 자금 흐름 상위 여부(스크리너에서만, 없으면 판단 불가)"""
    F = fmt or _fmt_price
    stg = ict.get('stage') if ict else "NO_ZONE"
    wi = (ict or {}).get('wict') or {}
    wst = wi.get('state', "W_NA")
    blocked = wst == "W_DOWN" and WEEKLY_DOWN_RISK_MULT <= 0

    def v(kind, tranche, *reasons, trigger=None, **extra):
        rs = list(reasons)
        if kind == "BUY" and wst == "W_DOWN":
            rs.append(f"⚠️ 주봉은 아직 하락 추세(주봉 CHoCH 전)라 단기 반등일 수 있습니다 → 리스크 ×{WEEKLY_DOWN_RISK_MULT:g}"
                      + (f" · 주봉 전환 기준선 {F(wi['level'])} ({wi['dist'] * 100:+.1f}%)" if wi.get('level') else ""))
        elif kind == "BUY" and wst == "W_SWEEP":
            rs.append(f"주봉 저점 스윕 후 회복(전환 초기 신호) → 리스크 ×{WEEKLY_SWEEP_RISK_MULT:g}")
        out = dict(kind=kind, tranche=tranche, reasons=rs, trigger=trigger, grade=None, checks=None, missing=[])
        out.update(extra)
        return out

    def graded(res):
        """BUY 결론에 등급을 매겨 A는 그대로, B는 조건부(리스크 절반), C는 매수 보류로 바꿈"""
        chk = ict_checks(info, ict) + [("flow", flow_ok, "섹터 자금 흐름 상위" if flow_ok else ("섹터 흐름 하위" if flow_ok is False else "스크리너에서 확인"))]
        g, miss = grade_from(chk)
        res.update(grade=g, checks=chk, missing=miss)
        if not GRADE_ACTIVE:
            res['grade_ref'] = True
            res['reasons'].append(f"📋 참고 등급 {g}" + (f" (빠진 조건: {', '.join(miss)})" if miss else "")
                                  + " — 검증 결과 등급이 성과를 가르지 못해 매수 판단에는 쓰지 않습니다.")
            return res
        if g == "A":
            res['reasons'].insert(0, "🅰️ A급 신호: 위치·순서·주봉이 모두 맞고 확인 조건도 충분합니다.")
        elif g == "B":
            res['reasons'].insert(0, f"🟡 B급(조건부): 빠진 조건 — {', '.join(miss)} → 리스크를 ×{GRADE_B_MULT:g}로 줄여 진입하거나, 조건이 채워질 때까지 기다립니다.")
        else:
            res.update(kind="WAIT", tranche=None)
            res['reasons'] = [f"C급: 구조(ICT 단계)는 맞지만 조건이 부족합니다 — 빠진 조건: {', '.join(miss)}",
                              "이 신호로는 사지 않습니다. 다음 단계(CHoCH 돌파 등)나 더 좋은 자리를 기다립니다."]
        return res

    if stg in ("SWEEP", "CHOCH", "RETEST") and blocked:
        return v("NO", None, "일봉 ICT 매수 조건은 나왔지만 주봉이 하락 추세(CHoCH 전)라 설정상 매수를 차단합니다.",
                 "WEEKLY_DOWN_RISK_MULT 값을 0보다 크게 바꾸면 리스크를 줄여 매수할 수 있습니다.")
    if stg == "COOLDOWN":
        cd = ict['cooldown']
        return v("WAIT", None, f"최근 {cd['age']}봉 전에 이 종목에서 손절(스윕 저점 이탈)이 났습니다 — 지금 신호({ICT_STAGE_LABEL.get(cd['was'], cd['was'])})는 건너뜁니다.",
                 f"앞으로 {cd['left']}봉 동안은 새로 사지 않습니다. 검증 결과 손절 직후 다시 들어간 매수는 85% 이상 다시 손절됐습니다.")
    if stg == "INVALID":
        return v("NO", None, "스윕 저점 아래로 종가가 내려가 바닥 시나리오가 무효가 됐습니다.",
                 "이 셋업으로 산 물량은 전량 정리하고, 새 스윕·구조가 나올 때까지 기다립니다.")
    if stg == "SWEEP":
        s = ict['sweep']
        n_ok = sum(bool(x) for x in (ict.get('filters') or {}).values())
        if s['age'] == 0 and wst == "W_DOWN" and n_ok <= 2:
            reclaim = s['level'] + 0.1 * ict['atr']
            return v("WAIT", None, f"{s['ctx']}에서 직전 저점({F(s['level'])})을 쓸고 오늘 종가로 회복했습니다.",
                     f"주봉이 하락 추세이고 확인 신호가 약해(보조 필터 {n_ok}개) 하루 더 확인합니다 — "
                     f"다음 거래일 종가가 {F(reclaim)} 위에서 유지되면 🟢 1차 매수로 바뀝니다.",
                     trigger=reclaim, confirm=reclaim)
        return graded(v("BUY", 1, f"{s['ctx']}에서 직전 저점({F(s['level'])})을 쓸고 종가로 회복했습니다 — 무릎 1차 진입 자리입니다.",
                        "아래 사다리의 1차(30%)를 사고, 체결 즉시 무효화 손절가에 스탑 주문을 걸어 두세요."))
    if stg == "SWEEP_EXT":
        s = ict['sweep']
        up = (ict['px'] / s['rec_close'] - 1) * 100
        return v("WAIT", None, f"{_tdate(s['time'])} 1차 신호(종가 {F(s['rec_close'])}) 이후 이미 {up:+.1f}% 올라 지금 사면 추격입니다.",
                 f"1차 신호 가격 {F(s['rec_close'])} 근처로 되돌리면 1차(30%)를 삽니다. 손절가는 그대로입니다.", trigger=s['rec_close'])
    if stg == "RECLAIM_FAIL":
        sf = ict['sweep_failing']
        reclaim = sf['level'] + 0.1 * ict['atr']
        return v("WAIT", None, f"{_tdate(sf['time'])}에 기준선 {F(sf['level'])}을 쓸고 회복했지만, 이후 종가가 다시 그 아래로 내려갔습니다 (회복 실패).",
                 f"종가가 {F(reclaim)} 위로 다시 올라오면 1차가 재성립합니다. 종가가 {F(sf['stop'])} 아래면 이 시나리오는 무효입니다.",
                 trigger=reclaim)
    if stg == "CHOCH":
        return graded(v("BUY", 2, f"{ict['choch']['src']}: 직전 반등 고점({F(ict['choch']['level'])})을 돌파·안착했습니다 — 2차 진입 자리입니다.",
                        "1차를 이미 샀다면 2차(30%)를 추가하고, 못 샀다면 2차만 삽니다. 손절가는 그대로 둡니다."))
    if stg == "RETEST":
        return graded(v("BUY", 3, f"CHoCH 후 돌파선({F(ict['retest']['level'])})까지 되돌림 후 지지 중입니다 — 3차 진입 자리입니다.",
                        "3차(40%)를 추가합니다. 이후에는 더 사지 않고 💼 포지션 관리 탭으로 관리합니다."))
    if stg == "PENDING":
        return v("WAIT", None, "스윕 후 돌파가 나왔지만 종가 안착(2봉) 확인 중입니다.", "안착이 확인되면 2차 매수로 바뀝니다. 미리 사지 마세요.",
                 trigger=ict['choch']['level'])
    if stg in ("CHOCH_EXT", "POST"):
        lv = ict['choch']['level']
        return v("WAIT", None, "CHoCH 이후 이미 올라가 지금 사면 추격입니다.",
                 f"돌파선({F(lv)}) 근처로 되돌림이 오면 3차(40%)를 삽니다.", trigger=lv)
    if stg == "IN_ZONE":
        z = ict['near'][0]
        return v("WAIT", None, f"{z['kind']}({F(z['lo'])}~{F(z['hi'])})에 도달했습니다 — 관심 종목입니다.",
                 "아직 사지 않습니다. 직전 저점을 아래로 쓸고 종가가 다시 올라오면(스윕 후 회복) 1차 매수 신호가 나옵니다.")
    base = action_verdict(info, plan_trig)
    if base['kind'] == "BUY":
        base = dict(kind="WAIT", tranche=None, trigger=base.get('trigger'),
                    reasons=["돌파는 확정됐지만 ICT 매수 맥락(수요 구간·할인 구간에서의 스윕)이 없습니다.",
                             "이 전략에서는 사지 않고 관망합니다."])
    base.setdefault('tranche', None)
    base.setdefault('grade', None)
    base.setdefault('missing', [])
    return base


def ict_ladder(info, ict, equity, rate=1.0, qty_dec=0, grade=None):
    """
    3단 분할 매수 사다리 (1회 계획): 1차 30% 스윕 후 회복 · 2차 30% CHoCH 확정 · 3차 40% 돌파선 리테스트
      손절  : 스윕 저점 − 0.25 ATR (스윕 전이면 직전 저점 또는 수요 구간 하단 기준 예상값) — 모든 차수 공통
      총액  : 세 차수가 모두 체결된 뒤 손절되면 총자산 × 1% × (200일선 하락 시 0.5) × (보조 필터 2개 미만 0.75)만 잃도록 역산
      규칙  : 지난 단계는 건너뛰고(남은 차수로 몰아 사지 않음), 손절가 아래에서는 추가 매수 금지
    가격은 표시 통화(원) 기준으로 반환.
    """
    if not ict or not ict.get('ok'):
        return None
    if ict.get('stage') == "INVALID":
        return dict(err="스윕 저점이 깨져 이 셋업은 무효가 됐습니다. 보유 물량은 정리하고, 새 스윕이 나올 때까지 사다리를 만들지 않습니다.",
                    sl=None, stop_src="-")
    if ict.get('stage') == "NO_ZONE":
        return dict(err="지금은 ICT 매수 맥락(주봉 수요 구간 근처)이 아니라 사다리를 만들지 않습니다. "
                        "가격이 주봉 수요 구간까지 내려오면 1차 예약 가격과 손절가가 계산됩니다.", sl=None, stop_src="-")
    L, smc = info['levels'], info.get('smc') or {}
    a = ict['atr']
    px = ict['px']
    stg = ict['stage']
    s, ch, rt = ict.get('sweep'), ict.get('choch'), ict.get('retest')

    # 손절
    stop, stop_src = _ict_stop(ict, L)
    sf = ict.get('sweep_failing') if stg == "RECLAIM_FAIL" else None

    # 차수별 가격 · 상태 · 짧은 조건 문구 (cond: 화면 카드용)
    if s and stg == "SWEEP_EXT":
        e1, st1, c1 = s['rec_close'], "대기: 1차 신호 가격까지 되돌림 시 (지정가)", "이 가격까지 되돌림 시"
    elif s:
        e1, st1, c1 = (px, "지금", "지금") if stg == "SWEEP" else (s['rec_close'], "지남", "지남")
    elif sf:
        e1, st1, c1 = sf['level'] + 0.1 * a, "대기: 종가가 이 가격 위로 다시 올라오면 (재회복)", "종가가 이 위로 재회복 시"
    else:
        liq = ict.get('liq')
        lvl = liq if (liq and liq > stop) else px
        e1, st1, c1 = lvl, "대기: 이 가격 아래로 쓸고 내려갔다가 종가로 다시 회복 시 (스윕 후 회복)", "아래로 쓸고 종가 회복 시"
    src = s or sf
    ch_known = False
    if src and not ch:
        brk0 = src.get('choch_level')
        if brk0 and brk0 > e1:
            e2, st2, c2 = brk0 + 0.1 * a, "대기: 종가가 이 가격 위로 돌파·안착 시 (CHoCH)", "종가 돌파 후 2일 안착 시"
            e3, st3, c3 = brk0, "대기: CHoCH 후 돌파선까지 되돌림 시", "돌파 후 되돌림 지지 시"
            ch_known = True
    if ch:
        brk = ch['level']
        if stg == "CHOCH":
            e2, st2, c2 = px, "지금", "지금"
        elif stg == "PENDING":
            e2, st2, c2 = max(px, brk), "대기: 종가 안착(2봉) 확인 후", "안착 확인 후"
        else:
            e2, st2, c2 = brk, "지남", "지남"
        e3 = brk
        st3, c3 = ("지금", "지금") if stg == "RETEST" else ("대기: 돌파선까지 되돌림 후 지지 시 (지정가)", "되돌림 후 지지 시")
    elif not ch_known:
        # 스윕 전: CHoCH 기준선은 스윕이 나와야 정해짐 → 1차 가격 + ATR로 '예상'만 표시
        e2, st2, c2 = e1 + 1.5 * a, "예상: 스윕 후 반등 고점 돌파 시 (스윕 후 재계산)", "스윕 후 재계산"
        e3, st3, c3 = e1 + 1.0 * a, "예상: CHoCH 후 되돌림 시 (스윕 후 재계산)", "스윕 후 재계산"
    if e3 > e2:
        e3 = e2

    wi = ict.get('wict') or {}
    mult, notes = 1.0, []
    if wi.get('mult', 1.0) < 1.0:
        mult *= max(wi['mult'], 0.0)
        notes.append(f"주봉 {W_STATE_META[wi['state']]['short']} ×{wi['mult']:g}")
    n_ok = sum(bool(x) for x in ict['filters'].values())
    if n_ok < 2:
        mult *= ICT_WEAK_FILTER_MULT
        notes.append(f"보조 필터 {n_ok}개 ×{ICT_WEAK_FILTER_MULT:g}")
    if grade == "B":
        mult *= GRADE_B_MULT
        notes.append(f"B급 ×{GRADE_B_MULT:g}")
    risk_eff = RISK_PER_TRADE * mult

    P = [e1 * rate, e2 * rate, e3 * rate]
    SL = stop * rate
    if risk_eff <= 0:
        return dict(err="설정상 주봉 하락 추세에서는 매수하지 않습니다 (WEEKLY_DOWN_RISK_MULT = 0).", sl=SL, stop_src=stop_src)
    stats = [st1, st2, st3]
    bad = [i + 1 for i, p in enumerate(P) if p <= SL]
    if any(stats[k - 1] == "지금" for k in bad) or len(bad) == 3:
        return dict(err=f"{', '.join(map(str, bad))}차 가격이 손절가보다 낮거나 같아 사다리를 만들 수 없습니다.", sl=SL, stop_src=stop_src)
    for k in bad:                                   # 예약 차수만 손절가 아래 → 그 차수는 생략
        stats[k - 1] = "생략: 손절가 아래 가격"
        P[k - 1] = SL * 1.0001
    st1, st2, st3 = stats
    loss = [(p * (1 + SPLIT_COST) - SL * (1 - SPLIT_COST)) / p for p in P]
    R = equity * risk_eff / 100
    amts = [R * wt / lf for wt, lf in zip(ICT_LADDER, loss)]               # 차수별 금액 = (손실 한도 × 배분) ÷ 손절까지 손실률
    cap = equity * MAX_POSITION_PCT / 100
    capped = sum(amts) > cap
    if capped:
        amts = [x * cap / sum(amts) for x in amts]
    rows, cum_loss = [], 0.0
    for k, (p, wt, lf, stt, cnd, amt0) in enumerate(zip(P, ICT_LADDER, loss, (st1, st2, st3), (c1, c2, c3), amts), start=1):
        q = _qty_floor(amt0 / p, qty_dec)
        amt = q * p
        live = stt != "지남" and not stt.startswith("생략")
        if live:
            cum_loss += amt * lf
        rows.append(dict(n=k, weight=wt, price=p, qty=q, amount=amt, status=stt, cond=cnd, live=live,
                         cum_loss=cum_loss, cum_loss_pct=cum_loss / equity * 100))
    now = next((r for r in rows if r['status'] == "지금"), None)
    total = sum(r['amount'] for r in rows if r['live'])                    # 실제 체결 수량 기준 (지난 차수 제외)
    return dict(err=None, rows=rows, sl=SL, sl_pct=(SL / (px * rate) - 1) * 100, stop_src=stop_src, total=total,
                risk_eff=risk_eff, notes=notes, capped=capped, now=now, filters_ok=n_ok)



# ==========================================
# ICT 화면 구성: 결론 배너 · ICT 분석 상자 · 3단 사다리 주문표
# ==========================================
def _tdate(t):
    try:
        return pd.Timestamp(t).strftime("%Y-%m-%d")
    except Exception:
        return "-"


GRADE_STYLE = {"A": dict(title="🟢 A급 매수", bg="#2e7d32"), "B": dict(title="🟡 B급 조건부 매수", bg="#8d6e00")}


def render_ict_banner(v, lad, info, ict, curr, rate, qty_dec, equity, basis=None, premium=None):
    sty = VERDICT_STYLE[v['kind']]
    title, bg = sty['title'], sty['bg']
    if v['kind'] == "BUY" and v.get('grade') in GRADE_STYLE and GRADE_ACTIVE:
        title, bg = GRADE_STYLE[v['grade']]['title'], GRADE_STYLE[v['grade']]['bg']
    elif v.get('grade') == "C":
        title = "⏳ C급 신호 · 매수 보류"
    elif v.get('confirm'):
        title = "⏳ 확인 대기 — 내일 종가로 1차 확정"
    if v['kind'] == "BUY" and v.get('tranche'):
        title += f" — {v['tranche']}차"
    basis_html = f'<div style="font-size:13px;opacity:.85;margin-top:8px">🕒 기준 봉: {basis["label"]}</div>' if basis and basis.get('label') else ""
    st.markdown(
        f"""<div style="background:{bg};color:#fff;padding:16px 20px;border-radius:12px;margin:2px 0 10px 0">
<div style="font-size:28px;font-weight:800;line-height:1.25">{title}</div>
<div style="font-size:15px;opacity:.95;margin-top:6px">{'<br>'.join(v['reasons'])}</div>{basis_html}</div>""",
        unsafe_allow_html=True)
    if basis and basis.get('stale'):
        st.warning("장이 끝났는데 오늘 봉이 아직 데이터에 없습니다 (데이터 지연). 휴장일이 아니라면 잠시 뒤 새로고침하세요 — 지금 판정은 직전 거래일 기준입니다.")
    if v.get('checks'):
        mk = lambda ok: "✅" if ok else ("❌" if ok is False else "⬜")
        st.caption(("등급 체크: " if GRADE_ACTIVE else "참고 등급 체크 (매수 판단에 미사용): ") + " · ".join(f"{mk(ok)} {GRADE_NAMES[k]}({d})" for k, ok, d in v['checks']))
    small = curr * rate < 1000
    m1, m2, m3, m4 = st.columns(4)
    ok = lad is not None and not lad.get('err')
    if v['kind'] == "BUY" and ok and lad.get('now'):
        r = lad['now']
        m1.metric(f"① 지금 {r['n']}차 매수", _won(r['price'], small), _qty_txt(r['qty'], qty_dec), delta_color="off")
        m2.metric("② 공통 손절가 (무효화)", _won(lad['sl'], small), f"{lad['sl_pct']:+.1f}%", delta_color="off")
        m3.metric("③ 이번 매수 금액", _won(r['amount']), f"총자산의 {r['amount'] / equity * 100:.1f}%", delta_color="off")
        left = [x for x in lad['rows'] if x['n'] > r['n']]
        m4.metric("④ 남은 차수", f"{len(left)}개" if left else "없음", " · ".join(f"{x['n']}차 {_won(x['price'], small)}" for x in left) or None,
                  delta_color="off")
    elif v.get('confirm') and ok:
        r1 = lad['rows'][0]
        m1.metric("① 내일 확인할 것", f"종가 {_won(v['confirm'] * rate, small)} 위", "유지되면 1차 확정", delta_color="off")
        m2.metric("② 그때 1차 매수", _won(r1['price'], small), _qty_txt(r1['qty'], qty_dec), delta_color="off")
        m3.metric("③ 공통 손절가", _won(lad['sl'], small), f"{lad['sl_pct']:+.1f}%", delta_color="off")
        m4.metric("④ 지금 권장 비중", "0%")
    elif v['kind'] == "WAIT" and ok:
        nxt = next((x for x in lad['rows'] if x['status'].startswith("대기")), None)
        names = {1: "스윕 후 회복", 2: "CHoCH 돌파", 3: "리테스트"}
        m1.metric("① 다음 매수", f"{nxt['n']}차 · {names[nxt['n']]}" if nxt else "-", None)
        m2.metric("② 기준 가격", _won(nxt['price'], small) if nxt else "-", (nxt.get('cond') if nxt else None), delta_color="off")
        m3.metric("③ 예상 손절가", _won(lad['sl'], small), f"{lad['sl_pct']:+.1f}%", delta_color="off")
        m4.metric("④ 그때 수량 · 지금 비중", _qty_txt(nxt['qty'], qty_dec) if nxt else "-", "지금은 0%", delta_color="off")
    else:
        rg = (ict or {}).get('range') or {}
        L = info.get('levels') or {}
        m1.metric("① 지금 권장 비중", "0%")
        wi = (ict or {}).get('wict') or {}
        wm = W_STATE_META.get(wi.get('state', "W_NA"))
        m2.metric("② 주봉 추세", f"{wm['icon']} {wm['short']}",
                  (f"전환 기준선까지 {wi['dist'] * 100:+.1f}%" if (wi.get('dist') is not None and wi.get('state') in ("W_DOWN", "W_SWEEP"))
                   else (f"레인지 {rg['pos'] * 100:.0f}%" if rg else None)), delta_color="off")
        m3.metric("③ 직전 저점 (ISSL)", _won(L['issl'] * rate, small) if L.get('issl') else "N/A")
        z = (ict or {}).get('zones') or []
        m4.metric("④ 가장 가까운 수요 구간", f"{_won(z[0]['lo'] * rate, small)}~" if z else "없음",
                  f"현재가 대비 {(z[0]['hi'] / curr - 1) * 100:+.1f}%" if z else None, delta_color="off")
    if premium and ok and premium.get('prem') is not None:
        p = premium['prem']
        live = [x for x in lad['rows'] if x['live']]
        st.caption(f"🇰🇷 업비트에서 주문할 때 (김프 {p * 100:+.2f}% 반영): "
                   + " · ".join(f"{x['n']}차 ≈ {_won(x['price'] * (1 + p), small)}" for x in live)
                   + f" · 손절 ≈ {_won(lad['sl'] * (1 + p), small)}  (앱 가격은 해외 USDT 시세 × 환율)")
    st.caption("⚠️ 수요 구간·할인 구간·3박자는 **관심 종목 기준**입니다. 실제 매수는 결론이 **🟢 매수**(스윕 후 회복 → CHoCH → 리테스트 단계)일 때만 하세요. "
               + ("B급은 리스크를 절반으로 줄인 조건부 매수, C급은 매수하지 않습니다." if GRADE_ACTIVE else
                  f"손절폭은 최소 {ICT_MIN_STOP_PCT:g}%(또는 {ICT_MIN_STOP_ATR:g} ATR), 손절 후 {ICT_COOLDOWN_N}봉은 쉽니다."))


def render_ict_box(ict, info, rate):
    if not ict or not ict.get('ok'):
        st.caption("ICT 분석에 필요한 데이터(일봉 60봉 이상)가 부족합니다.")
        return
    small = ict['px'] * rate < 1000
    W = lambda x: _won(x * rate, small) if x is not None else "-"
    mk = lambda ok: "✅" if ok else "⬜"
    with st.container(border=True):
        st.markdown("##### 🧭 ICT 분석 (주봉 구간 → 일봉 진입)")
        wk, rg = ict['weekly'], ict['range']
        wi = ict.get('wict') or {}
        wm = W_STATE_META.get(wi.get('state', "W_NA"))
        line = f"- **주봉 추세 판정**: {wm['icon']} **{wm['label']}**"
        if wi.get('level'):
            line += (f" · 주봉 CHoCH 기준선({wi.get('level_src', '마지막 반등 고점')}) {W(wi['level'])} (현재가 대비 {wi['dist'] * 100:+.1f}%)"
                     + (f" · {_tdate(wi['choch_time'])} 돌파" if wi.get('choch_time') is not None and wi['state'] == "W_CHOCH" else ""))
        if wi.get('sweep'):
            sw = wi['sweep']
            line += f" · {_tdate(sw['time'])} 주봉 저점 {W(sw['level'])} 아래 {W(sw['low'])}까지 스윕 후 회복"
        st.markdown(line)
        st.caption("주봉 추세가 하락(CHoCH 전)이면 일봉 반등은 단기 반등일 수 있어 사다리 리스크를 "
                   f"×{WEEKLY_DOWN_RISK_MULT:g}, 주봉 스윕 후 회복이면 ×{WEEKLY_SWEEP_RISK_MULT:g}로 줄입니다. 주봉 CHoCH 이후에는 정상 리스크입니다.")
        st.markdown(f"- **주봉 스윙 구조**: {wk.get('bias', '판단 불가')}"
                    + (f" · 고점 {'상승(HH)' if wk.get('hh') else '하락(LH)'} · 저점 {'상승(HL)' if wk.get('hl') else '하락(LL)'}" if wk.get('hh') is not None else ""))
        if rg:
            zone_txt = "OTE(62~79% 되돌림)" if rg['ote_lo'] <= ict['px'] <= rg['ote_hi'] else ("할인 구간" if rg['pos'] < 0.5 else "프리미엄 구간")
            st.markdown(f"- **딜링 레인지 (최근 {ICT_RANGE_W}주)**: {W(rg['lo'])} ~ {W(rg['hi'])} · 중간값 {W(rg['eq'])} · "
                        f"현재 위치 **{rg['pos'] * 100:.0f}% ({zone_txt})** · OTE {W(rg['ote_lo'])}~{W(rg['ote_hi'])}")
        if ict['zones']:
            st.markdown("- **주봉 수요 구간 (현재가 아래)**")
            for z in ict['zones']:
                tag = " ← 도달" if z in ict.get('near', []) else ""
                st.markdown(f"    - {z['kind']} {W(z['lo'])}~{W(z['hi'])} (현재가 대비 {(z['hi'] / ict['px'] - 1) * 100:+.1f}%, {_tdate(z['t'])}){tag}")
        else:
            st.markdown("- **주봉 수요 구간**: 현재가 아래에 유효한 구간이 없습니다 (불러온 기간 기준).")
        s, ch, rt, inv = ict['sweep'], ict['choch'], ict['retest'], ict['invalid']
        st.markdown("- **일봉 진행 상황**")
        st.markdown(f"    - {mk(s)} ① 스윕 후 회복: " + (f"{_tdate(s['time'])} 직전 저점 {W(s['level'])} 아래 {W(s['low'])}까지 쓸고 회복 ({s['ctx']})" if s
                    else ("스윕 실패(무효화) " + _tdate(inv['time']) if inv else "아직 없음")))
        st.markdown(f"    - {mk(ch and ch['bars_above'] >= CHOCH_SETTLE)} ② CHoCH: " + (f"{_tdate(ch['time'])} 돌파선 {W(ch['level'])} 상향 돌파 · {ch['bars_above']}봉 안착 ({ch['src']})" if ch else "아직 없음"))
        st.markdown(f"    - {mk(rt)} ③ 리테스트: " + (f"{_tdate(rt['time'])} 돌파선 {W(rt['level'])}까지 되돌림 후 지지" if rt else "아직 없음"))
        f = ict['filters']
        n_ok = sum(bool(x) for x in f.values())
        st.markdown(f"- **보조 필터 ({n_ok}/{len(f)})**: " + " · ".join(f"{'✓' if ok else '✗'} {k}" for k, ok in f.items()))
        st.caption("보조 필터는 점수로 더하지 않습니다. 2개 미만이면 사다리 리스크를 0.75배로 줄입니다. "
                   "수요 구간은 강한 주봉 상승 직전의 마지막 음봉(오더블록)과 주봉 가격 공백(FVG)으로 계산합니다.")


def render_ladder(v, lad, curr, rate, qty_dec, equity):
    with st.expander("🪜 3단 분할 매수 사다리 (손실 한도 30% 스윕 후 회복 · 30% CHoCH · 40% 리테스트 · 공통 손절)", expanded=v['kind'] == "BUY"):
        if lad is None:
            st.caption("사다리를 만들 데이터가 부족합니다.")
            return
        if lad.get('err'):
            (st.info if "맥락" in lad['err'] else st.error)(lad['err'])
            return
        small = curr * rate < 1000
        names = {1: "1차 · 스윕 후 회복", 2: "2차 · CHoCH 확정", 3: "3차 · 돌파선 리테스트"}
        rows = [{"차수": f"{names[r['n']]} (손실 한도 {int(r['weight'] * 100)}%)", "상태": r['status'], "가격": _won(r['price'], small),
                 "수량": _qty_txt(r['qty'], qty_dec), "금액": _won(r['amount']),
                 "여기까지 산 뒤 손절 시 손실": (f"{_won(r['cum_loss'])} ({r['cum_loss_pct']:.2f}%)" if r['live'] else "-")} for r in lad['rows']]
        rows.append({"차수": "🛑 공통 손절 (무효화)", "상태": "1차 체결 즉시 스탑 주문 · 모든 차수 공통", "가격": _won(lad['sl'], small),
                     "수량": "전량", "금액": "-", "여기까지 산 뒤 손절 시 손실": "-"})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.markdown(f"- 계획 총액(남은 차수 모두 체결 시, 실제 수량 기준): **{_won(lad['total'])}** (총자산의 {lad['total'] / equity * 100:.1f}%) · "
                    f"적용 리스크 {lad['risk_eff']:.2f}%" + (f" ({', '.join(lad['notes'])})" if lad['notes'] else "")
                    + f" · 손절 기준: {lad['stop_src']}")
        st.markdown("- **배분 방식**: 손실 한도(총자산 × 리스크)를 30/30/40으로 나눠, 각 차수가 손절 시 자기 몫만 잃도록 수량을 정합니다. "
                    "손절가에 가까운 1차(무릎)일수록 같은 손실로 더 많이 살 수 있어 금액이 커집니다.")
        st.markdown("- **규칙**: ① '지남' 차수는 건너뜁니다 (남은 차수로 몰아 사지 않음) ② 손절가 아래에서는 절대 추가 매수 금지 "
                    "③ 손절가에 닿으면 몇 차까지 샀든 전량 정리 ④ 같은 종목에서 연속 2번 무효화되면 새 고점·저점 구조가 나올 때까지 쉬기")
        if lad['capped']:
            st.warning(f"계산된 금액이 한 종목 최대 비중({MAX_POSITION_PCT:g}%)을 넘어 {MAX_POSITION_PCT:g}%로 제한했습니다.")
        if any(r['qty'] == 0 for r in lad['rows'] if r['live']):
            st.warning("수량이 0인 차수가 있습니다 (1주 가격이 배정 금액보다 큼). 이 종목은 차수를 줄이거나 매수하지 않는 편이 맞습니다.")
        if v['kind'] != "BUY":
            st.info("지금은 🟢 매수 실행 상태가 아니므로 이 표는 **조건이 충족됐을 때의 예약 계획**입니다. 지금 주문하지 마세요.")
        st.caption(f"※ 손실에는 편도 {SPLIT_COST * 100:.2f}%의 수수료·슬리피지를 포함했습니다. 총자산은 사이드바에서 바꿀 수 있습니다.")



LT_SPLIT = (0.3, 0.3, 0.4)     # 장기 적립 추가 매수 배분: 지금(또는 첫 구간) / 다음 구간 / 깊은 구간


def long_term_plan(ict, v):
    """
    장기 적립 모드 (손절 없음): 주봉 수요 구간에 나눠 담는 계획 (종목 통화 가격)
      1차 30% : 지금 🟢 신호면 현재가, 아니면 가장 가까운 수요 구간 상단
      2차 30% : 그 아래 구간 중간 (없으면 OTE 중간)
      3차 40% : 더 깊은 구간 중간 (없으면 레인지 하단 + 25%)
      재검토선 : 사용한 가장 깊은 구간 하단 — 주봉 종가가 이 아래면 추가 매수를 멈추고 계획을 다시 봄 (자동 매도 아님)
      추세 확인 : 주봉 CHoCH 기준선 위 주봉 종가 → 체결 안 된 금액을 그때 넣음
    """
    if not ict or not ict.get('ok'):
        return None
    px, rg, wi = ict['px'], ict.get('range') or {}, ict.get('wict') or {}
    zs = sorted([z for z in (ict.get('zones') or []) if z['hi'] < px * 0.995], key=lambda z: z['hi'], reverse=True)
    steps = []
    if v.get('kind') == "BUY":
        steps.append((px, "지금 (앱 🟢 신호)"))
    elif ict.get('near'):
        steps.append((px, "지금 (수요 구간 안)"))
    elif zs:
        z = zs.pop(0)
        steps.append((z['hi'], f"{z['kind']} 상단 (지정가)"))
    def add(pr, why):                               # 앞 차수보다 3% 이상 낮을 때만 (간격이 너무 좁으면 같은 자리)
        if len(steps) < 3 and (not steps or pr < steps[-1][0] * 0.97) and pr < px:
            steps.append((pr, why))
    for z in zs:
        add((z['lo'] + z['hi']) / 2, f"{z['kind']} 중간 (지정가)")
    if rg:
        add((rg['ote_lo'] + rg['ote_hi']) / 2, "주봉 OTE 중간 (지정가)")
        add(rg['lo'] + 0.25 * (rg['hi'] - rg['lo']), "레인지 하단 25% (지정가)")
    steps = steps[:3]
    if not steps:
        return None
    used = [z for z in (ict.get('zones') or []) if any(z['lo'] * 0.995 <= p_ <= z['hi'] * 1.005 for p_, _w in steps)]
    review = min([z['lo'] for z in used], default=steps[-1][0] * 0.92)
    trend = wi.get('level') if wi.get('state') in ("W_DOWN", "W_SWEEP") else None
    return dict(steps=steps, review=review, trend=trend, wstate=wi.get('state', "W_NA"))


def render_long_term(ict, v, rate, qty_dec, premium=None, key="lt"):
    plan = long_term_plan(ict, v)
    with st.expander("🏦 장기 적립 모드 (손절 없음 · 주봉 구간에 나눠 담기)", expanded=False):
        if plan is None:
            st.caption("장기 적립 계획을 만들 데이터가 부족합니다.")
            return
        small = ict['px'] * rate < 1000
        p_adj = (1 + premium['prem']) if (premium and premium.get('prem') is not None) else 1.0
        a1, a2 = st.columns(2)
        with a1:
            budget = st.number_input("추가 매수 예산 (원)", min_value=0.0, value=0.0, step=100_000.0, format="%.0f", key=f"{key}_bud")
        with a2:
            weekly = st.number_input("매주 적립액 (원)", min_value=0.0, value=100_000.0, step=10_000.0, format="%.0f", key=f"{key}_wk")
        budget = budget if isinstance(budget, (int, float)) else 0.0
        weekly = weekly if isinstance(weekly, (int, float)) else 0.0
        rows, q_tot = [], 0.0
        for k, ((pr, why), wt) in enumerate(zip(plan['steps'], LT_SPLIT), start=1):
            krw = pr * rate * p_adj
            amt = budget * wt
            q = _qty_floor(amt / krw, qty_dec) if krw > 0 and amt > 0 else 0.0
            q_tot += q
            rows.append({"차수": f"{k}차 ({int(wt * 100)}%)", "조건": why, "가격": _won(krw, small), "금액": _won(amt) if amt else "-",
                         "예상 수량": _qty_txt(q, qty_dec) if q else "-"})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        if budget > 0 and q_tot > 0:
            st.caption(f"세 차수 모두 체결되면 평균 매수가 약 {_won(budget / q_tot, small)}")
        st.markdown(f"- **매주 적립**: 가격과 상관없이 매주 {_won(weekly)}씩 삽니다. 떨어질 때 멈추지 않는 것이 핵심입니다.")
        st.markdown(f"- **재검토선**: 주봉 종가가 **{_won(plan['review'] * rate * p_adj, small)}** 아래로 마감하면 추가 매수를 멈추고 계획을 다시 봅니다 (자동 매도 아님 · 적립은 계속).")
        if plan['trend']:
            st.markdown(f"- **추세 확인 매수**: 지정가가 체결되지 않은 채 주봉 종가가 **{_won(plan['trend'] * rate * p_adj, small)}**(주봉 CHoCH) 위로 마감하면 남은 금액을 그때 넣습니다.")
        else:
            st.markdown("- **주봉 추세**: 이미 상승·전환 상태라 지정가가 안 오면 매주 적립만 이어갑니다.")
        st.caption("장기 보유용입니다. 단기 사다리의 손절 규칙은 적용하지 않습니다. 3년 이상 쓰지 않을 돈만, 전체 자산에서 비중 상한을 정해 두고 넣으세요."
                   + (f" 가격은 업비트 기준(김프 {premium['prem'] * 100:+.2f}% 반영)입니다." if p_adj != 1.0 else ""))


def render_dashboard(df, c1, c2, show_hist, rate_text=1.0, rate_chart=1.0, is_commodity=False,
                     captions=(), tf_label=None, htf=None, macro=None, qty_dec=0, premium=None, long_mode=False):
    curr = float(df['close'].iloc[-1])
    fibs = calculate_fibonacci_levels(df, is_commodity=is_commodity)
    bs = bar_status(df) if tf_label == "1일" else None
    dfa = closed_bars(df) if tf_label == "1일" else df          # 일봉 판정은 ICT·참고 지표 모두 마감 봉 기준으로 통일
    # 거시 추세 필터: 일봉은 데이터 안의 200일선(스크리너와 동일), 그 외 프레임은 기존 상위 프레임 방식
    if tf_label == "1일":
        pack, mtf = None, daily_regime(dfa)
    else:
        pack = prepare_htf(htf, tf_label)
        t_last = _naive(df['time']).iloc[-1].to_datetime64().astype('datetime64[ns]')
        mtf = mtf_at(pack, t_last)
    info = analyze_market_cycle(dfa, mtf, macro)
    meta = PHASE_META[info['phase']]
    plan_now = build_trade_plan(info) if info['entry'] in ("TRIGGER", "BREAKOUT") or info['phase'] == "상승" else None
    use_ict = tf_label == "1일"
    if use_ict:                                   # ICT 메인: 주봉 구간 + 일봉 스윕·CHoCH·리테스트 사다리
        ict = ict_analysis(dfa, info)
        v = ict_verdict(info, ict, plan_now if info['entry'] == "TRIGGER" else None, fmt=_disp_fmt(rate_text, curr))
        lad = ict_ladder(info, ict, EQUITY, rate_text, qty_dec, grade=v.get('grade') if GRADE_ACTIVE else None)
        # 사다리 가격은 표시 통화(rate_text) 기준 → 원래 통화로 되돌려 차트(rate_chart)에 그림
        extra = [(lad['sl'] / rate_text, "무효화 손절", "#ef5350")] if (lad and not lad.get('err') and rate_text) else []
        if lad and not lad.get('err'):
            extra += [(r['price'] / rate_text, f"{r['n']}차", "#42a5f5") for r in lad['rows'] if r['live']]
        zones = ict.get('zones') if ict else None
        fig_plan = None
    else:                                          # 분봉·주봉·월봉: 기존 결론 + 2분할 플랜
        ict, lad, extra, zones = None, None, [], None
        v = action_verdict(info, plan_now if info['entry'] == "TRIGGER" else None)
        plan2 = entry_plan_2split(info, curr, EQUITY, rate_text, qty_dec)
        fig_plan = plan_now if v['kind'] == "BUY" else None      # 차트의 손절·익절선은 🟢일 때만

    with c1:
        if use_ict:
            render_ict_banner(v, lad, info, ict, curr, rate_text, qty_dec, EQUITY, basis=bs, premium=premium)
            render_ladder(v, lad, curr, rate_text, qty_dec, EQUITY)
            if long_mode:
                render_long_term(ict, v, rate_text, qty_dec, premium)
            render_ict_box(ict, info, rate_text)
        else:
            render_action_banner(v, plan2, info, curr, rate_text, qty_dec, EQUITY)
            render_entry_plan(v, plan2, info, curr, rate_text, qty_dec, EQUITY)
            st.caption("ℹ️ ICT 사다리(스윕 → CHoCH → 리테스트)는 '1일' 프레임에서 계산됩니다.")
        for cap in captions:
            st.caption(cap)
        st.divider()
        st.markdown(f"###### 📎 참고 지표 — 국면 {meta['icon']} {info['phase']} ({info['score']}/100점)")
        tot = info['totals']
        st.caption("국면 일치도 — " + " | ".join(f"{k} {tot[k]}" for k in ["매집", "상승", "분배", "하락"])
                   + " · 점수는 '현재 위치 설명'이며 매수 신호가 아닙니다.")
        if info['adjust']:
            st.caption("보정: " + " · ".join(info['adjust']))
        if info['bars'] < 80:
            st.caption("⚠️ 데이터가 짧아 국면 판별 신뢰도가 낮습니다.")
        render_fact_box(info['facts'])
        render_phase_legend()
        render_trade_guide(info, fibs, curr, rate_text)

    with c2:
        history = None
        if show_hist:
            try:
                with st.spinner(f"과거 국면 연산 중... (최근 {HIST_BARS}봉)"):
                    history = cached_phase_history(df, pack, daily=(tf_label == "1일"))
            except Exception as e:
                st.error(f"과거 국면 계산 실패 ({type(e).__name__}): {e}")
        st.plotly_chart(build_chart(df, fibs, rate_chart, history, info, fig_plan, extra_lines=extra, zones=zones), width="stretch")

def render_dashboard_safe(df, c1, c2, show_hist, rate_chart=1.0, is_commodity=False, **kw):
    """고급 분석에서 어떤 예외가 나도 대시보드가 죽지 않도록, 기본 기술적 지표만으로 안전하게 폴백"""
    try:
        render_dashboard(df, c1, c2, show_hist, rate_chart=rate_chart, is_commodity=is_commodity, **kw)
    except Exception as e:
        with c1:
            st.error(f"🔴 절대 매수 금지 — 분석 엔진 오류({type(e).__name__})로 판정하지 못했습니다. 판정할 수 없을 때는 사지 않습니다.")
            st.caption("아래는 참고용 기본 지표입니다.")
            try:
                px = float(df['close'].iloc[-1])
                e20, e50, rsi = _val(df, 'EMA_20'), _val(df, 'EMA_50'), _val(df, 'RSI')
                tr = "정배열" if (e20 and e50 and px > e20 > e50) else "역배열" if (e20 and e50 and px < e20 < e50) else "혼조"
                st.markdown(f"- 현재가 대비 이평선 상태: **{tr}**")
                if rsi is not None:
                    st.markdown(f"- RSI(14): **{rsi:.1f}**")
            except Exception:
                st.markdown("- 기본 지표도 계산하지 못했습니다.")
        with c2:
            try:
                fibs = calculate_fibonacci_levels(df, is_commodity=is_commodity)
                st.plotly_chart(build_chart(df, fibs, rate_chart), width="stretch")
            except Exception:
                st.error("차트를 그리지 못했습니다.")

# ==========================================
# 봉 마감 판정: 장이 열려 있을 때만 마지막 봉을 '진행 중'으로 보고 판정에서 뺌
#   국내(KRX) 09:00~15:30 KST · 미국 09:30~16:00 뉴욕 시간 · 코인 매일 UTC 0시(한국 오전 9시) 마감
# ==========================================
MKT_SESSION = {"KRX": ("Asia/Seoul", (9, 0), (15, 30)), "US": ("America/New_York", (9, 30), (16, 0)),
               "CRYPTO": ("UTC", (0, 0), (24, 0))}
SESSION_BUFFER_MIN = 20     # 장 마감 후 데이터가 확정될 때까지 기다리는 시간 (분)


def market_of(ticker):
    t = str(ticker).upper()
    if "/" in t:
        return "CRYPTO"
    if t.endswith(".KS") or t.endswith(".KQ"):
        return "KRX"
    return "US"


def tag_df(df, mkt, tf_label):
    """로더가 만든 데이터에 시장·프레임 정보를 붙임 (봉 마감 판정용). 캐시를 거쳐도 유지됨"""
    if df is not None and not df.empty:
        df.attrs['mkt'], df.attrs['tf'] = mkt, tf_label
    return df


def _bar_date(t, zone):
    ts = pd.Timestamp(t)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(zone)
    return ts.date()


def bar_status(df):
    """
    마지막 봉 상태: dict(live, last, label, stale)
      live  : 장이 열려 있어 아직 진행 중인 봉인가 (이 경우만 판정에서 제외)
      stale : 장이 끝났는데 오늘 봉이 데이터에 없음 (데이터 지연 · 휴장일이면 정상)
    시장 정보가 없는 데이터(백테스트 등)는 모든 봉을 마감 봉으로 봅니다.
    """
    out = dict(live=False, last=None, label="", stale=False, mkt=None)
    if df is None or df.empty or 'time' not in df.columns:
        return out
    mkt, tf = df.attrs.get('mkt'), df.attrs.get('tf', "1일")
    out['mkt'] = mkt
    last = pd.Timestamp(df['time'].iloc[-1])
    out['last'] = last
    if mkt not in MKT_SESSION:
        out['label'] = f"{last:%m/%d} 봉 기준"
        return out
    zone, (oh, om), (ch, cm) = MKT_SESSION[mkt]
    now = pd.Timestamp.now(tz=zone)
    d_last, today = _bar_date(last, zone), now.date()
    if tf != "1일":
        out['live'] = True                      # 분봉·주봉·월봉의 마지막 봉은 항상 진행 중으로 간주
        out['label'] = "마지막 봉 진행 중"
        return out
    if mkt == "CRYPTO":
        out['live'] = d_last >= today
        out['label'] = f"{d_last:%m/%d} 일봉 진행 중 (오전 9시 마감)" if out['live'] else f"{d_last:%m/%d} 마감 봉 (확정)"
        if out['live'] and len(df) > 1:
            prev = _bar_date(df['time'].iloc[-2], zone)
            out['label'] = f"{prev:%m/%d} 마감 봉 기준 (확정) · 오늘 봉은 오전 9시에 마감"
        return out
    close_t = now.normalize() + pd.Timedelta(hours=ch, minutes=cm + SESSION_BUFFER_MIN)
    open_t = now.normalize() + pd.Timedelta(hours=oh, minutes=om)
    weekday = now.weekday() < 5
    out['live'] = bool(d_last == today and now < close_t)
    if out['live']:
        prev = _bar_date(df['time'].iloc[-2], zone) if len(df) > 1 else d_last
        out['label'] = f"{prev:%m/%d} 종가 기준 (확정) · 오늘 봉은 아직 마감 전이라 판정에서 제외"
    else:
        out['label'] = f"{d_last:%m/%d} 종가 기준 (확정)"
        out['stale'] = bool(weekday and now >= close_t and d_last < today) or bool(weekday and open_t <= now and d_last < today and (today - d_last).days > 3)
    return out


def closed_bars(df):
    """판정에 쓸 '마감된 봉'만 남긴 데이터 (진행 중인 마지막 봉만 제외, 이미 제외됐으면 그대로)"""
    if df is None or len(df) < 2:
        return df
    return df.iloc[:-1] if bar_status(df)['live'] else df


# ==========================================
# 데이터 로더 (모든 외부 호출 예외 처리 · 타임아웃 · 다중 소스 폴백)
# ==========================================
CRYPTO_EXCHANGES = ["binance", "okx", "bybit", "upbit", "kraken", "coinbase"]
EXCHANGE_LIMIT = {"upbit": 200}  # 거래소별 1회 조회 상한

CRYPTO_DAILY_BARS = 730     # 코인 일봉은 약 2년치를 여러 번 나눠 받아 주봉 수요 구간을 충분히 계산


def _fetch_crypto_history(ex, symbol, tf, lim):
    """일봉 약 2년치를 since 기준으로 나눠 조회. 실패하면 한 번만 조회한 결과로 대체"""
    try:
        day = 86_400_000
        since = ex.milliseconds() - CRYPTO_DAILY_BARS * day
        out = []
        for _ in range(int(CRYPTO_DAILY_BARS / max(lim, 1)) + 2):
            part = ex.fetch_ohlcv(symbol, timeframe=tf, since=since, limit=lim)
            if not part:
                break
            out += part
            nxt = part[-1][0] + day
            if nxt <= since or len(part) < 2:
                break
            since = nxt
            if since > ex.milliseconds():
                break
        if len(out) >= 120:
            return out
    except Exception:
        pass
    return ex.fetch_ohlcv(symbol, timeframe=tf, limit=lim)


@st.cache_data(ttl=60)
def load_crypto(symbol, tf_label):
    """여러 거래소를 순서대로 시도. (df, 사용한 거래소, 에러요약) 반환. 전부 실패해도 예외를 내지 않음."""
    tf_map = {"15분": "15m", "1시간": "1h", "4시간": "4h", "1일": "1d", "주봉": "1w", "월봉": "1M"}
    tf = tf_map.get(tf_label)
    errors = []
    if tf is None:
        return pd.DataFrame(), None, f"지원하지 않는 프레임: {tf_label}"
    for name in CRYPTO_EXCHANGES:
        try:
            ex = getattr(ccxt, name)({"timeout": 8000, "enableRateLimit": True})
            lim = min(300, EXCHANGE_LIMIT.get(name, 300))
            bars = _fetch_crypto_history(ex, symbol, tf, lim) if tf == "1d" else ex.fetch_ohlcv(symbol, timeframe=tf, limit=lim)
            if not bars:
                errors.append(f"{name}: 빈 데이터")
                continue
            df = pd.DataFrame(bars, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df = df.drop_duplicates('time').sort_values('time').dropna(subset=['close']).reset_index(drop=True)
            df['volume'] = df['volume'].fillna(0)
            df = add_indicators(df)
            return tag_df(df, "CRYPTO", tf_label), name, None
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}")
    return pd.DataFrame(), None, " / ".join(errors)

@st.cache_data(ttl=120, show_spinner=False)
def upbit_premium(symbol):
    """업비트 원화 가격이 해외 USDT 시세 × 환율보다 얼마나 비싼지(김프). 실패하면 None"""
    try:
        base = str(symbol).split("/")[0].upper()
        up = ccxt.upbit({"timeout": 5000}).fetch_ticker(f"{base}/KRW")['last']
        glob = None
        for name in ("binance", "okx", "bybit"):
            try:
                glob = getattr(ccxt, name)({"timeout": 5000}).fetch_ticker(f"{base}/USDT")['last']
                if glob:
                    break
            except Exception:
                continue
        if not up or not glob or not KRW_RATE:
            return None
        return dict(prem=float(up) / (float(glob) * KRW_RATE) - 1, upbit=float(up), usdt=float(glob))
    except Exception:
        return None


def load_yf_data(ticker, tf_label):
    """주식/원자재 공통 yfinance 로더. (df, 에러메시지) 반환"""
    tf_map = {"15분": "15m", "1시간": "60m", "4시간": "60m", "1일": "1d", "주봉": "1wk", "월봉": "1mo"}
    yf_tf = tf_map.get(tf_label)
    if yf_tf is None:
        return pd.DataFrame(), f"지원하지 않는 프레임: {tf_label}"
    period = "60d" if yf_tf in ["15m", "60m"] else ("2y" if yf_tf == "1d" else "10y")
    try:
        df = yf.download(ticker, period=period, interval=yf_tf, progress=False, timeout=10)
        out, err = _prep_yf_frame(df, tf_label)
        if tf_label == "1일" and out is not None and not out.empty:
            out = _patch_today_bar(out, ticker)
        return tag_df(out, market_of(ticker), tf_label), err
    except Exception as e:
        return pd.DataFrame(), f"{type(e).__name__}: {e}"


def _patch_today_bar(df, ticker):
    """
    야후 일봉에 오늘 봉이 아직 없을 때(국내 종목에서 종종 늦음) 1시간봉으로 오늘 봉을 만들어 붙임.
    장중이면 '진행 중' 봉으로 판정에서 빠지고, 장 마감 후면 확정 봉으로 쓰입니다. 실패하면 원래 데이터 그대로.
    """
    try:
        mkt = market_of(ticker)
        zone = MKT_SESSION[mkt][0]
        now = pd.Timestamp.now(tz=zone)
        if now.weekday() >= 5:
            return df
        last = _bar_date(df['time'].iloc[-1], zone)
        if last >= now.date():
            return df
        oh, om = MKT_SESSION[mkt][1]
        if now < now.normalize() + pd.Timedelta(hours=oh, minutes=om):
            return df
        h = yf.download(ticker, period="5d", interval="60m", progress=False, timeout=10)
        if h is None or h.empty:
            return df
        if isinstance(h.columns, pd.MultiIndex):
            h.columns = h.columns.get_level_values(0)
        idx = h.index.tz_convert(zone) if h.index.tz is not None else h.index.tz_localize(zone)
        today = h[idx.date == now.date()].dropna(subset=['Close'])
        if today.empty:
            return df
        t0 = df['time'].iloc[-1]
        new_t = pd.Timestamp(now.date())
        if getattr(t0, 'tzinfo', None) is not None:
            new_t = new_t.tz_localize(t0.tzinfo)
        row = dict(time=new_t, open=float(today['Open'].iloc[0]), high=float(today['High'].max()),
                   low=float(today['Low'].min()), close=float(today['Close'].iloc[-1]), volume=float(today['Volume'].fillna(0).sum()))
        base = df[['time', 'open', 'high', 'low', 'close', 'volume']]
        merged = pd.concat([base, pd.DataFrame([row])], ignore_index=True)
        out = add_indicators(merged)
        out.attrs['patched'] = True
        return out
    except Exception:
        return df

def _prep_yf_frame(df, tf_label):
    """yfinance 응답(단일 종목) → 앱 공통 형식 + 지표. (df, 에러메시지) 반환. 단일 조회와 스크리너 묶음 조회가 함께 사용"""
    try:
        if df is None or df.empty:
            return pd.DataFrame(), "yfinance에서 빈 데이터가 반환되었습니다."
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df.columns = [str(c).lower() for c in df.columns]
        df = df.rename(columns={'date': 'time', 'datetime': 'time', 'index': 'time'})
        if 'time' not in df.columns:   # yfinance 반환 형태가 바뀌어도 시간 축을 찾아서 쓰도록 보강
            cand = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
            if not cand:
                return pd.DataFrame(), "yfinance 응답에서 시간 컬럼을 찾지 못했습니다."
            df = df.rename(columns={cand[0]: 'time'})
        missing = [c for c in ('open', 'high', 'low', 'close', 'volume') if c not in df.columns]
        if missing:
            return pd.DataFrame(), f"yfinance 응답에 필요한 컬럼이 없습니다: {', '.join(missing)}"

        df = df.dropna(subset=['close']).reset_index(drop=True)       # 값이 비어 있는 마지막 봉(nan) 제거
        df['volume'] = df['volume'].fillna(0)
        if df.empty:
            return pd.DataFrame(), "yfinance에서 유효한 가격이 없습니다."
        if tf_label == "4시간":
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time')
            df = df.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna().reset_index()

        df = add_indicators(df)
        return df, None
    except Exception as e:
        return pd.DataFrame(), f"{type(e).__name__}: {e}"

@st.cache_data(ttl=300)
def load_yf_cached(ticker, tf_label):
    return load_yf_data(ticker, tf_label)

def load_htf_safe(loader, up_tf):
    """상위 프레임 로드. 어떤 이유로든 실패하면 None (MTF 필터만 생략하고 계속 진행)"""
    if not up_tf:
        return None
    try:
        res = loader(up_tf)
        d = res[0]
        return d if (d is not None and len(d) > 0) else None
    except Exception:
        return None


# ==========================================
# 💼 포지션 관리(매도 판단): 매수 판정 지표를 매도 관점으로 역활용 (가격은 종목 통화 기준)
# ==========================================
SELL_TRAIL_N = 22          # 트레일링 기준 고점을 찾는 최근 봉 수 (약 1개월)

def sell_analysis(df, avg_px, qty, qty_dec=0):
    """
    보유 종목 청산 분석.
      ① 1차 구조적 익절가 (50%): 현재가보다 1 ATR 이상 위의 가장 가까운 매물대 저항
      ② 잔여 50% 트레일링 스탑: 최근 22봉 최고가 − 3 ATR (매일 갱신, 올리기만)
      ③ 위험 알림
         🔴 즉시 전량 매도 : 종가가 트레일링 스탑 아래 마감 / 직전 저점(ISSL) 종가 이탈(구조 붕괴)
         🟠 절반 익절      : 고점 징후 2개 이상 (분배 국면, 약세 다이버전스, 고점권 대량 윗꼬리, OBV 약세, VAH 돌파 실패, 바로 위 저항)
         🟡 보유 · 주의    : 고점 징후 1개
         🟢 보유 유지      : 해당 없음
    고점 징후(분배 등)는 예측력이 검증되지 않아 '즉시 매도'가 아니라 '절반 익절' 단계로만 씁니다.
    """
    dc = closed_bars(df)                                # 마감 확정 봉 (장중이면 오늘 봉 제외)
    mtf = daily_regime(dc)
    info = analyze_market_cycle(dc, mtf, None, False)
    F = _features(dc.tail(200))
    L = info['levels']
    atr = L.get('atr_abs') or 0.0
    px = float(df['close'].iloc[-1])                    # 현재가(장중 포함)
    last_close = float(dc['close'].iloc[-1])
    hh = float(dc['high'].tail(SELL_TRAIL_N).max())
    trail = hh - TP3_TRAIL_ATR * atr if atr else None
    tp, tp_lab = structural_target(L, px, atr) if atr else (None, "-")

    hard, soft = [], []
    if trail is not None and last_close < trail:
        hard.append(f"종가가 트레일링 스탑({_fmt_price(trail)}) 아래로 마감했습니다.")
    if info['entry'] in ("WHIPSAW", "DOWN"):
        hard.append("직전 저점(ISSL)을 종가로 이탈해 상승 구조가 무너졌습니다 (Bearish CHoCH).")
    if info['phase'] == "분배":
        soft.append(f"분배 국면 판정 ({info['score']}점): 고점권 거래량·윗꼬리·고점 하락 징후")
    if F.bear_rsi or F.bear_macd:
        soft.append("약세 다이버전스: 가격은 고점을 높였는데 " + " · ".join(
            k for k, ok in (("RSI", F.bear_rsi), ("MACD", F.bear_macd)) if ok) + "는 낮아짐")
    if F.dist_climax:
        soft.append("고점권 대량 거래 + 긴 윗꼬리: 물량을 넘기는 흔적")
    elif F.wick_cluster:
        soft.append(f"고점권에서 최근 10봉 중 윗꼬리 캔들 {F.wick_cnt}개")
    if F.bear_obv:
        soft.append("가격은 버티는데 OBV(거래량 흐름)가 줄어듦")
    if F.va_fail:
        soft.append("매물대 상단(VAH) 돌파 실패 후 되밀림")
    e20 = _val(dc, 'EMA_20')
    if e20 and last_close < e20 and len(dc) > 1 and float(dc['close'].iloc[-2]) >= _val(dc, 'EMA_20', -2, e20):
        soft.append(f"종가가 20일선({_fmt_price(e20)}) 아래로 막 내려옴")
    if len(dc) > 21:
        lb = dc.iloc[-1]
        vmed = float(dc['volume'].iloc[-21:-1].median())
        if lb['close'] < lb['open'] and vmed > 0 and lb['volume'] >= 1.5 * vmed and last_close >= float(dc['close'].tail(60).quantile(0.7)):
            soft.append("고점권에서 거래량이 터진 음봉 (평소의 1.5배 이상)")
    near = [(v, lab) for lab, v in (L.get('resistances') or []) if v is not None and px < v <= px + 0.5 * atr]
    if near:
        v0, lab0 = min(near)
        soft.append(f"바로 위 저항({lab0} {_fmt_price(v0)})에 근접")

    if hard:
        level = "SELL_ALL"
    elif len(soft) >= 2:
        level = "SELL_HALF"
    elif len(soft) == 1:
        level = "CAUTION"
    else:
        level = "HOLD"

    q1 = _qty_floor(qty * 0.5, qty_dec)
    q2 = qty - q1
    return dict(info=info, px=px, last_close=last_close, atr=atr, trail=trail, trail_base=hh, tp=tp, tp_label=tp_lab,
                hard=hard, soft=soft, level=level, q1=q1, q2=q2, mtf=mtf,
                pnl=(px - avg_px) * qty, pnl_pct=(px / avg_px - 1) * 100 if avg_px else None,
                pnl_tp=(tp - avg_px) * q1 if tp is not None else None,
                pnl_trail=(trail - avg_px) * q2 if trail is not None else None,
                pnl_trail_all=(trail - avg_px) * qty if trail is not None else None)


SELL_STYLE = {
    "SELL_ALL":  dict(title="🔴 즉시 전량 매도", bg="#c62828"),
    "SELL_HALF": dict(title="🟠 절반 익절 (비중 축소)", bg="#e65100"),
    "CAUTION":   dict(title="🟡 보유 · 주의", bg="#b28704"),
    "HOLD":      dict(title="🟢 보유 유지", bg="#2e7d32"),
}


POS_STYLE = {
    "STOP":    dict(title="🔴 손절: 전량 매도", bg="#c62828"),
    "EXIT":    dict(title="🔴 전량 매도 (이익 보호)", bg="#c62828"),
    "HALF":    dict(title="🟠 절반 익절 (비중 축소)", bg="#e65100"),
    "TP1":     dict(title="🟠 1차 익절 (50%)", bg="#e65100"),
    "ADD":     dict(title="🔵 추가 매수", bg="#1565c0"),
    "CAUTION": dict(title="🟡 보유 · 주의", bg="#b28704"),
    "HOLD":    dict(title="🟢 보유 유지", bg="#2e7d32"),
}
POS_TP1_R = 2.0        # 1차 익절 최소 거리 (평단 − 손절가의 2배)
POS_MODES = ["사다리 1차까지 매수", "사다리 2차까지 매수", "사다리 3차까지 매수", "사다리와 무관하게 보유"]


def position_suggest_stop(df, avg_px, qty, bought, qty_dec=0, equity=10_000_000.0, rate=1.0):
    """포지션 관리 탭의 '내 손절가' 기본값 (종목 통화). 사다리면 무효화 손절, 아니면 트레일링"""
    r = sell_analysis(df, avg_px, qty, qty_dec)
    if bought:
        ict = ict_analysis(df, r['info'])
        lad = ict_ladder(r['info'], ict, equity, rate, qty_dec)
        if lad and not lad.get('err') and lad.get('sl'):
            return lad['sl'] / rate, "사다리 무효화 손절 (스윕 저점 − 0.25 ATR)"
        if r['info']['levels'].get('issl') and r['info']['levels']['issl'] < r['px']:
            return r['info']['levels']['issl'] - ICT_STOP_BUF_ATR * r['atr'], "직전 저점 − 0.25 ATR"
    if r['trail']:
        return r['trail'], f"트레일링 (최근 {SELL_TRAIL_N}봉 고점 − {TP3_TRAIL_ATR:g} ATR)"
    return r['px'] - 2 * r['atr'], "현재가 − 2 ATR"


POS_EXIT_MODES = ["추세 꺾임 분할 매도 (상승 추세가 꺾일 때 나눠 팔기)", "목표가 절반 익절 (평단 + 손절폭 2배 이상 첫 저항)"]


def position_plan(df, avg_px, qty, bought, tp1_done, my_stop, qty_dec=0, equity=10_000_000.0, rate=1.0, exit_mode="trend"):
    """
    보유 종목 하나의 '지금 할 일'과 손절가 (가격은 종목 통화, 손실 금액은 원화 환산도 함께)
      bought   : 0 = 사다리와 무관, 1·2·3 = 사다리 몇 차까지 샀는지
      tp1_done : 1차 익절(50%)을 이미 했는지
      my_stop  : 매수할 때 정한 손절가 (내리지 않음)
    손절가 단계: 1차 익절 전 = 내 손절가 → 1차 익절 후 = max(내 손절가, 평단가(본절), 트레일링)
    판단 순서: 손절 → 전량 매도 → 절반 익절 → 1차 익절 → 추가 매수 → 주의 → 보유
    """
    r = sell_analysis(df, avg_px, qty, qty_dec)
    info = r['info']
    ict = ict_analysis(df, info)
    lad = ict_ladder(info, ict, equity, rate, qty_dec)
    px, lc, atr = r['px'], r['last_close'], r['atr']
    L = info['levels']
    # 1차 익절 목표: 평단 + 2R(R = 평단 − 내 손절가) 이상에 있는 첫 저항, 없으면 평단 + 2R
    risk = avg_px - my_stop if my_stop < avg_px else (atr or avg_px * 0.05)
    floor_tp = avg_px + POS_TP1_R * risk
    cands = sorted((v, lab) for lab, v in (L.get('resistances') or []) if v is not None and np.isfinite(v) and v >= floor_tp)
    tp_hold, tp_lab = cands[0] if cands else (floor_tp, f"손익비 {POS_TP1_R:g}R")
    trail = r['trail']
    trend = exit_mode == "trend"
    be_line, be_hit = None, False
    if trend and bought:
        chl = ((ict or {}).get('choch') or {}).get('level') or ((ict or {}).get('sweep') or {}).get('choch_level')
        if not chl and lad and not lad.get('err'):
            r2 = next((x for x in lad['rows'] if x['n'] == 2), None)
            chl = r2['price'] / rate if r2 else None
        if not chl and L.get('ibsl') and atr and L['ibsl'] <= avg_px + 4 * atr:
            chl = L['ibsl']
        be_line = chl if (chl and chl > avg_px) else None
        be_hit = bool(be_line and lc >= be_line)
        tp_hold = None                                   # 추세 모드는 목표가 익절 없음
    if bought == 0:                      # 사다리와 무관: 기존 매도 규칙(트레일링 · 구조 붕괴)
        cands2 = [(my_stop, "내 손절가")] + ([(trail, "트레일링")] if trail else [])
        stop, stop_kind = max(cands2, key=lambda x: x[0])
        tp_hold = None
    elif tp1_done or be_hit:
        cands = [(my_stop, "내 손절가"), (avg_px, "본절(평단가)")] + ([(trail, "트레일링")] if (trail and trail < lc) else [])
        stop, stop_kind = max(cands, key=lambda x: x[0])
    else:
        stop, stop_kind = my_stop, "내 손절가 (매수 때 정한 값)"
    one_unit = qty_dec == 0 and qty < 2
    reasons, action, add_row = [], "HOLD", None
    if lc < stop:
        action = "STOP" if stop < avg_px else "EXIT"
        reasons.append(f"종가 {_fmt_price(lc)}가 손절가 {_fmt_price(stop)}({stop_kind}) 아래로 마감했습니다. 다음 봉에 전량 매도합니다.")
    elif r['hard'] and (tp1_done or bought == 0 or be_hit):
        action = "EXIT"
        reasons += r['hard'] + (["상승 추세가 꺾였습니다 — 남은 물량을 모두 정리합니다."] if trend else [])
    elif not tp1_done and len(r['soft']) >= 2 and one_unit:
        action = "CAUTION"
        reasons += r['soft']
        if trail and trail < lc and trail > stop:
            stop, stop_kind = trail, "트레일링 (1주라 나눠 팔 수 없어 손절선을 올림)"
        reasons.append("추세 꺾임 경고가 2개 이상이지만 1주라 나눠 팔 수 없습니다 → 손절선을 트레일링까지 올리고, 구조가 무너지면 전량 매도합니다.")
    elif not tp1_done and len(r['soft']) >= 2:
        action = "HALF"
        reasons += r['soft'] + (["추세 꺾임 경고가 2개 이상입니다 — 절반을 팔고, 나머지는 구조가 무너질 때까지 들고 갑니다."] if trend else [])
    elif not tp1_done and tp_hold and px >= tp_hold:
        action = "TP1"
        reasons.append(f"1차 익절가 {_fmt_price(tp_hold)}({tp_lab})에 도달했습니다. 절반을 팔고 남은 물량의 손절가를 평단가 이상으로 올리세요.")
    else:
        if bought in (1, 2) and lad and not lad.get('err'):
            add_row = next((x for x in lad['rows'] if x['n'] == bought + 1 and x['status'] == "지금"), None)
        if add_row and px > stop:
            action = "ADD"
            reasons.append(f"사다리 {add_row['n']}차 조건이 지금 충족됐습니다. 손절가는 그대로 두고 {add_row['n']}차를 추가합니다.")
        elif r['soft']:
            action = "CAUTION"
            reasons += r['soft']
            reasons.append("징후가 하나뿐이라 아직 팔지 않고, 손절가를 지키며 지켜봅니다.")
        else:
            reasons.append("손절가 위에서 정상 보유 중입니다. 손절 주문을 걸어 둔 채 기다립니다.")
    if r['hard'] and action not in ("STOP", "EXIT"):
        reasons.append("⚠️ " + " ".join(r['hard']) + " — 다만 추세 확인(본절 이동) 전에는 '내 손절가'를 기준으로 판단합니다.")
    if trend and bought and be_line and not be_hit and action in ("HOLD", "CAUTION", "ADD"):
        reasons.append(f"종가가 {_fmt_price(be_line)}(돌파선) 위로 올라서면 손절가를 평단가(본절) 이상으로 올립니다.")
    loss = (stop - avg_px) * qty
    return dict(r=r, ict=ict, lad=lad, action=action, reasons=reasons, stop=stop, stop_kind=stop_kind,
                dist=(stop / px - 1), loss=loss, loss_krw=loss * rate, loss_pct=loss * rate / equity * 100,
                tp=tp_hold, tp_label=tp_lab, trail=trail, add_row=add_row, px=px, last_close=lc,
                trend=trend, be_line=be_line, be_hit=be_hit,
                pnl=(px - avg_px) * qty, pnl_pct=(px / avg_px - 1) * 100)


# ==========================================
# 🔎 스크리너: 종목 목록 (섹터별 대표 종목 · 직접 고쳐 쓰세요)
# ==========================================
SCREEN_UNIVERSE = {
    "🇰🇷 국내 (KOSPI)": {
        "반도체": [("005930.KS", "삼성전자"), ("000660.KS", "SK하이닉스"), ("042700.KS", "한미반도체"), ("009150.KS", "삼성전기"), ("000990.KS", "DB하이텍")],
        "2차전지": [("373220.KS", "LG에너지솔루션"), ("006400.KS", "삼성SDI"), ("005490.KS", "POSCO홀딩스"), ("051910.KS", "LG화학"), ("003670.KS", "포스코퓨처엠")],
        "방산·항공우주": [("012450.KS", "한화에어로스페이스"), ("047810.KS", "한국항공우주"), ("064350.KS", "현대로템"), ("079550.KS", "LIG넥스원"), ("272210.KS", "한화시스템")],
        "조선": [("009540.KS", "HD한국조선해양"), ("329180.KS", "HD현대중공업"), ("042660.KS", "한화오션"), ("010140.KS", "삼성중공업"), ("443060.KS", "HD현대마린솔루션")],
        "전력기기": [("267260.KS", "HD현대일렉트릭"), ("010120.KS", "LS ELECTRIC"), ("298040.KS", "효성중공업"), ("034020.KS", "두산에너빌리티"), ("006260.KS", "LS")],
        "원전·유틸리티": [("015760.KS", "한국전력"), ("052690.KS", "한전기술"), ("051600.KS", "한전KPS"), ("000720.KS", "현대건설"), ("036460.KS", "한국가스공사")],
        "자동차": [("005380.KS", "현대차"), ("000270.KS", "기아"), ("012330.KS", "현대모비스"), ("204320.KS", "HL만도"), ("086280.KS", "현대글로비스")],
        "바이오": [("207940.KS", "삼성바이오로직스"), ("068270.KS", "셀트리온"), ("000100.KS", "유한양행"), ("326030.KS", "SK바이오팜"), ("128940.KS", "한미약품")],
        "인터넷·게임": [("035420.KS", "NAVER"), ("035720.KS", "카카오"), ("259960.KS", "크래프톤"), ("036570.KS", "엔씨소프트"), ("251270.KS", "넷마블")],
        "금융": [("105560.KS", "KB금융"), ("055550.KS", "신한지주"), ("086790.KS", "하나금융지주"), ("316140.KS", "우리금융지주"), ("138040.KS", "메리츠금융지주")],
    },
    "🇺🇸 미국": {
        "빅테크": [("AAPL", "애플"), ("MSFT", "마이크로소프트"), ("GOOGL", "알파벳"), ("AMZN", "아마존"), ("META", "메타")],
        "반도체·AI": [("NVDA", "엔비디아"), ("AVGO", "브로드컴"), ("TSM", "TSMC"), ("AMD", "AMD"), ("MU", "마이크론")],
        "소프트웨어·AI": [("PLTR", "팔란티어"), ("ORCL", "오라클"), ("CRM", "세일즈포스"), ("NOW", "서비스나우"), ("ADBE", "어도비")],
        "사이버보안": [("CRWD", "크라우드스트라이크"), ("PANW", "팔로알토"), ("FTNT", "포티넷"), ("ZS", "지스케일러"), ("NET", "클라우드플레어")],
        "방산": [("LMT", "록히드마틴"), ("RTX", "RTX"), ("NOC", "노스롭그루먼"), ("GD", "제너럴다이내믹스"), ("LHX", "L3해리스")],
        "항공우주·우주": [("GE", "GE에어로스페이스"), ("BA", "보잉"), ("HWM", "하우멧"), ("RKLB", "로켓랩"), ("ASTS", "AST스페이스모바일")],
        "전력·원전": [("GEV", "GE버노바"), ("CEG", "컨스텔레이션"), ("VST", "비스트라"), ("NEE", "넥스트에라"), ("OKLO", "오클로")],
        "금융": [("JPM", "JP모건"), ("V", "비자"), ("MA", "마스터카드"), ("BAC", "뱅크오브아메리카"), ("GS", "골드만삭스")],
        "헬스케어": [("LLY", "일라이릴리"), ("UNH", "유나이티드헬스"), ("JNJ", "존슨앤존슨"), ("ABBV", "애브비"), ("MRK", "머크")],
        "에너지": [("XOM", "엑슨모빌"), ("CVX", "쉐브론"), ("COP", "코노코필립스"), ("SLB", "SLB"), ("EOG", "EOG")],
        "소비·모빌리티": [("TSLA", "테슬라"), ("COST", "코스트코"), ("WMT", "월마트"), ("HD", "홈디포"), ("UBER", "우버")],
        "ETF·지수": [("SPY", "S&P500"), ("QQQ", "나스닥100"), ("DIA", "다우"), ("IWM", "러셀2000"), ("VTI", "미국 전체")],
        "ETF·레버리지(3x)": [("SOXL", "반도체 3x"), ("TQQQ", "나스닥 3x"), ("UPRO", "S&P500 3x"), ("TECL", "기술주 3x"), ("TNA", "러셀2000 3x")],
        "ETF·섹터테마": [("SOXX", "반도체"), ("SMH", "반도체(반에크)"), ("ITA", "항공우주·방산"), ("XLE", "에너지"), ("ARKK", "혁신기술"),
                       ("XLK", "기술주"), ("XLF", "금융"), ("XLV", "헬스케어")],
        "ETF·배당채권금": [("SCHD", "배당성장"), ("JEPI", "커버드콜 S&P"), ("JEPQ", "커버드콜 나스닥"), ("VYM", "고배당"), ("VTV", "가치주"),
                        ("TLT", "미국채 20년+"), ("IEF", "미국채 7-10년"), ("GLD", "금")],
    },
    "🪙 코인": {
        "대장·메이저 알트": [("BTC/USDT", "비트코인"), ("ETH/USDT", "이더리움"), ("BNB/USDT", "BNB"), ("SOL/USDT", "솔라나"), ("XRP/USDT", "리플"),
                         ("DOGE/USDT", "도지코인"), ("ADA/USDT", "에이다"), ("TRX/USDT", "트론"), ("LINK/USDT", "체인링크"), ("AVAX/USDT", "아발란체")],
    },
}
def tab_groups(market, etf=None):
    """트레이딩 룸 탭용 종목 목록: SCREEN_UNIVERSE를 그대로 읽어 {섹터: ["티커 (이름)", ...]} 형태로 변환.
    etf=True면 ETF 그룹만, False면 ETF 그룹 제외, None이면 전부. (종목은 SCREEN_UNIVERSE 한곳에서만 고치면 됨)"""
    out = {}
    for sec, items in SCREEN_UNIVERSE.get(market, {}).items():
        is_etf = sec.startswith("ETF")
        if etf is True and not is_etf or etf is False and is_etf:
            continue
        out[sec.replace("ETF·", "") if etf else sec] = [f"{t} ({n})" for t, n in items]
    return out


CANDIDATE_STAGES = ["🟢 1차: 스윕 후 회복", "🟢 2차: CHoCH 확정", "🟢 3차: 리테스트", "⏳ CHoCH 안착 대기",
                    "⏳ 1차 되돌림 대기", "⏳ 3차 리테스트 대기", "🪤 회복 실패 · 재회복 대기", "🧊 손절 후 휴식", "🎯 수요 구간 도달",
                    "🧲 셋업", "🔵 매집 관찰"]          # 무릎 매수 후보 (우선순위 순)
STAGE_ORDER = CANDIDATE_STAGES + ["🎯 트리거(셋업O)", "⏳ 안착 대기", "📈 돌파(셋업X)", "🚫 추격 구간", "🟢 상승", "🟣 분배",
                                  "⛔ 200일선 하락 중", "🪤 무효화(스윕 실패)", "🪤 휩쏘", "🔴 하락", "⚠️ 데이터 오류"]
BROKE_STAGES = ("🟢 2차: CHoCH 확정", "🟢 3차: 리테스트", "⏳ CHoCH 안착 대기", "⏳ 3차 리테스트 대기",
                "🎯 트리거(셋업O)", "📈 돌파(셋업X)", "⏳ 안착 대기", "🚫 추격 구간")          # 이미 돌파선을 넘은 단계


def classify_stage(info):
    """기존 엔진의 진입 상태·국면 → 스크리너 단계. 셋업 있는 돌파(TRIGGER)와 셋업 없는 돌파(BREAKOUT)를 분리"""
    entry, phase = info.get("entry"), info.get("phase")
    if entry in ("TRIGGER", "BREAKOUT"):
        if info.get("extended"):
            return "🚫 추격 구간"
        return "🎯 트리거(셋업O)" if entry == "TRIGGER" else "📈 돌파(셋업X)"
    if entry == "SETUP":
        return "🧲 셋업"
    if entry == "PENDING":
        return "⏳ 안착 대기"
    if entry == "WHIPSAW":
        return "🪤 휩쏘"
    if entry == "BLOCKED":
        return "⛔ 200일선 하락 중"
    if entry == "DOWN":
        return "🔴 하락"
    return {"매집": "🔵 매집 관찰", "상승": "🟢 상승", "분배": "🟣 분배"}.get(phase, "🔴 하락")


def _ret_since(df, days):
    t = pd.to_datetime(df["time"])
    past = df[t <= t.iloc[-1] - pd.Timedelta(days=days)]
    if past.empty:
        return None
    return float(df["close"].iloc[-1] / past["close"].iloc[-1] - 1)


def _flow_metrics(df):
    """종목 하나의 자금 유입 지표: 거래대금 증가율(최근 20일 ÷ 그 전 120일 평균 − 1), CMF20, 50·200일선 위치"""
    c, v = df["close"].astype(float), df["volume"].astype(float) if "volume" in df.columns else None
    out = dict(거래대금증가=None, CMF=None, 오십일선거리=None, 이백일선위=None)
    if v is not None and len(df) >= 60 and v.tail(20).sum() > 0:
        dv = c * v
        prev = dv.iloc[-140:-20] if len(dv) >= 140 else dv.iloc[:-20]
        if len(prev) >= 20 and prev.mean() > 0:
            out["거래대금증가"] = float(dv.tail(20).mean() / prev.mean() - 1)
        h, l = df["high"].astype(float), df["low"].astype(float)
        mfm = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
        vv = v.tail(20)
        if vv.sum() > 0:
            out["CMF"] = float((mfm.tail(20).fillna(0) * vv).sum() / vv.sum())
    if len(c) >= 50:
        out["오십일선거리"] = float(c.iloc[-1] / c.rolling(50).mean().iloc[-1] - 1)
    if len(c) >= 200:
        out["이백일선위"] = bool(c.iloc[-1] > c.rolling(200).mean().iloc[-1])
    return out


def _verdict_label(v):
    if v["kind"] == "BUY" and v.get("grade") in GRADE_STYLE and GRADE_ACTIVE:
        t = GRADE_STYLE[v["grade"]]["title"]
    elif v.get("grade") == "C":
        t = "⏳ C급 · 매수 보류"
    elif v.get("confirm"):
        t = "⏳ 확인 대기"
    else:
        t = VERDICT_STYLE[v["kind"]]["title"]
    return t + (f" {v['tranche']}차" if v.get("tranche") else "")


def finalize_grades(res, ov):
    """섹터 자금 흐름(상위 1/3 = 통과)을 8번째 조건으로 넣어 등급·결론을 다시 매김 (스크리너 전용)"""
    if res.empty or "_chk" not in res.columns:
        return res
    ranks = {} if ov is None or ov.empty else {(m, s_): r for m, s_, r in zip(ov["시장"], ov["섹터"], ov["흐름순위"])}
    valid = [r for r in ranks.values() if r is not None and np.isfinite(r)]
    top_cut = max(3, int(np.ceil(max(valid) / 3))) if valid else None
    res = res.copy()
    for i, r in res.iterrows():
        chk = r.get("_chk")
        if not isinstance(chk, list) or not chk:
            continue
        rk = ranks.get((r["시장"], r["섹터"]))
        flow = None if (top_cut is None or rk is None or not np.isfinite(rk)) else bool(rk <= top_cut)
        g, miss = grade_from([(k, ok, "") for k, ok in chk] + [("flow", flow, "")])
        res.at[i, "등급"], res.at[i, "빠진조건"] = g, ", ".join(miss)
        if GRADE_ACTIVE and (r.get("_kind") == "BUY" or (r.get("_kind") == "WAIT" and r.get("등급") == "C")):
            tr = r.get("_tranche")
            if not (isinstance(tr, (int, float)) and np.isfinite(tr)):
                tr = {"🟢 1차: 스윕 후 회복": 1, "🟢 2차: CHoCH 확정": 2, "🟢 3차: 리테스트": 3}.get(r["단계"])
            res.at[i, "결론"] = (GRADE_STYLE[g]["title"] + (f" {int(tr)}차" if tr else "")) if g in GRADE_STYLE else "⏳ C급 · 매수 보류"
    return res


def screen_row(df, market, sector, ticker, name):
    """종목 하나를 상세 화면(1일)과 같은 규칙으로 판정: 기존 엔진 + 일봉 200일선 필터 + ICT 사다리 단계 + 신호 등급"""
    row = dict(시장=market, 섹터=sector, 티커=ticker, 종목=name)
    try:
        bs = bar_status(df)
        row['기준봉'] = bs['label']
        df = closed_bars(df)                       # 상세 화면과 같은 '마감 봉' 기준
        mtf = daily_regime(df)
        info = analyze_market_cycle(df, mtf, None, False)
        ict = ict_analysis(df, info)
    except Exception as e:
        row.update(단계="⚠️ 데이터 오류", 결론="-", 비고=f"{type(e).__name__}")
        return row
    L = info["levels"]
    price = L.get("price")
    stage = ICT_STAGE_LABEL.get(ict.get("stage")) or classify_stage(info)
    plan_trig = build_trade_plan(info) if info["entry"] == "TRIGGER" else None
    verdict = ict_verdict(info, ict, plan_trig)
    lad = ict_ladder(info, ict, 1_000_000.0, 1.0, None)                # 손절선·다음 가격 계산용 (금액은 쓰지 않음)
    stop = lad["sl"] if (lad and not lad.get("err")) else None
    nxt = next((r for r in (lad or {}).get("rows", []) if r["status"] == "지금" or r["status"].startswith("대기")), None) \
        if (lad and not lad.get("err")) else None
    broke = stage in BROKE_STAGES
    ref = build_trade_plan(info, entry=price, assume_trigger=True) if price else None
    target = L["ibsl"] if (L.get("ibsl") and price and L["ibsl"] > price and not broke) else (ref["tp_struct"] if ref else None)
    tri = info.get("triple") or {}
    close = df["close"]
    tv = (close * df["volume"]).tail(20).mean() if "volume" in df.columns else None
    zn = (ict.get("zones") or [None])[0]
    rg = ict.get("range") or {}
    row.update(
        결론=_verdict_label(verdict),
        등급=verdict.get("grade") or "-", 빠진조건=", ".join(verdict.get("missing") or []),
        _chk=[(k, ok) for k, ok, _d in (verdict.get("checks") or []) if k != "flow"] or None,
        _kind=verdict["kind"], _tranche=verdict.get("tranche"),
        단계=stage, 국면=info["phase"], 국면점수=info["score"], 매집점수=info["totals"].get("매집", 0),
        진바닥3박자="".join("✓" if tri.get(k) else "✗" for k in ("crashed", "consolidating", "hl")),
        현재가=price, 돌파완료=broke,
        트리거까지=None if (broke or (nxt and nxt["status"] == "지금")) else ((nxt["price"] / price - 1) if (nxt and price) else None),
        다음차수=(f"{nxt['n']}차" + (" (지금)" if nxt["status"] == "지금" else "") if nxt else "-"),
        손절선까지=(stop / price - 1) if (stop and price) else None,
        여력위험비=((target - price) / (price - stop)) if (stop and target and price and target > price > stop) else None,
        레인지위치=rg.get("pos"), 수요구간=(f"{_fmt_price(zn['lo'])}~{_fmt_price(zn['hi'])}" if zn else "-"),
        보조필터=sum(bool(x) for x in (ict.get("filters") or {}).values()),
        이백일선=("-" if mtf.get("state") == "n/a" else
                ("위" if mtf["gap_pct"] >= 0 else "아래") + ("↗" if mtf["slope_pct"] > 0 else "↘")),
        수익률1M=_ret_since(df, 30), 수익률3M=_ret_since(df, 91), 거래대금=tv, 비고="",
        주봉=(lambda m: f"{m['icon']} {m['short']}")(W_STATE_META.get(((ict.get("wict") or {}).get("state")) or "W_NA", W_STATE_META["W_NA"])))
    row.update(_flow_metrics(df))
    return row


def _yf_batch(tickers, chunk=20):
    """yfinance 묶음 조회를 chunk개씩 나눠서 실행 (메모리·요청 제한 대비). {티커: 원본 DataFrame}"""
    frames = {}
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        try:
            big = yf.download(part, period="2y", interval="1d", group_by="ticker", progress=False, threads=True, timeout=30)
        except Exception:
            continue
        if big is None or big.empty:
            continue
        for t in part:
            if isinstance(big.columns, pd.MultiIndex):
                if t in big.columns.get_level_values(0):
                    frames[t] = big[t].copy()
            elif len(part) == 1:
                frames[t] = big.copy()
        del big
    return frames


@st.cache_data(ttl=3600, show_spinner=False)
def scan_market(market):
    """시장 하나 전체 스캔 (1시간 캐시). 주식은 yfinance 20종목 단위 묶음 조회, 코인은 거래소 순차 조회"""
    groups = SCREEN_UNIVERSE[market]
    rows = []
    if market == "🪙 코인":
        for sector, items in groups.items():
            for sym, name in items:
                try:
                    df, _src, err = load_crypto(sym, "1일")
                except Exception as e:
                    df, err = pd.DataFrame(), type(e).__name__
                if df is None or df.empty:
                    rows.append(dict(시장=market, 섹터=sector, 티커=sym, 종목=name, 단계="⚠️ 데이터 오류", 결론="-", 비고=str(err)[:60]))
                else:
                    rows.append(screen_row(df, market, sector, sym, name))
        return rows
    tickers = [t for items in groups.values() for t, _ in items]
    frames = _yf_batch(tickers)
    for sector, items in groups.items():
        for t, name in items:
            df, err = pd.DataFrame(), "데이터 없음"
            try:
                sub = frames.pop(t, None)                      # 처리한 원본은 바로 버려 메모리 절약
                if sub is not None:
                    sub = sub.dropna(subset=[c for c in sub.columns if str(c).lower() == "close"])
                    if len(sub) >= 120:
                        df, err = _prep_yf_frame(sub, "1일")
                        df = tag_df(df, market_of(t), "1일")
                    else:
                        err = f"데이터 부족({len(sub)}봉)"
            except Exception as e:
                df, err = pd.DataFrame(), type(e).__name__
            if df is None or df.empty:
                rows.append(dict(시장=market, 섹터=sector, 티커=t, 종목=name, 단계="⚠️ 데이터 오류", 결론="-", 비고=str(err)[:60]))
            else:
                rows.append(screen_row(df, market, sector, t, name))
    return rows


FLOW_W = {"r3": 0.35, "r1": 0.25, "surge": 0.25, "cmf": 0.15}     # 자금 흐름 점수 가중치 (백분위 순위의 가중 평균)
FLOW_QUAD = {"주도": "🔥 주도", "약화": "🌗 주도 · 쉬어가는 중", "개선": "🌱 새로 유입", "침체": "🧊 소외"}


def _pct_rank(sr):
    sr = pd.to_numeric(sr, errors="coerce")
    return sr.rank(pct=True, method="average") * 100


def flow_score_frame(df, r3="r3", r1="r1", surge="surge", cmf="cmf"):
    """여러 그룹(섹터·테마)의 3개월·1개월 수익률, 거래대금 증가, CMF → 백분위 순위의 가중 평균 = 자금 흐름 점수(0~100)
    국면: 3개월·1개월 순위가 모두 상위 절반 = 주도, 3개월만 상위 = 약화(쉬어가는 중), 1개월만 상위 = 개선(새로 유입), 둘 다 하위 = 침체.
    순위는 상대값이라 '이 목록 안에서' 어디가 상대적으로 강한지를 뜻합니다 (절대적인 상승·하락이 아님)."""
    d = df.copy()
    rk = {k: _pct_rank(d[col]) for k, col in (("r3", r3), ("r1", r1), ("surge", surge), ("cmf", cmf))}
    num = sum(FLOW_W[k] * rk[k].fillna(50) for k in rk)
    d["흐름점수"] = num.round(0)
    q3, q1 = rk["r3"].fillna(50), rk["r1"].fillna(50)
    d["흐름국면"] = np.where(q3 >= 50, np.where(q1 >= 50, FLOW_QUAD["주도"], FLOW_QUAD["약화"]),
                         np.where(q1 >= 50, FLOW_QUAD["개선"], FLOW_QUAD["침체"]))
    d["흐름순위"] = d["흐름점수"].rank(ascending=False, method="min").astype(int)
    return d


def _is_flow_sector(sec):
    """자금 흐름 순위에 넣을 그룹: 실제 업종만 (지수·레버리지·배당채권 ETF 묶음은 업종이 아니므로 제외)"""
    return not str(sec).startswith("ETF·") or sec == "ETF·섹터테마"


def sector_overview(res):
    """섹터 한눈에 보기 + 자금 흐름 순위: 3개월·1개월 수익률(중앙값), 거래대금 증가, CMF → 흐름 점수 · 국면"""
    ok = res[res["단계"] != "⚠️ 데이터 오류"]
    if ok.empty:
        return pd.DataFrame()
    out = []
    for (mk, sec), g in ok.groupby(["시장", "섹터"], sort=False):
        cnt = g["국면"].value_counts()
        num = lambda col: pd.to_numeric(g[col], errors="coerce") if col in g.columns else pd.Series(dtype=float)
        out.append({
            "시장": mk, "섹터": sec, "종목수": len(g),
            "후보": int(g["단계"].isin(CANDIDATE_STAGES).sum()),
            "🟢매수": int(g["결론"].astype(str).str.startswith("🟢").sum()),
            "🔵매집": int(cnt.get("매집", 0)), "🟢상승": int(cnt.get("상승", 0)),
            "🟣분배": int(cnt.get("분배", 0)), "🔴하락": int(cnt.get("하락", 0)),
            "200일선 위": f"{g['이백일선'].astype(str).str.startswith('위').mean() * 100:.0f}%",
            "주봉 상승": f"{g['주봉'].astype(str).str.contains('상승|전환').mean() * 100:.0f}%" if "주봉" in g.columns else "-",
            "평균 1개월": num("수익률1M").median(), "평균 3개월": num("수익률3M").median(),
            "거래대금 증가": num("거래대금증가").median(), "CMF": num("CMF").median(),
        })
    ov = pd.DataFrame(out)
    fl = ov[ov["섹터"].map(_is_flow_sector)]
    if len(fl) >= 3:
        sc = flow_score_frame(fl, "평균 3개월", "평균 1개월", "거래대금 증가", "CMF")
        ov = ov.join(sc[["흐름점수", "흐름국면", "흐름순위"]])
    for col in ("흐름점수", "흐름국면", "흐름순위"):
        if col not in ov.columns:
            ov[col] = None
    ov["_k"] = pd.to_numeric(ov["흐름점수"], errors="coerce").fillna(-1)
    return ov.sort_values(["_k", "후보"], ascending=[False, False]).drop(columns="_k")


def _fmt_pct(v, signed=True):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    return f"{v * 100:+.1f}%" if signed else f"{v * 100:.1f}%"


def _fmt_tv(v, market):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    if market == "🇰🇷 국내 (KOSPI)":
        return f"₩{v / 1e8:,.0f}억"
    return f"${v / 1e6:,.0f}M"


def _fmt_px(v, market):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    if market == "🇰🇷 국내 (KOSPI)":
        return f"₩{v:,.0f}"
    return f"${v:,.2f}" if v >= 1 else f"${v:,.4f}"


# ==========================================
# 🌊 글로벌 자금 흐름: 섹터·테마·국가·자산 ETF 비교 (모두 달러 기준이라 SPY 대비 상대강도를 바로 비교 가능)
# ==========================================
FLOW_UNIVERSE = {
    "미국 업종": [("XLK", "기술"), ("XLC", "통신·인터넷"), ("XLY", "임의소비"), ("XLP", "필수소비"), ("XLF", "금융"), ("XLV", "헬스케어"),
              ("XLI", "산업재"), ("XLE", "에너지"), ("XLB", "소재"), ("XLU", "유틸리티"), ("XLRE", "부동산")],
    "테마": [("SMH", "반도체"), ("IGV", "소프트웨어"), ("CIBR", "사이버보안"), ("BOTZ", "로봇·AI"), ("ITA", "방산·항공우주"),
           ("UFO", "우주"), ("URA", "우라늄·원자력"), ("NLR", "원자력 발전"), ("GRID", "전력망 인프라"), ("TAN", "태양광"),
           ("LIT", "2차전지·리튬"), ("XBI", "바이오"), ("KWEB", "중국 인터넷"), ("ARKK", "혁신기술")],
    "국가": [("SPY", "미국"), ("EWY", "한국"), ("EWJ", "일본"), ("EWT", "대만"), ("FXI", "중국"), ("INDA", "인도"),
           ("VGK", "유럽"), ("EEM", "신흥국")],
    "자산": [("TLT", "미국 장기채"), ("IEF", "미국 중기채"), ("GLD", "금"), ("SLV", "은"), ("USO", "원유"), ("UUP", "달러"),
           ("BTC-USD", "비트코인"), ("ETH-USD", "이더리움")],
}
FLOW_BENCH = "SPY"
# 흐름 테마 → 스크리너 섹터 (흐름이 강한 곳에서 '싸게 살' 종목을 바로 찾기 위한 연결)
FLOW_TO_SECTORS = {
    "XLK": ["빅테크", "반도체·AI", "소프트웨어·AI"], "XLC": ["빅테크", "인터넷·게임"], "XLY": ["소비·모빌리티", "자동차"],
    "XLF": ["금융"], "XLV": ["헬스케어", "바이오"], "XLI": ["전력기기", "조선", "항공우주·우주"], "XLE": ["에너지"],
    "XLU": ["전력·원전", "원전·유틸리티"], "SMH": ["반도체", "반도체·AI"], "IGV": ["소프트웨어·AI", "인터넷·게임"],
    "CIBR": ["사이버보안"], "BOTZ": ["반도체·AI", "소프트웨어·AI"], "ITA": ["방산", "방산·항공우주", "항공우주·우주"],
    "UFO": ["항공우주·우주", "방산·항공우주"], "URA": ["전력·원전", "원전·유틸리티"], "NLR": ["전력·원전", "원전·유틸리티"],
    "GRID": ["전력기기", "전력·원전"], "LIT": ["2차전지"], "XBI": ["바이오", "헬스케어"],
    "EWY": ["반도체", "자동차", "조선", "금융"], "BTC-USD": ["대장·메이저 알트"], "ETH-USD": ["대장·메이저 알트"],
}
FLOW_PULLBACK_MIN = 60      # '강한 흐름 속 눌림'으로 볼 최소 흐름 점수


@st.cache_data(ttl=3600, show_spinner=False)
def scan_flow():
    """FLOW_UNIVERSE 전체를 한 번에 조회해 그룹별 흐름 지표 계산 (1시간 캐시). 실패한 티커는 건너뜀"""
    items = [(grp, t, n) for grp, lst in FLOW_UNIVERSE.items() for t, n in lst]
    frames = _yf_batch([t for _, t, _ in items])
    data = {}
    for _grp, t, _n in items:
        sub = frames.pop(t, None)
        if sub is None:
            continue
        try:
            sub = sub.dropna(subset=[c for c in sub.columns if str(c).lower() == "close"])
            if len(sub) >= 130:
                d, _e = _prep_yf_frame(sub, "1일")
                if d is not None and not d.empty:
                    data[t] = d
        except Exception:
            continue
    bench = data.get(FLOW_BENCH)
    b1, b3 = (_ret_since(bench, 30), _ret_since(bench, 91)) if bench is not None else (None, None)
    rows = []
    for grp, t, n in items:
        d = data.get(t)
        if d is None:
            rows.append(dict(그룹=grp, 티커=t, 이름=n, 오류=True))
            continue
        r1, r3, r6 = _ret_since(d, 30), _ret_since(d, 91), _ret_since(d, 182)
        rs = lambda r, b: ((1 + r) / (1 + b) - 1) if (r is not None and b is not None) else None
        fm = _flow_metrics(d)
        rows.append(dict(그룹=grp, 티커=t, 이름=n, 오류=False, 현재가=float(d["close"].iloc[-1]),
                         r1=r1, r3=r3, r6=r6, rs1=rs(r1, b1), rs3=rs(r3, b3),
                         surge=fm["거래대금증가"], cmf=fm["CMF"], d50=fm["오십일선거리"], above200=fm["이백일선위"],
                         spark=d["close"].tail(130).tolist()))
    return rows


def flow_table(rows):
    """scan_flow 결과 → 흐름 점수·국면·눌림 여부가 붙은 표 (오류 티커 제외)"""
    df = pd.DataFrame([r for r in rows if not r.get("오류")])
    if df.empty:
        return df
    df = flow_score_frame(df, "r3", "r1", "surge", "cmf")
    df["눌림"] = (df["흐름점수"] >= FLOW_PULLBACK_MIN) & df["흐름국면"].isin([FLOW_QUAD["주도"], FLOW_QUAD["약화"]]) & \
                 (df["above200"].fillna(False).astype(bool)) & \
                 ((pd.to_numeric(df["r1"], errors="coerce") < 0) | (pd.to_numeric(df["d50"], errors="coerce").abs() <= 0.03))
    return df.sort_values("흐름점수", ascending=False)


# ==========================================
# 사이드바 네비게이션
# ==========================================
st.sidebar.title("🧭 PRO 퀀트 네비게이션")
tabs = ["🪙 암호화폐 (Crypto)", "📈 국내 주식 (KOSPI)", "🦅 미국 주식 (US Stock)",
        "📊 미국 ETF (US ETF)", "🥇 금·은 원자재 (Commodities)", "🌊 자금 흐름 (Money Flow)", "🔎 스크리너 (Screener)", "💼 포지션 관리 (Position)"]
market_choice = st.sidebar.radio("트레이딩 룸 선택", tabs)

# ---------- 자금 관리: 결론 배너 · 사다리 · 포지션 관리가 함께 사용 ----------
st.sidebar.divider()
st.sidebar.markdown("##### 💰 자금 관리")
_eq = st.sidebar.number_input("총 투자자산 (원)", min_value=0.0, value=10_000_000.0, step=1_000_000.0, format="%.0f",
                              help="현금만이 아니라 이 전략에 쓰는 자산 전체(보유 종목 평가액 + 현금)를 넣으세요. 수량은 이 금액 기준으로 계산됩니다.")
EQUITY = float(_eq) if isinstance(_eq, (int, float)) and _eq > 0 else 10_000_000.0
_rk = st.sidebar.number_input("1회 허용 손실 (%)", min_value=0.25, max_value=5.0, value=1.0, step=0.25, format="%.2f",
                              help="손절됐을 때 총자산의 몇 %까지 잃을지. 큰 계좌는 1%, 100만~300만 원대 작은 계좌는 2% 정도가 현실적입니다.")
RISK_PER_TRADE = float(_rk) if isinstance(_rk, (int, float)) and _rk > 0 else 1.0
US_QTY_DEC = US_QTY_DEC_MAX if st.sidebar.checkbox("미국 주식 소수점 매수 (토스 등)", value=True,
                                                   help="끄면 미국 주식도 1주 단위로 계산합니다.") else 0
st.sidebar.caption(f"1회 허용 손실 {RISK_PER_TRADE:g}% = ₩{EQUITY * RISK_PER_TRADE / 100:,.0f} · 한 종목 최대 {MAX_POSITION_PCT:g}% · "
                   "총자산은 현금이 아니라 투자 자산 전체 기준")
TF_OPTIONS = ["15분", "1시간", "4시간", "1일", "주봉", "월봉"]

# ---------- 사이드바 하단: 선택한 트레이딩 룸에 맞는 매크로 지표만 표시 ----------
BTC_DOM, FNG = None, None      # 암호화폐 탭에서만 조회 (탭 본문에서도 사용)

def _sidebar_krw():
    st.sidebar.metric("USD/KRW 환율", f"₩{KRW_RATE:,.0f}")
    st.sidebar.caption(f"환율 출처: {KRW_SRC}")

st.sidebar.divider()
st.sidebar.markdown("##### 🌍 매크로 지표")

if market_choice == "🪙 암호화폐 (Crypto)":
    BTC_DOM = get_btc_dominance()
    FNG = get_fear_greed()
    st.sidebar.metric("BTC 도미넌스", f"{BTC_DOM:.2f}%" if BTC_DOM is not None else "N/A")
    st.sidebar.metric("공포·탐욕 지수 (F&G)", f"{FNG['value']} ({FNG['label']})" if FNG else "N/A")

elif market_choice == "📈 국내 주식 (KOSPI)":
    _sidebar_krw()

elif market_choice == "🦅 미국 주식 (US Stock)":
    _sidebar_krw()
    _macro_metric("VIX 변동성 지수", get_vix(), "{:.2f}", "{:+.2f}", delta_color="inverse")

elif market_choice == "📊 미국 ETF (US ETF)":
    _sidebar_krw()
    _macro_metric("미국 10년물 국채 금리", get_us10y(), "{:.2f}%", "{:+.2f}%p", delta_color="inverse")

elif market_choice == "🥇 금·은 원자재 (Commodities)":
    _sidebar_krw()
    _macro_metric("달러 인덱스 (DXY)", get_dxy(), "{:.2f}", "{:+.2f}", delta_color="inverse")

elif market_choice == "🌊 자금 흐름 (Money Flow)":
    _sidebar_krw()
    _macro_metric("VIX 변동성 지수", get_vix(), "{:.2f}", "{:+.2f}", delta_color="inverse")
    _macro_metric("달러 인덱스 (DXY)", get_dxy(), "{:.2f}", "{:+.2f}", delta_color="inverse")
    _macro_metric("미국 10년물 국채 금리", get_us10y(), "{:.2f}%", "{:+.2f}%p", delta_color="inverse")

elif market_choice == "🔎 스크리너 (Screener)":
    _sidebar_krw()
    _macro_metric("VIX 변동성 지수", get_vix(), "{:.2f}", "{:+.2f}", delta_color="inverse")

elif market_choice == "💼 포지션 관리 (Position)":
    _sidebar_krw()

st.sidebar.caption("조회 실패 시 N/A로 표시되며, 국면 점수에는 반영하지 않습니다.")


# ==========================================
# 1. 🪙 암호화폐 마켓
# ==========================================
if market_choice == "🪙 암호화폐 (Crypto)":
    st.title("🪙 암호화폐 현물 퀀트 봇")

    btc_d, fng = BTC_DOM, FNG   # 수치는 사이드바에 고정 표시, 이 탭에서는 해석만 제공
    if btc_d is None: st.info("ℹ️ 도미넌스 조회에 실패해 이 지표는 생략하고 기본 기술 지표로만 분석합니다.")
    elif btc_d > 55.0: st.warning("⚠️ **[수급 경고]** 도미넌스가 매우 높습니다. 알트코인 수급 이탈에 주의하세요.")
    elif btc_d < 45.0: st.success("🔥 **[알트 시즌]** 도미넌스가 낮아 알트코인으로 자금이 순환될 확률이 높습니다.")
    else: st.info("⚖️ 도미넌스 중립 구간입니다. 개별 차트에 집중하세요.")

    crypto_tabs = [t for items in SCREEN_UNIVERSE["🪙 코인"].values() for t, _ in items] + ["직접 검색"]
    selected_crypto = st.radio("관심 심볼", crypto_tabs, horizontal=True)
    target_symbol = st.text_input("심볼 커스텀 입력", value="DOGE/USDT").upper() if selected_crypto == "직접 검색" else selected_crypto.split(" ")[0]

    c1, c2 = st.columns([1.3, 2])
    with c2:
        selected_tf = st.radio("프레임 선택", TF_OPTIONS, index=3, horizontal=True, key="c_tf")
        show_hist = st.checkbox("과거 국면 표시 (검증용)", key="c_hist")

    df, source, err = load_crypto(target_symbol, selected_tf)
    if df.empty:
        st.error(f"데이터 로드 실패 → {err}")
    else:
        htf_df = load_htf_safe(lambda tf: load_crypto(target_symbol, tf), UPPER_TF.get(selected_tf))
        render_dashboard_safe(df, c1, c2, show_hist, rate_text=KRW_RATE, rate_chart=1.0,
                              captions=(f"데이터 소스: {source} · 환율: {KRW_SRC}",), tf_label=selected_tf, htf=htf_df,
                              macro={"fng": fng["value"] if fng else None}, qty_dec=CRYPTO_QTY_DEC,
                              premium=upbit_premium(target_symbol) if selected_tf == "1일" else None, long_mode=True)

# ==========================================
# 2. 📈 국내 주식 (KOSPI) 마켓
# ==========================================
elif market_choice == "📈 국내 주식 (KOSPI)":
    st.title("📈 KOSPI 주도주 현물 매수 봇")

    kospi_sectors = tab_groups("🇰🇷 국내 (KOSPI)")      # 스크리너와 같은 목록 (SCREEN_UNIVERSE)

    vols = get_sector_volumes(kospi_sectors)
    if sum(vols.values()) > 0:
        try:
            fig_pie = go.Figure(data=[go.Pie(labels=list(vols.keys()), values=list(vols.values()), hole=.4)])
            fig_pie.update_layout(title="국내 시장 섹터별 자금 유입(거래대금) 현황", height=300, margin=dict(t=30, b=0, l=0, r=0))
            st.plotly_chart(fig_pie, width="stretch")
        except Exception:
            st.caption("섹터 차트를 그리지 못해 생략합니다.")
    else:
        st.caption("섹터별 거래대금 조회에 실패해 이 차트는 생략합니다.")

    sel_sec = st.selectbox("📂 시장 주도 섹터", list(kospi_sectors.keys()))
    sel_tick = st.radio(f"🏆 {sel_sec} 거래대금 상위 Top 5", kospi_sectors[sel_sec], horizontal=True).split(" ")[0]

    c1, c2 = st.columns([1.3, 2])
    with c2:
        selected_tf = st.radio("프레임 선택", TF_OPTIONS, index=3, horizontal=True, key="s_tf")
        show_hist = st.checkbox("과거 국면 표시 (검증용)", key="s_hist")

    df, err = load_yf_cached(sel_tick, selected_tf)
    if df.empty:
        st.error(f"데이터 로드 실패 → {err}")
    else:
        htf_df = load_htf_safe(lambda tf: load_yf_cached(sel_tick, tf), UPPER_TF.get(selected_tf))
        render_dashboard_safe(df, c1, c2, show_hist, rate_text=1.0, rate_chart=1.0, tf_label=selected_tf, htf=htf_df)

# ==========================================
# 3. 🦅 미국 주식 (US Stock)
# ==========================================
elif market_choice == "🦅 미국 주식 (US Stock)":
    st.title("🦅 미국 주식 주도주 퀀트 봇")

    us_sectors = tab_groups("🇺🇸 미국", etf=False)      # 스크리너와 같은 목록 (ETF 그룹은 ETF 탭에)

    vols = get_sector_volumes(us_sectors)
    if sum(vols.values()) > 0:
        try:
            fig_pie = go.Figure(data=[go.Pie(labels=list(vols.keys()), values=list(vols.values()), hole=.4)])
            fig_pie.update_layout(title="미국 시장 섹터별 자금 유입(거래대금) 현황", height=300, margin=dict(t=30, b=0, l=0, r=0))
            st.plotly_chart(fig_pie, width="stretch")
        except Exception:
            st.caption("섹터 차트를 그리지 못해 생략합니다.")
    else:
        st.caption("섹터별 거래대금 조회에 실패해 이 차트는 생략합니다.")

    sel_sec = st.selectbox("📂 미국 주도 섹터", list(us_sectors.keys()))
    sel_tick = st.radio(f"🏆 {sel_sec} 거래대금 상위 Top 5", us_sectors[sel_sec], horizontal=True).split(" ")[0]

    c1, c2 = st.columns([1.3, 2])
    with c2:
        selected_tf = st.radio("프레임 선택", TF_OPTIONS, index=3, horizontal=True, key="us_tf")
        show_hist = st.checkbox("과거 국면 표시 (검증용)", key="us_hist")

    df, err = load_yf_cached(sel_tick, selected_tf)
    if df.empty:
        st.error(f"데이터 로드 실패 → {err}")
    else:
        htf_df = load_htf_safe(lambda tf: load_yf_cached(sel_tick, tf), UPPER_TF.get(selected_tf))
        render_dashboard_safe(df, c1, c2, show_hist, rate_text=KRW_RATE, rate_chart=KRW_RATE,
                              captions=(f"환율 적용: 1 USD = {KRW_RATE:,.0f} KRW ({KRW_SRC})",),
                              tf_label=selected_tf, htf=htf_df, qty_dec=US_QTY_DEC)

# ==========================================
# 4. 📊 미국 ETF (US ETF)
# ==========================================
elif market_choice == "📊 미국 ETF (US ETF)":
    st.title("📊 미국 ETF 퀀트 봇")

    etf_groups = tab_groups("🇺🇸 미국", etf=True)       # 스크리너와 같은 목록 (SCREEN_UNIVERSE의 ETF 그룹)

    sel_grp = st.selectbox("📂 ETF 카테고리", list(etf_groups.keys()))
    sel_tick = st.radio(f"🏆 {sel_grp}", etf_groups[sel_grp], horizontal=True).split(" ")[0]
    if "채권" in sel_grp and sel_tick in ("TLT", "IEF"):
        st.caption("ℹ️ 채권 ETF는 금리와 반대로 움직입니다. 수급 기반 국면 판별의 해석이 주식과 다를 수 있으니 참고용으로만 보세요.")
    if "레버리지" in sel_grp:
        st.warning("⚠️ 레버리지 ETF는 하루 변동의 3배로 움직이고, 횡보장에서는 가치가 깎입니다(변동성 손실). 단기·소액으로만 다루세요.")

    c1, c2 = st.columns([1.3, 2])
    with c2:
        selected_tf = st.radio("프레임 선택", TF_OPTIONS, index=3, horizontal=True, key="etf_tf")
        show_hist = st.checkbox("과거 국면 표시 (검증용)", key="etf_hist")

    df, err = load_yf_cached(sel_tick, selected_tf)
    if df.empty:
        st.error(f"데이터 로드 실패 → {err}")
    else:
        htf_df = load_htf_safe(lambda tf: load_yf_cached(sel_tick, tf), UPPER_TF.get(selected_tf))
        render_dashboard_safe(df, c1, c2, show_hist, rate_text=KRW_RATE, rate_chart=KRW_RATE,
                              captions=(f"환율 적용: 1 USD = {KRW_RATE:,.0f} KRW ({KRW_SRC})",),
                              tf_label=selected_tf, htf=htf_df, qty_dec=US_QTY_DEC)

# ==========================================
# 5. 🥇 금·은 원자재 (Commodities)
# ==========================================
elif market_choice == "🥇 금·은 원자재 (Commodities)":
    st.title("🥇 금/은 안전자산 현물 봇")

    com_tick = st.radio("원자재 선택", ["GC=F (금 Gold)", "SI=F (은 Silver)"], horizontal=True).split(" ")[0]

    c1, c2 = st.columns([1.3, 2])
    with c2:
        selected_tf = st.radio("프레임 선택", TF_OPTIONS, index=3, horizontal=True, key="com_tf")
        show_hist = st.checkbox("과거 국면 표시 (검증용)", key="com_hist")

    df, err = load_yf_cached(com_tick, selected_tf)
    if df.empty:
        st.error(f"데이터 로드 실패 → {err}")
    else:
        htf_df = load_htf_safe(lambda tf: load_yf_cached(com_tick, tf), UPPER_TF.get(selected_tf))
        render_dashboard_safe(df, c1, c2, show_hist, rate_text=KRW_RATE, rate_chart=KRW_RATE, is_commodity=True,
                              captions=(f"환율 적용: 1 USD = {KRW_RATE:,.0f} KRW ({KRW_SRC}) · 온스당",),
                              tf_label=selected_tf, htf=htf_df)

# ==========================================
# 6. 🔎 스크리너 (한눈에 보기 → 후보 선별 → 심층 분석)
# ==========================================
elif market_choice == "🔎 스크리너 (Screener)":
    st.title("🔎 섹터·종목 스크리너")
    st.caption("① 섹터 흐름을 한눈에 보고 → ② 매집이 유효한 후보만 추린 뒤 → ③ 기존 대시보드로 심층 분석합니다. "
               "판정은 다른 탭과 같은 엔진(일봉)을 쓰며, 상위 프레임(MTF) 필터는 ③ 심층 분석에서 확인합니다.")

    mk_all = list(SCREEN_UNIVERSE.keys())
    n_all = {m: sum(len(v) for v in SCREEN_UNIVERSE[m].values()) for m in mk_all}
    s1, s2 = st.columns([3, 1])
    with s1:
        markets = st.multiselect("스캔할 시장", mk_all, default=mk_all, format_func=lambda m: f"{m} ({n_all[m]}종목)")
    with s2:
        st.write("")
        run = st.button("🔄 스캔 시작", type="primary")
    st.caption("결과는 1시간 동안 저장됩니다. 처음 스캔은 1~2분 걸릴 수 있습니다.")

    if "scan_rows" not in st.session_state:
        st.session_state["scan_rows"], st.session_state["scan_markets"] = None, []
    if run and markets:
        rows = []
        prog = st.progress(0.0, text="스캔 준비 중...")
        for i, m in enumerate(markets):
            prog.progress(i / len(markets), text=f"{m} 스캔 중... ({i + 1}/{len(markets)})")
            try:
                rows += scan_market(m)
            except Exception as e:
                st.warning(f"{m} 스캔 실패 ({type(e).__name__}) — 나머지 시장은 계속 진행합니다.")
        prog.progress(1.0, text="스캔 완료")
        st.session_state["scan_rows"], st.session_state["scan_markets"] = rows, markets
        st.session_state["scan_time"] = pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M")

    rows = st.session_state.get("scan_rows")
    if not rows:
        st.info("시장을 고르고 **🔄 스캔 시작**을 누르세요.")
    else:
        res = pd.DataFrame(rows)
        for col in ("결론", "국면", "국면점수", "매집점수", "진바닥3박자", "현재가", "돌파완료", "트리거까지", "다음차수", "손절선까지",
                    "여력위험비", "레인지위치", "수요구간", "보조필터", "이백일선", "수익률1M", "수익률3M", "거래대금", "비고",
                    "주봉", "거래대금증가", "CMF", "오십일선거리", "이백일선위", "등급", "빠진조건", "_chk", "_kind", "_tranche", "기준봉"):
            if col not in res.columns:
                res[col] = None
        res["_ord"] = res["단계"].map({s: i for i, s in enumerate(STAGE_ORDER)}).fillna(99)
        bad = res[res["단계"] == "⚠️ 데이터 오류"]
        st.caption(f"스캔 시각 {st.session_state.get('scan_time', '')} · {len(res)}종목 중 {len(res) - len(bad)}종목 판정"
                   + (f" · 데이터 오류 {len(bad)}종목 (맨 아래 전체 결과에서 확인)" if len(bad) else ""))

        ov = sector_overview(res)
        res = finalize_grades(res, ov)
        basis = res["기준봉"].dropna().astype(str)
        if len(basis):
            st.caption("🕒 판정 기준 봉: " + " / ".join(f"{k} ({v}종목)" for k, v in basis.value_counts().head(3).items()))

        # ---------- 🅰️ 등급 보드 ----------
        st.subheader("🅰️ 등급 보드" + ("" if GRADE_ACTIVE else " (참고용 · 검증 중)"))
        if not GRADE_ACTIVE:
            st.caption("과거 검증(43종목·2019~2026)에서 A급이 B급보다 성과가 나빠, 등급은 지금 매수 판단에 쓰지 않고 참고로만 보여줍니다. "
                       "실제 매수 여부는 '결론' 열(🟢 매수 실행)을 따르세요.")
        live_buy = res[res["_kind"].astype(str).eq("BUY") | res["결론"].astype(str).str.contains("A급|B급|C급")]
        a_b = live_buy[live_buy["등급"] == "A"]
        b_b = live_buy[live_buy["등급"] == "B"]
        c_b = live_buy[live_buy["등급"] == "C"]
        if a_b.empty:
            st.info("오늘 A급 신호는 없습니다. (A급 = 위치·순서·주봉 통과 + 나머지 조건 실패 1개 이하)"
                    + ("" if GRADE_ACTIVE else " 매수 신호 자체는 아래 ② 후보의 🟢 결론을 보세요."))
        else:
            for _, r_ in a_b.sort_values("_ord").iterrows():
                with st.container(border=True):
                    st.markdown(f"**{r_['종목']} ({r_['티커']})** · {r_['섹터']} · {r_['결론']} · {r_['단계']}")
                    rp = r_.get("레인지위치")
                    rp_txt = "-" if rp is None or not np.isfinite(rp) else f"{rp * 100:.0f}%"
                    st.caption(f"현재가 {_fmt_px(r_['현재가'], r_['시장'])} · 손절선까지 {_fmt_pct(r_['손절선까지'])} · "
                               f"레인지 {rp_txt} · 주봉 {r_.get('주봉', '-')} · 다음 단계: ③ 심층 분석에서 사다리 확인")
        if not b_b.empty:
            with st.expander(f"🟡 B급 조건부 ({len(b_b)}개) — 빠진 조건이 채워지면 A급", expanded=a_b.empty):
                st.dataframe(pd.DataFrame({"종목": b_b["종목"] + " (" + b_b["티커"] + ")", "섹터": b_b["섹터"], "결론": b_b["결론"],
                                           "단계": b_b["단계"], "빠진 조건": b_b["빠진조건"],
                                           "손절선까지": b_b["손절선까지"].map(_fmt_pct)}), hide_index=True, width="stretch")
        if not c_b.empty:
            st.caption(f"⏳ C급 {len(c_b)}개: " + ", ".join(f"{a} ({m})" for a, m in zip(c_b["종목"], c_b["빠진조건"])) + " — 구조만 맞고 조건이 부족해 매수하지 않습니다.")

        # ---------- ① 섹터 자금 흐름 순위 ----------
        st.subheader("① 섹터 자금 흐름 순위 (돈이 몰리는 순)")
        if ov.empty:
            st.warning("판정된 종목이 없습니다.")
        else:
            show = pd.DataFrame({
                "순위": ov["흐름순위"].map(lambda v: "-" if v is None or not np.isfinite(v) else f"{int(v)}위"),
                "시장": ov["시장"], "섹터": ov["섹터"],
                "흐름 점수": ov["흐름점수"].map(lambda v: "-" if v is None or not np.isfinite(v) else f"{v:.0f}"),
                "흐름 국면": ov["흐름국면"].fillna("-"),
                "3개월(중앙값)": ov["평균 3개월"].map(_fmt_pct), "1개월(중앙값)": ov["평균 1개월"].map(_fmt_pct),
                "거래대금 증가": ov["거래대금 증가"].map(_fmt_pct),
                "CMF": ov["CMF"].map(lambda v: "-" if v is None or not np.isfinite(v) else f"{v:+.2f}"),
                "200일선 위": ov["200일선 위"], "주봉 상승·전환": ov["주봉 상승"],
                "후보": ov["후보"], "🟢매수": ov["🟢매수"], "종목수": ov["종목수"],
            })
            st.dataframe(show, hide_index=True, width="stretch")
            st.caption(f"흐름 점수 = 스캔한 섹터들 사이의 백분위 순위 가중 평균 (3개월 수익률 {FLOW_W['r3']:.0%} · 1개월 {FLOW_W['r1']:.0%} · "
                       f"거래대금 증가 {FLOW_W['surge']:.0%} · CMF {FLOW_W['cmf']:.0%}, 섹터 안 종목들의 중앙값). "
                       "🔥 주도 = 길게도 짧게도 강함 · 🌗 쉬어가는 중 = 강했는데 최근 눌림(가장 노릴 만한 곳) · 🌱 새로 유입 · 🧊 소외. "
                       "지수·레버리지·배당채권 ETF 묶음은 업종이 아니라 순위에서 뺐습니다. 국가·자산까지 포함한 큰 흐름은 🌊 자금 흐름 탭에서 보세요.")

        # ---------- ② 매집 유효 후보 ----------
        st.subheader("② 무릎 매수 후보")
        f1, f3 = st.columns([3, 1])
        with f1:
            stages = st.multiselect("단계 (위에서부터 우선순위)", CANDIDATE_STAGES, default=CANDIDATE_STAGES)
        with f3:
            st.write("")
            no_lev = st.checkbox("레버리지 ETF 제외", value=True)
        cand = res[res["단계"].isin(stages or [])].copy()
        if no_lev:
            cand = cand[cand["섹터"] != "ETF·레버리지(3x)"]
        # 정렬: 단계 우선순위 → 섹터 자금 흐름 순위 → 여력/위험(손익비) → 거래대금 (검증되지 않은 매집 점수는 정렬에 쓰지 않음)
        fmap = {} if ov.empty else {(m_, s_): (r_, q_, sc_) for m_, s_, r_, q_, sc_ in
                                    zip(ov["시장"], ov["섹터"], ov["흐름순위"], ov["흐름국면"], ov["흐름점수"])}
        n_sec = max([v[0] for v in fmap.values() if v[0] is not None and np.isfinite(v[0])] or [1])
        cand["_frank"] = [fmap.get((m_, s_), (None,))[0] for m_, s_ in zip(cand["시장"], cand["섹터"])]
        cand["_frank"] = pd.to_numeric(cand["_frank"], errors="coerce").fillna(n_sec + 1)
        cand["_rr"] = pd.to_numeric(cand["여력위험비"], errors="coerce").fillna(-1)
        cand["_tv"] = pd.to_numeric(cand["거래대금"], errors="coerce").fillna(0)
        cand["_g"] = cand["등급"].map({"A": 0, "B": 1, "C": 3}).fillna(2)
        cand = cand.sort_values(["_g", "_ord", "_frank", "_rr", "_tv"], ascending=[True, True, True, False, False])
        flow_txt = lambda m_, s_: (lambda v: "-" if v is None or v[0] is None or not np.isfinite(v[0]) else f"{int(v[0])}위 {v[1]}")(fmap.get((m_, s_)))

        # ---------- 🏆 추천: 돈이 몰리는 섹터 안의 무릎 매수 후보 ----------
        top_cut = max(3, int(np.ceil(n_sec / 3)))
        rec = cand[(cand["_frank"] <= top_cut) & (~cand["결론"].astype(str).str.startswith("🔴"))].head(10)
        st.markdown(f"##### 🏆 추천 (자금 흐름 상위 {top_cut}개 섹터 안의 후보)")
        if rec.empty:
            st.info("흐름 상위 섹터에서 아직 무릎 매수 후보가 없습니다. 강한 섹터는 대개 이미 올라 있어서, 눌림이 올 때까지 기다리는 게 맞습니다.")
        else:
            for i, (_, r_) in enumerate(rec.iterrows(), 1):
                st.markdown(f"{i}. **{r_['종목']}** ({r_['티커']}) · {r_['섹터']} — 섹터 흐름 {flow_txt(r_['시장'], r_['섹터'])} · "
                            f"{r_['결론']} · {r_['단계']} · 주봉 {r_.get('주봉', '-')} · 레인지 위치 "
                            + ("-" if r_.get("레인지위치") is None or not np.isfinite(r_["레인지위치"]) else f"{r_['레인지위치'] * 100:.0f}%")
                            + f" · 손절선까지 {_fmt_pct(r_['손절선까지'])}")
            st.caption("순서: 단계(🟢 매수 실행 > 대기 > 관찰) → 섹터 흐름 순위 → 여력/위험. 같은 섹터에서 여러 개면 하나만 고르세요. "
                       "주봉이 🔴 하락이면 사다리 리스크가 절반으로 줄어듭니다.")
        if cand.empty:
            st.info("조건에 맞는 후보가 없습니다. 하락장에서는 후보가 없는 게 정상일 수 있습니다 — 억지로 찾지 마세요.")
        else:
            view = pd.DataFrame({
                "등급": cand["등급"].fillna("-"), "결론": cand["결론"], "단계": cand["단계"],
                "종목": cand["종목"] + " (" + cand["티커"] + ")", "섹터": cand["섹터"], "빠진 조건": cand["빠진조건"].fillna(""),
                "섹터 흐름": [flow_txt(m_, s_) for m_, s_ in zip(cand["시장"], cand["섹터"])],
                "주봉 추세": cand["주봉"].fillna("-"),
                "3박자(이탈·수축·HL)": cand["진바닥3박자"],
                "현재가": [_fmt_px(v, m) for v, m in zip(cand["현재가"], cand["시장"])],
                "다음 매수가까지": [("지금" if "(지금)" in str(nx) else "돌파 완료" if (b_ and (v is None or not np.isfinite(v))) else _fmt_pct(v))
                                for b_, v, nx in zip(cand["돌파완료"].fillna(False), cand["트리거까지"], cand["다음차수"])],
                "다음 차수": cand["다음차수"].fillna("-"),
                "손절선까지": cand["손절선까지"].map(_fmt_pct),
                "레인지 위치": cand["레인지위치"].map(lambda v: "-" if v is None or not np.isfinite(v) else f"{v * 100:.0f}%"),
                "주봉 수요 구간": cand["수요구간"].fillna("-"),
                "보조필터": cand["보조필터"].fillna(0).astype(int).astype(str) + "/7",
                "여력/위험(배)": cand["여력위험비"].map(lambda v: "-" if v is None or not np.isfinite(v) else f"{v:.1f}"),
                "200일선": cand["이백일선"], "거래대금(20일)": [_fmt_tv(v, m) for v, m in zip(cand["거래대금"], cand["시장"])],
                "1개월": cand["수익률1M"].map(_fmt_pct),
                "국면(참고)": cand["국면"] + " " + cand["국면점수"].fillna(0).astype(int).astype(str) + "점",
            })
            st.dataframe(view, hide_index=True, width="stretch")
            st.caption("· 손절선 = ICT 사다리의 공통 무효화 손절 (스윕 저점 − 0.25 ATR, 스윕 전이면 예상값) · "
                       "여력/위험 = (목표 − 현재가) ÷ (현재가 − 손절선), 목표는 돌파 전이면 돌파선(IBSL), 돌파 후면 다음 매물대 저항 · "
                       "레인지 위치 = 최근 52주 범위에서의 위치 (50% 아래가 할인 구간) · 주봉 추세 🔴 하락(주봉 CHoCH 전)이면 리스크 절반")
            st.caption("· 🟢는 ICT 사다리 단계(1차 스윕 후 회복 · 2차 CHoCH · 3차 리테스트)에서만 나옵니다. 상세 판단은 ③ 심층 분석(1일)에서 확인하세요.")
            dup = cand["섹터"].value_counts()
            dup = dup[dup >= 2]
            if len(dup):
                st.warning("같은 섹터에서 후보가 여러 개입니다: " + ", ".join(f"{s} {n}개" for s, n in dup.items())
                           + " — 같은 섹터는 사실상 같은 베팅이니 하나만 고르세요.")
            if (cand["섹터"] == "ETF·레버리지(3x)").any():
                st.warning("레버리지 ETF는 횡보장에서 가치가 깎이고(변동성 손실) 하루 만에 크게 움직입니다. 단기·소액으로만 다루세요.")

        with st.expander("📋 전체 결과 (모든 단계) · CSV 저장", expanded=False):
            allv = res.sort_values(["_ord", "시장", "섹터"]).drop(columns=[c for c in ("_ord", "_rr", "_tv", "_frank", "_g", "_chk", "_kind", "_tranche") if c in res.columns])
            st.dataframe(allv, hide_index=True, width="stretch")
            st.download_button("⬇️ 오늘 스캔 결과 CSV로 저장 (나중에 성과 확인용)",
                               allv.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"screen_{st.session_state.get('scan_time', '')[:10]}.csv", mime="text/csv")

        # ---------- ③ 심층 분석 ----------
        st.subheader("③ 심층 분석")
        pool = cand if not cand.empty else res[res["단계"] != "⚠️ 데이터 오류"]
        if st.checkbox("후보가 아닌 종목도 목록에 보기", value=cand.empty):
            pool = res[res["단계"] != "⚠️ 데이터 오류"].sort_values(["_ord", "시장", "섹터"])
        if pool.empty:
            st.info("분석할 종목이 없습니다.")
        else:
            opts = list(zip(pool["시장"], pool["티커"], pool["종목"], pool["단계"], pool["결론"].fillna("-")))
            pick = st.selectbox("종목 선택", opts, format_func=lambda o: f"{o[4]} · {o[3]} · {o[2]} ({o[1]}) · {o[0]}")
            p_mk, p_tk, p_nm = pick[0], pick[1], pick[2]
            st.markdown(f"#### {p_nm} ({p_tk})")
            c1, c2 = st.columns([1.3, 2])
            with c2:
                selected_tf = st.radio("프레임 선택", TF_OPTIONS, index=3, horizontal=True, key="scr_tf")
                show_hist = st.checkbox("과거 국면 표시 (검증용)", key="scr_hist")
            if p_mk == "🪙 코인":
                df, source, err = load_crypto(p_tk, selected_tf)
                if df.empty:
                    st.error(f"데이터 로드 실패 → {err}")
                else:
                    htf_df = load_htf_safe(lambda tf: load_crypto(p_tk, tf), UPPER_TF.get(selected_tf))
                    render_dashboard_safe(df, c1, c2, show_hist, rate_text=KRW_RATE, rate_chart=1.0,
                                          captions=(f"데이터 소스: {source} · 환율: {KRW_SRC}",), tf_label=selected_tf,
                                          htf=htf_df, qty_dec=CRYPTO_QTY_DEC,
                                          premium=upbit_premium(p_tk) if selected_tf == "1일" else None, long_mode=True)
            else:
                df, err = load_yf_cached(p_tk, selected_tf)
                if df.empty:
                    st.error(f"데이터 로드 실패 → {err}")
                else:
                    rate = 1.0 if p_mk == "🇰🇷 국내 (KOSPI)" else KRW_RATE
                    caps = () if rate == 1.0 else (f"환율 적용: 1 USD = {KRW_RATE:,.0f} KRW ({KRW_SRC})",)
                    htf_df = load_htf_safe(lambda tf: load_yf_cached(p_tk, tf), UPPER_TF.get(selected_tf))
                    render_dashboard_safe(df, c1, c2, show_hist, rate_text=rate, rate_chart=rate, captions=caps,
                                          tf_label=selected_tf, htf=htf_df, qty_dec=(0 if rate == 1.0 else US_QTY_DEC))

# ==========================================
# 6-0. 🌊 글로벌 자금 흐름 (어디로 돈이 몰리는가 → 그 안에서 싸게 살 곳)
# ==========================================
elif market_choice == "🌊 자금 흐름 (Money Flow)":
    st.title("🌊 글로벌 자금 흐름")
    st.caption("업종 · 테마 · 국가 · 자산 ETF를 같은 달러 기준으로 비교해 큰돈이 어디로 움직이는지 봅니다. "
               "흐름이 강한 곳을 고른 뒤, 그 안에서 **눌림(할인 구간 · 수요 구간)** 에 온 종목을 스크리너와 종목 화면에서 찾습니다.")
    try:
        with st.spinner("ETF 데이터를 불러오는 중... (1시간 저장)"):
            frows = scan_flow()
        ft = flow_table(frows)
    except Exception as e:
        frows, ft = [], pd.DataFrame()
        st.error(f"자금 흐름 데이터를 불러오지 못했습니다 ({type(e).__name__}). 잠시 뒤 다시 시도하세요.")
    if ft is None or ft.empty:
        st.info("표시할 데이터가 없습니다.")
    else:
        miss = [r["티커"] for r in frows if r.get("오류")]
        bench = ft[ft["티커"] == FLOW_BENCH]
        lead = ft[ft["흐름국면"] == FLOW_QUAD["주도"]]
        k1, k2, k3, k4 = st.columns(4)
        top = ft.iloc[0]
        k1.metric("가장 강한 흐름", f"{top['이름']} ({top['티커']})", f"점수 {top['흐름점수']:.0f}", delta_color="off")
        k2.metric("🔥 주도 그룹 수", f"{len(lead)}개 / {len(ft)}개")
        if not bench.empty:
            k3.metric("기준: S&P500 3개월", _fmt_pct(bench.iloc[0]["r3"]), f"1개월 {_fmt_pct(bench.iloc[0]['r1'])}", delta_color="off")
        risk_on = ft[ft["티커"].isin(["SMH", "XLK", "ARKK", "BTC-USD"])]["흐름점수"].mean()
        risk_off = ft[ft["티커"].isin(["TLT", "GLD", "XLU", "XLP", "UUP"])]["흐름점수"].mean()
        if np.isfinite(risk_on) and np.isfinite(risk_off):
            k4.metric("위험 선호 (성장 − 방어)", f"{risk_on - risk_off:+.0f}점",
                      "성장·위험자산 쪽으로 쏠림" if risk_on > risk_off + 10 else "방어자산 쪽으로 쏠림" if risk_off > risk_on + 10 else "뚜렷한 쏠림 없음",
                      delta_color="off")

        # ---------- ① 흐름 순위 ----------
        st.subheader("① 자금 흐름 순위")
        grp_pick = st.radio("보기", ["전체"] + list(FLOW_UNIVERSE.keys()), horizontal=True, key="flow_grp")
        fv = ft if grp_pick == "전체" else ft[ft["그룹"] == grp_pick]
        try:
            topn = fv.head(15).iloc[::-1]
            fig = go.Figure(data=[go.Bar(x=topn["흐름점수"], y=topn["이름"] + " (" + topn["티커"] + ")", orientation="h",
                                         marker_color=["#e53935" if q == FLOW_QUAD["주도"] else "#fb8c00" if q == FLOW_QUAD["약화"]
                                                       else "#43a047" if q == FLOW_QUAD["개선"] else "#90a4ae" for q in topn["흐름국면"]],
                                         text=topn["흐름국면"], textposition="inside")])
            fig.update_layout(height=max(320, 26 * len(topn)), margin=dict(l=10, r=10, t=10, b=10), xaxis_title="흐름 점수 (0~100)",
                              template="plotly_dark", xaxis=dict(range=[0, 100]))
            st.plotly_chart(fig, width="stretch")
        except Exception:
            pass
        tbl = pd.DataFrame({
            "순위": fv["흐름순위"], "그룹": fv["그룹"], "이름": fv["이름"] + " (" + fv["티커"] + ")",
            "흐름 점수": fv["흐름점수"].astype(int), "국면": fv["흐름국면"],
            "1개월": fv["r1"].map(_fmt_pct), "3개월": fv["r3"].map(_fmt_pct), "6개월": fv["r6"].map(_fmt_pct),
            "S&P 대비 3개월": fv["rs3"].map(_fmt_pct), "S&P 대비 1개월": fv["rs1"].map(_fmt_pct),
            "거래대금 증가": fv["surge"].map(_fmt_pct), "CMF(매집 압력)": fv["cmf"].map(lambda v: "-" if v is None or not np.isfinite(v) else f"{v:+.2f}"),
            "50일선 대비": fv["d50"].map(_fmt_pct), "200일선": fv["above200"].map(lambda b: "위" if b is True else "아래" if b is False else "-"),
            "강한 흐름 속 눌림": fv["눌림"].map(lambda b: "🎯" if b else ""),
            "추세(6개월)": fv["spark"],
        })
        try:
            st.dataframe(tbl, hide_index=True, width="stretch",
                         column_config={"추세(6개월)": st.column_config.LineChartColumn("추세(6개월)"),
                                        "흐름 점수": st.column_config.ProgressColumn("흐름 점수", min_value=0, max_value=100, format="%d")})
        except Exception:
            st.dataframe(tbl.drop(columns=["추세(6개월)"]), hide_index=True, width="stretch")
        st.caption(f"흐름 점수 = 이 목록 안에서의 백분위 순위를 가중 평균 (3개월 수익률 {FLOW_W['r3']:.0%} · 1개월 {FLOW_W['r1']:.0%} · "
                   f"거래대금 증가 {FLOW_W['surge']:.0%} · CMF {FLOW_W['cmf']:.0%}). 국면: 🔥 주도 = 3개월·1개월 모두 상위 절반 · "
                   "🌗 쉬어가는 중 = 3개월은 강한데 최근 1개월이 약함(눌림 후보) · 🌱 새로 유입 = 최근 1개월만 강함 · 🧊 소외 = 둘 다 약함. "
                   "거래대금 증가 = 최근 20일 평균 ÷ 그 전 120일 평균 − 1 · CMF = 종가가 하루 범위 위쪽에서 끝난 날의 거래량 비중(+면 매수 우위).")
        if miss:
            st.caption("조회 실패로 빠진 티커: " + ", ".join(miss))

        # ---------- ② 순환 지도 ----------
        st.subheader("② 순환 지도 (3개월 강도 × 최근 1개월 강도)")
        try:
            sc = ft.dropna(subset=["rs3", "rs1"])
            fig2 = go.Figure()
            for q, col in ((FLOW_QUAD["주도"], "#e53935"), (FLOW_QUAD["약화"], "#fb8c00"), (FLOW_QUAD["개선"], "#43a047"), (FLOW_QUAD["침체"], "#90a4ae")):
                g = sc[sc["흐름국면"] == q]
                if not g.empty:
                    fig2.add_trace(go.Scatter(x=g["rs3"] * 100, y=g["rs1"] * 100, mode="markers+text", name=q, text=g["이름"],
                                              textposition="top center", marker=dict(size=11, color=col)))
            fig2.add_hline(y=0, line_dash="dot", line_color="#888")
            fig2.add_vline(x=0, line_dash="dot", line_color="#888")
            fig2.update_layout(height=520, template="plotly_dark", margin=dict(l=10, r=10, t=10, b=10),
                               xaxis_title="S&P500 대비 3개월 (%)", yaxis_title="S&P500 대비 1개월 (%)")
            st.plotly_chart(fig2, width="stretch")
            st.caption("오른쪽 위 = 길게도 짧게도 강한 곳(주도) · 오른쪽 아래 = 강했지만 최근 쉬는 곳(눌림 매수를 노릴 곳) · "
                       "왼쪽 위 = 새로 돈이 들어오기 시작한 곳 · 왼쪽 아래 = 소외. 보통 시계 반대 방향(개선 → 주도 → 약화 → 침체)으로 돕니다.")
        except Exception:
            st.caption("순환 지도를 그리지 못했습니다.")

        # ---------- ③ 흐름 → 싸게 살 곳 ----------
        st.subheader("③ 흐름이 강한 곳에서 싸게 살 곳 찾기")
        pb = ft[ft["눌림"]]
        if pb.empty:
            st.info(f"지금은 '흐름 점수 {FLOW_PULLBACK_MIN}점 이상 · 3개월 강도 상위 · 200일선 위 · 최근 1개월 하락 또는 50일선 ±3% 이내'인 눌림 그룹이 없습니다. "
                    "강한 곳은 이미 많이 올랐거나, 쉬는 곳은 추세가 꺾인 상태입니다 — 추격하지 말고 눌림을 기다리세요.")
        else:
            st.success("🎯 강한 흐름 속 눌림: " + " · ".join(f"**{r['이름']}**({r['티커']}, {r['흐름점수']:.0f}점, 1개월 {_fmt_pct(r['r1'])})"
                                                      for _, r in pb.iterrows()))
        hot = ft[(ft["흐름점수"] >= FLOW_PULLBACK_MIN) & ft["티커"].isin(FLOW_TO_SECTORS.keys())]
        secs = []
        for _, r in hot.iterrows():
            for sname in FLOW_TO_SECTORS[r["티커"]]:
                if sname not in secs:
                    secs.append(sname)
        if secs:
            st.markdown("**흐름이 강한 테마와 연결된 스크리너 섹터**: " + " · ".join(secs))
        rows_sc = st.session_state.get("scan_rows")
        if rows_sc and secs:
            rs_df = pd.DataFrame(rows_sc)
            if "섹터" in rs_df.columns and "단계" in rs_df.columns:
                cc = rs_df[rs_df["섹터"].isin(secs) & rs_df["단계"].isin(CANDIDATE_STAGES)].copy()
                if cc.empty:
                    st.info("마지막 스크리너 결과에서 이 섹터들의 매수 후보는 없습니다. (강한 섹터가 아직 눌리지 않았다는 뜻일 수 있습니다)")
                else:
                    cc["_ord"] = cc["단계"].map({x: i for i, x in enumerate(STAGE_ORDER)}).fillna(99)
                    cc = cc.sort_values("_ord")
                    st.dataframe(pd.DataFrame({"결론": cc["결론"], "단계": cc["단계"], "종목": cc["종목"] + " (" + cc["티커"] + ")",
                                               "섹터": cc["섹터"], "레인지 위치": cc.get("레인지위치", pd.Series(dtype=float)).map(
                                                   lambda v: "-" if v is None or not np.isfinite(v) else f"{v * 100:.0f}%"),
                                               "주봉": cc.get("주봉", "-")}), hide_index=True, width="stretch")
                    st.caption(f"마지막 스크리너 스캔({st.session_state.get('scan_time', '')}) 기준. 상세 판단은 🔎 스크리너 ③ 심층 분석에서 확인하세요.")
        elif secs:
            st.info("🔎 스크리너 탭에서 스캔을 한 번 돌리면, 여기서 이 섹터들의 매수 후보를 바로 보여줍니다.")
        st.caption("※ '저평가'는 재무 지표가 아니라 기술적 위치(눌림 · 할인 구간 · 수요 구간)를 뜻합니다. "
                   "흐름 점수는 과거 수익률 기반이라 방향이 바뀌는 시점은 늦게 잡힙니다 — 순위는 '어디를 볼지'를 정하는 용도로 쓰세요.")

# ==========================================
# 7. 💼 포지션 관리 (손절 · 추가 매수 · 익절 · 매도를 한 화면에서)
# ==========================================
elif market_choice == "💼 포지션 관리 (Position)":
    st.title("💼 포지션 관리")
    st.caption("이미 산 종목 하나를 넣으면 지금 할 일(손절 · 추가 매수 · 익절 · 보유)과 손절가, 손절 시 손실 금액을 계산합니다. "
               "새로 살 계획은 각 종목 화면의 🪜 사다리에서, 산 뒤의 관리는 이 탭에서 합니다. (일봉 기준, 장 마감 후 확인 권장)")

    s1, s2 = st.columns([1, 2])
    with s1:
        mk = st.selectbox("시장", list(SCREEN_UNIVERSE.keys()), key="pos_mk")
    with s2:
        opts = [f"{t} ({n})" for items in SCREEN_UNIVERSE[mk].values() for t, n in items] + ["직접 입력"]
        pick = st.selectbox("보유 종목", opts, key="pos_pick")
    tk = (st.text_input("티커 직접 입력 (예: 005930.KS, AAPL, BTC/USDT)", value="", key="pos_tk") or "").strip().upper() \
        if pick == "직접 입력" else pick.split(" ")[0]
    is_kr, is_coin = mk == "🇰🇷 국내 (KOSPI)", mk == "🪙 코인"
    unit = "원" if is_kr else ("USDT" if is_coin else "달러")
    qdec = CRYPTO_QTY_DEC if is_coin else (US_QTY_DEC if mk == "🇺🇸 미국" else 0)
    fx = 1.0 if is_kr else KRW_RATE

    a1, a2 = st.columns(2)
    with a1:
        avg_px = st.number_input(f"평단가 ({unit})", min_value=0.0, value=0.0, step=1.0 if is_kr else 0.01,
                                 format="%.0f" if is_kr else "%.4f", key="sell_avg")
    with a2:
        qty = st.number_input("보유 수량", min_value=0.0, value=0.0, step=1.0 if qdec == 0 else 0.0001,
                              format="%.0f" if qdec == 0 else "%.4f", key="sell_qty")
    b1, b2 = st.columns([2, 1])
    with b1:
        mode = st.radio("어떻게 샀나요?", POS_MODES, index=0, horizontal=True, key="pos_mode")
        exit_sel = st.radio("매도 방식", POS_EXIT_MODES, index=0, horizontal=True, key="pos_exit",
                            help="추세 꺾임: 목표가 없이 상승 추세를 따라가다 꺾임 경고 2개면 절반, 구조가 무너지면 나머지 매도 · "
                                 "목표가: 평단 + 손절폭의 2배 이상 첫 저항에서 절반 익절")
    exit_mode = "trend" if exit_sel == POS_EXIT_MODES[0] else "target"
    with b2:
        st.write("")
        tp1_done = st.checkbox("절반 매도 이미 함" if exit_mode == "trend" else "1차 익절(50%) 이미 함", value=False, key="pos_tp1")
    bought = 0 if mode == POS_MODES[3] else POS_MODES.index(mode) + 1
    avg_px = avg_px if isinstance(avg_px, (int, float)) else 0.0
    qty = qty if isinstance(qty, (int, float)) else 0.0
    tp1_done = bool(tp1_done)

    if not tk:
        st.info("티커를 입력하세요.")
    else:
        if is_coin:
            df, _src, err = load_crypto(tk, "1일")
        else:
            df, err = load_yf_cached(tk, "1일")
        if df is None or df.empty:
            st.error(f"⛔ 데이터 로드 실패 → {err}")
        elif avg_px <= 0 or qty <= 0:
            st.info(f"현재가 {_fmt_price(float(df['close'].iloc[-1]))} {unit} · 평단가와 보유 수량을 입력하면 관리 계획을 계산합니다.")
        else:
            try:
                sug, sug_src = position_suggest_stop(df, avg_px, qty, bought, qdec, EQUITY, fx)
                my_stop = st.number_input(f"🛑 내 손절가 ({unit}) — 매수할 때 정한 값이 있으면 그 값을 넣으세요", min_value=0.0,
                                          value=float(round(sug, 4 if not is_kr else 0)), format="%.0f" if is_kr else "%.4f",
                                          key=f"pos_stop_{tk}_{bought}", help=f"기본값: {sug_src}. 손절가는 올리기만 하고 내리지 않습니다.")
                my_stop = my_stop if isinstance(my_stop, (int, float)) and my_stop > 0 else sug
                pp = position_plan(df, avg_px, qty, bought, tp1_done, my_stop, qdec, EQUITY, fx, exit_mode=exit_mode)
            except Exception as e:
                pp = None
                st.error(f"분석 실패 ({type(e).__name__}: {e}) — 판단이 안 될 때는 매수할 때 정한 손절가를 지키세요.")
            if pp:
                def money(v):
                    if v is None:
                        return "-"
                    return f"₩{v:,.0f}" if is_kr else f"${v:,.2f} (≈₩{v * fx:,.0f})"
                def pxs(v):
                    return "-" if v is None else (f"₩{v:,.0f}" if is_kr else f"${v:,.4g}" if v < 1 else f"${v:,.2f}")
                def qtxt(q):
                    return _qty_txt(q, qdec)
                sty = POS_STYLE[pp['action']]
                title = sty['title'] + (f" — {pp['add_row']['n']}차" if pp['action'] == "ADD" and pp['add_row'] else "")
                st.markdown(
                    f"""<div style="background:{sty['bg']};color:#fff;padding:16px 20px;border-radius:12px;margin:6px 0 10px 0">
<div style="font-size:28px;font-weight:800;line-height:1.25">{title}</div>
<div style="font-size:15px;opacity:.95;margin-top:6px">{'<br>'.join(pp['reasons'])}</div></div>""", unsafe_allow_html=True)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("현재가", pxs(pp['px']), f"평단 대비 {pp['pnl_pct']:+.1f}%")
                m2.metric("🛑 손절가", pxs(pp['stop']), f"현재가 대비 {pp['dist'] * 100:+.1f}%", delta_color="off")
                m3.metric("손절 시 손익", f"₩{pp['loss_krw']:,.0f}", f"총자산의 {pp['loss_pct']:+.2f}%", delta_color="off")
                m4.metric("평가손익", money(pp['pnl']))
                st.caption(f"손절가 기준: {pp['stop_kind']} · 총자산은 사이드바 값(₩{EQUITY:,.0f}) 기준")
                if pp['loss_pct'] < -RISK_PER_TRADE * 1.5:
                    st.warning(f"손절 시 손실이 총자산의 {abs(pp['loss_pct']):.1f}%로, 1회 허용 손실({RISK_PER_TRADE:g}%)보다 큽니다. "
                               "다음부터는 수량을 줄이거나, 지금 일부를 줄여 손실 한도를 맞추는 것을 검토하세요.")

                st.markdown("##### 📋 지금 걸어 둘 주문")
                rows = [{"주문": "🛑 손절 (스탑 감시 주문)", "가격": pxs(pp['stop']), "수량": qtxt(qty), "체결 시 손익": money(pp['loss'])}]
                if pp['action'] in ("STOP", "EXIT"):
                    rows = [{"주문": "🔴 다음 봉 시가에 전량 매도", "가격": pxs(pp['px']), "수량": qtxt(qty), "체결 시 손익": money(pp['pnl'])}]
                elif pp['action'] == "HALF":
                    q1 = _qty_floor(qty * 0.5, qdec)
                    rows.insert(0, {"주문": "🟠 지금 절반 매도", "가격": pxs(pp['px']), "수량": qtxt(q1), "체결 시 손익": money((pp['px'] - avg_px) * q1)})
                elif pp['action'] == "TP1":
                    q1 = _qty_floor(qty * 0.5, qdec)
                    rows.insert(0, {"주문": f"🟠 지금 절반 매도 (1차 익절 · {pp['tp_label']} 도달)", "가격": pxs(pp['px']), "수량": qtxt(q1),
                                    "체결 시 손익": money((pp['px'] - avg_px) * q1)})
                    rows.append({"주문": "⬆️ 매도 후 남은 물량 손절가", "가격": pxs(max(avg_px, pp['stop'])), "수량": qtxt(qty - q1), "체결 시 손익": "본절 이상"})
                elif not tp1_done and pp['tp']:
                    q1 = _qty_floor(qty * 0.5, qdec)
                    rows.append({"주문": f"🎯 1차 익절 지정가 ({pp['tp_label']})", "가격": pxs(pp['tp']), "수량": qtxt(q1),
                                 "체결 시 손익": money((pp['tp'] - avg_px) * q1)})
                if pp.get('be_line') and not pp.get('be_hit') and pp['action'] not in ("STOP", "EXIT"):
                    rows.append({"주문": "⬆️ 본절 이동 기준 (종가 돌파 시 손절가 → 평단가)", "가격": pxs(pp['be_line']), "수량": "-", "체결 시 손익": "-"})
                if pp['action'] == "ADD" and pp['add_row']:
                    ar = pp['add_row']
                    rows.append({"주문": f"🔵 {ar['n']}차 추가 매수", "가격": pxs(ar['price'] / fx),
                                 "수량": qtxt(ar['qty']), "체결 시 손익": "-"})
                if (tp1_done or bought == 0 or pp.get('be_hit')) and pp['trail']:
                    rows.append({"주문": "📈 트레일링 (매일 올리기만)", "가격": pxs(pp['trail']), "수량": "남은 물량 전부", "체결 시 손익": money((pp['trail'] - avg_px) * qty)})
                st.dataframe(pd.DataFrame(rows), hide_index=True)

                st.markdown("##### 🧭 손절가가 바뀌는 순서")
                if bought == 0:
                    st.markdown(f"- 사다리와 무관하게 산 종목은 **내 손절가와 트레일링({pxs(pp['trail'])}) 중 높은 값**을 손절가로 씁니다. "
                                "트레일링은 매일 올리기만 하고, 구조 붕괴 신호가 나오면 전량, 고점 징후가 2개 이상이면 절반을 정리합니다.")
                if bought and pp.get('trend'):
                    st.markdown(f"1. **추세 확인 전**: 매수 때 정한 손절가({pxs(my_stop)})를 그대로 둡니다. 내리지 않고, 그 아래에서 추가 매수하지 않습니다.")
                    st.markdown(f"2. **종가가 {pxs(pp['be_line']) if pp.get('be_line') else '돌파선'} 위로 올라서면**: 손절가를 **평단가({pxs(avg_px)}) 이상**으로 올립니다." + (" ✅ 완료" if pp.get('be_hit') else ""))
                    st.markdown(f"3. **상승 추세 동안**: 트레일링({pxs(pp['trail'])} · 최근 {SELL_TRAIL_N}봉 고점 − {TP3_TRAIL_ATR:g} ATR)을 따라 손절가를 올리며 보유합니다.")
                    st.markdown("4. **꺾임 경고 2개 이상**(20일선 이탈·약세 다이버전스·고점 윗꼬리·거래량 동반 음봉 등) → **절반 매도** · "
                                "**구조 붕괴**(직전 저점 종가 이탈) 또는 **트레일링 이탈** → **나머지 매도**")
                elif bought:
                    st.markdown(f"1. **1차 익절 전**: 매수 때 정한 손절가({pxs(my_stop)})를 그대로 둡니다. 가격이 빠져도 **내리지 않고, 그 아래에서 추가 매수하지 않습니다.**")
                    st.markdown(f"2. **1차 익절({pxs(pp['tp']) if pp['tp'] else '-'} · 평단 + 손절폭의 {POS_TP1_R:g}배 이상 첫 저항) 후**: 손절가를 **평단가({pxs(avg_px)}) 이상**으로 올려 남은 물량을 무위험으로 만듭니다.")
                    st.markdown(f"3. **그 뒤**: 트레일링({pxs(pp['trail'])} · 최근 {SELL_TRAIL_N}봉 고점 − {TP3_TRAIL_ATR:g} ATR)이 더 높아지면 그 값으로 계속 올립니다.")

                ict = pp['ict']
                wi = (ict or {}).get('wict') or {}
                wm = W_STATE_META.get(wi.get('state', "W_NA"))
                with st.expander("🔍 판단 근거 (주봉 추세 · ICT 단계 · 매도 신호)", expanded=pp['action'] not in ("HOLD",)):
                    st.markdown(f"- 주봉 추세: {wm['icon']} {wm['label']}" + (f" · 전환 기준선 {pxs(wi['level'])} ({wi['dist'] * 100:+.1f}%)" if wi.get('level') else ""))
                    st.markdown(f"- 일봉 ICT 단계: {ICT_STAGE_LABEL.get(ict.get('stage'), ict.get('stage'))}")
                    st.markdown(f"- 구조 붕괴 신호: " + (" · ".join(pp['r']['hard']) if pp['r']['hard'] else "없음"))
                    st.markdown(f"- 꺾임 경고·고점 징후 ({len(pp['r']['soft'])}개, 2개 이상이면 절반 매도): " + (" · ".join(pp['r']['soft']) if pp['r']['soft'] else "없음"))

                fibs = calculate_fibonacci_levels(df)
                lines = [(avg_px, "평단가", "#7e57c2"), (pp['stop'], "손절가", "#ef5350")]
                if pp['tp'] and not tp1_done:
                    lines.append((pp['tp'], "1차 익절(50%)", "#26a69a"))
                if pp['trail'] and (tp1_done or bought == 0 or pp.get('be_hit')):
                    lines.append((pp['trail'], "트레일링", "#ffa726"))
                if pp.get('be_line') and not pp.get('be_hit'):
                    lines.append((pp['be_line'], "본절 이동 기준", "#26a69a"))
                st.plotly_chart(build_chart(df, fibs, 1.0, None, pp['r']['info'], None, extra_lines=lines,
                                            zones=(ict or {}).get('zones')), width="stretch")
                st.caption("※ 규칙 기반 참고 도구이며 수익을 보장하지 않습니다. 투자 판단과 책임은 본인에게 있습니다.")
