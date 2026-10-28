import streamlit as st
import ccxt
import yfinance as yf
import pandas as pd
import numpy as np
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
TP_WEIGHTS = (0.4, 0.4, 0.2)   # 3분할 청산 비중 (1차 / 2차 / 3차)
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
    if veto:
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
        if mtf['state'] == "n/a":
            f_ema += f" 상위 프레임({lab}) 데이터가 부족해 거시 추세 필터는 적용되지 않았습니다."
        else:
            ap = f"({mtf['span']}EMA 근사)" if mtf['approx'] else f"({mtf['span']}EMA)"
            if mtf['state'] == "bear":
                f_ema += (f" 상위 프레임({lab}) 장기 추세{ap}가 하락 중이라 매집 신호를 "
                          + ("무효 처리합니다(데드캣 바운스 경계)." if MTF_VETO_RATIO >= 1 else "감점합니다."))
            elif mtf['state'] == "bull":
                f_ema += f" 상위 프레임({lab}) 장기 추세{ap}는 상승입니다."
            else:
                f_ema += f" 상위 프레임({lab}) 장기 추세{ap}는 뚜렷하지 않습니다."

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
            "risk_cap": risk_cap, "scale": scale, "conf": conf, "notes": notes,
            "position_pct": position_pct, "blocked": blocked}

@st.cache_data(ttl=300)
def cached_phase_history(df, htf_pack=None, bars=HIST_BARS, min_start=60):
    """최근 bars봉만, 각 시점에서 '그 시점까지의 데이터'로 다시 판별한 국면 구간 (검증용 · 연산량 제한)"""
    start = max(len(df) - bars, min_start)
    if start >= len(df):
        return []
    tn = _naive(df['time']).to_numpy(dtype='datetime64[ns]')
    labels = []
    for i in range(start, len(df)):
        labels.append(analyze_market_cycle(df.iloc[:i + 1], mtf_at(htf_pack, tn[i]), None, False)['phase'])
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
    with st.expander("📌 시장 국면 가이드 (매집 → 상승 → 분배 → 하락)", expanded=True):
        st.markdown("- 🔵 **매집**: `[바닥권 횡보 · 거래량 감소 · 저점 상승]` - 셋업 확인 후 CHoCH 트리거에서만 매수")
        st.markdown("- 🟢 **상승**: `[정배열 · 고점/저점 상승]` - 보유 유지, 눌림목만 매수 (추격 주의)")
        st.markdown("- 🟣 **분배**: `[고점권 거래량 폭증 · 윗꼬리 · 고점 하락]` - 신규 매수 금지, 분할 매도")
        st.markdown("- 🔴 **하락**: `[역배열 · 거래량 동반 하락]` - 매수 보류 (떨어지는 칼날)")
        st.markdown("- 🧲 **Phase 1 셋업**: `[장기선 이탈 + 변동성 수축 + 소진 신호]` → 관심 종목 편입 (매수 아님)")
        st.markdown("- 🎯 **Phase 2 트리거**: `[종가가 직전 반등 고점(IBSL)을 돌파·안착 = Bullish CHoCH]` → 실전 매수 타점")
        st.markdown("- 🅰️ **A급 트리거**: `[SSL 유동성 스윕 + 모멘텀 다이버전스 + CHoCH]` 3중 교차 검증 → 최대 비중")
        st.markdown("- 💰 **청산**: `1차 40%(2ATR, 도달 시 손절선을 본절로)` → `2차 40%(저항·VAH)` → `3차 20%(트레일링 스탑)`")

def render_fact_box(facts):
    with st.container(border=True):
        st.markdown("##### 📝 4대 팩트체크 + SMC")
        st.markdown(f"- **[추세 EMA]**: {facts['EMA']}")
        st.markdown(f"- **[위치 POS]**: {facts['POS']}")
        st.markdown(f"- **[파동 WAVE]**: {facts['WAVE']}")
        st.markdown(f"- **[수급 VOL]**: {facts['VOL']}")
        st.markdown(f"- **[구조 SMC]**: {facts['SMC']}")

def _render_plan(plan, rate, title):
    def fmt(v): return f"₩{v * rate:,.0f}"
    st.markdown(f"**{title}**")
    st.markdown(f"- 진입 기준가: {fmt(plan['entry'])}")
    st.markdown(f"- 🛑 권장 손절선(무효화 라인): {fmt(plan['sl'])} ({plan['sl_pct']:+.1f}%) · 기준: {plan['sl_kind']}")
    st.markdown("- 🎯 **3분할 청산**")
    st.markdown(f"    - 1차 (물량 {plan['tp1_w'] * 100:.0f}%, 생존형 · {plan['tp1_label']}): "
                f"{fmt(plan['tp1'])} ({plan['tp1_pct']:+.1f}%) · 손익비 1:{plan['rr_tp1']:.1f}")
    st.markdown(f"    - ⚠️ 1차 도달 시 **잔여 물량의 손절선을 진입가({fmt(plan['entry'])})로 상향 조정**해 "
                "무위험(Risk-Free) 트레이딩으로 전환하세요.")
    st.markdown(f"    - 2차 (물량 {plan['tp2_w'] * 100:.0f}%, 구조적 · {plan['tp2_label']}): "
                f"{fmt(plan['tp2'])} ({plan['tp2_pct']:+.1f}%) · 손익비 1:{plan['rr_tp2']:.1f}")
    st.markdown(f"    - 3차 (물량 {plan['tp3_w'] * 100:.0f}%, 추세 추종): {plan['tp3_rule']}")
    st.markdown(f"- ⚖️ 분할 청산 기대 손익비: 1 : {plan['rr_blend']:.1f} "
                f"(3차는 2차 수준 청산으로 보수적 가정) · 손절폭 {plan['risk_pct']:.1f}%")
    if plan['blocked']:
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
            _render_plan(plan, rate, "[실전 매매 계획]")
        else:
            st.markdown("- ATR을 계산할 수 없어 손절·익절 라인을 산정하지 못했습니다.")
        st.markdown(f"- 리테스트 참고선: IBSL {fmt(smc['choch']['level'])} (종가 이탈 시 신호 실패)")
        if info['phase'] in ("상승",):
            st.markdown("**[3분할 매도 타점]**")
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
                _render_plan(preview, rate, "[트리거 발생 시 예상 계획 (미리보기)]")
        st.markdown(f"- 무효화: 종가가 직전 저점(ISSL {fmt(smc['issl'])}) 아래로 내려가면 셋업 폐기")
    elif entry == "WHIPSAW":
        st.error("🪤 가짜 반등(휩쏘): 직전 저점을 종가 기준으로 이탈했습니다. 매수 점수를 0점으로 삭감했고 매수를 보류하세요.")
    elif entry == "BLOCKED":
        st.error("🚫 Bullish CHoCH가 나왔지만 상위 프레임이 하락 추세입니다. 데드캣 바운스로 보고 매수를 보류하세요.")
    elif ph == "분배":
        st.error("🟣 분배 국면입니다. 신규 매수를 금지하고 보유분은 분할 매도를 검토하세요.")
        st.markdown("**[3분할 매도 타점]**")
        st.markdown(f"- {sell_line}")
    elif ph == "상승":
        st.markdown("**[눌림목 참고 계획]** (CHoCH 트리거 없는 기존 상승 추세 · 비중 보수적)")
        plan = build_trade_plan(info)
        if plan:
            _render_plan(plan, rate, "눌림목 진입 시 예상 계획")
        st.markdown("**[3분할 매도 타점]** (고점권 접근 시 분할 매도 준비)")
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

def build_chart(df, fibs, rate=1.0, history=None, info=None, plan=None):
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
        fig.add_hline(y=plan['tp1'] * rate, line_dash="dot", line_color="#26a69a", line_width=2,
                      annotation_text="TP1 익절", annotation_position="right", row=1, col=1)

    colors = ['#26a69a' if r['close'] >= r['open'] else '#ef5350' for _, r in df.iterrows()]
    fig.add_trace(go.Bar(x=df['time'], y=df['volume'], marker_color=colors, name='Volume'), row=2, col=1)
    fig.update_layout(height=600, margin=dict(l=0, r=0, t=0, b=0), xaxis_rangeslider_visible=False)
    return fig

def render_dashboard(df, c1, c2, show_hist, rate_text=1.0, rate_chart=1.0, is_commodity=False,
                     captions=(), tf_label=None, htf=None, macro=None):
    curr = float(df['close'].iloc[-1])
    fibs = calculate_fibonacci_levels(df, is_commodity=is_commodity)
    pack = prepare_htf(htf, tf_label)
    t_last = _naive(df['time']).iloc[-1].to_datetime64().astype('datetime64[ns]')
    info = analyze_market_cycle(df, mtf_at(pack, t_last), macro)
    meta = PHASE_META[info['phase']]

    with c1:
        st.subheader(f"현재 국면: {meta['icon']} {info['phase']} ({info['score']}/100점 일치)")
        tot = info['totals']
        st.caption("국면 일치도 — " + " | ".join(f"{k} {tot[k]}" for k in ["매집", "상승", "분배", "하락"]))
        if info['adjust']:
            st.caption("보정: " + " · ".join(info['adjust']))
        for cap in captions:
            st.caption(cap)
        if info['bars'] < 80:
            st.caption("⚠️ 데이터가 짧아 국면 판별 신뢰도가 낮습니다.")
        render_fact_box(info['facts'])
        render_phase_legend()
        st.divider()
        render_trade_guide(info, fibs, curr, rate_text)

    with c2:
        history = None
        if show_hist:
            try:
                with st.spinner(f"과거 국면 연산 중... (최근 {HIST_BARS}봉)"):
                    history = cached_phase_history(df, pack)
            except Exception as e:
                st.error(f"과거 국면 계산 실패 ({type(e).__name__}): {e}")
        plan = None
        if info['entry'] in ("TRIGGER", "BREAKOUT") or info['phase'] == "상승":
            try:
                plan = build_trade_plan(info)
            except Exception:
                plan = None
        st.plotly_chart(build_chart(df, fibs, rate_chart, history, info, plan), width="stretch")

def render_dashboard_safe(df, c1, c2, show_hist, rate_chart=1.0, is_commodity=False, **kw):
    """고급 분석에서 어떤 예외가 나도 대시보드가 죽지 않도록, 기본 기술적 지표만으로 안전하게 폴백"""
    try:
        render_dashboard(df, c1, c2, show_hist, rate_chart=rate_chart, is_commodity=is_commodity, **kw)
    except Exception as e:
        with c1:
            st.warning(f"⚠️ 고급 분석 엔진 오류({type(e).__name__})로 기본 기술적 지표만 표시합니다.")
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
# 데이터 로더 (모든 외부 호출 예외 처리 · 타임아웃 · 다중 소스 폴백)
# ==========================================
CRYPTO_EXCHANGES = ["binance", "okx", "bybit", "upbit", "kraken", "coinbase"]
EXCHANGE_LIMIT = {"upbit": 200}  # 거래소별 1회 조회 상한

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
            bars = ex.fetch_ohlcv(symbol, timeframe=tf, limit=min(300, EXCHANGE_LIMIT.get(name, 300)))
            if not bars:
                errors.append(f"{name}: 빈 데이터")
                continue
            df = pd.DataFrame(bars, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df = add_indicators(df)
            return df, name, None
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}")
    return pd.DataFrame(), None, " / ".join(errors)

def load_yf_data(ticker, tf_label):
    """주식/원자재 공통 yfinance 로더. (df, 에러메시지) 반환"""
    tf_map = {"15분": "15m", "1시간": "60m", "4시간": "60m", "1일": "1d", "주봉": "1wk", "월봉": "1mo"}
    yf_tf = tf_map.get(tf_label)
    if yf_tf is None:
        return pd.DataFrame(), f"지원하지 않는 프레임: {tf_label}"
    period = "60d" if yf_tf in ["15m", "60m"] else ("2y" if yf_tf == "1d" else "10y")
    try:
        df = yf.download(ticker, period=period, interval=yf_tf, progress=False, timeout=10)
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
# 사이드바 네비게이션
# ==========================================
st.sidebar.title("🧭 PRO 퀀트 네비게이션")
tabs = ["🪙 암호화폐 (Crypto)", "📈 국내 주식 (KOSPI)", "🦅 미국 주식 (US Stock)",
        "📊 미국 ETF (US ETF)", "🥇 금·은 원자재 (Commodities)"]
market_choice = st.sidebar.radio("트레이딩 룸 선택", tabs)
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

    crypto_tabs = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "직접 검색"]
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
                              macro={"fng": fng["value"] if fng else None})

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
        "자동차": ["005380.KS (현대차)", "000270.KS (기아)", "012330.KS (현대모비스)", "204320.KS (HL만도)", "018880.KS (한온시스템)"]
    }

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

    us_sectors = {
        "테크/소프트웨어": ["AAPL (애플)", "MSFT (마이크로소프트)", "GOOGL (구글)", "META (메타)", "AMZN (아마존)"],
        "반도체/AI": ["NVDA (엔비디아)", "TSM (TSMC)", "AVGO (브로드컴)", "AMD (AMD)", "QCOM (퀄컴)"],
        "금융/핀테크": ["JPM (JP모건)", "V (비자)", "MA (마스터카드)", "BAC (뱅크오브아메리카)", "WFC (웰스파고)"],
        "헬스케어": ["LLY (일라이릴리)", "UNH (유나이티드헬스)", "JNJ (존슨앤존슨)", "ABBV (애브비)", "MRK (머크)"],
        "전기차/에너지": ["TSLA (테슬라)", "XOM (엑슨모빌)", "CVX (쉐브론)", "NEE (넥스테라)", "COP (코노코필립스)"]
    }

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
                              tf_label=selected_tf, htf=htf_df)

# ==========================================
# 4. 📊 미국 ETF (US ETF)
# ==========================================
elif market_choice == "📊 미국 ETF (US ETF)":
    st.title("📊 미국 ETF 퀀트 봇")

    etf_groups = {
        "시장 지수": ["SPY (S&P 500)", "QQQ (나스닥 100)", "DIA (다우존스)", "IWM (러셀 2000)"],
        "섹터/테마": ["SOXX (반도체)", "XLK (기술주)", "XLF (금융)", "XLV (헬스케어)"],
        "배당/가치": ["SCHD (배당성장)", "VYM (고배당)", "VTV (가치주)"],
        "채권/안전자산": ["TLT (미 국채 20년+)", "IEF (미 국채 7-10년)", "GLD (금)"],
    }

    sel_grp = st.selectbox("📂 ETF 카테고리", list(etf_groups.keys()))
    sel_tick = st.radio(f"🏆 {sel_grp}", etf_groups[sel_grp], horizontal=True).split(" ")[0]
    if sel_grp == "채권/안전자산":
        st.caption("ℹ️ 채권 ETF는 금리와 반대로 움직입니다. 수급 기반 국면 판별의 해석이 주식과 다를 수 있으니 참고용으로만 보세요.")

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
                              tf_label=selected_tf, htf=htf_df)

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
