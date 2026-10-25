import streamlit as st
import ccxt
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta  # noqa: F401  (df.ta 접근자 등록용)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests

st.set_page_config(page_title="PRO 퀀트 트레이딩 대시보드", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# 공통 유틸리티
# ==========================================
@st.cache_data(ttl=3600)
def get_krw_exchange_rate():
    """USD/KRW 환율 (yfinance). 실패 시 1350원 고정값 사용."""
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
    """EMA, 변동성(ATR%), 누적 거래량(OBV) 계산"""
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    pc = df['close'].shift(1)
    tr = pd.concat([df['high'] - df['low'], (df['high'] - pc).abs(), (df['low'] - pc).abs()], axis=1).max(axis=1)
    df['ATR_PCT'] = tr.rolling(14).mean() / df['close']
    df['OBV'] = (np.sign(df['close'].diff()).fillna(0) * df['volume']).cumsum()
    return df

# ==========================================
# 시장 국면 엔진: 매집 → 상승 → 분배 → 하락
# ==========================================
# 세력의 의도를 직접 알 수는 없으므로, 가격·거래량에 남는 흔적으로 국면을 추정합니다.
# 각 국면을 4개 영역(추세/위치/파동/수급)에서 25점씩, 총 100점 만점으로 채점해 가장 높은 국면을 선택합니다.
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

def analyze_market_cycle(df):
    d = df.tail(200)
    n = len(d)
    price = float(d['close'].iloc[-1])
    hi, lo = float(d['high'].max()), float(d['low'].min())
    rng = max(hi - lo, 1e-12)
    pos = (price - lo) / rng  # 0 = 구간 바닥, 1 = 구간 천장

    # ---------- 1. 추세 (EMA) ----------
    ema20 = _val(d, 'EMA_20', default=price)
    ema50 = _val(d, 'EMA_50', default=price)
    ema50_prev = _val(d, 'EMA_50', -11, default=ema50)
    atr_s = d['ATR_PCT'].dropna() if 'ATR_PCT' in d.columns else pd.Series(dtype=float)
    atr_med = max(float(atr_s.median()) if len(atr_s) else 0.02, 1e-4)
    atr_now = float(atr_s.iloc[-1]) if len(atr_s) else atr_med
    slope = (ema50 / ema50_prev - 1) / atr_med if ema50_prev else 0.0  # 변동성 대비 기울기
    comp = atr_now / atr_med < 0.85  # 변동성 수축

    up = price > ema20 > ema50
    down = price < ema20 < ema50
    flat = abs(slope) < 1.0
    crack = price < ema20 and ema20 > ema50
    rebound = ema20 < price < ema50

    t = {
        "하락": 25 if (down and slope < -1.0) else 15 if down else 8 if (price < ema50 and slope < 0) else 0,
        "매집": 25 if (flat and not up) else 15 if rebound else 0,
        "상승": 25 if (up and slope >= 1.0) else 15 if up else 8 if price > ema20 else 0,
        "분배": 25 if crack else 15 if (flat and price >= ema50) else 0,
    }

    # ---------- 2. 위치 (최근 범위 내 현재가 위치) ----------
    p = {
        "하락": 25 if pos <= 0.5 else 12 if pos <= 0.7 else 0,
        "매집": 25 if pos <= 0.3 else 12 if pos <= 0.45 else 0,
        "상승": 25 if pos >= 0.45 else 10 if pos >= 0.3 else 0,
        "분배": 25 if pos >= 0.7 else 12 if pos >= 0.55 else 0,
    }

    # ---------- 3. 파동 (고점/저점 구조) ----------
    piv_l = d['low'].where(d['low'] == d['low'].rolling(11, center=True).min()).dropna().values
    piv_h = d['high'].where(d['high'] == d['high'].rolling(11, center=True).max()).dropna().values
    enough = len(piv_h) >= 2 and len(piv_l) >= 2
    HH = enough and piv_h[-1] > piv_h[-2]
    LH = enough and piv_h[-1] < piv_h[-2]
    HL = enough and piv_l[-1] > piv_l[-2]
    LL = enough and piv_l[-1] < piv_l[-2]
    if enough:
        w = {
            "하락": 25 if (LH and LL) else 12 if LL else 0,
            "매집": 25 if (HL and not HH) else 10 if (not HL and not LL) else 0,
            "상승": 25 if (HH and HL) else 12 if HL else 0,
            "분배": 25 if (LH and not LL) else 10 if (HH and not HL) else 0,
        }
    else:
        w = {"하락": 5, "매집": 10, "상승": 5, "분배": 10}

    # ---------- 4. 수급 (거래량 흔적) ----------
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

    bull_div = bear_div = False
    if n > 25 and 'OBV' in d.columns:
        pr = price / float(d['close'].iloc[-21]) - 1
        obv_chg = float(d['OBV'].iloc[-1] - d['OBV'].iloc[-21])
        big = abs(obv_chg) > vol_base * 2
        bull_div = big and pos <= 0.45 and pr <= 0.01 and obv_chg > 0
        bear_div = big and pos >= 0.6 and pr >= -0.01 and obv_chg < 0

    v = {
        "하락": 25 if (ud < 0.8 and not (sell_climax or bull_div)) else 10 if ud < 1.0 else 0,
        "매집": min(25, (15 if sell_climax else 0) + (15 if bull_div else 0) + (10 if (dry or comp) else 0) + (5 if ud >= 1.0 else 0)),
        "상승": 25 if (ud > 1.2 and vr >= 0.9) else 12 if ud > 1.0 else 0,
        "분배": min(25, (15 if dist_climax else 0) + (15 if bear_div else 0) + (10 if (ud < 1.0 and pos >= 0.6) else 0)),
    }

    # ---------- 종합 ----------
    totals = {k: t[k] + p[k] + w[k] + v[k] for k in PHASE_ORDER}
    phase = max(PHASE_ORDER, key=lambda k: totals[k])
    top = totals[phase]
    late = phase == "하락" and (sell_climax or bull_div or totals["매집"] >= top - 10)

    # ---------- 팩트체크 문구 (관찰된 상태 설명) ----------
    if up:
        f_ema = "20·50 이평선 위 정배열이며 중기 이평선이 " + ("가파르게 " if slope >= 1.0 else "완만하게 ") + "우상향 중입니다."
    elif down:
        f_ema = "20·50 이평선 아래 역배열로 " + ("이평선이 가파르게 하락하며 " if slope < -1.0 else "") + "하방 압력이 이어지고 있습니다."
    elif crack:
        f_ema = "상승 추세 중 20 이평선이 깨지며 첫 균열이 나타났습니다."
    elif flat:
        f_ema = "이평선이 평평하게 수렴하며 방향성을 탐색하는 횡보 구간입니다."
    elif rebound:
        f_ema = "50 이평선 아래이지만 20 이평선을 회복하며 반등을 시도 중입니다."
    else:
        f_ema = "20 이평선 아래로 밀려 있으나 추세는 뚜렷하지 않습니다."

    pct = f"{pos * 100:.0f}%"
    if pos <= 0.2:
        f_pos = f"최근 {n}봉 범위의 하단 {pct} 지점으로 바닥권입니다."
    elif pos <= 0.45:
        f_pos = f"최근 {n}봉 범위의 하단 {pct} 지점으로 저가 구간입니다."
    elif pos < 0.7:
        f_pos = f"최근 {n}봉 범위의 {pct} 지점으로 중간대입니다."
    else:
        f_pos = f"최근 {n}봉 범위의 {pct} 지점으로 고점권입니다."

    if not enough:
        f_wave = "아직 뚜렷한 고점/저점 파동이 형성되지 않아 방향성을 탐색 중입니다."
    elif HH and HL:
        f_wave = "고점과 저점을 모두 높이는 상승 파동(HH/HL)이 살아 있습니다."
    elif LH and LL:
        f_wave = "고점과 저점이 모두 낮아지는 하락 파동(LH/LL)이 이어지고 있습니다."
    elif HL and not HH:
        f_wave = "고점은 못 넘었지만 저점을 높이고 있습니다(저점 상승). 매집 가능성을 시사합니다."
    elif LH and not LL:
        f_wave = "저점은 지켰지만 고점이 낮아지고 있습니다(고점 하락). 분배 가능성을 시사합니다."
    elif HH and not HL:
        f_wave = "고점은 높였지만 저점이 따라 오르지 못해 상승 힘이 약해지는 모습입니다."
    else:
        f_wave = "고점과 저점이 뚜렷한 방향 없이 혼조를 보이고 있습니다."

    if sell_climax:
        f_vol = "저점권에서 투매성 대량 거래량(셀링 클라이맥스)이 나왔습니다. 매도 물량이 소진되는 신호일 수 있습니다."
    elif bull_div:
        f_vol = "가격은 눌려 있는데 거래량 흐름(OBV)은 오르는 강세 다이버전스입니다. 조용한 매집 가능성이 있습니다."
    elif dist_climax:
        f_vol = "고점권에서 대량 거래량에도 윗꼬리가 길어지며 물량이 넘어가는 흔적(분배)이 보입니다."
    elif bear_div:
        f_vol = "가격은 버티는데 거래량 흐름(OBV)은 꺾이는 약세 다이버전스입니다."
    elif ud > 1.2:
        f_vol = "상승봉 거래량이 하락봉보다 우세해 매수 수급이 강합니다."
    elif ud < 0.8:
        f_vol = "하락봉 거래량이 우세해 매도 압력이 큽니다."
    else:
        f_vol = "상승·하락 거래량이 비슷하게 균형을 이루고 있습니다."
    if dry or comp:
        f_vol += " 거래량·변동성이 줄어드는 중입니다."

    return {
        "phase": phase, "score": top, "totals": totals, "late": late, "hl": bool(HL),
        "weak": top < 50, "bars": n,
        "facts": {"EMA": f_ema, "POS": f_pos, "WAVE": f_wave, "VOL": f_vol},
    }

@st.cache_data(ttl=300)
def cached_phase_history(df, start=60):
    """과거 각 시점에서 '그 시점까지의 데이터만' 보고 판별한 국면 구간 목록 (검증용)"""
    labels = []
    for i in range(start, len(df)):
        labels.append(analyze_market_cycle(df.iloc[:i + 1])['phase'])
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

def calculate_fibonacci_levels(df, is_commodity=False, window=120):
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

def render_fact_box(facts):
    with st.container(border=True):
        st.markdown("##### 📝 4대 팩트체크")
        st.markdown(f"- **[추세 EMA]**: {facts['EMA']}")
        st.markdown(f"- **[위치 POS]**: {facts['POS']}")
        st.markdown(f"- **[파동 WAVE]**: {facts['WAVE']}")
        st.markdown(f"- **[수급 VOL]**: {facts['VOL']}")

def render_trade_guide(info, fibs, curr, rate):
    def fmt(v):
        return f"₩{v * rate:,.0f}"

    buy_line = " | ".join(f"{lab}: {fmt(p)}" + (" (도달)" if curr <= p else "") for lab, p in fibs['buy'])
    sell_line = " | ".join(f"{lab}: {fmt(p)}" + (" (도달)" if curr >= p else "") for lab, p in fibs['sell'])
    ph = info['phase']

    if ph == "하락":
        if info['late']:
            st.warning("🟠 하락 말기 신호가 보입니다. 바닥 확정 전이므로 3차 타점 부근에서만 소액 탐색을 고려하세요.")
            lab, p = fibs['buy'][2]
            st.markdown("**[바닥 탐색 타점]**")
            st.markdown(f"- {lab}: {fmt(p)} | 무효화 기준(전저점): {fmt(fibs['low'])} 이탈 시 재평가")
        else:
            st.error("📉 하락 국면입니다. 매집 신호(투매 거래량 · 저점 상승 · 강세 다이버전스)가 확인될 때까지 현금 비중을 유지하고 매수를 보류하세요.")
    elif ph == "매집":
        size = "저점 상승 확인 · 정상 진입" if info['hl'] else "저점 상승 확인 전 · 비중 축소"
        st.markdown(f"**[3분할 매수 타점]** (매집 구간 · {size})")
        st.markdown(f"- {buy_line}")
        st.markdown(f"- 무효화 기준(전저점): {fmt(fibs['low'])} 이탈 시 매집 가설 폐기")
    elif ph == "상승":
        st.markdown("**[눌림목 매수 타점]** (상승 국면 · 추격 매수 주의)")
        st.markdown(f"- {buy_line}")
        st.markdown("**[3분할 매도 타점]** (고점권 접근 시 분할 매도 준비)")
        st.markdown(f"- {sell_line}")
    else:  # 분배
        st.error("🟣 분배 국면입니다. 신규 매수를 금지하고 보유분은 분할 매도를 검토하세요.")
        st.markdown("**[3분할 매도 타점]**")
        st.markdown(f"- {sell_line}")

    if info['weak']:
        st.caption("신호가 약해 국면 전환기일 수 있습니다. 단정적인 판단은 피하세요.")
    st.caption("※ 규칙 기반 참고 지표이며, 투자 판단과 책임은 본인에게 있습니다.")

def build_chart(df, fibs, rate=1.0, history=None):
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
    colors = ['#26a69a' if r['close'] >= r['open'] else '#ef5350' for _, r in df.iterrows()]
    fig.add_trace(go.Bar(x=df['time'], y=df['volume'], marker_color=colors, name='Volume'), row=2, col=1)
    fig.update_layout(height=600, margin=dict(l=0, r=0, t=0, b=0), xaxis_rangeslider_visible=False)
    return fig

def render_dashboard(df, c1, c2, show_hist, rate_text=1.0, rate_chart=1.0, is_commodity=False, captions=()):
    curr = float(df['close'].iloc[-1])
    fibs = calculate_fibonacci_levels(df, is_commodity=is_commodity)
    info = analyze_market_cycle(df)
    meta = PHASE_META[info['phase']]

    with c1:
        st.subheader(f"현재 국면: {meta['icon']} {info['phase']} ({info['score']}/100점 일치)")
        tot = info['totals']
        st.caption("국면 일치도 — " + " | ".join(f"{k} {tot[k]}" for k in ["매집", "상승", "분배", "하락"]))
        for cap in captions:
            st.caption(cap)
        if info['bars'] < 80:
            st.caption("⚠️ 데이터가 짧아 국면 판별 신뢰도가 낮습니다.")
        render_fact_box(info['facts'])
        render_phase_legend()
        st.divider()
        render_trade_guide(info, fibs, curr, rate_text)

    with c2:
        history = cached_phase_history(df) if show_hist else None
        st.plotly_chart(build_chart(df, fibs, rate_chart, history), width="stretch")

def load_yf_data(ticker, tf_label):
    """주식/원자재 공통 yfinance 로더. (df, 에러메시지) 반환"""
    tf_map = {"15분": "15m", "1시간": "60m", "4시간": "60m", "1일": "1d", "주봉": "1wk", "월봉": "1mo"}
    yf_tf = tf_map[tf_label]
    period = "60d" if yf_tf in ["15m", "60m"] else ("2y" if yf_tf == "1d" else "10y")
    try:
        df = yf.download(ticker, period=period, interval=yf_tf, progress=False)
        if df.empty:
            return pd.DataFrame(), "yfinance에서 빈 데이터가 반환되었습니다."
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df.columns = [str(c).lower() for c in df.columns]
        df = df.rename(columns={'date': 'time', 'datetime': 'time'})

        if tf_label == "4시간":
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time')
            df = df.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna().reset_index()

        df = add_indicators(df)
        return df, None
    except Exception as e:
        return pd.DataFrame(), f"{type(e).__name__}: {e}"


# ==========================================
# 사이드바 네비게이션
# ==========================================
st.sidebar.title("🧭 PRO 퀀트 네비게이션")
tabs = ["🪙 암호화폐 (Crypto)", "📈 국내 주식 (KOSPI)", "🦅 미국 주식 (US Stock)", "🥇 금·은 원자재 (Commodities)"]
market_choice = st.sidebar.radio("트레이딩 룸 선택", tabs)
TF_OPTIONS = ["15분", "1시간", "4시간", "1일", "주봉", "월봉"]


# ==========================================
# 1. 🪙 암호화폐 마켓
# ==========================================
if market_choice == "🪙 암호화폐 (Crypto)":
    st.title("🪙 암호화폐 현물 퀀트 봇")

    btc_d = get_btc_dominance()
    col_d1, col_d2 = st.columns([1, 3])
    with col_d1: st.metric(label="BTC 도미넌스", value=f"{btc_d:.2f}%")
    with col_d2:
        if btc_d > 55.0: st.warning("⚠️ **[수급 경고]** 도미넌스가 매우 높습니다. 알트코인 수급 이탈(흡성대법)에 주의하세요.")
        elif btc_d < 45.0: st.success("🔥 **[알트 시즌]** 도미넌스가 낮아 알트코인으로 자금이 순환될 확률이 높습니다.")
        else: st.info("⚖️ 도미넌스 중립 구간입니다. 개별 차트에 집중하세요.")

    crypto_tabs = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "직접 검색"]
    selected_crypto = st.radio("관심 심볼", crypto_tabs, horizontal=True)
    target_symbol = st.text_input("심볼 커스텀 입력", value="DOGE/USDT").upper() if selected_crypto == "직접 검색" else selected_crypto.split(" ")[0]

    @st.cache_data(ttl=60)
    def load_crypto(symbol, tf_label):
        """여러 거래소를 순서대로 시도. (df, 사용한 거래소, 에러요약) 반환"""
        tf_map = {"15분": "15m", "1시간": "1h", "4시간": "4h", "1일": "1d", "주봉": "1w", "월봉": "1M"}
        tf = tf_map[tf_label]
        errors = []
        for name in ["binance", "okx", "bybit", "kraken", "coinbase"]:
            try:
                ex = getattr(ccxt, name)()
                bars = ex.fetch_ohlcv(symbol, timeframe=tf, limit=300)
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

    c1, c2 = st.columns([1.3, 2])
    with c2:
        selected_tf = st.radio("프레임 선택", TF_OPTIONS, index=3, horizontal=True, key="c_tf")
        show_hist = st.checkbox("과거 국면 표시 (검증용)", key="c_hist")

    df, source, err = load_crypto(target_symbol, selected_tf)
    if df.empty:
        st.error(f"데이터 로드 실패 → {err}")
    else:
        render_dashboard(df, c1, c2, show_hist, rate_text=KRW_RATE, rate_chart=1.0,
                         captions=(f"데이터 소스: {source}",))

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
        fig_pie = go.Figure(data=[go.Pie(labels=list(vols.keys()), values=list(vols.values()), hole=.4)])
        fig_pie.update_layout(title="국내 시장 섹터별 자금 유입(거래대금) 현황", height=300, margin=dict(t=30, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, width="stretch")

    sel_sec = st.selectbox("📂 시장 주도 섹터", list(kospi_sectors.keys()))
    sel_tick = st.radio(f"🏆 {sel_sec} 거래대금 상위 Top 5", kospi_sectors[sel_sec], horizontal=True).split(" ")[0]

    @st.cache_data(ttl=300)
    def load_stock(ticker, tf_label):
        return load_yf_data(ticker, tf_label)

    c1, c2 = st.columns([1.3, 2])
    with c2:
        selected_tf = st.radio("프레임 선택", TF_OPTIONS, index=3, horizontal=True, key="s_tf")
        show_hist = st.checkbox("과거 국면 표시 (검증용)", key="s_hist")

    df, err = load_stock(sel_tick, selected_tf)
    if df.empty:
        st.error(f"데이터 로드 실패 → {err}")
    else:
        render_dashboard(df, c1, c2, show_hist, rate_text=1.0, rate_chart=1.0)

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
        fig_pie = go.Figure(data=[go.Pie(labels=list(vols.keys()), values=list(vols.values()), hole=.4)])
        fig_pie.update_layout(title="미국 시장 섹터별 자금 유입(거래대금) 현황", height=300, margin=dict(t=30, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, width="stretch")

    sel_sec = st.selectbox("📂 미국 주도 섹터", list(us_sectors.keys()))
    sel_tick = st.radio(f"🏆 {sel_sec} 거래대금 상위 Top 5", us_sectors[sel_sec], horizontal=True).split(" ")[0]

    @st.cache_data(ttl=300)
    def load_us_stock(ticker, tf_label):
        return load_yf_data(ticker, tf_label)

    c1, c2 = st.columns([1.3, 2])
    with c2:
        selected_tf = st.radio("프레임 선택", TF_OPTIONS, index=3, horizontal=True, key="us_tf")
        show_hist = st.checkbox("과거 국면 표시 (검증용)", key="us_hist")

    df, err = load_us_stock(sel_tick, selected_tf)
    if df.empty:
        st.error(f"데이터 로드 실패 → {err}")
    else:
        render_dashboard(df, c1, c2, show_hist, rate_text=KRW_RATE, rate_chart=KRW_RATE,
                         captions=(f"환율 적용: 1 USD = {KRW_RATE:,.0f} KRW",))

# ==========================================
# 4. 🥇 금·은 원자재 (Commodities)
# ==========================================
elif market_choice == "🥇 금·은 원자재 (Commodities)":
    st.title("🥇 금/은 안전자산 현물 봇")

    com_tick = st.radio("원자재 선택", ["GC=F (금 Gold)", "SI=F (은 Silver)"], horizontal=True).split(" ")[0]

    @st.cache_data(ttl=300)
    def load_commodity(ticker, tf_label):
        return load_yf_data(ticker, tf_label)

    c1, c2 = st.columns([1.3, 2])
    with c2:
        selected_tf = st.radio("프레임 선택", TF_OPTIONS, index=3, horizontal=True, key="com_tf")
        show_hist = st.checkbox("과거 국면 표시 (검증용)", key="com_hist")

    df, err = load_commodity(com_tick, selected_tf)
    if df.empty:
        st.error(f"데이터 로드 실패 → {err}")
    else:
        render_dashboard(df, c1, c2, show_hist, rate_text=KRW_RATE, rate_chart=KRW_RATE, is_commodity=True,
                         captions=(f"환율 적용: 1 USD = {KRW_RATE:,.0f} KRW (온스당)",))
