import streamlit as st
import ccxt
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import requests

st.set_page_config(page_title="PRO 퀀트 트레이딩 대시보드", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# 공통 유틸리티 및 4대 핵심 지표 엔진 (Pro Mode)
# ==========================================
@st.cache_data(ttl=3600)
def get_krw_exchange_rate():
    try:
        exchange = ccxt.binance()
        return exchange.fetch_ticker('USDT/KRW')['last']
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
                except:
                    pass
            vols[sec] = sec_vol
        return vols
    except:
        return {s: 0 for s in sectors_dict.keys()}

def calculate_fibonacci_levels(df, is_commodity=False):
    recent_df = df.tail(60)
    max_h = recent_df['high'].max()
    min_l = recent_df['low'].min()
    diff = max_h - min_l
    
    if is_commodity:
        return {'high': max_h, 'low': min_l, 'fib_1': max_h - diff * 0.236, 'fib_2': max_h - diff * 0.382, 'fib_3': max_h - diff * 0.500}
    else:
        return {'high': max_h, 'low': min_l, 'fib_1': max_h - diff * 0.382, 'fib_2': max_h - diff * 0.618, 'fib_3': max_h - diff * 0.786}

def analyze_4_core_indicators(df, current_price, fib_levels):
    score_total = 0
    fact_checks = {}

    # 1. 추세 (EMA)
    ema20 = df['EMA_20'].iloc[-1] if 'EMA_20' in df.columns else current_price
    ema50 = df['EMA_50'].iloc[-1] if 'EMA_50' in df.columns else current_price
    if current_price > ema20 and ema20 > ema50:
        score_ema = 25
        fact_checks['EMA'] = "단기 20 이평선과 중기 50 이평선 위에서 완벽한 정배열 상승 추세를 유지 중입니다."
    elif current_price > ema20:
        score_ema = 10
        fact_checks['EMA'] = "중기 50 이평선 아래에 있으나, 단기 20 이평선을 돌파하며 추세 반전을 시도하고 있습니다."
    else:
        score_ema = 0
        fact_checks['EMA'] = "단기 20 이평선 아래로 캔들이 이탈하며 하방 압력이 가중되는 역배열 상태입니다."
    score_total += score_ema

    # 2. 지지/저항 (Fibonacci)
    if current_price >= fib_levels['fib_1']: 
        score_fib = 25
        fact_checks['FIB'] = "상단 피보나치 되돌림 구간 위에서 굳건하게 매수세가 방어하며 상승 여력을 키우고 있습니다."
    elif current_price >= fib_levels['fib_2']: 
        score_fib = 20
        fact_checks['FIB'] = "주요 피보나치 0.618 황금 분할선 구간에서 매물대를 소화하며 일시적 지지 테스트 중입니다."
    elif current_price >= fib_levels['fib_3']: 
        score_fib = 10
        fact_checks['FIB'] = "전저점(최후 방어선) 부근까지 밀렸으나, 추가 하락을 막기 위해 간신히 버티고 있습니다."
    else: 
        score_fib = 0
        fact_checks['FIB'] = "핵심 지지선이 전면 붕괴되며 투심이 악화되어 추가 하락 위험이 매우 높은 상태입니다."
    score_total += score_fib

    # 3. 파동 카운팅 (HH/HL)
    pivot_highs = df[df['is_pivot_high']]['high'].dropna().values
    pivot_lows = df[df['is_pivot_low']]['low'].dropna().values
    if len(pivot_highs) >= 2 and len(pivot_lows) >= 2:
        if pivot_highs[-1] > pivot_highs[-2] and pivot_lows[-1] > pivot_lows[-2]:
            score_wave = 25
            fact_checks['WAVE'] = "고점과 저점을 지속적으로 높이는 전형적인 상승 채널(HH/HL)을 형성하며 파동이 살아있습니다."
        elif pivot_lows[-1] > pivot_lows[-2]:
            score_wave = 15
            fact_checks['WAVE'] = "고점 돌파는 멈췄으나, 저점을 꾸준히 높이며 에너지를 응축하는 수렴(HL) 국면입니다."
        else:
            score_wave = 0
            fact_checks['WAVE'] = "고점과 저점이 모두 낮아지는 전형적인 하락 파동(LH/LL) 구조로 하락세가 지속 중입니다."
    else:
        score_wave = 10
        fact_checks['WAVE'] = "아직 명확한 고점/저점 파동이 형성되지 않아 방향성을 탐색하는 횡보 장세입니다."
    score_total += score_wave

    # 4. 수급 (Volume)
    recent = df.tail(5)
    up_v = recent[recent['close'] >= recent['open']]['volume'].mean()
    dn_v = recent[recent['close'] < recent['open']]['volume'].mean()
    if math.isnan(up_v): up_v = 0
    if math.isnan(dn_v): dn_v = 0
    
    if up_v > dn_v * 1.2: 
        score_vol = 25
        fact_checks['VOL'] = "상승 반전 캔들에서 대량의 거래량이 터지며 주도 매수 수급이 강하게 유입되었습니다."
    elif up_v > dn_v: 
        score_vol = 15
        fact_checks['VOL'] = "매수 거래량이 매도 거래량을 소폭 상회하며 긍정적인 수급 흐름을 쥐고 있습니다."
    else: 
        score_vol = 0
        fact_checks['VOL'] = "하락 캔들에서 대량의 매도 거래량이 쏟아지며 시장 수급이 급격히 이탈하고 있습니다."
    score_total += score_vol

    if score_total >= 70: verdict = "🟢 적극 매수 / 눌림목 구간"
    elif score_total >= 50: verdict = "🟡 관망 및 중립 (비중 축소)"
    else: verdict = "🔴 위험 경고 및 매수 금지"

    return score_total, verdict, fact_checks

def render_score_legend():
    with st.expander("📌 자산 시장 매수 추천 컷오프 (Pro 가이드)", expanded=True):
        st.markdown("- 🟢 **70점 이상**: `[적극 매수 / 눌림목 구간]` - 3분할 매수 정상 진행")
        st.markdown("- 🟡 **50점 ~ 69점**: `[관망 및 중립]` - 시장 불안정. 보수적 접근 요망")
        st.markdown("- 🔴 **50점 미만**: `[위험 경고 및 매수 금지]` - 현금 비중 100% 강제 (떨어지는 칼날)")


# ==========================================
# 사이드바 네비게이션
# ==========================================
st.sidebar.title("🧭 PRO 퀀트 네비게이션")
tabs = ["🪙 암호화폐 (Crypto)", "📈 국내 주식 (KOSPI)", "🦅 미국 주식 (US Stock)", "🥇 금·은 원자재 (Commodities)"]
market_choice = st.sidebar.radio("트레이딩 룸 선택", tabs)


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
        tf_map = {"15분": "15m", "1시간": "1h", "4시간": "4h", "1일": "1d", "주봉": "1w", "월봉": "1M"}
        exchange = ccxt.binance()
        try:
            bars = exchange.fetch_ohlcv(symbol, timeframe=tf_map[tf_label], limit=250)
            df = pd.DataFrame(bars, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df.ta.ema(length=20, append=True)
            df.ta.ema(length=50, append=True)
            df['is_pivot_low'] = df['low'] == df['low'].rolling(11, center=True).min()
            df['is_pivot_high'] = df['high'] == df['high'].rolling(11, center=True).max()
            return df
        except: return pd.DataFrame()

    c1, c2 = st.columns([1.3, 2])
    with c2:
        tf_options = ["15분", "1시간", "4시간", "1일", "주봉", "월봉"]
        selected_tf = st.radio("프레임 선택", tf_options, index=3, horizontal=True, key="c_tf")

    df = load_crypto(target_symbol, selected_tf)
    if not df.empty:
        curr_price = df['close'].iloc[-1]
        fibs = calculate_fibonacci_levels(df)
        score, verdict, facts = analyze_4_core_indicators(df, curr_price, fibs)
        
        with c1:
            st.subheader(f"종합 스코어: {score}점 {verdict}")
            with st.container(border=True):
                st.markdown("##### 📝 4대 팩트체크")
                st.markdown(f"- **[추세 EMA]**: {facts['EMA']}")
                st.markdown(f"- **[지저 FIB]**: {facts['FIB']}")
                st.markdown(f"- **[파동 WAVE]**: {facts['WAVE']}")
                st.markdown(f"- **[수급 VOL]**: {facts['VOL']}")

            render_score_legend()
            st.divider()

            if score < 50:
                st.error("📉 투심이 붕괴되었습니다. 이익 극대화를 위해 매수 가이드를 원천 차단합니다.")
            else:
                st.markdown(f"**[3분할 매수 타점]** ({'비중 축소' if score < 70 else '정상 진입'})")
                st.markdown(f"- 1차(추세탑승): ₩{fibs['fib_1']*KRW_RATE:,.0f} | 2차(강력방어): ₩{fibs['fib_2']*KRW_RATE:,.0f} | 3차(바닥매집): ₩{fibs['fib_3']*KRW_RATE:,.0f}")
        
        with c2:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
            fig.add_trace(go.Candlestick(x=df['time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
            if 'EMA_20' in df.columns: fig.add_trace(go.Scatter(x=df['time'], y=df['EMA_20'], line=dict(color='orange', width=1), name='EMA 20'), row=1, col=1)
            fig.add_hline(y=fibs['fib_1'], line_dash="dot", line_color="orange", row=1, col=1)
            fig.add_hline(y=fibs['fib_3'], line_dash="dot", line_color="red", row=1, col=1)
            
            colors = ['#26a69a' if r['close'] >= r['open'] else '#ef5350' for _, r in df.iterrows()]
            fig.add_trace(go.Bar(x=df['time'], y=df['volume'], marker_color=colors, name='Volume'), row=2, col=1)
            fig.update_layout(height=600, margin=dict(l=0, r=0, t=0, b=0), xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

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
        st.plotly_chart(fig_pie, use_container_width=True)

    sel_sec = st.selectbox("📂 시장 주도 섹터", list(kospi_sectors.keys()))
    sel_tick = st.radio(f"🏆 {sel_sec} 거래대금 상위 Top 5", kospi_sectors[sel_sec], horizontal=True).split(" ")[0]

    @st.cache_data(ttl=300)
    def load_stock(ticker, tf_label):
        tf_map = {"15분": "15m", "1시간": "60m", "4시간": "60m", "1일": "1d", "주봉": "1wk", "월봉": "1mo"}
        yf_tf = tf_map[tf_label]
        period = "60d" if yf_tf in ["15m", "60m"] else ("1y" if yf_tf == "1d" else "5y")
        
        df = yf.download(ticker, period=period, interval=yf_tf, progress=False)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        date_col = 'date' if 'date' in df.columns else 'datetime'
        df = df.rename(columns={date_col: 'time'})
        
        if tf_label == "4시간":
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time')
            df = df.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna().reset_index()

        df.ta.ema(length=20, append=True)
        df.ta.ema(length=50, append=True)
        df['is_pivot_low'] = df['low'] == df['low'].rolling(11, center=True).min()
        df['is_pivot_high'] = df['high'] == df['high'].rolling(11, center=True).max()
        return df

    c1, c2 = st.columns([1.3, 2])
    with c2:
        tf_options = ["15분", "1시간", "4시간", "1일", "주봉", "월봉"]
        selected_tf = st.radio("프레임 선택", tf_options, index=3, horizontal=True, key="s_tf")

    df = load_stock(sel_tick, selected_tf)
    if not df.empty:
        curr = df['close'].iloc[-1]
        fibs = calculate_fibonacci_levels(df)
        score, verdict, facts = analyze_4_core_indicators(df, curr, fibs)
        
        with c1:
            st.subheader(f"종합 스코어: {score}점 {verdict}")
            with st.container(border=True):
                st.markdown("##### 📝 4대 팩트체크")
                st.markdown(f"- **[추세 EMA]**: {facts['EMA']}")
                st.markdown(f"- **[지저 FIB]**: {facts['FIB']}")
                st.markdown(f"- **[파동 WAVE]**: {facts['WAVE']}")
                st.markdown(f"- **[수급 VOL]**: {facts['VOL']}")
            
            render_score_legend()
            st.divider()

            if score < 50: 
                st.error("📉 하락 징후 포착. 칼날을 피하기 위해 현금 비중 100%를 유지하십시오.")
            else: 
                st.markdown(f"**[3분할 매수 타점]** ({'보수적 접근 요망' if score < 70 else '정상 진입'})")
                st.markdown(f"- 1차(0.382): ₩{fibs['fib_1']:,.0f} | 2차(0.618): ₩{fibs['fib_2']:,.0f} | 3차(0.786): ₩{fibs['fib_3']:,.0f}")
        with c2:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
            fig.add_trace(go.Candlestick(x=df['time'], open=df['open'], high=df['high'], low=df['low'], close=df['close']), row=1, col=1)
            if 'ema_20' in df.columns: fig.add_trace(go.Scatter(x=df['time'], y=df['ema_20'], line=dict(color='orange', width=1), name='EMA 20'), row=1, col=1)
            fig.add_hline(y=fibs['fib_1'], line_dash="dot", line_color="orange", row=1, col=1)
            fig.add_hline(y=fibs['fib_3'], line_dash="dot", line_color="red", row=1, col=1)
            colors = ['#26a69a' if r['close'] >= r['open'] else '#ef5350' for _, r in df.iterrows()]
            fig.add_trace(go.Bar(x=df['time'], y=df['volume'], marker_color=colors, name='Volume'), row=2, col=1)
            fig.update_layout(height=600, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

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
        st.plotly_chart(fig_pie, use_container_width=True)

    sel_sec = st.selectbox("📂 미국 주도 섹터", list(us_sectors.keys()))
    sel_tick = st.radio(f"🏆 {sel_sec} 거래대금 상위 Top 5", us_sectors[sel_sec], horizontal=True).split(" ")[0]

    @st.cache_data(ttl=300)
    def load_us_stock(ticker, tf_label):
        tf_map = {"15분": "15m", "1시간": "60m", "4시간": "60m", "1일": "1d", "주봉": "1wk", "월봉": "1mo"}
        yf_tf = tf_map[tf_label]
        period = "60d" if yf_tf in ["15m", "60m"] else ("1y" if yf_tf == "1d" else "5y")
        
        df = yf.download(ticker, period=period, interval=yf_tf, progress=False)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={'date': 'time', 'datetime':'time'})
        
        if tf_label == "4시간":
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time')
            df = df.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna().reset_index()

        df.ta.ema(length=20, append=True)
        df.ta.ema(length=50, append=True)
        df['is_pivot_low'] = df['low'] == df['low'].rolling(11, center=True).min()
        df['is_pivot_high'] = df['high'] == df['high'].rolling(11, center=True).max()
        return df

    c1, c2 = st.columns([1.3, 2])
    with c2:
        tf_options = ["15분", "1시간", "4시간", "1일", "주봉", "월봉"]
        selected_tf = st.radio("프레임 선택", tf_options, index=3, horizontal=True, key="us_tf")

    df = load_us_stock(sel_tick, selected_tf)
    if not df.empty:
        curr = df['close'].iloc[-1]
        fibs = calculate_fibonacci_levels(df)
        score, verdict, facts = analyze_4_core_indicators(df, curr, fibs)
        
        with c1:
            st.subheader(f"종합 스코어: {score}점 {verdict}")
            st.caption(f"환율 적용: 1 USD = {KRW_RATE:,.0f} KRW")
            with st.container(border=True):
                st.markdown("##### 📝 4대 팩트체크")
                st.markdown(f"- **[추세 EMA]**: {facts['EMA']}")
                st.markdown(f"- **[지저 FIB]**: {facts['FIB']}")
                st.markdown(f"- **[파동 WAVE]**: {facts['WAVE']}")
                st.markdown(f"- **[수급 VOL]**: {facts['VOL']}")
            
            render_score_legend()
            st.divider()

            if score < 50: 
                st.error("📉 하락세 우위. 현금 100% 관망 요망.")
            else: 
                st.markdown(f"**[원화 3분할 매수 타점]** ({'비중 축소' if score < 70 else '정상 진입'})")
                st.markdown(f"- 1차: ₩{fibs['fib_1']*KRW_RATE:,.0f} | 2차: ₩{fibs['fib_2']*KRW_RATE:,.0f} | 3차: ₩{fibs['fib_3']*KRW_RATE:,.0f}")
        with c2:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
            fig.add_trace(go.Candlestick(x=df['time'], open=df['open']*KRW_RATE, high=df['high']*KRW_RATE, low=df['low']*KRW_RATE, close=df['close']*KRW_RATE), row=1, col=1)
            if 'ema_20' in df.columns: fig.add_trace(go.Scatter(x=df['time'], y=df['ema_20']*KRW_RATE, line=dict(color='orange', width=1), name='EMA 20'), row=1, col=1)
            fig.add_hline(y=fibs['fib_1']*KRW_RATE, line_dash="dot", line_color="orange", row=1, col=1)
            fig.add_hline(y=fibs['fib_3']*KRW_RATE, line_dash="dot", line_color="red", row=1, col=1)
            colors = ['#26a69a' if r['close'] >= r['open'] else '#ef5350' for _, r in df.iterrows()]
            fig.add_trace(go.Bar(x=df['time'], y=df['volume'], marker_color=colors, name='Volume'), row=2, col=1)
            fig.update_layout(height=600, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 4. 🥇 금·은 원자재 (Commodities)
# ==========================================
elif market_choice == "🥇 금·은 원자재 (Commodities)":
    st.title("🥇 금/은 안전자산 현물 봇")
    
    com_tick = st.radio("원자재 선택", ["GC=F (금 Gold)", "SI=F (은 Silver)"], horizontal=True).split(" ")[0]

    @st.cache_data(ttl=300)
    def load_commodity(ticker, tf_label):
        tf_map = {"15분": "15m", "1시간": "60m", "4시간": "60m", "1일": "1d", "주봉": "1wk", "월봉": "1mo"}
        yf_tf = tf_map[tf_label]
        period = "60d" if yf_tf in ["15m", "60m"] else ("1y" if yf_tf == "1d" else "5y")
        
        df = yf.download(ticker, period=period, interval=yf_tf, progress=False)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={'date': 'time', 'datetime':'time'})
        
        if tf_label == "4시간":
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time')
            df = df.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna().reset_index()

        df.ta.ema(length=20, append=True)
        df.ta.ema(length=50, append=True)
        df['is_pivot_low'] = df['low'] == df['low'].rolling(11, center=True).min()
        df['is_pivot_high'] = df['high'] == df['high'].rolling(11, center=True).max()
        return df

    c1, c2 = st.columns([1.3, 2])
    with c2:
        tf_options = ["15분", "1시간", "4시간", "1일", "주봉", "월봉"]
        selected_tf = st.radio("프레임 선택", tf_options, index=3, horizontal=True, key="com_tf")

    df = load_commodity(com_tick, selected_tf)
    if not df.empty:
        curr = df['close'].iloc[-1]
        fibs = calculate_fibonacci_levels(df, is_commodity=True)
        score, verdict, facts = analyze_4_core_indicators(df, curr, fibs)
        
        with c1:
            st.subheader(f"종합 스코어: {score}점 {verdict}")
            st.caption(f"환율 적용: 1 USD = {KRW_RATE:,.0f} KRW (온스당)")
            with st.container(border=True):
                st.markdown("##### 📝 4대 팩트체크")
                st.markdown(f"- **[추세 EMA]**: {facts['EMA']}")
                st.markdown(f"- **[지저 FIB]**: {facts['FIB']}")
                st.markdown(f"- **[파동 WAVE]**: {facts['WAVE']}")
                st.markdown(f"- **[수급 VOL]**: {facts['VOL']}")
            
            render_score_legend()
            st.divider()

            if score < 50: 
                st.error("📉 추세 이탈. 안전 자산 편입 보류 요망.")
            else: 
                st.markdown("**[얕은 3분할 타점]** (원자재 특수 로직 적용)")
                st.markdown(f"- 1차(0.236): ₩{fibs['fib_1']*KRW_RATE:,.0f} | 2차(0.382): ₩{fibs['fib_2']*KRW_RATE:,.0f} | 3차(0.50): ₩{fibs['fib_3']*KRW_RATE:,.0f}")
        with c2:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
            fig.add_trace(go.Candlestick(x=df['time'], open=df['open']*KRW_RATE, high=df['high']*KRW_RATE, low=df['low']*KRW_RATE, close=df['close']*KRW_RATE), row=1, col=1)
            if 'ema_20' in df.columns: fig.add_trace(go.Scatter(x=df['time'], y=df['ema_20']*KRW_RATE, line=dict(color='orange', width=1), name='EMA 20'), row=1, col=1)
            fig.add_hline(y=fibs['fib_1']*KRW_RATE, line_dash="dot", line_color="orange", row=1, col=1)
            fig.add_hline(y=fibs['fib_3']*KRW_RATE, line_dash="dot", line_color="red", row=1, col=1)
            colors = ['#26a69a' if r['close'] >= r['open'] else '#ef5350' for _, r in df.iterrows()]
            fig.add_trace(go.Bar(x=df['time'], y=df['volume'], marker_color=colors, name='Volume'), row=2, col=1)
            fig.update_layout(height=600, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)