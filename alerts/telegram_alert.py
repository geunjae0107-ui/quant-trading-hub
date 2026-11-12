#!/usr/bin/env python3
"""
텔레그램 매수 타점 알림 — 앱(app.py)의 스크리너와 '똑같은 코드'로 143종목을 판정해서 🟢 매수 타점만 보냅니다.
  · 아침(07:30): 미국장 마감 후 → 미국 신호 + 오늘 아침 주문할 국내 신호 다시 알림
  · 저녁(16:45): 국내장 마감 후 → 내일 아침 주문할 국내 신호
  · 앱의 데이터 로더 · 봉 마감 규칙 · ICT 판정 · 실적 5일 가드 · 실전 대상 구분을 그대로 씀 (앱 화면과 같은 결론)
  · 처음 알리는 신호에는 🆕 표시 (state.json에 이전 알림 기억)

필요한 환경 변수: TELEGRAM_TOKEN (BotFather에서 받은 봇 토큰) · TELEGRAM_CHAT_ID (내 채팅 ID)
실행:
  python alerts/telegram_alert.py                 # 판정 후 텔레그램 전송
  python alerts/telegram_alert.py --dry-run       # 보내지 않고 메시지만 화면에 출력 (처음 확인용)
  python alerts/telegram_alert.py --chat-id       # 봇에게 아무 말이나 보낸 뒤 실행 → 내 채팅 ID 확인
"""
import argparse, json, os, sys, time, types
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
MARK = "# ==========================================\n# 사이드바 네비게이션"
KST = timezone(timedelta(hours=9))
MAX_LEN = 3800                              # 텔레그램 한 메시지 4096자 제한 → 여유 두고 나눔


# ───────────────────────── 앱 엔진 불러오기 (화면 부분 제외 · 데이터는 진짜로 받음) ─────────────────────────
def load_engine(app_path):
    import warnings
    warnings.filterwarnings("ignore")

    class _St:                               # streamlit 대신: 캐시는 그냥 함수 실행, 화면 함수는 아무것도 안 함
        def cache_data(self, *a, **k):
            return a[0] if (a and callable(a[0]) and not k) else (lambda f: f)
        cache_resource = cache_data
        def __getattr__(self, name):
            return lambda *a, **k: None
    sys.modules["streamlit"] = _St()
    for m in ("plotly", "plotly.graph_objects"):
        sys.modules[m] = types.ModuleType(m)
    sys.modules["plotly.subplots"] = types.SimpleNamespace(make_subplots=lambda **k: None)
    sys.modules["plotly"].graph_objects = sys.modules["plotly.graph_objects"]
    def _raise(*a, **k):
        raise RuntimeError("패키지 없음")
    for m, fake in (("yfinance", types.SimpleNamespace(Ticker=_raise, download=_raise)), ("ccxt", types.ModuleType("ccxt")),
                    ("requests", types.SimpleNamespace(get=_raise, post=_raise))):
        try:
            __import__(m)
        except ImportError:                   # 설치 안 된 환경(테스트용)에서만 대체. 실제 실행은 pip install -r requirements.txt
            print(f"  ⚠️ {m} 패키지가 없어 해당 데이터는 받지 않습니다 (pip install {m})")
            sys.modules[m] = fake
    try:
        import pandas_ta  # noqa: F401
        ok = hasattr(pd.DataFrame, "ta")
    except Exception:
        ok = False
    if not ok:                                # pandas_ta가 없거나 깨졌을 때: 앱이 쓰는 ema만 대신 제공
        sys.modules["pandas_ta"] = types.ModuleType("pandas_ta")
        if not hasattr(pd.DataFrame, "ta"):
            @pd.api.extensions.register_dataframe_accessor("ta")
            class _TA:
                def __init__(self, o): self.o = o
                def ema(self, length, append=True):
                    self.o[f"EMA_{length}"] = self.o["close"].ewm(span=length, adjust=False, min_periods=length).mean()
    src = open(app_path, encoding="utf-8").read()
    if MARK not in src:
        sys.exit("app.py에서 '사이드바 네비게이션' 구분선을 찾지 못했습니다.")
    ns = {"__name__": "engine"}
    exec(compile(src[: src.index(MARK)], app_path, "exec"), ns)
    return ns


def use_csv_data(ns, folder, asof=None):
    """테스트용: 인터넷 대신 폴더의 CSV(time,open,high,low,close,volume)로 판정"""
    def read(t):
        p = os.path.join(folder, t.replace("/", "_").replace("=", "_").replace("^", "_") + ".csv")
        if not os.path.exists(p):
            return None
        d = pd.read_csv(p, parse_dates=["time"])
        if asof:
            d = d[d["time"] <= pd.Timestamp(asof)]
        return d
    def yf_batch(tickers, chunk=20):
        out = {}
        for t in tickers:
            d = read(t)
            if d is not None:
                out[t] = d.set_index("time").rename(columns=str.capitalize)
        return out
    def load_crypto(sym, tf):
        d = read(sym)
        if d is None:
            return pd.DataFrame(), "csv", "없음"
        df, err = ns["_prep_yf_frame"](d.set_index("time").rename(columns=str.capitalize), "1일")
        return ns["tag_df"](df, "CRYPTO", "1일", sym), "csv", err
    ns["_yf_batch"], ns["load_crypto"] = yf_batch, load_crypto
    ns["earnings_days"] = lambda t: (None, None)


# ───────────────────────── 판정 ─────────────────────────
def scan_all(ns, only=None):
    rows = []
    for market in ns["SCREEN_UNIVERSE"]:
        if only:                              # 테스트용: 일부 종목만
            uni = {s: [x for x in it if x[0] in only] for s, it in ns["SCREEN_UNIVERSE"][market].items()}
            uni = {s: it for s, it in uni.items() if it}
            if not uni:
                continue
            ns["SCREEN_UNIVERSE"][market] = uni
        t0 = time.time()
        part = ns["scan_market"](market)
        rows += part
        print(f"  {market}: {len(part)}종목 ({time.time() - t0:.0f}초)", flush=True)
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    try:
        res = ns["apply_earnings"](res)       # 🟢인데 실적 5일 안 → '📅 실적 후 판단' (앱과 같은 규칙)
    except Exception as e:
        print(f"  (실적 일정 조회 실패: {type(e).__name__} — 실적 가드 없이 진행)")
    return res


def fmt_px(t, v):
    if v is None or not np.isfinite(v):
        return "-"
    if str(t).endswith((".KS", ".KQ")):
        return f"₩{v:,.0f}"
    return f"${v:,.2f}" if v >= 1 else f"${v:,.4f}"


def build_message(ns, res, state, now):
    is_am = now.hour < 12
    head = f"📡 매수 타점 알림 · {now:%m/%d}({'월화수목금토일'[now.weekday()]}) {now:%H:%M} {'아침' if is_am else '저녁'}"
    if res is None or res.empty:
        return [head + "\n\n데이터를 받지 못했습니다. 앱에서 직접 확인하세요."], set()
    kind = res.get("_kind", pd.Series("", index=res.index)).astype(str)
    concl = res.get("결론", pd.Series("", index=res.index)).astype(str)
    biko = res.get("비고", pd.Series("", index=res.index)).astype(str)

    def scope(r):
        ok, why = ns["trade_scope"](r["티커"], r.get("섹터"))
        if ok and "신규 상장" in str(r.get("비고", "")):
            return False, "상장 1년 미만"
        return ok, why

    buys, refs, keys = [], [], set()
    for _, r in res[kind == "BUY"].iterrows():
        ok, why = scope(r)
        tr = r.get("_tranche")
        tr = int(tr) if isinstance(tr, (int, float)) and np.isfinite(tr) else None
        key = f"{r['티커']}|{tr}"                      # 같은 종목·같은 차수를 10일 안에 이미 알렸으면 🆕 아님
        keys.add(key)
        last = state.get(key)
        new = last is None or (now.date() - datetime.fromisoformat(last).date()).days > 10
        (buys if ok else refs).append((r, tr, new, why))
    earn = res[concl.str.contains("실적 후 판단")]
    conf = res[concl.str.contains("확인 대기")]
    errs = int(res.get("단계", pd.Series("", index=res.index)).astype(str).str.contains("오류").sum())

    L = [head, ""]
    if buys:
        L.append(f"🟢 <b>실전 대상 매수 타점 {len(buys)}개</b>")
        buys.sort(key=lambda x: (not x[2], str(x[0].get("시장")), x[1] or 9))
        for r, tr, new, _ in buys:
            px = r.get("현재가")
            sd = r.get("손절선까지")
            stop = px * (1 + sd) if (px is not None and sd is not None and np.isfinite(sd)) else None
            loss1 = (px - stop) if stop else None
            wk = str(r.get("주봉", ""))
            L.append(f"{'🆕 ' if new else ''}<b>{r['종목']}</b> ({r['티커']}) · {tr or '-'}차 · {r.get('섹터', '')}")
            L.append(f"   현재 {fmt_px(r['티커'], px)} · 손절 {fmt_px(r['티커'], stop)}"
                     + (f" ({sd * 100:+.1f}%)" if sd is not None and np.isfinite(sd) else "")
                     + (f" · 1주 손실 {fmt_px(r['티커'], loss1)}" if loss1 else ""))
            extra = [x for x in (wk if "하락" in wk else "", str(r.get("기준봉", ""))) if x]
            if extra:
                L.append("   " + " · ".join(extra))
    else:
        L.append("🟢 실전 대상 매수 타점: <b>없음</b>")
    if len(earn):
        L += ["", "📅 실적 후 판단 (지금 사지 않음): " + ", ".join(f"{r['종목']}({r.get('실적', '')})" for _, r in earn.head(10).iterrows())]
    if len(conf):
        L += ["", "⏳ 다음 종가로 1차 확인: " + ", ".join(str(r["종목"]) for _, r in conf.head(12).iterrows())]
    if refs:
        L += ["", "🏷️ 참고용 신호 (매매 안 함): " + ", ".join(f"{r['종목']}" for r, *_ in refs[:15])]
    L += ["", "※ 주문 전 앱에서 다시 확인 · 동시 3종목·같은 섹터 1종목 · 체결 즉시 손절 주문"
          + (" · 미국 주식 손절은 달러로" if any(not str(r["티커"]).endswith((".KS", ".KQ")) for r, *_ in buys) else "")]
    if errs:
        L.append(f"※ 데이터 오류 {errs}종목 (앱에서 확인)")
    msgs, cur = [], ""
    for line in L:
        if len(cur) + len(line) + 1 > MAX_LEN:
            msgs.append(cur)
            cur = ""
        cur += line + "\n"
    if cur.strip():
        msgs.append(cur)
    return msgs, keys


# ───────────────────────── 텔레그램 ─────────────────────────
def tg_send(token, chat_id, text):
    import requests
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data=dict(chat_id=chat_id, text=text, parse_mode="HTML", disable_web_page_preview="true"), timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"텔레그램 전송 실패 HTTP {r.status_code}: {r.text[:200]}")


def tg_chat_id(token):
    import requests
    r = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=30).json()
    ids = {(u.get("message") or {}).get("chat", {}).get("id") for u in r.get("result", [])}
    ids.discard(None)
    if not ids:
        print("받은 메시지가 없습니다. 텔레그램에서 내 봇에게 아무 말이나 보낸 뒤 다시 실행하세요.")
    for i in ids:
        print(f"내 채팅 ID: {i}  → TELEGRAM_CHAT_ID 로 등록")


def main():
    ap = argparse.ArgumentParser(description="텔레그램 매수 타점 알림")
    ap.add_argument("--app", default=os.path.join(HERE, "..", "app.py"))
    ap.add_argument("--state", default=os.path.join(HERE, "state.json"))
    ap.add_argument("--dry-run", action="store_true", help="보내지 않고 화면에만 출력")
    ap.add_argument("--chat-id", action="store_true", help="내 채팅 ID 확인")
    ap.add_argument("--data-dir", default=None, help="테스트용: 인터넷 대신 CSV 폴더 사용")
    ap.add_argument("--asof", default=None, help="테스트용: 이 날짜까지만")
    ap.add_argument("--only", nargs="*", help="테스트용: 이 티커만")
    a = ap.parse_args()
    token, chat = os.environ.get("TELEGRAM_TOKEN", "").strip(), os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if a.chat_id:
        if not token:
            sys.exit("TELEGRAM_TOKEN 환경 변수가 없습니다.")
        return tg_chat_id(token)
    if not a.dry_run and not (token and chat):
        sys.exit("TELEGRAM_TOKEN · TELEGRAM_CHAT_ID 환경 변수가 필요합니다 (먼저 --dry-run 으로 메시지를 확인해 보세요).")
    if not os.path.exists(a.app):
        alt = os.path.join(HERE, "..", "app_legacy.py")
        if not os.path.exists(alt):
            sys.exit(f"앱 파일을 찾지 못했습니다: {a.app}")
        a.app = alt
    now = datetime.now(KST)
    print(f"판정 시작 · {now:%Y-%m-%d %H:%M} KST")
    ns = load_engine(a.app)
    print(f"엔진 {ns.get('APP_VERSION', '?')}")
    if a.data_dir:
        use_csv_data(ns, a.data_dir, a.asof)
    res = scan_all(ns, set(a.only) if a.only else None)
    try:
        state = dict(json.load(open(a.state, encoding="utf-8")).get("alerted", {}))     # {티커|차수: 마지막 알림 날짜}
    except Exception:
        state = {}
    msgs, keys = build_message(ns, res, state, now)
    for m in msgs:
        print("\n" + m)
        if not a.dry_run:
            tg_send(token, chat, m)
            time.sleep(1)
    if not a.dry_run:
        try:
            for k in keys:
                state[k] = now.date().isoformat()
            cut = now.date() - timedelta(days=60)
            state = {k: v for k, v in state.items() if datetime.fromisoformat(v).date() >= cut}
            json.dump(dict(alerted=state, updated=now.isoformat()), open(a.state, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        except Exception:
            pass
        print(f"\n전송 완료 ({len(msgs)}개 메시지)")


if __name__ == "__main__":
    main()
