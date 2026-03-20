"""Deep thinking: BTC->Alt lag edge의 본질과 한계."""
import numpy as np, pandas as pd, sys
sys.path.insert(0, '.')
import yfinance as yf

cost = 0.0018
print("=" * 80)
print("  DEEP THINK: edge의 본질은 무엇인가?")
print("=" * 80)
print()

# === 가설: BTC가 급변하면 알트는 "늦게" 따라간다 ===
# 이게 사실이면: BTC spike 시점의 알트 수익 < 다음 바 알트 수익
# 즉, 알트가 아직 덜 움직인 상태에서 진입 가능

btc = yf.Ticker("BTC-USD").history(period="180d", interval="1h")
btc.columns = [c.lower() for c in btc.columns]
btc_c = btc["close"].values; btc_ret = pd.Series(btc_c).pct_change().values
n = len(btc_c)

coins = {}
for coin, sym in [("SOL","SOL-USD"),("ETH","ETH-USD"),("XRP","XRP-USD"),("ADA","ADA-USD")]:
    df = yf.Ticker(sym).history(period="180d", interval="1h")
    df.columns = [c.lower() for c in df.columns]
    coins[coin] = df["close"].values

print("[1] BTC spike 시점에서 알트가 얼마나 '이미' 움직였는가?")
print()
print("    BTC 1h 수익률 vs 같은 시간 알트 수익률 (동시성)")
print()

for thresh in [0.012, 0.015]:
    print("  BTC |ret| > %.1f%%:" % (thresh*100))
    for coin in ["SOL","ETH","XRP","ADA"]:
        cn = min(n, len(coins[coin]))
        br = btc_ret[-cn:]
        cr = pd.Series(coins[coin][-cn:]).pct_change().values

        # BTC spike bars
        spikes = [(i, br[i]) for i in range(cn//3, cn) if not np.isnan(br[i]) and abs(br[i]) > thresh]

        if not spikes: continue

        # Same bar: BTC moved X%, alt moved Y%
        # Ratio = Y/X shows how much alt "already" moved
        ratios = []
        for i, b in spikes:
            if np.isnan(cr[i]): continue
            if abs(b) > 0.001:
                ratios.append(cr[i] / b)  # >1 means alt moved MORE, <1 means LESS

        avg_ratio = np.mean(ratios) if ratios else 0
        # Next bar alt return (what we actually capture)
        next_rets = [cr[i+1] * (1 if b > 0 else -1) for i, b in spikes if i+1 < cn and not np.isnan(cr[i+1])]
        avg_next = np.mean(next_rets) if next_rets else 0

        print("    %s: same_bar_ratio=%.2f (alt/btc), next_bar=%+.3f%%" % (coin, avg_ratio, avg_next*100))
    print()

print()
print("[2] 진짜 'lag'이 있는가? 시간대별 상관관계")
print()

# Cross-correlation: BTC ret vs ALT ret at different lags
for coin in ["SOL", "ETH"]:
    cn = min(n, len(coins[coin]))
    br = btc_ret[-cn:]
    cr = pd.Series(coins[coin][-cn:]).pct_change().values

    valid = ~np.isnan(br) & ~np.isnan(cr)
    br_v = br[valid]; cr_v = cr[valid]

    print("  BTC vs %s cross-correlation:" % coin)
    for lag in range(-3, 4):
        if lag >= 0:
            corr = np.corrcoef(br_v[:len(br_v)-max(1,lag)], cr_v[lag:len(cr_v)] if lag > 0 else cr_v[:len(cr_v)])[0,1]
        else:
            corr = np.corrcoef(br_v[-lag:], cr_v[:len(cr_v)+lag])[0,1]
        marker = " <-- peak" if abs(corr) == max(abs(np.corrcoef(br_v[:len(br_v)-1], cr_v[1:])[0,1]),
                                                    abs(np.corrcoef(br_v, cr_v)[0,1])) else ""
        print("    lag=%+d: corr=%.4f%s" % (lag, corr, " <-- SIMULTANEOUS" if lag==0 else ""))
    print()

print()
print("[3] BTC spike의 '원인'은 무엇인가?")
print()
print("  BTC 1h >1.2%가 발생하는 시간대 분포:")
spikes_all = [(i, btc_ret[i]) for i in range(n) if not np.isnan(btc_ret[i]) and abs(btc_ret[i]) > 0.012]
hours = [btc.index[i].hour for i, _ in spikes_all if i < len(btc.index)]
hour_counts = pd.Series(hours).value_counts().sort_index()
for h in range(0, 24, 2):
    cnt = sum(1 for hh in hours if h <= hh < h+2)
    bar = "#" * cnt
    print("    %02d-%02d UTC: %2d  %s" % (h, h+2, cnt, bar))
print()
print("  -> 특정 시간대에 spike가 집중되면 '뉴스 이벤트' 성격")
print("  -> 균등하면 '랜덤 변동성' 성격")
print()

print("[4] Edge의 지속 가능성: 왜 이 edge가 사라지지 않는가?")
print()
print("  가설 A: 알트 유동성이 BTC보다 낮아서 price discovery가 느림")
print("    -> 대형 거래소에서 SOL/ETH 유동성 충분")
print("    -> 이 가설이면 유동성 좋은 ETH에서 edge 작아야")

# Check: ETH vs SOL edge comparison
for coin in ["SOL", "ETH"]:
    cn = min(n, len(coins[coin]))
    br = btc_ret[-cn:]
    cr = pd.Series(coins[coin][-cn:]).pct_change().values
    split = cn // 3

    results = []
    pe = 0
    for i in range(max(split,2), cn-7):
        if i < pe: continue
        if np.isnan(br[i]) or abs(br[i]) < 0.012: continue
        side = "BUY" if br[i] > 0 else "SELL"
        next_ret = cr[i+1] if i+1 < cn and not np.isnan(cr[i+1]) else 0
        pnl = next_ret * (1 if side=="BUY" else -1)
        results.append(pnl - cost/6)  # rough cost for 1 bar
        pe = i + 2  # just skip next bar

    arr = np.array(results)
    print("    %s: n=%d avg=%+.4f%%" % (coin, len(arr), np.mean(arr)*100))

print()
print("  가설 B: 알트 트레이더가 BTC를 보고 반응하는 데 시간이 걸림")
print("    -> 자동화 봇이 이미 있으므로 이 lag은 매우 짧을 것")
print("    -> 1h bar 해상도로는 이 lag을 포착하기 어려울 수 있음")
print()
print("  가설 C: BTC 급변은 '정보 이벤트'이고 알트는 '가치 재평가' 필요")
print("    -> 이 경우 lag은 real (정보 처리 시간)")
print("    -> 6h 이내에 대부분 반영 완료")
print()

print("[5] 실행 현실성: 실제로 이 전략을 돌리면?")
print()
print("  a) BTC 1h bar close 기다리기: 매 정시")
print("     -> 봇이 매시 정각에 BTC close 확인")
print("     -> |ret| > 1.2%면 알트 4코인 진입")
print("     -> 문제: 정시에 수백 봇이 동시 진입 -> 슬리피지")
print()
print("  b) 실시간 모니터링 (개선):")
print("     -> BTC 가격을 1초마다 체크")
print("     -> 직전 1h open 대비 1.2% 이상 변동 감지 시 즉시 진입")
print("     -> bar close 안 기다림 -> 더 빠른 진입 가능")
print("     -> 하지만 bar 중간 spike가 bar close에서 사라질 수 있음")
print("       (spike 후 되돌림 = false signal)")
print()

# Test: how often does BTC spike mid-bar but close flat?
print("  c) Mid-bar spike vs Close spike 괴리:")
# Can't test with 1h data alone, but we can check:
# High/Low range vs close-to-close return
btc_h = btc["high"].values; btc_l = btc["low"].values; btc_o = btc["open"].values
intrabar_max = np.maximum((btc_h - btc_o) / btc_o, (btc_o - btc_l) / btc_o)
close_ret = np.abs(btc_ret)
# Bars where intrabar move > 1.2% but close-to-close < 0.6%
false_spikes = 0; true_spikes = 0
for i in range(n):
    if np.isnan(intrabar_max[i]) or np.isnan(close_ret[i]): continue
    if intrabar_max[i] > 0.012:
        if close_ret[i] < 0.006:
            false_spikes += 1
        else:
            true_spikes += 1

print("     Intrabar >1.2%% but close <0.6%%: %d (false signals)" % false_spikes)
print("     Intrabar >1.2%% and close >0.6%%: %d (true signals)" % true_spikes)
print("     False signal rate: %.0f%%" % (false_spikes / max(false_spikes+true_spikes,1) * 100))
print()

print("[6] 근본 질문: +0.28%/trade로 $500에서 의미 있는 수익?")
print()
for lev in [3, 5]:
    avg_pnl = 0.0028 * lev
    trades_per_month = 69 / 6  # 69 trades in 6 months
    monthly_dollar = 500 * avg_pnl * trades_per_month
    annual = monthly_dollar * 12
    print("  %dx: %+.2f%%/trade, ~%.0f trades/mo = $%+.0f/mo ($%+.0f/yr)" % (
        lev, avg_pnl*100, trades_per_month, monthly_dollar, annual))
    print("       $500 -> $%.0f (1yr, compound)" % (500 * (1 + avg_pnl) ** (trades_per_month * 12)))

print()
print("=" * 80)
print("  FINAL ASSESSMENT")
print("=" * 80)
print()
print("  Edge 존재 여부: CONDITIONAL YES")
print("    - 'BTC 롱' 아님 (하락장에서도 작동)")
print("    - BTC->알트 momentum continuation lag에서 발생")
print("    - 180일 하락장(-38.5%)에서도 양수")
print("    - FOLLOW vs COUNTER: +0.28% vs -0.52% (명확한 차이)")
print()
print("  Edge 크기: SMALL")
print("    - 1x: +0.28%/trade (수수료 차감 후)")
print("    - 3x: +0.84%/trade")
print("    - 월 ~11.5건 (1코인) * 4코인 = 46건 (상관관계 보정 시 ~15건)")
print()
print("  위험:")
print("    - 후반 90일 edge 약화 (0.62% -> 0.16%)")
print("    - n=69 (통계 경계선)")
print("    - 실행 슬리피지 미반영")
print("    - 레버리지 시 MDD 증폭")
print()
print("  솔직한 판단:")
print("    이건 '확실한 수익 전략'이 아니라")
print("    '통계적으로 약한 edge가 있을 수 있는 가설'이다.")
print("    Paper trading 2-4주로 실증하지 않으면 판단 불가.")
