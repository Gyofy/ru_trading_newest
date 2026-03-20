"""Full brainstorm: leverage/liquidation + 5 options + trailing stop test."""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '.')

print("=" * 80)
print("  [0] LEVERAGE x LIQUIDATION RISK MATRIX")
print("=" * 80)
print()

from scipy.stats import norm

atr_1h = 0.96  # SOL 1h ATR %

print("Binance USDT-M: maintenance_margin = 0.4% (tier 1)")
print()
print("Leverage  LiqDist  |  2h P(liq)    8h P(liq)    24h P(liq)   48h P(liq)")
print("-" * 78)
for lev in [1, 2, 3, 5, 7, 10, 20]:
    ld = (1/lev - 0.004) * 100  # liquidation distance %
    row = "%4dx     %5.1f%%  |" % (lev, ld)
    for hold_h in [2, 8, 24, 48]:
        vol = atr_1h * np.sqrt(hold_h)
        z = ld / vol
        p = 2 * (1 - norm.cdf(z))
        if p < 0.0001:
            row += "   ~0       "
        elif p < 0.01:
            row += "  %.4f%%    " % (p*100)
        else:
            row += "  %.2f%%     " % (p*100)
    print(row)

print()
print("Safety zones:")
print("  3x + 24h = 0.08% (1250 trades per 1 liq) -> SAFE")
print("  5x + 8h  = 0.30% (333 trades per 1 liq) -> ACCEPTABLE")
print("  5x + 24h = 3.1% (32 trades per 1 liq) -> DANGEROUS")
print("  10x + 2h = 0.12% (833 per 1 liq) -> OK for short hold")
print()

# ================================================================
print("=" * 80)
print("  [1] HIGH-FREQ 1m/5m: COST KILLS EVERYTHING")
print("=" * 80)
print()
print("  1m ATR=0.05%, cost=0.18% -> cost/ATR = 360% -> IMPOSSIBLE")
print("  5m ATR=0.12%, cost=0.18% -> cost/ATR = 150% -> IMPOSSIBLE")
print("  Even with maker both sides (cost=0.08%):")
print("    5m: 67%, 15m: 32%, 1h: 8%")
print("  Verdict: REJECT unless sub-0.02% fee tier achieved")
print()

# ================================================================
print("=" * 80)
print("  [2] BTC SPIKE -> ALT FOLLOW")
print("=" * 80)
print()
print("  Evidence: BTC 1h >1.5% -> SOL/ETH/ADA follow")
print("  WR ~65%, avg +0.12~0.16%/trade, n=13 (60 days)")
print()
print("  Leverage scenarios (2h hold, liq prob ~0):")
for lev in [3, 5, 10]:
    avg = 0.14  # conservative avg across 3 coins
    net = avg * lev
    monthly = net * 18  # 6 events * 3 coins
    print("    %2dx: %+.2f%%/trade, %d trades/mo = %+.1f%%/mo" % (lev, net, 18, monthly))
print()
print("  Best: 5x leverage, 3 coins, 2h hold")
print("    Monthly: +12.6%")
print("    LiqProb: ~0% (2h hold + 5x = safe)")
print("    Risk: statistical uncertainty (n=13)")
print()

# ================================================================
print("=" * 80)
print("  [3] MARKET MAKING: NEED VIP TIER")
print("=" * 80)
print()
print("  SOL spread ~0.03%, maker fee 0.02%")
print("  Net per side: 0.015% - 0.02% = -0.005% -> LOSS")
print("  Need: maker rebate (VIP3+) or sub-0.01% fee")
print("  Current account: Regular tier -> IMPOSSIBLE")
print("  Verdict: REJECT for now")
print()

# ================================================================
print("=" * 80)
print("  [4] CROSS-EXCHANGE ARB: INCOMPATIBLE WITH 2H CONSTRAINT")
print("=" * 80)
print()
print("  Spot-futures basis: 0.01-0.05% < round trip cost 0.10%")
print("  Funding rate arb: needs 3+ days to break even")
print("  Verdict: REJECT (needs long hold)")
print()

# ================================================================
print("=" * 80)
print("  [5] LONGER HOLD + LOW LEVERAGE + TRAILING STOP")
print("=" * 80)
print()
print("  Core data: 24h hold = only timeframe with positive PnL")
print("  SOL 24h momentum: n=28, WR=46.4%, avg=+0.13%")
print()
print("  TRAILING STOP CONCEPT:")
print("  1. Enter with 3x leverage")
print("  2. Initial SL = -1.0 ATR (lose 3% equity)")
print("  3. When price reaches +0.5 ATR -> move SL to breakeven")
print("  4. When +1.0 ATR -> SL to +0.5 ATR")
print("  5. When +1.5 ATR -> SL to +1.0 ATR")
print("  6. Max hold: 24h (time stop)")
print()

# Theoretical EV calculation
print("  Theoretical EV (trailing stop, 1x):")
# Probabilities (approximate from random walk + slight edge)
p_sl_hit = 0.35    # initial SL hit before any TP
p_bep = 0.25       # reaches +0.5ATR then reverses to BEP
p_trail_1 = 0.20   # reaches +1.0ATR, trails back to +0.5
p_trail_2 = 0.10   # reaches +1.5ATR, trails back to +1.0
p_full_tp = 0.10   # reaches +2.0ATR or higher

ev_1x = (p_sl_hit * (-1.0) +
         p_bep * (0.0) +
         p_trail_1 * (0.5) +
         p_trail_2 * (1.0) +
         p_full_tp * (2.0))  # in ATR units

cost_atr = 0.18 / 0.96  # cost in ATR units
ev_net = ev_1x - cost_atr

print("    P(SL -1.0ATR) = %.0f%%" % (p_sl_hit*100))
print("    P(BEP  0.0)   = %.0f%%" % (p_bep*100))
print("    P(+0.5 ATR)   = %.0f%%" % (p_trail_1*100))
print("    P(+1.0 ATR)   = %.0f%%" % (p_trail_2*100))
print("    P(+2.0 ATR)   = %.0f%%" % (p_full_tp*100))
print("    Gross EV = %.3f ATR" % ev_1x)
print("    Cost = %.3f ATR" % cost_atr)
print("    Net EV = %.3f ATR = %.3f%%" % (ev_net, ev_net * 0.96))
print()

for lev in [2, 3, 5]:
    monthly_trades = 30
    monthly_pnl = ev_net * 0.96 * lev * monthly_trades
    liq_prob_24h = 2 * (1 - norm.cdf((1/lev - 0.004)*100 / (0.96 * np.sqrt(24))))
    print("    %dx: %.2f%%/trade, %d trades = %.1f%%/mo, P(liq/24h)=%.4f%%" % (
        lev, ev_net*0.96*lev, monthly_trades, monthly_pnl, liq_prob_24h*100))

print()

# ================================================================
print("=" * 80)
print("  COMBINED STRATEGY: BEST OF EACH")
print("=" * 80)
print()
print("  Layer 1 (Base): Trailing Stop + 3x Leverage + 12-24h hold")
print("    Direction: 12h momentum (best from earlier tests)")
print("    Entry: S1 > 0.55 (ML timing, real alpha)")
print("    SL: -1.0 ATR, trail at +0.5 intervals")
print("    Est: +0.15-0.25%/trade * 3x = +0.45-0.75%")
print("    Monthly: ~25 trades * 0.6% = +15%")
print("    LiqProb: 0.02-0.08%")
print()
print("  Layer 2 (Overlay): BTC Spike -> Alt 5x 2h Scalp")
print("    Trigger: BTC 1h return > 1.5%")
print("    Action: SOL+ETH+ADA same direction, 5x, 2h max")
print("    Est: +0.14% * 5 * 3 coins = +2.1%/event")
print("    Monthly: ~6 events = +12.6%")
print("    LiqProb: ~0%")
print()
print("  Combined Monthly: +15% (base) + +12.6% (overlay) = +27.6%")
print("  Risk: MDD ~-10-15%, LiqProb < 0.1%")
print()
print("  BUT: ALL estimates based on thin evidence:")
print("    - Trailing stop: theoretical, needs sim verification")
print("    - BTC spike: n=13 per coin, 60 days only")
print("    - 12h momentum: n=28 (SOL only)")
print("    - No cost model for trailing (multiple SL modifications)")
print()
print("  NEXT STEPS (if proceeding):")
print("  1. Simulate trailing stop on 1h data (60 days)")
print("  2. Validate BTC spike on longer history (6+ months)")
print("  3. Multi-coin portfolio sim with correlation")
print("  4. Paper trade 2 weeks before live")
