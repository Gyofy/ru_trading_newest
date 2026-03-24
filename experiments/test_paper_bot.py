"""Quick test of paper bot single cycle."""
import sys, os
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_tsmom_paper import PaperBot, LOCK_FILE, COIN_LIST

LOCK_FILE.unlink(missing_ok=True)
bot = PaperBot()

print('=== v5.3 Single Cycle Test ===')
for coin in COIN_LIST:
    df = bot.fetch_4h(coin)
    if df.empty:
        print(f'  {coin}: EMPTY')
        continue
    sig, info = bot.generate_signal(df, coin)
    side = 'LONG' if sig == 1 else ('SHORT' if sig == -1 else 'FLAT')
    reason = info.get('reason', info.get('side', ''))
    stype = info.get('strategy_type', '') if sig != 0 else ''
    rsi = df['rsi_14'].iloc[-1]
    close = df['close'].iloc[-1]
    print(f'  {coin:5s}: {side:5s} | {reason:20s} {stype:10s} | ${close:.2f} RSI={rsi:.1f}')

bot._release_lock()
print('OK')
