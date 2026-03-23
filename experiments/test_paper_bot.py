"""Quick test of paper bot single cycle."""
import sys, os
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_tsmom_paper import PaperBot

bot = PaperBot()

print('=== Single Cycle Test ===')
for coin in ['BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'DOT', 'LINK']:
    df = bot.fetch_4h(coin)
    if df.empty:
        print(f'  {coin}: EMPTY')
        continue
    sig, info = bot.generate_signal(df, coin)
    side = 'LONG' if sig == 1 else ('SHORT' if sig == -1 else 'FLAT')
    reason = info.get('reason', info.get('side', ''))
    rsi = df['rsi_14'].iloc[-1]
    close = df['close'].iloc[-1]
    print(f'  {coin}: {side:5s} | {reason:20s} | close=${close:.2f} rsi={rsi:.1f}')

print('Bot verified OK')
