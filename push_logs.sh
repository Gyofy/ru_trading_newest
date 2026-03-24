#!/bin/bash
# 트레이딩 로그 자동 push (수동 실행 or cron)
# Usage: bash push_logs.sh

cd "$(dirname "$0")"

# Stage trade logs
git add -f data/reports/tsmom_paper/trades.jsonl \
           data/reports/tsmom_paper/state.json \
           data/reports/tsmom_paper/signal_log.jsonl \
           data/reports/tsmom_paper/trade_analysis.jsonl \
           data/reports/tsmom_paper/trajectories.jsonl \
           trading_result/ \
           2>/dev/null

# Check if there are changes
if git diff --cached --quiet; then
    echo "No log changes to push."
    exit 0
fi

# Commit with timestamp
TIMESTAMP=$(date +%Y-%m-%d_%H:%M)
TRADES=$(wc -l < data/reports/tsmom_paper/trades.jsonl 2>/dev/null || echo 0)
EQUITY=$(python -c "import json; print(f'{json.load(open(\"data/reports/tsmom_paper/state.json\"))[\"equity\"]:.2f}')" 2>/dev/null || echo "?")

git commit -m "Trade log update: ${TIMESTAMP} | ${TRADES} trades | \$${EQUITY}"
git push newest main

echo "Logs pushed: ${TRADES} trades, equity \$${EQUITY}"
