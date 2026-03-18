#!/bin/bash
# Demo Trading 세션 모니터 — 5분마다 상태를 session_log에 기록

LOG_DIR="/home/wlsry/Ru_trading/data/reports/demo_trading"
SESSION_LOG="$LOG_DIR/session_$(date +%Y%m%d).log"
DEMO_LOG="$LOG_DIR/demo_trading.log"
JSONL="$LOG_DIR/demo_log.jsonl"

echo "=== SESSION MONITOR START $(date '+%Y-%m-%d %H:%M:%S KST') ===" >> "$SESSION_LOG"

while true; do
    TS=$(date '+%Y-%m-%d %H:%M:%S KST')
    PID=$(pgrep -f "run_demo_trading.py" | head -1)

    if [ -z "$PID" ]; then
        echo "[$TS] [WARN] demo_trading 프로세스 없음 — 종료됨?" >> "$SESSION_LOG"
        sleep 60
        continue
    fi

    # JSONL에서 최신 상태 파싱
    LAST=$(tail -1 "$JSONL" 2>/dev/null)
    TRADES=$(echo "$LAST" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('trades','-'))" 2>/dev/null)
    EQUITY=$(echo "$LAST" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(f\"{d.get('equity',0):.2f}\")" 2>/dev/null)
    DOT_P=$(echo "$LAST"  | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(f\"{d.get('prices',{}).get('DOT',0):.4f}\")" 2>/dev/null)
    ADA_P=$(echo "$LAST"  | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(f\"{d.get('prices',{}).get('ADA',0):.4f}\")" 2>/dev/null)
    OPEN=$(echo "$LAST"   | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('open_positions','-'))" 2>/dev/null)

    # 최근 이벤트 (신호/거래)
    RECENT=$(tail -5 "$DEMO_LOG" 2>/dev/null | grep -E "학습 완료|신호 없음|S1 미달|LONG|SHORT|OPEN|CLOSE|TRADE|새 4h바" | tail -3)

    echo "[$TS] PID=$PID | equity=$EQUITY USDT | trades=$TRADES | open=$OPEN | DOT=$DOT_P ADA=$ADA_P" >> "$SESSION_LOG"
    if [ -n "$RECENT" ]; then
        echo "$RECENT" | while read line; do echo "    >> $line" >> "$SESSION_LOG"; done
    fi

    sleep 300  # 5분마다
done
