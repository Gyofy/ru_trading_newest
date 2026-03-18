#!/usr/bin/env python3
"""Discord Webhook 리포트 — live_trading_v2/events.jsonl 모니터링"""
import json, os, time, requests
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

WEBHOOK_URL = os.environ.get(
    "DISCORD_WEBHOOK_URL",
    "https://discord.com/api/webhooks/1483779094859481108/cUiWiOPBJQfQXRiRIPBOJ0EoGOgOIVxhzzEexFHxRxF1YkZQ6iHXIlKcNUxa3hif4fcq",
)
EVENTS_FILE  = Path("/home/wlsry/Ru_trading/data/reports/live_trading_v2/events.jsonl")
POSITIONS_FILE = Path("/home/wlsry/Ru_trading/data/reports/live_trading_v2/positions.json")

VERSION = "CLAUDE_CRYPTO_AGENT v4.3"

# ── 상태 ──────────────────────────────────────────────────────
state = {
    "equity": 0.0,
    "cycle": 0,
    "positions": {},       # coin → {side, entry, qty, sl, tp}
    "cycle_models": {},    # coin → cv_score  (현 사이클)
    "cycle_signals": [],   # [{coin, side, p_trade, result}]
    "total_trades": 0,
    "total_pnl": 0.0,
    "wins": 0,
    "losses": 0,
}

# ── 전송 ──────────────────────────────────────────────────────
def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def send_embed(embeds: list):
    try:
        requests.post(WEBHOOK_URL, json={"embeds": embeds}, timeout=8)
    except Exception as e:
        print(f"[Discord] 전송 실패: {e}")

def footer():
    return {"text": f"{VERSION} | {_now_utc()}"}

# ── 리포트 빌더 ───────────────────────────────────────────────

def report_bot_started(d: dict):
    mode = d.get("mode", "").upper()
    equity = d.get("equity", 0)
    coins = d.get("coins", [])
    color = 0x3498db

    embed = {
        "title": f"🚀 봇 시작 — {mode} 모드",
        "color": color,
        "fields": [
            {"name": "잔고", "value": f"**{equity:.2f} USDT**", "inline": True},
            {"name": "모드", "value": mode, "inline": True},
            {"name": "버전", "value": VERSION, "inline": True},
            {"name": "거래 코인", "value": " · ".join(coins), "inline": False},
        ],
        "footer": footer(),
    }
    send_embed([embed])


def report_position_opened(d: dict):
    coin   = d.get("coin", "")
    side   = d.get("side", "BUY")
    entry  = d.get("entry", 0)
    qty    = d.get("qty", 0)
    sl     = d.get("sl", 0)
    tp     = d.get("tp", 0)
    p_trade = d.get("p_trade", 0)
    p_dir   = d.get("p_dir", 0)
    mode   = d.get("mode", "live").upper()

    notional = entry * qty
    sl_pct  = abs(entry - sl) / entry * 100
    tp_pct  = abs(tp - entry) / entry * 100
    rr      = tp_pct / sl_pct if sl_pct > 0 else 0

    side_label = "🟢 LONG" if side == "BUY" else "🔴 SHORT"
    color = 0x2ecc71 if side == "BUY" else 0xe74c3c

    # 포지션 상태 저장
    state["positions"][coin] = {"side": side, "entry": entry, "qty": qty, "sl": sl, "tp": tp}

    embed = {
        "title": f"📋 진입 리포트 — {coin} {side_label}",
        "color": color,
        "fields": [
            {"name": "진입가", "value": f"**${entry:.4f}**", "inline": True},
            {"name": "수량",   "value": f"{qty} ({notional:.2f} USDT)", "inline": True},
            {"name": "모드",   "value": mode, "inline": True},
            {"name": "SL",     "value": f"${sl:.4f}  (-{sl_pct:.2f}%)", "inline": True},
            {"name": "TP",     "value": f"${tp:.4f}  (+{tp_pct:.2f}%)", "inline": True},
            {"name": "R:R",    "value": f"1 : {rr:.2f}", "inline": True},
            {"name": "S1 확률 (거래여부)", "value": f"{p_trade*100:.1f}%", "inline": True},
            {"name": "S2 확률 (방향)",     "value": f"{p_dir*100:.1f}%",   "inline": True},
        ],
        "footer": footer(),
    }
    send_embed([embed])


def report_position_closed(d: dict):
    coin    = d.get("coin", "")
    reason  = d.get("reason", "")
    side    = d.get("side", "")
    entry   = d.get("entry", 0)
    exit_p  = d.get("exit", 0)
    pnl     = d.get("pnl_usdt", 0)
    pnl_pct = d.get("pnl_pct", 0) * 100
    bars    = d.get("bars", 0)

    state["total_trades"] += 1
    state["total_pnl"]    += pnl
    if pnl >= 0:
        state["wins"] += 1
    else:
        state["losses"] += 1
    state["positions"].pop(coin, None)

    win_rate = state["wins"] / state["total_trades"] * 100 if state["total_trades"] else 0

    reason_map = {"tp": "🎯 TP 도달", "sl": "🛑 SL 도달", "ttl": "⏱ TTL 만료", "manual": "🖐 수동"}
    reason_label = reason_map.get(reason, f"❓ {reason}")

    result_emoji = "✅ 수익" if pnl >= 0 else "❌ 손실"
    color = 0x2ecc71 if pnl >= 0 else 0xe74c3c
    side_label = "LONG" if side == "BUY" else "SHORT"

    embed = {
        "title": f"📋 청산 리포트 — {coin} {side_label}",
        "color": color,
        "fields": [
            {"name": "청산 사유",  "value": reason_label, "inline": True},
            {"name": "결과",       "value": result_emoji,  "inline": True},
            {"name": "보유기간",   "value": f"{bars}봉 ({bars*4}h)", "inline": True},
            {"name": "진입가",     "value": f"${entry:.4f}",  "inline": True},
            {"name": "청산가",     "value": f"${exit_p:.4f}", "inline": True},
            {"name": "PnL",        "value": f"**{pnl:+.4f} USDT** ({pnl_pct:+.2f}%)", "inline": True},
            {"name": "누적 거래",  "value": f"{state['total_trades']}건", "inline": True},
            {"name": "승률",       "value": f"{win_rate:.0f}% ({state['wins']}W/{state['losses']}L)", "inline": True},
            {"name": "누적 PnL",   "value": f"{state['total_pnl']:+.4f} USDT", "inline": True},
        ],
        "footer": footer(),
    }
    send_embed([embed])


def report_cycle_complete(d: dict):
    cycle   = d.get("cycle", 0)
    elapsed = d.get("elapsed_s", 0)
    equity  = d.get("equity", state["equity"])
    open_coins = d.get("positions", [])

    state["cycle"]  = cycle
    state["equity"] = equity

    # 모델 CV 스코어 정리
    model_lines = []
    for coin, score in sorted(state["cycle_models"].items()):
        bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
        model_lines.append(f"`{coin:<4}` {bar} {score:.3f}")
    models_str = "\n".join(model_lines) if model_lines else "—"

    # 현재 오픈 포지션
    if open_coins:
        pos_lines = []
        for coin in open_coins:
            p = state["positions"].get(coin, {})
            side_label = "🟢 L" if p.get("side") == "BUY" else "🔴 S"
            pos_lines.append(f"{side_label} **{coin}** @ ${p.get('entry', 0):.4f}")
        pos_str = "\n".join(pos_lines)
    else:
        pos_str = "없음"

    # 당 사이클 신호 요약
    if state["cycle_signals"]:
        sig_lines = []
        for s in state["cycle_signals"]:
            icon = "✅" if s.get("entered") else "⚪"
            sig_lines.append(f"{icon} {s['coin']} {s['side']} (p={s['p_trade']:.2f})")
        sig_str = "\n".join(sig_lines)
    else:
        sig_str = "신호 없음"

    win_rate = state["wins"] / state["total_trades"] * 100 if state["total_trades"] else 0

    color = 0x5865f2  # Discord blurple

    embed = {
        "title": f"📊 사이클 #{cycle} 리포트",
        "color": color,
        "fields": [
            {"name": "잔고",          "value": f"**{equity:.2f} USDT**", "inline": True},
            {"name": "오픈 포지션",   "value": f"{len(open_coins)}개",   "inline": True},
            {"name": "소요시간",      "value": f"{elapsed:.1f}s",        "inline": True},
            {"name": "─── 모델 CV 스코어 ───", "value": models_str, "inline": False},
            {"name": "─── 이번 사이클 신호 ───", "value": sig_str, "inline": False},
            {"name": "─── 포지션 현황 ───", "value": pos_str, "inline": False},
            {"name": "누적 거래",     "value": f"{state['total_trades']}건", "inline": True},
            {"name": "승률",          "value": f"{win_rate:.0f}%",        "inline": True},
            {"name": "누적 PnL",      "value": f"{state['total_pnl']:+.4f} USDT", "inline": True},
        ],
        "footer": footer(),
    }
    send_embed([embed])

    # 다음 사이클을 위해 초기화
    state["cycle_models"]  = {}
    state["cycle_signals"] = []


def report_kill_switch(d: dict):
    embed = {
        "title": "⛔ 킬스위치 발동",
        "description": f"**사유:** {d.get('reason', '알 수 없음')}\n\n즉시 확인이 필요합니다.",
        "color": 0xe74c3c,
        "footer": footer(),
    }
    send_embed([embed])


def report_error(d: dict):
    msg = str(d.get("message", ""))[:800]
    embed = {
        "title": "⚠️ 에러 발생",
        "description": f"```\n{msg}\n```",
        "color": 0xe67e22,
        "footer": footer(),
    }
    send_embed([embed])


# ── 이벤트 처리 ───────────────────────────────────────────────

def handle_event(d: dict):
    ev = d.get("event", "")

    if ev == "bot_started":
        state["equity"] = d.get("equity", 0)
        report_bot_started(d)

    elif ev == "model_trained":
        state["cycle_models"][d.get("coin", "")] = d.get("cv_score", 0)

    elif ev == "position_opened":
        # 신호 기록
        state["cycle_signals"].append({
            "coin": d.get("coin"), "side": d.get("side"),
            "p_trade": d.get("p_trade", 0), "entered": True,
        })
        report_position_opened(d)

    elif ev == "partial_tp":
        stage   = d.get("stage", 0)
        coin    = d.get("coin", "")
        pnl     = d.get("pnl_usdt", 0)
        pnl_pct = d.get("pnl_pct", 0) * 100
        rem     = d.get("remaining_qty", 0)
        new_sl  = d.get("new_sl", 0)
        stage_label = {1: "TP1 (33%)", 2: "TP2 (33%)", 3: "TP3 (잔여)"}.get(stage, f"TP{stage}")
        sl_label = "BEP(손익분기)" if stage == 1 else "TP1 가격"
        embed = {
            "title": f"🎯 분할 익절 — {coin} {stage_label}",
            "color": 0x2ecc71,
            "fields": [
                {"name": "청산가",     "value": f"${d.get('exit', 0):.4f}", "inline": True},
                {"name": "PnL",        "value": f"**{pnl:+.4f} USDT** ({pnl_pct:+.2f}%)", "inline": True},
                {"name": "잔여 수량",  "value": f"{rem}", "inline": True},
                {"name": f"SL → {sl_label}", "value": f"${new_sl:.4f}", "inline": False},
            ],
            "footer": footer(),
        }
        send_embed([embed])
        return

    elif ev == "position_closed":
        report_position_closed(d)

    elif ev == "cycle_complete":
        # 사이클 상태만 업데이트, Discord 전송 안 함
        state["cycle"]  = d.get("cycle", state["cycle"])
        state["equity"] = d.get("equity", state["equity"])
        state["cycle_models"]  = {}
        state["cycle_signals"] = []

    elif ev == "kill_switch":
        report_kill_switch(d)

    elif ev == "error":
        report_error(d)


# ── 메인 루프 ─────────────────────────────────────────────────

def main():
    print(f"[Discord Notifier] 시작 — {EVENTS_FILE}")

    # (시작 알림 없음 — 매수/매도 시에만 전송)

    # 기존 파일 끝부터 읽기
    file_pos = EVENTS_FILE.stat().st_size if EVENTS_FILE.exists() else 0

    while True:
        try:
            if EVENTS_FILE.exists():
                with open(EVENTS_FILE, "r") as f:
                    f.seek(file_pos)
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            d = json.loads(line)
                            handle_event(d)
                            print(f"[Discord] 처리: {d.get('event')} {d.get('coin','')}")
                        except json.JSONDecodeError:
                            pass
                    file_pos = f.tell()
        except Exception as e:
            print(f"[Discord] 오류: {e}")

        time.sleep(5)


if __name__ == "__main__":
    main()
