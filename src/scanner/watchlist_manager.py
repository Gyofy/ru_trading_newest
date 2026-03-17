"""Watchlist Manager - 사람이 선택한 코인 관리.

Universe Scan 리포트를 보고 사람이 2~3개 코인을 선택하면,
이 모듈이 watchlist.json을 관리한다.

흐름:
  1. run_universe_scan.py → 리포트 생성
  2. 사람이 리포트 보고 코인 선택
  3. watchlist_manager.py → watchlist.json 업데이트
  4. (나중에) auto_trader가 watchlist.json의 활성 코인만 모니터링
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

WATCHLIST_PATH = Path("config/watchlist.json")
MAX_ACTIVE = 3  # 동시 활성 코인 수 상한


def _load() -> dict:
    """watchlist.json 로드. 없으면 기본 구조 반환."""
    if WATCHLIST_PATH.exists():
        with open(WATCHLIST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "active": [],
        "history": [],
        "updated_at": None,
    }


def _save(data: dict) -> Path:
    """watchlist.json 저장."""
    WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = datetime.now().isoformat()
    with open(WATCHLIST_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return WATCHLIST_PATH


def get_active() -> list[dict]:
    """현재 활성 코인 목록 반환."""
    return _load().get("active", [])


def get_active_symbols() -> list[str]:
    """활성 코인 심볼만 반환 (예: ["SOL", "AAVE"])."""
    return [c["coin"] for c in get_active()]


def activate(coins: list[str], reason: str = "") -> dict:
    """코인을 watchlist에 활성화.

    Args:
        coins: 활성화할 코인 목록 (예: ["SOL", "AAVE", "FIL"])
        reason: 선택 이유 (리포트 기반 메모)

    Returns:
        업데이트된 watchlist
    """
    coins = [c.upper() for c in coins]

    if len(coins) > MAX_ACTIVE:
        raise ValueError(f"최대 {MAX_ACTIVE}개까지만 활성화 가능 (요청: {len(coins)}개)")

    data = _load()
    now = datetime.now().isoformat()

    # 기존 활성 코인 → history로 이동
    for old in data["active"]:
        old["deactivated_at"] = now
        old["deactivated_reason"] = "replaced"
        data["history"].append(old)

    # 새 활성 코인 설정
    data["active"] = [
        {
            "coin": c,
            "activated_at": now,
            "reason": reason,
        }
        for c in coins
    ]

    # history 최근 50개만 유지
    data["history"] = data["history"][-50:]

    _save(data)
    print(f"[Watchlist] 활성화: {', '.join(coins)}")
    return data


def deactivate(coin: str, reason: str = "") -> dict:
    """특정 코인 비활성화."""
    coin = coin.upper()
    data = _load()
    now = datetime.now().isoformat()

    new_active = []
    for c in data["active"]:
        if c["coin"] == coin:
            c["deactivated_at"] = now
            c["deactivated_reason"] = reason
            data["history"].append(c)
        else:
            new_active.append(c)

    data["active"] = new_active
    data["history"] = data["history"][-50:]

    _save(data)
    print(f"[Watchlist] 비활성화: {coin}")
    return data


def show() -> str:
    """현재 watchlist 상태를 문자열로 반환."""
    data = _load()
    lines = ["=== Watchlist Status ==="]

    if not data["active"]:
        lines.append("(활성 코인 없음)")
    else:
        for c in data["active"]:
            lines.append(f"  [{c['coin']}] since {c['activated_at'][:16]}")
            if c.get("reason"):
                lines.append(f"    reason: {c['reason']}")

    if data.get("updated_at"):
        lines.append(f"\nLast updated: {data['updated_at'][:16]}")

    # 최근 history 5개
    recent = data.get("history", [])[-5:]
    if recent:
        lines.append("\nRecent history:")
        for h in reversed(recent):
            lines.append(f"  {h['coin']} | {h.get('activated_at', '?')[:10]} -> {h.get('deactivated_at', '?')[:10]} | {h.get('deactivated_reason', '')}")

    return "\n".join(lines)


# === CLI 인터페이스 ===
if __name__ == "__main__":
    import sys

    args = sys.argv[1:]

    if not args or args[0] == "show":
        print(show())

    elif args[0] == "activate" and len(args) >= 2:
        coins = [a.upper() for a in args[1:] if not a.startswith("--")]
        reason_parts = [a for a in args[1:] if a.startswith("--reason=")]
        reason = reason_parts[0].split("=", 1)[1] if reason_parts else ""
        activate(coins, reason)
        print(show())

    elif args[0] == "deactivate" and len(args) >= 2:
        deactivate(args[1])
        print(show())

    else:
        print("Usage:")
        print("  python watchlist_manager.py show")
        print("  python watchlist_manager.py activate SOL AAVE FIL --reason='Top 3 from report'")
        print("  python watchlist_manager.py deactivate FIL")
