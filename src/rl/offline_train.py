"""Offline training: signal_log.jsonl -> trained LinUCB.

Usage:
    python -m src.rl.offline_train
    python -m src.rl.offline_train --alpha 1.5 --gamma 0.99
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.rl.bandit import LinUCB, ACTION_NAMES, SIZING_MAP
from src.rl.state_builder import STATE_DIM, STATE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("offline_train")

DEFAULT_LOG = Path("data/reports/live_trading_v2/signal_log.jsonl")
DEFAULT_MODEL = Path("data/models/rl/linucb_v1.joblib")


def load_training_data(log_path: Path) -> list[dict]:
    """Load signal log and pair signals with results.

    Returns list of {state, action, reward, coin, ts} dicts.
    """
    signals = {}  # ts -> signal record
    results = {}  # original_ts -> result record

    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "state" in rec and "action" in rec:
                signals[rec["ts"]] = rec
            elif "original_ts" in rec and "resolved" in rec:
                results[rec["original_ts"]] = rec

    # Pair signals with results
    training_data = []
    for ts, sig in signals.items():
        res = results.get(ts)
        if res is None or not res.get("resolved"):
            continue

        # Skip: RL accepted but downstream risk gate rejected
        if sig["action"] > 0 and not sig.get("executed", True):
            continue

        state = np.array(sig["state"], dtype=np.float64)
        action = sig["action"]
        reward = res["result_reward"]

        training_data.append({
            "state": state,
            "action": action,
            "reward": reward,
            "coin": sig["coin"],
            "ts": ts,
        })

    return training_data


def train(
    data: list[dict],
    alpha: float = 1.0,
    gamma: float = 0.995,
) -> LinUCB:
    """Train LinUCB on paired (state, action, reward) data."""
    bandit = LinUCB(state_dim=STATE_DIM, n_actions=len(ACTION_NAMES),
                    alpha=alpha, gamma=gamma)

    # Sort by timestamp (chronological order matters for forgetting)
    data.sort(key=lambda d: d["ts"])

    for d in data:
        bandit.update(d["state"], d["action"], d["reward"])

    return bandit


def evaluate(bandit: LinUCB, data: list[dict]) -> dict:
    """Evaluate bandit on held-out data."""
    if not data:
        return {}

    rewards_by_action = {a: [] for a in range(bandit.K)}
    correct_accepts = 0
    correct_rejects = 0
    total = 0

    for d in data:
        total += 1
        actual_action = d["action"]
        actual_reward = d["reward"]
        predicted_action = bandit.select_action(d["state"])
        rewards_by_action[actual_action].append(actual_reward)

        # Was the action correct?
        if actual_action == 0 and actual_reward > 0:
            correct_rejects += 1
        elif actual_action > 0 and actual_reward > 0:
            correct_accepts += 1

    return {
        "total_samples": total,
        "per_action": {
            ACTION_NAMES[a]: {
                "count": len(r),
                "mean_reward": round(np.mean(r), 4) if r else 0,
                "std_reward": round(np.std(r), 4) if r else 0,
            }
            for a, r in rewards_by_action.items()
        },
        "correct_accepts": correct_accepts,
        "correct_rejects": correct_rejects,
    }


def main():
    parser = argparse.ArgumentParser(description="Offline LinUCB training")
    parser.add_argument("--input", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.995)
    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"Signal log not found: {args.input}")
        sys.exit(1)

    # Load data
    data = load_training_data(args.input)
    logger.info(f"Loaded {len(data)} resolved signals from {args.input}")

    if len(data) < 50:
        logger.warning(f"Only {len(data)} samples — consider waiting for more data")

    # Split: 80% train, 20% eval (chronological)
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    eval_data = data[split_idx:]

    # Train
    bandit = train(train_data, alpha=args.alpha, gamma=args.gamma)
    logger.info(f"Trained on {len(train_data)} samples")
    logger.info(f"Updates per action: {bandit.n_updates}")

    # Evaluate
    if eval_data:
        metrics = evaluate(bandit, eval_data)
        logger.info(f"Eval ({len(eval_data)} samples):")
        for action_name, stats in metrics["per_action"].items():
            logger.info(f"  {action_name}: n={stats['count']} mean={stats['mean_reward']:+.4f}")

    # Feature importance
    for a in range(bandit.K):
        if bandit.n_updates[a] > 0:
            fi = bandit.feature_importance(a)
            top5 = sorted(fi.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            logger.info(f"  {ACTION_NAMES[a]} top features: {top5}")

    # Save
    bandit.save(args.output)
    logger.info(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
