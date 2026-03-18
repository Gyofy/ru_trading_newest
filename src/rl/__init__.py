"""RL Meta-Strategy Layer (v4.3).

LinUCB Contextual Bandit that learns when to trust ML signals.
Sits between predict_2stage() and pre_trade_gate().

Modules:
  state_builder   -- build 30-dim state vector from pipeline data
  signal_logger   -- log every signal + counterfactual for offline learning
  bandit          -- LinUCB with Sherman-Morrison, forgetting, intercept
  rl_gate         -- decision wrapper with warmup + v4.2 fallback
  counterfactual  -- estimate PnL of rejected signals
  offline_train   -- CLI: signal_log.jsonl -> trained LinUCB
"""
