"""Execution Layer — 실전 매매 실행 계층.

Components:
  - OrderLedger: SQLite 주문/체결 영속화
  - PositionStateMachine: 심볼별 상태 머신
  - RiskEngine: Stop-distance 사이징 + 리스크 게이트
  - ExchangeAdapter: Binance USDT-M Futures ccxt 인터페이스
  - FeatureStore: 학습 동일 피처 파이프라인
  - LiveEngine: 메인 오케스트레이터
  - Watchdog: 독립 감시 프로세스
"""
