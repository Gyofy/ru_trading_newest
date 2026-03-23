# Binance Public Data — 수집 스펙

현재 파이프라인이 OHLCV만으로 합성(BVC)하고 있는 CVD/OFI/VPIN을
진짜 체결 데이터로 대체하기 위한 수집 계획.

소스: https://github.com/binance/binance-public-data
경로: `data/futures/um/daily/{type}/{SYMBOL}/`

---

## Tier 1: 핵심 (이것만 있어도 판이 달라짐)

### 1. Aggregate Trades (aggTrades)
- **용도**: 진짜 CVD/OFI/VPIN 계산 (현재 BVC 근사 → 정확도 +30~40%)
- **코인**: BTC, ETH, SOL, XRP, ADA, DOT, LINK (7개)
- **기간**: 최소 180일, 이상적 365일
- **해상도**: 개별 체결(tick-level) 또는 1분 집계
- **경로**: `data/futures/um/daily/aggTrades/{SYMBOL}/`

| 컬럼 | 설명 |
|------|------|
| timestamp | ms epoch |
| price | 체결가 |
| quantity | 수량 |
| is_buyer_maker | true=매도체결(taker sell), false=매수체결(taker buy) |

### 2. Funding Rate
- **용도**: 포지션 편향 감지 + 비용 정밀 계산 + 이벤트 시그널
- **코인**: 동일 7개
- **기간**: 365일
- **해상도**: 8시간 (하루 3회)
- **경로**: `data/futures/um/daily/fundingRate/{SYMBOL}/`

| 컬럼 | 설명 |
|------|------|
| calc_time | 정산 시각 |
| symbol | BTCUSDT 등 |
| funding_rate | float (양수=롱 지불, 음수=숏 지불) |
| mark_price | 마크 가격 |

### 3. Open Interest (openInterestHist)
- **용도**: 포지셔닝 과열/청산 리스크 감지
- **코인**: 동일 7개
- **기간**: 365일
- **해상도**: 5분 (→ 1h/4h 리샘플)
- **경로**: `data/futures/um/daily/openInterestHist/{SYMBOL}/`

| 컬럼 | 설명 |
|------|------|
| timestamp | ms epoch |
| symbol | BTCUSDT 등 |
| sum_open_interest | 총 OI (계약 수) |
| sum_open_interest_value | 총 OI (USDT 환산) |

---

## Tier 2: 중요 (전략 다양화에 필요)

### 4. Orderbook Snapshot (bookDepth)
- **용도**: 실제 bid-ask 스프레드, depth imbalance, 유동성 프로파일
- **코인**: BTC, ETH, SOL (유동성 상위 3개 우선)
- **기간**: 90일 (용량 큼)
- **해상도**: 100ms~1s 스냅샷 (또는 10분 집계)
- **경로**: `data/futures/um/daily/bookDepth/{SYMBOL}/` (top5/10/20레벨)
- **주의**: 용량이 매우 큼 (코인당 하루 수백MB~수GB). Top-5 레벨만 수집 권장.

| 컬럼 | 설명 |
|------|------|
| timestamp | ms epoch |
| bids | [[price, qty], ...] 상위 5~20레벨 |
| asks | [[price, qty], ...] 상위 5~20레벨 |

### 5. Liquidation Data (forceOrders)
- **용도**: 청산 캐스케이드 감지, 극단 이벤트 시그널
- **코인**: 동일 7개
- **기간**: 180일
- **해상도**: 개별 이벤트
- **경로**: `data/futures/um/daily/forceOrders/{SYMBOL}/`

| 컬럼 | 설명 |
|------|------|
| timestamp | ms epoch |
| symbol | BTCUSDT |
| side | LONG/SHORT (청산당한 쪽) |
| quantity | 청산 수량 |
| price | 청산가 |

### 6. Long/Short Ratio (globalLongShortAccountRatio)
- **용도**: 리테일 vs 고래 포지션 편향
- **코인**: 동일 7개
- **기간**: 365일
- **해상도**: 5분 (→ 4h 리샘플)
- **경로**: `data/futures/um/daily/globalLongShortAccountRatio/{SYMBOL}/`

| 컬럼 | 설명 |
|------|------|
| timestamp | ms epoch |
| symbol | BTCUSDT |
| long_account | 롱 계정 비율 |
| short_account | 숏 계정 비율 |
| long_short_ratio | 비율 |

---

## Tier 3: 보조 (있으면 좋음)

### 7. Taker Buy/Sell Volume (takerlongshortRatio)
- **용도**: 시장가 매수/매도 비율 (aggTrades 대안, 경량)
- **해상도**: 5분
- **경로**: `data/futures/um/daily/takerlongshortRatio/{SYMBOL}/`

### 8. OHLCV 고해상도 (1분봉)
- **용도**: 1m/5m 해상도 전략, 인트라바 변동성 분석
- **코인**: BTC, ETH, SOL
- **기간**: 90일
- **경로**: `data/futures/um/daily/klines/{SYMBOL}/1m/`

---

## 다운로드 용량 추정

| 데이터 | 코인 수 | 기간 | 예상 용량 (압축) |
|--------|---------|------|-----------------|
| AggTrades | 7 | 365일 | ~50-100GB |
| Funding Rate | 7 | 365일 | ~10MB |
| Open Interest | 7 | 365일 | ~500MB |
| Book Depth (top5) | 3 | 90일 | ~30-50GB |
| Liquidation | 7 | 180일 | ~100MB |
| Long/Short Ratio | 7 | 365일 | ~200MB |
| Taker Volume | 7 | 365일 | ~200MB |
| 1m Klines | 3 | 90일 | ~2GB |

---

## 최소 시작 세트 (권장)

디스크/시간 제약 고려 시 우선순위:

| 우선순위 | 데이터 | 용량 | 임팩트 |
|---------|--------|------|--------|
| 1 | Funding Rate (7코인, 365일) | ~10MB | 비용모델 정밀화 + 이벤트 시그널 |
| 2 | Open Interest (7코인, 365일) | ~500MB | 포지셔닝 과열 감지 |
| 3 | Long/Short Ratio (7코인, 365일) | ~200MB | 리테일 편향 시그널 |
| 4 | AggTrades (BTC만, 180일) | ~15GB | 진짜 CVD/OFI 검증 |
| 5 | Liquidation (7코인, 180일) | ~100MB | 캐스케이드 시그널 |

1~3번 합계 ~700MB — 바로 수집 가능한 크기.
이것만으로 현재 OHLCV-only 파이프라인에 3종류의 새 정보원이 추가됨.

---

## 다운로드 스크립트

```bash
# 최소 세트 다운로드
python src/data/crawlers/binance_public_data_downloader.py \
  --types funding_rate open_interest long_short_ratio \
  --symbols BTCUSDT ETHUSDT SOLUSDT XRPUSDT ADAUSDT DOTUSDT LINKUSDT \
  --days 365 \
  --output data/raw/binance_public

# AggTrades (BTC, 180일) — 별도 실행 권장
python src/data/crawlers/binance_public_data_downloader.py \
  --types agg_trades \
  --symbols BTCUSDT \
  --days 180 \
  --output data/raw/binance_public
```
