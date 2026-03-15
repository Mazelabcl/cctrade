# Roadmap

## ✅ Phase 1 — Foundation (DONE)
- SQLite DB + SQLAlchemy models
- Binance data fetcher (incremental)
- Fractal detection, HTF levels, Fibonacci, Volume Profile
- Touch tracking with temporal integrity
- Chart Champions color scheme (CC golden pocket, Igor quarters)

## ✅ Phase 2 — UI & Charts (DONE)
- TradingView Lightweight Charts with level overlays
- Chart fix: recent levels prioritized over oldest
- Level filters (type, source TF, naked/touched)
- Foundation Pipeline wizard (4-step: fetch → detect → touch → visualize)
- Settings UI with per-step progress tracking

## 🔄 Phase 3 — Data Completeness (IN PROGRESS)
- [x] 1d/1w/1M candles from 2020
- [x] 1h/4h candles from 2024
- [ ] 1h/4h candles from 2020 (fetching now)
- [ ] VP optimization: store 1m candles in DB (incremental on re-run)
- [ ] VWAP levels: session, weekly, anchored from fractals

## 📋 Phase 4 — Feature Engineering (NEXT)
- Distance features: how far is price from each level type?
- Confluence score: how many levels cluster near price?
- Level quality features: naked/touched, age, touch count
- VWAP interaction features
- VP position: above/below POC, in/out of value area
- HTF bias: weekly/monthly trend direction
- Candle structure: body%, wick ratio, volume vs average

## 📋 Phase 5 — Backtest Improvement
- Level Quality Score backtest: fractal-at-level hit rate per type
- Output: reliability score per level type → feeds into feature weights
- Review and align individual_level_backtest.py with fractal prediction goal
- Walk-forward validation to avoid lookahead bias

## 📋 Phase 6 — ML Model v2
- Feature importance analysis (which levels matter most?)
- Model comparison: RandomForest vs XGBoost vs LightGBM
- Handle class imbalance (SMOTE / class weights)
- Hyperparameter tuning
- Precision/recall optimization (tune threshold for false signal reduction)

## 📋 Phase 7 — Live Signal Generation
- Real-time BTCUSDT monitoring on 1h/4h/1d
- Predict fractal probability on each new candle close
- Signal dashboard: LONG/SHORT/FLAT with confidence
- Alert system (webhook / Telegram)

## 📋 Phase 8 — Paper Trading & Validation
- Paper trade signals with defined SL/TP rules
- Track live performance vs backtested performance
- Refine signal thresholds based on live results

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Daily candles as primary ML timeframe | Enough data, less noise than 1h/4h |
| 2020–2024-06 as training set | Covers full bull/bear cycle |
| 2024-06 onward as test | Post-halving regime, truly unseen |
| Fractals as target | Observable, objective, tradable |
| Naked levels weighted higher | Chart Champions: untested levels react stronger |
| No lookahead in features | All features use candle N-1 data to predict N |
