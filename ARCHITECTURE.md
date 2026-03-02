# Options Trader — Complete Architecture Reference

> **Last updated:** 2026-03-02  
> **Total files:** ~55 Python files | **~25,500 lines**  
> **Runtime:** Python 3.x + Conda (`options-trader` env)

---

## System Overview

An ML-driven options trading bot for Indian markets (NSE/BSE) via Zerodha Kite Connect. Runs in two modes:
- **Bot mode** (`python run.py --bot --paper`) — Automated scanning, signal generation, order execution
- **CLI mode** (`python run.py --cli`) — Interactive command shell for manual analysis and trading

### Core Data Flow (Live Trading)

```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐     ┌───────────────┐
│ Kite Connect │────▶│ DataFetcher  │────▶│ ML Predictor   │────▶│ ML Signal Gen │
│   (Auth)     │     │ (spot, chain,│     │ (XGB ensemble, │     │ (direction →  │
│              │     │ OI, hist)    │     │  ternary pred) │     │  strategy →   │
└─────────────┘     └──────────────┘     └────────────────┘     │  option legs) │
       │                   │                                     └───────┬───────┘
       │            ┌──────────────┐                                     │
       │            │ WebSocket    │◀─── real-time ticks                 ▼
       │            │ Manager      │                             ┌───────────────┐
       │            └──────┬───────┘                             │ Order Manager │
       │                   │                                     │ (paper/live)  │
       │                   ▼                                     └───────┬───────┘
       │            ┌──────────────┐                                     │
       │            │  Position    │◀────────────────────────────────────┘
       │            │  Tracker     │──── SL/target/trailing SL checks
       │            │  (WS + poll) │──── signal-based intelligent exits
       │            └──────┬───────┘
       │                   │
       │                   ▼
       │            ┌──────────────┐     ┌───────────────┐
       │            │  Database    │     │ Notifications │
       │            │  (SQLite)    │     │ (Telegram/WA) │
       │            └──────────────┘     └───────────────┘
       │                   ▲
       │            ┌──────────────┐
       └───────────▶│  Feedback    │──── drift detection → auto-retrain
                    │  Collector   │
                    └──────────────┘
```

---

## File-by-File Reference

### 1. Entry Points

| File | Lines | Purpose |
|------|-------|---------|
| `run.py` | 46 | Argparse entry: `--bot` / `--paper` / `--cli` |
| `bot.py` | 901 | **Main orchestrator** — login → scan loop → signals → orders → monitoring |
| `cli.py` | 2,498 | Interactive `cmd.Cmd` shell: market analysis, manual trading, ML training |
| `demo_signal.py` | 149 | Demo script showing paper trade signal format |

#### `bot.py` — Key Methods
- `start()` / `stop()` — lifecycle management
- `_run_loop()` → `_scan_and_trade()` every `signal_scan_interval` (300s)
- `_meets_auto_trade_criteria()` — strategy-specific entry gates:
  - Directional: RR ≥ 1.5 (live) / 0.5 (paper)
  - Credit spreads: confidence ≥ 75% (live) / 52% (paper)
- Registers callbacks: `sl_hit`, `trailing_sl_hit`, `target_hit`, `position_closed`, `signal_exit`
- Starts background threads: auto-retrain monitor, live feature collection

---

### 2. `auth/` — Authentication (~378 lines)

| File | Lines | Purpose |
|------|-------|---------|
| `kite_auth.py` | 156 | Kite Connect session management, token persistence (`data/session.json`) |
| `auto_login.py` | 207 | Headless Selenium + TOTP auto-login to Zerodha |

**Key functions:** `connect()`, `get_kite()`, `is_authenticated()`, `logout()`  
**Credentials:** `.env` → `KITE_API_KEY`, `KITE_API_SECRET`, `KITE_USER_ID`, `KITE_PASSWORD`, `TOTP_SECRET`

---

### 3. `config/` — Configuration (~464 lines)

#### `settings.py` — All Constants

| Config Block | Key Values |
|---|---|
| `KITE_CONFIG` | API key/secret from `.env` |
| `TRADING_CONFIG` | `max_positions=5`, `paper_max_positions=15`, `capital_per_trade=₹1.5L`, `max_loss_per_day=₹10K`, SL=30%, target=50%, trailing SL enabled |
| `MARKET_HOURS` | `trading_start=10:00`, `trading_end=15:00`, `carry_overnight=True`, `auto_square_off=False`, skip first 105min (until 11:00) |
| `BOT_CONFIG` | `signal_scan_interval=300s`, `position_poll_interval=5s` (batched), `signal_exit_enabled=True`, `use_websocket=True` |
| `GREEKS_EXIT_CONFIG` | Delta/theta/vega/gamma exit thresholds, DTE exit at 2 days |
| `UNDERLYING_ASSETS` | NIFTY (lot=25, 50pt), BANKNIFTY (lot=15, 100pt), FINNIFTY (lot=25, 50pt), SENSEX (lot=10, 100pt, BFO) |
| `STRATEGY_CONFIG` | 10 enabled strategies, `otm_offset=1` |
| `LIQUIDITY_GUARD` | Min volume (1000 idx / 500 stock), min OI (5000 idx / 1000 stock), max spread 5% |
| `EVENT_REGIME_CONFIG` | Risk-off breadth + intraday shock + put/call volume spike scoring, with configurable `flip_mode`, hysteresis thresholds, and confidence penalty |
| `ML_CONFIG` | `training_symbols`: NIFTY, BANKNIFTY, SENSEX, AXISBANK, HDFCBANK, RELIANCE, SBIN; `model_type=ensemble`, `optuna_trials=50`, guardrails, feedback, auto-retrain configs |

**Utility functions:** `get_lot_size()`, `get_options_exchange()`, `get_strike_interval()`, `get_instrument_token()`, `get_asset_by_name()`

#### `watchlist.json` — Stock Universe
Defines stock-specific lot sizes, instrument tokens, equity tokens for historical data.

---

### 4. `core/` — Infrastructure (~2,785 lines)

| File | Lines | Purpose |
|------|-------|---------|
| `logger.py` | 87 | Rotating file loggers: `trading_bot.log` (10MB) + `trades.log` (5MB) |
| `notifications.py` | 219 | Telegram + WhatsApp (CallMeBot) alerts for trades, signals, daily summary |
| `options_pricer.py` | 649 | 3-tier pricing: QuantLib → py_vollib → manual Black-Scholes. IV calc, Greeks, full analysis |
| `database.py` | 1,340 | SQLite with 11 tables: trades, orders, signals, daily_pnl, position_status_log, ml_features, ml_predictions, ml_model_performance, ml_feature_importance, market_data_cache, ml_training_jobs |
| `utils.py` | 483 | Market timing, expiry calculation (weekly/monthly), strike selection, position sizing |

#### Database Tables

| Table | Purpose |
|---|---|
| `trades` | All executed trades (entry/exit prices, P&L, strategy, direction, `exit_reason`) |
| `orders` | Individual order records per leg |
| `signals` | Generated signals (before execution decision) |
| `daily_pnl` | End-of-day P&L snapshots |
| `position_status_log` | Periodic position status (every 15 min) |
| `ml_features` | Training features for ML |
| `ml_predictions` | Every ML prediction logged |
| `ml_model_performance` | Model accuracy/F1/Sharpe tracking |
| `ml_feature_importance` | Feature importance per model |
| `market_data_cache` | Cached market data to reduce API calls |
| `ml_training_jobs` | Training job history and status |

---

### 5. `data/` — Market Data (~2,830 lines)

| File | Lines | Purpose |
|------|-------|---------|
| `data_fetcher.py` | ~1,360 | **Central data hub** — Kite API with rate limiting (3 req/sec) |
| `websocket_manager.py` | 532 | Kite WebSocket for real-time tick streaming |
| `nse_bhavcopy.py` | 608 | NSE bhavcopy historical data via `jugaad-data` |
| `nse_downloader.py` | 348 | Direct HTTP download from NSE archives |

#### `DataFetcher` Key Methods

| Method | Returns | Used By |
|---|---|---|
| `get_spot_price(underlying)` | float | Signal gen, strategies |
| `get_ltp(symbol)` | float | Position tracker (single) |
| `get_ltp_batch(symbols)` | Dict[str→float] | **Position tracker (batched polling)** |
| `get_options_chain(underlying)` | DataFrame | Strategy analysis, signal gen |
| `get_options_chain_with_greeks()` | DataFrame + Greeks | CLI display |
| `get_oi_data(underlying)` | Dict (PCR, max pain, sentiment, total call/put volume, put/call volume ratio) | Signal gen, event overlay, strategies |
| `get_historical_data(symbol, interval, days)` | DataFrame | Analysis, ML features |
| `get_historical_analysis(underlying)` | Dict (trend, RSI, ATR, SMA, S/R) | **Daily swing analysis** (uses `day` candles) |
| `get_intraday_analysis(underlying)` | Dict (VWAP, EMA9/21, RSI5m, bias) | **5-min entry timing** |
| `get_volatility_metrics(underlying)` | Dict (HV, IV, percentile, regime) | Strategy selection |

**Rate limiter:** `_throttle_api_call()` — sliding window, 3 calls/sec max with 50ms safety buffer. Applied to all Kite API calls.

#### `WebSocketManager`
- Manages `KiteTicker` connection, auto-reconnect, health monitoring (30s heartbeat)
- `register_callback("price_update", fn)` — position tracker registers here
- `subscribe_symbols()` / `subscribe_tokens()` — instrument subscription
- Provides `get_ltp(symbol)` from in-memory cache (no API call)

---

### 6. `execution/` — Order Execution (~1,832 lines)

| File | Lines | Purpose |
|------|-------|---------|
| `order_manager.py` | 679 | Order placement (paper simulation + live Kite API) |
| `position_tracker.py` | ~1,148 | Position monitoring with dual-mode: WebSocket + polling fallback |

#### `OrderManager`
- `execute_signal(signal, paper)` → places orders for each leg
- Paper mode: simulates fill at LTP ± slippage
- Live mode: `kite.place_order()` with limit/market
- `has_duplicate_position(signal)` — prevents re-entry
- Tracks `active_executions: Dict[str, TradeExecution]`

#### `PositionTracker` — Architecture

```
┌────────────────────────────────────────────────────┐
│                 PositionTracker                     │
│                                                    │
│  ┌──────────┐   ┌───────────┐   ┌──────────────┐  │
│  │ WebSocket │   │ Polling   │   │ Signal Exit  │  │
│  │ Callback  │   │ Thread    │   │ Thread       │  │
│  │ (on tick) │   │ (5s loop) │   │ (60s loop)   │  │
│  └─────┬─────┘   └─────┬─────┘   └──────┬───────┘  │
│        │               │                │          │
│        ▼               ▼                ▼          │
│  ┌─────────────────────────────────────────────┐   │
│  │ _update_metrics_and_check()                 │   │
│  │  → SL check (hard stop)                     │   │
│  │  → Trailing SL (HWM-based, 70% protect)     │   │
│  │  → Target check                             │   │
│  │  → Greeks-based exit (delta/theta/vega)     │   │
│  │  → Status logging every 15 min              │   │
│  └─────────────────────────────────────────────┘   │
│                                                    │
│  Callbacks: sl_hit 🔴, trailing_sl_hit 🟡,        │
│             target_hit 🟢, signal_exit 🔵          │
└────────────────────────────────────────────────────┘
```

**Polling is batched:** `_check_all_positions()` collects all leg symbols → single `get_ltp_batch()` call → distributes prices. 1 API call per poll cycle regardless of position count.

**Trailing SL:**
- Activates at 30% of target profit reached
- Tracks peak unrealized P&L (high-water mark)
- Locks 70% of peak profit as floor
- Profitable trailing SL exits classified as wins

---

### 7. `signals/` — Signal Generation (~2,021 lines)

| File | Lines | Purpose |
|------|-------|---------|
| `ml_signal_generator.py` | ~700 | **Primary signal engine** — ML prediction → strategy selection → option legs |
| `exit_signal_generator.py` | 895 | Intelligent exit signals (9 independent checks) |
| `signal_generator.py` | 434 | Legacy rule-based generator (not used in bot mode) |

#### `MLSignalGenerator` — Signal Pipeline

```
For each symbol in (UNDERLYING_ASSETS + watchlist):
  1. get_spot_price() ───────────────────────── current price
  2. get_options_chain() ────────────────────── strikes, premiums, OI
  3. get_oi_data() ──────────────────────────── PCR, max pain, sentiment
  4. get_volatility_metrics() ───────────────── IV, HV, percentile, regime
  5. get_historical_analysis() ──────────────── daily SMA, RSI, ATR, trend
  6. feature_engineer.extract_features() ────── 61 features
  7. predictor.predict() ────────────────────── BULLISH/NEUTRAL/BEARISH + confidence
  8. _build_global_event_context() ──────────── breadth + shock + put/call volume spike state
  9. _apply_event_regime_overlay() ──────────── flip/block/penalize confidence on risk-off regime
  10. _confirm_trend() ───────────────────────── check last 5 DB labels align
  11. get_intraday_analysis() ────────────────── VWAP/EMA/RSI5m → intraday bias
  12. DIRECTION_STRATEGY_MAP lookup ──────────── pick strategy by direction × IV
  13. strategy.analyze() ─────────────────────── construct StrategySignal with legs
```

#### Event-Regime Overlay (MVP)

- Builds a shared market context per scan from three proxies:
  1. **Breadth risk-off** across tracked symbols
  2. **Short-horizon intraday shock** (rapid downside moves)
  3. **Put/Call volume spike** from options-chain aggregates
- Uses hysteresis thresholds (`entry_threshold` / `exit_threshold`) to avoid rapid flip-flop.
- Applies one of: **flip direction**, **block new entries**, or **confidence penalty**, based on `EVENT_REGIME_CONFIG`.
- Emits explicit audit logs (`[EVENT_FLIP] ...`) when direction is transformed due to event regime.

#### Strategy Selection Map (direction × IV regime)

| Direction | Low IV | Normal IV | High IV |
|---|---|---|---|
| **BULLISH** | `bull_call_spread` | `long_call` | `bull_put_spread` (credit) |
| **BEARISH** | `bear_put_spread` | `long_put` | `bear_call_spread` (credit) |
| **NEUTRAL** | — | `iron_condor` | `short_put` / `short_call` |

#### `ExitSignalGenerator` — 9 Exit Checks

1. **Target/SL** — hard price-based
2. **Trend reversal** — SMA crossover flip
3. **Sentiment reversal** — PCR regime change
4. **Momentum reversal** — RSI flip (stocks: momentum, indices: mean-reversion)
5. **Support/resistance** — price breaks key levels
6. **OI shift** — significant OI buildup against position
7. **IV exit** — IV crush or spike
8. **Thesis invalidation** — original entry conditions no longer valid
9. **ML exit** — ML prediction contradicts position direction

---

### 8. `strategies/` — Strategy Library (~2,502 lines)

| File | Lines | Purpose |
|------|-------|---------|
| `base_strategy.py` | 291 | Abstract base, `StrategyType` enum (13 types), `OptionLeg`/`StrategySignal` dataclasses, `LIQUIDITY_GUARD` checks |
| `directional.py` | 619 | LongCall, LongPut, ShortCall, ShortPut |
| `spreads.py` | 806 | BullCallSpread, BearPutSpread, BearCallSpread, BullPutSpread |
| `volatility.py` | 520 | IronCondor, Straddle, Strangle |
| `catalogue.py` | 236 | Strategy registry/factory, `analyze_all()`, `get_best_signal()` |

#### Strategy Details

**Directional (4):**
| Strategy | Direction | IV Preference | Min Confidence | Risk Type |
|---|---|---|---|---|
| Long Call | Bullish | Low IV (cheap premium) | — | Capped (premium paid) |
| Long Put | Bearish | Low IV | — | Capped |
| Short Call | Neutral/Bearish | High IV | 70% | **Unlimited upside risk** |
| Short Put | Neutral/Bullish | High IV | 70% | Naked put (large but bounded) |

**Spreads (4):**
| Strategy | Direction | Type | Spread Width | Risk |
|---|---|---|---|---|
| Bull Call Spread | Bullish | Debit | ATR-based (`max(2, round(ATR/interval))`) | Capped (net debit) |
| Bear Put Spread | Bearish | Debit | ATR-based | Capped |
| Bear Call Spread | Bearish | Credit | ATR-based OTM + hedge | Capped (spread width - credit) |
| Bull Put Spread | Bullish | Credit | ATR-based OTM + hedge | Capped |

**Spread width fix:** `actual_interval` derived from real chain (`min(diffs)`) — not hardcoded. SBIN gets 4 strikes (₹20 width, not 1 strike / ₹5).

**Volatility (3):**
| Strategy | Direction | Legs | Index Only? |
|---|---|---|---|
| Iron Condor | Neutral | 4 (sell OTM call+put spreads) | No |
| Straddle | Neutral (exploding/contracting) | 2 (ATM call+put) | Yes |
| Strangle | Neutral | 2 (OTM call+put) | Yes |

**Liquidity Guard** (all strategies): Before entry, checks each leg for:
- Min volume: 1000 (index) / 500 (stock)
- Min OI: 5000 (index) / 1000 (stock)
- Max bid-ask spread: 5% of mid-price

---

### 9. `ml/` — Machine Learning (~10,800+ lines, 22 files)

#### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    ML PIPELINE                          │
│                                                         │
│  DATA SOURCES                FEATURE ENGINEERING        │
│  ┌──────────────┐           ┌──────────────────┐       │
│  │ NSE Bhavcopy │──────────▶│ unified_features │       │
│  │ (historical) │   ┌──────▶│ (50 features)    │       │
│  └──────────────┘   │       └────────┬─────────┘       │
│  ┌──────────────┐   │              │                   │
│  │ Live Feature │───┘              ▼                   │
│  │ Snapshots    │           ┌──────────────┐           │
│  └──────────────┘           │ Model Trainer│           │
│  ┌──────────────┐           │ (XGB+LGB+RF) │           │
│  │ Kite OHLCV   │──────────▶│ + Optuna     │           │
│  │ (API)        │           └──────┬───────┘           │
│  └──────────────┘                  │                   │
│                                    ▼                   │
│                           ┌──────────────┐              │
│  INFERENCE                │ Model Files  │              │
│  ┌──────────────┐         │ (.joblib)    │              │
│  │ Feature Eng  │         └──────┬───────┘              │
│  │ (61 features)│                │                      │
│  └──────┬───────┘                ▼                      │
│         │                ┌──────────────┐               │
│         └───────────────▶│ Predictor    │               │
│                          │ (ensemble)   │               │
│                          │ + guardrails │               │
│                          └──────┬───────┘               │
│                                 │                       │
│                                 ▼                       │
│                          {BULLISH|NEUTRAL|BEARISH}       │
│                          + confidence (0.0—1.0)         │
│                                                         │
│  FEEDBACK LOOP                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Feedback     │─▶│ Auto Retrain │─▶│ Model        │  │
│  │ Collector    │  │ (background) │  │ Registry     │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

#### Ternary Prediction System

| Label | Class | Meaning |
|---|---|---|
| 0 | BEARISH | Price expected to drop beyond threshold |
| 1 | NEUTRAL | Price stays within ±threshold (dead zone) |
| 2 | BULLISH | Price expected to rise beyond threshold |

Threshold is **Optuna-optimized per symbol** (search range: 0.1%–2.0%).

#### Feature Engineering

**Live features** (`feature_engineer.py`): 61 features in 6 categories:
1. **Price (15):** returns (1/5/10/20d), gap, range, candle body/shadow, price vs SMA20/50, 52w distance
2. **Technical (15):** SMA/EMA crossovers, RSI, MACD, Bollinger %B/width, Stochastic, ADX, Williams %R, ATR
3. **Options/Greeks (12):** IV, IV percentile, IV/HV ratio, position delta/gamma/theta/vega, ATM IV, skew
4. **OI Sentiment (6):** PCR, PCR change, max pain distance, OI buildup, call/put OI change
5. **Volatility (6):** HV 10/20, HV ratio, ATR, vol regime, vol trend
6. **Time/Calendar (7):** DTE, day of week, hour, monthly expiry distance, weekly flag, rollover, session

**Unified features** (`unified_features.py`): 50 features — subset compatible with both historical bhavcopy and live data.

#### Model Training

| Component | Details |
|---|---|
| Models | XGBoost, LightGBM, Random Forest |
| Ensemble | Weighted: XGB 50% + LGB 30% + RF 20% |
| Optuna | 50 trials per model (hyperparams), 200 trials for ensemble weights, 30 for label threshold |
| Parallel | 3 models train concurrently via `ThreadPoolExecutor` (cores/3 each) |
| Validation | Walk-forward (5 splits) + TimeSeriesSplit |
| Storage | `.joblib` files in `data/ml_models/`, tracked via MLflow |

#### ML File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `feature_engineer.py` | 865 | 61 live features extraction |
| `unified_features.py` | 488 | 50-feature bridge between historical and live |
| `model_trainer.py` | 1,222 | XGB/LGB/RF + Optuna training |
| `predictor.py` | 524 | Ensemble inference with caching (TTL=60s) |
| `guardrails.py` | 360 | 7 risk rules: sacred SL, max positions, drawdown breaker, model freshness |
| `full_pipeline.py` | 1,094 | Bhavcopy → Greeks proxies → unified features → per-symbol models |
| `data_collector.py` | 512 | Kite OHLCV collection + caching |
| `historical_data_collector.py` | 1,377 | NSE bhavcopy download + F&O processing |
| `historical_trainer.py` | 472 | Bhavcopy → ternary labels → training |
| `historical_predictor.py` | 372 | Per-symbol `.joblib` model loading + prediction |
| `live_feature_collector.py` | 664 | Background 15-min snapshot collection during market hours |
| `backtester.py` | 648 | Walk-forward backtesting with cost model |
| `evaluator.py` | 699 | Trading metrics (Sharpe, Sortino, drawdown, profit factor) + model comparison |
| `feedback_collector.py` | 399 | Prediction outcome logging + drift detection |
| `feedback_evaluator.py` | 676 | Correlate predictions with financial (₹) outcomes |
| `feedback_trainer.py` | 278 | Feedback-weighted retraining |
| `auto_retrain.py` | 408 | Background auto-retrain (min 50 samples, 7-day interval) |
| `model_registry.py` | 455 | Version control: dev → staging → production → archived + A/B testing |
| `mlflow_tracker.py` | 566 | MLflow experiment tracking (local file store) |
| `paper_trading_runner.py` | 585 | Paper trading orchestration for ML validation |
| `train_with_live_data.py` | 354 | Combined bhavcopy + live snapshot training |

#### Guardrails (7 Rules)

1. **Stop-loss is sacred** — ML cannot delay or override SL exits
2. **Confidence bounds** — ML adjustment capped at ±0.3
3. **Min ML confidence** — block trade if ML < 0.4
4. **Position sizing unchanged** — ML cannot change quantity
5. **Max positions respected** — hard cap from config
6. **Drawdown circuit breaker** — pause ML if daily loss > 5% of capital
7. **Model freshness** — fall back to rule-based if model > 14 days old

---

### 10. Hybrid Scanning Architecture

The bot uses a **two-layer scan** approach:

| Layer | Data Source | Interval | Purpose |
|-------|-----------|----------|---------|
| **Daily ML** | `get_historical_analysis()` (day candles) | 300s scan | Direction (swing-level): trend, RSI, ATR, S/R |
| **Intraday Timing** | `get_intraday_analysis()` (5-min candles) | Same scan | Entry timing: VWAP, EMA 9/21, RSI-5m, momentum |

**Logic:** ML says "go BULLISH" (daily), but intraday timing says "currently BEARISH" → **wait** for better entry. Only enter when both layers agree or intraday is NEUTRAL.

---

### 11. Key Configuration Tuning Points

| Parameter | Location | Current Value | What It Controls |
|---|---|---|---|
| `max_positions` | `TRADING_CONFIG` | 5 (live), 15 (paper) | Max concurrent trades |
| `capital_per_trade` | `TRADING_CONFIG` | ₹1.5L | Position sizing |
| `max_loss_per_day` | `TRADING_CONFIG` | ₹10K | Daily circuit breaker |
| `default_sl_percent` | `TRADING_CONFIG` | 30% | Stop loss on premium |
| `default_target_percent` | `TRADING_CONFIG` | 50% | Target profit on premium |
| `trailing_sl_percent` | `TRADING_CONFIG` | 30% | Trail at 30% below peak (protect 70%) |
| `signal_scan_interval` | `BOT_CONFIG` | 300s (5 min) | Time between market scans |
| `position_poll_interval` | `BOT_CONFIG` | 5s | Batched polling fallback interval |
| `min_confidence_for_trade` | `ML_CONFIG` | 0.50 | Minimum to take any trade |
| `optuna_trials` | `ML_CONFIG` | 50 | Hyperparameter search trials |
| `skip_first_minutes` | `MARKET_HOURS` | 105 (until 11:00) | Skip morning noise |
| `no_trade_after` | `MARKET_HOURS` | 14:00 | No new trades after 2 PM |
| `entry_threshold` / `exit_threshold` | `EVENT_REGIME_CONFIG.risk_off` | 1.0 / 0.8 | Risk-off regime activation/deactivation hysteresis |
| `flip_mode` | `EVENT_REGIME_CONFIG` | `flip_or_block` | Event handling policy (flip/block/penalize) |

---

### 12. Deployment Roadmap (Planned — Not Implemented)

Three-phase capital scaling plan to reach ₹1L/month target:

| Phase | Capital | Max Pos | Daily Loss | Strategy Mix | Duration |
|-------|---------|---------|------------|-------------|----------|
| **Phase 1** (Testing) | ₹2.5L | 3 | ₹5K | 60% index directional, 40% stock spreads, 0% naked | ~2-3 months |
| **Phase 2** (Scaling) | ₹10L | 6 | ₹20K | 60% index, 30% spreads, 10% cash-secured puts | Months 3-5 |
| **Phase 3** (Full) | ₹25L | 10 | ₹50K | 60% index, 30% spreads, 10% high-conviction sells | Month 6+ |

**Aggressiveness mode** (passive/moderate/aggressive) should be **ML-driven** — auto-selected based on recent win rate, drawdown history, VIX, and market regime. Not manually configured.

---

### 13. Known Issues / Technical Debt

| Issue | Location | Impact |
|---|---|---|
| `int64` JSON serialization error | Snapshot saving | Warning in logs, non-fatal |
| MLflow local metadata warnings | `mlruns/artifacts` malformed metadata files | Warning noise in logs, does not block inference/trading |
| `options_trader/` package (planned restructure) | Never implemented | Current flat structure works fine |
| BSE instruments limited chain | BullPutSpread for SENSEX | "Strikes not found" when ATR pushes too far OTM |
| Model confidence low (~51%) | Most symbols | Signals generated but skipped below 52% threshold |
