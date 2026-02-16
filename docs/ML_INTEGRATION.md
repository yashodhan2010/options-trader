# Machine Learning Integration Guide

This document describes the ML-powered trading system integrated into the Options Trader bot.

**IMPORTANT: As of v2.0, the bot operates in ML-Only Mode. All trading signals are derived exclusively from ML model predictions. Rule-based signals have been deprecated.**

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Components](#components)
6. [CLI Commands](#cli-commands)
7. [Training Pipeline](#training-pipeline)
8. [Model Comparison & Evaluation](#model-comparison--evaluation)
9. [Paper Trading](#paper-trading)
10. [Risk Guardrails](#risk-guardrails)
11. [MLflow Tracking](#mlflow-tracking)
12. [Feedback Loop](#feedback-loop)
13. [A/B Testing](#ab-testing)
14. [Troubleshooting](#troubleshooting)

---

## Overview

The ML module is the **sole source of trading signals**. No rule-based signal generation - ML drives everything.

Key features:

- **ML-Only Signals**: All entry signals come from ML predictions
- **Ternary Prediction**: BEARISH (0), NEUTRAL (1), BULLISH (2) with Optuna-optimized thresholds
- **Ensemble Models**: XGBoost + LightGBM + Random Forest with Optuna-tuned weights (data-driven, not hardcoded)
- **34+ Features**: Price, technicals, Greeks, OI sentiment, volatility, time-based
- **Optuna Deep Integration**: Hyperparameter tuning (50 trials + MedianPruner), ensemble weight optimization (200 trials), label threshold optimization (30 trials)
- **Parallel Training**: Concurrent sub-model training via ThreadPoolExecutor
- **Risk Guardrails**: Stop-loss protection, confidence bounds, circuit breaker
- **MLflow Tracking**: Full experiment and model versioning
- **Feedback Loop**: Drift detection and auto-retraining triggers
- **Model Comparison**: Trading-specific metrics (Sharpe, Profit Factor, Win Rate)
- **Full Pipeline Training**: End-to-end pipeline via `ml train-full` CLI command

### How It Works (ML-Only Mode)

```
Market Data --> Feature Engineering --> ML Prediction --> Strategy Selection --> Trade Signal
                                              |
                                              v
                                    Direction: BULLISH/NEUTRAL/BEARISH (ternary)
                                    Confidence: 0.0 to 1.0
                                    Threshold: Optuna-optimized per symbol
```

The ML system:
1. Predicts market direction using **ternary labels** (BEARISH=0, NEUTRAL=1, BULLISH=2)
2. The NEUTRAL dead-zone threshold is Optuna-optimized per symbol (range: 0.1%–2.0%)
3. Maps direction to appropriate strategy based on IV regime
4. Generates option legs using strategy execution
5. Filters trades below minimum confidence threshold
6. NEUTRAL predictions are explicitly skipped (no trade)

**No rule-based signals** - if ML model is not loaded, no signals are generated.

---

## Architecture

```
signals/
├── ml_signal_generator.py   # ML-Only signal generator (PRIMARY)
├── signal_generator.py      # Legacy rule-based (DEPRECATED)
└── exit_signal_generator.py # Exit signals (still rule-based for safety)

ml/
├── __init__.py              # Module exports
├── feature_engineer.py      # 34+ feature extraction
├── unified_features.py      # Unified feature definitions
├── data_collector.py        # Historical data caching
├── model_trainer.py         # XGBoost/LightGBM/RF + Optuna (parallel training)
├── full_pipeline.py         # End-to-end training pipeline with Optuna
├── evaluator.py             # Trading metrics & model comparison
├── predictor.py             # Ensemble inference (ternary)
├── backtester.py            # Historical simulation
├── feedback_collector.py    # Outcome logging & drift
├── feedback_trainer.py      # Feedback-based retraining
├── feedback_evaluator.py    # Feedback evaluation
├── model_registry.py        # Version control & A/B testing
├── mlflow_tracker.py        # MLflow integration
├── guardrails.py            # Risk management
├── auto_retrain.py          # Automatic retraining triggers
├── historical_data_collector.py  # Historical data fetching
├── historical_trainer.py    # Train on historical data
├── historical_predictor.py  # Historical predictions
├── live_feature_collector.py # Real-time feature extraction
└── paper_trading_runner.py  # Paper trading orchestration
```

### Data Flow (ML-Only)

```
+------------------+    +-------------------+    +------------------+
|  Data Collector  |--->|  Feature Engineer |--->|  Model Trainer   |
|  (3mo history)   |    |   (34+ features)  |    | (Optuna tuning)  |
+------------------+    +-------------------+    +--------+---------+
                                                         |
                                                         v
+------------------+    +-------------------+    +------------------+
| ML Signal        |<---|    Predictor      |<---|  Model Registry  |
| Generator        |    | (ensemble + grd)  |    | (versioning)     |
+--------+---------+    +-------------------+    +------------------+
         |
         v
+------------------+    +-------------------+    +------------------+
|   Execution      |--->|Feedback Collector |--->|  Drift Detection |
|                  |    | (log outcomes)    |    | (retrain alert)  |
+------------------+    +-------------------+    +------------------+
```

---

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

New ML dependencies:
- `scikit-learn>=1.2.0`
- `xgboost>=1.7.0`
- `lightgbm>=4.0.0`
- `optuna>=3.0.0`
- `mlflow>=2.5.0`

### 2. Initialize Database

The ML tables are created automatically on first run. Tables added:
- `ml_features` - Feature snapshots at entry/exit
- `ml_predictions` - Prediction logs with outcomes
- `ml_model_performance` - Model metrics and backtest results
- `ml_feature_importance` - Feature importance rankings
- `market_data_cache` - Historical OHLCV data
- `ml_training_jobs` - Training job history

### 3. Enable ML

In `config/settings.py`:

```python
ML_CONFIG = {
    "enabled": True,  # Set to True to activate
    # ... other settings
}
```

---

## Configuration

All ML settings are in `config/settings.py` under `ML_CONFIG`:

```python
ML_CONFIG = {
    # Master switch
    "enabled": True,
    
    # Confidence blending
    "confidence_weight": 0.5,  # 50% ML, 50% rule-based
    "min_blended_confidence": 0.55,  # Min confidence to trade
    
    # Training
    "optuna_trials": 50,  # Hyperparameter trials per model
    "min_training_samples": 100,  # Min samples for training
    "training_lookback_days": 90,  # Use 3 months of data
    "parallel_training": True,  # Train XGB/LGB/RF concurrently
    
    # Ensemble weights (Optuna-optimized, not hardcoded)
    # These are default starting values; actual weights are
    # determined by Optuna optimization (200 trials, F1-based)
    "ensemble_weights": {
        "xgboost": 0.5,
        "lightgbm": 0.3,
        "random_forest": 0.2,
    },
    
    # Ternary labeling
    # Threshold defines the NEUTRAL dead-zone (±threshold)
    # Optimized per symbol via Optuna (30 trials, range: 0.001 to 0.02)
    "label_threshold": 0.005,  # Default ±0.5%
    
    # Optuna integration
    "optuna": {
        "n_trials_hyperparams": 50,     # Trials for model hyperparameters
        "n_trials_weights": 200,        # Trials for ensemble weight optimization
        "n_trials_threshold": 30,       # Trials for label threshold optimization
        "pruner": "MedianPruner",       # Early stopping for bad trials
        "sampler": "TPESampler",        # Bayesian optimization
    },
    
    # Guardrails
    "guardrails": {
        "stop_loss_sacred": True,  # NEVER override stop-loss
        "max_confidence_adjustment": 0.3,  # Max ±30% adjustment
        "min_ml_confidence": 0.4,  # Ignore if ML < 40%
        "circuit_breaker_losses": 3,  # Pause after 3 consecutive losses
        "max_model_age_days": 14,  # Retrain if model > 2 weeks old
    },
    
    # Paper trading
    "paper_trading": {
        "initial_capital": 100000,
        "max_position_pct": 0.10,  # 10% per position
        "max_open_trades": 5,
        "drawdown_alert_pct": 0.10,  # Alert at 10% drawdown
        "win_rate_alert": 0.40,  # Alert if win rate < 40%
    },
    
    # Feedback loop
    "feedback": {
        "log_all_predictions": True,
        "log_features_at_entry": True,
        "log_features_at_exit": True,
        "drift_detection_enabled": True,
        "drift_threshold": 0.10,  # Retrain if accuracy drops 10%
    },
}
```

---

## CLI Commands

All ML commands are accessed via `ml <command>` in the CLI:

| Command | Description |
|---------|-------------|
| `ml status` | Show ML system status, model info, and recent performance |
| `ml train [SYMBOL]` | Train model with Optuna optimization (all symbols if omitted) |
| `ml train-full [START END]` | Full pipeline: data collection → features → ternary labels → Optuna threshold/weights optimization |
| `ml train-best [CONFIG]` | Train all symbols with best configuration (default: rf_aggressive) |
| `ml compare SYMBOL` | Compare 12 model configurations with trading metrics |
| `ml backtest SYMBOL` | Run historical backtest on trained model |
| `ml paper start` | Start paper trading session |
| `ml paper stop` | End session and show summary |
| `ml paper stats` | Show current session statistics |
| `ml predict SYMBOL` | Get ML prediction for underlying |
| `ml features SYMBOL` | Show extracted features for underlying |
| `ml drift` | Check for model drift |

### Training Best Configuration

The `train-best` command trains all symbols with a specific optimized configuration:

```bash
# Train with default (rf_aggressive) - highest Sharpe Ratio
ml train-best

# Train with specific configuration
ml train-best xgb_balanced
ml train-best lgb_aggressive
```

**Available Configurations:**

| Config | Model | Description |
|--------|-------|-------------|
| `rf_aggressive` | Random Forest | Deep trees (max_depth=20), low regularization |
| `rf_balanced` | Random Forest | Balanced depth and regularization |
| `rf_conservative` | Random Forest | Shallow trees, high regularization |
| `xgb_aggressive` | XGBoost | High learning rate (0.1), deep trees |
| `xgb_balanced` | XGBoost | Moderate settings |
| `lgb_aggressive` | LightGBM | High learning rate, many leaves |
| `lgb_balanced` | LightGBM | Balanced settings |

---

## Components

### Feature Engineer

Extracts 34 normalized features from market data:

| Category | Count | Examples |
|----------|-------|----------|
| Price | 10 | Returns, ATR, range, VWAP distance |
| Technicals | 10 | RSI, MACD, Bollinger, EMA crossovers |
| Options/Greeks | 6 | IV, delta, gamma, theta, vega |
| Volatility | 4 | HV, IV/HV ratio, vol regime |
| Time/Calendar | 4 | DTE, day of week, expiry proximity |

```python
from ml import get_feature_engineer

fe = get_feature_engineer()
features = fe.extract_features(
    spot_price=21500,
    market_data={"oi_data": {...}, "volatility": {...}},
    underlying="NIFTY",
    strategy_type="LONG_CALL"
)
# Returns: {"price_return_1d": 0.45, "rsi_14": 0.62, ...}
```

### Data Collector

Fetches and caches 3 months of historical data:

```python
from ml import get_data_collector

collector = get_data_collector()

# Collect initial data
await collector.collect_historical_data("NIFTY", days=90)

# Daily update
await collector.update_data("NIFTY")

# Get training DataFrame
df = collector.get_training_dataframe("NIFTY", days=90)
```

### Model Trainer

Trains ensemble models with Optuna hyperparameter optimization:

```python
from ml import get_model_trainer

trainer = get_model_trainer()

# Train direction prediction model
result = trainer.train_direction_model(
    underlying="NIFTY",
    strategy_type="LONG_CALL",  # Optional filter
    optimize=True,  # Run Optuna
    n_trials=50
)

print(f"Accuracy: {result['metrics']['accuracy']:.2%}")
print(f"Best params: {result['best_params']}")
```

### Predictor

Makes predictions with ensemble and guardrails:

```python
from ml import get_predictor

predictor = get_predictor()

# Get ML prediction
prediction = predictor.predict(features, "NIFTY", "LONG_CALL")
print(f"Direction: {prediction.direction}")
print(f"Confidence: {prediction.confidence:.2%}")

# With guardrails (recommended)
prediction = predictor.predict_with_guardrails(
    features=features,
    underlying="NIFTY",
    strategy_type="LONG_CALL",
    rule_confidence=0.72  # From rule-based signal
)
print(f"Blended confidence: {prediction.blended_confidence:.2%}")
```

### Backtester

Simulates historical performance with realistic costs:

```python
from ml.backtester import Backtester

bt = Backtester()

# Backtest ML model
result = bt.run_ml_backtest(
    underlying="NIFTY",
    start_date=datetime(2024, 10, 1),
    end_date=datetime(2024, 12, 31)
)

print(f"Total P&L: ₹{result.total_pnl:,.2f}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"Win Rate: {result.win_rate:.1%}")
print(f"Max Drawdown: {result.max_drawdown:.1%}")
```

---

## Model Comparison & Evaluation

The `evaluator.py` module provides trading-specific metrics for model comparison, going beyond standard ML accuracy.

### Trading Metrics

The `TradingMetrics` dataclass tracks 18 metrics across three categories:

**Trading Metrics:**
| Metric | Description |
|--------|-------------|
| `total_return` | Total percentage return |
| `sharpe_ratio` | Risk-adjusted return (annualized) |
| `max_drawdown` | Maximum peak-to-trough decline |
| `profit_factor` | Gross profit / Gross loss |
| `win_rate` | Winning trades / Total trades |
| `avg_win` | Average winning trade P&L |
| `avg_loss` | Average losing trade P&L |
| `risk_reward_ratio` | Average win / Average loss |
| `calmar_ratio` | Annual return / Max drawdown |

**Signal Quality Metrics:**
| Metric | Description |
|--------|-------------|
| `bullish_precision` | Accuracy of bullish predictions |
| `bearish_precision` | Accuracy of bearish predictions |
| `false_signal_rate` | Percentage of wrong signals |

**Risk Metrics:**
| Metric | Description |
|--------|-------------|
| `var_95` | Value at Risk (95% confidence) |
| `expected_shortfall` | Average loss beyond VaR |
| `volatility` | Standard deviation of returns |

### Running Model Comparison

Compare 12 pre-configured models to find the best:

```bash
# Via CLI
ml compare RELIANCE
```

```python
# Via Python
from ml.evaluator import ModelComparator, TradingEvaluator

comparator = ModelComparator()

# Compare all configurations
results_df = comparator.compare_configurations(
    X=features,           # Feature matrix
    y=labels,             # Target labels
    feature_names=names,  # Feature names
    prices=price_series   # For P&L simulation
)

# Get best configuration
best = comparator.get_best_configuration(
    metric='sharpe_ratio',
    secondary_metrics=['profit_factor', 'win_rate']
)

print(f"Best model: {best['config_name']}")
print(f"Sharpe: {best['sharpe_ratio']:.2f}")
```

### Model Configurations Tested

The comparator tests 12 configurations:

| Category | Variants |
|----------|----------|
| XGBoost | conservative, balanced, aggressive |
| LightGBM | conservative, balanced, aggressive |
| Random Forest | conservative, balanced, aggressive |
| Ensemble | xgb_heavy, balanced, lgb_heavy |

### Using TradingEvaluator Directly

```python
from ml.evaluator import TradingEvaluator

evaluator = TradingEvaluator(
    transaction_cost=0.001,  # 0.1% per trade
    slippage=0.0005          # 0.05% slippage
)

# Evaluate a trained model
metrics = evaluator.evaluate(
    model=trained_model,
    X_test=X_test,
    y_test=y_test,
    prices=price_series
)

print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
print(f"Win Rate: {metrics.win_rate:.1%}")
print(f"Profit Factor: {metrics.profit_factor:.2f}")
```

---

## Training Pipeline

### Full Pipeline Training (Recommended)

The `ml train-full` command runs the complete end-to-end pipeline:

```bash
# Full pipeline with default dates (3 months lookback)
ml train-full

# Full pipeline with custom date range
ml train-full 2025-01-01 2025-12-31
```

The full pipeline performs:
1. **Data Collection**: Fetches OHLCV + OI data for all watchlist symbols
2. **Feature Engineering**: Extracts 34+ normalized features
3. **Ternary Labeling**: Creates BEARISH/NEUTRAL/BULLISH labels with dead-zone
4. **Optuna Threshold Optimization**: Finds optimal NEUTRAL dead-zone width per symbol (30 trials)
5. **Model Training**: Trains XGB/LGB/RF with Optuna hyperparameter tuning (50 trials each, MedianPruner)
6. **Parallel Sub-Model Training**: XGB/LGB/RF train concurrently via ThreadPoolExecutor
7. **Ensemble Weight Optimization**: Optuna finds optimal blend weights (200 trials, F1-based)
8. **Model Saving**: Saves ensemble model with metadata to `data/ml_models/`

### Initial Training (One-time Setup)

```python
import asyncio
from ml import (
    get_data_collector,
    get_model_trainer,
    get_model_registry
)

async def initial_setup():
    # 1. Collect historical data
    collector = get_data_collector()
    for underlying in ["NIFTY", "BANKNIFTY"]:
        await collector.collect_historical_data(underlying, days=90)
    
    # 2. Train models
    trainer = get_model_trainer()
    for underlying in ["NIFTY", "BANKNIFTY"]:
        result = trainer.train_direction_model(
            underlying=underlying,
            optimize=True,
            n_trials=50
        )
        print(f"{underlying}: Accuracy={result['metrics']['accuracy']:.2%}")
    
    # 3. Promote best model to production
    registry = get_model_registry()
    registry.promote_model(result['model_version'], "production")

asyncio.run(initial_setup())
```

### Scheduled Retraining

Add to your cron or scheduler:

```python
# Weekly retraining (e.g., every Sunday)
def weekly_retrain():
    from ml import get_data_collector, get_model_trainer, get_feedback_collector
    
    collector = get_data_collector()
    trainer = get_model_trainer()
    feedback = get_feedback_collector()
    
    # Check if retraining needed
    if not feedback.should_retrain():
        return
    
    # Update data and retrain
    for underlying in ["NIFTY", "BANKNIFTY"]:
        collector.update_data(underlying)
        trainer.train_direction_model(underlying, optimize=True)
    
    # Reset baseline after retraining
    feedback.reset_baseline()
```

---

## Paper Trading

### Starting a Session

```python
from ml import get_paper_trading_runner

runner = get_paper_trading_runner()

# Start new session
session_id = runner.start_session(
    model_version="v1.0.0",  # Optional
    initial_capital=100000
)

print(f"Session started: {session_id}")
```

### Processing Signals

```python
# When a rule-based signal is generated
result = runner.process_signal(
    underlying="NIFTY",
    strategy_type="LONG_CALL",
    spot_price=21500,
    market_data={"oi_data": {...}, "volatility": {...}},
    rule_confidence=0.72,
    target_price=22000,
    stop_loss_price=21200
)

if result["action"] == "TRADE":
    print(f"Paper trade opened: {result['trade_id']}")
else:
    print(f"Skipped: {result['reason']}")
```

### Updating Positions

```python
# Called periodically (e.g., every minute)
closed_trades = runner.update_positions({
    "NIFTY": 21650,
    "BANKNIFTY": 48200
})

for trade in closed_trades:
    print(f"Closed {trade['trade_id']}: P&L ₹{trade['pnl']:.2f}")
```

### Ending Session

```python
summary = runner.end_session()

print(f"Total P&L: ₹{summary['total_pnl']:,.2f}")
print(f"Win Rate: {summary['win_rate']:.1%}")
print(f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {summary['max_drawdown']:.1%}")
```

---

## Risk Guardrails

The guardrails system ensures ML predictions don't cause catastrophic losses:

### 1. Stop-Loss Sacred Rule

**ML can NEVER override a stop-loss exit.** This is hardcoded and non-configurable.

```python
# In guardrails.py
def check_exit_signal(self, ...):
    if rule_exit_reason == "STOP_LOSS":
        return True, 1.0, "Stop-loss is sacred"
```

### 2. Confidence Bounds

ML can only adjust confidence by ±30% (configurable):

```python
# Rule confidence: 0.70
# ML confidence: 0.30 (very low)
# Adjustment: max -0.30 → Final: 0.40 (not 0.30)
```

### 3. Minimum ML Confidence

If ML confidence < 40%, the ML signal is ignored entirely:

```python
if ml_confidence < 0.4:
    return rule_confidence  # Use rule-based only
```

### 4. Circuit Breaker

After 3 consecutive losses, ML signals are paused for review:

```python
guardrails = get_guardrails()
guardrails.record_loss()  # Call on each loss

if guardrails.is_circuit_breaker_active():
    print("ML paused due to consecutive losses")
```

### 5. Model Staleness

Models older than 14 days trigger retraining alerts:

```python
if model_age_days > 14:
    logger.warning("Model is stale, consider retraining")
```

### 6. Direction Agreement

If ML and rule-based signals disagree on direction, confidence is reduced:

```python
# Rule: BULLISH, ML: BEARISH
# → Confidence reduced by 50%
```

### 7. Extreme Confidence Penalty

Very low (<0.3) or very high (>0.9) ML confidence is penalized:

```python
# Overconfident predictions are often wrong
if ml_confidence > 0.9:
    penalty = 0.1  # Reduce by 10%
```

---

## MLflow Tracking

### Viewing Experiments

Start the MLflow UI:

```bash
mlflow ui --port 5000
```

Open http://localhost:5000 to view:
- Training runs with hyperparameters
- Validation metrics (accuracy, precision, recall, F1, AUC)
- Backtest metrics (Sharpe, win rate, profit factor)
- Feature importance plots
- Model artifacts

### Programmatic Access

```python
from ml.mlflow_tracker import get_mlflow_tracker

tracker = get_mlflow_tracker()

# Start a training run
with tracker.start_training_run("nifty_direction_v2") as run:
    # ... training code ...
    tracker.log_validation_metrics({"accuracy": 0.68, "f1_score": 0.65})
    tracker.log_backtest_metrics({"sharpe_ratio": 1.5, "win_rate": 0.58})
    tracker.log_model(model, "direction_model")

# Compare runs
runs = mlflow.search_runs(experiment_ids=["1"])
print(runs[["run_id", "metrics.accuracy", "metrics.sharpe_ratio"]])
```

---

## Feedback Loop

### How It Works

1. **Entry**: Features are logged when a trade is opened
2. **Exit**: Features and P&L are logged when trade closes
3. **Analysis**: Prediction accuracy is tracked per model
4. **Drift Detection**: If accuracy drops >10%, retraining is recommended
5. **Auto-Retraining**: Model is automatically retrained from trade outcomes

### Target Variable

**Initial training** uses ternary price direction labels with an Optuna-optimized threshold:

| Label | Value | Meaning |
|-------|-------|---------|
| BEARISH | 0 | Next-day return < -threshold |
| NEUTRAL | 1 | Next-day return within ±threshold (dead zone) |
| BULLISH | 2 | Next-day return > +threshold |

The threshold is optimized per symbol (range: 0.1%–2.0%, 30 Optuna trials).

**Feedback-based retraining** uses actual trade P&L as the target:

| Outcome | Target Value | Meaning |
|---------|--------------|---------|
| WIN | 2 | Trade profit > 1% |
| BREAKEVEN | 1 | Trade P&L between -1% and +1% |
| LOSS | 0 | Trade loss > 1% |

This enables the model to learn from **real trading performance** rather than just historical price movements.

### Auto-Retraining Configuration

```python
# In config/settings.py
ML_CONFIG = {
    "auto_retrain": {
        "enabled": True,                     # Enable automatic retraining
        "min_samples": 50,                   # Minimum trade outcomes to retrain
        "interval_days": 7,                  # Retrain every N days if enough data
        "drift_threshold": 0.15,             # Accuracy drop threshold for retrain
        "auto_promote": False,               # Auto-promote if accuracy > 55%
        "use_feedback_target": True,         # Use trade P&L as target
        "check_interval_seconds": 3600,      # Background check interval (1 hour)
    },
}
```

### CLI Commands for Retraining

```bash
# Check retrain status and conditions
ml retrain-status

# Manually trigger feedback-based retrain
ml retrain

# Force retrain even if conditions not met
ml retrain --force
```

### Manual Feedback Logging

```python
from ml import get_feedback_collector

feedback = get_feedback_collector()

# Log at trade entry
feedback.log_entry_features(
    execution_id="trade_123",
    underlying="NIFTY",
    strategy_type="LONG_CALL",
    features=features,
    spot_price=21500
)

# Log prediction
feedback.log_prediction(
    execution_id="trade_123",
    underlying="NIFTY",
    strategy_type="LONG_CALL",
    model_version="v1.0.0",
    model_type="ensemble",
    direction_prediction="BULLISH",
    ml_confidence=0.72,
    rule_confidence=0.68,
    blended_confidence=0.70
)

# Log outcome at exit
feedback.log_outcome(
    execution_id="trade_123",
    actual_pnl=1500,
    actual_pnl_percent=3.2,
    trade_duration_seconds=7200
)
```

### Auto-Retrainer API

```python
from ml import get_auto_retrainer

retrainer = get_auto_retrainer()

# Check if retrain conditions are met
conditions = retrainer.check_retrain_conditions()
print(f"Should retrain: {conditions['should_retrain']}")
print(f"Available samples: {conditions['available_samples']}")
print(f"Drift detected: {conditions['drift_detected']}")

# Get feedback training data (uses trade P&L as target)
X, y, feature_names = retrainer.get_feedback_training_data()

# Manually trigger retraining
result = retrainer.retrain_from_feedback(force=True)
if result['success']:
    print(f"New model: {result['model_version']}")

# Start background monitor (auto-retrain when conditions met)
retrainer.start_background_monitor(check_interval=3600)  # Check every hour
```

### Checking Performance

```python
stats = feedback.get_performance_stats(days=30)
print(f"Accuracy: {stats['accuracy']:.1%}")
print(f"Total predictions: {stats['total_predictions']}")

if feedback.should_retrain():
    print("Retraining recommended!")
```

---

## A/B Testing

Compare two model versions in production:

```python
from ml import get_model_registry

registry = get_model_registry()

# Start A/B test
registry.start_ab_test(
    version_a="v1.0.0",  # Control (current production)
    version_b="v1.1.0",  # Test (new model)
    traffic_split=0.5    # 50% traffic to each
)

# During trading, get assigned model
model_version = registry.get_ab_test_model()

# After 1 week, end test and promote winner
results = registry.end_ab_test(promote_winner=True)
print(f"Winner: {results['winner']} ({results['winner_version']})")
```

---

## Troubleshooting

### ML Not Activating

1. Check `ML_CONFIG["enabled"]` is `True`
2. Verify ML dependencies installed: `pip install xgboost lightgbm optuna mlflow`
3. Check logs for import errors

### Low Accuracy

1. Ensure sufficient training data (>100 samples)
2. Try more Optuna trials (increase `optuna_trials`)
3. Check for data quality issues
4. Review feature importance for unexpected patterns

### Model Not Loading

1. Check `ML_MODELS_DIR` path exists
2. Verify model files are present: `ls ml_models/`
3. Check pickle compatibility (same Python version)

### High Latency

1. Reduce feature count (disable expensive features)
2. Use single model instead of ensemble
3. Enable prediction caching
4. Check database query performance

### Circuit Breaker Activated

```python
from ml import get_guardrails

guardrails = get_guardrails()

# Check status
if guardrails.is_circuit_breaker_active():
    print("Circuit breaker active due to consecutive losses")
    
# Reset after review
guardrails.reset_circuit_breaker()
```

### MLflow UI Not Starting

```bash
# Check if port is in use
netstat -an | findstr 5000

# Use different port
mlflow ui --port 5001
```

---

## Best Practices

1. **Start with paper trading** for at least 2 weeks before live
2. **Keep guardrails strict** initially (low confidence adjustment)
3. **Monitor daily** during initial deployment
4. **Retrain weekly** with fresh data
5. **Use A/B testing** before promoting new models
6. **Never disable stop-loss protection**
7. **Log everything** for post-trade analysis
8. **Run model comparison** before choosing a configuration
9. **Use trading metrics** (Sharpe, Win Rate) not just accuracy

---

## Quick Reference

### CLI Commands

```bash
# Training
ml train                    # Train all symbols with Optuna
ml train RELIANCE           # Train specific symbol
ml train-full               # Full pipeline: data + features + Optuna threshold/weights
ml train-full 2025-01-01 2025-12-31  # Full pipeline with custom dates
ml train-best               # Train all with rf_aggressive
ml train-best xgb_balanced  # Train with specific config

# Comparison & Analysis
ml compare RELIANCE         # Compare 12 model configs
ml backtest NIFTY           # Run historical backtest
ml predict NIFTY            # Get ML prediction (ternary: BULL/NEUTRAL/BEAR)
ml features NIFTY           # Show extracted features

# Paper Trading
ml paper start              # Start session
ml paper stats              # Check progress
ml paper stop               # End and show summary

# Status & Monitoring
ml status                   # ML system status
ml drift                    # Check for model drift

# Retraining
ml retrain-status           # Check retrain conditions
ml retrain                  # Feedback-based retrain
ml retrain --force          # Force retrain
```

### Python API

```python
# Enable ML
ML_CONFIG["enabled"] = True

# Get components
from ml import (
    get_feature_engineer,
    get_predictor,
    get_model_trainer,
    get_data_collector,
    get_feedback_collector,
    get_model_registry,
    get_paper_trading_runner,
    get_guardrails,
    TradingEvaluator,
    ModelComparator,
    TradingMetrics
)

# Train model
trainer = get_model_trainer()
trainer.train_direction_model("NIFTY", optimize=True)

# Train with specific params (no Optuna)
model, metrics, version = trainer.train_with_params(
    X=features, y=labels, feature_names=names,
    model_type="rf",
    params={"n_estimators": 100, "max_depth": 20}
)

# Compare models
comparator = ModelComparator()
results = comparator.compare_configurations(X, y, names, prices)
best = comparator.get_best_configuration('sharpe_ratio')

# Get prediction
predictor = get_predictor()
pred = predictor.predict_with_guardrails(features, "NIFTY", "LONG_CALL", 0.7)

# Paper trade
runner = get_paper_trading_runner()
runner.start_session()
runner.process_signal(...)
summary = runner.end_session()
```

---

*Last updated: February 16, 2026*
