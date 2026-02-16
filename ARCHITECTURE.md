# Options Trader Architecture

## New Package Structure (v2.1)

The codebase has been reorganized into a clean, modular MLOps architecture with advanced Optuna integration, parallel training, and ternary prediction system.

```
options-trader/
├── run.py                    # Main entry point
├── cli.py                    # Legacy CLI (still works)
├── bot.py                    # Automated trading bot
├── requirements.txt
├── .env                      # API keys and config
│
├── options_trader/           # NEW: Main Python package
│   ├── __init__.py
│   │
│   ├── config/               # Configuration
│   │   ├── __init__.py
│   │   └── settings.py       # All settings consolidated
│   │
│   ├── core/                 # Core utilities
│   │   ├── __init__.py
│   │   ├── logger.py         # Logging
│   │   ├── database.py       # SQLite database
│   │   ├── utils.py          # Helper functions
│   │   ├── options_pricer.py # Greeks/IV calculations
│   │   └── notifications.py  # Telegram alerts
│   │
│   ├── broker/               # NEW: Broker integration
│   │   ├── __init__.py
│   │   ├── auth.py           # Kite authentication
│   │   ├── data_fetcher.py   # Market data
│   │   └── websocket.py      # Real-time streaming
│   │
│   ├── ml/                   # Machine Learning
│   │   ├── __init__.py
│   │   ├── data/             # Data collection & features
│   │   │   └── __init__.py
│   │   ├── training/         # Model training
│   │   │   └── __init__.py
│   │   ├── inference/        # Prediction & guardrails
│   │   │   └── __init__.py
│   │   ├── registry/         # Model versioning
│   │   │   └── __init__.py
│   │   └── monitoring/       # Performance tracking
│   │       └── __init__.py
│   │
│   ├── strategies/           # Trading strategies
│   │   └── __init__.py       # All strategy exports
│   │
│   ├── signals/              # Signal generation
│   │   └── __init__.py       # All signal exports
│   │
│   ├── execution/            # Order execution
│   │   └── __init__.py       # All execution exports
│   │
│   └── cli/                  # CLI modules
│       ├── __init__.py
│       └── main.py           # CLI entry point
│
├── artifacts/                # NEW: All ML artifacts
│   ├── models/               # Trained models
│   ├── mlflow/               # MLflow tracking
│   └── cache/                # Cached data
│
├── config/                   # Legacy: Still used for watchlist
│   ├── settings.py           # Original settings (backwards compat)
│   └── watchlist.json        # Stock watchlist
│
├── data/                     # Runtime data
│   ├── trading_bot.db        # SQLite database
│   ├── session.json          # Kite session
│   └── ml_models/            # Legacy model storage
│
├── logs/                     # Log files
│   ├── trading_bot.log
│   └── trades.log
│
├── docs/                     # Documentation
│   ├── ML_INTEGRATION.md
│   └── STRATEGIES.md
│
└── (legacy modules)          # Still work, will be deprecated
    ├── auth/
    ├── core/
    ├── data/
    ├── ml/
    ├── signals/
    ├── strategies/
    └── execution/
```

## Usage

### Running the CLI
```bash
# Interactive mode
python run.py
# or
python cli.py

# Automated bot
python run.py --bot
python run.py --bot --paper  # Paper trading
```

### Importing from New Structure
```python
# Configuration
from options_trader.config import settings
from options_trader.config.settings import ML_CONFIG, TRADING_CONFIG

# Core utilities
from options_trader.core import logger, Database
from options_trader.core.utils import is_trading_allowed, get_market_status

# Broker
from options_trader.broker import connect, get_kite, DataFetcher

# ML
from options_trader.ml import MLPredictor, ModelTrainer, FeatureEngineer
from options_trader.ml.data import HistoricalDataCollector
from options_trader.ml.training import Backtester
from options_trader.ml.inference import get_predictor, get_guardrails

# Strategies
from options_trader.strategies import StrategyType, StrategyCatalogue

# Signals
from options_trader.signals import SignalGenerator, MLSignalGenerator

# Execution
from options_trader.execution import OrderManager, PositionTracker
```

### Backwards Compatibility

The legacy import paths still work:
```python
# These still work
from config.settings import ML_CONFIG
from core.logger import logger
from data.data_fetcher import data_fetcher
from ml.predictor import get_predictor
from auth.kite_auth import connect
```

## ML Pipeline

### Ternary Prediction System

The ML system predicts three classes instead of binary up/down:
- **BEARISH (0)**: Price expected to drop beyond threshold
- **NEUTRAL (1)**: Price expected to stay within ±threshold (dead zone)
- **BULLISH (2)**: Price expected to rise beyond threshold

The threshold is Optuna-optimized per symbol (search range: 0.1%–2.0%).

### Optuna Deep Integration

| Component | Optuna Usage |
|-----------|--------------|
| Hyperparameters | 50 trials per model (XGB/LGB/RF) with MedianPruner |
| Ensemble Weights | 200 trials to optimize XGB/LGB/RF blend weights (F1-based) |
| Label Threshold | 30 trials to optimize ternary NEUTRAL dead-zone width |

### Parallel Training

XGBoost, LightGBM, and Random Forest train concurrently via `ThreadPoolExecutor`.
CPU cores are divided equally among sub-models (e.g., 12 cores → 4 per model).
Enabled via `parallel_training: True` in settings.

### Statistical Upgrades

- **RSI Split**: Stocks use momentum (RSI trend-following), indices use mean-reversion
- **Liquidity Guard**: Checks volume, OI, and bid-ask spread per leg before entry
- **ATR-Based Spread Width**: Spread widths adapt to current volatility via ATR
- **Expiry Selection**: Weekly expiry for indices, monthly expiry for stocks

### Trailing Stop Loss

High-watermark trailing system:
- Activates at 30% of target profit
- Tracks peak unrealized P&L
- Protects 70% of peak profit
- Separate `trailing_sl_hit` callback (🟡) vs hard `sl_hit` (🔴)
- Correctly classifies profitable trailing SL exits as wins

### Sub-packages

1. **ml.data** - Data Collection & Feature Engineering
   - `FeatureEngineer`: Extract features from market data
   - `DataCollector`: General data collection
   - `HistoricalDataCollector`: NSE/Kite historical data
   - `LiveFeatureCollector`: Real-time feature extraction
   - `UnifiedFeatures`: Standardized feature definitions

2. **ml.training** - Model Training
   - `ModelTrainer`: XGBoost, LightGBM, RF, Ensemble training with Optuna-tuned weights
   - `FullPipeline`: End-to-end training (data → features → ternary labels → Optuna threshold/weight optimization)
   - `Backtester`: Walk-forward backtesting
   - Parallel sub-model training via `ThreadPoolExecutor`

3. **ml.inference** - Prediction
   - `MLPredictor`: Ternary prediction (BEARISH=0, NEUTRAL=1, BULLISH=2) with caching
   - `MLGuardrails`: Risk management and circuit breakers

4. **ml.registry** - Model Management
   - `ModelRegistry`: Save/load models, versioning, promotion

5. **ml.monitoring** - Performance Tracking
   - `ModelEvaluator`: Accuracy, Sharpe, drawdown metrics
   - `FeedbackCollector`: Log predictions and outcomes

## Configuration

All configuration is now centralized in `options_trader/config/settings.py`:

- `KITE_CONFIG`: API credentials
- `TRADING_CONFIG`: Position limits, capital
- `MARKET_HOURS`: Trading hours, expiry settings
- `ML_CONFIG`: Model training, guardrails
- `STRATEGY_CONFIG`: Enabled strategies
- `DATABASE_CONFIG`: SQLite path
- `LOGGING_CONFIG`: Log settings

## Migration Notes

1. The new `options_trader/` package uses re-exports for backwards compatibility
2. Legacy modules in root directory still work
3. Artifacts now consolidated in `artifacts/` directory
4. MLflow tracking moved to `artifacts/mlflow/`
5. Models moved to `artifacts/models/`
