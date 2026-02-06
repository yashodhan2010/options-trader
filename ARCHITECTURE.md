# Options Trader Architecture

## New Package Structure (v2.0)

The codebase has been reorganized into a clean, modular MLOps architecture.

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

### Sub-packages

1. **ml.data** - Data Collection & Feature Engineering
   - `FeatureEngineer`: Extract features from market data
   - `DataCollector`: General data collection
   - `HistoricalDataCollector`: NSE/Kite historical data
   - `LiveFeatureCollector`: Real-time feature extraction

2. **ml.training** - Model Training
   - `ModelTrainer`: XGBoost, LightGBM, RF, Ensemble training
   - `Backtester`: Walk-forward backtesting

3. **ml.inference** - Prediction
   - `MLPredictor`: Make predictions with caching
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
