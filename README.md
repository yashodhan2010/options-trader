# Options Trading Bot 🚀

A comprehensive, modular options trading bot for Indian markets using Zerodha Kite Connect API. This bot provides automated trading capabilities with support for multiple options strategies, **ML-driven signal generation**, advanced Greeks calculation, and robust risk management.

**v2.0: ML-Only Mode** - All trading signals are now derived exclusively from ML model predictions.

## Features

### 🤖 Machine Learning (Primary - ML-Only Mode)
- **ML-Driven Signals**: All entry signals come from ML predictions
- **No Rule-Based Signals**: ML is the sole source of trading decisions
- **Ternary Prediction**: BULLISH, NEUTRAL, or BEARISH with Optuna-optimized thresholds
- **Ensemble Models**: XGBoost + LightGBM + Random Forest with Optuna-tuned weights
- **34+ Features**: Price, technicals, Greeks, OI sentiment, volatility
- **Optuna Deep Integration**: Hyperparameter tuning, ensemble weight optimization, label threshold optimization, MedianPruner for early stopping
- **Parallel Training**: Concurrent sub-model training via ThreadPoolExecutor
- **Model Comparison**: Compare 12 configurations with trading-specific metrics
- **Trading Metrics**: Sharpe Ratio, Profit Factor, Win Rate, Max Drawdown
- **MLflow Tracking**: Experiment versioning and model registry
- **Risk Guardrails**: Stop-loss protection, confidence bounds, circuit breaker
- **Paper Trading Mode**: Simulation with feedback loop
- **Full Pipeline Training**: End-to-end pipeline with data collection, feature engineering, labeling, and Optuna optimization
- See [docs/ML_INTEGRATION.md](docs/ML_INTEGRATION.md) for full guide

### 📊 Strategy Catalogue
- **Directional Strategies**: Long Call, Long Put, Short Call, Short Put
- **Spread Strategies**: Bull Call Spread, Bear Put Spread
- **Neutral/Volatility Strategies**: Iron Condor, Straddle, Strangle
- Strategy selection based on ML direction and IV regime

### 📈 Market Analysis
- Real-time analysis of options metrics
- Open Interest (OI) based sentiment analysis
- Volatility regime detection
- Put-Call Ratio (PCR) analysis
- Max Pain calculation
- Historical Data Analysis for feature extraction

### 🧮 Options Pricing & Greeks
- **QuantLib-based** accurate Greeks calculation
- Implied Volatility (IV) using Newton-Raphson solver
- Full Greeks: Delta, Gamma, Theta, Vega, Rho
- Strategy-level combined Greeks
- **py_vollib** fallback for IV calculation

### ⚡ Execution
- Automated order placement via Kite Connect
- Stop Loss and Target management
- **High-Watermark Trailing Stop Loss** with activation threshold and profit protection
- Separate logging for trailing SL vs hard SL exits
- Position tracking and monitoring
- Paper trading mode for testing
- **Real-time WebSocket monitoring** for instant exit signals
- **Position persistence** for overnight recovery

### 🛡️ Risk Management
- Configurable stop loss and target percentages
- **Differentiated targets**: Higher reward for stocks vs indices
- Daily loss limits
- Maximum positions limit
- Position sizing based on risk
- **Portfolio-level Greeks monitoring**
- **Periodic status updates** (every 15 minutes, configurable)

### 📊 Statistical Upgrades
- **RSI Momentum vs Reversion**: Stocks use momentum (RSI trend-following), indices use mean-reversion
- **Liquidity Guard**: Volume, OI, and bid-ask spread checks per leg before entry
- **ATR-Based Dynamic Spread Width**: Spread widths adapt to current volatility via ATR
- **Expiry Selection**: Weekly expiry for indices, monthly expiry for stocks

### 📋 Stock Watchlist
- Pre-configured stocks with lot sizes
- AXISBANK, HDFCBANK, IDFCFIRSTB, RELIANCE, SBIN
- Easy to add/remove stocks via `config/watchlist.json`

## Project Structure

```
options-trader/
├── auth/
│   ├── __init__.py
│   └── kite_auth.py          # Kite Connect authentication
├── config/
│   ├── __init__.py
│   ├── settings.py           # Configuration settings
│   └── watchlist.json        # Stock watchlist with lot sizes
├── core/
│   ├── __init__.py
│   ├── database.py           # SQLite database for persistence
│   ├── logger.py             # Logging utilities
│   ├── notifications.py      # Notification system (Telegram ready)
│   ├── options_pricer.py     # QuantLib-based Greeks calculator
│   └── utils.py              # Helper functions
├── data/
│   ├── __init__.py
│   ├── data_fetcher.py       # Market data fetching
│   └── websocket_manager.py  # Real-time WebSocket streaming
├── docs/
│   ├── STRATEGIES.md         # Strategy documentation
│   └── ML_INTEGRATION.md     # ML module guide
├── execution/
│   ├── __init__.py
│   ├── order_manager.py      # Order placement and management
│   └── position_tracker.py   # Position monitoring with WebSocket
├── signals/
│   ├── __init__.py           # Package exports
│   ├── ml_signal_generator.py # ML-Only signal generator (PRIMARY)
│   ├── signal_generator.py   # Legacy rule-based (DEPRECATED)
│   └── exit_signal_generator.py # Exit signals
├── ml/                       # Machine Learning module
│   ├── __init__.py           # Module exports
│   ├── feature_engineer.py   # 34+ feature extraction
│   ├── unified_features.py   # Unified feature definitions
│   ├── data_collector.py     # Historical data caching
│   ├── model_trainer.py      # XGBoost/LightGBM/RF + Optuna (parallel training)
│   ├── full_pipeline.py      # End-to-end training pipeline with Optuna
│   ├── evaluator.py          # Trading metrics & model comparison
│   ├── predictor.py          # Ensemble inference (ternary: BEAR/NEUTRAL/BULL)
│   ├── backtester.py         # Historical simulation
│   ├── feedback_collector.py # Outcome logging & drift
│   ├── feedback_trainer.py   # Feedback-based retraining
│   ├── feedback_evaluator.py # Feedback evaluation
│   ├── model_registry.py     # Version control & A/B testing
│   ├── mlflow_tracker.py     # MLflow integration
│   ├── guardrails.py         # Risk management
│   ├── auto_retrain.py       # Automatic retraining triggers
│   ├── historical_data_collector.py  # Historical data fetching
│   ├── historical_trainer.py # Train on historical data
│   ├── historical_predictor.py # Historical predictions
│   ├── live_feature_collector.py # Real-time feature extraction
│   └── paper_trading_runner.py # Paper trading orchestration
├── signals/
│   ├── __init__.py
│   ├── signal_generator.py   # Signal generation (ML-enhanced)
│   └── exit_signal_generator.py # Exit signals (ML-enhanced)
├── strategies/
│   ├── __init__.py
│   ├── base_strategy.py      # Base strategy class
│   ├── catalogue.py          # Strategy registry
│   ├── directional.py        # Long/Short Call/Put
│   ├── spreads.py            # Bull/Bear Spreads
│   └── volatility.py         # Iron Condor, Straddle, Strangle
├── logs/                     # Log files directory
├── ml_models/                # Trained ML models (NEW)
├── mlruns/                   # MLflow tracking data (NEW)
├── bot.py                    # Main bot orchestrator
├── cli.py                    # Interactive CLI
├── demo_signal.py            # Demo/testing utilities
├── requirements.txt
├── .env.example
└── README.md
```

## Installation

### Prerequisites
- Python 3.9+
- Zerodha Kite Connect API subscription

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd options-trader
```

2. **Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your Kite API credentials
```

## Configuration

Edit `.env` file with your credentials:

```env
KITE_API_KEY=your_api_key
KITE_API_SECRET=your_api_secret
KITE_REDIRECT_URL=http://localhost:5000/callback
```

### Trading Configuration

Edit `config/settings.py` to customize:

- **Underlying Assets**: NIFTY, BANKNIFTY, FINNIFTY
- **Risk Parameters**: Stop loss %, Target %, Max positions (live) and paper trading cap
- **Strategy Settings**: Enabled strategies, strike selection
- **Market Hours**: Trading windows

### Market Hours Configuration

The bot respects market timing to avoid initial volatility:

```python
MARKET_HOURS = {
    "market_open": "09:15",       # NSE opens
    "trading_start": "09:30",     # Bot starts trading (15 min buffer)
    "trading_end": "15:15",       # Stop new positions
    "square_off_time": "15:20",   # Force close all positions
    "market_close": "15:30",      # NSE closes
    
    "no_trade_days": [],          # Holidays to skip
    "expiry_day_trading": True,   # Trade on expiry?
    "expiry_early_exit_time": "14:30",  # Earlier exit on expiry
}
```

| Time | Action |
|------|--------|
| 09:15 | Market opens - bot monitors only |
| **09:30** | **Trading begins** (volatility settled) |
| 15:15 | No new positions |
| 15:20 | Auto square-off all positions |
| 15:30 | Market closes |

Check current market status in CLI:
```
OptionsTrader> market
```

## Usage

### Interactive CLI Mode

```bash
python cli.py
```

Available commands:
- `login` - Login to Kite Connect
- `status` - Show account status
- `market` - Show market timing status
- `overview NIFTY` - Market overview
- `signals` - Generate trading signals
- `strategies` - List available strategies
- `trade NIFTY long_call` - Execute a specific trade
- `positions` - Show active positions
- `close all` - Close all positions
- `paper on/off` - Toggle paper trading
- `chain NIFTY` - Show options chain
- `greeks NIFTY` - Show options chain with Greeks (Delta, Gamma, Theta, Vega, IV)
- `iv NIFTY` - Show IV analysis and regime
- `history NIFTY` - Show historical trend analysis (RSI, momentum)
- `watchlist` - Show current stock watchlist
- `risk` - Show portfolio risk metrics (combined Greeks)
- `pnl` - Show today's P&L summary

**ML Commands:**
- `ml status` - Show ML model status and performance
- `ml train [SYMBOL]` - Train ML model with Optuna optimization
- `ml train-full [START_DATE END_DATE]` - Full pipeline: data collection + feature engineering + Optuna threshold/weights optimization
- `ml train-best [CONFIG]` - Train all symbols with best config (e.g., rf_aggressive)
- `ml compare SYMBOL` - Compare 12 model configurations with trading metrics
- `ml backtest SYMBOL` - Run historical backtest
- `ml paper start/stop/stats` - Paper trading session management
- `ml predict SYMBOL` - Get ML prediction for underlying
- `ml features SYMBOL` - Show extracted features
- `ml drift` - Check for model drift

### Bot Mode (Automated)

```bash
# Paper trading (default)
python bot.py --auto-trade

# Live trading
python bot.py --auto-trade --live

# Specific underlyings
python bot.py --underlyings NIFTY BANKNIFTY --auto-trade

# Market overview only
python bot.py --overview
```

## Strategies

### 1. Long Call
- **Outlook**: Bullish
- **Conditions**: Low IV, Bullish sentiment
- **Risk**: Limited (premium paid)
- **Reward**: Unlimited

### 2. Long Put
- **Outlook**: Bearish
- **Conditions**: Low IV, Bearish sentiment
- **Risk**: Limited (premium paid)
- **Reward**: Significant (until underlying reaches zero)

### 3. Short Call
- **Outlook**: Neutral to Bearish
- **Conditions**: High IV, Resistance levels
- **Risk**: Unlimited
- **Reward**: Limited (premium received)

### 4. Short Put
- **Outlook**: Neutral to Bullish
- **Conditions**: High IV, Support levels
- **Risk**: Significant
- **Reward**: Limited (premium received)

### 5. Bull Call Spread
- **Outlook**: Moderately Bullish
- **Structure**: Buy lower strike call, Sell higher strike call
- **Risk**: Limited (net debit)
- **Reward**: Limited (strike difference - debit)

### 6. Bear Put Spread
- **Outlook**: Moderately Bearish
- **Structure**: Buy higher strike put, Sell lower strike put
- **Risk**: Limited (net debit)
- **Reward**: Limited (strike difference - debit)

### 7. Iron Condor
- **Outlook**: Neutral (range-bound)
- **Structure**: Short strangle + long wings for protection
- **Risk**: Limited
- **Reward**: Limited (net credit)

### 8. Straddle
- **Outlook**: High volatility expected
- **Structure**: Buy ATM call and put
- **Risk**: Limited (total premium)
- **Reward**: Unlimited

### 9. Strangle
- **Outlook**: High volatility expected
- **Structure**: Buy OTM call and put
- **Risk**: Limited (total premium)
- **Reward**: Unlimited

## Signal Generation

The signal generator analyzes:

1. **Open Interest Data**
   - Put-Call Ratio (PCR)
   - Max Pain level
   - Max OI strikes (support/resistance)

2. **Volatility Metrics**
   - Historical Volatility (HV)
   - Implied Volatility (IV)
   - IV/HV ratio
   - Volatility regime classification

3. **Signal Generation**
   - Put-Call Ratio (PCR)
   - Max Pain level
   - Max OI strikes (support/resistance)

4. **Historical Data Analysis**
   - Trend detection (BULLISH/BEARISH/NEUTRAL)
   - RSI (Relative Strength Index)
   - Price momentum
   - Confidence boost for signals

## Options Pricing & Greeks

The bot uses **QuantLib** for accurate options pricing:

```python
from core.options_pricer import OptionsPricer

pricer = OptionsPricer()

# Calculate IV from market price
iv = pricer.calculate_iv(
    spot=24000, strike=24000, option_type="CE",
    market_price=250, time_to_expiry=0.05, risk_free_rate=0.065
)

# Calculate Greeks
greeks = pricer.calculate_greeks(
    spot=24000, strike=24000, option_type="CE",
    iv=0.15, time_to_expiry=0.05
)
# Returns: {delta, gamma, theta, vega, rho}

# Full analysis
analysis = pricer.full_analysis(
    spot=24000, strike=24000, option_type="CE",
    market_price=250, time_to_expiry=0.05
)
# Returns: theoretical_price, iv, greeks
```

## WebSocket Monitoring

Real-time price streaming for instant exit signals:

```python
# Enable WebSocket mode in config
BOT_CONFIG = {
    "use_websocket": True,  # Enable WebSocket monitoring
    "position_poll_interval": 5,  # Fallback poll interval
    "position_status_interval": 900,  # Status update interval (15 min)
    "persist_positions": True,  # Persist positions for recovery
}
```

## Stock Watchlist

Configure stocks in `config/watchlist.json`:

```json
{
    "enabled": true,
    "assets": [
        {"name": "AXISBANK", "equity_token": 1510401, "lot_size": 625},
        {"name": "HDFCBANK", "equity_token": 341249, "lot_size": 550},
        {"name": "RELIANCE", "equity_token": 738561, "lot_size": 500}
    ]
}
```

## Risk Management

### Stop Loss & Target
- Automatic SL/Target calculation based on strategy
- **High-watermark trailing stop loss**: Activates at 30% of target, protects 70% of peak profit
- Separate `TRAILING_SL_HIT` event (🟡) vs hard `SL_HIT` (🔴) for accurate P&L classification
- Differentiated targets: stocks get higher reward targets than indices (`is_index` flag)
- Customizable percentages in config

### Position Limits
- Maximum concurrent positions
- Capital allocation per trade
- Daily loss limit

### Greeks-Based Exit Management (NEW)

Dynamic exit logic using real-time Greeks:

| Exit Type | Trigger | Purpose |
|-----------|---------|---------|
| **Delta Exit** | Long: Δ < 0.10, Short: Δ > 0.90 | Exit dying or dangerous options |
| **Theta Exit** | Daily decay > 50% of remaining profit | Avoid time decay erosion |
| **Vega Exit** | IV drops > 20% from entry | Protect against IV crush |
| **Gamma Tighten** | Γ > 0.05 | Tighten SL when delta moves fast |
| **DTE Exit** | Days to expiry < 2 | Force exit before expiry |
| **Profit Lock** | Profit > 50% of target | Lock 30% of unrealized gains |

```python
# Configure in config/settings.py
GREEKS_EXIT_CONFIG = {
    "enabled": True,                    # Master switch
    "delta_exit_enabled": True,
    "min_delta_long": 0.10,             # Exit if delta falls below
    "theta_exit_enabled": True,
    "theta_decay_threshold": 0.5,       # Exit if decay > 50% of profit
    "vega_exit_enabled": True,
    "iv_drop_percent": 20,              # Exit on IV crush
    "gamma_tighten_enabled": True,
    "profit_lock_enabled": True,
}
```

Use `greeks_settings` in CLI to view/toggle:
```
OptionsTrader> greeks_settings
OptionsTrader> greeks_settings toggle
```

## API Reference

### SignalGenerator

```python
from signals import signal_generator

# Generate signals for all underlyings
signals = signal_generator.generate_signals()

# Generate for specific underlying
signals = signal_generator.generate_signals("NIFTY")

# Get market overview
overview = signal_generator.get_market_overview("NIFTY")
```

### OrderManager

```python
from execution import order_manager

# Enable paper trading
order_manager.set_paper_trading(True)

# Execute a signal
execution = order_manager.execute_signal(signal)

# Close position
order_manager.close_position(execution_id)
```

### StrategyCatalogue

```python
from strategies import create_catalogue

# Create catalogue for an underlying
catalogue = create_catalogue("NIFTY")

# Analyze with specific strategy
signal = catalogue.get_strategy(StrategyType.LONG_CALL).analyze(chain, metrics)

# Get all signals
signals = catalogue.analyze_all(chain, metrics)
```

### ML Module (NEW)

```python
from ml import (
    get_predictor,
    get_model_trainer,
    get_paper_trading_runner,
    get_feedback_collector
)

# Train ML model with Optuna optimization
trainer = get_model_trainer()
result = trainer.train_direction_model("NIFTY", optimize=True, n_trials=50)
print(f"Accuracy: {result['metrics']['accuracy']:.2%}")

# Get ML-enhanced prediction
predictor = get_predictor()
prediction = predictor.predict_with_guardrails(
    features=features,
    underlying="NIFTY",
    strategy_type="LONG_CALL",
    rule_confidence=0.72
)
print(f"Blended confidence: {prediction.blended_confidence:.2%}")

# Start paper trading session
runner = get_paper_trading_runner()
runner.start_session(initial_capital=100000)
# ... trade signals ...
summary = runner.end_session()
print(f"Win rate: {summary['win_rate']:.1%}, Sharpe: {summary['sharpe_ratio']:.2f}")

# Check model performance
feedback = get_feedback_collector()
stats = feedback.get_performance_stats(days=30)
print(f"ML Accuracy: {stats['accuracy']:.1%}")
```

📖 **Full ML documentation**: [docs/ML_INTEGRATION.md](docs/ML_INTEGRATION.md)

## Disclaimer

⚠️ **IMPORTANT**: This software is for educational purposes only. 

- Options trading involves substantial risk of loss
- Past performance does not guarantee future results
- Always paper trade first before using real money
- The authors are not responsible for any financial losses

## License

MIT License - see LICENSE file for details

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Support

For issues and feature requests, please open an issue on GitHub.