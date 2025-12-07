# Options Trading Bot 🚀

A comprehensive, modular options trading bot for Indian markets using Zerodha Kite Connect API. This bot provides automated trading capabilities with support for multiple options strategies, real-time signal generation, advanced Greeks calculation, and robust risk management.

## Features

### 📊 Strategy Catalogue
- **Directional Strategies**: Long Call, Long Put, Short Call, Short Put
- **Spread Strategies**: Bull Call Spread, Bear Put Spread
- **Neutral/Volatility Strategies**: Iron Condor, Straddle, Strangle

### 📈 Signal Generation
- Real-time analysis of options metrics
- Open Interest (OI) based sentiment analysis
- Volatility regime detection
- Put-Call Ratio (PCR) analysis
- Max Pain calculation
- **Historical Data Analysis** for confidence scoring (trend, RSI, momentum)

### 🧮 Options Pricing & Greeks (NEW)
- **QuantLib-based** accurate Greeks calculation
- Implied Volatility (IV) using Newton-Raphson solver
- Full Greeks: Delta, Gamma, Theta, Vega, Rho
- Strategy-level combined Greeks
- **py_vollib** fallback for IV calculation

### ⚡ Execution
- Automated order placement via Kite Connect
- Stop Loss and Target management
- Trailing Stop Loss support
- Position tracking and monitoring
- Paper trading mode for testing
- **Real-time WebSocket monitoring** for instant exit signals
- **Position persistence** for overnight recovery

### 🛡️ Risk Management
- Configurable stop loss and target percentages
- Daily loss limits
- Maximum positions limit
- Position sizing based on risk
- **Portfolio-level Greeks monitoring**
- **Periodic status updates** (every 15 minutes, configurable)

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
├── execution/
│   ├── __init__.py
│   ├── order_manager.py      # Order placement and management
│   └── position_tracker.py   # Position monitoring with WebSocket
├── signals/
│   ├── __init__.py
│   └── signal_generator.py   # Signal generation
├── strategies/
│   ├── __init__.py
│   ├── base_strategy.py      # Base strategy class
│   ├── catalogue.py          # Strategy registry
│   ├── directional.py        # Long/Short Call/Put
│   ├── spreads.py            # Bull/Bear Spreads
│   └── volatility.py         # Iron Condor, Straddle, Strangle
├── logs/                     # Log files directory
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
- **Risk Parameters**: Stop loss %, Target %, Max positions
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
- `market` - **Show market timing status** (NEW)
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
- Trailing stop loss for profit protection
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