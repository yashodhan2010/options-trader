"""
Configuration settings for the Options Trading Bot
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
CONFIG_DIR = Path(__file__).parent

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Kite Connect API Configuration
KITE_CONFIG = {
    "api_key": os.getenv("KITE_API_KEY", ""),
    "api_secret": os.getenv("KITE_API_SECRET", ""),
    "redirect_url": os.getenv("KITE_REDIRECT_URL", "http://localhost:5000/callback"),
    "access_token_path": DATA_DIR / "access_token.json",
}

# Trading Configuration
TRADING_CONFIG = {
    "exchange": "NFO",  # NFO for options
    "default_quantity": 1,  # Lot size multiplier
    "max_positions": 5,
    "paper_max_positions": 15,  # Relaxed cap for paper trading (set higher than live)
    "capital_per_trade": 150000,  # INR
    "max_loss_per_day": 10000,  # INR
    "default_sl_percent": 30,  # Stop loss percentage
    "default_target_percent": 50,  # Target percentage
    "trailing_sl_enabled": True,
    "trailing_sl_percent": 30,        # Trail at 30% below peak profit (protect 70%)
    "trailing_sl_activation_pct": 0.3,  # Start trailing after 30% of target reached
}

# Market Hours Configuration
MARKET_HOURS = {
    # Market opens at 9:15 - data collection starts here
    "market_open": "09:15",             # NSE market open time (data collection active)
    "trading_start": "10:00",           # Bot starts trading (skip morning volatility noise)
    "trading_end": "15:00",             # Stop taking new positions (avoid late-day theta)
    "market_close": "15:30",            # NSE market close time
    
    # Aliases for backwards compatibility
    "start": "09:15",                   # Alias for market_open
    "end": "15:30",                     # Alias for market_close
    "pre_open_start": "09:00",          # Pre-open session start
    "pre_open_end": "09:08",            # Pre-open session end
    "no_trade_before": "10:00",         # No trades before 10 AM (morning noise)
    "no_trade_after": "14:00",          # No new trades after 2 PM
    
    # Auto-exit settings
    "auto_exit_after_close": True,      # Automatically exit bot after market closes
    "auto_exit_buffer_minutes": 5,      # Minutes after market_close to exit (15:30 + 5 = 15:35)
    
    # Overnight position handling
    "auto_square_off": False,           # Set True to force close at square_off_time
    "square_off_time": "15:20",         # Only used if auto_square_off is True
    "carry_overnight": True,            # Allow positions to be carried overnight
    
    # Day-specific settings
    "no_trade_days": [],                # List of dates to skip (holidays, special days)
    "early_close_days": [],             # Days with early market close
    
    # Initial volatility handling
    "skip_first_minutes": 105,          # Minutes after open to skip (9:15 + 105 = 11:00)
    "skip_last_minutes": 90,            # Minutes before close to stop new trades (15:30 - 90 = 14:00)
    
    # Expiry day settings
    "expiry_day_trading": True,         # Trade on expiry days?
    "expiry_day_square_off": True,      # Force square-off on expiry day (recommended)
    "expiry_early_exit_time": "14:30",  # Exit positions earlier on expiry
    
    # Monthly vs Weekly expiry settings
    "use_monthly_expiry_only": True,    # Only trade monthly expiry options (not weekly)
}

# Bot Scan Configuration
BOT_CONFIG = {
    "signal_scan_interval": 900,        # Seconds between signal scans (15 minutes for better data)
    "position_poll_interval": 5,        # Seconds between position checks (exit monitoring)
    "position_status_interval": 900,    # Seconds between status updates (15 minutes = 900s)
    "max_signals_per_scan": 3,          # Maximum signals to generate per scan
    "min_signal_gap_minutes": 15,       # Minimum gap between signals for same underlying
    "use_websocket": True,              # Use WebSocket for real-time exit monitoring
    "persist_positions": True,          # Save positions to database for overnight recovery
    
    # Signal-based intelligent exit system
    "signal_exit_enabled": True,        # Enable reversal/thesis-based exits
    "signal_exit_interval": 60,         # Seconds between signal exit checks (less frequent, more expensive)
    "signal_exit_min_confidence": 0.70, # Minimum confidence to trigger exit
}

# Greeks-Based Exit Configuration (NEW)
GREEKS_EXIT_CONFIG = {
    "enabled": True,                    # Enable Greeks-based exit logic
    
    # Delta-based exits
    "delta_exit_enabled": True,
    "min_delta_long": 0.10,             # Exit long options if delta falls below this
    "max_delta_short": 0.90,            # Exit short options if delta rises above this
    
    # Theta-based exits (time decay management)
    "theta_exit_enabled": True,
    "theta_decay_threshold": 0.5,       # Exit if daily theta > 50% of remaining profit potential
    "days_to_expiry_exit": 2,           # Force exit when DTE falls below this
    
    # Vega-based exits (IV crush protection)
    "vega_exit_enabled": True,
    "iv_drop_percent": 20,              # Exit if IV drops more than 20% from entry
    
    # Gamma-based stop tightening
    "gamma_tighten_enabled": True,
    "gamma_threshold": 0.05,            # When gamma > threshold, tighten stops
    "gamma_sl_tighten_percent": 20,     # Reduce SL distance by 20% when gamma is high
    
    # Combined profit protection
    "profit_lock_enabled": True,
    "profit_lock_threshold": 0.5,       # When profit reaches 50% of target
    "profit_lock_percent": 0.3,         # Lock 30% of unrealized profit as new floor
}

# Underlying Assets Configuration
UNDERLYING_ASSETS = {
    "NIFTY": {
        "symbol": "NIFTY 50",
        "exchange": "NSE",
        "options_exchange": "NFO",  # NSE F&O
        "lot_size": 25,
        "tick_size": 0.05,
        "strike_interval": 50,
        "expiry_day": "Thursday",  # Weekly expiry
        "instrument_token": 256265,  # NSE NIFTY 50 index token
    },
    "BANKNIFTY": {
        "symbol": "NIFTY BANK",
        "exchange": "NSE",
        "options_exchange": "NFO",  # NSE F&O
        "lot_size": 15,
        "tick_size": 0.05,
        "strike_interval": 100,
        "expiry_day": "Wednesday",  # Weekly expiry
        "instrument_token": 260105,  # NSE NIFTY BANK index token
    },
    "FINNIFTY": {
        "symbol": "NIFTY FIN SERVICE",
        "exchange": "NSE",
        "options_exchange": "NFO",  # NSE F&O
        "lot_size": 25,
        "tick_size": 0.05,
        "strike_interval": 50,
        "expiry_day": "Tuesday",
        "instrument_token": 257801,  # NSE NIFTY FIN SERVICE index token
    },
    "SENSEX": {
        "symbol": "SENSEX",
        "exchange": "BSE",
        "options_exchange": "BFO",  # BSE F&O
        "lot_size": 10,
        "tick_size": 0.05,
        "strike_interval": 100,
        "expiry_day": "Friday",  # Weekly expiry
        "instrument_token": 265,  # BSE SENSEX index token
    },
}

# Options Metrics Thresholds
METRICS_CONFIG = {
    "oi_change_threshold": 10,  # Percentage change in OI
    "volume_spike_multiplier": 2,  # Volume spike detection
    "iv_percentile_high": 80,  # High IV percentile
    "iv_percentile_low": 20,  # Low IV percentile
    "pcr_bullish_threshold": 0.7,  # Put-Call Ratio for bullish
    "pcr_bearish_threshold": 1.3,  # Put-Call Ratio for bearish
    "max_bid_ask_spread": 5,  # Maximum bid-ask spread
}

# Strategy Configuration
STRATEGY_CONFIG = {
    "enabled_strategies": [
        "long_call",
        "long_put",
        "short_call",
        "short_put",
        "bull_call_spread",
        "bear_put_spread",
        "bear_call_spread",   # Credit spread - SELL call with BUY hedge
        "bull_put_spread",    # Credit spread - SELL put with BUY hedge
        "iron_condor",
        "straddle",
        "strangle",
    ],
    "default_days_to_expiry": 7,
    "min_days_to_expiry": 2,
    "max_days_to_expiry": 30,
    "strike_selection_mode": "atm",  # atm, otm, itm
    "otm_offset": 1,  # Number of strikes OTM
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_path": LOGS_DIR / "trading_bot.log",
    "max_bytes": 10485760,  # 10MB
    "backup_count": 5,
}

# Database Configuration (SQLite for simplicity)
DATABASE_CONFIG = {
    "path": DATA_DIR / "trading_bot.db",
}

# ============================================================================
# ML CONFIGURATION
# ============================================================================

# ML Models Directory
ML_MODELS_DIR = DATA_DIR / "ml_models"
ML_MODELS_DIR.mkdir(exist_ok=True)

ML_CONFIG = {
    # Master switch
    "enabled": os.getenv("ML_ENABLED", "true").lower() == "true",
    
    # =========================================================================
    # TRAINING SYMBOLS - Add your 5-6 symbols here for ML training
    # =========================================================================
    # These are the symbols the ML model will be trained on
    # Can be indices (from UNDERLYING_ASSETS) or stocks (from watchlist)
    "training_symbols": [
        "NIFTY",           # Index (NFO)
        "BANKNIFTY",       # Index (NFO)
        "SENSEX",          # Index (BFO)
        "AXISBANK",        # Stock
        "HDFCBANK",        # Stock
        "RELIANCE",        # Stock
        "SBIN",            # Stock
    ],
    
    # Model paths
    "model_path": ML_MODELS_DIR,
    "mlflow_tracking_uri": str(BASE_DIR / "mlruns"),
    "mlflow_enabled": os.getenv("MLFLOW_ENABLED", "true").lower() == "true",
    
    # Confidence blending
    "confidence_weight": 0.5,           # Weight of ML vs rule-based (0.5 = equal)
    "min_confidence_for_trade": 0.50,   # Minimum blended confidence to take trade (lowered for testing)
    
    # Model training
    "model_type": "ensemble",           # 'xgboost', 'lightgbm', 'rf', 'ensemble'
    "retrain_interval_days": 7,         # Days between model retraining
    "min_training_samples": 100,        # Minimum samples needed for training
    "validation_split": 0.2,            # Validation set size
    "walk_forward_splits": 5,           # Number of walk-forward splits
    
    # Optuna optimization
    "optuna_trials": 50,                # Number of Optuna trials
    "optuna_timeout": 3600,             # Max seconds for optimization
    "optuna_pruning": True,             # Enable early stopping of bad trials
    
    # Ensemble weights (if using ensemble)
    "ensemble_weights": {
        "xgboost": 0.5,
        "lightgbm": 0.3,
        "random_forest": 0.2,
    },
    
    # Feature engineering
    "feature_set": "full",              # 'minimal', 'standard', 'full'
    "lookback_periods": [5, 10, 20, 50],
    "normalize_features": True,
    
    # Prediction caching
    "prediction_cache_seconds": 60,     # Cache predictions for this long
    
    # Data collection
    "historical_days": 180,             # Days of historical data (Kite allows up to 2000 for daily)
    "data_update_interval": 86400,      # Seconds between data updates (1 day)
    
    # Guardrails (risk management)
    "guardrails": {
        "max_confidence_adjustment": 0.3,    # Max ML can adjust confidence ±
        "min_ml_confidence": 0.4,            # Block trade if ML confidence below
        "max_model_age_days": 14,            # Use rule-based if model older
        "daily_loss_threshold_percent": 5.0, # Pause ML if daily loss exceeds
        "drawdown_circuit_breaker_percent": 10.0,  # Emergency stop
    },
    
    # Paper trading
    "paper_trading": {
        "enabled": True,
        "duration_days": 30,                 # Paper trading duration
        "min_trades_for_promotion": 20,      # Min trades before model promotion
        "min_accuracy_for_promotion": 0.55,  # Min accuracy to promote model
        "min_sharpe_for_promotion": 1.0,     # Min Sharpe ratio to promote
    },
    
    # Feedback loop
    "feedback": {
        "log_all_predictions": True,         # Log every prediction to DB
        "log_features_at_entry": True,       # Store features when trade opens
        "log_features_at_exit": True,        # Store features when trade closes
        "drift_detection_enabled": True,     # Monitor for model drift
        "drift_threshold": 0.1,              # Retrain if accuracy drops by this
    },
    
    # Auto-Retraining from Feedback
    "auto_retrain": {
        "enabled": True,                     # Enable automatic retraining
        "min_samples": 50,                   # Minimum trade outcomes to retrain
        "interval_days": 7,                  # Retrain every N days if enough data
        "drift_threshold": 0.15,             # Accuracy drop threshold for retrain
        "auto_promote": False,               # Auto-promote if accuracy > 55%
        "use_feedback_target": True,         # Use trade P&L as target (not next-day return)
        "check_interval_seconds": 3600,      # Background check interval (1 hour)
    },
}

# Notification Configuration
NOTIFICATION_CONFIG = {
    "telegram_enabled": os.getenv("TELEGRAM_ENABLED", "false").lower() == "true",
    "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
    "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID", ""),
    # WhatsApp via CallMeBot (free)
    "whatsapp_enabled": os.getenv("WHATSAPP_ENABLED", "false").lower() == "true",
    "whatsapp_phone": os.getenv("WHATSAPP_PHONE", ""),
    "whatsapp_apikey": os.getenv("WHATSAPP_APIKEY", ""),
}


# ============================================================================
# WATCHLIST CONFIGURATION
# ============================================================================

def load_watchlist() -> Dict:
    """Load watchlist from JSON file."""
    watchlist_path = CONFIG_DIR / "watchlist.json"
    if watchlist_path.exists():
        with open(watchlist_path, "r") as f:
            return json.load(f)
    return {"enabled": False, "assets": []}


def get_watchlist_assets() -> List[Dict]:
    """Get list of enabled assets from watchlist."""
    watchlist = load_watchlist()
    if not watchlist.get("enabled", False):
        return []
    return [asset for asset in watchlist.get("assets", []) if asset.get("enabled", True)]


def get_watchlist_symbols() -> List[str]:
    """Get list of symbol names from watchlist."""
    return [asset["name"] for asset in get_watchlist_assets()]


def get_asset_by_name(name: str) -> Optional[Dict]:
    """Get asset details by name."""
    for asset in get_watchlist_assets():
        if asset["name"].upper() == name.upper():
            return asset
    return None


def get_instrument_token(name: str) -> Optional[int]:
    """
    Get instrument token for a symbol.
    For watchlist stocks, returns equity_token (for historical data).
    For indices, returns the instrument_token from UNDERLYING_ASSETS.
    """
    # Check indices first
    if name in UNDERLYING_ASSETS:
        return UNDERLYING_ASSETS[name].get("instrument_token")
    
    # Check watchlist (returns equity_token for stocks)
    asset = get_asset_by_name(name)
    if asset:
        return asset.get("equity_token") or asset.get("instrument_token")
    
    return None


def get_equity_token(name: str) -> Optional[int]:
    """
    Get equity (NSE) token for historical data analysis.
    
    Args:
        name: Symbol name
        
    Returns:
        Equity instrument token for NSE
    """
    asset = get_asset_by_name(name)
    if asset:
        return asset.get("equity_token")
    return None


def is_in_watchlist(name: str) -> bool:
    """Check if a symbol is in the enabled watchlist."""
    watchlist = load_watchlist()
    if not watchlist.get("enabled", False):
        return True  # If watchlist disabled, allow all
    return name.upper() in [a["name"].upper() for a in get_watchlist_assets()]


def get_lot_size(name: str) -> int:
    """
    Get lot size for an underlying.
    Checks UNDERLYING_ASSETS first (for indices), then watchlist (for stocks).
    
    Args:
        name: Symbol name
        
    Returns:
        Lot size (defaults to 25 if not found)
    """
    # Check indices first
    if name in UNDERLYING_ASSETS:
        return UNDERLYING_ASSETS[name].get("lot_size", 25)
    
    # Check watchlist
    asset = get_asset_by_name(name)
    if asset:
        return asset.get("lot_size", 25)
    
    return 25  # Default


def get_options_exchange(name: str) -> str:
    """
    Get the F&O exchange for an underlying.
    SENSEX options trade on BFO (BSE F&O), everything else on NFO (NSE F&O).
    
    Args:
        name: Symbol name
        
    Returns:
        Exchange string ('NFO' or 'BFO')
    """
    if name in UNDERLYING_ASSETS:
        return UNDERLYING_ASSETS[name].get("options_exchange", "NFO")
    
    # Check watchlist for stock-specific exchange override
    asset = get_asset_by_name(name)
    if asset:
        return asset.get("options_exchange", asset.get("exchange", "NFO"))
    
    return "NFO"  # Default


def get_strike_interval(name: str) -> int:
    """
    Get strike interval for an underlying.
    
    Args:
        name: Symbol name
        
    Returns:
        Strike interval (defaults to 50)
    """
    if name in UNDERLYING_ASSETS:
        return UNDERLYING_ASSETS[name].get("strike_interval", 50)
    return 50  # Default for stocks


# Load watchlist on module import
WATCHLIST = load_watchlist()
WATCHLIST_SYMBOLS = get_watchlist_symbols()
