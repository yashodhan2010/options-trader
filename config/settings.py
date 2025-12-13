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
    "trailing_sl_percent": 10,
}

# Market Hours Configuration
MARKET_HOURS = {
    # Market opens at 9:15 but we wait for initial volatility to settle
    "market_open": "09:15",             # NSE market open time
    "trading_start": "09:30",           # Bot starts trading (after initial volatility)
    "trading_end": "15:15",             # Stop taking new positions (15 min before close)
    "market_close": "15:30",            # NSE market close time
    
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
    "skip_first_minutes": 15,           # Minutes after open to skip (9:15 + 15 = 9:30)
    "skip_last_minutes": 15,            # Minutes before close to stop new trades
    
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
        "lot_size": 25,
        "tick_size": 0.05,
        "expiry_day": "Thursday",  # Weekly expiry
    },
    "BANKNIFTY": {
        "symbol": "NIFTY BANK",
        "exchange": "NSE",
        "lot_size": 15,
        "tick_size": 0.05,
        "expiry_day": "Wednesday",  # Weekly expiry
    },
    "FINNIFTY": {
        "symbol": "NIFTY FIN SERVICE",
        "exchange": "NSE",
        "lot_size": 25,
        "tick_size": 0.05,
        "expiry_day": "Tuesday",
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

# Market Hours
MARKET_HOURS = {
    "start": "09:15",
    "end": "15:30",
    "pre_open_start": "09:00",
    "pre_open_end": "09:08",
    "no_trade_before": "09:20",  # Avoid first 5 minutes
    "no_trade_after": "15:15",  # Avoid last 15 minutes
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

# Notification Configuration
NOTIFICATION_CONFIG = {
    "telegram_enabled": os.getenv("TELEGRAM_ENABLED", "false").lower() == "true",
    "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
    "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID", ""),
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


# Load watchlist on module import
WATCHLIST = load_watchlist()
WATCHLIST_SYMBOLS = get_watchlist_symbols()
