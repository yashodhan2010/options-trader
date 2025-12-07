"""
Utility functions for the Options Trading Bot
"""
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import json
from pathlib import Path


def get_expiry_date(underlying: str, days_to_expiry: Optional[int] = None) -> datetime:
    """
    Calculate the next expiry date for an underlying asset.
    
    Args:
        underlying: The underlying asset (NIFTY, BANKNIFTY, etc.)
        days_to_expiry: Specific days to expiry (optional)
        
    Returns:
        The next expiry date
    """
    from config.settings import UNDERLYING_ASSETS
    
    today = datetime.now()
    expiry_days = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
    }
    
    asset_config = UNDERLYING_ASSETS.get(underlying, {})
    expiry_day = asset_config.get("expiry_day", "Thursday")
    target_day = expiry_days.get(expiry_day, 3)
    
    current_day = today.weekday()
    days_ahead = target_day - current_day
    
    if days_ahead <= 0:
        days_ahead += 7
    
    if days_to_expiry is not None:
        # Find expiry closest to requested days
        expiry = today + timedelta(days=days_ahead)
        while (expiry - today).days < days_to_expiry:
            expiry += timedelta(days=7)
        return expiry
    
    return today + timedelta(days=days_ahead)


def get_strike_price(
    spot_price: float,
    underlying: str,
    option_type: str,
    strike_mode: str = "atm",
    offset: int = 0,
) -> float:
    """
    Calculate the appropriate strike price based on mode.
    
    Args:
        spot_price: Current spot price
        underlying: The underlying asset
        option_type: 'CE' or 'PE'
        strike_mode: 'atm', 'otm', 'itm'
        offset: Number of strikes to offset
        
    Returns:
        The calculated strike price
    """
    from config.settings import UNDERLYING_ASSETS
    
    # Get strike interval based on underlying
    strike_intervals = {
        "NIFTY": 50,
        "BANKNIFTY": 100,
        "FINNIFTY": 50,
    }
    
    interval = strike_intervals.get(underlying, 50)
    atm_strike = round(spot_price / interval) * interval
    
    if strike_mode == "atm":
        return atm_strike
    elif strike_mode == "otm":
        if option_type == "CE":
            return atm_strike + (offset * interval)
        else:
            return atm_strike - (offset * interval)
    elif strike_mode == "itm":
        if option_type == "CE":
            return atm_strike - (offset * interval)
        else:
            return atm_strike + (offset * interval)
    
    return atm_strike


def format_option_symbol(
    underlying: str,
    expiry: datetime,
    strike: float,
    option_type: str,
) -> str:
    """
    Format the option trading symbol.
    
    Args:
        underlying: The underlying asset
        expiry: Expiry date
        strike: Strike price
        option_type: 'CE' or 'PE'
        
    Returns:
        Formatted option symbol
    """
    # Format: NIFTY23DEC21500CE
    expiry_str = expiry.strftime("%y%b").upper()
    day = expiry.strftime("%d")
    
    return f"{underlying}{expiry_str}{day}{int(strike)}{option_type}"


def calculate_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float = 0.05,
    option_type: str = "CE",
) -> Dict[str, float]:
    """
    Calculate option Greeks using Black-Scholes model.
    
    Args:
        spot: Current spot price
        strike: Strike price
        time_to_expiry: Time to expiry in years
        volatility: Implied volatility
        risk_free_rate: Risk-free interest rate
        option_type: 'CE' or 'PE'
        
    Returns:
        Dictionary containing Delta, Gamma, Theta, Vega
    """
    import math
    from scipy.stats import norm
    
    if time_to_expiry <= 0:
        return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0}
    
    d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * math.sqrt(time_to_expiry))
    d2 = d1 - volatility * math.sqrt(time_to_expiry)
    
    # Delta
    if option_type == "CE":
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1
    
    # Gamma
    gamma = norm.pdf(d1) / (spot * volatility * math.sqrt(time_to_expiry))
    
    # Theta (per day)
    theta_part1 = -(spot * norm.pdf(d1) * volatility) / (2 * math.sqrt(time_to_expiry))
    if option_type == "CE":
        theta = (theta_part1 - risk_free_rate * strike * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)) / 365
    else:
        theta = (theta_part1 + risk_free_rate * strike * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2)) / 365
    
    # Vega (for 1% change in volatility)
    vega = spot * norm.pdf(d1) * math.sqrt(time_to_expiry) / 100
    
    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta, 4),
        "vega": round(vega, 4),
    }


def save_json(data: Any, filepath: Path) -> None:
    """Save data to JSON file."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: Path) -> Any:
    """Load data from JSON file."""
    if filepath.exists():
        with open(filepath, "r") as f:
            return json.load(f)
    return None


def is_market_open() -> bool:
    """Check if the market is currently open (9:15 - 15:30)."""
    from config.settings import MARKET_HOURS
    
    now = datetime.now()
    
    # Check if weekend
    if now.weekday() >= 5:
        return False
    
    # Check holiday list
    today_str = now.strftime("%Y-%m-%d")
    if today_str in MARKET_HOURS.get("no_trade_days", []):
        return False
    
    market_open = datetime.strptime(MARKET_HOURS.get("market_open", "09:15"), "%H:%M").time()
    market_close = datetime.strptime(MARKET_HOURS.get("market_close", "15:30"), "%H:%M").time()
    
    return market_open <= now.time() <= market_close


def is_trading_allowed() -> bool:
    """
    Check if NEW trades are allowed based on configured restrictions.
    This is different from is_market_open() - we may want to stop new trades
    before market close to manage positions.
    """
    from config.settings import MARKET_HOURS
    
    if not is_market_open():
        return False
    
    now = datetime.now()
    
    # Trading window (after initial volatility settles, before close)
    trading_start = datetime.strptime(MARKET_HOURS.get("trading_start", "09:30"), "%H:%M").time()
    trading_end = datetime.strptime(MARKET_HOURS.get("trading_end", "15:15"), "%H:%M").time()
    
    return trading_start <= now.time() <= trading_end


def should_square_off() -> bool:
    """
    Check if it's time to consider square off.
    Note: This just checks the time - actual square off decision 
    is made in bot.py based on auto_square_off and expiry settings.
    """
    from config.settings import MARKET_HOURS
    
    now = datetime.now()
    
    # Check if weekend (shouldn't have positions anyway)
    if now.weekday() >= 5:
        return False
    
    # Check if market is even open
    if not is_market_open():
        return False
    
    # Use expiry time if it's expiry day, otherwise regular square off time
    if is_expiry_day():
        square_off_time = datetime.strptime(
            MARKET_HOURS.get("expiry_early_exit_time", "14:30"), "%H:%M"
        ).time()
    else:
        square_off_time = datetime.strptime(
            MARKET_HOURS.get("square_off_time", "15:20"), "%H:%M"
        ).time()
    
    return now.time() >= square_off_time


def is_expiry_day(underlying: str = None) -> bool:
    """Check if today is an expiry day."""
    from config.settings import UNDERLYING_ASSETS
    
    today = datetime.now()
    day_name = today.strftime("%A")  # Monday, Tuesday, etc.
    
    if underlying:
        asset_config = UNDERLYING_ASSETS.get(underlying, {})
        expiry_day = asset_config.get("expiry_day", "Thursday")
        return day_name == expiry_day
    
    # Check if it's expiry for any underlying
    for asset, config in UNDERLYING_ASSETS.items():
        if day_name == config.get("expiry_day", "Thursday"):
            return True
    
    return False


def get_time_to_market_open() -> Optional[timedelta]:
    """Get time remaining until market opens (trading start, not 9:15)."""
    from config.settings import MARKET_HOURS
    
    now = datetime.now()
    
    # If weekend, return None
    if now.weekday() >= 5:
        return None
    
    trading_start = datetime.strptime(MARKET_HOURS.get("trading_start", "09:30"), "%H:%M")
    trading_start_today = now.replace(
        hour=trading_start.hour, 
        minute=trading_start.minute, 
        second=0, 
        microsecond=0
    )
    
    if now.time() < trading_start.time():
        return trading_start_today - now
    
    return None  # Market already open


def get_time_to_market_close() -> Optional[timedelta]:
    """Get time remaining until market closes."""
    from config.settings import MARKET_HOURS
    
    now = datetime.now()
    
    if now.weekday() >= 5:
        return None
    
    market_close = datetime.strptime(MARKET_HOURS.get("market_close", "15:30"), "%H:%M")
    market_close_today = now.replace(
        hour=market_close.hour,
        minute=market_close.minute,
        second=0,
        microsecond=0
    )
    
    if now.time() < market_close.time():
        return market_close_today - now
    
    return None  # Market already closed


def get_market_status() -> Dict[str, Any]:
    """Get comprehensive market status."""
    from config.settings import MARKET_HOURS
    
    now = datetime.now()
    
    status = {
        "current_time": now.strftime("%H:%M:%S"),
        "is_weekend": now.weekday() >= 5,
        "is_market_open": is_market_open(),
        "is_trading_allowed": is_trading_allowed(),
        "should_square_off": should_square_off(),
        "is_expiry_day": is_expiry_day(),
    }
    
    # Add timing info
    if not status["is_market_open"]:
        time_to_open = get_time_to_market_open()
        if time_to_open:
            status["time_to_open"] = str(time_to_open).split('.')[0]
            status["status_message"] = f"Market opens in {status['time_to_open']}"
        else:
            status["status_message"] = "Market closed"
    elif not status["is_trading_allowed"]:
        trading_start = MARKET_HOURS.get("trading_start", "09:30")
        trading_end = MARKET_HOURS.get("trading_end", "15:15")
        
        if now.time() < datetime.strptime(trading_start, "%H:%M").time():
            status["status_message"] = f"Waiting for initial volatility to settle (trading starts at {trading_start})"
        else:
            status["status_message"] = f"No new trades after {trading_end}"
    elif status["should_square_off"]:
        status["status_message"] = "⚠️ Square off time - closing positions"
    else:
        time_to_close = get_time_to_market_close()
        if time_to_close:
            status["time_to_close"] = str(time_to_close).split('.')[0]
        status["status_message"] = "✅ Trading active"
    
    return status


def calculate_position_size(
    capital: float,
    risk_per_trade: float,
    stop_loss_points: float,
    lot_size: int,
) -> int:
    """
    Calculate the number of lots based on risk management.
    
    Args:
        capital: Available capital
        risk_per_trade: Maximum risk per trade (percentage)
        stop_loss_points: Stop loss in points
        lot_size: Lot size for the underlying
        
    Returns:
        Number of lots to trade
    """
    risk_amount = capital * (risk_per_trade / 100)
    risk_per_lot = stop_loss_points * lot_size
    
    if risk_per_lot <= 0:
        return 1
    
    lots = int(risk_amount / risk_per_lot)
    return max(1, lots)
