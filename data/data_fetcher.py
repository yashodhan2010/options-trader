"""
Data Fetcher - Fetch market data, options chain, and metrics from Kite
Enhanced with Greeks and IV calculations using QuantLib.
"""
from datetime import datetime, timedelta, date
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import threading
import time
from collections import defaultdict

from auth.kite_auth import get_kite, is_authenticated
from core.logger import logger
from core.options_pricer import options_pricer, OptionPriceResult
from config.settings import (
    UNDERLYING_ASSETS, METRICS_CONFIG,
    get_asset_by_name, get_instrument_token, is_in_watchlist,
    get_options_exchange, get_strike_interval
)


class DataFetcher:
    """
    Fetches market data, options chain, and calculates metrics.
    Includes Greeks and IV calculations via QuantLib/py_vollib.
    """
    
    def __init__(self):
        self.kite = None
        self._instruments_cache: Dict[str, List[dict]] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._options_chain_cache: Dict[str, pd.DataFrame] = {}
        
        # API rate limiter: Kite allows ~3 req/sec for most endpoints
        self._rate_limit_lock = threading.Lock()
        self._api_call_times: list = []  # Timestamps of recent API calls
        self._max_calls_per_second = 3  # Kite Connect limit
    
    def _ensure_connected(self) -> bool:
        """Ensure Kite connection is established."""
        if not self.kite:
            if is_authenticated():
                self.kite = get_kite()
        return self.kite is not None
    
    def _load_instruments(self, exchange: str = "NFO") -> List[dict]:
        """
        Load instruments from Kite.
        
        Args:
            exchange: Exchange to fetch instruments for
            
        Returns:
            List of instruments
        """
        cache_key = exchange
        now = datetime.now()
        
        # Use cache if less than 1 hour old
        if (
            cache_key in self._instruments_cache
            and self._cache_timestamp
            and (now - self._cache_timestamp).seconds < 3600
        ):
            return self._instruments_cache[cache_key]
        
        if not self._ensure_connected():
            logger.error("Not connected to Kite")
            return []
        
        try:
            instruments = self.kite.instruments(exchange)
            self._instruments_cache[cache_key] = instruments
            self._cache_timestamp = now
            logger.info(f"Loaded {len(instruments)} instruments from {exchange}")
            return instruments
        except Exception as e:
            logger.error(f"Failed to load instruments: {e}")
            return []
    
    def get_spot_price(self, underlying: str) -> Optional[float]:
        """
        Get current spot price for an underlying.
        
        Args:
            underlying: The underlying asset (NIFTY, BANKNIFTY, etc.)
            
        Returns:
            Current spot price
        """
        if not self._ensure_connected():
            return None
        
        try:
            # Check if it's in UNDERLYING_ASSETS (indices)
            asset_config = UNDERLYING_ASSETS.get(underlying, {})
            
            if asset_config:
                # Index asset
                symbol = asset_config.get("symbol", underlying)
                exchange = asset_config.get("exchange", "NSE")
            else:
                # Check watchlist (stocks)
                watchlist_asset = get_asset_by_name(underlying)
                if watchlist_asset:
                    symbol = underlying
                    exchange = "NSE"
                else:
                    symbol = underlying
                    exchange = "NSE"
            
            self._throttle_api_call()
            quote = self.kite.quote(f"{exchange}:{symbol}")
            return quote[f"{exchange}:{symbol}"]["last_price"]
        except Exception as e:
            logger.error(f"Failed to get spot price for {underlying}: {e}")
            return None
    
    def _throttle_api_call(self) -> None:
        """Rate-limit API calls to stay within Kite's 3 req/sec limit."""
        with self._rate_limit_lock:
            now = time.monotonic()
            # Remove timestamps older than 1 second
            self._api_call_times = [t for t in self._api_call_times if now - t < 1.0]
            
            if len(self._api_call_times) >= self._max_calls_per_second:
                # Wait until the oldest call in the window expires
                sleep_time = 1.0 - (now - self._api_call_times[0]) + 0.05  # 50ms buffer
                if sleep_time > 0:
                    logger.debug(f"[RATE_LIMIT] Throttling API call for {sleep_time:.3f}s")
                    time.sleep(sleep_time)
            
            self._api_call_times.append(time.monotonic())
    
    def get_ltp(self, symbol: str, exchange: str = "NFO") -> Optional[float]:
        """
        Get last traded price for a symbol.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange (auto-detected for SENSEX options → BFO)
            
        Returns:
            Last traded price
        """
        if not self._ensure_connected():
            return None
        
        try:
            # Auto-detect BFO exchange for SENSEX options
            if exchange == "NFO" and symbol.startswith("SENSEX"):
                exchange = "BFO"
            
            self._throttle_api_call()
            quote = self.kite.ltp(f"{exchange}:{symbol}")
            return quote[f"{exchange}:{symbol}"]["last_price"]
        except Exception as e:
            logger.error(f"Failed to get LTP for {symbol}: {e}")
            return None
    
    def get_ltp_batch(self, symbols: List[str], exchange: str = "NFO") -> Dict[str, Optional[float]]:
        """
        Get LTP for multiple symbols in a SINGLE API call.
        Kite's ltp() supports up to ~500 instruments per call.
        
        Args:
            symbols: List of trading symbols
            exchange: Default exchange (auto-detected for SENSEX → BFO)
            
        Returns:
            Dict mapping symbol → last_price (None if unavailable)
        """
        result: Dict[str, Optional[float]] = {s: None for s in symbols}
        
        if not symbols or not self._ensure_connected():
            return result
        
        try:
            # Build exchange-qualified keys, auto-detect BFO for SENSEX
            qualified = []
            for sym in symbols:
                ex = exchange
                if ex == "NFO" and sym.startswith("SENSEX"):
                    ex = "BFO"
                qualified.append(f"{ex}:{sym}")
            
            # Kite allows batching in a single ltp() call
            self._throttle_api_call()
            quotes = self.kite.ltp(*qualified)
            
            for sym, qkey in zip(symbols, qualified):
                if qkey in quotes:
                    result[sym] = quotes[qkey].get("last_price")
            
            logger.debug(f"[BATCH_LTP] Fetched {len(quotes)}/{len(symbols)} prices in 1 API call")
        except Exception as e:
            logger.error(f"Failed batch LTP for {len(symbols)} symbols: {e}")
        
        return result
    
    def get_quote(self, symbol: str, exchange: str = "NFO") -> Optional[dict]:
        """
        Get detailed quote for a symbol.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange (auto-detected for SENSEX options → BFO)
            
        Returns:
            Quote dictionary
        """
        if not self._ensure_connected():
            return None
        
        try:
            # Auto-detect BFO exchange for SENSEX options
            if exchange == "NFO" and symbol.startswith("SENSEX"):
                exchange = "BFO"
            
            self._throttle_api_call()
            quote = self.kite.quote(f"{exchange}:{symbol}")
            return quote[f"{exchange}:{symbol}"]
        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            return None
    
    def get_instrument_token(self, symbol: str, exchange: str = "NFO") -> Optional[int]:
        """
        Get instrument token for a trading symbol.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange (NFO for options, NSE for stocks, BFO for BSE F&O)
            
        Returns:
            Instrument token or None
        """
        # Auto-detect BFO exchange for SENSEX options
        if exchange == "NFO" and symbol.startswith("SENSEX"):
            exchange = "BFO"
        
        # Check watchlist first
        watchlist_asset = get_asset_by_name(symbol)
        if watchlist_asset:
            token = watchlist_asset.get("instrument_token")
            if token:
                return token
        
        # Check underlying assets
        for asset_name, config in UNDERLYING_ASSETS.items():
            if config.get("symbol") == symbol:
                return config.get("instrument_token")
        
        # Search in cached instruments
        instruments = self._load_instruments(exchange)
        for inst in instruments:
            if inst.get("tradingsymbol") == symbol:
                return inst.get("instrument_token")
        
        # Try NSE/BSE if F&O exchange didn't find it
        if exchange in ("NFO", "BFO"):
            fallback_exchange = "BSE" if exchange == "BFO" else "NSE"
            instruments = self._load_instruments(fallback_exchange)
            for inst in instruments:
                if inst.get("tradingsymbol") == symbol:
                    return inst.get("instrument_token")
        
        logger.warning(f"No instrument token found for {symbol}")
        return None
    
    def get_options_chain(
        self,
        underlying: str,
        expiry_date: Optional[datetime] = None,
        num_strikes: int = 10,
    ) -> pd.DataFrame:
        """
        Get options chain for an underlying.
        
        Args:
            underlying: The underlying asset
            expiry_date: Specific expiry date (optional)
            num_strikes: Number of strikes above and below ATM
            
        Returns:
            DataFrame with options chain data
        """
        if not self._ensure_connected():
            return pd.DataFrame()
        
        try:
            def _normalize_expiry(expiry_value):
                """Convert expiry to a date object for reliable comparisons."""
                if expiry_value is None:
                    return None
                if isinstance(expiry_value, datetime):
                    return expiry_value.date()
                if isinstance(expiry_value, date):
                    return expiry_value
                try:
                    return datetime.fromisoformat(str(expiry_value)).date()
                except Exception:
                    return None
            
            # Get current spot price
            spot_price = self.get_spot_price(underlying)
            if not spot_price:
                return pd.DataFrame()
            
            # Load instruments from the correct F&O exchange
            options_exchange = get_options_exchange(underlying)
            instruments = self._load_instruments(options_exchange)
            
            # Filter for options of this underlying
            options = [
                i for i in instruments
                if i["name"] == underlying
                and i["instrument_type"] in ["CE", "PE"]
            ]
            
            target_expiry_date = None
            if expiry_date:
                target_expiry_date = expiry_date.date() if isinstance(expiry_date, datetime) else expiry_date
                options = [
                    o for o in options
                    if _normalize_expiry(o["expiry"]) == target_expiry_date
                ]
            else:
                # Determine expiry mode: indices use weekly, stocks use monthly
                from config.settings import MARKET_HOURS
                
                # Per-underlying expiry logic:
                # Indices (in UNDERLYING_ASSETS) → weekly expiry for faster theta decay
                # Stocks → monthly expiry for better liquidity and wider strikes
                is_index = underlying in UNDERLYING_ASSETS
                use_monthly = not is_index  # Stocks monthly, indices weekly
                
                expiries = sorted({
                    _normalize_expiry(o["expiry"]) for o in options
                    if _normalize_expiry(o["expiry"]) is not None
                })
                if expiries:
                    if use_monthly:
                        # Find monthly expiry (last Thursday of each month)
                        from core.utils import get_monthly_expiry_date
                        monthly_expiry = get_monthly_expiry_date()
                        
                        # Find the expiry closest to monthly expiry date
                        target_expiry = None
                        for exp in expiries:
                            if exp == monthly_expiry.date():
                                target_expiry = exp
                                break
                        
                        # If exact match not found, use closest expiry >= monthly
                        if not target_expiry:
                            for exp in expiries:
                                if exp >= monthly_expiry.date():
                                    target_expiry = exp
                                    break
                        
                        # Fallback to nearest expiry if no monthly found
                        if not target_expiry:
                            target_expiry = expiries[0]
                            logger.warning(
                                f"Monthly expiry not found for {underlying}, using nearest: {target_expiry}"
                            )
                        else:
                            logger.debug(f"Using monthly expiry for {underlying}: {target_expiry}")
                        
                        options = [
                            o for o in options
                            if _normalize_expiry(o["expiry"]) == target_expiry
                        ]
                    else:
                        # Use nearest expiry (weekly) for indices — faster theta decay
                        nearest_expiry = expiries[0]
                        logger.debug(f"Using weekly expiry for INDEX {underlying}: {nearest_expiry}")
                        options = [
                            o for o in options
                            if _normalize_expiry(o["expiry"]) == nearest_expiry
                        ]
            
            # Get ATM strike - determine strike interval based on underlying
            strike_interval = get_strike_interval(underlying)
            if underlying not in UNDERLYING_ASSETS:
                # For stocks, derive interval from the near-ATM chain (not deep-OTM wide gaps)
                strikes = sorted(set(o["strike"] for o in options))
                if len(strikes) >= 2:
                    # Use minimum gap (near-ATM interval) for tighter, more accurate filtering
                    diffs = [strikes[i+1] - strikes[i] for i in range(len(strikes)-1)]
                    strike_interval = min(diffs) if diffs else 50
                else:
                    strike_interval = 50  # Default
            
            atm_strike = round(spot_price / strike_interval) * strike_interval
            
            # Filter strikes around ATM
            min_strike = atm_strike - (num_strikes * strike_interval)
            max_strike = atm_strike + (num_strikes * strike_interval)
            
            options = [
                o for o in options
                if min_strike <= o["strike"] <= max_strike
            ]
            
            # Get quotes for all options (use correct exchange: BFO for SENSEX, NFO otherwise)
            symbols = [f"{options_exchange}:{o['tradingsymbol']}" for o in options]
            
            if not symbols:
                return pd.DataFrame()
            
            self._throttle_api_call()
            quotes = self.kite.quote(symbols)
            
            # Build options chain DataFrame
            chain_data = []
            for opt in options:
                symbol = f"{options_exchange}:{opt['tradingsymbol']}"
                quote = quotes.get(symbol, {})
                
                chain_data.append({
                    "symbol": opt["tradingsymbol"],
                    "strike": opt["strike"],
                    "option_type": opt["instrument_type"],
                    "expiry": opt["expiry"],
                    "ltp": quote.get("last_price", 0),
                    "bid": quote.get("depth", {}).get("buy", [{}])[0].get("price", 0),
                    "ask": quote.get("depth", {}).get("sell", [{}])[0].get("price", 0),
                    "oi": quote.get("oi", 0),
                    "oi_change": quote.get("oi_day_high", 0) - quote.get("oi_day_low", 0),
                    "volume": quote.get("volume", 0),
                    "iv": self._calculate_iv(quote, spot_price, opt),
                    "lot_size": opt["lot_size"],
                    "instrument_token": opt["instrument_token"],
                })
            
            df = pd.DataFrame(chain_data)
            
            if not df.empty:
                df = df.sort_values(["strike", "option_type"])
            
            logger.info(f"Fetched options chain for {underlying}: {len(df)} options")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get options chain: {e}")
            return pd.DataFrame()
    
    def _calculate_iv(self, quote: dict, spot: float, option: dict) -> float:
        """
        Calculate implied volatility using QuantLib/py_vollib.
        
        Args:
            quote: Quote data from Kite
            spot: Spot price of underlying
            option: Option instrument data
            
        Returns:
            Implied volatility as percentage
        """
        ltp = quote.get("last_price", 0)
        strike = option["strike"]
        expiry = option.get("expiry")
        option_type = option.get("instrument_type", "CE")
        
        if ltp <= 0 or not expiry:
            return 0.0
        
        try:
            # Convert expiry to date if needed
            if hasattr(expiry, 'date'):
                expiry_date = expiry.date()
            else:
                expiry_date = expiry
            
            # Use the options pricer for accurate IV
            iv = options_pricer.calculate_iv(
                option_price=ltp,
                spot_price=spot,
                strike=strike,
                expiry_date=expiry_date,
                option_type=option_type,
            )
            
            return round(iv * 100, 2)  # Return as percentage
            
        except Exception as e:
            logger.debug(f"IV calculation failed for {option.get('tradingsymbol')}: {e}")
            return 0.0
    
    def get_option_greeks(
        self,
        symbol: str,
        spot_price: float,
        strike: float,
        expiry_date: date,
        market_price: float,
        option_type: str = "CE",
    ) -> Dict:
        """
        Get full Greeks analysis for an option.
        
        Args:
            symbol: Option trading symbol
            spot_price: Current spot price
            strike: Strike price
            expiry_date: Expiry date
            market_price: Current market price of the option
            option_type: 'CE' for Call, 'PE' for Put
            
        Returns:
            Dictionary with IV, Greeks, and pricing analysis
        """
        try:
            result = options_pricer.full_analysis(
                spot_price=spot_price,
                strike=strike,
                expiry_date=expiry_date,
                market_price=market_price,
                option_type=option_type,
            )
            
            return {
                "symbol": symbol,
                **result.to_dict(),
            }
            
        except Exception as e:
            logger.error(f"Greeks calculation failed for {symbol}: {e}")
            return {}
    
    def get_options_chain_with_greeks(
        self,
        underlying: str,
        expiry_date: Optional[datetime] = None,
        num_strikes: int = 10,
    ) -> pd.DataFrame:
        """
        Get options chain enriched with Greeks for each option.
        
        Args:
            underlying: The underlying asset
            expiry_date: Specific expiry date (optional)
            num_strikes: Number of strikes above and below ATM
            
        Returns:
            DataFrame with options chain including Greeks
        """
        # Get basic options chain
        chain = self.get_options_chain(underlying, expiry_date, num_strikes)
        
        if chain.empty:
            return chain
        
        # Get spot price
        spot_price = self.get_spot_price(underlying)
        if not spot_price:
            return chain
        
        # Add Greeks for each option
        greeks_data = []
        for _, row in chain.iterrows():
            try:
                expiry = row['expiry']
                if hasattr(expiry, 'date'):
                    expiry_date_val = expiry.date()
                else:
                    expiry_date_val = expiry
                
                # Calculate IV
                iv = options_pricer.calculate_iv(
                    option_price=row['ltp'],
                    spot_price=spot_price,
                    strike=row['strike'],
                    expiry_date=expiry_date_val,
                    option_type=row['option_type'],
                )
                
                # Calculate Greeks
                if iv > 0:
                    greeks = options_pricer.calculate_greeks(
                        spot_price=spot_price,
                        strike=row['strike'],
                        expiry_date=expiry_date_val,
                        volatility=iv,
                        option_type=row['option_type'],
                    )
                    
                    greeks_data.append({
                        'iv': round(iv * 100, 2),
                        'delta': round(greeks.delta, 4),
                        'gamma': round(greeks.gamma, 6),
                        'theta': round(greeks.theta, 4),
                        'vega': round(greeks.vega, 4),
                    })
                else:
                    greeks_data.append({
                        'iv': 0,
                        'delta': 0,
                        'gamma': 0,
                        'theta': 0,
                        'vega': 0,
                    })
                    
            except Exception as e:
                logger.debug(f"Greeks calc error for {row['symbol']}: {e}")
                greeks_data.append({
                    'iv': row.get('iv', 0),
                    'delta': 0,
                    'gamma': 0,
                    'theta': 0,
                    'vega': 0,
                })
        
        # Add Greeks columns to DataFrame
        greeks_df = pd.DataFrame(greeks_data)
        chain = chain.reset_index(drop=True)
        chain['iv'] = greeks_df['iv']
        chain['delta'] = greeks_df['delta']
        chain['gamma'] = greeks_df['gamma']
        chain['theta'] = greeks_df['theta']
        chain['vega'] = greeks_df['vega']
        
        logger.info(f"Enriched options chain with Greeks for {underlying}")
        return chain
    
    def get_strategy_greeks(self, legs: List[Dict], spot_price: float) -> Dict:
        """
        Calculate net Greeks for a multi-leg strategy.
        
        Args:
            legs: List of leg dictionaries
            spot_price: Current spot price
            
        Returns:
            Dictionary with net Greeks and position analysis
        """
        return options_pricer.calculate_strategy_greeks(legs, spot_price)
    
    def get_iv_percentile(
        self,
        underlying: str,
        current_iv: float,
        lookback_days: int = 252,
    ) -> Dict:
        """
        Calculate IV percentile and rank.
        
        Args:
            underlying: Underlying asset
            current_iv: Current IV value
            lookback_days: Days to look back for comparison
            
        Returns:
            Dictionary with IV percentile, rank, and historical stats
        """
        # In production, you'd fetch historical IV data
        # For now, return estimated values based on typical ranges
        
        # Typical IV ranges for NSE indices
        iv_ranges = {
            "NIFTY": {"low": 10, "high": 35, "median": 15},
            "BANKNIFTY": {"low": 12, "high": 45, "median": 18},
            "FINNIFTY": {"low": 11, "high": 40, "median": 16},
            "SENSEX": {"low": 10, "high": 35, "median": 15},
        }
        
        range_data = iv_ranges.get(underlying, {"low": 15, "high": 50, "median": 25})
        
        # Calculate percentile (simplified)
        iv_range = range_data["high"] - range_data["low"]
        if iv_range > 0:
            percentile = ((current_iv - range_data["low"]) / iv_range) * 100
            percentile = max(0, min(100, percentile))
        else:
            percentile = 50
        
        # Determine IV regime
        if percentile < 25:
            regime = "LOW"
        elif percentile < 50:
            regime = "BELOW_AVERAGE"
        elif percentile < 75:
            regime = "ABOVE_AVERAGE"
        else:
            regime = "HIGH"
        
        return {
            "current_iv": current_iv,
            "percentile": round(percentile, 1),
            "regime": regime,
            "historical_low": range_data["low"],
            "historical_high": range_data["high"],
            "historical_median": range_data["median"],
        }
    
    def get_oi_data(self, underlying: str) -> Dict[str, Any]:
        """
        Get Open Interest analysis for an underlying.
        
        Args:
            underlying: The underlying asset
            
        Returns:
            Dictionary with OI analysis
        """
        chain = self.get_options_chain(underlying)
        
        if chain.empty:
            return {}
        
        # Separate calls and puts
        calls = chain[chain["option_type"] == "CE"]
        puts = chain[chain["option_type"] == "PE"]
        
        # Calculate PCR
        total_call_oi = calls["oi"].sum()
        total_put_oi = puts["oi"].sum()
        pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 0
        
        # Find max pain
        strikes = chain["strike"].unique()
        max_pain_strike = None
        min_pain = float("inf")
        
        spot = self.get_spot_price(underlying)
        
        for strike in strikes:
            call_oi = calls[calls["strike"] == strike]["oi"].sum()
            put_oi = puts[puts["strike"] == strike]["oi"].sum()
            
            # Calculate pain at this strike
            call_pain = sum((s - strike) * calls[calls["strike"] == s]["oi"].sum() 
                          for s in strikes if s < strike)
            put_pain = sum((strike - s) * puts[puts["strike"] == s]["oi"].sum() 
                         for s in strikes if s > strike)
            
            total_pain = call_pain + put_pain
            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = strike
        
        # Find max OI strikes
        max_call_oi_strike = calls.loc[calls["oi"].idxmax()]["strike"] if not calls.empty else None
        max_put_oi_strike = puts.loc[puts["oi"].idxmax()]["strike"] if not puts.empty else None
        
        return {
            "pcr": round(pcr, 2),
            "total_call_oi": total_call_oi,
            "total_put_oi": total_put_oi,
            "max_pain": max_pain_strike,
            "max_call_oi_strike": max_call_oi_strike,
            "max_put_oi_strike": max_put_oi_strike,
            "spot": spot,
            "sentiment": self._interpret_oi(pcr, max_call_oi_strike, max_put_oi_strike, spot),
        }
    
    def _interpret_oi(
        self,
        pcr: float,
        max_call_strike: float,
        max_put_strike: float,
        spot: float,
    ) -> str:
        """Interpret OI data for market sentiment."""
        if pcr > METRICS_CONFIG["pcr_bearish_threshold"]:
            sentiment = "BULLISH"  # High PCR often indicates bullish reversal
        elif pcr < METRICS_CONFIG["pcr_bullish_threshold"]:
            sentiment = "BEARISH"  # Low PCR indicates bearish
        else:
            sentiment = "NEUTRAL"
        
        # Adjust based on max OI strikes
        if max_call_strike and max_put_strike and spot:
            if spot > max_call_strike:
                sentiment = "STRONGLY_BULLISH"
            elif spot < max_put_strike:
                sentiment = "STRONGLY_BEARISH"
        
        return sentiment
    
    def get_historical_data(
        self,
        symbol: str,
        interval: str = "5minute",
        days: int = 5,
        exchange: str = "NSE",
    ) -> pd.DataFrame:
        """
        Get historical OHLC data.
        
        Args:
            symbol: Trading symbol
            interval: Candle interval (minute, 5minute, 15minute, day, etc.)
            days: Number of days of history
            exchange: Exchange
            
        Returns:
            DataFrame with OHLC data
        """
        if not self._ensure_connected():
            return pd.DataFrame()
        
        try:
            # First check if it's an index with a known instrument token
            from config.settings import UNDERLYING_ASSETS, get_instrument_token
            
            instrument_token = None
            
            # Check if symbol is an index with stored instrument_token
            if symbol in UNDERLYING_ASSETS:
                instrument_token = UNDERLYING_ASSETS[symbol].get("instrument_token")
            
            # If not an index or no token, look up in instruments
            if not instrument_token:
                instruments = self._load_instruments(exchange)
                instrument = next(
                    (i for i in instruments if i["tradingsymbol"] == symbol),
                    None
                )
                
                if not instrument:
                    # Try with get_instrument_token as fallback
                    instrument_token = get_instrument_token(symbol)
                    if not instrument_token:
                        logger.error(f"Instrument not found: {symbol}")
                        return pd.DataFrame()
                else:
                    instrument_token = instrument["instrument_token"]
            
            from_date = datetime.now() - timedelta(days=days)
            to_date = datetime.now()
            
            self._throttle_api_call()
            data = self.kite.historical_data(
                instrument_token,
                from_date,
                to_date,
                interval,
            )
            
            df = pd.DataFrame(data)
            if not df.empty:
                df.set_index("date", inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            return pd.DataFrame()
    
    def get_volatility_metrics(self, underlying: str) -> Dict[str, float]:
        """
        Calculate volatility metrics for an underlying.
        
        Args:
            underlying: The underlying asset
            
        Returns:
            Dictionary with volatility metrics
        """
        # For indices with instrument_token, use the underlying key directly
        asset_config = UNDERLYING_ASSETS.get(underlying, {})
        if asset_config and "instrument_token" in asset_config:
            symbol = underlying
        elif asset_config:
            symbol = asset_config.get("symbol", underlying)
        else:
            symbol = underlying
        
        # Get historical data (use correct exchange for the underlying)
        exchange = asset_config.get("exchange", "NSE")
        hist_data = self.get_historical_data(symbol, "day", 30, exchange)
        
        if hist_data.empty:
            return {}
        
        # Calculate returns
        hist_data["returns"] = hist_data["close"].pct_change()
        
        # Historical volatility (annualized)
        hv_20 = hist_data["returns"].tail(20).std() * (252 ** 0.5) * 100
        hv_10 = hist_data["returns"].tail(10).std() * (252 ** 0.5) * 100
        
        # Get ATM IV from options chain
        chain = self.get_options_chain(underlying, num_strikes=2)
        spot = self.get_spot_price(underlying)
        
        atm_iv = 0
        if not chain.empty and spot:
            strike_interval = get_strike_interval(underlying)
            atm_strike = round(spot / strike_interval) * strike_interval
            atm_options = chain[chain["strike"] == atm_strike]
            if not atm_options.empty:
                atm_iv = atm_options["iv"].mean()
        
        return {
            "hv_20": round(hv_20, 2),
            "hv_10": round(hv_10, 2),
            "atm_iv": round(atm_iv, 2),
            "iv_hv_ratio": round(atm_iv / hv_20, 2) if hv_20 > 0 else 0,
            "volatility_regime": self._classify_volatility(atm_iv, hv_20),
        }
    
    def _classify_volatility(self, iv: float, hv: float) -> str:
        """Classify volatility regime."""
        if iv > METRICS_CONFIG["iv_percentile_high"]:
            return "HIGH_IV"
        elif iv < METRICS_CONFIG["iv_percentile_low"]:
            return "LOW_IV"
        elif iv > hv * 1.2:
            return "IV_ELEVATED"
        elif iv < hv * 0.8:
            return "IV_DEPRESSED"
        return "NORMAL"
    
    def get_historical_analysis(self, underlying: str, days: int = 30) -> Dict[str, Any]:
        """
        Comprehensive historical analysis for an underlying.
        Analyzes trend, momentum, support/resistance, and recent performance.
        
        Args:
            underlying: The underlying asset
            days: Number of days to analyze
            
        Returns:
            Dictionary with historical analysis metrics
        """
        # For indices with instrument_token, use the underlying key directly
        # For stocks, use the trading symbol
        asset_config = UNDERLYING_ASSETS.get(underlying, {})
        if asset_config and "instrument_token" in asset_config:
            # Index with direct instrument token - pass underlying key
            symbol = underlying
        elif asset_config:
            # Stock - use trading symbol
            symbol = asset_config.get("symbol", underlying)
        else:
            symbol = underlying
        
        # Get historical data - use the correct exchange for the underlying
        hist_exchange = asset_config.get("exchange", "NSE") if asset_config else "NSE"
        hist_data = self.get_historical_data(symbol, "day", days, hist_exchange)
        
        if hist_data.empty or len(hist_data) < 10:
            logger.warning(f"Insufficient historical data for {underlying}")
            return {}
        
        try:
            analysis = {}
            
            # Current price
            current_price = hist_data["close"].iloc[-1]
            analysis["current_price"] = current_price
            
            # ============ TREND ANALYSIS ============
            # Simple Moving Averages
            hist_data["sma_5"] = hist_data["close"].rolling(window=5).mean()
            hist_data["sma_10"] = hist_data["close"].rolling(window=10).mean()
            hist_data["sma_20"] = hist_data["close"].rolling(window=20).mean()
            
            sma_5 = hist_data["sma_5"].iloc[-1]
            sma_10 = hist_data["sma_10"].iloc[-1]
            sma_20 = hist_data["sma_20"].iloc[-1] if len(hist_data) >= 20 else sma_10
            
            # Trend determination
            if current_price > sma_5 > sma_10 > sma_20:
                trend = "STRONG_UPTREND"
                trend_score = 1.0
            elif current_price > sma_10 > sma_20:
                trend = "UPTREND"
                trend_score = 0.7
            elif current_price < sma_5 < sma_10 < sma_20:
                trend = "STRONG_DOWNTREND"
                trend_score = -1.0
            elif current_price < sma_10 < sma_20:
                trend = "DOWNTREND"
                trend_score = -0.7
            else:
                trend = "SIDEWAYS"
                trend_score = 0.0
            
            analysis["trend"] = trend
            analysis["trend_score"] = trend_score
            analysis["sma_5"] = round(sma_5, 2)
            analysis["sma_10"] = round(sma_10, 2)
            analysis["sma_20"] = round(sma_20, 2)
            
            # Price vs SMA (how far from moving averages)
            analysis["price_vs_sma20_pct"] = round(((current_price - sma_20) / sma_20) * 100, 2)
            
            # ============ MOMENTUM ANALYSIS ============
            # RSI Calculation
            hist_data["returns"] = hist_data["close"].pct_change()
            hist_data["gain"] = hist_data["returns"].apply(lambda x: x if x > 0 else 0)
            hist_data["loss"] = hist_data["returns"].apply(lambda x: abs(x) if x < 0 else 0)
            
            avg_gain = hist_data["gain"].rolling(window=14).mean().iloc[-1]
            avg_loss = hist_data["loss"].rolling(window=14).mean().iloc[-1]
            
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 100
            
            analysis["rsi"] = round(rsi, 2)
            
            if rsi > 70:
                analysis["rsi_signal"] = "OVERBOUGHT"
            elif rsi < 30:
                analysis["rsi_signal"] = "OVERSOLD"
            else:
                analysis["rsi_signal"] = "NEUTRAL"
            
            # Recent momentum (5-day return)
            returns_5d = ((current_price / hist_data["close"].iloc[-6]) - 1) * 100 if len(hist_data) >= 6 else 0
            returns_10d = ((current_price / hist_data["close"].iloc[-11]) - 1) * 100 if len(hist_data) >= 11 else 0
            
            analysis["returns_5d"] = round(returns_5d, 2)
            analysis["returns_10d"] = round(returns_10d, 2)
            
            # Momentum score
            if returns_5d > 3:
                momentum = "STRONG_BULLISH"
                momentum_score = 1.0
            elif returns_5d > 1:
                momentum = "BULLISH"
                momentum_score = 0.5
            elif returns_5d < -3:
                momentum = "STRONG_BEARISH"
                momentum_score = -1.0
            elif returns_5d < -1:
                momentum = "BEARISH"
                momentum_score = -0.5
            else:
                momentum = "NEUTRAL"
                momentum_score = 0.0
            
            analysis["momentum"] = momentum
            analysis["momentum_score"] = momentum_score
            
            # ============ VOLATILITY ANALYSIS ============
            # Historical volatility
            daily_returns = hist_data["returns"].dropna()
            hv_10 = daily_returns.tail(10).std() * (252 ** 0.5) * 100
            hv_20 = daily_returns.tail(20).std() * (252 ** 0.5) * 100 if len(daily_returns) >= 20 else hv_10
            
            analysis["hv_10"] = round(hv_10, 2)
            analysis["hv_20"] = round(hv_20, 2)
            
            # Average True Range (ATR) for stop loss sizing
            hist_data["tr"] = pd.concat([
                hist_data["high"] - hist_data["low"],
                abs(hist_data["high"] - hist_data["close"].shift(1)),
                abs(hist_data["low"] - hist_data["close"].shift(1))
            ], axis=1).max(axis=1)
            
            atr_14 = hist_data["tr"].rolling(window=14).mean().iloc[-1]
            analysis["atr_14"] = round(atr_14, 2)
            analysis["atr_percent"] = round((atr_14 / current_price) * 100, 2)
            
            # ============ SUPPORT/RESISTANCE ============
            # Recent high/low
            high_20 = hist_data["high"].tail(20).max()
            low_20 = hist_data["low"].tail(20).min()
            
            analysis["resistance_20d"] = round(high_20, 2)
            analysis["support_20d"] = round(low_20, 2)
            
            # Position in range (0 = at support, 1 = at resistance)
            range_position = (current_price - low_20) / (high_20 - low_20) if high_20 != low_20 else 0.5
            analysis["range_position"] = round(range_position, 2)
            
            if range_position > 0.8:
                analysis["price_zone"] = "NEAR_RESISTANCE"
            elif range_position < 0.2:
                analysis["price_zone"] = "NEAR_SUPPORT"
            else:
                analysis["price_zone"] = "MID_RANGE"
            
            # ============ VOLUME ANALYSIS ============
            avg_volume_20 = hist_data["volume"].tail(20).mean()
            recent_volume = hist_data["volume"].tail(5).mean()
            volume_ratio = recent_volume / avg_volume_20 if avg_volume_20 > 0 else 1
            
            analysis["avg_volume_20d"] = int(avg_volume_20)
            analysis["volume_ratio"] = round(volume_ratio, 2)
            
            if volume_ratio > 1.5:
                analysis["volume_signal"] = "HIGH_VOLUME"
            elif volume_ratio < 0.7:
                analysis["volume_signal"] = "LOW_VOLUME"
            else:
                analysis["volume_signal"] = "NORMAL"
            
            # ============ OVERALL HISTORICAL SENTIMENT ============
            # Combine all factors for overall historical sentiment
            combined_score = (trend_score * 0.4) + (momentum_score * 0.4) + (
                0.2 if analysis["rsi_signal"] == "OVERSOLD" else 
                -0.2 if analysis["rsi_signal"] == "OVERBOUGHT" else 0
            )
            
            if combined_score > 0.5:
                analysis["historical_sentiment"] = "BULLISH"
            elif combined_score > 0.2:
                analysis["historical_sentiment"] = "MILDLY_BULLISH"
            elif combined_score < -0.5:
                analysis["historical_sentiment"] = "BEARISH"
            elif combined_score < -0.2:
                analysis["historical_sentiment"] = "MILDLY_BEARISH"
            else:
                analysis["historical_sentiment"] = "NEUTRAL"
            
            analysis["historical_score"] = round(combined_score, 2)
            
            # Confidence boost from historical data
            analysis["confidence_boost"] = self._calculate_confidence_boost(analysis)
            
            # Include raw DataFrame for ML feature extraction
            analysis["df"] = hist_data
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in historical analysis for {underlying}: {e}")
            return {}
    
    def _calculate_confidence_boost(self, analysis: Dict) -> float:
        """
        Calculate confidence boost based on historical analysis.
        Returns a value between -0.2 and +0.2 to add to base confidence.
        """
        boost = 0.0
        
        # Trend alignment boost
        if analysis.get("trend") in ["STRONG_UPTREND", "STRONG_DOWNTREND"]:
            boost += 0.08
        elif analysis.get("trend") in ["UPTREND", "DOWNTREND"]:
            boost += 0.04
        
        # Momentum alignment boost
        if analysis.get("momentum") in ["STRONG_BULLISH", "STRONG_BEARISH"]:
            boost += 0.06
        elif analysis.get("momentum") in ["BULLISH", "BEARISH"]:
            boost += 0.03
        
        # RSI extreme levels (potential reversal - reduce confidence)
        if analysis.get("rsi_signal") in ["OVERBOUGHT", "OVERSOLD"]:
            boost -= 0.02
        
        # Volume confirmation
        if analysis.get("volume_signal") == "HIGH_VOLUME":
            boost += 0.04
        
        # Cap the boost
        return max(-0.2, min(0.2, boost))

    # ========== INTRADAY ANALYSIS (5-minute candles) ==========

    def get_intraday_analysis(self, underlying: str) -> Dict[str, Any]:
        """
        Intraday analysis using 5-minute candles for entry timing.

        Computes:
        - VWAP and price position relative to VWAP
        - Intraday micro-trend (EMA 9/21 on 5-min candles)
        - 5-min RSI for overbought/oversold micro-timing
        - Intraday momentum (last 5 candles vs previous 5)
        - Candle quality (body-to-wick ratio of recent candles)

        Args:
            underlying: The underlying asset

        Returns:
            Dictionary with intraday metrics, empty dict on failure
        """
        # Determine symbol / exchange for the underlying
        asset_config = UNDERLYING_ASSETS.get(underlying, {})
        if asset_config and "instrument_token" in asset_config:
            symbol = underlying
        elif asset_config:
            symbol = asset_config.get("symbol", underlying)
        else:
            symbol = underlying
        hist_exchange = asset_config.get("exchange", "NSE") if asset_config else "NSE"

        # Fetch 5-minute candles for today + yesterday (need prior data for pre-market indicators)
        intraday = self.get_historical_data(symbol, "5minute", days=2, exchange=hist_exchange)

        if intraday.empty or len(intraday) < 10:
            logger.debug(f"Insufficient intraday data for {underlying} ({len(intraday)} candles)")
            return {}

        try:
            result: Dict[str, Any] = {}
            latest = intraday.iloc[-1]
            price = latest["close"]
            result["intraday_price"] = round(price, 2)

            # ---------- VWAP ----------
            # VWAP = cumulative(TP * volume) / cumulative(volume)
            # Reset per day: filter to today's candles only
            today = datetime.now().date()
            today_mask = intraday.index.date == today  # type: ignore[attr-defined]
            today_df = intraday[today_mask].copy()

            if len(today_df) >= 2:
                tp = (today_df["high"] + today_df["low"] + today_df["close"]) / 3
                cum_tp_vol = (tp * today_df["volume"]).cumsum()
                cum_vol = today_df["volume"].cumsum()
                vwap = (cum_tp_vol / cum_vol.replace(0, float("nan"))).iloc[-1]
                if pd.notna(vwap) and vwap > 0:
                    result["vwap"] = round(vwap, 2)
                    result["price_vs_vwap_pct"] = round(((price - vwap) / vwap) * 100, 3)
                    result["above_vwap"] = price > vwap
                else:
                    result["vwap"] = None
                    result["price_vs_vwap_pct"] = 0
                    result["above_vwap"] = None
            else:
                result["vwap"] = None
                result["price_vs_vwap_pct"] = 0
                result["above_vwap"] = None

            # ---------- Micro-trend: EMA 9 / 21 on 5-min close ----------
            intraday["ema_9"] = intraday["close"].ewm(span=9, adjust=False).mean()
            intraday["ema_21"] = intraday["close"].ewm(span=21, adjust=False).mean()

            ema_9 = intraday["ema_9"].iloc[-1]
            ema_21 = intraday["ema_21"].iloc[-1]
            result["ema_9"] = round(ema_9, 2)
            result["ema_21"] = round(ema_21, 2)

            if price > ema_9 > ema_21:
                micro_trend = "UP"
                micro_score = 1.0
            elif price < ema_9 < ema_21:
                micro_trend = "DOWN"
                micro_score = -1.0
            elif price > ema_21:
                micro_trend = "WEAK_UP"
                micro_score = 0.3
            elif price < ema_21:
                micro_trend = "WEAK_DOWN"
                micro_score = -0.3
            else:
                micro_trend = "FLAT"
                micro_score = 0.0

            result["micro_trend"] = micro_trend
            result["micro_trend_score"] = micro_score

            # ---------- 5-min RSI (14-period) ----------
            returns = intraday["close"].pct_change()
            gains = returns.clip(lower=0)
            losses = (-returns).clip(lower=0)
            avg_gain = gains.rolling(14).mean().iloc[-1]
            avg_loss = losses.rolling(14).mean().iloc[-1]

            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi_5m = 100 - (100 / (1 + rs))
            else:
                rsi_5m = 100.0

            result["rsi_5m"] = round(rsi_5m, 2)
            if rsi_5m > 75:
                result["rsi_5m_signal"] = "OVERBOUGHT"
            elif rsi_5m < 25:
                result["rsi_5m_signal"] = "OVERSOLD"
            else:
                result["rsi_5m_signal"] = "NEUTRAL"

            # ---------- Intraday momentum (last 5 vs previous 5 candles) ----------
            if len(intraday) >= 10:
                recent_5_ret = (intraday["close"].iloc[-1] / intraday["close"].iloc[-6] - 1) * 100
                prev_5_ret = (intraday["close"].iloc[-6] / intraday["close"].iloc[-11] - 1) * 100
                result["recent_5c_return_pct"] = round(recent_5_ret, 3)
                result["prev_5c_return_pct"] = round(prev_5_ret, 3)
                result["momentum_accelerating"] = recent_5_ret > prev_5_ret
            else:
                result["recent_5c_return_pct"] = 0
                result["prev_5c_return_pct"] = 0
                result["momentum_accelerating"] = False

            # ---------- Candle quality (average body-to-range ratio) ----------
            last_5 = intraday.tail(5)
            body = (last_5["close"] - last_5["open"]).abs()
            candle_range = last_5["high"] - last_5["low"]
            body_ratio = (body / candle_range.replace(0, float("nan"))).mean()
            result["avg_body_ratio"] = round(body_ratio, 3) if pd.notna(body_ratio) else 0.5

            # ---------- Intraday high/low range ----------
            if len(today_df) >= 2:
                result["intraday_high"] = round(today_df["high"].max(), 2)
                result["intraday_low"] = round(today_df["low"].min(), 2)
                day_range = result["intraday_high"] - result["intraday_low"]
                if day_range > 0:
                    result["intraday_range_position"] = round(
                        (price - result["intraday_low"]) / day_range, 3
                    )
                else:
                    result["intraday_range_position"] = 0.5
            else:
                result["intraday_high"] = price
                result["intraday_low"] = price
                result["intraday_range_position"] = 0.5

            # ---------- Overall intraday timing signal ----------
            # Bullish: above VWAP + micro UP + RSI not overbought
            # Bearish: below VWAP + micro DOWN + RSI not oversold
            bullish_score = 0
            bearish_score = 0

            if result.get("above_vwap"):
                bullish_score += 1
            elif result.get("above_vwap") is False:
                bearish_score += 1

            if micro_trend in ("UP", "WEAK_UP"):
                bullish_score += 1
            elif micro_trend in ("DOWN", "WEAK_DOWN"):
                bearish_score += 1

            if result["rsi_5m_signal"] == "OVERBOUGHT":
                bullish_score -= 1   # Caution against buying
            elif result["rsi_5m_signal"] == "OVERSOLD":
                bearish_score -= 1   # Caution against selling

            if result.get("momentum_accelerating"):
                # Momentum direction matters
                if result.get("recent_5c_return_pct", 0) > 0:
                    bullish_score += 1
                else:
                    bearish_score += 1

            if bullish_score >= 2:
                result["intraday_bias"] = "BULLISH"
            elif bearish_score >= 2:
                result["intraday_bias"] = "BEARISH"
            else:
                result["intraday_bias"] = "NEUTRAL"

            result["intraday_bull_score"] = bullish_score
            result["intraday_bear_score"] = bearish_score

            logger.debug(
                f"Intraday {underlying}: bias={result['intraday_bias']} "
                f"VWAP={'above' if result.get('above_vwap') else 'below'} "
                f"micro={micro_trend} RSI5m={rsi_5m:.0f}"
            )

            return result

        except Exception as e:
            logger.error(f"Error in intraday analysis for {underlying}: {e}")
            return {}


# Singleton instance
data_fetcher = DataFetcher()
