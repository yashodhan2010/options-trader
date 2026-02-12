"""
Historical Data Collector - Fetch and process market data for ML training.

Data Sources:
1. NSE Equity Bhavcopy - Daily OHLCV for stocks (works)
2. NSE Option Chain API - Live options data during market hours (works)
3. Kite Historical API - OHLCV for stocks and indices (works)

NOTE: NSE F&O bhavcopy archives are no longer publicly accessible.
For options data (IV, OI, Greeks), use:
- Live collection via NSE API during market hours
- Kite API for historical OHLCV

Features generated:
- OHLCV features from market data
- Technical indicators (RSI, MACD, Bollinger, etc.)
- IV calculation using Black-Scholes/Black-76 (from live options chain)
- Greeks (Delta, Gamma, Theta, Vega) calculation
- OI analysis and PCR ratios (from live options chain)
"""

import os
import io
import zipfile
import gzip
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.settings import (
    UNDERLYING_ASSETS, ML_CONFIG, DATABASE_CONFIG
)
from core.database import database
from core.logger import logger
from core.options_pricer import options_pricer

# Try to import jugaad-data as fallback for NSE downloads
try:
    from jugaad_data.nse import bhavcopy_save, bhavcopy_fo_save
    JUGAAD_AVAILABLE = True
except ImportError:
    JUGAAD_AVAILABLE = False
    logger.debug("jugaad-data not installed. Using direct NSE downloads only.")


# NSE Archive configuration
NSE_ARCHIVE_BASE = "https://archives.nseindia.com"
BHAVCOPY_PATH = "/content/historical/DERIVATIVES/{year}/{month}/fo{day}{month}{year}bhav.csv.zip"
EQUITY_BHAVCOPY_PATH = "/content/historical/EQUITIES/{year}/{month}/cm{day}{month}{year}bhav.csv.zip"

# Request headers to mimic browser
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "https://www.nseindia.com/",
}

# Month abbreviations for NSE URLs
MONTH_ABBR = {
    1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN",
    7: "JUL", 8: "AUG", 9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC"
}

# Risk-free rate for IV calculation (approximate)
RISK_FREE_RATE = 0.065  # 6.5% for India


class HistoricalDataCollector:
    """
    Collect historical F&O data from NSE bhavcopy archives.
    
    Downloads daily bhavcopy files, parses derivative data,
    calculates IV and Greeks, and generates ML features.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the collector.
        
        Args:
            cache_dir: Directory to cache downloaded files
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/bhavcopy_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update(NSE_HEADERS)
        
        # Watchlist symbols
        self.symbols = self._get_watchlist_symbols()
        
        # Ensure database table exists
        self._ensure_table()
        
        logger.info(f"HistoricalDataCollector initialized - {len(self.symbols)} symbols")
    
    def _get_watchlist_symbols(self) -> List[str]:
        """Get symbols from watchlist config."""
        try:
            from config.settings import CONFIG_DIR
            watchlist_path = CONFIG_DIR / "watchlist.json"
            
            if watchlist_path.exists():
                with open(watchlist_path) as f:
                    watchlist = json.load(f)
                    
                    if "assets" in watchlist:
                        symbols = [
                            asset["name"] 
                            for asset in watchlist["assets"] 
                            if asset.get("enabled", True)
                        ]
                    else:
                        symbols = watchlist.get("underlyings", [])
                    
                    for idx in ["NIFTY", "BANKNIFTY"]:
                        if idx not in symbols:
                            symbols.append(idx)
                    
                    if symbols:
                        return symbols
                        
        except Exception as e:
            logger.warning(f"Could not load watchlist: {e}")
        
        return list(UNDERLYING_ASSETS.keys()) + ["RELIANCE", "HDFCBANK", "SBIN", "AXISBANK"]
    
    def _ensure_table(self) -> None:
        """Ensure the feature snapshots table has the source column."""
        try:
            conn = database._get_connection()
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='ml_feature_snapshots'
            """)
            
            if not cursor.fetchone():
                # Table doesn't exist - will be created by live_feature_collector
                logger.warning("ml_feature_snapshots table doesn't exist. Run 'ml collect start' first.")
                return
            
            # Add source column if not exists (for distinguishing live vs historical)
            try:
                cursor.execute("ALTER TABLE ml_feature_snapshots ADD COLUMN source TEXT DEFAULT 'live'")
                conn.commit()
            except:
                pass  # Column already exists
            
            logger.info("ml_feature_snapshots table ready for historical data")
            
        except Exception as e:
            logger.error(f"Failed to ensure table: {e}")
    
    def _get_nse_session(self) -> requests.Session:
        """Get session with NSE cookies."""
        try:
            # First hit the main page to get cookies
            self.session.get("https://www.nseindia.com", timeout=10)
            time.sleep(0.5)
        except Exception as e:
            logger.warning(f"Failed to get NSE session: {e}")
        return self.session
    
    def load_from_file(self, file_path: str, target_date: Optional[date] = None) -> Optional[pd.DataFrame]:
        """
        Load bhavcopy from a manually downloaded file.
        
        Use this when NSE blocks automated downloads. Download the file manually
        from NSE website and load it here.
        
        Args:
            file_path: Path to the downloaded file (.csv or .csv.zip)
            target_date: Optional date to use for caching (extracted from filename if not provided)
            
        Returns:
            DataFrame with bhavcopy data or None if failed
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None
        
        try:
            # Check if it's a zip file
            if file_path.suffix.lower() == '.zip' or '.zip' in file_path.name.lower():
                with zipfile.ZipFile(file_path) as zf:
                    csv_name = zf.namelist()[0]
                    with zf.open(csv_name) as f:
                        df = pd.read_csv(f)
            else:
                df = pd.read_csv(file_path)
            
            # Extract date from filename if not provided
            # Format: fo{dd}{MON}{yyyy}bhav.csv.zip
            if target_date is None:
                filename = file_path.stem.replace('.csv', '')
                try:
                    # Try parsing foXXMONYYYYbhav format
                    if filename.startswith('fo') and 'bhav' in filename.lower():
                        date_part = filename[2:].replace('bhav', '').replace('BHAV', '')
                        day = int(date_part[:2])
                        mon = date_part[2:5].upper()
                        year = int(date_part[5:9])
                        month = list(MONTH_ABBR.values()).index(mon) + 1
                        target_date = date(year, month, day)
                except:
                    target_date = date.today()
            
            # Cache the data
            cache_file = self.cache_dir / f"fo_{target_date.isoformat()}.csv"
            df.to_csv(cache_file, index=False)
            logger.info(f"Loaded bhavcopy from {file_path}: {len(df)} records, cached as {cache_file.name}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load bhavcopy from {file_path}: {e}")
            return None
    
    def load_multiple_files(self, file_paths: List[str]) -> Dict[date, pd.DataFrame]:
        """
        Load multiple bhavcopy files.
        
        Args:
            file_paths: List of paths to bhavcopy files
            
        Returns:
            Dictionary mapping dates to DataFrames
        """
        results = {}
        
        for file_path in file_paths:
            df = self.load_from_file(file_path)
            if df is not None:
                # Get date from cache filename that was created
                # This is a bit hacky but works
                try:
                    cached = list(self.cache_dir.glob("fo_*.csv"))
                    if cached:
                        latest = max(cached, key=lambda x: x.stat().st_mtime)
                        date_str = latest.stem.replace('fo_', '')
                        target_date = date.fromisoformat(date_str)
                        results[target_date] = df
                except:
                    pass
        
        logger.info(f"Loaded {len(results)} bhavcopy files")
        return results
    
    def download_bhavcopy(self, target_date: date) -> Optional[pd.DataFrame]:
        """
        Download F&O bhavcopy for a specific date.
        
        Tries multiple URL patterns as NSE changes them sometimes.
        
        Args:
            target_date: Date to download data for
            
        Returns:
            DataFrame with bhavcopy data or None if failed
        """
        # Format: fo{dd}{MON}{yyyy}bhav.csv.zip in path /{yyyy}/{MON}/
        dd = f"{target_date.day:02d}"
        mon = MONTH_ABBR[target_date.month]  # JAN, FEB, etc.
        yyyy = str(target_date.year)
        mm = f"{target_date.month:02d}"  # 01, 02, etc.
        
        cache_file = self.cache_dir / f"fo_{target_date.isoformat()}.csv"
        
        # Check cache first
        if cache_file.exists():
            logger.debug(f"Loading from cache: {cache_file}")
            return pd.read_csv(cache_file)
        
        # Try multiple URL patterns - NSE uses month abbreviation (DEC) not number (12)
        url_patterns = [
            # Standard archive format with month abbreviation in path
            f"https://archives.nseindia.com/content/historical/DERIVATIVES/{yyyy}/{mon}/fo{dd}{mon}{yyyy}bhav.csv.zip",
            f"https://nsearchives.nseindia.com/content/historical/DERIVATIVES/{yyyy}/{mon}/fo{dd}{mon}{yyyy}bhav.csv.zip",
            # Alternative with numeric month in path (older format)
            f"https://archives.nseindia.com/content/historical/DERIVATIVES/{yyyy}/{mm}/fo{dd}{mon}{yyyy}bhav.csv.zip",
            f"https://nsearchives.nseindia.com/content/historical/DERIVATIVES/{yyyy}/{mm}/fo{dd}{mon}{yyyy}bhav.csv.zip",
            # Alternative paths
            f"https://nsearchives.nseindia.com/archives/fo/bhav/fo{dd}{mon}{yyyy}bhav.csv.zip",
            f"https://archives.nseindia.com/archives/fo/bhav/fo{dd}{mon}{yyyy}bhav.csv.zip",
        ]
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/"
        }
        
        for url in url_patterns:
            try:
                response = requests.get(url, headers=headers, timeout=15)
                
                if response.status_code == 200:
                    # Extract CSV from zip
                    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                        csv_name = zf.namelist()[0]
                        with zf.open(csv_name) as f:
                            df = pd.read_csv(f)
                    
                    # Cache the data
                    df.to_csv(cache_file, index=False)
                    logger.info(f"Downloaded bhavcopy for {target_date}: {len(df)} records from {url.split('/')[2]}")
                    return df
                    
            except zipfile.BadZipFile:
                continue  # Not a valid zip, try next URL
            except Exception as e:
                logger.debug(f"URL failed: {url} - {e}")
                continue
        
        # Fallback: Try jugaad-data library if direct download failed
        if JUGAAD_AVAILABLE:
            try:
                logger.info(f"Trying jugaad-data fallback for {target_date}...")
                jugaad_dir = self.cache_dir / "jugaad_temp"
                jugaad_dir.mkdir(parents=True, exist_ok=True)
                
                bhavcopy_fo_save(target_date, str(jugaad_dir))
                
                # Find the downloaded file
                for f in jugaad_dir.glob("*.csv"):
                    df = pd.read_csv(f)
                    df.to_csv(cache_file, index=False)
                    f.unlink()  # Clean up temp file
                    logger.info(f"Downloaded bhavcopy via jugaad-data for {target_date}: {len(df)} records")
                    return df
                    
            except Exception as e:
                if "404" in str(e) or "holiday" in str(e).lower():
                    logger.debug(f"{target_date} is likely a holiday")
                else:
                    logger.debug(f"jugaad-data fallback failed for {target_date}: {e}")
        
        logger.warning(f"Bhavcopy not available for {target_date} (holiday or NSE blocking). Use load_from_file() for manual upload.")
        return None
    
    def download_equity_bhavcopy(self, target_date: date) -> Optional[pd.DataFrame]:
        """
        Download equity (cash market) bhavcopy for spot prices.
        
        Uses the new NSE format (sec_bhavdata_full) which is publicly accessible.
        
        Args:
            target_date: Date to download data for
            
        Returns:
            DataFrame with equity bhavcopy data or None if failed
        """
        cache_file = self.cache_dir / f"cm_{target_date.isoformat()}.csv"
        
        # Check cache first
        if cache_file.exists():
            logger.debug(f"Loading from cache: {cache_file}")
            return pd.read_csv(cache_file)
        
        # NEW FORMAT: sec_bhavdata_full_DDMMYYYY.csv (this works!)
        date_str = target_date.strftime('%d%m%Y')
        new_url = f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{date_str}.csv"
        
        # OLD FORMAT (fallback)
        year = target_date.year
        month = MONTH_ABBR[target_date.month]
        day = f"{target_date.day:02d}"
        old_url = f"{NSE_ARCHIVE_BASE}/content/historical/EQUITIES/{year}/{month}/cm{day}{month}{year}bhav.csv.zip"
        
        # Try new format first (sec_bhavdata_full - this works!)
        try:
            session = self._get_nse_session()
            response = session.get(new_url, timeout=30)
            
            if response.status_code == 200 and len(response.content) > 1000:
                df = pd.read_csv(io.BytesIO(response.content))
                df.to_csv(cache_file, index=False)
                logger.info(f"Downloaded equity bhavcopy for {target_date}: {len(df)} records (new format)")
                return df
            else:
                logger.debug(f"New format failed: HTTP {response.status_code}")
                
        except Exception as e:
            logger.debug(f"Error with new equity format for {target_date}: {e}")
        
        # Try old archive format
        try:
            session = self._get_nse_session()
            response = session.get(old_url, timeout=30)
            
            if response.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                    csv_name = zf.namelist()[0]
                    with zf.open(csv_name) as f:
                        df = pd.read_csv(f)
                
                df.to_csv(cache_file, index=False)
                logger.info(f"Downloaded equity bhavcopy for {target_date}: {len(df)} records (old format)")
                return df
                
            elif response.status_code == 404:
                logger.debug(f"Equity bhavcopy not found for {target_date} via direct download")
            else:
                logger.debug(f"Direct download failed: HTTP {response.status_code}")
                
        except Exception as e:
            logger.debug(f"Error downloading equity bhavcopy for {target_date}: {e}")
        
        # Fallback: Try jugaad-data library
        if JUGAAD_AVAILABLE:
            try:
                logger.info(f"Trying jugaad-data fallback for equity bhavcopy {target_date}...")
                jugaad_dir = self.cache_dir / "jugaad_temp"
                jugaad_dir.mkdir(parents=True, exist_ok=True)
                
                bhavcopy_save(target_date, str(jugaad_dir))
                
                # Find the downloaded file
                for f in jugaad_dir.glob("*.csv"):
                    df = pd.read_csv(f)
                    df.to_csv(cache_file, index=False)
                    f.unlink()  # Clean up temp file
                    logger.info(f"Downloaded equity bhavcopy via jugaad-data for {target_date}: {len(df)} records")
                    return df
                    
            except Exception as e:
                if "404" in str(e) or "holiday" in str(e).lower():
                    logger.debug(f"{target_date} is likely a holiday")
                else:
                    logger.debug(f"jugaad-data equity fallback failed for {target_date}: {e}")
        
        logger.warning(f"Equity bhavcopy not available for {target_date}")
        return None
    
    def get_spot_prices(self, equity_df: pd.DataFrame, fo_df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract spot prices from equity bhavcopy or index futures.
        
        Args:
            equity_df: Equity bhavcopy DataFrame
            fo_df: F&O bhavcopy DataFrame
            
        Returns:
            Dictionary of symbol -> spot price
        """
        spot_prices = {}
        
        # Get equity prices
        if equity_df is not None:
            for symbol in self.symbols:
                if symbol in ["NIFTY", "BANKNIFTY", "FINNIFTY"]:
                    continue  # Handle indices separately
                
                row = equity_df[equity_df["SYMBOL"] == symbol]
                if not row.empty:
                    spot_prices[symbol] = float(row.iloc[0]["CLOSE"])
        
        # Get index prices from futures (current month)
        if fo_df is not None:
            for idx_symbol in ["NIFTY", "BANKNIFTY", "FINNIFTY"]:
                if idx_symbol not in self.symbols:
                    continue
                
                # Find current month future
                idx_futures = fo_df[
                    (fo_df["SYMBOL"] == idx_symbol) & 
                    (fo_df["INSTRUMENT"] == "FUTIDX")
                ]
                
                if not idx_futures.empty:
                    # Get nearest expiry
                    idx_futures = idx_futures.copy()
                    idx_futures["EXPIRY_DT"] = pd.to_datetime(idx_futures["EXPIRY_DT"])
                    nearest = idx_futures.sort_values("EXPIRY_DT").iloc[0]
                    spot_prices[idx_symbol] = float(nearest["CLOSE"])
        
        return spot_prices
    
    def calculate_iv_and_greeks(
        self,
        option_type: str,
        spot: float,
        strike: float,
        expiry_date: date,
        option_price: float,
        trade_date: date
    ) -> Dict[str, float]:
        """
        Calculate IV and Greeks for an option using Black-Scholes/Black-76.
        
        Args:
            option_type: 'CE' for call, 'PE' for put
            spot: Spot/underlying price
            strike: Strike price
            expiry_date: Option expiry date
            option_price: Option premium/close price
            trade_date: Date of the trade
            
        Returns:
            Dictionary with iv, delta, gamma, theta, vega
        """
        result = {
            "iv": 0.0,
            "delta": 0.0,
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0
        }
        
        try:
            # Calculate time to expiry in years
            days_to_expiry = (expiry_date - trade_date).days
            if days_to_expiry <= 0:
                return result
            
            t = days_to_expiry / 365.0
            
            # Use options_pricer for IV calculation
            is_call = option_type == "CE"
            
            # Calculate IV using Newton-Raphson or bisection
            iv = self._calculate_implied_volatility(
                spot, strike, t, RISK_FREE_RATE, option_price, is_call
            )
            
            if iv and iv > 0:
                result["iv"] = iv
                
                # Calculate Greeks with the IV
                greeks = options_pricer.calculate_greeks(
                    spot_price=spot,
                    strike_price=strike,
                    time_to_expiry=t,
                    risk_free_rate=RISK_FREE_RATE,
                    volatility=iv,
                    is_call=is_call
                )
                
                if greeks:
                    result["delta"] = greeks.get("delta", 0.0)
                    result["gamma"] = greeks.get("gamma", 0.0)
                    result["theta"] = greeks.get("theta", 0.0)
                    result["vega"] = greeks.get("vega", 0.0)
                    
        except Exception as e:
            logger.debug(f"Error calculating IV/Greeks: {e}")
        
        return result
    
    def _calculate_implied_volatility(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        market_price: float,
        is_call: bool,
        max_iterations: int = 100,
        precision: float = 1e-6
    ) -> Optional[float]:
        """
        Calculate implied volatility using bisection method.
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            market_price: Market option price
            is_call: True for call, False for put
            
        Returns:
            Implied volatility or None if not found
        """
        from scipy.stats import norm
        import math
        
        if market_price <= 0 or T <= 0 or S <= 0 or K <= 0:
            return None
        
        def black_scholes_price(sigma: float) -> float:
            """Calculate BS price for given volatility."""
            try:
                d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
                d2 = d1 - sigma * math.sqrt(T)
                
                if is_call:
                    price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
                else:
                    price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
                
                return price
            except:
                return 0.0
        
        # Bisection search
        low_vol = 0.01
        high_vol = 5.0
        
        for _ in range(max_iterations):
            mid_vol = (low_vol + high_vol) / 2
            mid_price = black_scholes_price(mid_vol)
            
            if abs(mid_price - market_price) < precision:
                return mid_vol
            
            if mid_price > market_price:
                high_vol = mid_vol
            else:
                low_vol = mid_vol
            
            if high_vol - low_vol < precision:
                break
        
        return (low_vol + high_vol) / 2
    
    def build_options_chain(
        self,
        fo_df: pd.DataFrame,
        symbol: str,
        spot_price: float,
        trade_date: date
    ) -> pd.DataFrame:
        """
        Build options chain from bhavcopy data with IV and Greeks.
        
        Args:
            fo_df: F&O bhavcopy DataFrame
            symbol: Underlying symbol
            spot_price: Spot price
            trade_date: Date of the data
            
        Returns:
            DataFrame with options chain including IV and Greeks
        """
        # Filter options for this symbol
        instrument_type = "OPTIDX" if symbol in ["NIFTY", "BANKNIFTY", "FINNIFTY"] else "OPTSTK"
        
        options = fo_df[
            (fo_df["SYMBOL"] == symbol) & 
            (fo_df["INSTRUMENT"] == instrument_type)
        ].copy()
        
        if options.empty:
            return pd.DataFrame()
        
        # Parse expiry dates
        options["EXPIRY_DT"] = pd.to_datetime(options["EXPIRY_DT"])
        
        # Get nearest expiry (weekly for indices, monthly for stocks)
        min_expiry = options["EXPIRY_DT"].min()
        options = options[options["EXPIRY_DT"] == min_expiry]
        
        expiry_date = min_expiry.date()
        
        # Calculate IV and Greeks for each option
        greeks_data = []
        for _, row in options.iterrows():
            strike = float(row["STRIKE_PR"])
            option_type = row["OPTION_TYP"]
            close_price = float(row["CLOSE"])
            
            iv_greeks = self.calculate_iv_and_greeks(
                option_type=option_type,
                spot=spot_price,
                strike=strike,
                expiry_date=expiry_date,
                option_price=close_price,
                trade_date=trade_date
            )
            
            greeks_data.append({
                "strike": strike,
                "option_type": option_type,
                "close": close_price,
                "open": float(row.get("OPEN", 0)),
                "high": float(row.get("HIGH", 0)),
                "low": float(row.get("LOW", 0)),
                "oi": int(row.get("OPEN_INT", 0)),
                "volume": int(row.get("CONTRACTS", 0)),
                "expiry": expiry_date,
                **iv_greeks
            })
        
        return pd.DataFrame(greeks_data)
    
    def calculate_oi_metrics(self, options_chain: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate OI-based metrics from options chain.
        
        Args:
            options_chain: DataFrame with options data
            
        Returns:
            Dictionary with OI metrics
        """
        metrics = {
            "pcr": 0.0,
            "call_oi_total": 0,
            "put_oi_total": 0,
            "max_pain": 0.0,
            "call_oi_change": 0.0,
            "put_oi_change": 0.0,
        }
        
        if options_chain.empty:
            return metrics
        
        calls = options_chain[options_chain["option_type"] == "CE"]
        puts = options_chain[options_chain["option_type"] == "PE"]
        
        call_oi = calls["oi"].sum() if not calls.empty else 0
        put_oi = puts["oi"].sum() if not puts.empty else 0
        
        metrics["call_oi_total"] = int(call_oi)
        metrics["put_oi_total"] = int(put_oi)
        
        if call_oi > 0:
            metrics["pcr"] = put_oi / call_oi
        
        # Calculate max pain (strike where total loss is minimum)
        if not calls.empty and not puts.empty:
            strikes = sorted(set(calls["strike"].tolist() + puts["strike"].tolist()))
            min_loss = float("inf")
            max_pain_strike = 0
            
            for strike in strikes:
                call_loss = sum(
                    max(0, strike - c["strike"]) * c["oi"] 
                    for _, c in calls.iterrows()
                )
                put_loss = sum(
                    max(0, p["strike"] - strike) * p["oi"]
                    for _, p in puts.iterrows()
                )
                total_loss = call_loss + put_loss
                
                if total_loss < min_loss:
                    min_loss = total_loss
                    max_pain_strike = strike
            
            metrics["max_pain"] = max_pain_strike
        
        return metrics
    
    def generate_features(
        self,
        symbol: str,
        spot_price: float,
        options_chain: pd.DataFrame,
        oi_metrics: Dict[str, float],
        trade_date: date
    ) -> Dict[str, float]:
        """
        Generate ML features from historical data.
        
        Args:
            symbol: Underlying symbol
            spot_price: Spot/close price
            options_chain: Options chain with IV and Greeks
            oi_metrics: OI-based metrics
            trade_date: Date of the data
            
        Returns:
            Dictionary of feature name -> value
        """
        features = {}
        
        # Basic price feature
        features["spot_price"] = spot_price
        
        # OI Features
        features["pcr"] = oi_metrics.get("pcr", 0.0)
        features["call_oi_total"] = oi_metrics.get("call_oi_total", 0)
        features["put_oi_total"] = oi_metrics.get("put_oi_total", 0)
        features["max_pain"] = oi_metrics.get("max_pain", 0.0)
        
        if not options_chain.empty:
            # IV Features
            calls = options_chain[options_chain["option_type"] == "CE"]
            puts = options_chain[options_chain["option_type"] == "PE"]
            
            if not calls.empty:
                features["iv_call_mean"] = calls["iv"].mean()
                features["iv_call_atm"] = self._get_atm_value(calls, spot_price, "iv")
            else:
                features["iv_call_mean"] = 0.0
                features["iv_call_atm"] = 0.0
            
            if not puts.empty:
                features["iv_put_mean"] = puts["iv"].mean()
                features["iv_put_atm"] = self._get_atm_value(puts, spot_price, "iv")
            else:
                features["iv_put_mean"] = 0.0
                features["iv_put_atm"] = 0.0
            
            # Average IV
            all_ivs = options_chain[options_chain["iv"] > 0]["iv"]
            features["iv_current"] = all_ivs.mean() if not all_ivs.empty else 0.0
            
            # IV skew (put IV - call IV for ATM)
            features["iv_skew"] = features["iv_put_atm"] - features["iv_call_atm"]
            
            # Greeks aggregation (ATM options)
            atm_calls = self._filter_atm(calls, spot_price)
            atm_puts = self._filter_atm(puts, spot_price)
            
            features["delta_call_atm"] = atm_calls["delta"].mean() if not atm_calls.empty else 0.5
            features["delta_put_atm"] = atm_puts["delta"].mean() if not atm_puts.empty else -0.5
            features["gamma_atm"] = (
                (atm_calls["gamma"].mean() if not atm_calls.empty else 0) +
                (atm_puts["gamma"].mean() if not atm_puts.empty else 0)
            ) / 2
            features["theta_atm"] = (
                (atm_calls["theta"].mean() if not atm_calls.empty else 0) +
                (atm_puts["theta"].mean() if not atm_puts.empty else 0)
            ) / 2
            features["vega_atm"] = (
                (atm_calls["vega"].mean() if not atm_calls.empty else 0) +
                (atm_puts["vega"].mean() if not atm_puts.empty else 0)
            ) / 2
            
            # Volume metrics
            features["call_volume_total"] = calls["volume"].sum() if not calls.empty else 0
            features["put_volume_total"] = puts["volume"].sum() if not puts.empty else 0
            features["volume_pcr"] = (
                features["put_volume_total"] / features["call_volume_total"]
                if features["call_volume_total"] > 0 else 0
            )
            
            # Days to expiry
            if not options_chain.empty:
                expiry = options_chain.iloc[0]["expiry"]
                if isinstance(expiry, date):
                    features["dte"] = (expiry - trade_date).days
                else:
                    features["dte"] = 7  # Default
            else:
                features["dte"] = 7
        
        # Time features
        features["day_of_week"] = trade_date.weekday()
        features["day_of_month"] = trade_date.day
        features["is_expiry_week"] = 1 if features.get("dte", 7) <= 7 else 0
        features["is_monthly_expiry"] = 1 if trade_date.day >= 25 else 0
        
        return features
    
    def _get_atm_value(self, options: pd.DataFrame, spot: float, column: str) -> float:
        """Get value of ATM option for a column."""
        if options.empty:
            return 0.0
        
        options = options.copy()
        options["distance"] = abs(options["strike"] - spot)
        atm = options.nsmallest(1, "distance")
        
        return float(atm[column].iloc[0]) if not atm.empty else 0.0
    
    def _filter_atm(self, options: pd.DataFrame, spot: float, n_strikes: int = 3) -> pd.DataFrame:
        """Filter to ATM options (within n strikes)."""
        if options.empty:
            return options
        
        options = options.copy()
        options["distance"] = abs(options["strike"] - spot)
        return options.nsmallest(n_strikes, "distance")
    
    def process_date(self, target_date: date) -> Dict[str, Any]:
        """
        Process historical data for a single date.
        
        Args:
            target_date: Date to process
            
        Returns:
            Dictionary with processing results
        """
        result = {
            "date": target_date,
            "collected": 0,
            "errors": 0,
            "symbols": {}
        }
        
        # Check if already processed
        existing = self._get_existing_snapshots(target_date)
        if existing:
            logger.info(f"Date {target_date} already has {len(existing)} snapshots")
            result["skipped"] = len(existing)
            return result
        
        # Download bhavcopy
        fo_df = self.download_bhavcopy(target_date)
        if fo_df is None:
            result["errors"] = 1
            result["error_msg"] = "Failed to download F&O bhavcopy"
            return result
        
        # Download equity bhavcopy for spot prices
        equity_df = self.download_equity_bhavcopy(target_date)
        
        # Get spot prices
        spot_prices = self.get_spot_prices(equity_df, fo_df)
        
        # Process each symbol
        for symbol in self.symbols:
            try:
                spot_price = spot_prices.get(symbol)
                if not spot_price:
                    logger.warning(f"No spot price for {symbol} on {target_date}")
                    continue
                
                # Build options chain with IV and Greeks
                options_chain = self.build_options_chain(fo_df, symbol, spot_price, target_date)
                
                # Calculate OI metrics
                oi_metrics = self.calculate_oi_metrics(options_chain)
                
                # Generate features
                features = self.generate_features(
                    symbol, spot_price, options_chain, oi_metrics, target_date
                )
                
                # Store in database
                self._store_snapshot(
                    symbol=symbol,
                    snapshot_time=datetime.combine(target_date, datetime.min.time().replace(hour=15, minute=30)),
                    spot_price=spot_price,
                    features=features,
                    source="historical"
                )
                
                result["collected"] += 1
                result["symbols"][symbol] = {
                    "spot": spot_price,
                    "features": len(features),
                    "has_options": not options_chain.empty,
                    "pcr": oi_metrics.get("pcr", 0)
                }
                
            except Exception as e:
                logger.error(f"Error processing {symbol} for {target_date}: {e}")
                result["errors"] += 1
        
        logger.info(f"Processed {target_date}: {result['collected']} symbols, {result['errors']} errors")
        return result
    
    def _get_existing_snapshots(self, target_date: date) -> List[str]:
        """Get list of symbols already processed for a date."""
        try:
            conn = database._get_connection()
            cursor = conn.cursor()
            
            date_start = datetime.combine(target_date, datetime.min.time())
            date_end = datetime.combine(target_date, datetime.max.time())
            
            cursor.execute("""
                SELECT DISTINCT underlying 
                FROM ml_feature_snapshots 
                WHERE snapshot_time >= ? AND snapshot_time <= ?
                AND source = 'historical'
            """, (date_start, date_end))
            
            return [row[0] for row in cursor.fetchall()]
            
        except Exception as e:
            logger.error(f"Error checking existing snapshots: {e}")
            return []
    
    def _store_snapshot(
        self,
        symbol: str,
        snapshot_time: datetime,
        spot_price: float,
        features: Dict[str, float],
        source: str = "historical"
    ) -> bool:
        """Store a feature snapshot in the database."""
        try:
            conn = database._get_connection()
            cursor = conn.cursor()
            
            metadata = {
                "source": source,
                "collected_at": datetime.now().isoformat()
            }
            
            cursor.execute("""
                INSERT INTO ml_feature_snapshots 
                (underlying, snapshot_time, spot_price, features, metadata, source)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                snapshot_time,
                spot_price,
                json.dumps(features),
                json.dumps(metadata),
                source
            ))
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to store snapshot: {e}")
            return False
    
    def collect_date_range(
        self,
        start_date: date,
        end_date: date,
        skip_weekends: bool = True,
        delay_seconds: float = 1.0
    ) -> Dict[str, Any]:
        """
        Collect historical data for a date range.
        
        Args:
            start_date: Start date
            end_date: End date
            skip_weekends: Skip Saturday and Sunday
            delay_seconds: Delay between requests to avoid rate limiting
            
        Returns:
            Summary of collection results
        """
        results = {
            "start_date": start_date,
            "end_date": end_date,
            "total_dates": 0,
            "processed_dates": 0,
            "total_collected": 0,
            "total_errors": 0,
            "skipped_dates": 0,
            "dates": {}
        }
        
        current_date = start_date
        while current_date <= end_date:
            # Skip weekends
            if skip_weekends and current_date.weekday() >= 5:
                current_date += timedelta(days=1)
                continue
            
            results["total_dates"] += 1
            
            try:
                date_result = self.process_date(current_date)
                
                if "skipped" in date_result:
                    results["skipped_dates"] += 1
                else:
                    results["processed_dates"] += 1
                    results["total_collected"] += date_result["collected"]
                    results["total_errors"] += date_result["errors"]
                
                results["dates"][current_date.isoformat()] = date_result
                
            except Exception as e:
                logger.error(f"Failed to process {current_date}: {e}")
                results["total_errors"] += 1
            
            # Rate limiting
            time.sleep(delay_seconds)
            current_date += timedelta(days=1)
        
        logger.info(
            f"Collection complete: {results['processed_dates']} dates, "
            f"{results['total_collected']} snapshots, {results['total_errors']} errors"
        )
        
        return results
    
    def fill_missing_dates(self, lookback_days: int = 30) -> Dict[str, Any]:
        """
        Find and fill missing dates in the database.
        
        Args:
            lookback_days: Number of days to look back
            
        Returns:
            Summary of filled dates
        """
        end_date = date.today() - timedelta(days=1)  # Yesterday
        start_date = end_date - timedelta(days=lookback_days)
        
        # Get dates that already have data
        existing_dates = self._get_dates_with_data(start_date, end_date)
        
        # Find missing dates (excluding weekends)
        missing_dates = []
        current = start_date
        while current <= end_date:
            if current.weekday() < 5 and current not in existing_dates:
                missing_dates.append(current)
            current += timedelta(days=1)
        
        if not missing_dates:
            logger.info("No missing dates found")
            return {"missing_dates": 0, "filled": 0}
        
        logger.info(f"Found {len(missing_dates)} missing dates, filling...")
        
        # Fill missing dates
        results = {
            "missing_dates": len(missing_dates),
            "filled": 0,
            "errors": 0,
            "dates": []
        }
        
        for missing_date in missing_dates:
            try:
                result = self.process_date(missing_date)
                if result["collected"] > 0:
                    results["filled"] += 1
                results["dates"].append({
                    "date": missing_date.isoformat(),
                    "collected": result["collected"]
                })
            except Exception as e:
                logger.error(f"Failed to fill {missing_date}: {e}")
                results["errors"] += 1
            
            time.sleep(1.0)  # Rate limiting
        
        return results
    
    def _get_dates_with_data(self, start_date: date, end_date: date) -> set:
        """Get set of dates that have historical data."""
        try:
            conn = database._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT DISTINCT DATE(snapshot_time) 
                FROM ml_feature_snapshots 
                WHERE snapshot_time >= ? AND snapshot_time <= ?
            """, (start_date, end_date))
            
            return {
                datetime.strptime(row[0], "%Y-%m-%d").date() 
                for row in cursor.fetchall()
            }
            
        except Exception as e:
            logger.error(f"Error getting dates with data: {e}")
            return set()
    
    def collect_from_kite(self, days: int = 30, interval: str = "day") -> Dict[str, Any]:
        """
        Collect historical data using Kite API instead of NSE bhavcopy.
        
        This is an alternative when NSE blocks direct bhavcopy downloads.
        Uses Kite's historical data API to get spot/futures OHLCV data.
        
        NOTE: This only collects spot/futures data. For full options chain
        data with IV/Greeks, use the live collector during market hours
        or manually download bhavcopy files from NSE website.
        
        Args:
            days: Number of days of historical data to collect
            interval: Candle interval (day, 15minute, 5minute, etc.)
            
        Returns:
            Dictionary with collection statistics
        """
        from data.data_fetcher import data_fetcher
        from ml.feature_engineer import FeatureEngineer
        
        results = {
            "method": "kite_api",
            "days_requested": days,
            "interval": interval,
            "symbols_collected": 0,
            "total_candles": 0,
            "snapshots_created": 0,
            "errors": []
        }
        
        feature_engineer = FeatureEngineer()
        
        for symbol in self.symbols:
            try:
                logger.info(f"Collecting Kite historical data for {symbol}...")
                
                # Determine correct exchange for the symbol
                from config.settings import UNDERLYING_ASSETS
                asset_cfg = UNDERLYING_ASSETS.get(symbol, {})
                exchange = asset_cfg.get("exchange", "NSE")
                
                # Get historical OHLCV from Kite
                hist_df = data_fetcher.get_historical_data(
                    symbol=symbol,
                    interval=interval,
                    days=days,
                    exchange=exchange
                )
                
                if hist_df.empty:
                    logger.warning(f"No historical data for {symbol}")
                    results["errors"].append(f"{symbol}: No data returned")
                    continue
                
                results["total_candles"] += len(hist_df)
                results["symbols_collected"] += 1
                
                # Create a simplified snapshot for each day
                # Note: This won't have full options chain data
                for idx, row in hist_df.iterrows():
                    try:
                        snapshot_time = idx if isinstance(idx, datetime) else datetime.combine(idx, datetime.min.time())
                        spot_price = float(row.get("close", row.get("ltp", 0)))
                        
                        if spot_price <= 0:
                            continue
                        
                        # Build basic features from OHLCV
                        features = {
                            "spot_price": spot_price,
                            "open": float(row.get("open", spot_price)),
                            "high": float(row.get("high", spot_price)),
                            "low": float(row.get("low", spot_price)),
                            "close": spot_price,
                            "volume": float(row.get("volume", 0)),
                            "day_range": float(row.get("high", spot_price) - row.get("low", spot_price)),
                            "day_range_pct": (row.get("high", spot_price) - row.get("low", spot_price)) / spot_price * 100 if spot_price > 0 else 0,
                            # Add time features
                            "hour": snapshot_time.hour if hasattr(snapshot_time, 'hour') else 12,
                            "day_of_week": snapshot_time.weekday() if hasattr(snapshot_time, 'weekday') else 0,
                            # Mark as partial data
                            "data_source": "kite_historical",
                            "has_options_data": False
                        }
                        
                        # Store snapshot
                        self._store_snapshot(
                            underlying=symbol,
                            snapshot_time=snapshot_time,
                            spot_price=spot_price,
                            features=features,
                            source="kite_historical"
                        )
                        results["snapshots_created"] += 1
                        
                    except Exception as e:
                        logger.debug(f"Error processing row for {symbol}: {e}")
                        continue
                
                logger.info(f"Collected {len(hist_df)} candles for {symbol}")
                
            except Exception as e:
                logger.error(f"Error collecting Kite data for {symbol}: {e}")
                results["errors"].append(f"{symbol}: {str(e)}")
        
        logger.info(f"Kite historical collection complete: {results['symbols_collected']} symbols, {results['snapshots_created']} snapshots")
        return results
    
    def _store_snapshot(
        self, 
        underlying: str, 
        snapshot_time: datetime, 
        spot_price: float, 
        features: Dict[str, Any],
        source: str = "historical"
    ) -> bool:
        """Store a feature snapshot in the database."""
        try:
            conn = database._get_connection()
            cursor = conn.cursor()
            
            # Calculate feature metadata
            feature_count = len(features)
            has_options_data = 1 if features.get('has_options_data', False) else 0
            has_oi_data = 1 if features.get('total_oi', 0) > 0 else 0
            has_greeks = 1 if features.get('atm_iv', 0) > 0 else 0
            
            cursor.execute("""
                INSERT INTO ml_feature_snapshots 
                (underlying, snapshot_time, spot_price, features_json, feature_count, 
                 has_options_data, has_oi_data, has_greeks, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                underlying,
                snapshot_time.isoformat(),
                spot_price,
                json.dumps(features),
                feature_count,
                has_options_data,
                has_oi_data,
                has_greeks,
                source
            ))
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error storing snapshot: {e}")
            return False
    
    def get_collection_status(self) -> Dict[str, Any]:
        """Get status of historical data collection."""
        try:
            conn = database._get_connection()
            cursor = conn.cursor()
            
            # Total snapshots by source
            cursor.execute("SELECT COUNT(*) FROM ml_feature_snapshots WHERE source = 'historical'")
            total_historical = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM ml_feature_snapshots WHERE source = 'kite_historical'")
            total_kite = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM ml_feature_snapshots WHERE source = 'live' OR source IS NULL")
            total_live = cursor.fetchone()[0]
            
            # Date range
            cursor.execute("""
                SELECT MIN(DATE(snapshot_time)), MAX(DATE(snapshot_time))
                FROM ml_feature_snapshots WHERE source IN ('historical', 'kite_historical')
            """)
            date_range = cursor.fetchone()
            
            # Unique symbols
            cursor.execute("SELECT DISTINCT underlying FROM ml_feature_snapshots WHERE source IN ('historical', 'kite_historical')")
            symbols = [row[0] for row in cursor.fetchall()]
            
            # Cache size
            cache_files = list(self.cache_dir.glob("*.csv"))
            
            return {
                "historical_snapshots": total_historical,
                "kite_historical_snapshots": total_kite,
                "live_snapshots": total_live,
                "total_snapshots": total_historical + total_kite + total_live,
                "date_range": {
                    "start": date_range[0] if date_range[0] else None,
                    "end": date_range[1] if date_range[1] else None
                },
                "symbols": symbols,
                "cache_files": len(cache_files),
                "cache_size_mb": sum(f.stat().st_size for f in cache_files) / (1024 * 1024)
            }
            
        except Exception as e:
            logger.error(f"Error getting collection status: {e}")
            return {"error": str(e)}


# Singleton instance
_collector: Optional[HistoricalDataCollector] = None


def get_historical_collector() -> HistoricalDataCollector:
    """Get the singleton historical data collector instance."""
    global _collector
    if _collector is None:
        _collector = HistoricalDataCollector()
    return _collector
