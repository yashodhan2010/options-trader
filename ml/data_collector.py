"""
Historical Data Collector for ML Training

Fetches and caches 3+ months of historical market data for ML model training.
Handles rate limiting, incremental updates, and data validation.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

from config.settings import ML_CONFIG, UNDERLYING_ASSETS, DATA_DIR
from core.database import database
from core.logger import logger


class DataCollector:
    """
    Collect and cache historical market data for ML training.
    
    Features:
    - Fetch OHLCV data for underlying assets
    - Incremental updates (only fetch missing dates)
    - Rate limiting to respect API limits
    - Data validation and cleaning
    - Cache to SQLite for fast access
    """
    
    def __init__(self, data_fetcher=None):
        """
        Initialize data collector.
        
        Args:
            data_fetcher: Optional DataFetcher instance (lazy loaded if not provided)
        """
        self._data_fetcher = data_fetcher
        self.historical_days = ML_CONFIG.get("historical_days", 180)
        self.rate_limit_delay = 0.5  # Seconds between API calls
        self.batch_size = 400  # Days per API call (Kite allows 2000 for daily candles)
        
        logger.info(f"DataCollector initialized for {self.historical_days} days of history")
    
    @property
    def data_fetcher(self):
        """Lazy load data fetcher."""
        if self._data_fetcher is None:
            from data.data_fetcher import data_fetcher
            self._data_fetcher = data_fetcher
        return self._data_fetcher
    
    def collect_historical_data(
        self,
        symbols: List[str] = None,
        days: int = None,
        force_refresh: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect historical OHLCV data for all symbols.
        
        Args:
            symbols: List of symbols to collect (default: all underlying assets)
            days: Number of historical days (default: from config)
            force_refresh: If True, re-fetch all data regardless of cache
            
        Returns:
            Dict mapping symbol to DataFrame of OHLCV data
        """
        symbols = symbols or list(UNDERLYING_ASSETS.keys())
        days = days or self.historical_days
        
        results = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        for symbol in symbols:
            try:
                logger.info(f"Collecting data for {symbol}...")
                
                # Check cache first
                if not force_refresh:
                    cached_data = self._get_cached_data(symbol, start_date, end_date)
                    if cached_data is not None and len(cached_data) > days * 0.9:
                        logger.info(f"Using cached data for {symbol}: {len(cached_data)} records")
                        results[symbol] = cached_data
                        continue
                
                # Fetch from API
                df = self._fetch_historical_ohlcv(symbol, start_date, end_date)
                
                if df is not None and len(df) > 0:
                    # Validate and clean
                    df = self._validate_and_clean(df)
                    
                    # Cache to database
                    self._cache_data(symbol, df)
                    
                    results[symbol] = df
                    logger.info(f"Collected {len(df)} records for {symbol}")
                else:
                    logger.warning(f"No data fetched for {symbol}")
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.error(f"Error collecting data for {symbol}: {e}")
        
        return results
    
    def update_data(self, symbols: List[str] = None) -> Dict[str, int]:
        """
        Incrementally update cached data with latest prices.
        
        Args:
            symbols: List of symbols to update (default: all)
            
        Returns:
            Dict mapping symbol to number of new records added
        """
        symbols = symbols or list(UNDERLYING_ASSETS.keys())
        updates = {}
        
        for symbol in symbols:
            try:
                # Find latest cached date
                cached = database.get_cached_market_data(symbol, interval="day")
                
                if cached:
                    # Parse dates and find latest
                    dates = [datetime.strptime(r["date"], "%Y-%m-%d") for r in cached if r.get("date")]
                    latest_cached = max(dates) if dates else datetime.now() - timedelta(days=self.historical_days)
                else:
                    latest_cached = datetime.now() - timedelta(days=self.historical_days)
                
                # Fetch from day after latest cached to today
                start_date = latest_cached + timedelta(days=1)
                end_date = datetime.now()
                
                if start_date >= end_date:
                    logger.debug(f"{symbol} data is up to date")
                    updates[symbol] = 0
                    continue
                
                df = self._fetch_historical_ohlcv(symbol, start_date, end_date)
                
                if df is not None and len(df) > 0:
                    self._cache_data(symbol, df)
                    updates[symbol] = len(df)
                    logger.info(f"Updated {symbol} with {len(df)} new records")
                else:
                    updates[symbol] = 0
                
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.error(f"Error updating {symbol}: {e}")
                updates[symbol] = -1
        
        return updates
    
    def get_training_dataframe(
        self,
        symbols: List[str] = None,
        days: int = None,
        include_technicals: bool = True
    ) -> pd.DataFrame:
        """
        Get consolidated DataFrame ready for ML training.
        
        Args:
            symbols: List of symbols to include
            days: Number of days of history
            include_technicals: Whether to calculate technical indicators
            
        Returns:
            DataFrame with all data consolidated
        """
        symbols = symbols or list(UNDERLYING_ASSETS.keys())
        days = days or self.historical_days
        
        all_data = []
        
        for symbol in symbols:
            try:
                # Get cached data
                start_date = datetime.now() - timedelta(days=days)
                df = self._get_cached_data(symbol, start_date, datetime.now())
                
                if df is None or len(df) == 0:
                    # Try to fetch if not cached
                    collected = self.collect_historical_data([symbol], days)
                    df = collected.get(symbol)
                
                if df is not None and len(df) > 0:
                    df = df.copy()
                    df["symbol"] = symbol
                    
                    if include_technicals:
                        df = self._add_technical_indicators(df)
                    
                    all_data.append(df)
                    
            except Exception as e:
                logger.error(f"Error getting training data for {symbol}: {e}")
        
        if not all_data:
            return pd.DataFrame()
        
        return pd.concat(all_data, ignore_index=True)
    
    def _fetch_historical_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from API."""
        try:
            # Calculate days
            days = (end_date - start_date).days
            
            # Get the trading symbol from UNDERLYING_ASSETS mapping
            trading_symbol = UNDERLYING_ASSETS.get(symbol, {}).get("symbol", symbol)
            exchange = UNDERLYING_ASSETS.get(symbol, {}).get("exchange", "NSE")
            
            # Use data_fetcher's historical data method
            df = self.data_fetcher.get_historical_data(
                symbol=trading_symbol,
                interval="day",
                days=min(days, self.batch_size),
                exchange=exchange
            )
            
            if df is None or len(df) == 0:
                return None
            
            # If we need more than batch_size days, fetch in batches
            if days > self.batch_size:
                all_data = [df]
                remaining_days = days - self.batch_size
                current_end = start_date + timedelta(days=remaining_days)
                
                while remaining_days > 0:
                    time.sleep(self.rate_limit_delay)
                    
                    batch_days = min(remaining_days, self.batch_size)
                    batch_df = self.data_fetcher.get_historical_data(
                        symbol=trading_symbol,
                        interval="day",
                        days=batch_days,
                        exchange=exchange
                    )
                    
                    if batch_df is not None and len(batch_df) > 0:
                        all_data.append(batch_df)
                    
                    remaining_days -= batch_days
                
                if all_data:
                    df = pd.concat(all_data, ignore_index=False)
                    df = df[~df.index.duplicated(keep='last')]
                    df = df.sort_index()
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None
    
    def _get_cached_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Get data from cache."""
        try:
            records = database.get_cached_market_data(
                symbol=symbol,
                interval="day",
                start_date=start_date,
                end_date=end_date
            )
            
            if not records:
                return None
            
            df = pd.DataFrame(records)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            df = df.sort_index()
            
            # Select relevant columns
            columns = ["open", "high", "low", "close", "volume"]
            available_cols = [c for c in columns if c in df.columns]
            
            return df[available_cols]
            
        except Exception as e:
            logger.error(f"Error getting cached data for {symbol}: {e}")
            return None
    
    def _cache_data(self, symbol: str, df: pd.DataFrame) -> None:
        """Cache data to database."""
        try:
            for idx, row in df.iterrows():
                date = idx if isinstance(idx, datetime) else pd.to_datetime(idx)
                
                database.save_market_data(
                    symbol=symbol,
                    data_type="index" if symbol in UNDERLYING_ASSETS else "equity",
                    interval="day",
                    date=date,
                    ohlcv={
                        "open": row.get("open", 0),
                        "high": row.get("high", 0),
                        "low": row.get("low", 0),
                        "close": row.get("close", 0),
                        "volume": row.get("volume", 0),
                    }
                )
        except Exception as e:
            logger.error(f"Error caching data for {symbol}: {e}")
    
    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean OHLCV data."""
        if df is None or len(df) == 0:
            return df
        
        df = df.copy()
        
        # Remove rows with missing critical data
        required_cols = ["open", "high", "low", "close"]
        available_cols = [c for c in required_cols if c in df.columns]
        
        if available_cols:
            df = df.dropna(subset=available_cols)
        
        # Remove rows with invalid prices (zero or negative)
        for col in available_cols:
            df = df[df[col] > 0]
        
        # Ensure high >= low
        if "high" in df.columns and "low" in df.columns:
            df = df[df["high"] >= df["low"]]
        
        # Forward fill small gaps (max 2 days)
        df = df.ffill(limit=2)
        
        # Sort by index
        df = df.sort_index()
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to DataFrame."""
        if df is None or len(df) < 20:
            return df
        
        df = df.copy()
        
        try:
            closes = df["close"].values
            highs = df["high"].values if "high" in df.columns else closes
            lows = df["low"].values if "low" in df.columns else closes
            
            # Returns
            df["return_1d"] = df["close"].pct_change() * 100
            df["return_5d"] = df["close"].pct_change(5) * 100
            df["return_10d"] = df["close"].pct_change(10) * 100
            df["return_20d"] = df["close"].pct_change(20) * 100
            
            # SMAs
            df["sma_5"] = df["close"].rolling(5).mean()
            df["sma_10"] = df["close"].rolling(10).mean()
            df["sma_20"] = df["close"].rolling(20).mean()
            df["sma_50"] = df["close"].rolling(50).mean()
            
            # SMA crossovers
            df["sma_5_10_cross"] = (df["sma_5"] > df["sma_10"]).astype(int)
            df["sma_10_20_cross"] = (df["sma_10"] > df["sma_20"]).astype(int)
            
            # RSI
            df["rsi_14"] = self._calculate_rsi_series(df["close"], 14)
            
            # ATR
            df["atr_14"] = self._calculate_atr_series(df, 14)
            df["atr_percent"] = df["atr_14"] / df["close"] * 100
            
            # Volatility (HV)
            df["hv_10"] = df["return_1d"].rolling(10).std() * np.sqrt(252)
            df["hv_20"] = df["return_1d"].rolling(20).std() * np.sqrt(252)
            
            # Bollinger Bands
            df["bb_middle"] = df["close"].rolling(20).mean()
            df["bb_std"] = df["close"].rolling(20).std()
            df["bb_upper"] = df["bb_middle"] + 2 * df["bb_std"]
            df["bb_lower"] = df["bb_middle"] - 2 * df["bb_std"]
            df["bb_percent_b"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
            df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"] * 100
            
            # MACD
            ema_12 = df["close"].ewm(span=12, adjust=False).mean()
            ema_26 = df["close"].ewm(span=26, adjust=False).mean()
            df["macd"] = ema_12 - ema_26
            df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
            df["macd_histogram"] = df["macd"] - df["macd_signal"]
            
            # Stochastic
            df["stoch_k"] = self._calculate_stochastic_series(df, 14)
            df["stoch_d"] = df["stoch_k"].rolling(3).mean()
            
            # Williams %R
            df["williams_r"] = self._calculate_williams_r_series(df, 14)
            
            # Price position
            df["price_vs_sma20"] = (df["close"] / df["sma_20"] - 1) * 100
            
            # Volume ratio (if volume exists)
            if "volume" in df.columns:
                df["volume_sma_20"] = df["volume"].rolling(20).mean()
                df["volume_ratio"] = df["volume"] / df["volume_sma_20"]
            
            # Target variable: next day return (for supervised learning)
            df["target_return_1d"] = df["return_1d"].shift(-1)
            df["target_direction"] = np.where(df["target_return_1d"] > 0.5, 1, 
                                             np.where(df["target_return_1d"] < -0.5, -1, 0))
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
        
        return df
    
    def _calculate_rsi_series(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI series."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_atr_series(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR series."""
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def _calculate_stochastic_series(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Stochastic %K series."""
        lowest_low = df["low"].rolling(window=period).min()
        highest_high = df["high"].rolling(window=period).max()
        
        stoch_k = 100 * (df["close"] - lowest_low) / (highest_high - lowest_low)
        
        return stoch_k
    
    def _calculate_williams_r_series(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R series."""
        highest_high = df["high"].rolling(window=period).max()
        lowest_low = df["low"].rolling(window=period).min()
        
        williams_r = -100 * (highest_high - df["close"]) / (highest_high - lowest_low)
        
        return williams_r
    
    def get_data_stats(self, symbol: str = None) -> Dict:
        """
        Get statistics about cached data.
        
        Args:
            symbol: Optional symbol to filter by
            
        Returns:
            Dict with data statistics
        """
        symbols = [symbol] if symbol else list(UNDERLYING_ASSETS.keys())
        stats = {}
        
        for sym in symbols:
            cached = database.get_cached_market_data(sym, interval="day")
            
            if cached:
                dates = [r.get("date") for r in cached if r.get("date")]
                stats[sym] = {
                    "record_count": len(cached),
                    "earliest_date": min(dates) if dates else None,
                    "latest_date": max(dates) if dates else None,
                    "days_covered": len(set(dates)),
                }
            else:
                stats[sym] = {
                    "record_count": 0,
                    "earliest_date": None,
                    "latest_date": None,
                    "days_covered": 0,
                }
        
        return stats


# Singleton instance
_data_collector: Optional[DataCollector] = None


def get_data_collector() -> DataCollector:
    """Get or create the singleton data collector instance."""
    global _data_collector
    if _data_collector is None:
        _data_collector = DataCollector()
    return _data_collector
