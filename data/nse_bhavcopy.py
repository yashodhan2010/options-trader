"""
NSE Bhavcopy Historical Data Collector

Downloads and processes NSE bhavcopy files using jugaad-data library.
Handles holidays, 404 errors, and rate limiting gracefully.

Features:
- Downloads equity and F&O bhavcopy data
- Extracts options chain data (strike, IV, OI, LTP)
- Caches processed data to SQLite
- Supports incremental updates
"""

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from pathlib import Path
import time
import os
from typing import Dict, List, Optional, Tuple

from core.logger import logger
from core.database import database
from config.settings import DATA_DIR, UNDERLYING_ASSETS

# Try to import jugaad-data
try:
    from jugaad_data.nse import bhavcopy_save, bhavcopy_fo_save
    JUGAAD_AVAILABLE = True
except ImportError:
    JUGAAD_AVAILABLE = False
    logger.warning("jugaad-data not installed. Run: pip install jugaad-data")


class NSEBhavcopyCollector:
    """
    Collect historical NSE data from bhavcopy files.
    
    Downloads:
    - Equity bhavcopy: Daily OHLCV for all NSE stocks
    - F&O bhavcopy: Options chain data (strike, OI, volume, prices)
    
    Usage:
        collector = NSEBhavcopyCollector()
        collector.download_historical(start_date, end_date)
        df = collector.get_options_data("NIFTY", lookback_days=90)
    """
    
    def __init__(self, download_dir: str = None):
        """
        Initialize bhavcopy collector.
        
        Args:
            download_dir: Directory to store downloaded files (default: data/bhavcopies)
        """
        self.download_dir = Path(download_dir) if download_dir else Path(DATA_DIR) / "bhavcopies"
        self.equity_dir = self.download_dir / "equity"
        self.fo_dir = self.download_dir / "fo"
        
        # Create directories
        self.equity_dir.mkdir(parents=True, exist_ok=True)
        self.fo_dir.mkdir(parents=True, exist_ok=True)
        
        # Rate limiting
        self.request_delay = 1.0  # seconds between requests
        
        # Track downloaded dates
        self.downloaded_dates: set = set()
        self._load_downloaded_dates()
        
        logger.info(f"NSEBhavcopyCollector initialized. Download dir: {self.download_dir}")
    
    def _load_downloaded_dates(self):
        """Load list of already downloaded dates from disk."""
        # Check equity directory for existing files
        for f in self.equity_dir.glob("*.csv"):
            try:
                # Filename format: cm01JAN2025bhav.csv
                date_str = f.stem[2:11]  # Extract date portion
                dt = datetime.strptime(date_str, "%d%b%Y").date()
                self.downloaded_dates.add(dt)
            except:
                pass
        
        # Check F&O directory
        for f in self.fo_dir.glob("*.csv"):
            try:
                date_str = f.stem[2:11]
                dt = datetime.strptime(date_str, "%d%b%Y").date()
                self.downloaded_dates.add(dt)
            except:
                pass
        
        logger.info(f"Found {len(self.downloaded_dates)} previously downloaded dates")
    
    def download_historical(
        self,
        start_date: date,
        end_date: date,
        include_equity: bool = True,
        include_fo: bool = True,
        skip_existing: bool = True
    ) -> Dict[str, int]:
        """
        Download bhavcopy files for a date range.
        
        Args:
            start_date: Start date for download
            end_date: End date for download
            include_equity: Download equity bhavcopy
            include_fo: Download F&O bhavcopy
            skip_existing: Skip dates already downloaded
            
        Returns:
            Dict with counts: {'equity_success': n, 'fo_success': n, 'failed': n}
        """
        if not JUGAAD_AVAILABLE:
            logger.error("jugaad-data not installed. Cannot download bhavcopy.")
            return {'equity_success': 0, 'fo_success': 0, 'failed': 0}
        
        # Generate business days
        dates = pd.bdate_range(start=start_date, end=end_date)
        
        results = {'equity_success': 0, 'fo_success': 0, 'failed': 0, 'skipped': 0}
        
        logger.info(f"Downloading bhavcopies for {len(dates)} business days ({start_date} to {end_date})")
        
        for single_date in dates:
            date_obj = single_date.date()
            
            # Skip if already downloaded
            if skip_existing and date_obj in self.downloaded_dates:
                results['skipped'] += 1
                continue
            
            logger.info(f"Downloading for {date_obj.strftime('%Y-%m-%d')}...")
            
            # Download equity bhavcopy
            if include_equity:
                try:
                    bhavcopy_save(date_obj, str(self.equity_dir))
                    results['equity_success'] += 1
                    logger.debug(f"✅ Equity bhavcopy downloaded for {date_obj}")
                except Exception as e:
                    if "404" in str(e) or "holiday" in str(e).lower():
                        logger.debug(f"⏭️ {date_obj} is likely a holiday (equity)")
                    else:
                        logger.warning(f"❌ Equity download failed for {date_obj}: {e}")
                        results['failed'] += 1
            
            # Download F&O bhavcopy
            if include_fo:
                try:
                    bhavcopy_fo_save(date_obj, str(self.fo_dir))
                    results['fo_success'] += 1
                    logger.debug(f"✅ F&O bhavcopy downloaded for {date_obj}")
                    self.downloaded_dates.add(date_obj)
                except Exception as e:
                    if "404" in str(e) or "holiday" in str(e).lower():
                        logger.debug(f"⏭️ {date_obj} is likely a holiday (F&O)")
                    else:
                        logger.warning(f"❌ F&O download failed for {date_obj}: {e}")
                        results['failed'] += 1
            
            # Rate limiting - be respectful to NSE servers
            time.sleep(self.request_delay)
        
        logger.info(f"Download complete: {results}")
        return results
    
    def download_last_n_days(self, days: int = 180, include_fo: bool = True) -> Dict[str, int]:
        """
        Download bhavcopy for the last N days.
        
        Args:
            days: Number of days to download
            include_fo: Include F&O data
            
        Returns:
            Download results dict
        """
        end_date = date.today() - timedelta(days=1)  # Yesterday (today may not be available)
        start_date = end_date - timedelta(days=days)
        
        return self.download_historical(start_date, end_date, include_fo=include_fo)
    
    def parse_fo_bhavcopy(self, file_path: Path) -> pd.DataFrame:
        """
        Parse a single F&O bhavcopy CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame with parsed F&O data
        """
        try:
            df = pd.read_csv(file_path)
            
            # Standardize column names
            df.columns = df.columns.str.strip().str.upper()
            
            # Common column mappings
            column_map = {
                'SYMBOL': 'symbol',
                'EXPIRY_DT': 'expiry',
                'STRIKE_PR': 'strike',
                'OPTION_TYP': 'option_type',
                'OPEN': 'open',
                'HIGH': 'high',
                'LOW': 'low',
                'CLOSE': 'close',
                'SETTLE_PR': 'settle_price',
                'CONTRACTS': 'contracts',
                'VAL_INLAKH': 'value_lakh',
                'OPEN_INT': 'oi',
                'CHG_IN_OI': 'oi_change',
                'TIMESTAMP': 'date'
            }
            
            df = df.rename(columns=column_map)
            
            # Filter for options only (CE/PE)
            if 'option_type' in df.columns:
                df = df[df['option_type'].isin(['CE', 'PE'])]
            
            # Parse dates
            if 'expiry' in df.columns:
                df['expiry'] = pd.to_datetime(df['expiry'], format='%d-%b-%Y', errors='coerce')
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], format='%d-%b-%Y', errors='coerce')
            
            return df
            
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return pd.DataFrame()
    
    def parse_equity_bhavcopy(self, file_path: Path) -> pd.DataFrame:
        """
        Parse a single equity bhavcopy CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame with parsed equity data
        """
        try:
            df = pd.read_csv(file_path)
            
            # Standardize column names
            df.columns = df.columns.str.strip().str.upper()
            
            column_map = {
                'SYMBOL': 'symbol',
                'SERIES': 'series',
                'OPEN': 'open',
                'HIGH': 'high',
                'LOW': 'low',
                'CLOSE': 'close',
                'LAST': 'last',
                'PREVCLOSE': 'prev_close',
                'TOTTRDQTY': 'volume',
                'TOTTRDVAL': 'turnover',
                'TIMESTAMP': 'date',
                'TOTALTRADES': 'trades'
            }
            
            df = df.rename(columns=column_map)
            
            # Filter for EQ series only (ignore BE, BL, etc.)
            if 'series' in df.columns:
                df = df[df['series'] == 'EQ']
            
            # Parse dates
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], format='%d-%b-%Y', errors='coerce')
            
            return df
            
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return pd.DataFrame()
    
    def get_options_data(
        self,
        symbol: str,
        start_date: date = None,
        end_date: date = None,
        lookback_days: int = 90
    ) -> pd.DataFrame:
        """
        Get processed options data for a symbol.
        
        Args:
            symbol: Underlying symbol (e.g., "NIFTY", "BANKNIFTY", "RELIANCE")
            start_date: Start date (default: lookback_days ago)
            end_date: End date (default: today)
            lookback_days: Days to look back if start_date not specified
            
        Returns:
            DataFrame with options data (date, strike, option_type, close, oi, oi_change)
        """
        end_date = end_date or date.today()
        start_date = start_date or (end_date - timedelta(days=lookback_days))
        
        all_data = []
        
        # Process all F&O bhavcopy files in date range
        for file_path in sorted(self.fo_dir.glob("*.csv")):
            try:
                # Extract date from filename
                date_str = file_path.stem[2:11]
                file_date = datetime.strptime(date_str, "%d%b%Y").date()
                
                if start_date <= file_date <= end_date:
                    df = self.parse_fo_bhavcopy(file_path)
                    
                    if len(df) > 0 and 'symbol' in df.columns:
                        # Filter for requested symbol
                        symbol_data = df[df['symbol'] == symbol]
                        if len(symbol_data) > 0:
                            all_data.append(symbol_data)
                            
            except Exception as e:
                logger.debug(f"Error processing {file_path}: {e}")
        
        if not all_data:
            logger.warning(f"No options data found for {symbol}")
            return pd.DataFrame()
        
        result = pd.concat(all_data, ignore_index=True)
        result = result.sort_values(['date', 'expiry', 'strike'])
        
        logger.info(f"Retrieved {len(result)} options records for {symbol}")
        return result
    
    def get_equity_data(
        self,
        symbol: str,
        start_date: date = None,
        end_date: date = None,
        lookback_days: int = 180
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data for an equity symbol.
        
        Args:
            symbol: Stock symbol (e.g., "RELIANCE", "TCS")
            start_date: Start date
            end_date: End date
            lookback_days: Days to look back if start_date not specified
            
        Returns:
            DataFrame with OHLCV data
        """
        end_date = end_date or date.today()
        start_date = start_date or (end_date - timedelta(days=lookback_days))
        
        all_data = []
        
        for file_path in sorted(self.equity_dir.glob("*.csv")):
            try:
                date_str = file_path.stem[2:11]
                file_date = datetime.strptime(date_str, "%d%b%Y").date()
                
                if start_date <= file_date <= end_date:
                    df = self.parse_equity_bhavcopy(file_path)
                    
                    if len(df) > 0 and 'symbol' in df.columns:
                        symbol_data = df[df['symbol'] == symbol]
                        if len(symbol_data) > 0:
                            all_data.append(symbol_data)
                            
            except Exception as e:
                logger.debug(f"Error processing {file_path}: {e}")
        
        if not all_data:
            logger.warning(f"No equity data found for {symbol}")
            return pd.DataFrame()
        
        result = pd.concat(all_data, ignore_index=True)
        result = result.sort_values('date')
        
        logger.info(f"Retrieved {len(result)} equity records for {symbol}")
        return result
    
    def calculate_daily_oi_metrics(
        self,
        symbol: str,
        trade_date: date,
        spot_price: float = None
    ) -> Dict:
        """
        Calculate OI-based metrics for a given date.
        
        Args:
            symbol: Underlying symbol
            trade_date: Date to calculate metrics for
            spot_price: Current spot price (for max pain calculation)
            
        Returns:
            Dict with PCR, max_pain, call_oi, put_oi, etc.
        """
        options_data = self.get_options_data(
            symbol, 
            start_date=trade_date, 
            end_date=trade_date
        )
        
        if options_data.empty:
            return {}
        
        # Get nearest expiry
        if 'expiry' in options_data.columns:
            nearest_expiry = options_data['expiry'].min()
            options_data = options_data[options_data['expiry'] == nearest_expiry]
        
        calls = options_data[options_data['option_type'] == 'CE']
        puts = options_data[options_data['option_type'] == 'PE']
        
        total_call_oi = calls['oi'].sum() if 'oi' in calls.columns else 0
        total_put_oi = puts['oi'].sum() if 'oi' in puts.columns else 0
        
        # Put-Call Ratio
        pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 1.0
        
        # Max Pain calculation
        max_pain = self._calculate_max_pain(options_data, spot_price)
        
        # OI changes
        call_oi_change = calls['oi_change'].sum() if 'oi_change' in calls.columns else 0
        put_oi_change = puts['oi_change'].sum() if 'oi_change' in puts.columns else 0
        
        return {
            'date': trade_date,
            'symbol': symbol,
            'pcr': round(pcr, 3),
            'max_pain': max_pain,
            'total_call_oi': int(total_call_oi),
            'total_put_oi': int(total_put_oi),
            'call_oi_change': int(call_oi_change),
            'put_oi_change': int(put_oi_change),
            'call_oi_change_percent': round(call_oi_change / total_call_oi * 100, 2) if total_call_oi > 0 else 0,
            'put_oi_change_percent': round(put_oi_change / total_put_oi * 100, 2) if total_put_oi > 0 else 0
        }
    
    def _calculate_max_pain(self, options_data: pd.DataFrame, spot_price: float = None) -> float:
        """Calculate max pain strike from options data."""
        if options_data.empty or 'strike' not in options_data.columns:
            return spot_price or 0
        
        strikes = options_data['strike'].unique()
        min_pain = float('inf')
        max_pain_strike = spot_price or strikes[len(strikes)//2]
        
        for strike in strikes:
            total_pain = 0
            
            # Pain for call writers
            calls = options_data[(options_data['option_type'] == 'CE') & (options_data['strike'] < strike)]
            if len(calls) > 0 and 'oi' in calls.columns:
                total_pain += (calls['oi'] * (strike - calls['strike'])).sum()
            
            # Pain for put writers
            puts = options_data[(options_data['option_type'] == 'PE') & (options_data['strike'] > strike)]
            if len(puts) > 0 and 'oi' in puts.columns:
                total_pain += (puts['oi'] * (puts['strike'] - strike)).sum()
            
            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = strike
        
        return max_pain_strike
    
    def build_training_dataset(
        self,
        symbols: List[str] = None,
        start_date: date = None,
        end_date: date = None,
        lookback_days: int = 180
    ) -> pd.DataFrame:
        """
        Build a comprehensive training dataset with all available features.
        
        Combines:
        - Equity OHLCV data
        - Options chain data (IV, OI)
        - OI metrics (PCR, max pain)
        
        Args:
            symbols: List of symbols to include
            start_date: Start date
            end_date: End date
            lookback_days: Lookback period
            
        Returns:
            DataFrame ready for feature engineering
        """
        symbols = symbols or list(UNDERLYING_ASSETS.keys())
        end_date = end_date or date.today()
        start_date = start_date or (end_date - timedelta(days=lookback_days))
        
        all_records = []
        
        for symbol in symbols:
            logger.info(f"Building dataset for {symbol}...")
            
            # Get equity data
            equity_df = self.get_equity_data(symbol, start_date, end_date)
            
            if equity_df.empty:
                logger.warning(f"No equity data for {symbol}, skipping")
                continue
            
            # Get options data
            options_df = self.get_options_data(symbol, start_date, end_date)
            
            # Process each trading day
            for _, row in equity_df.iterrows():
                trade_date = row['date'].date() if isinstance(row['date'], pd.Timestamp) else row['date']
                
                record = {
                    'symbol': symbol,
                    'date': trade_date,
                    'open': row.get('open'),
                    'high': row.get('high'),
                    'low': row.get('low'),
                    'close': row.get('close'),
                    'volume': row.get('volume'),
                }
                
                # Add OI metrics if options data available
                if not options_df.empty:
                    oi_metrics = self.calculate_daily_oi_metrics(
                        symbol, trade_date, row.get('close')
                    )
                    record.update(oi_metrics)
                
                all_records.append(record)
        
        if not all_records:
            return pd.DataFrame()
        
        result = pd.DataFrame(all_records)
        result = result.sort_values(['symbol', 'date'])
        
        logger.info(f"Built training dataset with {len(result)} records for {len(symbols)} symbols")
        return result
    
    def cache_to_database(self, df: pd.DataFrame, table_name: str = "bhavcopy_data"):
        """
        Cache processed bhavcopy data to SQLite database.
        
        Args:
            df: DataFrame to cache
            table_name: Table name in database
        """
        try:
            conn = database.get_connection()
            df.to_sql(table_name, conn, if_exists='append', index=False)
            logger.info(f"Cached {len(df)} records to {table_name}")
        except Exception as e:
            logger.error(f"Error caching to database: {e}")


# Singleton instance
_bhavcopy_collector = None


def get_bhavcopy_collector() -> NSEBhavcopyCollector:
    """Get singleton bhavcopy collector instance."""
    global _bhavcopy_collector
    if _bhavcopy_collector is None:
        _bhavcopy_collector = NSEBhavcopyCollector()
    return _bhavcopy_collector


# CLI helper for manual download
def download_historical_bhavcopy(
    start_date: date,
    end_date: date,
    download_dir: str = None
) -> Dict[str, int]:
    """
    Convenience function to download historical bhavcopy data.
    
    Args:
        start_date: Start date
        end_date: End date
        download_dir: Optional custom download directory
        
    Returns:
        Download results dict
    """
    collector = NSEBhavcopyCollector(download_dir)
    return collector.download_historical(start_date, end_date)


if __name__ == "__main__":
    # Example usage
    from datetime import date
    
    # Download last 30 days
    collector = NSEBhavcopyCollector()
    
    # Download data
    start = date(2025, 12, 1)
    end = date(2025, 12, 31)
    
    print(f"Downloading bhavcopy data from {start} to {end}...")
    results = collector.download_historical(start, end)
    print(f"Results: {results}")
    
    # Get options data for NIFTY
    nifty_options = collector.get_options_data("NIFTY", lookback_days=30)
    print(f"\nNIFTY options data: {len(nifty_options)} records")
    
    # Calculate OI metrics
    if not nifty_options.empty:
        metrics = collector.calculate_daily_oi_metrics("NIFTY", end, 24000)
        print(f"OI Metrics: {metrics}")
