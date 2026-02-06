"""
NSE Data Downloader

Downloads F&O bhavcopy from NSE archives using direct requests.
No session/cookie management needed for archives.nseindia.com.

Archive Availability (as of Jan 2026):
- F&O Bhavcopy: Available from ~2018 to mid-2024
- For recent data, use live option chain API during market hours
"""

import requests
import zipfile
import io
import time
import pandas as pd
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, Dict, List

# Headers for direct archive requests
ARCHIVE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/zip,application/octet-stream,*/*",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive"
}

# Headers for NSE API (requires session)
API_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/"
}


class NSEDownloader:
    """
    Download data from NSE archives and APIs.
    
    F&O bhavcopy downloads use direct requests (no session needed).
    Option chain API requires session with cookies.
    """
    
    def __init__(self, cache_dir: str = "data/nse_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._api_session = None
        
    def download_fo_bhavcopy(self, target_date: date, verbose: bool = True) -> Optional[pd.DataFrame]:
        """
        Download F&O bhavcopy for a specific date.
        
        Uses direct request to archives.nseindia.com (no session needed).
        
        Args:
            target_date: Date to download
            verbose: Print progress messages
            
        Returns:
            DataFrame with F&O bhavcopy data, or None if not available
        """
        cache_file = self.cache_dir / f"fo_bhavcopy_{target_date.isoformat()}.csv"
        
        # Check cache
        if cache_file.exists():
            if verbose:
                print(f"{target_date}: Loading from cache")
            return pd.read_csv(cache_file)
        
        # Format date - CRITICAL: use MON (e.g., JAN) not mm in URL path
        dd = f"{target_date.day:02d}"
        mon = target_date.strftime("%b").upper()
        yyyy = str(target_date.year)
        
        file_name = f"fo{dd}{mon}{yyyy}bhav.csv.zip"
        url = f"https://archives.nseindia.com/content/historical/DERIVATIVES/{yyyy}/{mon}/{file_name}"
        
        try:
            r = requests.get(url, headers=ARCHIVE_HEADERS, timeout=20)
            
            if r.status_code != 200:
                if verbose:
                    print(f"{target_date}: HTTP {r.status_code}")
                return None
            
            # Validate ZIP magic bytes
            if not r.content.startswith(b"PK"):
                if verbose:
                    print(f"{target_date}: Invalid ZIP (holiday or unavailable)")
                return None
            
            # Extract CSV from ZIP
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                csv_file = z.namelist()[0]
                df = pd.read_csv(z.open(csv_file))
            
            # Cache it
            df.to_csv(cache_file, index=False)
            
            if verbose:
                print(f"{target_date}: Downloaded {len(df)} records")
            return df
            
        except Exception as e:
            if verbose:
                print(f"{target_date}: Error - {e}")
            return None
    
    def download_equity_bhavcopy(self, target_date: date, verbose: bool = True) -> Optional[pd.DataFrame]:
        """
        Download equity bhavcopy (cash market).
        
        Uses direct request - no session needed.
        """
        cache_file = self.cache_dir / f"eq_bhavcopy_{target_date.isoformat()}.csv"
        
        if cache_file.exists():
            if verbose:
                print(f"{target_date}: Loading equity from cache")
            return pd.read_csv(cache_file)
        
        date_str = target_date.strftime('%d%m%Y')
        url = f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{date_str}.csv"
        
        try:
            r = requests.get(url, headers=ARCHIVE_HEADERS, timeout=20)
            
            if r.status_code == 200 and len(r.content) > 1000:
                df = pd.read_csv(io.BytesIO(r.content))
                df.to_csv(cache_file, index=False)
                if verbose:
                    print(f"{target_date}: Equity {len(df)} records")
                return df
                
        except Exception as e:
            if verbose:
                print(f"{target_date}: Equity error - {e}")
        
        return None
    
    def _get_api_session(self):
        """Get or create session for NSE API (requires cookies)."""
        if self._api_session is None:
            self._api_session = requests.Session()
            self._api_session.headers.update(API_HEADERS)
            # Must visit homepage first for API access
            self._api_session.get("https://www.nseindia.com", timeout=10)
        return self._api_session
    
    def download_option_chain(self, symbol: str = "NIFTY") -> Optional[pd.DataFrame]:
        """
        Download live option chain from NSE API.
        
        NOTE: Only works during market hours (9:15-15:30 IST).
        Requires session with cookies.
        """
        session = self._get_api_session()
        
        # Index vs Stock
        if symbol in ["NIFTY", "BANKNIFTY", "FINNIFTY", "NIFTYIT"]:
            url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
        else:
            url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"
        
        try:
            r = session.get(url, timeout=20)
            
            if r.status_code == 200:
                data = r.json()
                records = data.get("records", {})
                chain_data = records.get("data", [])
                
                if chain_data:
                    rows = []
                    for item in chain_data:
                        strike = item.get("strikePrice")
                        expiry = item.get("expiryDate")
                        
                        ce = item.get("CE", {})
                        pe = item.get("PE", {})
                        
                        if ce:
                            rows.append({
                                "symbol": symbol,
                                "expiry": expiry,
                                "strike": strike,
                                "option_type": "CE",
                                "oi": ce.get("openInterest", 0),
                                "oi_change": ce.get("changeinOpenInterest", 0),
                                "volume": ce.get("totalTradedVolume", 0),
                                "iv": ce.get("impliedVolatility", 0),
                                "ltp": ce.get("lastPrice", 0),
                                "bid": ce.get("bidprice", 0),
                                "ask": ce.get("askPrice", 0),
                            })
                        
                        if pe:
                            rows.append({
                                "symbol": symbol,
                                "expiry": expiry,
                                "strike": strike,
                                "option_type": "PE",
                                "oi": pe.get("openInterest", 0),
                                "oi_change": pe.get("changeinOpenInterest", 0),
                                "volume": pe.get("totalTradedVolume", 0),
                                "iv": pe.get("impliedVolatility", 0),
                                "ltp": pe.get("lastPrice", 0),
                                "bid": pe.get("bidprice", 0),
                                "ask": pe.get("askPrice", 0),
                            })
                    
                    df = pd.DataFrame(rows)
                    print(f"Option chain: {len(df)} records for {symbol}")
                    return df
                    
        except Exception as e:
            print(f"Error: {e}")
        
        return None
    
    def download_date_range(
        self, 
        start_date: date, 
        end_date: date,
        data_type: str = "fo",
        delay: float = 1.0
    ) -> Dict[date, pd.DataFrame]:
        """
        Download data for a date range.
        
        Args:
            start_date: Start date
            end_date: End date
            data_type: "fo" for F&O bhavcopy, "eq" for equity
            delay: Seconds between requests (be nice to NSE)
            
        Returns:
            Dict mapping dates to DataFrames
        """
        results = {}
        
        # Generate business days (NSE holidays will return None)
        dates = pd.bdate_range(start=start_date, end=end_date)
        
        print(f"Downloading {data_type.upper()} data: {start_date} to {end_date}")
        print(f"Business days to process: {len(dates)}")
        
        for i, dt in enumerate(dates):
            d = dt.date()
            
            if data_type == "fo":
                df = self.download_fo_bhavcopy(d, verbose=False)
            else:
                df = self.download_equity_bhavcopy(d, verbose=False)
            
            if df is not None:
                results[d] = df
                print(f"  [{i+1}/{len(dates)}] {d}: ✓ {len(df)} records")
            else:
                print(f"  [{i+1}/{len(dates)}] {d}: - (holiday or unavailable)")
            
            time.sleep(delay)
        
        print(f"\nCompleted: {len(results)}/{len(dates)} days downloaded")
        return results
    
    def bulk_download_fo_historical(
        self,
        symbols: Optional[List[str]] = None,
        start_date: date = date(2024, 1, 1),
        end_date: date = date(2024, 5, 31)
    ) -> pd.DataFrame:
        """
        Bulk download F&O historical data and filter by symbols.
        
        Args:
            symbols: List of symbols to filter (e.g., ["NIFTY", "BANKNIFTY"])
                    If None, returns all instruments
            start_date: Start date (default Jan 2024)
            end_date: End date (default May 2024 - archive limit)
            
        Returns:
            Combined DataFrame with all data
        """
        all_data = []
        
        # Download all available dates
        date_data = self.download_date_range(start_date, end_date, "fo")
        
        for d, df in date_data.items():
            df = df.copy()
            df["DATE"] = d
            
            # Filter by symbols if specified
            if symbols and "SYMBOL" in df.columns:
                df = df[df["SYMBOL"].isin(symbols)]
            
            all_data.append(df)
        
        if not all_data:
            print("No data downloaded!")
            return pd.DataFrame()
        
        combined = pd.concat(all_data, ignore_index=True)
        print(f"\nTotal records: {len(combined)}")
        
        # Save combined file
        output_file = self.cache_dir / f"fo_combined_{start_date}_{end_date}.csv"
        combined.to_csv(output_file, index=False)
        print(f"Saved to: {output_file}")
        
        return combined


def test_downloader():
    """Test the NSE downloader."""
    dl = NSEDownloader()
    
    print("=" * 60)
    print("NSE Downloader Test")
    print("=" * 60)
    
    # Test F&O bhavcopy (historical - should work)
    print("\n1. F&O Bhavcopy (Jan 15, 2024) - Historical archives...")
    fo_df = dl.download_fo_bhavcopy(date(2024, 1, 15))
    if fo_df is not None:
        print(f"   ✓ {len(fo_df)} records")
        print(f"   Columns: {list(fo_df.columns)}")
    else:
        print("   ✗ Failed")
    
    # Test equity bhavcopy (recent)
    print("\n2. Equity Bhavcopy (Dec 30, 2025) - Recent data...")
    eq_df = dl.download_equity_bhavcopy(date(2025, 12, 30))
    if eq_df is not None:
        print(f"   ✓ {len(eq_df)} records")
    else:
        print("   ✗ Failed")
    
    # Test option chain (only works during market hours)
    print("\n3. Option Chain API (NIFTY) - Live data...")
    oc_df = dl.download_option_chain("NIFTY")
    if oc_df is not None and len(oc_df) > 0:
        print(f"   ✓ {len(oc_df)} records")
    else:
        print("   ✗ (Only works during market hours 9:15-15:30 IST)")
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("  - F&O Archives: Available Jan 2018 - May 2024")
    print("  - Equity Bhavcopy: Available for recent dates")
    print("  - Option Chain: Live data during market hours only")
    print("=" * 60)


if __name__ == "__main__":
    test_downloader()
