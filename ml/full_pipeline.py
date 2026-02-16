"""
Full Historical Training Pipeline

Downloads NSE F&O bhavcopy for all symbols in watchlist,
calculates features including Greeks proxies, and trains per-symbol models.

For symbols not available in NSE bhavcopy (e.g. SENSEX on BSE),
automatically falls back to Kite Historical API for OHLCV data.

Features:
1. Downloads historical data for all watchlist symbols + indices
2. Falls back to Kite API for symbols without bhavcopy data
3. Calculates UNIFIED features (compatible with live prediction)
4. Trains individual models per symbol
5. Monthly update pipeline for continuous learning

IMPORTANT: Uses UnifiedFeatureDefinition to ensure historical models
work with live data from FeatureEngineer.
"""

import pandas as pd
import numpy as np
import json
import joblib
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from data.nse_downloader import NSEDownloader
from core.logger import logger
from ml.unified_features import (
    UnifiedFeatureDefinition,
    HistoricalFeatureAdapter,
    get_unified_feature_names
)


class GreeksCalculator:
    """
    Calculate Greek-like features from historical data.
    
    Since we don't have real-time IV/Greeks in bhavcopy,
    we derive proxy features from available data.
    """
    
    @staticmethod
    def estimate_iv_proxy(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Estimate IV proxy from historical volatility.
        
        Uses Parkinson's range-based volatility estimator which is
        more efficient than close-to-close volatility.
        """
        if 'high' not in df.columns or 'low' not in df.columns:
            return pd.Series(0, index=df.index)
        
        # Parkinson volatility (range-based)
        log_hl = np.log(df['high'] / df['low'])
        parkinson = np.sqrt((1 / (4 * np.log(2))) * (log_hl ** 2).rolling(window).mean())
        
        # Annualize (252 trading days)
        iv_proxy = parkinson * np.sqrt(252)
        
        return iv_proxy
    
    @staticmethod
    def estimate_delta_proxy(df: pd.DataFrame) -> pd.Series:
        """
        Estimate delta proxy based on momentum and OI patterns.
        
        Positive delta proxy = bullish momentum
        Negative delta proxy = bearish momentum
        """
        if 'close' not in df.columns:
            return pd.Series(0.5, index=df.index)
        
        # Price momentum (normalized)
        returns = df['close'].pct_change(5)
        momentum = returns.rolling(10).mean()
        
        # Normalize to 0-1 range (like delta)
        delta_proxy = (momentum - momentum.min()) / (momentum.max() - momentum.min() + 1e-10)
        delta_proxy = delta_proxy.fillna(0.5)
        
        return delta_proxy
    
    @staticmethod
    def estimate_gamma_proxy(df: pd.DataFrame) -> pd.Series:
        """
        Estimate gamma proxy from volatility of returns.
        
        High gamma = high sensitivity to price changes
        """
        if 'close' not in df.columns:
            return pd.Series(0, index=df.index)
        
        returns = df['close'].pct_change()
        
        # Gamma proxy = acceleration of returns
        gamma_proxy = returns.diff().abs().rolling(5).mean()
        
        return gamma_proxy.fillna(0)
    
    @staticmethod
    def estimate_theta_proxy(df: pd.DataFrame) -> pd.Series:
        """
        Estimate theta proxy from time decay patterns.
        
        Uses options OI decay patterns when available.
        """
        if 'total_oi' not in df.columns:
            return pd.Series(0, index=df.index)
        
        # OI decay rate (proxy for time decay)
        oi_change = df['total_oi'].pct_change()
        theta_proxy = -oi_change.rolling(5).mean()  # Negative because theta decays
        
        return theta_proxy.fillna(0)
    
    @staticmethod
    def estimate_vega_proxy(df: pd.DataFrame) -> pd.Series:
        """
        Estimate vega proxy from volatility sensitivity.
        """
        if 'close' not in df.columns:
            return pd.Series(0, index=df.index)
        
        returns = df['close'].pct_change()
        
        # Vega proxy = sensitivity to volatility changes
        vol_short = returns.rolling(5).std()
        vol_long = returns.rolling(20).std()
        
        vega_proxy = (vol_short - vol_long).abs()
        
        return vega_proxy.fillna(0)
    
    @staticmethod
    def calculate_all_greeks(df: pd.DataFrame) -> pd.DataFrame:
        """Add all Greek proxies to dataframe."""
        df = df.copy()
        
        df['iv_proxy'] = GreeksCalculator.estimate_iv_proxy(df)
        df['delta_proxy'] = GreeksCalculator.estimate_delta_proxy(df)
        df['gamma_proxy'] = GreeksCalculator.estimate_gamma_proxy(df)
        df['theta_proxy'] = GreeksCalculator.estimate_theta_proxy(df)
        df['vega_proxy'] = GreeksCalculator.estimate_vega_proxy(df)
        
        # IV percentile (where current IV sits in historical range)
        df['iv_percentile'] = df['iv_proxy'].rolling(60).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) if len(x) > 0 else 50
        ).fillna(50)
        
        # IV rank (current IV vs min/max)
        iv_min = df['iv_proxy'].rolling(60).min()
        iv_max = df['iv_proxy'].rolling(60).max()
        df['iv_rank'] = (df['iv_proxy'] - iv_min) / (iv_max - iv_min + 1e-10)
        df['iv_rank'] = df['iv_rank'].fillna(0.5)
        
        return df


class FullPipelineTrainer:
    """
    Full training pipeline for all symbols.
    """
    
    # Default symbols (indices + stocks)
    INDEX_SYMBOLS = ["NIFTY", "BANKNIFTY", "FINNIFTY", "SENSEX"]
    
    def __init__(
        self,
        watchlist_path: str = "config/watchlist.json",
        cache_dir: str = "data/nse_cache",
        model_dir: str = "data/ml_models"
    ):
        self.watchlist_path = Path(watchlist_path)
        self.cache_dir = Path(cache_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.downloader = NSEDownloader(cache_dir)
        self.greeks_calc = GreeksCalculator()
        self.feature_adapter = HistoricalFeatureAdapter()
        
        # Use unified feature names
        self.feature_names = get_unified_feature_names()
        
        # Load symbols from watchlist
        self.symbols = self._load_symbols()
        logger.info(f"Loaded {len(self.symbols)} symbols: {self.symbols}")
        logger.info(f"Using {len(self.feature_names)} unified features")
    
    def _load_symbols(self) -> List[str]:
        """Load symbols from watchlist + indices."""
        symbols = list(self.INDEX_SYMBOLS)
        
        try:
            with open(self.watchlist_path) as f:
                watchlist = json.load(f)
            
            for asset in watchlist.get("assets", []):
                if asset.get("enabled", True):
                    name = asset.get("name")
                    if name and name not in symbols:
                        symbols.append(name)
        except Exception as e:
            logger.warning(f"Could not load watchlist: {e}")
        
        return symbols
    
    def _fetch_kite_ohlcv(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """
        Fetch historical OHLCV from Kite API for symbols without bhavcopy data.
        
        Used as a fallback for BSE symbols (e.g. SENSEX) that don't appear
        in NSE F&O bhavcopy archives.
        """
        from data.data_fetcher import data_fetcher
        from config.settings import UNDERLYING_ASSETS
        
        asset_cfg = UNDERLYING_ASSETS.get(symbol, {})
        exchange = asset_cfg.get("exchange", "NSE")
        
        logger.info(f"Fetching {days} days of Kite data for {symbol} (exchange={exchange})")
        
        df = data_fetcher.get_historical_data(
            symbol=symbol, interval="day", days=days, exchange=exchange
        )
        
        if df.empty:
            return pd.DataFrame()
        
        # Ensure required columns
        for col in ("open", "high", "low", "close", "volume"):
            if col not in df.columns:
                df[col] = 0.0
        
        # Reset index (Kite returns date as index)
        if df.index.name == "date" or isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        
        if "date" in df.columns:
            df = df.sort_values("date").reset_index(drop=True)
        
        df["symbol"] = symbol
        logger.info(f"Kite fallback: {len(df)} candles for {symbol}")
        return df

    def download_all_historical(
        self,
        start_date: date = date(2024, 1, 1),
        end_date: date = date(2024, 5, 31)
    ) -> pd.DataFrame:
        """
        Download F&O bhavcopy for all dates and filter to our symbols.
        """
        logger.info(f"Downloading historical data: {start_date} to {end_date}")
        logger.info(f"Symbols: {self.symbols}")
        
        all_data = []
        dates = pd.bdate_range(start=start_date, end=end_date)
        
        for i, dt in enumerate(dates):
            d = dt.date()
            df = self.downloader.download_fo_bhavcopy(d, verbose=False)
            
            if df is not None:
                df["DATE"] = d
                # Filter to our symbols
                df = df[df["SYMBOL"].isin(self.symbols)]
                all_data.append(df)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Progress: {i+1}/{len(dates)} days downloaded")
            
            import time
            time.sleep(0.3)
        
        if not all_data:
            logger.error("No data downloaded!")
            return pd.DataFrame()
        
        combined = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total raw records: {len(combined)}")
        
        # Save raw data
        raw_file = self.cache_dir / f"raw_fo_data_{start_date}_{end_date}.csv"
        combined.to_csv(raw_file, index=False)
        logger.info(f"Raw data saved to: {raw_file}")
        
        return combined
    
    def process_symbol_data(self, fo_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Process F&O data for a single symbol.
        
        Aggregates futures and options data into daily features.
        """
        symbol_data = fo_df[fo_df["SYMBOL"] == symbol].copy()
        
        if symbol_data.empty:
            return pd.DataFrame()
        
        processed = []
        
        for dt in sorted(symbol_data["DATE"].unique()):
            day_data = symbol_data[symbol_data["DATE"] == dt]
            
            # Get futures (FUTIDX for indices, FUTSTK for stocks)
            futures = day_data[day_data["INSTRUMENT"].isin(["FUTIDX", "FUTSTK"])]
            if futures.empty:
                continue
            
            # Get nearest expiry future
            futures = futures.sort_values("EXPIRY_DT")
            fut = futures.iloc[0]
            
            # Get options
            options = day_data[day_data["INSTRUMENT"].isin(["OPTIDX", "OPTSTK"])]
            
            # Aggregate OI by option type
            calls = options[options["OPTION_TYP"] == "CE"]
            puts = options[options["OPTION_TYP"] == "PE"]
            
            call_oi = calls["OPEN_INT"].sum()
            put_oi = puts["OPEN_INT"].sum()
            call_oi_chg = calls["CHG_IN_OI"].sum()
            put_oi_chg = puts["CHG_IN_OI"].sum()
            
            # Strike-wise OI analysis (near ATM)
            spot_approx = fut["CLOSE"]
            
            # Find ATM options (within 2% of spot)
            atm_range = spot_approx * 0.02
            atm_calls = calls[abs(calls["STRIKE_PR"] - spot_approx) <= atm_range]
            atm_puts = puts[abs(puts["STRIKE_PR"] - spot_approx) <= atm_range]
            
            atm_call_oi = atm_calls["OPEN_INT"].sum()
            atm_put_oi = atm_puts["OPEN_INT"].sum()
            
            # OTM analysis
            otm_calls = calls[calls["STRIKE_PR"] > spot_approx * 1.02]
            otm_puts = puts[puts["STRIKE_PR"] < spot_approx * 0.98]
            
            otm_call_oi = otm_calls["OPEN_INT"].sum()
            otm_put_oi = otm_puts["OPEN_INT"].sum()
            
            # Calculate ratios
            pcr_oi = put_oi / (call_oi + 1e-10)
            pcr_volume = puts["CONTRACTS"].sum() / (calls["CONTRACTS"].sum() + 1e-10)
            
            # Max pain approximation (strike with max OI)
            if not options.empty:
                strike_oi = options.groupby("STRIKE_PR")["OPEN_INT"].sum()
                max_pain = strike_oi.idxmax() if len(strike_oi) > 0 else spot_approx
            else:
                max_pain = spot_approx
            
            row = {
                "symbol": symbol,
                "date": dt,
                
                # OHLCV
                "open": fut["OPEN"],
                "high": fut["HIGH"],
                "low": fut["LOW"],
                "close": fut["CLOSE"],
                "volume": fut.get("CONTRACTS", 0) * 100,  # Approximate
                
                # Futures OI
                "fut_oi": fut.get("OPEN_INT", 0),
                "fut_oi_change": fut.get("CHG_IN_OI", 0),
                
                # Options OI aggregates
                "call_oi": call_oi,
                "put_oi": put_oi,
                "call_oi_change": call_oi_chg,
                "put_oi_change": put_oi_chg,
                "total_oi": call_oi + put_oi + fut.get("OPEN_INT", 0),
                
                # OI ratios
                "pcr_oi": pcr_oi,
                "pcr_volume": pcr_volume,
                
                # Strike analysis
                "atm_call_oi": atm_call_oi,
                "atm_put_oi": atm_put_oi,
                "atm_pcr": atm_put_oi / (atm_call_oi + 1e-10),
                "otm_call_oi": otm_call_oi,
                "otm_put_oi": otm_put_oi,
                
                # Max pain
                "max_pain": max_pain,
                "max_pain_distance": (max_pain - spot_approx) / spot_approx,
            }
            
            processed.append(row)
        
        df = pd.DataFrame(processed)
        if not df.empty:
            df = df.sort_values("date").reset_index(drop=True)
        
        return df
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add UNIFIED features using HistoricalFeatureAdapter.
        
        This ensures the features match what FeatureEngineer produces
        for live prediction, enabling seamless model usage.
        """
        if df.empty:
            return df
        
        # Use the unified feature adapter
        return self.feature_adapter.extract_features(df)
    
    def create_labels(self, df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
        """Create prediction labels."""
        if df.empty:
            return df
        
        df = df.copy()
        
        # Future return
        df["future_return"] = df["close"].shift(-horizon) / df["close"] - 1
        
        # Binary label
        df["label"] = (df["future_return"] > 0).astype(float)
        
        # Drop rows without labels
        df = df.dropna(subset=["future_return"])
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare X, y arrays for training using UNIFIED feature set."""
        if df.empty:
            return np.array([]), np.array([]), []
        
        # Use unified feature names only
        feature_cols = [c for c in self.feature_names if c in df.columns]
        
        # Check for missing features
        missing = [c for c in self.feature_names if c not in df.columns]
        if missing:
            logger.debug(f"Missing unified features (will use 0): {missing[:5]}...")
        
        # Build feature matrix with unified features in order
        X_list = []
        for _, row in df.iterrows():
            features = [row.get(name, 0.0) for name in self.feature_names]
            # Replace NaN and inf
            features = [0.0 if (pd.isna(f) or np.isinf(f)) else f for f in features]
            X_list.append(features)
        
        X = np.array(X_list)
        y = df["label"].values
        
        return X, y, self.feature_names
    
    def train_symbol_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        symbol: str,
        feature_names: List[str]
    ) -> Dict:
        """Train model for a single symbol."""
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        if len(X) < 30:
            logger.warning(f"{symbol}: Insufficient data ({len(X)} samples)")
            return None
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Train on last fold
        train_idx, test_idx = list(tscv.split(X))[-1]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train Random Forest
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        }
        
        # Feature importance
        importance = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        
        logger.info(f"{symbol}: Acc={metrics['accuracy']:.2%}, F1={metrics['f1']:.2%}")
        
        return {
            "model": model,
            "metrics": metrics,
            "feature_importance": importance,
            "feature_names": feature_names,
            "n_samples": len(X),
            "symbol": symbol
        }
    
    def run_full_pipeline(
        self,
        start_date: date = date(2024, 1, 1),
        end_date: date = date(2024, 5, 31),
        force_download: bool = False,
        kite_fallback_days: int = 365
    ) -> Dict:
        """
        Run complete training pipeline for all symbols.
        
        For symbols not found in NSE bhavcopy (e.g. BSE symbols like SENSEX),
        automatically falls back to Kite Historical API.
        
        Args:
            start_date: Bhavcopy start date
            end_date: Bhavcopy end date
            force_download: Force re-download of bhavcopy
            kite_fallback_days: Days of Kite history for fallback symbols
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info("=" * 70)
        logger.info("FULL HISTORICAL TRAINING PIPELINE")
        logger.info(f"Date Range: {start_date} to {end_date}")
        logger.info(f"Symbols: {self.symbols}")
        logger.info("=" * 70)
        
        # Step 1: Download/load raw data
        raw_file = self.cache_dir / f"raw_fo_data_{start_date}_{end_date}.csv"
        
        if raw_file.exists() and not force_download:
            logger.info(f"Loading cached raw data: {raw_file}")
            fo_df = pd.read_csv(raw_file)
            fo_df["DATE"] = pd.to_datetime(fo_df["DATE"]).dt.date
        else:
            fo_df = self.download_all_historical(start_date, end_date)
        
        # Note: fo_df can be empty if bhavcopy download fails,
        # but Kite fallback may still work for some symbols
        if fo_df.empty:
            logger.warning("No bhavcopy data available, will try Kite fallback for all symbols")
            fo_df = pd.DataFrame()  # ensure it's a valid empty DF
        
        # Step 2: Process and train each symbol
        results = {}
        all_training_data = []
        
        for symbol in self.symbols:
            logger.info(f"\n{'='*40}")
            logger.info(f"Processing: {symbol}")
            logger.info(f"{'='*40}")
            
            # Process symbol data from bhavcopy
            sym_df = self.process_symbol_data(fo_df, symbol) if not fo_df.empty else pd.DataFrame()
            
            # Kite API fallback for symbols not in bhavcopy (e.g. SENSEX/BSE)
            if sym_df.empty or len(sym_df) < 20:
                logger.info(f"{symbol}: No bhavcopy data, trying Kite API fallback...")
                sym_df = self._fetch_kite_ohlcv(symbol, days=kite_fallback_days)
                if sym_df.empty or len(sym_df) < 20:
                    logger.warning(f"{symbol}: Insufficient data from both sources, skipping")
                    continue
                logger.info(f"{symbol}: Using {len(sym_df)} candles from Kite API")
            
            # Add features
            sym_df = self.add_features(sym_df)
            
            # Create labels
            sym_df = self.create_labels(sym_df)
            
            # Prepare for training
            X, y, feature_names = self.prepare_features(sym_df)
            
            if len(X) < 30:
                logger.warning(f"{symbol}: Insufficient samples after processing")
                continue
            
            logger.info(f"{symbol}: {len(X)} samples, {len(feature_names)} features")
            
            # Train model
            result = self.train_symbol_model(X, y, symbol, feature_names)
            
            if result:
                results[symbol] = result
                
                # Save model
                model_path = self.model_dir / f"{symbol}_model_{timestamp}.joblib"
                joblib.dump(result, model_path)
                logger.info(f"Model saved: {model_path.name}")
            
            # Store training data
            sym_df["_symbol"] = symbol
            all_training_data.append(sym_df)
        
        # Step 3: Save combined training data
        if all_training_data:
            combined_df = pd.concat(all_training_data, ignore_index=True)
            data_path = self.cache_dir / f"training_data_full_{start_date}_{end_date}.csv"
            combined_df.to_csv(data_path, index=False)
            logger.info(f"\nTraining data saved: {data_path}")
        
        # Step 4: Summary
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 70)
        
        for symbol, res in results.items():
            m = res["metrics"]
            logger.info(f"{symbol:12} | Acc: {m['accuracy']:.1%} | F1: {m['f1']:.1%} | Samples: {res['n_samples']}")
        
        # Save results summary
        summary = {
            "timestamp": timestamp,
            "date_range": (str(start_date), str(end_date)),
            "symbols": list(results.keys()),
            "metrics": {s: r["metrics"] for s, r in results.items()},
            "feature_names": feature_names if results else []
        }
        
        summary_path = self.model_dir / f"training_summary_{timestamp}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\nSummary saved: {summary_path}")
        
        return results


class MonthlyUpdatePipeline:
    """
    Monthly update pipeline for continuous learning.
    
    Instead of collecting data daily, downloads monthly bhavcopy
    and updates models periodically.
    """
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.trainer = FullPipelineTrainer()
        
    def get_last_update_date(self) -> Optional[date]:
        """Get date of last model update."""
        model_dir = self.base_dir / "ml_models"
        
        if not model_dir.exists():
            return None
        
        # Find most recent summary
        summaries = list(model_dir.glob("training_summary_*.json"))
        
        if not summaries:
            return None
        
        latest = max(summaries, key=lambda p: p.stat().st_mtime)
        
        try:
            with open(latest) as f:
                data = json.load(f)
            end_date = data.get("date_range", [None, None])[1]
            if end_date:
                return datetime.strptime(end_date, "%Y-%m-%d").date()
        except:
            pass
        
        return None
    
    def check_update_needed(self) -> Tuple[bool, Optional[date], Optional[date]]:
        """Check if monthly update is needed."""
        last_update = self.get_last_update_date()
        today = date.today()
        
        if last_update is None:
            # First run - train on last 5 months
            start = today - timedelta(days=150)
            # Cap at archive availability (May 2024)
            end = min(date(2024, 5, 31), today - timedelta(days=1))
            start = max(start, date(2024, 1, 1))
            return True, start, end
        
        # Check if 30+ days since last update
        days_since = (today - last_update).days
        
        if days_since >= 30:
            start = last_update + timedelta(days=1)
            end = min(date(2024, 5, 31), today - timedelta(days=1))
            
            if start < end:
                return True, start, end
        
        return False, None, None
    
    def run_monthly_update(self, force: bool = False) -> Optional[Dict]:
        """
        Run monthly update if needed.
        
        Args:
            force: Force update even if not due
        """
        needs_update, start_date, end_date = self.check_update_needed()
        
        if not needs_update and not force:
            logger.info("No update needed. Last update is recent.")
            return None
        
        if force:
            # Full retrain
            start_date = date(2024, 1, 1)
            end_date = date(2024, 5, 31)
        
        logger.info(f"Running monthly update: {start_date} to {end_date}")
        
        return self.trainer.run_full_pipeline(
            start_date=start_date,
            end_date=end_date,
            force_download=force
        )


def main():
    """Run the full training pipeline."""
    trainer = FullPipelineTrainer()
    
    # Train on Jan-May 2024 (available in archives)
    results = trainer.run_full_pipeline(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 5, 31),
        force_download=False
    )
    
    if results:
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE!")
        print("=" * 70)
        print(f"Models trained for {len(results)} symbols")
        
        avg_acc = np.mean([r["metrics"]["accuracy"] for r in results.values()])
        avg_f1 = np.mean([r["metrics"]["f1"] for r in results.values()])
        
        print(f"Average Accuracy: {avg_acc:.1%}")
        print(f"Average F1 Score: {avg_f1:.1%}")


if __name__ == "__main__":
    main()
