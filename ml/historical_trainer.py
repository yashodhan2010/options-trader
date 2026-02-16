"""
Historical Data Trainer

Downloads NSE F&O bhavcopy data and trains ML model with full feature set
including OI, OI change, and options-specific features.
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

from data.nse_downloader import NSEDownloader
from ml.feature_engineer import FeatureEngineer
from ml.model_trainer import ModelTrainer
from core.logger import logger


class HistoricalTrainer:
    """
    End-to-end trainer using NSE historical data.
    
    Downloads F&O bhavcopy, extracts features, and trains ML model.
    """
    
    def __init__(
        self,
        symbols: List[str] = None,
        cache_dir: str = "data/nse_cache"
    ):
        self.symbols = symbols or ["NIFTY", "BANKNIFTY"]
        self.downloader = NSEDownloader(cache_dir)
        self.feature_engineer = FeatureEngineer()
        self.cache_dir = Path(cache_dir)
        
    def download_historical_data(
        self,
        start_date: date = date(2024, 1, 1),
        end_date: date = date(2024, 5, 31),
        delay: float = 0.5
    ) -> pd.DataFrame:
        """
        Download F&O bhavcopy for date range.
        
        Args:
            start_date: Start date
            end_date: End date
            delay: Seconds between requests
            
        Returns:
            Combined DataFrame with all data
        """
        logger.info(f"Downloading F&O data: {start_date} to {end_date}")
        
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
                logger.info(f"[{i+1}/{len(dates)}] {d}: {len(df)} records")
            else:
                logger.debug(f"[{i+1}/{len(dates)}] {d}: holiday/unavailable")
            
            import time
            time.sleep(delay)
        
        if not all_data:
            logger.error("No data downloaded!")
            return pd.DataFrame()
        
        combined = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total records: {len(combined)}")
        
        return combined
    
    def process_fo_data(self, fo_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process F&O bhavcopy into OHLCV + OI features.
        
        Extracts:
        - Futures data (OHLCV)
        - Options OI aggregation
        - Put-Call OI ratio
        - OI change metrics
        """
        if fo_df.empty:
            return pd.DataFrame()
        
        processed = []
        
        for symbol in self.symbols:
            symbol_data = fo_df[fo_df["SYMBOL"] == symbol].copy()
            
            if symbol_data.empty:
                continue
            
            for dt in symbol_data["DATE"].unique():
                day_data = symbol_data[symbol_data["DATE"] == dt]
                
                # Get futures data (nearest expiry)
                futures = day_data[day_data["INSTRUMENT"].isin(["FUTIDX", "FUTSTK"])]
                if futures.empty:
                    continue
                
                # Sort by expiry and get nearest
                futures = futures.sort_values("EXPIRY_DT")
                fut_row = futures.iloc[0]
                
                # Get options data
                options = day_data[day_data["INSTRUMENT"].isin(["OPTIDX", "OPTSTK"])]
                
                # Aggregate OI by option type
                call_oi = options[options["OPTION_TYP"] == "CE"]["OPEN_INT"].sum()
                put_oi = options[options["OPTION_TYP"] == "PE"]["OPEN_INT"].sum()
                
                call_oi_chg = options[options["OPTION_TYP"] == "CE"]["CHG_IN_OI"].sum()
                put_oi_chg = options[options["OPTION_TYP"] == "PE"]["CHG_IN_OI"].sum()
                
                # Futures OI
                fut_oi = fut_row.get("OPEN_INT", 0)
                fut_oi_chg = fut_row.get("CHG_IN_OI", 0)
                
                # Calculate ratios
                pcr_oi = put_oi / call_oi if call_oi > 0 else 1.0
                
                row = {
                    "symbol": symbol,
                    "date": dt,
                    "open": fut_row["OPEN"],
                    "high": fut_row["HIGH"],
                    "low": fut_row["LOW"],
                    "close": fut_row["CLOSE"],
                    "volume": fut_row.get("CONTRACTS", 0) * 1000,  # Approx
                    
                    # OI features
                    "fut_oi": fut_oi,
                    "fut_oi_change": fut_oi_chg,
                    "call_oi": call_oi,
                    "put_oi": put_oi,
                    "call_oi_change": call_oi_chg,
                    "put_oi_change": put_oi_chg,
                    "pcr_oi": pcr_oi,
                    "total_oi": call_oi + put_oi + fut_oi,
                }
                
                processed.append(row)
        
        df = pd.DataFrame(processed)
        if not df.empty:
            df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
        
        return df
    
    def add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to processed data."""
        if df.empty:
            return df
        
        result = []
        
        for symbol in df["symbol"].unique():
            sym_df = df[df["symbol"] == symbol].copy()
            sym_df = sym_df.sort_values("date")
            
            # Price-based features
            sym_df["returns"] = sym_df["close"].pct_change()
            sym_df["log_returns"] = np.log(sym_df["close"] / sym_df["close"].shift(1))
            
            # Moving averages (shorter windows for limited data)
            for window in [3, 5, 10]:
                sym_df[f"sma_{window}"] = sym_df["close"].rolling(window, min_periods=1).mean()
                sym_df[f"ema_{window}"] = sym_df["close"].ewm(span=window, min_periods=1).mean()
            
            # Volatility
            sym_df["volatility_5"] = sym_df["returns"].rolling(5, min_periods=2).std()
            sym_df["volatility_10"] = sym_df["returns"].rolling(10, min_periods=3).std()
            
            # RSI (shorter window)
            delta = sym_df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(7, min_periods=3).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(7, min_periods=3).mean()
            rs = gain / loss.replace(0, np.nan)
            sym_df["rsi_7"] = 100 - (100 / (1 + rs))
            
            # MACD (shorter)
            ema8 = sym_df["close"].ewm(span=8, min_periods=3).mean()
            ema17 = sym_df["close"].ewm(span=17, min_periods=5).mean()
            sym_df["macd"] = ema8 - ema17
            sym_df["macd_signal"] = sym_df["macd"].ewm(span=5, min_periods=2).mean()
            
            # Bollinger Bands
            sym_df["bb_mid"] = sym_df["close"].rolling(10, min_periods=3).mean()
            bb_std = sym_df["close"].rolling(10, min_periods=3).std()
            sym_df["bb_upper"] = sym_df["bb_mid"] + 2 * bb_std
            sym_df["bb_lower"] = sym_df["bb_mid"] - 2 * bb_std
            sym_df["bb_width"] = (sym_df["bb_upper"] - sym_df["bb_lower"]) / sym_df["bb_mid"]
            
            # OI-based features
            sym_df["oi_change_ratio"] = sym_df["fut_oi_change"] / sym_df["fut_oi"].replace(0, 1)
            sym_df["pcr_oi_ma5"] = sym_df["pcr_oi"].rolling(5, min_periods=1).mean()
            sym_df["oi_trend"] = sym_df["fut_oi"].diff(3)
            
            # Price-OI relationship
            sym_df["price_change"] = sym_df["close"].pct_change()
            sym_df["oi_change_pct"] = sym_df["fut_oi_change"] / sym_df["fut_oi"].replace(0, 1)
            
            # Intraday range
            sym_df["range_pct"] = (sym_df["high"] - sym_df["low"]) / sym_df["close"]
            
            result.append(sym_df)
        
        return pd.concat(result, ignore_index=True)
    
    def create_labels(self, df: pd.DataFrame, horizon: int = 1, threshold: float = 0.005) -> pd.DataFrame:
        """
        Create classification labels based on future returns.
        
        Args:
            horizon: Days to look ahead
            threshold: Min return for bullish/bearish classification
        """
        if df.empty:
            return df
        
        result = []
        
        for symbol in df["symbol"].unique():
            sym_df = df[df["symbol"] == symbol].copy()
            sym_df = sym_df.sort_values("date")
            
            # Future return
            sym_df["future_return"] = sym_df["close"].shift(-horizon) / sym_df["close"] - 1
            
            # Classification: 0=BEARISH, 1=NEUTRAL, 2=BULLISH (matches DIRECTION_MAP)
            sym_df["label"] = 1  # Default NEUTRAL
            sym_df.loc[sym_df["future_return"] > threshold, "label"] = 2   # BULLISH
            sym_df.loc[sym_df["future_return"] < -threshold, "label"] = 0  # BEARISH
            
            # Binary classification (for simplicity)
            sym_df["label_binary"] = (sym_df["future_return"] > 0).astype(float)
            
            result.append(sym_df)
        
        combined = pd.concat(result, ignore_index=True)
        
        # Remove rows where we can't calculate the label (last row per symbol)
        combined = combined.dropna(subset=["future_return"])
        
        return combined
    
    def prepare_training_data(
        self,
        start_date: date = date(2024, 1, 1),
        end_date: date = date(2024, 5, 31)
    ) -> tuple:
        """
        Full pipeline: download, process, feature engineer, label.
        
        Returns:
            (X, y, feature_names, df)
        """
        # Step 1: Download
        logger.info("Step 1: Downloading historical data...")
        fo_df = self.download_historical_data(start_date, end_date)
        
        if fo_df.empty:
            return None, None, None, None
        
        # Step 2: Process F&O data
        logger.info("Step 2: Processing F&O data...")
        processed = self.process_fo_data(fo_df)
        logger.info(f"Processed {len(processed)} daily records")
        
        if processed.empty:
            logger.error("No processed data!")
            return None, None, None, None
        
        # Step 3: Add technical features
        logger.info("Step 3: Adding technical features...")
        featured = self.add_technical_features(processed)
        
        # Step 4: Create labels
        logger.info("Step 4: Creating labels...")
        labeled = self.create_labels(featured)
        
        # Step 5: Prepare for training
        logger.info("Step 5: Preparing training data...")
        
        # Feature columns (exclude non-features)
        exclude_cols = ["symbol", "date", "future_return", "label", "label_binary"]
        feature_cols = [c for c in labeled.columns if c not in exclude_cols]
        
        # Fill remaining NaN with 0 (for first few rows where rolling can't compute)
        labeled[feature_cols] = labeled[feature_cols].fillna(0)
        
        # Replace inf with large values
        labeled[feature_cols] = labeled[feature_cols].replace([np.inf, -np.inf], 0)
        
        X = labeled[feature_cols].values
        y = labeled["label_binary"].values
        
        logger.info(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Features: {feature_cols}")
        logger.info(f"Label distribution: 0={np.sum(y==0)}, 1={np.sum(y==1)}")
        
        return X, y, feature_cols, labeled
    
    def train_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        model_type: str = "random_forest"
    ) -> Dict:
        """Train ML model on prepared data."""
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False  # Time series split
        )
        
        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Train model
        if model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        n_classes = len(set(y_test) | set(y_pred))
        avg = "weighted"  # Always weighted — ternary labels {0,1,2} can produce {0,2} splits
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average=avg, zero_division=0),
            "recall": recall_score(y_test, y_pred, average=avg, zero_division=0),
            "f1": f1_score(y_test, y_pred, average=avg, zero_division=0),
        }
        
        logger.info(f"Model Performance:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")
        
        # Feature importance
        importance = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        
        logger.info("\nTop 10 Features:")
        for _, row in importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return {
            "model": model,
            "metrics": metrics,
            "feature_importance": importance,
            "feature_names": feature_names
        }
    
    def run_full_pipeline(
        self,
        start_date: date = date(2024, 1, 1),
        end_date: date = date(2024, 5, 31),
        save_model: bool = True
    ) -> Dict:
        """
        Run complete training pipeline.
        
        1. Download historical F&O data
        2. Process and engineer features
        3. Train model
        4. Save model
        """
        logger.info("=" * 60)
        logger.info("Historical ML Training Pipeline")
        logger.info(f"Symbols: {self.symbols}")
        logger.info(f"Date Range: {start_date} to {end_date}")
        logger.info("=" * 60)
        
        # Prepare data
        X, y, feature_names, df = self.prepare_training_data(start_date, end_date)
        
        if X is None:
            logger.error("Failed to prepare training data")
            return None
        
        # Train model
        result = self.train_model(X, y, feature_names)
        
        # Save model
        if save_model:
            import joblib
            from datetime import datetime
            
            model_dir = Path("data/ml_models")
            model_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = model_dir / f"fo_model_{timestamp}.joblib"
            
            joblib.dump({
                "model": result["model"],
                "feature_names": feature_names,
                "metrics": result["metrics"],
                "symbols": self.symbols,
                "date_range": (str(start_date), str(end_date))
            }, model_path)
            
            logger.info(f"\nModel saved to: {model_path}")
            result["model_path"] = str(model_path)
        
        # Save processed data
        data_path = self.cache_dir / f"training_data_{start_date}_{end_date}.csv"
        df.to_csv(data_path, index=False)
        logger.info(f"Training data saved to: {data_path}")
        
        return result


def main():
    """Run the historical training pipeline."""
    trainer = HistoricalTrainer(
        symbols=["NIFTY", "BANKNIFTY"]
    )
    
    # Train on Jan-May 2024 data
    result = trainer.run_full_pipeline(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 5, 31),
        save_model=True
    )
    
    if result:
        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Accuracy: {result['metrics']['accuracy']:.2%}")
        print(f"F1 Score: {result['metrics']['f1']:.2%}")
        print("=" * 60)


if __name__ == "__main__":
    main()
