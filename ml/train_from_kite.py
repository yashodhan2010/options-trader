"""
Train ML models using Kite Historical API data.

For symbols like SENSEX (BSE) that don't have NSE bhavcopy data,
this script fetches OHLCV history from Kite and trains models using
the same unified feature pipeline as full_pipeline.py.

Usage:
    python -m ml.train_from_kite                   # Train all symbols missing models
    python -m ml.train_from_kite --symbol SENSEX    # Train specific symbol
    python -m ml.train_from_kite --days 365         # Use 365 days of history
"""

import argparse
import joblib
import json
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from core.logger import logger
from config.settings import UNDERLYING_ASSETS, ML_CONFIG
from ml.unified_features import (
    HistoricalFeatureAdapter,
    get_unified_feature_names,
)


MODEL_DIR = Path("data/ml_models")


def get_symbols_without_models() -> List[str]:
    """Find training symbols that don't have any trained model."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    existing_models = set()
    for p in MODEL_DIR.glob("*_model_*.joblib"):
        # e.g. SENSEX_model_20260206_104219.joblib  →  SENSEX
        symbol = p.stem.rsplit("_model_", 1)[0]
        existing_models.add(symbol)
    
    training_symbols = ML_CONFIG.get("training_symbols", list(UNDERLYING_ASSETS.keys()))
    missing = [s for s in training_symbols if s not in existing_models]
    return missing


def fetch_kite_ohlcv(symbol: str, days: int = 365, interval: str = "day") -> pd.DataFrame:
    """
    Fetch historical OHLCV from Kite API for a given symbol.
    
    Uses instrument_token from UNDERLYING_ASSETS when available,
    falls back to exchange-based instrument lookup.
    """
    from data.data_fetcher import data_fetcher
    
    asset_cfg = UNDERLYING_ASSETS.get(symbol, {})
    exchange = asset_cfg.get("exchange", "NSE")
    
    logger.info(f"Fetching {days} days of {interval} data for {symbol} (exchange={exchange})")
    
    df = data_fetcher.get_historical_data(
        symbol=symbol,
        interval=interval,
        days=days,
        exchange=exchange,
    )
    
    if df.empty:
        logger.error(f"No historical data returned for {symbol}")
        return pd.DataFrame()
    
    # Ensure required columns exist
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            df[col] = 0.0
    
    # Reset index so 'date' becomes a column (Kite returns date as index)
    if df.index.name == "date" or isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
    
    # Sort chronologically
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
    
    logger.info(f"Fetched {len(df)} candles for {symbol} "
                f"(range: {df['date'].iloc[0]} -> {df['date'].iloc[-1]})")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply unified feature engineering on OHLCV data."""
    adapter = HistoricalFeatureAdapter()
    featured = adapter.extract_features(df)
    return featured


def create_labels(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Create next-day binary UP/DOWN labels."""
    df = df.copy()
    df["future_return"] = df["close"].shift(-horizon) / df["close"] - 1
    df["label"] = (df["future_return"] > 0).astype(float)
    df = df.dropna(subset=["future_return"])
    return df


def train_model(
    symbol: str,
    days: int = 365,
    min_samples: int = 30,
) -> Optional[Dict]:
    """
    Full train pipeline for one symbol using Kite data.
    
    1. Fetch OHLCV via Kite
    2. Engineer unified features
    3. Create labels
    4. Train RandomForest
    5. Save model
    """
    feature_names = get_unified_feature_names()
    
    # ── Step 1: fetch data ──
    raw = fetch_kite_ohlcv(symbol, days=days)
    if raw.empty:
        return None
    
    # ── Step 2: engineer features ──
    featured = engineer_features(raw)
    
    # ── Step 3: labels ──
    featured = create_labels(featured)
    
    total = len(featured)
    if total < min_samples:
        logger.warning(f"{symbol}: Only {total} samples after processing (need {min_samples}), skipping")
        return None
    
    # ── Step 4: prepare X, y ──
    available = [f for f in feature_names if f in featured.columns]
    missing = [f for f in feature_names if f not in featured.columns]
    if missing:
        logger.debug(f"Missing features (using 0): {missing[:5]}...")
        for f in missing:
            featured[f] = 0.0
    
    X = featured[feature_names].fillna(0).replace([np.inf, -np.inf], 0).values
    y = featured["label"].values
    
    up_count = int((y == 1).sum())
    down_count = int((y == 0).sum())
    logger.info(f"{symbol}: {total} samples  |  UP={up_count}  DOWN={down_count}")
    
    if up_count < 5 or down_count < 5:
        logger.warning(f"{symbol}: Severe class imbalance, skipping")
        return None
    
    # ── Step 5: time-series split ──
    n_splits = min(5, max(2, total // 30))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    train_idx, test_idx = list(tscv.split(X))[-1]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    logger.info(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")
    
    # ── Step 6: train RandomForest ──
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=8,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    
    logger.info(f"  Accuracy={metrics['accuracy']:.2%}  F1={metrics['f1']:.2%}")
    logger.info(f"  Top features: {', '.join(importance['feature'].head(5).tolist())}")
    
    # ── Step 7: save ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    result = {
        "model": model,
        "metrics": metrics,
        "feature_importance": importance,
        "feature_names": feature_names,
        "n_samples": total,
        "symbol": symbol,
        "data_source": "kite_historical",
        "training_days": days,
    }
    
    model_path = MODEL_DIR / f"{symbol}_model_{timestamp}.joblib"
    joblib.dump(result, model_path)
    logger.info(f"  Model saved: {model_path.name}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Train ML models from Kite historical data")
    parser.add_argument("--symbol", type=str, default=None,
                        help="Specific symbol to train (default: all missing)")
    parser.add_argument("--days", type=int, default=365,
                        help="Days of historical data (default: 365)")
    args = parser.parse_args()
    
    if args.symbol:
        symbols = [args.symbol]
    else:
        symbols = get_symbols_without_models()
        if not symbols:
            print("All training symbols already have models. Nothing to do.")
            return
    
    print(f"\n{'=' * 60}")
    print(f"KITE-BASED MODEL TRAINING")
    print(f"Symbols: {symbols}")
    print(f"History : {args.days} days")
    print(f"{'=' * 60}\n")
    
    results = {}
    for symbol in symbols:
        logger.info(f"\n{'-' * 40}")
        logger.info(f"Training: {symbol}")
        logger.info(f"{'-' * 40}")
        
        res = train_model(symbol, days=args.days)
        if res:
            results[symbol] = res
    
    # Summary
    print(f"\n{'=' * 60}")
    print("TRAINING SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Symbol':15} | {'Acc':>6} | {'F1':>6} | {'Samples':>7}")
    print("-" * 50)
    for sym, r in results.items():
        m = r["metrics"]
        print(f"{sym:15} | {m['accuracy']:5.1%} | {m['f1']:5.1%} | {r['n_samples']:7d}")
    
    if not results:
        print("No models trained.")


if __name__ == "__main__":
    main()
