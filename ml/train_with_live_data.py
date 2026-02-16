"""
Train models using collected live feature snapshots + historical bhavcopy data.

Combines:
1. Historical bhavcopy data (Jan-May 2024): ~100 samples/symbol
2. Live collected snapshots (Dec 2025 - Feb 2026): ~145 labeled samples/symbol

This gives models RECENT market behavior to learn from.
"""

import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from core.database import database
from core.logger import logger
from ml.unified_features import (
    UnifiedFeatureDefinition,
    LiveFeatureAdapter,
    get_unified_feature_names,
)
from ml.full_pipeline import FullPipelineTrainer


def load_live_snapshots() -> pd.DataFrame:
    """
    Load labeled feature snapshots from the database.
    
    - Drops NEUTRAL labels (ambiguous direction)
    - Deduplicates to 1 snapshot per symbol per day (uses the one closest to market open)
    - Uses binary label: UP=1, DOWN=0
    
    Returns DataFrame with one row per symbol-day, columns = unified feature names + label.
    """
    import sqlite3
    
    conn = sqlite3.connect("data/trading_bot.db")
    
    # Only load UP/DOWN labeled snapshots (skip NEUTRAL — ambiguous)
    query = """
        SELECT underlying, snapshot_time, spot_price, features_json,
               future_return_1d, label_direction, label_magnitude, source
        FROM ml_feature_snapshots
        WHERE label_direction IN ('UP', 'DOWN')
        ORDER BY underlying, snapshot_time
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    logger.info(f"Loaded {len(df)} UP/DOWN labeled snapshots from database")
    
    if df.empty:
        return pd.DataFrame()
    
    # Deduplicate: keep 1 snapshot per symbol per day
    # Multiple intraday snapshots share the same future_return_1d
    df["snapshot_time"] = pd.to_datetime(df["snapshot_time"], format="mixed", utc=True)
    df["date_only"] = df["snapshot_time"].dt.date
    
    # Keep the first snapshot of each day (closest to open — most predictive)
    df = df.sort_values("snapshot_time")
    df = df.drop_duplicates(subset=["underlying", "date_only"], keep="first")
    
    logger.info(f"After dedup (1 per symbol-day): {len(df)} snapshots")
    
    # Parse features JSON and map to unified names
    adapter = LiveFeatureAdapter()
    unified_names = get_unified_feature_names()
    
    rows = []
    for _, row in df.iterrows():
        try:
            live_features = json.loads(row["features_json"])
            unified = adapter.adapt(live_features)
            
            # Build row with unified features
            record = {name: unified.get(name, 0.0) for name in unified_names}
            record["symbol"] = row["underlying"]
            record["date"] = str(row["date_only"])
            record["close"] = row["spot_price"]
            
            # Binary label: UP=1, DOWN=0
            record["label"] = 1.0 if row["label_direction"] == "UP" else 0.0
            record["source"] = "live"
            record["future_return_1d"] = row["future_return_1d"]
            
            # Replace NaN/inf
            for name in unified_names:
                v = record[name]
                if pd.isna(v) or np.isinf(v):
                    record[name] = 0.0
            
            rows.append(record)
        except Exception as e:
            logger.debug(f"Skipping snapshot: {e}")
            continue
    
    result = pd.DataFrame(rows)
    logger.info(f"Parsed {len(result)} live snapshots into unified features")
    
    # Summary by symbol
    for sym in sorted(result["symbol"].unique()):
        sym_df = result[result["symbol"] == sym]
        up = (sym_df["label"] == 1).sum()
        down = (sym_df["label"] == 0).sum()
        logger.info(f"  {sym}: {len(sym_df)} days (UP={up}, DOWN={down})")
    
    return result


def load_historical_data() -> pd.DataFrame:
    """
    Load the pre-processed historical bhavcopy training data.
    """
    data_path = Path("data/nse_cache/training_data_full_2024-01-01_2024-05-31.csv")
    
    if not data_path.exists():
        logger.warning("No historical training data found. Run full_pipeline first.")
        return pd.DataFrame()
    
    df = pd.read_csv(data_path)
    df["source"] = "historical"
    
    logger.info(f"Loaded {len(df)} historical training samples")
    
    return df


def train_combined():
    """
    Train per-symbol models on combined historical + live data.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path("data/ml_models")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    unified_names = get_unified_feature_names()
    
    # Load both data sources
    logger.info("=" * 70)
    logger.info("COMBINED TRAINING: Historical + Live Data")
    logger.info("=" * 70)
    
    live_df = load_live_snapshots()
    hist_df = load_historical_data()
    
    if live_df.empty:
        logger.error("No live data available!")
        return None
    
    # Prepare historical data with matching columns
    hist_ready = pd.DataFrame()
    if not hist_df.empty:
        hist_ready = hist_df.copy()
        # Ensure label column exists and is named consistently
        if "label" not in hist_ready.columns and "future_return" in hist_ready.columns:
            hist_ready["label"] = (hist_ready["future_return"] > 0).astype(float)
        if "_symbol" in hist_ready.columns:
            hist_ready["symbol"] = hist_ready["_symbol"]
    
    # Get all symbols from live data
    symbols = sorted(live_df["symbol"].unique())
    
    logger.info(f"\nSymbols to train: {symbols}")
    logger.info(f"Historical samples: {len(hist_ready)}")
    logger.info(f"Live samples: {len(live_df)}")
    
    results = {}
    
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    for symbol in symbols:
        logger.info(f"\n{'='*40}")
        logger.info(f"Training: {symbol}")
        logger.info(f"{'='*40}")
        
        # Get symbol data from both sources
        sym_live = live_df[live_df["symbol"] == symbol].copy()
        
        sym_hist = pd.DataFrame()
        if not hist_ready.empty and "symbol" in hist_ready.columns:
            sym_hist = hist_ready[hist_ready["symbol"] == symbol].copy()
        
        n_hist = len(sym_hist)
        n_live = len(sym_live)
        
        logger.info(f"  Historical: {n_hist} samples")
        logger.info(f"  Live: {n_live} samples")
        
        # Combine: historical first (older), then live (newer)
        # This preserves time ordering for TimeSeriesSplit
        parts = []
        if not sym_hist.empty:
            parts.append(sym_hist)
        parts.append(sym_live)
        
        combined = pd.concat(parts, ignore_index=True)
        
        # Extract feature matrix
        available_features = [f for f in unified_names if f in combined.columns]
        missing_features = [f for f in unified_names if f not in combined.columns]
        
        if missing_features:
            logger.debug(f"  Missing features (using 0): {missing_features[:5]}...")
            for f in missing_features:
                combined[f] = 0.0
        
        # Build X, y
        X = combined[unified_names].fillna(0).replace([np.inf, -np.inf], 0).values
        y = combined["label"].values
        
        total_samples = len(X)
        logger.info(f"  Total combined: {total_samples} samples, {len(unified_names)} features")
        
        if total_samples < 30:
            logger.warning(f"  {symbol}: Insufficient data ({total_samples} samples), skipping")
            continue
        
        # Check class balance
        up_count = (y == 1).sum()
        down_count = (y == 0).sum()
        logger.info(f"  Class balance: UP={up_count}, DOWN={down_count}")
        
        if up_count < 5 or down_count < 5:
            logger.warning(f"  {symbol}: Severe class imbalance, skipping")
            continue
        
        # Time series split - train on last fold
        n_splits = min(5, max(2, total_samples // 30))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        train_idx, test_idx = list(tscv.split(X))[-1]
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        logger.info(f"  Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Train Random Forest (same config as full_pipeline)
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=8,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        n_classes = len(set(y_test) | set(y_pred))
        avg = "weighted"  # Always weighted — ternary labels {0,1,2} can produce {0,2} splits where pos_label=1 fails
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average=avg, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average=avg, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, average=avg, zero_division=0)),
        }
        
        # Feature importance (top 10)
        importance = pd.DataFrame({
            "feature": unified_names,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        
        top_features = importance.head(10)
        
        logger.info(f"  Acc={metrics['accuracy']:.2%}, F1={metrics['f1']:.2%}")
        logger.info(f"  Top features: {', '.join(top_features['feature'].tolist()[:5])}")
        
        result = {
            "model": model,
            "metrics": metrics,
            "feature_importance": importance,
            "feature_names": unified_names,
            "n_samples": total_samples,
            "n_historical": n_hist,
            "n_live": n_live,
            "symbol": symbol,
        }
        
        results[symbol] = result
        
        # Save model
        model_path = model_dir / f"{symbol}_model_{timestamp}.joblib"
        joblib.dump(result, model_path)
        logger.info(f"  Model saved: {model_path.name}")
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("COMBINED TRAINING SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Symbol':15} | {'Acc':>6} | {'F1':>6} | {'Hist':>5} | {'Live':>5} | {'Total':>5}")
    logger.info("-" * 65)
    
    for symbol, res in results.items():
        m = res["metrics"]
        logger.info(
            f"{symbol:15} | {m['accuracy']:5.1%} | {m['f1']:5.1%} | "
            f"{res['n_historical']:5d} | {res['n_live']:5d} | {res['n_samples']:5d}"
        )
    
    if results:
        avg_acc = np.mean([r["metrics"]["accuracy"] for r in results.values()])
        avg_f1 = np.mean([r["metrics"]["f1"] for r in results.values()])
        total = sum(r["n_samples"] for r in results.values())
        logger.info("-" * 65)
        logger.info(f"{'AVERAGE':15} | {avg_acc:5.1%} | {avg_f1:5.1%} | {'':>5} | {'':>5} | {total:5d}")
    
    # Save summary
    summary = {
        "timestamp": timestamp,
        "data_sources": ["historical_bhavcopy_2024", "live_snapshots_2025-2026"],
        "symbols": list(results.keys()),
        "metrics": {s: r["metrics"] for s, r in results.items()},
        "sample_counts": {
            s: {"historical": r["n_historical"], "live": r["n_live"], "total": r["n_samples"]}
            for s, r in results.items()
        },
        "feature_names": unified_names,
    }
    
    summary_path = model_dir / f"training_summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSummary saved: {summary_path}")
    
    return results


if __name__ == "__main__":
    results = train_combined()
    
    if results:
        avg_acc = np.mean([r["metrics"]["accuracy"] for r in results.values()])
        avg_f1 = np.mean([r["metrics"]["f1"] for r in results.values()])
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE!")
        print("=" * 70)
        print(f"Models trained for {len(results)} symbols")
        print(f"Average Accuracy: {avg_acc:.1%}")
        print(f"Average F1 Score: {avg_f1:.1%}")
    else:
        print("Training failed - no results")
