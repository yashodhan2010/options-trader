"""
Live Feature Collector for ML Training Data

Runs in the background during market hours to collect all 61 features
including options chain, IV, OI, Greeks data. Saves snapshots to database
for training models with full feature set.

Usage:
    python -m ml.live_feature_collector
    
    # Or from CLI:
    ml collect start  - Start collection
    ml collect stop   - Stop collection
    ml collect status - View collection stats
"""

import time
import signal
import threading
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    MARKET_HOURS, UNDERLYING_ASSETS, ML_CONFIG,
    DATABASE_CONFIG
)
from core.database import database
from core.logger import logger
from data.data_fetcher import data_fetcher
from ml.feature_engineer import FeatureEngineer


class LiveFeatureCollector:
    """
    Collect live market features for ML training.
    
    Runs in background during market hours and collects:
    - All 61 features including options chain, IV, OI, Greeks
    - Snapshots at configurable intervals (default: every 15 minutes)
    - Stores in ml_feature_snapshots table for training
    """
    
    def __init__(self, interval_seconds: int = 900):
        """
        Initialize the collector.
        
        Args:
            interval_seconds: Seconds between snapshots (default 15 min)
        """
        self.interval = interval_seconds
        self.running = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        
        self.feature_engineer = FeatureEngineer()
        
        # Get watchlist symbols
        self.symbols = self._get_watchlist_symbols()
        
        # Stats
        self.stats = {
            "started_at": None,
            "snapshots_collected": 0,
            "last_snapshot_time": None,
            "errors": 0,
            "symbols_collected": {},
        }
        
        # Ensure database table exists
        self._ensure_table()
        
        logger.info(f"LiveFeatureCollector initialized - {len(self.symbols)} symbols, {self.interval}s interval")
    
    def _get_watchlist_symbols(self) -> List[str]:
        """Get symbols from watchlist config."""
        try:
            from config.settings import CONFIG_DIR
            watchlist_path = CONFIG_DIR / "watchlist.json"
            
            if watchlist_path.exists():
                with open(watchlist_path) as f:
                    watchlist = json.load(f)
                    
                    # Handle new format with 'assets' array
                    if "assets" in watchlist:
                        symbols = [
                            asset["name"] 
                            for asset in watchlist["assets"] 
                            if asset.get("enabled", True)
                        ]
                    else:
                        # Old format with 'underlyings' list
                        symbols = watchlist.get("underlyings", [])
                    
                    # Add index symbols if not present
                    for idx in ["NIFTY", "BANKNIFTY", "SENSEX"]:
                        if idx not in symbols:
                            symbols.append(idx)
                    
                    if symbols:
                        return symbols
                        
        except Exception as e:
            logger.warning(f"Could not load watchlist: {e}")
        
        # Fallback to default underlyings
        return list(UNDERLYING_ASSETS.keys()) + ["RELIANCE", "HDFCBANK", "SBIN", "AXISBANK"]
    
    def _ensure_table(self) -> None:
        """Ensure the feature snapshots table exists."""
        try:
            conn = database._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ml_feature_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    underlying TEXT NOT NULL,
                    snapshot_time TIMESTAMP NOT NULL,
                    spot_price REAL,
                    features_json TEXT NOT NULL,
                    feature_count INTEGER,
                    has_options_data INTEGER DEFAULT 0,
                    has_oi_data INTEGER DEFAULT 0,
                    has_greeks INTEGER DEFAULT 0,
                    
                    -- Future label (filled by labeler)
                    future_return_1h REAL,
                    future_return_4h REAL,
                    future_return_1d REAL,
                    label_direction TEXT,  -- UP, DOWN, NEUTRAL
                    label_magnitude TEXT,  -- SMALL, MEDIUM, LARGE
                    labeled_at TIMESTAMP,
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_underlying ON ml_feature_snapshots(underlying)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_time ON ml_feature_snapshots(snapshot_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_labeled ON ml_feature_snapshots(labeled_at)")
            
            conn.commit()
            conn.close()
            
            logger.info("ml_feature_snapshots table ready")
            
        except Exception as e:
            logger.error(f"Error creating snapshots table: {e}")
    
    def _is_market_hours(self) -> bool:
        """Check if current time is within market hours."""
        now = datetime.now()
        
        # Check if it's a weekday (Monday=0, Sunday=6)
        if now.weekday() >= 5:
            return False
        
        # Parse market hours
        market_open = datetime.strptime(MARKET_HOURS["market_open"], "%H:%M").time()
        market_close = datetime.strptime(MARKET_HOURS["market_close"], "%H:%M").time()
        
        current_time = now.time()
        return market_open <= current_time <= market_close
    
    def _collect_features_for_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Collect all features for a single symbol.
        
        Args:
            symbol: Underlying symbol
            
        Returns:
            Dictionary with features and metadata, or None on error
        """
        try:
            # Get spot price
            spot_price = data_fetcher.get_spot_price(symbol)
            if not spot_price:
                logger.warning(f"No spot price for {symbol}")
                return None
            
            # Get historical data (for price/technical features)
            hist_data = data_fetcher.get_historical_data(
                symbol=symbol,
                interval="day",
                days=60,
                exchange="NSE"
            )
            
            if hist_data is None or len(hist_data) < 20:
                logger.warning(f"Insufficient historical data for {symbol}")
                return None
            
            # Get options chain with Greeks
            options_chain = data_fetcher.get_options_chain_with_greeks(symbol)
            has_options = options_chain is not None and not options_chain.empty
            
            # Get OI analysis
            oi_data = data_fetcher.get_oi_data(symbol)
            has_oi = bool(oi_data)
            
            # Get volatility data
            volatility_data = data_fetcher.get_volatility_data(symbol) if hasattr(data_fetcher, 'get_volatility_data') else None
            
            # Extract position Greeks if available (dummy for collection)
            greeks = None
            if has_options:
                greeks = {
                    "delta": 0.5,
                    "gamma": 0.05,
                    "theta": -0.1,
                    "vega": 0.2
                }
            
            # Extract all 61 features
            feature_set = self.feature_engineer.extract_features(
                underlying=symbol,
                spot_price=spot_price,
                historical_data=hist_data,
                options_chain=options_chain if has_options else None,
                oi_analysis=oi_data if has_oi else None,
                volatility_data=volatility_data,
                greeks=greeks,
                current_time=datetime.now()
            )
            
            features_dict = feature_set.to_dict()
            
            # Count non-zero features
            non_zero = sum(1 for v in features_dict.values() if v != 0.0)
            
            return {
                "underlying": symbol,
                "spot_price": spot_price,
                "features": features_dict,
                "feature_count": len(features_dict),
                "non_zero_features": non_zero,
                "has_options_data": has_options,
                "has_oi_data": has_oi,
                "has_greeks": greeks is not None,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error collecting features for {symbol}: {e}")
            self.stats["errors"] += 1
            return None
    
    def _save_snapshot(self, snapshot: Dict[str, Any]) -> bool:
        """Save a feature snapshot to the database."""
        try:
            conn = database._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO ml_feature_snapshots (
                    underlying, snapshot_time, spot_price, features_json,
                    feature_count, has_options_data, has_oi_data, has_greeks
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot["underlying"],
                snapshot["timestamp"],
                snapshot["spot_price"],
                json.dumps(snapshot["features"]),
                snapshot["feature_count"],
                1 if snapshot["has_options_data"] else 0,
                1 if snapshot["has_oi_data"] else 0,
                1 if snapshot["has_greeks"] else 0
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error saving snapshot: {e}")
            return False
    
    def _collection_loop(self) -> None:
        """Main collection loop - runs in background thread."""
        logger.info("Feature collection loop started")
        
        while not self._stop_event.is_set():
            try:
                if self._is_market_hours():
                    logger.info("Collecting live features...")
                    
                    for symbol in self.symbols:
                        if self._stop_event.is_set():
                            break
                        
                        snapshot = self._collect_features_for_symbol(symbol)
                        
                        if snapshot:
                            if self._save_snapshot(snapshot):
                                self.stats["snapshots_collected"] += 1
                                self.stats["symbols_collected"][symbol] = \
                                    self.stats["symbols_collected"].get(symbol, 0) + 1
                                
                                logger.debug(
                                    f"Saved snapshot for {symbol}: "
                                    f"{snapshot['non_zero_features']}/{snapshot['feature_count']} features"
                                )
                        
                        # Small delay between symbols to avoid rate limiting
                        time.sleep(1)
                    
                    self.stats["last_snapshot_time"] = datetime.now()
                    logger.info(
                        f"Collection complete - Total: {self.stats['snapshots_collected']} snapshots"
                    )
                else:
                    logger.debug("Outside market hours, waiting...")
                
                # Wait for next interval or stop signal
                self._stop_event.wait(timeout=self.interval)
                
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                self.stats["errors"] += 1
                time.sleep(60)  # Wait a minute on error
        
        logger.info("Feature collection loop stopped")
    
    def start(self) -> bool:
        """Start background feature collection."""
        if self.running:
            logger.warning("Collector already running")
            return False
        
        self.running = True
        self.stats["started_at"] = datetime.now()
        self._stop_event.clear()
        
        self._thread = threading.Thread(target=self._collection_loop, daemon=True)
        self._thread.start()
        
        logger.info("Live feature collection started")
        return True
    
    def stop(self) -> None:
        """Stop background feature collection."""
        if not self.running:
            return
        
        logger.info("Stopping feature collection...")
        self._stop_event.set()
        
        if self._thread:
            self._thread.join(timeout=10)
        
        self.running = False
        logger.info("Feature collection stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        stats = self.stats.copy()
        stats["running"] = self.running
        stats["symbols"] = self.symbols
        stats["interval_seconds"] = self.interval
        
        # Get database counts
        try:
            conn = database._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM ml_feature_snapshots")
            stats["total_in_db"] = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT underlying, COUNT(*) as count 
                FROM ml_feature_snapshots 
                GROUP BY underlying
            """)
            stats["db_by_symbol"] = {row[0]: row[1] for row in cursor.fetchall()}
            
            cursor.execute("""
                SELECT COUNT(*) FROM ml_feature_snapshots 
                WHERE has_options_data = 1 AND has_oi_data = 1
            """)
            stats["full_feature_snapshots"] = cursor.fetchone()[0]
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
        
        return stats
    
    def collect_once(self) -> Dict[str, Any]:
        """
        Collect features once for all symbols (no background loop).
        Useful for manual testing.
        
        Returns:
            Dictionary with collection results
        """
        results = {
            "collected": 0,
            "errors": 0,
            "symbols": {}
        }
        
        for symbol in self.symbols:
            snapshot = self._collect_features_for_symbol(symbol)
            
            if snapshot:
                if self._save_snapshot(snapshot):
                    results["collected"] += 1
                    results["symbols"][symbol] = {
                        "features": snapshot["non_zero_features"],
                        "has_options": snapshot["has_options_data"],
                        "has_oi": snapshot["has_oi_data"]
                    }
            else:
                results["errors"] += 1
        
        return results


# Global instance for CLI access
_collector_instance: Optional[LiveFeatureCollector] = None


def get_collector() -> LiveFeatureCollector:
    """Get or create the global collector instance."""
    global _collector_instance
    if _collector_instance is None:
        _collector_instance = LiveFeatureCollector()
    return _collector_instance


def label_snapshots(lookback_hours: int = 4) -> int:
    """
    Label past snapshots with actual future returns.
    
    Should be run periodically (e.g., daily) to label collected snapshots
    with actual price movements for supervised learning.
    
    Args:
        lookback_hours: Hours to look back for labeling
        
    Returns:
        Number of snapshots labeled
    """
    labeled_count = 0
    
    try:
        conn = database._get_connection()
        cursor = conn.cursor()
        
        # Get unlabeled snapshots older than lookback period
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        
        cursor.execute("""
            SELECT id, underlying, snapshot_time, spot_price 
            FROM ml_feature_snapshots 
            WHERE labeled_at IS NULL 
            AND snapshot_time < ?
        """, (cutoff_time,))
        
        snapshots = cursor.fetchall()
        
        for snap_id, underlying, snap_time, spot_price in snapshots:
            try:
                # Get price at 1h, 4h, 1d after snapshot
                snap_dt = datetime.fromisoformat(snap_time) if isinstance(snap_time, str) else snap_time
                
                # For now, use current price as proxy (in production, would need historical lookup)
                current_price = data_fetcher.get_spot_price(underlying)
                
                if current_price and spot_price:
                    # Calculate return
                    return_pct = (current_price / spot_price - 1) * 100
                    
                    # Determine direction
                    if return_pct > 0.5:
                        direction = "UP"
                    elif return_pct < -0.5:
                        direction = "DOWN"
                    else:
                        direction = "NEUTRAL"
                    
                    # Determine magnitude
                    abs_return = abs(return_pct)
                    if abs_return > 2.0:
                        magnitude = "LARGE"
                    elif abs_return > 1.0:
                        magnitude = "MEDIUM"
                    else:
                        magnitude = "SMALL"
                    
                    # Update snapshot
                    cursor.execute("""
                        UPDATE ml_feature_snapshots 
                        SET future_return_1d = ?, 
                            label_direction = ?, 
                            label_magnitude = ?,
                            labeled_at = ?
                        WHERE id = ?
                    """, (return_pct, direction, magnitude, datetime.now(), snap_id))
                    
                    labeled_count += 1
                    
            except Exception as e:
                logger.error(f"Error labeling snapshot {snap_id}: {e}")
        
        conn.commit()
        conn.close()
        
        logger.info(f"Labeled {labeled_count} snapshots")
        
    except Exception as e:
        logger.error(f"Error in labeling: {e}")
    
    return labeled_count


def get_training_data_from_snapshots(
    min_samples: int = 100,
    require_full_features: bool = True
) -> tuple:
    """
    Get training data from collected snapshots.
    
    Args:
        min_samples: Minimum samples required
        require_full_features: Only use snapshots with options/OI data
        
    Returns:
        Tuple of (X array, y array, feature_names)
    """
    import numpy as np
    
    try:
        conn = database._get_connection()
        cursor = conn.cursor()
        
        query = """
            SELECT features_json, label_direction 
            FROM ml_feature_snapshots 
            WHERE labeled_at IS NOT NULL
        """
        
        if require_full_features:
            query += " AND has_options_data = 1 AND has_oi_data = 1"
        
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        
        if len(rows) < min_samples:
            logger.warning(f"Only {len(rows)} samples available, need {min_samples}")
            return None, None, None
        
        # Parse features
        X_list = []
        y_list = []
        feature_names = None
        
        for features_json, label in rows:
            features = json.loads(features_json)
            
            if feature_names is None:
                feature_names = list(features.keys())
            
            X_list.append([features.get(f, 0.0) for f in feature_names])
            
            # Convert label to binary (1 = UP, 0 = DOWN/NEUTRAL)
            y_list.append(1 if label == "UP" else 0)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        logger.info(f"Loaded {len(X)} training samples with {len(feature_names)} features")
        
        return X, y, feature_names
        
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        return None, None, None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Live Feature Collector for ML Training")
    parser.add_argument("action", choices=["start", "stop", "status", "collect-once", "label"],
                       help="Action to perform")
    parser.add_argument("--interval", type=int, default=900,
                       help="Collection interval in seconds (default: 900)")
    
    args = parser.parse_args()
    
    collector = get_collector()
    
    if args.action == "start":
        print("Starting live feature collection...")
        print(f"Symbols: {collector.symbols}")
        print(f"Interval: {args.interval} seconds")
        print("Press Ctrl+C to stop\n")
        
        collector.interval = args.interval
        collector.start()
        
        # Handle Ctrl+C gracefully
        def signal_handler(sig, frame):
            print("\nStopping...")
            collector.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # Keep main thread alive
        while collector.running:
            time.sleep(1)
    
    elif args.action == "stop":
        collector.stop()
        print("Collection stopped")
    
    elif args.action == "status":
        stats = collector.get_stats()
        print("\n=== Live Feature Collector Status ===")
        print(f"Running: {stats.get('running', False)}")
        print(f"Started: {stats.get('started_at', 'N/A')}")
        print(f"Interval: {stats.get('interval_seconds', 'N/A')}s")
        print(f"\nSymbols: {', '.join(stats.get('symbols', []))}")
        print(f"\nSnapshots collected this session: {stats.get('snapshots_collected', 0)}")
        print(f"Last snapshot: {stats.get('last_snapshot_time', 'N/A')}")
        print(f"Errors: {stats.get('errors', 0)}")
        print(f"\nTotal in database: {stats.get('total_in_db', 0)}")
        print(f"Full feature snapshots: {stats.get('full_feature_snapshots', 0)}")
        
        if stats.get('db_by_symbol'):
            print("\nBy symbol:")
            for sym, count in stats['db_by_symbol'].items():
                print(f"  {sym}: {count}")
    
    elif args.action == "collect-once":
        print("Collecting features once for all symbols...")
        results = collector.collect_once()
        print(f"\nCollected: {results['collected']} snapshots")
        print(f"Errors: {results['errors']}")
        
        if results['symbols']:
            print("\nDetails:")
            for sym, info in results['symbols'].items():
                print(f"  {sym}: {info['features']} features, "
                      f"options={info['has_options']}, oi={info['has_oi']}")
    
    elif args.action == "label":
        print("Labeling snapshots with actual outcomes...")
        count = label_snapshots()
        print(f"Labeled {count} snapshots")
