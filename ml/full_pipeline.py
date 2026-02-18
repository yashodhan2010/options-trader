"""
Simplified Training Pipeline - Live Snapshots Only

Trains per-symbol models using ONLY live feature snapshots collected by the bot
from the ml_feature_snapshots table. Each snapshot includes real Greeks, OI, and IV
collected during market hours.

No historical data download - just database queries and model training.
"""

import pandas as pd
import numpy as np
import json
import joblib
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from config.settings import ML_CONFIG
from core.logger import logger

try:
    import optuna
    OPTUNA_AVAILABLE = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    OPTUNA_AVAILABLE = False


class SimplifiedPipelineTrainer:
    """
    Simplified training pipeline using only live snapshots from database.
    
    Data flow:
    1. Load labeled snapshots from ml_feature_snapshots table
    2. Filter by symbol and label_direction
    3. Scale features
    4. Train ensemble models (Random Forest + LightGBM + XGBoost)
    5. Save models with results
    """
    
    # Default symbols (indices + stocks)
    INDEX_SYMBOLS = ["NIFTY", "BANKNIFTY", "FINNIFTY", "SENSEX"]
    
    def __init__(
        self,
        watchlist_path: str = "config/watchlist.json",
        model_dir: str = "data/ml_models",
        db_path: str = "data/trading_bot.db"
    ):
        self.watchlist_path = Path(watchlist_path)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        
        # Load symbols
        self.symbols = self._load_symbols()
        logger.info(f"Loaded {len(self.symbols)} symbols: {self.symbols}")
    
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
    
    def _load_live_snapshots(self, symbol: str) -> pd.DataFrame:
        """
        Load labeled snapshots from database for a symbol.
        
        Returns DataFrame with:
        - Individual feature columns extracted from features_json
        - label_direction: UP, DOWN, or NEUTRAL
        - snapshot_time: When the snapshot was taken
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = f"""
                SELECT * FROM ml_feature_snapshots
                WHERE underlying = '{symbol}'
                AND label_direction IN ('UP', 'DOWN')
                ORDER BY snapshot_time
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                logger.warning(f"{symbol}: No labeled snapshots found")
                return pd.DataFrame()
            
            # Extract features from JSON
            df['features'] = df['features_json'].apply(lambda x: json.loads(x) if isinstance(x, str) else {})
            
            # Expand features into columns
            features_df = pd.json_normalize(df['features'])
            
            # Combine with original dataframe
            df = pd.concat([df.drop(columns=['features_json', 'features']), features_df], axis=1)
            
            logger.info(f"{symbol}: Loaded {len(df)} labeled snapshots with {len(features_df.columns)} features")
            return df
            
        except Exception as e:
            logger.error(f"{symbol}: Error loading snapshots - {e}")
            return pd.DataFrame()
    
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features and labels for training.
        
        Returns:
            X: Feature matrix (n_samples, n_features)
            y: Label vector (n_samples,) - 0=DOWN, 1=NEUTRAL, 2=UP
            feature_names: List of feature column names
        """
        # Define columns to exclude (metadata and labels)
        exclude_cols = {
            'id', 'underlying', 'snapshot_time', 'labeled_at', 'created_at',
            'label_direction', 'label_magnitude', 'source',
            'has_options_data', 'has_oi_data', 'has_greeks',
            'feature_count', 'spot_price', 'future_return_1h', 
            'future_return_4h', 'future_return_1d'
        }
        
        # Get all numeric feature columns
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols 
                       and df[col].dtype in ('float64', 'float32', 'int64', 'int32')]
        
        if not feature_cols:
            logger.warning("No numeric feature columns found")
            return np.array([]), np.array([]), []
        
        # Remove features that are mostly NaN (keep those with >30% non-null values)
        min_coverage = 0.3
        valid_features = [col for col in feature_cols 
                         if df[col].notna().sum() / len(df) >= min_coverage]
        
        if not valid_features:
            logger.warning("No features with sufficient data coverage")
            return np.array([]), np.array([]), []
        
        logger.info(f"  Using {len(valid_features)}/{len(feature_cols)} features with >{min_coverage*100:.0f}% coverage")
        
        # Create feature matrix and fill NaN with median
        X = df[valid_features].copy()
        X = X.fillna(X.median())
        X = X.values
        
        # Encode labels: DOWN=0, NEUTRAL=1, UP=2
        label_map = {'DOWN': 0, 'NEUTRAL': 1, 'UP': 2}
        y = df['label_direction'].map(label_map).values
        
        # Remove rows with NaN in features or labels
        valid_idx = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_idx]
        y = y[valid_idx].astype(int)
        
        if len(X) == 0:
            logger.warning(f"No valid samples after cleaning")
            return np.array([]), np.array([]), []
        
        logger.info(f"  After NaN handling: {len(X)} valid samples")
        
        # Scale features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        return X, y, valid_features
    
    def train_symbol_model(self, X: np.ndarray, y: np.ndarray, symbol: str, feature_names: List[str]) -> Optional[Dict]:
        """
        Train ensemble model for a symbol.
        
        Ensemble: Random Forest + LightGBM + XGBoost with equal weights.
        """
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
            
            try:
                import lightgbm as lgb
                LGBM_AVAILABLE = True
            except:
                LGBM_AVAILABLE = False
            
            try:
                from xgboost import XGBClassifier
                XGB_AVAILABLE = True
            except:
                XGB_AVAILABLE = False
            
            # Check which classes are present
            unique_classes = np.unique(y)
            logger.info(f"  Classes present: {unique_classes} (mapping: 0=DOWN, 1=NEUTRAL, 2=UP)")
            
            # Train/test split - only stratify if all classes are present
            stratify = y if len(unique_classes) >= 2 else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=stratify
            )
            
            # Train models
            models = {}
            
            # Random Forest (always available)
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            models['rf'] = rf
            
            # LightGBM (if available)
            if LGBM_AVAILABLE:
                try:
                    lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
                    lgb_model.fit(X_train, y_train)
                    models['lgb'] = lgb_model
                except Exception as lgb_e:
                    logger.warning(f"LightGBM training failed: {lgb_e}")
            
            # XGBoost (if available)
            if XGB_AVAILABLE:
                try:
                    xgb = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
                    xgb.fit(X_train, y_train)
                    models['xgb'] = xgb
                except Exception as xgb_e:
                    logger.warning(f"XGBoost training failed: {xgb_e}")
            
            # Ensemble voting
            if not models:
                logger.warning(f"{symbol}: No models trained successfully")
                return None
                
            y_pred_proba = None
            for name, model in models.items():
                proba = model.predict_proba(X_test)
                if y_pred_proba is None:
                    y_pred_proba = proba / len(models)
                else:
                    y_pred_proba += proba / len(models)
            
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            # Metrics (handle older sklearn versions that don't have zero_division)
            try:
                auc_roc = roc_auc_score(y_test, y_pred_proba[:, 1:] if y_pred_proba.shape[1] == 2 else y_pred_proba, multi_class='ovr', zero_division=0)
            except:
                try:
                    auc_roc = roc_auc_score(y_test, y_pred_proba[:, 1] if y_pred_proba.shape[1] == 2 else y_pred_proba, multi_class='ovr' if y_pred_proba.shape[1] > 2 else None, zero_division=0)
                except:
                    auc_roc = 0.0
            
            try:
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                    'auc_roc': auc_roc,
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                    'individual_metrics': {
                        name: {
                            'accuracy': accuracy_score(y_test, model.predict(X_test)),
                            'f1_score': f1_score(y_test, model.predict(X_test), average='weighted', zero_division=0)
                        }
                        for name, model in models.items()
                    },
                    'optimized_weights': {name: 1.0 / len(models) for name in models.keys()}
                }
            except TypeError:
                # Older sklearn version without zero_division
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted'),
                    'recall': recall_score(y_test, y_pred, average='weighted'),
                    'f1_score': f1_score(y_test, y_pred, average='weighted'),
                    'auc_roc': auc_roc,
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                    'individual_metrics': {
                        name: {
                            'accuracy': accuracy_score(y_test, model.predict(X_test)),
                            'f1_score': f1_score(y_test, model.predict(X_test), average='weighted')
                        }
                        for name, model in models.items()
                    },
                    'optimized_weights': {name: 1.0 / len(models) for name in models.keys()}
                }
            
            logger.info(f"{symbol} (ensemble {len(models)} models): "
                       f"Acc={metrics['accuracy']:.1%}, F1={metrics['f1_score']:.1%} ({len(y_test)} test samples)")
            
            result = {
                'models': models,
                'feature_names': feature_names,
                'metrics': metrics,
                'n_samples': len(X),
                'test_samples': len(X_test),
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"{symbol}: Training failed - {e}")
            return None
    
    def run_full_pipeline(self) -> Dict:
        """
        Train models for all symbols using live snapshots.
        
        Returns:
            Dict with results for each symbol
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info("=" * 70)
        logger.info("SIMPLIFIED TRAINING PIPELINE (Live Snapshots Only)")
        logger.info(f"Symbols: {self.symbols}")
        logger.info("=" * 70)
        
        results = {}
        
        for symbol in self.symbols:
            logger.info(f"\n{'='*40}")
            logger.info(f"Processing: {symbol}")
            logger.info(f"{'='*40}")
            
            # Load snapshots
            snap_df = self._load_live_snapshots(symbol)
            
            if snap_df.empty or len(snap_df) < 30:
                logger.warning(f"{symbol}: Insufficient snapshots (need >=30)")
                continue
            
            # Prepare features
            X, y, feature_names = self._prepare_features(snap_df)
            
            if len(X) < 30:
                logger.warning(f"{symbol}: Insufficient valid samples after preparation")
                continue
            
            logger.info(f"{symbol}: {len(X)} samples, {len(feature_names)} features")
            
            # Train
            result = self.train_symbol_model(X, y, symbol, feature_names)
            
            if result:
                results[symbol] = result
                
                # Save model
                model_path = self.model_dir / f"{symbol}_model_{timestamp}.joblib"
                joblib.dump(result, model_path)
                logger.info(f"Model saved: {model_path.name}")
        
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 70)
        
        for symbol, res in results.items():
            m = res["metrics"]
            logger.info(f"{symbol:12} | Acc: {m['accuracy']:.1%} | F1: {m['f1_score']:.1%} | "
                       f"Samples: {res['n_samples']}")
        
        # Save summary
        summary = {
            "timestamp": timestamp,
            "date_range": [str(date.today())],
            "symbols": list(results.keys()),
            "metrics": {s: r["metrics"] for s, r in results.items()},
            "feature_names": list(results.values())[0]["feature_names"] if results else []
        }
        
        summary_path = self.model_dir / f"training_summary_{timestamp}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"\nSummary saved: {summary_path}")
        
        return results


def run_full_pipeline(
    start_date: date = None,
    end_date: date = None,
    force_download: bool = False,
    **kwargs
) -> Dict:
    """
    Public function to run the training pipeline.
    
    Args ignored (kept for compatibility):
        start_date, end_date, force_download - not used
    
    Returns:
        Results dict with per-symbol models and metrics
    """
    trainer = SimplifiedPipelineTrainer()
    return trainer.run_full_pipeline()


if __name__ == "__main__":
    results = run_full_pipeline()
    logger.info(f"\nTraining complete! {len(results)} models trained.")
