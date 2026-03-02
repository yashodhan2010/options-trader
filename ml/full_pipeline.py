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
        
        # NOTE: Scaling moved to train_symbol_model to avoid train/test leakage
        # (scaler must be fit on training data only)
        
        return X, y, valid_features
    
    def train_symbol_model(self, X: np.ndarray, y: np.ndarray, symbol: str, feature_names: List[str]) -> Optional[Dict]:
        """
        Train ensemble model for a symbol.
        
        Uses TEMPORAL (chronological) train/test split with embargo gap to prevent
        data leakage from autocorrelated time-series samples.
        
        Ensemble: Random Forest + LightGBM + XGBoost with equal weights.
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
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
            
            # TEMPORAL train/test split (data is already in chronological order)
            # Add embargo gap between train and test to avoid near-duplicate leakage
            embargo_frac = 0.02  # ~2% gap between train and test
            train_end = int(len(X) * 0.8)
            embargo_size = max(1, int(len(X) * embargo_frac))
            test_start = min(train_end + embargo_size, len(X))
            
            X_train, y_train = X[:train_end], y[:train_end]
            X_test, y_test = X[test_start:], y[test_start:]
            
            logger.info(f"  Temporal split: {len(X_train)} train, {embargo_size} embargo, {len(X_test)} test")
            
            if len(X_test) < 5:
                logger.warning(f"{symbol}: Too few test samples after temporal split")
                return None

            # Class-balance guardrails (prevents unstable one-sided models in low-data regime)
            class_guard = ML_CONFIG.get("training_class_guard", {})
            if class_guard.get("enabled", True):
                train_classes, train_counts = np.unique(y_train, return_counts=True)
                test_classes, _ = np.unique(y_test, return_counts=True)

                if len(train_classes) < 2:
                    logger.warning(f"{symbol}: Skipping training - only one class in train split: {train_classes}")
                    return None

                if class_guard.get("require_test_class_diversity", True) and len(test_classes) < 2:
                    logger.warning(f"{symbol}: Skipping training - only one class in test split: {test_classes}")
                    return None

                minority_samples = int(np.min(train_counts))
                minority_ratio = minority_samples / max(1, len(y_train))
                min_minority_samples = int(class_guard.get("min_minority_samples", 20))
                min_minority_ratio = float(class_guard.get("min_minority_ratio", 0.10))

                if minority_samples < min_minority_samples or minority_ratio < min_minority_ratio:
                    logger.warning(
                        f"{symbol}: Skipping training - class imbalance too high "
                        f"(minority={minority_samples}, ratio={minority_ratio:.1%}, "
                        f"thresholds: samples>={min_minority_samples}, ratio>={min_minority_ratio:.1%})"
                    )
                    return None
            
            # Scale features - fit on TRAINING data only (prevents test data leakage)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Train models
            models = {}
            
            # Random Forest (always available) - balanced class weights for imbalanced data
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
            rf.fit(X_train, y_train)
            models['rf'] = rf
            
            # LightGBM (if available) - balanced class weights
            if LGBM_AVAILABLE:
                try:
                    lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1, is_unbalance=True)
                    lgb_model.fit(X_train, y_train)
                    models['lgb'] = lgb_model
                except Exception as lgb_e:
                    logger.warning(f"LightGBM training failed: {lgb_e}")
            
            # XGBoost (if available)
            if XGB_AVAILABLE:
                try:
                    # XGBoost requires consecutive labels 0..N-1
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    y_train_enc = le.fit_transform(y_train.astype(int))
                    
                    xgb = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
                    xgb.fit(X_train, y_train_enc)
                    xgb._label_encoder = le  # Store for inverse_transform during predict
                    models['xgb'] = xgb
                except Exception as xgb_e:
                    logger.warning(f"XGBoost training failed: {xgb_e}")
            
            # Ensemble voting
            if not models:
                logger.warning(f"{symbol}: No models trained successfully")
                return None
            
            ensemble_classes = np.unique(y_train)

            y_pred_proba = None
            for name, model in models.items():
                proba = model.predict_proba(X_test)

                if hasattr(model, '_label_encoder') and hasattr(model._label_encoder, 'classes_'):
                    model_classes = model._label_encoder.classes_
                elif hasattr(model, 'classes_'):
                    model_classes = model.classes_
                else:
                    model_classes = np.arange(proba.shape[1])

                aligned_proba = np.zeros((len(X_test), len(ensemble_classes)), dtype=float)
                class_to_idx = {cls: idx for idx, cls in enumerate(ensemble_classes)}

                for col_idx, class_label in enumerate(model_classes):
                    if class_label in class_to_idx:
                        aligned_proba[:, class_to_idx[class_label]] = proba[:, col_idx]

                row_sum = aligned_proba.sum(axis=1, keepdims=True)
                row_sum[row_sum == 0] = 1.0
                aligned_proba = aligned_proba / row_sum

                if y_pred_proba is None:
                    y_pred_proba = aligned_proba / len(models)
                else:
                    y_pred_proba += aligned_proba / len(models)
            
            # Map argmax column indices back to actual class labels
            y_pred = ensemble_classes[np.argmax(y_pred_proba, axis=1)]
            
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
                            'accuracy': accuracy_score(
                                y_test,
                                model._label_encoder.inverse_transform(model.predict(X_test).astype(int))
                                if hasattr(model, '_label_encoder') and hasattr(model._label_encoder, 'inverse_transform')
                                else model.predict(X_test)
                            ),
                            'f1_score': f1_score(
                                y_test,
                                model._label_encoder.inverse_transform(model.predict(X_test).astype(int))
                                if hasattr(model, '_label_encoder') and hasattr(model._label_encoder, 'inverse_transform')
                                else model.predict(X_test),
                                average='weighted',
                                zero_division=0,
                            )
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
                            'accuracy': accuracy_score(
                                y_test,
                                model._label_encoder.inverse_transform(model.predict(X_test).astype(int))
                                if hasattr(model, '_label_encoder') and hasattr(model._label_encoder, 'inverse_transform')
                                else model.predict(X_test)
                            ),
                            'f1_score': f1_score(
                                y_test,
                                model._label_encoder.inverse_transform(model.predict(X_test).astype(int))
                                if hasattr(model, '_label_encoder') and hasattr(model._label_encoder, 'inverse_transform')
                                else model.predict(X_test),
                                average='weighted'
                            )
                        }
                        for name, model in models.items()
                    },
                    'optimized_weights': {name: 1.0 / len(models) for name in models.keys()}
                }
            
            logger.info(f"{symbol} (ensemble {len(models)} models): "
                       f"Acc={metrics['accuracy']:.1%}, F1={metrics['f1_score']:.1%} ({len(y_test)} test samples)")
            
            result = {
                'models': models,
                'scaler': scaler,
                'ensemble_classes': ensemble_classes,
                'feature_names': feature_names,
                'metrics': metrics,
                'n_samples': len(X),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'embargo_samples': embargo_size,
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
