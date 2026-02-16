"""
ML Model Trainer with Optuna Hyperparameter Optimization

Trains XGBoost, LightGBM, and Random Forest models for options trading.
Supports walk-forward validation, ensemble creation, and MLflow tracking.
"""

import json
import os
import pickle
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd

from config.settings import ML_CONFIG, ML_MODELS_DIR
from core.database import database
from core.logger import logger

# ML library imports with fallbacks
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not installed. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not installed. Install with: pip install lightgbm")

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
    # Suppress noisy per-trial Optuna logs — we log our own summaries
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not installed. Install with: pip install optuna")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report
    )
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not installed. Install with: pip install scikit-learn")


class ModelTrainer:
    """
    Train ML models for options trading predictions.
    
    Features:
    - XGBoost, LightGBM, Random Forest classifiers
    - Optuna hyperparameter optimization
    - Walk-forward validation for time series
    - Ensemble model creation
    - MLflow experiment tracking
    - Model versioning and persistence
    """
    
    def __init__(self, mlflow_tracker=None):
        """
        Initialize model trainer.
        
        Args:
            mlflow_tracker: Optional MLflowTracker instance
        """
        self.mlflow_tracker = mlflow_tracker
        self.model_path = Path(ML_CONFIG.get("model_path", ML_MODELS_DIR))
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        self.optuna_trials = ML_CONFIG.get("optuna_trials", 50)
        self.optuna_timeout = ML_CONFIG.get("optuna_timeout", 3600)
        self.validation_split = ML_CONFIG.get("validation_split", 0.2)
        self.walk_forward_splits = ML_CONFIG.get("walk_forward_splits", 5)
        
        # Parallel training config
        self.parallel_training = ML_CONFIG.get("parallel_training", True)
        total_cores = os.cpu_count() or 4
        self.n_cores = total_cores
        # When training 3 models in parallel, each gets cores/3
        self.cores_per_model = max(1, total_cores // 3)
        
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.feature_names: List[str] = []
        
        logger.info(f"ModelTrainer initialized. Cores: {total_cores}, parallel: {self.parallel_training}, model_path: {self.model_path}")
    
    def train_direction_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str] = None,
        model_type: str = "xgboost",
        optimize: bool = True
    ) -> Tuple[Any, Dict[str, float], str]:
        """
        Train a direction prediction model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (1=bullish, 0=neutral, -1=bearish) or binary
            feature_names: List of feature names
            model_type: 'xgboost', 'lightgbm', 'rf', or 'ensemble'
            optimize: Whether to use Optuna optimization
            
        Returns:
            Tuple of (trained_model, metrics_dict, model_version)
        """
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        # Generate model version
        model_version = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Check minimum sample size
        min_samples = 20  # Minimum samples for reliable training
        if len(X) < min_samples:
            logger.warning(f"Insufficient samples ({len(X)}) for training. Minimum required: {min_samples}")
            raise ValueError(f"Insufficient training data: {len(X)} samples, need at least {min_samples}")
        
        # Scale features
        if self.scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        
        # Convert labels if needed (ensure non-negative for some classifiers)
        y_adjusted = y.copy()
        if y.min() < 0:
            y_adjusted = y + 1  # Convert -1,0,1 to 0,1,2
        
        # Check class balance - need at least 2 classes with some samples each
        unique_classes, class_counts = np.unique(y_adjusted, return_counts=True)
        if len(unique_classes) < 2:
            logger.warning(f"Only one class found in data. Classes: {unique_classes}, counts: {class_counts}")
            raise ValueError(f"Imbalanced data: only class {unique_classes[0]} present. Need data with both up and down days.")
        
        min_class_count = min(class_counts)
        if min_class_count < 5:
            logger.warning(f"Class imbalance: {dict(zip(unique_classes, class_counts))}. Minority class has only {min_class_count} samples.")
        
        # Split data (time-series aware)
        split_idx = int(len(X) * (1 - self.validation_split))
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y_adjusted[:split_idx], y_adjusted[split_idx:]
        
        logger.info(f"Training {model_type} model: {len(X_train)} train, {len(X_val)} val samples")
        
        # Start MLflow run
        if self.mlflow_tracker:
            self.mlflow_tracker.start_training_run(
                run_name=model_version,
                model_type=model_type,
                hyperparameters={"samples": len(X), "features": len(self.feature_names)}
            )
        
        try:
            if model_type == "ensemble":
                model, metrics = self._train_ensemble(X_train, y_train, X_val, y_val, optimize)
            elif model_type == "xgboost":
                model, metrics = self._train_xgboost(X_train, y_train, X_val, y_val, optimize)
            elif model_type == "lightgbm":
                model, metrics = self._train_lightgbm(X_train, y_train, X_val, y_val, optimize)
            elif model_type == "rf":
                model, metrics = self._train_random_forest(X_train, y_train, X_val, y_val, optimize)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Log metrics to MLflow
            if self.mlflow_tracker:
                self.mlflow_tracker.log_validation_metrics(
                    accuracy=metrics.get("accuracy", 0),
                    precision=metrics.get("precision", 0),
                    recall=metrics.get("recall", 0),
                    f1=metrics.get("f1_score", 0),
                    auc_roc=metrics.get("auc_roc", 0)
                )
            
            # Get and log feature importance
            feature_importance = self._get_feature_importance(model, model_type)
            if feature_importance and self.mlflow_tracker:
                self.mlflow_tracker.log_feature_importance(feature_importance)
            
            # Save model
            self._save_model(model, model_version, model_type, metrics)
            
            # Log model to MLflow
            if self.mlflow_tracker:
                self.mlflow_tracker.log_model(model, model_version, model_type, register=True)
                self.mlflow_tracker.end_run()
            
            # Save to database
            database.save_model_performance(
                model_version=model_version,
                model_type=model_type,
                training_samples=len(X),
                metrics=metrics
            )
            
            logger.info(f"Model {model_version} trained. Accuracy: {metrics.get('accuracy', 0):.4f}")
            
            return model, metrics, model_version
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            if self.mlflow_tracker:
                self.mlflow_tracker.end_run(status="FAILED")
            raise
    
    def train_with_params(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        model_type: str,
        params: Dict[str, Any],
        symbol: str = None,
        config_name: str = None
    ) -> Tuple[Any, Dict[str, float], str]:
        """
        Train a model with specific pre-defined parameters (no optimization).
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: List of feature names
            model_type: 'xgboost', 'lightgbm', or 'rf'
            params: Pre-defined hyperparameters
            symbol: Symbol being trained (for versioning)
            config_name: Configuration name (for versioning)
            
        Returns:
            Tuple of (trained_model, metrics_dict, model_version)
        """
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        # Generate model version with symbol and config
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if symbol and config_name:
            model_version = f"{symbol}_{config_name}_{timestamp}"
        else:
            model_version = f"{model_type}_custom_{timestamp}"
        
        # Scale features
        if self.scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        
        # Convert labels if needed
        y_adjusted = y.copy()
        if y.min() < 0:
            y_adjusted = y + 1
        
        # Split data
        split_idx = int(len(X) * (1 - self.validation_split))
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y_adjusted[:split_idx], y_adjusted[split_idx:]
        
        n_classes = len(np.unique(y_train))
        
        logger.info(f"Training {model_type} with custom params: {len(X_train)} train, {len(X_val)} val samples")
        
        try:
            if model_type == "rf":
                model = RandomForestClassifier(
                    random_state=42,
                    n_jobs=-1,
                    **params
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                metrics = self._calculate_metrics(y_val, y_pred, n_classes)
                
            elif model_type == "xgboost":
                if not XGBOOST_AVAILABLE:
                    raise ImportError("XGBoost not available")
                
                xgb_params = params.copy()
                xgb_params["objective"] = "multi:softmax" if n_classes > 2 else "binary:logistic"
                xgb_params["num_class"] = n_classes if n_classes > 2 else None
                xgb_params["random_state"] = 42
                xgb_params["n_jobs"] = -1
                
                # Remove None values
                xgb_params = {k: v for k, v in xgb_params.items() if v is not None}
                
                model = xgb.XGBClassifier(**xgb_params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                y_pred = model.predict(X_val)
                metrics = self._calculate_metrics(y_val, y_pred, n_classes)
                
            elif model_type == "lightgbm":
                if not LIGHTGBM_AVAILABLE:
                    raise ImportError("LightGBM not available")
                
                lgb_params = params.copy()
                lgb_params["objective"] = "multiclass" if n_classes > 2 else "binary"
                lgb_params["num_class"] = n_classes if n_classes > 2 else None
                lgb_params["random_state"] = 42
                lgb_params["n_jobs"] = -1
                lgb_params["verbose"] = -1
                
                # Remove None values
                lgb_params = {k: v for k, v in lgb_params.items() if v is not None}
                
                model = lgb.LGBMClassifier(**lgb_params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
                y_pred = model.predict(X_val)
                metrics = self._calculate_metrics(y_val, y_pred, n_classes)
                
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            metrics["best_params"] = params
            metrics["config_name"] = config_name
            metrics["symbol"] = symbol
            
            # Get feature importance
            feature_importance = self._get_feature_importance(model, model_type)
            
            # Save model
            self._save_model(model, model_version, model_type, metrics)
            
            # Save to database
            database.save_model_performance(
                model_version=model_version,
                model_type=model_type,
                training_samples=len(X),
                metrics=metrics
            )
            
            logger.info(f"Model {model_version} trained with custom params. Accuracy: {metrics.get('accuracy', 0):.4f}")
            
            return model, metrics, model_version
            
        except Exception as e:
            logger.error(f"Training with params failed: {e}")
            raise
    
    def _train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        optimize: bool = True,
        n_jobs: int = -1
    ) -> Tuple[Any, Dict[str, float]]:
        """Train XGBoost classifier."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available")
        
        n_classes = len(np.unique(y_train))
        
        if optimize and OPTUNA_AVAILABLE:
            best_params = self._optimize_xgboost(X_train, y_train, n_classes, n_jobs=n_jobs)
        else:
            best_params = {
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 100,
                "min_child_weight": 1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "gamma": 0,
                "reg_alpha": 0,
                "reg_lambda": 1,
            }
        
        # Determine objective
        if n_classes == 2:
            objective = "binary:logistic"
            eval_metric = "logloss"
        else:
            objective = "multi:softprob"
            eval_metric = "mlogloss"
        
        model = xgb.XGBClassifier(
            objective=objective,
            eval_metric=eval_metric,
            use_label_encoder=False,
            random_state=42,
            n_jobs=n_jobs,
            **best_params
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Calculate metrics
        y_pred = model.predict(X_val)
        metrics = self._calculate_metrics(y_val, y_pred, n_classes)
        metrics["best_params"] = best_params
        
        return model, metrics
    
    def _train_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        optimize: bool = True,
        n_jobs: int = -1
    ) -> Tuple[Any, Dict[str, float]]:
        """Train LightGBM classifier."""
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available")
        
        n_classes = len(np.unique(y_train))
        
        if optimize and OPTUNA_AVAILABLE:
            best_params = self._optimize_lightgbm(X_train, y_train, n_classes, n_jobs=n_jobs)
        else:
            best_params = {
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 100,
                "num_leaves": 31,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0,
                "reg_lambda": 1,
            }
        
        model = lgb.LGBMClassifier(
            objective="multiclass" if n_classes > 2 else "binary",
            random_state=42,
            verbose=-1,
            n_jobs=n_jobs,
            **best_params
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
        )
        
        y_pred = model.predict(X_val)
        metrics = self._calculate_metrics(y_val, y_pred, n_classes)
        metrics["best_params"] = best_params
        
        return model, metrics
    
    def _train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        optimize: bool = True,
        n_jobs: int = -1
    ) -> Tuple[Any, Dict[str, float]]:
        """Train Random Forest classifier."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not available")
        
        n_classes = len(np.unique(y_train))
        
        if optimize and OPTUNA_AVAILABLE:
            best_params = self._optimize_random_forest(X_train, y_train, n_jobs=n_jobs)
        else:
            best_params = {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
            }
        
        model = RandomForestClassifier(
            random_state=42,
            n_jobs=n_jobs,
            **best_params
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        metrics = self._calculate_metrics(y_val, y_pred, n_classes)
        metrics["best_params"] = best_params
        
        return model, metrics
    
    def _train_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        optimize: bool = True
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Train ensemble of XGBoost, LightGBM, and Random Forest with Optuna-tuned weights.
        
        When parallel_training is enabled, trains all 3 sub-models concurrently
        using ThreadPoolExecutor, allocating cores/3 threads per model.
        XGB/LGB/RF all release the GIL during native C++ computation,
        so thread-level parallelism is effective.
        """
        ensemble = {}
        ensemble_metrics = {}
        
        # Build list of models to train
        models_to_train = []
        if XGBOOST_AVAILABLE:
            models_to_train.append("xgboost")
        if LIGHTGBM_AVAILABLE:
            models_to_train.append("lightgbm")
        if SKLEARN_AVAILABLE:
            models_to_train.append("random_forest")
        
        if self.parallel_training and len(models_to_train) > 1:
            # --- PARALLEL TRAINING ---
            logger.info(f"Training {len(models_to_train)} models in parallel "
                        f"({self.cores_per_model} cores each, {self.n_cores} total)...")
            
            # Temporarily limit per-model cores to avoid oversubscription
            saved_n_jobs = self.cores_per_model
            
            def _train_model(model_name):
                t0 = time.time()
                if model_name == "xgboost":
                    model, metrics = self._train_xgboost(
                        X_train, y_train, X_val, y_val, optimize,
                        n_jobs=saved_n_jobs
                    )
                elif model_name == "lightgbm":
                    model, metrics = self._train_lightgbm(
                        X_train, y_train, X_val, y_val, optimize,
                        n_jobs=saved_n_jobs
                    )
                elif model_name == "random_forest":
                    model, metrics = self._train_random_forest(
                        X_train, y_train, X_val, y_val, optimize,
                        n_jobs=saved_n_jobs
                    )
                elapsed = time.time() - t0
                logger.info(f"  {model_name} done in {elapsed:.1f}s "
                            f"(acc={metrics.get('accuracy', 0):.1%})")
                return model_name, model, metrics
            
            t_start = time.time()
            with ThreadPoolExecutor(max_workers=len(models_to_train)) as executor:
                futures = {executor.submit(_train_model, name): name
                           for name in models_to_train}
                for future in as_completed(futures):
                    name, model, metrics = future.result()
                    ensemble[name] = model
                    ensemble_metrics[name] = metrics
            
            total_time = time.time() - t_start
            logger.info(f"All {len(models_to_train)} models trained in {total_time:.1f}s (parallel)")
        
        else:
            # --- SEQUENTIAL TRAINING ---
            for name in models_to_train:
                logger.info(f"Training {name} for ensemble...")
                t0 = time.time()
                if name == "xgboost":
                    ensemble[name], m = self._train_xgboost(
                        X_train, y_train, X_val, y_val, optimize
                    )
                elif name == "lightgbm":
                    ensemble[name], m = self._train_lightgbm(
                        X_train, y_train, X_val, y_val, optimize
                    )
                elif name == "random_forest":
                    ensemble[name], m = self._train_random_forest(
                        X_train, y_train, X_val, y_val, optimize
                    )
                ensemble_metrics[name] = m
                logger.info(f"  {name} done in {time.time()-t0:.1f}s")
        
        # Optimize ensemble weights with Optuna (or use config defaults)
        model_names = [n for n in ensemble if n not in ["weights", "scaler"]]
        n_classes = len(np.unique(y_train))
        
        if optimize and OPTUNA_AVAILABLE and len(model_names) >= 2:
            logger.info("Optimizing ensemble weights with Optuna...")
            weights = self._optimize_ensemble_weights(
                ensemble, model_names, X_val, y_val, n_classes
            )
        else:
            weights = ML_CONFIG.get("ensemble_weights", {
                "xgboost": 0.5,
                "lightgbm": 0.3,
                "random_forest": 0.2,
            })
        
        ensemble["weights"] = weights
        ensemble["scaler"] = self.scaler
        
        # Calculate ensemble predictions with optimized weights
        ensemble_probs = np.zeros((len(X_val), n_classes))
        total_weight = 0
        
        for name, model in ensemble.items():
            if name in ["weights", "scaler"]:
                continue
            
            weight = weights.get(name, 0.33)
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_val)
                ensemble_probs += weight * probs
                total_weight += weight
        
        if total_weight > 0:
            ensemble_probs /= total_weight
        
        ensemble_pred = np.argmax(ensemble_probs, axis=1)
        
        # Calculate ensemble metrics
        metrics = self._calculate_metrics(y_val, ensemble_pred, n_classes)
        metrics["individual_metrics"] = ensemble_metrics
        metrics["optimized_weights"] = weights
        
        return ensemble, metrics
    
    def _optimize_ensemble_weights(
        self,
        ensemble: Dict[str, Any],
        model_names: List[str],
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_classes: int
    ) -> Dict[str, float]:
        """Use Optuna to find optimal ensemble weights.
        
        Optimizes blending weights for each sub-model to maximize
        ensemble accuracy on the validation set.
        """
        # Pre-compute all model probabilities (avoid recalculating each trial)
        model_probs = {}
        for name in model_names:
            model = ensemble[name]
            if hasattr(model, "predict_proba"):
                model_probs[name] = model.predict_proba(X_val)
        
        if len(model_probs) < 2:
            logger.info("Only one model available, skipping weight optimization")
            return {name: 1.0 for name in model_probs}
        
        def objective(trial):
            # Suggest raw weights and normalize to sum to 1
            raw_weights = {}
            for name in model_probs:
                raw_weights[name] = trial.suggest_float(f"w_{name}", 0.05, 1.0)
            
            total = sum(raw_weights.values())
            weights = {name: w / total for name, w in raw_weights.items()}
            
            # Blend predictions
            blended = np.zeros((len(X_val), n_classes))
            for name, w in weights.items():
                blended += w * model_probs[name]
            
            preds = np.argmax(blended, axis=1)
            
            # Use F1 (weighted) as objective — better than accuracy for imbalanced classes
            avg = "binary" if n_classes == 2 else "weighted"
            return f1_score(y_val, preds, average=avg, zero_division=0)
        
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42)
        )
        
        # Weight optimization is fast (no model training), so run many trials
        study.optimize(objective, n_trials=200, timeout=30, show_progress_bar=False)
        
        # Extract and normalize best weights
        best = study.best_params
        raw = {name: best[f"w_{name}"] for name in model_probs}
        total = sum(raw.values())
        weights = {name: round(w / total, 3) for name, w in raw.items()}
        
        logger.info(f"Optimized ensemble weights: {weights} (F1={study.best_value:.4f})")
        return weights
    
    def _optimize_xgboost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_classes: int,
        n_jobs: int = -1
    ) -> Dict[str, Any]:
        """Optuna optimization for XGBoost with pruning."""
        use_pruning = ML_CONFIG.get("optuna_pruning", True)
        
        def objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 2),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 2),
            }
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model = xgb.XGBClassifier(
                    objective="binary:logistic" if n_classes == 2 else "multi:softprob",
                    use_label_encoder=False,
                    random_state=42,
                    n_jobs=n_jobs,
                    **params
                )
                
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                y_pred = model.predict(X_val)
                scores.append(accuracy_score(y_val, y_pred))
                
                # Report intermediate score for pruning
                if use_pruning:
                    trial.report(np.mean(scores), fold_idx)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
            
            return np.mean(scores)
        
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=1) if use_pruning else optuna.pruners.NopPruner()
        
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=pruner
        )
        
        study.optimize(
            objective,
            n_trials=self.optuna_trials,
            timeout=self.optuna_timeout,
            show_progress_bar=not self.parallel_training
        )
        
        n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        logger.info(f"XGBoost best params: {study.best_params} (pruned {n_pruned}/{len(study.trials)} trials, "
                    f"best={study.best_value:.4f}, n_jobs={n_jobs})")
        return study.best_params
    
    def _optimize_lightgbm(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_classes: int,
        n_jobs: int = -1
    ) -> Dict[str, Any]:
        """Optuna optimization for LightGBM with pruning."""
        use_pruning = ML_CONFIG.get("optuna_pruning", True)
        
        def objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 2),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 2),
            }
            
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model = lgb.LGBMClassifier(
                    objective="multiclass" if n_classes > 2 else "binary",
                    random_state=42,
                    verbose=-1,
                    n_jobs=n_jobs,
                    **params
                )
                
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
                y_pred = model.predict(X_val)
                scores.append(accuracy_score(y_val, y_pred))
                
                # Report intermediate score for pruning
                if use_pruning:
                    trial.report(np.mean(scores), fold_idx)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
            
            return np.mean(scores)
        
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=1) if use_pruning else optuna.pruners.NopPruner()
        
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=pruner
        )
        
        study.optimize(
            objective,
            n_trials=self.optuna_trials,
            timeout=self.optuna_timeout,
            show_progress_bar=not self.parallel_training
        )
        
        n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        logger.info(f"LightGBM best params: {study.best_params} (pruned {n_pruned}/{len(study.trials)} trials, "
                    f"best={study.best_value:.4f}, n_jobs={n_jobs})")
        return study.best_params
    
    def _optimize_random_forest(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_jobs: int = -1
    ) -> Dict[str, Any]:
        """Optuna optimization for Random Forest with pruning."""
        use_pruning = ML_CONFIG.get("optuna_pruning", True)
        
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            }
            
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model = RandomForestClassifier(random_state=42, n_jobs=n_jobs, **params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                scores.append(accuracy_score(y_val, y_pred))
                
                # Report intermediate score for pruning
                if use_pruning:
                    trial.report(np.mean(scores), fold_idx)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
            
            return np.mean(scores)
        
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=1) if use_pruning else optuna.pruners.NopPruner()
        
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=pruner
        )
        
        study.optimize(
            objective,
            n_trials=min(self.optuna_trials, 30),  # RF is slower
            timeout=self.optuna_timeout,
            show_progress_bar=not self.parallel_training
        )
        
        logger.info(f"Random Forest best params: {study.best_params} (pruned {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}/{len(study.trials)} trials, "
                    f"best={study.best_value:.4f}, n_jobs={n_jobs})")
        return study.best_params
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_classes: int
    ) -> Dict[str, float]:
        """Calculate classification metrics."""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
        }
        
        average = "binary" if n_classes == 2 else "weighted"
        
        try:
            metrics["precision"] = precision_score(y_true, y_pred, average=average, zero_division=0)
            metrics["recall"] = recall_score(y_true, y_pred, average=average, zero_division=0)
            metrics["f1_score"] = f1_score(y_true, y_pred, average=average, zero_division=0)
        except Exception as e:
            logger.warning(f"Error calculating precision/recall: {e}")
            metrics["precision"] = 0
            metrics["recall"] = 0
            metrics["f1_score"] = 0
        
        # AUC-ROC (only for binary classification)
        if n_classes == 2:
            try:
                metrics["auc_roc"] = roc_auc_score(y_true, y_pred)
            except Exception:
                metrics["auc_roc"] = 0.5
        else:
            metrics["auc_roc"] = 0
        
        # Confusion matrix
        try:
            cm = confusion_matrix(y_true, y_pred)
            metrics["confusion_matrix"] = cm.tolist()
        except Exception:
            metrics["confusion_matrix"] = []
        
        return metrics
    
    def _get_feature_importance(
        self,
        model: Any,
        model_type: str
    ) -> Dict[str, float]:
        """Extract feature importance from model."""
        importance = {}
        
        try:
            if model_type == "ensemble" and isinstance(model, dict):
                # Average importance across ensemble
                for name, m in model.items():
                    if name in ["weights", "scaler"]:
                        continue
                    sub_importance = self._get_feature_importance(m, name)
                    for feat, imp in sub_importance.items():
                        importance[feat] = importance.get(feat, 0) + imp / 3
            
            elif model_type == "xgboost" and hasattr(model, "feature_importances_"):
                for i, imp in enumerate(model.feature_importances_):
                    if i < len(self.feature_names):
                        importance[self.feature_names[i]] = float(imp)
            
            elif model_type == "lightgbm" and hasattr(model, "feature_importances_"):
                for i, imp in enumerate(model.feature_importances_):
                    if i < len(self.feature_names):
                        importance[self.feature_names[i]] = float(imp)
            
            elif model_type in ["rf", "random_forest"] and hasattr(model, "feature_importances_"):
                for i, imp in enumerate(model.feature_importances_):
                    if i < len(self.feature_names):
                        importance[self.feature_names[i]] = float(imp)
                        
        except Exception as e:
            logger.warning(f"Error getting feature importance: {e}")
        
        return importance
    
    def _save_model(
        self,
        model: Any,
        model_version: str,
        model_type: str,
        metrics: Dict[str, float]
    ) -> Path:
        """Save model to disk."""
        model_dir = self.model_path / model_version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_file = model_dir / "model.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(model, f)
        
        # Save scaler
        if self.scaler:
            scaler_file = model_dir / "scaler.pkl"
            with open(scaler_file, "wb") as f:
                pickle.dump(self.scaler, f)
        
        # Save feature names
        features_file = model_dir / "features.json"
        with open(features_file, "w") as f:
            json.dump(self.feature_names, f)
        
        # Save metadata
        metadata = {
            "model_version": model_version,
            "model_type": model_type,
            "trained_at": datetime.now().isoformat(),
            "metrics": metrics,
            "feature_count": len(self.feature_names),
        }
        metadata_file = model_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Model saved to {model_dir}")
        return model_dir
    
    def load_model(self, model_version: str) -> Tuple[Any, Any, List[str], Dict]:
        """
        Load a saved model.
        
        Args:
            model_version: Model version string
            
        Returns:
            Tuple of (model, scaler, feature_names, metadata)
        """
        model_dir = self.model_path / model_version
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model not found: {model_version}")
        
        # Load model
        model_file = model_dir / "model.pkl"
        with open(model_file, "rb") as f:
            model = pickle.load(f)
        
        # Load scaler
        scaler = None
        scaler_file = model_dir / "scaler.pkl"
        if scaler_file.exists():
            with open(scaler_file, "rb") as f:
                scaler = pickle.load(f)
        
        # Load feature names
        feature_names = []
        features_file = model_dir / "features.json"
        if features_file.exists():
            with open(features_file, "r") as f:
                feature_names = json.load(f)
        
        # Load metadata
        metadata = {}
        metadata_file = model_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
        
        return model, scaler, feature_names, metadata
    
    def get_latest_model(self, model_type: str = None) -> Optional[str]:
        """
        Get the latest model version.
        
        Args:
            model_type: Optional filter by model type
            
        Returns:
            Model version string or None
        """
        # Check database for active model
        active_models = database.get_model_performance(active_only=True)
        if active_models:
            return active_models[0].get("model_version")
        
        # Fall back to file system
        models = list(self.model_path.glob("*/metadata.json"))
        
        if not models:
            return None
        
        latest = None
        latest_time = None
        
        for metadata_file in models:
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                
                if model_type and metadata.get("model_type") != model_type:
                    continue
                
                trained_at = datetime.fromisoformat(metadata.get("trained_at", "2000-01-01"))
                
                if latest_time is None or trained_at > latest_time:
                    latest_time = trained_at
                    latest = metadata.get("model_version")
                    
            except Exception:
                continue
        
        return latest
    
    def walk_forward_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str = "xgboost",
        n_splits: int = None
    ) -> Dict[str, Any]:
        """
        Perform walk-forward validation.
        
        Args:
            X: Feature matrix
            y: Target labels
            model_type: Type of model to train
            n_splits: Number of splits (default from config)
            
        Returns:
            Dict with validation results
        """
        n_splits = n_splits or self.walk_forward_splits
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        results = {
            "splits": [],
            "mean_accuracy": 0,
            "mean_precision": 0,
            "mean_recall": 0,
            "mean_f1": 0,
        }
        
        for i, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(f"Walk-forward split {i+1}/{n_splits}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model without optimization for speed
            if model_type == "xgboost":
                model, metrics = self._train_xgboost(X_train, y_train, X_val, y_val, optimize=False)
            elif model_type == "lightgbm":
                model, metrics = self._train_lightgbm(X_train, y_train, X_val, y_val, optimize=False)
            else:
                model, metrics = self._train_random_forest(X_train, y_train, X_val, y_val, optimize=False)
            
            results["splits"].append({
                "split": i + 1,
                "train_size": len(X_train),
                "val_size": len(X_val),
                "metrics": metrics,
            })
        
        # Calculate means
        accuracies = [s["metrics"]["accuracy"] for s in results["splits"]]
        precisions = [s["metrics"]["precision"] for s in results["splits"]]
        recalls = [s["metrics"]["recall"] for s in results["splits"]]
        f1s = [s["metrics"]["f1_score"] for s in results["splits"]]
        
        results["mean_accuracy"] = np.mean(accuracies)
        results["mean_precision"] = np.mean(precisions)
        results["mean_recall"] = np.mean(recalls)
        results["mean_f1"] = np.mean(f1s)
        results["std_accuracy"] = np.std(accuracies)
        
        logger.info(f"Walk-forward validation: Mean accuracy = {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
        
        return results


# Singleton instance
_model_trainer: Optional[ModelTrainer] = None


def get_model_trainer() -> ModelTrainer:
    """Get or create the singleton model trainer instance."""
    global _model_trainer
    if _model_trainer is None:
        from ml.mlflow_tracker import get_mlflow_tracker
        _model_trainer = ModelTrainer(mlflow_tracker=get_mlflow_tracker())
    return _model_trainer
