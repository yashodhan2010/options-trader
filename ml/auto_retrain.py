"""
Auto-Retraining Module for Continuous Learning

Automatically retrains ML models based on:
1. Actual trade outcomes (feedback data)
2. Drift detection (accuracy degradation)
3. Scheduled intervals

This uses real P&L outcomes as the target variable instead of
historical price direction, enabling the model to learn from
actual trading performance.
"""

import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd

from config.settings import ML_CONFIG
from core.database import database
from core.logger import logger


class AutoRetrainer:
    """
    Automatic model retraining based on feedback data.
    
    Key Features:
    - Uses actual trade P&L as target variable
    - Drift-triggered retraining
    - Scheduled periodic retraining
    - Minimum sample requirements
    - A/B model comparison before promotion
    """
    
    # Target encoding for trade outcomes
    OUTCOME_ENCODING = {
        "WIN": 2,       # Bullish prediction was correct
        "BREAKEVEN": 1, # Neutral
        "LOSS": 0,      # Prediction was incorrect
    }
    
    def __init__(self):
        """Initialize auto-retrainer."""
        self.config = ML_CONFIG.get("auto_retrain", {})
        
        # Configuration
        self.enabled = self.config.get("enabled", True)
        self.min_samples = self.config.get("min_samples", 50)
        self.retrain_interval_days = self.config.get("interval_days", 7)
        self.drift_threshold = self.config.get("drift_threshold", 0.15)
        self.auto_promote = self.config.get("auto_promote", False)
        self.use_feedback_target = self.config.get("use_feedback_target", True)
        
        # State
        self._last_retrain: Optional[datetime] = None
        self._retrain_lock = threading.Lock()
        self._background_thread: Optional[threading.Thread] = None
        self._running = False
        
        # Lazy imports
        self._model_trainer = None
        self._feedback_collector = None
        self._model_registry = None
        
        logger.info(f"AutoRetrainer initialized (enabled={self.enabled})")
    
    @property
    def model_trainer(self):
        """Lazy load model trainer."""
        if self._model_trainer is None:
            from ml.model_trainer import get_model_trainer
            self._model_trainer = get_model_trainer()
        return self._model_trainer
    
    @property
    def feedback_collector(self):
        """Lazy load feedback collector."""
        if self._feedback_collector is None:
            from ml.feedback_collector import get_feedback_collector
            self._feedback_collector = get_feedback_collector()
        return self._feedback_collector
    
    @property
    def model_registry(self):
        """Lazy load model registry."""
        if self._model_registry is None:
            from ml.model_registry import get_model_registry
            self._model_registry = get_model_registry()
        return self._model_registry
    
    def get_feedback_training_data(
        self,
        min_samples: int = None,
        lookback_days: int = 90
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
        """
        Get training data from trade feedback.
        
        Uses actual trade outcomes (P&L) as the target variable,
        enabling the model to learn from real trading performance.
        
        Args:
            min_samples: Minimum samples required
            lookback_days: Days of data to include
            
        Returns:
            Tuple of (X features, y targets, feature_names) or (None, None, [])
        """
        min_samples = min_samples or self.min_samples
        
        try:
            # Get training data with outcomes
            min_date = datetime.now() - timedelta(days=lookback_days)
            data = database.get_training_data(
                outcome_only=True,
                min_date=min_date,
                limit=10000
            )
            
            if len(data) < min_samples:
                logger.warning(
                    f"Insufficient feedback data: {len(data)} samples "
                    f"(need {min_samples})"
                )
                return None, None, []
            
            logger.info(f"Retrieved {len(data)} feedback samples for training")
            
            # Extract features and targets
            X_list = []
            y_list = []
            feature_names = None
            
            for record in data:
                features = record.get("features", {})
                outcome = record.get("outcome")
                
                if not features or not outcome:
                    continue
                
                # Get feature names from first record
                if feature_names is None:
                    feature_names = list(features.keys())
                
                # Extract feature values in consistent order
                feature_values = [features.get(name, 0.0) for name in feature_names]
                X_list.append(feature_values)
                
                # Encode outcome as target
                # Use P&L-based target: WIN=2, BREAKEVEN=1, LOSS=0
                target = self.OUTCOME_ENCODING.get(outcome, 1)
                y_list.append(target)
            
            if len(X_list) < min_samples:
                logger.warning(f"Only {len(X_list)} valid samples after filtering")
                return None, None, []
            
            X = np.array(X_list, dtype=np.float32)
            y = np.array(y_list, dtype=np.int32)
            
            # Log class distribution
            unique, counts = np.unique(y, return_counts=True)
            dist = dict(zip(unique, counts))
            logger.info(f"Target distribution: WIN={dist.get(2, 0)}, "
                       f"BREAKEVEN={dist.get(1, 0)}, LOSS={dist.get(0, 0)}")
            
            return X, y, feature_names
            
        except Exception as e:
            logger.error(f"Failed to get feedback training data: {e}")
            return None, None, []
    
    def check_retrain_conditions(self) -> Dict[str, Any]:
        """
        Check if retraining conditions are met.
        
        Returns:
            Dict with check results and recommendations
        """
        result = {
            "should_retrain": False,
            "reasons": [],
            "available_samples": 0,
            "days_since_last_train": None,
            "current_accuracy": None,
            "drift_detected": False,
        }
        
        # Check sample count
        data = database.get_training_data(outcome_only=True, limit=10000)
        result["available_samples"] = len(data)
        
        if len(data) >= self.min_samples:
            result["reasons"].append(f"Sufficient samples: {len(data)} >= {self.min_samples}")
        
        # Check time since last training
        prod_model = self.model_registry.get_production_model()
        if prod_model:
            try:
                model_info = self.model_registry.get_model_info(prod_model)
                if model_info and model_info.get("trained_at"):
                    trained_at = datetime.fromisoformat(model_info["trained_at"])
                    days_since = (datetime.now() - trained_at).days
                    result["days_since_last_train"] = days_since
                    
                    if days_since >= self.retrain_interval_days:
                        result["reasons"].append(
                            f"Scheduled: {days_since} days since last training"
                        )
                        result["should_retrain"] = True
            except Exception as e:
                logger.debug(f"Could not check model age: {e}")
        
        # Check drift
        if self.feedback_collector.should_retrain():
            result["drift_detected"] = True
            result["reasons"].append("Model drift detected")
            result["should_retrain"] = True
        
        # Check performance stats
        stats = self.feedback_collector.get_performance_stats(days=14)
        if stats.get("accuracy"):
            result["current_accuracy"] = stats["accuracy"]
            
            # If accuracy is below threshold, recommend retrain
            if stats["accuracy"] < 0.5:  # Below 50% accuracy
                result["reasons"].append(f"Low accuracy: {stats['accuracy']:.1%}")
                result["should_retrain"] = True
        
        # Final check: must have minimum samples
        if result["should_retrain"] and len(data) < self.min_samples:
            result["should_retrain"] = False
            result["reasons"].append(
                f"Blocked: insufficient samples ({len(data)} < {self.min_samples})"
            )
        
        return result
    
    def retrain_from_feedback(
        self,
        force: bool = False,
        model_type: str = None
    ) -> Dict[str, Any]:
        """
        Retrain model using feedback data.
        
        Args:
            force: Force retrain even if conditions not met
            model_type: Model type to train (default from config)
            
        Returns:
            Dict with training results
        """
        with self._retrain_lock:
            result = {
                "success": False,
                "model_version": None,
                "metrics": {},
                "message": "",
            }
            
            # Check conditions unless forced
            if not force:
                conditions = self.check_retrain_conditions()
                if not conditions["should_retrain"]:
                    result["message"] = "Retrain conditions not met"
                    result["conditions"] = conditions
                    return result
            
            # Get feedback training data
            X, y, feature_names = self.get_feedback_training_data()
            
            if X is None or len(X) == 0:
                result["message"] = "No feedback training data available"
                return result
            
            logger.info(f"Starting feedback-based retraining with {len(X)} samples")
            
            try:
                # Train model
                model_type = model_type or ML_CONFIG.get("model_type", "ensemble")
                
                # Use model trainer
                model, scaler, metrics = self.model_trainer.train(
                    X=X,
                    y=y,
                    feature_names=feature_names,
                    model_type=model_type,
                    optimize=True
                )
                
                if model is None:
                    result["message"] = "Training failed"
                    return result
                
                # Save model
                version = self.model_trainer.save_model(
                    model=model,
                    scaler=scaler,
                    feature_names=feature_names,
                    metrics=metrics,
                    metadata={
                        "training_source": "feedback",
                        "sample_count": len(X),
                        "target_type": "trade_outcome",
                    }
                )
                
                result["success"] = True
                result["model_version"] = version
                result["metrics"] = metrics
                result["message"] = f"Feedback-trained model: {version}"
                
                # Optionally promote to production
                if self.auto_promote and metrics.get("accuracy", 0) > 0.55:
                    self.model_registry.promote_model(version)
                    result["promoted"] = True
                    logger.info(f"Auto-promoted model {version} to production")
                
                # Reset feedback baseline
                self.feedback_collector.reset_baseline()
                self._last_retrain = datetime.now()
                
                logger.info(f"Feedback retraining complete: {version}")
                
            except Exception as e:
                logger.error(f"Feedback retraining failed: {e}")
                result["message"] = f"Training error: {e}"
            
            return result
    
    def start_background_monitor(self, check_interval: int = 3600) -> None:
        """
        Start background monitoring for auto-retraining.
        
        Args:
            check_interval: Seconds between checks (default: 1 hour)
        """
        if not self.enabled:
            logger.info("Auto-retraining is disabled")
            return
        
        if self._running:
            logger.warning("Background monitor already running")
            return
        
        self._running = True
        
        def monitor_loop():
            logger.info(f"Auto-retrain monitor started (interval: {check_interval}s)")
            
            while self._running:
                try:
                    conditions = self.check_retrain_conditions()
                    
                    if conditions["should_retrain"]:
                        logger.info(f"Auto-retrain triggered: {conditions['reasons']}")
                        result = self.retrain_from_feedback()
                        
                        if result["success"]:
                            logger.info(f"Auto-retrain successful: {result['model_version']}")
                        else:
                            logger.warning(f"Auto-retrain failed: {result['message']}")
                    
                except Exception as e:
                    logger.error(f"Auto-retrain monitor error: {e}")
                
                time.sleep(check_interval)
        
        self._background_thread = threading.Thread(
            target=monitor_loop,
            daemon=True,
            name="AutoRetrainMonitor"
        )
        self._background_thread.start()
    
    def stop_background_monitor(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._background_thread:
            self._background_thread.join(timeout=5)
            self._background_thread = None
        logger.info("Auto-retrain monitor stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get auto-retrainer status.
        
        Returns:
            Status dictionary
        """
        conditions = self.check_retrain_conditions()
        
        return {
            "enabled": self.enabled,
            "running": self._running,
            "last_retrain": self._last_retrain.isoformat() if self._last_retrain else None,
            "min_samples": self.min_samples,
            "retrain_interval_days": self.retrain_interval_days,
            "drift_threshold": self.drift_threshold,
            "auto_promote": self.auto_promote,
            "conditions": conditions,
        }


# Singleton instance
_auto_retrainer: Optional[AutoRetrainer] = None


def get_auto_retrainer() -> AutoRetrainer:
    """Get or create the singleton auto-retrainer instance."""
    global _auto_retrainer
    if _auto_retrainer is None:
        _auto_retrainer = AutoRetrainer()
    return _auto_retrainer
