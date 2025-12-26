"""
ML Predictor for Options Trading

Provides inference from trained ML models with ensemble support,
confidence blending, and risk guardrails integration.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np

from config.settings import ML_CONFIG
from core.logger import logger


@dataclass
class MLPrediction:
    """Container for ML prediction results."""
    direction: str                  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    confidence: float               # 0.0 to 1.0
    probabilities: Dict[str, float] # Probability for each class
    model_version: str
    model_type: str
    timestamp: datetime
    feature_importance: Optional[Dict[str, float]] = None
    raw_prediction: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "direction": self.direction,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "model_version": self.model_version,
            "model_type": self.model_type,
            "timestamp": self.timestamp.isoformat(),
            "raw_prediction": self.raw_prediction,
        }


class MLPredictor:
    """
    ML prediction engine for options trading.
    
    Features:
    - Load and cache trained models
    - Ensemble prediction with configurable weights
    - Confidence blending with rule-based signals
    - Integration with trading guardrails
    - Prediction caching for performance
    """
    
    # Direction mappings
    DIRECTION_MAP = {
        0: "BEARISH",
        1: "NEUTRAL", 
        2: "BULLISH",
    }
    
    REVERSE_DIRECTION_MAP = {
        "BEARISH": 0,
        "NEUTRAL": 1,
        "BULLISH": 2,
    }
    
    def __init__(self):
        """Initialize the ML predictor."""
        self.model = None
        self.scaler = None
        self.feature_names: List[str] = []
        self.model_version: Optional[str] = None
        self.model_type: Optional[str] = None
        self.model_timestamp: Optional[datetime] = None
        
        # Prediction cache
        self._prediction_cache: Dict[str, Tuple[MLPrediction, datetime]] = {}
        self.cache_ttl = ML_CONFIG.get("prediction_cache_seconds", 60)
        
        # Guardrails (lazy loaded)
        self._guardrails = None
        
        # Model trainer for loading
        self._model_trainer = None
        
        self.enabled = ML_CONFIG.get("enabled", True)
        self.confidence_weight = ML_CONFIG.get("confidence_weight", 0.5)
        
        logger.info("MLPredictor initialized")
    
    @property
    def guardrails(self):
        """Lazy load guardrails."""
        if self._guardrails is None:
            from ml.guardrails import get_guardrails
            self._guardrails = get_guardrails()
        return self._guardrails
    
    @property
    def model_trainer(self):
        """Lazy load model trainer."""
        if self._model_trainer is None:
            from ml.model_trainer import get_model_trainer
            self._model_trainer = get_model_trainer()
        return self._model_trainer
    
    def load_model(self, model_version: str = None) -> bool:
        """
        Load a trained model for prediction.
        
        Args:
            model_version: Specific version to load, or None for latest
            
        Returns:
            True if model loaded successfully
        """
        try:
            if model_version is None:
                model_version = self.model_trainer.get_latest_model()
            
            if model_version is None:
                logger.warning("No trained model found")
                return False
            
            self.model, self.scaler, self.feature_names, metadata = \
                self.model_trainer.load_model(model_version)
            
            self.model_version = model_version
            self.model_type = metadata.get("model_type", "unknown")
            
            trained_at = metadata.get("trained_at")
            if trained_at:
                self.model_timestamp = datetime.fromisoformat(trained_at)
            else:
                self.model_timestamp = datetime.now()
            
            # Clear cache when loading new model
            self._prediction_cache.clear()
            
            logger.info(f"Loaded model: {model_version} ({self.model_type})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def predict(
        self,
        features: Dict[str, float],
        underlying: str = None,
        use_cache: bool = True
    ) -> Optional[MLPrediction]:
        """
        Make a prediction from features.
        
        Args:
            features: Dictionary of feature values
            underlying: Optional underlying symbol for caching
            use_cache: Whether to use cached predictions
            
        Returns:
            MLPrediction or None if prediction fails
        """
        if not self.enabled:
            return None
        
        # Check if model is loaded
        if self.model is None:
            if not self.load_model():
                return None
        
        # Check cache
        cache_key = f"{underlying}_{hash(frozenset(features.items()))}"
        if use_cache and cache_key in self._prediction_cache:
            cached_pred, cached_time = self._prediction_cache[cache_key]
            if (datetime.now() - cached_time).total_seconds() < self.cache_ttl:
                return cached_pred
        
        try:
            # Convert features to array in correct order
            X = self._features_to_array(features)
            
            # Scale features
            if self.scaler is not None:
                X = self.scaler.transform(X.reshape(1, -1))
            else:
                X = X.reshape(1, -1)
            
            # Make prediction
            if self.model_type == "ensemble" and isinstance(self.model, dict):
                prediction = self._ensemble_predict(X)
            else:
                prediction = self._single_model_predict(X)
            
            # Cache prediction
            if use_cache:
                self._prediction_cache[cache_key] = (prediction, datetime.now())
            
            return prediction
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None
    
    def _features_to_array(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dictionary to numpy array in correct order."""
        return np.array([features.get(name, 0.0) for name in self.feature_names])
    
    def _single_model_predict(self, X: np.ndarray) -> MLPrediction:
        """Make prediction with single model."""
        # Get probabilities
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[0]
        else:
            # Fallback for models without predict_proba
            pred = self.model.predict(X)[0]
            probs = np.zeros(3)
            probs[int(pred)] = 1.0
        
        # Get predicted class
        raw_pred = int(np.argmax(probs))
        direction = self.DIRECTION_MAP.get(raw_pred, "NEUTRAL")
        
        # Calculate confidence from probability distribution
        confidence = float(np.max(probs))
        
        # Create probability dict
        prob_dict = {
            "BEARISH": float(probs[0]) if len(probs) > 0 else 0,
            "NEUTRAL": float(probs[1]) if len(probs) > 1 else 0,
            "BULLISH": float(probs[2]) if len(probs) > 2 else 0,
        }
        
        return MLPrediction(
            direction=direction,
            confidence=confidence,
            probabilities=prob_dict,
            model_version=self.model_version,
            model_type=self.model_type,
            timestamp=datetime.now(),
            raw_prediction=raw_pred,
        )
    
    def _ensemble_predict(self, X: np.ndarray) -> MLPrediction:
        """Make prediction with ensemble of models."""
        weights = self.model.get("weights", {})
        ensemble_probs = np.zeros(3)
        total_weight = 0
        
        for name, model in self.model.items():
            if name in ["weights", "scaler"]:
                continue
            
            weight = weights.get(name, 0.33)
            
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[0]
                
                # Handle binary vs multiclass
                if len(probs) == 2:
                    # Binary: convert to 3-class (assume neutral in middle)
                    ensemble_probs[0] += weight * probs[0] * 0.5
                    ensemble_probs[1] += weight * 0.5
                    ensemble_probs[2] += weight * probs[1] * 0.5
                else:
                    ensemble_probs += weight * probs
                
                total_weight += weight
        
        if total_weight > 0:
            ensemble_probs /= total_weight
        
        # Get prediction
        raw_pred = int(np.argmax(ensemble_probs))
        direction = self.DIRECTION_MAP.get(raw_pred, "NEUTRAL")
        confidence = float(np.max(ensemble_probs))
        
        prob_dict = {
            "BEARISH": float(ensemble_probs[0]),
            "NEUTRAL": float(ensemble_probs[1]),
            "BULLISH": float(ensemble_probs[2]),
        }
        
        return MLPrediction(
            direction=direction,
            confidence=confidence,
            probabilities=prob_dict,
            model_version=self.model_version,
            model_type="ensemble",
            timestamp=datetime.now(),
            raw_prediction=raw_pred,
        )
    
    def predict_with_guardrails(
        self,
        features: Dict[str, float],
        rule_confidence: float,
        underlying: str,
        current_positions: int = 0,
        is_paper_mode: bool = False
    ) -> Tuple[Optional[MLPrediction], float, bool]:
        """
        Make prediction with guardrails applied.
        
        Args:
            features: Feature dictionary
            rule_confidence: Rule-based confidence score
            underlying: Underlying symbol
            current_positions: Number of current open positions
            is_paper_mode: Whether in paper trading mode
            
        Returns:
            Tuple of (prediction, blended_confidence, should_trade)
        """
        # Get ML prediction
        prediction = self.predict(features, underlying)
        
        if prediction is None:
            # ML unavailable, use rule-based only
            return None, rule_confidence, rule_confidence >= ML_CONFIG.get("min_confidence_for_trade", 0.55)
        
        # Check guardrails
        guardrail_result = self.guardrails.check_entry_signal(
            ml_confidence=prediction.confidence,
            rule_confidence=rule_confidence,
            current_positions=current_positions,
            model_timestamp=self.model_timestamp,
            is_paper_mode=is_paper_mode
        )
        
        if not guardrail_result.passed:
            logger.info(f"Guardrail blocked: {guardrail_result.message}")
            return prediction, rule_confidence, False
        
        # Blend confidence
        blended, was_adjusted = self.guardrails.blend_confidence(
            ml_confidence=prediction.confidence,
            rule_confidence=rule_confidence,
            ml_weight=self.confidence_weight
        )
        
        if was_adjusted:
            logger.debug(f"Confidence adjusted: ML={prediction.confidence:.2f}, "
                        f"Rule={rule_confidence:.2f}, Blended={blended:.2f}")
        
        # Check minimum confidence threshold
        min_confidence = ML_CONFIG.get("min_confidence_for_trade", 0.55)
        should_trade = blended >= min_confidence
        
        return prediction, blended, should_trade
    
    def predict_exit(
        self,
        features: Dict[str, float],
        current_pnl_percent: float,
        stop_loss_percent: float,
        target_percent: float,
        time_in_trade_hours: float
    ) -> Tuple[str, float, str]:
        """
        Predict exit recommendation.
        
        Args:
            features: Current market features
            current_pnl_percent: Current P&L percentage
            stop_loss_percent: Stop loss threshold
            target_percent: Target profit threshold
            time_in_trade_hours: Hours in trade
            
        Returns:
            Tuple of (recommendation, confidence, reason)
            recommendation: 'HOLD', 'EXIT', 'TRAIL'
        """
        # Get direction prediction
        prediction = self.predict(features)
        
        if prediction is None:
            return "HOLD", 0.5, "ML unavailable"
        
        # Basic exit logic based on prediction
        recommendation = "HOLD"
        confidence = prediction.confidence
        reason = ""
        
        # Check if prediction direction opposes current position
        # This is a simplified heuristic - can be enhanced with a dedicated exit model
        
        if prediction.direction == "BEARISH" and current_pnl_percent > 0:
            # Trend may be reversing, consider taking profit
            if prediction.confidence > 0.7:
                recommendation = "EXIT"
                reason = "ML predicts bearish reversal with high confidence"
            elif prediction.confidence > 0.5:
                recommendation = "TRAIL"
                reason = "ML predicts possible bearish move, tighten stops"
        
        elif prediction.direction == "BULLISH" and current_pnl_percent > 0:
            # Trend continuing, hold or trail
            recommendation = "TRAIL" if current_pnl_percent > target_percent * 0.5 else "HOLD"
            reason = "ML predicts bullish continuation"
        
        elif prediction.direction == "NEUTRAL":
            # Sideways expected
            if current_pnl_percent > target_percent * 0.3:
                recommendation = "TRAIL"
                reason = "ML predicts sideways, lock in partial profits"
            elif time_in_trade_hours > 4:
                recommendation = "EXIT"
                reason = "ML predicts sideways, time decay risk"
        
        # Time-based adjustments
        if time_in_trade_hours > 24 and prediction.direction != "BULLISH":
            recommendation = "EXIT"
            reason = "Extended time in trade with non-bullish outlook"
        
        return recommendation, confidence, reason
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "version": self.model_version,
            "type": self.model_type,
            "trained_at": self.model_timestamp.isoformat() if self.model_timestamp else None,
            "feature_count": len(self.feature_names),
            "cache_size": len(self._prediction_cache),
        }
    
    def clear_cache(self) -> int:
        """Clear prediction cache. Returns number of cleared entries."""
        count = len(self._prediction_cache)
        self._prediction_cache.clear()
        return count
    
    def is_model_stale(self) -> bool:
        """Check if the loaded model is stale (too old)."""
        if self.model_timestamp is None:
            return True
        
        max_age_days = ML_CONFIG.get("guardrails", {}).get("max_model_age_days", 14)
        age = datetime.now() - self.model_timestamp
        
        return age > timedelta(days=max_age_days)
    
    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get prediction statistics from cache."""
        if not self._prediction_cache:
            return {"count": 0}
        
        predictions = [p for p, _ in self._prediction_cache.values()]
        
        directions = [p.direction for p in predictions]
        confidences = [p.confidence for p in predictions]
        
        return {
            "count": len(predictions),
            "direction_counts": {
                "BULLISH": directions.count("BULLISH"),
                "BEARISH": directions.count("BEARISH"),
                "NEUTRAL": directions.count("NEUTRAL"),
            },
            "avg_confidence": np.mean(confidences) if confidences else 0,
            "min_confidence": min(confidences) if confidences else 0,
            "max_confidence": max(confidences) if confidences else 0,
        }


# Singleton instance
_predictor: Optional[MLPredictor] = None


def get_predictor() -> MLPredictor:
    """Get or create the singleton predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = MLPredictor()
    return _predictor
