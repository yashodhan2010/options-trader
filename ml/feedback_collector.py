"""
Feedback Collector for Continuous Learning

Collects trade outcomes, logs features, detects model drift,
and triggers retraining when needed.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import json
import numpy as np

from config.settings import ML_CONFIG
from core.database import database
from core.logger import logger


class FeedbackCollector:
    """
    Collect feedback from trades for continuous ML improvement.
    
    Features:
    - Log features at trade entry and exit
    - Track prediction outcomes
    - Detect model drift
    - Trigger retraining when accuracy drops
    - Calculate rolling performance metrics
    """
    
    def __init__(self):
        """Initialize feedback collector."""
        self.config = ML_CONFIG.get("feedback", {})
        
        self.log_predictions = self.config.get("log_all_predictions", True)
        self.log_entry_features = self.config.get("log_features_at_entry", True)
        self.log_exit_features = self.config.get("log_features_at_exit", True)
        self.drift_detection = self.config.get("drift_detection_enabled", True)
        self.drift_threshold = self.config.get("drift_threshold", 0.1)
        
        # Performance tracking
        self._recent_predictions: List[Dict] = []
        self._baseline_accuracy: Optional[float] = None
        
        logger.info("FeedbackCollector initialized")
    
    def log_entry_features(
        self,
        execution_id: str,
        underlying: str,
        strategy_type: str,
        features: Dict[str, float],
        spot_price: float
    ) -> bool:
        """
        Log features at trade entry for training data.
        
        Args:
            execution_id: Trade execution ID
            underlying: Underlying symbol
            strategy_type: Strategy type
            features: Feature dictionary
            spot_price: Current spot price
            
        Returns:
            Success status
        """
        if not self.log_entry_features:
            return True
        
        try:
            return database.save_ml_features(
                execution_id=execution_id,
                underlying=underlying,
                strategy_type=strategy_type,
                features=features,
                snapshot_type="entry",
                spot_price=spot_price
            )
        except Exception as e:
            logger.error(f"Failed to log entry features: {e}")
            return False
    
    def log_exit_features(
        self,
        execution_id: str,
        underlying: str,
        strategy_type: str,
        features: Dict[str, float],
        spot_price: float
    ) -> bool:
        """
        Log features at trade exit.
        
        Args:
            execution_id: Trade execution ID
            underlying: Underlying symbol
            strategy_type: Strategy type
            features: Feature dictionary
            spot_price: Current spot price
            
        Returns:
            Success status
        """
        if not self.log_exit_features:
            return True
        
        try:
            return database.save_ml_features(
                execution_id=execution_id,
                underlying=underlying,
                strategy_type=strategy_type,
                features=features,
                snapshot_type="exit",
                spot_price=spot_price
            )
        except Exception as e:
            logger.error(f"Failed to log exit features: {e}")
            return False
    
    def log_prediction(
        self,
        execution_id: str,
        underlying: str,
        strategy_type: str,
        model_version: str,
        model_type: str,
        direction_prediction: str,
        ml_confidence: float,
        rule_confidence: float,
        blended_confidence: float,
        predicted_pnl_range: tuple = None,
        top_features: Dict[str, float] = None
    ) -> bool:
        """
        Log an ML prediction for tracking.
        
        Args:
            execution_id: Trade execution ID
            underlying: Underlying symbol
            strategy_type: Strategy type
            model_version: Model version string
            model_type: Type of model
            direction_prediction: Predicted direction
            ml_confidence: ML confidence score
            rule_confidence: Rule-based confidence
            blended_confidence: Final blended confidence
            predicted_pnl_range: Optional (low, high) P&L range
            top_features: Optional top feature importance
            
        Returns:
            Success status
        """
        if not self.log_predictions:
            return True
        
        try:
            # Track in memory for drift detection
            self._recent_predictions.append({
                "execution_id": execution_id,
                "prediction": direction_prediction,
                "confidence": ml_confidence,
                "timestamp": datetime.now(),
            })
            
            # Keep only last 100 predictions
            if len(self._recent_predictions) > 100:
                self._recent_predictions = self._recent_predictions[-100:]
            
            return database.save_ml_prediction(
                execution_id=execution_id,
                underlying=underlying,
                strategy_type=strategy_type,
                model_version=model_version,
                model_type=model_type,
                direction_prediction=direction_prediction,
                confidence_score=ml_confidence,
                rule_confidence=rule_confidence,
                blended_confidence=blended_confidence,
                predicted_pnl_range=predicted_pnl_range,
                top_features=top_features
            )
        except Exception as e:
            logger.error(f"Failed to log prediction: {e}")
            return False
    
    def log_outcome(
        self,
        execution_id: str,
        actual_pnl: float,
        actual_pnl_percent: float,
        trade_duration_seconds: int
    ) -> bool:
        """
        Log trade outcome for feedback loop.
        
        Args:
            execution_id: Trade execution ID
            actual_pnl: Realized P&L
            actual_pnl_percent: P&L as percentage
            trade_duration_seconds: Trade duration
            
        Returns:
            Success status
        """
        try:
            # Determine outcome
            if actual_pnl_percent > 1.0:
                outcome = "WIN"
            elif actual_pnl_percent < -1.0:
                outcome = "LOSS"
            else:
                outcome = "BREAKEVEN"
            
            # Update ML features
            database.update_ml_features_outcome(
                execution_id=execution_id,
                actual_pnl=actual_pnl,
                actual_pnl_percent=actual_pnl_percent,
                outcome=outcome,
                trade_duration_seconds=trade_duration_seconds
            )
            
            # Update ML prediction with outcome
            # Determine if prediction was accurate
            prediction_accurate = outcome == "WIN"  # Simplified: profit = correct prediction
            
            database.update_ml_prediction_outcome(
                execution_id=execution_id,
                actual_outcome=outcome,
                actual_pnl=actual_pnl,
                prediction_accurate=prediction_accurate
            )
            
            # Update recent predictions tracking
            for pred in self._recent_predictions:
                if pred["execution_id"] == execution_id:
                    pred["outcome"] = outcome
                    pred["pnl"] = actual_pnl
                    pred["accurate"] = prediction_accurate
                    break
            
            # Check for drift after logging outcome
            if self.drift_detection:
                self._check_drift()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to log outcome: {e}")
            return False
    
    def _check_drift(self) -> None:
        """Check for model drift based on recent performance."""
        # Get predictions with outcomes
        completed = [p for p in self._recent_predictions if "outcome" in p]
        
        if len(completed) < 10:
            return  # Not enough data
        
        # Calculate recent accuracy
        recent_accurate = sum(1 for p in completed[-20:] if p.get("accurate", False))
        recent_accuracy = recent_accurate / min(20, len(completed))
        
        # Set baseline if not set
        if self._baseline_accuracy is None:
            self._baseline_accuracy = recent_accuracy
            return
        
        # Check drift
        accuracy_drop = self._baseline_accuracy - recent_accuracy
        
        if accuracy_drop > self.drift_threshold:
            logger.warning(
                f"Model drift detected! Accuracy dropped from "
                f"{self._baseline_accuracy:.2%} to {recent_accuracy:.2%}"
            )
            self._trigger_retraining_alert()
    
    def _trigger_retraining_alert(self) -> None:
        """Trigger alert for model retraining."""
        logger.warning("Model retraining recommended due to performance drift")
        # This could also trigger notifications, automated retraining, etc.
    
    def get_performance_stats(
        self,
        days: int = 30,
        model_version: str = None
    ) -> Dict[str, Any]:
        """
        Get performance statistics for the feedback loop.
        
        Args:
            days: Number of days to analyze
            model_version: Optional model version filter
            
        Returns:
            Dict with performance metrics
        """
        stats = database.get_prediction_accuracy(model_version, days)
        
        # Add recent performance from memory
        completed = [p for p in self._recent_predictions if "outcome" in p]
        
        if completed:
            recent_wins = sum(1 for p in completed if p.get("outcome") == "WIN")
            recent_pnl = sum(p.get("pnl", 0) for p in completed)
            
            stats["recent_predictions"] = len(completed)
            stats["recent_win_rate"] = recent_wins / len(completed)
            stats["recent_pnl"] = recent_pnl
        
        if self._baseline_accuracy:
            stats["baseline_accuracy"] = self._baseline_accuracy
        
        return stats
    
    def get_training_data_summary(self) -> Dict[str, Any]:
        """
        Get summary of available training data.
        
        Returns:
            Dict with training data statistics
        """
        try:
            # Get all training data records
            data = database.get_training_data(outcome_only=True, limit=10000)
            
            if not data:
                return {
                    "total_samples": 0,
                    "samples_with_outcome": 0,
                    "date_range": None,
                }
            
            outcomes = [d.get("outcome") for d in data if d.get("outcome")]
            win_count = sum(1 for o in outcomes if o == "WIN")
            loss_count = sum(1 for o in outcomes if o == "LOSS")
            
            dates = [d.get("snapshot_time") for d in data if d.get("snapshot_time")]
            
            return {
                "total_samples": len(data),
                "samples_with_outcome": len(outcomes),
                "win_samples": win_count,
                "loss_samples": loss_count,
                "win_rate": win_count / len(outcomes) if outcomes else 0,
                "earliest_date": min(dates) if dates else None,
                "latest_date": max(dates) if dates else None,
                "underlyings": list(set(d.get("underlying") for d in data if d.get("underlying"))),
                "strategies": list(set(d.get("strategy_type") for d in data if d.get("strategy_type"))),
            }
            
        except Exception as e:
            logger.error(f"Failed to get training data summary: {e}")
            return {"error": str(e)}
    
    def should_retrain(self) -> bool:
        """
        Determine if model should be retrained.
        
        Returns:
            True if retraining is recommended
        """
        stats = self.get_performance_stats()
        
        # Check accuracy drop
        if self._baseline_accuracy:
            current_accuracy = stats.get("accuracy", self._baseline_accuracy)
            if self._baseline_accuracy - current_accuracy > self.drift_threshold:
                return True
        
        # Check minimum samples for retraining
        min_samples = ML_CONFIG.get("min_training_samples", 100)
        training_summary = self.get_training_data_summary()
        
        if training_summary.get("samples_with_outcome", 0) >= min_samples * 1.5:
            # Have enough new data for retraining
            return True
        
        return False
    
    def reset_baseline(self) -> None:
        """Reset baseline accuracy (after retraining)."""
        self._baseline_accuracy = None
        self._recent_predictions = []
        logger.info("Feedback baseline reset")


# Singleton instance
_feedback_collector: Optional[FeedbackCollector] = None


def get_feedback_collector() -> FeedbackCollector:
    """Get or create the singleton feedback collector instance."""
    global _feedback_collector
    if _feedback_collector is None:
        _feedback_collector = FeedbackCollector()
    return _feedback_collector
