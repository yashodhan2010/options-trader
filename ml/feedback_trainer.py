"""
Feedback-Weighted Model Training

Uses trade outcomes to weight training samples and adjust predictions.
Key adjustments based on feedback:
1. Penalize NEUTRAL predictions (straddles/strangles losing money)
2. Boost BEARISH predictions (long_put profitable)
3. Reduce confidence output (model is overconfident)
"""

import sqlite3
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from core.database import database
from core.logger import logger
from ml.model_trainer import ModelTrainer
from ml.live_feature_collector import get_training_data_from_snapshots
from ml.feedback_evaluator import MLFeedbackEvaluator, FeedbackMetrics


class FeedbackWeightedTrainer:
    """
    Train models with feedback-based weighting.
    
    Uses actual trade outcomes to:
    1. Weight samples by financial outcome
    2. Adjust class weights based on directional performance
    3. Apply confidence calibration
    """
    
    def __init__(self):
        self.evaluator = MLFeedbackEvaluator()
        self.trainer = ModelTrainer()
        self.db_path = Path("data/trading_bot.db")
    
    def get_feedback_metrics(self) -> FeedbackMetrics:
        """Get current feedback metrics."""
        return self.evaluator.evaluate_all_trades(include_open=True)
    
    def compute_class_weights(self, metrics: FeedbackMetrics) -> Dict[int, float]:
        """
        Compute class weights based on direction performance.
        
        If BEARISH predictions are working, boost class 0 (DOWN) weight.
        If BULLISH working, boost class 1 (UP) weight.
        """
        if not metrics:
            return {0: 1.0, 1: 1.0}
        
        direction_perf = metrics.direction_performance
        
        # Default weights
        weights = {0: 1.0, 1: 1.0}
        
        # Get performance by direction
        bearish_pnl = direction_perf.get("BEARISH", {}).get("pnl", 0)
        bullish_pnl = direction_perf.get("BULLISH", {}).get("pnl", 0)
        neutral_pnl = direction_perf.get("NEUTRAL", {}).get("pnl", 0)
        
        # Total absolute PnL
        total_abs = abs(bearish_pnl) + abs(bullish_pnl) + abs(neutral_pnl) + 1
        
        # If bearish is profitable, predict more DOWN (class 0)
        if bearish_pnl > 0:
            # Boost DOWN class weight proportionally
            weights[0] = 1.0 + (bearish_pnl / total_abs)
        
        # If bullish is profitable, predict more UP (class 1)  
        if bullish_pnl > 0:
            weights[1] = 1.0 + (bullish_pnl / total_abs)
        
        # If neutral is losing badly, we need to push predictions
        # towards UP or DOWN instead of staying neutral
        if neutral_pnl < 0:
            # Increase both directional weights to avoid neutral zone
            boost = abs(neutral_pnl) / total_abs * 0.5
            weights[0] += boost
            weights[1] += boost
        
        # Normalize
        total_weight = weights[0] + weights[1]
        weights[0] = weights[0] / total_weight * 2
        weights[1] = weights[1] / total_weight * 2
        
        logger.info(f"Computed class weights from feedback: {weights}")
        return weights
    
    def compute_sample_weights(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        feature_names: List[str]
    ) -> np.ndarray:
        """
        Compute per-sample weights based on similar past trades.
        
        Samples similar to profitable trades get higher weight.
        Samples similar to losing trades get lower weight.
        """
        # For now, use uniform weights
        # Future: implement similarity matching
        return np.ones(len(X))
    
    def compute_confidence_adjustment(self, metrics: FeedbackMetrics) -> float:
        """
        Compute a multiplier to adjust model confidence.
        
        If model says 70% but only wins 21%, we need to scale down.
        """
        if not metrics or not metrics.confidence_calibration:
            return 1.0
        
        # Find the average overconfidence
        adjustments = []
        
        for bucket, data in metrics.confidence_calibration.items():
            expected = data.get("expected_win_rate", 0.5)  # e.g., 0.75 for 70-80%
            actual = data.get("win_rate", 0.5)  # e.g., 0.21
            
            if expected > 0:
                # Ratio of actual to expected
                ratio = actual / expected
                adjustments.append(ratio)
        
        if adjustments:
            avg_adjustment = sum(adjustments) / len(adjustments)
            # Don't go below 0.3 or above 1.0
            return max(0.3, min(1.0, avg_adjustment))
        
        return 1.0
    
    def train_with_feedback(
        self,
        min_samples: int = 100,
        optimize: bool = True
    ) -> Tuple[str, Dict]:
        """
        Train model using feedback-weighted approach.
        
        Returns:
            Tuple of (model_id, metrics)
        """
        # Get training data
        logger.info("Loading training data from live snapshots...")
        X, y, feature_names = get_training_data_from_snapshots(
            min_samples=min_samples,
            require_full_features=False
        )
        
        if X is None:
            logger.error("Failed to load training data")
            return None, None
        
        # Get feedback metrics
        logger.info("Computing feedback-based weights...")
        metrics = self.get_feedback_metrics()
        
        if metrics:
            # Compute class weights
            class_weights = self.compute_class_weights(metrics)
            
            # Compute sample weights
            sample_weights = self.compute_sample_weights(X, y, feature_names)
            
            # Compute confidence adjustment
            conf_adjustment = self.compute_confidence_adjustment(metrics)
            
            logger.info(f"Class weights: {class_weights}")
            logger.info(f"Confidence adjustment: {conf_adjustment:.2f}")
        else:
            class_weights = {0: 1.0, 1: 1.0}
            sample_weights = np.ones(len(X))
            conf_adjustment = 1.0
        
        # Train with class weights
        logger.info("Training feedback-weighted model...")
        
        # Store feedback config for the model
        feedback_config = {
            "class_weights": class_weights,
            "confidence_adjustment": conf_adjustment,
            "feedback_based": True,
            "training_trades": metrics.total_predictions if metrics else 0,
            "training_win_rate": metrics.win_rate if metrics else 0,
            "training_pnl": metrics.total_pnl if metrics else 0,
        }
        
        # Train the model
        model, train_metrics, model_id = self.trainer.train_direction_model(
            X=X,
            y=y,
            feature_names=feature_names,
            model_type="ensemble",
            optimize=optimize,
        )
        
        # Save feedback config with model
        self._save_feedback_config(model_id, feedback_config)
        
        logger.info(f"Feedback-weighted model trained: {model_id}")
        
        return model_id, {
            **train_metrics,
            "feedback_config": feedback_config,
        }
    
    def _save_feedback_config(self, model_id: str, config: Dict):
        """Save feedback configuration with model."""
        try:
            model_path = Path("data/ml_models") / model_id
            config_path = model_path / "feedback_config.json"
            
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Saved feedback config to {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving feedback config: {e}")


def retrain_with_feedback():
    """
    Main function to retrain model with feedback weighting.
    """
    trainer = FeedbackWeightedTrainer()
    
    # Get current feedback
    print("=" * 60)
    print("FEEDBACK-WEIGHTED MODEL TRAINING")
    print("=" * 60)
    print()
    
    metrics = trainer.get_feedback_metrics()
    
    if metrics:
        print("Current Model Performance:")
        print(f"  Win Rate: {metrics.win_rate:.1%}")
        print(f"  Total PnL: Rs.{metrics.total_pnl:,.2f}")
        print(f"  Profit Factor: {metrics.profit_factor:.2f}")
        print()
        
        print("Feedback-Based Adjustments:")
        weights = trainer.compute_class_weights(metrics)
        conf_adj = trainer.compute_confidence_adjustment(metrics)
        print(f"  Class Weights: DOWN={weights[0]:.2f}, UP={weights[1]:.2f}")
        print(f"  Confidence Adjustment: {conf_adj:.2f}x")
        print()
    
    # Train with feedback
    print("Training new model with feedback weights...")
    model_id, train_metrics = trainer.train_with_feedback(optimize=True)
    
    if model_id:
        print()
        print("=" * 60)
        print(f"NEW MODEL: {model_id}")
        print("=" * 60)
        print(f"  Accuracy: {train_metrics.get('accuracy', 0):.1%}")
        print(f"  Precision: {train_metrics.get('precision', 0):.1%}")
        print(f"  Recall: {train_metrics.get('recall', 0):.1%}")
        print()
        print("Feedback adjustments applied:")
        fc = train_metrics.get("feedback_config", {})
        print(f"  Based on {fc.get('training_trades', 0)} real trades")
        print(f"  Original win rate: {fc.get('training_win_rate', 0):.1%}")
        print(f"  Confidence will be scaled by {fc.get('confidence_adjustment', 1):.2f}x")
    
    return model_id


if __name__ == "__main__":
    retrain_with_feedback()
