"""
Model Registry for Version Control and A/B Testing

Manages model versions, staging/production promotion,
performance comparison, and rollback capabilities.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import shutil

from config.settings import ML_CONFIG, ML_MODELS_DIR
from core.database import database
from core.logger import logger


class ModelRegistry:
    """
    Registry for ML model version management.
    
    Features:
    - Model versioning and storage
    - Stage management (development, staging, production)
    - A/B testing support
    - Performance tracking
    - Rollback capabilities
    - Automatic model promotion
    """
    
    STAGES = ["development", "staging", "production", "archived"]
    
    def __init__(self):
        """Initialize model registry."""
        self.model_path = Path(ML_CONFIG.get("model_path", ML_MODELS_DIR))
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # A/B testing state
        self.ab_test_active = False
        self.ab_test_models: Dict[str, str] = {}  # {"A": version, "B": version}
        self.ab_test_traffic_split = 0.5  # 50/50 by default
        
        logger.info("ModelRegistry initialized")
    
    def register_model(
        self,
        model_version: str,
        model_type: str,
        metrics: Dict[str, float],
        backtest_metrics: Dict[str, float] = None,
        notes: str = None
    ) -> bool:
        """
        Register a new model version.
        
        Args:
            model_version: Model version string
            model_type: Type of model
            metrics: Validation metrics
            backtest_metrics: Optional backtest performance
            notes: Optional notes
            
        Returns:
            Success status
        """
        try:
            return database.save_model_performance(
                model_version=model_version,
                model_type=model_type,
                training_samples=metrics.get("training_samples", 0),
                metrics=metrics,
                backtest_metrics=backtest_metrics
            )
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return False
    
    def promote_model(
        self,
        model_version: str,
        target_stage: str = "production"
    ) -> bool:
        """
        Promote a model to a new stage.
        
        Args:
            model_version: Version to promote
            target_stage: Target stage
            
        Returns:
            Success status
        """
        if target_stage not in self.STAGES:
            logger.error(f"Invalid stage: {target_stage}")
            return False
        
        try:
            if target_stage == "production":
                # Set as active model
                return database.set_active_model(model_version)
            else:
                # Update stage in database
                # This would need an additional database method
                logger.info(f"Model {model_version} promoted to {target_stage}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            return False
    
    def get_production_model(self) -> Optional[str]:
        """
        Get the current production model version.
        
        Returns:
            Model version string or None
        """
        models = database.get_model_performance(active_only=True)
        if models:
            return models[0].get("model_version")
        return None
    
    def list_models(
        self,
        stage: str = None,
        model_type: str = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        List registered models.
        
        Args:
            stage: Optional stage filter
            model_type: Optional model type filter
            limit: Maximum number of models
            
        Returns:
            List of model metadata dicts
        """
        models = database.get_model_performance()
        
        if stage:
            models = [m for m in models if m.get("stage") == stage]
        
        if model_type:
            models = [m for m in models if m.get("model_type") == model_type]
        
        return models[:limit]
    
    def compare_models(
        self,
        version_a: str,
        version_b: str
    ) -> Dict[str, Any]:
        """
        Compare two model versions.
        
        Args:
            version_a: First model version
            version_b: Second model version
            
        Returns:
            Comparison dict
        """
        models_a = database.get_model_performance(model_version=version_a)
        models_b = database.get_model_performance(model_version=version_b)
        
        if not models_a or not models_b:
            return {"error": "One or both models not found"}
        
        model_a = models_a[0]
        model_b = models_b[0]
        
        comparison = {
            "version_a": version_a,
            "version_b": version_b,
            "metrics": {},
            "winner": None,
        }
        
        # Compare metrics
        metrics_to_compare = [
            "accuracy", "precision_score", "recall_score", "f1_score", "auc_roc",
            "backtest_sharpe", "backtest_win_rate", "backtest_profit_factor"
        ]
        
        a_score = 0
        b_score = 0
        
        for metric in metrics_to_compare:
            val_a = model_a.get(metric, 0) or 0
            val_b = model_b.get(metric, 0) or 0
            
            comparison["metrics"][metric] = {
                "a": val_a,
                "b": val_b,
                "winner": "A" if val_a > val_b else "B" if val_b > val_a else "TIE"
            }
            
            if val_a > val_b:
                a_score += 1
            elif val_b > val_a:
                b_score += 1
        
        comparison["winner"] = "A" if a_score > b_score else "B" if b_score > a_score else "TIE"
        comparison["a_score"] = a_score
        comparison["b_score"] = b_score
        
        return comparison
    
    def start_ab_test(
        self,
        version_a: str,
        version_b: str,
        traffic_split: float = 0.5
    ) -> bool:
        """
        Start an A/B test between two models.
        
        Args:
            version_a: Control model version
            version_b: Test model version
            traffic_split: Percentage of traffic to version B
            
        Returns:
            Success status
        """
        # Validate both models exist
        model_a = database.get_model_performance(model_version=version_a)
        model_b = database.get_model_performance(model_version=version_b)
        
        if not model_a or not model_b:
            logger.error("One or both models not found for A/B test")
            return False
        
        self.ab_test_active = True
        self.ab_test_models = {"A": version_a, "B": version_b}
        self.ab_test_traffic_split = traffic_split
        
        logger.info(f"A/B test started: A={version_a}, B={version_b}, split={traffic_split}")
        return True
    
    def get_ab_test_model(self) -> Optional[str]:
        """
        Get model version based on A/B test assignment.
        
        Returns:
            Model version to use
        """
        if not self.ab_test_active:
            return self.get_production_model()
        
        import random
        if random.random() < self.ab_test_traffic_split:
            return self.ab_test_models.get("B")
        return self.ab_test_models.get("A")
    
    def end_ab_test(self, promote_winner: bool = True) -> Dict[str, Any]:
        """
        End A/B test and optionally promote winner.
        
        Args:
            promote_winner: Whether to promote the winning model
            
        Returns:
            A/B test results
        """
        if not self.ab_test_active:
            return {"error": "No active A/B test"}
        
        version_a = self.ab_test_models.get("A")
        version_b = self.ab_test_models.get("B")
        
        # Compare live performance
        stats_a = database.get_prediction_accuracy(model_version=version_a, days=7)
        stats_b = database.get_prediction_accuracy(model_version=version_b, days=7)
        
        results = {
            "version_a": version_a,
            "version_b": version_b,
            "stats_a": stats_a,
            "stats_b": stats_b,
        }
        
        # Determine winner based on accuracy and P&L
        accuracy_a = stats_a.get("accuracy", 0)
        accuracy_b = stats_b.get("accuracy", 0)
        pnl_a = stats_a.get("total_pnl", 0)
        pnl_b = stats_b.get("total_pnl", 0)
        
        # Score: 60% accuracy, 40% P&L
        score_a = 0.6 * accuracy_a + 0.4 * (1 if pnl_a > pnl_b else 0)
        score_b = 0.6 * accuracy_b + 0.4 * (1 if pnl_b > pnl_a else 0)
        
        winner = "A" if score_a >= score_b else "B"
        winner_version = version_a if winner == "A" else version_b
        
        results["winner"] = winner
        results["winner_version"] = winner_version
        
        # Promote winner if requested
        if promote_winner:
            self.promote_model(winner_version, "production")
            results["promoted"] = True
        
        # End test
        self.ab_test_active = False
        self.ab_test_models = {}
        
        logger.info(f"A/B test ended. Winner: {winner} ({winner_version})")
        
        return results
    
    def should_auto_promote(
        self,
        model_version: str,
        min_trades: int = None,
        min_accuracy: float = None,
        min_sharpe: float = None
    ) -> bool:
        """
        Check if model meets auto-promotion criteria.
        
        Args:
            model_version: Model to check
            min_trades: Minimum trades required
            min_accuracy: Minimum accuracy required
            min_sharpe: Minimum Sharpe ratio required
            
        Returns:
            True if model should be promoted
        """
        paper_config = ML_CONFIG.get("paper_trading", {})
        min_trades = min_trades or paper_config.get("min_trades_for_promotion", 20)
        min_accuracy = min_accuracy or paper_config.get("min_accuracy_for_promotion", 0.55)
        min_sharpe = min_sharpe or paper_config.get("min_sharpe_for_promotion", 1.0)
        
        # Get model stats
        stats = database.get_prediction_accuracy(model_version=model_version, days=30)
        models = database.get_model_performance(model_version=model_version)
        
        if not models:
            return False
        
        model = models[0]
        
        # Check criteria
        total_trades = stats.get("total_predictions", 0)
        if total_trades < min_trades:
            logger.debug(f"Not enough trades: {total_trades} < {min_trades}")
            return False
        
        accuracy = stats.get("accuracy", 0)
        if accuracy < min_accuracy:
            logger.debug(f"Accuracy too low: {accuracy} < {min_accuracy}")
            return False
        
        sharpe = model.get("backtest_sharpe", 0) or 0
        if sharpe < min_sharpe:
            logger.debug(f"Sharpe too low: {sharpe} < {min_sharpe}")
            return False
        
        logger.info(f"Model {model_version} meets promotion criteria")
        return True
    
    def rollback(self) -> Optional[str]:
        """
        Rollback to previous production model.
        
        Returns:
            Previous model version or None
        """
        models = self.list_models(limit=5)
        
        if len(models) < 2:
            logger.warning("No previous model available for rollback")
            return None
        
        # Get current production
        current = self.get_production_model()
        
        # Find previous model
        for model in models:
            version = model.get("model_version")
            if version != current:
                self.promote_model(version, "production")
                logger.info(f"Rolled back to model: {version}")
                return version
        
        return None
    
    def archive_model(self, model_version: str) -> bool:
        """
        Archive a model version.
        
        Args:
            model_version: Version to archive
            
        Returns:
            Success status
        """
        model_dir = self.model_path / model_version
        archive_dir = self.model_path / "archived" / model_version
        
        if not model_dir.exists():
            return False
        
        try:
            archive_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(model_dir), str(archive_dir))
            logger.info(f"Model {model_version} archived")
            return True
        except Exception as e:
            logger.error(f"Failed to archive model: {e}")
            return False
    
    def get_model_lineage(self, model_version: str) -> Dict[str, Any]:
        """
        Get model lineage and training history.
        
        Args:
            model_version: Model version
            
        Returns:
            Lineage information
        """
        model_dir = self.model_path / model_version
        metadata_file = model_dir / "metadata.json"
        
        if not metadata_file.exists():
            return {"error": "Model metadata not found"}
        
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        
        # Get performance from database
        models = database.get_model_performance(model_version=model_version)
        if models:
            metadata["database_record"] = models[0]
        
        return metadata


# Singleton instance
_model_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """Get or create the singleton model registry instance."""
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry()
    return _model_registry
