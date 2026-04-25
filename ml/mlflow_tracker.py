"""
MLflow Tracking for Options Trading ML Models

Provides experiment tracking, model versioning, and performance monitoring.
Supports local file store with optional remote tracking server.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import pickle

from config.settings import BASE_DIR, ML_CONFIG
from core.logger import logger

# MLflow imports with fallback
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlflow.models.signature import infer_signature
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not installed. Install with: pip install mlflow")


class MLflowTracker:
    """
    MLflow experiment and model tracking.
    
    Features:
    - Experiment tracking for training runs
    - Model registry for version control
    - Metrics logging for backtests and paper trading
    - Artifact storage for models and feature importance
    """
    
    def __init__(self, experiment_name: str = "options_trading_ml"):
        self.experiment_name = experiment_name
        
        # Use proper file:// URI format for Windows paths
        mlruns_path = Path(ML_CONFIG.get("mlflow_tracking_uri", str(BASE_DIR / "mlruns")))
        self.tracking_uri = mlruns_path.as_uri()  # Converts to file:///C:/... format
        
        self.enabled = MLFLOW_AVAILABLE and ML_CONFIG.get("mlflow_enabled", True)
        
        if self.enabled:
            self._setup_mlflow()
        else:
            logger.info("MLflow tracking disabled or unavailable")
    
    def _setup_mlflow(self) -> None:
        """Set up MLflow tracking."""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            
            # Create or get experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                self.experiment_id = mlflow.create_experiment(
                    self.experiment_name
                )
            else:
                self.experiment_id = experiment.experiment_id
            
            mlflow.set_experiment(self.experiment_name)
            self.client = MlflowClient()
            
            logger.info(f"MLflow initialized: {self.tracking_uri}, experiment: {self.experiment_name}")
        except Exception as e:
            logger.error(f"Failed to initialize MLflow: {e}")
            self.enabled = False
    
    def start_training_run(
        self,
        run_name: str,
        model_type: str,
        hyperparameters: Dict[str, Any],
        tags: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """
        Start a new training run.
        
        Args:
            run_name: Name for this training run
            model_type: Type of model (xgboost, lightgbm, rf, ensemble)
            hyperparameters: Model hyperparameters
            tags: Additional tags
            
        Returns:
            Run ID or None if tracking disabled
        """
        if not self.enabled:
            return None
        
        try:
            run = mlflow.start_run(run_name=run_name)
            
            # Log model type and hyperparameters
            mlflow.log_param("model_type", model_type)
            for param_name, param_value in hyperparameters.items():
                # Handle nested dicts
                if isinstance(param_value, dict):
                    for k, v in param_value.items():
                        mlflow.log_param(f"{param_name}.{k}", v)
                else:
                    mlflow.log_param(param_name, param_value)
            
            # Log tags
            default_tags = {
                "training_date": datetime.now().isoformat(),
                "model_type": model_type,
            }
            if tags:
                default_tags.update(tags)
            mlflow.set_tags(default_tags)
            
            logger.info(f"Started MLflow run: {run.info.run_id}")
            return run.info.run_id
            
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
            return None
    
    def log_training_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """
        Log training metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number (for iteration tracking)
        """
        if not self.enabled:
            return
        
        try:
            for metric_name, value in metrics.items():
                if step is not None:
                    mlflow.log_metric(metric_name, value, step=step)
                else:
                    mlflow.log_metric(metric_name, value)
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
    
    def log_validation_metrics(
        self,
        accuracy: float,
        precision: float,
        recall: float,
        f1: float,
        auc_roc: float,
        confusion_matrix: Optional[List[List[int]]] = None
    ) -> None:
        """
        Log model validation metrics.
        
        Args:
            accuracy: Classification accuracy
            precision: Precision score
            recall: Recall score
            f1: F1 score
            auc_roc: Area under ROC curve
            confusion_matrix: Optional confusion matrix
        """
        if not self.enabled:
            return
        
        try:
            mlflow.log_metrics({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "auc_roc": auc_roc
            })
            
            if confusion_matrix:
                # Save confusion matrix as artifact
                cm_path = Path("confusion_matrix.json")
                with open(cm_path, "w") as f:
                    json.dump(confusion_matrix, f)
                mlflow.log_artifact(str(cm_path))
                cm_path.unlink()  # Clean up
                
        except Exception as e:
            logger.error(f"Failed to log validation metrics: {e}")
    
    def log_backtest_metrics(
        self,
        sharpe_ratio: float,
        sortino_ratio: float,
        max_drawdown: float,
        win_rate: float,
        profit_factor: float,
        total_trades: int,
        total_pnl: float,
        calmar_ratio: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None
    ) -> None:
        """
        Log backtest performance metrics.
        
        Args:
            sharpe_ratio: Risk-adjusted return
            sortino_ratio: Downside risk-adjusted return
            max_drawdown: Maximum peak-to-trough decline (%)
            win_rate: Percentage of winning trades
            profit_factor: Gross profit / Gross loss
            total_trades: Total number of trades
            total_pnl: Total P&L in currency
            calmar_ratio: Optional Calmar ratio
            avg_win: Average winning trade
            avg_loss: Average losing trade
        """
        if not self.enabled:
            return
        
        try:
            metrics = {
                "backtest_sharpe": sharpe_ratio,
                "backtest_sortino": sortino_ratio,
                "backtest_max_drawdown": max_drawdown,
                "backtest_win_rate": win_rate,
                "backtest_profit_factor": profit_factor,
                "backtest_total_trades": total_trades,
                "backtest_total_pnl": total_pnl,
            }
            
            if calmar_ratio is not None:
                metrics["backtest_calmar"] = calmar_ratio
            if avg_win is not None:
                metrics["backtest_avg_win"] = avg_win
            if avg_loss is not None:
                metrics["backtest_avg_loss"] = avg_loss
            
            mlflow.log_metrics(metrics)
            
        except Exception as e:
            logger.error(f"Failed to log backtest metrics: {e}")
    
    def log_feature_importance(
        self,
        feature_importance: Dict[str, float],
        top_n: int = 20
    ) -> None:
        """
        Log feature importance scores.
        
        Args:
            feature_importance: Dictionary of feature names and importance scores
            top_n: Number of top features to log individually
        """
        if not self.enabled:
            return
        
        try:
            # Sort by importance
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Log top N as metrics
            for i, (feature, importance) in enumerate(sorted_features[:top_n]):
                mlflow.log_metric(f"feature_importance_{i+1}_{feature}", importance)
            
            # Save full importance as artifact
            importance_path = Path("feature_importance.json")
            with open(importance_path, "w") as f:
                json.dump(dict(sorted_features), f, indent=2)
            mlflow.log_artifact(str(importance_path))
            importance_path.unlink()
            
        except Exception as e:
            logger.error(f"Failed to log feature importance: {e}")
    
    def log_model(
        self,
        model: Any,
        model_name: str,
        model_type: str,
        input_example: Optional[Any] = None,
        register: bool = False
    ) -> Optional[str]:
        """
        Log a trained model.
        
        Args:
            model: Trained model object
            model_name: Name for the model
            model_type: Type of model (xgboost, lightgbm, sklearn)
            input_example: Example input for signature inference
            register: Whether to register in model registry
            
        Returns:
            Model URI or None
        """
        if not self.enabled:
            return None
        
        try:
            # Determine the right logging method based on model type
            if model_type == "xgboost":
                import mlflow.xgboost
                model_info = mlflow.xgboost.log_model(
                    model, 
                    model_name,
                    registered_model_name=model_name if register else None
                )
            elif model_type == "lightgbm":
                import mlflow.lightgbm
                model_info = mlflow.lightgbm.log_model(
                    model,
                    model_name,
                    registered_model_name=model_name if register else None
                )
            else:
                import mlflow.sklearn
                model_info = mlflow.sklearn.log_model(
                    model,
                    model_name,
                    registered_model_name=model_name if register else None
                )
            
            logger.info(f"Model logged: {model_info.model_uri}")
            return model_info.model_uri
            
        except Exception as e:
            logger.error(f"Failed to log model: {e}")
            
            # Fallback: save as pickle artifact
            try:
                model_path = Path(f"{model_name}.pkl")
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
                mlflow.log_artifact(str(model_path))
                model_path.unlink()
                return f"artifacts/{model_name}.pkl"
            except Exception as e2:
                logger.error(f"Fallback model save also failed: {e2}")
                return None
    
    def end_run(self, status: str = "FINISHED") -> None:
        """
        End the current run.
        
        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        if not self.enabled:
            return
        
        try:
            mlflow.end_run(status=status)
        except Exception as e:
            logger.error(f"Failed to end MLflow run: {e}")
    
    def start_paper_trading_run(
        self,
        run_name: str,
        model_version: str,
        config: Dict[str, Any]
    ) -> Optional[str]:
        """
        Start a paper trading experiment run.
        
        Args:
            run_name: Name for the paper trading run
            model_version: Version of model being tested
            config: Paper trading configuration
            
        Returns:
            Run ID or None
        """
        if not self.enabled:
            return None
        
        try:
            run = mlflow.start_run(run_name=run_name)
            
            mlflow.set_tags({
                "run_type": "paper_trading",
                "model_version": model_version,
                "start_date": datetime.now().isoformat(),
            })
            
            mlflow.log_params({
                "paper_trading": True,
                "model_version": model_version,
                **{f"config_{k}": v for k, v in config.items() if not isinstance(v, (dict, list))}
            })
            
            return run.info.run_id
            
        except Exception as e:
            logger.error(f"Failed to start paper trading run: {e}")
            return None
    
    def log_daily_paper_trading_metrics(
        self,
        date: str,
        pnl: float,
        num_trades: int,
        win_rate: float,
        prediction_accuracy: float,
        cumulative_pnl: float
    ) -> None:
        """
        Log daily paper trading metrics.
        
        Args:
            date: Date string
            pnl: Daily P&L
            num_trades: Number of trades
            win_rate: Win rate for the day
            prediction_accuracy: ML prediction accuracy
            cumulative_pnl: Cumulative P&L to date
        """
        if not self.enabled:
            return
        
        try:
            # Use date as step for time series visualization
            day_num = int(datetime.fromisoformat(date).timestamp() / 86400)
            
            mlflow.log_metrics({
                "daily_pnl": pnl,
                "daily_trades": num_trades,
                "daily_win_rate": win_rate,
                "daily_prediction_accuracy": prediction_accuracy,
                "cumulative_pnl": cumulative_pnl,
            }, step=day_num)
            
        except Exception as e:
            logger.error(f"Failed to log daily metrics: {e}")
    
    def get_best_model(
        self,
        metric: str = "backtest_sharpe",
        model_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get the best model based on a metric.
        
        Args:
            metric: Metric to optimize
            model_name: Optional model name filter
            
        Returns:
            Dict with model info or None
        """
        if not self.enabled:
            return None
        
        try:
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string="attributes.status = 'FINISHED'",
                order_by=[f"metrics.{metric} DESC"],
                max_results=1
            )
            
            if len(runs) == 0:
                return None
            
            best_run = runs.iloc[0]
            return {
                "run_id": best_run.run_id,
                "metric_value": best_run[f"metrics.{metric}"],
                "params": {k.replace("params.", ""): v 
                          for k, v in best_run.items() if k.startswith("params.")},
                "model_type": best_run.get("params.model_type", "unknown")
            }
            
        except Exception as e:
            logger.error(f"Failed to get best model: {e}")
            return None
    
    def compare_models(
        self,
        run_ids: List[str],
        metrics: List[str]
    ) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Compare multiple model runs.
        
        Args:
            run_ids: List of run IDs to compare
            metrics: List of metrics to compare
            
        Returns:
            Dict mapping run_id to metric values
        """
        if not self.enabled:
            return None
        
        try:
            comparison = {}
            
            for run_id in run_ids:
                run = mlflow.get_run(run_id)
                comparison[run_id] = {
                    metric: run.data.metrics.get(metric, None)
                    for metric in metrics
                }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare models: {e}")
            return None
    
    def promote_model(
        self,
        model_name: str,
        version: int,
        stage: str = "Production"
    ) -> bool:
        """
        Promote a model version to a stage.
        
        Args:
            model_name: Registered model name
            version: Model version number
            stage: Target stage (Staging, Production, Archived)
            
        Returns:
            Success status
        """
        if not self.enabled:
            return False
        
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            logger.info(f"Model {model_name} v{version} promoted to {stage}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            return False


# Singleton instance
_tracker: Optional[MLflowTracker] = None


def get_mlflow_tracker(experiment_name: str = "options_trading_ml") -> MLflowTracker:
    """Get or create the singleton MLflow tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = MLflowTracker(experiment_name)
    return _tracker
