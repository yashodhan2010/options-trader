"""
Machine Learning Module for Options Trading Bot

This module provides ML-powered prediction, feature engineering,
model training with Optuna optimization, and MLflow experiment tracking.

Components:
    - feature_engineer: Extract and normalize 55+ features from market data
    - data_collector: Historical data caching to SQLite
    - model_trainer: XGBoost/LightGBM/RF training with Optuna
    - predictor: Ensemble inference with risk guardrails
    - backtester: Historical simulation engine
    - feedback_collector: Outcome logging and drift detection
    - model_registry: Version control and A/B testing
    - mlflow_tracker: MLflow experiment and model tracking
    - guardrails: Risk management constraints
    - paper_trading_runner: Orchestration for paper trading tests
    - evaluator: Trading-specific evaluation metrics and model comparison
    - auto_retrain: Automatic retraining from trade feedback
"""

# Core ML components
from ml.guardrails import TradingGuardrails, get_guardrails
from ml.feature_engineer import FeatureEngineer, get_feature_engineer
from ml.predictor import MLPredictor, get_predictor

# Singleton getters for easy access
from ml.data_collector import get_data_collector
from ml.model_trainer import get_model_trainer
from ml.feedback_collector import get_feedback_collector
from ml.model_registry import get_model_registry
from ml.paper_trading_runner import get_paper_trading_runner
from ml.mlflow_tracker import get_mlflow_tracker
from ml.evaluator import TradingEvaluator, ModelComparator, TradingMetrics
from ml.auto_retrain import AutoRetrainer, get_auto_retrainer

__all__ = [
    # Classes
    "FeatureEngineer",
    "MLPredictor",
    "TradingGuardrails",
    "TradingEvaluator",
    "ModelComparator",
    "TradingMetrics",
    "AutoRetrainer",
    # Singleton getters
    "get_feature_engineer",
    "get_predictor",
    "get_guardrails",
    "get_data_collector",
    "get_model_trainer",
    "get_feedback_collector",
    "get_model_registry",
    "get_paper_trading_runner",
    "get_mlflow_tracker",
    "get_auto_retrainer",
]
