"""
Custom Trading Evaluation Metrics and Model Comparison
Evaluates models using trading-specific metrics beyond standard ML metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    log_loss, brier_score_loss, precision_recall_curve, roc_curve
)
from dataclasses import dataclass
from core.logger import logger


@dataclass
class TradingMetrics:
    """Trading-specific evaluation metrics"""
    # Standard ML metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    
    # Trading-specific metrics
    win_rate: float
    profit_factor: float
    expected_value: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    
    # Signal quality metrics
    bullish_precision: float
    bearish_precision: float
    signal_confidence: float
    false_signal_rate: float
    
    # Risk metrics
    risk_reward_ratio: float
    avg_win: float
    avg_loss: float
    max_consecutive_losses: int
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for MLflow logging"""
        return {
            # Standard metrics
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "auc_roc": self.auc_roc,
            
            # Trading metrics
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "expected_value": self.expected_value,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "calmar_ratio": self.calmar_ratio,
            
            # Signal quality
            "bullish_precision": self.bullish_precision,
            "bearish_precision": self.bearish_precision,
            "signal_confidence": self.signal_confidence,
            "false_signal_rate": self.false_signal_rate,
            
            # Risk metrics
            "risk_reward_ratio": self.risk_reward_ratio,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "max_consecutive_losses": float(self.max_consecutive_losses),
        }


class TradingEvaluator:
    """
    Evaluates ML models using trading-specific metrics
    Simulates trading performance based on model predictions
    """
    
    def __init__(
        self,
        transaction_cost: float = 0.001,  # 0.1% per trade
        slippage: float = 0.0005,  # 0.05% slippage
        risk_free_rate: float = 0.06,  # 6% annual risk-free rate
    ):
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        prices: Optional[np.ndarray] = None,
    ) -> TradingMetrics:
        """
        Comprehensive evaluation with trading metrics
        
        Args:
            y_true: Actual direction (1=up, 0=down)
            y_pred: Predicted direction
            y_prob: Prediction probabilities (for confidence)
            prices: Price series for return simulation
        """
        # Standard ML metrics
        n_classes = len(set(y_true) | set(y_pred))
        avg = "weighted"  # Always weighted — ternary labels {0,1,2} can produce {0,2} splits
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=avg, zero_division=0)
        recall = recall_score(y_true, y_pred, average=avg, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)
        
        # AUC-ROC (needs probabilities)
        if y_prob is not None and len(np.unique(y_true)) > 1:
            try:
                # Always use multi-class OVR — handles both 2 and 3 class cases
                # with the full probability matrix
                if y_prob.ndim == 2:
                    auc_roc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
                else:
                    auc_roc = roc_auc_score(y_true, y_prob)
            except:
                auc_roc = 0.5
        else:
            auc_roc = 0.5
        
        # Trading simulation
        trading_results = self._simulate_trading(y_true, y_pred, prices)
        
        # Signal quality metrics
        signal_metrics = self._calculate_signal_quality(y_true, y_pred, y_prob)
        
        # Risk metrics
        risk_metrics = self._calculate_risk_metrics(trading_results['returns'])
        
        return TradingMetrics(
            # Standard
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_roc=auc_roc,
            
            # Trading
            win_rate=trading_results['win_rate'],
            profit_factor=trading_results['profit_factor'],
            expected_value=trading_results['expected_value'],
            sharpe_ratio=trading_results['sharpe_ratio'],
            max_drawdown=trading_results['max_drawdown'],
            calmar_ratio=trading_results['calmar_ratio'],
            
            # Signal quality
            bullish_precision=signal_metrics['bullish_precision'],
            bearish_precision=signal_metrics['bearish_precision'],
            signal_confidence=signal_metrics['signal_confidence'],
            false_signal_rate=signal_metrics['false_signal_rate'],
            
            # Risk
            risk_reward_ratio=risk_metrics['risk_reward_ratio'],
            avg_win=risk_metrics['avg_win'],
            avg_loss=risk_metrics['avg_loss'],
            max_consecutive_losses=risk_metrics['max_consecutive_losses'],
        )
    
    def _simulate_trading(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prices: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Simulate trading based on predictions"""
        
        # Generate synthetic returns if prices not provided
        if prices is None:
            # Assume 1% move per correct prediction, -1% for wrong
            returns = np.where(y_true == y_pred, 0.01, -0.01)
        else:
            # Calculate actual returns
            price_returns = np.diff(prices) / prices[:-1]
            # Align with predictions (predict at t, realize at t+1)
            if len(price_returns) >= len(y_pred):
                price_returns = price_returns[:len(y_pred)]
            else:
                price_returns = np.pad(price_returns, (0, len(y_pred) - len(price_returns)))
            
            # Long if predict up (1), short if predict down (0)
            position = np.where(y_pred == 1, 1, -1)
            returns = position * price_returns
        
        # Apply transaction costs
        returns = returns - self.transaction_cost - self.slippage
        
        # Calculate metrics
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        win_rate = len(wins) / len(returns) if len(returns) > 0 else 0
        
        total_wins = np.sum(wins) if len(wins) > 0 else 0
        total_losses = np.abs(np.sum(losses)) if len(losses) > 0 else 1e-10
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        expected_value = np.mean(returns) if len(returns) > 0 else 0
        
        # Sharpe Ratio (annualized)
        daily_rf = self.risk_free_rate / 252
        excess_returns = returns - daily_rf
        sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / (np.std(returns) + 1e-10)
        
        # Maximum Drawdown
        cumulative = np.cumprod(1 + returns)
        rolling_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = np.abs(np.min(drawdown)) if len(drawdown) > 0 else 0
        
        # Calmar Ratio (annualized return / max drawdown)
        annual_return = (cumulative[-1] ** (252 / len(returns)) - 1) if len(returns) > 0 else 0
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        return {
            'returns': returns,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'expected_value': expected_value,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
        }
    
    def _calculate_signal_quality(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Calculate signal quality metrics"""
        
        # Bullish precision (when we predict up, how often are we right?)
        bullish_mask = y_pred == 1
        if bullish_mask.sum() > 0:
            bullish_precision = (y_true[bullish_mask] == 1).mean()
        else:
            bullish_precision = 0
        
        # Bearish precision (when we predict down, how often are we right?)
        bearish_mask = y_pred == 0
        if bearish_mask.sum() > 0:
            bearish_precision = (y_true[bearish_mask] == 0).mean()
        else:
            bearish_precision = 0
        
        # Signal confidence (average probability of predictions)
        if y_prob is not None:
            signal_confidence = np.mean(np.maximum(y_prob, 1 - y_prob))
        else:
            signal_confidence = 0.5
        
        # False signal rate
        false_signals = np.sum(y_true != y_pred)
        false_signal_rate = false_signals / len(y_true) if len(y_true) > 0 else 0
        
        return {
            'bullish_precision': bullish_precision,
            'bearish_precision': bearish_precision,
            'signal_confidence': signal_confidence,
            'false_signal_rate': false_signal_rate,
        }
    
    def _calculate_risk_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate risk-related metrics"""
        
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        avg_win = np.mean(wins) if len(wins) > 0 else 0
        avg_loss = np.abs(np.mean(losses)) if len(losses) > 0 else 1e-10
        
        risk_reward_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        # Max consecutive losses
        max_consecutive_losses = 0
        current_streak = 0
        for r in returns:
            if r < 0:
                current_streak += 1
                max_consecutive_losses = max(max_consecutive_losses, current_streak)
            else:
                current_streak = 0
        
        return {
            'risk_reward_ratio': risk_reward_ratio,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_consecutive_losses': max_consecutive_losses,
        }


class ModelComparator:
    """
    Compares multiple model configurations and logs to MLflow
    Tests different combinations of:
    - Model types (XGBoost, LightGBM, RF, ensemble weights)
    - Feature sets
    - Lookback periods
    - Hyperparameter ranges
    """
    
    def __init__(self, mlflow_tracker=None):
        self.mlflow_tracker = mlflow_tracker
        self.evaluator = TradingEvaluator()
        self.comparison_results: List[Dict] = []
    
    def compare_configurations(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        prices: Optional[np.ndarray] = None,
        configurations: Optional[List[Dict]] = None,
    ) -> pd.DataFrame:
        """
        Compare multiple model configurations
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: List of feature names
            prices: Price series for realistic simulation
            configurations: List of configuration dicts to test
        """
        from sklearn.model_selection import TimeSeriesSplit
        
        if configurations is None:
            configurations = self._get_default_configurations()
        
        results = []
        tscv = TimeSeriesSplit(n_splits=3)
        
        for config in configurations:
            logger.info(f"Testing configuration: {config['name']}")
            
            try:
                # Get model based on configuration
                model = self._create_model(config)
                
                # Cross-validation
                fold_metrics = []
                for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    
                    # Skip if not enough samples or class imbalance
                    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                        continue
                    
                    # LabelEncoder remaps non-sequential labels (e.g. {0,2}) to {0,1}
                    # Required for XGBoost 3.x which needs sequential labels
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    y_train_enc = le.fit_transform(y_train.astype(int))
                    
                    # Train
                    model.fit(X_train, y_train_enc)
                    
                    # Predict and inverse_transform back to original labels
                    y_pred_enc = model.predict(X_test)
                    y_pred = le.inverse_transform(y_pred_enc)
                    y_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                    
                    # Get prices for this fold
                    fold_prices = prices[test_idx] if prices is not None else None
                    
                    # Evaluate
                    metrics = self.evaluator.evaluate(y_test, y_pred, y_prob, fold_prices)
                    fold_metrics.append(metrics.to_dict())
                
                if not fold_metrics:
                    continue
                
                # Average metrics across folds
                avg_metrics = {}
                for key in fold_metrics[0].keys():
                    avg_metrics[key] = np.mean([m[key] for m in fold_metrics])
                
                # Add configuration info
                result = {
                    'config_name': config['name'],
                    'model_type': config['model_type'],
                    **config.get('params', {}),
                    **avg_metrics,
                }
                results.append(result)
                
                # Log to MLflow if available
                if self.mlflow_tracker:
                    self._log_to_mlflow(config, avg_metrics)
                
            except Exception as e:
                logger.warning(f"Failed to test {config['name']}: {e}")
                continue
        
        self.comparison_results = results
        return pd.DataFrame(results)
    
    def _get_default_configurations(self) -> List[Dict]:
        """Get default configurations to test"""
        return [
            # XGBoost variations
            {
                'name': 'xgb_conservative',
                'model_type': 'xgboost',
                'params': {'max_depth': 3, 'learning_rate': 0.01, 'n_estimators': 200}
            },
            {
                'name': 'xgb_balanced',
                'model_type': 'xgboost',
                'params': {'max_depth': 5, 'learning_rate': 0.1, 'n_estimators': 150}
            },
            {
                'name': 'xgb_aggressive',
                'model_type': 'xgboost',
                'params': {'max_depth': 8, 'learning_rate': 0.2, 'n_estimators': 100}
            },
            
            # LightGBM variations
            {
                'name': 'lgb_conservative',
                'model_type': 'lightgbm',
                'params': {'max_depth': 3, 'learning_rate': 0.01, 'n_estimators': 200, 'num_leaves': 15}
            },
            {
                'name': 'lgb_balanced',
                'model_type': 'lightgbm',
                'params': {'max_depth': 5, 'learning_rate': 0.1, 'n_estimators': 150, 'num_leaves': 31}
            },
            {
                'name': 'lgb_aggressive',
                'model_type': 'lightgbm',
                'params': {'max_depth': 8, 'learning_rate': 0.2, 'n_estimators': 100, 'num_leaves': 63}
            },
            
            # Random Forest variations
            {
                'name': 'rf_conservative',
                'model_type': 'random_forest',
                'params': {'n_estimators': 200, 'max_depth': 5, 'min_samples_split': 10}
            },
            {
                'name': 'rf_balanced',
                'model_type': 'random_forest',
                'params': {'n_estimators': 150, 'max_depth': 10, 'min_samples_split': 5}
            },
            {
                'name': 'rf_aggressive',
                'model_type': 'random_forest',
                'params': {'n_estimators': 100, 'max_depth': 20, 'min_samples_split': 2}
            },
            
            # Ensemble variations (different weights)
            {
                'name': 'ensemble_xgb_heavy',
                'model_type': 'ensemble',
                'params': {'xgb_weight': 0.6, 'lgb_weight': 0.25, 'rf_weight': 0.15}
            },
            {
                'name': 'ensemble_balanced',
                'model_type': 'ensemble',
                'params': {'xgb_weight': 0.34, 'lgb_weight': 0.33, 'rf_weight': 0.33}
            },
            {
                'name': 'ensemble_lgb_heavy',
                'model_type': 'ensemble',
                'params': {'xgb_weight': 0.25, 'lgb_weight': 0.6, 'rf_weight': 0.15}
            },
        ]
    
    def _create_model(self, config: Dict):
        """Create model instance from configuration.
        
        Does NOT hardcode objective or num_class — lets sklearn wrappers
        auto-detect from the data. Labels are remapped to sequential by
        LabelEncoder in the CV loop so XGBoost 3.x is happy.
        """
        import xgboost as xgb
        import lightgbm as lgb
        from sklearn.ensemble import RandomForestClassifier, VotingClassifier
        
        model_type = config['model_type']
        params = config.get('params', {})
        
        if model_type == 'xgboost':
            return xgb.XGBClassifier(
                eval_metric='mlogloss',
                verbosity=0,
                **params
            )
        
        elif model_type == 'lightgbm':
            return lgb.LGBMClassifier(
                verbosity=-1,
                **params
            )
        
        elif model_type == 'random_forest':
            return RandomForestClassifier(
                random_state=42,
                **params
            )
        
        elif model_type == 'ensemble':
            xgb_weight = params.get('xgb_weight', 0.34)
            lgb_weight = params.get('lgb_weight', 0.33)
            rf_weight = params.get('rf_weight', 0.33)
            
            estimators = [
                ('xgb', xgb.XGBClassifier(
                    verbosity=0,
                    max_depth=5,
                    n_estimators=100
                )),
                ('lgb', lgb.LGBMClassifier(
                    verbosity=-1,
                    max_depth=5,
                    n_estimators=100
                )),
                ('rf', RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                ))
            ]
            
            return VotingClassifier(
                estimators=estimators,
                voting='soft',
                weights=[xgb_weight, lgb_weight, rf_weight]
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _log_to_mlflow(self, config: Dict, metrics: Dict):
        """Log configuration and metrics to MLflow"""
        if self.mlflow_tracker is None:
            return
        
        try:
            import mlflow
            
            with mlflow.start_run(run_name=config['name'], nested=True):
                # Log parameters
                mlflow.log_param("model_type", config['model_type'])
                mlflow.log_param("config_name", config['name'])
                for key, value in config.get('params', {}).items():
                    mlflow.log_param(key, value)
                
                # Log all metrics
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        mlflow.log_metric(key, value)
                
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")
    
    def get_best_configuration(
        self,
        metric: str = 'sharpe_ratio',
        secondary_metrics: Optional[List[str]] = None
    ) -> Dict:
        """
        Get the best configuration based on specified metric
        
        Args:
            metric: Primary metric to optimize
            secondary_metrics: Tie-breaker metrics
        """
        if not self.comparison_results:
            return {}
        
        df = pd.DataFrame(self.comparison_results)
        
        # Sort by primary metric (descending)
        df_sorted = df.sort_values(metric, ascending=False)
        
        # Apply secondary metrics as tie-breakers
        if secondary_metrics:
            df_sorted = df_sorted.sort_values(
                [metric] + secondary_metrics,
                ascending=[False] * (1 + len(secondary_metrics))
            )
        
        best = df_sorted.iloc[0].to_dict()
        
        logger.info(f"Best configuration: {best['config_name']}")
        logger.info(f"  {metric}: {best.get(metric, 'N/A')}")
        
        return best
    
    def generate_report(self) -> str:
        """Generate comparison report"""
        if not self.comparison_results:
            return "No comparison results available"
        
        df = pd.DataFrame(self.comparison_results)
        
        report = []
        report.append("=" * 60)
        report.append("MODEL COMPARISON REPORT")
        report.append("=" * 60)
        
        # Best by different metrics
        key_metrics = ['sharpe_ratio', 'profit_factor', 'win_rate', 'accuracy', 'f1_score']
        
        for metric in key_metrics:
            if metric in df.columns:
                best_idx = df[metric].idxmax()
                best_config = df.loc[best_idx, 'config_name']
                best_value = df.loc[best_idx, metric]
                report.append(f"\nBest {metric}: {best_config} ({best_value:.4f})")
        
        # Summary table
        report.append("\n" + "-" * 60)
        report.append("SUMMARY TABLE (sorted by Sharpe Ratio)")
        report.append("-" * 60)
        
        summary_cols = ['config_name', 'sharpe_ratio', 'profit_factor', 'win_rate', 'accuracy']
        summary_cols = [c for c in summary_cols if c in df.columns]
        
        summary = df[summary_cols].sort_values('sharpe_ratio', ascending=False)
        report.append(summary.to_string(index=False))
        
        return "\n".join(report)


def run_model_comparison(
    underlying: str,
    days: int = 180,
    mlflow_tracker=None
) -> Tuple[pd.DataFrame, str]:
    """
    Run comprehensive model comparison for a symbol
    
    Args:
        underlying: Symbol to train on
        days: Days of historical data
        mlflow_tracker: MLflow tracker instance
    
    Returns:
        Comparison results DataFrame and report string
    """
    from ml.data_collector import HistoricalDataCollector
    from ml.feature_engineer import FeatureEngineer
    from data.data_fetcher import DataFetcher
    
    logger.info(f"Running model comparison for {underlying}")
    
    # Collect data
    data_fetcher = DataFetcher()
    collector = HistoricalDataCollector(data_fetcher)
    feature_engineer = FeatureEngineer()
    
    # Get historical data
    df = collector.collect_historical_data(underlying, days=days)
    
    if df is None or len(df) < 50:
        logger.error(f"Insufficient data for {underlying}")
        return pd.DataFrame(), "Insufficient data"
    
    # Extract features
    X, y, feature_names = feature_engineer.extract_features_batch(df)
    
    if len(X) < 30:
        logger.error(f"Insufficient samples after feature extraction")
        return pd.DataFrame(), "Insufficient samples"
    
    # Get prices for realistic simulation
    prices = df['close'].values[-len(y):] if 'close' in df.columns else None
    
    # Run comparison
    comparator = ModelComparator(mlflow_tracker)
    results = comparator.compare_configurations(X, y, feature_names, prices)
    
    report = comparator.generate_report()
    
    # Get best config
    best = comparator.get_best_configuration(
        metric='sharpe_ratio',
        secondary_metrics=['profit_factor', 'win_rate']
    )
    
    return results, report, best
