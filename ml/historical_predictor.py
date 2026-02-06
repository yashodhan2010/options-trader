"""
Historical Model Predictor

Uses models trained on historical bhavcopy data for live prediction.
Loads per-symbol models and provides a unified prediction interface.

IMPORTANT: Uses UNIFIED features that are compatible with both:
- Historical training (from NSE bhavcopy)
- Live prediction (from FeatureEngineer)
"""

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from core.logger import logger
from ml.unified_features import (
    UnifiedFeatureDefinition,
    LiveFeatureAdapter,
    get_unified_feature_names
)


class HistoricalModelPredictor:
    """
    Predictor using models trained on historical NSE bhavcopy data.
    
    Features:
    - Loads per-symbol models trained via FullPipelineTrainer
    - Accepts BOTH unified features and live FeatureEngineer output
    - Provides ensemble predictions across multiple models
    - Calculates prediction confidence
    - Falls back gracefully when model unavailable
    """
    
    def __init__(
        self,
        model_dir: str = "data/ml_models",
        min_confidence: float = 0.6
    ):
        self.model_dir = Path(model_dir)
        self.min_confidence = min_confidence
        self.models: Dict[str, Dict] = {}
        self.feature_names: List[str] = get_unified_feature_names()
        self.live_adapter = LiveFeatureAdapter()
        
        self._load_models()
    
    def _load_models(self):
        """Load all available per-symbol models."""
        if not self.model_dir.exists():
            logger.warning(f"Model directory not found: {self.model_dir}")
            return
        
        # Find latest models for each symbol
        model_files = list(self.model_dir.glob("*_model_*.joblib"))
        
        # Group by symbol (first part before _model_)
        symbol_models: Dict[str, List[Path]] = {}
        
        for f in model_files:
            parts = f.stem.split("_model_")
            if len(parts) >= 2:
                symbol = parts[0]
                if symbol not in symbol_models:
                    symbol_models[symbol] = []
                symbol_models[symbol].append(f)
        
        # Load latest model for each symbol
        for symbol, files in symbol_models.items():
            latest = max(files, key=lambda p: p.stat().st_mtime)
            
            try:
                model_data = joblib.load(latest)
                self.models[symbol] = model_data
                
                if not self.feature_names and "feature_names" in model_data:
                    self.feature_names = model_data["feature_names"]
                
                logger.info(f"Loaded model for {symbol}: {latest.name}")
                
            except Exception as e:
                logger.error(f"Failed to load model {latest}: {e}")
        
        logger.info(f"Loaded {len(self.models)} symbol models")
        
        # Load training summary if available
        summaries = list(self.model_dir.glob("training_summary_*.json"))
        if summaries:
            latest_summary = max(summaries, key=lambda p: p.stat().st_mtime)
            try:
                with open(latest_summary) as f:
                    self.training_summary = json.load(f)
                logger.info(f"Training summary: {latest_summary.name}")
            except:
                self.training_summary = {}
    
    def get_available_symbols(self) -> List[str]:
        """Get list of symbols with trained models."""
        return list(self.models.keys())
    
    def get_model_metrics(self, symbol: str) -> Optional[Dict]:
        """Get training metrics for a symbol's model."""
        if symbol not in self.models:
            return None
        return self.models[symbol].get("metrics", {})
    
    def prepare_features(self, market_data: Dict, is_live_features: bool = False) -> Optional[np.ndarray]:
        """
        Prepare features from market data.
        
        Args:
            market_data: Dictionary with feature values
            is_live_features: If True, adapts from FeatureEngineer format
            
        Returns:
            Feature array in correct order
        """
        if not self.feature_names:
            logger.error("No feature names loaded")
            return None
        
        # If live features, adapt them to unified format
        if is_live_features:
            market_data = self.live_adapter.adapt(market_data)
        
        features = []
        for name in self.feature_names:
            value = market_data.get(name, 0)
            if pd.isna(value) or np.isinf(value):
                value = 0
            features.append(value)
        
        return np.array([features])
    
    def predict(
        self,
        symbol: str,
        market_data: Dict,
        return_proba: bool = True,
        is_live_features: bool = False
    ) -> Tuple[Optional[int], float]:
        """
        Make prediction for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "NIFTY", "RELIANCE")
            market_data: Dictionary with market features
            return_proba: Whether to return probability
            is_live_features: If True, market_data is from FeatureEngineer
            
        Returns:
            Tuple of (prediction, confidence)
            prediction: 1 = bullish, 0 = bearish, None = no signal
            confidence: Probability confidence [0, 1]
        """
        if symbol not in self.models:
            logger.warning(f"No model available for {symbol}")
            return None, 0.0
        
        model_data = self.models[symbol]
        model = model_data.get("model")
        
        if model is None:
            return None, 0.0
        
        # Prepare features (with adaptation if needed)
        X = self.prepare_features(market_data, is_live_features=is_live_features)
        if X is None:
            return None, 0.0
        
        try:
            if return_proba and hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                # proba[1] = probability of bullish (label=1)
                confidence = max(proba)
                prediction = 1 if proba[1] > 0.5 else 0
                
                # Apply minimum confidence threshold
                if confidence < self.min_confidence:
                    return None, confidence
                
                return prediction, proba[1] if prediction == 1 else proba[0]
            else:
                prediction = int(model.predict(X)[0])
                return prediction, 0.5  # No confidence without proba
                
        except Exception as e:
            logger.error(f"Prediction error for {symbol}: {e}")
            return None, 0.0
    
    def predict_from_live(
        self,
        symbol: str,
        feature_set
    ) -> Tuple[Optional[int], float]:
        """
        Convenience method to predict from FeatureEngineer output.
        
        Args:
            symbol: Trading symbol
            feature_set: FeatureSet from FeatureEngineer or dict
            
        Returns:
            Tuple of (prediction, confidence)
        """
        # Convert FeatureSet to dict if needed
        if hasattr(feature_set, 'to_dict'):
            features = feature_set.to_dict()
        elif hasattr(feature_set, 'features'):
            features = feature_set.features
        else:
            features = feature_set
        
        return self.predict(symbol, features, is_live_features=True)
    
    def predict_ensemble(
        self,
        symbol: str,
        market_data: Dict
    ) -> Dict[str, Any]:
        """
        Make ensemble prediction using symbol model + any related models.
        
        Returns detailed prediction info.
        """
        result = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "prediction": None,
            "direction": None,
            "confidence": 0.0,
            "model_available": symbol in self.models,
            "details": {}
        }
        
        prediction, confidence = self.predict(symbol, market_data)
        
        result["prediction"] = prediction
        result["confidence"] = confidence
        
        if prediction is not None:
            result["direction"] = "BULLISH" if prediction == 1 else "BEARISH"
            
            # Add model metrics
            metrics = self.get_model_metrics(symbol)
            if metrics:
                result["details"]["model_accuracy"] = metrics.get("accuracy")
                result["details"]["model_f1"] = metrics.get("f1")
        
        return result
    
    def get_signal_strength(
        self,
        symbol: str,
        market_data: Dict
    ) -> Tuple[str, float]:
        """
        Get signal strength for options trading.
        
        Returns:
            Tuple of (signal_type, strength)
            signal_type: "CE" for calls, "PE" for puts, "NEUTRAL"
            strength: Signal strength [0, 1]
        """
        prediction, confidence = self.predict(symbol, market_data)
        
        if prediction is None:
            return "NEUTRAL", 0.0
        
        if prediction == 1:
            return "CE", confidence  # Bullish -> Buy calls
        else:
            return "PE", confidence  # Bearish -> Buy puts


# Global instance
_predictor: Optional[HistoricalModelPredictor] = None


def get_historical_predictor() -> HistoricalModelPredictor:
    """Get global historical predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = HistoricalModelPredictor()
    return _predictor


# CLI interface
def main():
    """Test the historical model predictor."""
    predictor = HistoricalModelPredictor()
    
    print("\n" + "=" * 60)
    print("HISTORICAL MODEL PREDICTOR")
    print("=" * 60)
    
    print(f"\nAvailable symbols: {predictor.get_available_symbols()}")
    
    # Print metrics for each model
    print("\nModel Metrics:")
    print("-" * 40)
    
    for symbol in predictor.get_available_symbols():
        metrics = predictor.get_model_metrics(symbol)
        if metrics:
            print(f"{symbol:12} | Acc: {metrics.get('accuracy', 0):.1%} | F1: {metrics.get('f1', 0):.1%}")
    
    # Test prediction with sample data
    print("\n\nSample Prediction Test:")
    print("-" * 40)
    
    # Create dummy market data
    sample_data = {
        "open": 22000,
        "high": 22100,
        "low": 21900,
        "close": 22050,
        "volume": 1000000,
        "fut_oi": 500000,
        "fut_oi_change": 10000,
        "call_oi": 1000000,
        "put_oi": 800000,
        "call_oi_change": 20000,
        "put_oi_change": 15000,
        "total_oi": 1800000,
        "pcr_oi": 0.8,
        "pcr_volume": 0.75,
        "atm_call_oi": 50000,
        "atm_put_oi": 45000,
        "atm_pcr": 0.9,
        "otm_call_oi": 200000,
        "otm_put_oi": 180000,
        "max_pain": 22000,
        "max_pain_distance": 0.002,
        "returns": 0.001,
        "log_returns": 0.001,
        "range_pct": 0.009,
        "rsi": 55,
        "macd": 50,
        "macd_signal": 45,
        "macd_hist": 5,
        "bb_width": 0.02,
        "bb_position": 0.6,
        "iv_proxy": 0.15,
        "delta_proxy": 0.55,
        "gamma_proxy": 0.02,
        "theta_proxy": -0.01,
        "vega_proxy": 0.03,
        "iv_percentile": 50,
        "iv_rank": 0.5,
    }
    
    # Fill missing features with zeros
    for name in predictor.feature_names:
        if name not in sample_data:
            sample_data[name] = 0.0
    
    for symbol in predictor.get_available_symbols()[:3]:  # Test first 3
        result = predictor.predict_ensemble(symbol, sample_data)
        print(f"\n{symbol}:")
        print(f"  Direction:  {result['direction'] or 'NO SIGNAL'}")
        print(f"  Confidence: {result['confidence']:.1%}")


if __name__ == "__main__":
    main()
