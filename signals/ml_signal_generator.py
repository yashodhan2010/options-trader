"""
ML Signal Generator - ML-Only Signal Generation

All trading signals are derived from ML model predictions.
No rule-based signal generation - ML drives the entire signal flow.
"""
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

from data.data_fetcher import data_fetcher
from strategies.catalogue import StrategyCatalogue
from strategies.base_strategy import StrategySignal, StrategyType, TradeDirection, OptionLeg
from config.settings import (
    UNDERLYING_ASSETS, STRATEGY_CONFIG, ML_CONFIG,
    WATCHLIST, WATCHLIST_SYMBOLS, is_in_watchlist, get_watchlist_assets
)
from core.logger import logger
from core.utils import is_trading_allowed


def _get_ml_components():
    """Load ML components."""
    try:
        from ml.predictor import get_predictor
        from ml.feature_engineer import get_feature_engineer
        from ml.guardrails import get_guardrails
        
        return get_predictor(), get_feature_engineer(), get_guardrails()
    except ImportError as e:
        logger.error(f"Failed to import ML components: {e}")
        return None, None, None


class MLSignalGenerator:
    """
    ML-Only Signal Generator.
    
    All signals are driven by ML model predictions:
    1. ML model predicts direction (BULLISH/BEARISH/NEUTRAL) with confidence
    2. Direction maps to appropriate strategy type
    3. Strategy execution builds the actual option legs
    
    No rule-based signal generation - ML is the sole source of truth.
    """
    
    # Direction to strategy mapping based on IV regime
    # PRIORITY: Credit spreads first (sell premium, time decay in our favor)
    # Debit strategies only in LOW_IV where premium is cheap to buy
    DIRECTION_STRATEGY_MAP = {
        "BULLISH": {
            "LOW_IV": [StrategyType.BULL_PUT_SPREAD, StrategyType.BULL_CALL_SPREAD],
            "NORMAL": [StrategyType.BULL_PUT_SPREAD, StrategyType.BULL_CALL_SPREAD],
            "HIGH_IV": [StrategyType.BULL_PUT_SPREAD, StrategyType.SHORT_PUT],
        },
        "BEARISH": {
            "LOW_IV": [StrategyType.BEAR_CALL_SPREAD, StrategyType.BEAR_PUT_SPREAD],
            "NORMAL": [StrategyType.BEAR_CALL_SPREAD, StrategyType.BEAR_PUT_SPREAD],
            "HIGH_IV": [StrategyType.BEAR_CALL_SPREAD, StrategyType.SHORT_CALL],
        },
        "NEUTRAL": {
            "LOW_IV": [StrategyType.IRON_CONDOR],
            "NORMAL": [StrategyType.IRON_CONDOR],
            "HIGH_IV": [StrategyType.IRON_CONDOR, StrategyType.SHORT_CALL, StrategyType.SHORT_PUT],
        },
    }
    
    def __init__(self, underlyings: List[str] = None):
        """
        Initialize the ML signal generator.
        
        Args:
            underlyings: List of underlying assets to monitor.
                        If None, uses ML training symbols.
        """
        # Determine which assets to trade
        if underlyings:
            self.underlyings = underlyings
        elif WATCHLIST.get("enabled", False) and WATCHLIST_SYMBOLS:
            # Merge watchlist stocks with index symbols from UNDERLYING_ASSETS
            self.underlyings = list(WATCHLIST_SYMBOLS)  # Copy to avoid mutating
            for idx_symbol in UNDERLYING_ASSETS:
                if idx_symbol not in self.underlyings:
                    self.underlyings.append(idx_symbol)
            logger.info(f"Using watchlist + indices: {self.underlyings}")
        else:
            # Use ML training symbols as default
            self.underlyings = ML_CONFIG.get("training_symbols", list(UNDERLYING_ASSETS.keys()))
        
        self.catalogues: Dict[str, StrategyCatalogue] = {}
        self.last_signals: Dict[str, List[StrategySignal]] = {}
        self.signal_history: List[Dict] = []
        
        # ML components (lazy loaded)
        self._predictor = None
        self._feature_engineer = None
        self._guardrails = None
        self._ml_initialized = False
        
        # Configuration
        self.min_confidence = ML_CONFIG.get("min_confidence_for_trade", 0.55)
        self.model_loaded = False
        
        # Initialize strategy catalogues
        for underlying in self.underlyings:
            self.catalogues[underlying] = StrategyCatalogue(underlying)
        
        logger.info("ML-Only Signal Generator initialized")
        logger.info(f"Trading symbols: {self.underlyings}")
    
    def _init_ml(self) -> bool:
        """
        Initialize ML components.
        
        Returns:
            True if ML is ready for predictions
        """
        if self._ml_initialized:
            return self.model_loaded
        
        self._predictor, self._feature_engineer, self._guardrails = _get_ml_components()
        self._ml_initialized = True
        
        if not self._predictor or not self._feature_engineer:
            logger.error("ML components not available - no signals will be generated")
            return False
        
        # Try to load the model
        if self._predictor.load_model():
            self.model_loaded = True
            logger.info(f"ML model loaded: {self._predictor.model_version}")
            return True
        else:
            logger.error("No trained ML model found - train a model first")
            return False
    
    def generate_signals(
        self,
        underlying: Optional[str] = None,
        strategy_type: Optional[StrategyType] = None,
    ) -> List[StrategySignal]:
        """
        Generate trading signals based on ML predictions.
        
        Args:
            underlying: Specific underlying to analyze (None for all)
            strategy_type: Force specific strategy (overrides ML suggestion)
            
        Returns:
            List of generated signals
        """
        if not is_trading_allowed():
            logger.warning("Trading not allowed at this time")
            return []
        
        # Initialize ML if needed
        if not self._init_ml():
            logger.error("ML not ready - cannot generate signals")
            return []
        
        targets = [underlying] if underlying else self.underlyings
        all_signals = []
        
        for target in targets:
            try:
                signals = self._generate_ml_signals(target, strategy_type)
                all_signals.extend(signals)
                self.last_signals[target] = signals
                
                # Record in history
                for signal in signals:
                    self.signal_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "underlying": target,
                        "signal": signal.to_dict(),
                        "source": "ML",
                    })
                    
            except Exception as e:
                logger.error(f"Error generating ML signals for {target}: {e}")
                import traceback
                traceback.print_exc()
        
        # Sort all signals by confidence
        all_signals.sort(key=lambda s: s.confidence, reverse=True)
        
        logger.info(f"Generated {len(all_signals)} ML signals")
        return all_signals
    
    def _generate_ml_signals(
        self,
        underlying: str,
        force_strategy: Optional[StrategyType] = None,
    ) -> List[StrategySignal]:
        """
        Generate signals for a single underlying using ML prediction.
        
        Args:
            underlying: The underlying asset
            force_strategy: Force a specific strategy type
            
        Returns:
            List of signals for this underlying
        """
        logger.info(f"ML Analysis: {underlying}...")
        
        # Fetch market data
        spot = data_fetcher.get_spot_price(underlying)
        if not spot:
            logger.warning(f"Could not get spot price for {underlying}")
            return []
        
        # Get options chain
        options_chain = data_fetcher.get_options_chain(
            underlying,
            num_strikes=STRATEGY_CONFIG.get("otm_offset", 1) + 5,
        )
        
        if options_chain.empty:
            logger.warning(f"Empty options chain for {underlying}")
            return []
        
        # Get market metrics
        oi_data = data_fetcher.get_oi_data(underlying)
        volatility = data_fetcher.get_volatility_metrics(underlying)
        historical = data_fetcher.get_historical_analysis(underlying, days=30)
        
        market_data = {
            "spot": spot,
            "oi_data": oi_data,
            "volatility": volatility,
            "historical": historical,
        }
        
        # Extract features for ML prediction
        features = self._feature_engineer.extract_features(
            underlying=underlying,
            spot_price=spot,
            historical_data=historical.get("df") if isinstance(historical, dict) else historical,
            options_chain=options_chain,
            oi_analysis=oi_data,
            volatility_data=volatility,
        )
        
        if not features:
            logger.warning(f"Could not extract features for {underlying}")
            return []
        
        # Get ML prediction
        prediction = self._predictor.predict(features, underlying)
        
        if prediction is None:
            logger.warning(f"ML prediction failed for {underlying}")
            return []
        
        logger.info(
            f"ML Prediction for {underlying}: {prediction.direction} "
            f"(confidence: {prediction.confidence:.1%})"
        )
        
        # Check confidence threshold
        if prediction.confidence < self.min_confidence:
            logger.info(
                f"ML confidence {prediction.confidence:.1%} below threshold "
                f"{self.min_confidence:.1%} - skipping {underlying}"
            )
            return []
        
        # Trend confirmation: validate ML direction against recent prediction history
        if not self._confirm_trend(underlying, prediction.direction):
            logger.info(
                f"Trend confirmation failed for {underlying} ({prediction.direction}) - skipping"
            )
            return []
        
        # Determine strategy based on ML direction
        if force_strategy:
            strategy_types = [force_strategy]
        else:
            strategy_types = self._get_strategies_for_direction(
                prediction.direction,
                volatility.get("volatility_regime", "NORMAL")
            )
        
        # Generate signals for selected strategies
        signals = []
        catalogue = self.catalogues.get(underlying)
        
        if not catalogue:
            return []
        
        for strategy_type in strategy_types:
            strategy = catalogue.get_strategy(strategy_type)
            if not strategy:
                logger.info(f"  Strategy {strategy_type.value} not found in catalogue for {underlying}")
                continue
            
            try:
                # ML is driving the signal - skip strategy-level OI sentiment
                # and confidence gates (ML already validated direction & confidence)
                strategy.ml_override = True
                
                # Analyze with strategy
                signal = strategy.analyze(options_chain, market_data)
                
                # Reset override
                strategy.ml_override = False
                
                if signal:
                    # Override confidence with ML confidence
                    ml_signal = self._create_ml_signal(signal, prediction)
                    signals.append(ml_signal)
                    logger.info(
                        f"  Signal generated: {strategy_type.value} for {underlying} "
                        f"(ML confidence: {prediction.confidence:.1%})"
                    )
                else:
                    logger.info(
                        f"  Strategy {strategy_type.value} returned no signal for {underlying} "
                        f"(strategy conditions not met)"
                    )
                    
            except Exception as e:
                logger.error(f"Strategy {strategy_type.value} failed: {e}")
                import traceback
                traceback.print_exc()
                strategy.ml_override = False
        
        return signals
    
    def _confirm_trend(self, underlying: str, ml_direction: str) -> bool:
        """
        Validate ML prediction against recent feature snapshot labels from the database.
        Requires 60%+ of recent labels to align with the ML direction.
        
        Args:
            underlying: The underlying asset
            ml_direction: ML predicted direction (BULLISH/BEARISH/NEUTRAL)
            
        Returns:
            True if trend is confirmed or no history available
        """
        if ml_direction == "NEUTRAL":
            return True  # Neutral doesn't need trend confirmation
        
        try:
            import sqlite3
            import os
            
            db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "trading_bot.db")
            if not os.path.exists(db_path):
                return True  # No DB, skip confirmation
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ml_feature_snapshots'")
            if not cursor.fetchone():
                conn.close()
                return True
            
            # Get recent labels for this underlying (last 5 trading days)
            cursor.execute("""
                SELECT label_direction FROM ml_feature_snapshots 
                WHERE underlying = ? AND label_direction IS NOT NULL AND label_direction != 'NEUTRAL'
                ORDER BY snapshot_time DESC LIMIT 5
            """, (underlying,))
            
            rows = cursor.fetchall()
            conn.close()
            
            if len(rows) < 3:
                return True  # Not enough history, allow the trade
            
            labels = [r[0] for r in rows]
            
            # Map ML direction to expected label
            expected_label = "UP" if ml_direction == "BULLISH" else "DOWN"
            
            # Count alignment
            aligned = sum(1 for l in labels if l == expected_label)
            alignment_pct = aligned / len(labels)
            
            logger.info(
                f"Trend confirmation for {underlying}: {aligned}/{len(labels)} recent labels = "
                f"{expected_label} ({alignment_pct:.0%}), ML says {ml_direction}"
            )
            
            return alignment_pct >= 0.6  # Require 60%+ alignment
            
        except Exception as e:
            logger.warning(f"Trend confirmation check failed: {e}")
            return True  # On error, allow the trade
    
    def _get_strategies_for_direction(
        self,
        direction: str,
        iv_regime: str,
    ) -> List[StrategyType]:
        """
        Get appropriate strategies based on ML direction and IV regime.
        
        Args:
            direction: ML predicted direction (BULLISH/BEARISH/NEUTRAL)
            iv_regime: Current IV regime (LOW_IV/NORMAL/HIGH_IV)
            
        Returns:
            List of suitable strategy types
        """
        # Normalize IV regime
        if "LOW" in iv_regime.upper() or "DEPRESS" in iv_regime.upper():
            regime_key = "LOW_IV"
        elif "HIGH" in iv_regime.upper() or "ELEVAT" in iv_regime.upper():
            regime_key = "HIGH_IV"
        else:
            regime_key = "NORMAL"
        
        # Get strategies from mapping
        direction_map = self.DIRECTION_STRATEGY_MAP.get(direction, {})
        strategies = direction_map.get(regime_key, [])
        
        # Filter by enabled strategies
        enabled = STRATEGY_CONFIG.get("enabled_strategies", [])
        
        filtered = []
        for strategy_type in strategies:
            if strategy_type.value in enabled:
                filtered.append(strategy_type)
        
        # If no enabled strategies match, return first available
        if not filtered and strategies:
            filtered = [strategies[0]]
        
        return filtered
    
    def _create_ml_signal(
        self,
        base_signal: StrategySignal,
        prediction,
    ) -> StrategySignal:
        """
        Create a signal with ML prediction as the source of truth.
        
        Args:
            base_signal: Strategy-generated signal with option legs
            prediction: ML prediction result
            
        Returns:
            Enhanced signal with ML confidence
        """
        # Create new signal with ML confidence using actual StrategySignal fields
        ml_signal = StrategySignal(
            underlying=base_signal.underlying,
            strategy_type=base_signal.strategy_type,
            confidence=prediction.confidence,  # Use ML confidence
            legs=base_signal.legs,
            entry_time=base_signal.entry_time,
            expected_profit=base_signal.expected_profit,
            max_loss=base_signal.max_loss,
            stop_loss=base_signal.stop_loss,
            target=base_signal.target,
            rationale=f"[ML Signal] {prediction.direction} ({prediction.confidence:.1%}) - {base_signal.rationale}",
            metrics=base_signal.metrics.copy() if base_signal.metrics else {},
        )
        
        # Add ML metadata to metrics
        ml_signal.metrics["ml_direction"] = prediction.direction
        ml_signal.metrics["ml_confidence"] = prediction.confidence
        ml_signal.metrics["ml_probabilities"] = prediction.probabilities
        ml_signal.metrics["ml_model_version"] = prediction.model_version
        ml_signal.metrics["ml_model_type"] = prediction.model_type
        ml_signal.metrics["signal_source"] = "ML"
        
        return ml_signal
    
    def get_best_signal(self, underlying: Optional[str] = None) -> Optional[StrategySignal]:
        """
        Get the single best ML signal.
        
        Args:
            underlying: Specific underlying (None for any)
            
        Returns:
            Best signal or None
        """
        signals = self.generate_signals(underlying)
        return signals[0] if signals else None
    
    def get_market_overview(self, underlying: str) -> Dict[str, Any]:
        """
        Get market overview with ML prediction.
        
        Args:
            underlying: The underlying asset
            
        Returns:
            Dictionary with market overview and ML prediction
        """
        # Initialize ML if needed
        self._init_ml()
        
        spot = data_fetcher.get_spot_price(underlying)
        oi_data = data_fetcher.get_oi_data(underlying)
        volatility = data_fetcher.get_volatility_metrics(underlying)
        historical = data_fetcher.get_historical_analysis(underlying, days=30)
        options_chain = data_fetcher.get_options_chain(underlying)
        
        # Get ML prediction
        ml_prediction = None
        if self.model_loaded and self._feature_engineer:
            features = self._feature_engineer.extract_features(
                underlying=underlying,
                spot_price=spot,
                historical_data=historical.get("df") if isinstance(historical, dict) else historical,
                options_chain=options_chain,
                oi_analysis=oi_data,
                volatility_data=volatility,
            )
            
            if features:
                prediction = self._predictor.predict(features, underlying)
                if prediction:
                    ml_prediction = {
                        "direction": prediction.direction,
                        "confidence": prediction.confidence,
                        "probabilities": prediction.probabilities,
                        "model_version": prediction.model_version,
                    }
        
        return {
            "underlying": underlying,
            "spot": spot,
            "timestamp": datetime.now().isoformat(),
            "oi_analysis": {
                "pcr": oi_data.get("pcr"),
                "total_call_oi": oi_data.get("total_call_oi"),
                "total_put_oi": oi_data.get("total_put_oi"),
                "max_pain": oi_data.get("max_pain"),
                "sentiment": oi_data.get("sentiment"),
            },
            "volatility": {
                "hv_20": volatility.get("hv_20"),
                "atm_iv": volatility.get("atm_iv"),
                "regime": volatility.get("volatility_regime"),
            },
            "ml_prediction": ml_prediction,
            "recommended_strategies": self._get_strategies_for_direction(
                ml_prediction["direction"] if ml_prediction else "NEUTRAL",
                volatility.get("volatility_regime", "NORMAL")
            ) if ml_prediction else [],
        }
    
    def get_model_status(self) -> Dict[str, Any]:
        """
        Get ML model status.
        
        Returns:
            Dictionary with model status information
        """
        self._init_ml()
        
        if not self._predictor:
            return {
                "status": "error",
                "message": "ML predictor not available",
                "model_loaded": False,
            }
        
        return {
            "status": "ready" if self.model_loaded else "no_model",
            "model_loaded": self.model_loaded,
            "model_version": self._predictor.model_version if self.model_loaded else None,
            "model_type": self._predictor.model_type if self.model_loaded else None,
            "min_confidence": self.min_confidence,
            "underlyings": self.underlyings,
        }
    
    def clear_history(self) -> None:
        """Clear signal history."""
        self.signal_history = []
        self.last_signals = {}
        logger.info("Signal history cleared")


# Singleton instance
ml_signal_generator = MLSignalGenerator()

# Alias for backward compatibility - signal_generator now points to ML version
signal_generator = ml_signal_generator
