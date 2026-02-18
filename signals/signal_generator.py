"""
Signal Generator - Generates trading signals based on market analysis

Enhanced with ML-based confidence adjustment and signal filtering.
"""
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

from data.data_fetcher import data_fetcher
from strategies.catalogue import StrategyCatalogue
from strategies.base_strategy import StrategySignal, StrategyType
from config.settings import (
    UNDERLYING_ASSETS, STRATEGY_CONFIG, ML_CONFIG,
    WATCHLIST, WATCHLIST_SYMBOLS, is_in_watchlist, get_watchlist_assets
)
from core.logger import logger
from core.utils import is_trading_allowed


def _get_ml_components():
    """Lazy load ML components to avoid circular imports."""
    try:
        if not ML_CONFIG.get("enabled", False):
            return None, None, None
        
        from ml.predictor import get_predictor
        from ml.feature_engineer import get_feature_engineer
        from ml.guardrails import get_guardrails
        
        return get_predictor(), get_feature_engineer(), get_guardrails()
    except ImportError:
        return None, None, None


class SignalGenerator:
    """
    Generates trading signals by analyzing market data and running strategies.
    
    Enhanced with ML-based confidence adjustment and signal filtering.
    """
    
    def __init__(self, underlyings: List[str] = None):
        """
        Initialize the signal generator.
        
        Args:
            underlyings: List of underlying assets to monitor.
                        If None, uses watchlist if enabled, else UNDERLYING_ASSETS.
        """
        # Determine which assets to trade
        if underlyings:
            self.underlyings = underlyings
        elif WATCHLIST.get("enabled", False) and WATCHLIST_SYMBOLS:
            # Use watchlist symbols
            self.underlyings = WATCHLIST_SYMBOLS
            logger.info(f"Using watchlist: {self.underlyings}")
        else:
            # Fall back to default indices
            self.underlyings = list(UNDERLYING_ASSETS.keys())
        
        self.catalogues: Dict[str, StrategyCatalogue] = {}
        self.last_signals: Dict[str, List[StrategySignal]] = {}
        self.signal_history: List[Dict] = []
        
        # ML integration
        self.ml_enabled = ML_CONFIG.get("enabled", False)
        self.ml_confidence_weight = ML_CONFIG.get("confidence_weight", 0.5)
        self._predictor = None
        self._feature_engineer = None
        self._guardrails = None
        
        # Initialize strategy catalogues
        for underlying in self.underlyings:
            self.catalogues[underlying] = StrategyCatalogue(underlying)
        
        if self.ml_enabled:
            logger.info("ML-enhanced signal generation enabled")
    
    def _init_ml(self):
        """Initialize ML components on first use."""
        if self._predictor is None and self.ml_enabled:
            self._predictor, self._feature_engineer, self._guardrails = _get_ml_components()
    
    def generate_signals(
        self,
        underlying: Optional[str] = None,
        strategy_type: Optional[StrategyType] = None,
    ) -> List[StrategySignal]:
        """
        Generate trading signals for the specified underlying.
        
        Args:
            underlying: Specific underlying to analyze (None for all)
            strategy_type: Specific strategy to use (None for all)
            
        Returns:
            List of generated signals
        """
        if not is_trading_allowed():
            logger.warning("Trading not allowed at this time")
            return []
        
        targets = [underlying] if underlying else self.underlyings
        all_signals = []
        
        for target in targets:
            try:
                signals = self._analyze_underlying(target, strategy_type)
                all_signals.extend(signals)
                self.last_signals[target] = signals
                
                # Record in history
                for signal in signals:
                    self.signal_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "underlying": target,
                        "signal": signal.to_dict(),
                    })
                    
            except Exception as e:
                logger.error(f"Error generating signals for {target}: {e}")
        
        # Sort all signals by confidence
        all_signals.sort(key=lambda s: s.confidence, reverse=True)
        
        # Apply ML enhancement if enabled
        if self.ml_enabled and all_signals:
            all_signals = self._enhance_signals_with_ml(all_signals, targets)
        
        logger.info(f"Generated {len(all_signals)} total signals")
        return all_signals
    
    def _enhance_signals_with_ml(
        self,
        signals: List[StrategySignal],
        underlyings: List[str]
    ) -> List[StrategySignal]:
        """
        Enhance signals with ML predictions.
        
        Args:
            signals: Original rule-based signals
            underlyings: List of underlyings analyzed
            
        Returns:
            ML-enhanced signals
        """
        self._init_ml()
        
        if not self._predictor or not self._feature_engineer:
            return signals
        
        enhanced_signals = []
        
        for signal in signals:
            try:
                # Get market data for feature extraction
                underlying = signal.underlying
                spot = data_fetcher.get_spot_price(underlying)
                
                if not spot:
                    enhanced_signals.append(signal)
                    continue
                
                oi_data = data_fetcher.get_oi_data(underlying)
                volatility = data_fetcher.get_volatility_metrics(underlying)
                historical = data_fetcher.get_historical_analysis(underlying, days=30)
                options_chain = data_fetcher.get_options_chain(underlying)
                
                # Extract features
                features = self._feature_engineer.extract_features(
                    underlying=underlying,
                    spot_price=spot,
                    historical_data=historical.get("df") if isinstance(historical, dict) else historical,
                    options_chain=options_chain,
                    oi_analysis=oi_data,
                    volatility_data=volatility,
                )
                
                # Get ML prediction with guardrails
                prediction = self._predictor.predict_with_guardrails(
                    features=features,
                    underlying=underlying,
                    strategy_type=signal.strategy_type.value,
                    rule_confidence=signal.confidence
                )
                
                # Blend confidences
                original_confidence = signal.confidence
                blended_confidence = prediction.blended_confidence
                
                # Create enhanced signal (copy with updated confidence)
                enhanced_signal = StrategySignal(
                    underlying=signal.underlying,
                    strategy_type=signal.strategy_type,
                    legs=signal.legs,
                    entry_time=signal.entry_time,
                    confidence=blended_confidence,
                    expected_profit=signal.expected_profit,
                    max_loss=signal.max_loss,
                    stop_loss=signal.stop_loss,
                    target=signal.target,
                    rationale=signal.rationale + f" [ML: {prediction.confidence:.1%}, blend: {blended_confidence:.1%}]",
                    metrics=signal.metrics.copy() if signal.metrics else {},
                )
                
                # Store ML metadata in metrics
                enhanced_signal.metrics["ml_confidence"] = prediction.confidence
                enhanced_signal.metrics["ml_direction"] = prediction.direction
                enhanced_signal.metrics["ml_model_version"] = prediction.model_version
                enhanced_signal.metrics["original_confidence"] = original_confidence
                enhanced_signal.metrics["ml_feature_importance"] = prediction.feature_importance
                
                enhanced_signals.append(enhanced_signal)
                
                logger.debug(
                    f"ML enhanced {signal.strategy_type.value}: "
                    f"rule={original_confidence:.1%} -> blend={blended_confidence:.1%}"
                )
                
            except Exception as e:
                logger.warning(f"ML enhancement failed for signal: {e}")
                enhanced_signals.append(signal)
        
        # Re-sort by enhanced confidence
        enhanced_signals.sort(key=lambda s: s.confidence, reverse=True)
        
        return enhanced_signals
    
    def _analyze_underlying(
        self,
        underlying: str,
        strategy_type: Optional[StrategyType] = None,
    ) -> List[StrategySignal]:
        """
        Analyze a single underlying and generate signals.
        
        Args:
            underlying: The underlying asset
            strategy_type: Specific strategy to use
            
        Returns:
            List of signals for this underlying
        """
        logger.info(f"Analyzing {underlying}...")
        
        # Fetch market data
        spot = data_fetcher.get_spot_price(underlying)
        if not spot:
            logger.warning(f"Could not get spot price for {underlying}")
            return []
        
        # Get options chain
        options_chain = data_fetcher.get_options_chain(
            underlying,
            num_strikes=15,  # 15 strikes each side: enough depth for credit spread OTM + hedge
        )
        
        if options_chain.empty:
            logger.warning(f"Empty options chain for {underlying}")
            return []
        
        # Get market metrics
        oi_data = data_fetcher.get_oi_data(underlying)
        volatility = data_fetcher.get_volatility_metrics(underlying)
        
        # Get historical analysis for improved confidence
        historical = data_fetcher.get_historical_analysis(underlying, days=30)
        
        metrics = {
            "spot": spot,
            "oi_data": oi_data,
            "volatility": volatility,
            "historical": historical,
        }
        
        logger.debug(f"Metrics for {underlying}: OI sentiment={oi_data.get('sentiment')}, "
                    f"IV regime={volatility.get('volatility_regime')}, "
                    f"Trend={historical.get('trend')}, RSI={historical.get('rsi')}")
        
        # Get strategy catalogue
        catalogue = self.catalogues.get(underlying)
        if not catalogue:
            return []
        
        # Generate signals
        if strategy_type:
            strategy = catalogue.get_strategy(strategy_type)
            if strategy:
                signal = strategy.analyze(options_chain, metrics)
                return [signal] if signal else []
            return []
        
        return catalogue.analyze_all(options_chain, metrics)
    
    def get_sentiment_signals(
        self,
        underlying: str,
        sentiment: str,
    ) -> List[StrategySignal]:
        """
        Get signals that match a specific sentiment.
        
        Args:
            underlying: The underlying asset
            sentiment: Target sentiment
            
        Returns:
            List of matching signals
        """
        # Fetch data
        spot = data_fetcher.get_spot_price(underlying)
        if not spot:
            return []
        
        options_chain = data_fetcher.get_options_chain(underlying)
        oi_data = data_fetcher.get_oi_data(underlying)
        volatility = data_fetcher.get_volatility_metrics(underlying)
        
        metrics = {
            "spot": spot,
            "oi_data": oi_data,
            "volatility": volatility,
        }
        
        catalogue = self.catalogues.get(underlying)
        if not catalogue:
            return []
        
        return catalogue.get_signals_by_sentiment(options_chain, metrics, sentiment)
    
    def get_best_signal(self, underlying: Optional[str] = None) -> Optional[StrategySignal]:
        """
        Get the single best signal.
        
        Args:
            underlying: Specific underlying (None for any)
            
        Returns:
            Best signal or None
        """
        signals = self.generate_signals(underlying)
        return signals[0] if signals else None
    
    def get_market_overview(self, underlying: str) -> Dict[str, Any]:
        """
        Get a comprehensive market overview for an underlying.
        
        Args:
            underlying: The underlying asset
            
        Returns:
            Dictionary with market overview
        """
        spot = data_fetcher.get_spot_price(underlying)
        oi_data = data_fetcher.get_oi_data(underlying)
        volatility = data_fetcher.get_volatility_metrics(underlying)
        
        # Get options chain for additional analysis
        chain = data_fetcher.get_options_chain(underlying)
        
        return {
            "underlying": underlying,
            "spot": spot,
            "timestamp": datetime.now().isoformat(),
            "oi_analysis": {
                "pcr": oi_data.get("pcr"),
                "total_call_oi": oi_data.get("total_call_oi"),
                "total_put_oi": oi_data.get("total_put_oi"),
                "max_pain": oi_data.get("max_pain"),
                "max_call_oi_strike": oi_data.get("max_call_oi_strike"),
                "max_put_oi_strike": oi_data.get("max_put_oi_strike"),
                "sentiment": oi_data.get("sentiment"),
            },
            "volatility": {
                "hv_20": volatility.get("hv_20"),
                "hv_10": volatility.get("hv_10"),
                "atm_iv": volatility.get("atm_iv"),
                "iv_hv_ratio": volatility.get("iv_hv_ratio"),
                "regime": volatility.get("volatility_regime"),
            },
            "options_chain_summary": {
                "total_options": len(chain) if not chain.empty else 0,
                "expiry": chain["expiry"].iloc[0].isoformat() if not chain.empty else None,
            },
            "recommended_strategies": self._recommend_strategies(oi_data, volatility),
        }
    
    def _recommend_strategies(
        self,
        oi_data: Dict,
        volatility: Dict,
    ) -> List[str]:
        """
        Recommend strategies based on market conditions.
        
        Args:
            oi_data: Open interest data
            volatility: Volatility metrics
            
        Returns:
            List of recommended strategy names
        """
        sentiment = oi_data.get("sentiment", "NEUTRAL")
        iv_regime = volatility.get("volatility_regime", "NORMAL")
        
        recommendations = []
        
        # Directional recommendations
        if sentiment in ["BULLISH", "STRONGLY_BULLISH"]:
            if iv_regime in ["LOW_IV", "IV_DEPRESSED"]:
                recommendations.append("Long Call - Low IV favorable for buying")
            recommendations.append("Bull Call Spread - Limited risk bullish play")
            if iv_regime in ["HIGH_IV", "IV_ELEVATED"]:
                recommendations.append("Short Put - Collect premium with bullish bias")
        
        elif sentiment in ["BEARISH", "STRONGLY_BEARISH"]:
            if iv_regime in ["LOW_IV", "IV_DEPRESSED"]:
                recommendations.append("Long Put - Low IV favorable for buying")
            recommendations.append("Bear Put Spread - Limited risk bearish play")
            if iv_regime in ["HIGH_IV", "IV_ELEVATED"]:
                recommendations.append("Short Call - Collect premium with bearish bias")
        
        else:  # Neutral
            if iv_regime in ["HIGH_IV", "IV_ELEVATED"]:
                recommendations.append("Iron Condor - Profit from high IV and range-bound market")
                recommendations.append("Short Strangle - Collect premium in range-bound market")
            else:
                recommendations.append("Long Straddle - Profit from expected volatility expansion")
        
        return recommendations
    
    def clear_history(self) -> None:
        """Clear signal history."""
        self.signal_history = []
        self.last_signals = {}
        logger.info("Signal history cleared")


# Singleton instance
signal_generator = SignalGenerator()
