"""
Strategy Catalogue - Registry and management of all strategies
"""
from typing import Dict, List, Optional, Type
import pandas as pd

from .base_strategy import BaseStrategy, StrategyType, StrategySignal
from .directional import (
    LongCallStrategy, LongPutStrategy, ShortCallStrategy, ShortPutStrategy
)
from .spreads import BullCallSpreadStrategy, BearPutSpreadStrategy
from .volatility import IronCondorStrategy, StraddleStrategy, StrangleStrategy
from config.settings import STRATEGY_CONFIG
from core.logger import logger


class StrategyCatalogue:
    """
    Registry and factory for all trading strategies.
    """
    
    # Strategy registry mapping type to class
    STRATEGY_REGISTRY: Dict[StrategyType, Type[BaseStrategy]] = {
        StrategyType.LONG_CALL: LongCallStrategy,
        StrategyType.LONG_PUT: LongPutStrategy,
        StrategyType.SHORT_CALL: ShortCallStrategy,
        StrategyType.SHORT_PUT: ShortPutStrategy,
        StrategyType.BULL_CALL_SPREAD: BullCallSpreadStrategy,
        StrategyType.BEAR_PUT_SPREAD: BearPutSpreadStrategy,
        StrategyType.IRON_CONDOR: IronCondorStrategy,
        StrategyType.STRADDLE: StraddleStrategy,
        StrategyType.STRANGLE: StrangleStrategy,
    }
    
    def __init__(self, underlying: str):
        self.underlying = underlying
        self.strategies: Dict[StrategyType, BaseStrategy] = {}
        self._initialize_strategies()
    
    def _initialize_strategies(self) -> None:
        """Initialize all enabled strategies."""
        enabled = STRATEGY_CONFIG.get("enabled_strategies", [])
        
        for strategy_name in enabled:
            try:
                strategy_type = StrategyType(strategy_name)
                strategy_class = self.STRATEGY_REGISTRY.get(strategy_type)
                
                if strategy_class:
                    # Handle strategies with additional parameters
                    if strategy_type == StrategyType.STRADDLE:
                        # Create both long and short straddle
                        self.strategies[StrategyType.STRADDLE] = StraddleStrategy(
                            self.underlying, is_short=False
                        )
                    elif strategy_type == StrategyType.STRANGLE:
                        self.strategies[StrategyType.STRANGLE] = StrangleStrategy(
                            self.underlying, is_short=False
                        )
                    else:
                        self.strategies[strategy_type] = strategy_class(self.underlying)
                    
                    logger.debug(f"Initialized strategy: {strategy_name}")
                    
            except ValueError:
                logger.warning(f"Unknown strategy: {strategy_name}")
    
    def get_strategy(self, strategy_type: StrategyType) -> Optional[BaseStrategy]:
        """
        Get a specific strategy instance.
        
        Args:
            strategy_type: Type of strategy to get
            
        Returns:
            Strategy instance or None
        """
        return self.strategies.get(strategy_type)
    
    def list_strategies(self) -> List[Dict]:
        """
        List all available strategies.
        
        Returns:
            List of strategy info dictionaries
        """
        return [
            {
                "type": strategy_type.value,
                "name": strategy.name,
                "description": strategy.description,
            }
            for strategy_type, strategy in self.strategies.items()
        ]
    
    def analyze_all(
        self,
        options_chain: pd.DataFrame,
        metrics: Dict,
    ) -> List[StrategySignal]:
        """
        Run all strategies and collect signals.
        
        Args:
            options_chain: Options chain DataFrame
            metrics: Market metrics
            
        Returns:
            List of generated signals
        """
        signals = []
        
        for strategy_type, strategy in self.strategies.items():
            try:
                signal = strategy.analyze(options_chain, metrics)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error in {strategy.name}: {e}")
        
        # Sort by confidence
        signals.sort(key=lambda s: s.confidence, reverse=True)
        
        logger.info(f"Generated {len(signals)} signals from {len(self.strategies)} strategies")
        return signals
    
    def get_best_signal(
        self,
        options_chain: pd.DataFrame,
        metrics: Dict,
    ) -> Optional[StrategySignal]:
        """
        Get the best signal from all strategies.
        
        Args:
            options_chain: Options chain DataFrame
            metrics: Market metrics
            
        Returns:
            Best signal or None
        """
        signals = self.analyze_all(options_chain, metrics)
        
        if not signals:
            return None
        
        # Return highest confidence signal
        return signals[0]
    
    def get_signals_by_sentiment(
        self,
        options_chain: pd.DataFrame,
        metrics: Dict,
        sentiment: str,
    ) -> List[StrategySignal]:
        """
        Get signals matching a specific sentiment.
        
        Args:
            options_chain: Options chain DataFrame
            metrics: Market metrics
            sentiment: Target sentiment (BULLISH, BEARISH, NEUTRAL)
            
        Returns:
            List of matching signals
        """
        # Map sentiment to appropriate strategies
        sentiment_strategies = {
            "BULLISH": [StrategyType.LONG_CALL, StrategyType.BULL_CALL_SPREAD, StrategyType.SHORT_PUT],
            "STRONGLY_BULLISH": [StrategyType.LONG_CALL, StrategyType.BULL_CALL_SPREAD],
            "BEARISH": [StrategyType.LONG_PUT, StrategyType.BEAR_PUT_SPREAD, StrategyType.SHORT_CALL],
            "STRONGLY_BEARISH": [StrategyType.LONG_PUT, StrategyType.BEAR_PUT_SPREAD],
            "NEUTRAL": [StrategyType.IRON_CONDOR, StrategyType.STRADDLE, StrategyType.STRANGLE],
        }
        
        target_types = sentiment_strategies.get(sentiment, [])
        signals = []
        
        for strategy_type in target_types:
            strategy = self.strategies.get(strategy_type)
            if strategy:
                try:
                    signal = strategy.analyze(options_chain, metrics)
                    if signal:
                        signals.append(signal)
                except Exception as e:
                    logger.error(f"Error in {strategy.name}: {e}")
        
        signals.sort(key=lambda s: s.confidence, reverse=True)
        return signals
    
    def add_custom_strategy(
        self,
        strategy_type: StrategyType,
        strategy: BaseStrategy,
    ) -> None:
        """
        Add a custom strategy to the catalogue.
        
        Args:
            strategy_type: Type identifier for the strategy
            strategy: Strategy instance
        """
        self.strategies[strategy_type] = strategy
        logger.info(f"Added custom strategy: {strategy.name}")


# Factory function
def create_catalogue(underlying: str) -> StrategyCatalogue:
    """Create a strategy catalogue for an underlying."""
    return StrategyCatalogue(underlying)
