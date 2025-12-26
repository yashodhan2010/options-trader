"""
Exit Signal Generator - Systematic exit strategy based on market conditions and reversal signals.

Enhanced with ML-based exit probability prediction.

Instead of static percentage targets, this module generates intelligent exit signals by:
1. Detecting trend reversals in the underlying
2. Checking if the original trade thesis is still valid
3. Monitoring sentiment shifts (OI changes, PCR reversals)
4. Time-based exits (theta decay, DTE thresholds)
5. Volatility-based exits (IV crush/expansion)
6. Technical indicator reversals (RSI, momentum)
7. Support/Resistance breaches
8. ML-based exit probability prediction (NEW)
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from strategies.base_strategy import StrategySignal, StrategyType, TradeDirection
from data.data_fetcher import data_fetcher
from config.settings import TRADING_CONFIG, GREEKS_EXIT_CONFIG, ML_CONFIG
from core.logger import logger


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


class ExitReason(Enum):
    """Categorized exit reasons for analysis."""
    # Profit-based
    TARGET_HIT = "TARGET_HIT"
    PROFIT_LOCK = "PROFIT_LOCK"
    TRAILING_STOP = "TRAILING_STOP"
    
    # Loss-based
    STOP_LOSS = "STOP_LOSS"
    MAX_LOSS = "MAX_LOSS"
    
    # Signal-based (NEW)
    TREND_REVERSAL = "TREND_REVERSAL"
    SENTIMENT_REVERSAL = "SENTIMENT_REVERSAL"
    MOMENTUM_REVERSAL = "MOMENTUM_REVERSAL"
    RSI_REVERSAL = "RSI_REVERSAL"
    THESIS_INVALIDATED = "THESIS_INVALIDATED"
    
    # ML-based (NEW)
    ML_EXIT_SIGNAL = "ML_EXIT_SIGNAL"
    
    # Technical-based
    SUPPORT_BREACH = "SUPPORT_BREACH"
    RESISTANCE_BREACH = "RESISTANCE_BREACH"
    OI_SHIFT = "OI_SHIFT"
    
    # Greeks-based
    DELTA_EXIT = "DELTA_EXIT"
    THETA_DECAY = "THETA_DECAY"
    IV_CRUSH = "IV_CRUSH"
    IV_EXPANSION = "IV_EXPANSION"
    GAMMA_RISK = "GAMMA_RISK"
    DTE_EXIT = "DTE_EXIT"
    
    # Time-based
    END_OF_DAY = "END_OF_DAY"
    EXPIRY_APPROACH = "EXPIRY_APPROACH"
    
    # Manual
    MANUAL_EXIT = "MANUAL_EXIT"
    SQUARE_OFF = "SQUARE_OFF"


@dataclass
class ExitSignal:
    """
    Exit signal with detailed reasoning and confidence.
    """
    execution_id: str
    should_exit: bool
    reason: ExitReason
    confidence: float  # 0.0 to 1.0
    urgency: str  # "LOW", "MEDIUM", "HIGH", "IMMEDIATE"
    current_pnl: float
    expected_pnl_if_hold: float  # Estimated P&L if we don't exit
    rationale: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def is_urgent(self) -> bool:
        return self.urgency in ["HIGH", "IMMEDIATE"]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "should_exit": self.should_exit,
            "reason": self.reason.value,
            "confidence": self.confidence,
            "urgency": self.urgency,
            "current_pnl": self.current_pnl,
            "expected_pnl_if_hold": self.expected_pnl_if_hold,
            "rationale": self.rationale,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
        }


# Exit configuration for different strategy types
EXIT_CONFIG = {
    # Directional strategies - sensitive to trend reversals
    StrategyType.LONG_CALL: {
        "trend_sensitive": True,
        "sentiment_sensitive": True,
        "reversal_directions": ["BEARISH", "STRONGLY_BEARISH"],  # Exit if sentiment flips
        "rsi_overbought_exit": 75,  # Exit when RSI goes overbought (profit booking)
        "rsi_oversold_hold": 30,    # Don't exit if still oversold
        "momentum_reversal_exit": True,
        "support_breach_exit": True,
        "min_profit_for_signal_exit": 0.2,  # Only signal-exit if 20%+ profit
    },
    StrategyType.LONG_PUT: {
        "trend_sensitive": True,
        "sentiment_sensitive": True,
        "reversal_directions": ["BULLISH", "STRONGLY_BULLISH"],
        "rsi_oversold_exit": 25,   # Exit when RSI goes oversold (profit booking)
        "rsi_overbought_hold": 70,
        "momentum_reversal_exit": True,
        "resistance_breach_exit": True,
        "min_profit_for_signal_exit": 0.2,
    },
    StrategyType.SHORT_CALL: {
        "trend_sensitive": True,
        "sentiment_sensitive": True,
        "reversal_directions": ["BULLISH", "STRONGLY_BULLISH"],  # Danger for short call
        "resistance_breach_exit": True,  # Exit if resistance breaks
        "iv_crush_target": True,   # Target IV crush
        "theta_decay_target": True,
        "min_profit_for_signal_exit": 0.3,
    },
    StrategyType.SHORT_PUT: {
        "trend_sensitive": True,
        "sentiment_sensitive": True,
        "reversal_directions": ["BEARISH", "STRONGLY_BEARISH"],
        "support_breach_exit": True,
        "iv_crush_target": True,
        "theta_decay_target": True,
        "min_profit_for_signal_exit": 0.3,
    },
    # Spread strategies - less sensitive to small moves
    StrategyType.BULL_CALL_SPREAD: {
        "trend_sensitive": True,
        "sentiment_sensitive": False,  # Spreads are more forgiving
        "reversal_directions": ["STRONGLY_BEARISH"],  # Only strong reversal
        "min_profit_for_signal_exit": 0.4,
    },
    StrategyType.BEAR_PUT_SPREAD: {
        "trend_sensitive": True,
        "sentiment_sensitive": False,
        "reversal_directions": ["STRONGLY_BULLISH"],
        "min_profit_for_signal_exit": 0.4,
    },
    StrategyType.BEAR_CALL_SPREAD: {
        "trend_sensitive": True,
        "sentiment_sensitive": False,
        "reversal_directions": ["STRONGLY_BULLISH"],
        "resistance_breach_exit": True,
        "iv_crush_target": True,
        "min_profit_for_signal_exit": 0.3,
    },
    StrategyType.BULL_PUT_SPREAD: {
        "trend_sensitive": True,
        "sentiment_sensitive": False,
        "reversal_directions": ["STRONGLY_BEARISH"],
        "support_breach_exit": True,
        "iv_crush_target": True,
        "min_profit_for_signal_exit": 0.3,
    },
    # Neutral strategies - sensitive to breakouts
    StrategyType.IRON_CONDOR: {
        "trend_sensitive": False,
        "sentiment_sensitive": True,
        "reversal_directions": ["STRONGLY_BULLISH", "STRONGLY_BEARISH"],  # Any strong move is bad
        "breakout_exit": True,
        "iv_crush_target": True,
        "theta_decay_target": True,
        "min_profit_for_signal_exit": 0.3,
    },
    StrategyType.STRADDLE: {
        "trend_sensitive": False,
        "sentiment_sensitive": False,
        "breakout_target": True,  # Looking for big move
        "iv_expansion_target": True,
        "min_profit_for_signal_exit": 0.3,
    },
    StrategyType.STRANGLE: {
        "trend_sensitive": False,
        "sentiment_sensitive": False,
        "breakout_target": True,
        "iv_expansion_target": True,
        "min_profit_for_signal_exit": 0.3,
    },
}


class ExitSignalGenerator:
    """
    Generates intelligent exit signals based on market conditions,
    not just static percentage targets.
    
    Enhanced with ML-based exit probability prediction.
    """
    
    def __init__(self):
        self.entry_conditions: Dict[str, Dict] = {}  # Store conditions at entry
        self.last_check: Dict[str, datetime] = {}
        
        # ML integration
        self.ml_enabled = ML_CONFIG.get("enabled", False)
        self._predictor = None
        self._feature_engineer = None
        self._guardrails = None
        
        if self.ml_enabled:
            logger.info("ML-enhanced exit signal generation enabled")
    
    def _init_ml(self):
        """Initialize ML components on first use."""
        if self._predictor is None and self.ml_enabled:
            self._predictor, self._feature_engineer, self._guardrails = _get_ml_components()
        
    def store_entry_conditions(
        self, 
        execution_id: str, 
        signal: StrategySignal,
        market_data: Dict[str, Any]
    ) -> None:
        """
        Store market conditions at entry for comparison during exit checks.
        Call this when a new position is opened.
        
        Args:
            execution_id: Unique execution ID
            signal: The strategy signal that was executed
            market_data: Current market data (spot, OI, volatility, historical)
        """
        self.entry_conditions[execution_id] = {
            "signal": signal,
            "timestamp": datetime.now(),
            "spot": market_data.get("spot", 0),
            "sentiment": market_data.get("oi_data", {}).get("sentiment", "NEUTRAL"),
            "pcr": market_data.get("oi_data", {}).get("pcr", 1.0),
            "iv_regime": market_data.get("volatility", {}).get("volatility_regime", "NORMAL"),
            "avg_iv": market_data.get("volatility", {}).get("avg_iv", 0),
            "trend": market_data.get("historical", {}).get("trend", "NEUTRAL"),
            "momentum": market_data.get("historical", {}).get("momentum", "NEUTRAL"),
            "rsi": market_data.get("historical", {}).get("rsi", 50),
            "max_call_oi_strike": market_data.get("oi_data", {}).get("max_call_oi_strike"),
            "max_put_oi_strike": market_data.get("oi_data", {}).get("max_put_oi_strike"),
        }
        logger.debug(f"Stored entry conditions for {execution_id}: {signal.strategy_type.value}")
    
    def generate_exit_signal(
        self,
        execution_id: str,
        signal: StrategySignal,
        current_pnl: float,
        current_prices: Dict[str, float],
    ) -> Optional[ExitSignal]:
        """
        Generate an exit signal by analyzing current market conditions
        against entry conditions and the original trade thesis.
        
        Args:
            execution_id: Unique execution ID
            signal: The original strategy signal
            current_pnl: Current P&L of the position
            current_prices: Current prices for each leg
            
        Returns:
            ExitSignal if exit is recommended, None otherwise
        """
        # Get entry conditions
        entry = self.entry_conditions.get(execution_id, {})
        if not entry:
            logger.debug(f"No entry conditions for {execution_id}, using signal defaults")
            entry = {"signal": signal, "timestamp": signal.entry_time}
        
        # Get strategy-specific exit configuration
        strategy_config = EXIT_CONFIG.get(signal.strategy_type, {})
        
        # Fetch current market data
        underlying = signal.underlying
        try:
            current_data = self._fetch_current_market_data(underlying)
        except Exception as e:
            logger.debug(f"Could not fetch market data for exit check: {e}")
            return None
        
        # Calculate profit ratio
        max_profit = signal.expected_profit if signal.expected_profit > 0 else signal.target
        profit_ratio = current_pnl / max_profit if max_profit > 0 else 0
        
        # Run all exit checks
        exit_signals = []
        
        # 1. Traditional target/SL checks (keep as baseline)
        if current_pnl >= signal.target:
            exit_signals.append(self._create_exit_signal(
                execution_id, ExitReason.TARGET_HIT, 1.0, "MEDIUM",
                current_pnl, current_pnl,
                f"Target of Rs.{signal.target:.2f} reached"
            ))
        
        if current_pnl <= -signal.stop_loss:
            exit_signals.append(self._create_exit_signal(
                execution_id, ExitReason.STOP_LOSS, 1.0, "IMMEDIATE",
                current_pnl, current_pnl * 1.5,  # Expect further loss if held
                f"Stop loss of Rs.{signal.stop_loss:.2f} hit"
            ))
        
        # 2. TREND REVERSAL CHECK
        if strategy_config.get("trend_sensitive", False):
            trend_exit = self._check_trend_reversal(
                execution_id, signal, entry, current_data, current_pnl, profit_ratio, strategy_config
            )
            if trend_exit:
                exit_signals.append(trend_exit)
        
        # 3. SENTIMENT REVERSAL CHECK
        if strategy_config.get("sentiment_sensitive", False):
            sentiment_exit = self._check_sentiment_reversal(
                execution_id, signal, entry, current_data, current_pnl, profit_ratio, strategy_config
            )
            if sentiment_exit:
                exit_signals.append(sentiment_exit)
        
        # 4. MOMENTUM/RSI REVERSAL CHECK
        momentum_exit = self._check_momentum_reversal(
            execution_id, signal, entry, current_data, current_pnl, profit_ratio, strategy_config
        )
        if momentum_exit:
            exit_signals.append(momentum_exit)
        
        # 5. SUPPORT/RESISTANCE BREACH CHECK
        sr_exit = self._check_support_resistance(
            execution_id, signal, entry, current_data, current_pnl, strategy_config
        )
        if sr_exit:
            exit_signals.append(sr_exit)
        
        # 6. OI SHIFT CHECK (Max OI strike changes)
        oi_exit = self._check_oi_shift(
            execution_id, signal, entry, current_data, current_pnl, strategy_config
        )
        if oi_exit:
            exit_signals.append(oi_exit)
        
        # 7. IV-BASED EXITS
        iv_exit = self._check_iv_exit(
            execution_id, signal, entry, current_data, current_pnl, strategy_config
        )
        if iv_exit:
            exit_signals.append(iv_exit)
        
        # 8. THESIS INVALIDATION (Comprehensive check)
        thesis_exit = self._check_thesis_validity(
            execution_id, signal, entry, current_data, current_pnl, profit_ratio
        )
        if thesis_exit:
            exit_signals.append(thesis_exit)
        
        # 9. ML-BASED EXIT CHECK (NEW)
        if self.ml_enabled:
            ml_exit = self._check_ml_exit(
                execution_id, signal, entry, current_data, current_pnl, profit_ratio
            )
            if ml_exit:
                exit_signals.append(ml_exit)
        
        # Return highest priority exit signal
        if exit_signals:
            # Sort by urgency and confidence
            urgency_order = {"IMMEDIATE": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
            exit_signals.sort(key=lambda x: (urgency_order.get(x.urgency, 4), -x.confidence))
            return exit_signals[0]
        
        return None
    
    def _fetch_current_market_data(self, underlying: str) -> Dict[str, Any]:
        """Fetch current market data for the underlying."""
        spot = data_fetcher.get_spot_price(underlying)
        oi_data = data_fetcher.get_oi_analysis(underlying)
        volatility = data_fetcher.get_volatility_data(underlying)
        historical = data_fetcher.get_historical_analysis(underlying, days=5)
        
        return {
            "spot": spot,
            "oi_data": oi_data or {},
            "volatility": volatility or {},
            "historical": historical or {},
        }
    
    def _check_trend_reversal(
        self,
        execution_id: str,
        signal: StrategySignal,
        entry: Dict,
        current: Dict,
        current_pnl: float,
        profit_ratio: float,
        config: Dict,
    ) -> Optional[ExitSignal]:
        """Check if trend has reversed against the position."""
        entry_trend = entry.get("trend", "NEUTRAL")
        current_trend = current.get("historical", {}).get("trend", "NEUTRAL")
        
        # Define trend reversals for each direction
        bullish_trends = ["UPTREND", "STRONG_UPTREND"]
        bearish_trends = ["DOWNTREND", "STRONG_DOWNTREND"]
        
        is_bullish_strategy = signal.strategy_type in [
            StrategyType.LONG_CALL, StrategyType.BULL_CALL_SPREAD, 
            StrategyType.BULL_PUT_SPREAD, StrategyType.SHORT_PUT
        ]
        is_bearish_strategy = signal.strategy_type in [
            StrategyType.LONG_PUT, StrategyType.BEAR_PUT_SPREAD,
            StrategyType.BEAR_CALL_SPREAD, StrategyType.SHORT_CALL
        ]
        
        reversal_detected = False
        reason = ""
        urgency = "MEDIUM"
        
        if is_bullish_strategy:
            # For bullish strategies, bearish trend reversal is dangerous
            if current_trend in bearish_trends and entry_trend in bullish_trends:
                reversal_detected = True
                reason = f"Trend reversed from {entry_trend} to {current_trend}"
                urgency = "HIGH" if current_trend == "STRONG_DOWNTREND" else "MEDIUM"
        
        elif is_bearish_strategy:
            # For bearish strategies, bullish trend reversal is dangerous
            if current_trend in bullish_trends and entry_trend in bearish_trends:
                reversal_detected = True
                reason = f"Trend reversed from {entry_trend} to {current_trend}"
                urgency = "HIGH" if current_trend == "STRONG_UPTREND" else "MEDIUM"
        
        if reversal_detected:
            # Only exit on reversal if we have some profit or loss is small
            min_profit = config.get("min_profit_for_signal_exit", 0.2)
            if profit_ratio >= min_profit or current_pnl > 0:
                return self._create_exit_signal(
                    execution_id, ExitReason.TREND_REVERSAL, 0.75, urgency,
                    current_pnl, current_pnl * 0.5,  # Expect profit to decrease
                    reason,
                    {"entry_trend": entry_trend, "current_trend": current_trend}
                )
        
        return None
    
    def _check_sentiment_reversal(
        self,
        execution_id: str,
        signal: StrategySignal,
        entry: Dict,
        current: Dict,
        current_pnl: float,
        profit_ratio: float,
        config: Dict,
    ) -> Optional[ExitSignal]:
        """Check if OI-based sentiment has reversed."""
        entry_sentiment = entry.get("sentiment", "NEUTRAL")
        current_sentiment = current.get("oi_data", {}).get("sentiment", "NEUTRAL")
        
        reversal_directions = config.get("reversal_directions", [])
        
        if current_sentiment in reversal_directions and entry_sentiment not in reversal_directions:
            min_profit = config.get("min_profit_for_signal_exit", 0.2)
            
            if profit_ratio >= min_profit or current_pnl > 0:
                urgency = "HIGH" if "STRONGLY" in current_sentiment else "MEDIUM"
                
                return self._create_exit_signal(
                    execution_id, ExitReason.SENTIMENT_REVERSAL, 0.70, urgency,
                    current_pnl, current_pnl * 0.6,
                    f"Sentiment reversed from {entry_sentiment} to {current_sentiment}",
                    {"entry_sentiment": entry_sentiment, "current_sentiment": current_sentiment}
                )
        
        return None
    
    def _check_momentum_reversal(
        self,
        execution_id: str,
        signal: StrategySignal,
        entry: Dict,
        current: Dict,
        current_pnl: float,
        profit_ratio: float,
        config: Dict,
    ) -> Optional[ExitSignal]:
        """Check if momentum or RSI indicates reversal."""
        current_rsi = current.get("historical", {}).get("rsi", 50)
        current_momentum = current.get("historical", {}).get("momentum", "NEUTRAL")
        entry_momentum = entry.get("momentum", "NEUTRAL")
        
        is_bullish_strategy = signal.strategy_type in [
            StrategyType.LONG_CALL, StrategyType.BULL_CALL_SPREAD, 
            StrategyType.BULL_PUT_SPREAD, StrategyType.SHORT_PUT
        ]
        
        # RSI-based exit for profit booking
        if is_bullish_strategy:
            overbought_level = config.get("rsi_overbought_exit", 75)
            if current_rsi >= overbought_level and current_pnl > 0:
                return self._create_exit_signal(
                    execution_id, ExitReason.RSI_REVERSAL, 0.65, "MEDIUM",
                    current_pnl, current_pnl * 0.8,
                    f"RSI overbought at {current_rsi:.1f}, booking profits",
                    {"rsi": current_rsi, "threshold": overbought_level}
                )
        else:
            oversold_level = config.get("rsi_oversold_exit", 25)
            if current_rsi <= oversold_level and current_pnl > 0:
                return self._create_exit_signal(
                    execution_id, ExitReason.RSI_REVERSAL, 0.65, "MEDIUM",
                    current_pnl, current_pnl * 0.8,
                    f"RSI oversold at {current_rsi:.1f}, booking profits",
                    {"rsi": current_rsi, "threshold": oversold_level}
                )
        
        # Momentum reversal check
        if config.get("momentum_reversal_exit", False):
            bullish_momentum = ["BULLISH", "STRONG_BULLISH"]
            bearish_momentum = ["BEARISH", "STRONG_BEARISH"]
            
            if is_bullish_strategy:
                if current_momentum in bearish_momentum and entry_momentum in bullish_momentum:
                    if profit_ratio >= 0.2:
                        return self._create_exit_signal(
                            execution_id, ExitReason.MOMENTUM_REVERSAL, 0.60, "MEDIUM",
                            current_pnl, current_pnl * 0.7,
                            f"Momentum reversed from {entry_momentum} to {current_momentum}",
                            {"entry_momentum": entry_momentum, "current_momentum": current_momentum}
                        )
        
        return None
    
    def _check_support_resistance(
        self,
        execution_id: str,
        signal: StrategySignal,
        entry: Dict,
        current: Dict,
        current_pnl: float,
        config: Dict,
    ) -> Optional[ExitSignal]:
        """Check if key support/resistance levels have been breached."""
        current_spot = current.get("spot", 0)
        entry_spot = entry.get("spot", 0)
        
        max_call_oi = entry.get("max_call_oi_strike")  # Resistance at entry
        max_put_oi = entry.get("max_put_oi_strike")   # Support at entry
        
        is_bullish_strategy = signal.strategy_type in [
            StrategyType.LONG_CALL, StrategyType.BULL_CALL_SPREAD, 
            StrategyType.BULL_PUT_SPREAD, StrategyType.SHORT_PUT
        ]
        
        # Support breach (bearish) - dangerous for bullish strategies
        if config.get("support_breach_exit", False) and max_put_oi:
            if current_spot < max_put_oi and entry_spot >= max_put_oi:
                if is_bullish_strategy:
                    return self._create_exit_signal(
                        execution_id, ExitReason.SUPPORT_BREACH, 0.80, "HIGH",
                        current_pnl, current_pnl - abs(current_pnl) * 0.5,
                        f"Support at {max_put_oi} breached, spot now {current_spot:.2f}",
                        {"support": max_put_oi, "spot": current_spot}
                    )
        
        # Resistance breach (bullish) - dangerous for bearish strategies
        if config.get("resistance_breach_exit", False) and max_call_oi:
            if current_spot > max_call_oi and entry_spot <= max_call_oi:
                if not is_bullish_strategy:
                    return self._create_exit_signal(
                        execution_id, ExitReason.RESISTANCE_BREACH, 0.80, "HIGH",
                        current_pnl, current_pnl - abs(current_pnl) * 0.5,
                        f"Resistance at {max_call_oi} breached, spot now {current_spot:.2f}",
                        {"resistance": max_call_oi, "spot": current_spot}
                    )
        
        return None
    
    def _check_oi_shift(
        self,
        execution_id: str,
        signal: StrategySignal,
        entry: Dict,
        current: Dict,
        current_pnl: float,
        config: Dict,
    ) -> Optional[ExitSignal]:
        """Check if max OI strikes have shifted significantly."""
        entry_call_oi = entry.get("max_call_oi_strike")
        entry_put_oi = entry.get("max_put_oi_strike")
        current_call_oi = current.get("oi_data", {}).get("max_call_oi_strike")
        current_put_oi = current.get("oi_data", {}).get("max_put_oi_strike")
        
        # Significant shift detection
        if entry_call_oi and current_call_oi and entry_put_oi and current_put_oi:
            entry_range = entry_call_oi - entry_put_oi
            current_range = current_call_oi - current_put_oi
            
            # Range expansion/contraction
            if entry_range > 0:
                range_change = (current_range - entry_range) / entry_range
                
                # Significant range expansion (volatility expectation increased)
                if abs(range_change) > 0.3:  # 30% change
                    return self._create_exit_signal(
                        execution_id, ExitReason.OI_SHIFT, 0.55, "LOW",
                        current_pnl, current_pnl,
                        f"OI range shifted significantly ({range_change:.1%})",
                        {"entry_range": entry_range, "current_range": current_range}
                    )
        
        return None
    
    def _check_iv_exit(
        self,
        execution_id: str,
        signal: StrategySignal,
        entry: Dict,
        current: Dict,
        current_pnl: float,
        config: Dict,
    ) -> Optional[ExitSignal]:
        """Check for IV-based exit conditions."""
        entry_iv = entry.get("avg_iv", 0)
        current_iv = current.get("volatility", {}).get("avg_iv", 0)
        
        if entry_iv <= 0 or current_iv <= 0:
            return None
        
        iv_change_pct = ((current_iv - entry_iv) / entry_iv) * 100
        
        # Determine if position benefits from IV crush or expansion
        is_long_option = any(leg.direction == TradeDirection.BUY for leg in signal.legs)
        is_short_option = any(leg.direction == TradeDirection.SELL for leg in signal.legs)
        
        # IV Crush (good for short options, bad for long)
        if config.get("iv_crush_target", False) and is_short_option:
            if iv_change_pct < -15:  # IV dropped 15%+
                if current_pnl > 0:
                    return self._create_exit_signal(
                        execution_id, ExitReason.IV_CRUSH, 0.70, "MEDIUM",
                        current_pnl, current_pnl * 1.1,
                        f"IV crushed {abs(iv_change_pct):.1f}%, booking theta profits",
                        {"entry_iv": entry_iv, "current_iv": current_iv, "change": iv_change_pct}
                    )
        
        # IV Expansion (good for long options in straddle/strangle)
        if config.get("iv_expansion_target", False) and is_long_option:
            if iv_change_pct > 20:  # IV increased 20%+
                if current_pnl > 0:
                    return self._create_exit_signal(
                        execution_id, ExitReason.IV_EXPANSION, 0.65, "MEDIUM",
                        current_pnl, current_pnl * 0.9,
                        f"IV expanded {iv_change_pct:.1f}%, booking vega profits",
                        {"entry_iv": entry_iv, "current_iv": current_iv, "change": iv_change_pct}
                    )
        
        # IV Crush hurting long options
        if is_long_option and iv_change_pct < -20:
            return self._create_exit_signal(
                execution_id, ExitReason.IV_CRUSH, 0.60, "MEDIUM",
                current_pnl, current_pnl * 0.7,
                f"IV crushed {abs(iv_change_pct):.1f}%, reducing long option value",
                {"entry_iv": entry_iv, "current_iv": current_iv, "change": iv_change_pct}
            )
        
        return None
    
    def _check_thesis_validity(
        self,
        execution_id: str,
        signal: StrategySignal,
        entry: Dict,
        current: Dict,
        current_pnl: float,
        profit_ratio: float,
    ) -> Optional[ExitSignal]:
        """
        Comprehensive check if the original trade thesis is still valid.
        Exit if multiple conditions have invalidated the original reasoning.
        """
        invalidation_score = 0
        invalidation_reasons = []
        
        entry_sentiment = entry.get("sentiment", "NEUTRAL")
        current_sentiment = current.get("oi_data", {}).get("sentiment", "NEUTRAL")
        entry_trend = entry.get("trend", "NEUTRAL")
        current_trend = current.get("historical", {}).get("trend", "NEUTRAL")
        entry_iv_regime = entry.get("iv_regime", "NORMAL")
        current_iv_regime = current.get("volatility", {}).get("volatility_regime", "NORMAL")
        
        # Check sentiment alignment
        bullish_sentiments = ["BULLISH", "STRONGLY_BULLISH"]
        bearish_sentiments = ["BEARISH", "STRONGLY_BEARISH"]
        
        is_bullish_strategy = signal.strategy_type in [
            StrategyType.LONG_CALL, StrategyType.BULL_CALL_SPREAD, 
            StrategyType.BULL_PUT_SPREAD, StrategyType.SHORT_PUT
        ]
        is_bearish_strategy = signal.strategy_type in [
            StrategyType.LONG_PUT, StrategyType.BEAR_PUT_SPREAD,
            StrategyType.BEAR_CALL_SPREAD, StrategyType.SHORT_CALL
        ]
        is_neutral_strategy = signal.strategy_type in [
            StrategyType.IRON_CONDOR, StrategyType.STRADDLE, StrategyType.STRANGLE
        ]
        
        # 1. Sentiment misalignment
        if is_bullish_strategy and current_sentiment in bearish_sentiments:
            invalidation_score += 2
            invalidation_reasons.append(f"Sentiment now {current_sentiment}")
        elif is_bearish_strategy and current_sentiment in bullish_sentiments:
            invalidation_score += 2
            invalidation_reasons.append(f"Sentiment now {current_sentiment}")
        elif is_neutral_strategy and current_sentiment in ["STRONGLY_BULLISH", "STRONGLY_BEARISH"]:
            invalidation_score += 2
            invalidation_reasons.append(f"Strong directional sentiment: {current_sentiment}")
        
        # 2. Trend misalignment
        bullish_trends = ["UPTREND", "STRONG_UPTREND"]
        bearish_trends = ["DOWNTREND", "STRONG_DOWNTREND"]
        
        if is_bullish_strategy and current_trend in bearish_trends:
            invalidation_score += 2
            invalidation_reasons.append(f"Trend now {current_trend}")
        elif is_bearish_strategy and current_trend in bullish_trends:
            invalidation_score += 2
            invalidation_reasons.append(f"Trend now {current_trend}")
        
        # 3. IV regime change
        high_iv = ["HIGH_IV", "IV_ELEVATED"]
        low_iv = ["LOW_IV", "IV_DEPRESSED"]
        
        is_selling_premium = signal.strategy_type in [
            StrategyType.SHORT_CALL, StrategyType.SHORT_PUT,
            StrategyType.BEAR_CALL_SPREAD, StrategyType.BULL_PUT_SPREAD,
            StrategyType.IRON_CONDOR
        ]
        is_buying_premium = signal.strategy_type in [
            StrategyType.LONG_CALL, StrategyType.LONG_PUT,
            StrategyType.STRADDLE, StrategyType.STRANGLE
        ]
        
        if is_selling_premium and entry_iv_regime in high_iv and current_iv_regime in low_iv:
            # IV crushed - good for sellers, might want to exit
            invalidation_score += 1
            invalidation_reasons.append("IV regime shifted (crush complete)")
        elif is_buying_premium and entry_iv_regime in low_iv and current_iv_regime in high_iv:
            # IV expanded - might want to take profits
            invalidation_score += 1
            invalidation_reasons.append("IV expanded (vega profit available)")
        
        # Thesis invalidated if score >= 3
        if invalidation_score >= 3:
            # Only exit if we have profit or small loss
            if profit_ratio >= 0.1 or current_pnl > 0:
                return self._create_exit_signal(
                    execution_id, ExitReason.THESIS_INVALIDATED, 0.75, "MEDIUM",
                    current_pnl, current_pnl * 0.5,
                    f"Trade thesis invalidated: {', '.join(invalidation_reasons)}",
                    {"invalidation_score": invalidation_score, "reasons": invalidation_reasons}
                )
        
        return None
    
    def _check_ml_exit(
        self,
        execution_id: str,
        signal: StrategySignal,
        entry: Dict,
        current: Dict,
        current_pnl: float,
        profit_ratio: float,
    ) -> Optional[ExitSignal]:
        """
        Check if ML model recommends exit.
        
        Uses ML predictor to assess current market conditions
        and recommend exit based on predicted adverse movement.
        
        Note: ML never overrides stop-loss (handled by guardrails).
        """
        self._init_ml()
        
        if not self._predictor or not self._feature_engineer:
            return None
        
        try:
            underlying = signal.underlying
            spot = current.get("spot", 0)
            
            if not spot:
                return None
            
            # Extract current features
            features = self._feature_engineer.extract_features(
                spot_price=spot,
                market_data=current,
                underlying=underlying,
                strategy_type=signal.strategy_type.value
            )
            
            # Get ML exit prediction
            exit_prediction = self._predictor.predict_exit(
                features=features,
                underlying=underlying,
                strategy_type=signal.strategy_type.value,
                current_pnl_percent=profit_ratio * 100,
                entry_features=entry.get("features", {})
            )
            
            if exit_prediction is None:
                return None
            
            # ML exit signal only if:
            # 1. High confidence exit prediction (>0.7)
            # 2. Position is in profit (protect profits) OR small loss (<2%)
            # 3. Never on stop-loss path (guardrails handle that)
            
            min_exit_confidence = ML_CONFIG.get("guardrails", {}).get("min_ml_confidence", 0.6)
            
            if exit_prediction.confidence >= 0.7 and exit_prediction.direction == "EXIT":
                # Only trigger ML exit if in profit or small loss
                if profit_ratio >= 0 or profit_ratio >= -0.02:
                    urgency = "HIGH" if exit_prediction.confidence >= 0.85 else "MEDIUM"
                    
                    return self._create_exit_signal(
                        execution_id,
                        ExitReason.ML_EXIT_SIGNAL,
                        exit_prediction.confidence,
                        urgency,
                        current_pnl,
                        current_pnl * 0.7,  # Expected deterioration if held
                        f"ML model recommends exit (confidence: {exit_prediction.confidence:.1%})",
                        {
                            "ml_confidence": exit_prediction.confidence,
                            "ml_model_version": exit_prediction.model_version,
                            "top_features": exit_prediction.feature_importance,
                        }
                    )
            
            return None
            
        except Exception as e:
            logger.debug(f"ML exit check failed: {e}")
            return None
    
    def _create_exit_signal(
        self,
        execution_id: str,
        reason: ExitReason,
        confidence: float,
        urgency: str,
        current_pnl: float,
        expected_pnl: float,
        rationale: str,
        metrics: Dict = None,
    ) -> ExitSignal:
        """Helper to create an ExitSignal."""
        return ExitSignal(
            execution_id=execution_id,
            should_exit=True,
            reason=reason,
            confidence=confidence,
            urgency=urgency,
            current_pnl=current_pnl,
            expected_pnl_if_hold=expected_pnl,
            rationale=rationale,
            metrics=metrics or {},
        )
    
    def cleanup_execution(self, execution_id: str) -> None:
        """Clean up stored data for a closed execution."""
        if execution_id in self.entry_conditions:
            del self.entry_conditions[execution_id]
        if execution_id in self.last_check:
            del self.last_check[execution_id]


# Singleton instance
exit_signal_generator = ExitSignalGenerator()
