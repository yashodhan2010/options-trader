"""
Directional Strategies - Long Call, Long Put, Short Call, Short Put
"""
from datetime import datetime
from typing import Optional, Dict, Any

import pandas as pd

from .base_strategy import (
    BaseStrategy, StrategyType, StrategySignal, OptionLeg, TradeDirection
)
from config.settings import TRADING_CONFIG, STRATEGY_CONFIG, METRICS_CONFIG, UNDERLYING_ASSETS, get_lot_size
from core.logger import logger


class LongCallStrategy(BaseStrategy):
    """
    Long Call Strategy - Buy a call option.
    Bullish outlook, limited risk, unlimited profit potential.
    """
    
    def __init__(self, underlying: str):
        super().__init__(underlying)
        self.name = "Long Call"
        self.description = "Buy a call option for bullish outlook"
        self.strategy_type = StrategyType.LONG_CALL
    
    def analyze(self, options_chain: pd.DataFrame, metrics: Dict[str, Any]) -> Optional[StrategySignal]:
        """Analyze and generate signal for long call."""
        # Validate conditions
        is_valid, reason = self.validate_conditions(metrics)
        if not is_valid:
            logger.debug(f"Long Call validation failed: {reason}")
            return None
        
        # Check for bullish sentiment
        oi_data = metrics.get("oi_data", {})
        volatility = metrics.get("volatility", {})
        spot = metrics.get("spot", 0)
        
        sentiment = oi_data.get("sentiment", "NEUTRAL")
        if not self.ml_override and sentiment not in ["BULLISH", "STRONGLY_BULLISH"]:
            return None
        
        # Prefer low IV for buying options
        iv_regime = volatility.get("volatility_regime", "NORMAL")
        if not self.ml_override and iv_regime == "HIGH_IV":
            logger.debug("High IV - not ideal for long call")
            return None
        
        # Get ATM or slightly OTM call
        calls = options_chain[options_chain["option_type"] == "CE"]
        if calls.empty:
            return None
        
        # Find ATM strike
        strike_interval = 50 if self.underlying in ["NIFTY", "FINNIFTY"] else 100
        atm_strike = round(spot / strike_interval) * strike_interval
        
        # Get OTM strike based on config
        offset = STRATEGY_CONFIG.get("otm_offset", 1)
        target_strike = atm_strike + (offset * strike_interval)
        
        option = calls[calls["strike"] == target_strike]
        if option.empty:
            option = calls[calls["strike"] == atm_strike]
        
        if option.empty:
            return None
        
        option = option.iloc[0]
        
        # Liquidity guard
        is_liquid, liq_reason = self.check_leg_liquidity(option)
        if not is_liquid:
            logger.info(f"Long Call {self.underlying}: {liq_reason}")
            return None
        
        # Get historical data for enhanced confidence
        historical = metrics.get("historical", {})
        
        # Calculate confidence based on multiple factors including historical
        confidence = self._calculate_confidence(oi_data, volatility, sentiment, historical)
        
        if not self.ml_override and confidence < self.min_confidence:
            return None
        
        # Create option leg
        lot_size = get_lot_size(self.underlying)
        quantity = lot_size * TRADING_CONFIG.get("default_quantity", 1)
        
        leg = OptionLeg(
            symbol=option["symbol"],
            strike=option["strike"],
            option_type="CE",
            expiry=option["expiry"],
            direction=TradeDirection.BUY,
            quantity=quantity,
            entry_price=option["ltp"],
            instrument_token=option.get("instrument_token"),
        )
        
        # Calculate SL and target
        entry_price = option["ltp"] * quantity
        stop_loss = self.calculate_stop_loss(entry_price, iv_regime=iv_regime)
        target = self.calculate_target(entry_price, iv_regime=iv_regime)
        
        # Create signal
        signal = StrategySignal(
            strategy_type=self.strategy_type,
            underlying=self.underlying,
            legs=[leg],
            entry_time=datetime.now(),
            confidence=confidence,
            expected_profit=target,
            max_loss=entry_price,  # Max loss is premium paid
            stop_loss=stop_loss,
            target=target,
            rationale=self._build_rationale(sentiment, iv_regime, oi_data, historical),
            metrics={
                "pcr": oi_data.get("pcr"),
                "sentiment": sentiment,
                "iv_regime": iv_regime,
                "max_pain": oi_data.get("max_pain"),
                "trend": historical.get("trend"),
                "rsi": historical.get("rsi"),
                "momentum": historical.get("momentum"),
            },
        )
        
        logger.info(f"Long Call signal generated: {option['symbol']} @ {option['ltp']}")
        return signal
    
    def _calculate_confidence(
        self, oi_data: Dict, volatility: Dict, sentiment: str, historical: Dict = None
    ) -> float:
        """Calculate signal confidence including historical analysis."""
        confidence = 0.5
        
        # Sentiment boost from OI
        if sentiment == "STRONGLY_BULLISH":
            confidence += 0.15
        elif sentiment == "BULLISH":
            confidence += 0.10
        
        # PCR analysis
        pcr = oi_data.get("pcr", 1)
        if pcr > 1.2:  # High put writing
            confidence += 0.05
        
        # IV analysis - prefer low IV for buying
        iv_regime = volatility.get("volatility_regime", "NORMAL")
        if iv_regime in ["LOW_IV", "IV_DEPRESSED"]:
            confidence += 0.10
        elif iv_regime == "NORMAL":
            confidence += 0.05
        
        # Historical analysis boost
        if historical:
            # Trend alignment (bullish strategy needs uptrend)
            trend = historical.get("trend", "")
            if trend in ["STRONG_UPTREND", "UPTREND"]:
                confidence += 0.10
            elif trend in ["STRONG_DOWNTREND", "DOWNTREND"]:
                confidence -= 0.10
            
            # Momentum alignment
            momentum = historical.get("momentum", "")
            if momentum in ["STRONG_BULLISH", "BULLISH"]:
                confidence += 0.08
            elif momentum in ["STRONG_BEARISH", "BEARISH"]:
                confidence -= 0.08
            
            # RSI consideration
            rsi = historical.get("rsi", 50)
            if rsi < 30:  # Oversold - good for bullish
                confidence += 0.05
            elif rsi > 70:  # Overbought - risky for bullish
                confidence -= 0.05
            
            # Volume confirmation
            if historical.get("volume_signal") == "HIGH_VOLUME":
                confidence += 0.03
        
        return max(0.0, min(confidence, 1.0))
    
    def _build_rationale(
        self, sentiment: str, iv_regime: str, oi_data: Dict, historical: Dict
    ) -> str:
        """Build detailed rationale string."""
        parts = []
        
        # OI sentiment
        parts.append(f"OI Sentiment: {sentiment}")
        parts.append(f"PCR: {oi_data.get('pcr', 'N/A')}")
        parts.append(f"IV: {iv_regime}")
        
        # Historical data
        if historical:
            if historical.get("trend"):
                parts.append(f"Trend: {historical.get('trend')}")
            if historical.get("rsi"):
                parts.append(f"RSI: {historical.get('rsi')}")
            if historical.get("momentum"):
                parts.append(f"Momentum: {historical.get('momentum')}")
            if historical.get("returns_5d"):
                parts.append(f"5D Return: {historical.get('returns_5d')}%")
        
        return " | ".join(parts)
    
    def calculate_stop_loss(self, entry_price: float, **kwargs) -> float:
        """Calculate stop loss based on IV regime - tighter in low IV, wider in high IV."""
        iv_regime = kwargs.get("iv_regime", "NORMAL")
        if iv_regime in ["HIGH_IV", "IV_ELEVATED"]:
            sl_percent = 40  # Wider SL in high IV (more noise)
        elif iv_regime in ["LOW_IV", "IV_DEPRESSED"]:
            sl_percent = 25  # Tighter SL in low IV (less noise)
        else:
            sl_percent = TRADING_CONFIG.get("default_sl_percent", 30)
        return entry_price * (sl_percent / 100)
    
    def calculate_target(self, entry_price: float, **kwargs) -> float:
        """Calculate target based on IV regime. Stocks get higher targets than indices."""
        iv_regime = kwargs.get("iv_regime", "NORMAL")
        if iv_regime in ["LOW_IV", "IV_DEPRESSED"]:
            target_percent = 60  # Higher target when vol expansion expected
        elif iv_regime in ["HIGH_IV", "IV_ELEVATED"]:
            target_percent = 35  # Lower target in high IV (premium already inflated)
        else:
            target_percent = TRADING_CONFIG.get("default_target_percent", 50)
        # Stocks: scale up target by 40% (higher margin = higher reward expectation)
        if not self.is_index:
            target_percent = int(target_percent * 1.4)
        return entry_price * (target_percent / 100)


class LongPutStrategy(BaseStrategy):
    """
    Long Put Strategy - Buy a put option.
    Bearish outlook, limited risk, significant profit potential.
    """
    
    def __init__(self, underlying: str):
        super().__init__(underlying)
        self.name = "Long Put"
        self.description = "Buy a put option for bearish outlook"
        self.strategy_type = StrategyType.LONG_PUT
    
    def analyze(self, options_chain: pd.DataFrame, metrics: Dict[str, Any]) -> Optional[StrategySignal]:
        """Analyze and generate signal for long put."""
        is_valid, reason = self.validate_conditions(metrics)
        if not is_valid:
            return None
        
        oi_data = metrics.get("oi_data", {})
        volatility = metrics.get("volatility", {})
        spot = metrics.get("spot", 0)
        
        sentiment = oi_data.get("sentiment", "NEUTRAL")
        if not self.ml_override and sentiment not in ["BEARISH", "STRONGLY_BEARISH"]:
            return None
        
        # Prefer low IV for buying options
        iv_regime = volatility.get("volatility_regime", "NORMAL")
        if not self.ml_override and iv_regime == "HIGH_IV":
            return None
        
        puts = options_chain[options_chain["option_type"] == "PE"]
        if puts.empty:
            return None
        
        strike_interval = 50 if self.underlying in ["NIFTY", "FINNIFTY"] else 100
        atm_strike = round(spot / strike_interval) * strike_interval
        
        offset = STRATEGY_CONFIG.get("otm_offset", 1)
        target_strike = atm_strike - (offset * strike_interval)
        
        option = puts[puts["strike"] == target_strike]
        if option.empty:
            option = puts[puts["strike"] == atm_strike]
        
        if option.empty:
            return None
        
        option = option.iloc[0]
        
        # Liquidity guard
        is_liquid, liq_reason = self.check_leg_liquidity(option)
        if not is_liquid:
            logger.info(f"Long Put {self.underlying}: {liq_reason}")
            return None
        
        confidence = self._calculate_confidence(oi_data, volatility, sentiment)
        
        if not self.ml_override and confidence < self.min_confidence:
            return None
        
        lot_size = get_lot_size(self.underlying)
        quantity = lot_size * TRADING_CONFIG.get("default_quantity", 1)
        
        leg = OptionLeg(
            symbol=option["symbol"],
            strike=option["strike"],
            option_type="PE",
            expiry=option["expiry"],
            direction=TradeDirection.BUY,
            quantity=quantity,
            entry_price=option["ltp"],
            instrument_token=option.get("instrument_token"),
        )
        
        entry_price = option["ltp"] * quantity
        stop_loss = self.calculate_stop_loss(entry_price, iv_regime=iv_regime)
        target = self.calculate_target(entry_price, iv_regime=iv_regime)
        
        signal = StrategySignal(
            strategy_type=self.strategy_type,
            underlying=self.underlying,
            legs=[leg],
            entry_time=datetime.now(),
            confidence=confidence,
            expected_profit=target,
            max_loss=entry_price,
            stop_loss=stop_loss,
            target=target,
            rationale=f"Bearish sentiment ({sentiment}), IV regime: {iv_regime}, PCR: {oi_data.get('pcr', 0)}",
            metrics={
                "pcr": oi_data.get("pcr"),
                "sentiment": sentiment,
                "iv_regime": iv_regime,
                "max_pain": oi_data.get("max_pain"),
            },
        )
        
        logger.info(f"Long Put signal generated: {option['symbol']} @ {option['ltp']}")
        return signal
    
    def _calculate_confidence(self, oi_data: Dict, volatility: Dict, sentiment: str) -> float:
        """Calculate signal confidence."""
        confidence = 0.5
        
        if sentiment == "STRONGLY_BEARISH":
            confidence += 0.2
        elif sentiment == "BEARISH":
            confidence += 0.1
        
        pcr = oi_data.get("pcr", 1)
        if pcr < 0.8:
            confidence += 0.1
        
        iv_regime = volatility.get("volatility_regime", "NORMAL")
        if iv_regime in ["LOW_IV", "IV_DEPRESSED"]:
            confidence += 0.15
        elif iv_regime == "NORMAL":
            confidence += 0.05
        
        return min(confidence, 1.0)
    
    def calculate_stop_loss(self, entry_price: float, **kwargs) -> float:
        """Calculate stop loss based on IV regime."""
        iv_regime = kwargs.get("iv_regime", "NORMAL")
        if iv_regime in ["HIGH_IV", "IV_ELEVATED"]:
            sl_percent = 40
        elif iv_regime in ["LOW_IV", "IV_DEPRESSED"]:
            sl_percent = 25
        else:
            sl_percent = TRADING_CONFIG.get("default_sl_percent", 30)
        return entry_price * (sl_percent / 100)
    
    def calculate_target(self, entry_price: float, **kwargs) -> float:
        """Calculate target based on IV regime. Stocks get higher targets than indices."""
        iv_regime = kwargs.get("iv_regime", "NORMAL")
        if iv_regime in ["LOW_IV", "IV_DEPRESSED"]:
            target_percent = 60
        elif iv_regime in ["HIGH_IV", "IV_ELEVATED"]:
            target_percent = 35
        else:
            target_percent = TRADING_CONFIG.get("default_target_percent", 50)
        # Stocks: scale up target by 40%
        if not self.is_index:
            target_percent = int(target_percent * 1.4)
        return entry_price * (target_percent / 100)


class ShortCallStrategy(BaseStrategy):
    """
    Short Call Strategy - Sell a call option.
    Neutral to bearish outlook, limited profit, unlimited risk.
    Requires higher margin.
    """
    
    def __init__(self, underlying: str):
        super().__init__(underlying)
        self.name = "Short Call"
        self.description = "Sell a call option for neutral/bearish outlook"
        self.strategy_type = StrategyType.SHORT_CALL
        self.min_confidence = 0.7  # Higher threshold for selling
    
    def analyze(self, options_chain: pd.DataFrame, metrics: Dict[str, Any]) -> Optional[StrategySignal]:
        """Analyze and generate signal for short call."""
        is_valid, reason = self.validate_conditions(metrics)
        if not is_valid:
            return None
        
        oi_data = metrics.get("oi_data", {})
        volatility = metrics.get("volatility", {})
        spot = metrics.get("spot", 0)
        
        sentiment = oi_data.get("sentiment", "NEUTRAL")
        if not self.ml_override and sentiment in ["BULLISH", "STRONGLY_BULLISH"]:
            return None
        
        # Prefer high IV for selling options
        iv_regime = volatility.get("volatility_regime", "NORMAL")
        if not self.ml_override and iv_regime in ["LOW_IV", "IV_DEPRESSED"]:
            return None
        
        calls = options_chain[options_chain["option_type"] == "CE"]
        if calls.empty:
            return None
        
        strike_interval = 50 if self.underlying in ["NIFTY", "FINNIFTY"] else 100
        atm_strike = round(spot / strike_interval) * strike_interval
        
        # Sell OTM call (resistance level)
        max_call_oi_strike = oi_data.get("max_call_oi_strike", atm_strike + strike_interval * 2)
        target_strike = max_call_oi_strike
        
        option = calls[calls["strike"] == target_strike]
        if option.empty:
            return None
        
        option = option.iloc[0]
        
        # Liquidity guard
        is_liquid, liq_reason = self.check_leg_liquidity(option)
        if not is_liquid:
            logger.info(f"Short Call {self.underlying}: {liq_reason}")
            return None
        
        confidence = self._calculate_confidence(oi_data, volatility, sentiment)
        
        if not self.ml_override and confidence < self.min_confidence:
            return None
        
        lot_size = get_lot_size(self.underlying)
        quantity = lot_size * TRADING_CONFIG.get("default_quantity", 1)
        
        leg = OptionLeg(
            symbol=option["symbol"],
            strike=option["strike"],
            option_type="CE",
            expiry=option["expiry"],
            direction=TradeDirection.SELL,
            quantity=quantity,
            entry_price=option["ltp"],
            instrument_token=option.get("instrument_token"),
        )
        
        premium_received = option["ltp"] * quantity
        stop_loss = self.calculate_stop_loss(premium_received)
        target = self.calculate_target(premium_received)
        
        signal = StrategySignal(
            strategy_type=self.strategy_type,
            underlying=self.underlying,
            legs=[leg],
            entry_time=datetime.now(),
            confidence=confidence,
            expected_profit=target,
            max_loss=premium_received * 3,  # Risk management
            stop_loss=stop_loss,
            target=target,
            rationale=f"Bearish/Neutral sentiment, High IV ({iv_regime}), Resistance at {target_strike}",
            metrics={
                "pcr": oi_data.get("pcr"),
                "sentiment": sentiment,
                "iv_regime": iv_regime,
                "max_call_oi_strike": max_call_oi_strike,
            },
        )
        
        logger.info(f"Short Call signal generated: {option['symbol']} @ {option['ltp']}")
        return signal
    
    def _calculate_confidence(self, oi_data: Dict, volatility: Dict, sentiment: str) -> float:
        confidence = 0.5
        
        if sentiment == "STRONGLY_BEARISH":
            confidence += 0.15
        elif sentiment in ["BEARISH", "NEUTRAL"]:
            confidence += 0.1
        
        iv_regime = volatility.get("volatility_regime", "NORMAL")
        if iv_regime in ["HIGH_IV", "IV_ELEVATED"]:
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def calculate_stop_loss(self, entry_price: float, **kwargs) -> float:
        # For short options, SL is when premium increases
        return entry_price * 2  # Exit if premium doubles
    
    def calculate_target(self, entry_price: float, **kwargs) -> float:
        # Target percentage of premium decay; stocks get higher target
        target_pct = 0.50 if self.is_index else 0.65
        return entry_price * target_pct


class ShortPutStrategy(BaseStrategy):
    """
    Short Put Strategy - Sell a put option.
    Neutral to bullish outlook, limited profit, significant risk.
    """
    
    def __init__(self, underlying: str):
        super().__init__(underlying)
        self.name = "Short Put"
        self.description = "Sell a put option for neutral/bullish outlook"
        self.strategy_type = StrategyType.SHORT_PUT
        self.min_confidence = 0.7
    
    def analyze(self, options_chain: pd.DataFrame, metrics: Dict[str, Any]) -> Optional[StrategySignal]:
        """Analyze and generate signal for short put."""
        is_valid, reason = self.validate_conditions(metrics)
        if not is_valid:
            return None
        
        oi_data = metrics.get("oi_data", {})
        volatility = metrics.get("volatility", {})
        spot = metrics.get("spot", 0)
        
        sentiment = oi_data.get("sentiment", "NEUTRAL")
        if not self.ml_override and sentiment in ["BEARISH", "STRONGLY_BEARISH"]:
            return None
        
        iv_regime = volatility.get("volatility_regime", "NORMAL")
        if not self.ml_override and iv_regime in ["LOW_IV", "IV_DEPRESSED"]:
            return None
        
        puts = options_chain[options_chain["option_type"] == "PE"]
        if puts.empty:
            return None
        
        strike_interval = 50 if self.underlying in ["NIFTY", "FINNIFTY"] else 100
        atm_strike = round(spot / strike_interval) * strike_interval
        
        # Sell at max put OI (support level)
        max_put_oi_strike = oi_data.get("max_put_oi_strike", atm_strike - strike_interval * 2)
        target_strike = max_put_oi_strike
        
        option = puts[puts["strike"] == target_strike]
        if option.empty:
            return None
        
        option = option.iloc[0]
        
        # Liquidity guard
        is_liquid, liq_reason = self.check_leg_liquidity(option)
        if not is_liquid:
            logger.info(f"Short Put {self.underlying}: {liq_reason}")
            return None
        
        confidence = self._calculate_confidence(oi_data, volatility, sentiment)
        
        if not self.ml_override and confidence < self.min_confidence:
            return None
        
        lot_size = get_lot_size(self.underlying)
        quantity = lot_size * TRADING_CONFIG.get("default_quantity", 1)
        
        leg = OptionLeg(
            symbol=option["symbol"],
            strike=option["strike"],
            option_type="PE",
            expiry=option["expiry"],
            direction=TradeDirection.SELL,
            quantity=quantity,
            entry_price=option["ltp"],
            instrument_token=option.get("instrument_token"),
        )
        
        premium_received = option["ltp"] * quantity
        stop_loss = self.calculate_stop_loss(premium_received)
        target = self.calculate_target(premium_received)
        
        signal = StrategySignal(
            strategy_type=self.strategy_type,
            underlying=self.underlying,
            legs=[leg],
            entry_time=datetime.now(),
            confidence=confidence,
            expected_profit=target,
            max_loss=premium_received * 3,
            stop_loss=stop_loss,
            target=target,
            rationale=f"Bullish/Neutral sentiment, High IV ({iv_regime}), Support at {target_strike}",
            metrics={
                "pcr": oi_data.get("pcr"),
                "sentiment": sentiment,
                "iv_regime": iv_regime,
                "max_put_oi_strike": max_put_oi_strike,
            },
        )
        
        logger.info(f"Short Put signal generated: {option['symbol']} @ {option['ltp']}")
        return signal
    
    def _calculate_confidence(self, oi_data: Dict, volatility: Dict, sentiment: str) -> float:
        confidence = 0.5
        
        if sentiment == "STRONGLY_BULLISH":
            confidence += 0.15
        elif sentiment in ["BULLISH", "NEUTRAL"]:
            confidence += 0.1
        
        iv_regime = volatility.get("volatility_regime", "NORMAL")
        if iv_regime in ["HIGH_IV", "IV_ELEVATED"]:
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def calculate_stop_loss(self, entry_price: float, **kwargs) -> float:
        return entry_price * 2
    
    def calculate_target(self, entry_price: float, **kwargs) -> float:
        # Target percentage of premium decay; stocks get higher target
        target_pct = 0.50 if self.is_index else 0.65
        return entry_price * target_pct
