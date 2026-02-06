"""
Neutral/Volatility Strategies - Iron Condor, Straddle, Strangle
"""
from datetime import datetime
from typing import Optional, Dict, Any

import pandas as pd

from .base_strategy import (
    BaseStrategy, StrategyType, StrategySignal, OptionLeg, TradeDirection
)
from config.settings import TRADING_CONFIG, UNDERLYING_ASSETS, get_lot_size
from core.logger import logger


class IronCondorStrategy(BaseStrategy):
    """
    Iron Condor - Sell OTM put spread + Sell OTM call spread.
    Neutral outlook, profit from low volatility and time decay.
    """
    
    def __init__(self, underlying: str):
        super().__init__(underlying)
        self.name = "Iron Condor"
        self.description = "Sell OTM put spread and OTM call spread"
        self.strategy_type = StrategyType.IRON_CONDOR
        self.min_confidence = 0.65
    
    def analyze(self, options_chain: pd.DataFrame, metrics: Dict[str, Any]) -> Optional[StrategySignal]:
        """Analyze and generate signal for iron condor."""
        is_valid, reason = self.validate_conditions(metrics)
        if not is_valid:
            return None
        
        oi_data = metrics.get("oi_data", {})
        volatility = metrics.get("volatility", {})
        spot = metrics.get("spot", 0)
        
        sentiment = oi_data.get("sentiment", "NEUTRAL")
        iv_regime = volatility.get("volatility_regime", "NORMAL")
        
        # Best for neutral sentiment and high IV
        if sentiment in ["STRONGLY_BULLISH", "STRONGLY_BEARISH"]:
            return None
        
        if iv_regime in ["LOW_IV", "IV_DEPRESSED"]:
            return None
        
        calls = options_chain[options_chain["option_type"] == "CE"].sort_values("strike")
        puts = options_chain[options_chain["option_type"] == "PE"].sort_values("strike")
        
        if len(calls) < 2 or len(puts) < 2:
            return None
        
        strike_interval = 50 if self.underlying in ["NIFTY", "FINNIFTY"] else 100
        atm_strike = round(spot / strike_interval) * strike_interval
        
        # Use max OI strikes as short strikes
        max_call_strike = oi_data.get("max_call_oi_strike", atm_strike + 2 * strike_interval)
        max_put_strike = oi_data.get("max_put_oi_strike", atm_strike - 2 * strike_interval)
        
        # Define strikes
        sell_call_strike = max_call_strike
        buy_call_strike = sell_call_strike + strike_interval
        sell_put_strike = max_put_strike
        buy_put_strike = sell_put_strike - strike_interval
        
        # Get options
        sell_call = calls[calls["strike"] == sell_call_strike]
        buy_call = calls[calls["strike"] == buy_call_strike]
        sell_put = puts[puts["strike"] == sell_put_strike]
        buy_put = puts[puts["strike"] == buy_put_strike]
        
        if any(opt.empty for opt in [sell_call, buy_call, sell_put, buy_put]):
            return None
        
        sell_call = sell_call.iloc[0]
        buy_call = buy_call.iloc[0]
        sell_put = sell_put.iloc[0]
        buy_put = buy_put.iloc[0]
        
        # Calculate premiums
        call_spread_credit = sell_call["ltp"] - buy_call["ltp"]
        put_spread_credit = sell_put["ltp"] - buy_put["ltp"]
        net_credit = call_spread_credit + put_spread_credit
        
        # Max loss is width of spread minus credit
        spread_width = strike_interval
        max_loss = spread_width - net_credit
        
        if net_credit <= 0 or max_loss <= 0:
            return None
        
        confidence = self._calculate_confidence(oi_data, volatility, sentiment, net_credit, max_loss)
        
        if confidence < self.min_confidence:
            return None
        
        lot_size = get_lot_size(self.underlying)
        quantity = lot_size * TRADING_CONFIG.get("default_quantity", 1)
        
        legs = [
            OptionLeg(
                symbol=sell_call["symbol"],
                strike=sell_call["strike"],
                option_type="CE",
                expiry=sell_call["expiry"],
                direction=TradeDirection.SELL,
                quantity=quantity,
                entry_price=sell_call["ltp"],
            ),
            OptionLeg(
                symbol=buy_call["symbol"],
                strike=buy_call["strike"],
                option_type="CE",
                expiry=buy_call["expiry"],
                direction=TradeDirection.BUY,
                quantity=quantity,
                entry_price=buy_call["ltp"],
            ),
            OptionLeg(
                symbol=sell_put["symbol"],
                strike=sell_put["strike"],
                option_type="PE",
                expiry=sell_put["expiry"],
                direction=TradeDirection.SELL,
                quantity=quantity,
                entry_price=sell_put["ltp"],
            ),
            OptionLeg(
                symbol=buy_put["symbol"],
                strike=buy_put["strike"],
                option_type="PE",
                expiry=buy_put["expiry"],
                direction=TradeDirection.BUY,
                quantity=quantity,
                entry_price=buy_put["ltp"],
            ),
        ]
        
        total_credit = net_credit * quantity
        stop_loss = self.calculate_stop_loss(total_credit, max_loss=max_loss * quantity)
        target = self.calculate_target(total_credit)
        
        signal = StrategySignal(
            strategy_type=self.strategy_type,
            underlying=self.underlying,
            legs=legs,
            entry_time=datetime.now(),
            confidence=confidence,
            expected_profit=total_credit,
            max_loss=max_loss * quantity,
            stop_loss=stop_loss,
            target=target,
            rationale=f"Iron Condor: PE {buy_put_strike}/{sell_put_strike} - CE {sell_call_strike}/{buy_call_strike}",
            metrics={
                "net_credit": net_credit,
                "max_loss": max_loss,
                "breakeven_lower": sell_put_strike - net_credit,
                "breakeven_upper": sell_call_strike + net_credit,
                "profit_range": f"{sell_put_strike} - {sell_call_strike}",
            },
        )
        
        logger.info(f"Iron Condor signal: {sell_put_strike}/{sell_call_strike}")
        return signal
    
    def _calculate_confidence(
        self, oi_data: Dict, volatility: Dict, sentiment: str,
        net_credit: float, max_loss: float
    ) -> float:
        confidence = 0.5
        
        if sentiment == "NEUTRAL":
            confidence += 0.15
        
        iv_regime = volatility.get("volatility_regime", "NORMAL")
        if iv_regime in ["HIGH_IV", "IV_ELEVATED"]:
            confidence += 0.15
        
        rr_ratio = net_credit / max_loss if max_loss > 0 else 0
        if rr_ratio > 0.5:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def calculate_stop_loss(self, entry_credit: float, max_loss: float = None, **kwargs) -> float:
        # Exit when loss equals credit received
        return entry_credit
    
    def calculate_target(self, entry_credit: float, **kwargs) -> float:
        return entry_credit * 0.5  # Target 50% of max profit


class StraddleStrategy(BaseStrategy):
    """
    Long Straddle - Buy ATM call and ATM put.
    Expects big move in either direction.
    """
    
    def __init__(self, underlying: str, is_short: bool = False):
        super().__init__(underlying)
        self.is_short = is_short
        self.name = "Short Straddle" if is_short else "Long Straddle"
        self.description = "Sell ATM call and put" if is_short else "Buy ATM call and put"
        self.strategy_type = StrategyType.STRADDLE
    
    def analyze(self, options_chain: pd.DataFrame, metrics: Dict[str, Any]) -> Optional[StrategySignal]:
        """Analyze and generate signal for straddle."""
        # Straddles require high liquidity - restrict to indices only
        INDEX_UNDERLYINGS = ["NIFTY", "BANKNIFTY", "FINNIFTY", "NIFTY 50", "NIFTY BANK"]
        if self.underlying not in INDEX_UNDERLYINGS:
            logger.debug(f"Straddle skipped for {self.underlying} - index-only strategy")
            return None
        
        is_valid, reason = self.validate_conditions(metrics)
        if not is_valid:
            return None
        
        oi_data = metrics.get("oi_data", {})
        volatility = metrics.get("volatility", {})
        spot = metrics.get("spot", 0)
        
        iv_regime = volatility.get("volatility_regime", "NORMAL")
        
        # Long straddle: prefer low IV (expect volatility expansion)
        # Short straddle: prefer high IV (expect volatility contraction)
        if self.is_short:
            if iv_regime in ["LOW_IV", "IV_DEPRESSED"]:
                return None
        else:
            if iv_regime in ["HIGH_IV", "IV_ELEVATED"]:
                return None
        
        strike_interval = 50 if self.underlying in ["NIFTY", "FINNIFTY"] else 100
        atm_strike = round(spot / strike_interval) * strike_interval
        
        calls = options_chain[options_chain["option_type"] == "CE"]
        puts = options_chain[options_chain["option_type"] == "PE"]
        
        call_option = calls[calls["strike"] == atm_strike]
        put_option = puts[puts["strike"] == atm_strike]
        
        if call_option.empty or put_option.empty:
            return None
        
        call_option = call_option.iloc[0]
        put_option = put_option.iloc[0]
        
        total_premium = call_option["ltp"] + put_option["ltp"]
        
        confidence = self._calculate_confidence(oi_data, volatility)
        
        if confidence < self.min_confidence:
            return None
        
        lot_size = get_lot_size(self.underlying)
        quantity = lot_size * TRADING_CONFIG.get("default_quantity", 1)
        
        direction = TradeDirection.SELL if self.is_short else TradeDirection.BUY
        
        legs = [
            OptionLeg(
                symbol=call_option["symbol"],
                strike=atm_strike,
                option_type="CE",
                expiry=call_option["expiry"],
                direction=direction,
                quantity=quantity,
                entry_price=call_option["ltp"],
            ),
            OptionLeg(
                symbol=put_option["symbol"],
                strike=atm_strike,
                option_type="PE",
                expiry=put_option["expiry"],
                direction=direction,
                quantity=quantity,
                entry_price=put_option["ltp"],
            ),
        ]
        
        total_cost = total_premium * quantity
        
        if self.is_short:
            expected_profit = total_cost * 0.5
            max_loss = total_cost * 2
        else:
            expected_profit = total_cost
            max_loss = total_cost
        
        signal = StrategySignal(
            strategy_type=self.strategy_type,
            underlying=self.underlying,
            legs=legs,
            entry_time=datetime.now(),
            confidence=confidence,
            expected_profit=expected_profit,
            max_loss=max_loss,
            stop_loss=self.calculate_stop_loss(total_cost),
            target=self.calculate_target(total_cost),
            rationale=f"{'Short' if self.is_short else 'Long'} Straddle @ {atm_strike}, Premium: {total_premium:.2f}",
            metrics={
                "strike": atm_strike,
                "total_premium": total_premium,
                "breakeven_upper": atm_strike + total_premium,
                "breakeven_lower": atm_strike - total_premium,
            },
        )
        
        logger.info(f"{'Short' if self.is_short else 'Long'} Straddle signal @ {atm_strike}")
        return signal
    
    def _calculate_confidence(self, oi_data: Dict, volatility: Dict) -> float:
        confidence = 0.5
        
        iv_regime = volatility.get("volatility_regime", "NORMAL")
        
        if self.is_short:
            if iv_regime in ["HIGH_IV", "IV_ELEVATED"]:
                confidence += 0.2
        else:
            if iv_regime in ["LOW_IV", "IV_DEPRESSED"]:
                confidence += 0.2
        
        return min(confidence, 1.0)
    
    def calculate_stop_loss(self, entry_price: float, **kwargs) -> float:
        if self.is_short:
            return entry_price  # Exit when loss equals credit
        return entry_price * 0.5  # 50% loss for long
    
    def calculate_target(self, entry_price: float, **kwargs) -> float:
        if self.is_short:
            return entry_price * 0.5  # 50% of credit
        return entry_price  # 100% profit for long


class StrangleStrategy(BaseStrategy):
    """
    Strangle - Buy/Sell OTM call and OTM put.
    Similar to straddle but with different strikes.
    """
    
    def __init__(self, underlying: str, is_short: bool = False):
        super().__init__(underlying)
        self.is_short = is_short
        self.name = "Short Strangle" if is_short else "Long Strangle"
        self.description = "Sell OTM call and put" if is_short else "Buy OTM call and put"
        self.strategy_type = StrategyType.STRANGLE
    
    def analyze(self, options_chain: pd.DataFrame, metrics: Dict[str, Any]) -> Optional[StrategySignal]:
        """Analyze and generate signal for strangle."""
        # Strangles require high liquidity - restrict to indices only
        INDEX_UNDERLYINGS = ["NIFTY", "BANKNIFTY", "FINNIFTY", "NIFTY 50", "NIFTY BANK"]
        if self.underlying not in INDEX_UNDERLYINGS:
            logger.debug(f"Strangle skipped for {self.underlying} - index-only strategy")
            return None
        
        is_valid, reason = self.validate_conditions(metrics)
        if not is_valid:
            return None
        
        oi_data = metrics.get("oi_data", {})
        volatility = metrics.get("volatility", {})
        spot = metrics.get("spot", 0)
        
        iv_regime = volatility.get("volatility_regime", "NORMAL")
        
        if self.is_short:
            if iv_regime in ["LOW_IV", "IV_DEPRESSED"]:
                return None
        else:
            if iv_regime in ["HIGH_IV", "IV_ELEVATED"]:
                return None
        
        strike_interval = 50 if self.underlying in ["NIFTY", "FINNIFTY"] else 100
        atm_strike = round(spot / strike_interval) * strike_interval
        
        # Use max OI strikes for short strangle
        if self.is_short:
            call_strike = oi_data.get("max_call_oi_strike", atm_strike + 2 * strike_interval)
            put_strike = oi_data.get("max_put_oi_strike", atm_strike - 2 * strike_interval)
        else:
            call_strike = atm_strike + strike_interval
            put_strike = atm_strike - strike_interval
        
        calls = options_chain[options_chain["option_type"] == "CE"]
        puts = options_chain[options_chain["option_type"] == "PE"]
        
        call_option = calls[calls["strike"] == call_strike]
        put_option = puts[puts["strike"] == put_strike]
        
        if call_option.empty or put_option.empty:
            return None
        
        call_option = call_option.iloc[0]
        put_option = put_option.iloc[0]
        
        total_premium = call_option["ltp"] + put_option["ltp"]
        
        confidence = self._calculate_confidence(oi_data, volatility)
        
        if confidence < self.min_confidence:
            return None
        
        lot_size = get_lot_size(self.underlying)
        quantity = lot_size * TRADING_CONFIG.get("default_quantity", 1)
        
        direction = TradeDirection.SELL if self.is_short else TradeDirection.BUY
        
        legs = [
            OptionLeg(
                symbol=call_option["symbol"],
                strike=call_strike,
                option_type="CE",
                expiry=call_option["expiry"],
                direction=direction,
                quantity=quantity,
                entry_price=call_option["ltp"],
            ),
            OptionLeg(
                symbol=put_option["symbol"],
                strike=put_strike,
                option_type="PE",
                expiry=put_option["expiry"],
                direction=direction,
                quantity=quantity,
                entry_price=put_option["ltp"],
            ),
        ]
        
        total_cost = total_premium * quantity
        
        if self.is_short:
            expected_profit = total_cost * 0.5
            max_loss = total_cost * 3
        else:
            expected_profit = total_cost * 1.5
            max_loss = total_cost
        
        signal = StrategySignal(
            strategy_type=self.strategy_type,
            underlying=self.underlying,
            legs=legs,
            entry_time=datetime.now(),
            confidence=confidence,
            expected_profit=expected_profit,
            max_loss=max_loss,
            stop_loss=self.calculate_stop_loss(total_cost),
            target=self.calculate_target(total_cost),
            rationale=f"{'Short' if self.is_short else 'Long'} Strangle: {put_strike}PE / {call_strike}CE",
            metrics={
                "call_strike": call_strike,
                "put_strike": put_strike,
                "total_premium": total_premium,
                "breakeven_upper": call_strike + total_premium,
                "breakeven_lower": put_strike - total_premium,
            },
        )
        
        logger.info(f"{'Short' if self.is_short else 'Long'} Strangle signal: {put_strike}/{call_strike}")
        return signal
    
    def _calculate_confidence(self, oi_data: Dict, volatility: Dict) -> float:
        confidence = 0.5
        
        iv_regime = volatility.get("volatility_regime", "NORMAL")
        
        if self.is_short:
            if iv_regime in ["HIGH_IV", "IV_ELEVATED"]:
                confidence += 0.2
        else:
            if iv_regime in ["LOW_IV", "IV_DEPRESSED"]:
                confidence += 0.2
        
        return min(confidence, 1.0)
    
    def calculate_stop_loss(self, entry_price: float, **kwargs) -> float:
        if self.is_short:
            return entry_price
        return entry_price * 0.5
    
    def calculate_target(self, entry_price: float, **kwargs) -> float:
        if self.is_short:
            return entry_price * 0.5
        return entry_price
