"""
Spread Strategies - Bull Call Spread, Bear Put Spread
"""
from datetime import datetime
from typing import Optional, Dict, Any, List

import pandas as pd

from .base_strategy import (
    BaseStrategy, StrategyType, StrategySignal, OptionLeg, TradeDirection
)
from config.settings import TRADING_CONFIG, STRATEGY_CONFIG, UNDERLYING_ASSETS, get_lot_size
from core.logger import logger


class BullCallSpreadStrategy(BaseStrategy):
    """
    Bull Call Spread - Buy lower strike call, sell higher strike call.
    Moderately bullish, limited risk, limited profit.
    """
    
    def __init__(self, underlying: str):
        super().__init__(underlying)
        self.name = "Bull Call Spread"
        self.description = "Buy lower strike call, sell higher strike call"
        self.strategy_type = StrategyType.BULL_CALL_SPREAD
    
    def analyze(self, options_chain: pd.DataFrame, metrics: Dict[str, Any]) -> Optional[StrategySignal]:
        """Analyze and generate signal for bull call spread."""
        is_valid, reason = self.validate_conditions(metrics)
        if not is_valid:
            return None
        
        oi_data = metrics.get("oi_data", {})
        volatility = metrics.get("volatility", {})
        spot = metrics.get("spot", 0)
        
        sentiment = oi_data.get("sentiment", "NEUTRAL")
        if sentiment not in ["BULLISH", "STRONGLY_BULLISH", "NEUTRAL"]:
            return None
        
        calls = options_chain[options_chain["option_type"] == "CE"].sort_values("strike")
        if len(calls) < 2:
            return None
        
        strike_interval = 50 if self.underlying in ["NIFTY", "FINNIFTY"] else 100
        atm_strike = round(spot / strike_interval) * strike_interval
        
        # Buy ATM/slightly ITM, Sell OTM
        buy_strike = atm_strike
        sell_strike = atm_strike + (2 * strike_interval)
        
        buy_option = calls[calls["strike"] == buy_strike]
        sell_option = calls[calls["strike"] == sell_strike]
        
        if buy_option.empty or sell_option.empty:
            return None
        
        buy_option = buy_option.iloc[0]
        sell_option = sell_option.iloc[0]
        
        # Calculate net debit
        net_debit = buy_option["ltp"] - sell_option["ltp"]
        max_profit = (sell_strike - buy_strike) - net_debit
        
        if max_profit <= 0:
            return None
        
        confidence = self._calculate_confidence(oi_data, volatility, sentiment, net_debit, max_profit)
        
        if confidence < self.min_confidence:
            return None
        
        lot_size = get_lot_size(self.underlying)
        quantity = lot_size * TRADING_CONFIG.get("default_quantity", 1)
        
        legs = [
            OptionLeg(
                symbol=buy_option["symbol"],
                strike=buy_option["strike"],
                option_type="CE",
                expiry=buy_option["expiry"],
                direction=TradeDirection.BUY,
                quantity=quantity,
                entry_price=buy_option["ltp"],
                instrument_token=buy_option.get("instrument_token"),
            ),
            OptionLeg(
                symbol=sell_option["symbol"],
                strike=sell_option["strike"],
                option_type="CE",
                expiry=sell_option["expiry"],
                direction=TradeDirection.SELL,
                quantity=quantity,
                entry_price=sell_option["ltp"],
                instrument_token=sell_option.get("instrument_token"),
            ),
        ]
        
        total_debit = net_debit * quantity
        stop_loss = self.calculate_stop_loss(total_debit)
        target = self.calculate_target(max_profit * quantity)
        
        signal = StrategySignal(
            strategy_type=self.strategy_type,
            underlying=self.underlying,
            legs=legs,
            entry_time=datetime.now(),
            confidence=confidence,
            expected_profit=max_profit * quantity,
            max_loss=total_debit,
            stop_loss=stop_loss,
            target=target,
            rationale=f"Bull Call Spread: {buy_strike}/{sell_strike}, Net debit: {net_debit:.2f}",
            metrics={
                "net_debit": net_debit,
                "max_profit": max_profit,
                "breakeven": buy_strike + net_debit,
                "risk_reward": max_profit / net_debit if net_debit > 0 else 0,
            },
        )
        
        logger.info(f"Bull Call Spread signal: {buy_strike}/{sell_strike} for {net_debit:.2f}")
        return signal
    
    def _calculate_confidence(
        self, oi_data: Dict, volatility: Dict, sentiment: str, 
        net_debit: float, max_profit: float
    ) -> float:
        confidence = 0.5
        
        if sentiment == "STRONGLY_BULLISH":
            confidence += 0.15
        elif sentiment == "BULLISH":
            confidence += 0.1
        
        # Better risk/reward increases confidence
        rr_ratio = max_profit / net_debit if net_debit > 0 else 0
        if rr_ratio > 2:
            confidence += 0.15
        elif rr_ratio > 1.5:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def calculate_stop_loss(self, entry_price: float, **kwargs) -> float:
        return entry_price * 0.5  # Exit at 50% loss of debit
    
    def calculate_target(self, max_profit: float, **kwargs) -> float:
        return max_profit * 0.7  # Target 70% of max profit


class BearPutSpreadStrategy(BaseStrategy):
    """
    Bear Put Spread - Buy higher strike put, sell lower strike put.
    Moderately bearish, limited risk, limited profit.
    """
    
    def __init__(self, underlying: str):
        super().__init__(underlying)
        self.name = "Bear Put Spread"
        self.description = "Buy higher strike put, sell lower strike put"
        self.strategy_type = StrategyType.BEAR_PUT_SPREAD
    
    def analyze(self, options_chain: pd.DataFrame, metrics: Dict[str, Any]) -> Optional[StrategySignal]:
        """Analyze and generate signal for bear put spread."""
        is_valid, reason = self.validate_conditions(metrics)
        if not is_valid:
            return None
        
        oi_data = metrics.get("oi_data", {})
        volatility = metrics.get("volatility", {})
        spot = metrics.get("spot", 0)
        
        sentiment = oi_data.get("sentiment", "NEUTRAL")
        if sentiment not in ["BEARISH", "STRONGLY_BEARISH", "NEUTRAL"]:
            return None
        
        puts = options_chain[options_chain["option_type"] == "PE"].sort_values("strike")
        if len(puts) < 2:
            return None
        
        strike_interval = 50 if self.underlying in ["NIFTY", "FINNIFTY"] else 100
        atm_strike = round(spot / strike_interval) * strike_interval
        
        # Buy ATM/slightly ITM, Sell OTM
        buy_strike = atm_strike
        sell_strike = atm_strike - (2 * strike_interval)
        
        buy_option = puts[puts["strike"] == buy_strike]
        sell_option = puts[puts["strike"] == sell_strike]
        
        if buy_option.empty or sell_option.empty:
            return None
        
        buy_option = buy_option.iloc[0]
        sell_option = sell_option.iloc[0]
        
        # Calculate net debit
        net_debit = buy_option["ltp"] - sell_option["ltp"]
        max_profit = (buy_strike - sell_strike) - net_debit
        
        if max_profit <= 0:
            return None
        
        confidence = self._calculate_confidence(oi_data, volatility, sentiment, net_debit, max_profit)
        
        if confidence < self.min_confidence:
            return None
        
        lot_size = get_lot_size(self.underlying)
        quantity = lot_size * TRADING_CONFIG.get("default_quantity", 1)
        
        legs = [
            OptionLeg(
                symbol=buy_option["symbol"],
                strike=buy_option["strike"],
                option_type="PE",
                expiry=buy_option["expiry"],
                direction=TradeDirection.BUY,
                quantity=quantity,
                entry_price=buy_option["ltp"],
                instrument_token=buy_option.get("instrument_token"),
            ),
            OptionLeg(
                symbol=sell_option["symbol"],
                strike=sell_option["strike"],
                option_type="PE",
                expiry=sell_option["expiry"],
                direction=TradeDirection.SELL,
                quantity=quantity,
                entry_price=sell_option["ltp"],
                instrument_token=sell_option.get("instrument_token"),
            ),
        ]
        
        total_debit = net_debit * quantity
        stop_loss = self.calculate_stop_loss(total_debit)
        target = self.calculate_target(max_profit * quantity)
        
        signal = StrategySignal(
            strategy_type=self.strategy_type,
            underlying=self.underlying,
            legs=legs,
            entry_time=datetime.now(),
            confidence=confidence,
            expected_profit=max_profit * quantity,
            max_loss=total_debit,
            stop_loss=stop_loss,
            target=target,
            rationale=f"Bear Put Spread: {buy_strike}/{sell_strike}, Net debit: {net_debit:.2f}",
            metrics={
                "net_debit": net_debit,
                "max_profit": max_profit,
                "breakeven": buy_strike - net_debit,
                "risk_reward": max_profit / net_debit if net_debit > 0 else 0,
            },
        )
        
        logger.info(f"Bear Put Spread signal: {buy_strike}/{sell_strike} for {net_debit:.2f}")
        return signal
    
    def _calculate_confidence(
        self, oi_data: Dict, volatility: Dict, sentiment: str,
        net_debit: float, max_profit: float
    ) -> float:
        confidence = 0.5
        
        if sentiment == "STRONGLY_BEARISH":
            confidence += 0.15
        elif sentiment == "BEARISH":
            confidence += 0.1
        
        rr_ratio = max_profit / net_debit if net_debit > 0 else 0
        if rr_ratio > 2:
            confidence += 0.15
        elif rr_ratio > 1.5:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def calculate_stop_loss(self, entry_price: float, **kwargs) -> float:
        return entry_price * 0.5
    
    def calculate_target(self, max_profit: float, **kwargs) -> float:
        return max_profit * 0.7
