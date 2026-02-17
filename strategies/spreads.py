"""
Spread Strategies - Bull Call Spread, Bear Put Spread, Bear Call Spread (Credit), Bull Put Spread (Credit)
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
            logger.debug(f"Bull Call Spread {self.underlying}: validation failed - {reason}")
            return None
        
        oi_data = metrics.get("oi_data", {})
        volatility = metrics.get("volatility", {})
        spot = metrics.get("spot", 0)
        
        sentiment = oi_data.get("sentiment", "NEUTRAL")
        if not self.ml_override and sentiment not in ["BULLISH", "STRONGLY_BULLISH", "NEUTRAL"]:
            logger.info(f"Bull Call Spread {self.underlying}: skipped - OI sentiment is {sentiment}")
            return None
        
        calls = options_chain[options_chain["option_type"] == "CE"].sort_values("strike")
        if len(calls) < 2:
            logger.info(f"Bull Call Spread {self.underlying}: skipped - not enough CE options")
            return None
        
        strike_interval = 50 if self.underlying in ["NIFTY", "FINNIFTY"] else 100
        atm_strike = round(spot / strike_interval) * strike_interval
        
        # Buy ATM/slightly ITM, Sell OTM (ATR-based spread width)
        available_strikes = sorted(calls["strike"].unique())
        
        # Derive actual strike interval from the chain (NSE uses 5/10/25/50/100)
        if len(available_strikes) >= 2:
            chain_diffs = [available_strikes[i+1] - available_strikes[i] for i in range(len(available_strikes)-1)]
            actual_interval = min(chain_diffs)
        else:
            actual_interval = strike_interval
        
        # ATR-based spread width: span at least 1 ATR so the spread covers normal movement
        historical = metrics.get("historical", {})
        atr_14 = historical.get("atr_14", actual_interval * 5) if isinstance(historical, dict) else actual_interval * 5
        min_spread_strikes = max(2, round(atr_14 / actual_interval))
        
        logger.debug(f"Bull Call Spread {self.underlying}: chain_interval={actual_interval}, ATR={atr_14:.1f}, min_spread_strikes={min_spread_strikes}")
        buy_strike = min(available_strikes, key=lambda s: abs(s - atm_strike)) if available_strikes else atm_strike
        sell_candidates = [s for s in available_strikes if s > buy_strike]
        # Pick the strike at least min_spread_strikes away (ATR-based)
        if len(sell_candidates) >= min_spread_strikes:
            sell_strike = sell_candidates[min_spread_strikes - 1]
        elif sell_candidates:
            sell_strike = sell_candidates[-1]  # Farthest available
        else:
            return None
        
        buy_option = calls[calls["strike"] == buy_strike]
        sell_option = calls[calls["strike"] == sell_strike]
        
        if buy_option.empty or sell_option.empty:
            logger.info(f"Bull Call Spread {self.underlying}: skipped - strikes {buy_strike}/{sell_strike} not found")
            return None
        
        buy_option = buy_option.iloc[0]
        sell_option = sell_option.iloc[0]
        
        # Liquidity guard
        for leg_row in [buy_option, sell_option]:
            is_liquid, liq_reason = self.check_leg_liquidity(leg_row)
            if not is_liquid:
                logger.info(f"Bull Call Spread {self.underlying}: {liq_reason}")
                return None
        
        # Calculate net debit
        net_debit = buy_option["ltp"] - sell_option["ltp"]
        strike_width = sell_strike - buy_strike
        max_profit = strike_width - net_debit
        
        if max_profit <= 0 or net_debit <= 0:
            return None
        
        # Realistic expected profit: Stocks have higher margins so target more of the spread
        # Indices: capture 35% of max profit; Stocks: capture 50%
        realistic_profit_pct = 0.35 if self.is_index else 0.50
        expected_profit_per_share = max_profit * realistic_profit_pct
        
        # Risk/Reward based on realistic expectation
        realistic_rr = expected_profit_per_share / net_debit if net_debit > 0 else 0
        
        confidence = self._calculate_confidence(oi_data, volatility, sentiment, net_debit, max_profit)
        
        if not self.ml_override and confidence < self.min_confidence:
            logger.info(f"Bull Call Spread {self.underlying}: skipped - strategy confidence {confidence:.2f} < {self.min_confidence}")
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
        realistic_expected = expected_profit_per_share * quantity
        stop_loss = self.calculate_stop_loss(total_debit)
        target = self.calculate_target(realistic_expected)
        
        signal = StrategySignal(
            strategy_type=self.strategy_type,
            underlying=self.underlying,
            legs=legs,
            entry_time=datetime.now(),
            confidence=confidence,
            expected_profit=realistic_expected,
            max_loss=total_debit,
            stop_loss=stop_loss,
            target=target,
            rationale=f"Bull Call Spread: {buy_strike}/{sell_strike}, Net debit: {net_debit:.2f}",
            metrics={
                "net_debit": net_debit,
                "max_profit_per_share": max_profit,
                "expected_profit_per_share": expected_profit_per_share,
                "strike_width": strike_width,
                "breakeven": buy_strike + net_debit,
                "theoretical_rr": max_profit / net_debit if net_debit > 0 else 0,
                "realistic_rr": realistic_rr,
            },
        )
        
        logger.info(f"Bull Call Spread signal: {buy_strike}/{sell_strike} for {net_debit:.2f}, Realistic RR: {realistic_rr:.2f}")
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
        # Indices: 70% of realistic expected; Stocks: 85% (higher margin justifies higher target)
        target_pct = 0.70 if self.is_index else 0.85
        return max_profit * target_pct


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
            logger.debug(f"Bear Put Spread {self.underlying}: validation failed - {reason}")
            return None
        
        oi_data = metrics.get("oi_data", {})
        volatility = metrics.get("volatility", {})
        spot = metrics.get("spot", 0)
        
        sentiment = oi_data.get("sentiment", "NEUTRAL")
        if not self.ml_override and sentiment not in ["BEARISH", "STRONGLY_BEARISH", "NEUTRAL"]:
            logger.info(f"Bear Put Spread {self.underlying}: skipped - OI sentiment is {sentiment}")
            return None
        
        puts = options_chain[options_chain["option_type"] == "PE"].sort_values("strike")
        if len(puts) < 2:
            logger.info(f"Bear Put Spread {self.underlying}: skipped - not enough PE options")
            return None
        
        strike_interval = 50 if self.underlying in ["NIFTY", "FINNIFTY"] else 100
        atm_strike = round(spot / strike_interval) * strike_interval
        
        # Buy ATM/slightly ITM, Sell OTM (ATR-based spread width)
        available_strikes = sorted(puts["strike"].unique())
        
        # Derive actual strike interval from the chain (NSE uses 5/10/25/50/100)
        if len(available_strikes) >= 2:
            chain_diffs = [available_strikes[i+1] - available_strikes[i] for i in range(len(available_strikes)-1)]
            actual_interval = min(chain_diffs)
        else:
            actual_interval = strike_interval
        
        # ATR-based spread width: span at least 1 ATR so the spread covers normal movement
        historical = metrics.get("historical", {})
        atr_14 = historical.get("atr_14", actual_interval * 5) if isinstance(historical, dict) else actual_interval * 5
        min_spread_strikes = max(2, round(atr_14 / actual_interval))
        
        logger.debug(f"Bear Put Spread {self.underlying}: chain_interval={actual_interval}, ATR={atr_14:.1f}, min_spread_strikes={min_spread_strikes}")
        buy_strike = min(available_strikes, key=lambda s: abs(s - atm_strike)) if available_strikes else atm_strike
        sell_candidates = [s for s in available_strikes if s < buy_strike]
        # Pick the strike at least min_spread_strikes away below buy_strike
        if len(sell_candidates) >= min_spread_strikes:
            sell_strike = sell_candidates[-min_spread_strikes]  # From sorted ascending
        elif sell_candidates:
            sell_strike = sell_candidates[0]  # Farthest available below
        else:
            sell_strike = None
        
        if sell_strike is None:
            logger.info(f"Bear Put Spread {self.underlying}: skipped - no OTM put strike below {buy_strike}")
            return None
        
        buy_option = puts[puts["strike"] == buy_strike]
        sell_option = puts[puts["strike"] == sell_strike]
        
        if buy_option.empty or sell_option.empty:
            logger.info(f"Bear Put Spread {self.underlying}: skipped - strikes {buy_strike}/{sell_strike} not found")
            return None
        
        buy_option = buy_option.iloc[0]
        sell_option = sell_option.iloc[0]
        
        # Liquidity guard
        for leg_row in [buy_option, sell_option]:
            is_liquid, liq_reason = self.check_leg_liquidity(leg_row)
            if not is_liquid:
                logger.info(f"Bear Put Spread {self.underlying}: {liq_reason}")
                return None
        
        # Calculate net debit
        net_debit = buy_option["ltp"] - sell_option["ltp"]
        strike_width = buy_strike - sell_strike
        max_profit = strike_width - net_debit
        
        if max_profit <= 0 or net_debit <= 0:
            return None
        
        # Realistic expected profit: Stocks have higher margins so target more of the spread
        # Indices: capture 35% of max profit; Stocks: capture 50%
        realistic_profit_pct = 0.35 if self.is_index else 0.50
        expected_profit_per_share = max_profit * realistic_profit_pct
        
        # Risk/Reward based on realistic expectation
        realistic_rr = expected_profit_per_share / net_debit if net_debit > 0 else 0
        
        confidence = self._calculate_confidence(oi_data, volatility, sentiment, net_debit, max_profit)
        
        if not self.ml_override and confidence < self.min_confidence:
            logger.info(f"Bear Put Spread {self.underlying}: skipped - strategy confidence {confidence:.2f} < {self.min_confidence}")
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
        realistic_expected = expected_profit_per_share * quantity
        stop_loss = self.calculate_stop_loss(total_debit)
        target = self.calculate_target(realistic_expected)
        
        signal = StrategySignal(
            strategy_type=self.strategy_type,
            underlying=self.underlying,
            legs=legs,
            entry_time=datetime.now(),
            confidence=confidence,
            expected_profit=realistic_expected,
            max_loss=total_debit,
            stop_loss=stop_loss,
            target=target,
            rationale=f"Bear Put Spread: {buy_strike}/{sell_strike}, Net debit: {net_debit:.2f}",
            metrics={
                "net_debit": net_debit,
                "max_profit_per_share": max_profit,
                "expected_profit_per_share": expected_profit_per_share,
                "strike_width": strike_width,
                "breakeven": buy_strike - net_debit,
                "theoretical_rr": max_profit / net_debit if net_debit > 0 else 0,
                "realistic_rr": realistic_rr,
            },
        )
        
        logger.info(f"Bear Put Spread signal: {buy_strike}/{sell_strike} for {net_debit:.2f}, Realistic RR: {realistic_rr:.2f}")
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
        # Indices: 70% of realistic expected; Stocks: 85%
        target_pct = 0.70 if self.is_index else 0.85
        return max_profit * target_pct


class BearCallSpreadStrategy(BaseStrategy):
    """
    Bear Call Spread (Credit Spread) - SELL lower strike call, BUY higher strike call.
    Main trade is a SELL with a hedge. Bearish/Neutral outlook, profit from premium decay.
    Collects net credit, limited risk, limited profit.
    """
    
    def __init__(self, underlying: str):
        super().__init__(underlying)
        self.name = "Bear Call Spread"
        self.description = "Sell lower strike call, buy higher strike call (credit spread)"
        self.strategy_type = StrategyType.BEAR_CALL_SPREAD
        self.min_confidence = 0.65  # Higher threshold for credit spreads
    
    def analyze(self, options_chain: pd.DataFrame, metrics: Dict[str, Any]) -> Optional[StrategySignal]:
        """Analyze and generate signal for bear call spread (credit)."""
        is_valid, reason = self.validate_conditions(metrics)
        if not is_valid:
            logger.debug(f"Bear Call Spread {self.underlying}: validation failed - {reason}")
            return None
        
        oi_data = metrics.get("oi_data", {})
        volatility = metrics.get("volatility", {})
        spot = metrics.get("spot", 0)
        
        sentiment = oi_data.get("sentiment", "NEUTRAL")
        # Best for bearish or neutral sentiment (we want price to stay below short strike)
        if not self.ml_override and sentiment in ["BULLISH", "STRONGLY_BULLISH"]:
            logger.info(f"Bear Call Spread {self.underlying}: skipped - OI sentiment is {sentiment}")
            return None
        
        # Credit spreads work in any IV regime - higher IV gives better premiums
        iv_regime = volatility.get("volatility_regime", "NORMAL")
        
        calls = options_chain[options_chain["option_type"] == "CE"].sort_values("strike")
        if len(calls) < 2:
            logger.info(f"Bear Call Spread {self.underlying}: skipped - not enough CE options")
            return None
        
        strike_interval = 50 if self.underlying in ["NIFTY", "FINNIFTY"] else 100
        atm_strike = round(spot / strike_interval) * strike_interval
        
        # Derive actual strike interval from the chain (NSE uses 5/10/25/50/100)
        available_strikes = sorted(calls["strike"].unique())
        if len(available_strikes) >= 2:
            chain_diffs = [available_strikes[i+1] - available_strikes[i] for i in range(len(available_strikes)-1)]
            actual_interval = min(chain_diffs)
        else:
            actual_interval = strike_interval
        
        # ATR-based minimum OTM distance for credit spreads
        historical = metrics.get("historical", {})
        atr_14 = historical.get("atr_14", actual_interval * 5) if isinstance(historical, dict) else actual_interval * 5
        min_otm_strikes = max(2, round(atr_14 / actual_interval))
        
        # Use max call OI as resistance - sell at or near this level
        max_call_oi_strike = oi_data.get("max_call_oi_strike", atm_strike + 2 * strike_interval)
        
        # Validate max_call_oi_strike is reasonable (within 10 intervals of ATM)
        if max_call_oi_strike is None or abs(max_call_oi_strike - atm_strike) > 10 * strike_interval:
            max_call_oi_strike = atm_strike + 2 * strike_interval
        
        # Ensure sell strike is at least min_otm_strikes away from ATM (ATR-based safety)
        min_sell_strike = atm_strike + min_otm_strikes * actual_interval
        sell_strike = max(max_call_oi_strike, min_sell_strike)
        
        # SELL lower strike call (at resistance), BUY higher strike call (hedge)
        # Hedge width: at least 2 strikes or half ATR for meaningful risk cap
        min_hedge_strikes = max(2, min_otm_strikes // 2)
        
        # Find nearest available strikes in the chain
        sell_strike = min(available_strikes, key=lambda s: abs(s - sell_strike)) if available_strikes else sell_strike
        buy_candidates = [s for s in available_strikes if s > sell_strike]
        if len(buy_candidates) >= min_hedge_strikes:
            buy_strike = buy_candidates[min_hedge_strikes - 1]
        elif buy_candidates:
            buy_strike = buy_candidates[-1]
        else:
            buy_strike = sell_strike + actual_interval
        
        logger.debug(f"Bear Call Spread {self.underlying}: chain_interval={actual_interval}, ATR={atr_14:.1f}, OTM={min_otm_strikes} strikes, hedge={min_hedge_strikes} strikes")
        
        sell_option = calls[calls["strike"] == sell_strike]
        buy_option = calls[calls["strike"] == buy_strike]
        
        if sell_option.empty or buy_option.empty:
            logger.info(f"Bear Call Spread {self.underlying}: skipped - strikes {sell_strike}/{buy_strike} not found in chain")
            return None
        
        sell_option = sell_option.iloc[0]
        buy_option = buy_option.iloc[0]
        
        # Liquidity guard
        for leg_row in [sell_option, buy_option]:
            is_liquid, liq_reason = self.check_leg_liquidity(leg_row)
            if not is_liquid:
                logger.info(f"Bear Call Spread {self.underlying}: {liq_reason}")
                return None
        
        # Calculate net credit (sell premium > buy premium)
        net_credit = sell_option["ltp"] - buy_option["ltp"]
        spread_width = buy_strike - sell_strike
        max_loss = spread_width - net_credit
        
        if net_credit <= 0 or max_loss <= 0:
            logger.info(f"Bear Call Spread {self.underlying}: skipped - unfavorable credit ({net_credit:.2f})")
            return None
        
        # Risk/Reward for credit spreads
        rr_ratio = net_credit / max_loss if max_loss > 0 else 0
        
        confidence = self._calculate_confidence(oi_data, volatility, sentiment, net_credit, max_loss)
        
        if not self.ml_override and confidence < self.min_confidence:
            logger.info(f"Bear Call Spread {self.underlying}: skipped - strategy confidence {confidence:.2f} < {self.min_confidence}")
            return None
        
        lot_size = get_lot_size(self.underlying)
        quantity = lot_size * TRADING_CONFIG.get("default_quantity", 1)
        
        # SELL leg first (main trade), then BUY leg (hedge)
        legs = [
            OptionLeg(
                symbol=sell_option["symbol"],
                strike=sell_option["strike"],
                option_type="CE",
                expiry=sell_option["expiry"],
                direction=TradeDirection.SELL,  # Main trade - SELL
                quantity=quantity,
                entry_price=sell_option["ltp"],
                instrument_token=sell_option.get("instrument_token"),
            ),
            OptionLeg(
                symbol=buy_option["symbol"],
                strike=buy_option["strike"],
                option_type="CE",
                expiry=buy_option["expiry"],
                direction=TradeDirection.BUY,  # Hedge
                quantity=quantity,
                entry_price=buy_option["ltp"],
                instrument_token=buy_option.get("instrument_token"),
            ),
        ]
        
        total_credit = net_credit * quantity
        total_max_loss = max_loss * quantity
        stop_loss = self.calculate_stop_loss(total_credit, max_loss=total_max_loss)
        target = self.calculate_target(total_credit)
        
        signal = StrategySignal(
            strategy_type=self.strategy_type,
            underlying=self.underlying,
            legs=legs,
            entry_time=datetime.now(),
            confidence=confidence,
            expected_profit=total_credit,  # Max profit is credit received
            max_loss=total_max_loss,
            stop_loss=stop_loss,
            target=target,
            rationale=f"Bear Call Spread (SELL {sell_strike} / BUY {buy_strike}), Credit: {net_credit:.2f}, Resistance at {sell_strike}",
            metrics={
                "net_credit": net_credit,
                "max_loss_per_share": max_loss,
                "spread_width": spread_width,
                "breakeven": sell_strike + net_credit,
                "risk_reward": rr_ratio,
                "iv_regime": iv_regime,
                "resistance_strike": max_call_oi_strike,
            },
        )
        
        logger.info(f"Bear Call Spread signal: SELL {sell_strike} / BUY {buy_strike} for credit {net_credit:.2f}")
        return signal
    
    def _calculate_confidence(
        self, oi_data: Dict, volatility: Dict, sentiment: str,
        net_credit: float, max_loss: float
    ) -> float:
        confidence = 0.5
        
        # Sentiment alignment (bearish/neutral is good)
        if sentiment == "STRONGLY_BEARISH":
            confidence += 0.15
        elif sentiment == "BEARISH":
            confidence += 0.12
        elif sentiment == "NEUTRAL":
            confidence += 0.08
        
        # High IV is great for selling premium
        iv_regime = volatility.get("volatility_regime", "NORMAL")
        if iv_regime in ["HIGH_IV", "IV_ELEVATED"]:
            confidence += 0.15
        elif iv_regime == "NORMAL":
            confidence += 0.05
        
        # Risk/Reward ratio
        rr_ratio = net_credit / max_loss if max_loss > 0 else 0
        if rr_ratio > 0.5:
            confidence += 0.10
        elif rr_ratio > 0.3:
            confidence += 0.05
        
        return min(confidence, 1.0)
    
    def calculate_stop_loss(self, entry_credit: float, max_loss: float = None, **kwargs) -> float:
        # Exit when loss equals credit received (breakeven on loss side)
        return entry_credit
    
    def calculate_target(self, entry_credit: float, **kwargs) -> float:
        # Indices: 50% of credit; Stocks: 65% (higher margin justifies higher target)
        target_pct = 0.50 if self.is_index else 0.65
        return entry_credit * target_pct


class BullPutSpreadStrategy(BaseStrategy):
    """
    Bull Put Spread (Credit Spread) - SELL higher strike put, BUY lower strike put.
    Main trade is a SELL with a hedge. Bullish/Neutral outlook, profit from premium decay.
    Collects net credit, limited risk, limited profit.
    """
    
    def __init__(self, underlying: str):
        super().__init__(underlying)
        self.name = "Bull Put Spread"
        self.description = "Sell higher strike put, buy lower strike put (credit spread)"
        self.strategy_type = StrategyType.BULL_PUT_SPREAD
        self.min_confidence = 0.65
    
    def analyze(self, options_chain: pd.DataFrame, metrics: Dict[str, Any]) -> Optional[StrategySignal]:
        """Analyze and generate signal for bull put spread (credit)."""
        is_valid, reason = self.validate_conditions(metrics)
        if not is_valid:
            logger.debug(f"Bull Put Spread {self.underlying}: validation failed - {reason}")
            return None
        
        oi_data = metrics.get("oi_data", {})
        volatility = metrics.get("volatility", {})
        spot = metrics.get("spot", 0)
        
        sentiment = oi_data.get("sentiment", "NEUTRAL")
        # Best for bullish or neutral sentiment (we want price to stay above short strike)
        if not self.ml_override and sentiment in ["BEARISH", "STRONGLY_BEARISH"]:
            logger.info(f"Bull Put Spread {self.underlying}: skipped - OI sentiment is {sentiment}")
            return None
        
        # Credit spreads work in any IV regime - higher IV gives better premiums
        iv_regime = volatility.get("volatility_regime", "NORMAL")
        
        puts = options_chain[options_chain["option_type"] == "PE"].sort_values("strike")
        if len(puts) < 2:
            logger.info(f"Bull Put Spread {self.underlying}: skipped - not enough PE options")
            return None
        
        strike_interval = 50 if self.underlying in ["NIFTY", "FINNIFTY"] else 100
        atm_strike = round(spot / strike_interval) * strike_interval
        
        # Derive actual strike interval from the chain (NSE uses 5/10/25/50/100)
        available_strikes = sorted(puts["strike"].unique())
        if len(available_strikes) >= 2:
            chain_diffs = [available_strikes[i+1] - available_strikes[i] for i in range(len(available_strikes)-1)]
            actual_interval = min(chain_diffs)
        else:
            actual_interval = strike_interval
        
        # ATR-based minimum OTM distance for credit spreads
        historical = metrics.get("historical", {})
        atr_14 = historical.get("atr_14", actual_interval * 5) if isinstance(historical, dict) else actual_interval * 5
        min_otm_strikes = max(2, round(atr_14 / actual_interval))
        
        # Use max put OI as support - sell at or near this level
        max_put_oi_strike = oi_data.get("max_put_oi_strike", atm_strike - 2 * strike_interval)
        
        # Validate max_put_oi_strike is reasonable (within 10 intervals of ATM)
        if max_put_oi_strike is None or max_put_oi_strike <= 0 or abs(max_put_oi_strike - atm_strike) > 10 * strike_interval:
            max_put_oi_strike = atm_strike - 2 * strike_interval
        
        # Ensure sell strike is at least min_otm_strikes away from ATM (ATR-based safety)
        max_sell_strike = atm_strike - min_otm_strikes * actual_interval
        sell_strike = min(max_put_oi_strike, max_sell_strike)
        
        # SELL higher strike put (at support), BUY lower strike put (hedge)
        # Hedge width: at least 2 strikes or half ATR for meaningful risk cap
        min_hedge_strikes = max(2, min_otm_strikes // 2)
        
        # Find nearest available strikes in the chain
        sell_strike = min(available_strikes, key=lambda s: abs(s - sell_strike)) if available_strikes else sell_strike
        buy_candidates = [s for s in available_strikes if s < sell_strike]
        if len(buy_candidates) >= min_hedge_strikes:
            buy_strike = buy_candidates[-min_hedge_strikes]
        elif buy_candidates:
            buy_strike = buy_candidates[0]
        else:
            buy_strike = sell_strike - actual_interval
        
        logger.debug(f"Bull Put Spread {self.underlying}: chain_interval={actual_interval}, ATR={atr_14:.1f}, OTM={min_otm_strikes} strikes, hedge={min_hedge_strikes} strikes")
        
        sell_option = puts[puts["strike"] == sell_strike]
        buy_option = puts[puts["strike"] == buy_strike]
        
        if sell_option.empty or buy_option.empty:
            logger.info(f"Bull Put Spread {self.underlying}: skipped - strikes {sell_strike}/{buy_strike} not found in chain")
            return None
        
        sell_option = sell_option.iloc[0]
        buy_option = buy_option.iloc[0]
        
        # Liquidity guard
        for leg_row in [sell_option, buy_option]:
            is_liquid, liq_reason = self.check_leg_liquidity(leg_row)
            if not is_liquid:
                logger.info(f"Bull Put Spread {self.underlying}: {liq_reason}")
                return None
        
        # Calculate net credit (sell premium > buy premium)
        net_credit = sell_option["ltp"] - buy_option["ltp"]
        spread_width = sell_strike - buy_strike
        max_loss = spread_width - net_credit
        
        if net_credit <= 0 or max_loss <= 0:
            logger.info(f"Bull Put Spread {self.underlying}: skipped - unfavorable credit ({net_credit:.2f})")
            return None
        
        # Risk/Reward for credit spreads
        rr_ratio = net_credit / max_loss if max_loss > 0 else 0
        
        confidence = self._calculate_confidence(oi_data, volatility, sentiment, net_credit, max_loss)
        
        if not self.ml_override and confidence < self.min_confidence:
            logger.info(f"Bull Put Spread {self.underlying}: skipped - strategy confidence {confidence:.2f} < {self.min_confidence}")
            return None
        
        lot_size = get_lot_size(self.underlying)
        quantity = lot_size * TRADING_CONFIG.get("default_quantity", 1)
        
        # SELL leg first (main trade), then BUY leg (hedge)
        legs = [
            OptionLeg(
                symbol=sell_option["symbol"],
                strike=sell_option["strike"],
                option_type="PE",
                expiry=sell_option["expiry"],
                direction=TradeDirection.SELL,  # Main trade - SELL
                quantity=quantity,
                entry_price=sell_option["ltp"],
                instrument_token=sell_option.get("instrument_token"),
            ),
            OptionLeg(
                symbol=buy_option["symbol"],
                strike=buy_option["strike"],
                option_type="PE",
                expiry=buy_option["expiry"],
                direction=TradeDirection.BUY,  # Hedge
                quantity=quantity,
                entry_price=buy_option["ltp"],
                instrument_token=buy_option.get("instrument_token"),
            ),
        ]
        
        total_credit = net_credit * quantity
        total_max_loss = max_loss * quantity
        stop_loss = self.calculate_stop_loss(total_credit, max_loss=total_max_loss)
        target = self.calculate_target(total_credit)
        
        signal = StrategySignal(
            strategy_type=self.strategy_type,
            underlying=self.underlying,
            legs=legs,
            entry_time=datetime.now(),
            confidence=confidence,
            expected_profit=total_credit,  # Max profit is credit received
            max_loss=total_max_loss,
            stop_loss=stop_loss,
            target=target,
            rationale=f"Bull Put Spread (SELL {sell_strike} / BUY {buy_strike}), Credit: {net_credit:.2f}, Support at {sell_strike}",
            metrics={
                "net_credit": net_credit,
                "max_loss_per_share": max_loss,
                "spread_width": spread_width,
                "breakeven": sell_strike - net_credit,
                "risk_reward": rr_ratio,
                "iv_regime": iv_regime,
                "support_strike": max_put_oi_strike,
            },
        )
        
        logger.info(f"Bull Put Spread signal: SELL {sell_strike} / BUY {buy_strike} for credit {net_credit:.2f}")
        return signal
    
    def _calculate_confidence(
        self, oi_data: Dict, volatility: Dict, sentiment: str,
        net_credit: float, max_loss: float
    ) -> float:
        confidence = 0.5
        
        # Sentiment alignment (bullish/neutral is good)
        if sentiment == "STRONGLY_BULLISH":
            confidence += 0.15
        elif sentiment == "BULLISH":
            confidence += 0.12
        elif sentiment == "NEUTRAL":
            confidence += 0.08
        
        # High IV is great for selling premium
        iv_regime = volatility.get("volatility_regime", "NORMAL")
        if iv_regime in ["HIGH_IV", "IV_ELEVATED"]:
            confidence += 0.15
        elif iv_regime == "NORMAL":
            confidence += 0.05
        
        # Risk/Reward ratio
        rr_ratio = net_credit / max_loss if max_loss > 0 else 0
        if rr_ratio > 0.5:
            confidence += 0.10
        elif rr_ratio > 0.3:
            confidence += 0.05
        
        return min(confidence, 1.0)
    
    def calculate_stop_loss(self, entry_credit: float, max_loss: float = None, **kwargs) -> float:
        # Exit when loss equals credit received
        return entry_credit
    
    def calculate_target(self, entry_credit: float, **kwargs) -> float:
        # Indices: 50% of credit; Stocks: 65% (higher margin justifies higher target)
        target_pct = 0.50 if self.is_index else 0.65
        return entry_credit * target_pct