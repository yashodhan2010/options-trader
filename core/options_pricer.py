"""
Options Pricer - Greeks and IV calculations using QuantLib and py_vollib
Provides accurate pricing, Greeks, and implied volatility calculations.
"""
import math
from datetime import datetime, date
from typing import Optional, Dict, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

# Try importing QuantLib (preferred for accuracy)
try:
    import QuantLib as ql
    HAS_QUANTLIB = True
except ImportError:
    HAS_QUANTLIB = False

# Try importing py_vollib (faster for IV)
try:
    from py_vollib.black_scholes import black_scholes as bs_price
    from py_vollib.black_scholes.greeks.analytical import delta, gamma, theta, vega, rho
    from py_vollib.black_scholes.implied_volatility import implied_volatility as vollib_iv
    HAS_VOLLIB = True
except ImportError:
    HAS_VOLLIB = False

from core.logger import logger


class OptionType(Enum):
    CALL = "CE"
    PUT = "PE"


@dataclass
class Greeks:
    """Container for option Greeks."""
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "delta": round(self.delta, 4),
            "gamma": round(self.gamma, 6),
            "theta": round(self.theta, 4),
            "vega": round(self.vega, 4),
            "rho": round(self.rho, 4),
        }


@dataclass
class OptionPriceResult:
    """Complete option pricing result."""
    theoretical_price: float
    intrinsic_value: float
    time_value: float
    implied_volatility: float
    greeks: Greeks
    moneyness: str  # ITM, ATM, OTM
    days_to_expiry: int
    
    def to_dict(self) -> Dict:
        return {
            "theoretical_price": round(self.theoretical_price, 2),
            "intrinsic_value": round(self.intrinsic_value, 2),
            "time_value": round(self.time_value, 2),
            "implied_volatility": round(self.implied_volatility * 100, 2),  # As percentage
            "greeks": self.greeks.to_dict(),
            "moneyness": self.moneyness,
            "days_to_expiry": self.days_to_expiry,
        }


class OptionsPricer:
    """
    Options pricing engine using QuantLib and py_vollib.
    
    Features:
    - Black-Scholes pricing
    - Implied Volatility calculation
    - Full Greeks (Delta, Gamma, Theta, Vega, Rho)
    - Supports both European and American options
    """
    
    # Risk-free rate (RBI repo rate approximation)
    DEFAULT_RISK_FREE_RATE = 0.065  # 6.5%
    
    # Dividend yield for indices
    DEFAULT_DIVIDEND_YIELD = 0.015  # 1.5%
    
    def __init__(self, risk_free_rate: float = None, dividend_yield: float = None):
        """
        Initialize the options pricer.
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 6.5%)
            dividend_yield: Annual dividend yield (default: 1.5%)
        """
        self.risk_free_rate = risk_free_rate or self.DEFAULT_RISK_FREE_RATE
        self.dividend_yield = dividend_yield or self.DEFAULT_DIVIDEND_YIELD
        
        logger.info(f"OptionsPricer initialized - QuantLib: {HAS_QUANTLIB}, py_vollib: {HAS_VOLLIB}")
    
    def calculate_iv(
        self,
        option_price: float,
        spot_price: float,
        strike: float,
        expiry_date: date,
        option_type: str = "CE",
        market_price: float = None,
    ) -> float:
        """
        Calculate Implied Volatility from market price.
        
        Args:
            option_price: Market price of the option
            spot_price: Current spot price of underlying
            strike: Strike price
            expiry_date: Expiry date
            option_type: 'CE' for Call, 'PE' for Put
            
        Returns:
            Implied volatility as a decimal (e.g., 0.25 for 25%)
        """
        price = market_price or option_price
        days_to_expiry = self._days_to_expiry(expiry_date)
        
        if days_to_expiry <= 0:
            return 0.0
        
        time_to_expiry = days_to_expiry / 365.0
        is_call = option_type.upper() in ["CE", "CALL", "C"]
        
        # Try py_vollib first (faster)
        if HAS_VOLLIB:
            try:
                flag = 'c' if is_call else 'p'
                iv = vollib_iv(
                    price, spot_price, strike, time_to_expiry,
                    self.risk_free_rate, flag
                )
                return iv
            except Exception:
                pass
        
        # Fallback to Newton-Raphson method
        try:
            iv = self._newton_raphson_iv(
                price, spot_price, strike, time_to_expiry, is_call
            )
            return iv
        except Exception as e:
            logger.warning(f"IV calculation failed: {e}")
            return 0.0
    
    def calculate_greeks(
        self,
        spot_price: float,
        strike: float,
        expiry_date: date,
        volatility: float,
        option_type: str = "CE",
    ) -> Greeks:
        """
        Calculate option Greeks.
        
        Args:
            spot_price: Current spot price
            strike: Strike price
            expiry_date: Expiry date
            volatility: Implied volatility (as decimal)
            option_type: 'CE' for Call, 'PE' for Put
            
        Returns:
            Greeks object with delta, gamma, theta, vega, rho
        """
        days_to_expiry = self._days_to_expiry(expiry_date)
        
        if days_to_expiry <= 0 or volatility <= 0:
            return Greeks()
        
        time_to_expiry = days_to_expiry / 365.0
        is_call = option_type.upper() in ["CE", "CALL", "C"]
        
        # Use QuantLib if available (most accurate)
        if HAS_QUANTLIB:
            return self._quantlib_greeks(
                spot_price, strike, time_to_expiry, volatility, is_call
            )
        
        # Use py_vollib
        if HAS_VOLLIB:
            return self._vollib_greeks(
                spot_price, strike, time_to_expiry, volatility, is_call
            )
        
        # Fallback to manual Black-Scholes
        return self._manual_greeks(
            spot_price, strike, time_to_expiry, volatility, is_call
        )
    
    def price_option(
        self,
        spot_price: float,
        strike: float,
        expiry_date: date,
        volatility: float,
        option_type: str = "CE",
    ) -> float:
        """
        Calculate theoretical option price using Black-Scholes.
        
        Args:
            spot_price: Current spot price
            strike: Strike price
            expiry_date: Expiry date
            volatility: Implied volatility (as decimal)
            option_type: 'CE' for Call, 'PE' for Put
            
        Returns:
            Theoretical option price
        """
        days_to_expiry = self._days_to_expiry(expiry_date)
        
        if days_to_expiry <= 0:
            # At expiry, return intrinsic value
            return self._intrinsic_value(spot_price, strike, option_type)
        
        time_to_expiry = days_to_expiry / 365.0
        is_call = option_type.upper() in ["CE", "CALL", "C"]
        
        if HAS_QUANTLIB:
            return self._quantlib_price(
                spot_price, strike, time_to_expiry, volatility, is_call
            )
        
        if HAS_VOLLIB:
            flag = 'c' if is_call else 'p'
            return bs_price(flag, spot_price, strike, time_to_expiry, 
                           self.risk_free_rate, volatility)
        
        return self._manual_bs_price(
            spot_price, strike, time_to_expiry, volatility, is_call
        )
    
    def full_analysis(
        self,
        spot_price: float,
        strike: float,
        expiry_date: date,
        market_price: float,
        option_type: str = "CE",
    ) -> OptionPriceResult:
        """
        Complete option analysis with IV, Greeks, and pricing.
        
        Args:
            spot_price: Current spot price
            strike: Strike price
            expiry_date: Expiry date
            market_price: Current market price of the option
            option_type: 'CE' for Call, 'PE' for Put
            
        Returns:
            OptionPriceResult with all calculations
        """
        days_to_expiry = self._days_to_expiry(expiry_date)
        is_call = option_type.upper() in ["CE", "CALL", "C"]
        
        # Calculate IV from market price
        iv = self.calculate_iv(market_price, spot_price, strike, expiry_date, option_type)
        
        # Calculate theoretical price
        theoretical_price = self.price_option(spot_price, strike, expiry_date, iv, option_type)
        
        # Calculate Greeks
        greeks = self.calculate_greeks(spot_price, strike, expiry_date, iv, option_type)
        
        # Calculate intrinsic and time value
        intrinsic = self._intrinsic_value(spot_price, strike, option_type)
        time_value = max(0, market_price - intrinsic)
        
        # Determine moneyness
        moneyness = self._get_moneyness(spot_price, strike, is_call)
        
        return OptionPriceResult(
            theoretical_price=theoretical_price,
            intrinsic_value=intrinsic,
            time_value=time_value,
            implied_volatility=iv,
            greeks=greeks,
            moneyness=moneyness,
            days_to_expiry=days_to_expiry,
        )
    
    def calculate_strategy_greeks(
        self,
        legs: list,
        spot_price: float,
    ) -> Dict:
        """
        Calculate net Greeks for a multi-leg strategy.
        
        Args:
            legs: List of leg dictionaries with strike, expiry, option_type, quantity, market_price
            spot_price: Current spot price
            
        Returns:
            Dictionary with net Greeks and individual leg Greeks
        """
        net_greeks = Greeks()
        leg_details = []
        
        for leg in legs:
            strike = leg.get("strike")
            expiry = leg.get("expiry")
            option_type = leg.get("option_type", "CE")
            quantity = leg.get("quantity", 1)
            market_price = leg.get("market_price", 0)
            direction = leg.get("direction", "BUY")
            
            # Parse expiry if string
            if isinstance(expiry, str):
                try:
                    expiry = datetime.strptime(expiry, "%Y-%m-%d").date()
                except:
                    continue
            
            # Get full analysis for this leg
            analysis = self.full_analysis(spot_price, strike, expiry, market_price, option_type)
            
            # Adjust for direction and quantity
            multiplier = quantity if direction == "BUY" else -quantity
            
            net_greeks.delta += analysis.greeks.delta * multiplier
            net_greeks.gamma += analysis.greeks.gamma * multiplier
            net_greeks.theta += analysis.greeks.theta * multiplier
            net_greeks.vega += analysis.greeks.vega * multiplier
            net_greeks.rho += analysis.greeks.rho * multiplier
            
            leg_details.append({
                "strike": strike,
                "option_type": option_type,
                "direction": direction,
                "quantity": quantity,
                "iv": round(analysis.implied_volatility * 100, 2),
                "greeks": analysis.greeks.to_dict(),
            })
        
        return {
            "net_greeks": net_greeks.to_dict(),
            "legs": leg_details,
            "position_delta": round(net_greeks.delta, 4),
            "position_gamma": round(net_greeks.gamma, 6),
            "position_theta": round(net_greeks.theta, 2),
            "position_vega": round(net_greeks.vega, 2),
        }
    
    # ========== QuantLib Implementation ==========
    
    def _quantlib_price(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        is_call: bool,
    ) -> float:
        """Calculate option price using QuantLib."""
        try:
            # Set up QuantLib objects
            today = ql.Date.todaysDate()
            ql.Settings.instance().evaluationDate = today
            
            # Option type
            option_type = ql.Option.Call if is_call else ql.Option.Put
            
            # Exercise
            expiry_date = today + int(time_to_expiry * 365)
            exercise = ql.EuropeanExercise(expiry_date)
            
            # Payoff
            payoff = ql.PlainVanillaPayoff(option_type, strike)
            
            # Option
            option = ql.VanillaOption(payoff, exercise)
            
            # Market data
            spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
            rate_handle = ql.YieldTermStructureHandle(
                ql.FlatForward(today, self.risk_free_rate, ql.Actual365Fixed())
            )
            div_handle = ql.YieldTermStructureHandle(
                ql.FlatForward(today, self.dividend_yield, ql.Actual365Fixed())
            )
            vol_handle = ql.BlackVolTermStructureHandle(
                ql.BlackConstantVol(today, ql.NullCalendar(), volatility, ql.Actual365Fixed())
            )
            
            # Process and engine
            process = ql.BlackScholesMertonProcess(spot_handle, div_handle, rate_handle, vol_handle)
            engine = ql.AnalyticEuropeanEngine(process)
            option.setPricingEngine(engine)
            
            return option.NPV()
            
        except Exception as e:
            logger.error(f"QuantLib pricing error: {e}")
            return self._manual_bs_price(spot, strike, time_to_expiry, volatility, is_call)
    
    def _quantlib_greeks(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        is_call: bool,
    ) -> Greeks:
        """Calculate Greeks using QuantLib."""
        try:
            today = ql.Date.todaysDate()
            ql.Settings.instance().evaluationDate = today
            
            option_type = ql.Option.Call if is_call else ql.Option.Put
            expiry_date = today + int(time_to_expiry * 365)
            
            exercise = ql.EuropeanExercise(expiry_date)
            payoff = ql.PlainVanillaPayoff(option_type, strike)
            option = ql.VanillaOption(payoff, exercise)
            
            spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
            rate_handle = ql.YieldTermStructureHandle(
                ql.FlatForward(today, self.risk_free_rate, ql.Actual365Fixed())
            )
            div_handle = ql.YieldTermStructureHandle(
                ql.FlatForward(today, self.dividend_yield, ql.Actual365Fixed())
            )
            vol_handle = ql.BlackVolTermStructureHandle(
                ql.BlackConstantVol(today, ql.NullCalendar(), volatility, ql.Actual365Fixed())
            )
            
            process = ql.BlackScholesMertonProcess(spot_handle, div_handle, rate_handle, vol_handle)
            engine = ql.AnalyticEuropeanEngine(process)
            option.setPricingEngine(engine)
            
            return Greeks(
                delta=option.delta(),
                gamma=option.gamma(),
                theta=option.theta() / 365,  # Daily theta
                vega=option.vega() / 100,    # Per 1% move
                rho=option.rho() / 100,      # Per 1% move
            )
            
        except Exception as e:
            logger.error(f"QuantLib Greeks error: {e}")
            return self._manual_greeks(spot, strike, time_to_expiry, volatility, is_call)
    
    # ========== py_vollib Implementation ==========
    
    def _vollib_greeks(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        is_call: bool,
    ) -> Greeks:
        """Calculate Greeks using py_vollib."""
        try:
            flag = 'c' if is_call else 'p'
            
            return Greeks(
                delta=delta(flag, spot, strike, time_to_expiry, self.risk_free_rate, volatility),
                gamma=gamma(flag, spot, strike, time_to_expiry, self.risk_free_rate, volatility),
                theta=theta(flag, spot, strike, time_to_expiry, self.risk_free_rate, volatility) / 365,
                vega=vega(flag, spot, strike, time_to_expiry, self.risk_free_rate, volatility) / 100,
                rho=rho(flag, spot, strike, time_to_expiry, self.risk_free_rate, volatility) / 100,
            )
        except Exception as e:
            logger.error(f"py_vollib Greeks error: {e}")
            return self._manual_greeks(spot, strike, time_to_expiry, volatility, is_call)
    
    # ========== Manual Black-Scholes Implementation ==========
    
    def _manual_bs_price(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        is_call: bool,
    ) -> float:
        """Manual Black-Scholes pricing."""
        if time_to_expiry <= 0 or volatility <= 0:
            return max(0, spot - strike) if is_call else max(0, strike - spot)
        
        d1, d2 = self._d1_d2(spot, strike, time_to_expiry, volatility)
        
        if is_call:
            price = (spot * math.exp(-self.dividend_yield * time_to_expiry) * norm.cdf(d1) -
                    strike * math.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(d2))
        else:
            price = (strike * math.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(-d2) -
                    spot * math.exp(-self.dividend_yield * time_to_expiry) * norm.cdf(-d1))
        
        return max(0, price)
    
    def _manual_greeks(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        is_call: bool,
    ) -> Greeks:
        """Manual Greeks calculation using Black-Scholes formulas."""
        if time_to_expiry <= 0 or volatility <= 0:
            return Greeks()
        
        d1, d2 = self._d1_d2(spot, strike, time_to_expiry, volatility)
        sqrt_t = math.sqrt(time_to_expiry)
        
        # Common terms
        exp_div = math.exp(-self.dividend_yield * time_to_expiry)
        exp_rate = math.exp(-self.risk_free_rate * time_to_expiry)
        pdf_d1 = norm.pdf(d1)
        
        # Delta
        if is_call:
            delta_val = exp_div * norm.cdf(d1)
        else:
            delta_val = exp_div * (norm.cdf(d1) - 1)
        
        # Gamma (same for calls and puts)
        gamma_val = (exp_div * pdf_d1) / (spot * volatility * sqrt_t)
        
        # Theta
        term1 = -(spot * volatility * exp_div * pdf_d1) / (2 * sqrt_t)
        if is_call:
            term2 = -self.risk_free_rate * strike * exp_rate * norm.cdf(d2)
            term3 = self.dividend_yield * spot * exp_div * norm.cdf(d1)
        else:
            term2 = self.risk_free_rate * strike * exp_rate * norm.cdf(-d2)
            term3 = -self.dividend_yield * spot * exp_div * norm.cdf(-d1)
        theta_val = (term1 + term2 + term3) / 365  # Daily theta
        
        # Vega (same for calls and puts)
        vega_val = spot * exp_div * sqrt_t * pdf_d1 / 100  # Per 1% IV move
        
        # Rho
        if is_call:
            rho_val = strike * time_to_expiry * exp_rate * norm.cdf(d2) / 100
        else:
            rho_val = -strike * time_to_expiry * exp_rate * norm.cdf(-d2) / 100
        
        return Greeks(
            delta=delta_val,
            gamma=gamma_val,
            theta=theta_val,
            vega=vega_val,
            rho=rho_val,
        )
    
    def _d1_d2(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
    ) -> Tuple[float, float]:
        """Calculate d1 and d2 for Black-Scholes."""
        sqrt_t = math.sqrt(time_to_expiry)
        
        d1 = (math.log(spot / strike) + 
              (self.risk_free_rate - self.dividend_yield + 0.5 * volatility ** 2) * time_to_expiry) / \
             (volatility * sqrt_t)
        
        d2 = d1 - volatility * sqrt_t
        
        return d1, d2
    
    def _newton_raphson_iv(
        self,
        market_price: float,
        spot: float,
        strike: float,
        time_to_expiry: float,
        is_call: bool,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> float:
        """Calculate IV using Newton-Raphson method."""
        # Initial guess using Brenner-Subrahmanyam approximation
        iv = math.sqrt(2 * math.pi / time_to_expiry) * market_price / spot
        iv = max(0.01, min(iv, 5.0))  # Clamp between 1% and 500%
        
        for _ in range(max_iterations):
            price = self._manual_bs_price(spot, strike, time_to_expiry, iv, is_call)
            
            # Vega
            d1, _ = self._d1_d2(spot, strike, time_to_expiry, iv)
            vega = spot * math.exp(-self.dividend_yield * time_to_expiry) * \
                   norm.pdf(d1) * math.sqrt(time_to_expiry)
            
            if vega < 1e-10:
                break
            
            diff = market_price - price
            if abs(diff) < tolerance:
                break
            
            iv = iv + diff / vega
            iv = max(0.001, min(iv, 5.0))
        
        return iv
    
    # ========== Helper Methods ==========
    
    def _days_to_expiry(self, expiry_date: date) -> int:
        """Calculate days to expiry."""
        if isinstance(expiry_date, str):
            try:
                expiry_date = datetime.strptime(expiry_date, "%Y-%m-%d").date()
            except:
                return 0
        
        if isinstance(expiry_date, datetime):
            expiry_date = expiry_date.date()
        
        today = date.today()
        delta = (expiry_date - today).days
        return max(0, delta)
    
    def _intrinsic_value(self, spot: float, strike: float, option_type: str) -> float:
        """Calculate intrinsic value."""
        is_call = option_type.upper() in ["CE", "CALL", "C"]
        if is_call:
            return max(0, spot - strike)
        else:
            return max(0, strike - spot)
    
    def _get_moneyness(self, spot: float, strike: float, is_call: bool) -> str:
        """Determine option moneyness."""
        ratio = spot / strike
        
        if 0.98 <= ratio <= 1.02:
            return "ATM"
        elif (is_call and spot > strike) or (not is_call and spot < strike):
            return "ITM"
        else:
            return "OTM"


# Singleton instance
options_pricer = OptionsPricer()
