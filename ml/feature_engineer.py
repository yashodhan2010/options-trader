"""
Feature Engineering for ML-Powered Options Trading

Extracts and normalizes 55+ features from market data for ML model training and inference.
Features are organized into categories: price, technicals, options/greeks, OI sentiment,
volatility, and time/calendar features.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

from core.logger import logger


@dataclass
class FeatureSet:
    """Container for extracted features."""
    features: Dict[str, float] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    underlying: str = ""
    version: str = "1.0"
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array in consistent order."""
        return np.array([self.features.get(name, 0.0) for name in self.feature_names])
    
    def to_dict(self) -> Dict[str, float]:
        """Get features as dictionary."""
        return self.features.copy()
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to single-row DataFrame."""
        return pd.DataFrame([self.features], columns=self.feature_names)


class FeatureEngineer:
    """
    Extract and normalize features for ML models.
    
    Feature Categories (55+ features):
    1. Price Features (15): OHLCV patterns, returns, gaps, ranges
    2. Technical Indicators (15): SMA/EMA crossovers, RSI, MACD, Bollinger, Stochastic, ADX
    3. Options/Greeks (12): IV, IV percentile, Greeks, skew
    4. OI Sentiment (6): PCR, OI changes, max pain distance
    5. Volatility (6): HV, ATR, IV/HV ratio, regime
    6. Time/Calendar (7): DTE, day of week, session, expiry indicators
    """
    
    # Standard feature names in order
    FEATURE_NAMES = [
        # Price Features (15)
        "return_1d", "return_5d", "return_10d", "return_20d",
        "gap_percent", "intraday_range_percent", "close_to_high_percent",
        "close_to_low_percent", "body_percent", "upper_shadow_percent",
        "lower_shadow_percent", "price_vs_sma20", "price_vs_sma50",
        "high_52w_distance", "low_52w_distance",
        
        # Technical Indicators (15)
        "sma_5_10_crossover", "sma_10_20_crossover", "ema_9_21_crossover",
        "rsi_14", "rsi_signal",  # -1 oversold, 0 neutral, 1 overbought
        "macd_histogram", "macd_signal_crossover",
        "bollinger_percent_b", "bollinger_width",
        "stochastic_k", "stochastic_d", "stochastic_signal",
        "adx_14", "williams_r", "atr_percent",
        
        # Options/Greeks (12)
        "iv_current", "iv_percentile", "iv_hv_ratio",
        "position_delta", "position_gamma", "position_theta",
        "position_vega", "atm_iv", "otm_call_iv",
        "otm_put_iv", "iv_skew", "put_call_iv_ratio",
        
        # OI Sentiment (6)
        "pcr", "pcr_change_1d", "max_pain_distance_percent",
        "oi_buildup_signal", "call_oi_change_percent", "put_oi_change_percent",
        
        # Volatility (6)
        "hv_10", "hv_20", "hv_ratio_10_20", "atr_14_value",
        "volatility_regime",  # -1 low, 0 normal, 1 high
        "volatility_trend",   # -1 decreasing, 0 stable, 1 increasing
        
        # Time/Calendar (7)
        "dte", "day_of_week", "hour_of_day", "days_to_monthly_expiry",
        "is_weekly_expiry", "is_rollover_week", "session_indicator",  # 0 open, 1 mid, 2 close
    ]
    
    def __init__(self):
        self.feature_names = self.FEATURE_NAMES.copy()
        self.scaler_fitted = False
        self.feature_stats: Dict[str, Dict[str, float]] = {}
        
        logger.info(f"FeatureEngineer initialized with {len(self.feature_names)} features")
    
    def extract_features(
        self,
        underlying: str,
        spot_price: float,
        historical_data: pd.DataFrame,
        options_chain: Optional[pd.DataFrame] = None,
        oi_analysis: Optional[Dict] = None,
        volatility_data: Optional[Dict] = None,
        greeks: Optional[Dict] = None,
        current_time: Optional[datetime] = None
    ) -> FeatureSet:
        """
        Extract all features from market data.
        
        Args:
            underlying: Underlying asset symbol
            spot_price: Current spot price
            historical_data: DataFrame with OHLCV data (index=date, columns=open,high,low,close,volume)
            options_chain: Optional DataFrame with options chain data
            oi_analysis: Optional dict with OI analysis (pcr, max_pain, etc.)
            volatility_data: Optional dict with volatility metrics
            greeks: Optional dict with position Greeks
            current_time: Current timestamp (defaults to now)
            
        Returns:
            FeatureSet with all extracted features
        """
        current_time = current_time or datetime.now()
        features = {}
        
        # Extract each category
        price_features = self._extract_price_features(spot_price, historical_data)
        features.update(price_features)
        
        technical_features = self._extract_technical_features(historical_data)
        features.update(technical_features)
        
        options_features = self._extract_options_features(
            options_chain, volatility_data, greeks, spot_price
        )
        features.update(options_features)
        
        oi_features = self._extract_oi_features(oi_analysis, spot_price)
        features.update(oi_features)
        
        volatility_features = self._extract_volatility_features(
            historical_data, volatility_data
        )
        features.update(volatility_features)
        
        time_features = self._extract_time_features(
            current_time, underlying, options_chain
        )
        features.update(time_features)
        
        # Ensure all features are present
        for name in self.feature_names:
            if name not in features:
                features[name] = 0.0
        
        return FeatureSet(
            features=features,
            feature_names=self.feature_names,
            timestamp=current_time,
            underlying=underlying
        )
    
    def extract_features_batch(
        self,
        df: pd.DataFrame,
        lookback: int = 50
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract features from historical OHLCV data for batch training.
        
        This method processes each row of the dataframe and extracts features
        using only data available up to that point (no lookahead bias).
        
        Args:
            df: DataFrame with OHLCV columns (open, high, low, close, volume)
            lookback: Minimum lookback period required for feature extraction
            
        Returns:
            Tuple of (X matrix, feature_names list)
        """
        if df is None or len(df) < lookback + 1:
            logger.warning(f"Insufficient data for feature extraction: {len(df) if df is not None else 0} rows")
            return np.array([]), []
        
        # Ensure column names are lowercase
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # Feature subset for OHLCV-only training
        training_features = [
            # Price Features
            "return_1d", "return_5d", "return_10d", "return_20d",
            "gap_percent", "intraday_range_percent", "close_to_high_percent",
            "close_to_low_percent", "body_percent", "upper_shadow_percent",
            "lower_shadow_percent", "price_vs_sma20", "price_vs_sma50",
            
            # Technical Indicators
            "sma_5_10_crossover", "sma_10_20_crossover", "ema_9_21_crossover",
            "rsi_14", "rsi_signal", "macd_histogram", "macd_signal_crossover",
            "bollinger_percent_b", "bollinger_width",
            "stochastic_k", "stochastic_d", "stochastic_signal",
            "adx_14", "williams_r", "atr_percent",
            
            # Volatility
            "hv_10", "hv_20", "hv_ratio_10_20", "atr_14_value",
            "volatility_regime", "volatility_trend",
        ]
        
        all_features = []
        
        for i in range(lookback, len(df)):
            # Get data up to current point (no lookahead)
            historical = df.iloc[:i+1]
            current_close = df.iloc[i]["close"]
            
            row_features = {}
            
            # Extract price features
            price_feats = self._extract_price_features(current_close, historical)
            row_features.update(price_feats)
            
            # Extract technical features
            tech_feats = self._extract_technical_features(historical)
            row_features.update(tech_feats)
            
            # Extract volatility features (OHLCV only)
            vol_feats = self._extract_volatility_features(historical, None)
            row_features.update(vol_feats)
            
            # Build feature vector in order
            feature_vector = [row_features.get(name, 0.0) for name in training_features]
            all_features.append(feature_vector)
        
        X = np.array(all_features)
        
        # Handle NaN/Inf values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        logger.info(f"Extracted {len(X)} samples with {len(training_features)} features")
        return X, training_features
    
    def _extract_price_features(
        self,
        spot_price: float,
        df: pd.DataFrame
    ) -> Dict[str, float]:
        """Extract price-based features."""
        features = {}
        
        if df is None or len(df) < 2:
            return {name: 0.0 for name in self.feature_names if name.startswith(("return", "gap", "intraday", "close_to", "body", "shadow", "price_vs", "high_52", "low_52"))}
        
        # Ensure we have required columns
        df = df.copy()
        if "close" not in df.columns:
            return features
        
        closes = df["close"].values
        opens = df["open"].values if "open" in df.columns else closes
        highs = df["high"].values if "high" in df.columns else closes
        lows = df["low"].values if "low" in df.columns else closes
        
        # Returns (percentage)
        if len(closes) >= 2:
            features["return_1d"] = (closes[-1] / closes[-2] - 1) * 100
        if len(closes) >= 6:
            features["return_5d"] = (closes[-1] / closes[-6] - 1) * 100
        if len(closes) >= 11:
            features["return_10d"] = (closes[-1] / closes[-11] - 1) * 100
        if len(closes) >= 21:
            features["return_20d"] = (closes[-1] / closes[-21] - 1) * 100
        
        # Gap (today's open vs yesterday's close)
        if len(closes) >= 2:
            features["gap_percent"] = (opens[-1] / closes[-2] - 1) * 100
        
        # Intraday range
        if highs[-1] > 0:
            features["intraday_range_percent"] = (highs[-1] - lows[-1]) / closes[-1] * 100
        
        # Close position within day's range
        day_range = highs[-1] - lows[-1]
        if day_range > 0:
            features["close_to_high_percent"] = (highs[-1] - closes[-1]) / day_range * 100
            features["close_to_low_percent"] = (closes[-1] - lows[-1]) / day_range * 100
        
        # Candlestick body
        features["body_percent"] = abs(closes[-1] - opens[-1]) / closes[-1] * 100
        
        # Shadows
        body_high = max(opens[-1], closes[-1])
        body_low = min(opens[-1], closes[-1])
        if day_range > 0:
            features["upper_shadow_percent"] = (highs[-1] - body_high) / day_range * 100
            features["lower_shadow_percent"] = (body_low - lows[-1]) / day_range * 100
        
        # Price vs SMA
        if len(closes) >= 20:
            sma_20 = np.mean(closes[-20:])
            features["price_vs_sma20"] = (spot_price / sma_20 - 1) * 100
        if len(closes) >= 50:
            sma_50 = np.mean(closes[-50:])
            features["price_vs_sma50"] = (spot_price / sma_50 - 1) * 100
        
        # 52-week high/low distance (approximate with available data)
        if len(highs) >= 20:
            high_period = np.max(highs)
            low_period = np.min(lows)
            features["high_52w_distance"] = (spot_price / high_period - 1) * 100
            features["low_52w_distance"] = (spot_price / low_period - 1) * 100
        
        return features
    
    def _extract_technical_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract technical indicator features."""
        features = {}
        
        if df is None or len(df) < 20:
            return features
        
        closes = df["close"].values
        highs = df["high"].values if "high" in df.columns else closes
        lows = df["low"].values if "low" in df.columns else closes
        
        # SMA crossovers
        if len(closes) >= 10:
            sma_5 = np.mean(closes[-5:])
            sma_10 = np.mean(closes[-10:])
            features["sma_5_10_crossover"] = 1 if sma_5 > sma_10 else -1
        
        if len(closes) >= 20:
            sma_10 = np.mean(closes[-10:])
            sma_20 = np.mean(closes[-20:])
            features["sma_10_20_crossover"] = 1 if sma_10 > sma_20 else -1
        
        # EMA crossover (9/21)
        if len(closes) >= 21:
            ema_9 = self._calculate_ema(closes, 9)
            ema_21 = self._calculate_ema(closes, 21)
            features["ema_9_21_crossover"] = 1 if ema_9 > ema_21 else -1
        
        # RSI
        if len(closes) >= 15:
            rsi = self._calculate_rsi(closes, 14)
            features["rsi_14"] = rsi
            if rsi > 70:
                features["rsi_signal"] = 1  # Overbought
            elif rsi < 30:
                features["rsi_signal"] = -1  # Oversold
            else:
                features["rsi_signal"] = 0  # Neutral
        
        # MACD (12, 26, 9)
        if len(closes) >= 35:
            macd_line, signal_line, histogram = self._calculate_macd(closes)
            features["macd_histogram"] = histogram
            features["macd_signal_crossover"] = 1 if macd_line > signal_line else -1
        
        # Bollinger Bands
        if len(closes) >= 20:
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(closes)
            bb_range = bb_upper - bb_lower
            if bb_range > 0:
                features["bollinger_percent_b"] = (closes[-1] - bb_lower) / bb_range
                features["bollinger_width"] = bb_range / bb_middle * 100
        
        # Stochastic Oscillator
        if len(closes) >= 14:
            stoch_k, stoch_d = self._calculate_stochastic(closes, highs, lows)
            features["stochastic_k"] = stoch_k
            features["stochastic_d"] = stoch_d
            if stoch_k > 80:
                features["stochastic_signal"] = 1  # Overbought
            elif stoch_k < 20:
                features["stochastic_signal"] = -1  # Oversold
            else:
                features["stochastic_signal"] = 0
        
        # ADX (Average Directional Index)
        if len(closes) >= 28:
            adx = self._calculate_adx(highs, lows, closes, 14)
            features["adx_14"] = adx
        
        # Williams %R
        if len(closes) >= 14:
            williams_r = self._calculate_williams_r(closes, highs, lows, 14)
            features["williams_r"] = williams_r
        
        # ATR Percent
        if len(closes) >= 15:
            atr = self._calculate_atr(highs, lows, closes, 14)
            features["atr_percent"] = atr / closes[-1] * 100
        
        return features
    
    def _extract_options_features(
        self,
        options_chain: Optional[pd.DataFrame],
        volatility_data: Optional[Dict],
        greeks: Optional[Dict],
        spot_price: float
    ) -> Dict[str, float]:
        """Extract options and Greeks features."""
        features = {}
        
        # Volatility data
        if volatility_data:
            features["iv_current"] = volatility_data.get("atm_iv", 0)
            features["atm_iv"] = volatility_data.get("atm_iv", 0)
            features["iv_hv_ratio"] = volatility_data.get("iv_hv_ratio", 1.0)
            
            # IV percentile (if available, else estimate)
            if "iv_percentile" in volatility_data:
                features["iv_percentile"] = volatility_data["iv_percentile"]
            else:
                # Rough estimate based on IV/HV ratio
                ratio = volatility_data.get("iv_hv_ratio", 1.0)
                features["iv_percentile"] = min(100, max(0, (ratio - 0.5) * 100))
        
        # Position Greeks
        if greeks:
            features["position_delta"] = greeks.get("delta", 0)
            features["position_gamma"] = greeks.get("gamma", 0)
            features["position_theta"] = greeks.get("theta", 0)
            features["position_vega"] = greeks.get("vega", 0)
        
        # Options chain analysis
        if options_chain is not None and len(options_chain) > 0:
            # IV skew (OTM put IV vs OTM call IV)
            try:
                atm_strike = round(spot_price / 50) * 50  # Round to nearest 50
                
                calls = options_chain[options_chain["option_type"] == "CE"]
                puts = options_chain[options_chain["option_type"] == "PE"]
                
                # OTM call IV (1-2 strikes above ATM)
                otm_calls = calls[calls["strike"] > atm_strike]
                if len(otm_calls) > 0:
                    features["otm_call_iv"] = otm_calls.iloc[0].get("iv", 0)
                
                # OTM put IV (1-2 strikes below ATM)
                otm_puts = puts[puts["strike"] < atm_strike]
                if len(otm_puts) > 0:
                    features["otm_put_iv"] = otm_puts.iloc[-1].get("iv", 0)
                
                # IV skew
                if features.get("otm_put_iv", 0) > 0 and features.get("otm_call_iv", 0) > 0:
                    features["iv_skew"] = features["otm_put_iv"] - features["otm_call_iv"]
                    features["put_call_iv_ratio"] = features["otm_put_iv"] / features["otm_call_iv"]
                    
            except Exception as e:
                logger.debug(f"Error calculating IV skew: {e}")
        
        return features
    
    def _extract_oi_features(
        self,
        oi_analysis: Optional[Dict],
        spot_price: float
    ) -> Dict[str, float]:
        """Extract OI sentiment features."""
        features = {}
        
        if not oi_analysis:
            return features
        
        # Put-Call Ratio
        features["pcr"] = oi_analysis.get("pcr", 1.0)
        
        # PCR change (if historical available)
        features["pcr_change_1d"] = oi_analysis.get("pcr_change_1d", 0)
        
        # Max pain distance
        max_pain = oi_analysis.get("max_pain", spot_price)
        features["max_pain_distance_percent"] = (spot_price / max_pain - 1) * 100
        
        # OI buildup signal
        sentiment = oi_analysis.get("sentiment", "NEUTRAL")
        if sentiment in ["STRONGLY_BULLISH", "BULLISH"]:
            features["oi_buildup_signal"] = 1
        elif sentiment in ["STRONGLY_BEARISH", "BEARISH"]:
            features["oi_buildup_signal"] = -1
        else:
            features["oi_buildup_signal"] = 0
        
        # Call/Put OI changes
        total_call_oi = oi_analysis.get("total_call_oi", 1)
        total_put_oi = oi_analysis.get("total_put_oi", 1)
        
        # Placeholder for change percentages (would need historical OI)
        features["call_oi_change_percent"] = oi_analysis.get("call_oi_change_percent", 0)
        features["put_oi_change_percent"] = oi_analysis.get("put_oi_change_percent", 0)
        
        return features
    
    def _extract_volatility_features(
        self,
        df: pd.DataFrame,
        volatility_data: Optional[Dict]
    ) -> Dict[str, float]:
        """Extract volatility features."""
        features = {}
        
        if df is not None and len(df) >= 20:
            closes = df["close"].values
            highs = df["high"].values if "high" in df.columns else closes
            lows = df["low"].values if "low" in df.columns else closes
            
            # Historical volatility (annualized)
            if len(closes) >= 11:
                returns = np.diff(np.log(closes[-11:]))
                features["hv_10"] = np.std(returns) * np.sqrt(252) * 100
            
            if len(closes) >= 21:
                returns = np.diff(np.log(closes[-21:]))
                features["hv_20"] = np.std(returns) * np.sqrt(252) * 100
            
            # HV ratio
            if features.get("hv_10", 0) > 0 and features.get("hv_20", 0) > 0:
                features["hv_ratio_10_20"] = features["hv_10"] / features["hv_20"]
            
            # ATR value
            if len(closes) >= 15:
                features["atr_14_value"] = self._calculate_atr(highs, lows, closes, 14)
        
        if volatility_data:
            # Volatility regime
            regime = volatility_data.get("volatility_regime", "NORMAL")
            if regime in ["HIGH_IV", "IV_ELEVATED"]:
                features["volatility_regime"] = 1
            elif regime in ["LOW_IV", "IV_DEPRESSED"]:
                features["volatility_regime"] = -1
            else:
                features["volatility_regime"] = 0
            
            # Volatility trend (based on IV/HV changes)
            iv_hv = volatility_data.get("iv_hv_ratio", 1.0)
            if iv_hv > 1.2:
                features["volatility_trend"] = 1  # Increasing/elevated
            elif iv_hv < 0.8:
                features["volatility_trend"] = -1  # Decreasing
            else:
                features["volatility_trend"] = 0
        
        return features
    
    def _extract_time_features(
        self,
        current_time: datetime,
        underlying: str,
        options_chain: Optional[pd.DataFrame]
    ) -> Dict[str, float]:
        """Extract time and calendar features."""
        features = {}
        
        # Days to expiry (estimate if not available from options chain)
        if options_chain is not None and len(options_chain) > 0 and "expiry" in options_chain.columns:
            try:
                nearest_expiry = options_chain["expiry"].min()
                if isinstance(nearest_expiry, str):
                    nearest_expiry = datetime.strptime(nearest_expiry, "%Y-%m-%d")
                elif hasattr(nearest_expiry, 'to_pydatetime'):
                    nearest_expiry = nearest_expiry.to_pydatetime()
                dte = (nearest_expiry - current_time).days
                features["dte"] = max(0, dte)
            except Exception:
                features["dte"] = 7  # Default
        else:
            # Estimate based on day of week
            days_to_thursday = (3 - current_time.weekday()) % 7
            features["dte"] = days_to_thursday if days_to_thursday > 0 else 7
        
        # Day of week (0=Monday, 4=Friday)
        features["day_of_week"] = current_time.weekday()
        
        # Hour of day (normalized 9-16 trading hours)
        features["hour_of_day"] = current_time.hour + current_time.minute / 60
        
        # Days to monthly expiry (last Thursday of month)
        features["days_to_monthly_expiry"] = self._days_to_monthly_expiry(current_time)
        
        # Is weekly expiry (not monthly)
        features["is_weekly_expiry"] = 1 if features["dte"] <= 7 and features.get("days_to_monthly_expiry", 0) > 7 else 0
        
        # Is rollover week (week before monthly expiry)
        features["is_rollover_week"] = 1 if features.get("days_to_monthly_expiry", 0) <= 7 else 0
        
        # Session indicator (0=opening, 1=mid-day, 2=closing)
        hour = current_time.hour
        if hour < 10:
            features["session_indicator"] = 0  # Opening
        elif hour >= 14:
            features["session_indicator"] = 2  # Closing
        else:
            features["session_indicator"] = 1  # Mid-day
        
        return features
    
    # ==================== Helper Methods ====================
    
    def _calculate_ema(self, data: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(data) < period:
            return np.mean(data)
        
        multiplier = 2 / (period + 1)
        ema = np.mean(data[:period])
        
        for price in data[period:]:
            ema = (price - ema) * multiplier + ema
        
        return ema
    
    def _calculate_rsi(self, closes: np.ndarray, period: int = 14) -> float:
        """Calculate RSI."""
        if len(closes) < period + 1:
            return 50.0
        
        deltas = np.diff(closes[-(period+1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(
        self,
        closes: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[float, float, float]:
        """Calculate MACD line, signal line, and histogram."""
        ema_fast = self._calculate_ema(closes, fast)
        ema_slow = self._calculate_ema(closes, slow)
        
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line (need historical MACD values)
        # Simplified: use current MACD as approximation
        signal_line = macd_line * 0.9  # Approximation
        
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _calculate_bollinger_bands(
        self,
        closes: np.ndarray,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands."""
        if len(closes) < period:
            return closes[-1] * 1.02, closes[-1], closes[-1] * 0.98
        
        sma = np.mean(closes[-period:])
        std = np.std(closes[-period:])
        
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        
        return upper, sma, lower
    
    def _calculate_stochastic(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        period: int = 14
    ) -> Tuple[float, float]:
        """Calculate Stochastic %K and %D."""
        if len(closes) < period:
            return 50.0, 50.0
        
        highest_high = np.max(highs[-period:])
        lowest_low = np.min(lows[-period:])
        
        if highest_high == lowest_low:
            return 50.0, 50.0
        
        stoch_k = (closes[-1] - lowest_low) / (highest_high - lowest_low) * 100
        
        # %D is 3-period SMA of %K (simplified)
        stoch_d = stoch_k  # Would need historical %K values
        
        return stoch_k, stoch_d
    
    def _calculate_adx(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14
    ) -> float:
        """Calculate Average Directional Index."""
        if len(closes) < period * 2:
            return 25.0  # Default neutral value
        
        # Calculate +DM and -DM
        plus_dm = np.zeros(len(closes) - 1)
        minus_dm = np.zeros(len(closes) - 1)
        
        for i in range(1, len(closes)):
            high_diff = highs[i] - highs[i-1]
            low_diff = lows[i-1] - lows[i]
            
            if high_diff > low_diff and high_diff > 0:
                plus_dm[i-1] = high_diff
            if low_diff > high_diff and low_diff > 0:
                minus_dm[i-1] = low_diff
        
        # Calculate TR
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1])
            )
        )
        
        # Smoothed values
        atr = np.mean(tr[-period:])
        if atr == 0:
            return 25.0
        
        plus_di = np.mean(plus_dm[-period:]) / atr * 100
        minus_di = np.mean(minus_dm[-period:]) / atr * 100
        
        # Calculate DX
        di_sum = plus_di + minus_di
        if di_sum == 0:
            return 25.0
        
        dx = abs(plus_di - minus_di) / di_sum * 100
        
        return dx  # Simplified ADX (would need smoothing for true ADX)
    
    def _calculate_williams_r(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        period: int = 14
    ) -> float:
        """Calculate Williams %R."""
        if len(closes) < period:
            return -50.0
        
        highest_high = np.max(highs[-period:])
        lowest_low = np.min(lows[-period:])
        
        if highest_high == lowest_low:
            return -50.0
        
        return (highest_high - closes[-1]) / (highest_high - lowest_low) * -100
    
    def _calculate_atr(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14
    ) -> float:
        """Calculate Average True Range."""
        if len(closes) < period + 1:
            return np.mean(highs[-period:] - lows[-period:])
        
        tr = np.zeros(period)
        
        for i in range(period):
            idx = -(period - i)
            high_low = highs[idx] - lows[idx]
            high_close = abs(highs[idx] - closes[idx-1])
            low_close = abs(lows[idx] - closes[idx-1])
            tr[i] = max(high_low, high_close, low_close)
        
        return np.mean(tr)
    
    def _days_to_monthly_expiry(self, current_date: datetime) -> int:
        """Calculate days to last Thursday of current month."""
        # Find last day of month
        if current_date.month == 12:
            next_month = current_date.replace(year=current_date.year + 1, month=1, day=1)
        else:
            next_month = current_date.replace(month=current_date.month + 1, day=1)
        
        last_day = next_month - timedelta(days=1)
        
        # Find last Thursday
        days_since_thursday = (last_day.weekday() - 3) % 7
        last_thursday = last_day - timedelta(days=days_since_thursday)
        
        days_remaining = (last_thursday - current_date).days
        
        # If past this month's expiry, calculate for next month
        if days_remaining < 0:
            if last_day.month == 12:
                next_month_end = datetime(last_day.year + 1, 2, 1) - timedelta(days=1)
            else:
                next_month_end = datetime(last_day.year, last_day.month + 2, 1) - timedelta(days=1)
            
            days_since_thursday = (next_month_end.weekday() - 3) % 7
            next_last_thursday = next_month_end - timedelta(days=days_since_thursday)
            days_remaining = (next_last_thursday - current_date).days
        
        return max(0, days_remaining)
    
    def normalize_features(
        self,
        features: FeatureSet,
        fit: bool = False
    ) -> FeatureSet:
        """
        Normalize features using z-score normalization.
        
        Args:
            features: FeatureSet to normalize
            fit: Whether to fit the scaler on this data
            
        Returns:
            Normalized FeatureSet
        """
        if fit or not self.scaler_fitted:
            # Initialize stats with defaults
            self.feature_stats = {}
            for name in self.feature_names:
                value = features.features.get(name, 0)
                self.feature_stats[name] = {
                    "mean": value,
                    "std": 1.0,  # Will be updated with more data
                    "min": value,
                    "max": value
                }
            self.scaler_fitted = True
        
        normalized = {}
        for name, value in features.features.items():
            stats = self.feature_stats.get(name, {"mean": 0, "std": 1})
            std = stats["std"] if stats["std"] > 0 else 1.0
            normalized[name] = (value - stats["mean"]) / std
        
        return FeatureSet(
            features=normalized,
            feature_names=features.feature_names,
            timestamp=features.timestamp,
            underlying=features.underlying,
            version=features.version
        )
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names in order."""
        return self.feature_names.copy()
    
    def get_feature_count(self) -> int:
        """Get total number of features."""
        return len(self.feature_names)


# Singleton instance
_feature_engineer: Optional[FeatureEngineer] = None


def get_feature_engineer() -> FeatureEngineer:
    """Get or create the singleton feature engineer instance."""
    global _feature_engineer
    if _feature_engineer is None:
        _feature_engineer = FeatureEngineer()
    return _feature_engineer
