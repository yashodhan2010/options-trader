"""
Unified Feature Set for Historical and Live ML Models

This module defines a single feature set that works for both:
1. Historical training (from NSE bhavcopy data)
2. Live prediction (from Kite API + WebSocket)

The unified set contains 50 features that can be calculated from both sources.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from core.logger import logger


@dataclass
class UnifiedFeatureSet:
    """Container for unified features."""
    features: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    symbol: str = ""
    source: str = ""  # "historical" or "live"
    
    def to_array(self, feature_names: List[str]) -> np.ndarray:
        """Convert to numpy array in specified order."""
        return np.array([self.features.get(name, 0.0) for name in feature_names])
    
    def to_dict(self) -> Dict[str, float]:
        """Get features as dictionary."""
        return self.features.copy()


class UnifiedFeatureDefinition:
    """
    Unified Feature Set Definition
    
    50 features organized into 6 categories:
    1. Price/Returns (10): Core price action features
    2. Technical Indicators (12): Standard technicals
    3. Options/Greeks (8): IV and Greek proxies (compatible with both sources)
    4. OI Sentiment (8): Put-Call ratio and OI analysis
    5. Volatility (6): Historical and implied volatility
    6. Momentum/Trend (6): Trend strength indicators
    """
    
    FEATURE_NAMES = [
        # === Price/Returns (10) ===
        "return_1d",              # 1-day return
        "return_5d",              # 5-day return  
        "return_10d",             # 10-day return
        "return_20d",             # 20-day return
        "intraday_range_pct",     # (high-low)/close
        "close_to_high_pct",      # (close-low)/(high-low)
        "gap_pct",                # Gap from previous close
        "price_vs_sma20",         # Price relative to 20-day SMA
        "price_vs_sma50",         # Price relative to 50-day SMA (or 20 if not available)
        "log_return",             # Log return
        
        # === Technical Indicators (12) ===
        "rsi_14",                 # 14-period RSI
        "rsi_signal",             # RSI signal (-1=oversold, 0=neutral, 1=overbought)
        "macd",                   # MACD line
        "macd_signal",            # MACD signal line
        "macd_histogram",         # MACD histogram
        "bb_position",            # Position within Bollinger Bands (0-1)
        "bb_width",               # Bollinger Band width
        "sma_crossover_5_20",     # SMA 5/20 crossover signal
        "ema_crossover_9_21",     # EMA 9/21 crossover signal
        "stochastic_k",           # Stochastic %K
        "stochastic_d",           # Stochastic %D
        "atr_pct",                # ATR as percentage of price
        
        # === Options/Greeks (8) ===
        "iv_current",             # Current IV (or proxy)
        "iv_percentile",          # IV percentile (0-100)
        "iv_rank",                # IV rank (0-1)
        "delta_proxy",            # Delta estimate (momentum-based)
        "gamma_proxy",            # Gamma estimate (acceleration)
        "theta_proxy",            # Theta estimate (time decay)
        "vega_proxy",             # Vega estimate (vol sensitivity)
        "iv_hv_ratio",            # IV to HV ratio
        
        # === OI Sentiment (8) ===
        "pcr",                    # Put-Call Ratio (OI-based)
        "pcr_ma5",                # 5-day PCR moving average
        "pcr_change",             # PCR change from previous day
        "max_pain_distance_pct",  # Distance to max pain
        "call_oi_change_pct",     # Call OI change %
        "put_oi_change_pct",      # Put OI change %
        "oi_buildup_signal",      # OI buildup signal
        "atm_pcr",                # ATM Put-Call ratio
        
        # === Volatility (6) ===
        "hv_10",                  # 10-day historical volatility
        "hv_20",                  # 20-day historical volatility
        "hv_ratio",               # HV 10/20 ratio
        "volatility_regime",      # Vol regime (-1=low, 0=normal, 1=high)
        "volatility_trend",       # Vol trend (-1=decreasing, 0=stable, 1=increasing)
        "range_volatility",       # Range-based volatility
        
        # === Momentum/Trend (6) ===
        "momentum_5d",            # 5-day momentum
        "momentum_10d",           # 10-day momentum
        "trend_strength",         # ADX-like trend strength
        "price_acceleration",     # Price acceleration
        "volume_trend",           # Volume trend (if available)
        "oi_trend",               # OI trend
    ]
    
    # Feature count
    N_FEATURES = len(FEATURE_NAMES)
    
    # Feature categories for analysis
    CATEGORIES = {
        "price_returns": FEATURE_NAMES[0:10],
        "technical": FEATURE_NAMES[10:22],
        "options_greeks": FEATURE_NAMES[22:30],
        "oi_sentiment": FEATURE_NAMES[30:38],
        "volatility": FEATURE_NAMES[38:44],
        "momentum_trend": FEATURE_NAMES[44:50],
    }


class HistoricalFeatureAdapter:
    """
    Adapts historical bhavcopy data to unified feature format.
    
    Takes processed bhavcopy DataFrame and outputs unified features.
    """
    
    def __init__(self):
        self.feature_names = UnifiedFeatureDefinition.FEATURE_NAMES
        
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract unified features from processed historical data.
        
        Args:
            df: DataFrame with processed bhavcopy data (from FullPipelineTrainer.process_symbol_data)
            
        Returns:
            DataFrame with unified feature columns
        """
        if df.empty:
            return df
        
        result = df.copy()
        
        # === Price/Returns ===
        result["return_1d"] = result["close"].pct_change()
        result["return_5d"] = result["close"].pct_change(5)
        result["return_10d"] = result["close"].pct_change(10)
        result["return_20d"] = result["close"].pct_change(20)
        result["intraday_range_pct"] = (result["high"] - result["low"]) / result["close"]
        result["close_to_high_pct"] = (result["close"] - result["low"]) / (result["high"] - result["low"] + 1e-10)
        result["gap_pct"] = (result["open"] - result["close"].shift(1)) / result["close"].shift(1)
        
        # SMAs
        result["sma_20"] = result["close"].rolling(20, min_periods=1).mean()
        result["sma_50"] = result["close"].rolling(50, min_periods=1).mean()
        result["price_vs_sma20"] = result["close"] / result["sma_20"] - 1
        result["price_vs_sma50"] = result["close"] / result["sma_50"] - 1
        result["log_return"] = np.log(result["close"] / result["close"].shift(1))
        
        # === Technical Indicators ===
        # RSI
        delta = result["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        result["rsi_14"] = 100 - (100 / (1 + rs))
        result["rsi_14"] = result["rsi_14"].fillna(50)
        result["rsi_signal"] = np.where(result["rsi_14"] < 30, -1, np.where(result["rsi_14"] > 70, 1, 0))
        
        # MACD
        ema12 = result["close"].ewm(span=12, min_periods=1).mean()
        ema26 = result["close"].ewm(span=26, min_periods=1).mean()
        result["macd"] = ema12 - ema26
        result["macd_signal"] = result["macd"].ewm(span=9, min_periods=1).mean()
        result["macd_histogram"] = result["macd"] - result["macd_signal"]
        
        # Bollinger Bands
        bb_mid = result["close"].rolling(20, min_periods=1).mean()
        bb_std = result["close"].rolling(20, min_periods=1).std().fillna(0)
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        result["bb_position"] = (result["close"] - bb_lower) / (bb_upper - bb_lower + 1e-10)
        result["bb_width"] = (bb_upper - bb_lower) / bb_mid
        
        # SMA/EMA Crossovers
        sma5 = result["close"].rolling(5, min_periods=1).mean()
        sma20 = result["close"].rolling(20, min_periods=1).mean()
        result["sma_crossover_5_20"] = np.where(sma5 > sma20, 1, np.where(sma5 < sma20, -1, 0))
        
        ema9 = result["close"].ewm(span=9, min_periods=1).mean()
        ema21 = result["close"].ewm(span=21, min_periods=1).mean()
        result["ema_crossover_9_21"] = np.where(ema9 > ema21, 1, np.where(ema9 < ema21, -1, 0))
        
        # Stochastic
        low_14 = result["low"].rolling(14, min_periods=1).min()
        high_14 = result["high"].rolling(14, min_periods=1).max()
        result["stochastic_k"] = 100 * (result["close"] - low_14) / (high_14 - low_14 + 1e-10)
        result["stochastic_d"] = result["stochastic_k"].rolling(3, min_periods=1).mean()
        
        # ATR
        tr1 = result["high"] - result["low"]
        tr2 = abs(result["high"] - result["close"].shift(1))
        tr3 = abs(result["low"] - result["close"].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=1).mean()
        result["atr_pct"] = atr / result["close"]
        
        # === Options/Greeks (Proxies from price data) ===
        # IV proxy from Parkinson volatility
        log_hl = np.log(result["high"] / result["low"])
        parkinson = np.sqrt((1 / (4 * np.log(2))) * (log_hl ** 2).rolling(20, min_periods=1).mean())
        result["iv_current"] = parkinson * np.sqrt(252)  # Annualized
        
        # IV percentile and rank
        result["iv_percentile"] = result["iv_current"].rolling(60, min_periods=1).apply(
            lambda x: 100 * (x.rank().iloc[-1] / len(x)) if len(x) > 0 else 50
        ).fillna(50)
        iv_min = result["iv_current"].rolling(60, min_periods=1).min()
        iv_max = result["iv_current"].rolling(60, min_periods=1).max()
        result["iv_rank"] = (result["iv_current"] - iv_min) / (iv_max - iv_min + 1e-10)
        
        # Delta proxy (momentum-based)
        momentum = result["return_1d"].rolling(10, min_periods=1).mean()
        result["delta_proxy"] = (momentum - momentum.min()) / (momentum.max() - momentum.min() + 1e-10)
        result["delta_proxy"] = result["delta_proxy"].fillna(0.5)
        
        # Gamma proxy (acceleration of returns)
        result["gamma_proxy"] = result["return_1d"].diff().abs().rolling(5, min_periods=1).mean()
        
        # Theta proxy (OI decay rate)
        if "total_oi" in result.columns:
            oi_change = result["total_oi"].pct_change()
            result["theta_proxy"] = -oi_change.rolling(5, min_periods=1).mean()
        else:
            result["theta_proxy"] = 0
        
        # Vega proxy (volatility sensitivity)
        vol_short = result["return_1d"].rolling(5, min_periods=1).std()
        vol_long = result["return_1d"].rolling(20, min_periods=1).std()
        result["vega_proxy"] = (vol_short - vol_long).abs()
        
        # IV/HV ratio
        hv_20 = result["return_1d"].rolling(20, min_periods=1).std() * np.sqrt(252)
        result["iv_hv_ratio"] = result["iv_current"] / (hv_20 + 1e-10)
        
        # === OI Sentiment ===
        if "pcr_oi" in result.columns:
            result["pcr"] = result["pcr_oi"]
        elif "call_oi" in result.columns and "put_oi" in result.columns:
            result["pcr"] = result["put_oi"] / (result["call_oi"] + 1e-10)
        else:
            result["pcr"] = 1.0
        
        result["pcr_ma5"] = result["pcr"].rolling(5, min_periods=1).mean()
        result["pcr_change"] = result["pcr"].diff()
        
        if "max_pain_distance" in result.columns:
            result["max_pain_distance_pct"] = result["max_pain_distance"]
        else:
            result["max_pain_distance_pct"] = 0
        
        if "call_oi_change" in result.columns and "call_oi" in result.columns:
            result["call_oi_change_pct"] = result["call_oi_change"] / (result["call_oi"].shift(1) + 1e-10)
            result["put_oi_change_pct"] = result["put_oi_change"] / (result["put_oi"].shift(1) + 1e-10)
        else:
            result["call_oi_change_pct"] = 0
            result["put_oi_change_pct"] = 0
        
        # OI buildup signal
        result["oi_buildup_signal"] = np.where(
            (result["call_oi_change_pct"] > 0) & (result["put_oi_change_pct"] > 0), 1,
            np.where((result["call_oi_change_pct"] < 0) & (result["put_oi_change_pct"] < 0), -1, 0)
        )
        
        if "atm_pcr" in result.columns:
            pass  # Already exists
        else:
            result["atm_pcr"] = result["pcr"]
        
        # === Volatility ===
        result["hv_10"] = result["return_1d"].rolling(10, min_periods=1).std() * np.sqrt(252)
        result["hv_20"] = result["return_1d"].rolling(20, min_periods=1).std() * np.sqrt(252)
        result["hv_ratio"] = result["hv_10"] / (result["hv_20"] + 1e-10)
        
        # Volatility regime
        hv_mean = result["hv_20"].rolling(60, min_periods=1).mean()
        hv_std = result["hv_20"].rolling(60, min_periods=1).std()
        result["volatility_regime"] = np.where(
            result["hv_20"] > hv_mean + hv_std, 1,
            np.where(result["hv_20"] < hv_mean - hv_std, -1, 0)
        )
        
        # Volatility trend
        hv_diff = result["hv_20"].diff(5)
        result["volatility_trend"] = np.where(hv_diff > 0.01, 1, np.where(hv_diff < -0.01, -1, 0))
        
        # Range volatility
        result["range_volatility"] = result["intraday_range_pct"].rolling(10, min_periods=1).mean()
        
        # === Momentum/Trend ===
        result["momentum_5d"] = result["close"] / result["close"].shift(5) - 1
        result["momentum_10d"] = result["close"] / result["close"].shift(10) - 1
        
        # Trend strength (ADX-like)
        up_move = result["high"].diff()
        down_move = -result["low"].diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        plus_di = pd.Series(plus_dm).rolling(14, min_periods=1).mean() / (atr + 1e-10) * 100
        minus_di = pd.Series(minus_dm).rolling(14, min_periods=1).mean() / (atr + 1e-10) * 100
        dx = abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
        result["trend_strength"] = dx.rolling(14, min_periods=1).mean().values
        
        # Price acceleration
        result["price_acceleration"] = result["return_1d"].diff()
        
        # Volume trend
        if "volume" in result.columns and result["volume"].sum() > 0:
            vol_ma = result["volume"].rolling(10, min_periods=1).mean()
            result["volume_trend"] = result["volume"] / (vol_ma + 1e-10) - 1
        else:
            result["volume_trend"] = 0
        
        # OI trend
        if "total_oi" in result.columns:
            oi_ma = result["total_oi"].rolling(5, min_periods=1).mean()
            result["oi_trend"] = result["total_oi"] / (oi_ma + 1e-10) - 1
        else:
            result["oi_trend"] = 0
        
        # Fill NaN and clip extremes
        for col in self.feature_names:
            if col in result.columns:
                result[col] = result[col].fillna(0).replace([np.inf, -np.inf], 0)
        
        return result
    
    def get_feature_array(self, df: pd.DataFrame) -> np.ndarray:
        """Get feature array for the last row."""
        if df.empty:
            return np.zeros(len(self.feature_names))
        
        row = df.iloc[-1]
        return np.array([row.get(name, 0.0) for name in self.feature_names])


class LiveFeatureAdapter:
    """
    Adapts live FeatureEngineer output to unified feature format.
    
    Maps the 61 FeatureEngineer features to the 50 unified features.
    """
    
    # Mapping from FeatureEngineer names to unified names
    FEATURE_MAPPING = {
        # Direct mappings
        "return_1d": "return_1d",
        "return_5d": "return_5d",
        "return_10d": "return_10d",
        "return_20d": "return_20d",
        "intraday_range_percent": "intraday_range_pct",
        "close_to_high_percent": "close_to_high_pct",
        "gap_percent": "gap_pct",
        "price_vs_sma20": "price_vs_sma20",
        "price_vs_sma50": "price_vs_sma50",
        
        # Technical
        "rsi_14": "rsi_14",
        "rsi_signal": "rsi_signal",
        "macd_histogram": "macd_histogram",
        "macd_signal_crossover": "macd_signal",
        "bollinger_percent_b": "bb_position",
        "bollinger_width": "bb_width",
        "sma_5_10_crossover": "sma_crossover_5_20",  # Close enough
        "ema_9_21_crossover": "ema_crossover_9_21",
        "stochastic_k": "stochastic_k",
        "stochastic_d": "stochastic_d",
        "atr_percent": "atr_pct",
        
        # Options/Greeks - use real values when available
        "iv_current": "iv_current",
        "iv_percentile": "iv_percentile",
        "iv_hv_ratio": "iv_hv_ratio",
        "position_delta": "delta_proxy",
        "position_gamma": "gamma_proxy",
        "position_theta": "theta_proxy",
        "position_vega": "vega_proxy",
        
        # OI Sentiment
        "pcr": "pcr",
        "pcr_change_1d": "pcr_change",
        "max_pain_distance_percent": "max_pain_distance_pct",
        "oi_buildup_signal": "oi_buildup_signal",
        "call_oi_change_percent": "call_oi_change_pct",
        "put_oi_change_percent": "put_oi_change_pct",
        
        # Volatility
        "hv_10": "hv_10",
        "hv_20": "hv_20",
        "hv_ratio_10_20": "hv_ratio",
        "volatility_regime": "volatility_regime",
        "volatility_trend": "volatility_trend",
    }
    
    def __init__(self):
        self.feature_names = UnifiedFeatureDefinition.FEATURE_NAMES
    
    def adapt(self, live_features: Dict[str, float]) -> Dict[str, float]:
        """
        Convert FeatureEngineer output to unified features.
        
        Args:
            live_features: Dictionary from FeatureEngineer.extract_features()
            
        Returns:
            Dictionary with unified feature names
        """
        unified = {}
        
        for live_name, unified_name in self.FEATURE_MAPPING.items():
            if live_name in live_features:
                unified[unified_name] = live_features[live_name]
        
        # Calculate any missing features
        for name in self.feature_names:
            if name not in unified:
                unified[name] = 0.0
        
        # Special calculations for features that need conversion
        # Log return
        if "return_1d" in unified:
            unified["log_return"] = np.log(1 + unified["return_1d"]) if unified["return_1d"] > -1 else 0
        
        # IV rank from percentile
        if "iv_percentile" in unified:
            unified["iv_rank"] = unified["iv_percentile"] / 100.0
        
        # MACD (histogram is already there, need raw MACD)
        if "macd_histogram" in unified:
            unified["macd"] = unified["macd_histogram"]  # Approximate
        
        # ATM PCR (use main PCR if not available)
        if "pcr" in unified and unified.get("atm_pcr", 0) == 0:
            unified["atm_pcr"] = unified["pcr"]
        
        # PCR MA (use current PCR if MA not available)
        if "pcr" in unified and unified.get("pcr_ma5", 0) == 0:
            unified["pcr_ma5"] = unified["pcr"]
        
        # Momentum from returns
        if "return_5d" in unified:
            unified["momentum_5d"] = unified["return_5d"]
        if "return_10d" in unified:
            unified["momentum_10d"] = unified["return_10d"]
        
        # Trend strength from ADX if available
        if "adx_14" in live_features:
            unified["trend_strength"] = live_features["adx_14"]
        
        # Range volatility from ATR
        if "atr_pct" in unified:
            unified["range_volatility"] = unified["atr_pct"]
        
        return unified
    
    def to_array(self, unified_features: Dict[str, float]) -> np.ndarray:
        """Convert unified features to array."""
        return np.array([unified_features.get(name, 0.0) for name in self.feature_names])


def get_unified_feature_names() -> List[str]:
    """Get the list of unified feature names."""
    return UnifiedFeatureDefinition.FEATURE_NAMES.copy()


def get_unified_feature_count() -> int:
    """Get the number of unified features."""
    return UnifiedFeatureDefinition.N_FEATURES
