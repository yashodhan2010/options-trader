"""
ML Signal Generator - ML-Only Signal Generation

All trading signals are derived from ML model predictions.
No rule-based signal generation - ML drives the entire signal flow.
"""
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import deque, Counter
import math
import pandas as pd

from data.data_fetcher import data_fetcher
from strategies.catalogue import StrategyCatalogue
from strategies.base_strategy import StrategySignal, StrategyType, TradeDirection, OptionLeg
from config.settings import (
    UNDERLYING_ASSETS, STRATEGY_CONFIG, ML_CONFIG, EVENT_REGIME_CONFIG,
    WATCHLIST, WATCHLIST_SYMBOLS, is_in_watchlist, get_watchlist_assets
)
from core.logger import logger
from core.utils import is_trading_allowed


def _get_ml_components():
    """Load ML components."""
    try:
        from ml.predictor import get_predictor
        from ml.feature_engineer import get_feature_engineer
        from ml.guardrails import get_guardrails
        
        return get_predictor(), get_feature_engineer(), get_guardrails()
    except ImportError as e:
        logger.error(f"Failed to import ML components: {e}")
        return None, None, None


class MLSignalGenerator:
    """
    ML-Only Signal Generator.
    
    All signals are driven by ML model predictions:
    1. ML model predicts direction (BULLISH/BEARISH/NEUTRAL) with confidence
    2. Direction maps to appropriate strategy type
    3. Strategy execution builds the actual option legs
    
    No rule-based signal generation - ML is the sole source of truth.
    
    RSI Logic (momentum vs reversion):
    - Stocks: RSI extremes indicate momentum (crash/rally) — avoid counter-trend trades
    - Indices: RSI extremes indicate mean-reversion — favor counter-trend trades
    """
    
    # Direction to strategy mapping based on IV regime
    # PRIORITY: Credit spreads first (sell premium, time decay in our favor)
    # Debit strategies only in LOW_IV where premium is cheap to buy
    DIRECTION_STRATEGY_MAP = {
        "BULLISH": {
            "LOW_IV": [StrategyType.BULL_PUT_SPREAD, StrategyType.BULL_CALL_SPREAD],
            "NORMAL": [StrategyType.BULL_PUT_SPREAD, StrategyType.BULL_CALL_SPREAD],
            "HIGH_IV": [StrategyType.BULL_PUT_SPREAD, StrategyType.SHORT_PUT],
        },
        "BEARISH": {
            "LOW_IV": [StrategyType.BEAR_CALL_SPREAD, StrategyType.BEAR_PUT_SPREAD],
            "NORMAL": [StrategyType.BEAR_CALL_SPREAD, StrategyType.BEAR_PUT_SPREAD],
            "HIGH_IV": [StrategyType.BEAR_CALL_SPREAD, StrategyType.SHORT_CALL],
        },
        "NEUTRAL": {
            "LOW_IV": [StrategyType.IRON_CONDOR],
            "NORMAL": [StrategyType.IRON_CONDOR],
            "HIGH_IV": [StrategyType.IRON_CONDOR, StrategyType.SHORT_CALL, StrategyType.SHORT_PUT],
        },
    }
    
    def __init__(self, underlyings: List[str] = None):
        """
        Initialize the ML signal generator.
        
        Args:
            underlyings: List of underlying assets to monitor.
                        If None, uses ML training symbols.
        """
        # Determine which assets to trade
        if underlyings:
            self.underlyings = underlyings
        elif WATCHLIST.get("enabled", False) and WATCHLIST_SYMBOLS:
            # Merge watchlist stocks with index symbols from UNDERLYING_ASSETS
            self.underlyings = list(WATCHLIST_SYMBOLS)  # Copy to avoid mutating
            for idx_symbol in UNDERLYING_ASSETS:
                if idx_symbol not in self.underlyings:
                    self.underlyings.append(idx_symbol)
            logger.info(f"Using watchlist + indices: {self.underlyings}")
        else:
            # Use ML training symbols as default
            self.underlyings = ML_CONFIG.get("training_symbols", list(UNDERLYING_ASSETS.keys()))
        
        self.catalogues: Dict[str, StrategyCatalogue] = {}
        self.last_signals: Dict[str, List[StrategySignal]] = {}
        self.signal_history: List[Dict] = []
        
        # ML components (lazy loaded)
        self._predictor = None
        self._feature_engineer = None
        self._guardrails = None
        self._ml_initialized = False
        
        # Configuration
        self.min_confidence = ML_CONFIG.get("min_confidence_for_trade", 0.55)
        self.model_loaded = False

        entropy_cfg = ML_CONFIG.get("direction_entropy_guard", {})
        self.entropy_guard_enabled = entropy_cfg.get("enabled", True)
        self.entropy_window_size = entropy_cfg.get("window_size", 40)
        self.entropy_min_samples = entropy_cfg.get("min_samples", 20)
        self.entropy_min_value = entropy_cfg.get("min_entropy", 0.35)
        self._direction_windows: Dict[str, deque] = {
            symbol: deque(maxlen=self.entropy_window_size) for symbol in self.underlyings
        }

        # Event-regime overlay state
        self.event_config = EVENT_REGIME_CONFIG
        self.event_enabled = self.event_config.get("enabled", False)
        risk_off_cfg = self.event_config.get("risk_off", {})
        pcv_cfg = risk_off_cfg.get("put_call_volume_spike", {})
        self._pcv_window_size = pcv_cfg.get("window_size", 24)
        self._pcv_windows: Dict[str, deque] = {
            symbol: deque(maxlen=self._pcv_window_size) for symbol in self.underlyings
        }
        self._risk_off_state: Dict[str, bool] = {symbol: False for symbol in self.underlyings}
        
        # Initialize strategy catalogues
        for underlying in self.underlyings:
            self.catalogues[underlying] = StrategyCatalogue(underlying)
        
        logger.info("ML-Only Signal Generator initialized")
        logger.info(f"Trading symbols: {self.underlyings}")
    
    def _init_ml(self) -> bool:
        """
        Initialize ML components.
        
        Returns:
            True if ML is ready for predictions
        """
        if self._ml_initialized:
            return self.model_loaded
        
        self._predictor, self._feature_engineer, self._guardrails = _get_ml_components()
        self._ml_initialized = True
        
        if not self._predictor or not self._feature_engineer:
            logger.error("ML components not available - no signals will be generated")
            return False
        
        # Try to load the model
        if self._predictor.load_model():
            self.model_loaded = True
            logger.info(f"ML model loaded: {self._predictor.model_version}")
            return True
        else:
            logger.error("No trained ML model found - train a model first")
            return False
    
    def generate_signals(
        self,
        underlying: Optional[str] = None,
        strategy_type: Optional[StrategyType] = None,
    ) -> List[StrategySignal]:
        """
        Generate trading signals based on ML predictions.
        
        Args:
            underlying: Specific underlying to analyze (None for all)
            strategy_type: Force specific strategy (overrides ML suggestion)
            
        Returns:
            List of generated signals
        """
        if not is_trading_allowed():
            logger.warning("Trading not allowed at this time")
            return []
        
        # Initialize ML if needed
        if not self._init_ml():
            logger.error("ML not ready - cannot generate signals")
            return []
        
        targets = [underlying] if underlying else self.underlyings
        all_signals = []
        global_event_context = self._build_global_event_context() if self.event_enabled else {}
        
        for target in targets:
            try:
                signals = self._generate_ml_signals(target, strategy_type, global_event_context)
                all_signals.extend(signals)
                self.last_signals[target] = signals
                
                # Record in history
                for signal in signals:
                    self.signal_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "underlying": target,
                        "signal": signal.to_dict(),
                        "source": "ML",
                    })
                    
            except Exception as e:
                logger.error(f"Error generating ML signals for {target}: {e}")
                import traceback
                traceback.print_exc()
        
        # Sort all signals by confidence
        all_signals.sort(key=lambda s: s.confidence, reverse=True)
        
        logger.info(f"Generated {len(all_signals)} ML signals")
        return all_signals
    
    def _generate_ml_signals(
        self,
        underlying: str,
        force_strategy: Optional[StrategyType] = None,
        global_event_context: Optional[Dict[str, Any]] = None,
    ) -> List[StrategySignal]:
        """
        Generate signals for a single underlying using ML prediction.
        
        Args:
            underlying: The underlying asset
            force_strategy: Force a specific strategy type
            
        Returns:
            List of signals for this underlying
        """
        logger.info(f"ML Analysis: {underlying}...")
        
        # Fetch market data
        spot = data_fetcher.get_spot_price(underlying)
        if not spot:
            logger.warning(f"Could not get spot price for {underlying}")
            return []
        
        # Get options chains across ALL valid expiries (DTE range from config)
        expiry_chains = data_fetcher.get_options_chains_multi_expiry(
            underlying,
            num_strikes=15,
        )
        
        if not expiry_chains:
            # Fallback to single-expiry method for backward compatibility
            single_chain = data_fetcher.get_options_chain(underlying, num_strikes=15)
            if single_chain.empty:
                logger.warning(f"Empty options chain for {underlying}")
                return []
            expiry_chains = {None: single_chain}
        
        # Get market metrics
        oi_data = data_fetcher.get_oi_data(underlying)
        volatility = data_fetcher.get_volatility_metrics(underlying)
        historical = data_fetcher.get_historical_analysis(underlying, days=30)
        
        market_data = {
            "spot": spot,
            "oi_data": oi_data,
            "volatility": volatility,
            "historical": historical,
        }
        
        # Use nearest expiry chain for ML feature extraction (direction prediction)
        nearest_chain = next(iter(expiry_chains.values()))
        
        # Extract features for ML prediction
        features = self._feature_engineer.extract_features(
            underlying=underlying,
            spot_price=spot,
            historical_data=historical.get("df") if isinstance(historical, dict) else historical,
            options_chain=nearest_chain,
            oi_analysis=oi_data,
            volatility_data=volatility,
        )
        
        if not features:
            logger.warning(f"Could not extract features for {underlying}")
            return []
        
        prediction, blended_confidence, should_trade = self._predictor.predict_with_guardrails(
            features=features,
            rule_confidence=self.min_confidence,
            underlying=underlying,
            current_positions=0,
            is_paper_mode=True,
        )
        
        if prediction is None:
            logger.warning(f"ML prediction failed for {underlying}")
            return []

        if getattr(prediction, "abstained", False):
            logger.info(f"Abstain band triggered for {underlying} - skipping trade")
            return []

        prediction.raw_ml_confidence = prediction.confidence
        prediction.rule_confidence = self.min_confidence
        prediction.blended_confidence = blended_confidence
        prediction.confidence = blended_confidence
        
        logger.info(
            f"ML Prediction for {underlying}: {prediction.direction} "
            f"(confidence: {prediction.confidence:.1%})"
        )
        
        if not should_trade:
            logger.info(
                f"Guardrails blocked trade for {underlying} "
                f"(ML: {prediction.raw_ml_confidence:.1%}, blended: {prediction.confidence:.1%})"
            )
            return []

        # Check confidence threshold
        if prediction.confidence < self.min_confidence:
            logger.info(
                f"ML confidence {prediction.confidence:.1%} below threshold "
                f"{self.min_confidence:.1%} - skipping {underlying}"
            )
            return []

        # Intraday context is used both for timing and event-regime flip detection
        intraday = data_fetcher.get_intraday_analysis(underlying)
        intraday_bias = intraday.get("intraday_bias", "NEUTRAL") if intraday else "NEUTRAL"

        # Event-regime overlay (proxy + flow based)
        event_outcome = self._apply_event_regime_overlay(
            underlying=underlying,
            prediction=prediction,
            oi_data=oi_data,
            intraday=intraday,
            global_event_context=global_event_context or {},
        )
        if event_outcome.get("blocked"):
            logger.warning(
                f"Event regime blocked {underlying}: direction={prediction.direction}, "
                f"reasons={event_outcome.get('reasons', [])}"
            )
            return []

        self._record_prediction_direction(underlying, prediction.direction)
        if not self._passes_direction_entropy_guard(underlying):
            logger.info(f"Direction entropy guard blocked trade for {underlying}")
            return []
        
        # Trend confirmation: validate ML direction against recent prediction history
        if not self._confirm_trend(underlying, prediction.direction):
            logger.info(
                f"Trend confirmation failed for {underlying} ({prediction.direction}) - skipping"
            )
            return []
        
        # ========== INTRADAY TIMING FILTER ==========
        # Daily ML gives direction (swing map), intraday candles gate entry timing.
        # Only enter when 5-min price action aligns or is neutral.
        
        if prediction.direction == "BULLISH" and intraday_bias == "BEARISH":
            logger.info(
                f"Intraday timing mismatch for {underlying}: ML=BULLISH but intraday=BEARISH "
                f"(VWAP={'above' if intraday.get('above_vwap') else 'below'}, "
                f"micro={intraday.get('micro_trend')}, RSI5m={intraday.get('rsi_5m', '?')}) "
                f"- waiting for better entry"
            )
            return []
        elif prediction.direction == "BEARISH" and intraday_bias == "BULLISH":
            logger.info(
                f"Intraday timing mismatch for {underlying}: ML=BEARISH but intraday=BULLISH "
                f"(VWAP={'above' if intraday.get('above_vwap') else 'below'}, "
                f"micro={intraday.get('micro_trend')}, RSI5m={intraday.get('rsi_5m', '?')}) "
                f"- waiting for better entry"
            )
            return []
        else:
            if intraday:
                logger.info(
                    f"Intraday timing OK for {underlying}: ML={prediction.direction}, "
                    f"intraday={intraday_bias} "
                    f"(VWAP={'above' if intraday.get('above_vwap') else 'below'}, "
                    f"micro={intraday.get('micro_trend')}, RSI5m={intraday.get('rsi_5m', '?')})"
                )
        
        # Store intraday data in market_data for strategies to use
        market_data["intraday"] = intraday
        
        # Determine strategy based on ML direction + RSI context
        rsi = historical.get("rsi", 50) if isinstance(historical, dict) else 50
        rsi_signal = historical.get("rsi_signal", "NEUTRAL") if isinstance(historical, dict) else "NEUTRAL"
        is_index = underlying in UNDERLYING_ASSETS
        
        if force_strategy:
            strategy_types = [force_strategy]
        else:
            strategy_types = self._get_strategies_for_direction(
                prediction.direction,
                volatility.get("volatility_regime", "NORMAL"),
                rsi=rsi,
                is_index=is_index,
            )
        
        # Generate signals for selected strategies across ALL expiry chains
        signals = []
        catalogue = self.catalogues.get(underlying)
        
        if not catalogue:
            return []
        
        for expiry_date, options_chain in expiry_chains.items():
            dte_label = f" (DTE: {(expiry_date - datetime.now().date()).days})" if expiry_date else ""
            
            for strategy_type in strategy_types:
                strategy = catalogue.get_strategy(strategy_type)
                if not strategy:
                    continue
                
                try:
                    strategy.ml_override = True
                    signal = strategy.analyze(options_chain, market_data)
                    strategy.ml_override = False
                    
                    if signal:
                        ml_signal = self._create_ml_signal(signal, prediction)
                        signals.append(ml_signal)
                        logger.info(
                            f"  Signal generated: {strategy_type.value} for {underlying}{dte_label} "
                            f"(ML confidence: {prediction.confidence:.1%})"
                        )
                    
                except Exception as e:
                    logger.error(f"Strategy {strategy_type.value} failed for {underlying}{dte_label}: {e}")
                    import traceback
                    traceback.print_exc()
                    strategy.ml_override = False
        
        return signals

    def _build_global_event_context(self) -> Dict[str, Any]:
        """Build global proxy-based event context for the current scan."""
        context: Dict[str, Any] = {"risk_off_score": 0.0, "reasons": []}
        try:
            risk_off_cfg = self.event_config.get("risk_off", {})
            breadth_cfg = risk_off_cfg.get("market_breadth", {})
            idx_symbols = breadth_cfg.get("index_symbols", [])
            bearish_threshold = breadth_cfg.get("bearish_return_5d_threshold", -0.8)
            min_bearish_count = breadth_cfg.get("min_bearish_count", 3)
            score_weight = breadth_cfg.get("score_weight", 1.0)

            bearish_count = 0
            returns_5d: Dict[str, float] = {}
            for symbol in idx_symbols:
                hist = data_fetcher.get_historical_analysis(symbol, days=30)
                if not hist:
                    continue
                ret5 = hist.get("returns_5d")
                if ret5 is None:
                    continue
                returns_5d[symbol] = float(ret5)
                if float(ret5) <= bearish_threshold:
                    bearish_count += 1

            context["index_returns_5d"] = returns_5d
            context["bearish_index_count"] = bearish_count
            if bearish_count >= min_bearish_count:
                context["risk_off_score"] += score_weight
                context["reasons"].append(
                    f"breadth_bearish={bearish_count}/{max(len(idx_symbols), 1)}"
                )

            commodity_cfg = self.event_config.get("commodity_proxy", {})
            commodity_symbols = commodity_cfg.get("symbols", [])
            if commodity_symbols:
                bull_th = commodity_cfg.get("bullish_return_5d_threshold", 1.5)
                min_bull = commodity_cfg.get("min_bullish_count", 1)
                comm_weight = commodity_cfg.get("score_weight", 0.5)
                commodity_bull = 0
                for symbol in commodity_symbols:
                    hist = data_fetcher.get_historical_analysis(symbol, days=30)
                    ret5 = (hist or {}).get("returns_5d")
                    if ret5 is not None and float(ret5) >= bull_th:
                        commodity_bull += 1
                if commodity_bull >= min_bull:
                    context["risk_off_score"] += comm_weight
                    context["reasons"].append(f"commodity_bull={commodity_bull}")

        except Exception as e:
            logger.debug(f"Failed to build event context: {e}")

        return context

    def _apply_event_regime_overlay(
        self,
        underlying: str,
        prediction,
        oi_data: Dict[str, Any],
        intraday: Dict[str, Any],
        global_event_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply event-regime scoring and optionally flip/block signals."""
        if not self.event_enabled:
            return {"active": False, "blocked": False, "reasons": []}

        risk_off_cfg = self.event_config.get("risk_off", {})
        entry_threshold = risk_off_cfg.get("entry_threshold", 2.0)
        exit_threshold = risk_off_cfg.get("exit_threshold", 1.0)
        shock_cfg = risk_off_cfg.get("intraday_shock", {})
        pcv_cfg = risk_off_cfg.get("put_call_volume_spike", {})

        score = float(global_event_context.get("risk_off_score", 0.0))
        reasons = list(global_event_context.get("reasons", []))

        recent_5c_ret = (intraday or {}).get("recent_5c_return_pct")
        if recent_5c_ret is not None and float(recent_5c_ret) <= shock_cfg.get("recent_5c_return_pct_threshold", -0.35):
            score += float(shock_cfg.get("score_weight", 0.7))
            reasons.append(f"intraday_shock={float(recent_5c_ret):.3f}%")

        if pcv_cfg.get("enabled", True):
            pcv_ratio = oi_data.get("put_call_volume_ratio") if oi_data else None
            if pcv_ratio is not None:
                window = self._pcv_windows.setdefault(underlying, deque(maxlen=self._pcv_window_size))
                baseline = (sum(window) / len(window)) if len(window) >= pcv_cfg.get("min_samples", 8) else None
                if baseline and baseline > 0:
                    multiplier = pcv_cfg.get("spike_multiplier", 1.6)
                    min_ratio = pcv_cfg.get("min_ratio", 1.2)
                    if float(pcv_ratio) >= max(min_ratio, baseline * multiplier):
                        score += float(pcv_cfg.get("score_weight", 1.2))
                        reasons.append(
                            f"pcv_spike={float(pcv_ratio):.2f} vs avg={baseline:.2f}"
                        )
                window.append(float(pcv_ratio))

        prev_state = self._risk_off_state.get(underlying, False)
        if prev_state:
            is_active = score >= exit_threshold
        else:
            is_active = score >= entry_threshold
        self._risk_off_state[underlying] = is_active

        if not is_active:
            return {"active": False, "blocked": False, "reasons": reasons, "score": score}

        sensitive = set(self.event_config.get("risk_off_sensitive_symbols", []))
        is_sensitive = underlying in sensitive or underlying in UNDERLYING_ASSETS
        flip_mode = self.event_config.get("flip_mode", "flip_to_bearish")

        if is_sensitive and prediction.direction == "BULLISH":
            if flip_mode == "block_bullish":
                return {
                    "active": True,
                    "blocked": True,
                    "reasons": reasons,
                    "score": score,
                }

            original_direction = prediction.direction
            prediction.direction = "BEARISH"
            penalty = float(self.event_config.get("confidence_penalty", 0.12))
            prediction.confidence = max(0.35, prediction.confidence - penalty)
            logger.warning(
                f"[EVENT_FLIP] {underlying}: {original_direction} -> {prediction.direction}, "
                f"score={score:.2f}, reasons={reasons}"
            )

        return {"active": True, "blocked": False, "reasons": reasons, "score": score}
    
    def _confirm_trend(self, underlying: str, ml_direction: str) -> bool:
        """
        Validate ML prediction against recent feature snapshot labels from the database.
        Requires 60%+ of recent labels to align with the ML direction.
        
        Args:
            underlying: The underlying asset
            ml_direction: ML predicted direction (BULLISH/BEARISH/NEUTRAL)
            
        Returns:
            True if trend is confirmed or no history available
        """
        if ml_direction == "NEUTRAL":
            return True  # Neutral doesn't need trend confirmation
        
        try:
            import sqlite3
            import os
            
            db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "trading_bot.db")
            if not os.path.exists(db_path):
                return True  # No DB, skip confirmation
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ml_feature_snapshots'")
            if not cursor.fetchone():
                conn.close()
                return True
            
            # Get recent labels for this underlying (last 5 trading days, max 10 days old)
            cursor.execute("""
                SELECT label_direction FROM ml_feature_snapshots 
                WHERE underlying = ? AND label_direction IS NOT NULL AND label_direction != 'NEUTRAL'
                  AND snapshot_time >= datetime('now', '-10 days')
                ORDER BY snapshot_time DESC LIMIT 5
            """, (underlying,))
            
            rows = cursor.fetchall()
            conn.close()
            
            if len(rows) < 3:
                return True  # Not enough recent history, allow the trade
            
            labels = [r[0] for r in rows]
            
            # Map ML direction to expected label
            expected_label = "UP" if ml_direction == "BULLISH" else "DOWN"
            
            # Count alignment
            aligned = sum(1 for l in labels if l == expected_label)
            alignment_pct = aligned / len(labels)
            
            logger.info(
                f"Trend confirmation for {underlying}: {aligned}/{len(labels)} recent labels = "
                f"{expected_label} ({alignment_pct:.0%}), ML says {ml_direction}"
            )
            
            return alignment_pct >= 0.6  # Require 60%+ alignment
            
        except Exception as e:
            logger.warning(f"Trend confirmation check failed: {e}")
            return True  # On error, allow the trade

    def _record_prediction_direction(self, underlying: str, direction: str) -> None:
        """Record predicted direction for entropy monitoring."""
        if underlying not in self._direction_windows:
            self._direction_windows[underlying] = deque(maxlen=self.entropy_window_size)
        self._direction_windows[underlying].append(direction)

    def _passes_direction_entropy_guard(self, underlying: str) -> bool:
        """Block trading if recent prediction directions collapse to near-single class."""
        if not self.entropy_guard_enabled:
            return True

        window = self._direction_windows.get(underlying)
        if not window or len(window) < self.entropy_min_samples:
            return True

        counts = Counter(window)
        total = sum(counts.values())
        if total == 0:
            return True

        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        if entropy < self.entropy_min_value:
            logger.warning(
                f"Low direction entropy for {underlying}: H={entropy:.3f} < {self.entropy_min_value:.3f}, "
                f"counts={dict(counts)}"
            )
            return False

        return True
    
    # Credit-only strategies for cautious RSI situations
    _CREDIT_ONLY = {
        "BULLISH": [StrategyType.BULL_PUT_SPREAD],
        "BEARISH": [StrategyType.BEAR_CALL_SPREAD],
    }

    def _get_strategies_for_direction(
        self,
        direction: str,
        iv_regime: str,
        rsi: float = 50.0,
        is_index: bool = False,
    ) -> List[StrategyType]:
        """
        Get appropriate strategies based on ML direction, IV regime, and RSI context.
        
        RSI-aware filtering (momentum vs mean-reversion):
        - Stocks follow momentum: RSI extremes aligned with ML direction confirm the move
          and allow full strategy list. RSI extremes opposing ML direction signal caution
          (e.g., RSI < 30 but ML says BULLISH = catching a falling knife) — restrict
          to credit-only (defined risk, time decay in our favor).
        - Indices mean-revert: RSI extremes opposing the recent trend are opportunities
          (e.g., RSI < 30 + BULLISH ML = bounce likely) — allow full strategy list.
          RSI extremes confirming the trend are stretched (e.g., RSI > 70 + BULLISH ML
          = overbought index) — restrict to credit-only.
        
        Args:
            direction: ML predicted direction (BULLISH/BEARISH/NEUTRAL)
            iv_regime: Current IV regime (LOW_IV/NORMAL/HIGH_IV)
            rsi: Current 14-period RSI value (0-100)
            is_index: True for index underlyings, False for stocks
            
        Returns:
            List of suitable strategy types
        """
        # Normalize IV regime
        if "LOW" in iv_regime.upper() or "DEPRESS" in iv_regime.upper():
            regime_key = "LOW_IV"
        elif "HIGH" in iv_regime.upper() or "ELEVAT" in iv_regime.upper():
            regime_key = "HIGH_IV"
        else:
            regime_key = "NORMAL"
        
        # Get base strategies from IV-direction mapping
        direction_map = self.DIRECTION_STRATEGY_MAP.get(direction, {})
        strategies = list(direction_map.get(regime_key, []))
        
        # ── RSI-aware filtering ──────────────────────────────────────
        RSI_OVERSOLD = 30
        RSI_OVERBOUGHT = 70

        rsi_caution = False  # True → restrict to credit-only

        if direction != "NEUTRAL" and (rsi < RSI_OVERSOLD or rsi > RSI_OVERBOUGHT):
            if is_index:
                # Indices mean-revert:
                #   RSI < 30 + BULLISH  → bounce likely (reversion) → full list ✓
                #   RSI > 70 + BEARISH  → drop likely (reversion)  → full list ✓
                #   RSI < 30 + BEARISH  → stretched down, ML says more → caution
                #   RSI > 70 + BULLISH  → stretched up, ML says more  → caution
                if (rsi < RSI_OVERSOLD and direction == "BEARISH") or \
                   (rsi > RSI_OVERBOUGHT and direction == "BULLISH"):
                    rsi_caution = True
                    logger.info(
                        f"RSI caution (INDEX mean-reversion): RSI={rsi:.1f} with "
                        f"{direction} — stretched, restricting to credit-only"
                    )
            else:
                # Stocks follow momentum:
                #   RSI < 30 + BEARISH  → momentum confirms → full list ✓
                #   RSI > 70 + BULLISH  → momentum confirms → full list ✓
                #   RSI < 30 + BULLISH  → catching falling knife → caution
                #   RSI > 70 + BEARISH  → shorting a strong rally   → caution
                if (rsi < RSI_OVERSOLD and direction == "BULLISH") or \
                   (rsi > RSI_OVERBOUGHT and direction == "BEARISH"):
                    rsi_caution = True
                    logger.info(
                        f"RSI caution (STOCK momentum): RSI={rsi:.1f} with "
                        f"{direction} — counter-momentum, restricting to credit-only"
                    )

        if rsi_caution:
            credit_list = self._CREDIT_ONLY.get(direction, [])
            if credit_list:
                strategies = list(credit_list)
        # ──────────────────────────────────────────────────────────────
        
        # Filter by enabled strategies
        enabled = STRATEGY_CONFIG.get("enabled_strategies", [])
        
        filtered = []
        for strategy_type in strategies:
            if strategy_type.value in enabled:
                filtered.append(strategy_type)
        
        # If no enabled strategies match, return first available
        if not filtered and strategies:
            filtered = [strategies[0]]
        
        return filtered
    
    def _create_ml_signal(
        self,
        base_signal: StrategySignal,
        prediction,
    ) -> StrategySignal:
        """
        Create a signal with ML prediction as the source of truth.
        
        Args:
            base_signal: Strategy-generated signal with option legs
            prediction: ML prediction result
            
        Returns:
            Enhanced signal with ML confidence
        """
        # Create new signal with ML confidence using actual StrategySignal fields
        ml_signal = StrategySignal(
            underlying=base_signal.underlying,
            strategy_type=base_signal.strategy_type,
            confidence=prediction.confidence,  # Use ML confidence
            legs=base_signal.legs,
            entry_time=base_signal.entry_time,
            expected_profit=base_signal.expected_profit,
            max_loss=base_signal.max_loss,
            stop_loss=base_signal.stop_loss,
            target=base_signal.target,
            rationale=f"[ML Signal] {prediction.direction} ({prediction.confidence:.1%}) - {base_signal.rationale}",
            metrics=base_signal.metrics.copy() if base_signal.metrics else {},
        )
        
        # Add ML metadata to metrics
        ml_signal.metrics["ml_direction"] = prediction.direction
        ml_signal.metrics["ml_confidence"] = getattr(prediction, "raw_ml_confidence", prediction.confidence)
        ml_signal.metrics["rule_confidence"] = getattr(prediction, "rule_confidence", self.min_confidence)
        ml_signal.metrics["blended_confidence"] = prediction.confidence
        ml_signal.metrics["ml_probabilities"] = prediction.probabilities
        ml_signal.metrics["ml_model_version"] = prediction.model_version
        ml_signal.metrics["ml_model_type"] = prediction.model_type
        ml_signal.metrics["signal_source"] = "ML"
        
        return ml_signal
    
    def get_best_signal(self, underlying: Optional[str] = None) -> Optional[StrategySignal]:
        """
        Get the single best ML signal.
        
        Args:
            underlying: Specific underlying (None for any)
            
        Returns:
            Best signal or None
        """
        signals = self.generate_signals(underlying)
        return signals[0] if signals else None
    
    def get_market_overview(self, underlying: str) -> Dict[str, Any]:
        """
        Get market overview with ML prediction.
        
        Args:
            underlying: The underlying asset
            
        Returns:
            Dictionary with market overview and ML prediction
        """
        # Initialize ML if needed
        self._init_ml()
        
        spot = data_fetcher.get_spot_price(underlying)
        oi_data = data_fetcher.get_oi_data(underlying)
        volatility = data_fetcher.get_volatility_metrics(underlying)
        historical = data_fetcher.get_historical_analysis(underlying, days=30)
        options_chain = data_fetcher.get_options_chain(underlying)
        
        # Get ML prediction
        ml_prediction = None
        if self.model_loaded and self._feature_engineer:
            features = self._feature_engineer.extract_features(
                underlying=underlying,
                spot_price=spot,
                historical_data=historical.get("df") if isinstance(historical, dict) else historical,
                options_chain=options_chain,
                oi_analysis=oi_data,
                volatility_data=volatility,
            )
            
            if features:
                prediction = self._predictor.predict(features, underlying)
                if prediction:
                    ml_prediction = {
                        "direction": prediction.direction,
                        "confidence": prediction.confidence,
                        "probabilities": prediction.probabilities,
                        "model_version": prediction.model_version,
                    }
        
        return {
            "underlying": underlying,
            "spot": spot,
            "timestamp": datetime.now().isoformat(),
            "oi_analysis": {
                "pcr": oi_data.get("pcr"),
                "total_call_oi": oi_data.get("total_call_oi"),
                "total_put_oi": oi_data.get("total_put_oi"),
                "max_pain": oi_data.get("max_pain"),
                "sentiment": oi_data.get("sentiment"),
            },
            "volatility": {
                "hv_20": volatility.get("hv_20"),
                "atm_iv": volatility.get("atm_iv"),
                "regime": volatility.get("volatility_regime"),
            },
            "ml_prediction": ml_prediction,
            "recommended_strategies": self._get_strategies_for_direction(
                ml_prediction["direction"] if ml_prediction else "NEUTRAL",
                volatility.get("volatility_regime", "NORMAL")
            ) if ml_prediction else [],
        }
    
    def get_model_status(self) -> Dict[str, Any]:
        """
        Get ML model status.
        
        Returns:
            Dictionary with model status information
        """
        self._init_ml()
        
        if not self._predictor:
            return {
                "status": "error",
                "message": "ML predictor not available",
                "model_loaded": False,
            }
        
        return {
            "status": "ready" if self.model_loaded else "no_model",
            "model_loaded": self.model_loaded,
            "model_version": self._predictor.model_version if self.model_loaded else None,
            "model_type": self._predictor.model_type if self.model_loaded else None,
            "min_confidence": self.min_confidence,
            "underlyings": self.underlyings,
        }

    def get_entropy_status(self, underlying: Optional[str] = None) -> Dict[str, Any]:
        """Get rolling direction entropy status for one or all symbols."""
        targets = [underlying] if underlying else self.underlyings
        result: Dict[str, Any] = {
            "enabled": self.entropy_guard_enabled,
            "window_size": self.entropy_window_size,
            "min_samples": self.entropy_min_samples,
            "min_entropy": self.entropy_min_value,
            "symbols": {},
        }

        for symbol in targets:
            window = self._direction_windows.get(symbol, deque(maxlen=self.entropy_window_size))
            counts = Counter(window)
            total = sum(counts.values())

            entropy = None
            if total > 0:
                entropy_val = 0.0
                for count in counts.values():
                    p = count / total
                    if p > 0:
                        entropy_val -= p * math.log2(p)
                entropy = entropy_val

            tradable = True
            reason = "ok"
            if self.entropy_guard_enabled:
                if total < self.entropy_min_samples:
                    tradable = False
                    reason = f"warming_up ({total}/{self.entropy_min_samples})"
                elif entropy is not None and entropy < self.entropy_min_value:
                    tradable = False
                    reason = f"low_entropy ({entropy:.3f} < {self.entropy_min_value:.3f})"

            result["symbols"][symbol] = {
                "samples": total,
                "distribution": dict(counts),
                "entropy": entropy,
                "tradable": tradable,
                "reason": reason,
            }

        return result
    
    def clear_history(self) -> None:
        """Clear signal history."""
        self.signal_history = []
        self.last_signals = {}
        logger.info("Signal history cleared")


# Singleton instance
ml_signal_generator = MLSignalGenerator()

# Alias for backward compatibility - signal_generator now points to ML version
signal_generator = ml_signal_generator
