"""
ML Predictor for Options Trading

Provides inference from trained ML models with ensemble support,
confidence blending, and risk guardrails integration.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import joblib

from config.settings import ML_CONFIG
from core.logger import logger


@dataclass
class MLPrediction:
    """Container for ML prediction results."""
    direction: str                  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    confidence: float               # 0.0 to 1.0
    probabilities: Dict[str, float] # Probability for each class
    model_version: str
    model_type: str
    timestamp: datetime
    feature_importance: Optional[Dict[str, float]] = None
    raw_prediction: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "direction": self.direction,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "model_version": self.model_version,
            "model_type": self.model_type,
            "timestamp": self.timestamp.isoformat(),
            "raw_prediction": self.raw_prediction,
        }


class MLPredictor:
    """
    ML prediction engine for options trading.
    
    Features:
    - Load and cache trained models
    - Ensemble prediction with configurable weights
    - Confidence blending with rule-based signals
    - Integration with trading guardrails
    - Prediction caching for performance
    """
    
    # Direction mappings
    DIRECTION_MAP = {
        0: "BEARISH",
        1: "NEUTRAL", 
        2: "BULLISH",
    }
    
    REVERSE_DIRECTION_MAP = {
        "BEARISH": 0,
        "NEUTRAL": 1,
        "BULLISH": 2,
    }
    
    def __init__(self):
        """Initialize the ML predictor."""
        self.model = None
        self.scaler = None
        self.feature_names: List[str] = []
        self.model_version: Optional[str] = None
        self.model_type: Optional[str] = None
        self.model_timestamp: Optional[datetime] = None
        self._class_labels: Optional[np.ndarray] = None
        self._symbol_model_cache: Dict[str, Dict[str, Any]] = {}
        self._active_symbol: Optional[str] = None
        self._global_model_bundle: Optional[Dict[str, Any]] = None
        
        # Feedback-based adjustments
        self.confidence_adjustment: float = 1.0  # Multiplier from feedback
        self.feedback_config: Dict = {}
        
        # Prediction cache
        self._prediction_cache: Dict[str, Tuple[MLPrediction, datetime]] = {}
        self.cache_ttl = ML_CONFIG.get("prediction_cache_seconds", 60)
        
        # Guardrails (lazy loaded)
        self._guardrails = None
        
        # Model trainer for loading
        self._model_trainer = None
        
        self.enabled = ML_CONFIG.get("enabled", True)
        self.confidence_weight = ML_CONFIG.get("confidence_weight", 0.5)
        self.prefer_symbol_models = ML_CONFIG.get("prefer_symbol_models", True)
        hybrid_cfg = ML_CONFIG.get("hybrid_routing", {})
        self.hybrid_enabled = hybrid_cfg.get("enabled", False)
        self.local_symbol_allowlist = {
            str(s).upper() for s in hybrid_cfg.get("local_symbol_allowlist", [])
        }
        self.force_global_symbols = {
            str(s).upper() for s in hybrid_cfg.get("force_global_symbols", [])
        }
        self.event_override_cfg = hybrid_cfg.get("event_override", {})
        self.event_override_enabled = bool(self.event_override_cfg.get("enabled", False))
        self.event_override_symbols = {
            str(s).upper() for s in self.event_override_cfg.get("symbols", [])
        }
        self.event_override_min_score = float(self.event_override_cfg.get("min_score", 2.0))
        self.event_override_min_local_confidence = float(
            self.event_override_cfg.get("min_local_confidence", 0.56)
        )
        self.event_override_confidence_penalty = float(
            self.event_override_cfg.get("confidence_penalty", 0.05)
        )
        self.event_thresholds = self.event_override_cfg.get("thresholds", {})
        self.event_weights = self.event_override_cfg.get("weights", {})

        abstain_band = ML_CONFIG.get("abstain_band", {})
        self.default_abstain_margin = abstain_band.get("default_margin", 0.08)
        self.min_top_probability = abstain_band.get("min_top_probability", 0.45)
        self.abstain_margin_by_symbol = abstain_band.get("by_symbol", {})

        feature_schema = ML_CONFIG.get("feature_schema", {})
        self.feature_schema_strict = feature_schema.get("strict_mode", True)
        self.min_feature_overlap_ratio = feature_schema.get("min_overlap_ratio", 0.5)
        self.warn_feature_overlap_ratio = feature_schema.get("warn_overlap_ratio", 0.75)
        self.max_missing_features = feature_schema.get("max_missing_features", 25)
        
        logger.info("MLPredictor initialized")
    
    @property
    def guardrails(self):
        """Lazy load guardrails."""
        if self._guardrails is None:
            from ml.guardrails import get_guardrails
            self._guardrails = get_guardrails()
        return self._guardrails
    
    @property
    def model_trainer(self):
        """Lazy load model trainer."""
        if self._model_trainer is None:
            from ml.model_trainer import get_model_trainer
            self._model_trainer = get_model_trainer()
        return self._model_trainer
    
    def load_model(self, model_version: str = None) -> bool:
        """
        Load a trained model for prediction.
        
        Args:
            model_version: Specific version to load, or None for latest
            
        Returns:
            True if model loaded successfully
        """
        try:
            if model_version is None:
                model_version = self.model_trainer.get_latest_model()
            
            if model_version is None:
                logger.warning("No trained model found")
                return False
            
            self.model, self.scaler, self.feature_names, metadata = \
                self.model_trainer.load_model(model_version)
            
            self.model_version = model_version
            self.model_type = metadata.get("model_type", "unknown")
            
            trained_at = metadata.get("trained_at")
            if trained_at:
                self.model_timestamp = datetime.fromisoformat(trained_at)
            else:
                self.model_timestamp = datetime.now()

            class_labels = metadata.get("class_labels") if isinstance(metadata, dict) else None
            self._class_labels = np.array(class_labels) if class_labels else None
            self._active_symbol = None

            self._global_model_bundle = {
                "model": self.model,
                "scaler": self.scaler,
                "feature_names": list(self.feature_names),
                "model_version": self.model_version,
                "model_type": self.model_type,
                "model_timestamp": self.model_timestamp,
                "class_labels": self._class_labels,
            }
            
            # Load feedback config if available
            self._load_feedback_config(model_version)
            
            # Clear cache when loading new model
            self._prediction_cache.clear()
            
            logger.info(f"Loaded model: {model_version} ({self.model_type}), confidence_adj={self.confidence_adjustment:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def _activate_model_bundle(self, underlying: str, bundle: Dict[str, Any]) -> bool:
        try:
            self.model = bundle.get("model")
            self.scaler = bundle.get("scaler")
            self.feature_names = bundle.get("feature_names", [])
            self.model_version = bundle.get("model_version")
            self.model_type = bundle.get("model_type", "unknown")
            self.model_timestamp = bundle.get("model_timestamp") or datetime.now()
            self._class_labels = bundle.get("class_labels")
            self._active_symbol = underlying
            self._prediction_cache.clear()
            return self.model is not None
        except Exception as e:
            logger.error(f"Failed to activate symbol model for {underlying}: {e}")
            return False

    def _activate_global_model(self) -> bool:
        """Switch active model bundle to global model."""
        try:
            if not self._global_model_bundle:
                if not self.load_model():
                    return False

            bundle = self._global_model_bundle or {}
            self.model = bundle.get("model")
            self.scaler = bundle.get("scaler")
            self.feature_names = bundle.get("feature_names", [])
            self.model_version = bundle.get("model_version")
            self.model_type = bundle.get("model_type")
            self.model_timestamp = bundle.get("model_timestamp")
            self._class_labels = bundle.get("class_labels")
            self._active_symbol = None
            self._prediction_cache.clear()
            return self.model is not None
        except Exception as e:
            logger.error(f"Failed to activate global model: {e}")
            return False

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            parsed = float(value)
            if not np.isfinite(parsed):
                return default
            return parsed
        except Exception:
            return default

    def _event_movement_score(self, features: Dict[str, float]) -> Tuple[float, List[str]]:
        """Compute event movement score from feature proxies."""
        score = 0.0
        reasons: List[str] = []

        thresholds = self.event_thresholds
        weights = self.event_weights

        gap = abs(self._safe_float(features.get("gap_percent", 0.0)))
        if gap >= float(thresholds.get("abs_gap_percent", 1.0)):
            score += float(weights.get("abs_gap_percent", 1.0))
            reasons.append(f"gap={gap:.2f}")

        atr_pct = abs(self._safe_float(features.get("atr_percent", 0.0)))
        if atr_pct >= float(thresholds.get("atr_percent", 2.5)):
            score += float(weights.get("atr_percent", 1.0))
            reasons.append(f"atr%={atr_pct:.2f}")

        intraday_range = abs(self._safe_float(features.get("intraday_range_percent", 0.0)))
        if intraday_range >= float(thresholds.get("intraday_range_percent", 1.8)):
            score += float(weights.get("intraday_range_percent", 1.0))
            reasons.append(f"range%={intraday_range:.2f}")

        ret1d = abs(self._safe_float(features.get("return_1d", 0.0)))
        if ret1d >= float(thresholds.get("abs_return_1d", 1.5)):
            score += float(weights.get("abs_return_1d", 1.0))
            reasons.append(f"ret1d={ret1d:.2f}")

        pcr = self._safe_float(features.get("pcr", 1.0), default=1.0)
        if pcr >= float(thresholds.get("pcr_spike", 1.35)):
            score += float(weights.get("pcr_spike", 1.0))
            reasons.append(f"pcr={pcr:.2f}")

        return score, reasons

    def _should_use_local_default(self, symbol: str) -> bool:
        if not self.prefer_symbol_models:
            return False
        if not symbol:
            return False
        symbol = symbol.upper()

        if self.hybrid_enabled:
            if symbol in self.force_global_symbols:
                return False
            if self.local_symbol_allowlist:
                return symbol in self.local_symbol_allowlist

        return True

    def _route_model_for_prediction(
        self,
        underlying: Optional[str],
        features: Dict[str, float],
    ) -> Dict[str, Any]:
        """Choose between local/global model and optional event override."""
        route = {
            "source": "global",
            "event_override": False,
            "event_score": 0.0,
            "event_reasons": [],
            "fallback_to_global": False,
        }

        symbol = (underlying or "").upper()

        # Ensure global is ready for fallback paths.
        if self.model is None and not self.load_model():
            return route
        if self._global_model_bundle is None:
            self._global_model_bundle = {
                "model": self.model,
                "scaler": self.scaler,
                "feature_names": list(self.feature_names),
                "model_version": self.model_version,
                "model_type": self.model_type,
                "model_timestamp": self.model_timestamp,
                "class_labels": self._class_labels,
            }

        if not symbol:
            self._activate_global_model()
            return route

        if self._should_use_local_default(symbol):
            if self._load_symbol_model_for_underlying(symbol):
                route["source"] = "local"
                return route
            self._activate_global_model()
            return route

        # Default global path for force-global / non-allowlist symbols.
        self._activate_global_model()

        if not self.hybrid_enabled or not self.event_override_enabled:
            return route

        if self.event_override_symbols and symbol not in self.event_override_symbols:
            return route

        score, reasons = self._event_movement_score(features)
        route["event_score"] = score
        route["event_reasons"] = reasons

        if score < self.event_override_min_score:
            return route

        if self._load_symbol_model_for_underlying(symbol):
            route["source"] = "local"
            route["event_override"] = True
            route["fallback_to_global"] = True
            logger.info(
                f"Hybrid route event override for {symbol}: score={score:.2f}, reasons={reasons}"
            )

        return route

    def _load_symbol_model_for_underlying(self, underlying: str) -> bool:
        if not underlying:
            return False

        if self._active_symbol == underlying and self.model is not None:
            return True

        cached = self._symbol_model_cache.get(underlying)
        if cached:
            return self._activate_model_bundle(underlying, cached)

        model_root = Path(ML_CONFIG.get("model_path", "data/ml_models"))
        if not model_root.exists():
            return False

        model_files = sorted(
            model_root.glob(f"{underlying}_model_*.joblib"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

        if not model_files:
            return False

        latest = model_files[0]

        try:
            artifact = joblib.load(latest)

            if isinstance(artifact, dict) and "models" in artifact:
                model_dict = dict(artifact.get("models", {}))
                default_weight = 1.0 / max(1, len(model_dict))
                weights = artifact.get("metrics", {}).get("optimized_weights", {})
                model_dict["weights"] = {
                    name: float(weights.get(name, default_weight))
                    for name in model_dict.keys()
                }

                class_labels = artifact.get("ensemble_classes")
                bundle = {
                    "model": model_dict,
                    "scaler": artifact.get("scaler"),
                    "feature_names": artifact.get("feature_names", []),
                    "model_version": latest.stem,
                    "model_type": "symbol_ensemble",
                    "model_timestamp": datetime.fromisoformat(artifact.get("timestamp")) if artifact.get("timestamp") else datetime.fromtimestamp(latest.stat().st_mtime),
                    "class_labels": np.array(class_labels) if class_labels is not None else None,
                }

            elif isinstance(artifact, dict) and "model" in artifact:
                class_labels = artifact.get("class_labels")
                bundle = {
                    "model": artifact.get("model"),
                    "scaler": artifact.get("scaler"),
                    "feature_names": artifact.get("feature_names", []),
                    "model_version": latest.stem,
                    "model_type": "symbol_single",
                    "model_timestamp": datetime.fromtimestamp(latest.stat().st_mtime),
                    "class_labels": np.array(class_labels) if class_labels is not None else None,
                }
            else:
                return False

            self._symbol_model_cache[underlying] = bundle
            loaded = self._activate_model_bundle(underlying, bundle)
            if loaded:
                logger.info(f"Loaded symbol model for {underlying}: {latest.name}")
            return loaded

        except Exception as e:
            logger.warning(f"Failed to load symbol model for {underlying}: {e}")
            return False
    
    def _load_feedback_config(self, model_version: str):
        """Load feedback configuration for confidence adjustment."""
        try:
            import json
            from pathlib import Path
            
            config_path = Path("data/ml_models") / model_version / "feedback_config.json"
            if config_path.exists():
                with open(config_path) as f:
                    self.feedback_config = json.load(f)
                
                self.confidence_adjustment = self.feedback_config.get("confidence_adjustment", 1.0)
                logger.info(f"Loaded feedback config: confidence adjustment = {self.confidence_adjustment:.2f}")
            else:
                self.confidence_adjustment = 1.0
                self.feedback_config = {}
                
        except Exception as e:
            logger.warning(f"Could not load feedback config: {e}")
            self.confidence_adjustment = 1.0
    
    def predict(
        self,
        features,
        underlying: str = None,
        use_cache: bool = True
    ) -> Optional[MLPrediction]:
        """
        Make a prediction from features.
        
        Args:
            features: Dictionary of feature values or FeatureSet object
            underlying: Optional underlying symbol for caching
            use_cache: Whether to use cached predictions
            
        Returns:
            MLPrediction or None if prediction fails
        """
        if not self.enabled:
            return None

        # Check if model is loaded
        if self.model is None and not self.load_model():
            return None
        
        # Convert FeatureSet to dict if needed
        if hasattr(features, 'to_dict'):
            features = features.to_dict()
        elif hasattr(features, 'features'):
            features = features.features

        route_info = self._route_model_for_prediction(underlying, features)
        
        # Check cache
        cache_key = (
            f"{self.model_version}_{underlying}_{route_info.get('source')}_"
            f"{int(route_info.get('event_override', False))}_{hash(frozenset(features.items()))}"
        )
        if use_cache and cache_key in self._prediction_cache:
            cached_pred, cached_time = self._prediction_cache[cache_key]
            if (datetime.now() - cached_time).total_seconds() < self.cache_ttl:
                return cached_pred
        
        try:
            # Convert features to array in correct order
            X = self._features_to_array(features)
            
            # Scale features
            if self.scaler is not None:
                X = self.scaler.transform(X.reshape(1, -1))
            else:
                X = X.reshape(1, -1)
            
            # Make prediction
            if isinstance(self.model, dict):
                prediction = self._ensemble_predict(X, underlying)
            else:
                prediction = self._single_model_predict(X, underlying)

            # Optional confidence floor for event-override local model.
            if (
                route_info.get("event_override")
                and route_info.get("fallback_to_global")
                and prediction.confidence < self.event_override_min_local_confidence
            ):
                if self._activate_global_model():
                    X_global = self._features_to_array(features)
                    if self.scaler is not None:
                        X_global = self.scaler.transform(X_global.reshape(1, -1))
                    else:
                        X_global = X_global.reshape(1, -1)

                    if isinstance(self.model, dict):
                        prediction = self._ensemble_predict(X_global, underlying)
                    else:
                        prediction = self._single_model_predict(X_global, underlying)

                    route_info["source"] = "global"
                    route_info["event_override"] = False
                    logger.info(
                        f"Hybrid route fallback to global for {underlying}: "
                        f"local_conf={prediction.confidence:.2f}"
                    )

            if route_info.get("event_override"):
                prediction.confidence = max(
                    0.1,
                    prediction.confidence - self.event_override_confidence_penalty,
                )

            prediction.route_source = route_info.get("source")
            prediction.event_override = bool(route_info.get("event_override"))
            prediction.event_score = float(route_info.get("event_score", 0.0))
            prediction.event_reasons = list(route_info.get("event_reasons", []))
            
            # Cache prediction
            if use_cache:
                self._prediction_cache[cache_key] = (prediction, datetime.now())
            
            return prediction
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None
    
    def _features_to_array(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dictionary to numpy array in correct order."""
        required = set(self.feature_names)
        provided = set(features.keys())

        overlap = len(required & provided) / max(1, len(required))
        missing_count = max(0, len(required) - len(required & provided))

        if overlap < self.warn_feature_overlap_ratio:
            logger.warning(
                f"Low feature overlap for model {self.model_version}: {overlap:.1%} "
                f"({len(required & provided)}/{len(required)})"
            )

        if self.feature_schema_strict:
            if overlap < self.min_feature_overlap_ratio:
                raise ValueError(
                    f"Feature overlap too low ({overlap:.1%} < {self.min_feature_overlap_ratio:.1%})"
                )
            if missing_count > self.max_missing_features:
                raise ValueError(
                    f"Missing features too high ({missing_count} > {self.max_missing_features})"
                )

        vector = []
        for name in self.feature_names:
            value = features.get(name, 0.0)
            try:
                value = float(value)
            except Exception:
                value = 0.0

            if not np.isfinite(value):
                value = 0.0

            vector.append(value)

        return np.array(vector, dtype=float)

    def _direction_bucket_for_class(self, class_label: Any, all_classes: np.ndarray) -> str:
        if isinstance(class_label, str):
            normalized = class_label.strip().upper()
            if normalized in {"BEARISH", "DOWN", "SELL", "PUT", "PE"}:
                return "BEARISH"
            if normalized in {"BULLISH", "UP", "BUY", "CALL", "CE"}:
                return "BULLISH"
            if normalized in {"NEUTRAL", "SIDEWAYS"}:
                return "NEUTRAL"

        numeric_classes = []
        for item in all_classes:
            try:
                numeric_classes.append(int(item))
            except Exception:
                continue

        try:
            value = int(class_label)
        except Exception:
            return "NEUTRAL"

        if len(set(numeric_classes)) == 2:
            lo, hi = sorted(set(numeric_classes))
            if value == lo:
                return "BEARISH"
            if value == hi:
                return "BULLISH"

        if value <= 0:
            return "BEARISH"
        if value >= 2:
            return "BULLISH"
        return "NEUTRAL"

    def _map_probabilities_to_direction_space(self, probs: np.ndarray, classes: np.ndarray) -> np.ndarray:
        mapped = np.zeros(3, dtype=float)

        for prob, class_label in zip(probs, classes):
            if not np.isfinite(prob):
                continue

            bucket = self._direction_bucket_for_class(class_label, classes)
            if bucket == "BEARISH":
                mapped[0] += float(prob)
            elif bucket == "BULLISH":
                mapped[2] += float(prob)
            else:
                mapped[1] += float(prob)

        total = float(mapped.sum())
        if total <= 0:
            mapped[1] = 1.0
            return mapped

        return mapped / total

    def _extract_model_probabilities(self, model: Any, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        classes = None

        if hasattr(model, "_label_encoder") and hasattr(model._label_encoder, "classes_"):
            classes = np.array(model._label_encoder.classes_)
        elif hasattr(model, "classes_"):
            classes = np.array(model.classes_)
        elif self._class_labels is not None and len(self._class_labels) > 0:
            classes = np.array(self._class_labels)

        if hasattr(model, "predict_proba"):
            probs = np.array(model.predict_proba(X)[0], dtype=float)
            if classes is None or len(classes) != len(probs):
                classes = np.arange(len(probs))
            return probs, classes

        pred = model.predict(X)[0]

        if classes is None or len(classes) == 0:
            classes = np.array([pred])

        probs = np.zeros(len(classes), dtype=float)
        try:
            pred_idx = int(np.where(classes == pred)[0][0])
        except Exception:
            pred_idx = 0
        probs[pred_idx] = 1.0
        return probs, classes

    def _get_abstain_margin(self, underlying: Optional[str]) -> float:
        if underlying and underlying in self.abstain_margin_by_symbol:
            return float(self.abstain_margin_by_symbol.get(underlying, self.default_abstain_margin))
        return float(self.default_abstain_margin)

    def _apply_abstain_rule(
        self,
        mapped_probs: np.ndarray,
        underlying: Optional[str] = None,
    ) -> Tuple[int, str, float, bool]:
        sorted_idx = np.argsort(mapped_probs)[::-1]
        top_idx = int(sorted_idx[0])
        top_prob = float(mapped_probs[top_idx])
        second_prob = float(mapped_probs[int(sorted_idx[1])]) if len(sorted_idx) > 1 else 0.0

        margin = top_prob - second_prob
        abstain_margin = self._get_abstain_margin(underlying)

        should_abstain = top_prob < self.min_top_probability or margin < abstain_margin
        if should_abstain:
            logger.info(
                f"Abstain triggered for {underlying or 'GLOBAL'}: top={top_prob:.2f}, "
                f"margin={margin:.2f}, min_top={self.min_top_probability:.2f}, min_margin={abstain_margin:.2f}"
            )
            return 1, "NEUTRAL", top_prob, True

        direction = self.DIRECTION_MAP.get(top_idx, "NEUTRAL")
        return top_idx, direction, top_prob, False
    
    def _single_model_predict(self, X: np.ndarray, underlying: Optional[str] = None) -> MLPrediction:
        """Make prediction with single model."""
        probs, classes = self._extract_model_probabilities(self.model, X)
        mapped_probs = self._map_probabilities_to_direction_space(probs, classes)
        
        raw_pred, direction, confidence, abstained = self._apply_abstain_rule(mapped_probs, underlying)
        
        # Apply feedback-based confidence adjustment
        adjusted_confidence = confidence * self.confidence_adjustment
        # Clamp to valid range
        adjusted_confidence = max(0.1, min(0.95, adjusted_confidence))
        
        # Create probability dict
        prob_dict = {
            "BEARISH": float(mapped_probs[0]),
            "NEUTRAL": float(mapped_probs[1]),
            "BULLISH": float(mapped_probs[2]),
        }
        
        prediction = MLPrediction(
            direction=direction,
            confidence=adjusted_confidence,
            probabilities=prob_dict,
            model_version=self.model_version,
            model_type=self.model_type,
            timestamp=datetime.now(),
            raw_prediction=raw_pred,
        )
        prediction.abstained = abstained
        return prediction
    
    def _ensemble_predict(self, X: np.ndarray, underlying: Optional[str] = None) -> MLPrediction:
        """Make prediction with ensemble of models."""
        weights = self.model.get("weights", {})
        ensemble_probs = np.zeros(3)
        total_weight = 0
        
        for name, model in self.model.items():
            if name in ["weights", "scaler", "class_labels"]:
                continue
            
            weight = weights.get(name, 0.33)

            try:
                probs, classes = self._extract_model_probabilities(model, X)
                mapped_probs = self._map_probabilities_to_direction_space(probs, classes)
                ensemble_probs += weight * mapped_probs
                total_weight += weight
            except Exception as e:
                logger.warning(f"Ensemble sub-model {name} failed: {e}")
        
        if total_weight > 0:
            ensemble_probs /= total_weight
        
        raw_pred, direction, confidence, abstained = self._apply_abstain_rule(ensemble_probs, underlying)
        
        # Apply feedback-based confidence adjustment
        adjusted_confidence = confidence * self.confidence_adjustment
        # Clamp to valid range
        adjusted_confidence = max(0.1, min(0.95, adjusted_confidence))
        
        prob_dict = {
            "BEARISH": float(ensemble_probs[0]),
            "NEUTRAL": float(ensemble_probs[1]),
            "BULLISH": float(ensemble_probs[2]),
        }
        
        prediction = MLPrediction(
            direction=direction,
            confidence=adjusted_confidence,
            probabilities=prob_dict,
            model_version=self.model_version,
            model_type=self.model_type or "ensemble",
            timestamp=datetime.now(),
            raw_prediction=raw_pred,
        )
        prediction.abstained = abstained
        return prediction
    
    def predict_with_guardrails(
        self,
        features: Dict[str, float],
        rule_confidence: float,
        underlying: str,
        current_positions: int = 0,
        is_paper_mode: bool = False
    ) -> Tuple[Optional[MLPrediction], float, bool]:
        """
        Make prediction with guardrails applied.
        
        Args:
            features: Feature dictionary
            rule_confidence: Rule-based confidence score
            underlying: Underlying symbol
            current_positions: Number of current open positions
            is_paper_mode: Whether in paper trading mode
            
        Returns:
            Tuple of (prediction, blended_confidence, should_trade)
        """
        # Get ML prediction
        prediction = self.predict(features, underlying)
        
        if prediction is None:
            # ML unavailable, use rule-based only
            return None, rule_confidence, rule_confidence >= ML_CONFIG.get("min_confidence_for_trade", 0.55)
        
        # Check guardrails
        guardrail_result = self.guardrails.check_entry_signal(
            ml_confidence=prediction.confidence,
            rule_confidence=rule_confidence,
            current_positions=current_positions,
            model_timestamp=self.model_timestamp,
            is_paper_mode=is_paper_mode
        )
        
        if not guardrail_result.passed:
            logger.info(f"Guardrail blocked: {guardrail_result.message}")
            return prediction, rule_confidence, False
        
        # Blend confidence
        blended, was_adjusted = self.guardrails.blend_confidence(
            ml_confidence=prediction.confidence,
            rule_confidence=rule_confidence,
            ml_weight=self.confidence_weight
        )
        
        if was_adjusted:
            logger.debug(f"Confidence adjusted: ML={prediction.confidence:.2f}, "
                        f"Rule={rule_confidence:.2f}, Blended={blended:.2f}")
        
        # Check minimum confidence threshold
        min_confidence = ML_CONFIG.get("min_confidence_for_trade", 0.55)
        should_trade = blended >= min_confidence
        
        return prediction, blended, should_trade
    
    def predict_exit(
        self,
        features: Dict[str, float],
        current_pnl_percent: float,
        stop_loss_percent: float,
        target_percent: float,
        time_in_trade_hours: float
    ) -> Tuple[str, float, str]:
        """
        Predict exit recommendation.
        
        Args:
            features: Current market features
            current_pnl_percent: Current P&L percentage
            stop_loss_percent: Stop loss threshold
            target_percent: Target profit threshold
            time_in_trade_hours: Hours in trade
            
        Returns:
            Tuple of (recommendation, confidence, reason)
            recommendation: 'HOLD', 'EXIT', 'TRAIL'
        """
        # Get direction prediction
        prediction = self.predict(features)
        
        if prediction is None:
            return "HOLD", 0.5, "ML unavailable"
        
        # Basic exit logic based on prediction
        recommendation = "HOLD"
        confidence = prediction.confidence
        reason = ""
        
        # Check if prediction direction opposes current position
        # This is a simplified heuristic - can be enhanced with a dedicated exit model
        
        if prediction.direction == "BEARISH" and current_pnl_percent > 0:
            # Trend may be reversing, consider taking profit
            if prediction.confidence > 0.7:
                recommendation = "EXIT"
                reason = "ML predicts bearish reversal with high confidence"
            elif prediction.confidence > 0.5:
                recommendation = "TRAIL"
                reason = "ML predicts possible bearish move, tighten stops"
        
        elif prediction.direction == "BULLISH" and current_pnl_percent > 0:
            # Trend continuing, hold or trail
            recommendation = "TRAIL" if current_pnl_percent > target_percent * 0.5 else "HOLD"
            reason = "ML predicts bullish continuation"
        
        elif prediction.direction == "NEUTRAL":
            # Sideways expected
            if current_pnl_percent > target_percent * 0.3:
                recommendation = "TRAIL"
                reason = "ML predicts sideways, lock in partial profits"
            elif time_in_trade_hours > 4:
                recommendation = "EXIT"
                reason = "ML predicts sideways, time decay risk"
        
        # Time-based adjustments
        if time_in_trade_hours > 24 and prediction.direction != "BULLISH":
            recommendation = "EXIT"
            reason = "Extended time in trade with non-bullish outlook"
        
        return recommendation, confidence, reason
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "version": self.model_version,
            "type": self.model_type,
            "trained_at": self.model_timestamp.isoformat() if self.model_timestamp else None,
            "feature_count": len(self.feature_names),
            "cache_size": len(self._prediction_cache),
        }
    
    def clear_cache(self) -> int:
        """Clear prediction cache. Returns number of cleared entries."""
        count = len(self._prediction_cache)
        self._prediction_cache.clear()
        return count
    
    def is_model_stale(self) -> bool:
        """Check if the loaded model is stale (too old)."""
        if self.model_timestamp is None:
            return True
        
        max_age_days = ML_CONFIG.get("guardrails", {}).get("max_model_age_days", 14)
        age = datetime.now() - self.model_timestamp
        
        return age > timedelta(days=max_age_days)
    
    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get prediction statistics from cache."""
        if not self._prediction_cache:
            return {"count": 0}
        
        predictions = [p for p, _ in self._prediction_cache.values()]
        
        directions = [p.direction for p in predictions]
        confidences = [p.confidence for p in predictions]
        
        return {
            "count": len(predictions),
            "direction_counts": {
                "BULLISH": directions.count("BULLISH"),
                "BEARISH": directions.count("BEARISH"),
                "NEUTRAL": directions.count("NEUTRAL"),
            },
            "avg_confidence": np.mean(confidences) if confidences else 0,
            "min_confidence": min(confidences) if confidences else 0,
            "max_confidence": max(confidences) if confidences else 0,
        }


# Singleton instance
_predictor: Optional[MLPredictor] = None


def get_predictor() -> MLPredictor:
    """Get or create the singleton predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = MLPredictor()
    return _predictor
