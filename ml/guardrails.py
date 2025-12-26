"""
Risk Management Guardrails for ML-Powered Trading

Ensures ML predictions cannot override critical risk management rules.
All guardrails are designed to protect capital first.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from enum import Enum

from config.settings import TRADING_CONFIG, ML_CONFIG
from core.logger import logger


class GuardrailViolation(Enum):
    """Types of guardrail violations."""
    NONE = "none"
    LOW_ML_CONFIDENCE = "low_ml_confidence"
    CONFIDENCE_ADJUSTMENT_EXCEEDED = "confidence_adjustment_exceeded"
    STOP_LOSS_OVERRIDE_BLOCKED = "stop_loss_override_blocked"
    MAX_POSITIONS_EXCEEDED = "max_positions_exceeded"
    DAILY_LOSS_EXCEEDED = "daily_loss_exceeded"
    MODEL_STALE = "model_stale"
    DRAWDOWN_CIRCUIT_BREAKER = "drawdown_circuit_breaker"
    POSITION_SIZE_CHANGE_BLOCKED = "position_size_change_blocked"


@dataclass
class GuardrailCheckResult:
    """Result of a guardrail check."""
    passed: bool
    violation: GuardrailViolation
    original_value: float
    adjusted_value: float
    message: str
    metadata: Dict[str, Any]


class TradingGuardrails:
    """
    Risk management guardrails for ML trading.
    
    Core principles:
    1. Stop-loss is sacred - ML cannot delay/override stop-loss exits
    2. Confidence bounds - ML adjustment capped at ±0.3
    3. Min ML confidence - Block trade if ML confidence < 0.4
    4. Position sizing unchanged - ML cannot change quantity/lot size
    5. Max positions respected - ML cannot exceed max_positions config
    6. Drawdown circuit breaker - Pause ML if daily loss > threshold
    7. Model freshness - Use rule-based if model > 14 days old
    """
    
    def __init__(self):
        self.config = ML_CONFIG.get("guardrails", {})
        
        # Guardrail parameters
        self.max_confidence_adjustment = self.config.get("max_confidence_adjustment", 0.3)
        self.min_ml_confidence = self.config.get("min_ml_confidence", 0.4)
        self.max_model_age_days = self.config.get("max_model_age_days", 14)
        self.daily_loss_threshold_percent = self.config.get("daily_loss_threshold_percent", 5.0)
        self.drawdown_circuit_breaker_percent = self.config.get("drawdown_circuit_breaker_percent", 10.0)
        
        # State tracking
        self.daily_pnl = 0.0
        self.daily_pnl_date = datetime.now().date()
        self.circuit_breaker_active = False
        self.violations_today: list = []
        
        logger.info("Trading guardrails initialized")
    
    def check_entry_signal(
        self,
        ml_confidence: float,
        rule_confidence: float,
        current_positions: int,
        model_timestamp: Optional[datetime] = None,
        is_paper_mode: bool = False
    ) -> GuardrailCheckResult:
        """
        Check if an entry signal passes all guardrails.
        
        Args:
            ml_confidence: ML model's confidence score (0-1)
            rule_confidence: Rule-based confidence score (0-1)
            current_positions: Number of currently open positions
            model_timestamp: When the model was last trained
            is_paper_mode: Whether in paper trading mode
            
        Returns:
            GuardrailCheckResult with pass/fail and adjusted confidence
        """
        max_positions = (
            TRADING_CONFIG.get("paper_max_positions", 15) if is_paper_mode 
            else TRADING_CONFIG.get("max_positions", 5)
        )
        
        # Check 1: Circuit breaker
        if self.circuit_breaker_active and not is_paper_mode:
            return GuardrailCheckResult(
                passed=False,
                violation=GuardrailViolation.DRAWDOWN_CIRCUIT_BREAKER,
                original_value=ml_confidence,
                adjusted_value=0.0,
                message="Circuit breaker active - ML trading paused due to excessive losses",
                metadata={"daily_pnl": self.daily_pnl}
            )
        
        # Check 2: Max positions
        if current_positions >= max_positions:
            return GuardrailCheckResult(
                passed=False,
                violation=GuardrailViolation.MAX_POSITIONS_EXCEEDED,
                original_value=ml_confidence,
                adjusted_value=0.0,
                message=f"Max positions ({max_positions}) reached",
                metadata={"current_positions": current_positions}
            )
        
        # Check 3: Model staleness
        if model_timestamp:
            model_age = datetime.now() - model_timestamp
            if model_age > timedelta(days=self.max_model_age_days):
                logger.warning(f"ML model is {model_age.days} days old, using rule-based only")
                return GuardrailCheckResult(
                    passed=True,
                    violation=GuardrailViolation.MODEL_STALE,
                    original_value=ml_confidence,
                    adjusted_value=rule_confidence,  # Use rule-based confidence
                    message=f"Model is {model_age.days} days old, falling back to rules",
                    metadata={"model_age_days": model_age.days}
                )
        
        # Check 4: Minimum ML confidence
        if ml_confidence < self.min_ml_confidence:
            return GuardrailCheckResult(
                passed=False,
                violation=GuardrailViolation.LOW_ML_CONFIDENCE,
                original_value=ml_confidence,
                adjusted_value=0.0,
                message=f"ML confidence {ml_confidence:.2f} below minimum {self.min_ml_confidence}",
                metadata={"threshold": self.min_ml_confidence}
            )
        
        # Check 5: Confidence adjustment bounds
        confidence_delta = ml_confidence - rule_confidence
        if abs(confidence_delta) > self.max_confidence_adjustment:
            # Cap the adjustment
            if confidence_delta > 0:
                adjusted = rule_confidence + self.max_confidence_adjustment
            else:
                adjusted = rule_confidence - self.max_confidence_adjustment
            
            return GuardrailCheckResult(
                passed=True,
                violation=GuardrailViolation.CONFIDENCE_ADJUSTMENT_EXCEEDED,
                original_value=ml_confidence,
                adjusted_value=adjusted,
                message=f"Confidence adjustment capped from {ml_confidence:.2f} to {adjusted:.2f}",
                metadata={
                    "original_delta": confidence_delta,
                    "max_delta": self.max_confidence_adjustment
                }
            )
        
        # All checks passed
        return GuardrailCheckResult(
            passed=True,
            violation=GuardrailViolation.NONE,
            original_value=ml_confidence,
            adjusted_value=ml_confidence,
            message="All guardrails passed",
            metadata={}
        )
    
    def check_exit_signal(
        self,
        ml_exit_recommendation: str,
        ml_exit_confidence: float,
        current_pnl_percent: float,
        stop_loss_percent: float,
        is_stop_loss_triggered: bool
    ) -> GuardrailCheckResult:
        """
        Check if an exit signal passes all guardrails.
        
        Critical rule: ML cannot delay a stop-loss exit.
        
        Args:
            ml_exit_recommendation: "HOLD", "EXIT", or "TRAIL"
            ml_exit_confidence: ML's confidence in the exit recommendation
            current_pnl_percent: Current P&L as percentage
            stop_loss_percent: Stop loss threshold percentage
            is_stop_loss_triggered: Whether rule-based stop-loss is triggered
            
        Returns:
            GuardrailCheckResult
        """
        # SACRED RULE: Stop-loss is never overridden
        if is_stop_loss_triggered:
            if ml_exit_recommendation == "HOLD":
                self._log_violation(GuardrailViolation.STOP_LOSS_OVERRIDE_BLOCKED)
                return GuardrailCheckResult(
                    passed=False,  # Block the ML recommendation
                    violation=GuardrailViolation.STOP_LOSS_OVERRIDE_BLOCKED,
                    original_value=ml_exit_confidence,
                    adjusted_value=1.0,  # Force exit
                    message="ML attempted to override stop-loss - BLOCKED",
                    metadata={
                        "current_pnl_percent": current_pnl_percent,
                        "stop_loss_percent": stop_loss_percent
                    }
                )
            else:
                # ML agrees with exit, proceed
                return GuardrailCheckResult(
                    passed=True,
                    violation=GuardrailViolation.NONE,
                    original_value=ml_exit_confidence,
                    adjusted_value=ml_exit_confidence,
                    message="Stop-loss triggered, ML agrees with exit",
                    metadata={}
                )
        
        # Non-stop-loss exits can be influenced by ML
        return GuardrailCheckResult(
            passed=True,
            violation=GuardrailViolation.NONE,
            original_value=ml_exit_confidence,
            adjusted_value=ml_exit_confidence,
            message="Exit signal passed guardrails",
            metadata={"recommendation": ml_exit_recommendation}
        )
    
    def check_position_sizing(
        self,
        ml_suggested_quantity: int,
        rule_quantity: int,
        lot_size: int
    ) -> GuardrailCheckResult:
        """
        Check position sizing. ML cannot change quantity.
        
        Args:
            ml_suggested_quantity: Quantity suggested by ML
            rule_quantity: Quantity from rule-based system
            lot_size: Contract lot size
            
        Returns:
            GuardrailCheckResult with original quantity preserved
        """
        if ml_suggested_quantity != rule_quantity:
            return GuardrailCheckResult(
                passed=True,  # Allow trade but with original quantity
                violation=GuardrailViolation.POSITION_SIZE_CHANGE_BLOCKED,
                original_value=float(ml_suggested_quantity),
                adjusted_value=float(rule_quantity),
                message=f"ML quantity {ml_suggested_quantity} overridden to {rule_quantity}",
                metadata={"lot_size": lot_size}
            )
        
        return GuardrailCheckResult(
            passed=True,
            violation=GuardrailViolation.NONE,
            original_value=float(rule_quantity),
            adjusted_value=float(rule_quantity),
            message="Position sizing within limits",
            metadata={}
        )
    
    def update_daily_pnl(self, pnl: float) -> None:
        """
        Update daily P&L and check circuit breaker.
        
        Args:
            pnl: P&L to add (positive or negative)
        """
        today = datetime.now().date()
        
        # Reset if new day
        if today != self.daily_pnl_date:
            self.daily_pnl = 0.0
            self.daily_pnl_date = today
            self.circuit_breaker_active = False
            self.violations_today = []
        
        self.daily_pnl += pnl
        
        # Check circuit breaker
        max_loss = TRADING_CONFIG.get("max_loss_per_day", 10000)
        loss_threshold = max_loss * (self.daily_loss_threshold_percent / 100)
        
        if self.daily_pnl < -loss_threshold:
            self.circuit_breaker_active = True
            logger.warning(
                f"Circuit breaker ACTIVATED: Daily P&L {self.daily_pnl:.2f} "
                f"exceeds threshold {-loss_threshold:.2f}"
            )
    
    def reset_circuit_breaker(self) -> None:
        """Manually reset the circuit breaker (admin action)."""
        self.circuit_breaker_active = False
        logger.info("Circuit breaker manually reset")
    
    def _log_violation(self, violation: GuardrailViolation) -> None:
        """Log a guardrail violation."""
        self.violations_today.append({
            "violation": violation.value,
            "timestamp": datetime.now().isoformat()
        })
        logger.warning(f"Guardrail violation: {violation.value}")
    
    def get_daily_stats(self) -> Dict[str, Any]:
        """Get daily guardrail statistics."""
        return {
            "date": self.daily_pnl_date.isoformat(),
            "daily_pnl": self.daily_pnl,
            "circuit_breaker_active": self.circuit_breaker_active,
            "violations_count": len(self.violations_today),
            "violations": self.violations_today
        }
    
    def blend_confidence(
        self,
        ml_confidence: float,
        rule_confidence: float,
        ml_weight: float = None
    ) -> Tuple[float, bool]:
        """
        Blend ML and rule-based confidence scores with guardrails.
        
        Args:
            ml_confidence: ML model confidence (0-1)
            rule_confidence: Rule-based confidence (0-1)
            ml_weight: Weight for ML (default from config)
            
        Returns:
            Tuple of (blended_confidence, was_adjusted)
        """
        if ml_weight is None:
            ml_weight = ML_CONFIG.get("confidence_weight", 0.5)
        
        # Calculate blended confidence
        blended = (ml_confidence * ml_weight) + (rule_confidence * (1 - ml_weight))
        
        # Apply bounds
        confidence_delta = blended - rule_confidence
        was_adjusted = False
        
        if abs(confidence_delta) > self.max_confidence_adjustment:
            if confidence_delta > 0:
                blended = rule_confidence + self.max_confidence_adjustment
            else:
                blended = rule_confidence - self.max_confidence_adjustment
            was_adjusted = True
        
        # Ensure bounds [0, 1]
        blended = max(0.0, min(1.0, blended))
        
        return blended, was_adjusted


# Singleton instance
_guardrails: Optional[TradingGuardrails] = None


def get_guardrails() -> TradingGuardrails:
    """Get or create the singleton guardrails instance."""
    global _guardrails
    if _guardrails is None:
        _guardrails = TradingGuardrails()
    return _guardrails
