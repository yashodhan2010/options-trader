"""
Paper Trading Runner for ML Model Testing

Orchestrates paper trading with ML integration, tracks performance,
and manages continuous feedback loop.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import json
import asyncio

from config.settings import ML_CONFIG
from core.database import database
from core.logger import logger


@dataclass
class PaperTrade:
    """Represents a paper trade."""
    id: str
    underlying: str
    strategy_type: str
    direction: str
    entry_time: datetime
    entry_price: float
    quantity: int
    target_price: float
    stop_loss_price: float
    ml_confidence: float
    rule_confidence: float
    blended_confidence: float
    model_version: str
    
    # Exit fields
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: float = 0.0
    pnl_percent: float = 0.0
    status: str = "OPEN"
    
    # Feature tracking
    entry_features: Dict[str, float] = field(default_factory=dict)
    exit_features: Dict[str, float] = field(default_factory=dict)


@dataclass
class PaperTradingSession:
    """Paper trading session state."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    model_version: str = ""
    initial_capital: float = 100000.0
    current_capital: float = 100000.0
    
    # Statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    peak_capital: float = 100000.0
    
    # Performance
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    
    # Daily tracking
    daily_pnl: Dict[str, float] = field(default_factory=dict)
    
    # Active trades
    active_trades: List[PaperTrade] = field(default_factory=list)
    closed_trades: List[PaperTrade] = field(default_factory=list)


class PaperTradingRunner:
    """
    Run paper trading with ML integration.
    
    Features:
    - Simulate trades without real capital
    - Track ML model performance in live conditions
    - Manage position sizing and risk
    - Daily/weekly performance reports
    - Automatic alerts for poor performance
    """
    
    def __init__(self):
        """Initialize paper trading runner."""
        self.config = ML_CONFIG.get("paper_trading", {})
        self.initial_capital = self.config.get("initial_capital", 100000)
        self.max_position_pct = self.config.get("max_position_pct", 0.1)
        self.max_open_trades = self.config.get("max_open_trades", 5)
        
        # Alert thresholds
        self.drawdown_alert_pct = self.config.get("drawdown_alert_pct", 0.10)
        self.win_rate_alert = self.config.get("win_rate_alert", 0.40)
        
        # Session
        self.session: Optional[PaperTradingSession] = None
        
        # Components (lazy load)
        self._predictor = None
        self._feature_engineer = None
        self._feedback_collector = None
        self._guardrails = None
        
        logger.info("PaperTradingRunner initialized")
    
    @property
    def predictor(self):
        """Lazy load predictor."""
        if self._predictor is None:
            from ml.predictor import get_predictor
            self._predictor = get_predictor()
        return self._predictor
    
    @property
    def feature_engineer(self):
        """Lazy load feature engineer."""
        if self._feature_engineer is None:
            from ml.feature_engineer import get_feature_engineer
            self._feature_engineer = get_feature_engineer()
        return self._feature_engineer
    
    @property
    def feedback_collector(self):
        """Lazy load feedback collector."""
        if self._feedback_collector is None:
            from ml.feedback_collector import get_feedback_collector
            self._feedback_collector = get_feedback_collector()
        return self._feedback_collector
    
    @property
    def guardrails(self):
        """Lazy load guardrails."""
        if self._guardrails is None:
            from ml.guardrails import get_guardrails
            self._guardrails = get_guardrails()
        return self._guardrails
    
    def start_session(
        self,
        model_version: str = None,
        initial_capital: float = None
    ) -> str:
        """
        Start a new paper trading session.
        
        Args:
            model_version: Optional model version override
            initial_capital: Optional initial capital override
            
        Returns:
            Session ID
        """
        session_id = f"paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        capital = initial_capital or self.initial_capital
        
        self.session = PaperTradingSession(
            session_id=session_id,
            start_time=datetime.now(),
            model_version=model_version or "",
            initial_capital=capital,
            current_capital=capital,
            peak_capital=capital,
        )
        
        logger.info(f"Paper trading session started: {session_id}")
        return session_id
    
    def end_session(self) -> Dict[str, Any]:
        """
        End current session and generate report.
        
        Returns:
            Session summary
        """
        if not self.session:
            return {"error": "No active session"}
        
        # Close all open trades at current market price
        for trade in self.session.active_trades:
            self._close_trade(
                trade,
                exit_price=trade.entry_price,  # Use entry as proxy
                exit_reason="SESSION_END"
            )
        
        self.session.end_time = datetime.now()
        
        summary = self._calculate_session_summary()
        
        logger.info(f"Paper trading session ended: {self.session.session_id}")
        
        return summary
    
    def process_signal(
        self,
        underlying: str,
        strategy_type: str,
        spot_price: float,
        market_data: Dict[str, Any],
        rule_confidence: float,
        target_price: float,
        stop_loss_price: float
    ) -> Dict[str, Any]:
        """
        Process a trading signal through ML pipeline.
        
        Args:
            underlying: Underlying symbol
            strategy_type: Strategy type
            spot_price: Current spot price
            market_data: Market data for features
            rule_confidence: Rule-based confidence
            target_price: Target price
            stop_loss_price: Stop loss price
            
        Returns:
            Dict with decision and trade details
        """
        if not self.session:
            self.start_session()
        
        result = {
            "action": "SKIP",
            "reason": "",
            "ml_confidence": 0.0,
            "blended_confidence": 0.0,
        }
        
        # Check max open trades
        if len(self.session.active_trades) >= self.max_open_trades:
            result["reason"] = "Max open trades reached"
            return result
        
        # Check position size limit
        position_value = self.session.current_capital * self.max_position_pct
        if position_value < 5000:  # Min position size
            result["reason"] = "Insufficient capital for position"
            return result
        
        try:
            # Extract features
            features = self.feature_engineer.extract_features(
                spot_price=spot_price,
                market_data=market_data,
                underlying=underlying,
                strategy_type=strategy_type
            )
            
            # Get ML prediction with guardrails
            prediction = self.predictor.predict_with_guardrails(
                features=features,
                underlying=underlying,
                strategy_type=strategy_type,
                rule_confidence=rule_confidence
            )
            
            result["ml_confidence"] = prediction.confidence
            result["blended_confidence"] = prediction.blended_confidence
            result["prediction"] = prediction.direction
            
            # Decision logic
            min_confidence = ML_CONFIG.get("min_blended_confidence", 0.55)
            
            if prediction.blended_confidence < min_confidence:
                result["action"] = "SKIP"
                result["reason"] = f"Confidence {prediction.blended_confidence:.2%} below threshold"
                return result
            
            if prediction.direction == "NEUTRAL":
                result["action"] = "SKIP"
                result["reason"] = "Neutral prediction"
                return result
            
            # Create paper trade
            trade_id = f"{self.session.session_id}_t{self.session.total_trades + 1}"
            quantity = int(position_value / spot_price)
            
            trade = PaperTrade(
                id=trade_id,
                underlying=underlying,
                strategy_type=strategy_type,
                direction=prediction.direction,
                entry_time=datetime.now(),
                entry_price=spot_price,
                quantity=quantity,
                target_price=target_price,
                stop_loss_price=stop_loss_price,
                ml_confidence=prediction.confidence,
                rule_confidence=rule_confidence,
                blended_confidence=prediction.blended_confidence,
                model_version=prediction.model_version,
                entry_features=features,
            )
            
            self.session.active_trades.append(trade)
            self.session.total_trades += 1
            
            # Log to feedback collector
            self.feedback_collector.log_entry_features(
                execution_id=trade_id,
                underlying=underlying,
                strategy_type=strategy_type,
                features=features,
                spot_price=spot_price
            )
            
            self.feedback_collector.log_prediction(
                execution_id=trade_id,
                underlying=underlying,
                strategy_type=strategy_type,
                model_version=prediction.model_version,
                model_type="ensemble",
                direction_prediction=prediction.direction,
                ml_confidence=prediction.confidence,
                rule_confidence=rule_confidence,
                blended_confidence=prediction.blended_confidence,
                top_features=prediction.feature_importance
            )
            
            result["action"] = "TRADE"
            result["trade_id"] = trade_id
            result["quantity"] = quantity
            result["reason"] = f"ML confidence: {prediction.confidence:.2%}"
            
            logger.info(f"Paper trade opened: {trade_id}, {prediction.direction} @ {spot_price}")
            
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
            result["action"] = "SKIP"
            result["reason"] = f"Error: {str(e)}"
        
        return result
    
    def update_positions(
        self,
        market_prices: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Update open positions with current prices.
        
        Args:
            market_prices: Dict of underlying -> current price
            
        Returns:
            List of closed trade summaries
        """
        if not self.session:
            return []
        
        closed = []
        
        for trade in self.session.active_trades[:]:  # Copy to allow removal
            underlying = trade.underlying
            current_price = market_prices.get(underlying)
            
            if current_price is None:
                continue
            
            # Check exit conditions
            exit_reason = None
            
            if trade.direction == "BULLISH":
                # Long position
                if current_price >= trade.target_price:
                    exit_reason = "TARGET_HIT"
                elif current_price <= trade.stop_loss_price:
                    exit_reason = "STOP_LOSS_HIT"
            else:
                # Short position (for bearish strategies)
                if current_price <= trade.target_price:
                    exit_reason = "TARGET_HIT"
                elif current_price >= trade.stop_loss_price:
                    exit_reason = "STOP_LOSS_HIT"
            
            if exit_reason:
                result = self._close_trade(trade, current_price, exit_reason)
                closed.append(result)
        
        # Check alerts
        self._check_alerts()
        
        return closed
    
    def _close_trade(
        self,
        trade: PaperTrade,
        exit_price: float,
        exit_reason: str
    ) -> Dict[str, Any]:
        """Close a paper trade."""
        trade.exit_time = datetime.now()
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        trade.status = "CLOSED"
        
        # Calculate P&L
        if trade.direction == "BULLISH":
            pnl_per_unit = exit_price - trade.entry_price
        else:
            pnl_per_unit = trade.entry_price - exit_price
        
        trade.pnl = pnl_per_unit * trade.quantity
        trade.pnl_percent = (pnl_per_unit / trade.entry_price) * 100
        
        # Update session
        self.session.current_capital += trade.pnl
        self.session.total_pnl += trade.pnl
        
        if trade.pnl > 0:
            self.session.winning_trades += 1
        else:
            self.session.losing_trades += 1
        
        # Update peak and drawdown
        if self.session.current_capital > self.session.peak_capital:
            self.session.peak_capital = self.session.current_capital
        
        current_drawdown = (self.session.peak_capital - self.session.current_capital) / self.session.peak_capital
        if current_drawdown > self.session.max_drawdown:
            self.session.max_drawdown = current_drawdown
        
        # Update daily P&L
        today = datetime.now().strftime("%Y-%m-%d")
        self.session.daily_pnl[today] = self.session.daily_pnl.get(today, 0) + trade.pnl
        
        # Move to closed trades
        if trade in self.session.active_trades:
            self.session.active_trades.remove(trade)
        self.session.closed_trades.append(trade)
        
        # Log to feedback collector
        trade_duration = int((trade.exit_time - trade.entry_time).total_seconds())
        self.feedback_collector.log_outcome(
            execution_id=trade.id,
            actual_pnl=trade.pnl,
            actual_pnl_percent=trade.pnl_percent,
            trade_duration_seconds=trade_duration
        )
        
        logger.info(f"Paper trade closed: {trade.id}, P&L: ₹{trade.pnl:.2f} ({trade.pnl_percent:.2%})")
        
        return {
            "trade_id": trade.id,
            "pnl": trade.pnl,
            "pnl_percent": trade.pnl_percent,
            "exit_reason": exit_reason,
        }
    
    def force_close_trade(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str = "MANUAL_CLOSE"
    ) -> Optional[Dict[str, Any]]:
        """Force close a trade by ID."""
        for trade in self.session.active_trades:
            if trade.id == trade_id:
                return self._close_trade(trade, exit_price, exit_reason)
        return None
    
    def _check_alerts(self) -> None:
        """Check for performance alerts."""
        if not self.session or self.session.total_trades < 5:
            return
        
        # Drawdown alert
        if self.session.max_drawdown > self.drawdown_alert_pct:
            logger.warning(
                f"⚠️ ALERT: Drawdown {self.session.max_drawdown:.1%} exceeds "
                f"threshold {self.drawdown_alert_pct:.1%}"
            )
        
        # Win rate alert
        self.session.win_rate = (
            self.session.winning_trades / self.session.total_trades
            if self.session.total_trades > 0 else 0
        )
        
        if self.session.win_rate < self.win_rate_alert:
            logger.warning(
                f"⚠️ ALERT: Win rate {self.session.win_rate:.1%} below "
                f"threshold {self.win_rate_alert:.1%}"
            )
    
    def _calculate_session_summary(self) -> Dict[str, Any]:
        """Calculate comprehensive session summary."""
        if not self.session:
            return {}
        
        s = self.session
        
        # Win/loss stats
        winning = [t for t in s.closed_trades if t.pnl > 0]
        losing = [t for t in s.closed_trades if t.pnl < 0]
        
        avg_win = sum(t.pnl for t in winning) / len(winning) if winning else 0
        avg_loss = sum(t.pnl for t in losing) / len(losing) if losing else 0
        
        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate Sharpe ratio from daily returns
        if len(s.daily_pnl) > 1:
            daily_returns = list(s.daily_pnl.values())
            import statistics
            try:
                mean_return = statistics.mean(daily_returns)
                std_return = statistics.stdev(daily_returns)
                sharpe = (mean_return / std_return) * (252 ** 0.5) if std_return > 0 else 0
            except:
                sharpe = 0
        else:
            sharpe = 0
        
        return {
            "session_id": s.session_id,
            "model_version": s.model_version,
            "start_time": s.start_time.isoformat(),
            "end_time": s.end_time.isoformat() if s.end_time else None,
            "duration_hours": (
                (s.end_time or datetime.now()) - s.start_time
            ).total_seconds() / 3600,
            "initial_capital": s.initial_capital,
            "final_capital": s.current_capital,
            "total_pnl": s.total_pnl,
            "total_pnl_pct": (s.current_capital - s.initial_capital) / s.initial_capital * 100,
            "total_trades": s.total_trades,
            "winning_trades": s.winning_trades,
            "losing_trades": s.losing_trades,
            "win_rate": s.winning_trades / s.total_trades if s.total_trades > 0 else 0,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "max_drawdown": s.max_drawdown,
            "sharpe_ratio": sharpe,
            "active_trades": len(s.active_trades),
            "daily_pnl": s.daily_pnl,
        }
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current session statistics."""
        return self._calculate_session_summary()
    
    def get_active_trades(self) -> List[Dict[str, Any]]:
        """Get list of active trades."""
        if not self.session:
            return []
        
        return [
            {
                "id": t.id,
                "underlying": t.underlying,
                "direction": t.direction,
                "entry_price": t.entry_price,
                "target": t.target_price,
                "stop_loss": t.stop_loss_price,
                "entry_time": t.entry_time.isoformat(),
                "ml_confidence": t.ml_confidence,
            }
            for t in self.session.active_trades
        ]


# Singleton instance
_paper_runner: Optional[PaperTradingRunner] = None


def get_paper_trading_runner() -> PaperTradingRunner:
    """Get or create the singleton paper trading runner instance."""
    global _paper_runner
    if _paper_runner is None:
        _paper_runner = PaperTradingRunner()
    return _paper_runner
