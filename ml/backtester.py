"""
Backtesting Engine for ML Options Trading Strategies

Simulates trading on historical data with realistic costs,
slippage, and performance metrics calculation.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from config.settings import ML_CONFIG, TRADING_CONFIG
from core.logger import logger


@dataclass
class BacktestTrade:
    """Represents a single trade in backtesting."""
    entry_time: datetime
    exit_time: Optional[datetime]
    underlying: str
    strategy_type: str
    direction: str              # 'LONG' or 'SHORT'
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    confidence: float
    ml_confidence: float
    
    # P&L
    gross_pnl: float = 0.0
    costs: float = 0.0
    net_pnl: float = 0.0
    pnl_percent: float = 0.0
    
    # Exit info
    exit_reason: str = ""
    is_winner: bool = False
    duration_hours: float = 0.0
    
    # ML info
    ml_prediction: str = ""
    prediction_correct: bool = False


@dataclass
class BacktestResult:
    """Container for backtesting results."""
    # Basic info
    start_date: datetime
    end_date: datetime
    duration_days: int
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # P&L metrics
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_duration_days: int = 0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Trade characteristics
    avg_trade_duration_hours: float = 0.0
    avg_confidence: float = 0.0
    
    # ML metrics
    ml_accuracy: float = 0.0
    ml_predictions_used: int = 0
    
    # Daily returns for analysis
    daily_returns: List[float] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    trades: List[BacktestTrade] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "duration_days": self.duration_days,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "total_pnl": self.total_pnl,
            "profit_factor": self.profit_factor,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "avg_trade_duration_hours": self.avg_trade_duration_hours,
            "ml_accuracy": self.ml_accuracy,
        }


class Backtester:
    """
    Backtest ML trading strategies on historical data.
    
    Features:
    - Realistic trading simulation with costs
    - Slippage modeling
    - Multiple exit conditions
    - Comprehensive performance metrics
    - Equity curve and drawdown analysis
    """
    
    # Cost parameters
    BROKERAGE_PER_ORDER = 20.0          # INR per order
    STT_RATE = 0.001                     # 0.1% Securities Transaction Tax
    SLIPPAGE_PERCENT = 0.05              # 0.05% slippage
    BID_ASK_SPREAD_PERCENT = 0.1         # 0.1% bid-ask spread
    
    def __init__(
        self,
        initial_capital: float = 1000000.0,
        max_positions: int = 5,
        position_size_percent: float = 10.0
    ):
        """
        Initialize backtester.
        
        Args:
            initial_capital: Starting capital in INR
            max_positions: Maximum concurrent positions
            position_size_percent: Percentage of capital per position
        """
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.position_size_percent = position_size_percent
        
        # State
        self.capital = initial_capital
        self.equity = initial_capital
        self.positions: List[BacktestTrade] = []
        self.closed_trades: List[BacktestTrade] = []
        self.equity_history: List[Tuple[datetime, float]] = []
        self.daily_pnl: Dict[str, float] = {}
        
        # Config
        self.stop_loss_percent = TRADING_CONFIG.get("default_sl_percent", 30) / 100
        self.target_percent = TRADING_CONFIG.get("default_target_percent", 50) / 100
        
        logger.info(f"Backtester initialized: capital={initial_capital}, "
                   f"max_positions={max_positions}")
    
    def run_backtest(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> BacktestResult:
        """
        Run backtest on historical data with signals.
        
        Args:
            data: DataFrame with OHLCV data (index=datetime)
            signals: DataFrame with entry signals (datetime, direction, confidence, ml_confidence)
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            BacktestResult with all metrics
        """
        # Reset state
        self._reset()
        
        # Filter dates
        if start_date:
            data = data[data.index >= start_date]
            signals = signals[signals.index >= start_date] if hasattr(signals, 'index') else signals
        if end_date:
            data = data[data.index <= end_date]
            signals = signals[signals.index <= end_date] if hasattr(signals, 'index') else signals
        
        if len(data) == 0:
            logger.warning("No data for backtest")
            return self._generate_result(start_date or datetime.now(), end_date or datetime.now())
        
        actual_start = data.index[0] if hasattr(data.index[0], 'isoformat') else datetime.now()
        actual_end = data.index[-1] if hasattr(data.index[-1], 'isoformat') else datetime.now()
        
        logger.info(f"Running backtest from {actual_start} to {actual_end}")
        
        # Convert signals to dict for faster lookup
        signal_dict = self._signals_to_dict(signals)
        
        # Iterate through each time step
        for timestamp, row in data.iterrows():
            current_price = row["close"]
            current_time = timestamp if isinstance(timestamp, datetime) else pd.to_datetime(timestamp)
            
            # 1. Check exits for open positions
            self._check_exits(current_time, current_price, row)
            
            # 2. Check for new entry signals
            if current_time in signal_dict:
                signal = signal_dict[current_time]
                self._process_entry_signal(current_time, current_price, signal, row)
            
            # 3. Update equity
            self._update_equity(current_time, current_price)
        
        # Close any remaining positions at end
        self._close_all_positions(actual_end, data.iloc[-1]["close"], "END_OF_BACKTEST")
        
        return self._generate_result(actual_start, actual_end)
    
    def run_ml_backtest(
        self,
        data: pd.DataFrame,
        predictor,
        feature_engineer,
        min_confidence: float = 0.55
    ) -> BacktestResult:
        """
        Run backtest using ML predictor on historical data.
        
        Args:
            data: DataFrame with OHLCV + technical indicators
            predictor: MLPredictor instance
            feature_engineer: FeatureEngineer instance
            min_confidence: Minimum confidence to enter trade
            
        Returns:
            BacktestResult
        """
        self._reset()
        
        if len(data) < 30:
            logger.warning("Insufficient data for ML backtest")
            return self._generate_result(datetime.now(), datetime.now())
        
        actual_start = data.index[0] if hasattr(data.index[0], 'isoformat') else datetime.now()
        actual_end = data.index[-1] if hasattr(data.index[-1], 'isoformat') else datetime.now()
        
        logger.info(f"Running ML backtest from {actual_start} to {actual_end}")
        
        for i in range(30, len(data)):  # Need lookback for features
            row = data.iloc[i]
            current_time = data.index[i]
            if not isinstance(current_time, datetime):
                current_time = pd.to_datetime(current_time)
            current_price = row["close"]
            
            # 1. Check exits
            self._check_exits(current_time, current_price, row)
            
            # 2. Get ML prediction
            if len(self.positions) < self.max_positions:
                # Extract features from row
                features = self._extract_features_from_row(row, data.iloc[max(0, i-30):i])
                
                # Get prediction
                prediction = predictor.predict(features)
                
                if prediction and prediction.confidence >= min_confidence:
                    if prediction.direction in ["BULLISH", "BEARISH"]:
                        signal = {
                            "direction": "LONG" if prediction.direction == "BULLISH" else "SHORT",
                            "confidence": prediction.confidence,
                            "ml_confidence": prediction.confidence,
                            "ml_prediction": prediction.direction,
                            "underlying": row.get("symbol", "NIFTY"),
                            "strategy_type": "ml_direction",
                        }
                        self._process_entry_signal(current_time, current_price, signal, row)
            
            # 3. Update equity
            self._update_equity(current_time, current_price)
        
        # Close remaining positions
        self._close_all_positions(actual_end, data.iloc[-1]["close"], "END_OF_BACKTEST")
        
        return self._generate_result(actual_start, actual_end)
    
    def _reset(self) -> None:
        """Reset backtester state."""
        self.capital = self.initial_capital
        self.equity = self.initial_capital
        self.positions = []
        self.closed_trades = []
        self.equity_history = []
        self.daily_pnl = {}
    
    def _signals_to_dict(self, signals: pd.DataFrame) -> Dict[datetime, Dict]:
        """Convert signals DataFrame to dict for fast lookup."""
        result = {}
        
        if signals is None or len(signals) == 0:
            return result
        
        for idx, row in signals.iterrows():
            timestamp = idx if isinstance(idx, datetime) else pd.to_datetime(idx)
            result[timestamp] = row.to_dict() if hasattr(row, 'to_dict') else dict(row)
        
        return result
    
    def _process_entry_signal(
        self,
        timestamp: datetime,
        price: float,
        signal: Dict,
        row: pd.Series
    ) -> None:
        """Process an entry signal."""
        if len(self.positions) >= self.max_positions:
            return
        
        direction = signal.get("direction", "LONG")
        confidence = signal.get("confidence", 0.5)
        ml_confidence = signal.get("ml_confidence", confidence)
        
        # Calculate position size
        position_value = self.capital * (self.position_size_percent / 100)
        
        # Apply slippage to entry
        entry_price = price * (1 + self.SLIPPAGE_PERCENT / 100) if direction == "LONG" else \
                      price * (1 - self.SLIPPAGE_PERCENT / 100)
        
        quantity = int(position_value / entry_price)
        if quantity <= 0:
            return
        
        # Calculate entry costs
        entry_costs = self.BROKERAGE_PER_ORDER + (entry_price * quantity * self.STT_RATE)
        
        # Create trade
        trade = BacktestTrade(
            entry_time=timestamp,
            exit_time=None,
            underlying=signal.get("underlying", "NIFTY"),
            strategy_type=signal.get("strategy_type", "direction"),
            direction=direction,
            entry_price=entry_price,
            exit_price=None,
            quantity=quantity,
            confidence=confidence,
            ml_confidence=ml_confidence,
            costs=entry_costs,
            ml_prediction=signal.get("ml_prediction", ""),
        )
        
        self.positions.append(trade)
        self.capital -= entry_costs  # Deduct entry costs
        
        logger.debug(f"Backtest entry: {direction} at {entry_price:.2f}, qty={quantity}")
    
    def _check_exits(
        self,
        timestamp: datetime,
        current_price: float,
        row: pd.Series
    ) -> None:
        """Check exit conditions for open positions."""
        positions_to_close = []
        
        for trade in self.positions:
            should_exit = False
            exit_reason = ""
            
            # Calculate current P&L
            if trade.direction == "LONG":
                pnl_percent = (current_price - trade.entry_price) / trade.entry_price
            else:
                pnl_percent = (trade.entry_price - current_price) / trade.entry_price
            
            # Check stop loss
            if pnl_percent <= -self.stop_loss_percent:
                should_exit = True
                exit_reason = "STOP_LOSS"
            
            # Check target
            elif pnl_percent >= self.target_percent:
                should_exit = True
                exit_reason = "TARGET_HIT"
            
            # Check time-based exit (max 5 days for options)
            trade_duration = timestamp - trade.entry_time
            if trade_duration.days >= 5:
                should_exit = True
                exit_reason = "TIME_EXIT"
            
            if should_exit:
                positions_to_close.append((trade, current_price, exit_reason))
        
        # Close positions
        for trade, exit_price, reason in positions_to_close:
            self._close_position(trade, timestamp, exit_price, reason)
    
    def _close_position(
        self,
        trade: BacktestTrade,
        timestamp: datetime,
        exit_price: float,
        reason: str
    ) -> None:
        """Close a position."""
        if trade not in self.positions:
            return
        
        # Apply slippage to exit
        if trade.direction == "LONG":
            actual_exit = exit_price * (1 - self.SLIPPAGE_PERCENT / 100)
        else:
            actual_exit = exit_price * (1 + self.SLIPPAGE_PERCENT / 100)
        
        # Calculate P&L
        if trade.direction == "LONG":
            gross_pnl = (actual_exit - trade.entry_price) * trade.quantity
        else:
            gross_pnl = (trade.entry_price - actual_exit) * trade.quantity
        
        # Exit costs
        exit_costs = self.BROKERAGE_PER_ORDER + (actual_exit * trade.quantity * self.STT_RATE)
        
        # Update trade
        trade.exit_time = timestamp
        trade.exit_price = actual_exit
        trade.gross_pnl = gross_pnl
        trade.costs += exit_costs
        trade.net_pnl = gross_pnl - trade.costs
        trade.pnl_percent = (trade.net_pnl / (trade.entry_price * trade.quantity)) * 100
        trade.exit_reason = reason
        trade.is_winner = trade.net_pnl > 0
        trade.duration_hours = (timestamp - trade.entry_time).total_seconds() / 3600
        
        # Check ML prediction accuracy
        if trade.ml_prediction:
            if trade.ml_prediction == "BULLISH" and trade.net_pnl > 0:
                trade.prediction_correct = True
            elif trade.ml_prediction == "BEARISH" and trade.net_pnl > 0:
                trade.prediction_correct = True
        
        # Move to closed trades
        self.positions.remove(trade)
        self.closed_trades.append(trade)
        
        # Update capital
        self.capital += (trade.entry_price * trade.quantity) + trade.net_pnl
        
        # Track daily P&L
        date_str = timestamp.strftime("%Y-%m-%d")
        self.daily_pnl[date_str] = self.daily_pnl.get(date_str, 0) + trade.net_pnl
        
        logger.debug(f"Backtest exit: {reason}, P&L={trade.net_pnl:.2f}")
    
    def _close_all_positions(
        self,
        timestamp: datetime,
        price: float,
        reason: str
    ) -> None:
        """Close all open positions."""
        for trade in list(self.positions):
            self._close_position(trade, timestamp, price, reason)
    
    def _update_equity(self, timestamp: datetime, current_price: float) -> None:
        """Update equity curve."""
        # Calculate unrealized P&L
        unrealized_pnl = 0
        for trade in self.positions:
            if trade.direction == "LONG":
                unrealized_pnl += (current_price - trade.entry_price) * trade.quantity
            else:
                unrealized_pnl += (trade.entry_price - current_price) * trade.quantity
        
        self.equity = self.capital + unrealized_pnl
        self.equity_history.append((timestamp, self.equity))
    
    def _extract_features_from_row(
        self,
        row: pd.Series,
        lookback_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Extract feature dictionary from DataFrame row."""
        features = {}
        
        # Direct features from row
        feature_columns = [
            "return_1d", "return_5d", "return_10d", "return_20d",
            "rsi_14", "macd_histogram", "bb_percent_b", "bb_width",
            "stoch_k", "stoch_d", "williams_r", "atr_percent",
            "hv_10", "hv_20", "price_vs_sma20", "volume_ratio"
        ]
        
        for col in feature_columns:
            if col in row.index:
                features[col] = float(row[col]) if pd.notna(row[col]) else 0.0
        
        # Calculate additional features if not present
        if "close" in row.index:
            features["spot_price"] = float(row["close"])
        
        return features
    
    def _generate_result(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> BacktestResult:
        """Generate backtest result with all metrics."""
        result = BacktestResult(
            start_date=start_date,
            end_date=end_date,
            duration_days=(end_date - start_date).days,
            trades=self.closed_trades,
        )
        
        if not self.closed_trades:
            return result
        
        # Trade statistics
        result.total_trades = len(self.closed_trades)
        result.winning_trades = sum(1 for t in self.closed_trades if t.is_winner)
        result.losing_trades = result.total_trades - result.winning_trades
        result.win_rate = result.winning_trades / result.total_trades if result.total_trades > 0 else 0
        
        # P&L metrics
        pnls = [t.net_pnl for t in self.closed_trades]
        winning_pnls = [p for p in pnls if p > 0]
        losing_pnls = [p for p in pnls if p < 0]
        
        result.total_pnl = sum(pnls)
        result.gross_profit = sum(winning_pnls)
        result.gross_loss = abs(sum(losing_pnls))
        result.profit_factor = result.gross_profit / result.gross_loss if result.gross_loss > 0 else float('inf')
        result.avg_win = np.mean(winning_pnls) if winning_pnls else 0
        result.avg_loss = np.mean(losing_pnls) if losing_pnls else 0
        result.largest_win = max(pnls) if pnls else 0
        result.largest_loss = min(pnls) if pnls else 0
        
        # Trade characteristics
        durations = [t.duration_hours for t in self.closed_trades]
        confidences = [t.confidence for t in self.closed_trades]
        result.avg_trade_duration_hours = np.mean(durations) if durations else 0
        result.avg_confidence = np.mean(confidences) if confidences else 0
        
        # ML metrics
        ml_trades = [t for t in self.closed_trades if t.ml_prediction]
        if ml_trades:
            result.ml_predictions_used = len(ml_trades)
            result.ml_accuracy = sum(1 for t in ml_trades if t.prediction_correct) / len(ml_trades)
        
        # Daily returns for risk metrics
        result.daily_returns = list(self.daily_pnl.values())
        
        # Equity curve
        result.equity_curve = [eq for _, eq in self.equity_history]
        
        # Risk metrics
        if result.equity_curve:
            result.max_drawdown = self._calculate_max_drawdown(result.equity_curve)
            
        if result.daily_returns and len(result.daily_returns) > 1:
            result.sharpe_ratio = self._calculate_sharpe_ratio(result.daily_returns)
            result.sortino_ratio = self._calculate_sortino_ratio(result.daily_returns)
            
        if result.max_drawdown > 0 and result.duration_days > 0:
            annual_return = (result.total_pnl / self.initial_capital) * (365 / result.duration_days)
            result.calmar_ratio = annual_return / (result.max_drawdown / 100)
        
        return result
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown percentage."""
        if not equity_curve:
            return 0
        
        peak = equity_curve[0]
        max_dd = 0
        
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak * 100
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_sharpe_ratio(
        self,
        returns: List[float],
        risk_free_rate: float = 0.06
    ) -> float:
        """Calculate Sharpe ratio (annualized)."""
        if not returns or len(returns) < 2:
            return 0
        
        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        if std_return == 0:
            return 0
        
        # Assume daily returns, annualize
        daily_rf = risk_free_rate / 252
        sharpe = (mean_return - daily_rf) / std_return * np.sqrt(252)
        
        return sharpe
    
    def _calculate_sortino_ratio(
        self,
        returns: List[float],
        risk_free_rate: float = 0.06
    ) -> float:
        """Calculate Sortino ratio (annualized)."""
        if not returns or len(returns) < 2:
            return 0
        
        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        
        # Downside deviation
        downside_returns = returns_array[returns_array < 0]
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_std = np.std(downside_returns)
        
        if downside_std == 0:
            return float('inf')
        
        daily_rf = risk_free_rate / 252
        sortino = (mean_return - daily_rf) / downside_std * np.sqrt(252)
        
        return sortino


# Singleton instance
_backtester: Optional[Backtester] = None


def get_backtester(
    initial_capital: float = 1000000.0,
    max_positions: int = 5
) -> Backtester:
    """Get or create the singleton backtester instance."""
    global _backtester
    if _backtester is None:
        _backtester = Backtester(initial_capital, max_positions)
    return _backtester
