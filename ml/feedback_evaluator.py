"""
ML Feedback Evaluator - Correlate ML Predictions with Financial Outcomes

This module evaluates ML model performance based on actual trade outcomes,
not just directional accuracy. It measures:
1. Prediction accuracy in financial terms (Rs. made/lost per prediction)
2. Confidence calibration (do 80% confident predictions win 80% of the time?)
3. Strategy-specific performance by ML direction
4. Open position evaluation with current market prices
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

from core.database import database
from core.logger import logger
from data.data_fetcher import data_fetcher


@dataclass
class PredictionOutcome:
    """Represents a single ML prediction with its financial outcome."""
    trade_id: int
    underlying: str
    strategy_type: str
    predicted_direction: str  # BULLISH, BEARISH, NEUTRAL
    ml_confidence: float
    entry_time: datetime
    exit_time: Optional[datetime]
    realized_pnl: float
    unrealized_pnl: float
    status: str  # ACTIVE, CLOSED
    prediction_correct: Optional[bool]  # Based on financial outcome
    expected_profit: float
    max_loss: float


@dataclass
class FeedbackMetrics:
    """Aggregated feedback metrics for model evaluation."""
    total_predictions: int
    correct_predictions: int
    accuracy: float
    
    total_pnl: float
    avg_pnl_per_trade: float
    win_rate: float
    
    # Financial accuracy by confidence bucket
    confidence_calibration: Dict[str, Dict]
    
    # By direction
    direction_performance: Dict[str, Dict]
    
    # By strategy
    strategy_performance: Dict[str, Dict]
    
    # Profit factor
    gross_profit: float
    gross_loss: float
    profit_factor: float
    
    # Risk metrics
    avg_winner: float
    avg_loser: float
    expectancy: float


class MLFeedbackEvaluator:
    """
    Evaluates ML model performance based on actual financial outcomes.
    
    Key Metrics:
    1. Financial Accuracy: Did the prediction make money?
    2. Confidence Calibration: Are high-confidence predictions more profitable?
    3. Direction Accuracy: Does BULLISH prediction lead to profits on bullish strategies?
    4. Strategy Fit: Which strategies work best with ML signals?
    """
    
    def __init__(self):
        self.db_path = Path("data/trading_bot.db")
        self._ensure_feedback_table()
    
    def _ensure_feedback_table(self):
        """Create ml_prediction_feedback table if not exists."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ml_prediction_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id INTEGER UNIQUE,
                    underlying TEXT,
                    strategy_type TEXT,
                    predicted_direction TEXT,
                    actual_direction TEXT,
                    ml_confidence REAL,
                    spot_at_entry REAL,
                    spot_at_exit REAL,
                    price_change_percent REAL,
                    realized_pnl REAL,
                    unrealized_pnl REAL,
                    prediction_correct INTEGER,
                    financial_accuracy REAL,
                    entry_time TEXT,
                    exit_time TEXT,
                    evaluated_at TEXT,
                    model_version TEXT,
                    FOREIGN KEY (trade_id) REFERENCES trades(id)
                )
            """)
            
            # Add index for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_underlying 
                ON ml_prediction_feedback(underlying)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_direction 
                ON ml_prediction_feedback(predicted_direction)
            """)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error creating feedback table: {e}")
    
    def evaluate_all_trades(self, include_open: bool = True) -> FeedbackMetrics:
        """
        Evaluate all trades and compute feedback metrics.
        
        Args:
            include_open: Whether to include open positions with unrealized PnL
            
        Returns:
            FeedbackMetrics with comprehensive evaluation
        """
        outcomes = self._gather_trade_outcomes(include_open)
        
        if not outcomes:
            logger.warning("No trades to evaluate")
            return None
        
        # Store feedback in database
        self._store_feedback(outcomes)
        
        # Compute metrics
        return self._compute_metrics(outcomes)
    
    def _gather_trade_outcomes(self, include_open: bool = True) -> List[PredictionOutcome]:
        """Gather all trade outcomes with their ML prediction context."""
        outcomes = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all trades
            query = """
                SELECT id, underlying, strategy_type, signal_data, 
                       realized_pnl, status, entry_time, exit_time
                FROM trades
                WHERE signal_data IS NOT NULL
            """
            if not include_open:
                query += " AND status = 'CLOSED'"
            
            cursor.execute(query)
            trades = cursor.fetchall()
            
            for trade in trades:
                trade_id, underlying, strategy, signal_data, pnl, status, entry_time, exit_time = trade
                
                try:
                    signal = json.loads(signal_data) if signal_data else {}
                except:
                    signal = {}
                
                # Infer ML direction from strategy type
                predicted_direction = self._infer_direction_from_strategy(strategy)
                
                # Get confidence from signal
                ml_confidence = signal.get("confidence", 0.5)
                expected_profit = signal.get("expected_profit", 0)
                max_loss = signal.get("max_loss", 0)
                
                # Calculate unrealized PnL for open positions
                unrealized_pnl = 0
                if status == "ACTIVE":
                    unrealized_pnl = self._calculate_unrealized_pnl(underlying, signal)
                
                # Determine if prediction was correct based on financial outcome
                total_pnl = (pnl or 0) + unrealized_pnl
                prediction_correct = self._is_prediction_correct(
                    predicted_direction, strategy, total_pnl
                )
                
                outcome = PredictionOutcome(
                    trade_id=trade_id,
                    underlying=underlying,
                    strategy_type=strategy,
                    predicted_direction=predicted_direction,
                    ml_confidence=ml_confidence,
                    entry_time=datetime.fromisoformat(entry_time) if entry_time else None,
                    exit_time=datetime.fromisoformat(exit_time) if exit_time else None,
                    realized_pnl=pnl or 0,
                    unrealized_pnl=unrealized_pnl,
                    status=status,
                    prediction_correct=prediction_correct,
                    expected_profit=expected_profit,
                    max_loss=max_loss,
                )
                outcomes.append(outcome)
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error gathering trade outcomes: {e}")
        
        return outcomes
    
    def _infer_direction_from_strategy(self, strategy_type: str) -> str:
        """Infer the ML predicted direction from the strategy executed."""
        bullish = ["long_call", "bull_call_spread", "bull_put_spread", "short_put"]
        bearish = ["long_put", "bear_put_spread", "bear_call_spread", "short_call"]
        neutral = ["straddle", "strangle", "iron_condor", "iron_butterfly"]
        
        strategy_lower = strategy_type.lower()
        
        if strategy_lower in bullish:
            return "BULLISH"
        elif strategy_lower in bearish:
            return "BEARISH"
        elif strategy_lower in neutral:
            return "NEUTRAL"
        else:
            return "UNKNOWN"
    
    def _is_prediction_correct(
        self, 
        predicted_direction: str, 
        strategy_type: str, 
        total_pnl: float
    ) -> bool:
        """
        Determine if prediction was correct based on financial outcome.
        
        For directional strategies: Correct if profitable
        For neutral strategies: Correct if profitable (volatility prediction was right)
        """
        # Simple rule: prediction is correct if we made money
        # This is the ultimate test - did the ML signal result in profit?
        return total_pnl > 0
    
    def _calculate_unrealized_pnl(self, underlying: str, signal: Dict) -> float:
        """Calculate unrealized PnL for open position."""
        try:
            legs = signal.get("legs", [])
            if not legs:
                return 0
            
            unrealized = 0
            
            for leg in legs:
                entry_price = leg.get("entry_price", 0)
                quantity = leg.get("quantity", 0)
                direction = leg.get("direction", "BUY")
                symbol = leg.get("symbol", "")
                
                # Get current price
                current_price = self._get_current_option_price(symbol)
                if current_price is None:
                    continue
                
                # Calculate PnL
                if direction == "BUY":
                    leg_pnl = (current_price - entry_price) * quantity
                else:  # SELL
                    leg_pnl = (entry_price - current_price) * quantity
                
                unrealized += leg_pnl
            
            return unrealized
            
        except Exception as e:
            logger.error(f"Error calculating unrealized PnL: {e}")
            return 0
    
    def _get_current_option_price(self, symbol: str) -> Optional[float]:
        """Get current price for an option symbol."""
        try:
            # Try to get from Kite
            quote = data_fetcher.kite.quote(f"NFO:{symbol}")
            if quote and f"NFO:{symbol}" in quote:
                return quote[f"NFO:{symbol}"].get("last_price", 0)
        except Exception as e:
            logger.debug(f"Could not get price for {symbol}: {e}")
        return None
    
    def _store_feedback(self, outcomes: List[PredictionOutcome]):
        """Store prediction feedback in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for outcome in outcomes:
                # Get spot prices if available
                spot_entry = 0
                spot_exit = 0
                price_change = 0
                
                total_pnl = outcome.realized_pnl + outcome.unrealized_pnl
                
                cursor.execute("""
                    INSERT OR REPLACE INTO ml_prediction_feedback 
                    (trade_id, underlying, strategy_type, predicted_direction,
                     ml_confidence, realized_pnl, unrealized_pnl, prediction_correct,
                     entry_time, exit_time, evaluated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    outcome.trade_id,
                    outcome.underlying,
                    outcome.strategy_type,
                    outcome.predicted_direction,
                    outcome.ml_confidence,
                    outcome.realized_pnl,
                    outcome.unrealized_pnl,
                    1 if outcome.prediction_correct else 0,
                    outcome.entry_time.isoformat() if outcome.entry_time else None,
                    outcome.exit_time.isoformat() if outcome.exit_time else None,
                    datetime.now().isoformat(),
                ))
            
            conn.commit()
            conn.close()
            logger.info(f"Stored feedback for {len(outcomes)} trades")
            
        except Exception as e:
            logger.error(f"Error storing feedback: {e}")
    
    def _compute_metrics(self, outcomes: List[PredictionOutcome]) -> FeedbackMetrics:
        """Compute comprehensive feedback metrics."""
        
        total = len(outcomes)
        if total == 0:
            return None
        
        # Basic accuracy
        correct = sum(1 for o in outcomes if o.prediction_correct)
        accuracy = correct / total
        
        # PnL metrics
        total_pnl = sum(o.realized_pnl + o.unrealized_pnl for o in outcomes)
        avg_pnl = total_pnl / total
        
        winners = [o for o in outcomes if (o.realized_pnl + o.unrealized_pnl) > 0]
        losers = [o for o in outcomes if (o.realized_pnl + o.unrealized_pnl) <= 0]
        
        win_rate = len(winners) / total if total > 0 else 0
        
        gross_profit = sum(o.realized_pnl + o.unrealized_pnl for o in winners)
        gross_loss = abs(sum(o.realized_pnl + o.unrealized_pnl for o in losers))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        avg_winner = gross_profit / len(winners) if winners else 0
        avg_loser = gross_loss / len(losers) if losers else 0
        
        # Expectancy = (Win% × Avg Win) - (Loss% × Avg Loss)
        expectancy = (win_rate * avg_winner) - ((1 - win_rate) * avg_loser)
        
        # Confidence calibration
        confidence_calibration = self._compute_confidence_calibration(outcomes)
        
        # Direction performance
        direction_performance = self._compute_direction_performance(outcomes)
        
        # Strategy performance
        strategy_performance = self._compute_strategy_performance(outcomes)
        
        return FeedbackMetrics(
            total_predictions=total,
            correct_predictions=correct,
            accuracy=accuracy,
            total_pnl=total_pnl,
            avg_pnl_per_trade=avg_pnl,
            win_rate=win_rate,
            confidence_calibration=confidence_calibration,
            direction_performance=direction_performance,
            strategy_performance=strategy_performance,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            profit_factor=profit_factor,
            avg_winner=avg_winner,
            avg_loser=avg_loser,
            expectancy=expectancy,
        )
    
    def _compute_confidence_calibration(self, outcomes: List[PredictionOutcome]) -> Dict:
        """
        Compute win rate and avg PnL for each confidence bucket.
        
        This shows if high confidence predictions are actually more accurate.
        """
        buckets = {
            "50-60%": {"range": (0.50, 0.60), "trades": [], "wins": 0, "pnl": 0},
            "60-70%": {"range": (0.60, 0.70), "trades": [], "wins": 0, "pnl": 0},
            "70-80%": {"range": (0.70, 0.80), "trades": [], "wins": 0, "pnl": 0},
            "80-90%": {"range": (0.80, 0.90), "trades": [], "wins": 0, "pnl": 0},
            "90-100%": {"range": (0.90, 1.00), "trades": [], "wins": 0, "pnl": 0},
        }
        
        for outcome in outcomes:
            conf = outcome.ml_confidence
            pnl = outcome.realized_pnl + outcome.unrealized_pnl
            
            for bucket_name, bucket_data in buckets.items():
                low, high = bucket_data["range"]
                if low <= conf < high or (high == 1.0 and conf == 1.0):
                    bucket_data["trades"].append(outcome)
                    bucket_data["pnl"] += pnl
                    if pnl > 0:
                        bucket_data["wins"] += 1
                    break
        
        # Compute metrics for each bucket
        calibration = {}
        for bucket_name, bucket_data in buckets.items():
            n = len(bucket_data["trades"])
            if n > 0:
                calibration[bucket_name] = {
                    "count": n,
                    "win_rate": bucket_data["wins"] / n,
                    "total_pnl": bucket_data["pnl"],
                    "avg_pnl": bucket_data["pnl"] / n,
                    "expected_win_rate": sum(bucket_data["range"]) / 2,  # Midpoint
                }
        
        return calibration
    
    def _compute_direction_performance(self, outcomes: List[PredictionOutcome]) -> Dict:
        """Compute performance metrics by predicted direction."""
        directions = {}
        
        for outcome in outcomes:
            direction = outcome.predicted_direction
            if direction not in directions:
                directions[direction] = {"trades": 0, "wins": 0, "pnl": 0}
            
            pnl = outcome.realized_pnl + outcome.unrealized_pnl
            directions[direction]["trades"] += 1
            directions[direction]["pnl"] += pnl
            if pnl > 0:
                directions[direction]["wins"] += 1
        
        # Compute rates
        for direction, data in directions.items():
            n = data["trades"]
            data["win_rate"] = data["wins"] / n if n > 0 else 0
            data["avg_pnl"] = data["pnl"] / n if n > 0 else 0
        
        return directions
    
    def _compute_strategy_performance(self, outcomes: List[PredictionOutcome]) -> Dict:
        """Compute performance metrics by strategy type."""
        strategies = {}
        
        for outcome in outcomes:
            strategy = outcome.strategy_type
            if strategy not in strategies:
                strategies[strategy] = {"trades": 0, "wins": 0, "pnl": 0}
            
            pnl = outcome.realized_pnl + outcome.unrealized_pnl
            strategies[strategy]["trades"] += 1
            strategies[strategy]["pnl"] += pnl
            if pnl > 0:
                strategies[strategy]["wins"] += 1
        
        # Compute rates
        for strategy, data in strategies.items():
            n = data["trades"]
            data["win_rate"] = data["wins"] / n if n > 0 else 0
            data["avg_pnl"] = data["pnl"] / n if n > 0 else 0
        
        return strategies
    
    def get_model_score(self, metrics: FeedbackMetrics) -> float:
        """
        Compute a single score (0-100) for the model based on feedback.
        
        This score can be used to compare models and decide on retraining.
        """
        if not metrics:
            return 0
        
        # Weighted components
        win_rate_score = metrics.win_rate * 30  # Max 30 points
        
        # Profit factor score (cap at 3.0 for max points)
        pf_score = min(metrics.profit_factor / 3.0, 1.0) * 25  # Max 25 points
        
        # Positive expectancy bonus
        if metrics.expectancy > 0:
            expectancy_score = min(metrics.expectancy / 1000, 1.0) * 25  # Max 25 points
        else:
            expectancy_score = 0
        
        # Confidence calibration bonus (are high confidence = high win rate?)
        calibration_score = self._compute_calibration_score(metrics.confidence_calibration) * 20
        
        return win_rate_score + pf_score + expectancy_score + calibration_score
    
    def _compute_calibration_score(self, calibration: Dict) -> float:
        """
        Score how well calibrated the confidence is.
        
        Perfect calibration: 70% confidence = 70% win rate
        """
        if not calibration:
            return 0
        
        errors = []
        for bucket_name, data in calibration.items():
            expected = data.get("expected_win_rate", 0.5)
            actual = data.get("win_rate", 0.5)
            error = abs(expected - actual)
            errors.append(error)
        
        avg_error = sum(errors) / len(errors) if errors else 0.5
        
        # Convert error to score (0 error = 1.0 score)
        return max(0, 1.0 - avg_error * 2)
    
    def print_feedback_report(self, metrics: FeedbackMetrics = None):
        """Print a comprehensive feedback report."""
        if metrics is None:
            metrics = self.evaluate_all_trades(include_open=True)
        
        if not metrics:
            print("No trades to evaluate")
            return
        
        print("=" * 70)
        print("ML MODEL FEEDBACK EVALUATION REPORT")
        print("=" * 70)
        print()
        
        print("📊 OVERALL PERFORMANCE")
        print("-" * 40)
        print(f"  Total Trades Evaluated: {metrics.total_predictions}")
        print(f"  Win Rate: {metrics.win_rate:.1%}")
        print(f"  Total PnL: Rs.{metrics.total_pnl:,.2f}")
        print(f"  Avg PnL/Trade: Rs.{metrics.avg_pnl_per_trade:,.2f}")
        print()
        
        print("💰 PROFITABILITY METRICS")
        print("-" * 40)
        print(f"  Gross Profit: Rs.{metrics.gross_profit:,.2f}")
        print(f"  Gross Loss: Rs.{metrics.gross_loss:,.2f}")
        print(f"  Profit Factor: {metrics.profit_factor:.2f}")
        print(f"  Avg Winner: Rs.{metrics.avg_winner:,.2f}")
        print(f"  Avg Loser: Rs.{metrics.avg_loser:,.2f}")
        print(f"  Expectancy: Rs.{metrics.expectancy:,.2f}/trade")
        print()
        
        print("🎯 CONFIDENCE CALIBRATION")
        print("-" * 40)
        print("  (Does high confidence = high profit?)")
        for bucket, data in metrics.confidence_calibration.items():
            expected_wr = data.get("expected_win_rate", 0) * 100
            actual_wr = data.get("win_rate", 0) * 100
            count = data.get("count", 0)
            avg_pnl = data.get("avg_pnl", 0)
            calibration = "✅" if abs(expected_wr - actual_wr) < 15 else "⚠️"
            print(f"  {bucket}: {count} trades, {actual_wr:.0f}% win rate, Rs.{avg_pnl:,.0f}/trade {calibration}")
        print()
        
        print("🧭 DIRECTION PERFORMANCE")
        print("-" * 40)
        print("  (How accurate is each directional prediction?)")
        for direction, data in metrics.direction_performance.items():
            trades = data["trades"]
            wr = data["win_rate"] * 100
            pnl = data["pnl"]
            emoji = "✅" if pnl > 0 else "❌"
            print(f"  {emoji} {direction}: {trades} trades, {wr:.0f}% win rate, Rs.{pnl:,.2f}")
        print()
        
        print("📈 STRATEGY PERFORMANCE")
        print("-" * 40)
        for strategy, data in sorted(metrics.strategy_performance.items(), 
                                     key=lambda x: x[1]["pnl"], reverse=True):
            trades = data["trades"]
            wr = data["win_rate"] * 100
            pnl = data["pnl"]
            emoji = "✅" if pnl > 0 else "❌"
            print(f"  {emoji} {strategy}: {trades} trades, {wr:.0f}% win rate, Rs.{pnl:,.2f}")
        print()
        
        # Model score
        score = self.get_model_score(metrics)
        print("🏆 MODEL SCORE")
        print("-" * 40)
        print(f"  Overall Score: {score:.1f}/100")
        if score >= 70:
            print("  Rating: EXCELLENT - Model is performing well")
        elif score >= 50:
            print("  Rating: GOOD - Model is profitable but can improve")
        elif score >= 30:
            print("  Rating: FAIR - Consider retraining with more data")
        else:
            print("  Rating: POOR - Model needs significant improvement")
        print()
        
        print("💡 RECOMMENDATIONS")
        print("-" * 40)
        
        # Generate recommendations
        recs = self._generate_recommendations(metrics)
        for i, rec in enumerate(recs, 1):
            print(f"  {i}. {rec}")
        
        print()
        print("=" * 70)
    
    def _generate_recommendations(self, metrics: FeedbackMetrics) -> List[str]:
        """Generate actionable recommendations based on metrics."""
        recs = []
        
        # Check confidence calibration
        for bucket, data in metrics.confidence_calibration.items():
            expected = data.get("expected_win_rate", 0.5)
            actual = data.get("win_rate", 0.5)
            if actual < expected - 0.2:
                recs.append(f"Model overconfident in {bucket} range - consider lowering confidence thresholds")
        
        # Check direction performance
        for direction, data in metrics.direction_performance.items():
            if data["pnl"] < 0:
                recs.append(f"Avoid {direction} predictions until model improves (losing Rs.{abs(data['pnl']):,.0f})")
        
        # Check strategy performance
        worst_strategies = sorted(metrics.strategy_performance.items(), 
                                 key=lambda x: x[1]["pnl"])[:2]
        for strategy, data in worst_strategies:
            if data["pnl"] < 0:
                recs.append(f"Disable {strategy} strategy (losing Rs.{abs(data['pnl']):,.0f})")
        
        # Profit factor
        if metrics.profit_factor < 1.0:
            recs.append("Model is net unprofitable - urgent retraining needed")
        elif metrics.profit_factor < 1.5:
            recs.append("Profit factor low - consider tightening entry criteria")
        
        if not recs:
            recs.append("Model performing well - continue monitoring")
        
        return recs


def get_feedback_evaluator() -> MLFeedbackEvaluator:
    """Get singleton instance of feedback evaluator."""
    if not hasattr(get_feedback_evaluator, "_instance"):
        get_feedback_evaluator._instance = MLFeedbackEvaluator()
    return get_feedback_evaluator._instance


if __name__ == "__main__":
    evaluator = MLFeedbackEvaluator()
    evaluator.print_feedback_report()
