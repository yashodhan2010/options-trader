"""
Database models for persisting trades and signals
"""
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import json

from config.settings import DATABASE_CONFIG
from core.logger import logger


class Database:
    """
    SQLite database for persisting trade data.
    """
    
    def __init__(self, db_path: Path = None):
        self.db_path = db_path or DATABASE_CONFIG["path"]
        self._ensure_tables()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _ensure_tables(self) -> None:
        """Create tables if they don't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                execution_id TEXT UNIQUE,
                underlying TEXT,
                strategy_type TEXT,
                signal_data TEXT,
                entry_time TIMESTAMP,
                exit_time TIMESTAMP,
                status TEXT,
                realized_pnl REAL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Orders table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT,
                execution_id TEXT,
                symbol TEXT,
                transaction_type TEXT,
                quantity INTEGER,
                order_type TEXT,
                price REAL,
                filled_price REAL,
                status TEXT,
                placed_at TIMESTAMP,
                filled_at TIMESTAMP,
                message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (execution_id) REFERENCES trades (execution_id)
            )
        """)
        
        # Signals table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                underlying TEXT,
                strategy_type TEXT,
                confidence REAL,
                signal_data TEXT,
                executed INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Daily PnL table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_pnl (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE UNIQUE,
                total_pnl REAL,
                num_trades INTEGER,
                num_winners INTEGER,
                num_losers INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Position Status Log table (for periodic updates)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS position_status_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                execution_id TEXT,
                underlying TEXT,
                strategy_type TEXT,
                current_pnl REAL,
                unrealized_pnl REAL,
                current_prices TEXT,
                time_in_trade_seconds INTEGER,
                stop_loss REAL,
                target REAL,
                status TEXT,
                logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (execution_id) REFERENCES trades (execution_id)
            )
        """)
        
        # ============================================================================
        # ML TABLES
        # ============================================================================
        
        # ML Feature Snapshots (for training data)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ml_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                execution_id TEXT,
                underlying TEXT,
                strategy_type TEXT,
                snapshot_time TIMESTAMP,
                snapshot_type TEXT DEFAULT 'entry',
                
                -- All features stored as JSON for flexibility
                features_json TEXT,
                
                -- Key features for quick queries (denormalized)
                spot_price REAL,
                rsi_14 REAL,
                macd_histogram REAL,
                iv_current REAL,
                iv_percentile REAL,
                pcr REAL,
                hv_20 REAL,
                atr_percent REAL,
                dte INTEGER,
                trend_score REAL,
                momentum_score REAL,
                
                -- Outcome (filled after trade closes)
                actual_pnl REAL,
                actual_pnl_percent REAL,
                outcome TEXT,
                trade_duration_seconds INTEGER,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (execution_id) REFERENCES trades (execution_id)
            )
        """)
        
        # ML Predictions Log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ml_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                execution_id TEXT,
                underlying TEXT,
                strategy_type TEXT,
                prediction_time TIMESTAMP,
                
                -- Model info
                model_version TEXT,
                model_type TEXT,
                
                -- Predictions
                direction_prediction TEXT,
                confidence_score REAL,
                rule_confidence REAL,
                blended_confidence REAL,
                predicted_pnl_range_low REAL,
                predicted_pnl_range_high REAL,
                exit_recommendation TEXT,
                exit_confidence REAL,
                
                -- Feature importance (top 5 as JSON)
                top_features_json TEXT,
                
                -- Outcome (filled after trade closes)
                actual_outcome TEXT,
                actual_pnl REAL,
                prediction_accurate INTEGER,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (execution_id) REFERENCES trades (execution_id)
            )
        """)
        
        # Model Performance Tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ml_model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version TEXT UNIQUE,
                model_type TEXT,
                trained_at TIMESTAMP,
                training_samples INTEGER,
                
                -- Validation metrics
                accuracy REAL,
                precision_score REAL,
                recall_score REAL,
                f1_score REAL,
                auc_roc REAL,
                
                -- Trading metrics (from backtest)
                backtest_win_rate REAL,
                backtest_sharpe REAL,
                backtest_sortino REAL,
                backtest_max_drawdown REAL,
                backtest_profit_factor REAL,
                backtest_total_trades INTEGER,
                backtest_total_pnl REAL,
                
                -- Live/paper performance (updated periodically)
                live_predictions INTEGER DEFAULT 0,
                live_accuracy REAL,
                live_pnl_contribution REAL,
                
                -- Model status
                is_active INTEGER DEFAULT 0,
                stage TEXT DEFAULT 'development',
                notes TEXT,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Feature Importance History
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ml_feature_importance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version TEXT,
                feature_name TEXT,
                importance_score REAL,
                importance_rank INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Market Data Cache (for historical data storage)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                data_type TEXT,
                interval TEXT,
                date DATE,
                
                -- OHLCV data
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                
                -- Options data (if applicable)
                oi INTEGER,
                iv REAL,
                
                -- Metadata
                source TEXT DEFAULT 'kite',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                UNIQUE(symbol, data_type, interval, date)
            )
        """)
        
        # ML Training Jobs Log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ml_training_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT UNIQUE,
                model_type TEXT,
                status TEXT DEFAULT 'pending',
                
                -- Configuration
                config_json TEXT,
                
                -- Timing
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                duration_seconds INTEGER,
                
                -- Results
                best_params_json TEXT,
                best_score REAL,
                num_trials INTEGER,
                
                -- Errors
                error_message TEXT,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indices for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ml_features_execution ON ml_features(execution_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ml_features_underlying ON ml_features(underlying)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ml_features_time ON ml_features(snapshot_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ml_predictions_execution ON ml_predictions(execution_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ml_predictions_model ON ml_predictions(model_version)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data_cache(symbol, date)")
        
        conn.commit()
        conn.close()
        logger.debug("Database tables ensured")
    
    def save_trade(
        self,
        execution_id: str,
        underlying: str,
        strategy_type: str,
        signal_data: Dict,
        status: str = "ACTIVE",
    ) -> bool:
        """
        Save a new trade to the database.
        
        Args:
            execution_id: Unique execution ID
            underlying: Underlying asset
            strategy_type: Type of strategy
            signal_data: Full signal data as dictionary
            status: Trade status
            
        Returns:
            True if successful
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO trades (execution_id, underlying, strategy_type, signal_data, entry_time, status)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                execution_id,
                underlying,
                strategy_type,
                json.dumps(signal_data),
                datetime.now().isoformat(),
                status,
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to save trade: {e}")
            return False
    
    def update_trade(
        self,
        execution_id: str,
        status: Optional[str] = None,
        exit_time: Optional[datetime] = None,
        realized_pnl: Optional[float] = None,
    ) -> bool:
        """
        Update an existing trade.
        
        Args:
            execution_id: Execution ID to update
            status: New status
            exit_time: Exit timestamp
            realized_pnl: Realized P&L
            
        Returns:
            True if successful
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            updates = []
            params = []
            
            if status:
                updates.append("status = ?")
                params.append(status)
            
            if exit_time:
                updates.append("exit_time = ?")
                params.append(exit_time.isoformat())
            
            if realized_pnl is not None:
                updates.append("realized_pnl = ?")
                params.append(realized_pnl)
            
            if updates:
                params.append(execution_id)
                cursor.execute(f"""
                    UPDATE trades SET {', '.join(updates)} WHERE execution_id = ?
                """, params)
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to update trade: {e}")
            return False
    
    def save_order(self, order_data: Dict) -> bool:
        """
        Save an order to the database.
        
        Args:
            order_data: Order data dictionary
            
        Returns:
            True if successful
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO orders (
                    order_id, execution_id, symbol, transaction_type,
                    quantity, order_type, price, filled_price, status,
                    placed_at, filled_at, message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                order_data.get("order_id"),
                order_data.get("execution_id"),
                order_data.get("symbol"),
                order_data.get("transaction_type"),
                order_data.get("quantity"),
                order_data.get("order_type"),
                order_data.get("price"),
                order_data.get("filled_price"),
                order_data.get("status"),
                order_data.get("placed_at"),
                order_data.get("filled_at"),
                order_data.get("message"),
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to save order: {e}")
            return False
    
    def save_signal(
        self,
        underlying: str,
        strategy_type: str,
        confidence: float,
        signal_data: Dict,
        executed: bool = False,
    ) -> bool:
        """
        Save a signal to the database.
        
        Args:
            underlying: Underlying asset
            strategy_type: Strategy type
            confidence: Signal confidence
            signal_data: Full signal data
            executed: Whether signal was executed
            
        Returns:
            True if successful
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO signals (underlying, strategy_type, confidence, signal_data, executed)
                VALUES (?, ?, ?, ?, ?)
            """, (
                underlying,
                strategy_type,
                confidence,
                json.dumps(signal_data),
                1 if executed else 0,
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to save signal: {e}")
            return False
    
    def get_trades(
        self,
        status: Optional[str] = None,
        underlying: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict]:
        """
        Get trades with optional filters.
        
        Args:
            status: Filter by status
            underlying: Filter by underlying
            start_date: Filter from date
            end_date: Filter to date
            
        Returns:
            List of trade dictionaries
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            query = "SELECT * FROM trades WHERE 1=1"
            params = []
            
            if status:
                query += " AND status = ?"
                params.append(status)
            
            if underlying:
                query += " AND underlying = ?"
                params.append(underlying)
            
            if start_date:
                query += " AND entry_time >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND entry_time <= ?"
                params.append(end_date.isoformat())
            
            query += " ORDER BY created_at DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to get trades: {e}")
            return []
    
    def get_daily_stats(self, date: datetime = None) -> Dict:
        """
        Get trading statistics for a specific day.
        
        Args:
            date: Date to get stats for (default: today)
            
        Returns:
            Statistics dictionary
        """
        date = date or datetime.now()
        date_str = date.strftime("%Y-%m-%d")
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Get all closed trades for the day
            cursor.execute("""
                SELECT realized_pnl FROM trades 
                WHERE DATE(entry_time) = ? AND status = 'CLOSED'
            """, (date_str,))
            
            rows = cursor.fetchall()
            conn.close()
            
            pnls = [row["realized_pnl"] for row in rows]
            
            return {
                "date": date_str,
                "total_pnl": sum(pnls),
                "num_trades": len(pnls),
                "num_winners": len([p for p in pnls if p > 0]),
                "num_losers": len([p for p in pnls if p < 0]),
                "win_rate": len([p for p in pnls if p > 0]) / len(pnls) * 100 if pnls else 0,
                "avg_winner": sum([p for p in pnls if p > 0]) / len([p for p in pnls if p > 0]) if [p for p in pnls if p > 0] else 0,
                "avg_loser": sum([p for p in pnls if p < 0]) / len([p for p in pnls if p < 0]) if [p for p in pnls if p < 0] else 0,
            }
            
        except Exception as e:
            logger.error(f"Failed to get daily stats: {e}")
            return {}
    
    def get_strategy_performance(self, days: int = 30) -> List[Dict]:
        """
        Get performance breakdown by strategy.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            List of strategy performance dictionaries
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    strategy_type,
                    COUNT(*) as num_trades,
                    SUM(realized_pnl) as total_pnl,
                    AVG(realized_pnl) as avg_pnl,
                    SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as winners,
                    SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) as losers
                FROM trades
                WHERE status = 'CLOSED' 
                AND entry_time >= datetime('now', ?)
                GROUP BY strategy_type
            """, (f"-{days} days",))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to get strategy performance: {e}")
            return []
    
    def get_active_trades(self) -> List[Dict]:
        """
        Get all trades with ACTIVE status (for overnight position recovery).
        
        Returns:
            List of active trade dictionaries with parsed signal_data
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM trades WHERE status = 'ACTIVE' ORDER BY entry_time ASC
            """)
            
            rows = cursor.fetchall()
            conn.close()
            
            trades = []
            for row in rows:
                trade = dict(row)
                # Parse signal_data JSON
                if trade.get("signal_data"):
                    try:
                        trade["signal_data"] = json.loads(trade["signal_data"])
                    except:
                        pass
                trades.append(trade)
            
            return trades
            
        except Exception as e:
            logger.error(f"Failed to get active trades: {e}")
            return []
    
    def log_position_status(
        self,
        execution_id: str,
        underlying: str,
        strategy_type: str,
        current_pnl: float,
        unrealized_pnl: float,
        current_prices: Dict[str, float],
        time_in_trade_seconds: int,
        stop_loss: float,
        target: float,
        status: str = "ACTIVE",
    ) -> bool:
        """
        Log periodic position status update.
        
        Args:
            execution_id: Execution ID
            underlying: Underlying asset
            strategy_type: Strategy type
            current_pnl: Current P&L
            unrealized_pnl: Unrealized P&L
            current_prices: Dictionary of symbol -> current price
            time_in_trade_seconds: Time in trade in seconds
            stop_loss: Current stop loss
            target: Current target
            status: Trade status
            
        Returns:
            True if successful
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO position_status_log (
                    execution_id, underlying, strategy_type, current_pnl,
                    unrealized_pnl, current_prices, time_in_trade_seconds,
                    stop_loss, target, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                execution_id,
                underlying,
                strategy_type,
                current_pnl,
                unrealized_pnl,
                json.dumps(current_prices),
                time_in_trade_seconds,
                stop_loss,
                target,
                status,
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to log position status: {e}")
            return False
    
    def get_position_status_history(
        self,
        execution_id: str,
        limit: int = 100,
    ) -> List[Dict]:
        """
        Get position status history for an execution.
        
        Args:
            execution_id: Execution ID
            limit: Maximum records to return
            
        Returns:
            List of status log entries
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM position_status_log 
                WHERE execution_id = ?
                ORDER BY logged_at DESC
                LIMIT ?
            """, (execution_id, limit))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to get position status history: {e}")
            return []
    
    # ============================================================================
    # ML-RELATED DATABASE METHODS
    # ============================================================================
    
    def save_ml_features(
        self,
        execution_id: str,
        underlying: str,
        strategy_type: str,
        features: Dict[str, float],
        snapshot_type: str = "entry",
        spot_price: float = None,
    ) -> bool:
        """
        Save ML feature snapshot for a trade.
        
        Args:
            execution_id: Trade execution ID
            underlying: Underlying asset
            strategy_type: Strategy type
            features: Dictionary of feature values
            snapshot_type: 'entry', 'exit', or 'periodic'
            spot_price: Current spot price
            
        Returns:
            Success status
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO ml_features (
                    execution_id, underlying, strategy_type, snapshot_time, snapshot_type,
                    features_json, spot_price, rsi_14, macd_histogram, iv_current,
                    iv_percentile, pcr, hv_20, atr_percent, dte, trend_score, momentum_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                execution_id,
                underlying,
                strategy_type,
                datetime.now().isoformat(),
                snapshot_type,
                json.dumps(features),
                spot_price or features.get("spot_price", 0),
                features.get("rsi_14", 0),
                features.get("macd_histogram", 0),
                features.get("iv_current", 0),
                features.get("iv_percentile", 0),
                features.get("pcr", 0),
                features.get("hv_20", 0),
                features.get("atr_percent", 0),
                features.get("dte", 0),
                features.get("trend_score", 0),
                features.get("momentum_score", 0),
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to save ML features: {e}")
            return False
    
    def update_ml_features_outcome(
        self,
        execution_id: str,
        actual_pnl: float,
        actual_pnl_percent: float,
        outcome: str,
        trade_duration_seconds: int
    ) -> bool:
        """
        Update ML features with trade outcome for feedback loop.
        
        Args:
            execution_id: Trade execution ID
            actual_pnl: Realized P&L
            actual_pnl_percent: P&L as percentage
            outcome: 'WIN', 'LOSS', or 'BREAKEVEN'
            trade_duration_seconds: Duration of trade
            
        Returns:
            Success status
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE ml_features 
                SET actual_pnl = ?, actual_pnl_percent = ?, outcome = ?, trade_duration_seconds = ?
                WHERE execution_id = ? AND snapshot_type = 'entry'
            """, (actual_pnl, actual_pnl_percent, outcome, trade_duration_seconds, execution_id))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to update ML features outcome: {e}")
            return False
    
    def save_ml_prediction(
        self,
        execution_id: str,
        underlying: str,
        strategy_type: str,
        model_version: str,
        model_type: str,
        direction_prediction: str,
        confidence_score: float,
        rule_confidence: float,
        blended_confidence: float,
        predicted_pnl_range: Tuple[float, float] = None,
        exit_recommendation: str = None,
        top_features: Dict[str, float] = None,
    ) -> bool:
        """
        Save an ML prediction for tracking.
        
        Args:
            execution_id: Trade execution ID
            underlying: Underlying asset
            strategy_type: Strategy type
            model_version: Model version string
            model_type: Type of model
            direction_prediction: 'BULLISH', 'BEARISH', 'NEUTRAL'
            confidence_score: ML confidence (0-1)
            rule_confidence: Rule-based confidence (0-1)
            blended_confidence: Final blended confidence
            predicted_pnl_range: (low, high) predicted P&L
            exit_recommendation: Exit recommendation if any
            top_features: Top feature importance dict
            
        Returns:
            Success status
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            pnl_low, pnl_high = predicted_pnl_range if predicted_pnl_range else (None, None)
            
            cursor.execute("""
                INSERT INTO ml_predictions (
                    execution_id, underlying, strategy_type, prediction_time,
                    model_version, model_type, direction_prediction, confidence_score,
                    rule_confidence, blended_confidence, predicted_pnl_range_low,
                    predicted_pnl_range_high, exit_recommendation, top_features_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                execution_id,
                underlying,
                strategy_type,
                datetime.now().isoformat(),
                model_version,
                model_type,
                direction_prediction,
                confidence_score,
                rule_confidence,
                blended_confidence,
                pnl_low,
                pnl_high,
                exit_recommendation,
                json.dumps(top_features) if top_features else None,
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to save ML prediction: {e}")
            return False
    
    def update_ml_prediction_outcome(
        self,
        execution_id: str,
        actual_outcome: str,
        actual_pnl: float,
        prediction_accurate: bool
    ) -> bool:
        """
        Update ML prediction with actual outcome.
        
        Args:
            execution_id: Trade execution ID
            actual_outcome: 'WIN', 'LOSS', 'BREAKEVEN'
            actual_pnl: Realized P&L
            prediction_accurate: Whether prediction was correct
            
        Returns:
            Success status
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE ml_predictions 
                SET actual_outcome = ?, actual_pnl = ?, prediction_accurate = ?
                WHERE execution_id = ?
            """, (actual_outcome, actual_pnl, 1 if prediction_accurate else 0, execution_id))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to update ML prediction outcome: {e}")
            return False
    
    def save_model_performance(
        self,
        model_version: str,
        model_type: str,
        training_samples: int,
        metrics: Dict[str, float],
        backtest_metrics: Dict[str, float] = None,
    ) -> bool:
        """
        Save or update model performance metrics.
        
        Args:
            model_version: Model version string
            model_type: Type of model
            training_samples: Number of training samples
            metrics: Validation metrics dict
            backtest_metrics: Optional backtest metrics dict
            
        Returns:
            Success status
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            backtest = backtest_metrics or {}
            
            cursor.execute("""
                INSERT OR REPLACE INTO ml_model_performance (
                    model_version, model_type, trained_at, training_samples,
                    accuracy, precision_score, recall_score, f1_score, auc_roc,
                    backtest_win_rate, backtest_sharpe, backtest_sortino,
                    backtest_max_drawdown, backtest_profit_factor,
                    backtest_total_trades, backtest_total_pnl, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_version,
                model_type,
                datetime.now().isoformat(),
                training_samples,
                metrics.get("accuracy", 0),
                metrics.get("precision", 0),
                metrics.get("recall", 0),
                metrics.get("f1_score", 0),
                metrics.get("auc_roc", 0),
                backtest.get("win_rate", 0),
                backtest.get("sharpe", 0),
                backtest.get("sortino", 0),
                backtest.get("max_drawdown", 0),
                backtest.get("profit_factor", 0),
                backtest.get("total_trades", 0),
                backtest.get("total_pnl", 0),
                datetime.now().isoformat(),
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model performance: {e}")
            return False
    
    def get_training_data(
        self,
        underlying: str = None,
        strategy_type: str = None,
        min_date: datetime = None,
        max_date: datetime = None,
        outcome_only: bool = True,
        limit: int = 10000
    ) -> List[Dict]:
        """
        Get ML training data from feature snapshots.
        
        Args:
            underlying: Filter by underlying (optional)
            strategy_type: Filter by strategy (optional)
            min_date: Minimum date filter
            max_date: Maximum date filter
            outcome_only: Only return records with outcomes
            limit: Maximum records to return
            
        Returns:
            List of feature dictionaries with outcomes
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            query = "SELECT * FROM ml_features WHERE snapshot_type = 'entry'"
            params = []
            
            if outcome_only:
                query += " AND outcome IS NOT NULL"
            
            if underlying:
                query += " AND underlying = ?"
                params.append(underlying)
            
            if strategy_type:
                query += " AND strategy_type = ?"
                params.append(strategy_type)
            
            if min_date:
                query += " AND snapshot_time >= ?"
                params.append(min_date.isoformat())
            
            if max_date:
                query += " AND snapshot_time <= ?"
                params.append(max_date.isoformat())
            
            query += f" ORDER BY snapshot_time DESC LIMIT {limit}"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            results = []
            for row in rows:
                row_dict = dict(row)
                # Parse features JSON
                if row_dict.get("features_json"):
                    row_dict["features"] = json.loads(row_dict["features_json"])
                results.append(row_dict)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get training data: {e}")
            return []
    
    def get_model_performance(
        self,
        model_version: str = None,
        active_only: bool = False
    ) -> List[Dict]:
        """
        Get model performance records.
        
        Args:
            model_version: Specific version to get
            active_only: Only return active models
            
        Returns:
            List of performance records
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            query = "SELECT * FROM ml_model_performance WHERE 1=1"
            params = []
            
            if model_version:
                query += " AND model_version = ?"
                params.append(model_version)
            
            if active_only:
                query += " AND is_active = 1"
            
            query += " ORDER BY trained_at DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to get model performance: {e}")
            return []
    
    def set_active_model(self, model_version: str) -> bool:
        """
        Set a model as the active model (deactivates others).
        
        Args:
            model_version: Version to activate
            
        Returns:
            Success status
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Deactivate all models
            cursor.execute("UPDATE ml_model_performance SET is_active = 0")
            
            # Activate specified model
            cursor.execute(
                "UPDATE ml_model_performance SET is_active = 1, stage = 'production' WHERE model_version = ?",
                (model_version,)
            )
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to set active model: {e}")
            return False
    
    def save_market_data(
        self,
        symbol: str,
        data_type: str,
        interval: str,
        date: datetime,
        ohlcv: Dict[str, float],
        oi: int = None,
        iv: float = None,
    ) -> bool:
        """
        Cache market data for historical analysis.
        
        Args:
            symbol: Trading symbol
            data_type: 'equity', 'index', 'option'
            interval: 'day', '15minute', etc.
            date: Date of the data
            ohlcv: Dict with open, high, low, close, volume
            oi: Optional open interest
            iv: Optional implied volatility
            
        Returns:
            Success status
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO market_data_cache (
                    symbol, data_type, interval, date, open, high, low, close, volume, oi, iv
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                data_type,
                interval,
                date.strftime("%Y-%m-%d") if isinstance(date, datetime) else date,
                ohlcv.get("open", 0),
                ohlcv.get("high", 0),
                ohlcv.get("low", 0),
                ohlcv.get("close", 0),
                ohlcv.get("volume", 0),
                oi,
                iv,
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to save market data: {e}")
            return False
    
    def get_cached_market_data(
        self,
        symbol: str,
        interval: str = "day",
        start_date: datetime = None,
        end_date: datetime = None,
    ) -> List[Dict]:
        """
        Get cached market data.
        
        Args:
            symbol: Trading symbol
            interval: Data interval
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            List of OHLCV data dicts
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            query = "SELECT * FROM market_data_cache WHERE symbol = ? AND interval = ?"
            params = [symbol, interval]
            
            if start_date:
                query += " AND date >= ?"
                params.append(start_date.strftime("%Y-%m-%d"))
            
            if end_date:
                query += " AND date <= ?"
                params.append(end_date.strftime("%Y-%m-%d"))
            
            query += " ORDER BY date ASC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to get cached market data: {e}")
            return []
    
    def get_prediction_accuracy(
        self,
        model_version: str = None,
        days: int = 30
    ) -> Dict[str, float]:
        """
        Calculate prediction accuracy statistics.
        
        Args:
            model_version: Filter by model version
            days: Number of days to look back
            
        Returns:
            Dict with accuracy metrics
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            min_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            query = """
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN prediction_accurate = 1 THEN 1 ELSE 0 END) as correct,
                    AVG(actual_pnl) as avg_pnl,
                    SUM(actual_pnl) as total_pnl
                FROM ml_predictions 
                WHERE prediction_time >= ? AND actual_outcome IS NOT NULL
            """
            params = [min_date]
            
            if model_version:
                query += " AND model_version = ?"
                params.append(model_version)
            
            cursor.execute(query, params)
            row = cursor.fetchone()
            conn.close()
            
            if row and row["total"] > 0:
                return {
                    "total_predictions": row["total"],
                    "correct_predictions": row["correct"] or 0,
                    "accuracy": (row["correct"] or 0) / row["total"],
                    "avg_pnl": row["avg_pnl"] or 0,
                    "total_pnl": row["total_pnl"] or 0,
                }
            
            return {
                "total_predictions": 0,
                "correct_predictions": 0,
                "accuracy": 0,
                "avg_pnl": 0,
                "total_pnl": 0,
            }
            
        except Exception as e:
            logger.error(f"Failed to get prediction accuracy: {e}")
            return {}


# Singleton instance
database = Database()
