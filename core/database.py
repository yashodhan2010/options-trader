"""
Database models for persisting trades and signals
"""
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
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


# Singleton instance
database = Database()
