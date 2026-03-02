"""
Order Manager - Handles order placement, modifications, and execution
"""
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import time

from auth.kite_auth import get_kite, is_authenticated
from strategies.base_strategy import StrategySignal, OptionLeg, TradeDirection
from config.settings import TRADING_CONFIG, UNDERLYING_ASSETS, get_options_exchange
from core.logger import logger


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "PENDING"
    PLACED = "PLACED"
    COMPLETE = "COMPLETE"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"
    SL_M = "SL-M"


@dataclass
class Order:
    """Represents a single order."""
    order_id: str = ""
    symbol: str = ""
    exchange: str = "NFO"
    transaction_type: str = ""  # BUY or SELL
    quantity: int = 0
    order_type: OrderType = OrderType.MARKET
    price: float = 0
    trigger_price: float = 0
    status: OrderStatus = OrderStatus.PENDING
    placed_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    filled_price: float = 0
    message: str = ""
    parent_signal_id: str = ""
    leg_index: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "exchange": self.exchange,
            "transaction_type": self.transaction_type,
            "quantity": self.quantity,
            "order_type": self.order_type.value,
            "price": self.price,
            "trigger_price": self.trigger_price,
            "status": self.status.value,
            "placed_at": self.placed_at.isoformat() if self.placed_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "filled_price": self.filled_price,
            "message": self.message,
        }


@dataclass
class TradeExecution:
    """Represents a complete trade execution from a signal."""
    signal: StrategySignal
    execution_id: str = ""
    orders: List[Order] = field(default_factory=list)
    sl_orders: List[Order] = field(default_factory=list)
    target_orders: List[Order] = field(default_factory=list)
    status: str = "PENDING"
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    realized_pnl: float = 0
    current_pnl: float = 0
    
    def is_complete(self) -> bool:
        return all(o.status == OrderStatus.COMPLETE for o in self.orders)
    
    def is_failed(self) -> bool:
        return any(o.status in [OrderStatus.REJECTED, OrderStatus.FAILED] for o in self.orders)


class OrderManager:
    """
    Manages order placement, stop loss, and target execution.
    """
    
    def __init__(self):
        self.kite = None
        self.active_executions: Dict[str, TradeExecution] = {}
        self.order_history: List[Order] = []
        self.is_paper_trading: bool = True  # Start with paper trading
    
    def _ensure_connected(self) -> bool:
        """Ensure Kite connection is established."""
        if not self.kite:
            if is_authenticated():
                self.kite = get_kite()
        return self.kite is not None
    
    def set_paper_trading(self, enabled: bool) -> None:
        """Enable or disable paper trading mode."""
        self.is_paper_trading = enabled
        logger.info(f"Paper trading mode: {'enabled' if enabled else 'disabled'}")
    
    def has_duplicate_position(self, signal: StrategySignal) -> bool:
        """
        Check if there's already an open position with same underlying, 
        strategy type, and strike prices.
        
        Args:
            signal: The signal to check for duplicates
            
        Returns:
            True if duplicate exists, False otherwise
        """
        # Get strikes from the new signal
        new_strikes = set(leg.strike for leg in signal.legs)
        
        for exec_id, execution in self.active_executions.items():
            if execution.status != "ACTIVE":
                continue
            
            existing_signal = execution.signal
            
            # Check if same underlying and strategy type
            if (existing_signal.underlying == signal.underlying and 
                existing_signal.strategy_type == signal.strategy_type):
                
                # Check if same strikes
                existing_strikes = set(leg.strike for leg in existing_signal.legs)
                
                if new_strikes == existing_strikes:
                    logger.warning(
                        f"Duplicate position exists: {exec_id} "
                        f"({signal.underlying} {signal.strategy_type.value} @ {sorted(new_strikes)})"
                    )
                    return True
        
        return False
    
    def execute_signal(
        self,
        signal: StrategySignal,
        order_type: OrderType = OrderType.MARKET,
    ) -> TradeExecution:
        """
        Execute a trading signal by placing orders for all legs.
        
        Args:
            signal: The strategy signal to execute
            order_type: Type of order to place
            
        Returns:
            TradeExecution object with order details
        """
        execution_id = f"{signal.strategy_type.value}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        execution = TradeExecution(signal=signal, execution_id=execution_id)
        
        # Check for duplicate positions
        if self.has_duplicate_position(signal):
            execution.status = "DUPLICATE"
            logger.warning(f"Skipping duplicate trade: {signal.strategy_type.value} for {signal.underlying}")
            return execution
        
        logger.info(f"Executing signal: {signal.strategy_type.value} for {signal.underlying}")
        
        try:
            # Determine the correct F&O exchange for this signal's underlying
            signal_exchange = get_options_exchange(signal.underlying)
            
            # Place orders for each leg
            for i, leg in enumerate(signal.legs):
                order = self._place_leg_order(leg, order_type, execution_id, i, exchange=signal_exchange)
                execution.orders.append(order)
                
                if order.status == OrderStatus.REJECTED:
                    execution.status = "FAILED"
                    logger.error(f"Order rejected for {leg.symbol}: {order.message}")
                    # Cancel any already placed orders
                    self._cancel_execution(execution)
                    return execution
            
            # Wait for orders to complete
            if not self.is_paper_trading:
                self._wait_for_completion(execution)
            
            if execution.is_complete() or self.is_paper_trading:
                execution.status = "ACTIVE"
                execution.entry_time = datetime.now()
                
                # Place stop loss and target orders
                self._place_sl_target_orders(execution)
                
                # Store active execution
                self.active_executions[execution_id] = execution
                
                # Persist to database for overnight recovery
                self._persist_trade_to_db(execution_id, signal)
                
                # Fetch current market data for exit signal tracking
                market_data = self._get_market_data_for_exit_tracking(signal.underlying)
                
                # Notify position tracker of new position for WebSocket subscription
                from execution.position_tracker import position_tracker
                position_tracker.on_new_position(execution_id, market_data)
                
                logger.info(f"Signal executed successfully: {execution_id}")
            else:
                execution.status = "PARTIAL"
                logger.warning(f"Partial execution: {execution_id}")
            
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            execution.status = "FAILED"
        
        return execution
    
    def _get_market_data_for_exit_tracking(self, underlying: str) -> dict:
        """
        Fetch current market data to store as entry conditions for exit signal tracking.
        
        Args:
            underlying: The underlying asset
            
        Returns:
            Dict with spot, OI, volatility, and historical data
        """
        try:
            from data.data_fetcher import data_fetcher
            return {
                "spot": data_fetcher.get_spot_price(underlying),
                "oi_data": data_fetcher.get_oi_analysis(underlying) or {},
                "volatility": data_fetcher.get_volatility_data(underlying) or {},
                "historical": data_fetcher.get_historical_analysis(underlying, days=5) or {},
            }
        except Exception as e:
            logger.debug(f"Could not fetch market data for exit tracking: {e}")
            return {}
    
    def _place_leg_order(
        self,
        leg: OptionLeg,
        order_type: OrderType,
        execution_id: str,
        leg_index: int,
        exchange: str = "NFO",
    ) -> Order:
        """
        Place order for a single leg.
        
        Args:
            leg: The option leg
            order_type: Type of order
            execution_id: Parent execution ID
            leg_index: Index of the leg
            exchange: F&O exchange (NFO or BFO)
            
        Returns:
            Order object
        """
        order = Order(
            symbol=leg.symbol,
            exchange=exchange,
            transaction_type=leg.direction.value,
            quantity=leg.quantity,
            order_type=order_type,
            price=leg.entry_price if order_type == OrderType.LIMIT else 0,
            parent_signal_id=execution_id,
            leg_index=leg_index,
        )
        
        if self.is_paper_trading:
            # Simulate order execution
            order.order_id = f"PAPER_{execution_id}_{leg_index}"
            order.status = OrderStatus.COMPLETE
            order.filled_price = leg.entry_price
            order.placed_at = datetime.now()
            order.filled_at = datetime.now()
            logger.info(f"[PAPER] Order placed: {leg.direction.value} {leg.quantity} {leg.symbol} @ {leg.entry_price}")
        else:
            # Place real order via Kite
            try:
                if not self._ensure_connected():
                    order.status = OrderStatus.FAILED
                    order.message = "Not connected to Kite"
                    return order
                
                order_id = self.kite.place_order(
                    variety="regular",
                    exchange=exchange,
                    tradingsymbol=leg.symbol,
                    transaction_type=leg.direction.value,
                    quantity=leg.quantity,
                    product="NRML",
                    order_type=order_type.value,
                    price=leg.entry_price if order_type == OrderType.LIMIT else None,
                )
                
                order.order_id = str(order_id)
                order.status = OrderStatus.PLACED
                order.placed_at = datetime.now()
                
                logger.info(f"Order placed: {order.order_id} - {leg.direction.value} {leg.quantity} {leg.symbol}")
                
            except Exception as e:
                order.status = OrderStatus.FAILED
                order.message = str(e)
                logger.error(f"Failed to place order: {e}")
        
        self.order_history.append(order)

        # Persist order for audit trail when we have an order ID
        if order.order_id:
            try:
                from core.database import database
                from config.settings import BOT_CONFIG

                if BOT_CONFIG.get("persist_positions", True):
                    database.save_order({
                        "order_id": order.order_id,
                        "execution_id": execution_id,
                        "symbol": order.symbol,
                        "transaction_type": order.transaction_type,
                        "quantity": order.quantity,
                        "price": order.filled_price or order.price,
                        "status": order.status.value,
                        "order_type": order.order_type.value,
                        "placed_at": order.placed_at or datetime.now(),
                    })
            except Exception as e:
                logger.debug(f"Failed to persist order {order.order_id}: {e}")

        return order
    
    def _wait_for_completion(self, execution: TradeExecution, timeout: int = 30) -> None:
        """Wait for all orders to complete."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            all_complete = True
            
            for order in execution.orders:
                if order.status == OrderStatus.PLACED:
                    # Check order status
                    order_details = self._get_order_status(order.order_id)
                    if order_details:
                        if order_details.get("status") == "COMPLETE":
                            order.status = OrderStatus.COMPLETE
                            order.filled_price = order_details.get("average_price", 0)
                            order.filled_at = datetime.now()
                        elif order_details.get("status") == "REJECTED":
                            order.status = OrderStatus.REJECTED
                            order.message = order_details.get("status_message", "")
                        else:
                            all_complete = False
                    else:
                        all_complete = False
            
            if all_complete:
                break
            
            time.sleep(0.5)
    
    def _get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get order status from Kite."""
        if not self._ensure_connected():
            return None
        
        try:
            orders = self.kite.orders()
            for order in orders:
                if str(order.get("order_id")) == order_id:
                    return order
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
        
        return None
    
    def _place_sl_target_orders(self, execution: TradeExecution) -> None:
        """Place stop loss and target orders for an execution."""
        signal = execution.signal
        
        # For now, we'll monitor SL/target in the position tracker
        # In production, you might want to place GTT orders
        
        logger.info(f"SL: {signal.stop_loss:.2f}, Target: {signal.target:.2f}")
    
    def _persist_trade_to_db(self, execution_id: str, signal: StrategySignal) -> None:
        """
        Persist trade to database for overnight recovery.
        
        Args:
            execution_id: Execution ID
            signal: Strategy signal
        """
        try:
            from core.database import database
            from config.settings import BOT_CONFIG
            
            if not BOT_CONFIG.get("persist_positions", True):
                return
            
            signal_data = self.get_execution_signal_data(execution_id)
            if signal_data:
                database.save_trade(
                    execution_id=execution_id,
                    underlying=signal.underlying,
                    strategy_type=signal.strategy_type.value,
                    signal_data=signal_data,
                    status="ACTIVE",
                )
                logger.debug(f"Trade persisted to database: {execution_id}")
        except Exception as e:
            logger.error(f"Failed to persist trade to database: {e}")
    
    def _update_trade_in_db(
        self,
        execution_id: str,
        status: str,
        realized_pnl: float,
        exit_reason: Optional[str] = None,
    ) -> None:
        """
        Update trade status in database.
        
        Args:
            execution_id: Execution ID
            status: New status
            realized_pnl: Realized P&L
        """
        try:
            from core.database import database
            from config.settings import BOT_CONFIG
            
            if not BOT_CONFIG.get("persist_positions", True):
                return
            
            database.update_trade(
                execution_id=execution_id,
                status=status,
                exit_time=datetime.now(),
                realized_pnl=realized_pnl,
                exit_reason=exit_reason,
            )
            logger.debug(f"Trade updated in database: {execution_id} -> {status}")
        except Exception as e:
            logger.error(f"Failed to update trade in database: {e}")
    
    def _cancel_execution(self, execution: TradeExecution) -> None:
        """Cancel all orders in an execution."""
        for order in execution.orders:
            if order.status == OrderStatus.PLACED and order.order_id:
                self._cancel_order(order.order_id)
                order.status = OrderStatus.CANCELLED
    
    def _cancel_order(self, order_id: str) -> bool:
        """Cancel a specific order."""
        if self.is_paper_trading:
            logger.info(f"[PAPER] Order cancelled: {order_id}")
            return True
        
        if not self._ensure_connected():
            return False
        
        try:
            self.kite.cancel_order(variety="regular", order_id=order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    def close_position(
        self,
        execution_id: str,
        order_type: OrderType = OrderType.MARKET,
        reason: str = "manual",
    ) -> bool:
        """
        Close an active position.
        
        Args:
            execution_id: ID of the execution to close
            order_type: Type of order to use for closing
            
        Returns:
            True if successful
        """
        execution = self.active_executions.get(execution_id)
        if not execution:
            logger.warning(f"Execution not found: {execution_id}")
            return False
        
        logger.info(f"Closing position: {execution_id}")
        
        try:
            # Determine exchange from the signal's underlying
            exit_exchange = get_options_exchange(execution.signal.underlying)
            
            # Place exit orders for each leg (opposite direction)
            for i, leg in enumerate(execution.signal.legs):
                exit_direction = TradeDirection.SELL if leg.is_long else TradeDirection.BUY
                
                exit_leg = OptionLeg(
                    symbol=leg.symbol,
                    strike=leg.strike,
                    option_type=leg.option_type,
                    expiry=leg.expiry,
                    direction=exit_direction,
                    quantity=leg.quantity,
                    entry_price=leg.current_price or leg.entry_price,
                )
                
                order = self._place_leg_order(exit_leg, order_type, f"{execution_id}_exit", i, exchange=exit_exchange)
                
                if order.status in [OrderStatus.COMPLETE, OrderStatus.PLACED]:
                    # Calculate P&L
                    if leg.entry_price and order.filled_price:
                        pnl = (order.filled_price - leg.entry_price) * leg.quantity
                        if leg.is_short:
                            pnl = -pnl
                        execution.realized_pnl += pnl
            
            execution.status = "CLOSED"
            execution.exit_time = datetime.now()
            
            # Update database
            self._update_trade_in_db(execution_id, "CLOSED", execution.realized_pnl, reason)
            
            # Remove from active executions
            del self.active_executions[execution_id]
            
            logger.info(f"Position closed. Realized P&L: {execution.realized_pnl:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    def get_active_positions(self) -> List[Dict]:
        """Get all active positions."""
        return [
            {
                "execution_id": exec_id,
                "strategy": execution.signal.strategy_type.value,
                "underlying": execution.signal.underlying,
                "entry_time": execution.entry_time.isoformat() if execution.entry_time else None,
                "legs": [
                    {
                        "symbol": leg.symbol,
                        "direction": leg.direction.value,
                        "quantity": leg.quantity,
                        "entry_price": leg.entry_price,
                    }
                    for leg in execution.signal.legs
                ],
                "stop_loss": execution.signal.stop_loss,
                "target": execution.signal.target,
                "status": execution.status,
            }
            for exec_id, execution in self.active_executions.items()
        ]
    
    def get_order_history(self) -> List[Dict]:
        """Get order history."""
        return [order.to_dict() for order in self.order_history]
    
    def modify_sl_target(
        self,
        execution_id: str,
        new_sl: Optional[float] = None,
        new_target: Optional[float] = None,
    ) -> bool:
        """
        Modify stop loss or target for an active position.
        
        Args:
            execution_id: Execution ID
            new_sl: New stop loss value
            new_target: New target value
            
        Returns:
            True if successful
        """
        execution = self.active_executions.get(execution_id)
        if not execution:
            logger.warning(f"Execution not found: {execution_id}")
            return False
        
        if new_sl is not None:
            execution.signal.stop_loss = new_sl
            logger.info(f"Stop loss updated to {new_sl}")
        
        if new_target is not None:
            execution.signal.target = new_target
            logger.info(f"Target updated to {new_target}")
        
        return True
    
    def load_persisted_position(self, execution_id: str, signal_data: Dict) -> bool:
        """
        Load a persisted position from database (for overnight recovery).
        
        Args:
            execution_id: Execution ID
            signal_data: Signal data dictionary from database
            
        Returns:
            True if successful
        """
        try:
            from strategies.base_strategy import StrategySignal, OptionLeg, TradeDirection, StrategyType
            
            # Reconstruct legs
            legs = []
            for leg_data in signal_data.get("legs", []):
                leg = OptionLeg(
                    symbol=leg_data.get("symbol", ""),
                    strike=leg_data.get("strike", 0),
                    option_type=leg_data.get("option_type", "CE"),
                    expiry=leg_data.get("expiry", ""),
                    direction=TradeDirection(leg_data.get("direction", "BUY")),
                    quantity=leg_data.get("quantity", 0),
                    entry_price=leg_data.get("entry_price", 0),
                )
                leg.current_price = leg_data.get("current_price", leg.entry_price)
                legs.append(leg)
            
            # Parse entry time
            entry_time_str = signal_data.get("entry_time")
            if entry_time_str:
                try:
                    entry_time = datetime.fromisoformat(entry_time_str)
                except:
                    entry_time = datetime.now()
            else:
                entry_time = datetime.now()
            
            # Reconstruct signal
            # Note: risk_reward_ratio is a computed @property, not a constructor param
            signal = StrategySignal(
                strategy_type=StrategyType(signal_data.get("strategy_type", "LONG_CALL")),
                underlying=signal_data.get("underlying", ""),
                legs=legs,
                entry_time=entry_time,
                confidence=signal_data.get("confidence", 0),
                expected_profit=signal_data.get("expected_profit", signal_data.get("target", 0)),
                max_loss=signal_data.get("max_loss", signal_data.get("stop_loss", 0)),
                stop_loss=signal_data.get("stop_loss", 0),
                target=signal_data.get("target", 0),
                rationale=signal_data.get("rationale", "Loaded from database"),
            )
            
            # Create execution
            execution = TradeExecution(signal=signal, execution_id=execution_id)
            execution.status = "ACTIVE"
            execution.entry_time = entry_time
            
            # Add to active executions
            self.active_executions[execution_id] = execution
            
            logger.info(f"Loaded persisted position: {execution_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load persisted position: {e}")
            return False
    
    def get_execution_signal_data(self, execution_id: str) -> Optional[Dict]:
        """
        Get signal data for an execution (for database persistence).
        
        Args:
            execution_id: Execution ID
            
        Returns:
            Signal data dictionary
        """
        execution = self.active_executions.get(execution_id)
        if not execution:
            return None
        
        signal = execution.signal
        return {
            "strategy_type": signal.strategy_type.value,
            "underlying": signal.underlying,
            "legs": [
                {
                    "symbol": leg.symbol,
                    "strike": leg.strike,
                    "option_type": leg.option_type,
                    "expiry": leg.expiry.isoformat() if hasattr(leg.expiry, 'isoformat') else str(leg.expiry),
                    "direction": leg.direction.value,
                    "quantity": leg.quantity,
                    "entry_price": leg.entry_price,
                    "current_price": leg.current_price,
                }
                for leg in signal.legs
            ],
            "confidence": signal.confidence,
            "expected_profit": signal.expected_profit,
            "max_loss": signal.max_loss,
            "risk_reward_ratio": signal.risk_reward_ratio,
            "stop_loss": signal.stop_loss,
            "target": signal.target,
            "rationale": signal.rationale,
            "entry_time": execution.entry_time.isoformat() if execution.entry_time else None,
        }


# Singleton instance
order_manager = OrderManager()
