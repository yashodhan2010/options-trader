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
    gtt_order_ids: List[int] = field(default_factory=list)
    trading_mode: str = "PAPER"
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
    
    @property
    def trading_mode(self) -> str:
        """Return 'PAPER' or 'LIVE' based on current mode."""
        return "PAPER" if self.is_paper_trading else "LIVE"
    
    def _ensure_connected(self) -> bool:
        """Ensure Kite connection is established."""
        if not self.kite:
            if is_authenticated():
                self.kite = get_kite()
        return self.kite is not None
    
    def _get_underlying_from_execution(self, execution_id: str) -> Optional[str]:
        """Get underlying symbol from an active or exit execution."""
        # Strip _exit suffix to find the parent execution
        base_id = execution_id.replace("_exit", "")
        execution = self.active_executions.get(base_id)
        if execution:
            return execution.signal.underlying
        return None

    def set_paper_trading(self, enabled: bool) -> None:
        """Enable or disable paper trading mode."""
        self.is_paper_trading = enabled
        logger.info(f"Paper trading mode: {'enabled' if enabled else 'disabled'}")
    
    def _check_margin_available(self, signal: StrategySignal) -> bool:
        """
        Check if sufficient margin is available before placing live orders.
        
        Uses Kite's basket_order_margins API to get the real combined margin
        (with spread benefit). Falls back to per-leg order_margins if the
        basket API fails, and to a conservative estimate as last resort.
        
        Args:
            signal: The strategy signal to check
            
        Returns:
            True if margin is sufficient
        """
        if not self._ensure_connected():
            logger.error("Cannot check margin - not connected")
            return False
        
        try:
            from auth.kite_auth import get_margins
            
            margins = get_margins()
            if not margins:
                logger.error("Could not fetch account margins")
                return False
            
            equity_margin = margins.get("equity", {})
            available = equity_margin.get("available", {}).get("live_balance", 0)
            
            signal_exchange = get_options_exchange(signal.underlying)
            
            # Build basket order params for Kite margin API
            basket_params = []
            for leg in signal.legs:
                basket_params.append({
                    "exchange": signal_exchange,
                    "tradingsymbol": leg.symbol,
                    "transaction_type": leg.direction.value,
                    "quantity": leg.quantity,
                    "product": "NRML",
                    "order_type": "LIMIT",
                    "price": leg.entry_price or 0,
                })
            
            # Primary: basket_order_margins — real combined margin with spread benefit
            required_margin = None
            try:
                basket_result = self.kite.basket_order_margins(
                    basket_params, consider_positions=True, mode="compact"
                )
                if isinstance(basket_result, dict):
                    required_margin = (
                        basket_result.get("final", {}).get("total", 0)
                        or basket_result.get("initial", {}).get("total", 0)
                    )
                if required_margin:
                    logger.info(f"Basket margin (combined): Rs.{required_margin:,.2f}")
            except Exception as e:
                logger.warning(f"Basket margin API failed: {e} — falling back to per-leg")
            
            # Fallback: sum of per-leg order_margins
            if not required_margin:
                try:
                    leg_margins = self.kite.order_margins(basket_params)
                    required_margin = sum(m.get("total", 0) for m in leg_margins)
                    logger.info(f"Per-leg margin (sum): Rs.{required_margin:,.2f}")
                except Exception as e:
                    logger.warning(f"Per-leg margin API also failed: {e}")
            
            # Last resort: conservative estimate (buy premium * 2)
            if not required_margin:
                required_margin = sum(
                    (leg.entry_price or 0) * leg.quantity
                    for leg in signal.legs if leg.is_long
                ) * 2
                logger.info(f"Fallback margin estimate: Rs.{required_margin:,.2f}")
            
            # 10% buffer for price fluctuation between check and execution
            total_required = required_margin * 1.1
            
            if available < total_required:
                logger.warning(
                    f"Insufficient margin: available Rs.{available:,.2f}, "
                    f"required Rs.{total_required:,.2f} (margin Rs.{required_margin:,.2f} + 10% buffer)"
                )
                return False
            
            logger.info(
                f"Margin check passed: available Rs.{available:,.2f}, "
                f"required Rs.{total_required:,.2f}"
            )
            return True
            
        except Exception as e:
            logger.error(f"Margin check failed: {e}")
            return False
    
    def has_duplicate_position(self, signal: StrategySignal) -> bool:
        """
        Check if there's already an open position with the same underlying
        AND same strategy type in the SAME trading mode. Different strategies
        on the same underlying are allowed (different risk/reward profiles).
        Paper and live positions are independent — a paper trade does not
        block a live trade.
        
        Args:
            signal: The signal to check for duplicates
            
        Returns:
            True if duplicate exists, False otherwise
        """
        for exec_id, execution in self.active_executions.items():
            if execution.status != "ACTIVE":
                continue
            
            # Only check duplicates within the same trading mode
            if execution.trading_mode != self.trading_mode:
                continue
            
            existing_signal = execution.signal
            
            if (existing_signal.underlying == signal.underlying and
                existing_signal.strategy_type == signal.strategy_type):
                logger.warning(
                    f"Duplicate position exists: {exec_id} "
                    f"({signal.underlying} {signal.strategy_type.value}) [{self.trading_mode}]"
                )
                return True
        
        return False
    
    def execute_signal(
        self,
        signal: StrategySignal,
        order_type: OrderType = None,
    ) -> TradeExecution:
        """
        Execute a trading signal by placing orders for all legs.
        
        For live trading, uses the order type from TRADING_CONFIG:
        - LIMIT (default): Places limit order at LTP + slippage tolerance for better fills
        - MARKET: Immediate fill but no price control
        
        Args:
            signal: The strategy signal to execute
            order_type: Type of order to place
            
        Returns:
            TradeExecution object with order details
        """
        execution_id = f"{signal.strategy_type.value}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        execution = TradeExecution(signal=signal, execution_id=execution_id, trading_mode=self.trading_mode)
        
        # Check for duplicate positions
        if self.has_duplicate_position(signal):
            execution.status = "DUPLICATE"
            logger.warning(f"Skipping duplicate trade: {signal.strategy_type.value} for {signal.underlying}")
            return execution
        
        # Determine order type: use config default for live, MARKET for paper
        if order_type is None:
            if self.is_paper_trading:
                order_type = OrderType.MARKET
            else:
                entry_type_str = TRADING_CONFIG.get("entry_order_type", "LIMIT")
                order_type = OrderType.LIMIT if entry_type_str == "LIMIT" else OrderType.MARKET
        
        # For LIMIT orders in live mode, adjust entry prices with slippage tolerance
        if not self.is_paper_trading and order_type == OrderType.LIMIT:
            slippage_pct = TRADING_CONFIG.get("limit_slippage_pct", 1.0) / 100
            for leg in signal.legs:
                if leg.entry_price and leg.entry_price > 0:
                    if leg.is_long:
                        # Buying: willing to pay up to slippage% above LTP
                        leg.entry_price = round(leg.entry_price * (1 + slippage_pct), 2)
                    else:
                        # Selling: willing to accept slippage% below LTP
                        leg.entry_price = round(leg.entry_price * (1 - slippage_pct), 2)
                    # Kite NFO tick size is 0.05
                    leg.entry_price = round(leg.entry_price / 0.05) * 0.05
        
        logger.info(f"Executing signal: {signal.strategy_type.value} for {signal.underlying} (order_type={order_type.value})")
        
        try:
            # Pre-flight DTE check — reject trades with insufficient days to expiry
            # High-confidence signals (>= 80%) get a lower DTE floor (5 days)
            from config.settings import STRATEGY_CONFIG
            confidence = signal.confidence if hasattr(signal, 'confidence') else 0
            high_conf_threshold = STRATEGY_CONFIG.get("high_confidence_threshold", 0.80)
            if confidence >= high_conf_threshold:
                min_dte = STRATEGY_CONFIG.get("high_confidence_min_dte", 5)
            else:
                min_dte = STRATEGY_CONFIG.get("min_days_to_expiry", 20)
            today = datetime.now().date()
            for leg in signal.legs:
                leg_expiry = leg.expiry.date() if isinstance(leg.expiry, datetime) else leg.expiry
                dte = (leg_expiry - today).days
                if dte < min_dte:
                    execution.status = "FAILED"
                    logger.warning(
                        f"DTE too low for {leg.symbol}: {dte} days "
                        f"(min: {min_dte}, confidence: {confidence:.0%}). Rejecting trade."
                    )
                    return execution

            # Pre-flight margin check for live trading
            if not self.is_paper_trading:
                if not self._check_margin_available(signal):
                    execution.status = "FAILED"
                    logger.error(f"Insufficient margin for {signal.strategy_type.value} on {signal.underlying}")
                    return execution
            
            # Determine the correct F&O exchange for this signal's underlying
            signal_exchange = get_options_exchange(signal.underlying)
            
            # For live trading, reorder legs: place BUY (hedge) legs before
            # SELL legs so the exchange recognises the spread and applies
            # combined margin instead of naked-short margin.
            ordered_legs = list(enumerate(signal.legs))
            if not self.is_paper_trading:
                ordered_legs.sort(key=lambda x: 0 if x[1].is_long else 1)
            
            # Place orders for each leg
            for i, leg in ordered_legs:
                order = self._place_leg_order(leg, order_type, execution_id, i, exchange=signal_exchange)
                execution.orders.append(order)
                
                if order.status in (OrderStatus.REJECTED, OrderStatus.FAILED):
                    execution.status = "FAILED"
                    logger.error(f"Order rejected for {leg.symbol}: {order.message}")
                    # Cancel any already placed orders
                    self._cancel_execution(execution)
                    return execution
                
                # For live spreads, wait briefly for the BUY leg to be
                # acknowledged before sending the SELL leg so the exchange
                # has the hedge on record for margin benefit.
                if not self.is_paper_trading and leg.is_long:
                    time.sleep(0.5)
            
            # Wait for orders to complete
            if not self.is_paper_trading:
                limit_timeout = TRADING_CONFIG.get("limit_timeout_seconds", 15)
                self._wait_for_completion(execution, timeout=limit_timeout if order_type == OrderType.LIMIT else 30)
                
                # If LIMIT orders didn't fill, cancel and retry as MARKET
                if order_type == OrderType.LIMIT and not execution.is_complete():
                    unfilled = [o for o in execution.orders if o.status == OrderStatus.PLACED]
                    if unfilled:
                        logger.warning(f"{len(unfilled)} LIMIT order(s) unfilled after {limit_timeout}s — converting to MARKET")
                        for order in unfilled:
                            self._cancel_order(order.order_id)
                            order.status = OrderStatus.CANCELLED
                        
                        # Re-place unfilled legs as MARKET
                        for order in unfilled:
                            leg = signal.legs[order.leg_index]
                            market_order = self._place_leg_order(
                                leg, OrderType.MARKET, execution_id, order.leg_index, exchange=signal_exchange
                            )
                            # Replace the cancelled order in the list
                            idx = execution.orders.index(order)
                            execution.orders[idx] = market_order
                        
                        # Wait for MARKET orders
                        self._wait_for_completion(execution, timeout=30)
            
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
                # Cancel all placed orders to avoid orphaned legs on the exchange
                self._cancel_execution(execution)
                logger.info(f"Cancelled orphaned orders for partial execution: {execution_id}")
            
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            execution.status = "FAILED"
            # Cancel any orders that made it to the exchange before the error
            if execution.orders:
                self._cancel_execution(execution)
        
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
        # Stock options don't allow MARKET orders on Kite — convert to LIMIT
        # with market protection (5% slippage) for non-index underlyings.
        effective_order_type = order_type
        effective_price = leg.entry_price if order_type == OrderType.LIMIT else 0

        if order_type == OrderType.MARKET and not self.is_paper_trading:
            # Determine underlying from execution_id or symbol
            underlying = self._get_underlying_from_execution(execution_id)
            if underlying and underlying not in UNDERLYING_ASSETS:
                # Stock option — MARKET orders blocked, use LIMIT with 5% protection
                ref_price = leg.entry_price or 0
                if ref_price > 0:
                    protection_pct = 0.05
                    if leg.direction.value == "BUY":
                        effective_price = round(ref_price * (1 + protection_pct), 2)
                    else:
                        effective_price = round(max(ref_price * (1 - protection_pct), 0.05), 2)
                    effective_order_type = OrderType.LIMIT
                    logger.info(
                        f"Stock option: MARKET→LIMIT with protection. "
                        f"{leg.direction.value} {leg.symbol} @ {effective_price} (ref: {ref_price})"
                    )

        order = Order(
            symbol=leg.symbol,
            exchange=exchange,
            transaction_type=leg.direction.value,
            quantity=leg.quantity,
            order_type=effective_order_type,
            price=effective_price,
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
                    order_type=effective_order_type.value,
                    price=effective_price if effective_order_type == OrderType.LIMIT else None,
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
                        "trading_mode": self.trading_mode,
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
        """
        Place exchange-level GTT stop-loss orders for crash protection.
        
        GTT orders sit on the exchange and trigger even if the bot is down.
        The bot's software-based SL/target monitoring (position_tracker) handles
        normal exits; GTT is a safety net for catastrophic scenarios only.
        
        Converts the strategy-level SL (absolute Rs P&L) to per-leg price triggers.
        """
        signal = execution.signal
        
        if self.is_paper_trading:
            logger.info(f"[PAPER] SL: {signal.stop_loss:.2f}, Target: {signal.target:.2f}")
            return
        
        if not TRADING_CONFIG.get("place_gtt_stop_loss", True):
            logger.info(f"GTT SL disabled. Software SL: {signal.stop_loss:.2f}, Target: {signal.target:.2f}")
            return
        
        if not self._ensure_connected():
            logger.error("Cannot place GTT orders — Kite not connected")
            return
        
        buffer_pct = TRADING_CONFIG.get("gtt_sl_buffer_pct", 2.0) / 100
        exchange = get_options_exchange(signal.underlying)
        num_legs = max(len(signal.legs), 1)
        
        for leg in signal.legs:
            if not leg.entry_price or leg.entry_price <= 0:
                continue
            
            # Strategy SL is total Rs. Divide equally among legs to get per-leg SL.
            sl_per_leg = signal.stop_loss / num_legs
            sl_price_change = sl_per_leg / max(leg.quantity, 1)
            
            if leg.is_long:
                # Long leg loses when price drops → GTT SELL to close
                trigger_price = leg.entry_price - sl_price_change * (1 + buffer_pct)
                trigger_price = max(round(trigger_price / 0.05) * 0.05, 0.05)
                # LIMIT price slightly below trigger for fill certainty
                order_price = max(trigger_price - 0.05, 0.05)
                txn_type = "SELL"
            else:
                # Short leg loses when price rises → GTT BUY to close
                trigger_price = leg.entry_price + sl_price_change * (1 + buffer_pct)
                trigger_price = round(trigger_price / 0.05) * 0.05
                # LIMIT price slightly above trigger for fill certainty
                order_price = trigger_price + 0.05
                txn_type = "BUY"
            
            try:
                gtt_response = self.kite.place_gtt(
                    trigger_type="single",
                    tradingsymbol=leg.symbol,
                    exchange=exchange,
                    trigger_values=[trigger_price],
                    last_price=leg.entry_price,
                    orders=[{
                        "exchange": exchange,
                        "tradingsymbol": leg.symbol,
                        "transaction_type": txn_type,
                        "quantity": leg.quantity,
                        "order_type": "LIMIT",
                        "product": "NRML",
                        "price": order_price,
                    }]
                )
                gtt_id = gtt_response.get("trigger_id")
                if gtt_id:
                    execution.gtt_order_ids.append(gtt_id)
                    logger.info(
                        f"GTT SL placed for {leg.symbol}: trigger={trigger_price:.2f}, "
                        f"{txn_type} {leg.quantity} @ {order_price:.2f} (GTT ID: {gtt_id})"
                    )
            except Exception as e:
                logger.error(f"Failed to place GTT for {leg.symbol}: {e}")
        
        logger.info(
            f"SL: {signal.stop_loss:.2f}, Target: {signal.target:.2f} "
            f"(GTT IDs: {execution.gtt_order_ids})"
        )
    
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
                    trading_mode=self.trading_mode,
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
        """Cancel placed orders and unwind any already-filled legs.
        
        When a spread execution goes PARTIAL (e.g. BUY filled, SELL rejected
        for margin), we must:
        1. Cancel any orders still sitting on the exchange (PLACED)
        2. Re-check PLACED orders that may have filled in the meantime
        3. Exit (reverse-trade) any legs that already filled so we don't
           leave naked orphaned positions
        """
        filled_orders = []
        
        for order in execution.orders:
            if order.status == OrderStatus.PLACED and order.order_id:
                # Try to cancel — may fail if it filled between check and cancel
                cancelled = self._cancel_order(order.order_id)
                if cancelled:
                    order.status = OrderStatus.CANCELLED
                else:
                    # Failed to cancel — order may have filled, re-check status
                    details = self._get_order_status(order.order_id)
                    if details and details.get("status") == "COMPLETE":
                        order.status = OrderStatus.COMPLETE
                        order.filled_price = details.get("average_price", 0)
                        order.filled_at = datetime.now()
            
            if order.status == OrderStatus.COMPLETE:
                filled_orders.append(order)
        
        # Unwind filled legs by placing reverse orders
        if filled_orders and not self.is_paper_trading:
            signal_exchange = get_options_exchange(execution.signal.underlying)
            is_stock_option = execution.signal.underlying not in UNDERLYING_ASSETS
            for order in filled_orders:
                reverse_direction = "SELL" if order.transaction_type == "BUY" else "BUY"
                logger.warning(
                    f"Unwinding filled orphan: {reverse_direction} {order.quantity} "
                    f"{order.symbol} (was {order.transaction_type} @ {order.filled_price})"
                )
                try:
                    # Stock options don't allow MARKET orders — use LIMIT with 5% protection
                    if is_stock_option and order.filled_price:
                        protection_pct = 0.05
                        if reverse_direction == "BUY":
                            unwind_price = round(order.filled_price * (1 + protection_pct), 2)
                        else:
                            unwind_price = round(max(order.filled_price * (1 - protection_pct), 0.05), 2)
                        self.kite.place_order(
                            variety="regular",
                            exchange=signal_exchange,
                            tradingsymbol=order.symbol,
                            transaction_type=reverse_direction,
                            quantity=order.quantity,
                            product="NRML",
                            order_type="LIMIT",
                            price=unwind_price,
                        )
                        logger.info(f"Unwind LIMIT order placed: {reverse_direction} {order.quantity} {order.symbol} @ {unwind_price}")
                    else:
                        self.kite.place_order(
                            variety="regular",
                            exchange=signal_exchange,
                            tradingsymbol=order.symbol,
                            transaction_type=reverse_direction,
                            quantity=order.quantity,
                            product="NRML",
                            order_type="MARKET",
                        )
                        logger.info(f"Unwind order placed: {reverse_direction} {order.quantity} {order.symbol}")
                except Exception as e:
                    logger.error(f"CRITICAL: Failed to unwind orphan {order.symbol}: {e}")
    
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
    
    def _cancel_gtt_orders(self, execution: TradeExecution) -> None:
        """Cancel all GTT orders for an execution to prevent double exits."""
        if self.is_paper_trading or not execution.gtt_order_ids:
            return
        
        if not self._ensure_connected():
            logger.error("Cannot cancel GTT orders — Kite not connected")
            return
        
        for gtt_id in execution.gtt_order_ids:
            try:
                self.kite.delete_gtt(gtt_id)
                logger.info(f"GTT cancelled: {gtt_id}")
            except Exception as e:
                # GTT may have already triggered or expired
                logger.warning(f"Failed to cancel GTT {gtt_id}: {e}")
        
        execution.gtt_order_ids.clear()
    
    def check_gtt_triggers(self) -> List[str]:
        """
        Check if any GTT orders have been triggered on the exchange.
        Returns list of execution_ids whose GTTs have fired.
        
        This detects the case where Zerodha fires a per-leg GTT SL,
        so the bot can exit the remaining legs of the spread.
        """
        if self.is_paper_trading or not self._ensure_connected():
            return []
        
        triggered_executions = []
        
        try:
            gtt_orders = self.kite.get_gtts()
        except Exception as e:
            logger.error(f"Failed to fetch GTT list: {e}")
            return []
        
        # Build a lookup: gtt_id -> status
        gtt_status_map = {}
        for gtt in gtt_orders:
            gtt_id = gtt.get("id")
            status = gtt.get("status")
            if gtt_id:
                gtt_status_map[gtt_id] = status
        
        for exec_id, execution in list(self.active_executions.items()):
            if execution.status != "ACTIVE" or not execution.gtt_order_ids:
                continue
            
            for gtt_id in execution.gtt_order_ids:
                status = gtt_status_map.get(gtt_id)
                if status == "triggered":
                    logger.warning(
                        f"GTT {gtt_id} TRIGGERED for {exec_id} "
                        f"({execution.signal.underlying} {execution.signal.strategy_type.value}) — "
                        f"must exit remaining legs"
                    )
                    if exec_id not in triggered_executions:
                        triggered_executions.append(exec_id)
                    break  # One triggered GTT is enough to flag this execution
        
        return triggered_executions
    
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
        
        # Cancel any GTT orders first to prevent double execution
        self._cancel_gtt_orders(execution)
        
        try:
            # Determine exchange from the signal's underlying
            exit_exchange = get_options_exchange(execution.signal.underlying)
            
            exit_orders = []
            any_failed = False
            
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
                exit_orders.append((order, leg))
                
                if order.status in [OrderStatus.FAILED, OrderStatus.REJECTED]:
                    logger.error(f"Exit order FAILED for {leg.symbol}: {order.message}")
                    any_failed = True
            
            # In live mode, wait for exit orders to fill before calculating P&L
            if not self.is_paper_trading:
                exit_execution = TradeExecution(signal=execution.signal, execution_id=f"{execution_id}_exit")
                exit_execution.orders = [o for o, _ in exit_orders]
                self._wait_for_completion(exit_execution, timeout=60)
            
            # Calculate P&L from actual fill prices
            for order, leg in exit_orders:
                if order.status == OrderStatus.COMPLETE and leg.entry_price and order.filled_price:
                    pnl = (order.filled_price - leg.entry_price) * leg.quantity
                    if leg.is_short:
                        pnl = -pnl
                    execution.realized_pnl += pnl
                elif order.status not in [OrderStatus.COMPLETE]:
                    any_failed = True
                    logger.error(f"Exit order not filled for {leg.symbol}: status={order.status.value}")
            
            if any_failed:
                # Don't remove position from tracking if exit orders failed
                logger.error(f"Some exit orders failed for {execution_id} — position kept in active tracking")
                return False
            
            execution.status = "CLOSED"
            execution.exit_time = datetime.now()
            
            # Update database
            self._update_trade_in_db(execution_id, "CLOSED", execution.realized_pnl, reason)
            
            # Remove from active executions
            del self.active_executions[execution_id]
            
            logger.info(f"Position closed. Realized P&L: {execution.realized_pnl:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error closing position {execution_id}: {e}")
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
    
    def register_manual_position(
        self,
        underlying: str,
        strategy_type_str: str,
        legs_data: List[Dict],
        stop_loss: float,
        target: float,
        confidence: float = 0.5,
        rationale: str = "Manual import",
    ) -> Optional[str]:
        """
        Register a position that was placed manually on Kite.
        Creates the execution in memory and persists to DB so the
        position tracker can monitor SL/target/signal exits.

        Args:
            underlying: e.g. "NIFTY"
            strategy_type_str: e.g. "bear_call_spread"
            legs_data: List of dicts with keys:
                symbol, strike, option_type, expiry (YYYY-MM-DD),
                direction (BUY/SELL), quantity, entry_price
            stop_loss: Stop loss amount (positive Rs.)
            target: Target profit amount (positive Rs.)
            confidence: Signal confidence (0-1)
            rationale: Text note

        Returns:
            execution_id on success, None on failure
        """
        from strategies.base_strategy import StrategySignal, OptionLeg, TradeDirection, StrategyType

        try:
            strat_type = StrategyType(strategy_type_str)
        except ValueError:
            logger.error(f"Unknown strategy type: {strategy_type_str}")
            return None

        legs = []
        for ld in legs_data:
            try:
                expiry = datetime.strptime(ld["expiry"], "%Y-%m-%d") if isinstance(ld["expiry"], str) else ld["expiry"]
            except Exception:
                expiry = datetime.now()
            legs.append(OptionLeg(
                symbol=ld["symbol"],
                strike=float(ld["strike"]),
                option_type=ld["option_type"],
                expiry=expiry,
                direction=TradeDirection(ld["direction"]),
                quantity=int(ld["quantity"]),
                entry_price=float(ld["entry_price"]),
            ))

        # Calculate expected profit / max loss from legs
        total_credit = sum(l.entry_price * l.quantity for l in legs if l.is_short)
        total_debit = sum(l.entry_price * l.quantity for l in legs if l.is_long)
        net_credit = total_credit - total_debit

        signal = StrategySignal(
            strategy_type=strat_type,
            underlying=underlying,
            legs=legs,
            entry_time=datetime.now(),
            confidence=confidence,
            expected_profit=target,
            max_loss=stop_loss,
            stop_loss=stop_loss,
            target=target,
            rationale=rationale,
        )

        execution_id = f"{strategy_type_str}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        execution = TradeExecution(
            signal=signal,
            execution_id=execution_id,
            trading_mode=self.trading_mode,
        )
        execution.status = "ACTIVE"
        execution.entry_time = datetime.now()

        # Create synthetic COMPLETE orders so close_position can reverse them
        for i, leg in enumerate(legs):
            order = Order(
                order_id=f"MANUAL_{execution_id}_{i}",
                symbol=leg.symbol,
                exchange=get_options_exchange(underlying),
                transaction_type=leg.direction.value,
                quantity=leg.quantity,
                order_type=OrderType.LIMIT,
                price=leg.entry_price,
                status=OrderStatus.COMPLETE,
                placed_at=datetime.now(),
                filled_at=datetime.now(),
                filled_price=leg.entry_price,
                parent_signal_id=execution_id,
                leg_index=i,
            )
            execution.orders.append(order)

        self.active_executions[execution_id] = execution

        # Persist to DB
        self._persist_trade_to_db(execution_id, signal)

        # Notify position tracker for WebSocket subscription
        from execution.position_tracker import position_tracker
        position_tracker.on_new_position(execution_id)

        logger.info(f"Manual position registered: {execution_id}")
        return execution_id

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
            
            # Restore GTT order IDs
            execution.gtt_order_ids = signal_data.get("gtt_order_ids", [])
            
            # Add to active executions
            self.active_executions[execution_id] = execution
            
            logger.info(f"Loaded persisted position: {execution_id} (GTTs: {execution.gtt_order_ids})")
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
            "gtt_order_ids": list(execution.gtt_order_ids) if execution.gtt_order_ids else [],
        }


# Singleton instance
order_manager = OrderManager()
