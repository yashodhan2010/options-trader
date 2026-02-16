"""
Position Tracker - Monitors and manages open positions
Supports both polling and WebSocket-based real-time monitoring.
Includes periodic status updates and position persistence for overnight positions.
Now includes signal-based intelligent exit system.
"""
from datetime import datetime
from typing import Dict, List, Optional, Callable, Set
import threading
import time

from data.data_fetcher import data_fetcher
from execution.order_manager import order_manager, OrderType
from strategies.base_strategy import StrategySignal
from config.settings import TRADING_CONFIG, BOT_CONFIG, GREEKS_EXIT_CONFIG
from core.logger import logger, trade_logger
from core.database import database
from core.utils import is_market_open


# Lazy import to avoid circular dependency
def get_exit_signal_generator():
    from signals.exit_signal_generator import exit_signal_generator
    return exit_signal_generator


class PositionTracker:
    """
    Tracks and monitors all open positions for SL/Target hits.
    Supports two modes:
    - Polling mode: Checks positions at regular intervals (default)
    - WebSocket mode: Real-time monitoring via WebSocket (recommended)
    
    Also provides:
    - Periodic status updates every 15 minutes (configurable)
    - Position persistence for overnight recovery
    - Signal-based intelligent exits (reversal detection)
    """
    
    def __init__(self):
        self.is_running: bool = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.status_thread: Optional[threading.Thread] = None
        self.poll_interval: int = BOT_CONFIG.get("position_poll_interval", 5)  # From config
        self.status_interval: int = BOT_CONFIG.get("position_status_interval", 900)  # 15 minutes
        self.trailing_sl_enabled: bool = TRADING_CONFIG.get("trailing_sl_enabled", True)
        self.trailing_sl_percent: float = TRADING_CONFIG.get("trailing_sl_percent", 30)
        self.trailing_sl_activation_pct: float = TRADING_CONFIG.get("trailing_sl_activation_pct", 0.3)
        self.trailing_stop_levels: Dict[str, float] = {}  # execution_id -> trail floor (profit to lock in)
        self.persist_positions: bool = BOT_CONFIG.get("persist_positions", True)
        self.callbacks: Dict[str, List[Callable]] = {
            "sl_hit": [],
            "target_hit": [],
            "position_closed": [],
            "status_update": [],  # New callback for status updates
            "greeks_exit": [],    # Greeks-based exit callback
            "signal_exit": [],    # Signal-based intelligent exit callback
        }
        self.position_metrics: Dict[str, Dict] = {}
        self.entry_greeks: Dict[str, Dict] = {}  # Store entry Greeks for comparison
        self.last_status_update: Optional[datetime] = None
        
        # Signal-based exit system (runs in dedicated thread to avoid blocking WebSocket)
        self.signal_exit_enabled: bool = BOT_CONFIG.get("signal_exit_enabled", True)
        self.signal_exit_interval: int = BOT_CONFIG.get("signal_exit_interval", 60)  # Check every 60s
        self.last_signal_check: Dict[str, datetime] = {}
        self.signal_exit_thread: Optional[threading.Thread] = None
        self.signal_exit_lock: threading.Lock = threading.Lock()
        
        # WebSocket mode
        self.use_websocket: bool = BOT_CONFIG.get("use_websocket", True)
        self.websocket_manager = None
        self.subscribed_tokens: Set[int] = set()
        
        # Paper trading mode - bypass market hour checks
        self.paper_trading: bool = False
    
    def set_websocket_manager(self, ws_manager) -> None:
        """
        Set the WebSocket manager for real-time monitoring.
        
        Args:
            ws_manager: WebSocketTicker instance
        """
        self.websocket_manager = ws_manager
        logger.info("WebSocket manager attached to position tracker")
    
    def start_monitoring(self, use_websocket: bool = None, paper_trading: bool = False) -> None:
        """
        Start the position monitoring.
        
        Args:
            use_websocket: Override default WebSocket setting
            paper_trading: If True, bypass market hour checks
        """
        if self.is_running:
            logger.warning("Position tracker already running")
            return
        
        if use_websocket is not None:
            self.use_websocket = use_websocket
        
        self.paper_trading = paper_trading
        self.is_running = True
        
        # Load any existing positions from database (overnight recovery)
        self._load_persisted_positions()
        
        if self.use_websocket and self.websocket_manager:
            # WebSocket mode - register callback for price updates
            self.websocket_manager.register_callback("price_update", self._on_price_update)
            self._subscribe_active_positions()
            mode_str = "WebSocket mode"
        else:
            mode_str = f"Polling mode, interval: {self.poll_interval}s"
        
        # Always start polling thread as fallback (even in WebSocket mode)
        # This ensures prices update even if WebSocket subscriptions fail
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        if self.paper_trading:
            mode_str += " [PAPER - market hours bypassed]"
        
        logger.info(f"Position tracker started ({mode_str} + polling fallback)")
        
        # Start periodic status update thread
        self.status_thread = threading.Thread(target=self._status_update_loop, daemon=True)
        self.status_thread.start()
        logger.info(f"Position status updates enabled (interval: {self.status_interval}s / {self.status_interval // 60} min)")
        
        # Start dedicated signal exit thread (separate from WebSocket for performance)
        if self.signal_exit_enabled:
            self.signal_exit_thread = threading.Thread(target=self._signal_exit_loop, daemon=True)
            self.signal_exit_thread.start()
            logger.info(f"Signal exit monitoring enabled (interval: {self.signal_exit_interval}s)")
    
    def stop_monitoring(self) -> None:
        """Stop the position monitoring."""
        self.is_running = False
        
        if self.use_websocket and self.websocket_manager:
            # Unsubscribe from tokens
            if self.subscribed_tokens:
                self.websocket_manager.unsubscribe(list(self.subscribed_tokens))
            self.subscribed_tokens.clear()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        
        if self.status_thread:
            self.status_thread.join(timeout=10)
        
        if self.signal_exit_thread:
            self.signal_exit_thread.join(timeout=10)
            
        logger.info("Position tracker stopped")
    
    def _subscribe_active_positions(self) -> None:
        """Subscribe to WebSocket for all active position instruments."""
        if not self.websocket_manager:
            return
        
        active_positions = order_manager.get_active_positions()
        tokens_to_subscribe = set()
        
        for position in active_positions:
            execution_id = position["execution_id"]
            execution = order_manager.active_executions.get(execution_id)
            
            if execution:
                for leg in execution.signal.legs:
                    # Get instrument token for the symbol
                    token = self._get_instrument_token(leg.symbol)
                    if token:
                        tokens_to_subscribe.add(token)
        
        # Subscribe to new tokens only
        new_tokens = tokens_to_subscribe - self.subscribed_tokens
        if new_tokens:
            self.websocket_manager.subscribe(list(new_tokens), mode="ltp")
            self.subscribed_tokens.update(new_tokens)
            logger.debug(f"Subscribed to {len(new_tokens)} new instruments")
    
    def _get_instrument_token(self, symbol: str) -> Optional[int]:
        """
        Get instrument token for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Instrument token or None
        """
        # This would typically fetch from instrument master
        # For now, return None and let data_fetcher handle it
        return data_fetcher.get_instrument_token(symbol) if hasattr(data_fetcher, 'get_instrument_token') else None
    
    def _on_price_update(self, token: int, price: float) -> None:
        """
        Callback for WebSocket price updates.
        Called on every tick for subscribed instruments.
        
        Args:
            token: Instrument token
            price: Latest price
        """
        if not self.is_running:
            return
        
        # Check positions that have this token
        self._check_positions_for_token(token, price)
    
    def _check_positions_for_token(self, token: int, price: float) -> None:
        """
        Check all positions that have an instrument matching this token.
        
        Args:
            token: Instrument token
            price: Latest price
        """
        active_positions = order_manager.get_active_positions()
        
        for position in active_positions:
            execution_id = position["execution_id"]
            execution = order_manager.active_executions.get(execution_id)
            
            if not execution:
                continue
            
            # Check if this position has the updated token
            position_has_token = False
            for leg in execution.signal.legs:
                leg_token = self._get_instrument_token(leg.symbol)
                if leg_token == token:
                    leg.current_price = price
                    position_has_token = True
            
            # If position has this token, recalculate and check SL/Target
            if position_has_token:
                self._check_position_realtime(execution_id, execution)
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop (Polling mode only)."""
        while self.is_running:
            try:
                # In paper trading mode, always check positions
                # In live mode, only check during market hours
                if self.paper_trading or is_market_open():
                    self._check_all_positions()
                time.sleep(self.poll_interval)
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                time.sleep(self.poll_interval)
    
    def _check_all_positions(self) -> None:
        """Check all active positions for SL/Target (Polling mode)."""
        active_positions = order_manager.get_active_positions()
        
        for position in active_positions:
            try:
                self._check_position_polling(position)
            except Exception as e:
                logger.error(f"Error checking position {position['execution_id']}: {e}")
    
    def _check_position_polling(self, position: Dict) -> None:
        """
        Check a single position for SL/Target hit (Polling mode).
        Fetches latest prices via API.
        
        Args:
            position: Position dictionary
        """
        execution_id = position["execution_id"]
        execution = order_manager.active_executions.get(execution_id)
        
        if not execution:
            return
        
        signal = execution.signal
        
        # Get current prices for all legs via API
        current_prices = {}
        total_pnl = 0
        
        for leg in signal.legs:
            current_price = data_fetcher.get_ltp(leg.symbol)
            
            # In paper trading mode, if API returns no price, simulate with entry price
            if current_price:
                current_prices[leg.symbol] = current_price
                leg.current_price = current_price
            elif self.paper_trading:
                # Use entry price with small random fluctuation for paper trading
                import random
                # Simulate price within +/- 2% of entry price
                fluctuation = random.uniform(-0.02, 0.02)
                simulated_price = leg.entry_price * (1 + fluctuation)
                current_prices[leg.symbol] = simulated_price
                leg.current_price = simulated_price
                logger.debug(f"[PAPER] Simulated price for {leg.symbol}: Rs.{simulated_price:.2f}")
            else:
                # Fallback to entry price if no current price available
                if leg.entry_price:
                    current_prices[leg.symbol] = leg.entry_price
                    leg.current_price = leg.entry_price
            
            total_pnl += leg.pnl()
        
        # Store metrics and check SL/Target
        self._update_metrics_and_check(execution_id, execution, total_pnl, current_prices)
    
    def _check_position_realtime(self, execution_id: str, execution) -> None:
        """
        Check a single position for SL/Target hit (WebSocket mode).
        Uses cached prices from WebSocket.
        
        Args:
            execution_id: Execution ID
            execution: TradeExecution object
        """
        signal = execution.signal
        
        # Calculate P&L from current prices (already updated)
        current_prices = {}
        total_pnl = 0
        
        for leg in signal.legs:
            if leg.current_price:
                current_prices[leg.symbol] = leg.current_price
            elif self.paper_trading:
                # Simulate price for paper trading if no websocket price
                import random
                fluctuation = random.uniform(-0.02, 0.02)
                simulated_price = leg.entry_price * (1 + fluctuation)
                current_prices[leg.symbol] = simulated_price
                leg.current_price = simulated_price
            elif leg.entry_price:
                # Fallback to entry price
                current_prices[leg.symbol] = leg.entry_price
                leg.current_price = leg.entry_price
            
            total_pnl += leg.pnl()
        
        # Store metrics and check SL/Target
        self._update_metrics_and_check(execution_id, execution, total_pnl, current_prices)
    
    def _update_metrics_and_check(
        self, 
        execution_id: str, 
        execution, 
        total_pnl: float,
        current_prices: Dict[str, float]
    ) -> None:
        """
        Update position metrics and check for SL/Target.
        
        Args:
            execution_id: Execution ID
            execution: TradeExecution object
            total_pnl: Current total P&L
            current_prices: Dict of symbol -> current price
        """
        signal = execution.signal
        
        # Store metrics
        self.position_metrics[execution_id] = {
            "current_pnl": total_pnl,
            "last_update": datetime.now().isoformat(),
            "current_prices": current_prices,
        }
        
        # Check stop loss (initial hard SL)
        if total_pnl <= -signal.stop_loss:
            logger.warning(f"Stop loss hit for {execution_id}! P&L: {total_pnl:.2f}")
            self._trigger_exit(execution_id, "SL_HIT", total_pnl)
            self._notify("sl_hit", execution_id, total_pnl)
            return
        
        # Check trailing stop loss - locks in profits
        if self.trailing_sl_enabled and execution_id in self.trailing_stop_levels:
            trail_level = self.trailing_stop_levels[execution_id]
            if trail_level > 0 and total_pnl <= trail_level:
                logger.info(
                    f"Trailing SL hit for {execution_id}! P&L: {total_pnl:.2f} "
                    f"(trail floor: {trail_level:.2f})"
                )
                self._trigger_exit(execution_id, "TRAILING_SL_HIT", total_pnl)
                self._notify("sl_hit", execution_id, total_pnl)
                return
        
        # Check target
        if total_pnl >= signal.target:
            logger.info(f"Target hit for {execution_id}! P&L: {total_pnl:.2f}")
            self._trigger_exit(execution_id, "TARGET_HIT", total_pnl)
            self._notify("target_hit", execution_id, total_pnl)
            return
        
        # Update trailing stop loss (ratchet upward as profits grow)
        if self.trailing_sl_enabled and total_pnl > 0:
            self._update_trailing_sl(execution_id, execution, total_pnl)
        
        # Greeks-based exit checks
        if GREEKS_EXIT_CONFIG.get("enabled", False):
            exit_reason = self._check_greeks_exit(execution_id, execution, total_pnl)
            if exit_reason:
                logger.info(f"Greeks-based exit for {execution_id}: {exit_reason}")
                self._trigger_exit(execution_id, exit_reason, total_pnl)
                self._notify("greeks_exit", execution_id, total_pnl, exit_reason)
                return
        
        # NOTE: Signal-based exits are handled by a dedicated thread (_signal_exit_loop)
        # to avoid blocking WebSocket callbacks. Not called inline here.
    
    def _check_signal_exit(
        self,
        execution_id: str,
        execution,
        total_pnl: float,
        current_prices: Dict[str, float],
        force: bool = False,
    ) -> None:
        """
        Check for signal-based intelligent exits using the ExitSignalGenerator.
        This checks for trend reversals, sentiment shifts, thesis invalidation, etc.
        
        NOTE: This is an expensive operation (fetches market data). 
        In WebSocket mode, this is rate-limited and runs asynchronously.
        
        Args:
            execution_id: Execution ID
            execution: TradeExecution object
            total_pnl: Current total P&L
            current_prices: Dict of symbol -> current price
            force: If True, bypass rate limiting
        """
        # Rate limit signal checks (they're more expensive than simple SL/Target)
        now = datetime.now()
        last_check = self.last_signal_check.get(execution_id)
        if not force and last_check and (now - last_check).seconds < self.signal_exit_interval:
            return
        
        self.last_signal_check[execution_id] = now
        
        try:
            exit_generator = get_exit_signal_generator()
            exit_signal = exit_generator.generate_exit_signal(
                execution_id=execution_id,
                signal=execution.signal,
                current_pnl=total_pnl,
                current_prices=current_prices,
            )
            
            if exit_signal and exit_signal.should_exit:
                # Log the exit signal details
                logger.info(f"[EXIT SIGNAL] {execution_id}")
                logger.info(f"   Reason:     {exit_signal.reason.value}")
                logger.info(f"   Confidence: {exit_signal.confidence:.1%}")
                logger.info(f"   Urgency:    {exit_signal.urgency}")
                logger.info(f"   Current P&L: Rs.{exit_signal.current_pnl:.2f}")
                logger.info(f"   Rationale:  {exit_signal.rationale}")
                
                # Only act on high confidence or urgent signals
                should_exit = (
                    exit_signal.confidence >= 0.70 or
                    exit_signal.urgency in ["HIGH", "IMMEDIATE"]
                )
                
                if should_exit:
                    reason_str = f"SIGNAL_{exit_signal.reason.value}"
                    self._trigger_exit(execution_id, reason_str, total_pnl)
                    self._notify("signal_exit", execution_id, total_pnl, exit_signal)
                else:
                    # Log as advisory but don't exit
                    logger.info(f"   [ADVISORY] Signal suggests exit but below threshold - monitoring")
                    
        except Exception as e:
            logger.debug(f"Signal exit check failed for {execution_id}: {e}")
    
    def _signal_exit_loop(self) -> None:
        """
        Dedicated loop for signal-based exit checks.
        
        Runs separately from WebSocket callbacks to avoid blocking real-time price updates.
        This is an expensive operation (fetches market data, analyzes indicators) so it runs
        at a configurable interval (default 60s) rather than on every price tick.
        """
        logger.debug("Signal exit loop started")
        
        while self.is_running:
            try:
                # Sleep first, then check (allows immediate shutdown)
                time.sleep(self.signal_exit_interval)
                
                if not self.is_running:
                    break
                
                # Check if we have active positions
                active_executions = order_manager.active_executions.copy()
                if not active_executions:
                    continue
                
                logger.debug(f"Signal exit check: scanning {len(active_executions)} position(s)")
                
                # Check each active position for signal-based exits
                for execution_id, execution in active_executions.items():
                    if not self.is_running:
                        break
                    
                    try:
                        # Get current prices from cached metrics
                        with self.signal_exit_lock:
                            metrics = self.position_metrics.get(execution_id, {})
                        
                        current_prices = metrics.get("current_prices", {})
                        current_pnl = metrics.get("current_pnl", 0)
                        
                        # If no prices yet, skip (wait for WebSocket/poll to update)
                        if not current_prices:
                            continue
                        
                        # Run the signal exit check (force=True bypasses rate limit since we control timing)
                        self._check_signal_exit(
                            execution_id=execution_id,
                            execution=execution,
                            total_pnl=current_pnl,
                            current_prices=current_prices,
                            force=True,  # We control the timing, so bypass internal rate limiting
                        )
                        
                    except Exception as e:
                        logger.debug(f"Signal exit check error for {execution_id}: {e}")
                
            except Exception as e:
                logger.error(f"Signal exit loop error: {e}")
                time.sleep(5)  # Brief pause on error
        
        logger.debug("Signal exit loop stopped")

    def on_new_position(self, execution_id: str, market_data: Dict = None) -> None:
        """
        Called when a new position is opened.
        Subscribes to WebSocket for the position's instruments.
        Also stores entry Greeks and market conditions for intelligent exit checks.
        
        Args:
            execution_id: Execution ID of new position
            market_data: Optional market data at entry time (spot, OI, volatility, historical)
        """
        execution = order_manager.active_executions.get(execution_id)
        if not execution:
            return
        
        # Store entry Greeks for comparison during exit checks
        if GREEKS_EXIT_CONFIG.get("enabled", False):
            self.store_entry_greeks(execution_id, execution.signal)
        
        # Store entry conditions for signal-based exits
        if self.signal_exit_enabled and market_data:
            try:
                exit_generator = get_exit_signal_generator()
                exit_generator.store_entry_conditions(execution_id, execution.signal, market_data)
                logger.debug(f"Stored entry conditions for signal-based exits: {execution_id}")
            except Exception as e:
                logger.debug(f"Could not store entry conditions: {e}")
        
        # WebSocket subscription
        if not self.use_websocket or not self.websocket_manager:
            return
        
        tokens_to_subscribe = []
        for leg in execution.signal.legs:
            token = self._get_instrument_token(leg.symbol)
            if token and token not in self.subscribed_tokens:
                tokens_to_subscribe.append(token)
                self.subscribed_tokens.add(token)
        
        if tokens_to_subscribe:
            self.websocket_manager.subscribe(tokens_to_subscribe, mode="ltp")
            logger.debug(f"Subscribed to {len(tokens_to_subscribe)} instruments for new position")
    
    def on_position_closed(self, execution_id: str) -> None:
        """
        Called when a position is closed.
        Cleans up stored Greeks and optionally unsubscribes from WebSocket.
        
        Args:
            execution_id: Execution ID of closed position
        """
        # Clean up entry Greeks
        if execution_id in self.entry_greeks:
            del self.entry_greeks[execution_id]
        
        # Keep WebSocket subscriptions for now - may be reused
        # In production, implement token reference counting
        pass
    
    def _update_trailing_sl(
        self,
        execution_id: str,
        execution,
        current_pnl: float,
    ) -> None:
        """
        Update trailing stop loss to lock in profits as the trade moves favorably.
        
        Only activates after profit exceeds activation_pct of target (default 30%).
        Trails at trailing_sl_percent below peak profit (default 30% — protects 70%).
        
        Args:
            execution_id: Execution ID
            execution: TradeExecution object
            current_pnl: Current P&L (positive)
        """
        signal = execution.signal
        
        # Only start trailing after reaching activation threshold
        activation_threshold = signal.target * self.trailing_sl_activation_pct
        if current_pnl < activation_threshold:
            return
        
        # Calculate trail floor: protect (100 - trailing_sl_percent)% of current profit
        # e.g., with 30% trail: at P&L 1000 → trail floor = 700 (protect 70%)
        protection_pct = (100 - self.trailing_sl_percent) / 100
        new_trail_level = current_pnl * protection_pct
        
        # Only ratchet upward - never lower the trail floor
        current_trail = self.trailing_stop_levels.get(execution_id, 0)
        if new_trail_level > current_trail:
            self.trailing_stop_levels[execution_id] = new_trail_level
            logger.info(
                f"Trailing SL updated for {execution_id}: "
                f"lock in Rs.{new_trail_level:.2f} (P&L: {current_pnl:.2f}, "
                f"protect {protection_pct:.0%})"
            )
    
    def _check_greeks_exit(
        self,
        execution_id: str,
        execution,
        current_pnl: float,
    ) -> Optional[str]:
        """
        Check if position should exit based on Greeks.
        
        Returns exit reason string if exit should trigger, None otherwise.
        """
        signal = execution.signal
        config = GREEKS_EXIT_CONFIG
        
        try:
            # Get current Greeks for the position
            current_greeks = data_fetcher.get_strategy_greeks(signal)
            if not current_greeks:
                return None
            
            # Store entry Greeks if not already stored
            if execution_id not in self.entry_greeks:
                self.entry_greeks[execution_id] = {
                    "greeks": current_greeks.copy(),
                    "iv": current_greeks.get("avg_iv", 0),
                    "timestamp": datetime.now(),
                }
            
            entry_data = self.entry_greeks[execution_id]
            
            # 1. DELTA-BASED EXIT
            if config.get("delta_exit_enabled", False):
                delta = abs(current_greeks.get("delta", 0))
                is_long = any(leg.direction.value == "BUY" for leg in signal.legs)
                
                if is_long and delta < config.get("min_delta_long", 0.10):
                    # Long option losing sensitivity - becoming worthless
                    return "DELTA_TOO_LOW"
                
                if not is_long and delta > config.get("max_delta_short", 0.90):
                    # Short option too risky - deep ITM
                    return "DELTA_TOO_HIGH"
            
            # 2. THETA-BASED EXIT (Time Decay)
            if config.get("theta_exit_enabled", False):
                theta = abs(current_greeks.get("theta", 0))
                remaining_profit = signal.target - current_pnl
                
                if remaining_profit > 0:
                    # If daily theta decay exceeds threshold of remaining profit
                    if theta > remaining_profit * config.get("theta_decay_threshold", 0.5):
                        return "THETA_DECAY_HIGH"
                
                # Check days to expiry
                dte = current_greeks.get("dte", 30)
                if dte <= config.get("days_to_expiry_exit", 2):
                    return "DTE_TOO_LOW"
            
            # 3. VEGA-BASED EXIT (IV Crush)
            if config.get("vega_exit_enabled", False):
                entry_iv = entry_data.get("iv", 0)
                current_iv = current_greeks.get("avg_iv", 0)
                
                if entry_iv > 0 and current_iv > 0:
                    iv_change_pct = ((current_iv - entry_iv) / entry_iv) * 100
                    
                    # For long options, IV crush hurts
                    is_long = any(leg.direction.value == "BUY" for leg in signal.legs)
                    if is_long and iv_change_pct < -config.get("iv_drop_percent", 20):
                        return "IV_CRUSH"
            
            # 4. GAMMA-BASED STOP TIGHTENING
            if config.get("gamma_tighten_enabled", False) and current_pnl > 0:
                gamma = abs(current_greeks.get("gamma", 0))
                
                if gamma > config.get("gamma_threshold", 0.05):
                    # Tighten stop loss when gamma is high (fast-moving delta)
                    tighten_pct = config.get("gamma_sl_tighten_percent", 20) / 100
                    new_sl_distance = signal.stop_loss * (1 - tighten_pct)
                    
                    # Create tighter floor
                    tighter_sl = current_pnl - new_sl_distance
                    if tighter_sl > 0 and tighter_sl > signal.stop_loss:
                        old_sl = signal.stop_loss
                        signal.stop_loss = tighter_sl
                        logger.debug(f"Gamma-tightened SL for {execution_id}: {old_sl:.2f} -> {tighter_sl:.2f}")
            
            # 5. PROFIT LOCK (Dynamic Floor)
            if config.get("profit_lock_enabled", False):
                profit_ratio = current_pnl / signal.target if signal.target > 0 else 0
                
                if profit_ratio >= config.get("profit_lock_threshold", 0.5):
                    # Lock a percentage of profit
                    locked_profit = current_pnl * config.get("profit_lock_percent", 0.3)
                    
                    if locked_profit > signal.stop_loss:
                        old_sl = signal.stop_loss
                        signal.stop_loss = locked_profit
                        logger.debug(f"Profit-locked SL for {execution_id}: {old_sl:.2f} -> {locked_profit:.2f}")
            
        except Exception as e:
            logger.debug(f"Greeks exit check failed for {execution_id}: {e}")
        
        return None
    
    def store_entry_greeks(self, execution_id: str, signal: StrategySignal) -> None:
        """
        Store Greeks at entry time for comparison during exit checks.
        Call this when a new position is opened.
        """
        try:
            greeks = data_fetcher.get_strategy_greeks(signal)
            if greeks:
                self.entry_greeks[execution_id] = {
                    "greeks": greeks.copy(),
                    "iv": greeks.get("avg_iv", 0),
                    "timestamp": datetime.now(),
                }
                logger.debug(f"Stored entry Greeks for {execution_id}: D={greeks.get('delta', 0):.4f}")
        except Exception as e:
            logger.debug(f"Could not store entry Greeks: {e}")

    def _trigger_exit(
        self,
        execution_id: str,
        reason: str,
        pnl: float,
    ) -> None:
        """
        Trigger position exit.
        
        Args:
            execution_id: Execution ID
            reason: Exit reason
            pnl: P&L at exit
        """
        logger.info(f"Triggering exit for {execution_id}: {reason}")
        
        success = order_manager.close_position(execution_id, OrderType.MARKET)
        
        if success:
            self._notify("position_closed", execution_id, pnl, reason)
            
            # Clean up metrics and trailing state
            if execution_id in self.position_metrics:
                del self.position_metrics[execution_id]
            if execution_id in self.trailing_stop_levels:
                del self.trailing_stop_levels[execution_id]
    
    def register_callback(self, event: str, callback: Callable) -> None:
        """
        Register a callback for position events.
        
        Args:
            event: Event type (sl_hit, target_hit, position_closed)
            callback: Callback function
        """
        if event in self.callbacks:
            self.callbacks[event].append(callback)
    
    def _notify(self, event: str, *args) -> None:
        """Notify all registered callbacks for an event."""
        for callback in self.callbacks.get(event, []):
            try:
                callback(*args)
            except Exception as e:
                logger.error(f"Error in callback: {e}")
    
    def get_position_summary(self) -> List[Dict]:
        """
        Get summary of all active positions with current P&L.
        
        Returns:
            List of position summaries
        """
        summaries = []
        
        for position in order_manager.get_active_positions():
            execution_id = position["execution_id"]
            metrics = self.position_metrics.get(execution_id, {})
            
            summaries.append({
                **position,
                "current_pnl": metrics.get("current_pnl", 0),
                "last_update": metrics.get("last_update"),
            })
        
        return summaries
    
    def get_daily_pnl(self) -> float:
        """
        Calculate total P&L for the day.
        
        Returns:
            Total daily P&L
        """
        total_pnl = 0
        
        # Active positions
        for metrics in self.position_metrics.values():
            total_pnl += metrics.get("current_pnl", 0)
        
        # Closed positions (from order history)
        # This would typically come from a database in production
        
        return total_pnl
    
    def check_daily_loss_limit(self) -> bool:
        """
        Check if daily loss limit has been reached.
        
        Returns:
            True if loss limit reached
        """
        max_loss = TRADING_CONFIG.get("max_loss_per_day", 10000)
        daily_pnl = self.get_daily_pnl()
        
        if daily_pnl <= -max_loss:
            logger.warning(f"Daily loss limit reached! P&L: {daily_pnl:.2f}")
            return True
        
        return False
    
    def force_close_all(self, reason: str = "Manual close") -> None:
        """
        Force close all active positions.
        
        Args:
            reason: Reason for closing
        """
        logger.warning(f"Force closing all positions: {reason}")
        trade_logger.info(f"FORCE CLOSE ALL TRIGGERED | Reason: {reason}")
        
        active_positions = order_manager.get_active_positions()
        
        if not active_positions:
            logger.info("No active positions to close")
            return
        
        total_pnl = 0
        
        for position in active_positions:
            execution_id = position["execution_id"]
            execution = order_manager.active_executions.get(execution_id)
            metrics = self.position_metrics.get(execution_id, {})
            pnl = metrics.get("current_pnl", 0)
            
            # Log full trade details before closing
            if execution:
                signal = execution.signal
                entry_time = execution.entry_time
                time_str = "Unknown"
                if entry_time:
                    duration = datetime.now() - entry_time
                    time_str = self._format_duration(duration.total_seconds())
                
                result = "PROFIT" if pnl >= 0 else "LOSS"
                
                trade_logger.info("=" * 80)
                trade_logger.info(f"TRADE CLOSED")
                trade_logger.info("=" * 80)
                trade_logger.info(f"Execution ID: {execution_id}")
                trade_logger.info(f"Strategy:     {signal.strategy_type.value}")
                trade_logger.info(f"Underlying:   {signal.underlying}")
                trade_logger.info(f"Reason:       {reason}")
                trade_logger.info(f"Duration:     {time_str}")
                trade_logger.info(f"Mode:         {'PAPER' if order_manager.is_paper_trading else 'LIVE'}")
                trade_logger.info("-" * 40)
                trade_logger.info("LEGS:")
                for leg in signal.legs:
                    leg_pnl = leg.pnl() if hasattr(leg, 'pnl') else 0
                    trade_logger.info(
                        f"  {leg.direction.value:4} | {leg.symbol} | "
                        f"Qty: {leg.quantity} | Entry: Rs.{leg.entry_price:.2f} | "
                        f"Exit: Rs.{leg.current_price or leg.entry_price:.2f} | "
                        f"P&L: Rs.{leg_pnl:.2f}"
                    )
                trade_logger.info("-" * 40)
                trade_logger.info(f"P&L:          Rs.{pnl:.2f}")
                trade_logger.info(f"Result:       {result}")
                trade_logger.info("=" * 80)
                trade_logger.info("")
            
            order_manager.close_position(execution_id)
            self._notify("position_closed", execution_id, pnl, reason)
            total_pnl += pnl
        
        self.position_metrics.clear()
        logger.info(f"All {len(active_positions)} positions closed. Total P&L: Rs.{total_pnl:.2f}")
    
    # ========== Periodic Status Updates ==========
    
    def _status_update_loop(self) -> None:
        """Periodic loop to log and broadcast position status updates."""
        while self.is_running:
            try:
                time.sleep(self.status_interval)
                
                if not self.is_running:
                    break
                
                self._broadcast_status_update()
                
            except Exception as e:
                logger.error(f"Error in status update loop: {e}")
    
    def _broadcast_status_update(self) -> None:
        """Generate and broadcast status update for all active positions."""
        active_positions = order_manager.get_active_positions()
        
        self.last_status_update = datetime.now()
        
        logger.info("=" * 60)
        logger.info(f"[STATUS] POSITION STATUS UPDATE - {self.last_status_update.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)
        
        if not active_positions:
            logger.info("[STATUS] No open positions")
            logger.info("=" * 60)
            return
        
        total_unrealized_pnl = 0
        
        for position in active_positions:
            execution_id = position["execution_id"]
            execution = order_manager.active_executions.get(execution_id)
            
            if not execution:
                continue
            
            metrics = self.position_metrics.get(execution_id, {})
            current_pnl = metrics.get("current_pnl", 0)
            current_prices = metrics.get("current_prices", {})
            
            # Calculate time in trade
            entry_time = execution.entry_time
            if entry_time:
                time_in_trade = datetime.now() - entry_time
                time_str = self._format_duration(time_in_trade.total_seconds())
                time_in_seconds = int(time_in_trade.total_seconds())
            else:
                time_str = "Unknown"
                time_in_seconds = 0
            
            signal = execution.signal
            pnl_percent = (current_pnl / signal.stop_loss * 100) if signal.stop_loss else 0
            
            # Get Greeks for the position
            position_greeks = {"delta": 0, "gamma": 0, "theta": 0, "vega": 0}
            try:
                greeks_data = data_fetcher.get_strategy_greeks(signal)
                if greeks_data:
                    position_greeks = greeks_data
            except Exception as e:
                logger.debug(f"Could not fetch Greeks: {e}")
            
            # Log status
            pnl_indicator = "[+]" if current_pnl >= 0 else "[-]"
            logger.info(f"\n[POS] {execution_id}")
            logger.info(f"   Strategy: {signal.strategy_type.value} | Underlying: {signal.underlying}")
            logger.info(f"   Time in Trade: {time_str}")
            logger.info(f"   {pnl_indicator} Current P&L: Rs.{current_pnl:,.2f} ({pnl_percent:+.1f}% of SL)")
            logger.info(f"   SL: Rs.{signal.stop_loss:,.2f} | Target: Rs.{signal.target:,.2f}")
            logger.info(f"   Greeks: D={position_greeks.get('delta', 0):.4f} G={position_greeks.get('gamma', 0):.6f} T={position_greeks.get('theta', 0):.2f} V={position_greeks.get('vega', 0):.4f}")
            
            for leg in signal.legs:
                leg_pnl = leg.pnl() if hasattr(leg, 'pnl') else 0
                logger.info(f"   |-- {leg.direction.value} {leg.symbol}: Entry Rs.{leg.entry_price:.2f} -> Current Rs.{leg.current_price or 0:.2f} (P&L: Rs.{leg_pnl:.2f})")
            
            total_unrealized_pnl += current_pnl
            
            # Persist to database
            if self.persist_positions:
                database.log_position_status(
                    execution_id=execution_id,
                    underlying=signal.underlying,
                    strategy_type=signal.strategy_type.value,
                    current_pnl=current_pnl,
                    unrealized_pnl=current_pnl,
                    current_prices=current_prices,
                    time_in_trade_seconds=time_in_seconds,
                    stop_loss=signal.stop_loss,
                    target=signal.target,
                    status="ACTIVE",
                )
            
            # Notify callbacks
            self._notify("status_update", {
                "execution_id": execution_id,
                "underlying": signal.underlying,
                "strategy": signal.strategy_type.value,
                "current_pnl": current_pnl,
                "time_in_trade": time_str,
                "stop_loss": signal.stop_loss,
                "target": signal.target,
            })
        
        logger.info(f"\n[TOTAL] Total Unrealized P&L: Rs.{total_unrealized_pnl:,.2f}")
        logger.info("=" * 60)
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        elif seconds < 86400:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
        else:
            days = int(seconds // 86400)
            hours = int((seconds % 86400) // 3600)
            return f"{days}d {hours}h"
    
    # ========== Position Persistence ==========
    
    def _load_persisted_positions(self) -> None:
        """
        Load active positions from database on startup.
        Handles overnight positions and bot restarts.
        """
        if not self.persist_positions:
            return
        
        try:
            active_trades = database.get_active_trades()
            
            if not active_trades:
                logger.info("No persisted positions to load")
                return
            
            logger.info(f"[LOAD] Loading {len(active_trades)} persisted position(s) from database...")
            
            for trade in active_trades:
                execution_id = trade.get("execution_id")
                signal_data = trade.get("signal_data", {})
                
                if not execution_id or not signal_data:
                    continue
                
                # Check if already in active executions
                if execution_id in order_manager.active_executions:
                    logger.debug(f"Position {execution_id} already loaded")
                    continue
                
                # Reconstruct position in order manager
                loaded = order_manager.load_persisted_position(execution_id, signal_data)
                
                if loaded:
                    logger.info(f"[OK] Loaded position: {execution_id} ({trade.get('strategy_type')} on {trade.get('underlying')})")
                    
                    # Initialize metrics
                    self.position_metrics[execution_id] = {
                        "current_pnl": 0,
                        "last_update": datetime.now().isoformat(),
                        "current_prices": {},
                    }
                else:
                    logger.warning(f"[FAIL] Failed to load position: {execution_id}")
            
            logger.info(f"Finished loading persisted positions")
            
        except Exception as e:
            logger.error(f"Error loading persisted positions: {e}")
    
    def get_status_update_summary(self) -> Dict:
        """
        Get a summary of current position status.
        
        Returns:
            Dictionary with status summary
        """
        active_positions = order_manager.get_active_positions()
        total_pnl = sum(
            self.position_metrics.get(p["execution_id"], {}).get("current_pnl", 0)
            for p in active_positions
        )
        
        return {
            "active_positions": len(active_positions),
            "total_unrealized_pnl": total_pnl,
            "last_status_update": self.last_status_update.isoformat() if self.last_status_update else None,
            "status_interval_minutes": self.status_interval // 60,
            "positions": [
                {
                    "execution_id": p["execution_id"],
                    "underlying": p["underlying"],
                    "strategy": p["strategy"],
                    "current_pnl": self.position_metrics.get(p["execution_id"], {}).get("current_pnl", 0),
                }
                for p in active_positions
            ]
        }


# Singleton instance
position_tracker = PositionTracker()
