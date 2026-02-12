"""
Options Trading Bot - Main Orchestrator

All trading signals are ML-driven. No rule-based signal generation.
"""
import argparse
import time
from datetime import datetime
from typing import Optional, List
import threading

from auth.kite_auth import connect, get_kite, is_authenticated, get_profile, get_margins
from data.data_fetcher import data_fetcher
from data.websocket_manager import WebSocketTicker
from signals.ml_signal_generator import ml_signal_generator as signal_generator
from execution.order_manager import order_manager, OrderType
from execution.position_tracker import position_tracker
from strategies.base_strategy import StrategyType
from config.settings import (
    UNDERLYING_ASSETS, TRADING_CONFIG, MARKET_HOURS, NOTIFICATION_CONFIG, BOT_CONFIG, ML_CONFIG
)
from core.logger import logger, trade_logger
from core.utils import is_market_open, is_trading_allowed, should_square_off, get_market_status, is_expiry_day, should_auto_exit


class OptionsTradingBot:
    """
    Main orchestrator for the options trading bot.
    """
    
    def __init__(
        self,
        underlyings: List[str] = None,
        auto_trade: bool = False,
        paper_trading: bool = True,
        use_websocket: bool = True,
    ):
        """
        Initialize the trading bot.
        
        Args:
            underlyings: List of underlying assets to trade
            auto_trade: Enable automatic trade execution
            paper_trading: Enable paper trading mode
            use_websocket: Use WebSocket for real-time monitoring
        """
        self.underlyings = underlyings or list(UNDERLYING_ASSETS.keys())
        self.auto_trade = auto_trade
        self.paper_trading = paper_trading
        self.is_running = False
        self.scan_interval = BOT_CONFIG.get("signal_scan_interval", 60)  # From config
        self.use_websocket = use_websocket
        self.websocket_manager: Optional[WebSocketTicker] = None
        
        # Set trading mode
        order_manager.set_paper_trading(paper_trading)
        
        # Register callbacks
        self._register_callbacks()
        
        logger.info(f"Bot initialized for: {', '.join(self.underlyings)}")
        logger.info(f"Auto-trade: {auto_trade}, Paper trading: {paper_trading}")
        logger.info(f"WebSocket monitoring: {use_websocket}")
    
    def _register_callbacks(self) -> None:
        """Register callbacks for position events."""
        position_tracker.register_callback("sl_hit", self._on_sl_hit)
        position_tracker.register_callback("target_hit", self._on_target_hit)
        position_tracker.register_callback("position_closed", self._on_position_closed)
        position_tracker.register_callback("signal_exit", self._on_signal_exit)
    
    def _on_sl_hit(self, execution_id: str, pnl: float) -> None:
        """Handle stop loss hit."""
        logger.warning(f"[SL HIT] {execution_id}, Loss: Rs.{abs(pnl):.2f}")
        self._send_notification(f"Stop Loss Hit!\nPosition: {execution_id}\nLoss: Rs.{abs(pnl):.2f}")
        
        # Log to trade file
        trade_logger.info("=" * 80)
        trade_logger.info(f"STOP LOSS HIT")
        trade_logger.info(f"Execution ID: {execution_id}")
        trade_logger.info(f"P&L:          Rs.{pnl:.2f}")
        trade_logger.info(f"Result:       LOSS")
        trade_logger.info("=" * 80)
        trade_logger.info("")
    
    def _on_target_hit(self, execution_id: str, pnl: float) -> None:
        """Handle target hit."""
        logger.info(f"[TARGET HIT] {execution_id}, Profit: Rs.{pnl:.2f}")
        self._send_notification(f"Target Hit!\nPosition: {execution_id}\nProfit: Rs.{pnl:.2f}")
        
        # Log to trade file
        trade_logger.info("=" * 80)
        trade_logger.info(f"TARGET HIT")
        trade_logger.info(f"Execution ID: {execution_id}")
        trade_logger.info(f"P&L:          Rs.{pnl:.2f}")
        trade_logger.info(f"Result:       PROFIT")
        trade_logger.info("=" * 80)
        trade_logger.info("")
    
    def _on_position_closed(self, execution_id: str, pnl: float, reason: str) -> None:
        """Handle position closed."""
        logger.info(f"Position Closed: {execution_id}, P&L: Rs.{pnl:.2f}, Reason: {reason}")
        
        # Log to trade file
        result = "PROFIT" if pnl >= 0 else "LOSS"
        trade_logger.info("=" * 80)
        trade_logger.info(f"TRADE CLOSED")
        trade_logger.info(f"Execution ID: {execution_id}")
        trade_logger.info(f"Reason:       {reason}")
        trade_logger.info(f"P&L:          Rs.{pnl:.2f}")
        trade_logger.info(f"Result:       {result}")
        trade_logger.info("=" * 80)
        trade_logger.info("")
    
    def _on_signal_exit(self, execution_id: str, pnl: float, exit_signal) -> None:
        """Handle intelligent signal-based exit."""
        reason_str = exit_signal.reason.value if hasattr(exit_signal.reason, 'value') else str(exit_signal.reason)
        logger.info(f"[SIGNAL EXIT] {execution_id}, P&L: Rs.{pnl:.2f}, Reason: {reason_str}")
        
        # Send notification
        self._send_notification(
            f"Signal Exit Triggered!\n"
            f"Position: {execution_id}\n"
            f"Reason: {reason_str}\n"
            f"Confidence: {exit_signal.confidence:.1%}\n"
            f"P&L: Rs.{pnl:.2f}"
        )
        
        # Log to trade file
        result = "PROFIT" if pnl >= 0 else "LOSS"
        trade_logger.info("=" * 80)
        trade_logger.info(f"SIGNAL EXIT TRIGGERED")
        trade_logger.info("=" * 80)
        trade_logger.info(f"Execution ID: {execution_id}")
        trade_logger.info(f"Exit Reason:  {reason_str}")
        trade_logger.info(f"Confidence:   {exit_signal.confidence:.1%}")
        trade_logger.info(f"Urgency:      {exit_signal.urgency}")
        trade_logger.info(f"P&L:          Rs.{pnl:.2f}")
        trade_logger.info(f"Result:       {result}")
        trade_logger.info(f"Mode:         {'PAPER' if order_manager.is_paper_trading else 'LIVE'}")
        trade_logger.info("-" * 40)
        trade_logger.info(f"Rationale:    {exit_signal.rationale}")
        trade_logger.info("=" * 80)
        trade_logger.info("")

    def _send_notification(self, message: str) -> None:
        """Send notification via configured channels (Telegram, WhatsApp)."""
        from core.notifications import notification_service
        
        # Send via WhatsApp
        if NOTIFICATION_CONFIG.get("whatsapp_enabled"):
            notification_service.send_whatsapp(message)
        
        # Send via Telegram
        if NOTIFICATION_CONFIG.get("telegram_enabled"):
            notification_service.send_telegram(message)
        
        logger.info(f"[NOTIFICATION] {message}")
    
    def login(self) -> bool:
        """
        Perform Kite Connect login.
        
        Returns:
            True if login successful
        """
        if is_authenticated():
            logger.info("Already authenticated")
            profile = get_profile()
            if profile:
                logger.info(f"Logged in as: {profile.get('user_name')}")
            return True
        
        logger.info("Starting login process...")
        try:
            connect()
            return True
        except Exception as e:
            logger.error(f"Login failed: {e}")
            return False
    
    def start(self) -> None:
        """Start the trading bot."""
        logger.info("=" * 50)
        logger.info("Starting Options Trading Bot (ML-Only Mode)")
        logger.info("=" * 50)
        
        # Verify ML model is available
        ml_status = signal_generator.get_model_status()
        if not ml_status.get("model_loaded"):
            logger.error("=" * 50)
            logger.error("NO ML MODEL FOUND!")
            logger.error("This bot requires a trained ML model to generate signals.")
            logger.error("Please train a model first using:")
            logger.error("  python cli.py -> ml train <SYMBOL>")
            logger.error("  or: ml train-best <config>")
            logger.error("=" * 50)
            return
        
        logger.info(f"ML Model: {ml_status.get('model_version')} ({ml_status.get('model_type')})")
        logger.info(f"Min Confidence: {ml_status.get('min_confidence'):.1%}")
        
        # Login first
        if not self.login():
            logger.error("Login failed. Cannot start bot.")
            return
        
        # Verify connection
        profile = get_profile()
        margins = get_margins()
        
        if profile:
            logger.info(f"User: {profile.get('user_name')} ({profile.get('user_id')})")
        
        if margins:
            equity_margin = margins.get("equity", {})
            available = equity_margin.get("available", {}).get("live_balance", 0)
            logger.info(f"Available margin: Rs.{available:,.2f}")
        
        self.is_running = True
        
        # Initialize WebSocket if enabled
        if self.use_websocket:
            self._start_websocket()
        
        # Start auto-retrain monitor if enabled
        self._start_auto_retrain_monitor()
        
        # Start live ML feature collection (background thread)
        self._start_live_feature_collection()
        
        # Start position tracker (WebSocket or polling mode)
        # Pass paper_trading flag to bypass market hour checks
        position_tracker.start_monitoring(
            use_websocket=self.use_websocket,
            paper_trading=self.paper_trading
        )
        
        # Main loop
        self._run_loop()
    
    def _start_websocket(self) -> None:
        """Initialize and start WebSocket connection."""
        kite = get_kite()
        if not kite:
            logger.error("Cannot start WebSocket - Kite not connected")
            return
        
        try:
            self.websocket_manager = WebSocketTicker(kite)
            position_tracker.set_websocket_manager(self.websocket_manager)
            self.websocket_manager.start()
            logger.info("WebSocket connection started")
        except Exception as e:
            logger.error(f"Failed to start WebSocket: {e}")
            self.use_websocket = False
    
    def _run_loop(self) -> None:
        """Main bot loop with market timing awareness."""
        logger.info("=" * 70)
        logger.info("[BOT] OPTIONS TRADING BOT IS NOW RUNNING (ML-ONLY MODE)")
        logger.info("=" * 70)
        logger.info(f"Mode: {'PAPER TRADING [TEST]' if order_manager.is_paper_trading else '[!] LIVE TRADING [!]'}")
        logger.info(f"Signal Source: ML Model ({signal_generator._predictor.model_version if signal_generator._predictor else 'N/A'})")
        logger.info(f"Auto Trade: {'ENABLED [ON]' if self.auto_trade else 'DISABLED [OFF]'}")
        logger.info(f"Scan Interval: {self.scan_interval} seconds")
        logger.info(f"Overnight carry: {'ENABLED' if MARKET_HOURS.get('carry_overnight', True) else 'DISABLED'}")
        logger.info(f"Underlyings: {', '.join(self.underlyings)}")
        logger.info("=" * 70)
        logger.info("Press Ctrl+C to stop")
        logger.info("")
        
        last_status_log = None
        loop_count = 0
        
        while self.is_running:
            try:
                loop_count += 1
                
                # Get current market status
                market_status = get_market_status()
                now = datetime.now()
                
                # Log status every loop iteration for visibility
                current_time_str = now.strftime("%H:%M:%S")
                
                # Determine current state
                if is_trading_allowed():
                    state = "[ACTIVE] TRADING ACTIVE"
                elif is_market_open():
                    state = "[WAIT] MARKET OPEN (Outside trading window)"
                else:
                    state = "[CLOSED] MARKET CLOSED"
                
                # Log every iteration for paper trading or when state changes
                should_log = (last_status_log is None or 
                             (now - last_status_log).seconds >= 60 or  # Every minute
                             loop_count == 1)
                
                if should_log:
                    logger.info(f"[{current_time_str}] {state} | {market_status['status_message']}")
                    if is_expiry_day():
                        logger.info("[!] Today is an expiry day!")
                    
                    # Show time to next event
                    if market_status.get('time_to_open'):
                        logger.info(f"[TIME] Until trading starts: {market_status['time_to_open']}")
                    elif market_status.get('time_to_close'):
                        logger.info(f"[TIME] Until market close: {market_status['time_to_close']}")
                    
                    last_status_log = now
                
                # Square off check - only if enabled
                if should_square_off():
                    if MARKET_HOURS.get("auto_square_off", False):
                        logger.warning("[SQUARE OFF] Time reached - closing all positions")
                        position_tracker.force_close_all("SQUARE_OFF_TIME")
                        self.auto_trade = False
                        time.sleep(60)
                        continue
                    elif is_expiry_day() and MARKET_HOURS.get("expiry_day_square_off", True):
                        # Force square off on expiry day only
                        logger.warning("[EXPIRY] Expiry day square off - closing expiring positions")
                        position_tracker.force_close_all("EXPIRY_SQUARE_OFF")
                        self.auto_trade = False
                        time.sleep(60)
                        continue
                    else:
                        # Carry overnight - just log it
                        logger.info("[CARRY] Positions will be carried overnight")
                
                # Trading logic
                if is_trading_allowed():
                    logger.info(f"[SCAN] Scanning for trading signals... (Loop #{loop_count})")
                    self._scan_and_trade()
                elif is_market_open():
                    # Market is open but we're outside trading window
                    # Still monitor existing positions
                    logger.info(f"[WAIT] Waiting for trading window to start... Monitoring positions only.")
                else:
                    # Market is closed
                    # Check if we should auto-exit the bot
                    if should_auto_exit():
                        logger.info("=" * 60)
                        logger.info("[AUTO-EXIT] Market closed and auto-exit time reached.")
                        logger.info("[AUTO-EXIT] Shutting down bot gracefully...")
                        logger.info("=" * 60)
                        break  # Exit the main loop
                    
                    logger.info(f"[SLEEP] Market is closed. Waiting... (Next check in 60 seconds)")
                    # Sleep longer when market is closed
                    time.sleep(60)
                    continue
                
                # Check daily loss limit
                if position_tracker.check_daily_loss_limit():
                    logger.warning("[LIMIT] Daily loss limit reached. Stopping auto-trade.")
                    self.auto_trade = False
                
                logger.info(f"[TIMER] Sleeping for {self.scan_interval} seconds before next scan...\n")
                time.sleep(self.scan_interval)
                
            except KeyboardInterrupt:
                logger.info("Shutdown requested...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(self.scan_interval)
        
        self.stop()
    
    def _scan_and_trade(self) -> None:
        """Scan for signals and execute trades if auto-trade is enabled."""
        # Check if we can take more positions
        active_positions = len(order_manager.get_active_positions())
        live_cap = TRADING_CONFIG.get("max_positions", 5)
        paper_cap = TRADING_CONFIG.get("paper_max_positions")
        
        if order_manager.is_paper_trading:
            max_positions = paper_cap or live_cap
        else:
            max_positions = live_cap
        
        logger.info(f"   [POSITIONS] Active: {active_positions}/{max_positions}")
        
        if active_positions >= max_positions:
            logger.warning(f"   [!] Max positions ({max_positions}) reached - skipping signal generation")
            return
        
        # Get underlyings that already have active positions (for deduplication)
        active_underlyings = set()
        for pos in order_manager.get_active_positions():
            underlying = getattr(pos, 'underlying', None) or pos.get('underlying', None) if isinstance(pos, dict) else None
            if underlying:
                active_underlyings.add(underlying)
        
        if active_underlyings:
            logger.info(f"   [DEDUP] Active positions on: {', '.join(active_underlyings)}")
        
        # Generate signals
        logger.info(f"   [SCAN] Generating signals for: {', '.join(signal_generator.underlyings)}")
        signals = signal_generator.generate_signals()
        
        if not signals:
            logger.info("   [INFO] No signals generated (market conditions not favorable)")
            return
        
        # Log signals
        logger.info(f"   [SIGNALS] Generated {len(signals)} signal(s):")
        logger.info("")
        
        for i, signal in enumerate(signals, 1):
            logger.info(f"   Signal #{i}:")
            logger.info(
                f"      Strategy: {signal.strategy_type.value} | "
                f"{signal.underlying} | "
                f"Confidence: {signal.confidence:.2%} | "
                f"RR: {signal.risk_reward_ratio:.2f}"
            )
            logger.info(f"      Rationale: {signal.rationale}")
            logger.info(f"      SL: Rs.{signal.stop_loss:.2f} | Target: Rs.{signal.target:.2f}")
            for leg in signal.legs:
                logger.info(
                    f"      -> {leg.direction.value} {leg.symbol} @ Rs.{leg.entry_price:.2f}"
                )
            logger.info("")
        
        # Execute best signal if auto-trade enabled
        if self.auto_trade and signals:
            executed = False
            
            # Check all signals and execute the first one that meets criteria
            for signal in signals:
                # Position deduplication: skip if we already have a position on this underlying
                if signal.underlying in active_underlyings:
                    logger.info(f"   [DEDUP] Skipping {signal.strategy_type.value} on {signal.underlying} - already have active position")
                    continue
                
                if self._meets_auto_trade_criteria(signal):
                    logger.info(f"   [EXECUTE] Auto-executing signal: {signal.strategy_type.value}")
                    execution = order_manager.execute_signal(signal)
                    
                    if execution.status == "ACTIVE":
                        logger.info("   [OK] Trade executed successfully!")
                        
                        # Log to dedicated trade log file
                        self._log_trade_entry(signal, execution)
                        
                        self._send_notification(
                            f"New Trade Executed!\n"
                            f"Strategy: {signal.strategy_type.value}\n"
                            f"Underlying: {signal.underlying}\n"
                            f"SL: Rs.{signal.stop_loss:.2f}\n"
                            f"Target: Rs.{signal.target:.2f}"
                        )
                        executed = True
                        break  # Only execute one trade per scan
                    elif execution.status == "DUPLICATE":
                        logger.info(f"   [SKIP] Duplicate position already open for {signal.underlying} {signal.strategy_type.value}")
                        # Continue to check other signals
                        continue
                    else:
                        logger.error(f"   [FAIL] Trade execution failed: {execution.status}")
                        trade_logger.info(f"FAILED | {signal.strategy_type.value} | {signal.underlying} | Status: {execution.status}")
                else:
                    logger.info(
                        f"   [SKIP] Signal #{signals.index(signal)+1} ({signal.strategy_type.value}) - doesn't meet criteria"
                    )
            
            if not executed:
                logger.warning("   [!] None of the signals were executed (criteria not met or duplicates)")
        elif signals:
            logger.info("   [INFO] Auto-trade disabled - signals not executed")
    
    def _log_trade_entry(self, signal, execution) -> None:
        """Log trade entry to the dedicated trades.log file."""
        trade_logger.info("=" * 80)
        trade_logger.info(f"NEW TRADE OPENED")
        trade_logger.info("=" * 80)
        trade_logger.info(f"Execution ID: {execution.execution_id}")
        trade_logger.info(f"Strategy:     {signal.strategy_type.value}")
        trade_logger.info(f"Underlying:   {signal.underlying}")
        trade_logger.info(f"Confidence:   {signal.confidence:.1%}")
        trade_logger.info(f"Risk/Reward:  {signal.risk_reward_ratio:.2f}")
        trade_logger.info(f"Mode:         {'PAPER' if order_manager.is_paper_trading else 'LIVE'}")
        trade_logger.info("-" * 40)
        trade_logger.info("LEGS:")
        for leg in signal.legs:
            trade_logger.info(f"  {leg.direction.value:4} | {leg.symbol} | Qty: {leg.quantity} | Entry: Rs.{leg.entry_price:.2f}")
        trade_logger.info("-" * 40)
        trade_logger.info(f"Stop Loss:    Rs.{signal.stop_loss:.2f}")
        trade_logger.info(f"Target:       Rs.{signal.target:.2f}")
        trade_logger.info(f"Max Loss:     Rs.{signal.max_loss:.2f}")
        trade_logger.info(f"Exp Profit:   Rs.{signal.expected_profit:.2f}")
        trade_logger.info(f"Rationale:    {signal.rationale}")
        trade_logger.info("=" * 80)
        trade_logger.info("")

    def _meets_auto_trade_criteria(self, signal) -> bool:
        """
        Check if signal meets auto-trade criteria based on strategy type.
        
        Different strategies have different risk/reward profiles:
        - Directional (Long Call/Put): Require high RR (>= 1.5)
        - Credit strategies (Iron Condor, Short options): Require high confidence (>= 75%)
        - Volatility plays (Straddle/Strangle): Require moderate confidence (>= 70%)
        """
        from strategies.base_strategy import StrategyType
        
        strategy_type = signal.strategy_type
        confidence = signal.confidence
        rr = signal.risk_reward_ratio
        
        # Define criteria per strategy category
        # Directional strategies - need good RR since we're buying premium
        directional_strategies = [
            StrategyType.LONG_CALL,
            StrategyType.LONG_PUT,
        ]
        
        # Credit/selling strategies - high probability, lower RR is acceptable
        credit_strategies = [
            StrategyType.SHORT_CALL,
            StrategyType.SHORT_PUT,
            StrategyType.IRON_CONDOR,
            StrategyType.BEAR_CALL_SPREAD,
            StrategyType.BULL_PUT_SPREAD,
        ]
        
        # Debit spread strategies - balanced approach
        spread_strategies = [
            StrategyType.BULL_CALL_SPREAD,
            StrategyType.BEAR_PUT_SPREAD,
        ]
        
        # Volatility strategies - depend on big moves
        volatility_strategies = [
            StrategyType.STRADDLE,
            StrategyType.STRANGLE,
        ]
        
        # Apply criteria based on strategy type
        if strategy_type in directional_strategies:
            # Directional: Need confidence >= 70% AND RR >= 1.5
            meets_criteria = confidence >= 0.70 and rr >= 1.5
            criteria_desc = f"Conf: {confidence:.0%} >= 70%, RR: {rr:.2f} >= 1.5"
            
        elif strategy_type in credit_strategies:
            # Credit: Need confidence >= 75% AND RR >= 0.3 (lower RR acceptable)
            meets_criteria = confidence >= 0.75 and rr >= 0.3
            criteria_desc = f"Conf: {confidence:.0%} >= 75%, RR: {rr:.2f} >= 0.3"
            
        elif strategy_type in spread_strategies:
            # Spreads: Need confidence >= 70% AND RR >= 1.0
            meets_criteria = confidence >= 0.70 and rr >= 1.0
            criteria_desc = f"Conf: {confidence:.0%} >= 70%, RR: {rr:.2f} >= 1.0"
            
        elif strategy_type in volatility_strategies:
            # Volatility: Need confidence >= 70% AND RR >= 0.8
            meets_criteria = confidence >= 0.70 and rr >= 0.8
            criteria_desc = f"Conf: {confidence:.0%} >= 70%, RR: {rr:.2f} >= 0.8"
            
        else:
            # Default: Standard criteria
            meets_criteria = confidence >= 0.70 and rr >= 1.5
            criteria_desc = f"Conf: {confidence:.0%} >= 70%, RR: {rr:.2f} >= 1.5"
        
        if meets_criteria:
            logger.info(f"   [CRITERIA] {strategy_type.value}: PASSED ({criteria_desc})")
        else:
            logger.info(f"   [CRITERIA] {strategy_type.value}: FAILED ({criteria_desc})")
        
        return meets_criteria
    
    def _start_auto_retrain_monitor(self) -> None:
        """Start background auto-retrain monitor."""
        auto_retrain_config = ML_CONFIG.get("auto_retrain", {})
        
        if not auto_retrain_config.get("enabled", False):
            logger.info("Auto-retrain monitor disabled")
            return
        
        try:
            from ml import get_auto_retrainer
            
            retrainer = get_auto_retrainer()
            check_interval = auto_retrain_config.get("check_interval_seconds", 3600)
            retrainer.start_background_monitor(check_interval=check_interval)
            
            logger.info(f"Auto-retrain monitor started (interval: {check_interval}s)")
            
        except Exception as e:
            logger.warning(f"Could not start auto-retrain monitor: {e}")
    
    def _stop_auto_retrain_monitor(self) -> None:
        """Stop auto-retrain monitor."""
        try:
            from ml import get_auto_retrainer
            retrainer = get_auto_retrainer()
            retrainer.stop_background_monitor()
        except Exception:
            pass
    
    def _start_live_feature_collection(self) -> None:
        """Start background live ML feature collection for training data."""
        try:
            from ml.live_feature_collector import get_collector
            
            collector = get_collector()
            if not collector.running:
                collector.start()
                logger.info("Live ML feature collection started (background)")
            else:
                logger.info("Live ML feature collection already running")
                
        except Exception as e:
            logger.warning(f"Could not start live feature collection: {e}")
    
    def _stop_live_feature_collection(self) -> None:
        """Stop live ML feature collection."""
        try:
            from ml.live_feature_collector import get_collector
            collector = get_collector()
            collector.stop()
        except Exception:
            pass
    
    def stop(self) -> None:
        """Stop the trading bot."""
        logger.info("Stopping bot...")
        self.is_running = False
        
        # Stop auto-retrain monitor
        self._stop_auto_retrain_monitor()
        
        # Stop live feature collection
        self._stop_live_feature_collection()
        
        # Stop WebSocket
        if self.websocket_manager:
            self.websocket_manager.stop()
            logger.info("WebSocket connection stopped")
        
        # Stop position tracker
        position_tracker.stop_monitoring()
        
        logger.info("Bot stopped")
    
    def get_websocket_status(self) -> dict:
        """
        Get WebSocket connection status.
        
        Returns:
            Status dictionary
        """
        if not self.websocket_manager:
            return {"enabled": False, "connected": False}
        
        return {
            "enabled": self.use_websocket,
            "connected": self.websocket_manager.is_connected,
            "subscribed_tokens": len(position_tracker.subscribed_tokens),
            "latest_prices": self.websocket_manager.get_all_prices(),
        }
    
    def get_market_overview(self, underlying: str = None) -> None:
        """
        Print market overview for underlyings.
        
        Args:
            underlying: Specific underlying or None for all
        """
        targets = [underlying] if underlying else self.underlyings
        
        for target in targets:
            overview = signal_generator.get_market_overview(target)
            
            print(f"\n{'=' * 50}")
            print(f"Market Overview: {target}")
            print(f"{'=' * 50}")
            print(f"Spot Price: Rs.{overview.get('spot', 0):,.2f}")
            
            oi = overview.get("oi_analysis", {})
            print(f"\nOpen Interest Analysis:")
            print(f"  PCR: {oi.get('pcr', 0):.2f}")
            print(f"  Max Pain: {oi.get('max_pain')}")
            print(f"  Max Call OI: {oi.get('max_call_oi_strike')}")
            print(f"  Max Put OI: {oi.get('max_put_oi_strike')}")
            print(f"  Sentiment: {oi.get('sentiment')}")
            
            vol = overview.get("volatility", {})
            print(f"\nVolatility:")
            print(f"  HV(20): {vol.get('hv_20', 0):.2f}%")
            print(f"  ATM IV: {vol.get('atm_iv', 0):.2f}%")
            print(f"  IV/HV Ratio: {vol.get('iv_hv_ratio', 0):.2f}")
            print(f"  Regime: {vol.get('regime')}")
            
            print(f"\nRecommended Strategies:")
            for rec in overview.get("recommended_strategies", []):
                print(f"  - {rec}")
    
    def execute_strategy(
        self,
        underlying: str,
        strategy_type: str,
    ) -> None:
        """
        Manually execute a specific strategy.
        
        Args:
            underlying: The underlying asset
            strategy_type: Strategy type to execute
        """
        try:
            strat_type = StrategyType(strategy_type)
        except ValueError:
            logger.error(f"Unknown strategy: {strategy_type}")
            return
        
        signals = signal_generator.generate_signals(underlying, strat_type)
        
        if not signals:
            logger.info("No signal generated for this strategy")
            return
        
        signal = signals[0]
        logger.info(f"Signal: {signal.strategy_type.value}")
        logger.info(f"Confidence: {signal.confidence:.2f}")
        logger.info(f"Rationale: {signal.rationale}")
        
        for leg in signal.legs:
            logger.info(f"  {leg.direction.value} {leg.symbol} @ Rs.{leg.entry_price:.2f}")
        
        logger.info(f"SL: Rs.{signal.stop_loss:.2f}, Target: Rs.{signal.target:.2f}")
        
        # Confirm execution
        confirm = input("\nExecute trade? (y/n): ").strip().lower()
        if confirm == 'y':
            execution = order_manager.execute_signal(signal)
            if execution.status == "ACTIVE":
                logger.info("Trade executed successfully!")
            else:
                logger.error(f"Execution failed: {execution.status}")
    
    def show_positions(self) -> None:
        """Display current positions."""
        positions = position_tracker.get_position_summary()
        
        if not positions:
            print("\nNo active positions")
            return
        
        print(f"\n{'=' * 60}")
        print("Active Positions")
        print(f"{'=' * 60}")
        
        for pos in positions:
            print(f"\nExecution ID: {pos['execution_id']}")
            print(f"Strategy: {pos['strategy']}")
            print(f"Underlying: {pos['underlying']}")
            print(f"Entry Time: {pos['entry_time']}")
            print(f"Current P&L: Rs.{pos.get('current_pnl', 0):.2f}")
            print(f"Stop Loss: Rs.{pos['stop_loss']:.2f}")
            print(f"Target: Rs.{pos['target']:.2f}")
            print("Legs:")
            for leg in pos['legs']:
                print(f"  {leg['direction']} {leg['symbol']} x{leg['quantity']} @ Rs.{leg['entry_price']:.2f}")
        
        print(f"\nTotal Daily P&L: Rs.{position_tracker.get_daily_pnl():.2f}")
    
    def close_all_positions(self) -> None:
        """Close all active positions."""
        confirm = input("Close all positions? (y/n): ").strip().lower()
        if confirm == 'y':
            position_tracker.force_close_all("Manual close by user")
            logger.info("All positions closed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Options Trading Bot")
    parser.add_argument(
        "--underlyings",
        nargs="+",
        default=["NIFTY", "BANKNIFTY"],
        help="Underlyings to trade",
    )
    parser.add_argument(
        "--auto-trade",
        action="store_true",
        help="Enable automatic trade execution",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        default=True,
        help="Enable paper trading mode",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live trading mode (disables paper trading)",
    )
    parser.add_argument(
        "--overview",
        action="store_true",
        help="Show market overview and exit",
    )
    parser.add_argument(
        "--positions",
        action="store_true",
        help="Show current positions and exit",
    )
    
    args = parser.parse_args()
    
    # Determine paper trading mode
    paper_trading = not args.live
    
    # Create bot
    bot = OptionsTradingBot(
        underlyings=args.underlyings,
        auto_trade=args.auto_trade,
        paper_trading=paper_trading,
    )
    
    if args.overview:
        if bot.login():
            for underlying in args.underlyings:
                bot.get_market_overview(underlying)
        return
    
    if args.positions:
        if bot.login():
            bot.show_positions()
        return
    
    # Start the bot
    bot.start()


if __name__ == "__main__":
    main()
