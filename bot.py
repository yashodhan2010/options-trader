"""
Options Trading Bot - Main Orchestrator
"""
import argparse
import time
from datetime import datetime
from typing import Optional, List
import threading

from auth.kite_auth import connect, get_kite, is_authenticated, get_profile, get_margins
from data.data_fetcher import data_fetcher
from data.websocket_manager import WebSocketTicker
from signals.signal_generator import signal_generator
from execution.order_manager import order_manager, OrderType
from execution.position_tracker import position_tracker
from strategies.base_strategy import StrategyType
from config.settings import (
    UNDERLYING_ASSETS, TRADING_CONFIG, MARKET_HOURS, NOTIFICATION_CONFIG, BOT_CONFIG
)
from core.logger import logger
from core.utils import is_market_open, is_trading_allowed, should_square_off, get_market_status, is_expiry_day


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
    
    def _on_sl_hit(self, execution_id: str, pnl: float) -> None:
        """Handle stop loss hit."""
        logger.warning(f"🔴 SL Hit: {execution_id}, Loss: ₹{abs(pnl):.2f}")
        self._send_notification(f"Stop Loss Hit!\nPosition: {execution_id}\nLoss: ₹{abs(pnl):.2f}")
    
    def _on_target_hit(self, execution_id: str, pnl: float) -> None:
        """Handle target hit."""
        logger.info(f"🟢 Target Hit: {execution_id}, Profit: ₹{pnl:.2f}")
        self._send_notification(f"Target Hit!\nPosition: {execution_id}\nProfit: ₹{pnl:.2f}")
    
    def _on_position_closed(self, execution_id: str, pnl: float, reason: str) -> None:
        """Handle position closed."""
        logger.info(f"Position Closed: {execution_id}, P&L: ₹{pnl:.2f}, Reason: {reason}")
    
    def _send_notification(self, message: str) -> None:
        """Send notification via configured channels."""
        if NOTIFICATION_CONFIG.get("telegram_enabled"):
            # TODO: Implement Telegram notification
            pass
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
        logger.info("Starting Options Trading Bot")
        logger.info("=" * 50)
        
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
            logger.info(f"Available margin: ₹{available:,.2f}")
        
        self.is_running = True
        
        # Initialize WebSocket if enabled
        if self.use_websocket:
            self._start_websocket()
        
        # Start position tracker (WebSocket or polling mode)
        position_tracker.start_monitoring(use_websocket=self.use_websocket)
        
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
        logger.info("Bot is now running. Press Ctrl+C to stop.")
        logger.info(f"Overnight carry: {'ENABLED' if MARKET_HOURS.get('carry_overnight', True) else 'DISABLED'}")
        
        last_status_log = None
        
        while self.is_running:
            try:
                # Get current market status
                market_status = get_market_status()
                
                # Log status change periodically (every 5 minutes)
                now = datetime.now()
                if last_status_log is None or (now - last_status_log).seconds >= 300:
                    logger.info(f"📊 {market_status['status_message']}")
                    if is_expiry_day():
                        logger.info("⚠️  Today is an expiry day!")
                    last_status_log = now
                
                # Square off check - only if enabled
                if should_square_off():
                    if MARKET_HOURS.get("auto_square_off", False):
                        logger.warning("⏰ Square off time reached - closing all positions")
                        position_tracker.force_close_all("SQUARE_OFF_TIME")
                        self.auto_trade = False
                        time.sleep(60)
                        continue
                    elif is_expiry_day() and MARKET_HOURS.get("expiry_day_square_off", True):
                        # Force square off on expiry day only
                        logger.warning("⏰ Expiry day square off - closing expiring positions")
                        position_tracker.force_close_all("EXPIRY_SQUARE_OFF")
                        self.auto_trade = False
                        time.sleep(60)
                        continue
                    else:
                        # Carry overnight - just log it
                        logger.info("📦 Positions will be carried overnight")
                
                # Trading logic
                if is_trading_allowed():
                    self._scan_and_trade()
                elif is_market_open():
                    # Market is open but we're outside trading window
                    # Still monitor existing positions
                    logger.debug(f"Outside trading window: {market_status['status_message']}")
                else:
                    # Market is closed
                    logger.debug("Market closed - waiting...")
                    # Sleep longer when market is closed
                    time.sleep(60)
                    continue
                
                # Check daily loss limit
                if position_tracker.check_daily_loss_limit():
                    logger.warning("Daily loss limit reached. Stopping auto-trade.")
                    self.auto_trade = False
                
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
        logger.debug("Scanning for signals...")
        
        # Check if we can take more positions
        active_positions = len(order_manager.get_active_positions())
        max_positions = TRADING_CONFIG.get("max_positions", 5)
        
        if active_positions >= max_positions:
            logger.debug(f"Max positions ({max_positions}) reached")
            return
        
        # Generate signals
        signals = signal_generator.generate_signals()
        
        if not signals:
            logger.debug("No signals generated")
            return
        
        # Log signals
        for signal in signals:
            logger.info(
                f"Signal: {signal.strategy_type.value} | "
                f"{signal.underlying} | "
                f"Confidence: {signal.confidence:.2f} | "
                f"RR: {signal.risk_reward_ratio:.2f}"
            )
            logger.info(f"  Rationale: {signal.rationale}")
            for leg in signal.legs:
                logger.info(
                    f"  Leg: {leg.direction.value} {leg.symbol} @ ₹{leg.entry_price:.2f}"
                )
        
        # Execute best signal if auto-trade enabled
        if self.auto_trade and signals:
            best_signal = signals[0]
            
            # Additional validation before trading
            if best_signal.confidence >= 0.7 and best_signal.risk_reward_ratio >= 1.5:
                logger.info(f"Auto-executing signal: {best_signal.strategy_type.value}")
                execution = order_manager.execute_signal(best_signal)
                
                if execution.status == "ACTIVE":
                    logger.info("Trade executed successfully!")
                    self._send_notification(
                        f"New Trade Executed!\n"
                        f"Strategy: {best_signal.strategy_type.value}\n"
                        f"Underlying: {best_signal.underlying}\n"
                        f"SL: ₹{best_signal.stop_loss:.2f}\n"
                        f"Target: ₹{best_signal.target:.2f}"
                    )
                else:
                    logger.error(f"Trade execution failed: {execution.status}")
            else:
                logger.debug("Signal doesn't meet auto-trade criteria")
    
    def stop(self) -> None:
        """Stop the trading bot."""
        logger.info("Stopping bot...")
        self.is_running = False
        
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
            print(f"Spot Price: ₹{overview.get('spot', 0):,.2f}")
            
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
                print(f"  • {rec}")
    
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
            logger.info(f"  {leg.direction.value} {leg.symbol} @ ₹{leg.entry_price:.2f}")
        
        logger.info(f"SL: ₹{signal.stop_loss:.2f}, Target: ₹{signal.target:.2f}")
        
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
            print(f"Current P&L: ₹{pos.get('current_pnl', 0):.2f}")
            print(f"Stop Loss: ₹{pos['stop_loss']:.2f}")
            print(f"Target: ₹{pos['target']:.2f}")
            print("Legs:")
            for leg in pos['legs']:
                print(f"  {leg['direction']} {leg['symbol']} x{leg['quantity']} @ ₹{leg['entry_price']:.2f}")
        
        print(f"\nTotal Daily P&L: ₹{position_tracker.get_daily_pnl():.2f}")
    
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
