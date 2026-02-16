"""
Interactive CLI for the Options Trading Bot

All signals are ML-driven. No rule-based signal generation.
"""
import cmd
import sys
from datetime import datetime

from bot import OptionsTradingBot
from auth.kite_auth import connect, get_kite, is_authenticated, get_profile, get_margins, logout
from data.data_fetcher import data_fetcher
from signals.ml_signal_generator import ml_signal_generator as signal_generator
from execution.order_manager import order_manager
from execution.position_tracker import position_tracker
from strategies import StrategyType
from config.settings import UNDERLYING_ASSETS, ML_CONFIG
from core.logger import logger


class TradingCLI(cmd.Cmd):
    """Interactive command-line interface for the trading bot."""
    
    intro = """
+---------------------------------------------------------------+
|      OPTIONS TRADING BOT - Interactive Mode (ML-Only)         |
|                                                               |
|  All signals are driven by ML predictions.                    |
|  Type 'help' for available commands                           |
|  Type 'quit' to exit                                          |
+---------------------------------------------------------------+
    """
    prompt = "options> "
    
    def __init__(self):
        super().__init__()
        self.bot = None
        self.logged_in = False
    
    def do_login(self, arg):
        """Login to Kite Connect"""
        if is_authenticated():
            print("Already logged in!")
            profile = get_profile()
            if profile:
                print(f"User: {profile.get('user_name')}")
            return
        
        try:
            connect()
            self.logged_in = True
            print("Login successful!")
            profile = get_profile()
            if profile:
                print(f"Welcome, {profile.get('user_name')}!")
        except Exception as e:
            print(f"Login failed: {e}")
    
    def do_logout(self, arg):
        """Logout from Kite Connect"""
        logout()
        self.logged_in = False
        print("Logged out successfully")
    
    def do_status(self, arg):
        """Show connection and account status"""
        if is_authenticated():
            print("[OK] Connected to Kite")
            profile = get_profile()
            margins = get_margins()
            
            if profile:
                print(f"User: {profile.get('user_name')} ({profile.get('user_id')})")
            
            if margins:
                equity = margins.get("equity", {})
                available = equity.get("available", {}).get("live_balance", 0)
                used = equity.get("utilised", {}).get("debits", 0)
                print(f"Available Margin: Rs.{available:,.2f}")
                print(f"Used Margin: Rs.{used:,.2f}")
        else:
            print("[X] Not connected")
    
    def do_market(self, arg):
        """Show market timing status"""
        from core.utils import get_market_status, is_expiry_day
        from config.settings import MARKET_HOURS
        
        status = get_market_status()
        
        print(f"\n{'='*60}")
        print(f"MARKET STATUS")
        print(f"{'='*60}")
        print(f"Current Time: {status['current_time']}")
        print(f"\n[DATE] Today: {datetime.now().strftime('%A, %B %d, %Y')}")
        
        if status['is_weekend']:
            print("[X] Weekend - Markets Closed")
        else:
            print(f"\n[TIME] Configured Timings:")
            print(f"   Market Open:    {MARKET_HOURS.get('market_open', '09:15')}")
            print(f"   Trading Start:  {MARKET_HOURS.get('trading_start', '09:30')} (after initial volatility)")
            print(f"   Trading End:    {MARKET_HOURS.get('trading_end', '15:15')} (no new positions)")
            print(f"   Square Off:     {MARKET_HOURS.get('square_off_time', '15:20')}")
            print(f"   Market Close:   {MARKET_HOURS.get('market_close', '15:30')}")
            
            print(f"\n[STATUS] Current Status:")
            market_emoji = "[OPEN]" if status['is_market_open'] else "[CLOSED]"
            print(f"   {market_emoji} Market Open: {'Yes' if status['is_market_open'] else 'No'}")
            
            trading_emoji = "[OK]" if status['is_trading_allowed'] else "[WAIT]"
            print(f"   {trading_emoji} Trading Allowed: {'Yes' if status['is_trading_allowed'] else 'No'}")
            
            if status.get('time_to_open'):
                print(f"   [TIMER] Time to trading start: {status['time_to_open']}")
            if status.get('time_to_close'):
                print(f"   [TIMER] Time to market close: {status['time_to_close']}")
            
            # Overnight carry settings
            print(f"\n[POSITION] Position Carry Settings:")
            carry_overnight = MARKET_HOURS.get('carry_overnight', True)
            auto_square_off = MARKET_HOURS.get('auto_square_off', False)
            
            if carry_overnight:
                print(f"   [Y] Overnight Carry: ENABLED (positions will be held)")
            else:
                print(f"   [N] Overnight Carry: DISABLED")
            
            if auto_square_off:
                print(f"   [!] Auto Square Off: ENABLED at {MARKET_HOURS.get('square_off_time', '15:20')}")
            else:
                print(f"   [Y] Auto Square Off: DISABLED")
            
            if is_expiry_day():
                print(f"\n   [!] TODAY IS AN EXPIRY DAY!")
                if MARKET_HOURS.get('expiry_day_square_off', True):
                    print(f"   [!] Expiry positions will be squared off at {MARKET_HOURS.get('expiry_early_exit_time', '14:30')}")
                else:
                    print(f"   Expiry positions can be carried (if not weekly expiry)")
        
        print(f"\n[INFO] {status['status_message']}")
        print(f"{'='*60}")

    def do_spot(self, arg):
        """Get spot price. Usage: spot [NIFTY|BANKNIFTY|FINNIFTY]"""
        underlying = arg.strip().upper() or "NIFTY"
        
        if underlying not in UNDERLYING_ASSETS:
            print(f"Unknown underlying: {underlying}")
            return
        
        spot = data_fetcher.get_spot_price(underlying)
        if spot:
            print(f"{underlying} Spot: Rs.{spot:,.2f}")
        else:
            print("Failed to get spot price")
    
    def do_overview(self, arg):
        """Show market overview. Usage: overview [NIFTY|BANKNIFTY]"""
        underlying = arg.strip().upper() or "NIFTY"
        
        if underlying not in UNDERLYING_ASSETS:
            print(f"Unknown underlying: {underlying}")
            return
        
        overview = signal_generator.get_market_overview(underlying)
        
        print(f"\n{'=' * 50}")
        print(f"Market Overview: {underlying}")
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
        print(f"  Regime: {vol.get('regime')}")
        
        # ML Prediction
        ml_pred = overview.get("ml_prediction")
        if ml_pred:
            print(f"\nML Prediction:")
            print(f"  Direction: {ml_pred.get('direction')}")
            print(f"  Confidence: {ml_pred.get('confidence', 0):.1%}")
            print(f"  Model: {ml_pred.get('model_version')}")
        else:
            print(f"\nML Prediction: Not available (no model loaded)")
        
        print(f"\nRecommended Strategies (based on ML):")
        strategies = overview.get("recommended_strategies", [])
        if strategies:
            for strategy in strategies:
                if hasattr(strategy, 'value'):
                    print(f"  - {strategy.value}")
                else:
                    print(f"  - {strategy}")
        else:
            print("  No strategies recommended")
    
    def do_signals(self, arg):
        """Generate ML signals. Usage: signals [NIFTY|BANKNIFTY]"""
        underlying = arg.strip().upper() if arg.strip() else None
        
        # Check ML model status first
        ml_status = signal_generator.get_model_status()
        if not ml_status.get("model_loaded"):
            print("\n[!] No ML model loaded!")
            print("    Please train a model first: ml train <SYMBOL>")
            print("    Or use: ml train-best <config>")
            return
        
        print(f"\nGenerating ML signals...")
        print(f"Model: {ml_status.get('model_version')} ({ml_status.get('model_type')})")
        print(f"Min Confidence: {ml_status.get('min_confidence', 0):.1%}")
        print()
        
        signals = signal_generator.generate_signals(underlying)
        
        if not signals:
            print("No signals generated (ML confidence below threshold)")
            return
        
        print(f"\nGenerated {len(signals)} ML signal(s):\n")
        
        for i, signal in enumerate(signals, 1):
            ml_dir = signal.metrics.get('ml_direction', 'N/A')
            ml_conf = signal.metrics.get('ml_confidence', 0)
            print(f"{i}. {signal.strategy_type.value} ({signal.underlying})")
            print(f"   ML Direction: {ml_dir} | ML Confidence: {ml_conf:.1%}")
            print(f"   Risk/Reward: {signal.risk_reward_ratio:.2f}")
            print(f"   Rationale: {signal.rationale}")
            print(f"   SL: Rs.{signal.stop_loss:.2f} | Target: Rs.{signal.target:.2f}")
            for leg in signal.legs:
                print(f"   -> {leg.direction.value} {leg.symbol} @ Rs.{leg.entry_price:.2f}")
            print()
    
    def do_strategies(self, arg):
        """List available strategies"""
        print("\nAvailable Strategies:")
        print("-" * 40)
        for st in StrategyType:
            print(f"  - {st.value}")
    
    def do_trade(self, arg):
        """Execute a trade. Usage: trade <UNDERLYING> <STRATEGY>"""
        args = arg.strip().split()
        
        if len(args) < 2:
            print("Usage: trade <UNDERLYING> <STRATEGY>")
            print("Example: trade NIFTY long_call")
            return
        
        underlying = args[0].upper()
        strategy = args[1].lower()
        
        if underlying not in UNDERLYING_ASSETS:
            print(f"Unknown underlying: {underlying}")
            return
        
        try:
            strat_type = StrategyType(strategy)
        except ValueError:
            print(f"Unknown strategy: {strategy}")
            return
        
        signals = signal_generator.generate_signals(underlying, strat_type)
        
        if not signals:
            print("No signal generated for this strategy")
            return
        
        signal = signals[0]
        print(f"\nSignal: {signal.strategy_type.value}")
        print(f"Confidence: {signal.confidence:.2%}")
        print(f"Rationale: {signal.rationale}")
        
        for leg in signal.legs:
            print(f"  -> {leg.direction.value} {leg.symbol} @ Rs.{leg.entry_price:.2f}")
        
        print(f"\nSL: Rs.{signal.stop_loss:.2f} | Target: Rs.{signal.target:.2f}")
        
        confirm = input("\nExecute trade? (y/n): ").strip().lower()
        if confirm == 'y':
            execution = order_manager.execute_signal(signal)
            if execution.status == "ACTIVE":
                print("[OK] Trade executed successfully!")
            else:
                print(f"[X] Execution failed: {execution.status}")
    
    def do_positions(self, arg):
        """Show current positions"""
        positions = position_tracker.get_position_summary()
        
        if not positions:
            print("\nNo active positions")
            return
        
        print(f"\n{'=' * 60}")
        print("Active Positions")
        print(f"{'=' * 60}")
        
        for pos in positions:
            pnl = pos.get('current_pnl', 0)
            pnl_color = "[+]" if pnl >= 0 else "[-]"
            
            print(f"\n{pos['execution_id']}")
            print(f"  Strategy: {pos['strategy']} | {pos['underlying']}")
            print(f"  P&L: {pnl_color} Rs.{pnl:.2f}")
            print(f"  SL: Rs.{pos['stop_loss']:.2f} | Target: Rs.{pos['target']:.2f}")
        
        print(f"\n{'=' * 60}")
        print(f"Total P&L: Rs.{position_tracker.get_daily_pnl():.2f}")
    
    def do_close(self, arg):
        """Close a position. Usage: close <execution_id> or close all"""
        if not order_manager.is_paper_trading and not is_authenticated():
            print("Not logged in! Please login first.")
            return
        if arg.strip().lower() == 'all':
            confirm = input("Close ALL positions? (y/n): ").strip().lower()
            if confirm == 'y':
                position_tracker.force_close_all("Manual close")
                print("All positions closed")
        elif arg.strip():
            success = order_manager.close_position(arg.strip())
            if success:
                print("Position closed")
            else:
                print("Failed to close position")
        else:
            print("Usage: close <execution_id> or close all")
    
    def do_paper(self, arg):
        """Toggle paper trading. Usage: paper [on|off]"""
        if arg.strip().lower() == 'on':
            order_manager.set_paper_trading(True)
            print("Paper trading: ON")
        elif arg.strip().lower() == 'off':
            confirm = input("Enable LIVE trading? (type 'yes' to confirm): ").strip()
            if confirm == 'yes':
                order_manager.set_paper_trading(False)
                print("[!] LIVE trading: ON")
            else:
                print("Cancelled")
        else:
            mode = "ON" if order_manager.is_paper_trading else "OFF"
            print(f"Paper trading: {mode}")
    
    def do_greeks_settings(self, arg):
        """Show Greeks-based exit settings. Usage: greeks_settings [toggle]"""
        from config.settings import GREEKS_EXIT_CONFIG
        
        if arg.strip().lower() == 'toggle':
            GREEKS_EXIT_CONFIG["enabled"] = not GREEKS_EXIT_CONFIG.get("enabled", False)
            status = "ENABLED" if GREEKS_EXIT_CONFIG["enabled"] else "DISABLED"
            print(f"Greeks-based exits: {status}")
            return
        
        print(f"\n{'='*70}")
        print(f"GREEKS-BASED EXIT SETTINGS")
        print(f"{'='*70}")
        
        enabled = GREEKS_EXIT_CONFIG.get("enabled", False)
        print(f"\n[CONFIG] Master Switch: {'[Y] ENABLED' if enabled else '[N] DISABLED'}")
        
        print(f"\n[DELTA] Delta-Based Exits: {'ON' if GREEKS_EXIT_CONFIG.get('delta_exit_enabled') else 'OFF'}")
        print(f"   * Min Delta (Long): {GREEKS_EXIT_CONFIG.get('min_delta_long', 0.10)} - Exit if delta falls below")
        print(f"   * Max Delta (Short): {GREEKS_EXIT_CONFIG.get('max_delta_short', 0.90)} - Exit if delta rises above")
        
        print(f"\n[THETA] Theta-Based Exits: {'ON' if GREEKS_EXIT_CONFIG.get('theta_exit_enabled') else 'OFF'}")
        print(f"   * Decay Threshold: {GREEKS_EXIT_CONFIG.get('theta_decay_threshold', 0.5)*100}% of remaining profit")
        print(f"   * Min DTE: {GREEKS_EXIT_CONFIG.get('days_to_expiry_exit', 2)} days")
        
        print(f"\n[VEGA] Vega-Based Exits (IV Crush): {'ON' if GREEKS_EXIT_CONFIG.get('vega_exit_enabled') else 'OFF'}")
        print(f"   * IV Drop Threshold: {GREEKS_EXIT_CONFIG.get('iv_drop_percent', 20)}%")
        
        print(f"\n[GAMMA] Gamma-Based SL Tightening: {'ON' if GREEKS_EXIT_CONFIG.get('gamma_tighten_enabled') else 'OFF'}")
        print(f"   * Gamma Threshold: {GREEKS_EXIT_CONFIG.get('gamma_threshold', 0.05)}")
        print(f"   * SL Tighten: {GREEKS_EXIT_CONFIG.get('gamma_sl_tighten_percent', 20)}%")
        
        print(f"\n[PROFIT] Profit Lock: {'ON' if GREEKS_EXIT_CONFIG.get('profit_lock_enabled') else 'OFF'}")
        print(f"   * Threshold: {GREEKS_EXIT_CONFIG.get('profit_lock_threshold', 0.5)*100}% of target")
        print(f"   * Lock: {GREEKS_EXIT_CONFIG.get('profit_lock_percent', 0.3)*100}% of profit")
        
        print(f"\n{'='*70}")
        print("Use 'greeks_settings toggle' to enable/disable Greeks exits")

    def do_chain(self, arg):
        """Show options chain. Usage: chain <UNDERLYING>"""
        underlying = arg.strip().upper() or "NIFTY"
        
        if underlying not in UNDERLYING_ASSETS:
            print(f"Unknown underlying: {underlying}")
            return
        
        print(f"\nFetching options chain for {underlying}...")
        chain = data_fetcher.get_options_chain(underlying, num_strikes=5)
        
        if chain.empty:
            print("Failed to fetch options chain")
            return
        
        # Display calls and puts side by side
        calls = chain[chain["option_type"] == "CE"].sort_values("strike")
        puts = chain[chain["option_type"] == "PE"].sort_values("strike")
        
        print(f"\n{'CALLS':^40} | {'PUTS':^40}")
        print("-" * 85)
        print(f"{'Symbol':<15} {'LTP':>8} {'OI':>10} | {'OI':>10} {'LTP':>8} {'Symbol':<15}")
        print("-" * 85)
        
        for (_, call), (_, put) in zip(calls.iterrows(), puts.iterrows()):
            print(
                f"{call['symbol']:<15} {call['ltp']:>8.2f} {call['oi']:>10,} | "
                f"{put['oi']:>10,} {put['ltp']:>8.2f} {put['symbol']:<15}"
            )
    
    def do_greeks(self, arg):
        """Show options chain with Greeks. Usage: greeks <UNDERLYING>"""
        underlying = arg.strip().upper() or "NIFTY"
        
        # Check if it's in UNDERLYING_ASSETS or watchlist
        from config.settings import is_in_watchlist
        if underlying not in UNDERLYING_ASSETS and not is_in_watchlist(underlying):
            print(f"Unknown underlying: {underlying}")
            return
        
        print(f"\nFetching options chain with Greeks for {underlying}...")
        chain = data_fetcher.get_options_chain_with_greeks(underlying, num_strikes=5)
        
        if chain.empty:
            print("Failed to fetch options chain")
            return
        
        # Display with Greeks
        print(f"\n{'='*100}")
        print(f"OPTIONS CHAIN WITH GREEKS - {underlying}")
        print(f"{'='*100}")
        print(f"{'Symbol':<18} {'Type':^4} {'Strike':>8} {'LTP':>8} {'IV':>6} {'Delta':>7} {'Gamma':>8} {'Theta':>7} {'Vega':>6}")
        print("-" * 100)
        
        for _, row in chain.sort_values(['strike', 'option_type']).iterrows():
            print(
                f"{row['symbol']:<18} {row['option_type']:^4} {row['strike']:>8.0f} "
                f"{row['ltp']:>8.2f} {row.get('iv', 0):>5.1f}% {row.get('delta', 0):>7.4f} "
                f"{row.get('gamma', 0):>8.6f} {row.get('theta', 0):>7.2f} {row.get('vega', 0):>6.2f}"
            )
        print(f"{'='*100}")
    
    def do_iv(self, arg):
        """Check IV percentile for an underlying. Usage: iv <UNDERLYING>"""
        underlying = arg.strip().upper() or "NIFTY"
        
        print(f"\nFetching IV analysis for {underlying}...")
        
        # Get ATM option IV
        chain = data_fetcher.get_options_chain(underlying, num_strikes=3)
        if chain.empty:
            print("Failed to fetch data")
            return
        
        # Get ATM IV (average of ATM call and put)
        spot = data_fetcher.get_spot_price(underlying)
        if not spot:
            print("Failed to get spot price")
            return
        
        # Find closest strike to spot
        chain['distance'] = abs(chain['strike'] - spot)
        atm_options = chain.nsmallest(2, 'distance')
        atm_iv = atm_options['iv'].mean()
        
        # Get IV percentile
        iv_data = data_fetcher.get_iv_percentile(underlying, atm_iv)
        
        print(f"\n{'='*50}")
        print(f"IV ANALYSIS - {underlying}")
        print(f"{'='*50}")
        print(f"Spot Price:      Rs.{spot:,.2f}")
        print(f"ATM IV:          {atm_iv:.2f}%")
        print(f"IV Percentile:   {iv_data.get('percentile', 0):.1f}%")
        print(f"IV Regime:       {iv_data.get('regime', 'UNKNOWN')}")
        print(f"Historical Low:  {iv_data.get('historical_low', 0):.1f}%")
        print(f"Historical High: {iv_data.get('historical_high', 0):.1f}%")
        print(f"{'='*50}")
        
        # Strategy suggestions based on IV
        regime = iv_data.get('regime', '')
        print("\n[STRATEGY] Strategy Suggestions:")
        if regime == "HIGH":
            print("  * HIGH IV - Consider selling options (Short Straddle, Iron Condor)")
            print("  * Credit spreads may offer good premium")
        elif regime == "LOW":
            print("  * LOW IV - Consider buying options (Long Straddle, Long Strangle)")
            print("  * Debit spreads may be cheaper to enter")
        else:
            print("  * NORMAL IV - Directional strategies based on view")
    
    def do_watchlist(self, arg):
        """Show current watchlist stocks"""
        from config.settings import get_watchlist_assets, load_watchlist
        
        watchlist = load_watchlist()
        assets = get_watchlist_assets()
        
        print(f"\n{'='*60}")
        print(f"TRADING WATCHLIST")
        print(f"{'='*60}")
        print(f"Status: {'ENABLED' if watchlist.get('enabled') else 'DISABLED'}")
        print(f"{'='*60}")
        print(f"{'Stock':<15} {'Equity Token':>12} {'Lot Size':>10} {'Status':>10}")
        print("-" * 60)
        
        for asset in assets:
            status = "[Y]" if asset.get('enabled', True) else "[N]"
            print(f"{asset['name']:<15} {asset.get('equity_token', 'N/A'):>12} {asset.get('lot_size', 'N/A'):>10} {status:>10}")
        
        print(f"{'='*60}")
    
    def do_risk(self, arg):
        """Show portfolio risk metrics for active positions"""
        positions = order_manager.get_active_positions()
        
        if not positions:
            print("No active positions")
            return
        
        total_delta = 0
        total_gamma = 0
        total_theta = 0
        total_vega = 0
        total_capital = 0
        total_max_loss = 0
        
        print(f"\n{'='*80}")
        print(f"PORTFOLIO RISK METRICS")
        print(f"{'='*80}")
        
        for position in positions:
            execution_id = position['execution_id']
            execution = order_manager.executions.get(execution_id)
            if not execution:
                continue
            
            signal = execution.signal
            
            # Calculate Greeks
            try:
                greeks = data_fetcher.get_strategy_greeks(signal)
                total_delta += greeks.get('delta', 0)
                total_gamma += greeks.get('gamma', 0)
                total_theta += greeks.get('theta', 0)
                total_vega += greeks.get('vega', 0)
            except:
                pass
            
            total_capital += signal.capital_required
            total_max_loss += signal.stop_loss
        
        print(f"\n[GREEKS] Portfolio Greeks:")
        print(f"   Delta:  {total_delta:+.4f} (directional exposure)")
        print(f"   Gamma:  {total_gamma:+.6f} (delta sensitivity)")
        print(f"   Theta:  {total_theta:+.2f} Rs./day (time decay)")
        print(f"   Vega:   {total_vega:+.4f} (volatility sensitivity)")
        
        print(f"\n[CAPITAL] Capital Metrics:")
        print(f"   Capital Deployed: Rs.{total_capital:,.2f}")
        print(f"   Max Risk (SL):    Rs.{total_max_loss:,.2f}")
        print(f"   Active Positions: {len(positions)}")
        
        # Risk interpretation
        print(f"\n[RISK] Risk Interpretation:")
        if abs(total_delta) > 0.5:
            direction = "bullish" if total_delta > 0 else "bearish"
            print(f"   [!] Portfolio is {direction} biased (high delta)")
        else:
            print(f"   [OK] Portfolio is relatively delta neutral")
        
        if total_theta < -50:
            print(f"   [!] High time decay - losing Rs.{abs(total_theta):.0f}/day to theta")
        elif total_theta > 0:
            print(f"   [OK] Positive theta - earning Rs.{total_theta:.0f}/day from time decay")
        
        print(f"{'='*80}")
    
    def do_pnl(self, arg):
        """Show P&L summary for today"""
        from core.database import database
        
        # Get daily summary from database
        today = datetime.now().strftime('%Y-%m-%d')
        
        print(f"\n{'='*60}")
        print(f"P&L SUMMARY - {today}")
        print(f"{'='*60}")
        
        # Realized P&L (closed trades)
        try:
            trades = database.get_trades()
            today_trades = [t for t in trades if t.get('exit_time', '').startswith(today)]
            realized_pnl = sum(t.get('pnl', 0) for t in today_trades)
            closed_count = len(today_trades)
        except:
            realized_pnl = 0
            closed_count = 0
        
        # Unrealized P&L (active positions)
        positions = order_manager.get_active_positions()
        unrealized_pnl = 0
        
        for position in positions:
            execution_id = position['execution_id']
            execution = order_manager.executions.get(execution_id)
            if execution:
                unrealized_pnl += execution.current_pnl or 0
        
        total_pnl = realized_pnl + unrealized_pnl
        
        print(f"\n[CLOSED] Realized P&L:      Rs.{realized_pnl:>+12,.2f}  ({closed_count} trades)")
        print(f"[OPEN] Unrealized P&L:      Rs.{unrealized_pnl:>+12,.2f}  ({len(positions)} positions)")
        print(f"{'-'*60}")
        print(f"[TOTAL] Total P&L:          Rs.{total_pnl:>+12,.2f}")
        
        # Win/loss stats
        if today_trades:
            winners = [t for t in today_trades if t.get('pnl', 0) > 0]
            losers = [t for t in today_trades if t.get('pnl', 0) < 0]
            win_rate = len(winners) / len(today_trades) * 100 if today_trades else 0
            
            print(f"\n[STATS] Today's Stats:")
            print(f"   Win Rate:  {win_rate:.1f}% ({len(winners)}W / {len(losers)}L)")
            if winners:
                print(f"   Best Win:  Rs.{max(t.get('pnl', 0) for t in winners):,.2f}")
            if losers:
                print(f"   Worst Loss: Rs.{min(t.get('pnl', 0) for t in losers):,.2f}")
        
        print(f"{'='*60}")
    
    def do_trades(self, arg):
        """Show past trade history with P&L. Usage: trades [N] or trades all"""
        from core.database import database
        
        # Parse argument: number of trades to show, default 10
        arg = arg.strip().lower()
        show_all = arg == 'all'
        try:
            limit = int(arg) if arg and not show_all else 10
        except ValueError:
            limit = 10
        
        trades = database.get_trades(status="CLOSED")
        
        if not trades:
            print("No closed trades found in database.")
            return
        
        if not show_all:
            trades = trades[:limit]
        
        total_pnl = 0
        wins = 0
        losses = 0
        
        print(f"\n{'='*80}")
        print(f"TRADE HISTORY (showing {'all' if show_all else f'last {len(trades)}'} of {len(database.get_trades(status='CLOSED'))} closed trades)")
        print(f"{'='*80}")
        
        for t in trades:
            pnl = t.get('realized_pnl', 0) or t.get('pnl', 0) or 0
            total_pnl += pnl
            result = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "FLAT"
            if pnl > 0:
                wins += 1
            elif pnl < 0:
                losses += 1
            
            pnl_color = "+" if pnl >= 0 else ""
            
            exec_id = t.get('execution_id', 'N/A')
            underlying = t.get('underlying', 'N/A')
            strategy = t.get('strategy_type', t.get('strategy', 'N/A'))
            entry = t.get('entry_time', 'N/A')
            exit_t = t.get('exit_time', 'N/A')
            
            # Format entry/exit times for readability
            if entry and entry != 'N/A':
                try:
                    entry_dt = datetime.fromisoformat(str(entry))
                    entry = entry_dt.strftime('%Y-%m-%d %H:%M')
                except:
                    pass
            if exit_t and exit_t != 'N/A':
                try:
                    exit_dt = datetime.fromisoformat(str(exit_t))
                    exit_t = exit_dt.strftime('%Y-%m-%d %H:%M')
                except:
                    pass
            
            print(f"\n  [{result:4}] {strategy} | {underlying}")
            print(f"         P&L: {pnl_color}Rs.{pnl:,.2f}")
            print(f"         Entry: {entry} -> Exit: {exit_t}")
            print(f"         ID: {exec_id}")
        
        # Summary
        win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
        print(f"\n{'='*80}")
        print(f"SUMMARY")
        print(f"  Total P&L:  {'+'if total_pnl>=0 else ''}Rs.{total_pnl:,.2f}")
        print(f"  Win Rate:   {win_rate:.1f}% ({wins}W / {losses}L)")
        if wins > 0:
            avg_win = sum(t.get('realized_pnl', 0) or t.get('pnl', 0) or 0 for t in trades if (t.get('realized_pnl', 0) or t.get('pnl', 0) or 0) > 0) / wins
            print(f"  Avg Win:    +Rs.{avg_win:,.2f}")
        if losses > 0:
            avg_loss = sum(t.get('realized_pnl', 0) or t.get('pnl', 0) or 0 for t in trades if (t.get('realized_pnl', 0) or t.get('pnl', 0) or 0) < 0) / losses
            print(f"  Avg Loss:   Rs.{avg_loss:,.2f}")
        print(f"{'='*80}")
    
    def do_history(self, arg):
        """Show historical analysis for an underlying. Usage: history <UNDERLYING>"""
        underlying = arg.strip().upper() or "NIFTY"
        
        # Check if it's in UNDERLYING_ASSETS or watchlist
        from config.settings import is_in_watchlist
        if underlying not in UNDERLYING_ASSETS and not is_in_watchlist(underlying):
            print(f"Unknown underlying: {underlying}")
            return
        
        print(f"\nFetching historical analysis for {underlying}...")
        
        analysis = data_fetcher.get_historical_analysis(underlying)
        if not analysis:
            print("Failed to fetch historical data")
            return
        
        print(f"\n{'='*60}")
        print(f"HISTORICAL ANALYSIS - {underlying}")
        print(f"{'='*60}")
        
        # Trend info
        trend = analysis.get('trend', 'UNKNOWN')
        trend_emoji = {"BULLISH": "[UP]", "BEARISH": "[DN]", "NEUTRAL": "[--]"}.get(trend, "[??]")
        print(f"\n{trend_emoji} Trend: {trend}")
        print(f"   Trend Strength: {analysis.get('trend_strength', 0)*100:.1f}%")
        
        # RSI
        rsi = analysis.get('rsi', 50)
        rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
        print(f"\n[RSI] RSI: {rsi:.1f} ({rsi_status})")
        
        # Momentum
        mom = analysis.get('momentum', 0)
        mom_emoji = "[+]" if mom > 0 else "[-]"
        print(f"{mom_emoji} Momentum: {mom*100:+.2f}%")
        
        # Confidence
        confidence = analysis.get('confidence_boost', 0)
        print(f"\n[SIGNAL] Confidence Boost: {confidence*100:+.1f}%")
        
        # Recent price action
        print(f"\n[PRICE] Recent Highs/Lows:")
        print(f"   High (20D): Rs.{analysis.get('high_20d', 0):,.2f}")
        print(f"   Low (20D):  Rs.{analysis.get('low_20d', 0):,.2f}")
        print(f"   Current:    Rs.{analysis.get('current_price', 0):,.2f}")
        
        print(f"{'='*60}")
    
    def do_start(self, arg):
        """Start the trading bot in auto mode"""
        self.bot = OptionsTradingBot(
            underlyings=list(UNDERLYING_ASSETS.keys()),
            auto_trade=True,
            paper_trading=order_manager.is_paper_trading,
        )
        
        print("Starting bot in auto mode...")
        print("Press Ctrl+C to stop")
        
        try:
            self.bot.start()
        except KeyboardInterrupt:
            print("\nStopping bot...")
    
    # ========== ML COMMANDS ==========
    
    def do_ml(self, arg):
        """
        ML model commands. Usage:
            ml status              - Show ML status and model info
            ml train [UNDERLYING]  - Train ML model (default: all)
            ml backtest UNDERLYING - Run backtest on trained model
            ml paper start         - Start ML paper trading session
            ml paper stop          - Stop paper trading and show results
            ml paper stats         - Show current paper trading stats
            ml predict UNDERLYING  - Get ML prediction for underlying
            ml features UNDERLYING - Show extracted features
            ml drift               - Check for model drift
            ml compare UNDERLYING  - Compare model configurations with trading metrics
            ml train-best [MODEL]  - Train all symbols with best configuration (default: rf_aggressive)
            ml retrain             - Retrain from feedback data (trade outcomes)
            ml retrain-status      - Show auto-retrain status and conditions
            ml collect [start|stop|status|once] - Live feature collection for training
            ml label               - Label collected snapshots with outcomes
            ml historical [status|date|range|fill|kite] - Download NSE bhavcopy historical data
            ml backfill [DAYS]     - Backfill last N days of historical data (uses jugaad-data)
            ml train-full          - Train all symbols using NSE bhavcopy with 59 features
            ml train-monthly       - Run monthly update pipeline (download + train)
            ml models              - Show trained per-symbol models and metrics
        """
        args = arg.strip().split()
        
        if not args:
            print("Usage: ml <command> [args]")
            print("Commands: status, train, train-best, train-full, train-monthly,")
            print("          backtest, paper, predict, features, drift, compare,")
            print("          retrain, retrain-status, collect, label, historical,")
            print("          backfill, models")
            return
        
        cmd = args[0].lower()
        
        if cmd == "status":
            self._ml_status()
        elif cmd == "train":
            underlying = args[1].upper() if len(args) > 1 else None
            self._ml_train(underlying)
        elif cmd == "backtest":
            if len(args) < 2:
                print("Usage: ml backtest UNDERLYING")
                return
            self._ml_backtest(args[1].upper())
        elif cmd == "paper":
            if len(args) < 2:
                print("Usage: ml paper [start|stop|stats]")
                return
            self._ml_paper(args[1].lower())
        elif cmd == "predict":
            if len(args) < 2:
                print("Usage: ml predict UNDERLYING")
                return
            self._ml_predict(args[1].upper())
        elif cmd == "features":
            if len(args) < 2:
                print("Usage: ml features UNDERLYING")
                return
            self._ml_features(args[1].upper())
        elif cmd == "drift":
            self._ml_drift()
        elif cmd == "compare":
            if len(args) < 2:
                print("Usage: ml compare UNDERLYING")
                return
            self._ml_compare(args[1].upper())
        elif cmd == "train-best":
            model_type = args[1].lower() if len(args) > 1 else "rf_aggressive"
            self._ml_train_best(model_type)
        elif cmd == "retrain":
            force = "--force" in args or "-f" in args
            self._ml_retrain_feedback(force)
        elif cmd == "retrain-status":
            self._ml_retrain_status()
        elif cmd == "collect":
            action = args[1].lower() if len(args) > 1 else "status"
            self._ml_collect(action)
        elif cmd == "label":
            self._ml_label()
        elif cmd == "historical":
            action = args[1].lower() if len(args) > 1 else "status"
            self._ml_historical(action, args[2:] if len(args) > 2 else [])
        elif cmd == "backfill":
            days = int(args[1]) if len(args) > 1 else 30
            self._ml_backfill(days)
        elif cmd == "train-full":
            self._ml_train_full(args[1:])
        elif cmd == "train-monthly":
            force = "--force" in args or "-f" in args
            self._ml_train_monthly(force)
        elif cmd == "models":
            self._ml_models()
        else:
            print(f"Unknown ML command: {cmd}")
    
    def _ml_status(self):
        """Show ML system status."""
        print(f"\n{'='*60}")
        print("ML SYSTEM STATUS (ML-Only Mode)")
        print(f"{'='*60}")
        
        # Get signal generator status
        ml_status = signal_generator.get_model_status()
        
        print(f"\n[SIGNAL GENERATOR]")
        print(f"   Status: {ml_status.get('status', 'unknown').upper()}")
        print(f"   Model Loaded: {'YES' if ml_status.get('model_loaded') else 'NO'}")
        if ml_status.get('model_loaded'):
            print(f"   Model Version: {ml_status.get('model_version')}")
            print(f"   Model Type: {ml_status.get('model_type')}")
        print(f"   Min Confidence: {ml_status.get('min_confidence', 0):.1%}")
        print(f"   Trading Symbols: {', '.join(ml_status.get('underlyings', []))}")
        
        enabled = ML_CONFIG.get("enabled", False)
        print(f"\n[CONFIG] ML Enabled: {'YES' if enabled else 'NO'}")
        print(f"   Confidence Weight: {ML_CONFIG.get('confidence_weight', 0.5):.0%}")
        print(f"   Optuna Trials: {ML_CONFIG.get('optuna_trials', 50)}")
        
        # Show training symbols
        symbols = ML_CONFIG.get("training_symbols", [])
        print(f"\n[SYMBOLS] Training Symbols ({len(symbols)}):")
        for sym in symbols:
            print(f"   - {sym}")
        
        weights = ML_CONFIG.get("ensemble_weights", {})
        print(f"\n[ENSEMBLE] Model Weights:")
        print(f"   XGBoost: {weights.get('xgboost', 0.5):.0%}")
        print(f"   LightGBM: {weights.get('lightgbm', 0.3):.0%}")
        print(f"   Random Forest: {weights.get('random_forest', 0.2):.0%}")
        
        guardrails = ML_CONFIG.get("guardrails", {})
        print(f"\n[GUARDRAILS]")
        print(f"   Stop-Loss Sacred: {guardrails.get('stop_loss_sacred', True)}")
        print(f"   Max Confidence Adj: +/-{guardrails.get('max_confidence_adjustment', 0.3):.0%}")
        print(f"   Circuit Breaker: {guardrails.get('circuit_breaker_losses', 3)} losses")
        
        if enabled:
            try:
                from ml import get_model_registry, get_feedback_collector
                
                registry = get_model_registry()
                prod_model = registry.get_production_model()
                print(f"\n[REGISTRY] Production Model: {prod_model or 'None'}")
                
                feedback = get_feedback_collector()
                stats = feedback.get_performance_stats(days=7)
                if stats.get("total_predictions", 0) > 0:
                    print(f"\n[PERFORMANCE] Last 7 Days:")
                    print(f"   Predictions: {stats.get('total_predictions', 0)}")
                    print(f"   Accuracy: {stats.get('accuracy', 0):.1%}")
                    print(f"   Avg P&L: Rs.{stats.get('avg_pnl', 0):,.2f}")
            except Exception as e:
                print(f"\n[!] Could not load ML components: {e}")
        
        if not ml_status.get('model_loaded'):
            print(f"\n[!] WARNING: No ML model loaded!")
            print(f"    The bot cannot generate signals without a trained model.")
            print(f"    Train a model using: ml train <SYMBOL>")
            print(f"    Or use: ml train-best <config>")
        
        print(f"\n{'='*60}")
    
    def _ml_train(self, underlying=None):
        """Train ML model."""
        if not ML_CONFIG.get("enabled", False):
            print("ML is disabled. Enable it in config/settings.py")
            return
        
        print("\n[TRAINING] Starting ML model training...")
        
        try:
            from ml import get_model_trainer, get_data_collector, get_feature_engineer
            
            trainer = get_model_trainer()
            collector = get_data_collector()
            feature_engineer = get_feature_engineer()
            
            # Use training_symbols from ML_CONFIG, or specific underlying
            if underlying:
                underlyings = [underlying]
            else:
                underlyings = ML_CONFIG.get("training_symbols", list(UNDERLYING_ASSETS.keys()))
            
            print(f"[INFO] Training on {len(underlyings)} symbols: {', '.join(underlyings)}")
            
            for ul in underlyings:
                print(f"\n[{ul}] Collecting historical data...")
                
                # Collect data (synchronous call) - use config setting
                days = ML_CONFIG.get("historical_days", 180)
                data = collector.collect_historical_data(symbols=[ul], days=days)
                if not data or ul not in data:
                    print(f"[{ul}] Warning: No historical data collected, skipping...")
                    continue
                
                df = data[ul]
                print(f"[{ul}] Collected {len(df)} records")
                
                # Extract features using batch method (use smaller lookback for limited data)
                print(f"[{ul}] Extracting features...")
                lookback = min(20, len(df) - 15)  # Adaptive lookback based on data size
                if lookback < 15:
                    print(f"[{ul}] Warning: Not enough data ({len(df)} rows), need at least 35. Skipping...")
                    continue
                    
                X, feature_names = feature_engineer.extract_features_batch(df, lookback=lookback)
                
                if X is None or len(X) == 0:
                    print(f"[{ul}] Warning: Could not extract features, skipping...")
                    continue
                
                # Create target variable (predict next day direction)
                # 1 = price went up, 0 = price went down
                df_cols = [c.lower() for c in df.columns]
                close_col = 'close' if 'close' in df_cols else df.columns[df_cols.index('close')] if 'close' in df_cols else None
                
                if close_col is None:
                    print(f"[{ul}] Warning: No 'close' column found, skipping...")
                    continue
                
                # Create target: 1 if next day close > current close, 0 otherwise
                closes = df[close_col].values
                # Target for each day in X (X already starts from lookback)
                # X[i] corresponds to df row (lookback + i)
                # Target for X[i] is whether df row (lookback + i + 1) went up
                y_full = (closes[1:] > closes[:-1]).astype(int)  # Direction for each day
                y = y_full[lookback:]  # Align with X start point
                
                # Remove last sample (no future target available)
                if len(X) > len(y):
                    X = X[:len(y)]
                if len(y) > len(X):
                    y = y[:len(X)]
                
                # Need last one less since we predict "next day"
                X = X[:-1]
                y = y[1:]  # Shift target by 1 to predict next day
                
                if len(X) < 10:
                    print(f"[{ul}] Warning: Too few samples ({len(X)}), skipping...")
                    continue
                
                print(f"[{ul}] Training with Optuna optimization ({len(X)} samples, {len(feature_names)} features)...")
                
                model, metrics, model_version = trainer.train_direction_model(
                    X=X,
                    y=y,
                    feature_names=feature_names,
                    model_type="ensemble",
                    optimize=True
                )
                
                print(f"\n[{ul}] Training Complete!")
                print(f"   Model Version: {model_version}")
                print(f"   Accuracy: {metrics.get('accuracy', 0):.2%}")
                print(f"   F1 Score: {metrics.get('f1_score', 0):.2%}")
                print(f"   AUC-ROC: {metrics.get('auc_roc', 0):.2%}")
                
        except Exception as e:
            print(f"[ERROR] Training failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _ml_backtest(self, underlying):
        """Run backtest on ML model."""
        if underlying not in UNDERLYING_ASSETS:
            print(f"Unknown underlying: {underlying}")
            return
        
        print(f"\n[BACKTEST] Running backtest for {underlying}...")
        
        try:
            from ml.backtester import Backtester
            from datetime import timedelta
            
            bt = Backtester()
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            result = bt.run_ml_backtest(
                underlying=underlying,
                start_date=start_date,
                end_date=end_date
            )
            
            print(f"\n[RESULTS] Backtest Results ({start_date.date()} to {end_date.date()}):")
            print(f"   Total Trades: {result.total_trades}")
            print(f"   Win Rate: {result.win_rate:.1%}")
            print(f"   Total P&L: Rs.{result.total_pnl:,.2f}")
            print(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}")
            print(f"   Max Drawdown: {result.max_drawdown:.1%}")
            print(f"   Profit Factor: {result.profit_factor:.2f}")
            
        except Exception as e:
            print(f"[ERROR] Backtest failed: {e}")
    
    def _ml_paper(self, action):
        """Manage ML paper trading session."""
        try:
            from ml import get_paper_trading_runner
            
            runner = get_paper_trading_runner()
            
            if action == "start":
                capital = ML_CONFIG.get("paper_trading", {}).get("initial_capital", 100000)
                session_id = runner.start_session(initial_capital=capital)
                print(f"\n[PAPER] Paper trading session started!")
                print(f"   Session ID: {session_id}")
                print(f"   Initial Capital: Rs.{capital:,.2f}")
                print(f"\n   Use 'ml paper stats' to check progress")
                print(f"   Use 'ml paper stop' to end session")
                
            elif action == "stop":
                summary = runner.end_session()
                if "error" in summary:
                    print(f"[!] {summary['error']}")
                    return
                
                print(f"\n[PAPER] Session Ended!")
                print(f"{'='*50}")
                print(f"   Duration: {summary.get('duration_hours', 0):.1f} hours")
                print(f"   Total Trades: {summary.get('total_trades', 0)}")
                print(f"   Win Rate: {summary.get('win_rate', 0):.1%}")
                print(f"   Total P&L: Rs.{summary.get('total_pnl', 0):,.2f}")
                print(f"   Return: {summary.get('total_pnl_pct', 0):.2%}")
                print(f"   Sharpe Ratio: {summary.get('sharpe_ratio', 0):.2f}")
                print(f"   Max Drawdown: {summary.get('max_drawdown', 0):.1%}")
                print(f"{'='*50}")
                
            elif action == "stats":
                stats = runner.get_current_stats()
                if not stats:
                    print("[!] No active paper trading session")
                    return
                
                print(f"\n[PAPER] Current Session Stats:")
                print(f"   Session: {stats.get('session_id', 'N/A')}")
                print(f"   Capital: Rs.{stats.get('final_capital', 0):,.2f}")
                print(f"   P&L: Rs.{stats.get('total_pnl', 0):,.2f} ({stats.get('total_pnl_pct', 0):.2%})")
                print(f"   Trades: {stats.get('total_trades', 0)} (Win: {stats.get('winning_trades', 0)})")
                print(f"   Active: {stats.get('active_trades', 0)} positions")
                
            else:
                print(f"Unknown paper action: {action}")
                print("Use: ml paper [start|stop|stats]")
                
        except Exception as e:
            print(f"[ERROR] Paper trading error: {e}")
    
    def _ml_predict(self, underlying):
        """Get ML prediction for underlying."""
        if underlying not in UNDERLYING_ASSETS:
            print(f"Unknown underlying: {underlying}")
            return
        
        if not is_authenticated():
            print("Please login first to get live data")
            return
        
        print(f"\n[PREDICT] Getting ML prediction for {underlying}...")
        
        try:
            from ml import get_predictor, get_feature_engineer
            
            # Get market data
            spot = data_fetcher.get_spot_price(underlying)
            oi_data = data_fetcher.get_oi_data(underlying)
            volatility = data_fetcher.get_volatility_metrics(underlying)
            historical = data_fetcher.get_historical_analysis(underlying, days=30)
            
            market_data = {
                "oi_data": oi_data,
                "volatility": volatility,
                "historical": historical,
            }
            
            # Extract features
            fe = get_feature_engineer()
            features = fe.extract_features(
                spot_price=spot,
                market_data=market_data,
                underlying=underlying,
                strategy_type="LONG_CALL"
            )
            
            # Get prediction
            predictor = get_predictor()
            prediction = predictor.predict(features, underlying, "LONG_CALL")
            
            print(f"\n[RESULT] ML Prediction for {underlying}:")
            print(f"   Spot: Rs.{spot:,.2f}")
            print(f"   Direction: {prediction.direction}")
            print(f"   Confidence: {prediction.confidence:.1%}")
            print(f"   Model: {prediction.model_version}")
            
            if prediction.feature_importance:
                print(f"\n   Top Features:")
                for feat, imp in list(prediction.feature_importance.items())[:5]:
                    print(f"      {feat}: {imp:.3f}")
                    
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
    
    def _ml_features(self, underlying):
        """Show extracted features for underlying."""
        if underlying not in UNDERLYING_ASSETS:
            print(f"Unknown underlying: {underlying}")
            return
        
        if not is_authenticated():
            print("Please login first to get live data")
            return
        
        print(f"\n[FEATURES] Extracting features for {underlying}...")
        
        try:
            from ml import get_feature_engineer
            
            spot = data_fetcher.get_spot_price(underlying)
            oi_data = data_fetcher.get_oi_data(underlying)
            volatility = data_fetcher.get_volatility_metrics(underlying)
            historical = data_fetcher.get_historical_analysis(underlying, days=30)
            
            fe = get_feature_engineer()
            features = fe.extract_features(
                spot_price=spot,
                market_data={
                    "oi_data": oi_data,
                    "volatility": volatility,
                    "historical": historical,
                },
                underlying=underlying,
                strategy_type="LONG_CALL"
            )
            
            print(f"\n[{underlying}] Extracted {len(features)} Features:")
            print(f"{'='*50}")
            
            # Group by category
            categories = {
                "Price": [k for k in features if k.startswith("price_")],
                "Technical": [k for k in features if any(k.startswith(p) for p in ["rsi", "macd", "bb_", "ema", "sma"])],
                "Options": [k for k in features if any(k.startswith(p) for p in ["iv_", "delta", "gamma", "theta", "vega"])],
                "OI": [k for k in features if k.startswith("oi_")],
                "Volatility": [k for k in features if k.startswith("vol_")],
                "Time": [k for k in features if any(k.startswith(p) for p in ["time_", "dte", "day_"])],
            }
            
            for cat, keys in categories.items():
                if keys:
                    print(f"\n[{cat}]")
                    for key in keys[:5]:  # Show top 5 per category
                        print(f"   {key}: {features.get(key, 0):.4f}")
                    if len(keys) > 5:
                        print(f"   ... and {len(keys) - 5} more")
            
        except Exception as e:
            print(f"[ERROR] Feature extraction failed: {e}")
    
    def _ml_drift(self):
        """Check for model drift."""
        try:
            from ml import get_feedback_collector
            
            feedback = get_feedback_collector()
            stats = feedback.get_performance_stats(days=30)
            summary = feedback.get_training_data_summary()
            
            print(f"\n[DRIFT] Model Drift Analysis:")
            print(f"{'='*50}")
            
            print(f"\n[DATA] Training Data:")
            print(f"   Total Samples: {summary.get('total_samples', 0)}")
            print(f"   With Outcomes: {summary.get('samples_with_outcome', 0)}")
            print(f"   Win Rate: {summary.get('win_rate', 0):.1%}")
            
            print(f"\n[PERFORMANCE] Recent Performance:")
            print(f"   Predictions: {stats.get('total_predictions', 0)}")
            print(f"   Accuracy: {stats.get('accuracy', 0):.1%}")
            
            if feedback._baseline_accuracy:
                current = stats.get("accuracy", feedback._baseline_accuracy)
                drift = feedback._baseline_accuracy - current
                print(f"\n[DRIFT] Baseline: {feedback._baseline_accuracy:.1%}")
                print(f"   Current: {current:.1%}")
                print(f"   Drift: {drift:+.1%}")
                
                if drift > 0.1:
                    print(f"\n   [!] RETRAINING RECOMMENDED (drift > 10%)")
            
            if feedback.should_retrain():
                print(f"\n   [!] Model retraining is recommended")
            else:
                print(f"\n   [OK] Model performance is stable")
                
        except Exception as e:
            print(f"[ERROR] Drift check failed: {e}")
    
    def _ml_compare(self, underlying):
        """Compare model configurations with trading-specific metrics."""
        # Allow both indices (from UNDERLYING_ASSETS) and stocks (from training_symbols)
        training_symbols = ML_CONFIG.get("training_symbols", [])
        valid_symbols = list(UNDERLYING_ASSETS.keys()) + training_symbols
        
        if underlying not in valid_symbols:
            print(f"Unknown underlying: {underlying}")
            print(f"Valid symbols: {', '.join(valid_symbols)}")
            return
        
        print(f"\n{'='*60}")
        print(f"MODEL COMPARISON - {underlying}")
        print(f"{'='*60}")
        print("\nComparing 12 model configurations with trading metrics...")
        print("This will test: XGBoost, LightGBM, Random Forest, and Ensemble variants\n")
        
        try:
            from ml.evaluator import ModelComparator, TradingEvaluator
            from ml import get_data_collector, get_feature_engineer, get_mlflow_tracker
            
            collector = get_data_collector()
            feature_engineer = get_feature_engineer()
            mlflow_tracker = get_mlflow_tracker()
            
            # Collect data
            print(f"[1/4] Collecting historical data...")
            days = ML_CONFIG.get("historical_days", 180)
            data = collector.collect_historical_data(symbols=[underlying], days=days)
            
            if not data or underlying not in data:
                print(f"[ERROR] Could not collect data for {underlying}")
                return
            
            df = data[underlying]
            print(f"   Collected {len(df)} records")
            
            # Extract features
            print(f"[2/4] Extracting features...")
            lookback = min(20, len(df) - 15)
            X, feature_names = feature_engineer.extract_features_batch(df, lookback=lookback)
            
            if X is None or len(X) == 0:
                print(f"[ERROR] Could not extract features")
                return
            
            # Create target variable
            df_cols = [c.lower() for c in df.columns]
            close_col = 'close' if 'close' in df_cols else df.columns[df_cols.index('close')]
            closes = df[close_col].values
            y_full = (closes[1:] > closes[:-1]).astype(int)
            y = y_full[lookback:]
            
            if len(X) > len(y):
                X = X[:len(y)]
            if len(y) > len(X):
                y = y[:len(X)]
            
            X = X[:-1]
            y = y[1:]
            
            # Get prices for simulation
            prices = closes[lookback + 1:lookback + 1 + len(y)]
            
            print(f"   Samples: {len(X)}, Features: {len(feature_names)}")
            
            # Run comparison
            print(f"[3/4] Running model comparisons (12 configurations)...")
            
            import mlflow
            # Set up MLflow experiment for comparison
            mlflow.set_tracking_uri(f"file:///{ML_CONFIG.get('mlflow_uri', 'data/mlflow').replace(chr(92), '/')}")
            mlflow.set_experiment("model_comparison")
            
            with mlflow.start_run(run_name=f"compare_{underlying}"):
                mlflow.log_param("underlying", underlying)
                mlflow.log_param("samples", len(X))
                mlflow.log_param("features", len(feature_names))
                
                comparator = ModelComparator(mlflow_tracker)
                results_df = comparator.compare_configurations(
                    X, y, feature_names, prices
                )
            
            # Generate and display report
            print(f"[4/4] Generating comparison report...")
            report = comparator.generate_report()
            print(f"\n{report}")
            
            # Show best configuration
            best = comparator.get_best_configuration(
                metric='sharpe_ratio',
                secondary_metrics=['profit_factor', 'win_rate']
            )
            
            print(f"\n{'='*60}")
            print("RECOMMENDED CONFIGURATION")
            print(f"{'='*60}")
            print(f"\n   Best Model: {best.get('config_name', 'N/A')}")
            print(f"   Model Type: {best.get('model_type', 'N/A')}")
            print(f"\n   Key Metrics:")
            print(f"      Sharpe Ratio: {best.get('sharpe_ratio', 0):.4f}")
            print(f"      Profit Factor: {best.get('profit_factor', 0):.4f}")
            print(f"      Win Rate: {best.get('win_rate', 0):.2%}")
            print(f"      Max Drawdown: {best.get('max_drawdown', 0):.2%}")
            print(f"      Risk/Reward: {best.get('risk_reward_ratio', 0):.2f}")
            
            print(f"\n   Signal Quality:")
            print(f"      Bullish Precision: {best.get('bullish_precision', 0):.2%}")
            print(f"      Bearish Precision: {best.get('bearish_precision', 0):.2%}")
            print(f"      False Signal Rate: {best.get('false_signal_rate', 0):.2%}")
            
            print(f"\n[OK] Results logged to MLflow experiment 'model_comparison'")
            print(f"     View at: http://127.0.0.1:5000")
            
        except Exception as e:
            print(f"[ERROR] Comparison failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _ml_train_best(self, config_name="rf_aggressive"):
        """Train all symbols with the best configuration from comparison."""
        if not ML_CONFIG.get("enabled", False):
            print("ML is disabled. Enable it in config/settings.py")
            return
        
        # Define best configurations based on comparison results
        BEST_CONFIGS = {
            "rf_aggressive": {
                "model_type": "rf",
                "params": {
                    "n_estimators": 100,
                    "max_depth": 20,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "max_features": "sqrt"
                }
            },
            "rf_balanced": {
                "model_type": "rf",
                "params": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2,
                    "max_features": "sqrt"
                }
            },
            "rf_conservative": {
                "model_type": "rf",
                "params": {
                    "n_estimators": 100,
                    "max_depth": 5,
                    "min_samples_split": 10,
                    "min_samples_leaf": 5,
                    "max_features": "sqrt"
                }
            },
            "xgb_aggressive": {
                "model_type": "xgboost",
                "params": {
                    "n_estimators": 200,
                    "max_depth": 10,
                    "learning_rate": 0.1,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9
                }
            },
            "xgb_balanced": {
                "model_type": "xgboost",
                "params": {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8
                }
            },
            "lgb_aggressive": {
                "model_type": "lightgbm",
                "params": {
                    "n_estimators": 200,
                    "max_depth": 15,
                    "learning_rate": 0.1,
                    "num_leaves": 63
                }
            },
            "lgb_balanced": {
                "model_type": "lightgbm",
                "params": {
                    "n_estimators": 100,
                    "max_depth": 8,
                    "learning_rate": 0.05,
                    "num_leaves": 31
                }
            }
        }
        
        if config_name not in BEST_CONFIGS:
            print(f"Unknown configuration: {config_name}")
            print(f"Available: {', '.join(BEST_CONFIGS.keys())}")
            return
        
        config = BEST_CONFIGS[config_name]
        model_type = config["model_type"]
        params = config["params"]
        
        print(f"\n{'='*60}")
        print(f"TRAINING ALL SYMBOLS WITH BEST CONFIGURATION")
        print(f"{'='*60}")
        print(f"\nConfiguration: {config_name}")
        print(f"Model Type: {model_type}")
        print(f"Parameters: {params}")
        
        try:
            from ml import get_data_collector, get_feature_engineer, get_mlflow_tracker
            from ml.model_trainer import ModelTrainer
            import mlflow
            import numpy as np
            
            collector = get_data_collector()
            feature_engineer = get_feature_engineer()
            mlflow_tracker = get_mlflow_tracker()
            
            # Set up MLflow experiment
            mlflow.set_tracking_uri(f"file:///{ML_CONFIG.get('mlflow_uri', 'data/mlflow').replace(chr(92), '/')}")
            mlflow.set_experiment("best_config_training")
            
            # Get training symbols
            symbols = ML_CONFIG.get("training_symbols", list(UNDERLYING_ASSETS.keys()))
            print(f"\nTraining {len(symbols)} symbols: {', '.join(symbols)}")
            print(f"{'='*60}\n")
            
            results = {}
            
            for symbol in symbols:
                print(f"\n[{symbol}] Starting training...")
                
                try:
                    # Collect data
                    days = ML_CONFIG.get("historical_days", 180)
                    data = collector.collect_historical_data(symbols=[symbol], days=days)
                    
                    if not data or symbol not in data:
                        print(f"   [SKIP] No data for {symbol}")
                        results[symbol] = {"status": "skipped", "reason": "no data"}
                        continue
                    
                    df = data[symbol]
                    print(f"   Collected {len(df)} records")
                    
                    # Extract features
                    lookback = min(20, len(df) - 15)
                    if lookback < 15:
                        print(f"   [SKIP] Insufficient data ({len(df)} rows)")
                        results[symbol] = {"status": "skipped", "reason": "insufficient data"}
                        continue
                    
                    X, feature_names = feature_engineer.extract_features_batch(df, lookback=lookback)
                    
                    if X is None or len(X) == 0:
                        print(f"   [SKIP] Could not extract features")
                        results[symbol] = {"status": "skipped", "reason": "feature extraction failed"}
                        continue
                    
                    # Create target variable
                    df_cols = [c.lower() for c in df.columns]
                    close_col = 'close' if 'close' in df_cols else df.columns[df_cols.index('close')]
                    closes = df[close_col].values
                    y_full = (closes[1:] > closes[:-1]).astype(int)
                    y = y_full[lookback:]
                    
                    if len(X) > len(y):
                        X = X[:len(y)]
                    if len(y) > len(X):
                        y = y[:len(X)]
                    
                    X = X[:-1]
                    y = y[1:]
                    
                    if len(X) < 10:
                        print(f"   [SKIP] Too few samples ({len(X)})")
                        results[symbol] = {"status": "skipped", "reason": "too few samples"}
                        continue
                    
                    print(f"   Training {model_type} with {len(X)} samples...")
                    
                    # Train with specific configuration
                    trainer = ModelTrainer(mlflow_tracker)
                    
                    # Start MLflow run
                    with mlflow.start_run(run_name=f"{symbol}_{config_name}"):
                        mlflow.log_param("symbol", symbol)
                        mlflow.log_param("config", config_name)
                        mlflow.log_param("model_type", model_type)
                        mlflow.log_params(params)
                        mlflow.log_param("samples", len(X))
                        mlflow.log_param("features", len(feature_names))
                        
                        # Train model with fixed params (no Optuna optimization)
                        model, metrics, model_version = trainer.train_with_params(
                            X=X,
                            y=y,
                            feature_names=feature_names,
                            model_type=model_type,
                            params=params,
                            symbol=symbol,
                            config_name=config_name
                        )
                        
                        # Log metrics
                        mlflow.log_metrics({
                            "accuracy": metrics.get("accuracy", 0),
                            "precision": metrics.get("precision", 0),
                            "recall": metrics.get("recall", 0),
                            "f1_score": metrics.get("f1_score", 0),
                            "auc_roc": metrics.get("auc_roc", 0)
                        })
                    
                    print(f"   [OK] Accuracy: {metrics.get('accuracy', 0):.2%}, F1: {metrics.get('f1_score', 0):.2%}")
                    results[symbol] = {
                        "status": "success",
                        "model_version": model_version,
                        "accuracy": metrics.get("accuracy", 0),
                        "f1_score": metrics.get("f1_score", 0)
                    }
                    
                except Exception as e:
                    print(f"   [ERROR] {e}")
                    results[symbol] = {"status": "error", "reason": str(e)}
            
            # Summary
            print(f"\n{'='*60}")
            print("TRAINING SUMMARY")
            print(f"{'='*60}\n")
            
            success = sum(1 for r in results.values() if r["status"] == "success")
            skipped = sum(1 for r in results.values() if r["status"] == "skipped")
            errors = sum(1 for r in results.values() if r["status"] == "error")
            
            print(f"Configuration: {config_name}")
            print(f"Total Symbols: {len(symbols)}")
            print(f"Successful: {success}")
            print(f"Skipped: {skipped}")
            print(f"Errors: {errors}\n")
            
            print(f"{'Symbol':<12} {'Status':<10} {'Accuracy':<10} {'F1 Score':<10}")
            print("-" * 50)
            
            for symbol, result in results.items():
                status = result["status"]
                if status == "success":
                    acc = f"{result['accuracy']:.2%}"
                    f1 = f"{result['f1_score']:.2%}"
                else:
                    acc = "-"
                    f1 = result.get("reason", "-")[:20]
                print(f"{symbol:<12} {status:<10} {acc:<10} {f1:<10}")
            
            print(f"\n[OK] Results logged to MLflow experiment 'best_config_training'")
            print(f"     View at: http://127.0.0.1:5000")
            
        except Exception as e:
            print(f"\n[ERROR] Training failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _ml_retrain_feedback(self, force=False):
        """Retrain model using actual trade feedback (P&L outcomes)."""
        if not ML_CONFIG.get("enabled", False):
            print("ML is disabled. Enable it in config/settings.py")
            return
        
        print(f"\n{'='*60}")
        print("FEEDBACK-BASED RETRAINING")
        print(f"{'='*60}")
        print("\nThis trains the model using actual trade outcomes (P&L)")
        print("instead of historical price direction.\n")
        
        try:
            from ml import get_auto_retrainer
            
            retrainer = get_auto_retrainer()
            
            # Check conditions first
            if not force:
                print("[1/3] Checking retrain conditions...")
                conditions = retrainer.check_retrain_conditions()
                
                print(f"\n   Available Samples: {conditions['available_samples']}")
                print(f"   Minimum Required: {retrainer.min_samples}")
                
                if conditions.get('days_since_last_train'):
                    print(f"   Days Since Last Train: {conditions['days_since_last_train']}")
                
                if conditions.get('current_accuracy'):
                    print(f"   Current Accuracy: {conditions['current_accuracy']:.1%}")
                
                if conditions.get('drift_detected'):
                    print(f"   Drift Detected: YES")
                
                print(f"\n   Reasons for retrain:")
                for reason in conditions.get('reasons', []):
                    print(f"      - {reason}")
                
                if not conditions['should_retrain']:
                    print(f"\n[INFO] Retrain conditions not met.")
                    print("       Use 'ml retrain --force' to force retrain.")
                    return
            else:
                print("[1/3] Force mode - skipping condition check...")
            
            # Get feedback data
            print("\n[2/3] Loading feedback training data...")
            X, y, feature_names = retrainer.get_feedback_training_data()
            
            if X is None or len(X) == 0:
                print("\n[ERROR] No feedback training data available.")
                print("        Execute some trades first to collect feedback.")
                return
            
            print(f"   Samples: {len(X)}")
            print(f"   Features: {len(feature_names)}")
            
            # Distribution
            import numpy as np
            unique, counts = np.unique(y, return_counts=True)
            dist = dict(zip(unique, counts))
            print(f"   Target Distribution:")
            print(f"      WIN (2): {dist.get(2, 0)}")
            print(f"      BREAKEVEN (1): {dist.get(1, 0)}")
            print(f"      LOSS (0): {dist.get(0, 0)}")
            
            # Confirm
            if not force:
                confirm = input("\nProceed with retraining? (y/n): ").strip().lower()
                if confirm != 'y':
                    print("Cancelled.")
                    return
            
            # Retrain
            print("\n[3/3] Training model from feedback data...")
            result = retrainer.retrain_from_feedback(force=True)
            
            if result['success']:
                print(f"\n[OK] Retraining successful!")
                print(f"     Model Version: {result['model_version']}")
                print(f"\n   Metrics:")
                for key, value in result.get('metrics', {}).items():
                    if isinstance(value, float):
                        print(f"      {key}: {value:.4f}")
                    else:
                        print(f"      {key}: {value}")
                
                if result.get('promoted'):
                    print(f"\n   [OK] Model auto-promoted to production")
            else:
                print(f"\n[ERROR] Retraining failed: {result['message']}")
                
        except Exception as e:
            print(f"\n[ERROR] Retrain failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _ml_retrain_status(self):
        """Show auto-retrain status and conditions."""
        if not ML_CONFIG.get("enabled", False):
            print("ML is disabled. Enable it in config/settings.py")
            return
        
        print(f"\n{'='*60}")
        print("AUTO-RETRAIN STATUS")
        print(f"{'='*60}")
        
        try:
            from ml import get_auto_retrainer, get_feedback_collector
            
            retrainer = get_auto_retrainer()
            feedback = get_feedback_collector()
            
            status = retrainer.get_status()
            
            print(f"\n[CONFIGURATION]")
            print(f"   Enabled: {'YES' if status['enabled'] else 'NO'}")
            print(f"   Background Monitor: {'RUNNING' if status['running'] else 'STOPPED'}")
            print(f"   Min Samples: {status['min_samples']}")
            print(f"   Retrain Interval: {status['retrain_interval_days']} days")
            print(f"   Drift Threshold: {status['drift_threshold']:.1%}")
            print(f"   Auto-Promote: {'YES' if status['auto_promote'] else 'NO'}")
            
            if status.get('last_retrain'):
                print(f"\n   Last Retrain: {status['last_retrain']}")
            
            conditions = status.get('conditions', {})
            print(f"\n[CURRENT CONDITIONS]")
            print(f"   Available Samples: {conditions.get('available_samples', 0)}")
            print(f"   Should Retrain: {'YES' if conditions.get('should_retrain') else 'NO'}")
            print(f"   Drift Detected: {'YES' if conditions.get('drift_detected') else 'NO'}")
            
            if conditions.get('current_accuracy'):
                print(f"   Current Accuracy: {conditions['current_accuracy']:.1%}")
            
            if conditions.get('days_since_last_train'):
                print(f"   Days Since Training: {conditions['days_since_last_train']}")
            
            print(f"\n[REASONS]")
            for reason in conditions.get('reasons', []):
                print(f"   - {reason}")
            
            # Feedback data summary
            print(f"\n[FEEDBACK DATA]")
            summary = feedback.get_training_data_summary()
            print(f"   Total Records: {summary.get('total_samples', 0)}")
            print(f"   With Outcomes: {summary.get('samples_with_outcome', 0)}")
            print(f"   Win Rate: {summary.get('win_rate', 0):.1%}")
            
            if summary.get('underlyings'):
                print(f"   Symbols: {', '.join(summary['underlyings'])}")
            
            print(f"\n{'='*60}")
            print("COMMANDS:")
            print("   ml retrain         - Manually trigger retrain")
            print("   ml retrain --force - Force retrain even if conditions not met")
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"\n[ERROR] Could not get status: {e}")
            import traceback
            traceback.print_exc()
    
    def _ml_collect(self, action: str):
        """Manage live feature collection for ML training."""
        try:
            from ml.live_feature_collector import get_collector, label_snapshots
            
            collector = get_collector()
            
            if action == "start":
                print("\n" + "="*60)
                print("STARTING LIVE FEATURE COLLECTION")
                print("="*60)
                print(f"\nSymbols: {', '.join(collector.symbols)}")
                print(f"Interval: {collector.interval} seconds")
                print("\nThis will collect all 61 features including:")
                print("  - Price & Technical (34 features)")
                print("  - Options Chain & Greeks (12 features)")
                print("  - OI Sentiment (6 features)")
                print("  - Time/Calendar (7 features)")
                print("  - Volatility (2 additional features)")
                print("\nCollection runs in background during market hours.")
                print("Use 'ml collect status' to check progress.")
                print("Use 'ml collect stop' to stop collection.")
                
                if collector.start():
                    print("\n[SUCCESS] Collection started!")
                else:
                    print("\n[WARNING] Collector already running")
                    
            elif action == "stop":
                collector.stop()
                print("\n[SUCCESS] Collection stopped")
                
            elif action == "status":
                stats = collector.get_stats()
                
                print("\n" + "="*60)
                print("LIVE FEATURE COLLECTION STATUS")
                print("="*60)
                
                print(f"\n[COLLECTOR]")
                print(f"   Running: {'YES' if stats.get('running') else 'NO'}")
                print(f"   Started: {stats.get('started_at', 'N/A')}")
                print(f"   Interval: {stats.get('interval_seconds', 900)}s")
                print(f"   Symbols: {', '.join(stats.get('symbols', []))}")
                
                print(f"\n[SESSION STATS]")
                print(f"   Snapshots Collected: {stats.get('snapshots_collected', 0)}")
                print(f"   Last Snapshot: {stats.get('last_snapshot_time', 'N/A')}")
                print(f"   Errors: {stats.get('errors', 0)}")
                
                print(f"\n[DATABASE]")
                print(f"   Total Snapshots: {stats.get('total_in_db', 0)}")
                print(f"   Full Feature Snapshots: {stats.get('full_feature_snapshots', 0)}")
                
                if stats.get('db_by_symbol'):
                    print(f"\n   By Symbol:")
                    for sym, count in sorted(stats['db_by_symbol'].items()):
                        print(f"      {sym}: {count}")
                
                print(f"\n{'='*60}")
                print("COMMANDS:")
                print("   ml collect start  - Start background collection")
                print("   ml collect stop   - Stop collection")
                print("   ml collect once   - Collect once for all symbols")
                print("   ml label          - Label snapshots with outcomes")
                print(f"{'='*60}")
                
            elif action == "once":
                print("\nCollecting features once for all symbols...")
                results = collector.collect_once()
                
                print(f"\n[RESULTS]")
                print(f"   Collected: {results['collected']} snapshots")
                print(f"   Errors: {results['errors']}")
                
                if results['symbols']:
                    print(f"\n   Details:")
                    for sym, info in results['symbols'].items():
                        opts = "✓" if info['has_options'] else "✗"
                        oi = "✓" if info['has_oi'] else "✗"
                        print(f"      {sym}: {info['features']} features (Options: {opts}, OI: {oi})")
                        
            else:
                print(f"Unknown collect action: {action}")
                print("Usage: ml collect [start|stop|status|once]")
                
        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()
    
    def _ml_label(self):
        """Label collected snapshots with actual outcomes."""
        try:
            from ml.live_feature_collector import label_snapshots, get_training_data_from_snapshots
            
            print("\n" + "="*60)
            print("LABELING FEATURE SNAPSHOTS")
            print("="*60)
            
            print("\nLabeling snapshots with actual price movements...")
            count = label_snapshots(lookback_hours=4)
            
            print(f"\n[RESULTS]")
            print(f"   Labeled: {count} snapshots")
            
            # Check training data availability
            X, y, features = get_training_data_from_snapshots(min_samples=10, require_full_features=False)
            
            if X is not None:
                print(f"\n[TRAINING DATA AVAILABLE]")
                print(f"   Samples: {len(X)}")
                print(f"   Features: {len(features)}")
                print(f"   Positive (UP): {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
                print(f"   Negative (DOWN/NEUTRAL): {len(y)-sum(y)} ({(len(y)-sum(y))/len(y)*100:.1f}%)")
                
                if len(X) >= 100:
                    print(f"\n   [READY] Sufficient data for training with full features!")
                    print(f"   Run 'ml train-best' to retrain with collected data")
                else:
                    print(f"\n   [NEED MORE] Collect at least {100 - len(X)} more samples")
            else:
                print(f"\n   No labeled data available yet")
            
            print(f"\n{'='*60}")
            
        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()
    
    def _ml_historical(self, action: str, args: list):
        """Manage historical data collection from NSE bhavcopy."""
        try:
            from ml.historical_data_collector import get_historical_collector
            from datetime import date, timedelta
            
            collector = get_historical_collector()
            
            if action == "status":
                status = collector.get_collection_status()
                
                print("\n" + "="*60)
                print("HISTORICAL DATA COLLECTION STATUS")
                print("="*60)
                
                print(f"\n[DATABASE]")
                print(f"   Historical (NSE):  {status.get('historical_snapshots', 0)}")
                print(f"   Historical (Kite): {status.get('kite_historical_snapshots', 0)}")
                print(f"   Live Snapshots:    {status.get('live_snapshots', 0)}")
                print(f"   Total Snapshots:   {status.get('total_snapshots', 0)}")
                
                date_range = status.get('date_range', {})
                if date_range.get('start'):
                    print(f"\n[DATE RANGE]")
                    print(f"   Start: {date_range.get('start')}")
                    print(f"   End: {date_range.get('end')}")
                
                symbols = status.get('symbols', [])
                if symbols:
                    print(f"\n[SYMBOLS] ({len(symbols)})")
                    print(f"   {', '.join(symbols)}")
                
                print(f"\n[CACHE]")
                print(f"   Files: {status.get('cache_files', 0)}")
                print(f"   Size: {status.get('cache_size_mb', 0):.1f} MB")
                
                print(f"\n{'='*60}")
                print("COMMANDS:")
                print("   ml historical status            - Show collection status")
                print("   ml historical date YYYY-MM-DD   - Collect for specific date")
                print("   ml historical range START END   - Collect date range")
                print("   ml historical fill              - Fill missing dates (30 days)")
                print("   ml historical load FILE_PATH    - Load manually downloaded bhavcopy")
                print("   ml historical kite [DAYS]       - Collect via Kite API (default 30)")
                print("   ml backfill [DAYS]              - Backfill last N days")
                print(f"\n   NOTE: If NSE blocks downloads, use either:")
                print(f"         1. 'ml historical kite 30' - Uses Kite API (OHLCV only)")
                print(f"         2. Download manually from NSE and use 'ml historical load'")
                print(f"            https://www.nseindia.com/all-reports-derivatives")
                print(f"{'='*60}")
                
            elif action == "date":
                if not args:
                    print("Usage: ml historical date YYYY-MM-DD")
                    return
                
                target_date = date.fromisoformat(args[0])
                print(f"\nCollecting historical data for {target_date}...")
                
                result = collector.process_date(target_date)
                
                print(f"\n[RESULTS]")
                print(f"   Date: {result['date']}")
                print(f"   Collected: {result['collected']} snapshots")
                print(f"   Errors: {result['errors']}")
                
                if result.get('symbols'):
                    print(f"\n   Symbols:")
                    for sym, info in result['symbols'].items():
                        pcr = info.get('pcr', 0)
                        print(f"      {sym}: Spot={info.get('spot'):.2f}, Features={info.get('features')}, PCR={pcr:.2f}")
                
            elif action == "range":
                if len(args) < 2:
                    print("Usage: ml historical range START_DATE END_DATE")
                    print("Example: ml historical range 2025-12-01 2025-12-20")
                    return
                
                start_date = date.fromisoformat(args[0])
                end_date = date.fromisoformat(args[1])
                
                print(f"\nCollecting historical data from {start_date} to {end_date}...")
                print("This may take a while. Please wait...\n")
                
                results = collector.collect_date_range(start_date, end_date)
                
                print(f"\n[RESULTS]")
                print(f"   Date Range: {results['start_date']} to {results['end_date']}")
                print(f"   Total Dates: {results['total_dates']}")
                print(f"   Processed: {results['processed_dates']}")
                print(f"   Skipped (already exists): {results['skipped_dates']}")
                print(f"   Total Snapshots: {results['total_collected']}")
                print(f"   Errors: {results['total_errors']}")
                
            elif action == "fill":
                print("\nFinding and filling missing dates in last 30 days...")
                results = collector.fill_missing_dates(lookback_days=30)
                
                print(f"\n[RESULTS]")
                print(f"   Missing Dates Found: {results.get('missing_dates', 0)}")
                print(f"   Dates Filled: {results.get('filled', 0)}")
                print(f"   Errors: {results.get('errors', 0)}")
                
                if results.get('dates'):
                    print(f"\n   Filled Dates:")
                    for d in results['dates'][:10]:
                        print(f"      {d['date']}: {d['collected']} snapshots")
                    if len(results['dates']) > 10:
                        print(f"      ... and {len(results['dates']) - 10} more")
            
            elif action == "load":
                if not args:
                    print("Usage: ml historical load FILE_PATH [FILE_PATH2 ...]")
                    print("\nLoad manually downloaded bhavcopy files from NSE.")
                    print("Download from: https://www.nseindia.com/all-reports-derivatives")
                    print("\nExample:")
                    print("   ml historical load C:\\Downloads\\fo27DEC2024bhav.csv.zip")
                    return
                
                # Load each file
                from pathlib import Path
                loaded = 0
                for file_path in args:
                    file_path = file_path.strip('"').strip("'")
                    print(f"\nLoading: {file_path}")
                    df = collector.load_from_file(file_path)
                    if df is not None:
                        print(f"   Records: {len(df)}")
                        loaded += 1
                    else:
                        print(f"   Failed to load file")
                
                print(f"\n[RESULT] Loaded {loaded}/{len(args)} files")
                
                # Now process the loaded files
                if loaded > 0:
                    print("\nProcessing loaded files to extract features...")
                    status = collector.get_collection_status()
                    print(f"\n   Cache Files: {status.get('cache_files', 0)}")
                    print("\n   Run 'ml historical status' to see current data")
                    print("   Run 'ml historical fill' to process cached files")
            
            elif action == "kite":
                # Collect using Kite API
                days = 30
                if args:
                    try:
                        days = int(args[0])
                    except ValueError:
                        print("Usage: ml historical kite [DAYS]")
                        return
                
                print(f"\n{'='*60}")
                print("COLLECTING HISTORICAL DATA VIA KITE API")
                print(f"{'='*60}")
                print(f"\nDays: {days}")
                print(f"Symbols: {', '.join(collector.symbols)}")
                print(f"\nNOTE: This collects OHLCV data only (no options chain).")
                print("For full features, use live collector during market hours.\n")
                
                results = collector.collect_from_kite(days=days)
                
                print(f"\n[RESULTS]")
                print(f"   Method: {results.get('method')}")
                print(f"   Symbols Collected: {results.get('symbols_collected')}")
                print(f"   Total Candles: {results.get('total_candles')}")
                print(f"   Snapshots Created: {results.get('snapshots_created')}")
                
                if results.get('errors'):
                    print(f"\n[ERRORS]")
                    for err in results['errors'][:5]:
                        print(f"   - {err}")
                    if len(results['errors']) > 5:
                        print(f"   ... and {len(results['errors']) - 5} more")
                        
            else:
                print(f"Unknown historical action: {action}")
                print("Usage: ml historical [status|date|range|fill|load|kite]")
                
        except ValueError as e:
            print(f"\n[ERROR] Invalid date format. Use YYYY-MM-DD")
        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()
    
    def _ml_backfill(self, days: int = 30):
        """Backfill historical data for the last N days."""
        try:
            from ml.historical_data_collector import get_historical_collector
            from datetime import date, timedelta
            
            collector = get_historical_collector()
            
            end_date = date.today() - timedelta(days=1)  # Yesterday
            start_date = end_date - timedelta(days=days)
            
            print(f"\n" + "="*60)
            print(f"BACKFILLING HISTORICAL DATA")
            print(f"="*60)
            print(f"\nDate Range: {start_date} to {end_date} ({days} days)")
            print(f"Symbols: {', '.join(collector.symbols)}")
            print("\nDownloading from NSE archives and calculating IV/Greeks...")
            print("This may take several minutes. Please wait...\n")
            
            results = collector.collect_date_range(start_date, end_date)
            
            print(f"\n{'='*60}")
            print(f"BACKFILL COMPLETE")
            print(f"{'='*60}")
            print(f"\n[SUMMARY]")
            print(f"   Total Trading Days: {results['total_dates']}")
            print(f"   Days Processed: {results['processed_dates']}")
            print(f"   Days Skipped (already exists): {results['skipped_dates']}")
            print(f"   Total Snapshots Created: {results['total_collected']}")
            print(f"   Errors: {results['total_errors']}")
            
            # Show current status
            status = collector.get_collection_status()
            print(f"\n[DATABASE NOW]")
            print(f"   Total Historical: {status.get('historical_snapshots', 0)}")
            print(f"   Total Live: {status.get('live_snapshots', 0)}")
            print(f"   Grand Total: {status.get('total_snapshots', 0)}")
            
            if results['total_collected'] >= 50:
                print(f"\n[NEXT STEP] Run 'ml label' to label data, then 'ml train-best' to train model")
            
        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()
    
    def do_bhavcopy(self, arg):
        """
        Download NSE bhavcopy data using jugaad-data library.
        
        Usage:
            bhavcopy download DAYS    - Download last N days of bhavcopy data
            bhavcopy status           - Show downloaded data status
            bhavcopy date YYYY-MM-DD  - Download for a specific date
            bhavcopy range START END  - Download for date range
            
        Examples:
            bhavcopy download 30      - Download last 30 days
            bhavcopy date 2025-12-01  - Download for Dec 1, 2025
            bhavcopy range 2025-11-01 2025-11-30
        """
        args = arg.strip().split()
        
        if not args:
            print("Usage: bhavcopy <download|status|date|range> [args]")
            return
        
        cmd = args[0].lower()
        
        try:
            from data.nse_bhavcopy import get_bhavcopy_collector, JUGAAD_AVAILABLE
            from datetime import date, timedelta
            import pandas as pd
            
            if not JUGAAD_AVAILABLE:
                print("\n[ERROR] jugaad-data not installed!")
                print("Run: pip install jugaad-data")
                return
            
            collector = get_bhavcopy_collector()
            
            if cmd == "status":
                print("\n" + "="*60)
                print("BHAVCOPY DATA STATUS")
                print("="*60)
                
                # Count downloaded files
                equity_count = len(list(collector.equity_dir.glob("*.csv")))
                fo_count = len(list(collector.fo_dir.glob("*.csv")))
                
                print(f"\n[DOWNLOADED FILES]")
                print(f"   Equity bhavcopy: {equity_count} files")
                print(f"   F&O bhavcopy:    {fo_count} files")
                print(f"   Directory: {collector.download_dir}")
                
                if collector.downloaded_dates:
                    dates_list = sorted(collector.downloaded_dates)
                    print(f"\n[DATE RANGE]")
                    print(f"   Earliest: {dates_list[0]}")
                    print(f"   Latest:   {dates_list[-1]}")
                    print(f"   Total dates: {len(dates_list)}")
                
            elif cmd == "download":
                days = int(args[1]) if len(args) > 1 else 30
                
                end_date = date.today() - timedelta(days=1)
                start_date = end_date - timedelta(days=days)
                
                print(f"\n{'='*60}")
                print(f"DOWNLOADING BHAVCOPY DATA")
                print(f"{'='*60}")
                print(f"\nDate range: {start_date} to {end_date}")
                print(f"This will download F&O bhavcopy files...")
                print(f"Holidays will be skipped automatically.\n")
                
                results = collector.download_historical(start_date, end_date)
                
                print(f"\n{'='*60}")
                print(f"DOWNLOAD COMPLETE")
                print(f"{'='*60}")
                print(f"   Equity downloaded: {results.get('equity_success', 0)}")
                print(f"   F&O downloaded:    {results.get('fo_success', 0)}")
                print(f"   Skipped (cached):  {results.get('skipped', 0)}")
                print(f"   Failed:            {results.get('failed', 0)}")
                
            elif cmd == "date":
                if len(args) < 2:
                    print("Usage: bhavcopy date YYYY-MM-DD")
                    return
                
                target_date = date.fromisoformat(args[1])
                print(f"\nDownloading bhavcopy for {target_date}...")
                
                results = collector.download_historical(target_date, target_date)
                
                if results.get('fo_success', 0) > 0:
                    print(f"✅ Successfully downloaded F&O bhavcopy for {target_date}")
                else:
                    print(f"❌ Could not download for {target_date} (holiday or NSE blocking)")
                    
            elif cmd == "range":
                if len(args) < 3:
                    print("Usage: bhavcopy range START_DATE END_DATE")
                    print("Example: bhavcopy range 2025-11-01 2025-11-30")
                    return
                
                start_date = date.fromisoformat(args[1])
                end_date = date.fromisoformat(args[2])
                
                print(f"\n{'='*60}")
                print(f"DOWNLOADING BHAVCOPY DATA")
                print(f"{'='*60}")
                print(f"\nDate range: {start_date} to {end_date}")
                print(f"This may take a while...\n")
                
                results = collector.download_historical(start_date, end_date)
                
                print(f"\n{'='*60}")
                print(f"DOWNLOAD COMPLETE")
                print(f"{'='*60}")
                print(f"   Equity downloaded: {results.get('equity_success', 0)}")
                print(f"   F&O downloaded:    {results.get('fo_success', 0)}")
                print(f"   Skipped (cached):  {results.get('skipped', 0)}")
                print(f"   Failed:            {results.get('failed', 0)}")
                
            else:
                print(f"Unknown command: {cmd}")
                print("Usage: bhavcopy <download|status|date|range> [args]")
                
        except ImportError as e:
            print(f"\n[ERROR] Could not import bhavcopy module: {e}")
            print("Run: pip install jugaad-data")
        except ValueError as e:
            print(f"\n[ERROR] Invalid date format: {e}")
            print("Use YYYY-MM-DD format (e.g., 2025-12-01)")
        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()
    
    def _ml_train_full(self, args: list):
        """Train all symbols using NSE bhavcopy historical data with 59 features."""
        try:
            from ml.full_pipeline import FullPipelineTrainer
            from datetime import date
            
            print("\n" + "="*60)
            print("FULL TRAINING PIPELINE (NSE Bhavcopy)")
            print("="*60)
            
            # Parse optional date arguments
            if len(args) >= 2:
                start_date = date.fromisoformat(args[0])
                end_date = date.fromisoformat(args[1])
            else:
                # Default: Jan-May 2024 (known available in archives)
                start_date = date(2024, 1, 1)
                end_date = date(2024, 5, 31)
            
            print(f"\nDate Range: {start_date} to {end_date}")
            print("This downloads NSE bhavcopy data and trains per-symbol models.")
            print("\nFeatures include:")
            print("   - OHLCV, Futures OI")
            print("   - Options OI (calls/puts), PCR")
            print("   - ATM/OTM analysis, Max Pain")
            print("   - Technical indicators (RSI, MACD, BB)")
            print("   - Greek proxies (IV, Delta, Gamma, Theta, Vega)")
            print("\nStarting training pipeline...\n")
            
            trainer = FullPipelineTrainer()
            results = trainer.run_full_pipeline(
                start_date=start_date,
                end_date=end_date,
                force_download="--force" in args or "-f" in args
            )
            
            if results:
                print("\n" + "="*60)
                print("TRAINING COMPLETE!")
                print("="*60)
                print(f"Models trained: {len(results)}")
                
                import numpy as np
                avg_acc = np.mean([r["metrics"]["accuracy"] for r in results.values()])
                avg_f1 = np.mean([r["metrics"]["f1"] for r in results.values()])
                
                print(f"Average Accuracy: {avg_acc:.1%}")
                print(f"Average F1 Score: {avg_f1:.1%}")
                
                print("\nModels saved to: data/ml_models/")
                print("Use 'ml models' to see trained model details.")
            else:
                print("\n[ERROR] Training failed. Check logs for details.")
                
        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()
    
    def _ml_train_monthly(self, force: bool = False):
        """Run monthly update pipeline - download new data and retrain."""
        try:
            from ml.full_pipeline import MonthlyUpdatePipeline
            
            print("\n" + "="*60)
            print("MONTHLY UPDATE PIPELINE")
            print("="*60)
            
            pipeline = MonthlyUpdatePipeline()
            needs_update, start, end = pipeline.check_update_needed()
            
            if not needs_update and not force:
                print("\nNo update needed. Models are up to date.")
                print(f"Last update date: {pipeline.get_last_update_date()}")
                print("\nUse '--force' or '-f' to force retrain.")
                return
            
            if force:
                print("\nForcing full retrain...")
            else:
                print(f"\nUpdating with data from {start} to {end}...")
            
            results = pipeline.run_monthly_update(force=force)
            
            if results:
                print("\n[SUCCESS] Monthly update complete!")
                print(f"Models updated: {len(results)}")
            else:
                print("\n[INFO] No new data available or update not needed.")
                
        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()
    
    def _ml_models(self):
        """Show trained per-symbol models and their metrics."""
        try:
            from ml.historical_predictor import HistoricalModelPredictor
            from pathlib import Path
            import json
            
            print("\n" + "="*60)
            print("TRAINED SYMBOL MODELS")
            print("="*60)
            
            predictor = HistoricalModelPredictor()
            symbols = predictor.get_available_symbols()
            
            if not symbols:
                print("\nNo trained models found.")
                print("Run 'ml train-full' to train models from NSE bhavcopy data.")
                return
            
            print(f"\nLoaded {len(symbols)} symbol models:")
            print("-"*50)
            print(f"{'Symbol':<12} | {'Accuracy':<10} | {'F1 Score':<10} | {'Samples':<8}")
            print("-"*50)
            
            for symbol in symbols:
                metrics = predictor.get_model_metrics(symbol)
                if metrics:
                    model_data = predictor.models.get(symbol, {})
                    n_samples = model_data.get("n_samples", "?")
                    print(f"{symbol:<12} | {metrics.get('accuracy', 0):>8.1%} | {metrics.get('f1', 0):>8.1%} | {n_samples:>8}")
            
            print("-"*50)
            
            # Show feature count
            if predictor.feature_names:
                print(f"\nFeatures: {len(predictor.feature_names)}")
                print("Top features include: iv_proxy, pcr_oi, delta_proxy, rsi, macd")
            
            # Show latest training summary
            model_dir = Path("data/ml_models")
            summaries = list(model_dir.glob("training_summary_*.json"))
            if summaries:
                latest = max(summaries, key=lambda p: p.stat().st_mtime)
                with open(latest) as f:
                    summary = json.load(f)
                
                date_range = summary.get("date_range", [])
                print(f"\nTraining Data: {date_range[0]} to {date_range[1]}" if len(date_range) == 2 else "")
                print(f"Training Time: {summary.get('timestamp', 'unknown')}")
            
            print("\nCommands:")
            print("   ml train-full [START END]  - Train all symbols")
            print("   ml train-monthly           - Monthly update pipeline")
                
        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()
    
    def do_quit(self, arg):
        """Exit the CLI"""
        print("Goodbye!")
        return True
    
    def do_exit(self, arg):
        """Exit the CLI"""
        return self.do_quit(arg)
    
    def default(self, line):
        print(f"Unknown command: {line}")
        print("Type 'help' for available commands")


def main():
    """Run the interactive CLI."""
    try:
        cli = TradingCLI()
        cli.cmdloop()
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
