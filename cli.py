"""
Interactive CLI for the Options Trading Bot
"""
import cmd
import sys
from datetime import datetime

from bot import OptionsTradingBot
from auth.kite_auth import connect, get_kite, is_authenticated, get_profile, get_margins, logout
from data.data_fetcher import data_fetcher
from signals.signal_generator import signal_generator
from execution.order_manager import order_manager
from execution.position_tracker import position_tracker
from strategies import StrategyType
from config.settings import UNDERLYING_ASSETS
from core.logger import logger


class TradingCLI(cmd.Cmd):
    """Interactive command-line interface for the trading bot."""
    
    intro = """
╔═══════════════════════════════════════════════════════════════╗
║           OPTIONS TRADING BOT - Interactive Mode              ║
║                                                               ║
║  Type 'help' for available commands                           ║
║  Type 'quit' to exit                                          ║
╚═══════════════════════════════════════════════════════════════╝
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
            print("✓ Connected to Kite")
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
            print("✗ Not connected")
    
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
        
        print(f"\nRecommended Strategies:")
        for rec in overview.get("recommended_strategies", []):
            print(f"  - {rec}")
    
    def do_signals(self, arg):
        """Generate signals. Usage: signals [NIFTY|BANKNIFTY]"""
        underlying = arg.strip().upper() if arg.strip() else None
        
        print("\nGenerating signals...")
        signals = signal_generator.generate_signals(underlying)
        
        if not signals:
            print("No signals generated")
            return
        
        print(f"\nGenerated {len(signals)} signal(s):\n")
        
        for i, signal in enumerate(signals, 1):
            print(f"{i}. {signal.strategy_type.value} ({signal.underlying})")
            print(f"   Confidence: {signal.confidence:.2%}")
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
                print("✓ Trade executed successfully!")
            else:
                print(f"✗ Execution failed: {execution.status}")
    
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
