#!/usr/bin/env python
"""
Options Trader - Main Entry Point

Usage:
    python run.py          # Start interactive CLI
    python run.py --bot    # Start automated bot
    python run.py --help   # Show help
"""
import sys
import signal
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Options Trading Bot with ML-powered signals"
    )
    parser.add_argument(
        "--bot", "-b",
        action="store_true",
        help="Start the automated trading bot"
    )
    parser.add_argument(
        "--paper", "-p",
        action="store_true",
        help="Run in paper trading mode (implies --bot and --auto-trade)"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run in LIVE trading mode with real money (implies --bot)"
    )
    parser.add_argument(
        "--auto-trade", "-a",
        action="store_true",
        help="Enable automatic trade execution"
    )
    
    args = parser.parse_args()
    
    # --paper implies --bot + auto_trade
    if args.paper:
        args.bot = True
        args.auto_trade = True
    
    # --live implies --bot
    if args.live:
        args.bot = True
    
    if args.bot:
        # Determine trading mode
        if args.live:
            paper_trading = False
            auto_trade = args.auto_trade
        elif args.paper:
            paper_trading = True
            auto_trade = True
        else:
            paper_trading = True  # Default to paper for safety
            auto_trade = args.auto_trade
        
        # Safety confirmation for live trading
        if not paper_trading:
            print("\n" + "=" * 60)
            print("[WARNING] LIVE TRADING MODE")
            print("=" * 60)
            print("You are about to start the bot with REAL MONEY.")
            print(f"Auto-trade: {'ENABLED' if auto_trade else 'DISABLED'}")
            print("=" * 60)
            confirm = input("Type 'YES' to confirm live trading: ").strip()
            if confirm != "YES":
                print("Cancelled. Use --paper for paper trading.")
                return
        
        from bot import OptionsTradingBot
        bot = OptionsTradingBot(paper_trading=paper_trading, auto_trade=auto_trade)

        # Handle termination signals (Task Scheduler sends these)
        def _shutdown(signum, frame):
            bot.stop()
            sys.exit(0)

        signal.signal(signal.SIGTERM, _shutdown)
        signal.signal(signal.SIGINT, _shutdown)
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, _shutdown)

        bot.start()
    else:
        # Start interactive CLI
        from cli import TradingCLI
        cli = TradingCLI()
        try:
            cli.cmdloop()
        except KeyboardInterrupt:
            print("\nGoodbye!")


if __name__ == "__main__":
    main()
