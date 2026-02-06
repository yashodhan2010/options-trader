#!/usr/bin/env python
"""
Options Trader - Main Entry Point

Usage:
    python run.py          # Start interactive CLI
    python run.py --bot    # Start automated bot
    python run.py --help   # Show help
"""
import sys
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
        help="Run in paper trading mode"
    )
    
    args = parser.parse_args()
    
    if args.bot:
        # Start automated bot
        from bot import OptionsTradingBot
        bot = OptionsTradingBot(paper_trading=args.paper)
        bot.run()
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
