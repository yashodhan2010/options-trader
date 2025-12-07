"""
Demo script to show what a paper trading signal looks like
This creates a SIMULATED example signal to demonstrate the format
"""
from datetime import datetime
from strategies.base_strategy import StrategySignal, StrategyType, OptionLeg, TradeDirection

print("=" * 70)
print("ENHANCED PAPER SIGNAL EXAMPLE (WITH HISTORICAL ANALYSIS)")
print("=" * 70)
print("\nThis is what a real signal would look like during market hours:\n")

# Create a simulated Bull Call Spread signal for RELIANCE with historical data
signal = StrategySignal(
    strategy_type=StrategyType.BULL_CALL_SPREAD,
    underlying="RELIANCE",
    legs=[
        OptionLeg(
            symbol="RELIANCE24DEC1280CE",
            strike=1280.0,
            option_type="CE",
            expiry=datetime(2024, 12, 26),
            direction=TradeDirection.BUY,
            quantity=250,  # 1 lot
            entry_price=45.50,
            instrument_token=12345678,
        ),
        OptionLeg(
            symbol="RELIANCE24DEC1300CE",
            strike=1300.0,
            option_type="CE",
            expiry=datetime(2024, 12, 26),
            direction=TradeDirection.SELL,
            quantity=250,
            entry_price=32.25,
            instrument_token=12345679,
        ),
    ],
    entry_time=datetime.now(),
    confidence=0.82,  # Higher confidence with historical alignment
    expected_profit=3687.50,
    max_loss=3312.50,
    stop_loss=1656.25,
    target=2950.00,
    rationale="OI: BULLISH | PCR: 1.35 | IV: NORMAL | Trend: UPTREND | RSI: 58 | Momentum: BULLISH | 5D Return: 2.3%",
    metrics={
        # Real-time metrics
        "spot_price": 1275.40,
        "pcr": 1.35,
        "iv_percentile": 45,
        "sentiment": "BULLISH",
        # Historical metrics (NEW)
        "trend": "UPTREND",
        "trend_score": 0.7,
        "rsi": 58.2,
        "rsi_signal": "NEUTRAL",
        "momentum": "BULLISH",
        "momentum_score": 0.5,
        "returns_5d": 2.3,
        "returns_10d": 4.1,
        "sma_5": 1268.50,
        "sma_10": 1255.20,
        "sma_20": 1240.80,
        "support_20d": 1220.00,
        "resistance_20d": 1295.00,
        "atr_14": 18.50,
        "volume_signal": "NORMAL",
        "historical_sentiment": "BULLISH",
        "confidence_boost": 0.10,
    }
)

print(f"{'='*70}")
print(f"STRATEGY: {signal.strategy_type.value.upper().replace('_', ' ')}")
print(f"{'='*70}")
print(f"Underlying:      {signal.underlying}")
print(f"Spot Price:      Rs.{signal.metrics['spot_price']:,.2f}")
print(f"Confidence:      {signal.confidence:.0%}  <-- Enhanced with historical!")

print(f"\n{'-'*70}")
print("REAL-TIME ANALYSIS:")
print(f"{'-'*70}")
print(f"  OI Sentiment:    {signal.metrics['sentiment']}")
print(f"  PCR:             {signal.metrics['pcr']}")
print(f"  IV Percentile:   {signal.metrics['iv_percentile']}%")

print(f"\n{'-'*70}")
print("HISTORICAL ANALYSIS (NEW):")
print(f"{'-'*70}")
print(f"  Trend:           {signal.metrics['trend']} (Score: {signal.metrics['trend_score']})")
print(f"  RSI (14):        {signal.metrics['rsi']:.1f} ({signal.metrics['rsi_signal']})")
print(f"  Momentum:        {signal.metrics['momentum']} (Score: {signal.metrics['momentum_score']})")
print(f"  5-Day Return:    {signal.metrics['returns_5d']:+.1f}%")
print(f"  10-Day Return:   {signal.metrics['returns_10d']:+.1f}%")
print(f"  SMA 5/10/20:     {signal.metrics['sma_5']:.2f} / {signal.metrics['sma_10']:.2f} / {signal.metrics['sma_20']:.2f}")
print(f"  Support (20d):   Rs.{signal.metrics['support_20d']:,.2f}")
print(f"  Resistance (20d):Rs.{signal.metrics['resistance_20d']:,.2f}")
print(f"  ATR (14):        Rs.{signal.metrics['atr_14']:.2f}")
print(f"  Volume:          {signal.metrics['volume_signal']}")
print(f"  Historical Sent: {signal.metrics['historical_sentiment']}")
print(f"  Confidence Boost:{signal.metrics['confidence_boost']:+.0%}")

print(f"\n{'-'*70}")
print("TRADE LEGS:")
print(f"{'-'*70}")
for i, leg in enumerate(signal.legs, 1):
    action = "BUY " if leg.direction == TradeDirection.BUY else "SELL"
    cost = leg.entry_price * leg.quantity
    print(f"  Leg {i}: {action} {leg.option_type} Strike {leg.strike:.0f}")
    print(f"         Symbol: {leg.symbol}")
    print(f"         Price:  Rs.{leg.entry_price:.2f} x {leg.quantity} = Rs.{cost:,.2f}")
    print()

net_debit = (signal.legs[0].entry_price - signal.legs[1].entry_price) * signal.legs[0].quantity
max_profit_potential = (signal.legs[1].strike - signal.legs[0].strike - (signal.legs[0].entry_price - signal.legs[1].entry_price)) * signal.legs[0].quantity

print(f"{'-'*70}")
print("P&L SUMMARY:")
print(f"{'-'*70}")
print(f"  Net Debit (Cost):     Rs.{net_debit:,.2f}")
print(f"  Max Profit Potential: Rs.{max_profit_potential:,.2f}")
print(f"  Max Loss:             Rs.{signal.max_loss:,.2f}")
print(f"  Stop Loss Level:      Rs.{signal.stop_loss:,.2f}")
print(f"  Target Level:         Rs.{signal.target:,.2f}")
print(f"  Risk/Reward Ratio:    1:{signal.risk_reward_ratio:.2f}")

print(f"\n{'-'*70}")
print("TRADE RATIONALE:")
print(f"{'-'*70}")
print(f"  {signal.rationale}")

print(f"\n{'='*70}")
print("CONFIDENCE CALCULATION BREAKDOWN:")
print(f"{'='*70}")
print("""
  BASE CONFIDENCE:                 50%
  
  REAL-TIME FACTORS:
  + OI Sentiment (BULLISH):       +10%
  + PCR > 1.2:                     +5%
  + IV Regime (NORMAL):            +5%
                                  -----
  Subtotal:                        70%
  
  HISTORICAL FACTORS (NEW):
  + Trend (UPTREND):              +10%
  + Momentum (BULLISH):            +8%
  + RSI (58 - neutral):            +0%
  + Volume (NORMAL):               +0%
                                  -----
  FINAL CONFIDENCE:                82%
""")

print("=" * 70)
print("\nKEY INSIGHT: Historical analysis added +12% confidence because:")
print("  - Price is in UPTREND (above all SMAs)")
print("  - 5-day momentum is BULLISH (+2.3%)")
print("  - RSI at 58 is neutral (not overbought)")
print("  - This aligns with bullish OI sentiment")
print("=" * 70)
