# Options Trading Strategies Guide

A comprehensive guide to all trading strategies implemented in the Options Trading Bot.

---

## Table of Contents

1. [Strategy Overview](#strategy-overview)
2. [Directional Strategies](#directional-strategies)
   - [Long Call](#long-call)
   - [Long Put](#long-put)
   - [Short Call (Naked)](#short-call-naked)
   - [Short Put (Naked)](#short-put-naked)
3. [Spread Strategies (Debit)](#spread-strategies-debit)
   - [Bull Call Spread](#bull-call-spread)
   - [Bear Put Spread](#bear-put-spread)
4. [Credit Spread Strategies](#credit-spread-strategies)
   - [Bear Call Spread (Credit)](#bear-call-spread-credit)
   - [Bull Put Spread (Credit)](#bull-put-spread-credit)
5. [Neutral/Volatility Strategies](#neutralvolatility-strategies)
   - [Iron Condor](#iron-condor)
   - [Long Straddle](#long-straddle)
   - [Short Straddle](#short-straddle)
   - [Long Strangle](#long-strangle)
   - [Short Strangle](#short-strangle)
6. [Strategy Selection Logic](#strategy-selection-logic)
7. [Confidence Calculation](#confidence-calculation)
8. [Risk Management](#risk-management)

---

## Strategy Overview

| Strategy | Direction | Main Trade | Hedge | Market View | IV Preference | Risk Profile |
|----------|-----------|------------|-------|-------------|---------------|--------------|
| Long Call | Bullish | BUY Call | None | Up | Low IV | Limited Risk, Unlimited Profit |
| Long Put | Bearish | BUY Put | None | Down | Low IV | Limited Risk, High Profit |
| Short Call | Bearish/Neutral | SELL Call | None | Down/Flat | High IV | Unlimited Risk, Limited Profit |
| Short Put | Bullish/Neutral | SELL Put | None | Up/Flat | High IV | High Risk, Limited Profit |
| Bull Call Spread | Moderately Bullish | BUY Call | SELL Call (higher) | Up | Any | Limited Risk, Limited Profit |
| Bear Put Spread | Moderately Bearish | BUY Put | SELL Put (lower) | Down | Any | Limited Risk, Limited Profit |
| **Bear Call Spread** | Bearish/Neutral | **SELL Call** | BUY Call (higher) | Down/Flat | High IV | Limited Risk, Limited Profit |
| **Bull Put Spread** | Bullish/Neutral | **SELL Put** | BUY Put (lower) | Up/Flat | High IV | Limited Risk, Limited Profit |
| Iron Condor | Neutral | SELL Call + SELL Put | BUY Call + BUY Put | Range-bound | High IV | Limited Risk, Limited Profit |
| Long Straddle | High Volatility | BUY Call + BUY Put | None | Big Move | Low IV | Limited Risk, Unlimited Profit |
| Short Straddle | Low Volatility | SELL Call + SELL Put | None | Flat | High IV | Unlimited Risk, Limited Profit |
| Long Strangle | High Volatility | BUY Call + BUY Put (OTM) | None | Big Move | Low IV | Limited Risk, Unlimited Profit |
| Short Strangle | Low Volatility | SELL Call + SELL Put (OTM) | None | Range-bound | High IV | Unlimited Risk, Limited Profit |

---

## Directional Strategies

### Long Call

**Description**: Buy a call option expecting the underlying to rise significantly.

**Setup**:
- BUY 1 ATM/OTM Call

**When to Use**:
- Bullish or Strongly Bullish sentiment (based on OI analysis)
- Low to Normal IV (avoid buying expensive premium)
- RSI < 30 (oversold) adds confidence
- Strong uptrend on historical analysis

**P&L Profile**:
- Max Profit: Unlimited (as underlying rises)
- Max Loss: Premium paid
- Breakeven: Strike + Premium

**Example**:
```
NIFTY @ 24,000
BUY NIFTY 24,050 CE @ ₹150
Max Loss: ₹150 × 25 (lot) = ₹3,750
Breakeven: 24,200
```

---

### Long Put

**Description**: Buy a put option expecting the underlying to fall significantly.

**Setup**:
- BUY 1 ATM/OTM Put

**When to Use**:
- Bearish or Strongly Bearish sentiment
- Low to Normal IV
- RSI > 70 (overbought) adds confidence
- Downtrend on historical analysis

**P&L Profile**:
- Max Profit: Strike - Premium (if underlying goes to 0)
- Max Loss: Premium paid
- Breakeven: Strike - Premium

**Example**:
```
NIFTY @ 24,000
BUY NIFTY 23,950 PE @ ₹140
Max Loss: ₹140 × 25 = ₹3,500
Breakeven: 23,810
```

---

### Short Call (Naked)

**Description**: Sell a call option collecting premium, expecting underlying to stay flat or fall.

**Setup**:
- SELL 1 OTM Call (typically at max call OI strike - resistance)

**When to Use**:
- Bearish or Neutral sentiment
- HIGH IV (selling expensive premium)
- Strong resistance at strike (max call OI)

**P&L Profile**:
- Max Profit: Premium received
- Max Loss: Unlimited (as underlying rises)
- Breakeven: Strike + Premium

**Risk Warning**: ⚠️ Requires higher margin, unlimited risk potential.

**Example**:
```
NIFTY @ 24,000
SELL NIFTY 24,200 CE @ ₹80 (resistance at 24,200)
Max Profit: ₹80 × 25 = ₹2,000
Max Loss: Unlimited
```

---

### Short Put (Naked)

**Description**: Sell a put option collecting premium, expecting underlying to stay flat or rise.

**Setup**:
- SELL 1 OTM Put (typically at max put OI strike - support)

**When to Use**:
- Bullish or Neutral sentiment
- HIGH IV
- Strong support at strike (max put OI)

**P&L Profile**:
- Max Profit: Premium received
- Max Loss: Strike × Lot Size (if underlying goes to 0)
- Breakeven: Strike - Premium

**Risk Warning**: ⚠️ Significant downside risk, requires margin.

**Example**:
```
NIFTY @ 24,000
SELL NIFTY 23,800 PE @ ₹70 (support at 23,800)
Max Profit: ₹70 × 25 = ₹1,750
```

---

## Spread Strategies (Debit)

### Bull Call Spread

**Description**: Buy a lower strike call, sell a higher strike call. Reduces cost but caps profit.

**Setup**:
- BUY 1 ATM/ITM Call
- SELL 1 OTM Call (hedge)

**When to Use**:
- Moderately Bullish sentiment
- Any IV environment (spread neutralizes some IV risk)
- When you want to reduce premium outlay

**P&L Profile**:
- Max Profit: (Higher Strike - Lower Strike) - Net Debit
- Max Loss: Net Debit paid
- Breakeven: Lower Strike + Net Debit

**Example**:
```
NIFTY @ 24,000
BUY NIFTY 24,000 CE @ ₹200
SELL NIFTY 24,100 CE @ ₹140
Net Debit: ₹60
Max Profit: ₹100 - ₹60 = ₹40 per share
Max Loss: ₹60 × 25 = ₹1,500
```

---

### Bear Put Spread

**Description**: Buy a higher strike put, sell a lower strike put. Reduces cost but caps profit.

**Setup**:
- BUY 1 ATM/ITM Put
- SELL 1 OTM Put (hedge)

**When to Use**:
- Moderately Bearish sentiment
- Any IV environment
- When you want to reduce premium outlay

**P&L Profile**:
- Max Profit: (Higher Strike - Lower Strike) - Net Debit
- Max Loss: Net Debit paid
- Breakeven: Higher Strike - Net Debit

**Example**:
```
NIFTY @ 24,000
BUY NIFTY 24,000 PE @ ₹180
SELL NIFTY 23,900 PE @ ₹120
Net Debit: ₹60
Max Profit: ₹100 - ₹60 = ₹40 per share
```

---

## Credit Spread Strategies

### Bear Call Spread (Credit)

**Description**: **SELL** a lower strike call, BUY a higher strike call as hedge. Collect premium expecting underlying to stay below the short strike.

**Setup**:
- **SELL 1 OTM Call** (main trade - at resistance)
- BUY 1 further OTM Call (hedge)

**When to Use**:
- Bearish or Neutral sentiment
- HIGH IV (sell expensive premium)
- Max call OI acts as resistance
- Expect underlying to stay below short strike

**P&L Profile**:
- Max Profit: Net Credit received
- Max Loss: (Higher Strike - Lower Strike) - Net Credit
- Breakeven: Lower Strike + Net Credit

**Why This Strategy**:
- Main trade is a SELL (collect premium)
- Buy leg provides protection against unlimited loss
- Benefits from time decay (theta positive)
- Profits when market stays flat or falls

**Example**:
```
NIFTY @ 24,000 (Resistance at 24,200)
SELL NIFTY 24,200 CE @ ₹80  ← Main Trade
BUY NIFTY 24,300 CE @ ₹50   ← Hedge
Net Credit: ₹30
Max Profit: ₹30 × 25 = ₹750
Max Loss: (₹100 - ₹30) × 25 = ₹1,750
```

---

### Bull Put Spread (Credit)

**Description**: **SELL** a higher strike put, BUY a lower strike put as hedge. Collect premium expecting underlying to stay above the short strike.

**Setup**:
- **SELL 1 OTM Put** (main trade - at support)
- BUY 1 further OTM Put (hedge)

**When to Use**:
- Bullish or Neutral sentiment
- HIGH IV
- Max put OI acts as support
- Expect underlying to stay above short strike

**P&L Profile**:
- Max Profit: Net Credit received
- Max Loss: (Higher Strike - Lower Strike) - Net Credit
- Breakeven: Higher Strike - Net Credit

**Why This Strategy**:
- Main trade is a SELL (collect premium)
- Buy leg provides protection against large downside
- Benefits from time decay
- Profits when market stays flat or rises

**Example**:
```
NIFTY @ 24,000 (Support at 23,800)
SELL NIFTY 23,800 PE @ ₹70  ← Main Trade
BUY NIFTY 23,700 PE @ ₹45   ← Hedge
Net Credit: ₹25
Max Profit: ₹25 × 25 = ₹625
Max Loss: (₹100 - ₹25) × 25 = ₹1,875
```

---

## Neutral/Volatility Strategies

### Iron Condor

**Description**: Sell both an OTM call spread and an OTM put spread. Profit from range-bound market.

**Setup**:
- **SELL 1 OTM Call** (at resistance)
- BUY 1 further OTM Call (hedge)
- **SELL 1 OTM Put** (at support)
- BUY 1 further OTM Put (hedge)

**When to Use**:
- Neutral sentiment
- HIGH IV (collect expensive premium)
- Max call OI as upper bound, max put OI as lower bound
- Expect range-bound movement

**P&L Profile**:
- Max Profit: Total Net Credit
- Max Loss: Width of one spread - Net Credit
- Profit Zone: Between short strikes

**Example**:
```
NIFTY @ 24,000
SELL 24,200 CE @ ₹80, BUY 24,300 CE @ ₹50  (Call Spread Credit: ₹30)
SELL 23,800 PE @ ₹70, BUY 23,700 PE @ ₹45  (Put Spread Credit: ₹25)
Total Credit: ₹55
Profit Zone: 23,800 - 24,200
Max Profit: ₹55 × 25 = ₹1,375
```

---

### Long Straddle

**Description**: Buy ATM call and ATM put. Profit from big move in either direction.

**Setup**:
- BUY 1 ATM Call
- BUY 1 ATM Put

**When to Use**:
- Expect high volatility / big move
- LOW IV (buying cheap premium before volatility expansion)
- Major events upcoming (earnings, budget, etc.)

**P&L Profile**:
- Max Profit: Unlimited (big move in either direction)
- Max Loss: Total premium paid
- Breakeven: Strike ± Total Premium

---

### Short Straddle

**Description**: Sell ATM call and ATM put. Profit from low volatility / range-bound market.

**Setup**:
- **SELL 1 ATM Call**
- **SELL 1 ATM Put**

**When to Use**:
- Neutral sentiment, expect flat market
- HIGH IV (sell expensive premium, expect IV crush)

**Risk Warning**: ⚠️ Unlimited risk if market moves significantly.

---

### Long Strangle

**Description**: Buy OTM call and OTM put. Cheaper than straddle, needs bigger move.

**Setup**:
- BUY 1 OTM Call
- BUY 1 OTM Put

**When to Use**:
- Expect very high volatility
- LOW IV
- Want cheaper entry than straddle

---

### Short Strangle

**Description**: Sell OTM call and OTM put. Wider profit zone than short straddle.

**Setup**:
- **SELL 1 OTM Call** (at max call OI)
- **SELL 1 OTM Put** (at max put OI)

**When to Use**:
- Neutral sentiment
- HIGH IV
- Wider expected range than straddle

**Risk Warning**: ⚠️ Unlimited risk outside the range.

---

## Strategy Selection Logic

The bot automatically selects strategies based on:

1. **ML Prediction** (primary - ternary):
   - BULLISH → Long Call, Bull Call Spread, Bull Put Spread, Short Put
   - NEUTRAL → Iron Condor, Short Straddle, Short Strangle, Credit Spreads
   - BEARISH → Long Put, Bear Put Spread, Bear Call Spread, Short Call

2. **IV Regime**:
   - HIGH_IV → Sell strategies (Short Call/Put, Credit Spreads, Iron Condor)
   - LOW_IV → Buy strategies (Long Call/Put, Debit Spreads, Long Straddle)
   - NORMAL → All strategies considered

3. **Support/Resistance** (from max OI strikes):
   - Short strikes placed at max OI levels (natural S/R)

4. **Expiry Selection**:
   - **Indices** (NIFTY, BANKNIFTY, FINNIFTY): Weekly expiry (faster theta decay, tighter strikes)
   - **Stocks** (AXISBANK, HDFCBANK, etc.): Monthly expiry (more liquidity, wider availability)
   - **Multi-Expiry Evaluation**: All valid expiries within 5–45 DTE are evaluated; ML picks the best by confidence

5. **Spread Width** (ATR-based):
   - Spread width is dynamically set using the 14-period ATR of the underlying
   - Higher volatility → wider spreads, lower volatility → tighter spreads
   - Replaces fixed-width spread selection for better risk/reward adaptation

6. **Confidence-Gated DTE Floor**:
   - ML confidence ≥ 80% → minimum 5 DTE allowed (higher conviction, shorter-dated)
   - ML confidence < 80% → minimum 20 DTE required (more time buffer for lower conviction)
   - Enforced at order execution as defense-in-depth

---

## Confidence Calculation

Each signal includes a confidence score (0.0 - 1.0):

```
Base Confidence: 0.50

+ Sentiment Alignment:
  - Strong alignment: +0.15
  - Moderate alignment: +0.10

+ IV Regime Match:
  - Perfect match: +0.15 to +0.20
  - Good match: +0.05 to +0.10

+ Historical Trend Alignment:
  - Strong trend match: +0.10
  - Momentum match: +0.08

+ Risk/Reward Ratio:
  - R/R > 2: +0.15
  - R/R > 1.5: +0.10

+ RSI/Oversold-Overbought:
  - Favorable: +0.05

Minimum Confidence for Trade: 0.60 (0.70 for sell strategies)
```

---

## Risk Management

### Position Sizing
- Default: 1 lot per trade
- Max positions: 3 (live), 15 (paper trading)
- Capital per trade: ₹50,000 (sized for ~₹1L account)
- Duplicate blocking: same underlying + same strategy type rejected; different strategies on same underlying allowed

### Stop Loss Rules
| Strategy Type | Stop Loss Logic |
|---------------|-----------------|
| Long Options | 30% of premium paid |
| Short Options | Exit when premium doubles (100% loss) |
| Debit Spreads | 50% of net debit |
| Credit Spreads | Exit when loss equals credit received |

### Target Rules
| Strategy Type | Target Logic |
|---------------|--------------|
| Long Options | 50% profit on premium |
| Short Options | 50% of premium decay |
| Debit Spreads | 70% of max profit |
| Credit Spreads | 50% of credit received |

---

## Quick Reference: Sell-Primary Strategies

For traders who prefer **collecting premium** (theta decay):

| Strategy | Main Leg | Hedge | Best When |
|----------|----------|-------|-----------|
| Short Call | SELL Call | None | Bearish, High IV |
| Short Put | SELL Put | None | Bullish, High IV |
| Bear Call Spread | SELL Call | BUY Call | Bearish, High IV, Want limited risk |
| Bull Put Spread | SELL Put | BUY Put | Bullish, High IV, Want limited risk |
| Iron Condor | SELL Call + Put | BUY Call + Put | Neutral, High IV |
| Short Straddle | SELL Call + Put | None | Neutral, High IV, Expect no move |
| Short Strangle | SELL Call + Put (OTM) | None | Neutral, High IV, Wide range |

---

*Last Updated: March 2026*
