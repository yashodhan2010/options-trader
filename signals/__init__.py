"""
Signals package initialization

All signals are now ML-driven. The MLSignalGenerator is the sole source
of trading signals based on ML model predictions.
"""
# ML-Only Signal Generator (primary)
from .ml_signal_generator import MLSignalGenerator, ml_signal_generator, signal_generator

# Exit signal generator (rule-based exits for risk management)
from .exit_signal_generator import ExitSignalGenerator, exit_signal_generator, ExitReason, ExitSignal

# Legacy import (deprecated - use ml_signal_generator instead)
from .signal_generator import SignalGenerator
