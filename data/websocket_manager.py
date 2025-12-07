"""
WebSocket Ticker Manager - Real-time Price Streaming for Options Trading
Manages Kite Connect WebSocket connections for live price updates
Based on proven websocket_manager.py pattern
"""
import time
import threading
from typing import Dict, List, Optional, Callable, Set
from datetime import datetime
from kiteconnect import KiteTicker

from config.settings import KITE_CONFIG, get_watchlist_assets, get_instrument_token
from core.logger import logger


class WebSocketManager:
    """Manages Kite Connect WebSocket ticker for real-time price updates"""
    
    def __init__(self):
        self.api_key = KITE_CONFIG["api_key"]
        self.access_token = None
        
        # WebSocket connection
        self.kws: Optional[KiteTicker] = None
        self.is_connected = False
        self.connection_lock = threading.Lock()
        
        # Subscriptions
        self.subscribed_tokens: Set[int] = set()
        self.symbol_to_token: Dict[str, int] = {}
        self.token_to_symbol: Dict[int, str] = {}
        
        # Price cache - stores latest prices
        self.price_cache: Dict[str, Dict] = {}
        
        # Callbacks
        self.price_callbacks: List[Callable] = []
        self.connection_callbacks: List[Callable] = []
        
        # Connection health
        self.last_tick_time: Optional[datetime] = None
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.health_check_interval = 30  # seconds
        
        # Load instrument mapping from watchlist
        self._load_instrument_mapping()
    
    def _load_instrument_mapping(self):
        """Load instrument token mapping from watchlist.json"""
        try:
            # Load from watchlist
            watchlist_assets = get_watchlist_assets()
            
            for asset in watchlist_assets:
                symbol = asset["name"].upper()
                token = asset.get("instrument_token")
                
                if token:
                    self.symbol_to_token[symbol] = token
                    self.token_to_symbol[token] = symbol
            
            logger.info(f"Loaded {len(self.symbol_to_token)} instrument mappings from watchlist")
            
        except Exception as e:
            logger.error(f"Failed to load instrument mappings: {e}")
    
    def add_instrument_mapping(self, symbol: str, token: int):
        """Dynamically add instrument mapping (for options symbols)"""
        self.symbol_to_token[symbol.upper()] = token
        self.token_to_symbol[token] = symbol.upper()
    
    def get_instrument_token(self, symbol: str) -> Optional[int]:
        """Get instrument token for a symbol"""
        return self.symbol_to_token.get(symbol.upper())
    
    def get_symbol_from_token(self, token: int) -> Optional[str]:
        """Get symbol from instrument token"""
        return self.token_to_symbol.get(token)
    
    def add_price_callback(self, callback: Callable):
        """Add callback function for price updates"""
        if callback not in self.price_callbacks:
            self.price_callbacks.append(callback)
    
    def remove_price_callback(self, callback: Callable):
        """Remove a price callback"""
        if callback in self.price_callbacks:
            self.price_callbacks.remove(callback)
    
    def add_connection_callback(self, callback: Callable):
        """Add callback function for connection events"""
        if callback not in self.connection_callbacks:
            self.connection_callbacks.append(callback)
    
    def set_access_token(self, access_token: str):
        """Set access token for WebSocket connection"""
        self.access_token = access_token
    
    def connect(self) -> bool:
        """Establish WebSocket connection"""
        try:
            with self.connection_lock:
                if self.is_connected:
                    logger.warning("WebSocket already connected")
                    return True
                
                if not self.access_token:
                    # Try to get from kite_auth
                    from auth.kite_auth import get_kite, is_authenticated
                    if is_authenticated():
                        kite = get_kite()
                        self.access_token = kite.access_token
                    else:
                        logger.error("No access token available for WebSocket")
                        return False
                
                # Initialize KiteTicker
                self.kws = KiteTicker(self.api_key, self.access_token)
                
                # Set up event handlers
                self.kws.on_ticks = self._on_ticks
                self.kws.on_connect = self._on_connect
                self.kws.on_close = self._on_close
                self.kws.on_error = self._on_error
                self.kws.on_reconnect = self._on_reconnect
                self.kws.on_noreconnect = self._on_noreconnect
                
                # Connect
                logger.info("Connecting to Kite WebSocket...")
                self.kws.connect(threaded=True)
                
                # Wait for connection
                for i in range(10):  # Wait up to 10 seconds
                    if self.is_connected:
                        break
                    time.sleep(1)
                
                if self.is_connected:
                    logger.info("✅ WebSocket connected successfully")
                    self._start_health_monitor()
                    return True
                else:
                    logger.error("❌ WebSocket connection timeout")
                    return False
                
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect WebSocket"""
        try:
            with self.connection_lock:
                if self.kws:
                    self.kws.close()
                    self.kws = None
                self.is_connected = False
                self.subscribed_tokens.clear()
                logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"Error disconnecting WebSocket: {e}")
    
    def subscribe_symbols(self, symbols: List[str], mode: str = "ltp") -> bool:
        """
        Subscribe to symbols for real-time updates.
        
        Args:
            symbols: List of trading symbols
            mode: 'ltp' for last price only, 'quote' for full quote, 'full' for all data
        """
        try:
            if not self.is_connected:
                logger.warning("WebSocket not connected, cannot subscribe")
                return False
            
            # Convert symbols to tokens
            tokens = []
            for symbol in symbols:
                token = self.get_instrument_token(symbol)
                if token:
                    tokens.append(token)
                    self.subscribed_tokens.add(token)
                else:
                    logger.warning(f"No instrument token found for {symbol}")
            
            if tokens:
                self.kws.subscribe(tokens)
                
                # Set mode
                if mode == "ltp":
                    self.kws.set_mode(self.kws.MODE_LTP, tokens)
                elif mode == "quote":
                    self.kws.set_mode(self.kws.MODE_QUOTE, tokens)
                else:
                    self.kws.set_mode(self.kws.MODE_FULL, tokens)
                
                logger.info(f"Subscribed to {len(tokens)} instruments in {mode} mode")
                return True
            else:
                logger.warning("No valid tokens to subscribe")
                return False
            
        except Exception as e:
            logger.error(f"Failed to subscribe to symbols: {e}")
            return False
    
    def subscribe_tokens(self, tokens: List[int], mode: str = "ltp") -> bool:
        """Subscribe directly using instrument tokens (for options)"""
        try:
            if not self.is_connected:
                logger.warning("WebSocket not connected, cannot subscribe")
                return False
            
            self.kws.subscribe(tokens)
            self.subscribed_tokens.update(tokens)
            
            # Set mode
            if mode == "ltp":
                self.kws.set_mode(self.kws.MODE_LTP, tokens)
            elif mode == "quote":
                self.kws.set_mode(self.kws.MODE_QUOTE, tokens)
            else:
                self.kws.set_mode(self.kws.MODE_FULL, tokens)
            
            logger.info(f"Subscribed to {len(tokens)} tokens in {mode} mode")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to tokens: {e}")
            return False
    
    def unsubscribe_symbols(self, symbols: List[str]):
        """Unsubscribe from symbols"""
        try:
            if not self.is_connected:
                return
            
            tokens = []
            for symbol in symbols:
                token = self.get_instrument_token(symbol)
                if token and token in self.subscribed_tokens:
                    tokens.append(token)
                    self.subscribed_tokens.discard(token)
            
            if tokens:
                self.kws.unsubscribe(tokens)
                logger.info(f"Unsubscribed from {len(tokens)} instruments")
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe from symbols: {e}")
    
    def unsubscribe_tokens(self, tokens: List[int]):
        """Unsubscribe using instrument tokens"""
        try:
            if not self.is_connected:
                return
            
            valid_tokens = [t for t in tokens if t in self.subscribed_tokens]
            
            if valid_tokens:
                self.kws.unsubscribe(valid_tokens)
                for t in valid_tokens:
                    self.subscribed_tokens.discard(t)
                logger.info(f"Unsubscribed from {len(valid_tokens)} tokens")
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe from tokens: {e}")
    
    def get_ltp(self, symbol: str) -> Optional[float]:
        """Get last traded price from cache"""
        cached = self.price_cache.get(symbol.upper())
        if cached:
            return cached.get("ltp")
        return None
    
    def get_price_data(self, symbol: str) -> Optional[Dict]:
        """Get full price data from cache"""
        return self.price_cache.get(symbol.upper())
    
    def _on_connect(self, ws, response):
        """Handle WebSocket connection"""
        self.is_connected = True
        self.reconnect_attempts = 0
        logger.info("📡 WebSocket connected")
        
        # Notify connection callbacks
        for callback in self.connection_callbacks:
            try:
                callback("connected", response)
            except Exception as e:
                logger.error(f"Connection callback error: {e}")
    
    def _on_close(self, ws, code, reason):
        """Handle WebSocket disconnection"""
        self.is_connected = False
        logger.warning(f"📡 WebSocket disconnected: {code} - {reason}")
        
        # Notify connection callbacks
        for callback in self.connection_callbacks:
            try:
                callback("disconnected", {"code": code, "reason": reason})
            except Exception as e:
                logger.error(f"Disconnection callback error: {e}")
    
    def _on_error(self, ws, code, reason):
        """Handle WebSocket errors"""
        logger.error(f"❌ WebSocket error: {code} - {reason}")
        
        # Notify connection callbacks
        for callback in self.connection_callbacks:
            try:
                callback("error", {"code": code, "reason": reason})
            except Exception as e:
                logger.error(f"Error callback error: {e}")
    
    def _on_reconnect(self, ws, attempts_count):
        """Handle WebSocket reconnection"""
        self.reconnect_attempts = attempts_count
        logger.info(f"🔄 WebSocket reconnecting... (attempt {attempts_count})")
    
    def _on_noreconnect(self, ws):
        """Handle WebSocket failed reconnection"""
        logger.error("❌ WebSocket reconnection failed")
        self.is_connected = False
    
    def _on_ticks(self, ws, ticks):
        """Handle incoming price ticks"""
        try:
            self.last_tick_time = datetime.now()
            
            for tick in ticks:
                instrument_token = tick['instrument_token']
                symbol = self.get_symbol_from_token(instrument_token)
                
                # Extract relevant price data
                price_data = {
                    'symbol': symbol or f"TOKEN_{instrument_token}",
                    'instrument_token': instrument_token,
                    'ltp': tick.get('last_price', 0),
                    'volume': tick.get('volume', 0),
                    'timestamp': datetime.now(),
                    'change': tick.get('change', 0),
                    'change_percent': tick.get('change_percent', 0),
                    'open': tick.get('ohlc', {}).get('open', 0),
                    'high': tick.get('ohlc', {}).get('high', 0),
                    'low': tick.get('ohlc', {}).get('low', 0),
                    'close': tick.get('ohlc', {}).get('close', 0),
                    'bid': tick.get('depth', {}).get('buy', [{}])[0].get('price', 0),
                    'ask': tick.get('depth', {}).get('sell', [{}])[0].get('price', 0),
                }
                
                # Update price cache
                if symbol:
                    self.price_cache[symbol] = price_data
                self.price_cache[str(instrument_token)] = price_data
                
                # Notify all price callbacks
                for callback in self.price_callbacks:
                    try:
                        callback(price_data)
                    except Exception as e:
                        logger.error(f"Price callback error: {e}")
                
        except Exception as e:
            logger.error(f"Error processing ticks: {e}")
    
    def _start_health_monitor(self):
        """Start health monitoring thread"""
        def health_check():
            while self.is_connected:
                try:
                    time.sleep(self.health_check_interval)
                    
                    # Check if we've received ticks recently
                    if self.last_tick_time:
                        time_since_last_tick = (datetime.now() - self.last_tick_time).total_seconds()
                        if time_since_last_tick > 60:  # No ticks for 60 seconds
                            logger.warning(f"No ticks received for {time_since_last_tick:.0f} seconds")
                    
                except Exception as e:
                    logger.error(f"Health check error: {e}")
        
        health_thread = threading.Thread(target=health_check, daemon=True)
        health_thread.start()
    
    def get_connection_status(self) -> Dict:
        """Get current connection status"""
        return {
            'connected': self.is_connected,
            'subscribed_count': len(self.subscribed_tokens),
            'subscribed_tokens': list(self.subscribed_tokens),
            'reconnect_attempts': self.reconnect_attempts,
            'last_tick_time': self.last_tick_time.isoformat() if self.last_tick_time else None,
            'cached_prices': len(self.price_cache),
        }
    
    # ========== Wrapper methods for position_tracker compatibility ==========
    
    def start(self) -> bool:
        """Start WebSocket connection (alias for connect)"""
        return self.connect()
    
    def stop(self):
        """Stop WebSocket connection (alias for disconnect)"""
        self.disconnect()
    
    def subscribe(self, tokens: List[int], mode: str = "ltp") -> bool:
        """
        Subscribe to instrument tokens (wrapper for subscribe_tokens).
        
        Args:
            tokens: List of instrument tokens
            mode: Subscription mode (ltp, quote, full)
        
        Returns:
            True if successful
        """
        return self.subscribe_tokens(tokens, mode)
    
    def unsubscribe(self, tokens: List[int]):
        """
        Unsubscribe from instrument tokens (wrapper for unsubscribe_tokens).
        
        Args:
            tokens: List of instrument tokens
        """
        self.unsubscribe_tokens(tokens)
    
    def get_price(self, token: int) -> Optional[float]:
        """
        Get latest price for an instrument token.
        
        Args:
            token: Instrument token
            
        Returns:
            Latest price or None
        """
        data = self.price_cache.get(str(token))
        if data:
            return data.get('ltp')
        return None
    
    def get_all_prices(self) -> Dict[int, float]:
        """
        Get all cached prices.
        
        Returns:
            Dictionary of token -> price
        """
        prices = {}
        for key, data in self.price_cache.items():
            if key.isdigit():
                prices[int(key)] = data.get('ltp', 0)
        return prices
    
    def register_callback(self, event: str, callback: Callable):
        """
        Register a callback for events.
        
        Args:
            event: Event type ('price_update', 'connection')
            callback: Callback function
        """
        if event == "price_update":
            # Wrap callback to extract token and price
            def price_wrapper(price_data):
                token = price_data.get('instrument_token')
                price = price_data.get('ltp')
                if token and price:
                    callback(token, price)
            
            self.add_price_callback(price_wrapper)
        elif event == "connection":
            self.add_connection_callback(callback)
        else:
            logger.warning(f"Unknown callback event type: {event}")


# ========== WebSocketTicker class for bot.py compatibility ==========

class WebSocketTicker:
    """
    Wrapper class for WebSocket functionality.
    Used by bot.py and position_tracker for real-time price streaming.
    """
    
    def __init__(self, kite=None):
        """
        Initialize WebSocketTicker.
        
        Args:
            kite: KiteConnect instance (optional, uses global if not provided)
        """
        self.manager = websocket_manager
        
        if kite:
            self.manager.set_access_token(kite.access_token)
    
    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self.manager.is_connected
    
    def start(self) -> bool:
        """Start WebSocket connection."""
        return self.manager.connect()
    
    def stop(self):
        """Stop WebSocket connection."""
        self.manager.disconnect()
    
    def subscribe(self, tokens: List[int], mode: str = "ltp") -> bool:
        """Subscribe to instrument tokens."""
        return self.manager.subscribe_tokens(tokens, mode)
    
    def unsubscribe(self, tokens: List[int]):
        """Unsubscribe from instrument tokens."""
        self.manager.unsubscribe_tokens(tokens)
    
    def get_price(self, token: int) -> Optional[float]:
        """Get latest price for a token."""
        return self.manager.get_price(token)
    
    def get_all_prices(self) -> Dict[int, float]:
        """Get all cached prices."""
        return self.manager.get_all_prices()
    
    def register_callback(self, event: str, callback: Callable):
        """Register a callback for events."""
        self.manager.register_callback(event, callback)
    
    def get_status(self) -> Dict:
        """Get connection status."""
        return self.manager.get_connection_status()


# Singleton instance
websocket_manager = WebSocketManager()
