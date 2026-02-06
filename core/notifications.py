"""
Notification utilities for alerts
"""
import requests
from typing import Optional

from config.settings import NOTIFICATION_CONFIG
from core.logger import logger


class NotificationService:
    """
    Service for sending notifications via various channels.
    Supports: Telegram, WhatsApp (via CallMeBot)
    """
    
    def __init__(self):
        self.telegram_enabled = NOTIFICATION_CONFIG.get("telegram_enabled", False)
        self.telegram_token = NOTIFICATION_CONFIG.get("telegram_bot_token", "")
        self.telegram_chat_id = NOTIFICATION_CONFIG.get("telegram_chat_id", "")
        
        # WhatsApp via CallMeBot
        self.whatsapp_enabled = NOTIFICATION_CONFIG.get("whatsapp_enabled", False)
        self.whatsapp_phone = NOTIFICATION_CONFIG.get("whatsapp_phone", "")
        self.whatsapp_apikey = NOTIFICATION_CONFIG.get("whatsapp_apikey", "")
    
    def send_telegram(self, message: str, parse_mode: str = "HTML") -> bool:
        """
        Send a Telegram notification.
        
        Args:
            message: Message to send
            parse_mode: Parse mode (HTML or Markdown)
            
        Returns:
            True if successful
        """
        if not self.telegram_enabled:
            logger.debug("Telegram notifications disabled")
            return False
        
        if not self.telegram_token or not self.telegram_chat_id:
            logger.warning("Telegram not configured properly")
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": parse_mode,
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.debug("Telegram notification sent")
                return True
            else:
                logger.error(f"Telegram error: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")
            return False
    
    def send_whatsapp(self, message: str) -> bool:
        """
        Send a WhatsApp notification via CallMeBot.
        
        Setup: Send "I allow callmebot to send me messages" to +34 644 59 71 67
        on WhatsApp. You'll receive your API key.
        
        Args:
            message: Message to send (plain text, no HTML)
            
        Returns:
            True if successful
        """
        if not self.whatsapp_enabled:
            logger.debug("WhatsApp notifications disabled")
            return False
        
        if not self.whatsapp_phone or not self.whatsapp_apikey:
            logger.warning("WhatsApp not configured properly (need phone + apikey)")
            return False
        
        try:
            import urllib.parse
            encoded_msg = urllib.parse.quote_plus(message)
            url = (
                f"https://api.callmebot.com/whatsapp.php"
                f"?phone={self.whatsapp_phone}"
                f"&text={encoded_msg}"
                f"&apikey={self.whatsapp_apikey}"
            )
            
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                logger.debug("WhatsApp notification sent")
                return True
            else:
                logger.error(f"WhatsApp error ({response.status_code}): {response.text[:200]}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send WhatsApp notification: {e}")
            return False
    
    def _strip_html(self, message: str) -> str:
        """Strip HTML tags from message for WhatsApp (plain text only)."""
        import re
        # Replace <b>text</b> with *text* (WhatsApp bold)
        text = re.sub(r'<b>(.*?)</b>', r'*\1*', message)
        # Remove any remaining HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        return text.strip()
    
    def send_trade_alert(
        self,
        action: str,
        strategy: str,
        underlying: str,
        details: str,
        pnl: Optional[float] = None,
    ) -> None:
        """
        Send a trade alert notification.
        
        Args:
            action: Action (ENTRY, EXIT, SL_HIT, TARGET_HIT)
            strategy: Strategy name
            underlying: Underlying asset
            details: Additional details
            pnl: P&L if applicable
        """
        emoji_map = {
            "ENTRY": "🟢",
            "EXIT": "🔵",
            "SL_HIT": "🔴",
            "TARGET_HIT": "🎯",
        }
        
        emoji = emoji_map.get(action, "📊")
        
        message = f"""
{emoji} <b>{action}</b>

<b>Strategy:</b> {strategy}
<b>Underlying:</b> {underlying}
{details}
"""
        
        if pnl is not None:
            pnl_emoji = "📈" if pnl >= 0 else "📉"
            message += f"\n{pnl_emoji} <b>P&L:</b> ₹{pnl:,.2f}"
        
        self.send_telegram(message)
        self.send_whatsapp(self._strip_html(message))
    
    def send_daily_summary(
        self,
        date: str,
        total_pnl: float,
        num_trades: int,
        winners: int,
        losers: int,
    ) -> None:
        """
        Send daily trading summary.
        
        Args:
            date: Date string
            total_pnl: Total P&L
            num_trades: Number of trades
            winners: Number of winning trades
            losers: Number of losing trades
        """
        pnl_emoji = "📈" if total_pnl >= 0 else "📉"
        win_rate = (winners / num_trades * 100) if num_trades > 0 else 0
        
        message = f"""
📊 <b>Daily Summary - {date}</b>

{pnl_emoji} <b>Total P&L:</b> ₹{total_pnl:,.2f}

<b>Trades:</b> {num_trades}
<b>Winners:</b> {winners} ✅
<b>Losers:</b> {losers} ❌
<b>Win Rate:</b> {win_rate:.1f}%
"""
        
        self.send_telegram(message)
        self.send_whatsapp(self._strip_html(message))
    
    def send_signal_alert(
        self,
        strategy: str,
        underlying: str,
        confidence: float,
        rationale: str,
    ) -> None:
        """
        Send signal alert notification.
        
        Args:
            strategy: Strategy name
            underlying: Underlying asset
            confidence: Signal confidence
            rationale: Signal rationale
        """
        message = f"""
🔔 <b>New Signal</b>

<b>Strategy:</b> {strategy}
<b>Underlying:</b> {underlying}
<b>Confidence:</b> {confidence:.0%}
<b>Rationale:</b> {rationale}
"""
        
        self.send_telegram(message)
        self.send_whatsapp(self._strip_html(message))


# Singleton instance
notification_service = NotificationService()
