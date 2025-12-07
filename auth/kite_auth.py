"""
Kite Connect Authentication and Session Management
Based on proven connection pattern - stores token and reuses for the day
"""
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from kiteconnect import KiteConnect

from config.settings import KITE_CONFIG, DATA_DIR

# Set logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Basic constants
kite = KiteConnect(api_key=KITE_CONFIG["api_key"])
SESSION_FILE = DATA_DIR / "session.json"


def save_token(token_data: dict) -> None:
    """Save access token to a file for reuse throughout the day."""
    DATA_DIR.mkdir(exist_ok=True)
    with open(SESSION_FILE, "w") as f:
        json.dump(token_data, f)
    logger.info("Access token saved to file")


def load_token() -> Optional[dict]:
    """Load token from file if it exists."""
    if SESSION_FILE.exists():
        with open(SESSION_FILE, "r") as f:
            return json.load(f)
    return None


def get_login_url() -> str:
    """Return Kite login URL."""
    return kite.login_url()


def generate_access_token(request_token: str) -> KiteConnect:
    """Generate and save new access token."""
    data = kite.generate_session(request_token, api_secret=KITE_CONFIG["api_secret"])
    kite.set_access_token(data["access_token"])
    
    save_token({
        "access_token": data["access_token"],
        "date": datetime.today().strftime("%Y-%m-%d")
    })
    
    logger.info("New access token generated and saved")
    return kite


def connect() -> KiteConnect:
    """
    Connect to Kite API.
    Reuses existing token if valid for today, otherwise prompts for fresh login.
    
    Returns:
        Authenticated KiteConnect instance
    """
    global kite
    
    # Try to reuse existing token if valid
    data = load_token()
    if data and data.get("date") == datetime.today().strftime("%Y-%m-%d"):
        kite.set_access_token(data["access_token"])
        
        # Verify token is still valid
        try:
            profile = kite.profile()
            logger.info(f"✅ Reusing existing access token for {profile.get('user_name', 'user')}")
            return kite
        except Exception as e:
            logger.warning(f"Stored token invalid: {e}")
    
    # Else require fresh login
    login_url = get_login_url()
    print(f"\n{'='*60}")
    print("🔐 KITE CONNECT LOGIN REQUIRED")
    print(f"{'='*60}")
    print(f"\n📎 Click here to log in to Zerodha Kite Connect:\n{login_url}\n")
    request_token = input("Paste the `request_token` you got after login and hit Enter: ").strip()
    
    if not request_token:
        raise ValueError("Request token cannot be empty")
    
    kite = generate_access_token(request_token)
    return kite


def get_kite() -> KiteConnect:
    """
    Get the authenticated Kite instance.
    Call connect() first if not authenticated.
    
    Returns:
        KiteConnect instance
    """
    return kite


def is_authenticated() -> bool:
    """Check if currently authenticated with valid token."""
    data = load_token()
    if data and data.get("date") == datetime.today().strftime("%Y-%m-%d"):
        try:
            kite.set_access_token(data["access_token"])
            kite.profile()
            return True
        except Exception:
            return False
    return False


def get_profile() -> Optional[dict]:
    """Get user profile."""
    try:
        return kite.profile()
    except Exception as e:
        logger.error(f"Failed to get profile: {e}")
        return None


def get_margins() -> Optional[dict]:
    """Get account margins."""
    try:
        return kite.margins()
    except Exception as e:
        logger.error(f"Failed to get margins: {e}")
        return None


def logout() -> None:
    """Logout and clear stored session."""
    global kite
    try:
        if SESSION_FILE.exists():
            SESSION_FILE.unlink()
            logger.info("Session file deleted")
        kite = KiteConnect(api_key=KITE_CONFIG["api_key"])
        logger.info("Logged out successfully")
    except Exception as e:
        logger.error(f"Logout error: {e}")


if __name__ == "__main__":
    connect()
