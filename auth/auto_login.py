"""
Automated Kite Connect Login using Selenium + TOTP

Automates the full Zerodha login flow:
1. Opens Kite login page in headless Chrome
2. Enters user ID and password
3. Generates TOTP code using pyotp
4. Extracts request_token from redirect URL
5. Generates access_token via Kite API

Requires environment variables:
    KITE_USER_ID     - Zerodha client ID (e.g. AB1234)
    KITE_PASSWORD    - Zerodha login password
    KITE_TOTP_SECRET - TOTP secret key (base32 string from 2FA setup)
"""
import os
import time
import logging
from urllib.parse import urlparse, parse_qs
from typing import Optional

logger = logging.getLogger(__name__)


def _get_credentials() -> tuple:
    """
    Load Zerodha login credentials from environment variables.
    
    Returns:
        Tuple of (user_id, password, totp_secret)
        
    Raises:
        ValueError if any credential is missing
    """
    user_id = os.getenv("KITE_USER_ID", "").strip()
    password = os.getenv("KITE_PASSWORD", "").strip()
    totp_secret = os.getenv("KITE_TOTP_SECRET", "").strip()
    
    missing = []
    if not user_id:
        missing.append("KITE_USER_ID")
    if not password:
        missing.append("KITE_PASSWORD")
    if not totp_secret:
        missing.append("KITE_TOTP_SECRET")
    
    if missing:
        raise ValueError(
            f"Missing environment variables for auto-login: {', '.join(missing)}\n"
            f"Set them in your .env file or as system environment variables."
        )
    
    return user_id, password, totp_secret


def _generate_totp(secret: str) -> str:
    """Generate current TOTP code from secret key."""
    import pyotp
    totp = pyotp.TOTP(secret)
    return totp.now()


def auto_login(login_url: str) -> Optional[str]:
    """
    Perform automated Zerodha login using headless Chrome.
    
    Args:
        login_url: Kite Connect login URL
        
    Returns:
        request_token string, or None if auto-login fails
    """
    try:
        user_id, password, totp_secret = _get_credentials()
    except ValueError as e:
        logger.warning(f"Auto-login not configured: {e}")
        return None
    
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
    except ImportError:
        logger.warning("Selenium not installed. Run: pip install selenium")
        return None
    
    driver = None
    try:
        logger.info("Starting automated Zerodha login...")
        
        # Suppress verbose Selenium/urllib3 debug logging
        logging.getLogger("selenium").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        
        # Setup headless Chrome
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1280,720")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--log-level=3")  # Suppress console noise
        
        # Selenium 4+ has built-in driver manager
        driver = webdriver.Chrome(options=chrome_options)
        driver.set_page_load_timeout(30)
        wait = WebDriverWait(driver, 15)
        
        # Step 1: Open login page
        logger.info("Opening Kite login page...")
        driver.get(login_url)
        
        # Step 2: Enter User ID
        logger.info("Entering credentials...")
        user_id_field = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='text']#userid, input#userid"))
        )
        user_id_field.clear()
        user_id_field.send_keys(user_id)
        
        # Step 3: Enter Password
        password_field = driver.find_element(By.CSS_SELECTOR, "input[type='password']#password, input#password")
        password_field.clear()
        password_field.send_keys(password)
        
        # Step 4: Click Login button
        login_button = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
        login_button.click()
        
        # Step 5: Wait for TOTP page and enter TOTP
        logger.info("Entering TOTP...")
        totp_field = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='text']#userid, input[type='number'], input[label='External TOTP'], input.su-input-field"))
        )
        
        # Small delay to ensure page is ready
        time.sleep(1)
        
        # Re-locate TOTP field (page may have refreshed)
        totp_fields = driver.find_elements(By.CSS_SELECTOR, "input[type='number'], input[type='text']")
        totp_field = None
        for field in totp_fields:
            if field.is_displayed():
                totp_field = field
                break
        
        if not totp_field:
            logger.error("Could not find TOTP input field")
            return None
        
        totp_code = _generate_totp(totp_secret)
        totp_field.clear()
        totp_field.send_keys(totp_code)
        
        # Step 6: Submit TOTP (some pages auto-submit, some need button click)
        time.sleep(1)
        try:
            submit_buttons = driver.find_elements(By.CSS_SELECTOR, "button[type='submit']")
            for btn in submit_buttons:
                if btn.is_displayed():
                    btn.click()
                    break
        except Exception:
            pass  # Auto-submitted
        
        # Step 7: Wait for redirect and extract request_token
        logger.info("Waiting for redirect...")
        
        # Wait for URL to contain request_token (redirect to our callback URL)
        for _ in range(30):  # Max 30 seconds
            time.sleep(1)
            current_url = driver.current_url
            
            if "request_token" in current_url:
                parsed = urlparse(current_url)
                params = parse_qs(parsed.query)
                request_token = params.get("request_token", [None])[0]
                
                if request_token:
                    logger.info("Auto-login successful! Request token obtained.")
                    return request_token
            
            # Check for login errors
            try:
                error_elements = driver.find_elements(By.CSS_SELECTOR, ".error-message, .status-message.error")
                for elem in error_elements:
                    if elem.is_displayed() and elem.text.strip():
                        logger.error(f"Login error: {elem.text.strip()}")
                        return None
            except Exception:
                pass
        
        logger.error(f"Timeout waiting for redirect. Current URL: {driver.current_url}")
        return None
        
    except Exception as e:
        logger.error(f"Auto-login failed: {e}")
        return None
        
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass


def is_auto_login_configured() -> bool:
    """Check if auto-login credentials are set."""
    try:
        _get_credentials()
        return True
    except ValueError:
        return False
