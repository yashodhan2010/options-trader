"""
Data package initialization
"""
from .data_fetcher import DataFetcher, data_fetcher

# NSE downloader with browser mimicking
from .nse_downloader import NSEDownloader

# Legacy NSE Bhavcopy collector
try:
    from .nse_bhavcopy import (
        NSEBhavcopyCollector,
        get_bhavcopy_collector,
        download_historical_bhavcopy
    )
except ImportError:
    # jugaad-data may not be installed
    pass
