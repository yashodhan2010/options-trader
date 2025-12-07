"""
Core package initialization
"""
from .logger import logger, setup_logger
from .utils import *
from .database import Database, database
from .notifications import NotificationService, notification_service
