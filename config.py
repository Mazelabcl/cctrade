# config.py

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Credentials
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')

if not BINANCE_API_KEY or not BINANCE_API_SECRET:
    raise ValueError("API key and/or secret not found. Please set them as environment variables.")


# Global Constants
ROUNDING_PRECISION = 1  # USD
CONFLUENCE_THRESHOLD = 2  # Minimum number of overlapping levels to mark as confluence
HIT_COUNT_THRESHOLD = 3  # Levels hit less than this are considered valid
PRICE_RANGE = 100  # Price range for confluence zones

# Logging Configuration
LOG_FILE = 'trading_bot.log'

# Data Fetching Parameters
SYMBOL = 'BTCUSDT'
START_DATE = '1 Jan 2017'
END_DATE = '31 Dec 2024'
