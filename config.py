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
END_DATE = '30 Jun 2025'

# ML Prediction Configuration
PREDICTION_HORIZONS = {
    'hour': 1,      # Predict next 1 hour candle
    'day': 24,      # Predict next 24 hour candles (1 day)
    'week': 168,    # Predict next 168 hour candles (1 week)
    '15days': 360,  # Predict next 360 hour candles (15 days)
    'month': 720    # Predict next 720 hour candles (30 days)
}

# Default prediction settings
DEFAULT_PREDICTION_HORIZON = 'day'  # Default to daily prediction
DEFAULT_TIMEFRAME = 'hourly'         # Base timeframe for predictions

# ML Feature Engineering
CONFLUENCE_ZONE_WIDTH = 0.015  # 1.5% zone width for level clustering
MAX_LEVEL_DISTANCE = 0.05      # 5% max distance to consider levels relevant
LEVEL_STRENGTH_DECAY = 0.95    # Decay factor for older levels
MIN_CONFLUENCE_SCORE = 1.0     # Minimum score to consider confluence significant

# Target Variable Configuration
FRACTAL_TARGET_CLASSES = {
    0: 'no_fractal',      # No fractal formation
    1: 'bullish_fractal', # Swing low (fractal up)
    2: 'bearish_fractal'  # Swing high (fractal down)
}
