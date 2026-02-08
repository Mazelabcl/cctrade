import os
from dotenv import load_dotenv

load_dotenv()

basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key-change-in-production')

    # Database
    SQLALCHEMY_DATABASE_URI = os.getenv(
        'DATABASE_URL',
        f'sqlite:///{os.path.join(basedir, "instance", "tradebot.db")}'
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Binance API
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
    BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', '')

    # Trading
    SYMBOL = 'BTCUSDT'
    START_DATE = '1 Jan 2017'
    END_DATE = '30 Jun 2025'

    # Technical Analysis
    ROUNDING_PRECISION = 1
    CONFLUENCE_THRESHOLD = 2
    HIT_COUNT_THRESHOLD = 3
    PRICE_RANGE = 100

    # ML Prediction
    PREDICTION_HORIZONS = {
        'hour': 1,
        'day': 24,
        'week': 168,
        '15days': 360,
        'month': 720,
    }
    DEFAULT_PREDICTION_HORIZON = 'day'
    DEFAULT_TIMEFRAME = 'hourly'

    # ML Feature Engineering
    CONFLUENCE_ZONE_WIDTH = 0.015
    MAX_LEVEL_DISTANCE = 0.05
    LEVEL_STRENGTH_DECAY = 0.95
    MIN_CONFLUENCE_SCORE = 1.0

    # Target Variable
    FRACTAL_TARGET_CLASSES = {
        0: 'no_fractal',
        1: 'bullish_fractal',
        2: 'bearish_fractal',
    }

    # Scheduler
    SCHEDULER_API_ENABLED = False
    SCHEDULER_ENABLED = os.getenv('SCHEDULER_ENABLED', 'false').lower() == 'true'

    # Logging
    LOG_FILE = 'trading_bot.log'


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite://'  # in-memory


config_by_name = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
}
