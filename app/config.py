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

    # Foundation defaults (overridable via Settings UI)
    FOUNDATION_DATA_START = '2020-01-01'
    FOUNDATION_DATA_END = '2025-12-31'
    FOUNDATION_TRAIN_TEST_CUTOFF = '2024-06-01'
    FOUNDATION_FETCH_TIMEFRAMES = ['1d', '1w', '1M']
    FOUNDATION_HTF_TIMEFRAMES = ['1d', '1w', '1M']
    FOUNDATION_FRACTAL_TIMEFRAMES = ['1d', '1w', '1M']
    FOUNDATION_FIBONACCI_TIMEFRAMES = ['1d', '1w']
    FOUNDATION_VP_TIMEFRAMES = ['1d', '1w', '1M']
    FOUNDATION_ALL_TIMEFRAMES = ['1h', '4h', '1d', '1w', '1M']

    # Technical Analysis
    ROUNDING_PRECISION = 1
    CONFLUENCE_THRESHOLD = 2
    HIT_COUNT_THRESHOLD = 3
    PRICE_RANGE = 100
    TOUCHED_THRESHOLD = 1  # touches needed to consider a level "touched" (parametrizable)

    # Fibonacci — Chart Champions methodology
    CC_FIBONACCI_RATIOS = [('CC', 0.639)]       # Daniel's golden pocket (0.618-0.66 avg)
    IGOR_FIBONACCI_RATIOS = [('0.25', 0.25), ('0.50', 0.50), ('0.75', 0.75)]  # Igor's quarters

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

    # Backtesting
    BACKTEST_INITIAL_CASH = 100_000.0
    BACKTEST_COMMISSION = 0.001        # 0.1% per trade
    BACKTEST_RISK_PER_TRADE = 0.02     # 2% of portfolio
    BACKTEST_CONFIDENCE_THRESHOLD = 0.55
    BACKTEST_LEVEL_PROXIMITY_PCT = 0.02  # price within 2% of level
    BACKTEST_ATR_SL_MULT = 1.5         # stop-loss = 1.5x ATR
    BACKTEST_ATR_TP_MULT = 3.0         # take-profit = 3.0x ATR
    BACKTEST_INVALIDATE_ON_FIRST_TOUCH = True  # levels disappear on first touch in backtest

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
