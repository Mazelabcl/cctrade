"""Run backtest for PrevSession + VWAP levels on 4h."""
import sys
import os
import time
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'instance', 'tradebot.db')

# Force WAL checkpoint before starting
conn = sqlite3.connect(DB_PATH, timeout=30)
conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
conn.execute('PRAGMA journal_mode=WAL')
conn.close()
print("WAL checkpointed")

# Now run with SQLAlchemy
from app import create_app
from app.extensions import db as flask_db
from app.services.level_trade_backtest_db import run_level_trade_backtest

app = create_app()

# Set longer busy timeout in the engine
with app.app_context():
    flask_db.session.execute(flask_db.text('PRAGMA busy_timeout = 30000'))
    flask_db.session.execute(flask_db.text('PRAGMA journal_mode = WAL'))

    print('Running backtest on 4h...')
    start = time.time()
    results = run_level_trade_backtest(
        flask_db.session,
        exec_timeframe='4h',
        rr_ratios=[1.0, 2.0, 3.0],
        naked_only=True,
        timeout=100,
    )
    elapsed = time.time() - start
    print(f'Done in {elapsed:.0f}s, {len(results)} results')

    # Show ALL results sorted by WR
    print()
    print(f'{"Level Type":30s} {"TF":8s} {"Strategy":15s} {"WR%":>6s} {"PF":>6s} {"Trades":>6s}')
    print('-' * 75)
    for r in sorted(results, key=lambda x: -(x.get('win_rate', 0) or 0)):
        lt = r.get('level_type', '?')
        tf = r.get('level_source_timeframe', '?')
        strat = r.get('strategy_name', '?')
        wr = r.get('win_rate', 0) or 0
        pf = r.get('profit_factor', 0) or 0
        trades = r.get('total_trades', 0)
        if trades >= 5:
            print(f'  {lt:30s} {tf:8s} {strat:15s} {wr:6.1f} {pf:6.2f} {trades:6d}')
