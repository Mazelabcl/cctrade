"""Backtest views — run and review backtests."""
import threading
from flask import Blueprint, render_template, jsonify, request, abort, current_app
from ..extensions import db
from ..models import MLModel, BacktestResult, IndividualLevelBacktest, IndividualLevelTrade

backtest_bp = Blueprint('backtest', __name__, template_folder='../templates')


# ---------------------------------------------------------------------------
# Existing ML-based backtest routes
# ---------------------------------------------------------------------------

@backtest_bp.route('/')
def index():
    """Backtest index: model selector, config form, results table."""
    models = MLModel.query.order_by(MLModel.created_at.desc()).all()
    results = (
        BacktestResult.query
        .order_by(BacktestResult.created_at.desc())
        .limit(50)
        .all()
    )
    return render_template('backtest/index.html', models=models, results=results)


@backtest_bp.route('/<int:result_id>')
def detail(result_id):
    """Backtest detail: metrics, trade log, equity curve."""
    result = db.session.get(BacktestResult, result_id)
    if result is None:
        abort(404)
    model = db.session.get(MLModel, result.model_id)
    return render_template('backtest/detail.html', result=result, model=model)


# ---------------------------------------------------------------------------
# Individual Level Backtest routes
# ---------------------------------------------------------------------------

@backtest_bp.route('/individual-levels')
def individual_levels():
    """Individual level backtest dashboard."""
    backtests = (
        IndividualLevelBacktest.query
        .filter_by(status='completed')
        .order_by(IndividualLevelBacktest.win_rate.desc())
        .all()
    )
    backtests_json = [b.to_dict() for b in backtests]
    return render_template('backtest/individual_levels.html',
                           backtests=backtests, backtests_json=backtests_json)


@backtest_bp.route('/individual-levels/<int:bt_id>')
def individual_level_detail(bt_id):
    """Detail view for a single level-type backtest."""
    bt = db.session.get(IndividualLevelBacktest, bt_id)
    if bt is None:
        abort(404)
    trades = (
        IndividualLevelTrade.query
        .filter_by(backtest_id=bt_id)
        .order_by(IndividualLevelTrade.entry_time)
        .all()
    )
    return render_template('backtest/individual_level_detail.html', bt=bt, trades=trades)


# ---------------------------------------------------------------------------
# Individual Level Backtest API
# ---------------------------------------------------------------------------

@backtest_bp.route('/api/individual-levels/results')
def api_il_results():
    """JSON: all individual level backtest results."""
    backtests = IndividualLevelBacktest.query.filter_by(status='completed').all()
    return jsonify([b.to_dict() for b in backtests])


@backtest_bp.route('/api/individual-levels/<int:bt_id>/trades')
def api_il_trades(bt_id):
    """JSON: trades for a specific backtest."""
    bt = db.session.get(IndividualLevelBacktest, bt_id)
    if bt is None:
        abort(404)

    q = IndividualLevelTrade.query.filter_by(backtest_id=bt_id)

    # Optional filters
    direction = request.args.get('direction')
    if direction:
        q = q.filter_by(direction=direction.upper())
    exit_reason = request.args.get('exit_reason')
    if exit_reason:
        q = q.filter_by(exit_reason=exit_reason)
    result_filter = request.args.get('result')
    if result_filter == 'winning':
        q = q.filter(IndividualLevelTrade.pnl > 0)
    elif result_filter == 'losing':
        q = q.filter(IndividualLevelTrade.pnl <= 0)

    # Sorting
    sort = request.args.get('sort', 'entry_time')
    order = request.args.get('order', 'asc')
    sort_col = getattr(IndividualLevelTrade, sort, IndividualLevelTrade.entry_time)
    q = q.order_by(sort_col.desc() if order == 'desc' else sort_col.asc())

    limit = request.args.get('limit', 500, type=int)
    trades = q.limit(limit).all()
    return jsonify([t.to_dict() for t in trades])


@backtest_bp.route('/api/individual-levels/<int:bt_id>/equity-curve')
def api_il_equity_curve(bt_id):
    """JSON: equity curve data for charting."""
    trades = (
        IndividualLevelTrade.query
        .filter_by(backtest_id=bt_id)
        .order_by(IndividualLevelTrade.entry_time)
        .all()
    )
    curve = []
    cumulative = 0.0
    for t in trades:
        cumulative += t.pnl_pct if t.pnl_pct else 0.0
        curve.append({
            'time': t.entry_time.isoformat() if t.entry_time else None,
            'pnl_pct': t.pnl_pct,
            'cumulative_pct': cumulative * 100,
            'direction': t.direction,
            'exit_reason': t.exit_reason,
        })
    return jsonify(curve)


@backtest_bp.route('/api/individual-levels/<int:bt_id>/replay-data')
def api_il_replay(bt_id):
    """JSON: full replay data (candles + levels + trades) for chart visualization."""
    bt = db.session.get(IndividualLevelBacktest, bt_id)
    if bt is None:
        abort(404)

    trades = (
        IndividualLevelTrade.query
        .filter_by(backtest_id=bt_id)
        .order_by(IndividualLevelTrade.entry_time)
        .all()
    )

    return jsonify({
        'backtest': bt.to_dict(),
        'trades': [t.to_dict() for t in trades],
    })


@backtest_bp.route('/api/individual-levels/trade/<int:trade_id>/chart-data')
def api_il_trade_chart(trade_id):
    """JSON: candle data around a specific trade for chart visualization."""
    from pathlib import Path
    from ..services.individual_level_backtest import load_candles_csv
    import pandas as pd

    trade = db.session.get(IndividualLevelTrade, trade_id)
    if trade is None:
        abort(404)

    # Load candle data from CSV
    root = Path(__file__).resolve().parent.parent.parent
    datasets = root / 'datasets'
    candle_files = sorted(datasets.glob('ml_dataset_*.csv'))
    if not candle_files:
        return jsonify({'error': 'No dataset files'}), 404

    candles = load_candles_csv(*[str(f) for f in candle_files])

    # Get candles around the trade: 24h before entry to 24h after exit
    entry = pd.Timestamp(trade.entry_time)
    exit_t = pd.Timestamp(trade.exit_time) if trade.exit_time else entry
    margin = pd.Timedelta(hours=48)
    mask = (candles['open_time'] >= entry - margin) & (candles['open_time'] <= exit_t + margin)
    subset = candles[mask]

    chart_candles = []
    for _, row in subset.iterrows():
        chart_candles.append({
            'time': int(row['open_time'].timestamp()),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
        })

    return jsonify({
        'trade': trade.to_dict(),
        'candles': chart_candles,
        'markers': {
            'entry_time': int(entry.timestamp()),
            'entry_price': trade.entry_price,
            'exit_time': int(pd.Timestamp(trade.exit_time).timestamp()) if trade.exit_time else None,
            'exit_price': trade.exit_price,
            'stop_loss': trade.stop_loss,
            'take_profit': trade.take_profit,
            'direction': trade.direction,
            'exit_reason': trade.exit_reason,
        },
    })


@backtest_bp.route('/individual-levels/<int:bt_id>/trade/<int:trade_id>')
def individual_trade_chart(bt_id, trade_id):
    """Individual trade chart visualization page."""
    bt = db.session.get(IndividualLevelBacktest, bt_id)
    trade = db.session.get(IndividualLevelTrade, trade_id)
    if bt is None or trade is None:
        abort(404)

    # Get prev/next trade for navigation
    prev_trade = (
        IndividualLevelTrade.query
        .filter_by(backtest_id=bt_id)
        .filter(IndividualLevelTrade.entry_time < trade.entry_time)
        .order_by(IndividualLevelTrade.entry_time.desc())
        .first()
    )
    next_trade = (
        IndividualLevelTrade.query
        .filter_by(backtest_id=bt_id)
        .filter(IndividualLevelTrade.entry_time > trade.entry_time)
        .order_by(IndividualLevelTrade.entry_time.asc())
        .first()
    )

    return render_template('backtest/individual_trade_chart.html',
                           bt=bt, trade=trade,
                           prev_trade=prev_trade, next_trade=next_trade)


# ---------------------------------------------------------------------------
# Level Performance Backtest routes  (DB-backed, wick SL, configurable RR)
# ---------------------------------------------------------------------------

@backtest_bp.route('/level-performance')
def level_performance():
    """Level performance backtest dashboard."""
    # Fetch completed wick-RR backtests, newest first
    backtests = (
        IndividualLevelBacktest.query
        .filter(IndividualLevelBacktest.strategy_name.like('wick_rr_%'))
        .filter_by(status='completed')
        .order_by(IndividualLevelBacktest.created_at.desc())
        .all()
    )
    # Group into pivot: {(level_type, timeframe): {rr: record}}
    pivot: dict[tuple, dict] = {}
    rr_set: set[float] = set()
    for bt in backtests:
        try:
            rr = float(bt.parameters.get('rr', 0)) if bt.parameters else 0
        except (TypeError, ValueError):
            rr = 0.0
        key = (bt.level_type, bt.level_source_timeframe)
        pivot.setdefault(key, {})[rr] = bt
        rr_set.add(rr)
    rr_list = sorted(rr_set)
    rows = [
        {'level_type': lt, 'timeframe': tf, 'by_rr': rr_map}
        for (lt, tf), rr_map in sorted(pivot.items())
    ]
    return render_template('backtest/level_performance.html',
                           rows=rows, rr_list=rr_list)


@backtest_bp.route('/api/level-performance/run', methods=['POST'])
def api_lp_run():
    """Start a level performance backtest in a background thread."""
    from ..services.level_trade_backtest_db import run_level_trade_backtest
    from ..services import progress

    data       = request.get_json(silent=True) or {}
    exec_tf    = data.get('exec_timeframe', '1h')
    rr_ratios  = [float(r) for r in data.get('rr_ratios', [1.0, 2.0, 3.0])]
    tolerance  = float(data.get('tolerance_pct', 0.005))
    timeout    = int(data.get('timeout', 100))
    naked_only = bool(data.get('naked_only', True))

    if progress.get_state()['running']:
        return jsonify({'error': 'Another task is already running'}), 409

    progress.start()
    app = current_app._get_current_object()

    def _run():
        try:
            with app.app_context():
                def _cb(done, total, label):
                    progress.update(f'{done}/{total}', label)

                results = run_level_trade_backtest(
                    db.session,
                    exec_timeframe=exec_tf,
                    rr_ratios=rr_ratios,
                    tolerance_pct=tolerance,
                    timeout=timeout,
                    naked_only=naked_only,
                    progress_cb=_cb,
                )
                progress.set_result({'count': len(results)})
                progress.finish()
        except Exception as exc:
            progress.finish(error=str(exc))

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({'status': 'started', 'exec_timeframe': exec_tf, 'rr_ratios': rr_ratios})


@backtest_bp.route('/api/level-performance/results')
def api_lp_results():
    """JSON: all completed wick-RR backtest results."""
    backtests = (
        IndividualLevelBacktest.query
        .filter(IndividualLevelBacktest.strategy_name.like('wick_rr_%'))
        .filter_by(status='completed')
        .order_by(IndividualLevelBacktest.level_type,
                  IndividualLevelBacktest.level_source_timeframe)
        .all()
    )
    out = []
    for bt in backtests:
        d = bt.to_dict()
        try:
            d['rr'] = float(bt.parameters.get('rr', 0)) if bt.parameters else 0
        except (TypeError, ValueError):
            d['rr'] = 0.0
        out.append(d)
    return jsonify(out)


# ---------------------------------------------------------------------------
# CSV-based individual level backtest (legacy) — keep as-is
# ---------------------------------------------------------------------------

@backtest_bp.route('/api/individual-levels/run', methods=['POST'])
def api_il_run():
    """Run a new individual level backtest."""
    import os
    from pathlib import Path
    from ..services.individual_level_backtest import (
        load_candles_csv, load_levels_csv,
        run_individual_level_backtest,
        FixedPercentStrategy, ATRBasedStrategy, STRATEGIES,
    )

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON body'}), 400

    level_type = data.get('level_type', 'HTF')
    source_tf = data.get('source_timeframe', '1d')
    exec_tf = data.get('exec_timeframe', '1h')
    strategy_name = data.get('strategy', 'fixed_percent')
    params = data.get('parameters', {})
    tolerance = params.get('tolerance_pct', 0.02)
    timeout = params.get('timeout', 50)
    naked_only = params.get('naked_only', True)

    # Build strategy
    if strategy_name == 'atr_based':
        strategy = ATRBasedStrategy(
            atr_sl_mult=params.get('atr_sl_mult', 1.5),
            atr_tp_mult=params.get('atr_tp_mult', 3.0),
            timeout=timeout,
        )
    else:
        strategy = FixedPercentStrategy(
            sl_pct=params.get('sl_pct', 0.02),
            tp_pct=params.get('tp_pct', 0.04),
            timeout=timeout,
        )

    # Load CSVs
    root = Path(__file__).resolve().parent.parent.parent
    datasets = root / 'datasets'
    candle_files = sorted(datasets.glob('ml_dataset_*.csv'))
    level_files = sorted(datasets.glob('levels_dataset_*.csv'))

    if not candle_files or not level_files:
        return jsonify({'error': 'No dataset files found'}), 404

    candles = load_candles_csv(*[str(f) for f in candle_files])
    levels = load_levels_csv(*[str(f) for f in level_files])

    record, trades = run_individual_level_backtest(
        candles_df=candles,
        levels_df=levels,
        level_type=level_type,
        source_timeframe=source_tf,
        exec_timeframe=exec_tf,
        strategy=strategy,
        tolerance_pct=tolerance,
        timeout=timeout,
        naked_only=naked_only,
        db=db.session,
    )

    if record:
        return jsonify({'backtest_id': record.id, 'status': 'completed', **record.to_dict()})
    return jsonify({'error': 'No trades generated', 'status': 'empty'}), 200
