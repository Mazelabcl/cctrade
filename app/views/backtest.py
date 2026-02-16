"""Backtest views — run and review backtests."""
from flask import Blueprint, render_template, jsonify, request, abort
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
