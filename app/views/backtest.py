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


# ---------------------------------------------------------------------------
# Trade Explorer — visual trade browser with scoring
# ---------------------------------------------------------------------------

@backtest_bp.route('/trade-explorer')
def trade_explorer():
    """Trade Explorer: visually browse trades with chart, levels, and scoring."""
    return render_template('backtest/trade_explorer.html')


@backtest_bp.route('/api/trade-explorer/trades')
def api_trade_explorer_list():
    """JSON: filtered list of trades for the explorer."""
    from ..models import Candle, Level

    exec_tf = request.args.get('exec_tf', '4h')
    strategy = request.args.get('strategy', 'wick_rr_1.0')
    min_score = request.args.get('min_score', 0, type=float)
    level_type = request.args.get('level_type')
    result_filter = request.args.get('result')  # 'win', 'loss', or None
    offset = request.args.get('offset', 0, type=int)
    limit = request.args.get('limit', 50, type=int)

    # Build win rate cache for scoring
    from sqlalchemy import func as sa_func
    wr_results = (
        db.session.query(
            IndividualLevelBacktest.level_type,
            IndividualLevelBacktest.level_source_timeframe,
            sa_func.avg(IndividualLevelBacktest.win_rate),
        )
        .filter(
            IndividualLevelBacktest.status == 'completed',
            IndividualLevelBacktest.total_trades >= 10,
            IndividualLevelBacktest.strategy_name == 'wick_rr_1.0',
        )
        .group_by(
            IndividualLevelBacktest.level_type,
            IndividualLevelBacktest.level_source_timeframe,
        )
        .all()
    )
    wr_cache = {(lt, tf): wr / 100.0 for lt, tf, wr in wr_results}

    # Query trades
    q = (
        db.session.query(IndividualLevelTrade, IndividualLevelBacktest)
        .join(IndividualLevelBacktest)
        .filter(
            IndividualLevelBacktest.trade_execution_timeframe == exec_tf,
            IndividualLevelBacktest.status == 'completed',
            IndividualLevelBacktest.strategy_name == strategy,
            IndividualLevelTrade.exit_reason.in_(['TP_HIT', 'SL_HIT']),
        )
    )

    if level_type:
        q = q.filter(IndividualLevelBacktest.level_type == level_type)
    if result_filter == 'win':
        q = q.filter(IndividualLevelTrade.exit_reason == 'TP_HIT')
    elif result_filter == 'loss':
        q = q.filter(IndividualLevelTrade.exit_reason == 'SL_HIT')

    q = q.order_by(IndividualLevelTrade.entry_time.desc())
    all_trades = q.all()

    # Score each trade and filter
    scored = []
    for trade, bt in all_trades:
        key = (bt.level_type, bt.level_source_timeframe)
        level_wr = wr_cache.get(key, 0.35)
        level_score = level_wr * 10.0

        # Wick score (need candle data)
        wick_score = 0
        vol_score = 0
        if trade.volume_ratio:
            vol_score = min(max(trade.volume_ratio - 0.5, 0) * 0.8, 2.0)

        conf_score = min((trade.zone_confluence or 1) - 1, 3) * 1.0
        dist = trade.distance_to_level or 0
        prec_score = max(0, 2.0 - dist * 400)

        total_score = level_score + wick_score + vol_score + conf_score + prec_score

        if total_score >= min_score:
            scored.append({
                'id': trade.id,
                'entry_time': trade.entry_time.isoformat() if trade.entry_time else None,
                'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'level_price': getattr(trade, 'level_price', None) or trade.entry_price,
                'direction': trade.direction,
                'exit_reason': trade.exit_reason,
                'pnl_pct': round(trade.pnl_pct * 100, 2) if trade.pnl_pct else 0,
                'stop_loss': trade.stop_loss,
                'take_profit': trade.take_profit,
                'level_type': bt.level_type,
                'level_tf': bt.level_source_timeframe,
                'candles_held': trade.candles_held,
                'score': round(total_score, 1),
                'score_breakdown': {
                    'level': round(level_score, 1),
                    'volume': round(vol_score, 1),
                    'confluence': round(conf_score, 1),
                    'precision': round(prec_score, 1),
                },
                'volume_ratio': round(trade.volume_ratio, 2) if trade.volume_ratio else None,
                'distance_pct': round(dist * 100, 3),
                'confluence': trade.zone_confluence,
            })

    total = len(scored)
    page = scored[offset:offset + limit]

    return jsonify({
        'total': total,
        'offset': offset,
        'trades': page,
    })


@backtest_bp.route('/api/trade-explorer/<int:trade_id>/chart')
def api_trade_explorer_chart(trade_id):
    """JSON: candle + level data around a specific trade for charting."""
    from ..models import Candle, Level
    import pandas as pd

    trade = db.session.get(IndividualLevelTrade, trade_id)
    if trade is None:
        abort(404)

    bt = db.session.get(IndividualLevelBacktest, trade.backtest_id)
    exec_tf = bt.trade_execution_timeframe if bt else '4h'

    entry = trade.entry_time
    exit_t = trade.exit_time or entry

    # Load candles: 40 before entry, trade duration, 20 after exit
    from datetime import timedelta
    tf_hours = {'1h': 1, '4h': 4, '6h': 6, '8h': 8, '12h': 12, '1d': 24}
    h = tf_hours.get(exec_tf, 4)
    margin_before = timedelta(hours=h * 40)
    margin_after = timedelta(hours=h * 60)  # 60 bars after exit for TP analysis

    candles = (
        Candle.query
        .filter(
            Candle.timeframe == exec_tf,
            Candle.open_time >= entry - margin_before,
            Candle.open_time <= exit_t + margin_after,
        )
        .order_by(Candle.open_time)
        .all()
    )

    chart_candles = [{
        'time': int(c.open_time.timestamp()),
        'open': c.open,
        'high': c.high,
        'low': c.low,
        'close': c.close,
    } for c in candles]

    # Load nearby levels (D/W/M, near the trade price, naked at entry time)
    # NOTE: PrevSession/VP levels have first_touched_at always NULL (bug).
    # For PrevSession, only show the MOST RECENT per (type, timeframe) — "mobile" approach.
    # For structural levels (Fractal/HTF/CC), show all naked within range.
    price = trade.entry_price
    price_range = price * 0.05  # 5% around entry

    naked_filter = [
        Level.timeframe.in_(['daily', 'weekly', 'monthly']),
        Level.price_level.between(price - price_range, price + price_range),
        Level.created_at <= entry,
        db.or_(
            Level.first_touched_at.is_(None),
            Level.first_touched_at > entry,
        ),
    ]

    # Structural levels: Fractal, HTF, Fib_CC — all naked ones
    structural_types = ['Fractal_support', 'Fractal_resistance', 'HTF_level', 'Fib_CC']
    structural_levels = (
        Level.query
        .filter(*naked_filter, Level.level_type.in_(structural_types))
        .order_by(db.func.abs(Level.price_level - price))
        .limit(50)
        .all()
    )

    # PrevSession/VP: only the MOST RECENT per (type, timeframe) — mobile levels
    # This prevents showing 20+ PrevSession_VWAP from different days
    from sqlalchemy import func as sa_func
    prevsession_vp_types = [
        'PrevSession_High', 'PrevSession_Low', 'PrevSession_EQ',
        'PrevSession_25', 'PrevSession_75', 'PrevSession_VWAP',
        'PrevSession_VP_POC', 'PrevSession_VP_VAH', 'PrevSession_VP_VAL',
        'VP_POC', 'VP_VAH', 'VP_VAL',
    ]

    # Subquery: max created_at per (level_type, timeframe) before entry
    max_created_sq = (
        db.session.query(
            Level.level_type,
            Level.timeframe,
            sa_func.max(Level.created_at).label('max_created'),
        )
        .filter(
            Level.level_type.in_(prevsession_vp_types),
            Level.timeframe.in_(['daily', 'weekly', 'monthly']),
            Level.created_at <= entry,
        )
        .group_by(Level.level_type, Level.timeframe)
        .subquery()
    )

    mobile_levels = (
        Level.query
        .join(max_created_sq, db.and_(
            Level.level_type == max_created_sq.c.level_type,
            Level.timeframe == max_created_sq.c.timeframe,
            Level.created_at == max_created_sq.c.max_created,
        ))
        .filter(
            Level.price_level.between(price - price_range, price + price_range),
        )
        .order_by(db.func.abs(Level.price_level - price))
        .all()
    )

    # Igor fibs (show naked only, limited)
    igor_types = ['Fib_0.25', 'Fib_0.50', 'Fib_0.75']
    igor_levels = (
        Level.query
        .filter(*naked_filter, Level.level_type.in_(igor_types))
        .order_by(db.func.abs(Level.price_level - price))
        .limit(20)
        .all()
    )

    nearby_levels = structural_levels + mobile_levels + igor_levels

    chart_levels = [{
        'price': l.price_level,
        'type': l.level_type,
        'tf': l.timeframe,
        'created': int(l.created_at.timestamp()) if l.created_at else None,
        'touched': int(l.first_touched_at.timestamp()) if l.first_touched_at else None,
        'role': 'resistance' if l.price_level > trade.entry_price else 'support',
    } for l in nearby_levels]

    # Resolve level price: trade.level_price → Level table via level_id → entry_price
    level_price = getattr(trade, 'level_price', None)
    if not level_price and trade.level_id:
        level_obj = db.session.get(Level, trade.level_id)
        if level_obj:
            level_price = level_obj.price_level
    if not level_price:
        level_price = trade.entry_price

    # Compute analysis zone — directional: support expands DOWN, resistance UP
    zone_width = 0.015  # 1.5%
    if trade.direction == 'LONG':
        # Support zone: level down to level*(1-1.5%)
        zone_inner = level_price
        zone_outer = level_price * (1 - zone_width)
    else:
        # Resistance zone: level up to level*(1+1.5%)
        zone_inner = level_price
        zone_outer = level_price * (1 + zone_width)

    # Count levels in the zone — use the already-loaded nearby_levels
    zone_lo = min(zone_inner, zone_outer)
    zone_hi = max(zone_inner, zone_outer)
    zone_in_range = [l for l in nearby_levels if zone_lo <= l.price_level <= zone_hi]
    zone_level_count = len(zone_in_range)
    structural_set = {'Fractal_support', 'Fractal_resistance', 'HTF_level', 'Fib_CC'}
    zone_structural = sum(1 for l in zone_in_range if l.level_type in structural_set)
    zone_prevsession = sum(1 for l in zone_in_range if l.level_type.startswith('PrevSession'))
    zone_vp = sum(1 for l in zone_in_range if l.level_type.startswith('VP_'))

    # Entry/exit condition explanation
    risk_pct = abs(trade.entry_price - trade.stop_loss) / trade.entry_price * 100
    strategy = bt.strategy_name if bt else 'wick_rr_1.0'
    rr_str = strategy.replace('wick_rr_', '') if strategy else '1.0'

    entry_explanation = (
        f"Wick touched {bt.level_type} ({bt.level_source_timeframe}) at "
        f"${level_price:,.0f}. "
        f"SL = entry candle wick {'low' if trade.direction == 'LONG' else 'high'} "
        f"+ 0.1% buffer (${trade.stop_loss:,.0f}, risk {risk_pct:.2f}%). "
        f"TP = {rr_str}:1 RR (${trade.take_profit:,.0f})."
    ) if bt else ''

    exit_explanation = (
        f"{'TP hit' if trade.exit_reason == 'TP_HIT' else 'SL hit'} "
        f"at ${trade.exit_price:,.0f} after {trade.candles_held or '?'} candles."
    )

    return jsonify({
        'candles': chart_candles,
        'levels': chart_levels,
        'trade': {
            'entry_time': int(entry.timestamp()),
            'entry_price': trade.entry_price,
            'exit_time': int(exit_t.timestamp()),
            'exit_price': trade.exit_price,
            'stop_loss': trade.stop_loss,
            'take_profit': trade.take_profit,
            'direction': trade.direction,
            'exit_reason': trade.exit_reason,
            'level_price': level_price,
            'level_type': bt.level_type if bt else None,
            'level_tf': bt.level_source_timeframe if bt else None,
            'pnl_pct': round(trade.pnl_pct * 100, 2) if trade.pnl_pct else 0,
            'zone_inner': zone_inner,
            'zone_outer': zone_outer,
            'zone_level_count': zone_level_count,
            'zone_structural': zone_structural,
            'zone_prevsession': zone_prevsession,
            'zone_vp': zone_vp,
            'entry_explanation': entry_explanation,
            'exit_explanation': exit_explanation,
            'strategy': strategy,
            'annotations': trade.metadata_json if trade.metadata_json else {},
        },
    })


@backtest_bp.route('/api/trade-explorer/<int:trade_id>/annotations', methods=['POST'])
def api_trade_explorer_save_annotations(trade_id):
    """Save TP annotation targets for a trade."""
    trade = db.session.get(IndividualLevelTrade, trade_id)
    if trade is None:
        abort(404)

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON body'}), 400

    if not trade.metadata_json:
        trade.metadata_json = {}
    trade.metadata_json['tp_targets'] = data.get('tp_targets', [])
    db.session.commit()

    return jsonify({'status': 'saved', 'trade_id': trade_id})


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
