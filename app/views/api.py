import json
import logging
import queue
import threading
from datetime import datetime, timezone
from flask import Blueprint, jsonify, request, Response, current_app
from ..extensions import db
from ..models import Candle, Feature, Level, MLModel, Prediction, PipelineRun, BacktestResult

logger = logging.getLogger(__name__)

api_bp = Blueprint('api', __name__)

# SSE subscribers
_sse_subscribers: list[queue.Queue] = []
_sse_lock = threading.Lock()


def publish_sse(event: str, data: dict):
    """Publish an event to all SSE subscribers."""
    msg = f"event: {event}\ndata: {json.dumps(data)}\n\n"
    with _sse_lock:
        dead = []
        for q in _sse_subscribers:
            try:
                q.put_nowait(msg)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _sse_subscribers.remove(q)


@api_bp.route('/foundation-progress')
def foundation_progress():
    """Return real-time foundation pipeline progress (in-memory, no DB)."""
    from ..services.progress import get_state
    return jsonify(get_state())


@api_bp.route('/health')
def health():
    candle_count = db.session.query(Candle).filter_by(timeframe='1h').count()
    model_count = db.session.query(MLModel).count()
    latest_candle = (
        Candle.query.filter_by(timeframe='1h')
        .order_by(Candle.open_time.desc()).first()
    )
    return jsonify({
        'status': 'ok',
        'candles': candle_count,
        'models': model_count,
        'latest_candle': latest_candle.open_time.isoformat() if latest_candle else None,
    })


@api_bp.route('/stream')
def stream():
    """SSE endpoint for real-time pipeline status updates."""
    def generate():
        q = queue.Queue(maxsize=50)
        with _sse_lock:
            _sse_subscribers.append(q)
        try:
            # Send initial heartbeat
            yield "event: connected\ndata: {}\n\n"
            while True:
                try:
                    msg = q.get(timeout=30)
                    yield msg
                except queue.Empty:
                    yield ": heartbeat\n\n"
        except GeneratorExit:
            pass
        finally:
            with _sse_lock:
                if q in _sse_subscribers:
                    _sse_subscribers.remove(q)

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@api_bp.route('/candles')
def candles():
    tf = request.args.get('tf', '1h')
    start = request.args.get('start')
    end = request.args.get('end')
    limit = request.args.get('limit', 500, type=int)

    query = Candle.query.filter_by(timeframe=tf)
    if start:
        query = query.filter(Candle.open_time >= start)
    if end:
        query = query.filter(Candle.open_time <= end)

    rows = query.order_by(Candle.open_time.desc()).limit(limit).all()
    return jsonify([c.to_dict() for c in reversed(rows)])


@api_bp.route('/levels')
def levels():
    start = request.args.get('start')
    end = request.args.get('end')
    active_only = request.args.get('active_only', 'true') == 'true'
    level_type = request.args.get('type')
    timeframe = request.args.get('timeframe')    # comma-separated: daily,weekly
    source = request.args.get('source')          # comma-separated: htf,fibonacci
    naked_only = request.args.get('naked_only', 'false') == 'true'

    query = Level.query
    if active_only:
        query = query.filter(Level.invalidated_at.is_(None))
    if start:
        query = query.filter(Level.created_at >= start)
    if end:
        query = query.filter(Level.created_at <= end)
    if level_type:
        query = query.filter(Level.level_type.like(f'%{level_type}%'))
    if timeframe:
        tfs = [t.strip() for t in timeframe.split(',') if t.strip()]
        if tfs:
            query = query.filter(Level.timeframe.in_(tfs))
    if source:
        sources = [s.strip() for s in source.split(',') if s.strip()]
        if sources:
            query = query.filter(Level.source.in_(sources))
    if naked_only:
        query = query.filter(
            Level.support_touches == 0,
            Level.resistance_touches == 0,
        )

    rows = query.order_by(Level.price_level).all()
    return jsonify([l.to_dict() for l in rows])


@api_bp.route('/predictions')
def predictions_api():
    """Get predictions with optional filters."""
    model_id = request.args.get('model_id', type=int)
    limit = request.args.get('limit', 100, type=int)

    query = Prediction.query
    if model_id:
        query = query.filter_by(model_id=model_id)
    rows = query.order_by(Prediction.created_at.desc()).limit(limit).all()
    return jsonify([p.to_dict() for p in rows])


@api_bp.route('/predictions/overlay')
def predictions_overlay():
    """Get predictions formatted for chart overlay markers."""
    model_id = request.args.get('model_id', type=int)
    limit = request.args.get('limit', 500, type=int)

    query = db.session.query(Prediction, Candle).join(
        Candle, Prediction.candle_id == Candle.id
    )
    if model_id:
        query = query.filter(Prediction.model_id == model_id)

    rows = query.order_by(Candle.open_time.desc()).limit(limit).all()
    markers = []
    for pred, candle in reversed(rows):
        if pred.predicted_class == 0:
            continue
        markers.append({
            'time': candle.open_time.isoformat(),
            'predicted_class': pred.predicted_class,
            'confidence': pred.confidence,
            'actual_class': pred.actual_class,
        })
    return jsonify(markers)


@api_bp.route('/stats')
def stats():
    candle_count = db.session.query(Candle).filter_by(timeframe='1h').count()
    level_count = db.session.query(Level).count()
    active_level_count = db.session.query(Level).filter(Level.invalidated_at.is_(None)).count()
    feature_count = db.session.query(Feature).count()
    model_count = db.session.query(MLModel).count()
    prediction_count = db.session.query(Prediction).count()

    latest_candle = (
        Candle.query.filter_by(timeframe='1h')
        .order_by(Candle.open_time.desc())
        .first()
    )
    earliest_candle = (
        Candle.query.filter_by(timeframe='1h')
        .order_by(Candle.open_time)
        .first()
    )

    latest_run = (
        PipelineRun.query
        .order_by(PipelineRun.started_at.desc())
        .first()
    )

    # Fractal counts
    bullish_count = db.session.query(Candle).filter_by(
        timeframe='1h', bullish_fractal=True).count()
    bearish_count = db.session.query(Candle).filter_by(
        timeframe='1h', bearish_fractal=True).count()

    # Level breakdown by source
    level_by_source = dict(
        db.session.query(Level.source, db.func.count(Level.id))
        .group_by(Level.source).all()
    )

    return jsonify({
        'candle_count': candle_count,
        'level_count': level_count,
        'active_level_count': active_level_count,
        'feature_count': feature_count,
        'model_count': model_count,
        'prediction_count': prediction_count,
        'latest_candle': latest_candle.open_time.isoformat() if latest_candle else None,
        'earliest_candle': earliest_candle.open_time.isoformat() if earliest_candle else None,
        'latest_run': latest_run.to_dict() if latest_run else None,
        'bullish_fractals': bullish_count,
        'bearish_fractals': bearish_count,
        'level_by_source': level_by_source,
    })


@api_bp.route('/fetch-data', methods=['POST'])
def fetch_data():
    """Trigger data fetching from Binance in a background thread."""
    from ..tasks.data_sync import sync_candle_data

    app = current_app._get_current_object()

    def _work():
        with app.app_context():
            publish_sse('pipeline', {'status': 'running', 'type': 'data_fetch'})
            try:
                count = sync_candle_data()
                publish_sse('pipeline', {'status': 'completed', 'type': 'data_fetch',
                                         'new_candles': count})
            except Exception as e:
                publish_sse('pipeline', {'status': 'failed', 'type': 'data_fetch',
                                         'error': str(e)})

    thread = threading.Thread(target=_work, daemon=True)
    thread.start()
    return jsonify({'status': 'started', 'message': 'Data fetch started'})


@api_bp.route('/run-indicators', methods=['POST'])
def run_indicators():
    """Trigger indicator computation in a background thread."""
    from ..services.indicators import run_indicators as _run

    app = current_app._get_current_object()

    def _work():
        with app.app_context():
            publish_sse('pipeline', {'status': 'running', 'type': 'indicators'})
            try:
                result = _run(db.session)
                publish_sse('pipeline', {'status': 'completed', 'type': 'indicators', **result})
            except Exception as e:
                publish_sse('pipeline', {'status': 'failed', 'type': 'indicators', 'error': str(e)})

    thread = threading.Thread(target=_work, daemon=True)
    thread.start()
    return jsonify({'status': 'started', 'message': 'Indicator pipeline started'})


@api_bp.route('/compute-features', methods=['POST'])
def compute_features():
    """Trigger feature computation in background."""
    from ..services.feature_engine import compute_features as _compute

    app = current_app._get_current_object()

    def _work():
        with app.app_context():
            publish_sse('pipeline', {'status': 'running', 'type': 'features'})
            try:
                count = _compute(db.session)
                publish_sse('pipeline', {'status': 'completed', 'type': 'features',
                                         'features_computed': count})
            except Exception as e:
                publish_sse('pipeline', {'status': 'failed', 'type': 'features',
                                         'error': str(e)})

    thread = threading.Thread(target=_work, daemon=True)
    thread.start()
    return jsonify({'status': 'started', 'message': 'Feature computation started'})


@api_bp.route('/train-model', methods=['POST'])
def train_model_endpoint():
    """Trigger model training in background."""
    from ..services.ml_trainer import train_model as _train

    data = request.get_json() or {}
    algorithm = data.get('algorithm', 'random_forest')
    target = data.get('target', 'target_bullish')
    name = data.get('name')

    app = current_app._get_current_object()

    def _work():
        with app.app_context():
            publish_sse('pipeline', {'status': 'running', 'type': 'training',
                                     'algorithm': algorithm})
            try:
                model = _train(db.session, algorithm=algorithm,
                               target_column=target, name=name)
                publish_sse('pipeline', {
                    'status': 'completed', 'type': 'training',
                    'model_id': model.id, 'accuracy': model.accuracy,
                    'f1': model.f1_macro,
                })
            except Exception as e:
                publish_sse('pipeline', {'status': 'failed', 'type': 'training',
                                         'error': str(e)})

    thread = threading.Thread(target=_work, daemon=True)
    thread.start()
    return jsonify({'status': 'started', 'message': f'Training {algorithm} model...'})


@api_bp.route('/predict', methods=['POST'])
def predict_endpoint():
    """Run predictions with a trained model."""
    from ..services.ml_predictor import predict as _predict

    data = request.get_json() or {}
    model_id = data.get('model_id')

    try:
        results = _predict(db.session, model_id=model_id)
        return jsonify({'status': 'ok', 'predictions': len(results)})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 400


@api_bp.route('/run-foundation', methods=['POST'])
def run_foundation():
    """Trigger the foundation pipeline (fetch + multi-TF indicators + touch tracking)."""
    from ..tasks.pipeline_runner import run_foundation_pipeline

    app = current_app._get_current_object()

    def _work():
        with app.app_context():
            run_foundation_pipeline(app)

    thread = threading.Thread(target=_work, daemon=True)
    thread.start()
    return jsonify({'status': 'started', 'message': 'Foundation pipeline started'})


@api_bp.route('/foundation/preview')
def foundation_preview():
    """Return what data exists and what the pipeline would do (read-only)."""
    from ..models.setting import get_setting
    from ..tasks.pipeline_runner import _read_foundation_config

    cfg = _read_foundation_config()

    # Existing candle counts per timeframe
    existing_candles = {}
    for tf in cfg['fetch_tfs']:
        count = db.session.query(Candle).filter_by(symbol='BTCUSDT', timeframe=tf).count()
        earliest = db.session.query(Candle.open_time).filter_by(
            symbol='BTCUSDT', timeframe=tf
        ).order_by(Candle.open_time).first()
        latest = db.session.query(Candle.open_time).filter_by(
            symbol='BTCUSDT', timeframe=tf
        ).order_by(Candle.open_time.desc()).first()
        existing_candles[tf] = {
            'count': count,
            'from': str(earliest[0]) if earliest else None,
            'to': str(latest[0]) if latest else None,
        }

    # Existing level counts by type
    from sqlalchemy import func
    level_counts = dict(
        db.session.query(Level.level_type, func.count(Level.id))
        .group_by(Level.level_type).all()
    )

    return jsonify({
        'date_range': {'start': cfg['start_date'], 'end': cfg['end_date']},
        'train_test_cutoff': get_setting('train_test_cutoff', '2024-06-01'),
        'has_api_keys': bool(cfg['api_key'] and cfg['api_secret']),
        'fetch_timeframes': cfg['fetch_tfs'],
        'level_config': {
            'htf': cfg['htf_tfs'],
            'fractal': cfg['fractal_tfs'],
            'fibonacci': cfg['fib_tfs'],
            'vp': cfg['vp_tfs'],
        },
        'existing_candles': existing_candles,
        'existing_levels': level_counts,
    })


@api_bp.route('/foundation/fetch', methods=['POST'])
def foundation_fetch():
    """Step 1: Fetch candles only."""
    from ..tasks.pipeline_runner import run_foundation_fetch

    app = current_app._get_current_object()
    thread = threading.Thread(target=run_foundation_fetch, args=(app,), daemon=True)
    thread.start()
    return jsonify({'status': 'started', 'step': 'fetch'})


@api_bp.route('/foundation/detect-levels', methods=['POST'])
def foundation_detect_levels():
    """Step 2: Run level detection (fractal, HTF, Fibonacci, VP)."""
    from ..tasks.pipeline_runner import run_foundation_levels

    app = current_app._get_current_object()
    thread = threading.Thread(target=run_foundation_levels, args=(app,), daemon=True)
    thread.start()
    return jsonify({'status': 'started', 'step': 'detect-levels'})


@api_bp.route('/foundation/touch-tracking', methods=['POST'])
def foundation_touch_tracking():
    """Step 3: Reset and re-run touch tracking."""
    from ..tasks.pipeline_runner import run_foundation_touches

    app = current_app._get_current_object()
    thread = threading.Thread(target=run_foundation_touches, args=(app,), daemon=True)
    thread.start()
    return jsonify({'status': 'started', 'step': 'touch-tracking'})


@api_bp.route('/regenerate-fibs', methods=['POST'])
def regenerate_fibs():
    """Delete old Fibonacci levels and regenerate with CC/Igor split types."""
    from ..services.indicators import calculate_fibonacci_levels, _add_levels, _load_candle_df, _tf_label
    from ..services.level_tracker import run_touch_tracking
    from ..models.setting import get_setting

    app = current_app._get_current_object()

    def _work():
        with app.app_context():
            # Delete all existing Fibonacci levels
            deleted = db.session.query(Level).filter(Level.source == 'fibonacci').delete()
            db.session.commit()
            logger.info("Regenerate fibs: deleted %d old levels", deleted)

            # Regenerate for configured timeframes
            fib_tfs = get_setting('fibonacci_timeframes', '1d,1w').split(',')
            fib_tfs = [t for t in fib_tfs if t]

            for tf in fib_tfs:
                candles = _load_candle_df(db.session, 'BTCUSDT', tf)
                if candles.empty:
                    continue
                fibs = calculate_fibonacci_levels(candles, _tf_label(tf))
                added = _add_levels(db.session, fibs, skip_duplicates=True)
                logger.info("Regenerate fibs [%s]: %d new levels", tf, added)

            db.session.commit()

            # Reset touches on ALL levels and re-run touch tracking
            db.session.query(Level).update({
                Level.support_touches: 0,
                Level.resistance_touches: 0,
                Level.invalidated_at: None,
                Level.first_touched_at: None,
            })
            db.session.commit()

            fetch_tfs = get_setting('fetch_timeframes', '1d,1w,1M').split(',')
            touch_tf = '1d' if '1d' in fetch_tfs else ('1w' if '1w' in fetch_tfs else '1M')
            tt = run_touch_tracking(db.session, timeframe=touch_tf, symbol='BTCUSDT')
            logger.info("Regenerate fibs: touch tracking done — %s", tt)

    thread = threading.Thread(target=_work, daemon=True)
    thread.start()
    return jsonify({'status': 'started', 'message': 'Fibonacci regeneration started in background'})


@api_bp.route('/foundation-status')
def foundation_status():
    """Return foundation data coverage and level counts."""
    from ..models.setting import get_setting

    # Data coverage per timeframe
    data_coverage = {}
    timeframes = db.session.query(Candle.timeframe).distinct().all()
    for (tf,) in timeframes:
        count = db.session.query(Candle).filter_by(timeframe=tf).count()
        earliest = (
            db.session.query(Candle.open_time)
            .filter_by(timeframe=tf)
            .order_by(Candle.open_time)
            .first()
        )
        latest = (
            db.session.query(Candle.open_time)
            .filter_by(timeframe=tf)
            .order_by(Candle.open_time.desc())
            .first()
        )
        data_coverage[tf] = {
            'count': count,
            'from': earliest[0].isoformat() if earliest else None,
            'to': latest[0].isoformat() if latest else None,
        }

    # Levels by type and timeframe
    level_rows = (
        db.session.query(Level.source, Level.timeframe, db.func.count(Level.id))
        .filter(Level.invalidated_at.is_(None))
        .group_by(Level.source, Level.timeframe)
        .all()
    )
    levels_by_type_tf = {}
    for source, tf, count in level_rows:
        if source not in levels_by_type_tf:
            levels_by_type_tf[source] = {}
        levels_by_type_tf[source][tf] = count

    # Train/test split info
    cutoff = get_setting('train_test_cutoff', '2024-06-01')
    train_levels = db.session.query(Level).filter(
        Level.created_at < cutoff,
        Level.invalidated_at.is_(None),
    ).count()
    test_levels = db.session.query(Level).filter(
        Level.created_at >= cutoff,
        Level.invalidated_at.is_(None),
    ).count()

    # Check if foundation pipeline is running
    pipeline_running = (
        db.session.query(PipelineRun)
        .filter_by(pipeline_type='foundation', status='running')
        .first()
    ) is not None

    return jsonify({
        'data_coverage': data_coverage,
        'levels_by_type_tf': levels_by_type_tf,
        'train_test_cutoff': cutoff,
        'train_levels': train_levels,
        'test_levels': test_levels,
        'pipeline_running': pipeline_running,
    })


@api_bp.route('/run-pipeline', methods=['POST'])
def run_pipeline():
    """Trigger the full automated pipeline in background."""
    from ..tasks.pipeline_runner import run_full_pipeline

    app = current_app._get_current_object()

    def _work():
        with app.app_context():
            run_full_pipeline(app)

    thread = threading.Thread(target=_work, daemon=True)
    thread.start()
    return jsonify({'status': 'started', 'message': 'Full pipeline started'})


@api_bp.route('/backfill-actuals', methods=['POST'])
def backfill_actuals():
    """Backfill actual_class for predictions where outcome is known."""
    from ..tasks.accuracy_tracker import backfill_actuals as _backfill

    try:
        count = _backfill(db.session)
        return jsonify({'status': 'ok', 'updated': count})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 400


@api_bp.route('/run-backtest', methods=['POST'])
def run_backtest_endpoint():
    """Trigger a backtest in a background thread."""
    from ..services.backtest.runner import run_backtest as _run_bt

    data = request.get_json() or {}
    model_id = data.get('model_id')
    if not model_id:
        return jsonify({'status': 'error', 'error': 'model_id required'}), 400

    cash = data.get('cash', 100000)
    commission = data.get('commission', 0.001)
    risk = data.get('risk_per_trade', 0.02)
    confidence = data.get('confidence_threshold', 0.55)

    app = current_app._get_current_object()

    def _work():
        with app.app_context():
            publish_sse('pipeline', {'status': 'running', 'type': 'backtest',
                                     'model_id': model_id})
            try:
                result = _run_bt(
                    db.session, model_id=model_id,
                    initial_cash=cash, commission=commission,
                    risk_per_trade=risk, confidence_threshold=confidence,
                )
                publish_sse('pipeline', {
                    'status': 'completed', 'type': 'backtest',
                    'result_id': result.id,
                    'total_return': result.total_return,
                    'total_trades': result.total_trades,
                })
            except Exception as e:
                publish_sse('pipeline', {'status': 'failed', 'type': 'backtest',
                                         'error': str(e)})

    thread = threading.Thread(target=_work, daemon=True)
    thread.start()
    return jsonify({'status': 'started', 'message': f'Backtest started for model {model_id}'})


@api_bp.route('/backtest-results')
def backtest_results():
    """Get backtest results, optionally filtered by model."""
    model_id = request.args.get('model_id', type=int)
    limit = request.args.get('limit', 20, type=int)

    query = BacktestResult.query
    if model_id:
        query = query.filter_by(model_id=model_id)
    rows = query.order_by(BacktestResult.created_at.desc()).limit(limit).all()
    return jsonify([r.to_dict() for r in rows])


@api_bp.route('/toggle-live-sync', methods=['POST'])
def toggle_live_sync():
    """Start or stop the live data sync scheduler job."""
    from ..models.setting import get_setting, set_setting
    from ..tasks.scheduler import start_live_sync, stop_live_sync, is_live_sync_active

    data = request.get_json() or {}
    enabled = data.get('enabled', True)
    app = current_app._get_current_object()

    if enabled:
        interval = int(get_setting('live_sync_interval_minutes', '5'))
        set_setting('live_sync_enabled', 'true')
        start_live_sync(app, interval)
        return jsonify({'status': 'ok', 'message': f'Live sync started (every {interval}m)'})
    else:
        set_setting('live_sync_enabled', 'false')
        stop_live_sync()
        return jsonify({'status': 'ok', 'message': 'Live sync stopped'})


@api_bp.route('/sync-status')
def sync_status():
    """Return current live sync state and recent sync history."""
    from ..models.setting import get_setting
    from ..tasks.scheduler import is_live_sync_active
    from ..tasks.data_sync import get_last_fetch_time

    enabled = get_setting('live_sync_enabled', 'false') == 'true'
    interval = get_setting('live_sync_interval_minutes', '5')
    job_active = is_live_sync_active()
    last_fetch = get_last_fetch_time()

    # Recent sync history from PipelineRun
    recent_syncs = (
        PipelineRun.query
        .filter_by(pipeline_type='data_fetch')
        .order_by(PipelineRun.started_at.desc())
        .limit(15)
        .all()
    )

    return jsonify({
        'enabled': enabled,
        'interval_minutes': int(interval),
        'job_active': job_active,
        'last_fetch': last_fetch.isoformat() if last_fetch else None,
        'recent_syncs': [r.to_dict() for r in recent_syncs],
    })


@api_bp.route('/latest-signal')
def latest_signal():
    """Get the latest prediction with trade signal for the most recent candle."""
    from ..services.signal_generator import generate_signal

    # Find the most recent prediction
    latest_pred = (
        Prediction.query
        .join(Candle, Prediction.candle_id == Candle.id)
        .order_by(Candle.open_time.desc())
        .first()
    )

    if not latest_pred:
        return jsonify({'status': 'no_data', 'message': 'No predictions available'})

    candle = db.session.get(Candle, latest_pred.candle_id)
    feature = Feature.query.filter_by(candle_id=latest_pred.candle_id).first()

    # Find nearest support/resistance from active levels
    active_levels = Level.query.filter(Level.invalidated_at.is_(None)).all()
    nearest_support = None
    nearest_resistance = None
    for lvl in sorted(active_levels, key=lambda l: l.price_level, reverse=True):
        if lvl.price_level <= candle.close and nearest_support is None:
            nearest_support = lvl.price_level
    for lvl in sorted(active_levels, key=lambda l: l.price_level):
        if lvl.price_level > candle.close and nearest_resistance is None:
            nearest_resistance = lvl.price_level

    signal = generate_signal(latest_pred, candle, feature, nearest_support, nearest_resistance)

    return jsonify({
        'status': 'ok',
        'candle_time': candle.open_time.isoformat(),
        'candle_close': candle.close,
        'predicted_class': latest_pred.predicted_class,
        'prob_bullish': latest_pred.prob_bullish,
        'prob_bearish': latest_pred.prob_bearish,
        'prob_no_fractal': latest_pred.prob_no_fractal,
        'confidence': latest_pred.confidence,
        'signal': signal.signal,
        'entry_price': signal.entry_price,
        'stop_loss': signal.stop_loss,
        'take_profit': signal.take_profit,
        'atr': signal.atr,
        'reason': signal.reason,
    })


@api_bp.route('/pipeline-runs')
def pipeline_runs():
    limit = request.args.get('limit', 20, type=int)
    runs = (
        PipelineRun.query
        .order_by(PipelineRun.started_at.desc())
        .limit(limit)
        .all()
    )
    return jsonify([r.to_dict() for r in runs])
