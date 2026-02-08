import json
import queue
import threading
from datetime import datetime, timezone
from flask import Blueprint, jsonify, request, Response, current_app
from ..extensions import db
from ..models import Candle, Level, Feature, MLModel, PipelineRun

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


@api_bp.route('/health')
def health():
    return jsonify({'status': 'ok'})


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

    query = Level.query
    if active_only:
        query = query.filter(Level.invalidated_at.is_(None))
    if start:
        query = query.filter(Level.created_at >= start)
    if end:
        query = query.filter(Level.created_at <= end)
    if level_type:
        query = query.filter(Level.level_type.like(f'%{level_type}%'))

    rows = query.order_by(Level.price_level).all()
    return jsonify([l.to_dict() for l in rows])


@api_bp.route('/stats')
def stats():
    candle_count = db.session.query(Candle).filter_by(timeframe='1h').count()
    level_count = db.session.query(Level).count()
    active_level_count = db.session.query(Level).filter(Level.invalidated_at.is_(None)).count()
    feature_count = db.session.query(Feature).count()
    model_count = db.session.query(MLModel).count()

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
        'latest_candle': latest_candle.open_time.isoformat() if latest_candle else None,
        'earliest_candle': earliest_candle.open_time.isoformat() if earliest_candle else None,
        'latest_run': latest_run.to_dict() if latest_run else None,
        'bullish_fractals': bullish_count,
        'bearish_fractals': bearish_count,
        'level_by_source': level_by_source,
    })


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
    horizon = data.get('horizon', 'day')
    name = data.get('name')

    app = current_app._get_current_object()

    def _work():
        with app.app_context():
            publish_sse('pipeline', {'status': 'running', 'type': 'training',
                                     'algorithm': algorithm})
            try:
                model = _train(db.session, algorithm=algorithm,
                               prediction_horizon=horizon, name=name)
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
