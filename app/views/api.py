from flask import Blueprint, jsonify, request
from ..extensions import db
from ..models import Candle, Level, Feature, PipelineRun

api_bp = Blueprint('api', __name__)


@api_bp.route('/health')
def health():
    return jsonify({'status': 'ok'})


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

    candles = query.order_by(Candle.open_time.desc()).limit(limit).all()
    return jsonify([c.to_dict() for c in reversed(candles)])


@api_bp.route('/levels')
def levels():
    start = request.args.get('start')
    end = request.args.get('end')
    active_only = request.args.get('active_only', 'true') == 'true'

    query = Level.query
    if active_only:
        query = query.filter(Level.invalidated_at.is_(None))
    if start:
        query = query.filter(Level.created_at >= start)
    if end:
        query = query.filter(Level.created_at <= end)

    levels = query.order_by(Level.price_level).all()
    return jsonify([l.to_dict() for l in levels])


@api_bp.route('/stats')
def stats():
    candle_count = db.session.query(Candle).filter_by(timeframe='1h').count()
    level_count = db.session.query(Level).count()
    feature_count = db.session.query(Feature).count()

    latest_candle = (
        Candle.query.filter_by(timeframe='1h')
        .order_by(Candle.open_time.desc())
        .first()
    )

    latest_run = (
        PipelineRun.query
        .order_by(PipelineRun.started_at.desc())
        .first()
    )

    return jsonify({
        'candle_count': candle_count,
        'level_count': level_count,
        'feature_count': feature_count,
        'latest_candle': latest_candle.open_time.isoformat() if latest_candle else None,
        'latest_run': latest_run.to_dict() if latest_run else None,
    })
