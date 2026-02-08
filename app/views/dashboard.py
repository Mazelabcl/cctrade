from flask import Blueprint, render_template
from ..extensions import db
from ..models import Candle, Level, Feature, MLModel, Prediction, PipelineRun

dashboard_bp = Blueprint('dashboard', __name__)


@dashboard_bp.route('/')
def index():
    candle_count = db.session.query(Candle).filter_by(timeframe='1h').count()
    level_count = db.session.query(Level).count()
    active_level_count = db.session.query(Level).filter(Level.invalidated_at.is_(None)).count()
    feature_count = db.session.query(Feature).count()
    model_count = db.session.query(MLModel).count()

    # Fractal counts
    bullish_count = db.session.query(Candle).filter_by(
        timeframe='1h', bullish_fractal=True).count()
    bearish_count = db.session.query(Candle).filter_by(
        timeframe='1h', bearish_fractal=True).count()

    recent_runs = (
        db.session.query(PipelineRun)
        .order_by(PipelineRun.started_at.desc())
        .limit(10)
        .all()
    )

    timeframes = db.session.query(
        Candle.timeframe,
        db.func.count(Candle.id),
        db.func.min(Candle.open_time),
        db.func.max(Candle.open_time),
    ).group_by(Candle.timeframe).all()

    # Level breakdown by source
    level_by_source = db.session.query(
        Level.source,
        db.func.count(Level.id),
    ).group_by(Level.source).all()

    # Latest candle info
    latest_candle = (
        Candle.query.filter_by(timeframe='1h')
        .order_by(Candle.open_time.desc())
        .first()
    )

    # Latest prediction
    latest_prediction = (
        Prediction.query
        .order_by(Prediction.created_at.desc())
        .first()
    )

    # Prediction accuracy stats
    total_preds = db.session.query(Prediction).count()
    verified_preds = db.session.query(Prediction).filter(
        Prediction.actual_class.isnot(None)).count()
    correct_preds = db.session.query(Prediction).filter(
        Prediction.actual_class == Prediction.predicted_class).count()

    return render_template(
        'dashboard/index.html',
        candle_count=candle_count,
        level_count=level_count,
        active_level_count=active_level_count,
        feature_count=feature_count,
        model_count=model_count,
        bullish_count=bullish_count,
        bearish_count=bearish_count,
        recent_runs=recent_runs,
        timeframes=timeframes,
        level_by_source=level_by_source,
        latest_candle=latest_candle,
        latest_prediction=latest_prediction,
        total_preds=total_preds,
        verified_preds=verified_preds,
        correct_preds=correct_preds,
    )
