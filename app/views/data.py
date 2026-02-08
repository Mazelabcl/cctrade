from flask import Blueprint, render_template
from ..extensions import db
from ..models import Candle, Level, PipelineRun

data_bp = Blueprint('data', __name__)


@data_bp.route('/')
def status():
    timeframes = db.session.query(
        Candle.timeframe,
        db.func.count(Candle.id),
        db.func.min(Candle.open_time),
        db.func.max(Candle.open_time),
    ).group_by(Candle.timeframe).all()

    level_stats = db.session.query(
        Level.level_type,
        db.func.count(Level.id),
    ).group_by(Level.level_type).order_by(db.func.count(Level.id).desc()).all()

    level_by_source = db.session.query(
        Level.source,
        db.func.count(Level.id),
    ).group_by(Level.source).all()

    # Pipeline run history
    runs = (
        PipelineRun.query
        .order_by(PipelineRun.started_at.desc())
        .limit(20)
        .all()
    )

    return render_template(
        'data/status.html',
        timeframes=timeframes,
        level_stats=level_stats,
        level_by_source=level_by_source,
        runs=runs,
    )
