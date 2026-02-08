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
    ).group_by(Level.level_type).all()

    return render_template(
        'data/status.html',
        timeframes=timeframes,
        level_stats=level_stats,
    )
