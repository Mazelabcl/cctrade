from flask import Blueprint, render_template
from ..extensions import db
from ..models import Candle, Level, PipelineRun

dashboard_bp = Blueprint('dashboard', __name__)


@dashboard_bp.route('/')
def index():
    candle_count = db.session.query(Candle).filter_by(timeframe='1h').count()
    level_count = db.session.query(Level).count()
    active_level_count = db.session.query(Level).filter(Level.invalidated_at.is_(None)).count()
    recent_runs = (
        db.session.query(PipelineRun)
        .order_by(PipelineRun.started_at.desc())
        .limit(10)
        .all()
    )

    # Get timeframe coverage
    timeframes = db.session.query(
        Candle.timeframe,
        db.func.count(Candle.id),
        db.func.min(Candle.open_time),
        db.func.max(Candle.open_time),
    ).group_by(Candle.timeframe).all()

    return render_template(
        'dashboard/index.html',
        candle_count=candle_count,
        level_count=level_count,
        active_level_count=active_level_count,
        recent_runs=recent_runs,
        timeframes=timeframes,
    )
