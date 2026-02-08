from flask import Blueprint, render_template, request
from ..extensions import db
from ..models import Feature, Candle

features_bp = Blueprint('features', __name__)


@features_bp.route('/')
def status():
    feature_count = db.session.query(Feature).count()
    candle_count = db.session.query(Candle).filter_by(timeframe='1h').count()

    # Sample recent features
    recent = (
        db.session.query(Feature)
        .order_by(Feature.id.desc())
        .limit(20)
        .all()
    )

    return render_template(
        'features/status.html',
        feature_count=feature_count,
        candle_count=candle_count,
        recent=recent,
    )


@features_bp.route('/explore')
def explore():
    page = request.args.get('page', 1, type=int)
    per_page = 50

    pagination = (
        db.session.query(Feature)
        .order_by(Feature.id.desc())
        .paginate(page=page, per_page=per_page, error_out=False)
    )

    return render_template(
        'features/explore.html',
        features=pagination.items,
        pagination=pagination,
    )
