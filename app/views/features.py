from flask import Blueprint, render_template
from ..extensions import db
from ..models import Feature

features_bp = Blueprint('features', __name__)


@features_bp.route('/')
def status():
    feature_count = db.session.query(Feature).count()
    return render_template('features/status.html', feature_count=feature_count)
