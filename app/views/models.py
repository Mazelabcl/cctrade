from flask import Blueprint, render_template
from ..extensions import db
from ..models import MLModel

models_bp = Blueprint('models', __name__)


@models_bp.route('/')
def index():
    models = db.session.query(MLModel).order_by(MLModel.created_at.desc()).all()
    return render_template('models/train.html', models=models)
