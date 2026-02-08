from flask import Blueprint, render_template
from ..extensions import db
from ..models import MLModel, Prediction

models_bp = Blueprint('models', __name__)


@models_bp.route('/')
def index():
    models = db.session.query(MLModel).order_by(MLModel.created_at.desc()).all()
    return render_template('models/train.html', models=models)


@models_bp.route('/<int:model_id>')
def detail(model_id):
    model = db.session.query(MLModel).get_or_404(model_id)
    pred_count = db.session.query(Prediction).filter_by(model_id=model_id).count()
    return render_template('models/evaluate.html', model=model, pred_count=pred_count)


@models_bp.route('/predictions')
def predictions():
    recent = (
        db.session.query(Prediction)
        .order_by(Prediction.created_at.desc())
        .limit(100)
        .all()
    )
    return render_template('models/predict.html', predictions=recent)
