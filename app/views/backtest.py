"""Backtest views — run and review backtests."""
from flask import Blueprint, render_template
from ..extensions import db
from ..models import MLModel, BacktestResult

backtest_bp = Blueprint('backtest', __name__, template_folder='../templates')


@backtest_bp.route('/')
def index():
    """Backtest index: model selector, config form, results table."""
    models = MLModel.query.order_by(MLModel.created_at.desc()).all()
    results = (
        BacktestResult.query
        .order_by(BacktestResult.created_at.desc())
        .limit(50)
        .all()
    )
    return render_template('backtest/index.html', models=models, results=results)


@backtest_bp.route('/<int:result_id>')
def detail(result_id):
    """Backtest detail: metrics, trade log, equity curve."""
    result = db.session.get(BacktestResult, result_id)
    if result is None:
        from flask import abort
        abort(404)
    model = db.session.get(MLModel, result.model_id)
    return render_template('backtest/detail.html', result=result, model=model)
