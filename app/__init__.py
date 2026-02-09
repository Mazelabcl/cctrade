import logging
import os
from flask import Flask
from .config import config_by_name
from .extensions import db, scheduler


def create_app(config_name=None):
    if config_name is None:
        config_name = os.getenv('FLASK_ENV', 'development')

    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(config_by_name[config_name])

    os.makedirs(app.instance_path, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
    )

    db.init_app(app)
    scheduler.init_app(app)

    from .models import candle, level, feature, ml_model, prediction, pipeline_run, backtest_result  # noqa: F401

    from .views.dashboard import dashboard_bp
    from .views.data import data_bp
    from .views.charts import charts_bp
    from .views.features import features_bp
    from .views.models import models_bp
    from .views.api import api_bp
    from .views.backtest import backtest_bp

    app.register_blueprint(dashboard_bp)
    app.register_blueprint(data_bp, url_prefix='/data')
    app.register_blueprint(charts_bp, url_prefix='/charts')
    app.register_blueprint(features_bp, url_prefix='/features')
    app.register_blueprint(models_bp, url_prefix='/models')
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(backtest_bp, url_prefix='/backtest')

    with app.app_context():
        db.create_all()

    # Initialize scheduler (skipped in testing)
    from .tasks.scheduler import init_scheduler
    init_scheduler(app)

    return app
