from flask import Blueprint, render_template

charts_bp = Blueprint('charts', __name__)


@charts_bp.route('/')
def price():
    return render_template('charts/price.html')
