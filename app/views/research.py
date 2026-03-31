"""AutoResearch Dashboard — visual hub for all research findings and roadmap."""
from flask import Blueprint, render_template, jsonify
from ..extensions import db
from sqlalchemy import text
import json
from pathlib import Path

research_bp = Blueprint('research', __name__, template_folder='../templates')

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / 'scripts' / 'autoresearch' / 'results'


@research_bp.route('/')
def index():
    """Research dashboard."""
    # Live data inventory from DB
    candle_stats = db.session.execute(text("""
        SELECT timeframe, COUNT(*) as cnt,
               MIN(open_time) as first_date, MAX(open_time) as last_date
        FROM candles
        GROUP BY timeframe
        ORDER BY cnt DESC
    """)).fetchall()

    level_stats = db.session.execute(text("""
        SELECT level_type, timeframe, COUNT(*) as cnt,
               SUM(CASE WHEN first_touched_at IS NULL THEN 1 ELSE 0 END) as naked
        FROM levels
        GROUP BY level_type, timeframe
        ORDER BY cnt DESC
    """)).fetchall()

    feature_count = db.session.execute(text(
        "SELECT COUNT(*) FROM features"
    )).fetchone()[0]

    return render_template('research/index.html',
                           candle_stats=candle_stats,
                           level_stats=level_stats,
                           feature_count=feature_count)


@research_bp.route('/api/experiment-results')
def api_experiment_results():
    """JSON: parse best results from each experiment JSONL file."""
    results = {}
    for fpath in RESULTS_DIR.glob('*.jsonl'):
        name = fpath.stem
        best_fitness = 0
        best_metrics = {}
        n_experiments = 0
        n_improvements = 0

        for line in fpath.read_text().strip().split('\n'):
            if not line.strip():
                continue
            try:
                exp = json.loads(line)
                n_experiments += 1
                if exp.get('improved'):
                    n_improvements += 1
                metrics = exp.get('metrics', {})
                fitness = metrics.get('fitness', metrics.get('f1_macro', 0))
                if isinstance(fitness, (int, float)) and fitness > best_fitness:
                    best_fitness = fitness
                    best_metrics = metrics
            except json.JSONDecodeError:
                continue

        if n_experiments > 0:
            results[name] = {
                'experiments': n_experiments,
                'improvements': n_improvements,
                'best_fitness': round(best_fitness, 4) if best_fitness < 1000 else round(best_fitness, 0),
                'best_metrics': best_metrics,
            }

    return jsonify(results)
