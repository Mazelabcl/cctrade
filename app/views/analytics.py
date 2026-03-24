"""Analytics dashboard — visualize experiments, features, and backtest results."""
from flask import Blueprint, render_template, jsonify, request
from ..extensions import db
from ..models import Feature, Level
from ..models.individual_level_backtest import IndividualLevelBacktest, IndividualLevelTrade
from sqlalchemy import func as sa_func, text

analytics_bp = Blueprint('analytics', __name__, template_folder='../templates')


@analytics_bp.route('/')
def index():
    """Analytics dashboard with Plotly tabs."""
    return render_template('analytics/index.html')


# ---------------------------------------------------------------------------
# Tab 1: Backtest Results Overview
# ---------------------------------------------------------------------------

@analytics_bp.route('/api/backtest-summary')
def api_backtest_summary():
    """JSON: backtest results grouped by (level_type, timeframe, strategy)."""
    strategy = request.args.get('strategy', 'wick_rr_1.0')

    results = (
        db.session.query(
            IndividualLevelBacktest.level_type,
            IndividualLevelBacktest.level_source_timeframe,
            IndividualLevelBacktest.total_trades,
            IndividualLevelBacktest.win_rate,
            IndividualLevelBacktest.profit_factor,
            IndividualLevelBacktest.winning_trades,
            IndividualLevelBacktest.losing_trades,
        )
        .filter(
            IndividualLevelBacktest.status == 'completed',
            IndividualLevelBacktest.strategy_name == strategy,
        )
        .order_by(IndividualLevelBacktest.win_rate.desc())
        .all()
    )

    data = []
    for lt, tf, trades, wr, pf, wins, losses in results:
        data.append({
            'level_type': lt,
            'timeframe': tf,
            'total_trades': trades or 0,
            'win_rate': round(wr, 1) if wr else 0,
            'profit_factor': round(pf, 2) if pf and pf != float('inf') else 0,
            'winning_trades': wins or 0,
            'losing_trades': losses or 0,
        })

    # Available strategies
    strategies = [r[0] for r in db.session.query(
        IndividualLevelBacktest.strategy_name
    ).distinct().all()]

    return jsonify({'data': data, 'strategies': strategies})


# ---------------------------------------------------------------------------
# Tab 2: Feature Distribution
# ---------------------------------------------------------------------------

@analytics_bp.route('/api/feature-distribution')
def api_feature_distribution():
    """JSON: feature distribution stats and sample data for histograms."""
    # Get raw feature data for histograms (sample for performance)
    rows = db.session.execute(text("""
        SELECT support_distance_pct, resistance_distance_pct,
               support_confluence_score, resistance_confluence_score,
               target_bullish, target_bearish
        FROM features
        WHERE support_distance_pct IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 5000
    """)).fetchall()

    sup_dist = [r[0] for r in rows if r[0] is not None]
    res_dist = [r[1] for r in rows if r[1] is not None]
    sup_conf = [r[2] for r in rows if r[2] is not None]
    res_conf = [r[3] for r in rows if r[3] is not None]
    is_bull = [r[4] for r in rows]
    is_bear = [r[5] for r in rows]

    # Stats
    stats_row = db.session.execute(text("""
        SELECT COUNT(*) as total,
               AVG(support_distance_pct) as avg_sup_dist,
               AVG(resistance_distance_pct) as avg_res_dist,
               AVG(support_confluence_score) as avg_sup_conf,
               AVG(resistance_confluence_score) as avg_res_conf,
               MAX(support_confluence_score) as max_sup_conf,
               MAX(resistance_confluence_score) as max_res_conf
        FROM features
        WHERE support_distance_pct IS NOT NULL
    """)).fetchone()

    # Percentile distribution
    pct_rows = db.session.execute(text("""
        SELECT
            SUM(CASE WHEN support_distance_pct < 0.001 THEN 1 ELSE 0 END) as p01,
            SUM(CASE WHEN support_distance_pct < 0.002 THEN 1 ELSE 0 END) as p02,
            SUM(CASE WHEN support_distance_pct < 0.005 THEN 1 ELSE 0 END) as p05,
            SUM(CASE WHEN support_distance_pct < 0.01 THEN 1 ELSE 0 END) as p10,
            SUM(CASE WHEN support_distance_pct < 0.02 THEN 1 ELSE 0 END) as p20,
            COUNT(*) as total
        FROM features WHERE support_distance_pct IS NOT NULL
    """)).fetchone()

    return jsonify({
        'support_distance': sup_dist,
        'resistance_distance': res_dist,
        'support_confluence': sup_conf,
        'resistance_confluence': res_conf,
        'is_bullish': is_bull,
        'is_bearish': is_bear,
        'stats': {
            'total': stats_row[0],
            'avg_sup_dist': round(stats_row[1] * 100, 3) if stats_row[1] else 0,
            'avg_res_dist': round(stats_row[2] * 100, 3) if stats_row[2] else 0,
            'avg_sup_conf': round(stats_row[3], 2) if stats_row[3] else 0,
            'avg_res_conf': round(stats_row[4], 2) if stats_row[4] else 0,
            'max_sup_conf': stats_row[5] or 0,
            'max_res_conf': stats_row[6] or 0,
        },
        'distance_pcts': {
            'within_01': round(pct_rows[0] / pct_rows[5] * 100, 1) if pct_rows[5] else 0,
            'within_02': round(pct_rows[1] / pct_rows[5] * 100, 1) if pct_rows[5] else 0,
            'within_05': round(pct_rows[2] / pct_rows[5] * 100, 1) if pct_rows[5] else 0,
            'within_10': round(pct_rows[3] / pct_rows[5] * 100, 1) if pct_rows[5] else 0,
            'within_20': round(pct_rows[4] / pct_rows[5] * 100, 1) if pct_rows[5] else 0,
        },
    })


# ---------------------------------------------------------------------------
# Tab 3: Level Density Analysis
# ---------------------------------------------------------------------------

@analytics_bp.route('/api/level-density')
def api_level_density():
    """JSON: how many naked levels exist at different time points."""
    # Sample time points (monthly intervals)
    rows = db.session.execute(text("""
        SELECT DISTINCT strftime('%Y-%m', open_time) as month
        FROM candles
        WHERE timeframe = '4h'
        ORDER BY month
    """)).fetchall()
    months = [r[0] for r in rows]

    density = []
    for month in months:
        # Count naked structural levels at mid-month
        mid = f'{month}-15 00:00:00'

        structural = db.session.execute(text("""
            SELECT COUNT(*) FROM levels
            WHERE timeframe IN ('daily','weekly','monthly')
            AND level_type IN ('Fractal_support','Fractal_resistance','HTF_level','Fib_CC')
            AND created_at <= :t
            AND (first_touched_at IS NULL OR first_touched_at > :t)
        """), {'t': mid}).fetchone()[0]

        # Count mobile levels (most recent per type+tf)
        mobile = db.session.execute(text("""
            SELECT COUNT(*) FROM (
                SELECT level_type, timeframe, MAX(created_at) as max_c
                FROM levels
                WHERE timeframe IN ('daily','weekly','monthly')
                AND (level_type LIKE 'PrevSession%' OR level_type LIKE 'VP_%')
                AND created_at <= :t
                GROUP BY level_type, timeframe
            )
        """), {'t': mid}).fetchone()[0]

        density.append({
            'month': month,
            'structural': structural,
            'mobile': mobile,
            'total': structural + mobile,
        })

    # Level type breakdown (current state)
    type_counts = db.session.execute(text("""
        SELECT level_type, timeframe, COUNT(*) as cnt,
               SUM(CASE WHEN first_touched_at IS NOT NULL THEN 1 ELSE 0 END) as touched
        FROM levels
        WHERE timeframe IN ('daily','weekly','monthly')
        GROUP BY level_type, timeframe
        ORDER BY cnt DESC
    """)).fetchall()

    breakdown = [{'type': r[0], 'tf': r[1], 'total': r[2], 'touched': r[3]}
                 for r in type_counts]

    return jsonify({'density_over_time': density, 'type_breakdown': breakdown})


# ---------------------------------------------------------------------------
# Tab 4: Scoring Analysis
# ---------------------------------------------------------------------------

@analytics_bp.route('/api/scoring-analysis')
def api_scoring_analysis():
    """JSON: trade scores vs outcomes for scoring analysis."""
    strategy = request.args.get('strategy', 'wick_rr_1.0')

    # Get trades with their backtest info for scoring
    rows = db.session.execute(text("""
        SELECT t.entry_price, t.exit_price, t.pnl_pct, t.exit_reason,
               t.direction, t.candles_held, t.distance_to_level,
               t.zone_confluence, t.volume_ratio,
               b.level_type, b.level_source_timeframe, b.win_rate
        FROM individual_level_trades t
        JOIN individual_level_backtests b ON t.backtest_id = b.id
        WHERE b.status = 'completed'
        AND b.strategy_name = :strategy
        AND t.exit_reason IN ('TP_HIT', 'SL_HIT')
        ORDER BY RANDOM()
        LIMIT 5000
    """), {'strategy': strategy}).fetchall()

    trades = []
    for r in rows:
        level_wr = (r[11] or 35) / 100.0
        level_score = level_wr * 10.0
        vol_score = min(max((r[8] or 0.5) - 0.5, 0) * 0.8, 2.0)
        conf_score = min(((r[7] or 1) - 1), 3) * 1.0
        prec_score = max(0, 2.0 - (r[6] or 0) * 400)
        total_score = level_score + vol_score + conf_score + prec_score

        trades.append({
            'score': round(total_score, 1),
            'pnl_pct': round((r[2] or 0) * 100, 2),
            'result': 'WIN' if r[3] == 'TP_HIT' else 'LOSS',
            'direction': r[4],
            'level_type': r[9],
            'level_tf': r[10],
            'candles_held': r[5],
        })

    # Equity curve by score thresholds
    all_trades_sorted = db.session.execute(text("""
        SELECT t.pnl_pct, t.exit_reason, t.entry_time,
               b.level_type, b.level_source_timeframe, b.win_rate,
               t.distance_to_level, t.zone_confluence, t.volume_ratio
        FROM individual_level_trades t
        JOIN individual_level_backtests b ON t.backtest_id = b.id
        WHERE b.status = 'completed'
        AND b.strategy_name = :strategy
        AND t.exit_reason IN ('TP_HIT', 'SL_HIT')
        ORDER BY t.entry_time
    """), {'strategy': strategy}).fetchall()

    equity_curves = {}
    for threshold in [0, 7, 8, 9, 10]:
        cumulative = 0
        curve = []
        for r in all_trades_sorted:
            level_wr = (r[5] or 35) / 100.0
            score = level_wr * 10.0 + min(max((r[8] or 0.5) - 0.5, 0) * 0.8, 2.0) + \
                    min(((r[7] or 1) - 1), 3) * 1.0 + max(0, 2.0 - (r[6] or 0) * 400)
            if score >= threshold:
                cumulative += (r[0] or 0) * 100
                curve.append(round(cumulative, 2))
        equity_curves[str(threshold)] = curve

    return jsonify({'trades': trades, 'equity_curves': equity_curves})


# ---------------------------------------------------------------------------
# Tab 5: MFE Analysis (Maximum Favorable Excursion)
# ---------------------------------------------------------------------------

@analytics_bp.route('/api/mfe')
def api_mfe():
    """JSON: MFE analysis data from saved results files."""
    import json
    import pandas as pd
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent.parent
    data = {}

    for tf in ['4h', '1h', '30m', '15m']:
        fpath = root / f'scripts/mfe_results{"_" + tf if tf != "4h" else ""}.json'
        if not fpath.exists():
            fpath = root / f'scripts/mfe_results_{tf}.json'
        if not fpath.exists() and tf == '4h':
            fpath = root / 'scripts/mfe_results.json'
        if fpath.exists():
            with open(fpath) as f:
                results = json.load(f)

            df = pd.DataFrame(results)
            if df.empty:
                continue

            # RR distribution
            rr_thresholds = [1, 2, 3, 5, 10, 15, 20, 30, 50]
            rr_dist = []
            for rr in rr_thresholds:
                count = int((df['max_rr'] >= rr).sum())
                pct = round((df['max_rr'] >= rr).mean() * 100, 1)
                rr_dist.append({'rr': rr, 'count': count, 'pct': pct})

            # By level type
            by_type = df.groupby('level_type').agg(
                trades=('max_rr', 'count'),
                avg_max_rr=('max_rr', 'mean'),
                median_max_rr=('max_rr', 'median'),
                pct_above_5=('max_rr', lambda x: round((x >= 5).mean() * 100, 1)),
                pct_above_10=('max_rr', lambda x: round((x >= 10).mean() * 100, 1)),
                avg_candles=('candles_to_max', 'mean'),
            ).reset_index().sort_values('avg_max_rr', ascending=False)

            # Sample max_rr values for histogram
            sample = df['max_rr'].clip(upper=100).tolist()

            # Money left on table for winners
            winners = df[df['original_exit'] == 'TP_HIT']
            money_left = winners['money_left'].tolist() if not winners.empty else []

            data[tf] = {
                'total_trades': len(df),
                'sl_hit_pct': round((df['final_exit'] == 'SL_HIT').mean() * 100, 1),
                'mean_max_rr': round(df['max_rr'].mean(), 1),
                'median_max_rr': round(df['max_rr'].median(), 1),
                'p90_max_rr': round(df['max_rr'].quantile(0.90), 1),
                'rr_distribution': rr_dist,
                'by_level_type': by_type.to_dict('records'),
                'max_rr_histogram': sample[:5000],
                'money_left': money_left[:5000],
                'mean_candles_to_max': round(df['candles_to_max'].mean(), 0),
                'median_candles_to_max': round(df['candles_to_max'].median(), 0),
            }

    return jsonify(data)


# ---------------------------------------------------------------------------
# Tab 6: Backtest Breakdown by Source Timeframe
# ---------------------------------------------------------------------------

@analytics_bp.route('/api/backtest-breakdown')
def api_backtest_breakdown():
    """JSON: backtest WR broken down by level_type x source_tf x exec_tf."""
    strategy = request.args.get('strategy', 'wick_rr_1.0')

    rows = db.session.execute(text("""
        SELECT level_type, level_source_timeframe, trade_execution_timeframe,
               total_trades, win_rate, profit_factor
        FROM individual_level_backtests
        WHERE status = 'completed' AND strategy_name = :strategy
        AND total_trades >= 5
        ORDER BY level_type, level_source_timeframe, trade_execution_timeframe
    """), {'strategy': strategy}).fetchall()

    data = []
    for lt, stf, etf, trades, wr, pf in rows:
        pf_val = float(pf) if pf and pf != float('inf') else 0
        data.append({
            'level_type': lt,
            'source_tf': stf,
            'exec_tf': etf,
            'trades': trades,
            'win_rate': round(wr, 1) if wr else 0,
            'profit_factor': round(pf_val, 2) if pf_val < 100 else 99.99,
        })

    return jsonify(data)
