"""Backtest runner — wires up Cerebro, runs the backtest, persists results."""
import logging
from datetime import datetime, timezone

import backtrader as bt
from sqlalchemy.orm import Session

from ...config import Config
from ...models import MLModel, BacktestResult, PipelineRun
from .data_feed import FractalPandasData, build_backtest_dataframe
from .strategy import FractalStrategy

logger = logging.getLogger(__name__)


def run_backtest(
    db: Session,
    model_id: int,
    initial_cash: float = Config.BACKTEST_INITIAL_CASH,
    commission: float = Config.BACKTEST_COMMISSION,
    risk_per_trade: float = Config.BACKTEST_RISK_PER_TRADE,
    confidence_threshold: float = Config.BACKTEST_CONFIDENCE_THRESHOLD,
    level_proximity_pct: float = Config.BACKTEST_LEVEL_PROXIMITY_PCT,
    atr_sl_mult: float = Config.BACKTEST_ATR_SL_MULT,
    atr_tp_mult: float = Config.BACKTEST_ATR_TP_MULT,
) -> BacktestResult:
    """Run a full backtest for a trained model and persist results.

    Returns the BacktestResult record.
    """
    run = PipelineRun(
        pipeline_type='backtest',
        status='running',
        started_at=datetime.now(timezone.utc),
        metadata_json={'model_id': model_id, 'cash': initial_cash},
    )
    db.add(run)
    db.commit()

    try:
        ml_model = db.get(MLModel, model_id)
        if ml_model is None:
            raise ValueError(f"Model {model_id} not found")

        # Build data
        df = build_backtest_dataframe(
            db, model_id,
            confidence_threshold=confidence_threshold,
            level_proximity_pct=level_proximity_pct,
        )
        if df.empty:
            raise ValueError("No data available for backtest")

        # Set up Cerebro
        cerebro = bt.Cerebro()
        data = FractalPandasData(dataname=df)
        cerebro.adddata(data)

        cerebro.addstrategy(
            FractalStrategy,
            risk_pct=risk_per_trade,
            confidence_threshold=confidence_threshold,
        )

        cerebro.broker.setcash(initial_cash)
        cerebro.broker.setcommission(commission=commission)

        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe',
                            riskfreerate=0.0, annualize=True, timeframe=bt.TimeFrame.Days)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')

        # Run
        results = cerebro.run()
        strat = results[0]

        # Collect metrics
        final_value = cerebro.broker.getvalue()
        total_return = (final_value - initial_cash) / initial_cash

        sharpe_analysis = strat.analyzers.sharpe.get_analysis()
        sharpe = sharpe_analysis.get('sharperatio')
        if sharpe is not None:
            sharpe = float(sharpe)

        dd_analysis = strat.analyzers.drawdown.get_analysis()
        max_dd = dd_analysis.get('max', {}).get('drawdown', 0.0)
        if max_dd:
            max_dd = float(max_dd) / 100.0  # convert from percentage

        sqn_analysis = strat.analyzers.sqn.get_analysis()
        sqn_val = sqn_analysis.get('sqn')
        if sqn_val is not None:
            sqn_val = float(sqn_val)

        # Custom trade analysis from strategy
        custom = strat.get_trade_analysis()

        # Persist result
        result = BacktestResult(
            model_id=model_id,
            initial_cash=initial_cash,
            commission=commission,
            risk_per_trade=risk_per_trade,
            confidence_threshold=confidence_threshold,
            level_proximity_pct=level_proximity_pct,
            atr_sl_mult=atr_sl_mult,
            atr_tp_mult=atr_tp_mult,
            final_value=final_value,
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            total_trades=custom['total_trades'],
            win_rate=custom['win_rate'],
            profit_factor=custom['profit_factor'],
            avg_win=custom['avg_win'],
            avg_loss=custom['avg_loss'],
            sqn=sqn_val,
            trade_log=strat.trade_log,
        )
        db.add(result)

        # Update MLModel with backtest metrics
        ml_model.sharpe_ratio = sharpe
        ml_model.max_drawdown = max_dd
        ml_model.profit_factor = custom['profit_factor']
        ml_model.win_rate = custom['win_rate']
        ml_model.total_trades = custom['total_trades']
        ml_model.backtest_return = total_return

        run.status = 'completed'
        run.finished_at = datetime.now(timezone.utc)
        run.rows_processed = len(df)
        db.commit()

        logger.info(
            "Backtest complete for model %d: return=%.2f%%, sharpe=%.2f, "
            "trades=%d, win_rate=%.2f%%",
            model_id, total_return * 100, sharpe or 0,
            custom['total_trades'], custom['win_rate'] * 100,
        )
        return result

    except Exception as exc:
        run.status = 'failed'
        run.finished_at = datetime.now(timezone.utc)
        run.error_message = str(exc)
        db.commit()
        raise
