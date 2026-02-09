"""Add backtest_results table.

Revision ID: b4_backtest_results
Revises: b3_backtest_metrics
Create Date: 2026-02-09
"""
from alembic import op
import sqlalchemy as sa

revision = 'b4_backtest_results'
down_revision = 'b3_backtest_metrics'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'backtest_results',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('model_id', sa.Integer(), sa.ForeignKey('ml_models.id'), nullable=False),
        sa.Column('initial_cash', sa.Float(), nullable=False),
        sa.Column('commission', sa.Float()),
        sa.Column('risk_per_trade', sa.Float()),
        sa.Column('confidence_threshold', sa.Float()),
        sa.Column('level_proximity_pct', sa.Float()),
        sa.Column('atr_sl_mult', sa.Float()),
        sa.Column('atr_tp_mult', sa.Float()),
        sa.Column('final_value', sa.Float()),
        sa.Column('total_return', sa.Float()),
        sa.Column('sharpe_ratio', sa.Float()),
        sa.Column('max_drawdown', sa.Float()),
        sa.Column('total_trades', sa.Integer()),
        sa.Column('win_rate', sa.Float()),
        sa.Column('profit_factor', sa.Float()),
        sa.Column('avg_win', sa.Float()),
        sa.Column('avg_loss', sa.Float()),
        sa.Column('sqn', sa.Float()),
        sa.Column('trade_log', sa.JSON()),
        sa.Column('created_at', sa.DateTime(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table('backtest_results')
