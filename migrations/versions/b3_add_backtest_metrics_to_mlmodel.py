"""Add backtest metric columns to ml_models.

Revision ID: b3_backtest_metrics
Revises: b2_momentum_features
Create Date: 2026-02-09
"""
from alembic import op
import sqlalchemy as sa

revision = 'b3_backtest_metrics'
down_revision = 'b2_momentum_features'
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table('ml_models') as batch_op:
        batch_op.add_column(sa.Column('sharpe_ratio', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('max_drawdown', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('profit_factor', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('win_rate', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('total_trades', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('backtest_return', sa.Float(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('ml_models') as batch_op:
        batch_op.drop_column('backtest_return')
        batch_op.drop_column('total_trades')
        batch_op.drop_column('win_rate')
        batch_op.drop_column('profit_factor')
        batch_op.drop_column('max_drawdown')
        batch_op.drop_column('sharpe_ratio')
