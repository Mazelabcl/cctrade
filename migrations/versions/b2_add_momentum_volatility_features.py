"""Add momentum and volatility feature columns.

Revision ID: b2_momentum_features
Revises: a1702b7546dc
Create Date: 2026-02-09
"""
from alembic import op
import sqlalchemy as sa

revision = 'b2_momentum_features'
down_revision = 'a1702b7546dc'
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table('features') as batch_op:
        batch_op.add_column(sa.Column('rsi_14', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('macd_line', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('macd_signal', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('macd_histogram', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('bollinger_width', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('atr_14', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('momentum_12', sa.Float(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('features') as batch_op:
        batch_op.drop_column('momentum_12')
        batch_op.drop_column('atr_14')
        batch_op.drop_column('bollinger_width')
        batch_op.drop_column('macd_histogram')
        batch_op.drop_column('macd_signal')
        batch_op.drop_column('macd_line')
        batch_op.drop_column('rsi_14')
