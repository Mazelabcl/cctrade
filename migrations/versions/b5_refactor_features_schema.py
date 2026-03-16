"""Refactor features table: add targets + confluence, remove retail/dead columns.

Revision ID: b5_refactor_features
Revises: fb7889110deb
Create Date: 2026-03-15
"""
from alembic import op
import sqlalchemy as sa

revision = 'b5_refactor_features'
down_revision = 'fb7889110deb'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Wipe features, predictions (all invalid with new target/schema)
    op.execute('DELETE FROM features')
    op.execute('DELETE FROM predictions')
    # Deactivate all models (incompatible with new feature set)
    op.execute('UPDATE ml_models SET is_active = 0')

    with op.batch_alter_table('features') as batch_op:
        # Add new columns: targets + zone quality features
        batch_op.add_column(sa.Column('target_bullish', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('target_bearish', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('support_confluence_score', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('resistance_confluence_score', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('support_liquidity_consumed', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('resistance_liquidity_consumed', sa.Float(), nullable=True))

        # Drop zone boundary columns (absolute prices, useless for ML)
        batch_op.drop_column('support_zone_start')
        batch_op.drop_column('support_zone_end')
        batch_op.drop_column('resistance_zone_start')
        batch_op.drop_column('resistance_zone_end')

        # Drop dead columns (never populated)
        batch_op.drop_column('total_support_touches')
        batch_op.drop_column('total_resistance_touches')
        batch_op.drop_column('support_level_vector')
        batch_op.drop_column('resistance_level_vector')
        batch_op.drop_column('support_touched_vector')
        batch_op.drop_column('resistance_touched_vector')

        # Drop retail indicators
        batch_op.drop_column('rsi_14')
        batch_op.drop_column('macd_line')
        batch_op.drop_column('macd_signal')
        batch_op.drop_column('macd_histogram')
        batch_op.drop_column('bollinger_width')

        # Drop old zone count columns (replaced by confluence_score)
        batch_op.drop_column('support_daily_count')
        batch_op.drop_column('support_weekly_count')
        batch_op.drop_column('support_monthly_count')
        batch_op.drop_column('support_fib618_count')
        batch_op.drop_column('support_naked_count')
        batch_op.drop_column('resistance_daily_count')
        batch_op.drop_column('resistance_weekly_count')
        batch_op.drop_column('resistance_monthly_count')
        batch_op.drop_column('resistance_fib618_count')
        batch_op.drop_column('resistance_naked_count')


def downgrade() -> None:
    with op.batch_alter_table('features') as batch_op:
        # Re-add dropped columns
        batch_op.add_column(sa.Column('resistance_naked_count', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('resistance_fib618_count', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('resistance_monthly_count', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('resistance_weekly_count', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('resistance_daily_count', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('support_naked_count', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('support_fib618_count', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('support_monthly_count', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('support_weekly_count', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('support_daily_count', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('bollinger_width', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('macd_histogram', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('macd_signal', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('macd_line', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('rsi_14', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('resistance_touched_vector', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('support_touched_vector', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('resistance_level_vector', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('support_level_vector', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('total_resistance_touches', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('total_support_touches', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('resistance_zone_end', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('resistance_zone_start', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('support_zone_end', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('support_zone_start', sa.Float(), nullable=True))

        # Drop new columns
        batch_op.drop_column('resistance_liquidity_consumed')
        batch_op.drop_column('support_liquidity_consumed')
        batch_op.drop_column('resistance_confluence_score')
        batch_op.drop_column('support_confluence_score')
        batch_op.drop_column('target_bearish')
        batch_op.drop_column('target_bullish')
