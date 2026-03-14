"""add first_touched_at to levels

Revision ID: fb7889110deb
Revises: 44d9feded03e
Create Date: 2026-03-10 23:38:39.634132

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'fb7889110deb'
down_revision: Union[str, Sequence[str], None] = '44d9feded03e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('levels', sa.Column('first_touched_at', sa.DateTime(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('levels', 'first_touched_at')
