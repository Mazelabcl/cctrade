"""merge heads

Revision ID: 81777b4b3b49
Revises: b5_refactor_features, bfc812709a55
Create Date: 2026-03-31 18:50:31.556056

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '81777b4b3b49'
down_revision: Union[str, Sequence[str], None] = ('b5_refactor_features', 'bfc812709a55')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
