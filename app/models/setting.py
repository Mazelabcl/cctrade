"""Key-value settings store for runtime configuration."""
from ..extensions import db


class Setting(db.Model):
    __tablename__ = 'settings'

    key = db.Column(db.String(128), primary_key=True)
    value = db.Column(db.Text, nullable=True)

    def __repr__(self):
        return f'<Setting {self.key}>'


def get_setting(key: str, default: str = None) -> str | None:
    """Get a setting value by key, returning default if not found."""
    row = db.session.get(Setting, key)
    if row is None or row.value is None:
        return default
    return row.value


def set_setting(key: str, value: str) -> None:
    """Set a setting value, creating or updating as needed."""
    row = db.session.get(Setting, key)
    if row is None:
        row = Setting(key=key, value=value)
        db.session.add(row)
    else:
        row.value = value
    db.session.commit()
