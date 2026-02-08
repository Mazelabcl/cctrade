from datetime import datetime
from ..extensions import db


class PipelineRun(db.Model):
    __tablename__ = 'pipeline_runs'

    id = db.Column(db.Integer, primary_key=True)
    pipeline_type = db.Column(db.String(50), nullable=False)
    status = db.Column(db.String(20), nullable=False, default='running')
    started_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    finished_at = db.Column(db.DateTime)
    period_start = db.Column(db.DateTime)
    period_end = db.Column(db.DateTime)
    rows_processed = db.Column(db.Integer)
    error_message = db.Column(db.Text)
    metadata_json = db.Column('metadata', db.JSON)

    def __repr__(self):
        return f'<PipelineRun {self.pipeline_type} {self.status}>'

    def to_dict(self):
        return {
            'id': self.id,
            'pipeline_type': self.pipeline_type,
            'status': self.status,
            'started_at': self.started_at.isoformat(),
            'finished_at': self.finished_at.isoformat() if self.finished_at else None,
            'period_start': self.period_start.isoformat() if self.period_start else None,
            'period_end': self.period_end.isoformat() if self.period_end else None,
            'rows_processed': self.rows_processed,
            'error_message': self.error_message,
        }
