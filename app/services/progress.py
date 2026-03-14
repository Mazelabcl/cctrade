"""In-memory progress tracker for long-running pipelines.

Avoids hitting the database for status checks during heavy writes.
Thread-safe via simple dict + lock.
"""
import threading
from datetime import datetime, timezone

_lock = threading.Lock()
_state = {
    'running': False,
    'step': '',
    'detail': '',
    'log': [],       # list of {'time': ..., 'msg': ...}
    'started_at': None,
    'finished_at': None,
    'error': None,
}


def start():
    """Mark pipeline as started, clear previous state."""
    with _lock:
        _state['running'] = True
        _state['step'] = 'starting'
        _state['detail'] = ''
        _state['log'] = []
        _state['started_at'] = datetime.now(timezone.utc).isoformat()
        _state['finished_at'] = None
        _state['error'] = None


def update(step: str, detail: str = ''):
    """Update current step and add to log."""
    with _lock:
        _state['step'] = step
        _state['detail'] = detail
        _state['log'].append({
            'time': datetime.now(timezone.utc).isoformat(),
            'msg': f"[{step}] {detail}" if detail else step,
        })


def finish(error: str = None):
    """Mark pipeline as finished."""
    with _lock:
        _state['running'] = False
        _state['finished_at'] = datetime.now(timezone.utc).isoformat()
        if error:
            _state['error'] = error
            _state['step'] = 'failed'
            _state['log'].append({
                'time': datetime.now(timezone.utc).isoformat(),
                'msg': f"ERROR: {error}",
            })
        else:
            _state['step'] = 'done'
            _state['log'].append({
                'time': datetime.now(timezone.utc).isoformat(),
                'msg': 'Pipeline completed successfully.',
            })


def get_state() -> dict:
    """Get current state (thread-safe copy)."""
    with _lock:
        return {
            'running': _state['running'],
            'step': _state['step'],
            'detail': _state['detail'],
            'log': list(_state['log']),  # copy
            'started_at': _state['started_at'],
            'finished_at': _state['finished_at'],
            'error': _state['error'],
        }
