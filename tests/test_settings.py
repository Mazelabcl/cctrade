"""Tests for the Settings model, views, and API endpoints."""
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from app.models.setting import Setting, get_setting, set_setting


# --- Setting model helpers ---

def test_get_setting_returns_default_when_missing(app):
    """get_setting returns the default for a key that doesn't exist."""
    with app.app_context():
        assert get_setting('nonexistent') is None
        assert get_setting('nonexistent', 'fallback') == 'fallback'


def test_set_and_get_setting(app):
    """set_setting creates a row, get_setting retrieves it."""
    with app.app_context():
        set_setting('my_key', 'my_value')
        assert get_setting('my_key') == 'my_value'


def test_set_setting_updates_existing(app):
    """set_setting overwrites an existing key."""
    with app.app_context():
        set_setting('key', 'v1')
        set_setting('key', 'v2')
        assert get_setting('key') == 'v2'


def test_get_setting_returns_default_for_none_value(app):
    """get_setting returns default when value is explicitly None."""
    with app.app_context():
        from app.extensions import db
        row = Setting(key='null_key', value=None)
        db.session.add(row)
        db.session.commit()
        assert get_setting('null_key', 'default') == 'default'


# --- Settings page ---

def test_settings_page_loads(client):
    """GET /settings/ returns 200 with expected content."""
    response = client.get('/settings/')
    assert response.status_code == 200
    assert b'Settings' in response.data
    assert b'Binance API' in response.data
    assert b'Live Data Sync' in response.data


def test_settings_page_shows_masked_secret(app, client):
    """Settings page masks the API secret, showing only last 4 chars."""
    with app.app_context():
        set_setting('binance_api_secret', 'abcdefghijklmnop')

    response = client.get('/settings/')
    assert response.status_code == 200
    assert b'abcdefghijklmnop' not in response.data
    assert b'mnop' in response.data


def test_save_settings(client):
    """POST /settings/ saves API key and sync preferences."""
    response = client.post('/settings/', data={
        'api_key': 'test_api_key_123',
        'api_secret': 'test_secret_456',
        'live_sync_enabled': 'on',
        'live_sync_interval': '10',
        'sync_timeframes': '1h',
    }, follow_redirects=True)

    assert response.status_code == 200
    assert b'Settings saved' in response.data


def test_save_settings_persists_values(app, client):
    """POST /settings/ persists values retrievable via get_setting."""
    client.post('/settings/', data={
        'api_key': 'persist_key',
        'api_secret': 'persist_secret',
        'live_sync_interval': '15',
        'sync_timeframes': '1h',
    })

    with app.app_context():
        assert get_setting('binance_api_key') == 'persist_key'
        assert get_setting('binance_api_secret') == 'persist_secret'
        assert get_setting('live_sync_interval_minutes') == '15'
        assert get_setting('live_sync_enabled') == 'false'


def test_save_settings_does_not_overwrite_secret_with_mask(app, client):
    """POST with masked secret (starts with *) should not overwrite the real secret."""
    with app.app_context():
        set_setting('binance_api_secret', 'real_secret_value')

    client.post('/settings/', data={
        'api_key': 'some_key',
        'api_secret': '********alue',  # masked placeholder
        'live_sync_interval': '5',
        'sync_timeframes': '1h',
    })

    with app.app_context():
        assert get_setting('binance_api_secret') == 'real_secret_value'


def test_save_settings_sync_disabled_when_unchecked(app, client):
    """When live_sync_enabled checkbox is not sent, it saves as false."""
    with app.app_context():
        set_setting('live_sync_enabled', 'true')

    client.post('/settings/', data={
        'api_key': 'k',
        'api_secret': 's',
        'live_sync_interval': '5',
        'sync_timeframes': '1h',
        # no live_sync_enabled field — checkbox unchecked
    })

    with app.app_context():
        assert get_setting('live_sync_enabled') == 'false'


# --- Timeframes ---

def test_save_settings_multiple_timeframes(app, client):
    """POST with multiple timeframes saves them comma-separated."""
    from werkzeug.datastructures import MultiDict
    client.post('/settings/', data=MultiDict([
        ('api_key', 'k'),
        ('api_secret', 's'),
        ('live_sync_interval', '5'),
        ('sync_timeframes', '1h'),
        ('sync_timeframes', '4h'),
        ('sync_timeframes', '1d'),
    ]))

    with app.app_context():
        assert get_setting('sync_timeframes') == '1h,4h,1d'


def test_save_settings_no_timeframes_defaults_to_1h(app, client):
    """POST with no timeframes selected defaults to 1h."""
    client.post('/settings/', data={
        'api_key': 'k',
        'api_secret': 's',
        'live_sync_interval': '5',
        # no sync_timeframes
    })

    with app.app_context():
        assert get_setting('sync_timeframes') == '1h'


def test_settings_page_shows_timeframe_checkboxes(client):
    """Settings page includes timeframe checkboxes."""
    response = client.get('/settings/')
    assert response.status_code == 200
    assert b'tf_1h' in response.data
    assert b'tf_4h' in response.data
    assert b'tf_1d' in response.data


# --- Pipeline toggle ---

def test_save_settings_pipeline_toggle(app, client):
    """POST with run_full_pipeline_on_sync checked saves as true."""
    client.post('/settings/', data={
        'api_key': 'k',
        'api_secret': 's',
        'live_sync_interval': '5',
        'sync_timeframes': '1h',
        'run_full_pipeline_on_sync': 'on',
    })

    with app.app_context():
        assert get_setting('run_full_pipeline_on_sync') == 'true'


def test_save_settings_pipeline_toggle_off(app, client):
    """POST without run_full_pipeline_on_sync saves as false."""
    with app.app_context():
        set_setting('run_full_pipeline_on_sync', 'true')

    client.post('/settings/', data={
        'api_key': 'k',
        'api_secret': 's',
        'live_sync_interval': '5',
        'sync_timeframes': '1h',
    })

    with app.app_context():
        assert get_setting('run_full_pipeline_on_sync') == 'false'


# --- Test connection ---

def test_test_connection_no_keys(client):
    """POST /settings/test-connection fails when no keys configured."""
    response = client.post('/settings/test-connection')
    assert response.status_code == 400
    data = response.get_json()
    assert data['status'] == 'error'
    assert 'not configured' in data['message']


def test_test_connection_success(app, client):
    """POST /settings/test-connection returns ok on successful Binance call."""
    with app.app_context():
        set_setting('binance_api_key', 'key123')
        set_setting('binance_api_secret', 'secret456')

    mock_client = MagicMock()
    mock_client.get_system_status.return_value = {'msg': 'normal'}
    mock_client.get_account.return_value = {
        'canTrade': True,
        'balances': [{'free': '1.0', 'locked': '0.0'}],
    }

    with patch('binance.client.Client', return_value=mock_client):
        response = client.post('/settings/test-connection')

    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'ok'
    assert data['can_trade'] is True
    assert data['balances'] == 1


def test_test_connection_api_error(app, client):
    """POST /settings/test-connection returns error on Binance failure."""
    with app.app_context():
        set_setting('binance_api_key', 'key')
        set_setting('binance_api_secret', 'secret')

    with patch('binance.client.Client', side_effect=Exception('Invalid API key')):
        response = client.post('/settings/test-connection')

    assert response.status_code == 400
    data = response.get_json()
    assert data['status'] == 'error'
    assert 'Invalid API key' in data['message']


# --- Sync status API ---

def test_sync_status_defaults(client):
    """GET /api/sync-status returns defaults when nothing configured."""
    response = client.get('/api/sync-status')
    assert response.status_code == 200
    data = response.get_json()
    assert data['enabled'] is False
    assert data['interval_minutes'] == 5
    assert data['job_active'] is False
    assert data['last_fetch'] is None
    assert 'recent_syncs' in data
    assert isinstance(data['recent_syncs'], list)


def test_sync_status_reflects_settings(app, client):
    """GET /api/sync-status reflects DB settings."""
    with app.app_context():
        set_setting('live_sync_enabled', 'true')
        set_setting('live_sync_interval_minutes', '10')

    response = client.get('/api/sync-status')
    data = response.get_json()
    assert data['enabled'] is True
    assert data['interval_minutes'] == 10


def test_sync_status_includes_history(app, client):
    """GET /api/sync-status includes recent_syncs from PipelineRun."""
    from app.extensions import db
    from app.models import PipelineRun

    with app.app_context():
        run = PipelineRun(
            pipeline_type='data_fetch',
            status='completed',
            started_at=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
            finished_at=datetime(2024, 1, 1, 12, 0, 5, tzinfo=timezone.utc),
            rows_processed=42,
        )
        db.session.add(run)
        db.session.commit()

    response = client.get('/api/sync-status')
    data = response.get_json()
    assert len(data['recent_syncs']) == 1
    assert data['recent_syncs'][0]['rows_processed'] == 42
    assert data['recent_syncs'][0]['status'] == 'completed'


def test_sync_status_persisted_last_fetch(app, client):
    """GET /api/sync-status returns last_fetch from DB setting."""
    with app.app_context():
        set_setting('last_sync_at', '2024-06-15T10:30:00+00:00')

    response = client.get('/api/sync-status')
    data = response.get_json()
    assert data['last_fetch'] is not None
    assert '2024-06-15' in data['last_fetch']


# --- Toggle live sync API ---

def test_toggle_live_sync_enable(app, client):
    """POST /api/toggle-live-sync with enabled=true saves setting."""
    with patch('app.tasks.scheduler.start_live_sync'):
        response = client.post('/api/toggle-live-sync',
                               json={'enabled': True})

    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'ok'

    with app.app_context():
        assert get_setting('live_sync_enabled') == 'true'


def test_toggle_live_sync_disable(app, client):
    """POST /api/toggle-live-sync with enabled=false saves setting and stops sync."""
    with app.app_context():
        set_setting('live_sync_enabled', 'true')

    with patch('app.tasks.scheduler.stop_live_sync'):
        response = client.post('/api/toggle-live-sync',
                               json={'enabled': False})

    assert response.status_code == 200

    with app.app_context():
        assert get_setting('live_sync_enabled') == 'false'


# --- Latest signal API ---

def test_latest_signal_no_data(client):
    """GET /api/latest-signal returns no_data when no predictions exist."""
    response = client.get('/api/latest-signal')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'no_data'


def test_latest_signal_with_prediction(app, client):
    """GET /api/latest-signal returns signal data when predictions exist."""
    from app.extensions import db
    from app.models import Candle, Feature, Prediction, MLModel

    with app.app_context():
        candle = Candle(
            symbol='BTCUSDT', timeframe='1h',
            open_time=datetime(2024, 6, 1, 12, tzinfo=timezone.utc),
            open=65000, high=65500, low=64800, close=65200, volume=100,
        )
        db.session.add(candle)
        db.session.flush()

        feature = Feature(candle_id=candle.id, atr_14=500.0)
        db.session.add(feature)

        model = MLModel(
            name='test', algorithm='random_forest', version=1,
            prediction_horizon='day',
            file_path='/tmp/test.pkl', feature_names='[]',
        )
        db.session.add(model)
        db.session.flush()

        pred = Prediction(
            model_id=model.id, candle_id=candle.id,
            predicted_class=1,
            prob_no_fractal=0.2, prob_bullish=0.65, prob_bearish=0.15,
            confidence=0.65,
        )
        db.session.add(pred)
        db.session.commit()

    response = client.get('/api/latest-signal')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'ok'
    assert data['predicted_class'] == 1
    assert data['confidence'] == 0.65
    assert data['prob_bullish'] == 0.65
    assert data['signal'] in ('LONG', 'FLAT')  # depends on level proximity
    assert data['candle_close'] == 65200
    assert 'reason' in data


# --- data_sync reads from DB settings ---

def test_sync_candle_data_uses_db_keys(app):
    """sync_candle_data prefers DB settings over Flask config for API keys."""
    with app.app_context():
        set_setting('binance_api_key', 'db_key')
        set_setting('binance_api_secret', 'db_secret')

        with patch('app.tasks.data_sync.fetch_candles', return_value=5) as mock_fetch:
            from app.tasks.data_sync import sync_candle_data
            count = sync_candle_data()

        assert count == 5
        call_kwargs = mock_fetch.call_args[1]
        assert call_kwargs['api_key'] == 'db_key'
        assert call_kwargs['api_secret'] == 'db_secret'


def test_sync_candle_data_falls_back_to_config(app):
    """sync_candle_data falls back to Flask config when DB settings are empty."""
    with app.app_context():
        app.config['BINANCE_API_KEY'] = 'env_key'
        app.config['BINANCE_API_SECRET'] = 'env_secret'

        with patch('app.tasks.data_sync.fetch_candles', return_value=3) as mock_fetch:
            from app.tasks.data_sync import sync_candle_data
            count = sync_candle_data()

        assert count == 3
        call_kwargs = mock_fetch.call_args[1]
        assert call_kwargs['api_key'] == 'env_key'
        assert call_kwargs['api_secret'] == 'env_secret'

        # Clean up
        app.config['BINANCE_API_KEY'] = ''
        app.config['BINANCE_API_SECRET'] = ''


def test_sync_candle_data_uses_configured_timeframes(app):
    """sync_candle_data reads timeframes from DB setting."""
    with app.app_context():
        set_setting('binance_api_key', 'k')
        set_setting('binance_api_secret', 's')
        set_setting('sync_timeframes', '1h,4h')

        with patch('app.tasks.data_sync.fetch_candles', return_value=2) as mock_fetch:
            from app.tasks.data_sync import sync_candle_data
            count = sync_candle_data()

        assert count == 4  # 2 candles per timeframe x 2 timeframes
        assert mock_fetch.call_count == 2
        intervals = [c[1]['interval'] for c in mock_fetch.call_args_list]
        assert '1h' in intervals
        assert '4h' in intervals


def test_sync_candle_data_persists_last_sync(app):
    """sync_candle_data persists last_sync_at to DB."""
    with app.app_context():
        set_setting('binance_api_key', 'k')
        set_setting('binance_api_secret', 's')

        with patch('app.tasks.data_sync.fetch_candles', return_value=1):
            from app.tasks.data_sync import sync_candle_data
            sync_candle_data()

        assert get_setting('last_sync_at') is not None
        assert get_setting('last_sync_candles') == '1'
