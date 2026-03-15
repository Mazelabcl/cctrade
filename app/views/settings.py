"""Settings views — API keys, live sync configuration, foundation config."""
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app
from ..extensions import db
from ..models.setting import get_setting, set_setting

settings_bp = Blueprint('settings', __name__, template_folder='../templates')

AVAILABLE_TIMEFRAMES = ['1h', '4h', '1d']
ALL_TIMEFRAMES = ['1h', '4h', '1d', '1w', '1M']
LEVEL_TYPES = ['htf', 'fractal', 'fibonacci', 'vp']


def _get_foundation_config():
    """Read foundation config from DB settings with app config defaults."""
    cfg = current_app.config
    return {
        'data_start_date': get_setting('data_start_date', cfg.get('FOUNDATION_DATA_START', '2020-01-01')),
        'data_end_date': get_setting('data_end_date', cfg.get('FOUNDATION_DATA_END', '2025-12-31')),
        'train_test_cutoff': get_setting('train_test_cutoff', cfg.get('FOUNDATION_TRAIN_TEST_CUTOFF', '2024-06-01')),
        'fetch_timeframes': get_setting('fetch_timeframes', '1d,1w,1M').split(','),
        'htf_timeframes': get_setting('htf_timeframes', '1d,1w,1M').split(','),
        'fractal_timeframes': get_setting('fractal_timeframes', '1d,1w,1M').split(','),
        'fibonacci_timeframes': get_setting('fibonacci_timeframes', '1d,1w').split(','),
        'vp_timeframes': get_setting('vp_timeframes', '1d,1w,1M').split(','),
    }


@settings_bp.route('/')
def index():
    """Settings page: API keys, live sync toggle, foundation config."""
    api_key = get_setting('binance_api_key', '')
    api_secret = get_setting('binance_api_secret', '')
    live_sync_enabled = get_setting('live_sync_enabled', 'false') == 'true'
    live_sync_interval = get_setting('live_sync_interval_minutes', '5')
    sync_timeframes = get_setting('sync_timeframes', '1h').split(',')
    run_pipeline = get_setting('run_full_pipeline_on_sync', 'false') == 'true'

    # Mask the secret — show only last 4 chars
    masked_secret = ''
    if api_secret:
        masked_secret = '*' * 8 + api_secret[-4:]

    foundation = _get_foundation_config()

    return render_template('settings/index.html',
                           api_key=api_key,
                           masked_secret=masked_secret,
                           has_secret=bool(api_secret),
                           live_sync_enabled=live_sync_enabled,
                           live_sync_interval=live_sync_interval,
                           sync_timeframes=sync_timeframes,
                           available_timeframes=AVAILABLE_TIMEFRAMES,
                           run_pipeline=run_pipeline,
                           foundation=foundation,
                           all_timeframes=ALL_TIMEFRAMES)


@settings_bp.route('/', methods=['POST'])
def save():
    """Save API keys and sync preferences."""
    api_key = request.form.get('api_key', '').strip()
    api_secret = request.form.get('api_secret', '').strip()
    live_sync_enabled = request.form.get('live_sync_enabled') == 'on'
    live_sync_interval = request.form.get('live_sync_interval', '5')
    run_pipeline = request.form.get('run_full_pipeline_on_sync') == 'on'

    # Collect selected timeframes from checkboxes
    timeframes = request.form.getlist('sync_timeframes')
    if not timeframes:
        timeframes = ['1h']

    if api_key:
        set_setting('binance_api_key', api_key)

    # Only update secret if user entered a new one (not the masked placeholder)
    if api_secret and not api_secret.startswith('*'):
        set_setting('binance_api_secret', api_secret)

    set_setting('live_sync_enabled', 'true' if live_sync_enabled else 'false')
    set_setting('live_sync_interval_minutes', live_sync_interval)
    set_setting('sync_timeframes', ','.join(timeframes))
    set_setting('run_full_pipeline_on_sync', 'true' if run_pipeline else 'false')

    flash('Settings saved.', 'success')
    return redirect(url_for('settings.index'))


@settings_bp.route('/save-foundation', methods=['POST'])
def save_foundation():
    """Save foundation configuration."""
    set_setting('data_start_date', request.form.get('data_start_date', '2020-01-01'))
    set_setting('data_end_date', request.form.get('data_end_date', '2025-12-31'))
    set_setting('train_test_cutoff', request.form.get('train_test_cutoff', '2024-06-01'))

    # Only allow valid fetch timeframes
    valid_tfs = set(ALL_TIMEFRAMES)
    fetch_tfs = [t for t in request.form.getlist('fetch_timeframes') if t in valid_tfs] or ['1d']
    set_setting('fetch_timeframes', ','.join(fetch_tfs))

    for level_type in LEVEL_TYPES:
        tfs = [t for t in request.form.getlist(f'{level_type}_timeframes') if t in valid_tfs]
        set_setting(f'{level_type}_timeframes', ','.join(tfs))

    # Return JSON for AJAX calls, redirect for form submissions
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.accept_mimetypes.best == 'application/json':
        return jsonify({'status': 'ok', 'message': 'Foundation config saved'})

    flash('Foundation config saved.', 'success')
    return redirect(url_for('settings.index'))


@settings_bp.route('/test-connection', methods=['POST'])
def test_connection():
    """Test Binance API connectivity with stored keys."""
    api_key = get_setting('binance_api_key', '')
    api_secret = get_setting('binance_api_secret', '')

    if not api_key or not api_secret:
        return jsonify({'status': 'error', 'message': 'API keys not configured'}), 400

    try:
        from binance.client import Client
        client = Client(api_key, api_secret)
        status = client.get_system_status()
        account = client.get_account()
        balances = [b for b in account.get('balances', [])
                    if float(b['free']) > 0 or float(b['locked']) > 0]

        return jsonify({
            'status': 'ok',
            'system_status': status.get('msg', 'normal'),
            'can_trade': account.get('canTrade', False),
            'balances': len(balances),
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400
