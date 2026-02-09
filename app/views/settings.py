"""Settings views — API keys, live sync configuration."""
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from ..extensions import db
from ..models.setting import get_setting, set_setting

settings_bp = Blueprint('settings', __name__, template_folder='../templates')

AVAILABLE_TIMEFRAMES = ['1h', '4h', '1d']


@settings_bp.route('/')
def index():
    """Settings page: API keys, live sync toggle."""
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

    return render_template('settings/index.html',
                           api_key=api_key,
                           masked_secret=masked_secret,
                           has_secret=bool(api_secret),
                           live_sync_enabled=live_sync_enabled,
                           live_sync_interval=live_sync_interval,
                           sync_timeframes=sync_timeframes,
                           available_timeframes=AVAILABLE_TIMEFRAMES,
                           run_pipeline=run_pipeline)


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
