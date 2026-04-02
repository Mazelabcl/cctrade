#!/usr/bin/env python
"""Telegram Alert Bot — sends trade signals via Telegram.

Designed to run every 15 minutes via Windows Task Scheduler.
Fetches latest 15m candle, checks for confluence touch, sends alert.

Setup:
1. Talk to @BotFather on Telegram -> /newbot -> get TOKEN
2. Send a message to your bot, then visit:
   https://api.telegram.org/bot<TOKEN>/getUpdates
   to find your chat_id
3. Set in DB: python scripts/alert_bot.py --setup --token YOUR_TOKEN --chat-id YOUR_CHAT_ID
4. Test: python scripts/alert_bot.py --test
5. Schedule: create Windows Task Scheduler job running every 15 min

Usage:
    python scripts/alert_bot.py              # Normal run (check for signals)
    python scripts/alert_bot.py --test       # Send test message
    python scripts/alert_bot.py --setup --token TOKEN --chat-id CHAT_ID
"""
import sys
import os
import argparse
import logging
import urllib.request
import urllib.parse
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)


def send_telegram(token, chat_id, message):
    """Send a message via Telegram Bot API."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'HTML',
    }).encode('utf-8')

    req = urllib.request.Request(url, data=data)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            if result.get('ok'):
                logger.info("Telegram message sent")
                return True
            else:
                logger.error("Telegram API error: %s", result)
                return False
    except Exception as e:
        logger.error("Failed to send Telegram message: %s", e)
        return False


def check_and_alert():
    """Smart alerts: always sends a message with market status."""
    from app import create_app
    from app.extensions import db
    from app.models.setting import get_setting
    from app.services.confluence_signal import (
        check_for_signal, update_open_trades, get_market_status,
    )

    app = create_app()
    with app.app_context():
        token = get_setting('telegram_bot_token', '')
        chat_id = get_setting('telegram_chat_id', '')

        if not token or not chat_id:
            logger.warning("Telegram not configured. Run: python scripts/alert_bot.py --setup")
            return

        # Sync latest candles
        try:
            from app.tasks.data_sync import sync_candle_data
            count = sync_candle_data()
            if count > 0:
                logger.info("Synced %d new candles", count)
        except Exception as e:
            logger.warning("Sync failed (continuing with existing data): %s", e)

        # Check for new signal
        signal = check_for_signal(db.session)
        if signal:
            levels = json.loads(signal.level_types_json) if signal.level_types_json else []
            icon = '🟢' if signal.direction == 'LONG' else '🔴'
            msg = (
                f"<b>{icon} {signal.direction} SIGNAL!</b>\n\n"
                f"<b>Entry:</b> ${signal.entry_price:,.2f}\n"
                f"<b>Stop Loss:</b> ${signal.stop_loss:,.2f}\n"
                f"<b>Risk:</b> {signal.risk_pct:.3f}%\n"
                f"<b>Confluence:</b> {signal.confluence_score} types\n"
                f"<b>Levels:</b> {', '.join(levels)}\n"
            )
            send_telegram(token, chat_id, msg)

        # Update open trades
        closed = update_open_trades(db.session)
        for trade in closed:
            icon = '✅' if trade.pnl_r > 0 else '❌'
            msg = (
                f"<b>{icon} Trade Closed</b>\n\n"
                f"<b>{trade.direction}</b> from ${trade.entry_price:,.2f}\n"
                f"<b>Exit:</b> ${trade.exit_price:,.2f} ({trade.exit_reason})\n"
                f"<b>P&L:</b> {trade.pnl_r:+.2f}R "
                f"(${trade.pnl_r * 10:+.2f} at $10 risk)\n"
            )
            send_telegram(token, chat_id, msg)

        # Always send market status
        status = get_market_status(db.session)
        if status.get('status') == 'no_data':
            send_telegram(token, chat_id, "No candle data available.")
            return

        price = status['price']
        time_str = status.get('candle_time', '?')[:16]

        # Open trade update
        if status['open_trades']:
            for t in status['open_trades']:
                be_icon = '🛡' if t['breakeven_reached'] else '⏳'
                r_icon = '📈' if t['unrealized_r'] > 0 else '📉'
                msg = (
                    f"<b>{r_icon} Open: {t['direction']}</b> from ${t['entry_price']:,.2f}\n"
                    f"Current: ${price:,.2f} ({t['unrealized_r']:+.1f}R)\n"
                    f"{be_icon} BE: {'ACTIVE' if t['breakeven_reached'] else 'not yet'} "
                    f"| Trail SL: ${t['current_trail_sl']:,.2f}\n"
                    f"Bars: {t['bars_since_entry']}"
                )
                send_telegram(token, chat_id, msg)
            return  # Don't spam status when trade is open

        # Approaching zone alert
        if status['approaching_zones']:
            zone = status['approaching_zones'][0]
            direction = '⬇️ support' if zone['direction'] == 'support' else '⬆️ resistance'
            levels_text = '\n'.join(
                f"  - {l['type']} ({l['tf']}) @ ${l['price']:,.2f}"
                for l in zone['levels'][:5]
            )
            msg = (
                f"<b>⚠️ Approaching Zone!</b>\n\n"
                f"BTC: ${price:,.2f} -> {direction}\n"
                f"Zone @ ${zone['center_price']:,.2f} ({zone['dist_pct']:.2f}% away)\n"
                f"Confluence: {zone['confluence']} types\n"
                f"{levels_text}\n\n"
                f"<i>Prepare for entry</i>"
            )
            send_telegram(token, chat_id, msg)
            return

        # Normal status (no trade, no approaching zone)
        sup = status.get('nearest_support')
        res = status.get('nearest_resistance')
        sup_text = f"{sup['type']} ({sup['tf']}) @ ${sup['price']:,.2f} — {sup['dist_pct']:.2f}% below" if sup else "none"
        res_text = f"{res['type']} ({res['tf']}) @ ${res['price']:,.2f} — {res['dist_pct']:.2f}% above" if res else "none"

        msg = (
            f"<b>📊 Status</b> ({time_str})\n\n"
            f"BTC: ${price:,.2f}\n"
            f"Support: {sup_text}\n"
            f"Resistance: {res_text}\n"
            f"Open trades: {len(status['open_trades'])}"
        )
        send_telegram(token, chat_id, msg)


def setup(token, chat_id):
    """Store Telegram credentials in DB."""
    from app import create_app
    from app.extensions import db
    from app.models.setting import set_setting

    app = create_app()
    with app.app_context():
        set_setting('telegram_bot_token', token)
        set_setting('telegram_chat_id', chat_id)
        db.session.commit()
        print(f"Telegram configured: token={token[:10]}... chat_id={chat_id}")
        print("Test with: python scripts/alert_bot.py --test")


def test_message():
    """Send a test message with current market status."""
    from app import create_app
    from app.extensions import db
    from app.models.setting import get_setting
    from app.services.confluence_signal import get_market_status

    app = create_app()
    with app.app_context():
        token = get_setting('telegram_bot_token', '')
        chat_id = get_setting('telegram_chat_id', '')

        if not token or not chat_id:
            print("Telegram not configured. Run: python scripts/alert_bot.py --setup --token TOKEN --chat-id CHAT_ID")
            return

        # Try to get live market status
        try:
            status = get_market_status(db.session)
            price = status.get('price', '?')
            sup = status.get('nearest_support')
            res = status.get('nearest_resistance')
            sup_text = f"${sup['price']:,.2f} ({sup['type']} {sup['tf']}, {sup['dist_pct']:.2f}%)" if sup else "none"
            res_text = f"${res['price']:,.2f} ({res['type']} {res['tf']}, {res['dist_pct']:.2f}%)" if res else "none"
            zones = len(status.get('approaching_zones', []))

            msg = (
                f"<b>🤖 Tradebot Alert System — TEST</b>\n\n"
                f"Alerts are working!\n\n"
                f"<b>Current BTC:</b> ${price:,.2f}\n"
                f"<b>Nearest support:</b> {sup_text}\n"
                f"<b>Nearest resistance:</b> {res_text}\n"
                f"<b>Approaching zones:</b> {zones}\n\n"
                f"<i>You will receive status updates every 15 min</i>"
            )
        except Exception as e:
            msg = (
                f"<b>🤖 Tradebot Alert System — TEST</b>\n\n"
                f"Alerts are working!\n"
                f"<i>(No market data available: {e})</i>"
            )

        ok = send_telegram(token, chat_id, msg)
        print("Test message sent!" if ok else "Failed to send test message")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Telegram Alert Bot')
    parser.add_argument('--test', action='store_true', help='Send test message')
    parser.add_argument('--setup', action='store_true', help='Configure Telegram credentials')
    parser.add_argument('--token', type=str, help='Telegram bot token')
    parser.add_argument('--chat-id', type=str, help='Telegram chat ID')
    args = parser.parse_args()

    if args.setup:
        if not args.token or not args.chat_id:
            print("Usage: python scripts/alert_bot.py --setup --token YOUR_TOKEN --chat-id YOUR_CHAT_ID")
            sys.exit(1)
        setup(args.token, args.chat_id)
    elif args.test:
        test_message()
    else:
        check_and_alert()
