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
    """Check for new signals and send alerts."""
    from app import create_app
    from app.extensions import db
    from app.models.setting import get_setting
    from app.services.confluence_signal import check_for_signal, update_open_trades

    app = create_app()
    with app.app_context():
        token = get_setting('telegram_bot_token', '')
        chat_id = get_setting('telegram_chat_id', '')

        if not token or not chat_id:
            logger.warning("Telegram not configured. Run: python scripts/alert_bot.py --setup")
            return

        # First sync latest candles
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
            msg = (
                f"<b>{'🟢' if signal.direction == 'LONG' else '🔴'} "
                f"{signal.direction} Signal</b>\n\n"
                f"<b>Entry:</b> ${signal.entry_price:,.2f}\n"
                f"<b>Stop Loss:</b> ${signal.stop_loss:,.2f}\n"
                f"<b>Risk:</b> {signal.risk_pct:.3f}%\n"
                f"<b>Confluence:</b> {signal.confluence_score} types\n"
                f"<b>Levels:</b> {', '.join(levels)}\n\n"
                f"<i>15m Confluence Scalper | be_rr=2.75</i>"
            )
            send_telegram(token, chat_id, msg)

        # Update open trades
        closed = update_open_trades(db.session)
        for trade in closed:
            icon = '✅' if trade.pnl_r > 0 else '❌'
            msg = (
                f"<b>{icon} Trade Closed</b>\n\n"
                f"<b>Direction:</b> {trade.direction}\n"
                f"<b>Entry:</b> ${trade.entry_price:,.2f}\n"
                f"<b>Exit:</b> ${trade.exit_price:,.2f}\n"
                f"<b>Reason:</b> {trade.exit_reason}\n"
                f"<b>P&L:</b> {trade.pnl_r:+.2f}R "
                f"(${trade.pnl_r * 10:+.2f} at $10 risk)\n"
            )
            send_telegram(token, chat_id, msg)

        if not signal and not closed:
            logger.info("No new signals or closed trades")


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
    """Send a test message."""
    from app import create_app
    from app.extensions import db
    from app.models.setting import get_setting

    app = create_app()
    with app.app_context():
        token = get_setting('telegram_bot_token', '')
        chat_id = get_setting('telegram_chat_id', '')

        if not token or not chat_id:
            print("Telegram not configured. Run: python scripts/alert_bot.py --setup --token TOKEN --chat-id CHAT_ID")
            return

        msg = (
            "<b>🤖 Tradebot Alert System</b>\n\n"
            "Test message - alerts are working!\n"
            "You will receive signals from the 15m confluence scalper.\n\n"
            "<i>PF 3.11 | WR 39% | $268/mo ($10 risk)</i>"
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
