# data_fetching.py

import logging
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd
from tqdm import tqdm

def fetch_data(client, symbol, interval, start_str, end_str):
    """Fetch historical klines from Binance."""
    logging.info(f"Fetching {interval} data for {symbol} from {start_str} to {end_str}")
    try:
        klines = []
        limit = 1000  # Maximum records per request
        interval_milliseconds = {
            Client.KLINE_INTERVAL_1MINUTE: 60 * 1000,
            Client.KLINE_INTERVAL_1HOUR: 60 * 60 * 1000,
            Client.KLINE_INTERVAL_12HOUR: 12 * 60 * 60 * 1000,
            Client.KLINE_INTERVAL_1DAY: 24 * 60 * 60 * 1000,
            Client.KLINE_INTERVAL_1WEEK: 7 * 24 * 60 * 60 * 1000,
            Client.KLINE_INTERVAL_1MONTH: 30 * 24 * 60 * 60 * 1000,
        }
        start_timestamp = int(pd.to_datetime(start_str).timestamp() * 1000)
        end_timestamp = int(pd.to_datetime(end_str).timestamp() * 1000)
        all_klines = []
        total_requests = ((end_timestamp - start_timestamp) // (interval_milliseconds[interval] * limit)) + 1
        with tqdm(total=total_requests, desc=f"Fetching {interval} data") as pbar:
            while start_timestamp < end_timestamp:
                temp_klines = client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    startTime=start_timestamp,
                    endTime=end_timestamp,
                    limit=limit
                )
                if not temp_klines:
                    break
                all_klines.extend(temp_klines)
                start_timestamp = temp_klines[-1][6] + 1  # Use close_time of last candle +1 ms as new start
                pbar.update(1)
        if not all_klines:
            logging.warning(f"No data fetched for {interval} timeframe.")
            return pd.DataFrame()
        data = pd.DataFrame(all_klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        data['open_time'] = pd.to_datetime(data['open_time'], unit='ms')
        data['close_time'] = pd.to_datetime(data['close_time'], unit='ms')
        data[['open', 'high', 'low', 'close', 'volume']] = data[['open', 'high', 'low', 'close', 'volume']].astype(float)
        data.set_index('open_time', inplace=True)
        logging.info(f"Fetched {len(data)} records for {interval} timeframe.")
        return data
    except BinanceAPIException as e:
        logging.error(f"Error fetching data: {e}")
        return pd.DataFrame()
