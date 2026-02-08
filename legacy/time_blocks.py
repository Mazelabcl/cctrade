from typing import Dict
from datetime import datetime

def get_utc_block(candle_n1: dict) -> Dict:
    """
    Determine which 4-hour UTC block the candle N-1 belongs to.
    
    UTC Blocks (4 hours each):
    - Block 0: 00:00-03:59
    - Block 1: 04:00-07:59
    - Block 2: 08:00-11:59
    - Block 3: 12:00-15:59
    - Block 4: 16:00-19:59
    - Block 5: 20:00-23:59
    
    Args:
        candle_n1: Previous candle (N-1) with 'open_time' field in format 'YYYY-MM-DD HH:MM:SS'
        
    Returns:
        dict: Dictionary with time block:
            - utc_block: Integer 0-5 indicating which 4-hour block
    """
    try:
        # Convert string to datetime if it's not already
        if isinstance(candle_n1['open_time'], str):
            dt = datetime.strptime(candle_n1['open_time'], '%Y-%m-%d %H:%M:%S')
            print(f"Local time: {dt}")
            # Convert to UTC (assuming input is UTC-3)
            dt = dt.replace(hour=(dt.hour + 3) % 24)
            print(f"UTC time: {dt}")
        elif isinstance(candle_n1['open_time'], (int, float)):
            # Fallback for timestamp in milliseconds
            timestamp = candle_n1['open_time'] / 1000
            dt = datetime.utcfromtimestamp(timestamp)
        else:
            dt = candle_n1['open_time']
            
        # Get hour in UTC and determine block (4-hour blocks)
        block = dt.hour // 4
        print(f"Hour: {dt.hour}, Block: {block}")
        
        return {
            'utc_block': block
        }
        
    except Exception as e:
        print(f"Error determining UTC block: {e}")
        return {
            'utc_block': 0  # Default to block 0 in case of error
        }
