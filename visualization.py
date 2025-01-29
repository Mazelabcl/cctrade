# visualization.py

import logging
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np

def plot_timeframe_levels(prepared_data, timeframe):
    """Plot price data with levels for a specific timeframe."""
    data = prepared_data['data_timeframes'][timeframe]
    fractals = prepared_data['fractals'].get(timeframe, pd.DataFrame())
    htf_levels = prepared_data['htf_levels'].get(timeframe, pd.DataFrame())
    fib_levels = prepared_data['fib_levels'].get(timeframe, pd.DataFrame())
    
    if data.empty:
        logging.warning(f"No data available for {timeframe} timeframe. Skipping plotting.")
        return
    
    try:
        logging.info(f"Plotting {timeframe.capitalize()} timeframe levels")
        df = data.copy()
        df.set_index('open_time', inplace=True)
        # Prepare data for mplfinance
        mpf_data = df[['open', 'high', 'low', 'close']]
        # Prepare additional plots
        apds = []
        # Fractals
        if not fractals.empty:
            fractals = fractals.set_index('open_time')
            # Up fractals
            up_fractals = fractals[fractals['up_fractal'] == 1]
            if not up_fractals.empty:
                up_fib_series = pd.Series(np.nan, index=df.index)
                up_fib_series.loc[up_fractals.index] = up_fractals['low']
                apds.append(mpf.make_addplot(up_fib_series, type='scatter', markersize=100, marker='^', color='green'))
            # Down fractals
            down_fractals = fractals[fractals['down_fractal'] == 1]
            if not down_fractals.empty:
                down_fib_series = pd.Series(np.nan, index=df.index)
                down_fib_series.loc[down_fractals.index] = down_fractals['high']
                apds.append(mpf.make_addplot(down_fib_series, type='scatter', markersize=100, marker='v', color='red'))
        # HTF levels: Plot lines starting from their origin
        if not htf_levels.empty:
            for idx, row in htf_levels.iterrows():
                level = row['level']
                start_time = row['time']
                # Create a Series with NaN before start_time and level value from start_time onward
                htf_series = pd.Series(np.nan, index=df.index)
                htf_series.loc[df.index >= start_time] = level
                # Assign a unique color for each HTF level type
                if 'HTF_Level_daily' in row['type']:
                    color = 'blue'
                elif 'HTF_Level_weekly' in row['type']:
                    color = 'purple'
                elif 'HTF_Level_monthly' in row['type']:
                    color = 'brown'
                else:
                    color = 'orange'  # Default color
                apds.append(mpf.make_addplot(htf_series, color=color, linestyle='-', width=1.5))
        # Fibonacci levels (q1, q2, cc, q3) for Support and Resistance
        if not fib_levels.empty and 'anchor_index' in fib_levels.columns:
            logging.info(f"Plotting {len(fib_levels)} Fibonacci levels for {timeframe} timeframe.")
            for _, fib in fib_levels.iterrows():
                # Create a series for the fib level between start_time and end_time
                fib_series = pd.Series(np.nan, index=df.index)
                # Find the mask where the fib is active
                mask = (df.index >= fib['start_time']) & (df.index <= fib['end_time'])
                fib_series.loc[mask] = fib['fib_level']
                # Check if there are any non-NaN values to plot
                if not fib_series.isnull().all():
                    if fib['type'] == 'support':
                        color = 'green'  # Color for support Fib levels
                    else:
                        color = 'red'    # Color for resistance Fib levels
                    linestyle = '--' if fib['fib_type'] in ['q1', 'q2', 'q3', 'cc'] else '-.'
                    apds.append(mpf.make_addplot(fib_series, color=color, linestyle=linestyle, width=1.5))
        # Plot
        save_filename = f'BTCUSDT_{timeframe}_timeframe_with_levels.png'
        logging.info(f"Saving plot to {save_filename}")
        mpf.plot(
            mpf_data,
            type='candle',
            style='charles',
            addplot=apds,
            title=f'BTCUSDT {timeframe.capitalize()} Timeframe with Fib Levels',
            figsize=(15, 7),
            savefig=save_filename
        )
        logging.info(f"Plot saved successfully to {save_filename}")
        # Optionally display the plot
        # plt.show()
    except Exception as e:
        logging.error(f"Error plotting {timeframe} timeframe levels: {e}")

def visualize_volume_profile(prepared_data, timeframe):
    """Visualize volume profile for a specific timeframe."""
    data = prepared_data['data_timeframes'][timeframe]
    volume_profiles = prepared_data['volume_profiles'].get(timeframe, pd.DataFrame())
    
    if data.empty or volume_profiles.empty:
        logging.warning(f"No data available for {timeframe} timeframe volume profile. Skipping plotting.")
        return
    
    # Create figure with two subplots (main and volume profile)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw={'width_ratios': [3, 1]})
    
    # Get data
    volume_profile = volume_profiles
    
    # Plot horizontal volume profile
    volume_profile.plot(x='volume', y='price', kind='barh', ax=ax1, color='lightgray', alpha=0.5)
    
    # Add POC, VAH, and VAL lines
    ax1.axhline(y=volume_profile['poc'].iloc[0], color='red', linestyle='-', label='POC', linewidth=2)
    ax1.axhline(y=volume_profile['vah'].iloc[0], color='blue', linestyle='--', label='VAH', linewidth=1)
    ax1.axhline(y=volume_profile['val'].iloc[0], color='blue', linestyle='--', label='VAL', linewidth=1)
    
    # Fill Value Area
    ax1.fill_between(ax1.get_xlim(), volume_profile['val'].iloc[0], volume_profile['vah'].iloc[0], color='blue', alpha=0.1)
    
    # Customize plot
    ax1.set_title(f'Volume Profile - {timeframe} Timeframe')
    ax1.set_xlabel('Volume')
    ax1.set_ylabel('Price Level')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add text annotations for levels
    max_volume = volume_profile['volume'].max()
    ax1.text(max_volume * 1.05, volume_profile['poc'].iloc[0], f'POC: {volume_profile["poc"].iloc[0]:.2f}', verticalalignment='center')
    ax1.text(max_volume * 1.05, volume_profile['vah'].iloc[0], f'VAH: {volume_profile["vah"].iloc[0]:.2f}', verticalalignment='bottom')
    ax1.text(max_volume * 1.05, volume_profile['val'].iloc[0], f'VAL: {volume_profile["val"].iloc[0]:.2f}', verticalalignment='top')
    
    # Plot vertical volume profile on the right
    volume_profile.plot(x='price', y='volume', kind='bar', ax=ax2, color='lightgray', alpha=0.5, width=1.0)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    
    # Add POC, VAH, and VAL lines
    ax2.axvline(x=volume_profile.index.get_loc(volume_profile['price'] == volume_profile['poc'].iloc[0]).start, color='red', linestyle='-', label='POC', linewidth=2)
    ax2.axvline(x=volume_profile.index.get_loc(volume_profile['price'] == volume_profile['vah'].iloc[0]).start, color='blue', linestyle='--', label='VAH', linewidth=1)
    ax2.axvline(x=volume_profile.index.get_loc(volume_profile['price'] == volume_profile['val'].iloc[0]).start, color='blue', linestyle='--', label='VAL', linewidth=1)
    
    # Customize right plot
    ax2.set_title('Volume Distribution')
    ax2.set_xlabel('Price Level')
    ax2.set_ylabel('Volume')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'BTCUSDT_{timeframe}_Volume_Profile.png', dpi=300, bbox_inches='tight')
    logging.info(f"Volume Profile plot saved to BTCUSDT_{timeframe}_Volume_Profile.png")
    plt.close()
