import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os

def create_interactive_chart(df, timeframe, levels_data=None):
    """Create an interactive candlestick chart with volume."""
    try:
        print(f"Creating interactive chart for {timeframe} timeframe")
        print(f"Data for {timeframe}: {len(df)} rows")
        print(f"Creating figure for {timeframe} timeframe")
        
        # Create figure with secondary y-axis
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.03, subplot_titles=(f'BTCUSDT {timeframe}', 'Volume'),
                           row_heights=[0.7, 0.3])

        # Add candlestick
        fig.add_trace(go.Candlestick(x=df.index,
                                    open=df['open'].astype(float),
                                    high=df['high'].astype(float),
                                    low=df['low'].astype(float),
                                    close=df['close'].astype(float),
                                    name='OHLC'),
                                    row=1, col=1)

        # Add volume bar chart
        colors = ['red' if float(row['open']) > float(row['close']) else 'green' for _, row in df.iterrows()]
        
        fig.add_trace(go.Bar(x=df.index,
                            y=df['volume'].astype(float),
                            marker_color=colors,
                            name='Volume'),
                            row=2, col=1)

        # Add technical levels if available
        if levels_data:
            # Add HTF levels
            htf_levels = levels_data.get('htf_levels', {}).get(timeframe, pd.DataFrame())
            if not htf_levels.empty:
                for _, row in htf_levels.iterrows():
                    level = float(row['price_level'])
                    fig.add_hline(y=level, line_dash="dash", line_color="blue",
                                 annotation_text=f"HTF {level:.0f}",
                                 row=1, col=1)

            # Add Fibonacci levels
            fib_levels = levels_data.get('fib_levels', {}).get(timeframe, pd.DataFrame())
            if not fib_levels.empty:
                for _, row in fib_levels.iterrows():
                    level = float(row['price_level'])
                    fig.add_hline(y=level, line_dash="dot", line_color="purple",
                                 annotation_text=f"Fib {row['level_type']}",
                                 row=1, col=1)

            # Add Volume Profile levels
            vp_levels = levels_data.get('volume_profile', {}).get(timeframe, {})
            if vp_levels:
                for level_type, level in vp_levels.items():
                    if isinstance(level, (int, float)):
                        fig.add_hline(y=float(level), line_dash="solid", line_color="gray",
                                     annotation_text=f"VP {level_type}",
                                     row=1, col=1)

        # Update layout
        fig.update_layout(
            title=f'BTCUSDT {timeframe} Chart',
            yaxis_title='Price (USDT)',
            yaxis2_title='Volume',
            xaxis_rangeslider_visible=False
        )

        # Save to HTML
        output_file = f'BTCUSDT_{timeframe}_interactive.html'
        fig.write_html(output_file)
        print(f"Interactive chart saved to {output_file}")
        
    except Exception as e:
        print(f"Error creating interactive chart: {e}")

def create_all_timeframe_charts(prepared_data):
    """Create interactive charts for all timeframes."""
    for timeframe in ['daily', 'weekly', 'monthly']:
        df = prepared_data['data_timeframes'][timeframe]
        levels_data = {
            'htf_levels': prepared_data.get('htf_levels', {}).get(timeframe, []),
            'fib_levels': prepared_data.get('fib_levels', {}).get(timeframe, {}),
            'volume_profile': prepared_data.get('volume_profiles', {}).get(timeframe, {})
        }
        create_interactive_chart(df, timeframe, levels_data)
