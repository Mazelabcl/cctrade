import pandas as pd
import plotly.graph_objects as go
import random
from typing import List, Dict, Tuple
from datetime import datetime, timedelta

def create_candlestick_chart(
    candles_df: pd.DataFrame,
    center_index: int,
    window_size: int = 100
) -> go.Figure:
    """
    Create a candlestick chart centered on a specific candle.
    
    Args:
        candles_df: DataFrame with candle data (timestamp, open, high, low, close)
        center_index: Index of the candle to center on
        window_size: Number of candles to show (default 100)
    
    Returns:
        Plotly Figure object
    """
    # Calculate window bounds
    start_idx = max(0, center_index - window_size//2)
    end_idx = min(len(candles_df), start_idx + window_size)
    window_df = candles_df.iloc[start_idx:end_idx].copy()
    
    # Create candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=window_df['timestamp'],
        open=window_df['open'],
        high=window_df['high'],
        low=window_df['low'],
        close=window_df['close']
    )])
    
    return fig

def add_zone_to_chart(
    fig: go.Figure,
    zone: Dict,
    candle_time: datetime,
    is_support: bool = True,
    alpha: float = 0.1
) -> go.Figure:
    """
    Add a support or resistance zone to the chart.
    
    Args:
        fig: Plotly Figure object
        zone: Zone dictionary with zone_start, zone_end, level_prices
        candle_time: Timestamp of the N-2 candle that created this zone
        is_support: True if support zone, False if resistance
        alpha: Opacity of the zone (default 0.1)
    
    Returns:
        Updated Plotly Figure
    """
    if not zone:
        return fig
        
    # Add zone as a rectangle
    color = 'rgba(0,255,0,0.1)' if is_support else 'rgba(255,0,0,0.1)'
    fig.add_shape(
        type="rect",
        x0=candle_time,
        x1=candle_time + timedelta(hours=2),  # Show zone for 2 candles
        y0=zone['zone_start'],
        y1=zone['zone_end'],
        fillcolor=color,
        opacity=alpha,
        layer="below",
        line_width=0,
    )
    
    # Add individual levels as horizontal lines
    for price in zone['level_prices']:
        fig.add_shape(
            type="line",
            x0=candle_time,
            x1=candle_time + timedelta(hours=2),
            y0=price,
            y1=price,
            line=dict(
                color="green" if is_support else "red",
                width=1,
                dash="dot",
            ),
        )
    
    return fig

def add_swing_point_markers(
    fig: go.Figure,
    candles_df: pd.DataFrame,
    features_df: pd.DataFrame,
    index: int
) -> go.Figure:
    """Add swing high/low markers to the chart."""
    # Get features for this candle
    features = features_df.iloc[index]
    candle = candles_df.iloc[index]
    
    # Check if it's a swing point
    if pd.notna(features['fractal_timing_high']):
        # Add triangle marker above the high
        fig.add_trace(go.Scatter(
            x=[candle['timestamp']],
            y=[candle['high'] + (candle['high'] * 0.001)],  # Slightly above high
            mode='markers',
            marker=dict(
                symbol='triangle-down',
                size=12,
                color='red',
            ),
            name=f'Swing High (Time: {features["fractal_timing_high"]})',
            showlegend=False
        ))
    
    if pd.notna(features['fractal_timing_low']):
        # Add triangle marker below the low
        fig.add_trace(go.Scatter(
            x=[candle['timestamp']],
            y=[candle['low'] - (candle['low'] * 0.001)],  # Slightly below low
            mode='markers',
            marker=dict(
                symbol='triangle-up',
                size=12,
                color='green',
            ),
            name=f'Swing Low (Time: {features["fractal_timing_low"]})',
            showlegend=False
        ))
    
    return fig

def visualize_zone_sequence(
    candles_file: str = "ml_dataset.csv",
    features_file: str = "ml_features_dataset.csv",
    num_samples: int = 5,  # Number of sequences to visualize
    min_touches: int = 1  # Minimum number of levels touched
) -> None:
    """
    Create visualizations showing the sequence of zone creation and interaction.
    Shows:
    - N-2 candle (blue) and its zones
    - N-1 and N candles in gray for context
    
    Args:
        candles_file: Path to candles dataset
        features_file: Path to features dataset
        num_samples: Number of sequences to visualize
        min_touches: Minimum number of levels that must be touched
    """
    # Load datasets
    candles_df = pd.read_csv(candles_file)
    features_df = pd.read_csv(features_file)
    
    # Convert timestamps
    candles_df['timestamp'] = pd.to_datetime(candles_df['open_time'])
    
    # Find indices where zones were touched
    touched_indices = features_df[
        (features_df['total_support_touches'] >= min_touches) |
        (features_df['total_resistance_touches'] >= min_touches)
    ].index
    
    # Sample from touched indices (if we have enough)
    samples = random.sample(list(touched_indices), min(num_samples, len(touched_indices)))
    
    # Create visualization for each sample
    for i, idx in enumerate(samples):
        # Create figure
        fig = go.Figure()
        
        # Get candles for this sequence
        sequence_df = candles_df.iloc[idx-2:idx+1]
        
        # Plot N-2 candle in blue (this is our pivot)
        n2_candle = sequence_df.iloc[0]
        fig.add_candlestick(
            x=[n2_candle['timestamp']],
            open=[n2_candle['open']],
            high=[n2_candle['high']],
            low=[n2_candle['low']],
            close=[n2_candle['close']],
            increasing_line_color='blue',
            decreasing_line_color='blue',
            name='N-2 (Pivot)'
        )
        
        # Plot N-1 and N candles in gray
        other_candles = sequence_df.iloc[1:]
        fig.add_candlestick(
            x=other_candles['timestamp'],
            open=other_candles['open'],
            high=other_candles['high'],
            low=other_candles['low'],
            close=other_candles['close'],
            increasing_line_color='gray',
            decreasing_line_color='gray',
            name='Other Candles'
        )
        
        # Add zones from N-2 candle
        features = features_df.iloc[idx-2]
        
        # Support zone above N-2 candle
        if pd.notna(features['support_zone_start']):
            support_zone = {
                'zone_start': float(features['support_zone_start']),
                'zone_end': float(features['support_zone_end']),
                'level_prices': eval(features['support_level_prices'])
            }
            fig = add_zone_to_chart(
                fig=fig,
                zone=support_zone,
                candle_time=n2_candle['timestamp'],
                is_support=True,
                alpha=0.3
            )
        
        # Resistance zone below N-2 candle
        if pd.notna(features['resistance_zone_start']):
            resistance_zone = {
                'zone_start': float(features['resistance_zone_start']),
                'zone_end': float(features['resistance_zone_end']),
                'level_prices': eval(features['resistance_level_prices'])
            }
            fig = add_zone_to_chart(
                fig=fig,
                zone=resistance_zone,
                candle_time=n2_candle['timestamp'],
                is_support=False,
                alpha=0.3
            )
        
        # Add swing point markers
        for j in range(3):
            fig = add_swing_point_markers(
                fig=fig,
                candles_df=candles_df,
                features_df=features_df,
                index=idx-2+j
            )
        
        # Update layout
        fig.update_layout(
            title=f'Zone Sequence {i+1}',
            xaxis_title='Time',
            yaxis_title='Price',
            showlegend=True
        )
        
        # Save to HTML
        fig.write_html(f'zone_sequence_{i+1}.html')

if __name__ == "__main__":
    visualize_zone_sequence()
