# -*- coding: utf-8 -*-
"""
advanced_viz.py
Telemetry comparison and track visualizations.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import fastf1
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def get_driver_color(driver, session):
    """Helper to get team color for a driver."""
    try:
        # Try getting from session info
        if hasattr(session, 'get_driver'):
            drv_info = session.get_driver(driver)
            team_name = drv_info['TeamName']
            # You might want to import TEAM_COLORS here or use fastf1
            return fastf1.plotting.get_team_color(team_name, session=session)
    except:
        pass
    return '#888888' # Default gray

def plot_telemetry_comparison(session, driver1, driver2, lap_number=None):
    """
    Compare telemetry (Speed, Throttle, Brake, Gear, RPM) between two drivers.
    Returns a Plotly Figure.
    """
    try:
        laps = session.laps
        
        # Select laps
        if lap_number:
            lap1 = laps.pick_driver(driver1).pick_lap(lap_number)
            lap2 = laps.pick_driver(driver2).pick_lap(lap_number)
            title_suffix = f"Lap {lap_number}"
        else:
            lap1 = laps.pick_driver(driver1).pick_fastest()
            lap2 = laps.pick_driver(driver2).pick_fastest()
            title_suffix = "Fastest Lap"
            
        if lap1.empty or lap2.empty:
            return None

        # Get telemetry
        tel1 = lap1.get_car_data().add_distance()
        tel2 = lap2.get_car_data().add_distance()
        
        # Colors
        color1 = get_driver_color(driver1, session)
        color2 = get_driver_color(driver2, session)
        
        # Create subplots
        fig = make_subplots(
            rows=6, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=("Speed", "Throttle", "Brake", "Gear", "RPM", "Time Delta"),
            row_heights=[0.25, 0.1, 0.1, 0.1, 0.2, 0.25]
        )
        
        # 1. Speed
        fig.add_trace(go.Scatter(
            x=tel1['Distance'], y=tel1['Speed'],
            name=driver1, line=dict(color=color1), legendgroup=driver1
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=tel2['Distance'], y=tel2['Speed'],
            name=driver2, line=dict(color=color2), legendgroup=driver2
        ), row=1, col=1)
        
        # 2. Throttle
        fig.add_trace(go.Scatter(
            x=tel1['Distance'], y=tel1['Throttle'],
            name=driver1, line=dict(color=color1), showlegend=False, legendgroup=driver1
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=tel2['Distance'], y=tel2['Throttle'],
            name=driver2, line=dict(color=color2), showlegend=False, legendgroup=driver2
        ), row=2, col=1)
        
        # 3. Brake
        fig.add_trace(go.Scatter(
            x=tel1['Distance'], y=tel1['Brake'],
            name=driver1, line=dict(color=color1), showlegend=False, legendgroup=driver1,
            fill='tozeroy'
        ), row=3, col=1)
        
        fig.add_trace(go.Scatter(
            x=tel2['Distance'], y=tel2['Brake'],
            name=driver2, line=dict(color=color2), showlegend=False, legendgroup=driver2,
            fill='tozeroy', opacity=0.5
        ), row=3, col=1)
        
        # 4. Gear
        fig.add_trace(go.Scatter(
            x=tel1['Distance'], y=tel1['nGear'],
            name=driver1, line=dict(color=color1), showlegend=False, legendgroup=driver1,
            mode='lines' # Step line might be better but standard line is okay for now
        ), row=4, col=1)
        
        fig.add_trace(go.Scatter(
            x=tel2['Distance'], y=tel2['nGear'],
            name=driver2, line=dict(color=color2), showlegend=False, legendgroup=driver2
        ), row=4, col=1)
        
        # 5. RPM
        fig.add_trace(go.Scatter(
            x=tel1['Distance'], y=tel1['RPM'],
            name=driver1, line=dict(color=color1), showlegend=False, legendgroup=driver1
        ), row=5, col=1)
        
        fig.add_trace(go.Scatter(
            x=tel2['Distance'], y=tel2['RPM'],
            name=driver2, line=dict(color=color2), showlegend=False, legendgroup=driver2
        ), row=5, col=1)

        # Delta time calculation
        try:
            # Interpolate to find time diff
            # Need cumulative time
            t1 = tel1['Time'].dt.total_seconds()
            t2 = tel2['Time'].dt.total_seconds()
            d1 = tel1['Distance']
            d2 = tel2['Distance']
            
            # Interpolate t2 onto d1 axis
            # Note: Distances must be sorted, usually they are
            t2_interp = np.interp(d1, d2, t2)
            delta = t2_interp - t1 # Positive means driver1 is faster (smaller time)?
            # Let's say Delta = Driver 2 Time - Driver 1 Time
            # If Delta > 0, Driver 2 took longer -> Driver 1 is ahead.
            # Usually: Delta = Reference - Target.
            # Let's plot: Gap to Driver 1.
            # Gap = t2 - t1. 
            # If t2 > t1 (D2 is slower), Gap is +ve.
            gap = t2_interp - t1
            
            fig.add_trace(go.Scatter(
                x=d1, y=gap,
                name=f"Gap to {driver1}",
                line=dict(color='white', width=1),
                fill='tozeroy',
                fillcolor='rgba(255,255,255,0.1)',
                legendgroup='delta'
            ), row=6, col=1)
            
            fig.update_yaxes(title_text="Gap (s)", row=6, col=1)
            fig.update_xaxes(title_text="Distance (m)", row=6, col=1)
            
        except Exception as e:
            logger.warning(f"Could not calc delta: {e}")

        fig.update_layout(
            title=f"Telemetry Comparison: {driver1} vs {driver2} ({title_suffix})",
            height=1000,
            hovermode="x unified",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(0,0,0,0.5)"
            )
        )
        
        fig.update_yaxes(title_text="km/h", row=1, col=1)
        fig.update_yaxes(title_text="%", row=2, col=1, range=[0, 105])
        fig.update_yaxes(title_text="On/Off", row=3, col=1, range=[0, 1.1])
        fig.update_yaxes(title_text="Gear", row=4, col=1)
        fig.update_yaxes(title_text="RPM", row=5, col=1)
        # Row 6 handled above
        
        return fig
        
    except Exception as e:
        logger.error(f"Error plotting telemetry: {e}")
        return None

def plot_gear_shift_trace(session, driver=None):
    """
    Gear shift distribution and shift points.
    """
    try:
        laps = session.laps
        if driver:
            lap = laps.pick_driver(driver).pick_fastest()
        else:
            lap = laps.pick_fastest()
            
        tel = lap.get_telemetry()
        
        # Calculate gear shifts
        tel['GearChange'] = tel['nGear'].diff().fillna(0)
        shifts = tel[tel['GearChange'] != 0]
        
        fig = go.Figure()
        
        # Speed trace background
        fig.add_trace(go.Scatter(
            x=tel['Distance'], y=tel['Speed'],
            name='Speed',
            line=dict(color='gray', width=1),
            opacity=0.5
        ))
        
        # Upshifts
        upshifts = shifts[shifts['GearChange'] > 0]
        fig.add_trace(go.Scatter(
            x=upshifts['Distance'], y=upshifts['Speed'],
            mode='markers',
            name='Upshift',
            marker=dict(symbol='triangle-up', size=10, color='green')
        ))
        
        # Downshifts
        downshifts = shifts[shifts['GearChange'] < 0]
        fig.add_trace(go.Scatter(
            x=downshifts['Distance'], y=downshifts['Speed'],
            mode='markers',
            name='Downshift',
            marker=dict(symbol='triangle-down', size=10, color='red')
        ))
        
        fig.update_layout(
            title=f"Gear Shift Points - {lap['Driver']}",
            xaxis_title="Distance (m)",
            yaxis_title="Speed (km/h)",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error plotting gear shifts: {e}")
        return None

def plot_tyre_shape(session, driver):
    """
    Visualize tyre stints as concentric donuts or simple donut.
    Shows the tyre compound color and laps driven.
    """
    try:
        from config import TEAM_COLORS
        # Compound Colors (Standard Pirelli)
        comp_colors = {
            'SOFT': '#FF3333', 'MEDIUM': '#FFD700', 'HARD': '#FFFFFF', 
            'INTERMEDIATE': '#43B02A', 'WET': '#0067AD', 'UNKNOWN': '#888888'
        }
        
        # Get stints
        laps = session.laps.pick_driver(driver)
        stints = []
        current_stint = []
        last_compound = None
        
        # Identify stints logic (simplified)
        for i, lap in laps.iterrows():
            comp = lap.get('Compound', 'UNKNOWN')
            if pd.isna(comp): comp = 'UNKNOWN'
            
            if last_compound and comp != last_compound:
                stints.append({'compound': last_compound, 'laps': len(current_stint)})
                current_stint = []
            
            last_compound = comp
            current_stint.append(lap)
            
        if current_stint:
            stints.append({'compound': last_compound, 'laps': len(current_stint)})
            
        # Prepare Donut Data
        labels = []
        values = []
        colors = []
        
        for idx, s in enumerate(stints):
            c_name = str(s['compound']).upper()
            labels.append(f"Stint {idx+1} ({c_name})")
            values.append(s['laps'])
            colors.append(comp_colors.get(c_name, '#888'))
            
        fig = go.Figure(data=[go.Pie(
            labels=labels, 
            values=values, 
            hole=.6,
            marker_colors=colors,
            sort=False,
            textinfo='label+value',
            textposition='inside'
        )])
        
        fig.update_layout(
            title=f"Tyre Usage - {driver}",
            showlegend=False,
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
             annotations=[dict(text=driver, x=0.5, y=0.5, font_size=20, showarrow=False, font=dict(color='white'))]
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error plotting tyre shape: {e}")
        return None

def plot_circuit_context(session, highlight_sector=None):
    """
    Plot a mini-map of the circuit for context.
    Can highlight a specific sector or point.
    """
    try:
        lap = session.laps.pick_fastest()
        if lap is None: return None
        
        tel = lap.get_telemetry()
        x = tel['X']
        y = tel['Y']
        
        fig = go.Figure(go.Scatter(
            x=x, y=y,
            mode='lines',
            line=dict(color='gray', width=2),
            hoverinfo='skip'
        ))
        
        # Highlight logic (placeholder for sector highlighting)
        if highlight_sector:
             # Logic to find sector segment would go here
             pass
             
        fig.update_layout(
            height=200,
            width=200,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False, scaleanchor='y'),
            yaxis=dict(visible=False),
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False
        )
        return fig
    except Exception as e:
        logger.error(f"Error plotting circuit context: {e}")
        return None
