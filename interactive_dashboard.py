import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
import streamlit as st
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
from data_mapping import DataMapper
from portfolio_config import PortfolioConfig
from perfect_weight_calculator import PerfectWeightCalculator
from portfolio_simulation import PortfolioSimulation, SimulationConfig
from performance_analytics import PerformanceAnalytics
from reporting_visualization import PHPReporting

class InteractiveDashboard:
    """Interactive dashboard for PHP analysis using Plotly and Streamlit"""
    
    def __init__(self):
        self.data_mapper = DataMapper()
        self.portfolio_config = PortfolioConfig(self.data_mapper)
        self.perfect_weight_calc = PerfectWeightCalculator(
            self.data_mapper, self.portfolio_config)
        self.simulation_engine = PortfolioSimulation(
            self.data_mapper, self.portfolio_config, self.perfect_weight_calc)
        self.analytics_engine = PerformanceAnalytics()
        self.reporter = PHPReporting()
        
        # Dashboard state
        self.results_cache = {}
        
    def create_interactive_performance_dashboard(self, 
                                               analytics_results: Dict,
                                               mandate: str) -> go.Figure:
        """Create interactive performance dashboard with multiple views"""
        
        all_results = analytics_results.get('all_results', [])
        successful_results = [r for r in all_results if r.get('success', False)]
        
        if not successful_results:
            return self._create_empty_figure("No successful scenarios to display")
        
        # Extract data for visualization
        data_rows = []
        for result in successful_results:
            sim_results = result.get('simulation_results', {})
            config = result.get('config')
            
            if sim_results and config:
                data_rows.append({
                    'scenario_id': config.scenario_id,
                    'start_date': config.start_date,
                    'deviation': f"{config.permitted_deviation:.0%}",
                    'horizon': f"{config.investment_horizon_years}Y",
                    'rebalancing': config.rebalancing_frequency,
                    'transaction_cost': f"{config.transaction_cost_bps}bps",
                    'return': sim_results.get('annualized_return', 0),
                    'volatility': sim_results.get('volatility', 0),
                    'sharpe': sim_results.get('sharpe_ratio', 0),
                    'max_drawdown': sim_results.get('max_drawdown', 0),
                    'total_return': sim_results.get('total_return', 0),
                    'start_year': int(config.start_date[:4]) if config.start_date else 2000
                })
        
        df = pd.DataFrame(data_rows)
        
        # Create subplot dashboard
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                f'{mandate} PHP: Risk-Return Profile',
                f'{mandate} PHP: Return Distribution',
                f'{mandate} PHP: Performance by Parameter',
                f'{mandate} PHP: Drawdown Analysis',
                f'{mandate} PHP: Time Evolution',
                f'{mandate} PHP: Parameter Impact'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # Plot 1: Risk-Return Scatter (colored by Sharpe Ratio)
        fig.add_trace(
            go.Scatter(
                x=df['volatility'],
                y=df['return'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=df['sharpe'],
                    colorscale='Viridis',
                    colorbar=dict(title="Sharpe Ratio", x=0.47, len=0.25),
                    line=dict(width=1, color='white')
                ),
                text=df['scenario_id'],
                hovertemplate='<b>%{text}</b><br>' +
                             'Return: %{y:.1%}<br>' +
                             'Volatility: %{x:.1%}<br>' +
                             'Sharpe: %{marker.color:.2f}<extra></extra>',
                name='Scenarios'
            ),
            row=1, col=1
        )
        
        # Plot 2: Return Histogram
        fig.add_trace(
            go.Histogram(
                x=df['return'],
                nbinsx=30,
                name='Return Dist',
                marker_color='lightblue',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        # Add vertical lines for mean and median
        mean_return = df['return'].mean()
        median_return = df['return'].median()
        
        fig.add_vline(x=mean_return, line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: {mean_return:.1%}", row=1, col=2)
        fig.add_vline(x=median_return, line_dash="dash", line_color="orange",
                     annotation_text=f"Median: {median_return:.1%}", row=1, col=2)
        
        # Plot 3: Box plot by Deviation
        for deviation in df['deviation'].unique():
            subset = df[df['deviation'] == deviation]
            fig.add_trace(
                go.Box(
                    y=subset['return'],
                    name=deviation,
                    boxpoints='outliers',
                    jitter=0.3,
                    pointpos=-1.8
                ),
                row=2, col=1
            )
        
        # Plot 4: Drawdown vs Return
        # Map horizon strings to numeric values for color mapping
        horizon_color_map = {h: i for i, h in enumerate(df['horizon'].unique())}
        horizon_colors = [horizon_color_map[h] for h in df['horizon']]
        
        fig.add_trace(
            go.Scatter(
                x=df['max_drawdown'],
                y=df['return'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=horizon_colors,
                    colorscale='Plasma',
                    colorbar=dict(
                        title="Horizon", 
                        x=1.02, 
                        len=0.25,
                        tickmode='array',
                        tickvals=list(horizon_color_map.values()),
                        ticktext=list(horizon_color_map.keys())
                    ),
                    line=dict(width=1, color='white')
                ),
                text=df['scenario_id'],
                hovertemplate='<b>%{text}</b><br>' +
                             'Return: %{y:.1%}<br>' +
                             'Max Drawdown: %{x:.1%}<br>' +
                             'Horizon: %{customdata}<extra></extra>',
                customdata=df['horizon'],
                name='Risk-Return'
            ),
            row=2, col=2
        )
        
        # Plot 5: Performance by Start Year
        yearly_performance = df.groupby('start_year').agg({
            'return': ['mean', 'std', 'count'],
            'volatility': 'mean',
            'sharpe': 'mean'
        }).round(4)
        
        years = yearly_performance.index.tolist()
        avg_returns = yearly_performance[('return', 'mean')].tolist()
        return_stds = yearly_performance[('return', 'std')].tolist()
        
        fig.add_trace(
            go.Scatter(
                x=years,
                y=avg_returns,
                mode='lines+markers',
                marker=dict(size=8, color='blue'),
                line=dict(width=3),
                name='Avg Return',
                error_y=dict(
                    type='data',
                    array=return_stds,
                    visible=True
                )
            ),
            row=3, col=1
        )
        
        # Plot 6: Parameter Impact Heatmap (as scatter)
        # Create parameter combination score
        param_combinations = df.groupby(['deviation', 'horizon']).agg({
            'return': 'mean',
            'sharpe': 'mean',
            'max_drawdown': 'mean'
        }).reset_index()
        
        fig.add_trace(
            go.Scatter(
                x=param_combinations['deviation'],
                y=param_combinations['horizon'],
                mode='markers+text',
                marker=dict(
                    size=param_combinations['return'] * 500,  # Scale for visibility
                    color=param_combinations['sharpe'],
                    colorscale='RdYlGn',
                    colorbar=dict(title="Avg Sharpe", x=1.05, len=0.25),
                    line=dict(width=2, color='white')
                ),
                text=[f"{ret:.1%}" for ret in param_combinations['return']],
                textposition="middle center",
                hovertemplate='Deviation: %{x}<br>' +
                             'Horizon: %{y}<br>' +
                             'Avg Return: %{text}<br>' +
                             'Avg Sharpe: %{marker.color:.2f}<extra></extra>',
                name='Param Combinations'
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title=f'{mandate} PHP: Interactive Performance Dashboard',
            title_x=0.5,
            showlegend=False,
            plot_bgcolor='white',
            font=dict(size=10)
        )
        
        # Update axes labels and formatting
        fig.update_xaxes(title_text="Volatility", row=1, col=1, tickformat='.0%')
        fig.update_yaxes(title_text="Return", row=1, col=1, tickformat='.0%')
        
        fig.update_xaxes(title_text="Annualized Return", row=1, col=2, tickformat='.0%')
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        
        fig.update_xaxes(title_text="Deviation Limit", row=2, col=1)
        fig.update_yaxes(title_text="Annualized Return", row=2, col=1, tickformat='.0%')
        
        fig.update_xaxes(title_text="Maximum Drawdown", row=2, col=2, tickformat='.0%')
        fig.update_yaxes(title_text="Annualized Return", row=2, col=2, tickformat='.0%')
        
        fig.update_xaxes(title_text="Investment Start Year", row=3, col=1)
        fig.update_yaxes(title_text="Average Return", row=3, col=1, tickformat='.0%')
        
        fig.update_xaxes(title_text="Deviation Limit", row=3, col=2)
        fig.update_yaxes(title_text="Investment Horizon", row=3, col=2)
        
        return fig
    
    def create_scenario_comparison_chart(self, 
                                       scenarios: List[Dict],
                                       comparison_type: str = "performance") -> go.Figure:
        """Create interactive scenario comparison chart"""
        
        if not scenarios:
            return self._create_empty_figure("No scenarios provided for comparison")
        
        # Extract scenario data
        scenario_data = []
        for scenario in scenarios:
            if scenario.get('simulation_results'):
                sim_results = scenario['simulation_results']
                config = scenario.get('config')
                
                scenario_data.append({
                    'name': config.scenario_id if config else 'Unknown',
                    'start_date': config.start_date if config else 'Unknown',
                    'return': sim_results.get('annualized_return', 0),
                    'volatility': sim_results.get('volatility', 0),
                    'sharpe': sim_results.get('sharpe_ratio', 0),
                    'max_drawdown': sim_results.get('max_drawdown', 0),
                    'total_return': sim_results.get('total_return', 0),
                    'deviation': config.permitted_deviation if config else 0,
                    'horizon': config.investment_horizon_years if config else 0
                })
        
        if not scenario_data:
            return self._create_empty_figure("No valid scenario data for comparison")
        
        df = pd.DataFrame(scenario_data)
        
        if comparison_type == "performance":
            return self._create_performance_comparison(df)
        elif comparison_type == "risk":
            return self._create_risk_comparison(df)
        elif comparison_type == "weights":
            return self._create_weight_comparison(scenarios)
        else:
            return self._create_performance_comparison(df)
    
    def _create_performance_comparison(self, df: pd.DataFrame) -> go.Figure:
        """Create performance comparison chart"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Annualized Returns Comparison',
                'Risk-Adjusted Performance (Sharpe Ratios)',
                'Total Returns vs Maximum Drawdown',
                'Risk-Return Profile'
            ],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Plot 1: Returns Bar Chart
        fig.add_trace(
            go.Bar(
                x=df['name'],
                y=df['return'],
                name='Annualized Return',
                marker_color='lightblue',
                text=[f"{r:.1%}" for r in df['return']],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # Plot 2: Sharpe Ratio Bar Chart
        fig.add_trace(
            go.Bar(
                x=df['name'],
                y=df['sharpe'],
                name='Sharpe Ratio',
                marker_color='lightgreen',
                text=[f"{s:.2f}" for s in df['sharpe']],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        # Plot 3: Total Return vs Drawdown
        fig.add_trace(
            go.Scatter(
                x=df['max_drawdown'],
                y=df['total_return'],
                mode='markers+text',
                marker=dict(size=12, color='red'),
                text=[name[:15] + "..." if len(name) > 15 else name for name in df['name']],
                textposition='top center',
                name='Scenarios'
            ),
            row=2, col=1
        )
        
        # Plot 4: Risk-Return Scatter
        fig.add_trace(
            go.Scatter(
                x=df['volatility'],
                y=df['return'],
                mode='markers+text',
                marker=dict(
                    size=15,
                    color=df['sharpe'],
                    colorscale='Viridis',
                    colorbar=dict(title="Sharpe Ratio"),
                    line=dict(width=2, color='white')
                ),
                text=[name[:10] + "..." if len(name) > 10 else name for name in df['name']],
                textposition='top center',
                name='Risk-Return'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title='Scenario Performance Comparison',
            title_x=0.5,
            showlegend=False
        )
        
        # Update axes formatting
        fig.update_yaxes(tickformat='.1%', row=1, col=1)
        fig.update_yaxes(tickformat='.1%', row=2, col=1)
        fig.update_xaxes(tickformat='.1%', row=2, col=1)
        fig.update_yaxes(tickformat='.1%', row=2, col=2)
        fig.update_xaxes(tickformat='.1%', row=2, col=2)
        
        # Rotate x-axis labels
        fig.update_xaxes(tickangle=45, row=1, col=1)
        fig.update_xaxes(tickangle=45, row=1, col=2)
        
        return fig
    
    def _create_risk_comparison(self, df: pd.DataFrame) -> go.Figure:
        """Create risk comparison chart"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Volatility Comparison',
                'Maximum Drawdown Comparison',
                'Risk Distribution',
                'Risk Metrics Heatmap'
            ],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "violin"}, {"type": "scatter"}]]
        )
        
        # Plot 1: Volatility Bar Chart
        fig.add_trace(
            go.Bar(
                x=df['name'],
                y=df['volatility'],
                name='Volatility',
                marker_color='orange',
                text=[f"{v:.1%}" for v in df['volatility']],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # Plot 2: Max Drawdown Bar Chart
        fig.add_trace(
            go.Bar(
                x=df['name'],
                y=df['max_drawdown'],
                name='Max Drawdown',
                marker_color='red',
                text=[f"{d:.1%}" for d in df['max_drawdown']],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        # Plot 3: Risk Distribution (Violin Plot)
        fig.add_trace(
            go.Violin(
                y=df['volatility'],
                name='Volatility',
                box_visible=True,
                meanline_visible=True
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Violin(
                y=df['max_drawdown'],
                name='Max Drawdown',
                box_visible=True,
                meanline_visible=True
            ),
            row=2, col=1
        )
        
        # Plot 4: Risk Metrics Scatter (Normalized)
        # Normalize metrics for comparison
        vol_norm = (df['volatility'] - df['volatility'].min()) / (df['volatility'].max() - df['volatility'].min())
        dd_norm = (df['max_drawdown'] - df['max_drawdown'].min()) / (df['max_drawdown'].max() - df['max_drawdown'].min())
        
        fig.add_trace(
            go.Scatter(
                x=vol_norm,
                y=dd_norm,
                mode='markers+text',
                marker=dict(
                    size=15,
                    color=df['return'],
                    colorscale='RdYlGn',
                    colorbar=dict(title="Return"),
                    line=dict(width=2, color='white')
                ),
                text=[name[:8] + "..." if len(name) > 8 else name for name in df['name']],
                textposition='top center',
                name='Risk Profile'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title='Scenario Risk Comparison',
            title_x=0.5,
            showlegend=False
        )
        
        # Update axes formatting
        fig.update_yaxes(tickformat='.1%', row=1, col=1)
        fig.update_yaxes(tickformat='.1%', row=1, col=2)
        fig.update_yaxes(tickformat='.1%', row=2, col=1)
        
        # Rotate x-axis labels
        fig.update_xaxes(tickangle=45, row=1, col=1)
        fig.update_xaxes(tickangle=45, row=1, col=2)
        
        return fig
    
    def _create_weight_comparison(self, scenarios: List[Dict]) -> go.Figure:
        """Create weight allocation comparison chart"""
        
        if not scenarios or not any(s.get('perfect_weights') for s in scenarios):
            return self._create_empty_figure("No weight data available for comparison")
        
        # Extract weight data
        weight_data = []
        asset_names = set()
        
        for i, scenario in enumerate(scenarios):
            if scenario.get('perfect_weights'):
                config = scenario.get('config')
                scenario_name = config.scenario_id if config else f'Scenario {i+1}'
                
                for asset, weight in scenario['perfect_weights'].items():
                    weight_data.append({
                        'scenario': scenario_name,
                        'asset': asset,
                        'weight': weight,
                        'neutral_weight': scenario.get('neutral_weights', {}).get(asset, 0)
                    })
                    asset_names.add(asset)
        
        if not weight_data:
            return self._create_empty_figure("No valid weight data for comparison")
        
        df = pd.DataFrame(weight_data)
        df['weight_deviation'] = df['weight'] - df['neutral_weight']
        
        # Create comparison chart
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Perfect Weight Allocations',
                'Weight Deviations from Neutral',
                'Weight Allocation Heatmap',
                'Asset Weight Distribution'
            ],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "violin"}]]
        )
        
        # Plot 1: Stacked Bar Chart of Weights
        scenarios_list = df['scenario'].unique()
        colors = px.colors.qualitative.Set3[:len(asset_names)]
        
        for i, asset in enumerate(sorted(asset_names)):
            asset_data = df[df['asset'] == asset]
            fig.add_trace(
                go.Bar(
                    x=asset_data['scenario'],
                    y=asset_data['weight'],
                    name=asset[:15] + "..." if len(asset) > 15 else asset,
                    marker_color=colors[i % len(colors)]
                ),
                row=1, col=1
            )
        
        # Plot 2: Weight Deviations
        for i, asset in enumerate(sorted(asset_names)):
            asset_data = df[df['asset'] == asset]
            fig.add_trace(
                go.Bar(
                    x=asset_data['scenario'],
                    y=asset_data['weight_deviation'],
                    name=asset[:15] + "..." if len(asset) > 15 else asset,
                    marker_color=colors[i % len(colors)],
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Plot 3: Weight Heatmap (as scatter)
        pivot_data = df.pivot(index='asset', columns='scenario', values='weight')
        
        for i, scenario in enumerate(pivot_data.columns):
            fig.add_trace(
                go.Scatter(
                    x=[scenario] * len(pivot_data.index),
                    y=pivot_data.index,
                    mode='markers+text',
                    marker=dict(
                        size=[w*200 for w in pivot_data[scenario]],  # Scale for visibility
                        color=pivot_data[scenario],
                        colorscale='Blues',
                        colorbar=dict(title="Weight") if i == 0 else None,
                        showscale=i == 0
                    ),
                    text=[f"{w:.1%}" for w in pivot_data[scenario]],
                    textposition='middle center',
                    name=scenario,
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Plot 4: Asset Weight Distribution
        for asset in sorted(asset_names):
            asset_weights = df[df['asset'] == asset]['weight']
            fig.add_trace(
                go.Violin(
                    y=asset_weights,
                    name=asset[:12] + "..." if len(asset) > 12 else asset,
                    box_visible=True,
                    meanline_visible=True,
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1000,
            title='Scenario Weight Allocation Comparison',
            title_x=0.5,
            barmode='stack'
        )
        
        # Update axes formatting
        fig.update_yaxes(tickformat='.0%', row=1, col=1)
        fig.update_yaxes(tickformat='.0%', row=1, col=2)
        fig.update_yaxes(tickformat='.0%', row=2, col=2)
        
        # Add horizontal line at zero for deviations
        fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=2)
        
        # Rotate x-axis labels
        fig.update_xaxes(tickangle=45, row=1, col=1)
        fig.update_xaxes(tickangle=45, row=1, col=2)
        fig.update_xaxes(tickangle=45, row=2, col=1)
        
        return fig
    
    def create_parameter_sensitivity_analysis(self, 
                                            analytics_results: Dict,
                                            mandate: str) -> go.Figure:
        """Create parameter sensitivity analysis chart"""
        
        all_results = analytics_results.get('all_results', [])
        successful_results = [r for r in all_results if r.get('success', False)]
        
        if not successful_results:
            return self._create_empty_figure("No successful scenarios for sensitivity analysis")
        
        # Extract parameter data
        param_data = []
        for result in successful_results:
            sim_results = result.get('simulation_results', {})
            config = result.get('config')
            
            if sim_results and config:
                param_data.append({
                    'deviation': config.permitted_deviation,
                    'horizon': config.investment_horizon_years,
                    'rebalancing': config.rebalancing_frequency,
                    'transaction_cost': config.transaction_cost_bps,
                    'return': sim_results.get('annualized_return', 0),
                    'volatility': sim_results.get('volatility', 0),
                    'sharpe': sim_results.get('sharpe_ratio', 0),
                    'max_drawdown': sim_results.get('max_drawdown', 0)
                })
        
        df = pd.DataFrame(param_data)
        
        # Create sensitivity analysis
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Return Sensitivity to Deviation Limit',
                'Return Sensitivity to Transaction Costs',
                'Sharpe Ratio by Investment Horizon',
                'Parameter Interaction Effects'
            ]
        )
        
        # Plot 1: Return by Deviation
        deviation_analysis = df.groupby('deviation').agg({
            'return': ['mean', 'std', 'count'],
            'sharpe': 'mean',
            'volatility': 'mean'
        }).reset_index()
        
        fig.add_trace(
            go.Scatter(
                x=deviation_analysis['deviation'],
                y=deviation_analysis[('return', 'mean')],
                mode='lines+markers',
                marker=dict(size=10, color='blue'),
                line=dict(width=3),
                error_y=dict(
                    type='data',
                    array=deviation_analysis[('return', 'std')],
                    visible=True
                ),
                name='Mean Return'
            ),
            row=1, col=1
        )
        
        # Plot 2: Return by Transaction Cost
        tx_cost_analysis = df.groupby('transaction_cost').agg({
            'return': ['mean', 'std'],
            'sharpe': 'mean'
        }).reset_index()
        
        fig.add_trace(
            go.Scatter(
                x=tx_cost_analysis['transaction_cost'],
                y=tx_cost_analysis[('return', 'mean')],
                mode='lines+markers',
                marker=dict(size=10, color='red'),
                line=dict(width=3),
                error_y=dict(
                    type='data',
                    array=tx_cost_analysis[('return', 'std')],
                    visible=True
                ),
                name='Mean Return by TX Cost'
            ),
            row=1, col=2
        )
        
        # Plot 3: Sharpe by Horizon and Rebalancing
        for rebal_freq in df['rebalancing'].unique():
            subset = df[df['rebalancing'] == rebal_freq]
            horizon_analysis = subset.groupby('horizon')['sharpe'].mean()
            
            fig.add_trace(
                go.Bar(
                    x=horizon_analysis.index,
                    y=horizon_analysis.values,
                    name=rebal_freq,
                    text=[f"{s:.2f}" for s in horizon_analysis.values],
                    textposition='auto'
                ),
                row=2, col=1
            )
        
        # Plot 4: 3D Parameter Space (as bubble chart)
        param_summary = df.groupby(['deviation', 'horizon', 'rebalancing']).agg({
            'return': 'mean',
            'sharpe': 'mean',
            'volatility': 'mean'
        }).reset_index()
        
        for rebal in param_summary['rebalancing'].unique():
            subset = param_summary[param_summary['rebalancing'] == rebal]
            fig.add_trace(
                go.Scatter(
                    x=subset['deviation'],
                    y=subset['horizon'],
                    mode='markers',
                    marker=dict(
                        size=subset['return'] * 300,  # Scale for visibility
                        color=subset['sharpe'],
                        colorscale='RdYlGn',
                        colorbar=dict(title="Sharpe Ratio") if rebal == param_summary['rebalancing'].iloc[0] else None,
                        showscale=rebal == param_summary['rebalancing'].iloc[0],
                        line=dict(width=2, color='white')
                    ),
                    name=rebal,
                    hovertemplate=f'Rebalancing: {rebal}<br>' +
                                 'Deviation: %{x:.0%}<br>' +
                                 'Horizon: %{y} years<br>' +
                                 'Avg Return: %{marker.size:.1%}<br>' +
                                 'Avg Sharpe: %{marker.color:.2f}<extra></extra>'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            title=f'{mandate} PHP: Parameter Sensitivity Analysis',
            title_x=0.5,
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(title_text="Deviation Limit", tickformat='.0%', row=1, col=1)
        fig.update_yaxes(title_text="Average Return", tickformat='.0%', row=1, col=1)
        
        fig.update_xaxes(title_text="Transaction Cost (bps)", row=1, col=2)
        fig.update_yaxes(title_text="Average Return", tickformat='.0%', row=1, col=2)
        
        fig.update_xaxes(title_text="Investment Horizon", row=2, col=1)
        fig.update_yaxes(title_text="Average Sharpe Ratio", row=2, col=1)
        
        fig.update_xaxes(title_text="Deviation Limit", tickformat='.0%', row=2, col=2)
        fig.update_yaxes(title_text="Investment Horizon", row=2, col=2)
        
        return fig
    
    def _create_empty_figure(self, message: str) -> go.Figure:
        """Create empty figure with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor='white'
        )
        return fig
    
    def save_dashboard_html(self, fig: go.Figure, filename: str, title: str = "PHP Dashboard"):
        """Save dashboard as HTML file"""
        
        output_path = Path("reports/html") / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure HTML output
        config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': filename.replace('.html', ''),
                'height': 800,
                'width': 1200,
                'scale': 2
            }
        }
        
        # Save with custom layout
        fig.update_layout(
            title=title,
            title_x=0.5,
            font=dict(family="Arial, sans-serif", size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='white'
        )
        
        pyo.plot(fig, filename=str(output_path), auto_open=False, config=config)
        
        print(f"📊 Dashboard saved: {output_path}")
        return str(output_path)

# Streamlit App Components (if running with streamlit)
def create_streamlit_app():
    """Create Streamlit web application for PHP analysis"""
    
    st.set_page_config(
        page_title="PHP Analysis Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📊 Perfect Hindsight Portfolio Analysis Dashboard")
    st.markdown("---")
    
    # Initialize dashboard
    if 'dashboard' not in st.session_state:
        st.session_state.dashboard = InteractiveDashboard()
    
    dashboard = st.session_state.dashboard
    
    # Sidebar controls
    st.sidebar.header("Analysis Parameters")
    
    mandate = st.sidebar.selectbox(
        "Select Mandate",
        options=["6040", "EQFI", "EQFILA", "EQFILAIA"],
        index=1
    )
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        analysis_start = st.date_input("Analysis Start", value=pd.to_datetime("2010-01-01"))
    with col2:
        analysis_end = st.date_input("Analysis End", value=pd.to_datetime("2020-01-01"))
    
    sample_size = st.sidebar.slider("Sample Size (scenarios)", min_value=10, max_value=1000, value=100)
    
    # Generate analysis button
    if st.sidebar.button("🚀 Run Analysis", type="primary"):
        with st.spinner(f"Generating PHP analysis for {mandate}..."):
            try:
                # Generate report
                report = dashboard.reporter.generate_complete_php_report(
                    mandate=mandate,
                    analysis_start=analysis_start.strftime('%Y-%m-%d'),
                    analysis_end=analysis_end.strftime('%Y-%m-%d'),
                    sample_size=sample_size
                )
                
                st.session_state.current_report = report
                st.session_state.current_analytics = report['analytics_results']
                
                st.success(f"✅ Analysis completed! {report['total_scenarios']} scenarios analyzed.")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.session_state.current_report = None
    
    # Display results if available
    if hasattr(st.session_state, 'current_report') and st.session_state.current_report:
        report = st.session_state.current_report
        analytics = st.session_state.current_analytics
        
        # Executive Summary
        st.header("📋 Executive Summary")
        summary = report['executive_summary']
        
        col1, col2, col3, col4 = st.columns(4)
        
        summary_stats = summary.get('summary_statistics', {})
        with col1:
            avg_return = summary_stats.get('annualized_return', {}).get('mean', 0)
            st.metric("Average Return", f"{avg_return:.1%}")
        
        with col2:
            avg_volatility = summary_stats.get('volatility', {}).get('mean', 0)
            st.metric("Average Volatility", f"{avg_volatility:.1%}")
        
        with col3:
            avg_sharpe = summary_stats.get('sharpe_ratio', {}).get('mean', 0)
            st.metric("Average Sharpe", f"{avg_sharpe:.2f}")
        
        with col4:
            success_rate = summary['successful_scenarios'] / summary['total_scenarios_analyzed']
            st.metric("Success Rate", f"{success_rate:.1%}")
        
        # Key Findings
        st.subheader("🔍 Key Findings")
        for finding in summary['key_findings']:
            st.write(f"• {finding}")
        
        # Interactive Dashboard
        st.header("📊 Interactive Performance Dashboard")
        
        dashboard_fig = dashboard.create_interactive_performance_dashboard(analytics, mandate)
        st.plotly_chart(dashboard_fig, use_container_width=True)
        
        # Scenario Comparison
        st.header("⚖️ Scenario Comparison")
        
        # Let user select scenarios for comparison
        top_performers = analytics.get('top_performers', {}).get('top_10', [])
        if top_performers:
            comparison_scenarios = top_performers[:5]  # Top 5 for comparison
            
            comparison_type = st.radio(
                "Comparison Type",
                options=["performance", "risk", "weights"],
                horizontal=True
            )
            
            comparison_fig = dashboard.create_scenario_comparison_chart(
                comparison_scenarios, comparison_type
            )
            st.plotly_chart(comparison_fig, use_container_width=True)
        
        # Parameter Sensitivity
        st.header("🎯 Parameter Sensitivity Analysis")
        
        sensitivity_fig = dashboard.create_parameter_sensitivity_analysis(analytics, mandate)
        st.plotly_chart(sensitivity_fig, use_container_width=True)
        
        # Download Reports
        st.header("💾 Download Reports")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Download Executive Summary"):
                summary_json = json.dumps(summary, indent=2, default=str)
                st.download_button(
                    label="Download JSON",
                    data=summary_json,
                    file_name=f"{mandate}_executive_summary.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📊 Save Dashboard HTML"):
                html_path = dashboard.save_dashboard_html(
                    dashboard_fig, 
                    f"{mandate}_dashboard.html",
                    f"{mandate} PHP Analysis Dashboard"
                )
                st.success(f"Dashboard saved: {html_path}")
        
        with col3:
            if st.button("📈 Save Sensitivity HTML"):
                html_path = dashboard.save_dashboard_html(
                    sensitivity_fig,
                    f"{mandate}_sensitivity.html", 
                    f"{mandate} Parameter Sensitivity Analysis"
                )
                st.success(f"Sensitivity analysis saved: {html_path}")

if __name__ == "__main__":
    print("🚀 PHP Interactive Dashboard Module")
    print("Run with: streamlit run interactive_dashboard.py")
    
    # Test dashboard creation
    dashboard = InteractiveDashboard()
    print("✅ Dashboard initialized successfully")