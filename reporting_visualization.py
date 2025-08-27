import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import warnings
from pathlib import Path
import json
from dataclasses import asdict
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from data_mapping import DataMapper
from portfolio_config import PortfolioConfig
from perfect_weight_calculator import PerfectWeightCalculator
from portfolio_simulation import PortfolioSimulation, SimulationConfig
from performance_analytics import PerformanceAnalytics

class PHPReporting:
    """Module for generating comprehensive reports and visualizations for PHP analysis"""
    
    def __init__(self, 
                 data_mapper: DataMapper = None,
                 portfolio_config: PortfolioConfig = None,
                 perfect_weight_calc: PerfectWeightCalculator = None,
                 simulation_engine: PortfolioSimulation = None,
                 analytics_engine: PerformanceAnalytics = None):
        
        self.data_mapper = data_mapper or DataMapper()
        self.portfolio_config = portfolio_config or PortfolioConfig(self.data_mapper)
        self.perfect_weight_calc = perfect_weight_calc or PerfectWeightCalculator(
            self.data_mapper, self.portfolio_config)
        self.simulation_engine = simulation_engine or PortfolioSimulation(
            self.data_mapper, self.portfolio_config, self.perfect_weight_calc)
        self.analytics_engine = analytics_engine or PerformanceAnalytics()
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create output directories
        self.output_dir = Path("reports")
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "charts").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        (self.output_dir / "html").mkdir(exist_ok=True)
    
    def generate_complete_php_report(self, 
                                   mandate: str,
                                   analysis_start: str = "1996-01-01",
                                   analysis_end: str = "2025-05-31",
                                   sample_size: int = None) -> Dict[str, Any]:
        """
        Generate complete PHP analysis report for a mandate
        
        Args:
            mandate: Portfolio mandate to analyze
            analysis_start: Analysis start date
            analysis_end: Analysis end date  
            sample_size: Limit number of scenarios for testing (None = all scenarios)
        
        Returns:
            Dictionary containing all report components
        """
        print(f"🚀 Generating Complete PHP Report for {mandate}")
        print(f"   Analysis Period: {analysis_start} to {analysis_end}")
        
        # Step 1: Generate all scenarios
        print("📊 Step 1: Generating scenarios...")
        all_scenarios = self.simulation_engine.generate_all_php_scenarios(
            mandate, analysis_start, analysis_end
        )
        
        if sample_size and len(all_scenarios) > sample_size:
            print(f"   Using sample of {sample_size} scenarios (total: {len(all_scenarios)})")
            scenarios = all_scenarios[:sample_size]
        else:
            scenarios = all_scenarios
            print(f"   Processing all {len(scenarios)} scenarios")
        
        # Step 2: Run simulations
        print("⚙️ Step 2: Running simulations...")
        simulation_results = self.simulation_engine.run_scenario_batch(scenarios)
        
        # Step 3: Run analytics
        print("📈 Step 3: Running comprehensive analytics...")
        analytics_results = self.analytics_engine.run_comprehensive_analysis_from_results(
            simulation_results, f"{mandate}_comprehensive"
        )
        
        # Step 4: Generate all report components
        print("📋 Step 4: Generating report components...")
        
        report_components = {
            'mandate': mandate,
            'analysis_period': f"{analysis_start} to {analysis_end}",
            'total_scenarios': len(scenarios),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'analytics_results': analytics_results,
            'simulation_results': simulation_results,
            'executive_summary': self._generate_executive_summary(analytics_results, mandate),
            'performance_charts': self._create_performance_visualizations(analytics_results, mandate),
            'comparison_tables': self._create_comparison_tables(analytics_results, mandate),
            'scenario_deep_dive': self._create_scenario_deep_dive(analytics_results, mandate),
            'risk_analysis': self._create_risk_analysis_report(analytics_results, mandate)
        }
        
        # Step 5: Export reports
        print("💾 Step 5: Exporting reports...")
        self._export_all_reports(report_components)
        
        print(f"✅ Complete PHP Report Generated for {mandate}")
        print(f"   📁 Reports saved to: {self.output_dir}")
        
        return report_components
    
    def _generate_executive_summary(self, analytics_results: Dict, mandate: str) -> Dict[str, Any]:
        """Generate executive summary from analytics results"""
        
        summary_stats = analytics_results.get('summary_statistics', {})
        best_scenarios = analytics_results.get('top_performers', {}).get('top_10', [])
        param_analysis = analytics_results.get('parameter_analysis', {})
        
        # Key findings
        key_findings = []
        
        # Performance findings
        avg_return = summary_stats.get('annualized_return', {}).get('mean', 0)
        avg_excess = summary_stats.get('excess_return', {}).get('mean', 0)
        best_return = max([s.get('annualized_return', 0) for s in best_scenarios]) if best_scenarios else 0
        
        key_findings.append(f"Average PHP annualized return: {avg_return:.1%}")
        key_findings.append(f"Average excess return vs benchmark: {avg_excess:.1%}")
        key_findings.append(f"Best scenario achieved: {best_return:.1%} annualized")
        
        # Risk findings
        avg_volatility = summary_stats.get('volatility', {}).get('mean', 0)
        avg_max_dd = summary_stats.get('max_drawdown', {}).get('mean', 0)
        avg_sharpe = summary_stats.get('sharpe_ratio', {}).get('mean', 0)
        
        key_findings.append(f"Average volatility: {avg_volatility:.1%}")
        key_findings.append(f"Average maximum drawdown: {avg_max_dd:.1%}")
        key_findings.append(f"Average Sharpe ratio: {avg_sharpe:.2f}")
        
        # Parameter findings
        if param_analysis:
            best_deviation = param_analysis.get('optimal_deviation', 'N/A')
            best_horizon = param_analysis.get('optimal_horizon', 'N/A')
            best_rebalancing = param_analysis.get('optimal_rebalancing', 'N/A')
            
            key_findings.append(f"Optimal deviation limit: {best_deviation}")
            key_findings.append(f"Optimal investment horizon: {best_horizon}")
            key_findings.append(f"Optimal rebalancing frequency: {best_rebalancing}")
        
        # Success rate
        successful_scenarios = len([r for r in analytics_results.get('all_results', []) 
                                  if r.get('success', False)])
        total_scenarios = len(analytics_results.get('all_results', []))
        success_rate = successful_scenarios / total_scenarios if total_scenarios > 0 else 0
        
        key_findings.append(f"Scenario success rate: {success_rate:.1%}")
        
        return {
            'mandate': mandate,
            'analysis_date': datetime.now().strftime('%Y-%m-%d'),
            'total_scenarios_analyzed': total_scenarios,
            'successful_scenarios': successful_scenarios,
            'key_findings': key_findings,
            'summary_statistics': summary_stats,
            'methodology_notes': [
                "PHP uses perfect hindsight to allocate weights based on realized asset performance",
                "Transaction costs include 2x impact (buy and sell) as specified",
                "Hedge fund data forward-filled from monthly to daily frequency",
                "Weight constraints use absolute deviations from neutral weights",
                "Risk-free rate assumed at 2% for excess return calculations"
            ]
        }
    
    def _create_performance_visualizations(self, analytics_results: Dict, mandate: str) -> Dict[str, str]:
        """Create all performance visualization charts"""
        
        chart_files = {}
        
        try:
            # Chart 1: Return Distribution
            chart_files['return_distribution'] = self._plot_return_distribution(
                analytics_results, mandate
            )
            
            # Chart 2: Risk-Return Scatter
            chart_files['risk_return_scatter'] = self._plot_risk_return_scatter(
                analytics_results, mandate
            )
            
            # Chart 3: Parameter Heatmaps
            chart_files['parameter_heatmaps'] = self._plot_parameter_heatmaps(
                analytics_results, mandate
            )
            
            # Chart 4: Time Series Performance
            chart_files['time_series'] = self._plot_time_series_performance(
                analytics_results, mandate
            )
            
            # Chart 5: Drawdown Analysis
            chart_files['drawdown_analysis'] = self._plot_drawdown_analysis(
                analytics_results, mandate
            )
            
        except Exception as e:
            print(f"Warning: Error creating visualizations: {e}")
            chart_files['error'] = str(e)
        
        return chart_files
    
    def _plot_return_distribution(self, analytics_results: Dict, mandate: str) -> str:
        """Create return distribution histogram"""
        
        # Extract returns from all successful scenarios
        all_results = analytics_results.get('all_results', [])
        successful_results = [r for r in all_results if r.get('success', False)]
        
        if not successful_results:
            return "No successful scenarios to plot"
        
        returns = []
        excess_returns = []
        
        for result in successful_results:
            sim_results = result.get('simulation_results', {})
            if sim_results:
                returns.append(sim_results.get('annualized_return', 0))
                # Calculate excess return vs benchmark (assuming 6040 benchmark)
                benchmark_return = 0.08  # Placeholder - should be calculated
                excess_returns.append(sim_results.get('annualized_return', 0) - benchmark_return)
        
        # Create subplot figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Annualized Returns
        ax1.hist(returns, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(np.mean(returns), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(returns):.1%}')
        ax1.axvline(np.median(returns), color='orange', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(returns):.1%}')
        ax1.set_xlabel('Annualized Return')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'{mandate} PHP: Annualized Return Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add summary statistics
        stats_text = f'Scenarios: {len(returns)}\n'
        stats_text += f'Mean: {np.mean(returns):.1%}\n'
        stats_text += f'Std Dev: {np.std(returns):.1%}\n'
        stats_text += f'Min: {np.min(returns):.1%}\n'
        stats_text += f'Max: {np.max(returns):.1%}'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 2: Excess Returns
        ax2.hist(excess_returns, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(np.mean(excess_returns), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(excess_returns):.1%}')
        ax2.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5, 
                   label='Zero Excess')
        ax2.set_xlabel('Excess Return vs Benchmark')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'{mandate} PHP: Excess Return Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = f"{mandate}_return_distribution.png"
        filepath = self.output_dir / "charts" / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _plot_risk_return_scatter(self, analytics_results: Dict, mandate: str) -> str:
        """Create risk-return scatter plot"""
        
        all_results = analytics_results.get('all_results', [])
        successful_results = [r for r in all_results if r.get('success', False)]
        
        if not successful_results:
            return "No successful scenarios to plot"
        
        returns = []
        volatilities = []
        sharpe_ratios = []
        deviations = []
        horizons = []
        
        for result in successful_results:
            sim_results = result.get('simulation_results', {})
            config = result.get('config')
            
            if sim_results and config:
                returns.append(sim_results.get('annualized_return', 0))
                volatilities.append(sim_results.get('volatility', 0))
                sharpe_ratios.append(sim_results.get('sharpe_ratio', 0))
                deviations.append(config.permitted_deviation)
                horizons.append(config.investment_horizon_years)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # Plot 1: Risk-Return Scatter colored by Sharpe Ratio
        ax1 = plt.subplot(2, 2, 1)
        scatter = ax1.scatter(volatilities, returns, c=sharpe_ratios, cmap='viridis', 
                            alpha=0.6, s=30)
        ax1.set_xlabel('Volatility (Annualized)')
        ax1.set_ylabel('Return (Annualized)')
        ax1.set_title(f'{mandate} PHP: Risk-Return Profile (Colored by Sharpe Ratio)')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Sharpe Ratio')
        
        # Plot 2: Colored by Deviation Limit
        ax2 = plt.subplot(2, 2, 2)
        deviation_colors = [d*100 for d in deviations]  # Convert to percentage for coloring
        scatter2 = ax2.scatter(volatilities, returns, c=deviation_colors, cmap='plasma',
                             alpha=0.6, s=30)
        ax2.set_xlabel('Volatility (Annualized)')
        ax2.set_ylabel('Return (Annualized)')
        ax2.set_title(f'{mandate} PHP: Risk-Return Profile (Colored by Deviation Limit)')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='Deviation Limit (%)')
        
        # Plot 3: Colored by Investment Horizon
        ax3 = plt.subplot(2, 2, 3)
        scatter3 = ax3.scatter(volatilities, returns, c=horizons, cmap='coolwarm',
                             alpha=0.6, s=30)
        ax3.set_xlabel('Volatility (Annualized)')
        ax3.set_ylabel('Return (Annualized)')
        ax3.set_title(f'{mandate} PHP: Risk-Return Profile (Colored by Horizon)')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter3, ax=ax3, label='Investment Horizon (Years)')
        
        # Plot 4: Sharpe Ratio vs Volatility
        ax4 = plt.subplot(2, 2, 4)
        ax4.scatter(volatilities, sharpe_ratios, alpha=0.6, s=30, color='red')
        ax4.set_xlabel('Volatility (Annualized)')
        ax4.set_ylabel('Sharpe Ratio')
        ax4.set_title(f'{mandate} PHP: Sharpe Ratio vs Volatility')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        
        filename = f"{mandate}_risk_return_analysis.png"
        filepath = self.output_dir / "charts" / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _plot_parameter_heatmaps(self, analytics_results: Dict, mandate: str) -> str:
        """Create parameter analysis heatmaps"""
        
        all_results = analytics_results.get('all_results', [])
        successful_results = [r for r in all_results if r.get('success', False)]
        
        if not successful_results:
            return "No successful scenarios to plot"
        
        # Create DataFrame for analysis
        data_rows = []
        for result in successful_results:
            sim_results = result.get('simulation_results', {})
            config = result.get('config')
            
            if sim_results and config:
                data_rows.append({
                    'deviation': config.permitted_deviation,
                    'horizon': config.investment_horizon_years,
                    'rebalancing': config.rebalancing_frequency,
                    'transaction_cost_bps': config.transaction_cost_bps,
                    'return': sim_results.get('annualized_return', 0),
                    'volatility': sim_results.get('volatility', 0),
                    'sharpe': sim_results.get('sharpe_ratio', 0),
                    'max_drawdown': sim_results.get('max_drawdown', 0)
                })
        
        df = pd.DataFrame(data_rows)
        
        # Create heatmaps
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # Heatmap 1: Average Return by Deviation and Horizon
        pivot1 = df.groupby(['deviation', 'horizon'])['return'].mean().unstack()
        sns.heatmap(pivot1, annot=True, fmt='.1%', cmap='RdYlGn', ax=axes[0,0])
        axes[0,0].set_title(f'{mandate} PHP: Average Return by Deviation & Horizon')
        axes[0,0].set_xlabel('Investment Horizon (Years)')
        axes[0,0].set_ylabel('Deviation Limit')
        
        # Heatmap 2: Average Sharpe Ratio by Deviation and Rebalancing
        pivot2 = df.groupby(['deviation', 'rebalancing'])['sharpe'].mean().unstack()
        sns.heatmap(pivot2, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[0,1])
        axes[0,1].set_title(f'{mandate} PHP: Average Sharpe Ratio by Deviation & Rebalancing')
        axes[0,1].set_xlabel('Rebalancing Frequency')
        axes[0,1].set_ylabel('Deviation Limit')
        
        # Heatmap 3: Average Volatility by Horizon and Transaction Cost
        pivot3 = df.groupby(['horizon', 'transaction_cost_bps'])['volatility'].mean().unstack()
        sns.heatmap(pivot3, annot=True, fmt='.1%', cmap='RdYlBu_r', ax=axes[1,0])
        axes[1,0].set_title(f'{mandate} PHP: Average Volatility by Horizon & Transaction Cost')
        axes[1,0].set_xlabel('Transaction Cost (bps)')
        axes[1,0].set_ylabel('Investment Horizon (Years)')
        
        # Heatmap 4: Average Max Drawdown by Deviation and Transaction Cost
        pivot4 = df.groupby(['deviation', 'transaction_cost_bps'])['max_drawdown'].mean().unstack()
        sns.heatmap(pivot4, annot=True, fmt='.1%', cmap='RdYlBu', ax=axes[1,1])
        axes[1,1].set_title(f'{mandate} PHP: Average Max Drawdown by Deviation & Transaction Cost')
        axes[1,1].set_xlabel('Transaction Cost (bps)')
        axes[1,1].set_ylabel('Deviation Limit')
        
        plt.tight_layout()
        
        filename = f"{mandate}_parameter_heatmaps.png"
        filepath = self.output_dir / "charts" / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _plot_time_series_performance(self, analytics_results: Dict, mandate: str) -> str:
        """Create time series performance analysis"""
        
        # This would require extracting portfolio value time series from individual scenarios
        # For now, create a placeholder that shows performance by start date
        
        all_results = analytics_results.get('all_results', [])
        successful_results = [r for r in all_results if r.get('success', False)]
        
        if not successful_results:
            return "No successful scenarios to plot"
        
        # Group results by start year and calculate average performance
        performance_by_year = {}
        
        for result in successful_results:
            config = result.get('config')
            sim_results = result.get('simulation_results', {})
            
            if config and sim_results:
                start_year = int(config.start_date[:4])
                if start_year not in performance_by_year:
                    performance_by_year[start_year] = []
                
                performance_by_year[start_year].append({
                    'return': sim_results.get('annualized_return', 0),
                    'volatility': sim_results.get('volatility', 0),
                    'sharpe': sim_results.get('sharpe_ratio', 0),
                    'max_drawdown': sim_results.get('max_drawdown', 0)
                })
        
        # Calculate averages by year
        years = sorted(performance_by_year.keys())
        avg_returns = []
        avg_volatilities = []
        avg_sharpes = []
        avg_drawdowns = []
        
        for year in years:
            year_data = performance_by_year[year]
            avg_returns.append(np.mean([d['return'] for d in year_data]))
            avg_volatilities.append(np.mean([d['volatility'] for d in year_data]))
            avg_sharpes.append(np.mean([d['sharpe'] for d in year_data]))
            avg_drawdowns.append(np.mean([d['max_drawdown'] for d in year_data]))
        
        # Create time series plots
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # Plot 1: Average Returns by Start Year
        axes[0,0].plot(years, avg_returns, marker='o', linewidth=2, markersize=6)
        axes[0,0].set_xlabel('Investment Start Year')
        axes[0,0].set_ylabel('Average Annualized Return')
        axes[0,0].set_title(f'{mandate} PHP: Average Returns by Investment Start Year')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        # Plot 2: Average Volatility by Start Year
        axes[0,1].plot(years, avg_volatilities, marker='s', linewidth=2, markersize=6, color='red')
        axes[0,1].set_xlabel('Investment Start Year')
        axes[0,1].set_ylabel('Average Volatility')
        axes[0,1].set_title(f'{mandate} PHP: Average Volatility by Investment Start Year')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        # Plot 3: Average Sharpe Ratio by Start Year
        axes[1,0].plot(years, avg_sharpes, marker='^', linewidth=2, markersize=6, color='green')
        axes[1,0].set_xlabel('Investment Start Year')
        axes[1,0].set_ylabel('Average Sharpe Ratio')
        axes[1,0].set_title(f'{mandate} PHP: Average Sharpe Ratio by Investment Start Year')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Plot 4: Average Max Drawdown by Start Year
        axes[1,1].plot(years, avg_drawdowns, marker='D', linewidth=2, markersize=6, color='purple')
        axes[1,1].set_xlabel('Investment Start Year')
        axes[1,1].set_ylabel('Average Maximum Drawdown')
        axes[1,1].set_title(f'{mandate} PHP: Average Max Drawdown by Investment Start Year')
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        plt.tight_layout()
        
        filename = f"{mandate}_time_series_performance.png"
        filepath = self.output_dir / "charts" / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _plot_drawdown_analysis(self, analytics_results: Dict, mandate: str) -> str:
        """Create drawdown analysis charts"""
        
        all_results = analytics_results.get('all_results', [])
        successful_results = [r for r in all_results if r.get('success', False)]
        
        if not successful_results:
            return "No successful scenarios to plot"
        
        # Extract drawdown data
        max_drawdowns = []
        returns = []
        horizons = []
        deviations = []
        
        for result in successful_results:
            sim_results = result.get('simulation_results', {})
            config = result.get('config')
            
            if sim_results and config:
                max_drawdowns.append(sim_results.get('max_drawdown', 0))
                returns.append(sim_results.get('annualized_return', 0))
                horizons.append(config.investment_horizon_years)
                deviations.append(config.permitted_deviation)
        
        # Create drawdown analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # Plot 1: Drawdown Distribution
        axes[0,0].hist(max_drawdowns, bins=50, alpha=0.7, color='red', edgecolor='black')
        axes[0,0].axvline(np.mean(max_drawdowns), color='blue', linestyle='--', linewidth=2,
                         label=f'Mean: {np.mean(max_drawdowns):.1%}')
        axes[0,0].axvline(np.median(max_drawdowns), color='green', linestyle='--', linewidth=2,
                         label=f'Median: {np.median(max_drawdowns):.1%}')
        axes[0,0].set_xlabel('Maximum Drawdown')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title(f'{mandate} PHP: Maximum Drawdown Distribution')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        # Plot 2: Return vs Drawdown
        axes[0,1].scatter(max_drawdowns, returns, alpha=0.6, s=30)
        axes[0,1].set_xlabel('Maximum Drawdown')
        axes[0,1].set_ylabel('Annualized Return')
        axes[0,1].set_title(f'{mandate} PHP: Return vs Maximum Drawdown')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        axes[0,1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        # Calculate and display correlation
        correlation = np.corrcoef(max_drawdowns, returns)[0,1]
        axes[0,1].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                      transform=axes[0,1].transAxes, fontsize=12,
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 3: Drawdown by Horizon
        horizon_5_dd = [dd for dd, h in zip(max_drawdowns, horizons) if h == 5]
        horizon_10_dd = [dd for dd, h in zip(max_drawdowns, horizons) if h == 10]
        
        axes[1,0].hist([horizon_5_dd, horizon_10_dd], bins=30, alpha=0.7, 
                      label=['5-Year', '10-Year'], color=['blue', 'orange'])
        axes[1,0].set_xlabel('Maximum Drawdown')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title(f'{mandate} PHP: Drawdown Distribution by Investment Horizon')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        # Plot 4: Calmar Ratio (Return/Max Drawdown)
        calmar_ratios = [r/dd if dd > 0 else 0 for r, dd in zip(returns, max_drawdowns)]
        axes[1,1].hist(calmar_ratios, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[1,1].axvline(np.mean(calmar_ratios), color='red', linestyle='--', linewidth=2,
                         label=f'Mean: {np.mean(calmar_ratios):.2f}')
        axes[1,1].set_xlabel('Calmar Ratio (Return/Max Drawdown)')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title(f'{mandate} PHP: Calmar Ratio Distribution')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = f"{mandate}_drawdown_analysis.png"
        filepath = self.output_dir / "charts" / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _create_comparison_tables(self, analytics_results: Dict, mandate: str) -> Dict[str, pd.DataFrame]:
        """Create comparison tables for different scenarios"""
        
        tables = {}
        
        # Table 1: Top/Bottom Performers
        top_performers = analytics_results.get('top_performers', {}).get('top_10', [])
        bottom_performers = analytics_results.get('bottom_performers', {}).get('bottom_10', [])
        
        if top_performers:
            top_df = pd.DataFrame([
                {
                    'Scenario': r['config'].scenario_id if r.get('config') else 'Unknown',
                    'Return': r.get('annualized_return', 0),
                    'Volatility': r.get('volatility', 0),
                    'Sharpe': r.get('sharpe_ratio', 0),
                    'Max Drawdown': r.get('max_drawdown', 0),
                    'Deviation': r['config'].permitted_deviation if r.get('config') else 0,
                    'Horizon': f"{r['config'].investment_horizon_years}Y" if r.get('config') else 'Unknown'
                }
                for r in top_performers
            ])
            tables['top_performers'] = top_df
        
        if bottom_performers:
            bottom_df = pd.DataFrame([
                {
                    'Scenario': r['config'].scenario_id if r.get('config') else 'Unknown',
                    'Return': r.get('annualized_return', 0),
                    'Volatility': r.get('volatility', 0),
                    'Sharpe': r.get('sharpe_ratio', 0),
                    'Max Drawdown': r.get('max_drawdown', 0),
                    'Deviation': r['config'].permitted_deviation if r.get('config') else 0,
                    'Horizon': f"{r['config'].investment_horizon_years}Y" if r.get('config') else 'Unknown'
                }
                for r in bottom_performers
            ])
            tables['bottom_performers'] = bottom_df
        
        # Table 2: Summary by Parameters
        all_results = analytics_results.get('all_results', [])
        successful_results = [r for r in all_results if r.get('success', False)]
        
        if successful_results:
            param_data = []
            for result in successful_results:
                sim_results = result.get('simulation_results', {})
                config = result.get('config')
                
                if sim_results and config:
                    param_data.append({
                        'Deviation': f"{config.permitted_deviation:.0%}",
                        'Horizon': f"{config.investment_horizon_years}Y",
                        'Rebalancing': config.rebalancing_frequency,
                        'Transaction_Cost': f"{config.transaction_cost_bps}bps",
                        'Return': sim_results.get('annualized_return', 0),
                        'Volatility': sim_results.get('volatility', 0),
                        'Sharpe': sim_results.get('sharpe_ratio', 0),
                        'Max_Drawdown': sim_results.get('max_drawdown', 0)
                    })
            
            param_df = pd.DataFrame(param_data)
            
            # Create summary by deviation
            deviation_summary = param_df.groupby('Deviation').agg({
                'Return': ['mean', 'std', 'min', 'max'],
                'Volatility': ['mean', 'std'],
                'Sharpe': ['mean', 'std'],
                'Max_Drawdown': ['mean', 'max']
            }).round(4)
            
            tables['deviation_summary'] = deviation_summary
            
            # Create summary by horizon
            horizon_summary = param_df.groupby('Horizon').agg({
                'Return': ['mean', 'std', 'min', 'max'],
                'Volatility': ['mean', 'std'],
                'Sharpe': ['mean', 'std'],
                'Max_Drawdown': ['mean', 'max']
            }).round(4)
            
            tables['horizon_summary'] = horizon_summary
        
        return tables
    
    def _create_scenario_deep_dive(self, analytics_results: Dict, mandate: str) -> Dict[str, Any]:
        """Create detailed analysis of specific scenarios"""
        
        deep_dive = {}
        
        # Best and worst scenario analysis
        top_performers = analytics_results.get('top_performers', {}).get('top_10', [])
        bottom_performers = analytics_results.get('bottom_performers', {}).get('bottom_10', [])
        
        if top_performers:
            best_scenario = top_performers[0]
            deep_dive['best_scenario'] = {
                'scenario_id': best_scenario['config'].scenario_id if best_scenario.get('config') else 'Unknown',
                'performance_metrics': {
                    'annualized_return': best_scenario.get('annualized_return', 0),
                    'volatility': best_scenario.get('volatility', 0),
                    'sharpe_ratio': best_scenario.get('sharpe_ratio', 0),
                    'max_drawdown': best_scenario.get('max_drawdown', 0)
                },
                'configuration': {
                    'deviation_limit': best_scenario['config'].permitted_deviation if best_scenario.get('config') else 'Unknown',
                    'investment_horizon': best_scenario['config'].investment_horizon_years if best_scenario.get('config') else 'Unknown',
                    'rebalancing_frequency': best_scenario['config'].rebalancing_frequency if best_scenario.get('config') else 'Unknown',
                    'transaction_cost_bps': best_scenario['config'].transaction_cost_bps if best_scenario.get('config') else 'Unknown'
                },
                'perfect_weights': best_scenario.get('perfect_weights', {}),
                'asset_performance': best_scenario.get('asset_performance', {})
            }
        
        if bottom_performers:
            worst_scenario = bottom_performers[0]
            deep_dive['worst_scenario'] = {
                'scenario_id': worst_scenario['config'].scenario_id if worst_scenario.get('config') else 'Unknown',
                'performance_metrics': {
                    'annualized_return': worst_scenario.get('annualized_return', 0),
                    'volatility': worst_scenario.get('volatility', 0),
                    'sharpe_ratio': worst_scenario.get('sharpe_ratio', 0),
                    'max_drawdown': worst_scenario.get('max_drawdown', 0)
                },
                'configuration': {
                    'deviation_limit': worst_scenario['config'].permitted_deviation if worst_scenario.get('config') else 'Unknown',
                    'investment_horizon': worst_scenario['config'].investment_horizon_years if worst_scenario.get('config') else 'Unknown',
                    'rebalancing_frequency': worst_scenario['config'].rebalancing_frequency if worst_scenario.get('config') else 'Unknown',
                    'transaction_cost_bps': worst_scenario['config'].transaction_cost_bps if worst_scenario.get('config') else 'Unknown'
                },
                'perfect_weights': worst_scenario.get('perfect_weights', {}),
                'asset_performance': worst_scenario.get('asset_performance', {})
            }
        
        return deep_dive
    
    def _create_risk_analysis_report(self, analytics_results: Dict, mandate: str) -> Dict[str, Any]:
        """Create comprehensive risk analysis report"""
        
        all_results = analytics_results.get('all_results', [])
        successful_results = [r for r in all_results if r.get('success', False)]
        
        if not successful_results:
            return {'error': 'No successful scenarios for risk analysis'}
        
        # Extract risk metrics
        returns = [r.get('simulation_results', {}).get('annualized_return', 0) for r in successful_results]
        volatilities = [r.get('simulation_results', {}).get('volatility', 0) for r in successful_results]
        max_drawdowns = [r.get('simulation_results', {}).get('max_drawdown', 0) for r in successful_results]
        sharpe_ratios = [r.get('simulation_results', {}).get('sharpe_ratio', 0) for r in successful_results]
        
        # Risk analysis
        risk_analysis = {
            'return_statistics': {
                'mean': np.mean(returns),
                'median': np.median(returns),
                'std_dev': np.std(returns),
                'min': np.min(returns),
                'max': np.max(returns),
                'skewness': float(pd.Series(returns).skew()),
                'kurtosis': float(pd.Series(returns).kurtosis()),
                'var_95': np.percentile(returns, 5),
                'var_99': np.percentile(returns, 1)
            },
            'volatility_statistics': {
                'mean': np.mean(volatilities),
                'median': np.median(volatilities),
                'std_dev': np.std(volatilities),
                'min': np.min(volatilities),
                'max': np.max(volatilities)
            },
            'drawdown_statistics': {
                'mean': np.mean(max_drawdowns),
                'median': np.median(max_drawdowns),
                'std_dev': np.std(max_drawdowns),
                'min': np.min(max_drawdowns),
                'max': np.max(max_drawdowns),
                'scenarios_with_dd_over_20pct': sum([1 for dd in max_drawdowns if dd > 0.20]),
                'scenarios_with_dd_over_30pct': sum([1 for dd in max_drawdowns if dd > 0.30])
            },
            'sharpe_statistics': {
                'mean': np.mean(sharpe_ratios),
                'median': np.median(sharpe_ratios),
                'std_dev': np.std(sharpe_ratios),
                'min': np.min(sharpe_ratios),
                'max': np.max(sharpe_ratios),
                'scenarios_with_positive_sharpe': sum([1 for sr in sharpe_ratios if sr > 0])
            },
            'correlation_analysis': {
                'return_volatility_correlation': float(np.corrcoef(returns, volatilities)[0,1]),
                'return_drawdown_correlation': float(np.corrcoef(returns, max_drawdowns)[0,1]),
                'volatility_drawdown_correlation': float(np.corrcoef(volatilities, max_drawdowns)[0,1])
            }
        }
        
        return risk_analysis
    
    def _export_all_reports(self, report_components: Dict[str, Any]):
        """Export all report components to various formats"""
        
        mandate = report_components['mandate']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Export 1: Executive Summary as JSON
        summary_file = self.output_dir / "data" / f"{mandate}_executive_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(report_components['executive_summary'], f, indent=2, default=str)
        
        # Export 2: Comparison Tables as Excel
        excel_file = self.output_dir / "data" / f"{mandate}_comparison_tables_{timestamp}.xlsx"
        comparison_tables = report_components.get('comparison_tables', {})
        
        if comparison_tables:
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                for sheet_name, df in comparison_tables.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=True)
        
        # Export 3: Risk Analysis as JSON
        risk_file = self.output_dir / "data" / f"{mandate}_risk_analysis_{timestamp}.json"
        with open(risk_file, 'w') as f:
            json.dump(report_components['risk_analysis'], f, indent=2, default=str)
        
        # Export 4: Scenario Deep Dive as JSON
        scenario_file = self.output_dir / "data" / f"{mandate}_scenario_deep_dive_{timestamp}.json"
        with open(scenario_file, 'w') as f:
            json.dump(report_components['scenario_deep_dive'], f, indent=2, default=str)
        
        # Export 5: Complete Analytics Results
        analytics_file = self.output_dir / "data" / f"{mandate}_complete_analytics_{timestamp}.json"
        
        # Prepare analytics results for JSON export (handle non-serializable objects)
        analytics_export = {}
        for key, value in report_components['analytics_results'].items():
            if key == 'all_results':
                # Export only summary info for all results to avoid huge files
                analytics_export[key] = f"Total scenarios: {len(value)}"
            else:
                analytics_export[key] = value
        
        with open(analytics_file, 'w') as f:
            json.dump(analytics_export, f, indent=2, default=str)
        
        print(f"📁 Reports exported to:")
        print(f"   Executive Summary: {summary_file}")
        print(f"   Comparison Tables: {excel_file}")
        print(f"   Risk Analysis: {risk_file}")
        print(f"   Scenario Deep Dive: {scenario_file}")
        print(f"   Analytics Summary: {analytics_file}")

if __name__ == "__main__":
    # Test the reporting module
    print("🧪 Testing PHP Reporting Module")
    
    # Initialize reporting engine
    reporter = PHPReporting()
    
    # Generate a sample report (limited scenarios for testing)
    print("\n📊 Generating sample report for EQFI mandate...")
    
    try:
        report = reporter.generate_complete_php_report(
            mandate="EQFI",
            analysis_start="2010-01-01", 
            analysis_end="2020-01-01",
            sample_size=100  # Limit to 100 scenarios for testing
        )
        
        print("\n✅ Sample report generation completed successfully!")
        print(f"📈 Total scenarios analyzed: {report['total_scenarios']}")
        print(f"📁 Reports saved to: {reporter.output_dir}")
        
    except Exception as e:
        print(f"❌ Error generating sample report: {e}")
        import traceback
        traceback.print_exc()