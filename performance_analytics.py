import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import warnings
from scipy import stats
from dataclasses import dataclass
from portfolio_simulation import PortfolioSimulation, SimulationConfig
from portfolio_config import PortfolioConfig
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class ScenarioResult:
    """Container for individual scenario results with enhanced metrics"""
    config: SimulationConfig
    success: bool
    total_return: float
    annualized_return: float
    volatility: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    excess_return: float
    tracking_error: float
    information_ratio: float
    downside_deviation: float
    transaction_cost: float
    rebalancing_count: int
    forced_rebalancing_count: int
    benchmark_return: float
    benchmark_volatility: float
    error: str = None

class PerformanceAnalytics:
    """Comprehensive analytics framework for PHP performance analysis"""
    
    def __init__(self):
        self.simulation_engine = PortfolioSimulation()
        self.results_cache = {}
        
    def calculate_enhanced_metrics(self, 
                                 portfolio_returns: pd.Series, 
                                 benchmark_returns: pd.Series = None,
                                 risk_free_rate: float = 0.02) -> Dict[str, float]:
        """
        Calculate comprehensive risk-adjusted performance metrics
        
        Args:
            portfolio_returns: Daily portfolio returns
            benchmark_returns: Daily benchmark returns (optional)
            risk_free_rate: Annual risk-free rate (default 2%)
            
        Returns:
            Dictionary of enhanced metrics
        """
        if portfolio_returns.empty:
            return {}
        
        # Convert risk-free rate to daily
        daily_rf = risk_free_rate / 252
        
        # Basic metrics
        total_return = (1 + portfolio_returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        excess_returns = portfolio_returns - daily_rf
        sharpe_ratio = excess_returns.mean() * 252 / (excess_returns.std() * np.sqrt(252)) if excess_returns.std() > 0 else 0
        
        # Sortino ratio (using downside deviation)
        downside_returns = portfolio_returns[portfolio_returns < daily_rf] - daily_rf
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = excess_returns.mean() * 252 / downside_deviation if downside_deviation > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'downside_deviation': downside_deviation
        }
        
        # Benchmark-relative metrics
        if benchmark_returns is not None and not benchmark_returns.empty:
            # Align returns
            common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
            if len(common_dates) > 0:
                port_aligned = portfolio_returns.loc[common_dates]
                bench_aligned = benchmark_returns.loc[common_dates]
                
                # Excess returns
                excess_rets = port_aligned - bench_aligned
                excess_return = excess_rets.mean() * 252
                tracking_error = excess_rets.std() * np.sqrt(252)
                information_ratio = excess_return / tracking_error if tracking_error > 0 else 0
                
                # Benchmark metrics
                bench_total = (1 + bench_aligned).prod() - 1
                bench_annualized = (1 + bench_total) ** (252 / len(bench_aligned)) - 1
                bench_volatility = bench_aligned.std() * np.sqrt(252)
                
                metrics.update({
                    'excess_return': excess_return,
                    'tracking_error': tracking_error,
                    'information_ratio': information_ratio,
                    'benchmark_return': bench_annualized,
                    'benchmark_volatility': bench_volatility
                })
        
        return metrics
    
    def run_comprehensive_analysis(self, 
                                 mandate: str,
                                 analysis_start: str = "1996-01-01",
                                 analysis_end: str = "2025-05-31",
                                 max_scenarios: Optional[int] = None) -> List[ScenarioResult]:
        """
        Run comprehensive analysis across all scenarios for a mandate
        
        Args:
            mandate: Portfolio mandate to analyze
            analysis_start: Start date for analysis
            analysis_end: End date for analysis
            max_scenarios: Maximum number of scenarios to run (for testing)
            
        Returns:
            List of ScenarioResult objects
        """
        print(f"=== COMPREHENSIVE ANALYSIS: {mandate} ===")
        print(f"Period: {analysis_start} to {analysis_end}")
        
        # Generate all scenarios
        all_scenarios = self.simulation_engine.generate_all_php_scenarios(
            mandate, analysis_start, analysis_end
        )
        
        if max_scenarios:
            all_scenarios = all_scenarios[:max_scenarios]
        
        print(f"Total scenarios to analyze: {len(all_scenarios)}")
        
        results = []
        
        for i, scenario in enumerate(all_scenarios):
            if i % 100 == 0:
                print(f"Processing scenario {i+1}/{len(all_scenarios)}...")
            
            try:
                # Run simulation
                sim_result = self.simulation_engine.simulate_php_performance(scenario)
                
                if not sim_result['success']:
                    results.append(ScenarioResult(
                        config=scenario,
                        success=False,
                        error=sim_result['error'],
                        **{k: 0.0 for k in ['total_return', 'annualized_return', 'volatility', 
                                           'max_drawdown', 'sharpe_ratio', 'sortino_ratio', 
                                           'calmar_ratio', 'excess_return', 'tracking_error', 
                                           'information_ratio', 'downside_deviation', 
                                           'transaction_cost', 'benchmark_return', 'benchmark_volatility']},
                        rebalancing_count=0,
                        forced_rebalancing_count=0
                    ))
                    continue
                
                # Get portfolio returns
                portfolio_returns = sim_result['simulation_results']['portfolio_returns']
                
                # Get benchmark returns
                benchmark_returns = self.simulation_engine.portfolio_config.calculate_benchmark_6040(
                    scenario.start_date, scenario.end_date
                )
                
                # Calculate enhanced metrics
                enhanced_metrics = self.calculate_enhanced_metrics(
                    portfolio_returns, benchmark_returns
                )
                
                # Create result object
                result = ScenarioResult(
                    config=scenario,
                    success=True,
                    total_return=enhanced_metrics.get('total_return', 0),
                    annualized_return=enhanced_metrics.get('annualized_return', 0),
                    volatility=enhanced_metrics.get('volatility', 0),
                    max_drawdown=enhanced_metrics.get('max_drawdown', 0),
                    sharpe_ratio=enhanced_metrics.get('sharpe_ratio', 0),
                    sortino_ratio=enhanced_metrics.get('sortino_ratio', 0),
                    calmar_ratio=enhanced_metrics.get('calmar_ratio', 0),
                    excess_return=enhanced_metrics.get('excess_return', 0),
                    tracking_error=enhanced_metrics.get('tracking_error', 0),
                    information_ratio=enhanced_metrics.get('information_ratio', 0),
                    downside_deviation=enhanced_metrics.get('downside_deviation', 0),
                    transaction_cost=sim_result['simulation_results'].get('total_transaction_cost', 0),
                    rebalancing_count=len(sim_result['simulation_results'].get('rebalancing_dates', [])),
                    forced_rebalancing_count=len(sim_result['simulation_results'].get('forced_rebalancing_dates', [])),
                    benchmark_return=enhanced_metrics.get('benchmark_return', 0),
                    benchmark_volatility=enhanced_metrics.get('benchmark_volatility', 0),
                    error=None
                )
                
                results.append(result)
                
            except Exception as e:
                print(f"Error in scenario {scenario.scenario_id}: {e}")
                results.append(ScenarioResult(
                    config=scenario,
                    success=False,
                    error=str(e),
                    **{k: 0.0 for k in ['total_return', 'annualized_return', 'volatility', 
                                       'max_drawdown', 'sharpe_ratio', 'sortino_ratio', 
                                       'calmar_ratio', 'excess_return', 'tracking_error', 
                                       'information_ratio', 'downside_deviation', 
                                       'transaction_cost', 'benchmark_return', 'benchmark_volatility']},
                    rebalancing_count=0,
                    forced_rebalancing_count=0
                ))
        
        print(f"Analysis complete: {len([r for r in results if r.success])}/{len(results)} successful")
        
        # Cache results
        self.results_cache[mandate] = results
        
        return results
    
    def run_comprehensive_analysis_from_results(self, 
                                               simulation_results: List[Dict[str, Any]],
                                               analysis_name: str = "php_analysis") -> Dict[str, Any]:
        """
        Run comprehensive analysis on existing simulation results
        
        Args:
            simulation_results: List of simulation result dictionaries
            analysis_name: Name identifier for this analysis
            
        Returns:
            Dictionary containing all analysis results
        """
        print(f"=== COMPREHENSIVE ANALYSIS: {analysis_name} ===")
        
        # Extract successful results
        successful_results = [r for r in simulation_results if r.get('success', False)]
        total_results = len(simulation_results)
        success_count = len(successful_results)
        
        print(f"Processing {success_count}/{total_results} successful scenarios")
        
        if not successful_results:
            return {
                'all_results': simulation_results,
                'summary_statistics': {},
                'top_performers': {'top_10': [], 'bottom_10': []},
                'parameter_analysis': {},
                'distribution_analysis': {},
                'correlation_analysis': {},
                'error': 'No successful scenarios found'
            }
        
        # Convert to enhanced metrics format
        enhanced_results = []
        for result in successful_results:
            sim_results = result.get('simulation_results', {})
            config = result.get('config')
            
            if sim_results and config:
                enhanced_results.append({
                    'config': config,
                    'success': True,
                    'annualized_return': sim_results.get('annualized_return', 0),
                    'total_return': sim_results.get('total_return', 0),
                    'volatility': sim_results.get('volatility', 0),
                    'max_drawdown': sim_results.get('max_drawdown', 0),
                    'sharpe_ratio': sim_results.get('sharpe_ratio', 0),
                    'excess_return': sim_results.get('annualized_return', 0) - 0.08,  # vs benchmark
                    'transaction_cost': sim_results.get('total_transaction_cost', 0),
                    'rebalancing_count': len(sim_results.get('rebalancing_dates', [])),
                    'forced_rebalancing_count': len(sim_results.get('forced_rebalancing_dates', [])),
                    'perfect_weights': result.get('perfect_weights', {}),
                    'asset_performance': result.get('asset_performance', {})
                })
        
        # Generate summary statistics
        returns = [r['annualized_return'] for r in enhanced_results]
        volatilities = [r['volatility'] for r in enhanced_results]
        sharpe_ratios = [r['sharpe_ratio'] for r in enhanced_results]
        max_drawdowns = [r['max_drawdown'] for r in enhanced_results]
        excess_returns = [r['excess_return'] for r in enhanced_results]
        
        summary_statistics = {
            'annualized_return': {
                'mean': np.mean(returns), 'median': np.median(returns),
                'std': np.std(returns), 'min': np.min(returns), 'max': np.max(returns),
                'percentile_5': np.percentile(returns, 5),
                'percentile_95': np.percentile(returns, 95)
            },
            'volatility': {
                'mean': np.mean(volatilities), 'median': np.median(volatilities),
                'std': np.std(volatilities), 'min': np.min(volatilities), 'max': np.max(volatilities)
            },
            'sharpe_ratio': {
                'mean': np.mean(sharpe_ratios), 'median': np.median(sharpe_ratios),
                'std': np.std(sharpe_ratios), 'min': np.min(sharpe_ratios), 'max': np.max(sharpe_ratios)
            },
            'max_drawdown': {
                'mean': np.mean(max_drawdowns), 'median': np.median(max_drawdowns),
                'std': np.std(max_drawdowns), 'min': np.min(max_drawdowns), 'max': np.max(max_drawdowns)
            },
            'excess_return': {
                'mean': np.mean(excess_returns), 'median': np.median(excess_returns),
                'std': np.std(excess_returns), 'min': np.min(excess_returns), 'max': np.max(excess_returns)
            }
        }
        
        # Identify top and bottom performers
        enhanced_results_sorted = sorted(enhanced_results, key=lambda x: x['annualized_return'], reverse=True)
        
        top_performers = {
            'top_10': enhanced_results_sorted[:10],
            'bottom_10': enhanced_results_sorted[-10:] if len(enhanced_results_sorted) >= 10 else []
        }
        
        # Parameter analysis
        parameter_analysis = self._analyze_parameters(enhanced_results)
        
        # Distribution analysis
        distribution_analysis = {
            'return_distribution': {
                'skewness': float(stats.skew(returns)),
                'kurtosis': float(stats.kurtosis(returns)),
                'normality_test': float(stats.shapiro(returns[:min(len(returns), 5000)])[1])  # p-value
            },
            'positive_excess_scenarios': len([r for r in excess_returns if r > 0]),
            'high_sharpe_scenarios': len([r for r in sharpe_ratios if r > 1.0]),
            'low_drawdown_scenarios': len([r for r in max_drawdowns if r < 0.10])
        }
        
        # Correlation analysis
        correlation_matrix = np.corrcoef([returns, volatilities, sharpe_ratios, max_drawdowns])
        correlation_analysis = {
            'return_volatility': float(correlation_matrix[0, 1]),
            'return_sharpe': float(correlation_matrix[0, 2]),
            'return_drawdown': float(correlation_matrix[0, 3]),
            'volatility_drawdown': float(correlation_matrix[1, 3])
        }
        
        analysis_results = {
            'all_results': simulation_results,
            'enhanced_results': enhanced_results,
            'summary_statistics': summary_statistics,
            'top_performers': top_performers,
            'bottom_performers': {'bottom_10': enhanced_results_sorted[-10:]},
            'parameter_analysis': parameter_analysis,
            'distribution_analysis': distribution_analysis,
            'correlation_analysis': correlation_analysis,
            'scenario_count': {
                'total': total_results,
                'successful': success_count,
                'success_rate': success_count / total_results if total_results > 0 else 0
            }
        }
        
        print(f"✅ Analysis complete: {success_count}/{total_results} scenarios processed")
        
        return analysis_results
    
    def _analyze_parameters(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze optimal parameter combinations"""
        
        # Group by parameters
        param_groups = {}
        for result in results:
            config = result['config']
            key = (config.permitted_deviation, config.investment_horizon_years, 
                   config.rebalancing_frequency, config.transaction_cost_bps)
            
            if key not in param_groups:
                param_groups[key] = []
            param_groups[key].append(result)
        
        # Find optimal combinations
        param_performance = []
        for params, group_results in param_groups.items():
            avg_return = np.mean([r['annualized_return'] for r in group_results])
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in group_results])
            avg_drawdown = np.mean([r['max_drawdown'] for r in group_results])
            
            param_performance.append({
                'deviation': params[0],
                'horizon': params[1], 
                'rebalancing': params[2],
                'transaction_cost': params[3],
                'avg_return': avg_return,
                'avg_sharpe': avg_sharpe,
                'avg_drawdown': avg_drawdown,
                'scenario_count': len(group_results)
            })
        
        # Sort by Sharpe ratio
        param_performance.sort(key=lambda x: x['avg_sharpe'], reverse=True)
        
        optimal_params = param_performance[0] if param_performance else {}
        
        return {
            'optimal_deviation': f"{optimal_params.get('deviation', 0):.0%}",
            'optimal_horizon': f"{optimal_params.get('horizon', 0)}Y",
            'optimal_rebalancing': optimal_params.get('rebalancing', 'unknown'),
            'optimal_transaction_cost': f"{optimal_params.get('transaction_cost', 0)}bps",
            'parameter_performance': param_performance[:20]  # Top 20 combinations
        }
    
    def generate_performance_summary(self, results: List[ScenarioResult]) -> pd.DataFrame:
        """Generate summary statistics across all scenarios"""
        
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return pd.DataFrame()
        
        # Extract metrics into DataFrame
        data = []
        for result in successful_results:
            data.append({
                'scenario_id': result.config.scenario_id,
                'mandate': result.config.mandate,
                'horizon_years': result.config.investment_horizon_years,
                'deviation': result.config.permitted_deviation,
                'rebalancing': result.config.rebalancing_frequency,
                'transaction_cost_bps': result.config.transaction_cost_bps,
                'total_return': result.total_return,
                'annualized_return': result.annualized_return,
                'volatility': result.volatility,
                'max_drawdown': result.max_drawdown,
                'sharpe_ratio': result.sharpe_ratio,
                'sortino_ratio': result.sortino_ratio,
                'calmar_ratio': result.calmar_ratio,
                'excess_return': result.excess_return,
                'tracking_error': result.tracking_error,
                'information_ratio': result.information_ratio,
                'downside_deviation': result.downside_deviation,
                'transaction_cost': result.transaction_cost,
                'rebalancing_count': result.rebalancing_count,
                'forced_rebalancing_count': result.forced_rebalancing_count,
                'benchmark_return': result.benchmark_return,
                'benchmark_volatility': result.benchmark_volatility
            })
        
        df = pd.DataFrame(data)
        return df
    
    def print_scenario_analysis_summary(self, results: List[ScenarioResult]):
        """Print comprehensive summary of scenario analysis"""
        
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        print(f"\n=== SCENARIO ANALYSIS SUMMARY ===")
        print(f"Total Scenarios: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        if not successful:
            print("No successful scenarios to analyze")
            return
        
        # Convert to DataFrame for analysis
        df = self.generate_performance_summary(successful)
        
        # Summary statistics
        print(f"\n=== PERFORMANCE STATISTICS ===")
        metrics = ['annualized_return', 'volatility', 'max_drawdown', 'sharpe_ratio', 
                  'sortino_ratio', 'excess_return', 'information_ratio']
        
        summary_stats = df[metrics].describe()
        
        for metric in metrics:
            stats = summary_stats[metric]
            print(f"\n{metric.replace('_', ' ').title()}:")
            print(f"  Mean:   {stats['mean']:>8.2%}" if 'ratio' not in metric else f"  Mean:   {stats['mean']:>8.2f}")
            print(f"  Median: {stats['50%']:>8.2%}" if 'ratio' not in metric else f"  Median: {stats['50%']:>8.2f}")
            print(f"  Std:    {stats['std']:>8.2%}" if 'ratio' not in metric else f"  Std:    {stats['std']:>8.2f}")
            print(f"  Min:    {stats['min']:>8.2%}" if 'ratio' not in metric else f"  Min:    {stats['min']:>8.2f}")
            print(f"  Max:    {stats['max']:>8.2%}" if 'ratio' not in metric else f"  Max:    {stats['max']:>8.2f}")
        
        # Top performers
        print(f"\n=== TOP 10 PERFORMERS (by Annualized Return) ===")
        top_performers = df.nlargest(10, 'annualized_return')
        
        for i, (_, row) in enumerate(top_performers.iterrows(), 1):
            print(f"{i:2d}. {row['scenario_id'][:50]:<50} {row['annualized_return']:>8.1%}")
        
        # Worst performers  
        print(f"\n=== WORST 10 PERFORMERS (by Annualized Return) ===")
        worst_performers = df.nsmallest(10, 'annualized_return')
        
        for i, (_, row) in enumerate(worst_performers.iterrows(), 1):
            print(f"{i:2d}. {row['scenario_id'][:50]:<50} {row['annualized_return']:>8.1%}")
    
    def analyze_by_parameter(self, results: List[ScenarioResult], parameter: str) -> pd.DataFrame:
        """
        Analyze performance by specific parameter (e.g., deviation, rebalancing frequency)
        
        Args:
            results: List of scenario results
            parameter: Parameter to group by ('deviation', 'rebalancing', 'horizon_years', 'transaction_cost_bps')
            
        Returns:
            DataFrame with statistics by parameter value
        """
        successful = [r for r in results if r.success]
        if not successful:
            return pd.DataFrame()
        
        df = self.generate_performance_summary(successful)
        
        # Group by parameter
        grouped = df.groupby(parameter)
        
        # Calculate statistics for each group
        metrics = ['annualized_return', 'volatility', 'max_drawdown', 'sharpe_ratio', 
                  'sortino_ratio', 'excess_return', 'information_ratio']
        
        summary_data = []
        
        for param_value, group in grouped:
            row = {'parameter_value': param_value, 'count': len(group)}
            
            for metric in metrics:
                row[f'{metric}_mean'] = group[metric].mean()
                row[f'{metric}_median'] = group[metric].median()
                row[f'{metric}_std'] = group[metric].std()
                row[f'{metric}_min'] = group[metric].min()
                row[f'{metric}_max'] = group[metric].max()
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def correlation_analysis(self, results: List[ScenarioResult]) -> pd.DataFrame:
        """Analyze correlations between different metrics and parameters"""
        
        successful = [r for r in results if r.success]
        if not successful:
            return pd.DataFrame()
        
        df = self.generate_performance_summary(successful)
        
        # Select numeric columns for correlation
        numeric_cols = ['horizon_years', 'deviation', 'transaction_cost_bps',
                       'annualized_return', 'volatility', 'max_drawdown', 'sharpe_ratio',
                       'sortino_ratio', 'excess_return', 'information_ratio', 
                       'transaction_cost', 'rebalancing_count', 'forced_rebalancing_count']
        
        correlation_matrix = df[numeric_cols].corr()
        
        return correlation_matrix
    
    def performance_distribution_analysis(self, results: List[ScenarioResult]) -> Dict[str, Any]:
        """Analyze performance distributions and identify patterns"""
        
        successful = [r for r in results if r.success]
        if not successful:
            return {}
        
        df = self.generate_performance_summary(successful)
        
        analysis = {}
        
        # Return distribution analysis
        returns = df['annualized_return']
        analysis['return_distribution'] = {
            'mean': returns.mean(),
            'median': returns.median(),
            'std': returns.std(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'percentile_10': returns.quantile(0.1),
            'percentile_25': returns.quantile(0.25),
            'percentile_75': returns.quantile(0.75),
            'percentile_90': returns.quantile(0.9),
            'positive_return_pct': (returns > 0).mean(),
            'outperform_benchmark_pct': (df['excess_return'] > 0).mean()
        }
        
        # Risk distribution analysis
        volatility = df['volatility']
        drawdowns = df['max_drawdown']
        
        analysis['risk_distribution'] = {
            'volatility_mean': volatility.mean(),
            'volatility_median': volatility.median(),
            'drawdown_mean': drawdowns.mean(),
            'drawdown_median': drawdowns.median(),
            'low_vol_scenarios_pct': (volatility < volatility.median()).mean(),
            'high_drawdown_scenarios_pct': (drawdowns > 0.2).mean()  # >20% drawdown
        }
        
        # Risk-adjusted performance
        sharpe_ratios = df['sharpe_ratio']
        info_ratios = df['information_ratio']
        
        analysis['risk_adjusted_distribution'] = {
            'sharpe_mean': sharpe_ratios.mean(),
            'sharpe_median': sharpe_ratios.median(),
            'positive_sharpe_pct': (sharpe_ratios > 0).mean(),
            'high_sharpe_pct': (sharpe_ratios > 1.0).mean(),
            'info_ratio_mean': info_ratios.mean(),
            'info_ratio_median': info_ratios.median(),
            'positive_info_ratio_pct': (info_ratios > 0).mean()
        }
        
        # Transaction cost impact
        tx_costs = df['transaction_cost']
        analysis['transaction_cost_analysis'] = {
            'mean_tx_cost': tx_costs.mean(),
            'median_tx_cost': tx_costs.median(),
            'high_tx_cost_threshold': tx_costs.quantile(0.9),
            'correlation_return_txcost': df['annualized_return'].corr(tx_costs),
            'correlation_sharpe_txcost': df['sharpe_ratio'].corr(tx_costs)
        }
        
        return analysis
    
    def generate_executive_summary(self, results: List[ScenarioResult]) -> str:
        """Generate executive summary of PHP analysis"""
        
        successful = [r for r in results if r.success]
        if not successful:
            return "No successful scenarios to analyze."
        
        df = self.generate_performance_summary(successful)
        dist_analysis = self.performance_distribution_analysis(results)
        
        # Best and worst scenarios
        best_scenario = df.loc[df['annualized_return'].idxmax()]
        worst_scenario = df.loc[df['annualized_return'].idxmin()]
        
        summary = f"""
=== PERFECT HINDSIGHT PORTFOLIO EXECUTIVE SUMMARY ===

SCOPE:
- Mandate: {df['mandate'].iloc[0]}
- Total Scenarios Analyzed: {len(successful):,}
- Analysis Period: Multiple rolling periods with {df['horizon_years'].nunique()} investment horizons

KEY FINDINGS:

RETURNS:
- Average Annualized Return: {dist_analysis['return_distribution']['mean']:.1%}
- Median Annualized Return: {dist_analysis['return_distribution']['median']:.1%}
- Return Range: {dist_analysis['return_distribution']['percentile_10']:.1%} to {dist_analysis['return_distribution']['percentile_90']:.1%} (10th-90th percentile)
- Scenarios with Positive Returns: {dist_analysis['return_distribution']['positive_return_pct']:.1%}
- Scenarios Outperforming Benchmark: {dist_analysis['return_distribution']['outperform_benchmark_pct']:.1%}

RISK PROFILE:
- Average Volatility: {dist_analysis['risk_distribution']['volatility_mean']:.1%}
- Average Maximum Drawdown: {dist_analysis['risk_distribution']['drawdown_mean']:.1%}
- High Drawdown Scenarios (>20%): {dist_analysis['risk_distribution']['high_drawdown_scenarios_pct']:.1%}

RISK-ADJUSTED PERFORMANCE:
- Average Sharpe Ratio: {dist_analysis['risk_adjusted_distribution']['sharpe_mean']:.2f}
- Scenarios with Positive Sharpe Ratio: {dist_analysis['risk_adjusted_distribution']['positive_sharpe_pct']:.1%}
- High-Quality Scenarios (Sharpe > 1.0): {dist_analysis['risk_adjusted_distribution']['high_sharpe_pct']:.1%}
- Average Information Ratio: {dist_analysis['risk_adjusted_distribution']['info_ratio_mean']:.2f}

TRANSACTION COSTS:
- Average Transaction Cost: {dist_analysis['transaction_cost_analysis']['mean_tx_cost']:.2%}
- Correlation with Returns: {dist_analysis['transaction_cost_analysis']['correlation_return_txcost']:.2f}

BEST PERFORMING SCENARIO:
- ID: {best_scenario['scenario_id']}
- Annualized Return: {best_scenario['annualized_return']:.1%}
- Sharpe Ratio: {best_scenario['sharpe_ratio']:.2f}
- Max Drawdown: {best_scenario['max_drawdown']:.1%}

WORST PERFORMING SCENARIO:
- ID: {worst_scenario['scenario_id']}
- Annualized Return: {worst_scenario['annualized_return']:.1%}
- Sharpe Ratio: {worst_scenario['sharpe_ratio']:.2f}
- Max Drawdown: {worst_scenario['max_drawdown']:.1%}

STRATEGIC INSIGHTS:
- Perfect Hindsight portfolios show {'strong' if dist_analysis['return_distribution']['outperform_benchmark_pct'] > 0.6 else 'mixed' if dist_analysis['return_distribution']['outperform_benchmark_pct'] > 0.4 else 'weak'} outperformance potential
- Risk management is {'effective' if dist_analysis['risk_distribution']['high_drawdown_scenarios_pct'] < 0.3 else 'challenging'} with current constraints
- Transaction costs have {'significant' if abs(dist_analysis['transaction_cost_analysis']['correlation_return_txcost']) > 0.3 else 'moderate' if abs(dist_analysis['transaction_cost_analysis']['correlation_return_txcost']) > 0.1 else 'minimal'} impact on performance
        """
        
        return summary.strip()
    
    def export_results(self, results: List[ScenarioResult], filename: str = "php_analysis_results.csv"):
        """Export analysis results to CSV"""
        
        successful = [r for r in results if r.success]
        if not successful:
            print("No successful results to export")
            return
        
        df = self.generate_performance_summary(successful)
        df.to_csv(filename, index=False)
        print(f"Results exported to {filename}")
        print(f"Exported {len(df)} scenarios with {len(df.columns)} metrics")
    
    def rank_scenarios(self, results: List[ScenarioResult], 
                      primary_metric: str = 'sharpe_ratio',
                      secondary_metric: str = 'annualized_return',
                      top_n: int = 20) -> pd.DataFrame:
        """
        Rank scenarios by multiple criteria
        
        Args:
            results: List of scenario results
            primary_metric: Primary ranking metric
            secondary_metric: Secondary ranking metric (tiebreaker)
            top_n: Number of top scenarios to return
            
        Returns:
            DataFrame with top ranked scenarios
        """
        successful = [r for r in results if r.success]
        if not successful:
            return pd.DataFrame()
        
        df = self.generate_performance_summary(successful)
        
        # Rank by primary metric, then secondary metric
        ranked = df.sort_values([primary_metric, secondary_metric], ascending=[False, False])
        
        # Add ranking columns
        ranked['rank'] = range(1, len(ranked) + 1)
        ranked['percentile'] = ranked[primary_metric].rank(pct=True)
        
        return ranked.head(top_n)
    
    def filter_scenarios(self, results: List[ScenarioResult], 
                        filters: Dict[str, Any]) -> List[ScenarioResult]:
        """
        Filter scenarios by specified criteria
        
        Args:
            results: List of scenario results
            filters: Dictionary of filter criteria
                    e.g., {'sharpe_ratio_min': 1.0, 'max_drawdown_max': 0.15}
        
        Returns:
            Filtered list of scenario results
        """
        successful = [r for r in results if r.success]
        if not successful:
            return []
        
        df = self.generate_performance_summary(successful)
        
        # Apply filters
        mask = pd.Series([True] * len(df))
        
        for filter_name, filter_value in filters.items():
            if '_min' in filter_name:
                metric = filter_name.replace('_min', '')
                if metric in df.columns:
                    mask &= (df[metric] >= filter_value)
            elif '_max' in filter_name:
                metric = filter_name.replace('_max', '')
                if metric in df.columns:
                    mask &= (df[metric] <= filter_value)
            elif filter_name in df.columns:
                if isinstance(filter_value, list):
                    mask &= df[filter_name].isin(filter_value)
                else:
                    mask &= (df[filter_name] == filter_value)
        
        # Return filtered results
        filtered_indices = df[mask].index.tolist()
        return [successful[i] for i in filtered_indices]
    
    def create_performance_buckets(self, results: List[ScenarioResult],
                                 metric: str = 'annualized_return',
                                 n_buckets: int = 5) -> Dict[str, List[ScenarioResult]]:
        """
        Create performance buckets (quintiles, deciles, etc.) for analysis
        
        Args:
            results: List of scenario results
            metric: Metric to bucket by
            n_buckets: Number of buckets to create
            
        Returns:
            Dictionary with bucket labels and corresponding scenario results
        """
        successful = [r for r in results if r.success]
        if not successful:
            return {}
        
        df = self.generate_performance_summary(successful)
        
        # Create buckets
        df['bucket'] = pd.qcut(df[metric], q=n_buckets, labels=[f'Bucket_{i+1}' for i in range(n_buckets)])
        
        buckets = {}
        for bucket_name in df['bucket'].unique():
            bucket_indices = df[df['bucket'] == bucket_name].index.tolist()
            buckets[bucket_name] = [successful[i] for i in bucket_indices]
        
        return buckets
    
    def rolling_performance_analysis(self, results: List[ScenarioResult]) -> pd.DataFrame:
        """Analyze performance across different time periods"""
        
        successful = [r for r in results if r.success]
        if not successful:
            return pd.DataFrame()
        
        df = self.generate_performance_summary(successful)
        
        # Extract start year from scenario_id for time-based analysis
        df['start_year'] = df['scenario_id'].str.extract(r'_(\d{4})\d{2}_').astype(int)
        
        # Group by start year and calculate statistics
        yearly_stats = df.groupby('start_year').agg({
            'annualized_return': ['mean', 'median', 'std', 'min', 'max'],
            'volatility': ['mean', 'median'],
            'max_drawdown': ['mean', 'median'],
            'sharpe_ratio': ['mean', 'median'],
            'excess_return': ['mean', 'median']
        }).round(4)
        
        # Flatten column names
        yearly_stats.columns = ['_'.join(col).strip() for col in yearly_stats.columns.values]
        yearly_stats.reset_index(inplace=True)
        
        return yearly_stats
    
    def scenario_comparison_matrix(self, results: List[ScenarioResult],
                                 comparison_metrics: List[str] = None) -> pd.DataFrame:
        """Create comparison matrix for different scenario parameters"""
        
        if comparison_metrics is None:
            comparison_metrics = ['annualized_return', 'sharpe_ratio', 'max_drawdown']
        
        successful = [r for r in results if r.success]
        if not successful:
            return pd.DataFrame()
        
        df = self.generate_performance_summary(successful)
        
        # Create pivot table for each parameter combination
        comparison_data = []
        
        # Deviation vs Rebalancing Frequency
        for metric in comparison_metrics:
            pivot = df.pivot_table(
                values=metric,
                index='deviation',
                columns='rebalancing',
                aggfunc='mean'
            )
            
            for dev in pivot.index:
                for rebal in pivot.columns:
                    comparison_data.append({
                        'metric': metric,
                        'deviation': dev,
                        'rebalancing': rebal,
                        'value': pivot.loc[dev, rebal] if not pd.isna(pivot.loc[dev, rebal]) else 0
                    })
        
        return pd.DataFrame(comparison_data)
    
    def identify_optimal_parameters(self, results: List[ScenarioResult]) -> Dict[str, Any]:
        """Identify optimal parameter combinations across scenarios"""
        
        successful = [r for r in results if r.success]
        if not successful:
            return {}
        
        df = self.generate_performance_summary(successful)
        
        # Analyze by parameter combinations
        parameter_analysis = {}
        
        # Best deviation level
        deviation_analysis = df.groupby('deviation').agg({
            'annualized_return': 'mean',
            'sharpe_ratio': 'mean',
            'max_drawdown': 'mean',
            'excess_return': 'mean'
        }).round(4)
        
        parameter_analysis['optimal_deviation'] = {
            'by_return': deviation_analysis['annualized_return'].idxmax(),
            'by_sharpe': deviation_analysis['sharpe_ratio'].idxmax(),
            'by_drawdown': deviation_analysis['max_drawdown'].idxmin(),
            'analysis': deviation_analysis
        }
        
        # Best rebalancing frequency
        rebalancing_analysis = df.groupby('rebalancing').agg({
            'annualized_return': 'mean',
            'sharpe_ratio': 'mean',
            'max_drawdown': 'mean',
            'transaction_cost': 'mean'
        }).round(4)
        
        parameter_analysis['optimal_rebalancing'] = {
            'by_return': rebalancing_analysis['annualized_return'].idxmax(),
            'by_sharpe': rebalancing_analysis['sharpe_ratio'].idxmax(),
            'by_cost': rebalancing_analysis['transaction_cost'].idxmin(),
            'analysis': rebalancing_analysis
        }
        
        # Best transaction cost level
        tx_cost_analysis = df.groupby('transaction_cost_bps').agg({
            'annualized_return': 'mean',
            'sharpe_ratio': 'mean',
            'information_ratio': 'mean'
        }).round(4)
        
        parameter_analysis['optimal_transaction_cost'] = {
            'by_return': tx_cost_analysis['annualized_return'].idxmax(),
            'by_sharpe': tx_cost_analysis['sharpe_ratio'].idxmax(),
            'analysis': tx_cost_analysis
        }
        
        # Best investment horizon
        horizon_analysis = df.groupby('horizon_years').agg({
            'annualized_return': 'mean',
            'sharpe_ratio': 'mean',
            'volatility': 'mean'
        }).round(4)
        
        parameter_analysis['optimal_horizon'] = {
            'by_return': horizon_analysis['annualized_return'].idxmax(),
            'by_sharpe': horizon_analysis['sharpe_ratio'].idxmax(),
            'analysis': horizon_analysis
        }
        
        return parameter_analysis
    
    def print_comprehensive_analysis(self, results: List[ScenarioResult]):
        """Print comprehensive analysis including all capabilities"""
        
        print(self.generate_executive_summary(results))
        
        print(f"\n" + "="*80)
        print("DETAILED PARAMETER ANALYSIS")
        print("="*80)
        
        # Parameter optimization
        optimal_params = self.identify_optimal_parameters(results)
        
        for param_name, analysis in optimal_params.items():
            print(f"\n{param_name.upper().replace('_', ' ')}:")
            if 'by_return' in analysis:
                print(f"  Optimal by Return: {analysis['by_return']}")
            if 'by_sharpe' in analysis:
                print(f"  Optimal by Sharpe: {analysis['by_sharpe']}")
            if 'by_drawdown' in analysis:
                print(f"  Optimal by Drawdown: {analysis['by_drawdown']}")
            if 'by_cost' in analysis:
                print(f"  Optimal by Cost: {analysis['by_cost']}")
        
        # Top performers
        print(f"\n" + "="*80)
        print("TOP 10 SCENARIOS (by Sharpe Ratio)")
        print("="*80)
        
        top_scenarios = self.rank_scenarios(results, 'sharpe_ratio', 'annualized_return', 10)
        
        if not top_scenarios.empty:
            for i, (_, row) in enumerate(top_scenarios.iterrows(), 1):
                print(f"{i:2d}. {row['scenario_id'][:50]}")
                print(f"     Return: {row['annualized_return']:>6.1%}, Sharpe: {row['sharpe_ratio']:>5.2f}, Drawdown: {row['max_drawdown']:>6.1%}")
        
        # Distribution analysis
        dist_analysis = self.performance_distribution_analysis(results)
        
        print(f"\n" + "="*80)
        print("PERFORMANCE DISTRIBUTION DETAILS")
        print("="*80)
        
        print(f"\nReturn Distribution:")
        print(f"  Skewness: {dist_analysis['return_distribution']['skewness']:.2f}")
        print(f"  Kurtosis: {dist_analysis['return_distribution']['kurtosis']:.2f}")
        print(f"  25th-75th Percentile: {dist_analysis['return_distribution']['percentile_25']:.1%} to {dist_analysis['return_distribution']['percentile_75']:.1%}")
        
        print(f"\nTransaction Cost Impact:")
        print(f"  Mean Cost: {dist_analysis['transaction_cost_analysis']['mean_tx_cost']:.2%}")
        print(f"  90th Percentile Cost: {dist_analysis['transaction_cost_analysis']['high_tx_cost_threshold']:.2%}")
        print(f"  Correlation with Sharpe: {dist_analysis['transaction_cost_analysis']['correlation_sharpe_txcost']:.2f}")
        
        print(f"\n" + "="*80 + "\n")

if __name__ == "__main__":
    # Test the analytics framework
    analytics = PerformanceAnalytics()
    
    print("Testing comprehensive analytics framework...")
    
    # Run analysis on a small sample first
    results = analytics.run_comprehensive_analysis(
        mandate="EQFI",
        analysis_start="2010-01-01", 
        analysis_end="2015-01-01",
        max_scenarios=50  # Small sample for testing
    )
    
    # Print summary
    analytics.print_scenario_analysis_summary(results)