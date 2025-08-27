from portfolio_simulation import PortfolioSimulation, SimulationConfig
from portfolio_config import PortfolioConfig
import numpy as np
import pandas as pd

def test_simulation_scenarios():
    """Test 5 different simulation scenarios and calculate excess return/tracking error vs 6040 benchmark"""
    
    sim_engine = PortfolioSimulation()
    
    # Define 5 test scenarios
    test_scenarios = [
        SimulationConfig(
            mandate="EQFI",
            start_date="2008-01-01",
            end_date="2013-01-01",
            investment_horizon_years=5,
            permitted_deviation=0.20,
            rebalancing_frequency="quarterly",
            transaction_cost_bps=25,
            scenario_id="Financial_Crisis_Recovery"
        ),
        SimulationConfig(
            mandate="EQFILA",
            start_date="2015-01-01",
            end_date="2020-01-01",
            investment_horizon_years=5,
            permitted_deviation=0.10,
            rebalancing_frequency="monthly",
            transaction_cost_bps=50,
            scenario_id="Low_Rate_Environment"
        ),
        SimulationConfig(
            mandate="EQFI",
            start_date="2010-01-01",
            end_date="2020-01-01",
            investment_horizon_years=10,
            permitted_deviation=0.05,
            rebalancing_frequency="annual",
            transaction_cost_bps=100,
            scenario_id="Long_Bull_Market"
        ),
        SimulationConfig(
            mandate="EQFILAIA",
            start_date="2012-01-01",
            end_date="2017-01-01",
            investment_horizon_years=5,
            permitted_deviation=0.50,
            rebalancing_frequency="none",
            transaction_cost_bps=5,
            scenario_id="QE_Era_All_Assets"
        ),
        SimulationConfig(
            mandate="EQFILA",
            start_date="2018-01-01",
            end_date="2023-01-01",
            investment_horizon_years=5,
            permitted_deviation=0.20,
            rebalancing_frequency="quarterly",
            transaction_cost_bps=25,
            scenario_id="Recent_Volatility"
        )
    ]
    
    print("=== TESTING 5 SIMULATION SCENARIOS WITH BENCHMARK COMPARISON ===\n")
    
    for i, config in enumerate(test_scenarios, 1):
        print(f"=== SCENARIO {i}: {config.scenario_id} ===")
        print(f"Mandate: {config.mandate}")
        print(f"Period: {config.start_date} to {config.end_date}")
        print(f"Horizon: {config.investment_horizon_years} years")
        print(f"Deviation: ±{config.permitted_deviation:.0%}")
        print(f"Rebalancing: {config.rebalancing_frequency}")
        print(f"Transaction Cost: {config.transaction_cost_bps} bps")
        
        try:
            # Simulate PHP performance
            php_result = sim_engine.simulate_php_performance(config)
            
            if not php_result['success']:
                print(f"❌ PHP Simulation failed: {php_result['error']}\n")
                continue
            
            # Calculate 6040 benchmark for the same period
            benchmark_returns = sim_engine.portfolio_config.calculate_benchmark_6040(
                config.start_date, config.end_date
            )
            
            if benchmark_returns.empty:
                print(f"❌ Benchmark calculation failed\n")
                continue
            
            # Get PHP results
            php_sim = php_result['simulation_results']
            php_returns = php_sim['portfolio_returns']
            
            # Align PHP and benchmark returns to common dates
            common_dates = php_returns.index.intersection(benchmark_returns.index)
            if len(common_dates) == 0:
                print(f"❌ No overlapping dates between PHP and benchmark\n")
                continue
            
            php_aligned = php_returns.loc[common_dates]
            benchmark_aligned = benchmark_returns.loc[common_dates]
            
            # Calculate benchmark statistics
            benchmark_total_return = (1 + benchmark_aligned).prod() - 1
            
            # Calculate benchmark annualized return using actual time period
            start_date = pd.to_datetime(config.start_date)
            end_date = pd.to_datetime(config.end_date)
            time_period_years = (end_date - start_date).days / 365.25
            
            if time_period_years > 0:
                benchmark_annualized = ((1 + benchmark_total_return) ** (1 / time_period_years)) - 1
            else:
                benchmark_annualized = benchmark_total_return
                
            benchmark_volatility = benchmark_aligned.std() * np.sqrt(252)
            
            # Calculate benchmark max drawdown
            benchmark_cumulative = (1 + benchmark_aligned).cumprod()
            benchmark_peak = benchmark_cumulative.cummax()
            benchmark_drawdown = (benchmark_cumulative - benchmark_peak) / benchmark_peak
            benchmark_max_dd = abs(benchmark_drawdown.min())
            
            # Calculate excess return and tracking error
            excess_returns = php_aligned - benchmark_aligned
            annualized_excess_return = excess_returns.mean() * 252
            tracking_error = excess_returns.std() * np.sqrt(252)
            information_ratio = annualized_excess_return / tracking_error if tracking_error > 0 else 0
            
            # Print results
            print(f"\n{'Metric':<25} {'PHP':<12} {'6040 Bench':<12} {'Difference':<12}")
            print("-" * 65)
            print(f"{'Total Return':<25} {php_sim['total_return']:>11.1%} {benchmark_total_return:>11.1%} {php_sim['total_return'] - benchmark_total_return:>11.1%}")
            print(f"{'Annualized Return':<25} {php_sim['annualized_return']:>11.1%} {benchmark_annualized:>11.1%} {php_sim['annualized_return'] - benchmark_annualized:>11.1%}")
            print(f"{'Volatility':<25} {php_sim['volatility']:>11.1%} {benchmark_volatility:>11.1%} {php_sim['volatility'] - benchmark_volatility:>11.1%}")
            print(f"{'Max Drawdown':<25} {php_sim['max_drawdown']:>11.1%} {benchmark_max_dd:>11.1%} {php_sim['max_drawdown'] - benchmark_max_dd:>11.1%}")
            print(f"{'Sharpe Ratio':<25} {php_sim['sharpe_ratio']:>11.2f} {(benchmark_aligned.mean() * 252) / (benchmark_aligned.std() * np.sqrt(252)):>11.2f} {php_sim['sharpe_ratio'] - (benchmark_aligned.mean() * 252) / (benchmark_aligned.std() * np.sqrt(252)):>11.2f}")
            
            print(f"\n{'Excess Return Metrics':<25}")
            print("-" * 40)
            print(f"{'Annualized Excess Return':<25} {annualized_excess_return:>11.1%}")
            print(f"{'Tracking Error':<25} {tracking_error:>11.1%}")
            print(f"{'Information Ratio':<25} {information_ratio:>11.2f}")
            print(f"\nNote: Annualized excess return uses geometric mean of daily excess returns,")
            print(f"      which may differ from arithmetic difference of annualized returns")
            print(f"      (Arithmetic difference: {php_sim['annualized_return'] - benchmark_annualized:+.1%})")
            
            print(f"\n{'Transaction Metrics':<25}")
            print("-" * 40)
            print(f"{'Total Transaction Cost':<25} {php_sim['total_transaction_cost']:>11.2%}")
            print(f"{'Scheduled Rebalances':<25} {len(php_sim['rebalancing_dates']):>11d}")
            print(f"{'Forced Rebalances':<25} {len(php_sim['forced_rebalancing_dates']):>11d}")
            
            # Show perfect weights allocation
            perfect_weights = php_result['perfect_weights']
            neutral_weights = php_result['neutral_weights']
            print(f"\n{'Perfect Weight Allocation':<35}")
            print("-" * 50)
            for asset in neutral_weights:
                neutral = neutral_weights[asset]
                perfect = perfect_weights.get(asset, 0)
                deviation = perfect - neutral
                print(f"{asset[:30]:<30} {neutral:>6.1%} → {perfect:>6.1%} ({deviation:>+6.1%})")
            
        except Exception as e:
            print(f"❌ Error in scenario: {e}")
        
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    test_simulation_scenarios()