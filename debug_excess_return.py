from portfolio_simulation import PortfolioSimulation, SimulationConfig
import pandas as pd
import numpy as np

def debug_excess_return():
    """Debug the excess return calculation discrepancy"""
    
    sim_engine = PortfolioSimulation()
    
    # Test Scenario 1: Financial Crisis Recovery
    config = SimulationConfig(
        mandate="EQFI",
        start_date="2008-01-01",
        end_date="2013-01-01",
        investment_horizon_years=5,
        permitted_deviation=0.20,
        rebalancing_frequency="quarterly",
        transaction_cost_bps=25,
        scenario_id="Debug_Test"
    )
    
    print("=== DEBUGGING EXCESS RETURN CALCULATION ===\n")
    
    # Get PHP results
    php_result = sim_engine.simulate_php_performance(config)
    php_sim = php_result['simulation_results']
    php_returns = php_sim['portfolio_returns']
    
    print(f"PHP Annualized Return: {php_sim['annualized_return']:.3%}")
    print(f"PHP Time Period: {php_sim['time_period_years']:.2f} years")
    print(f"PHP Final Value: {php_sim['final_value']:.4f}")
    
    # Get benchmark results
    benchmark_returns = sim_engine.portfolio_config.calculate_benchmark_6040(
        config.start_date, config.end_date
    )
    
    # Calculate benchmark annualized return using same method as PHP
    start_date = pd.to_datetime(config.start_date)
    end_date = pd.to_datetime(config.end_date)
    time_period_years = (end_date - start_date).days / 365.25
    
    benchmark_total_return = (1 + benchmark_returns).prod() - 1
    benchmark_annualized = ((1 + benchmark_total_return) ** (1 / time_period_years)) - 1
    
    print(f"\nBenchmark Annualized Return: {benchmark_annualized:.3%}")
    print(f"Benchmark Total Return: {benchmark_total_return:.3%}")
    print(f"Benchmark Final Value: {1 + benchmark_total_return:.4f}")
    
    # Calculate excess return - Method 1 (difference of annualized returns)
    excess_method1 = php_sim['annualized_return'] - benchmark_annualized
    print(f"\nMethod 1 - Difference of Annualized Returns: {excess_method1:.3%}")
    
    # Align returns for daily calculations
    common_dates = php_returns.index.intersection(benchmark_returns.index)
    php_aligned = php_returns.loc[common_dates]
    benchmark_aligned = benchmark_returns.loc[common_dates]
    
    print(f"\nOverlapping days: {len(common_dates)}")
    print(f"PHP aligned period: {php_aligned.index[0]} to {php_aligned.index[-1]}")
    print(f"Benchmark aligned period: {benchmark_aligned.index[0]} to {benchmark_aligned.index[-1]}")
    
    # Calculate excess return - Method 2 (annualized daily excess returns)
    excess_returns_daily = php_aligned - benchmark_aligned
    excess_method2 = excess_returns_daily.mean() * 252
    
    print(f"\nMethod 2 - Annualized Daily Excess Returns: {excess_method2:.3%}")
    
    # Calculate what the annualized returns should be using the aligned data
    php_aligned_total = (1 + php_aligned).prod() - 1
    benchmark_aligned_total = (1 + benchmark_aligned).prod() - 1
    
    aligned_period_years = (php_aligned.index[-1] - php_aligned.index[0]).days / 365.25
    
    php_aligned_annualized = ((1 + php_aligned_total) ** (1 / aligned_period_years)) - 1
    benchmark_aligned_annualized = ((1 + benchmark_aligned_total) ** (1 / aligned_period_years)) - 1
    
    print(f"\nUsing Aligned Data:")
    print(f"PHP Aligned Annualized: {php_aligned_annualized:.3%}")
    print(f"Benchmark Aligned Annualized: {benchmark_aligned_annualized:.3%}")
    print(f"Aligned Period Excess: {php_aligned_annualized - benchmark_aligned_annualized:.3%}")
    
    # Show statistics
    print(f"\nDaily Statistics:")
    print(f"PHP daily mean: {php_aligned.mean():.6f} ({php_aligned.mean() * 252:.3%} annualized)")
    print(f"Benchmark daily mean: {benchmark_aligned.mean():.6f} ({benchmark_aligned.mean() * 252:.3%} annualized)")
    print(f"Excess daily mean: {excess_returns_daily.mean():.6f} ({excess_returns_daily.mean() * 252:.3%} annualized)")

if __name__ == "__main__":
    debug_excess_return()