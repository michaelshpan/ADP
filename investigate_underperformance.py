from portfolio_simulation import PortfolioSimulation, SimulationConfig
import pandas as pd
import numpy as np

def investigate_php_underperformance():
    """Investigate why Scenarios 4 and 5 show PHP underperformance"""
    
    sim_engine = PortfolioSimulation()
    
    # The two underperforming scenarios
    scenarios = [
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
    
    print("=== INVESTIGATING PHP UNDERPERFORMANCE ===\n")
    
    for scenario in scenarios:
        print(f"=== {scenario.scenario_id} ===")
        print(f"Mandate: {scenario.mandate}")
        print(f"Period: {scenario.start_date} to {scenario.end_date}")
        print(f"Deviation: ±{scenario.permitted_deviation:.0%}")
        print(f"Rebalancing: {scenario.rebalancing_frequency}")
        print(f"Transaction Cost: {scenario.transaction_cost_bps} bps\n")
        
        # Run PHP simulation
        php_result = sim_engine.simulate_php_performance(scenario)
        
        if not php_result['success']:
            print(f"❌ Error: {php_result['error']}\n")
            continue
        
        # Get detailed results
        asset_performance = php_result['asset_performance']
        perfect_weights = php_result['perfect_weights']
        neutral_weights = php_result['neutral_weights']
        sim_results = php_result['simulation_results']
        
        print("1. ASSET CLASS PERFORMANCE ANALYSIS")
        print("-" * 50)
        sorted_performance = sorted(asset_performance.items(), key=lambda x: x[1], reverse=True)
        
        for asset, performance in sorted_performance:
            neutral_wgt = neutral_weights.get(asset, 0)
            perfect_wgt = perfect_weights.get(asset, 0)
            weight_change = perfect_wgt - neutral_wgt
            contribution = perfect_wgt * performance
            
            print(f"{asset[:30]:<30}")
            print(f"  Performance: {performance:>8.1%}")
            print(f"  Weight: {neutral_wgt:>6.1%} → {perfect_wgt:>6.1%} ({weight_change:>+6.1%})")
            print(f"  Contribution: {contribution:>8.1%}")
            print()
        
        # Calculate total expected return from perfect weights
        expected_return = sum(perfect_weights[asset] * asset_performance[asset] 
                            for asset in perfect_weights if asset in asset_performance)
        
        print(f"Expected Total Return (no costs): {expected_return:.1%}")
        print(f"Actual PHP Total Return: {sim_results['total_return']:.1%}")
        print(f"Performance Drag: {sim_results['total_return'] - expected_return:.1%}")
        
        print(f"\n2. TRANSACTION COST BREAKDOWN")
        print("-" * 50)
        print(f"Total Transaction Cost: {sim_results['total_transaction_cost']:.3%}")
        print(f"Scheduled Rebalances: {len(sim_results['rebalancing_dates'])}")
        print(f"Forced Rebalances: {len(sim_results['forced_rebalancing_dates'])}")
        print(f"Total Rebalancing Events: {len(sim_results['transaction_costs'])}")
        
        if sim_results['transaction_costs']:
            total_tx_cost = sum([tx['cost'] for tx in sim_results['transaction_costs']])
            avg_tx_cost = total_tx_cost / len(sim_results['transaction_costs'])
            print(f"Average Cost per Rebalance: {avg_tx_cost:.3%}")
            
            # Show biggest transaction costs
            sorted_tx = sorted(sim_results['transaction_costs'], key=lambda x: x['cost'], reverse=True)
            print(f"\nLargest 5 Transaction Costs:")
            for i, tx in enumerate(sorted_tx[:5]):
                print(f"  {i+1}. {tx['date'].strftime('%Y-%m-%d')}: {tx['cost']:.3%} ({tx['reason']})")
        
        print(f"\n3. BENCHMARK COMPARISON")
        print("-" * 50)
        
        # Get 6040 benchmark performance
        benchmark_returns = sim_engine.portfolio_config.calculate_benchmark_6040(
            scenario.start_date, scenario.end_date
        )
        
        if not benchmark_returns.empty:
            benchmark_total = (1 + benchmark_returns).prod() - 1
            start_date = pd.to_datetime(scenario.start_date)
            end_date = pd.to_datetime(scenario.end_date)
            time_period = (end_date - start_date).days / 365.25
            benchmark_annualized = ((1 + benchmark_total) ** (1 / time_period)) - 1
            
            print(f"6040 Benchmark Total Return: {benchmark_total:.1%}")
            print(f"6040 Benchmark Annualized: {benchmark_annualized:.1%}")
            print(f"PHP vs Benchmark: {sim_results['total_return'] - benchmark_total:.1%}")
            
            # Show what 6040 allocation would have returned with same asset performance
            benchmark_expected = (0.6 * asset_performance.get('Russell 3000', 0) +
                                0.28 * asset_performance.get('ICE BofA US Treasury', 0) +
                                0.12 * asset_performance.get('ICE BofAML US Corporate', 0))
            
            print(f"Expected 6040 Return (from asset performance): {benchmark_expected:.1%}")
            print(f"Expected PHP Return (from asset performance): {expected_return:.1%}")
            print(f"Perfect Hindsight Advantage (before costs): {expected_return - benchmark_expected:.1%}")
            
        print(f"\n4. ANALYSIS & DIAGNOSIS")
        print("-" * 50)
        
        # Identify key issues
        issues = []
        
        # Check for negative performing assets with high allocations
        for asset, weight in perfect_weights.items():
            if weight > 0.1 and asset in asset_performance:  # >10% weight
                perf = asset_performance[asset]
                if perf < 0:
                    issues.append(f"High allocation ({weight:.0%}) to negative performer: {asset} ({perf:.1%})")
        
        # Check transaction costs
        if sim_results['total_transaction_cost'] > 0.01:  # >1%
            issues.append(f"High transaction costs: {sim_results['total_transaction_cost']:.2%}")
        
        # Check for extreme weight shifts
        for asset in neutral_weights:
            neutral_wgt = neutral_weights[asset]
            perfect_wgt = perfect_weights.get(asset, 0)
            shift = abs(perfect_wgt - neutral_wgt)
            if shift > 0.2:  # >20% weight shift
                direction = "to" if perfect_wgt > neutral_wgt else "from"
                issues.append(f"Extreme weight shift {direction} {asset}: {neutral_wgt:.0%} → {perfect_wgt:.0%}")
        
        if issues:
            print("Key Issues Identified:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
        else:
            print("No obvious structural issues identified.")
            print("Underperformance may be due to:")
            print("  - Poor asset class selection during this period")
            print("  - Timing/volatility drag in perfect weight implementation")
            print("  - Inherent limitations of the perfect hindsight approach")
        
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    investigate_php_underperformance()