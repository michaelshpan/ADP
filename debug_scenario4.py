from portfolio_simulation import PortfolioSimulation, SimulationConfig
import pandas as pd
import numpy as np

def debug_scenario4_performance():
    """Deep dive into why Scenario 4 shows -3.2% when holding strong performers"""
    
    sim_engine = PortfolioSimulation()
    
    config = SimulationConfig(
        mandate="EQFILAIA",
        start_date="2012-01-01",
        end_date="2017-01-01",
        investment_horizon_years=5,
        permitted_deviation=0.50,
        rebalancing_frequency="none",
        transaction_cost_bps=5,
        scenario_id="Debug_Scenario4"
    )
    
    print("=== DEBUGGING SCENARIO 4: THE MYSTERY OF -3.2% ===\n")
    
    # Get PHP results
    php_result = sim_engine.simulate_php_performance(config)
    perfect_weights = php_result['perfect_weights']
    sim_results = php_result['simulation_results']
    asset_performance = php_result['asset_performance']
    
    print("PERFECT WEIGHTS ALLOCATION:")
    print("-" * 40)
    for asset, weight in perfect_weights.items():
        if weight > 0:
            perf = asset_performance.get(asset, 0)
            print(f"{asset[:30]:<30}: {weight:>6.1%} (perf: {perf:>7.1%})")
    
    print(f"\nEXPECTED TOTAL RETURN: {sum(perfect_weights[asset] * asset_performance[asset] for asset in perfect_weights if asset in asset_performance):.1%}")
    print(f"ACTUAL TOTAL RETURN: {sim_results['total_return']:.1%}")
    
    # Load the actual simulation data to see what happened day by day
    returns_data = sim_engine._load_simulation_data(config)
    print(f"\nDATA PERIOD: {returns_data.index[0]} to {returns_data.index[-1]}")
    print(f"TRADING DAYS: {len(returns_data)}")
    
    # Check what assets are actually in the data
    print(f"\nAVAILABLE ASSETS IN DATA:")
    for i, asset in enumerate(returns_data.columns):
        print(f"  {i+1}. {asset}")
    
    # Check if the high-performing assets have data
    high_performers = ['Refinitiv Venture Capital', 'Refinitiv Private Equity Buyout']
    
    print(f"\nHIGH PERFORMER DATA CHECK:")
    for asset in high_performers:
        if asset in returns_data.columns:
            data = returns_data[asset].dropna()
            total_return = (1 + data).prod() - 1
            print(f"\n{asset}:")
            print(f"  Data available: YES")
            print(f"  Non-null days: {len(data)} / {len(returns_data)}")
            print(f"  First value: {data.iloc[0]:.6f}")
            print(f"  Last value: {data.iloc[-1]:.6f}")
            print(f"  Total return from daily data: {total_return:.1%}")
            print(f"  Expected performance: {asset_performance[asset]:.1%}")
            print(f"  Difference: {total_return - asset_performance[asset]:.1%}")
            
            # Show first 10 and last 10 returns
            print(f"  First 10 returns: {data.head(10).values}")
            print(f"  Last 10 returns: {data.tail(10).values}")
            
            # Check for suspicious patterns
            zero_returns = (data == 0).sum()
            print(f"  Zero return days: {zero_returns}")
            
            # Check volatility
            volatility = data.std() * np.sqrt(252)
            print(f"  Annualized volatility: {volatility:.1%}")
            
        else:
            print(f"\n{asset}: DATA NOT AVAILABLE")
    
    # Manual calculation - what should the portfolio return be?
    print(f"\nMANUAL VERIFICATION:")
    print("-" * 30)
    
    filtered_weights = {asset: weight for asset, weight in perfect_weights.items() 
                       if asset in returns_data.columns and weight > 0}
    
    if filtered_weights:
        # Normalize filtered weights
        total_filtered = sum(filtered_weights.values())
        normalized_weights = {asset: weight/total_filtered for asset, weight in filtered_weights.items()}
        
        print("NORMALIZED WEIGHTS FOR AVAILABLE ASSETS:")
        for asset, weight in normalized_weights.items():
            print(f"  {asset[:30]:<30}: {weight:>6.1%}")
        
        # Calculate what the return should be
        manual_total_return = 0
        for asset, weight in normalized_weights.items():
            asset_data = returns_data[asset].dropna()
            asset_total_return = (1 + asset_data).prod() - 1
            contribution = weight * asset_total_return
            manual_total_return += contribution
            print(f"  {asset[:30]:<30}: {asset_total_return:>7.1%} × {weight:>6.1%} = {contribution:>7.1%}")
        
        print(f"\nMANUAL CALCULATED TOTAL RETURN: {manual_total_return:.1%}")
        print(f"SIMULATION ACTUAL RETURN: {sim_results['total_return']:.1%}")
        print(f"DIFFERENCE: {sim_results['total_return'] - manual_total_return:.1%}")
    
    # Check if there are any missing or problematic data points
    print(f"\nDATA QUALITY CHECK:")
    print("-" * 25)
    
    for asset in returns_data.columns:
        data = returns_data[asset]
        missing_count = data.isna().sum()
        zero_count = (data == 0).sum()
        extreme_positive = (data > 0.5).sum()  # >50% daily return
        extreme_negative = (data < -0.5).sum()  # <-50% daily return
        
        print(f"{asset[:30]:<30}: Missing={missing_count:>4d}, Zeros={zero_count:>4d}, Extreme+={extreme_positive:>2d}, Extreme-={extreme_negative:>2d}")
    
    # Check the portfolio simulation step by step for first few days
    print(f"\nSIMULATION STEP-BY-STEP (First 10 days):")
    print("-" * 50)
    
    # Get the actual weights used in simulation (after filtering and normalization)
    available_assets = returns_data.columns.tolist()
    filtered_perfect_weights = {asset: weight for asset, weight in perfect_weights.items() 
                               if asset in available_assets}
    total_available_weight = sum(filtered_perfect_weights.values())
    if total_available_weight > 0:
        normalized_perfect_weights = {asset: weight / total_available_weight 
                                    for asset, weight in filtered_perfect_weights.items()}
    else:
        normalized_perfect_weights = {}
    
    print("WEIGHTS USED IN SIMULATION:")
    for asset, weight in normalized_perfect_weights.items():
        print(f"  {asset[:30]:<30}: {weight:>6.1%}")
    
    portfolio_value = 1.0
    print(f"\nDay 0: Portfolio Value = {portfolio_value:.6f}")
    
    for i in range(min(10, len(returns_data))):
        date = returns_data.index[i]
        daily_returns = returns_data.iloc[i]
        
        daily_portfolio_return = 0
        print(f"\nDay {i+1} ({date.strftime('%Y-%m-%d')}):")
        
        for asset, weight in normalized_perfect_weights.items():
            if asset in daily_returns.index:
                asset_return = daily_returns[asset]
                contribution = weight * asset_return
                daily_portfolio_return += contribution
                print(f"  {asset[:25]:<25}: {asset_return:>8.4f} × {weight:>6.1%} = {contribution:>8.4f}")
        
        portfolio_value *= (1 + daily_portfolio_return)
        print(f"  Portfolio Return: {daily_portfolio_return:>8.4f} ({daily_portfolio_return*100:>6.2f}%)")
        print(f"  Portfolio Value: {portfolio_value:>8.6f}")

if __name__ == "__main__":
    debug_scenario4_performance()