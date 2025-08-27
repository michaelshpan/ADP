import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
from data_mapping import DataMapper
from portfolio_config import PortfolioConfig

class PerfectWeightCalculator:
    """Module for calculating Perfect Hindsight Portfolio weights based on asset class performance"""
    
    def __init__(self, data_mapper: DataMapper = None, portfolio_config: PortfolioConfig = None):
        self.data_mapper = data_mapper or DataMapper()
        self.portfolio_config = portfolio_config or PortfolioConfig(self.data_mapper)
        
        # Scenario parameters from specs/scenarios.txt
        self.deviation_limits = [0.05, 0.10, 0.20, 0.50]  # ±5%, ±10%, ±20%, ±50% absolute
        self.investment_horizons = [5, 10]  # 5-year and 10-year
        
    def calculate_asset_performance(self, 
                                  mandate: str,
                                  start_date: str, 
                                  end_date: str) -> Dict[str, float]:
        """
        Calculate absolute performance for each asset class over the investment horizon
        
        CRITICAL: Must use IDENTICAL data as portfolio simulation
        
        Args:
            mandate: Portfolio mandate (6040, EQFI, EQFILA, EQFILAIA)
            start_date: Investment start date
            end_date: Investment end date
            
        Returns:
            Dictionary of asset class absolute returns
        """
        # Load all required data with daily frequency enforcement (same as simulation)
        data_dict = self.data_mapper.load_all_required_data(mandate, start_date, end_date, ensure_daily=True)
        
        asset_performance = {}
        
        for asset_name, data in data_dict.items():
            if data.empty:
                warnings.warn(f"No data available for {asset_name}")
                asset_performance[asset_name] = 0.0
                continue
                
            # Calculate total return over the period using the same method as simulation
            # This ensures consistency between perfect weight calculation and actual simulation
            start_level = data['LEVEL'].iloc[0]
            end_level = data['LEVEL'].iloc[-1]
            total_return = (end_level / start_level) - 1.0
            
            asset_performance[asset_name] = total_return
            
            # Performance calculation completed for asset
            
        return asset_performance
    
    def calculate_perfect_weights(self,
                                mandate: str,
                                asset_performance: Dict[str, float],
                                permitted_deviation: float) -> Dict[str, float]:
        """
        Calculate Perfect Weights using improved logic that prioritizes best performers
        
        New Logic:
        1. Sort assets by performance (best to worst)
        2. Allocate maximum allowed weight to best performers first
        3. Continue until all weight is allocated or all assets are at bounds
        4. Handle negative returns by setting to minimum weight
        
        Args:
            mandate: Portfolio mandate
            asset_performance: Dict of asset absolute returns
            permitted_deviation: Maximum absolute deviation from neutral weights (e.g., 0.05 for ±5%)
            
        Returns:
            Dictionary of perfect weights
        """
        # Get neutral weights for the mandate
        neutral_weights = self.portfolio_config.get_mandate_weights(mandate)
        
        # Filter performance to only include assets in the mandate
        filtered_performance = {asset: perf for asset, perf in asset_performance.items() 
                               if asset in neutral_weights}
        
        if not filtered_performance:
            warnings.warn(f"No matching assets found for mandate {mandate}")
            return neutral_weights
        
        # Calculate min/max weights using ABSOLUTE deviations
        weight_bounds = {}
        for asset in neutral_weights:
            neutral_weight = neutral_weights[asset]
            min_weight = max(0.0, neutral_weight - permitted_deviation)
            max_weight = min(1.0, neutral_weight + permitted_deviation)
            weight_bounds[asset] = (min_weight, max_weight)
        
        perfect_weights = {}
        
        # Step 1: Set negative performing assets to minimum weight
        negative_assets = {asset: perf for asset, perf in filtered_performance.items() if perf < 0}
        for asset, perf in negative_assets.items():
            min_weight, _ = weight_bounds[asset]
            perfect_weights[asset] = min_weight
        
        # Step 2: Calculate remaining weight to allocate
        allocated_weight = sum(perfect_weights.values())
        remaining_weight = 1.0 - allocated_weight
        
        # Step 3: Sort positive performing assets by performance (best first)
        positive_assets = {asset: perf for asset, perf in filtered_performance.items() if perf >= 0}
        sorted_positive = sorted(positive_assets.items(), key=lambda x: x[1], reverse=True)
        
        # Step 4: Allocate remaining weight to positive performers (best first)
        if sorted_positive and remaining_weight > 0:
            # Initialize all positive assets to their minimum weight
            for asset, _ in sorted_positive:
                min_weight, _ = weight_bounds[asset]
                perfect_weights[asset] = min_weight
                remaining_weight -= min_weight
            
            # Allocate remaining weight to best performers first, up to their max
            while remaining_weight > 0.001 and sorted_positive:  # Small threshold for floating point
                weight_allocated_this_round = 0
                
                for asset, perf in sorted_positive:
                    if remaining_weight <= 0.001:
                        break
                    
                    min_weight, max_weight = weight_bounds[asset]
                    current_weight = perfect_weights[asset]
                    
                    # How much more can this asset receive?
                    additional_capacity = max_weight - current_weight
                    
                    if additional_capacity > 0.001:
                        # Give this asset as much as possible (up to remaining weight)
                        additional_allocation = min(remaining_weight, additional_capacity)
                        perfect_weights[asset] += additional_allocation
                        remaining_weight -= additional_allocation
                        weight_allocated_this_round += additional_allocation
                
                # If no weight was allocated this round, all assets are at max - break
                if weight_allocated_this_round < 0.001:
                    break
            
            # If there's still remaining weight, distribute proportionally to all positive assets
            # that haven't hit their max (fallback for edge cases)
            if remaining_weight > 0.001:
                available_assets = []
                for asset, _ in sorted_positive:
                    min_weight, max_weight = weight_bounds[asset]
                    if perfect_weights[asset] < max_weight - 0.001:
                        available_assets.append(asset)
                
                if available_assets:
                    # Distribute remaining weight equally among available assets
                    per_asset_weight = remaining_weight / len(available_assets)
                    for asset in available_assets:
                        min_weight, max_weight = weight_bounds[asset]
                        additional = min(per_asset_weight, max_weight - perfect_weights[asset])
                        perfect_weights[asset] += additional
        
        # Handle case where all assets have zero performance
        elif not positive_assets and not negative_assets:
            # Use neutral weights within constraints
            for asset in filtered_performance:
                min_weight, max_weight = weight_bounds[asset]
                neutral_weight = neutral_weights[asset]
                perfect_weights[asset] = np.clip(neutral_weight, min_weight, max_weight)
        
        # Final normalization only if total is significantly off from 1.0
        total_weight = sum(perfect_weights.values())
        if abs(total_weight - 1.0) > 0.01:  # Only normalize if more than 1% off
            if total_weight > 0:
                perfect_weights = {asset: weight / total_weight 
                                  for asset, weight in perfect_weights.items()}
        
        return perfect_weights
    
    def generate_rolling_php_scenarios(self,
                                     mandate: str,
                                     analysis_start: str = "1996-01-01",
                                     analysis_end: str = "2025-05-31") -> List[Dict]:
        """
        Generate all rolling PHP scenarios based on specs/scenarios.txt
        
        Returns list of scenario configurations:
        - Investment horizon (5-year, 10-year)
        - Monthly rolling start dates
        - Permitted deviation levels
        """
        scenarios = []
        
        analysis_start_date = pd.to_datetime(analysis_start)
        analysis_end_date = pd.to_datetime(analysis_end)
        
        for horizon_years in self.investment_horizons:
            for deviation in self.deviation_limits:
                # Generate monthly rolling start dates
                current_date = analysis_start_date
                
                while current_date + pd.DateOffset(years=horizon_years) <= analysis_end_date:
                    end_date = current_date + pd.DateOffset(years=horizon_years)
                    
                    scenario = {
                        'mandate': mandate,
                        'start_date': current_date.strftime('%Y-%m-%d'),
                        'end_date': end_date.strftime('%Y-%m-%d'),
                        'investment_horizon_years': horizon_years,
                        'permitted_deviation': deviation,
                        'scenario_id': f"{mandate}_{current_date.strftime('%Y%m')}_{horizon_years}Y_{int(deviation*100)}pct"
                    }
                    scenarios.append(scenario)
                    
                    # Move to next month (first day of month)
                    current_date = current_date + pd.DateOffset(months=1)
                    current_date = current_date.replace(day=1)
        
        return scenarios
    
    def calculate_php_perfect_weights(self, scenario: Dict) -> Dict:
        """
        Calculate perfect weights for a specific PHP scenario
        
        Args:
            scenario: Scenario configuration dictionary
            
        Returns:
            Dictionary with scenario info and calculated perfect weights
        """
        # Calculate asset performance over the investment horizon
        asset_performance = self.calculate_asset_performance(
            scenario['mandate'],
            scenario['start_date'],
            scenario['end_date']
        )
        
        # Calculate perfect weights
        perfect_weights = self.calculate_perfect_weights(
            scenario['mandate'],
            asset_performance,
            scenario['permitted_deviation']
        )
        
        # Get neutral weights for comparison
        neutral_weights = self.portfolio_config.get_mandate_weights(scenario['mandate'])
        
        result = {
            **scenario,
            'asset_performance': asset_performance,
            'perfect_weights': perfect_weights,
            'neutral_weights': neutral_weights,
            'weight_deviations': {asset: perfect_weights.get(asset, 0) - neutral_weights.get(asset, 0)
                                 for asset in neutral_weights.keys()}
        }
        
        return result
    
    def print_perfect_weight_example(self, mandate: str = "EQFI", 
                                   start_date: str = "2010-01-01", 
                                   end_date: str = "2020-01-01",
                                   deviation: float = 0.05):
        """Print an example of perfect weight calculation"""
        print(f"=== PERFECT WEIGHT EXAMPLE ===")
        print(f"Mandate: {mandate}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Permitted Deviation: ±{deviation:.0%} (absolute)")
        
        # Show neutral weights and bounds
        neutral_weights = self.portfolio_config.get_mandate_weights(mandate)
        print(f"\n=== WEIGHT BOUNDS ===")
        print(f"{'Asset':<35} {'Neutral':<8} {'Min':<8} {'Max':<8}")
        print("-" * 65)
        for asset, neutral in neutral_weights.items():
            min_wgt = max(0.0, neutral - deviation)
            max_wgt = min(1.0, neutral + deviation)
            print(f"{asset:<35} {neutral:>7.1%} {min_wgt:>7.1%} {max_wgt:>7.1%}")
        
        # Calculate asset performance
        performance = self.calculate_asset_performance(mandate, start_date, end_date)
        print(f"\n=== ASSET PERFORMANCE ({start_date} to {end_date}) ===")
        for asset, perf in performance.items():
            print(f"{asset:<35}: {perf:>8.1%}")
        
        # Calculate perfect weights
        perfect_weights = self.calculate_perfect_weights(mandate, performance, deviation)
        
        print(f"\n=== WEIGHT COMPARISON ===")
        print(f"{'Asset':<35} {'Neutral':<8} {'Perfect':<8} {'Deviation':<10}")
        print("-" * 70)
        
        for asset in neutral_weights.keys():
            neutral = neutral_weights[asset]
            perfect = perfect_weights.get(asset, 0)
            deviation_val = perfect - neutral
            
            print(f"{asset:<35} {neutral:>7.1%} {perfect:>7.1%} {deviation_val:>9.1%}")
        
        print(f"\nTotal Perfect Weight: {sum(perfect_weights.values()):.1%}")

if __name__ == "__main__":
    # Test the perfect weight calculator
    calc = PerfectWeightCalculator()
    
    # Print example calculation
    calc.print_perfect_weight_example()
    
    # Test scenario generation
    scenarios = calc.generate_rolling_php_scenarios("EQFI", "2010-01-01", "2020-01-01")
    print(f"\n=== SCENARIO GENERATION TEST ===")
    print(f"Generated {len(scenarios)} scenarios for EQFI (2010-2020 sample)")
    print("First 3 scenarios:")
    for i, scenario in enumerate(scenarios[:3]):
        print(f"{i+1}: {scenario['scenario_id']}")
        print(f"   Period: {scenario['start_date']} to {scenario['end_date']}")
        print(f"   Deviation: ±{scenario['permitted_deviation']:.0%}")