import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
import os
import sys
from functools import partial
from data_mapping import DataMapper
from portfolio_config import PortfolioConfig
from perfect_weight_calculator import PerfectWeightCalculator

# Global function for multiprocessing (must be at module level for pickling)
def _simulate_scenario_worker(scenario_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker function for parallel scenario simulation
    
    This function must be defined at module level to be pickleable for multiprocessing
    """
    try:
        scenario = scenario_data['scenario']
        data_dir = scenario_data['data_dir']
        specs_dir = scenario_data['specs_dir']
        
        # Create fresh instances for each worker process
        data_mapper = DataMapper(data_dir, specs_dir)
        portfolio_config = PortfolioConfig(data_mapper)
        perfect_weight_calc = PerfectWeightCalculator(data_mapper, portfolio_config)
        simulation_engine = PortfolioSimulation(data_mapper, portfolio_config, perfect_weight_calc)
        
        # Run the simulation
        result = simulation_engine.simulate_php_performance(scenario)
        return result
        
    except Exception as e:
        # Return error result
        return {
            'config': scenario_data.get('scenario'),
            'success': False,
            'error': f"Worker error: {str(e)}",
            'simulation_results': None
        }

def _create_simulation_wrapper():
    """Create the simulation wrapper function for multiprocessing"""
    return _simulate_scenario_worker

@dataclass
class SimulationConfig:
    """Configuration for PHP simulation"""
    mandate: str
    start_date: str
    end_date: str
    investment_horizon_years: int
    permitted_deviation: float
    rebalancing_frequency: str  # 'monthly', 'quarterly', 'annual', 'none'
    transaction_cost_bps: int  # Transaction cost in basis points
    scenario_id: str

@dataclass 
class ParallelConfig:
    """Configuration for parallel processing"""
    use_parallel: bool = True
    max_workers: Optional[int] = None  # None = auto-detect
    chunk_size: Optional[int] = None  # None = auto-calculate
    fallback_on_error: bool = True  # Fall back to sequential on error
    progress_reporting: bool = True  # Show progress updates
    memory_limit_gb: Optional[float] = None  # Memory limit per worker (not implemented yet)

class PortfolioSimulation:
    """Module for simulating Perfect Hindsight Portfolio performance"""
    
    def __init__(self, 
                 data_mapper: DataMapper = None,
                 portfolio_config: PortfolioConfig = None,
                 perfect_weight_calc: PerfectWeightCalculator = None):
        
        self.data_mapper = data_mapper or DataMapper()
        self.portfolio_config = portfolio_config or PortfolioConfig(self.data_mapper)
        self.perfect_weight_calc = perfect_weight_calc or PerfectWeightCalculator(
            self.data_mapper, self.portfolio_config
        )
        
        # Scenario parameters from specs/scenarios.txt
        self.rebalancing_frequencies = ['monthly', 'quarterly', 'annual', 'none']
        self.transaction_costs_bps = [5, 25, 50, 100]  # 5bps, 25bps, 50bps, 100bps
        self.investment_horizons = [5, 10]  # 5-year and 10-year
        self.deviation_limits = [0.05, 0.10, 0.20, 0.50]  # ±5%, ±10%, ±20%, ±50%
    
    def generate_all_php_scenarios(self,
                                 mandate: str,
                                 analysis_start: str = "1996-01-01",
                                 analysis_end: str = "2025-05-31") -> List[SimulationConfig]:
        """
        Generate all possible PHP simulation scenarios
        
        Returns:
            List of SimulationConfig objects for all scenario combinations
        """
        scenarios = []
        
        analysis_start_date = pd.to_datetime(analysis_start)
        analysis_end_date = pd.to_datetime(analysis_end)
        
        for horizon_years in self.investment_horizons:
            for deviation in self.deviation_limits:
                for rebal_freq in self.rebalancing_frequencies:
                    for tx_cost_bps in self.transaction_costs_bps:
                        
                        # Generate monthly rolling start dates
                        current_date = analysis_start_date
                        
                        while current_date + pd.DateOffset(years=horizon_years) <= analysis_end_date:
                            end_date = current_date + pd.DateOffset(years=horizon_years)
                            
                            scenario_id = (f"{mandate}_{current_date.strftime('%Y%m')}_"
                                        f"{horizon_years}Y_{int(deviation*100)}pct_"
                                        f"{rebal_freq}_{tx_cost_bps}bps")
                            
                            config = SimulationConfig(
                                mandate=mandate,
                                start_date=current_date.strftime('%Y-%m-%d'),
                                end_date=end_date.strftime('%Y-%m-%d'),
                                investment_horizon_years=horizon_years,
                                permitted_deviation=deviation,
                                rebalancing_frequency=rebal_freq,
                                transaction_cost_bps=tx_cost_bps,
                                scenario_id=scenario_id
                            )
                            scenarios.append(config)
                            
                            # Move to next month (first day of month)
                            current_date = current_date + pd.DateOffset(months=1)
                            current_date = current_date.replace(day=1)
        
        return scenarios
    
    def simulate_php_performance(self, config: SimulationConfig) -> Dict[str, Any]:
        """
        Simulate a single PHP scenario performance
        
        Args:
            config: Simulation configuration
            
        Returns:
            Dictionary with simulation results including returns, weights, transactions, etc.
        """
        try:
            # Step 1: Calculate perfect weights based on asset performance
            asset_performance = self.perfect_weight_calc.calculate_asset_performance(
                config.mandate, config.start_date, config.end_date
            )
            
            perfect_weights = self.perfect_weight_calc.calculate_perfect_weights(
                config.mandate, asset_performance, config.permitted_deviation
            )
            
            neutral_weights = self.portfolio_config.get_mandate_weights(config.mandate)
            
            # Step 2: Load daily return data for simulation period
            returns_data = self._load_simulation_data(config)
            
            if returns_data.empty:
                raise ValueError(f"No data available for simulation period")
            
            # Step 3: Simulate portfolio with perfect weights, rebalancing, and transaction costs
            simulation_results = self._run_portfolio_simulation(
                returns_data, perfect_weights, config
            )
            
            # Step 4: Compile results
            results = {
                'config': config,
                'asset_performance': asset_performance,
                'perfect_weights': perfect_weights,
                'neutral_weights': neutral_weights,
                'simulation_results': simulation_results,
                'success': True,
                'error': None
            }
            
            return results
            
        except Exception as e:
            return {
                'config': config,
                'success': False,
                'error': str(e),
                'simulation_results': None
            }
    
    def _load_simulation_data(self, config: SimulationConfig) -> pd.DataFrame:
        """
        Load and prepare daily return data for simulation
        
        CRITICAL: Must use IDENTICAL date range and data as perfect weight calculation
        """
        # Load all required data with daily frequency enforcement for hedge fund
        data_dict = self.data_mapper.load_all_required_data(
            config.mandate, config.start_date, config.end_date, ensure_daily=True
        )
        
        # Calculate daily returns for each asset
        returns_dict = {}
        for asset_name, data in data_dict.items():
            if not data.empty:
                daily_returns = data['LEVEL'].pct_change()
                returns_dict[asset_name] = daily_returns
        
        # Combine into DataFrame and drop missing data
        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.dropna()
        
        # Validation: Ensure we have the expected date range
        if not returns_df.empty:
            actual_start = returns_df.index[0]
            actual_end = returns_df.index[-1]
            expected_start = pd.to_datetime(config.start_date)
            expected_end = pd.to_datetime(config.end_date)
            
            # Check for significant date misalignment (more than 30 days)
            start_diff = abs((actual_start - expected_start).days)
            end_diff = abs((actual_end - expected_end).days)
            
            if start_diff > 30 or end_diff > 30:
                print(f"Warning: Simulation data period mismatch for {config.scenario_id}")
                print(f"  Expected: {config.start_date} to {config.end_date}")
                print(f"  Actual: {actual_start.strftime('%Y-%m-%d')} to {actual_end.strftime('%Y-%m-%d')}")
        
        return returns_df
    
    def _run_portfolio_simulation(self, 
                                returns_data: pd.DataFrame,
                                target_weights: Dict[str, float],
                                config: SimulationConfig) -> Dict[str, Any]:
        """
        Run the actual portfolio simulation with rebalancing and transaction costs
        
        Logic:
        1. Start with target weights (perfect weights)
        2. Let weights drift daily with asset performance
        3. Rebalance according to frequency
        4. Apply transaction costs when rebalancing
        5. Force rebalancing if weights exceed permitted deviation
        """
        
        # Filter target weights to only include assets with data
        available_assets = returns_data.columns.tolist()
        filtered_weights = {asset: weight for asset, weight in target_weights.items() 
                           if asset in available_assets}
        
        if not filtered_weights:
            raise ValueError("No assets with data found in target weights")
        
        # Normalize weights for available assets
        total_weight = sum(filtered_weights.values())
        if total_weight > 0:
            filtered_weights = {asset: weight / total_weight 
                              for asset, weight in filtered_weights.items()}
        
        # Initialize tracking variables
        portfolio_values = []
        portfolio_weights_history = []
        transaction_costs = []
        rebalancing_dates = []
        forced_rebalancing_dates = []
        
        # Get neutral weights for constraint checking
        neutral_weights = self.portfolio_config.get_mandate_weights(config.mandate)
        
        # Initial portfolio value and weights
        portfolio_value = 1.0
        current_weights = filtered_weights.copy()
        
        portfolio_values.append(portfolio_value)
        portfolio_weights_history.append(current_weights.copy())
        
        # Get rebalancing schedule
        rebalance_dates = self._get_rebalancing_dates(
            returns_data.index, config.rebalancing_frequency
        )
        
        # Simulate each day
        for i, date in enumerate(returns_data.index[1:], 1):
            daily_returns = returns_data.iloc[i]
            
            # Calculate new asset values after daily returns
            new_weights = {}
            total_portfolio_return = 0.0
            
            for asset in current_weights:
                if asset in daily_returns.index:
                    asset_return = daily_returns[asset]
                    asset_contribution = current_weights[asset] * (1 + asset_return)
                    total_portfolio_return += current_weights[asset] * asset_return
                    new_weights[asset] = asset_contribution
                else:
                    # If no return data, assume no change
                    new_weights[asset] = current_weights[asset]
            
            # Update portfolio value
            portfolio_value *= (1 + total_portfolio_return)
            
            # Normalize weights after drift
            total_weight = sum(new_weights.values())
            if total_weight > 0:
                current_weights = {asset: weight / total_weight 
                                 for asset, weight in new_weights.items()}
            
            # Check if forced rebalancing is needed due to constraint violations
            forced_rebalance = self._check_constraint_violations(
                current_weights, neutral_weights, config.permitted_deviation
            )
            
            # Check if scheduled rebalancing is needed
            scheduled_rebalance = date in rebalance_dates
            
            # Execute rebalancing if needed
            if forced_rebalance or scheduled_rebalance:
                transaction_cost = self._calculate_transaction_cost(
                    current_weights, filtered_weights, config.transaction_cost_bps
                )
                
                # Apply transaction cost to portfolio value
                portfolio_value *= (1 - transaction_cost)
                
                # Record transaction
                transaction_costs.append({
                    'date': date,
                    'cost': transaction_cost,
                    'reason': 'forced' if forced_rebalance else 'scheduled'
                })
                
                if forced_rebalance:
                    forced_rebalancing_dates.append(date)
                if scheduled_rebalance:
                    rebalancing_dates.append(date)
                
                # Reset to target weights
                current_weights = filtered_weights.copy()
            
            # Record daily values
            portfolio_values.append(portfolio_value)
            portfolio_weights_history.append(current_weights.copy())
        
        # Calculate summary statistics
        portfolio_series = pd.Series(portfolio_values, index=returns_data.index)
        portfolio_returns = portfolio_series.pct_change().dropna()
        
        # Calculate time period in years for proper annualization
        start_date = returns_data.index[0]
        end_date = returns_data.index[-1]
        time_period_years = (end_date - start_date).days / 365.25
        
        # Total transaction costs
        total_transaction_cost = sum([tc['cost'] for tc in transaction_costs])
        
        # Calculate annualized return correctly
        total_return = portfolio_value - 1.0
        if time_period_years > 0:
            annualized_return = (portfolio_value ** (1 / time_period_years)) - 1.0
        else:
            annualized_return = total_return
        
        results = {
            'portfolio_values': portfolio_series,
            'portfolio_returns': portfolio_returns,
            'portfolio_weights_history': portfolio_weights_history,
            'transaction_costs': transaction_costs,
            'total_transaction_cost': total_transaction_cost,
            'rebalancing_dates': rebalancing_dates,
            'forced_rebalancing_dates': forced_rebalancing_dates,
            'final_value': portfolio_value,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'time_period_years': time_period_years,
            'volatility': portfolio_returns.std() * np.sqrt(252),
            'max_drawdown': self._calculate_max_drawdown(portfolio_series),
            'sharpe_ratio': ((portfolio_returns.mean() - 0.02 / 252) * 252) / (portfolio_returns.std() * np.sqrt(252)) if portfolio_returns.std() > 0 else 0
        }
        
        return results
    
    def _get_rebalancing_dates(self, 
                             date_index: pd.DatetimeIndex, 
                             frequency: str) -> List[pd.Timestamp]:
        """Get scheduled rebalancing dates based on frequency"""
        if frequency == 'none':
            return []
        elif frequency == 'monthly':
            return [date for date in date_index if date.is_month_end]
        elif frequency == 'quarterly':
            return [date for date in date_index 
                   if date.month in [3, 6, 9, 12] and date.is_month_end]
        elif frequency == 'annual':
            return [date for date in date_index 
                   if date.month == 12 and date.is_month_end]
        else:
            return []
    
    def _check_constraint_violations(self, 
                                   current_weights: Dict[str, float],
                                   neutral_weights: Dict[str, float],
                                   permitted_deviation: float) -> bool:
        """Check if any asset weight violates permitted deviation constraints"""
        for asset, current_weight in current_weights.items():
            if asset in neutral_weights:
                neutral_weight = neutral_weights[asset]
                max_allowed = min(1.0, neutral_weight + permitted_deviation)
                min_allowed = max(0.0, neutral_weight - permitted_deviation)
                
                if current_weight > max_allowed or current_weight < min_allowed:
                    return True
        return False
    
    def _calculate_transaction_cost(self, 
                                  current_weights: Dict[str, float],
                                  target_weights: Dict[str, float],
                                  cost_bps: int) -> float:
        """
        Calculate transaction cost based on weight changes
        
        Corrected formula: transaction_cost = (total_weight_rebalanced * cost_bps / 10000) * 2
        Logic: Pay transaction cost on both sides (buy and sell) of each rebalancing trade
        """
        total_weight_change = 0.0
        
        for asset in current_weights:
            current = current_weights.get(asset, 0)
            target = target_weights.get(asset, 0)
            total_weight_change += abs(target - current)
        
        # Corrected transaction cost calculation from specs
        transaction_cost = (total_weight_change * cost_bps / 10000) * 2
        
        return transaction_cost
    
    def _calculate_max_drawdown(self, portfolio_series: pd.Series) -> float:
        """Calculate maximum drawdown"""
        if portfolio_series.empty:
            return 0.0
        
        peak = portfolio_series.cummax()
        drawdown = (portfolio_series - peak) / peak
        max_drawdown = drawdown.min()
        
        return abs(max_drawdown)
    
    def simulate_scenarios_parallel(self, 
                                   scenarios: List[SimulationConfig],
                                   parallel_config: ParallelConfig = None,
                                   max_workers: int = None,
                                   use_parallel: bool = True,
                                   chunk_size: int = None) -> List[Dict[str, Any]]:
        """
        Simulate scenarios using parallel processing
        
        Args:
            scenarios: List of scenario configurations to simulate
            parallel_config: Configuration for parallel processing (None = defaults)
            max_workers: Maximum number of worker processes (deprecated, use parallel_config)
            use_parallel: If False, fall back to sequential processing (deprecated, use parallel_config)
            chunk_size: Batch size for each worker (deprecated, use parallel_config)
        
        Returns:
            List of simulation results in the same order as input scenarios
        """
        if not scenarios:
            print("⚠️ No scenarios provided for simulation")
            return []
        
        # Handle backward compatibility and create default config
        if parallel_config is None:
            parallel_config = ParallelConfig(
                use_parallel=use_parallel,
                max_workers=max_workers,
                chunk_size=chunk_size
            )
        
        # Determine optimal worker count
        if parallel_config.max_workers is None:
            parallel_config.max_workers = max(1, cpu_count() - 1)  # Leave one core free
        
        # For small batches, don't use parallel processing
        if len(scenarios) < parallel_config.max_workers or not parallel_config.use_parallel:
            if parallel_config.progress_reporting:
                print(f"🔄 Running {len(scenarios)} scenarios sequentially...")
            return self.run_scenario_batch(scenarios)
        
        # Calculate optimal chunk size
        if parallel_config.chunk_size is None:
            parallel_config.chunk_size = max(1, len(scenarios) // (parallel_config.max_workers * 2))
        
        if parallel_config.progress_reporting:
            print(f"🚀 Running {len(scenarios)} scenarios in parallel...")
            print(f"   Workers: {parallel_config.max_workers}")
            print(f"   Chunk size: {parallel_config.chunk_size}")
        
        try:
            # Create a wrapper function that can be pickled for multiprocessing
            simulation_func = _create_simulation_wrapper()
            
            # Prepare scenario data for multiprocessing
            scenario_data = []
            for scenario in scenarios:
                scenario_data.append({
                    'scenario': scenario,
                    'data_dir': self.data_mapper.data_dir,
                    'specs_dir': self.data_mapper.specs_dir
                })
            
            # Run parallel simulation with progress tracking
            with Pool(processes=parallel_config.max_workers) as pool:
                if parallel_config.progress_reporting and len(scenarios) > 10:
                    # Use imap for progress tracking on larger batches
                    results = []
                    total_scenarios = len(scenario_data)
                    
                    print(f"      Progress: 0/{total_scenarios} (0.0%)", end="", flush=True)
                    
                    for i, result in enumerate(pool.imap(simulation_func, scenario_data, chunksize=parallel_config.chunk_size)):
                        results.append(result)
                        
                        # Update progress at chunk intervals or key milestones
                        if (i + 1) % max(1, parallel_config.chunk_size) == 0 or (i + 1) % max(1, total_scenarios // 10) == 0 or (i + 1) == total_scenarios:
                            progress_pct = ((i + 1) / total_scenarios) * 100
                            print(f"\r      Progress: {i + 1}/{total_scenarios} ({progress_pct:.1f}%)", end="", flush=True)
                    
                    print()  # New line after progress complete
                else:
                    # Simple mapping for small batches
                    results = pool.map(simulation_func, scenario_data, chunksize=parallel_config.chunk_size)
            
            # Count successful simulations
            successful = sum(1 for r in results if r['success'])
            failed = len(results) - successful
            
            if parallel_config.progress_reporting:
                print(f"✅ Parallel simulation completed:")
                print(f"   Successful: {successful}")
                print(f"   Failed: {failed}")
            
            return results
            
        except Exception as e:
            if parallel_config.progress_reporting:
                print(f"❌ Parallel processing failed: {str(e)}")
            
            if parallel_config.fallback_on_error:
                if parallel_config.progress_reporting:
                    print(f"🔄 Falling back to sequential processing...")
                return self.run_scenario_batch(scenarios)
            else:
                raise e
    
    def run_scenario_batch(self, scenarios: List[SimulationConfig]) -> List[Dict[str, Any]]:
        """Run a batch of scenarios"""
        results = []
        
        for i, config in enumerate(scenarios):
            if i % 50 == 0:  # Progress reporting
                print(f"Processing scenario {i+1}/{len(scenarios)}: {config.scenario_id}")
            
            result = self.simulate_php_performance(config)
            results.append(result)
        
        return results
    
    def print_simulation_summary(self, result: Dict[str, Any]):
        """Print summary of a single simulation result"""
        if not result['success']:
            print(f"❌ {result['config'].scenario_id}: {result['error']}")
            return
        
        config = result['config']
        sim_results = result['simulation_results']
        
        print(f"✅ {config.scenario_id}")
        print(f"   Total Return: {sim_results['total_return']:.1%}")
        print(f"   Annualized Return: {sim_results['annualized_return']:.1%}")
        print(f"   Volatility: {sim_results['volatility']:.1%}")
        print(f"   Max Drawdown: {sim_results['max_drawdown']:.1%}")
        print(f"   Sharpe Ratio: {sim_results['sharpe_ratio']:.2f}")
        print(f"   Transaction Cost: {sim_results['total_transaction_cost']:.2%}")
        print(f"   Rebalances: {len(sim_results['rebalancing_dates'])}")
        print(f"   Forced Rebalances: {len(sim_results['forced_rebalancing_dates'])}")

if __name__ == "__main__":
    # Protect against multiprocessing issues on Windows/macOS
    mp.set_start_method('fork', force=True) if sys.platform != 'win32' else None
    
    # Test the portfolio simulation
    sim_engine = PortfolioSimulation()
    
    # Generate a small sample of scenarios for testing
    test_scenarios = sim_engine.generate_all_php_scenarios(
        "EQFI", "2010-01-01", "2020-01-01"
    )
    
    print(f"Generated {len(test_scenarios)} test scenarios")
    
    if test_scenarios:
        print("First scenario:")
        print(f"  {test_scenarios[0].scenario_id}")
        
        # Test single scenario simulation
        print(f"\n=== TESTING SINGLE SCENARIO ===")
        result = sim_engine.simulate_php_performance(test_scenarios[0])
        sim_engine.print_simulation_summary(result)
    else:
        print("No scenarios generated - check date range")