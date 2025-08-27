import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
from data_mapping import DataMapper

class PortfolioConfig:
    """Module for portfolio configuration and benchmark calculations"""
    
    def __init__(self, data_mapper: DataMapper = None):
        self.data_mapper = data_mapper or DataMapper()
        self.mandates = self._initialize_mandates()
        
    def _initialize_mandates(self) -> Dict[str, Dict]:
        """Initialize mandate configurations"""
        mandates = {}
        
        for mandate_name in ['6040', 'EQFI', 'EQFILA', 'EQFILAIA']:
            weights = self.data_mapper.get_mandate_weights(mandate_name)
            required_indices = self.data_mapper.get_required_indices(mandate_name)
            
            mandates[mandate_name] = {
                'name': mandate_name,
                'weights': weights,
                'indices': required_indices,
                'description': self._get_mandate_description(mandate_name)
            }
            
        return mandates
    
    def _get_mandate_description(self, mandate: str) -> str:
        """Get description for each mandate"""
        descriptions = {
            '6040': 'Benchmark - 60% Equity, 40% Fixed Income',
            'EQFI': 'Equity + Fixed Income',
            'EQFILA': 'Equity + Fixed Income + Liquid Alternatives',
            'EQFILAIA': 'Equity + Fixed Income + Liquid + Illiquid Alternatives'
        }
        return descriptions.get(mandate, 'Unknown mandate')
    
    def get_mandate_info(self, mandate: str) -> Dict:
        """Get complete information about a mandate"""
        if mandate not in self.mandates:
            raise ValueError(f"Unknown mandate: {mandate}")
        return self.mandates[mandate]
    
    def get_mandate_weights(self, mandate: str) -> Dict[str, float]:
        """Get weights for a specific mandate"""
        if mandate not in self.mandates:
            raise ValueError(f"Unknown mandate: {mandate}")
        return self.mandates[mandate]['weights']
    
    def validate_weights(self, mandate: str) -> bool:
        """Validate that weights sum to 1.0"""
        weights = self.mandates[mandate]['weights']
        total = sum(weights.values())
        return abs(total - 1.0) < 1e-6
    
    def calculate_portfolio_return(self, 
                                 mandate: str,
                                 start_date: str,
                                 end_date: str,
                                 weights: Dict[str, float] = None,
                                 rebalancing_freq: str = 'daily') -> pd.Series:
        """
        Calculate portfolio returns for a given mandate and time period
        
        Args:
            mandate: Portfolio mandate (6040, EQFI, EQFILA, EQFILAIA)
            start_date: Start date for calculation
            end_date: End date for calculation
            weights: Custom weights (if None, uses mandate's neutral weights)
            rebalancing_freq: 'daily', 'monthly', 'quarterly', 'annual', 'none'
        
        Returns:
            Portfolio return time series
        """
        if weights is None:
            weights = self.mandates[mandate]['weights']
        
        # Load all required data with daily frequency enforcement
        data_dict = self.data_mapper.load_all_required_data(mandate, start_date, end_date, ensure_daily=True)
        
        # Align all data to common dates
        returns_data = self._align_and_calculate_returns(data_dict)
        
        if returns_data.empty:
            raise ValueError(f"No overlapping data found for mandate {mandate}")
        
        # Calculate portfolio returns based on rebalancing frequency
        portfolio_returns = self._calculate_rebalanced_returns(
            returns_data, weights, rebalancing_freq
        )
        
        return portfolio_returns
    
    def _align_and_calculate_returns(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Align data from different indices and calculate returns"""
        returns_dict = {}
        
        for index_name, data in data_dict.items():
            # Calculate daily returns
            daily_returns = data['LEVEL'].pct_change()
            returns_dict[index_name] = daily_returns
        
        # Combine all returns into single DataFrame
        returns_df = pd.DataFrame(returns_dict)
        
        # Drop rows with any missing data
        returns_df = returns_df.dropna()
        
        return returns_df
    
    def _calculate_rebalanced_returns(self, 
                                    returns_data: pd.DataFrame,
                                    weights: Dict[str, float],
                                    rebalancing_freq: str) -> pd.Series:
        """Calculate portfolio returns with specified rebalancing frequency"""
        
        # Initialize portfolio with target weights
        portfolio_weights = pd.DataFrame(index=returns_data.index)
        for asset, weight in weights.items():
            if asset in returns_data.columns:
                portfolio_weights[asset] = weight
            else:
                warnings.warn(f"Asset {asset} not found in returns data")
        
        if portfolio_weights.empty:
            raise ValueError("No matching assets found in returns data")
        
        # Calculate returns based on rebalancing frequency
        if rebalancing_freq == 'daily':
            return self._daily_rebalanced_returns(returns_data, portfolio_weights)
        elif rebalancing_freq == 'none':
            return self._buy_and_hold_returns(returns_data, portfolio_weights)
        else:
            return self._periodic_rebalanced_returns(
                returns_data, portfolio_weights, rebalancing_freq
            )
    
    def _daily_rebalanced_returns(self, 
                                returns_data: pd.DataFrame,
                                weights_df: pd.DataFrame) -> pd.Series:
        """Calculate returns with daily rebalancing (constant weights)"""
        portfolio_returns = []
        
        for date in returns_data.index:
            daily_return = 0.0
            for asset in weights_df.columns:
                if asset in returns_data.columns:
                    weight = weights_df.loc[date, asset]
                    asset_return = returns_data.loc[date, asset]
                    daily_return += weight * asset_return
            portfolio_returns.append(daily_return)
        
        return pd.Series(portfolio_returns, index=returns_data.index)
    
    def _buy_and_hold_returns(self, 
                            returns_data: pd.DataFrame,
                            initial_weights: pd.DataFrame) -> pd.Series:
        """Calculate returns with no rebalancing (buy and hold)"""
        portfolio_values = []
        current_weights = initial_weights.iloc[0].copy()
        
        # Initialize portfolio value
        portfolio_value = 1.0
        portfolio_values.append(portfolio_value)
        
        for i, date in enumerate(returns_data.index[1:], 1):
            # Calculate new asset values
            asset_returns = {}
            for asset in current_weights.index:
                if asset in returns_data.columns:
                    asset_returns[asset] = 1 + returns_data.iloc[i][asset]
            
            # Update portfolio value
            portfolio_return = sum(current_weights[asset] * asset_returns.get(asset, 1.0) 
                                 for asset in current_weights.index)
            portfolio_value *= portfolio_return
            portfolio_values.append(portfolio_value)
            
            # Update weights (drift with performance)
            total_value = sum(current_weights[asset] * asset_returns.get(asset, 1.0) 
                            for asset in current_weights.index)
            for asset in current_weights.index:
                current_weights[asset] = (current_weights[asset] * 
                                        asset_returns.get(asset, 1.0)) / total_value
        
        # Convert values to returns
        portfolio_values = pd.Series(portfolio_values, index=returns_data.index)
        return portfolio_values.pct_change().dropna()
    
    def _periodic_rebalanced_returns(self, 
                                   returns_data: pd.DataFrame,
                                   target_weights: pd.DataFrame,
                                   rebalancing_freq: str) -> pd.Series:
        """Calculate returns with periodic rebalancing"""
        rebalance_dates = self._get_rebalancing_dates(returns_data.index, rebalancing_freq)
        
        portfolio_values = []
        current_weights = target_weights.iloc[0].copy()
        portfolio_value = 1.0
        portfolio_values.append(portfolio_value)
        
        for i, date in enumerate(returns_data.index[1:], 1):
            # Check if rebalancing date
            if date in rebalance_dates:
                current_weights = target_weights.iloc[i].copy()
            
            # Calculate portfolio return
            portfolio_return = 0.0
            asset_returns = {}
            
            for asset in current_weights.index:
                if asset in returns_data.columns:
                    asset_return = returns_data.iloc[i][asset]
                    asset_returns[asset] = 1 + asset_return
                    portfolio_return += current_weights[asset] * asset_return
            
            portfolio_value *= (1 + portfolio_return)
            portfolio_values.append(portfolio_value)
            
            # Update weights if not rebalancing (let them drift)
            if date not in rebalance_dates:
                total_value = sum(current_weights[asset] * asset_returns.get(asset, 1.0) 
                                for asset in current_weights.index)
                for asset in current_weights.index:
                    if asset in asset_returns:
                        current_weights[asset] = (current_weights[asset] * 
                                                asset_returns[asset]) / total_value
        
        portfolio_values = pd.Series(portfolio_values, index=returns_data.index)
        return portfolio_values.pct_change().dropna()
    
    def _get_rebalancing_dates(self, date_index: pd.DatetimeIndex, freq: str) -> List[pd.Timestamp]:
        """Get rebalancing dates based on frequency"""
        if freq == 'daily':
            return date_index.tolist()
        elif freq == 'monthly':
            return [date for date in date_index if date.is_month_end]
        elif freq == 'quarterly':
            return [date for date in date_index if date.month in [3, 6, 9, 12] and date.is_month_end]
        elif freq == 'annual':
            return [date for date in date_index if date.month == 12 and date.is_month_end]
        else:
            return []
    
    def calculate_benchmark_6040(self, start_date: str, end_date: str) -> pd.Series:
        """Calculate 6040 benchmark with daily rebalancing"""
        return self.calculate_portfolio_return('6040', start_date, end_date, rebalancing_freq='daily')
    
    def print_mandate_summary(self):
        """Print summary of all mandates"""
        print("=== PORTFOLIO MANDATE SUMMARY ===\n")
        
        for mandate_name, config in self.mandates.items():
            print(f"{mandate_name}: {config['description']}")
            print(f"Total Weight: {sum(config['weights'].values()):.1%}")
            print("Asset Allocation:")
            
            for asset, weight in config['weights'].items():
                print(f"  {asset:<35}: {weight:>6.1%}")
            
            print(f"Weight Validation: {'✓' if self.validate_weights(mandate_name) else '✗'}")
            print("-" * 60)

if __name__ == "__main__":
    # Test the portfolio configuration
    portfolio_config = PortfolioConfig()
    portfolio_config.print_mandate_summary()
    
    # Test benchmark calculation
    print("\n=== TESTING BENCHMARK CALCULATION ===")
    try:
        benchmark = portfolio_config.calculate_benchmark_6040("2020-01-01", "2020-12-31")
        print(f"6040 Benchmark calculated successfully")
        print(f"Sample period: {benchmark.index[0]} to {benchmark.index[-1]}")
        print(f"Total return: {(1 + benchmark).prod() - 1:.2%}")
        print(f"Annualized return: {benchmark.mean() * 252:.2%}")
        print(f"Annualized volatility: {benchmark.std() * np.sqrt(252):.2%}")
    except Exception as e:
        print(f"Error calculating benchmark: {e}")