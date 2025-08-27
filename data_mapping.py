import pandas as pd
import os
from typing import Dict, List, Optional
from pathlib import Path
import tempfile

class DataMapper:
    """Module for mapping data files to indices and loading data"""
    
    def __init__(self, data_dir: str = "data", specs_dir: str = "specs"):
        self.data_dir = data_dir
        self.specs_dir = specs_dir
        self.index_map = self._load_index_mapping()
        self.neutral_weights = self._load_neutral_weights()
        self.data_cache = {}
        self._forward_fill_messages_shown = set()  # Track which indices have shown forward-fill messages
        self._global_message_dir = Path(tempfile.gettempdir()) / "php_messages"  # Global message tracking
    
    def _load_index_mapping(self) -> Dict[str, str]:
        """Load index mapping from specs/index_map.csv"""
        mapping_path = os.path.join(self.specs_dir, "index_map.csv")
        df = pd.read_csv(mapping_path)
        return dict(zip(df['File'], df['Index']))
    
    def _load_neutral_weights(self) -> pd.DataFrame:
        """Load neutral weights from specs/neutral_weight.csv"""
        weights_path = os.path.join(self.specs_dir, "neutral_weight.csv")
        # Skip first row and use second row as header
        df = pd.read_csv(weights_path, skiprows=1)
        # Clean up the dataframe - handle the fact that Index column has empty rows
        df = df[df['Index'].notna()]
        df = df[df['Index'] != '']
        return df
    
    def get_available_indices(self) -> List[str]:
        """Get list of all available indices"""
        return list(self.index_map.values())
    
    def get_required_indices(self, mandate: str = None) -> List[str]:
        """Get required indices for a specific mandate or all mandates"""
        if mandate:
            if mandate not in ['6040', 'EQFI', 'EQFILA', 'EQFILAIA']:
                raise ValueError(f"Unknown mandate: {mandate}")
            
            # Filter for non-zero weights in the specified mandate
            weights = self.neutral_weights[self.neutral_weights[mandate].notna()]
            weights = weights[weights[mandate] != '']
            weights[mandate] = weights[mandate].astype(str).str.rstrip('%')
            weights = weights[weights[mandate].astype(float) > 0]
            return weights['Index'].tolist()
        else:
            # Return all indices used in any mandate
            all_indices = set()
            for mandate in ['6040', 'EQFI', 'EQFILA', 'EQFILAIA']:
                all_indices.update(self.get_required_indices(mandate))
            return list(all_indices)
    
    def validate_data_availability(self) -> Dict[str, bool]:
        """Check if all required indices have corresponding data files"""
        required_indices = self.get_required_indices()
        available_indices = self.get_available_indices()
        
        validation = {}
        for index in required_indices:
            validation[index] = index in available_indices
        
        return validation
    
    def get_data_file_for_index(self, index_name: str) -> Optional[str]:
        """Get data file name for a specific index"""
        for file, index in self.index_map.items():
            if index == index_name:
                return file
        return None
    
    def load_index_data(self, index_name: str, start_date: str = None, end_date: str = None, 
                       ensure_daily: bool = False) -> pd.DataFrame:
        """
        Load data for a specific index with optional daily frequency enforcement
        
        Args:
            index_name: Name of the index to load
            start_date: Start date filter
            end_date: End date filter  
            ensure_daily: If True, forward-fill monthly data to daily frequency
        """
        if index_name in self.data_cache:
            df = self.data_cache[index_name].copy()
        else:
            file_name = self.get_data_file_for_index(index_name)
            if not file_name:
                raise ValueError(f"No data file found for index: {index_name}")
            
            file_path = os.path.join(self.data_dir, file_name)
            df = pd.read_csv(file_path)
            
            # Standardize date column with explicit format handling
            # All data files use M/D/YY or MM/DD/YY format (2-digit years)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                # Use explicit format that handles both M/D/YY and MM/DD/YY
                df['DATE'] = pd.to_datetime(df['DATE'], format='%m/%d/%y')
            
            df = df.set_index('DATE')
            df = df.sort_index()
            
            # Handle Credit Suisse Hedge Fund Index monthly data
            if index_name == "Credit Suisse Hedge Fund Index" and ensure_daily:
                # Forward-fill monthly data to daily frequency
                if not df.empty:
                    # Create daily date range from first to last date
                    full_range = pd.date_range(start=df.index[0], end=df.index[-1], freq='D')
                    
                    # Reindex to daily frequency and forward-fill values
                    df = df.reindex(full_range)
                    df['LEVEL'] = df['LEVEL'].ffill()
                    
                    # Only show the message once globally across all processes
                    if self._should_show_forward_fill_message(index_name):
                        print(f"✅ Forward-filled {index_name} to daily frequency")
                        self._mark_forward_fill_message_shown(index_name)
            
            # Cache the processed data
            self.data_cache[index_name] = df.copy()
        
        # Filter by date range if specified
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        
        return df
    
    def load_all_required_data(self, mandate: str, start_date: str = None, end_date: str = None, 
                              ensure_daily: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Load all required data for a specific mandate
        
        Args:
            mandate: Portfolio mandate
            start_date: Start date filter
            end_date: End date filter
            ensure_daily: If True, forward-fill monthly data to daily frequency
        """
        required_indices = self.get_required_indices(mandate)
        data = {}
        
        for index in required_indices:
            try:
                data[index] = self.load_index_data(index, start_date, end_date, ensure_daily)
            except ValueError as e:
                print(f"Warning: {e}")
        
        return data
    
    def get_mandate_weights(self, mandate: str) -> Dict[str, float]:
        """Get neutral weights for a specific mandate"""
        if mandate not in ['6040', 'EQFI', 'EQFILA', 'EQFILAIA']:
            raise ValueError(f"Unknown mandate: {mandate}")
        
        weights = {}
        mandate_data = self.neutral_weights[self.neutral_weights[mandate].notna()]
        mandate_data = mandate_data[mandate_data[mandate] != '']
        
        for _, row in mandate_data.iterrows():
            weight_str = str(row[mandate]).rstrip('%')
            if weight_str and weight_str != 'nan':
                weights[row['Index']] = float(weight_str) / 100
        
        return weights
    
    def print_data_summary(self):
        """Print summary of available data and validation results"""
        print("=== DATA MAPPING SUMMARY ===")
        print(f"Available data files: {len(self.index_map)}")
        print(f"Available indices: {len(self.get_available_indices())}")
        
        print("\n=== INDEX MAPPING ===")
        for file, index in self.index_map.items():
            print(f"{file:<15} -> {index}")
        
        print("\n=== VALIDATION RESULTS ===")
        validation = self.validate_data_availability()
        for index, available in validation.items():
            status = "✓" if available else "✗"
            print(f"{status} {index}")
        
        print("\n=== MANDATE REQUIREMENTS ===")
        for mandate in ['6040', 'EQFI', 'EQFILA', 'EQFILAIA']:
            indices = self.get_required_indices(mandate)
            weights = self.get_mandate_weights(mandate)
            print(f"\n{mandate}:")
            for index in indices:
                weight = weights.get(index, 0) * 100
                print(f"  {index:<35}: {weight:>6.1f}%")
    
    def _should_show_forward_fill_message(self, index_name: str) -> bool:
        """Check if we should show the forward-fill message for this index"""
        try:
            # Create message directory if it doesn't exist
            self._global_message_dir.mkdir(exist_ok=True)
            
            # Check if message file exists
            message_file = self._global_message_dir / f"forward_fill_{index_name.replace(' ', '_')}.flag"
            return not message_file.exists()
        except:
            # Fall back to local tracking if file system approach fails
            return index_name not in self._forward_fill_messages_shown
    
    def _mark_forward_fill_message_shown(self, index_name: str):
        """Mark that we've shown the forward-fill message for this index"""
        try:
            # Create flag file
            self._global_message_dir.mkdir(exist_ok=True)
            message_file = self._global_message_dir / f"forward_fill_{index_name.replace(' ', '_')}.flag"
            message_file.touch()
        except:
            # Fall back to local tracking if file system approach fails
            self._forward_fill_messages_shown.add(index_name)
    
    def clear_forward_fill_flags(self):
        """Clear all forward-fill message flags for a new analysis session"""
        try:
            if self._global_message_dir.exists():
                for flag_file in self._global_message_dir.glob("forward_fill_*.flag"):
                    flag_file.unlink(missing_ok=True)
        except:
            pass
        self._forward_fill_messages_shown.clear()

if __name__ == "__main__":
    mapper = DataMapper()
    mapper.print_data_summary()