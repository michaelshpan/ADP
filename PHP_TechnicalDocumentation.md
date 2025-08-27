# Perfect Hindsight Portfolio (PHP) Analysis System
## Comprehensive Technical Documentation Report

## 1. SYSTEM OVERVIEW

### **What the System Does**
The Perfect Hindsight Portfolio (PHP) analysis system is a portfolio simulation platform that evaluates theoretical optimal portfolio allocation strategies using perfect hindsight of asset class performance against a static 60/40 benchmark. It simulates "perfect" investment decisions by allocating larger weights to asset classes that historically performed well, subject to practical constraints.

### **Core Capabilities**
- **Multi-Mandate Portfolio Analysis**: Supports 4 investment strategies (6040, EQFI, EQFILA, EQFILAIA)
- **Perfect Weight Calculation**: Determines optimal allocations based on realized performance
- **Comprehensive Simulation Engine**: Runs thousands of scenarios with different parameters
- **Advanced Performance Analytics**: Risk-adjusted returns, drawdowns, Sharpe ratios, benchmarks
- **Multi-Format Reporting**: Executive summaries, technical reports, visualization, interactive dashboards
- **High-Performance Computing**: Parallel processing with optimization techniques

## 2. ARCHITECTURE & STRUCTURE

### **System Architecture**
```
┌─────────────────────────────────────────┐
│         Master Orchestrator             │
│      (Central Coordination)             │
├─────────────────────────────────────────┤
│  Reporting  │  Dashboard  │ Analytics   │
├─────────────────────────────────────────┤
│      Performance Analytics Engine       │
├─────────────────────────────────────────┤
│ Portfolio │ Perfect Weight │ Portfolio  │
│Simulation │  Calculator    │   Config   │
├─────────────────────────────────────────┤
│       Data Mapping & Management         │
├─────────────────────────────────────────┤
│      Data Sources & Specifications      │
└─────────────────────────────────────────┘
```

### **Data Flow**
1. **Raw Data** → Data Mapper → Clean time series
2. **Mandate Specs** → Portfolio Config → Weight constraints
3. **Parameters** → Perfect Weight Calculator → Investment scenarios
4. **Scenarios** → Portfolio Simulation → Performance results
5. **Results** → Analytics Engine → Risk-adjusted metrics
6. **Analytics** → Reporting Engines → Charts and summaries

## 3. KEY MODULES ANALYSIS

### **Master Orchestrator** (`master_orchestrator.py` - 1,787 lines)
- **Purpose**: Central command and control for analysis pipelines
- **Key Methods**: `run_complete_php_analysis()`, `vs_bm()`, mandate comparisons
- **Features**: End-to-end workflow management, benchmark optimization

### **Portfolio Simulation** (`portfolio_simulation.py` - 614 lines)  
- **Purpose**: Core simulation engine with constraints
- **Key Features**: Daily rebalancing, transaction costs, parallel execution
- **Performance**: High-performance parallel processing with progress tracking

### **Perfect Weight Calculator** (`perfect_weight_calculator.py` - 319 lines)
- **Purpose**: Core PHP logic for optimal weight calculation based on asset class performance for full investment horizon
- **Algorithm**: Constraint-based allocation with performance-driven weighting
- **Output**: Perfect weights respecting absolute deviation limits

### **Performance Analytics** (`performance_analytics.py` - 1,004 lines)
- **Purpose**: Comprehensive risk-adjusted performance measurement
- **Metrics**: Sharpe ratio, Information ratio, Maximum drawdown, Calmar ratio
- **Analysis**: Parameter sensitivity, statistical distributions

### **Data Mapping** (`data_mapping.py` - 234 lines)
- **Purpose**: Data ingestion, validation, and transformation
- **Sources**: 11 CSV files with historical asset class returns
- **Features**: Missing data handling, frequency conversion, validation

## 4. DATA STRUCTURES & MODELS

### **Core Configuration**
```python
@dataclass
class SimulationConfig:
    mandate: str                    # EQFI, EQFILA, EQFILAIA, 6040
    start_date: str                # Investment period start
    end_date: str                  # Investment period end
    investment_horizon_years: int   # 5 or 10 years
    permitted_deviation: float      # ±5%, ±10%, ±20%, ±50%
    rebalancing_frequency: str      # monthly, quarterly, annual, no rebal
    transaction_cost_bps: int       # 5, 25, 50, 100 basis points (single way)
```

### **Result Formats**
- **JSON**: Structured analysis data with nested metrics
- **Excel**: Multi-sheet workbooks with comparison tables  
- **HTML**: Interactive Plotly dashboards
- **PNG**: High-resolution charts and visualizations

## 5. DEPENDENCIES & REQUIREMENTS

### **External Libraries**
- **pandas/numpy**: Data manipulation and numerical computing
- **matplotlib/plotly**: Static and interactive visualizations
- **scipy**: Statistical analysis and distributions
- **multiprocessing**: Parallel scenario execution
- **jinja2**: Report template rendering

### **Data Requirements**
- **11 CSV files**: Historical asset class returns (1996-2025)
- **Index mapping**: Maps data files to asset class names
- **Neutral weights**: Portfolio mandate specifications
- **System**: Python 3.8+, 8GB+ RAM, multi-core CPU recommended

## 6. KEY ALGORITHMS & CALCULATIONS

### **Perfect Weight Algorithm**
1. Sort assets by historical performance (best to worst)
2. Set underperforming assets to minimum allowed weight — if negative performance, allocate zero weight
3. Allocate maximum weight to best performers first
4. Then, allocate remaining weight to next best performers based on relative performance
5. Respect absolute deviation constraints (±5% to ±50%)
6. Normalize weights to sum to 100%

### **Performance Metrics**
- **Sharpe Ratio**: `(Return - Risk Free Rate) / Volatility` (Risk free rate = 2%)
- **Maximum Drawdown**: Peak-to-trough loss calculation
- **Information Ratio**: `Excess Return / Tracking Error`
- **Transaction Costs**: Double-sided costs on rebalancing

## 7. WORKFLOW & EXECUTION

### **Typical Execution Flow**
1. Load and validate data sources
2. Generate all possible scenario combinations based on start and end date of analysis and parameters
3. Calculate perfect weights for each scenario
4. Execute parallel portfolio simulations
5. Compute comprehensive analytics
6. Generate multi-format reports and visualizations

### **Performance Optimizations**
- **Parallel Processing**: Auto-scaling workers with progress tracking
- **Benchmark Optimization**: Smart caching eliminates redundant calculations
- **Memory Management**: Streaming results for large datasets
- **Scenario Deduplication**: Reduces computational complexity

## 8. OUTPUT & REPORTING

### **Report Types**
- **Executive Summaries**: High-level performance metrics and insights
- **Technical Reports**: Detailed statistical analysis with charts
- **Benchmark Comparisons**: Mandate vs 6040 portfolio analysis
- **Interactive Dashboards**: Web-based exploration tools

### **File Organization**
```
reports/
├── {mandate}_complete_analysis_{timestamp}/
├── vs_bm/{mandate}_vs_{benchmark}_{timestamp}/
├── charts/ (Visualizations)
├── executive/ (Leadership summaries)
└── data/ (JSON/Excel exports)
```

## 9. NOTABLE FEATURES 

### **Technical Specifications**
- **Absolute Deviation Constraints**: More intuitive than relative constraints
- **Multi-Dimensional Optimization**: 128+ parameter combinations per analysis
- **Forward-Fill Processing**: Automatic daily frequency conversion for Hedge Fund index with monthly data
- **Executive Report Generation**: Professional multi-format presentations

### **Advanced Capabilities**
- **Cross-Mandate Comparison**: Systematic strategy analysis
- **Real-Time Progress Tracking**: User feedback during long-running analyses
- **Intelligent Error Handling**: Graceful degradation and fallback mechanisms
- **Comprehensive Risk Analytics**: 15+ risk-adjusted performance metrics

## 10. EXTENSIBILITY & CUSTOMIZATION

### **Adding New Features**
- **New Mandates**: Add specifications in `neutral_weight.csv`
- **Custom Metrics**: Extend `PerformanceAnalytics` class
- **Additional Visualizations**: New chart types in reporting modules
- **Alternative Constraints**: Modify perfect weight calculation logic

### **Configuration Options**
- **Sample Size Limiting**: For testing and development
- **Parallel Processing**: Configurable workers and memory limits
- **Output Formats**: Flexible report generation combinations
- **Analysis Periods**: Custom date ranges and investment horizons

---

## **QUICK START GUIDE**

### **Basic Usage**
```python
# Initialize the system
from master_orchestrator import PHPMasterOrchestrator
orchestrator = PHPMasterOrchestrator()

# Run complete analysis
results = orchestrator.run_complete_php_analysis(
    mandate="EQFILA",
    analysis_start="2010-01-01",
    analysis_end="2020-01-01"
)

# Run benchmark comparison
comparison = orchestrator.vs_bm(
    mandate="EQFI",
    analysis_start="2015-01-01",
    analysis_end="2020-01-01",
    output_mode="both"
)
```

### **Command Line Interface**
```bash
# Complete PHP analysis
python master_orchestrator.py --mandate EQFILA --start 2010-01-01 --end 2020-01-01

# Benchmark comparison
python master_orchestrator.py --vs-bm EQFI --sample-size 100 --output-mode both

# Mandate comparison
python master_orchestrator.py --compare-mandates --start 2015-01-01 --end 2020-01-01
```

### **Key Configuration Files**
- `specs/neutral_weight.csv` - Portfolio mandate definitions
- `specs/index_map.csv` - Asset class mappings
- `data/data_*.csv` - Historical return data (11 files)

### **Output Locations**
- `reports/` - All analysis results and visualizations
- `reports/vs_bm/` - Benchmark comparison results
- `reports/charts/` - Visualization files
- `reports/executive/` - Leadership summaries

---
  
*Documentation Date: August 27, 2025*