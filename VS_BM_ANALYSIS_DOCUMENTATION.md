# 📊 Mandate vs Benchmark (vs_bm) Analysis Documentation

## Overview

The `vs_bm` analysis function provides detailed scenario-by-scenario comparison between any mandate (EQFI, EQFILA, EQFILAIA) and the 6040 benchmark portfolio. This analysis reveals how the mandate performs relative to the traditional 60/40 portfolio across thousands of investment scenarios.

---

## 🎯 Key Features

### **Direct Scenario Matching**
- Each mandate scenario is paired with a matching 6040 benchmark scenario
- **Perfect matching** on start dates, end dates, and investment horizons
- **Fair comparison**: 6040 uses daily rebalancing, no transaction costs, no deviations (pure benchmark)

### **Comprehensive Metrics**
- **Excess Return**: Mandate return - Benchmark return
- **Tracking Error**: Volatility of return differences
- **Max Drawdown Difference**: Relative drawdown comparison
- **Win Rate**: Percentage of scenarios where mandate outperforms

### **Advanced Statistics**
- **Information Ratio**: Excess return per unit of tracking error
- **Alpha**: Risk-adjusted excess return using CAPM
- **Beta**: Systematic risk relative to benchmark
- **Up/Down Capture Ratios**: Performance in different market conditions

---

## 🚀 Usage

### **Python API**

```python
from master_orchestrator import PHPMasterOrchestrator

orchestrator = PHPMasterOrchestrator()

# Basic usage
results = orchestrator.vs_bm(
    mandate="EQFILA",
    analysis_start="2015-01-01",
    analysis_end="2020-01-01"
)

# Advanced usage with options
results = orchestrator.vs_bm(
    mandate="EQFI",
    analysis_start="2010-01-01",
    analysis_end="2020-01-01",
    sample_size=100,                    # Limit scenarios for testing
    output_mode="both",                 # "individual", "aggregated", or "both"
    include_advanced_stats=True         # Include Information Ratio, Alpha, Beta
)
```

### **Command Line Interface**

```bash
# Basic analysis
python3 master_orchestrator.py --vs-bm EQFILA --start 2015-01-01 --end 2020-01-01

# With sampling and specific output
python3 master_orchestrator.py --vs-bm EQFI --sample-size 100 --output-mode aggregated

# Individual comparisons only
python3 master_orchestrator.py --vs-bm EQFILAIA --output-mode individual --skip-advanced-stats
```

---

## 📊 Output Structure

### **Individual Comparison Format**

Each scenario comparison includes:

```python
{
    'scenario_id': 'EQFILA_201001_5Y_5pct_monthly_5bps',
    'mandate_config': {
        'start_date': '2010-01-01',
        'end_date': '2015-01-01', 
        'horizon_years': 5,
        'deviation_limit': 0.05,
        'rebalancing_freq': 'monthly',
        'transaction_cost_bps': 5
    },
    'mandate_performance': {
        'total_return': 0.145,
        'annualized_return': 0.089,
        'volatility': 0.098,
        'max_drawdown': 0.117,
        'sharpe_ratio': 0.85
    },
    'benchmark_performance': {
        'total_return': 0.128,
        'annualized_return': 0.076,
        'volatility': 0.092,
        'max_drawdown': 0.125,
        'sharpe_ratio': 0.79
    },
    'relative_metrics': {
        'excess_return': 0.017,           # 1.7% excess return
        'tracking_error': 0.034,          # 3.4% tracking error
        'drawdown_difference': -0.008,    # 0.8% better max drawdown
        'sharpe_difference': 0.06,
        'winner': 'mandate'
    },
    'advanced_stats': {                   # If include_advanced_stats=True
        'information_ratio': 0.50,
        'alpha': 0.015,
        'beta': 1.02,
        'up_capture_ratio': 1.05,
        'down_capture_ratio': 0.95
    }
}
```

### **Aggregated Analysis Format**

Statistical summary across all scenarios:

```python
{
    'mandate': 'EQFILA',
    'benchmark': '6040',
    'summary': {
        'total_scenarios_compared': 1247,
        'mandate_win_rate': 0.73,         # 73% scenarios outperformed
        'benchmark_win_rate': 0.25,
        'tie_rate': 0.02
    },
    'excess_return_analysis': {
        'mean': 0.012,                    # 1.2% average excess return
        'median': 0.015,
        'std_dev': 0.028,
        'min': -0.045,
        'max': 0.078,
        'percentiles': {
            'p10': -0.018, 'p25': -0.002, 
            'p75': 0.031, 'p90': 0.045
        },
        'positive_excess_rate': 0.68,     # 68% positive excess returns
        'statistically_significant': True
    },
    'tracking_error_analysis': {
        'mean': 0.041,                    # 4.1% average tracking error
        'median': 0.038,
        'std_dev': 0.015
    },
    'relative_drawdown_analysis': {
        'mean_difference': -0.003,        # 0.3% better drawdowns on average  
        'better_drawdown_rate': 0.68      # 68% scenarios had better drawdowns
    },
    'risk_return_efficiency': {
        'information_ratio_portfolio': 0.29,
        'avg_sharpe_difference': 0.08
    },
    'parameter_sensitivity': {           # Which parameters drive outperformance
        'deviation_limit': {
            '0.05': {'avg_excess_return': 0.018, 'win_rate': 0.82},
            '0.1': {'avg_excess_return': 0.015, 'win_rate': 0.75},
            '0.2': {'avg_excess_return': 0.008, 'win_rate': 0.65}
        }
    }
}
```

---

## 🔍 Analysis Methodology

### **Benchmark Portfolio Design**

The 6040 benchmark portfolio uses:
- **Pure 60/40 allocation**: 60% equity, 40% fixed income
- **Daily rebalancing**: No weight drift allowed
- **No transaction costs**: Ideal frictionless trading
- **No deviation limits**: Always maintains exact 60/40 weights

This represents the theoretical performance ceiling of the traditional balanced portfolio strategy.

### **Scenario Matching Logic**

For each mandate scenario with parameters:
- Start Date: 2010-01-01
- End Date: 2015-01-01  
- Deviation: ±10%
- Rebalancing: Monthly
- Transaction Cost: 25bps

A matching 6040 benchmark scenario is created with:
- Start Date: 2010-01-01 ✅ (same)
- End Date: 2015-01-01 ✅ (same)
- Deviation: 0% (pure benchmark weights)
- Rebalancing: Daily (no drift)
- Transaction Cost: 0bps (frictionless)

### **Statistical Calculations**

- **Tracking Error**: `sqrt(variance(mandate_daily_returns - benchmark_daily_returns)) * sqrt(252)`
- **Information Ratio**: `excess_return / tracking_error`
- **Alpha (Jensen's)**: `mandate_return - (risk_free + beta * (benchmark_return - risk_free))`
- **Beta**: `covariance(mandate, benchmark) / variance(benchmark)`

---

## 📈 Interpretation Guide

### **Excess Return Analysis**
- **Positive mean**: Mandate outperforms on average
- **High positive excess rate**: Consistent outperformance
- **Statistical significance**: t-statistic > 1.96 indicates significance

### **Tracking Error Analysis**
- **Low tracking error**: Strategy closely follows benchmark
- **High tracking error**: More deviation, higher active risk

### **Information Ratio**
- **> 0.5**: Good active management
- **> 0.75**: Excellent active management
- **< 0.25**: Poor risk-adjusted performance

### **Parameter Sensitivity**
- **Higher win rates** indicate which parameters consistently outperform
- **Deviation limits**: Often 5-10% optimal for most mandates
- **Transaction costs**: Lower is generally better for active strategies

---

## ⚠️ Limitations & Assumptions

### **Model Limitations**
1. **Simplified Tracking Error**: Based on volatility estimates rather than full daily return correlation
2. **Static Correlation**: Uses 0.85 correlation assumption between mandate and benchmark
3. **Perfect Hindsight**: Both portfolios use perfect foresight (not realistic)

### **Implementation Assumptions**
1. **Daily Rebalancing**: Benchmark assumes frictionless daily rebalancing
2. **No Market Impact**: Transaction costs are linear, no price impact
3. **Data Availability**: All scenarios require complete data coverage

### **Statistical Assumptions**
1. **Normal Returns**: Statistical tests assume normal distribution
2. **Constant Volatility**: Does not account for regime changes
3. **Linear Beta**: CAPM assumptions may not hold across all periods

---

## 🛠️ Technical Implementation

### **Performance Characteristics**
- **Parallel Processing**: Utilizes multiple CPU cores automatically
- **Memory Efficient**: Processes scenarios in batches
- **Scalable**: Handles 1,000+ scenarios efficiently

### **Error Handling**
- **Graceful Degradation**: Continues analysis if some scenarios fail
- **Input Validation**: Validates mandate names and date ranges
- **Fallback Options**: Sequential processing if parallel fails

### **Output Options**
- **JSON Export**: Structured data for further analysis
- **Visualization Ready**: Data formatted for charting
- **Extensible**: Easy to add new metrics or output formats

---

## 📚 Examples & Use Cases

### **Portfolio Strategy Evaluation**
Compare EQFILA's diversification benefits vs traditional 60/40:
```python
results = orchestrator.vs_bm("EQFILA", "2000-01-01", "2020-01-01")
```

### **Parameter Optimization**
Find optimal deviation limits across market cycles:
```python
results = orchestrator.vs_bm("EQFI", output_mode="aggregated")
sensitivity = results['results']['aggregated_analysis']['parameter_sensitivity']
```

### **Risk Assessment**
Evaluate tracking error and drawdown trade-offs:
```python
results = orchestrator.vs_bm("EQFILAIA", include_advanced_stats=True)
risk_analysis = results['results']['aggregated_analysis']['tracking_error_analysis']
```

---

## 🔧 Troubleshooting

### **Common Issues**

**No scenarios generated**
- Check date range (need sufficient period for investment horizons)
- Ensure mandate name is correct (EQFI, EQFILA, EQFILAIA)

**Low success rate**
- Data availability issues for specific periods
- Try broader date range or different sample size

**Performance issues**
- Use sample_size parameter for testing
- Check system resources (parallel processing uses multiple cores)

### **Performance Optimization**

**For Testing**
```python
results = orchestrator.vs_bm(mandate, sample_size=50)  # Limit scenarios
```

**For Production**
```python
results = orchestrator.vs_bm(mandate, sample_size=None)  # All scenarios
```

---

*Generated by PHP Master Orchestrator vs_bm Analysis Module*
*Last Updated: August 27, 2025*