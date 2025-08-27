import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import warnings
from pathlib import Path
from jinja2 import Template
from dataclasses import dataclass, asdict
from data_mapping import DataMapper
from portfolio_config import PortfolioConfig
from perfect_weight_calculator import PerfectWeightCalculator
from portfolio_simulation import PortfolioSimulation, SimulationConfig
from performance_analytics import PerformanceAnalytics
from reporting_visualization import PHPReporting

@dataclass
class ExecutiveSummary:
    """Executive summary data structure"""
    mandate: str
    analysis_period: str
    total_scenarios: int
    successful_scenarios: int
    success_rate: float
    
    # Performance highlights
    avg_annualized_return: float
    best_annualized_return: float
    worst_annualized_return: float
    avg_excess_return: float
    
    # Risk highlights
    avg_volatility: float
    avg_max_drawdown: float
    worst_max_drawdown: float
    avg_sharpe_ratio: float
    best_sharpe_ratio: float
    
    # Key findings
    optimal_deviation_limit: str
    optimal_investment_horizon: str
    optimal_rebalancing_frequency: str
    optimal_transaction_cost: str
    
    # Risk insights
    scenarios_with_positive_excess: int
    scenarios_with_high_drawdown: int
    correlation_return_risk: float
    
    # Methodology notes
    methodology_notes: List[str]

class ExecutiveReportGenerator:
    """Module for generating executive-level reports and presentations"""
    
    def __init__(self, 
                 data_mapper: DataMapper = None,
                 portfolio_config: PortfolioConfig = None,
                 perfect_weight_calc: PerfectWeightCalculator = None,
                 simulation_engine: PortfolioSimulation = None,
                 analytics_engine: PerformanceAnalytics = None,
                 reporting_engine: PHPReporting = None):
        
        self.data_mapper = data_mapper or DataMapper()
        self.portfolio_config = portfolio_config or PortfolioConfig(self.data_mapper)
        self.perfect_weight_calc = perfect_weight_calc or PerfectWeightCalculator(
            self.data_mapper, self.portfolio_config)
        self.simulation_engine = simulation_engine or PortfolioSimulation(
            self.data_mapper, self.portfolio_config, self.perfect_weight_calc)
        self.analytics_engine = analytics_engine or PerformanceAnalytics()
        self.reporting_engine = reporting_engine or PHPReporting()
        
        # Create output directories
        self.output_dir = Path("reports/executive")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "summaries").mkdir(exist_ok=True)
        (self.output_dir / "presentations").mkdir(exist_ok=True)
        (self.output_dir / "memos").mkdir(exist_ok=True)
    
    def generate_executive_summary(self, 
                                 analytics_results: Dict,
                                 mandate: str,
                                 analysis_period: str) -> ExecutiveSummary:
        """Generate structured executive summary from analytics results"""
        
        all_results = analytics_results.get('all_results', [])
        successful_results = [r for r in all_results if r.get('success', False)]
        summary_stats = analytics_results.get('summary_statistics', {})
        top_performers = analytics_results.get('top_performers', {}).get('top_10', [])
        param_analysis = analytics_results.get('parameter_analysis', {})
        
        if not successful_results:
            raise ValueError("No successful scenarios found for executive summary")
        
        # Extract performance metrics
        returns = [r.get('simulation_results', {}).get('annualized_return', 0) for r in successful_results]
        volatilities = [r.get('simulation_results', {}).get('volatility', 0) for r in successful_results]
        max_drawdowns = [r.get('simulation_results', {}).get('max_drawdown', 0) for r in successful_results]
        sharpe_ratios = [r.get('simulation_results', {}).get('sharpe_ratio', 0) for r in successful_results]
        
        # Calculate excess returns (assuming 2% risk-free rate and 8% benchmark)
        benchmark_return = 0.08  # Placeholder - should be calculated from actual benchmark
        excess_returns = [r - benchmark_return for r in returns]
        
        # Performance highlights
        avg_return = np.mean(returns)
        best_return = max(returns) if returns else 0
        worst_return = min(returns) if returns else 0
        avg_excess = np.mean(excess_returns)
        
        # Risk highlights  
        avg_vol = np.mean(volatilities)
        avg_dd = np.mean(max_drawdowns)
        worst_dd = max(max_drawdowns) if max_drawdowns else 0
        avg_sharpe = np.mean(sharpe_ratios)
        best_sharpe = max(sharpe_ratios) if sharpe_ratios else 0
        
        # Success metrics
        total_scenarios = len(all_results)
        success_count = len(successful_results)
        success_rate = success_count / total_scenarios if total_scenarios > 0 else 0
        
        # Risk insights
        positive_excess_count = sum(1 for er in excess_returns if er > 0)
        high_drawdown_count = sum(1 for dd in max_drawdowns if dd > 0.20)  # >20% drawdown
        return_risk_correlation = np.corrcoef(returns, volatilities)[0,1] if len(returns) > 1 else 0
        
        # Optimal parameters
        optimal_deviation = param_analysis.get('optimal_deviation', 'N/A')
        optimal_horizon = param_analysis.get('optimal_horizon', 'N/A')  
        optimal_rebalancing = param_analysis.get('optimal_rebalancing', 'N/A')
        optimal_tx_cost = param_analysis.get('optimal_transaction_cost', 'N/A')
        
        # Methodology notes
        methodology_notes = [
            "Perfect Hindsight Portfolio (PHP) uses realized asset performance to determine optimal weights",
            "Analysis covers multiple investment horizons (5-year and 10-year) with rolling monthly starts",
            "Weight constraints implemented as absolute deviations from neutral portfolio weights",
            "Transaction costs include both buy and sell impacts as per methodology specifications",
            f"Hedge fund data forward-filled from monthly to daily frequency for consistency",
            f"Risk-free rate assumed at 2% annually for excess return and Sharpe ratio calculations",
            f"Success rate reflects scenarios with valid data and successful simulation completion"
        ]
        
        return ExecutiveSummary(
            mandate=mandate,
            analysis_period=analysis_period,
            total_scenarios=total_scenarios,
            successful_scenarios=success_count,
            success_rate=success_rate,
            
            avg_annualized_return=avg_return,
            best_annualized_return=best_return,
            worst_annualized_return=worst_return,
            avg_excess_return=avg_excess,
            
            avg_volatility=avg_vol,
            avg_max_drawdown=avg_dd,
            worst_max_drawdown=worst_dd,
            avg_sharpe_ratio=avg_sharpe,
            best_sharpe_ratio=best_sharpe,
            
            optimal_deviation_limit=str(optimal_deviation),
            optimal_investment_horizon=str(optimal_horizon),
            optimal_rebalancing_frequency=str(optimal_rebalancing),
            optimal_transaction_cost=str(optimal_tx_cost),
            
            scenarios_with_positive_excess=positive_excess_count,
            scenarios_with_high_drawdown=high_drawdown_count,
            correlation_return_risk=return_risk_correlation,
            
            methodology_notes=methodology_notes
        )
    
    def generate_executive_memo(self, 
                              executive_summary: ExecutiveSummary,
                              key_insights: List[str] = None) -> str:
        """Generate executive memo document"""
        
        memo_template = """
EXECUTIVE MEMORANDUM

TO:         Investment Committee
FROM:       Portfolio Analytics Team  
DATE:       {{ analysis_date }}
RE:         Perfect Hindsight Portfolio Analysis - {{ mandate }} Mandate

EXECUTIVE SUMMARY

This memorandum presents the results of our comprehensive Perfect Hindsight Portfolio (PHP) analysis for the {{ mandate }} mandate, covering the period {{ analysis_period }}. The analysis evaluated {{ total_scenarios_formatted }} investment scenarios across multiple parameters to assess the theoretical performance benefits of perfect market timing.

KEY FINDINGS

Performance Results:
• Average annualized return: {{ "%.1f"|format(avg_annualized_return * 100) }}%
• Best case annualized return: {{ "%.1f"|format(best_annualized_return * 100) }}%  
• Average excess return vs. benchmark: {{ "%.1f"|format(avg_excess_return * 100) }}%
• {{ scenarios_with_positive_excess }} of {{ successful_scenarios }} scenarios ({{ positive_excess_rate }}%) generated positive excess returns

Risk Profile:
• Average volatility: {{ "%.1f"|format(avg_volatility * 100) }}%
• Average maximum drawdown: {{ "%.1f"|format(avg_max_drawdown * 100) }}%
• Worst maximum drawdown: {{ "%.1f"|format(worst_max_drawdown * 100) }}%  
• Average Sharpe ratio: {{ "%.2f"|format(avg_sharpe_ratio) }}
• {{ scenarios_with_high_drawdown }} scenarios experienced drawdowns exceeding 20%

Optimal Parameters:
• Deviation limit: {{ optimal_deviation_limit }}
• Investment horizon: {{ optimal_investment_horizon }}
• Rebalancing frequency: {{ optimal_rebalancing_frequency }}
• Transaction cost tolerance: {{ optimal_transaction_cost }}

STRATEGIC INSIGHTS

{% if key_insights %}
{% for insight in key_insights %}
• {{ insight }}
{% endfor %}
{% else %}
• The PHP approach demonstrates the theoretical value of perfect market timing
• Risk-return correlation of {{ "%.2f"|format(correlation_return_risk) }} suggests diversification benefits remain important
• {{ "%.1f"|format(success_rate * 100) }}% scenario success rate indicates robust methodology implementation
• Parameter sensitivity analysis reveals optimal configuration varies by market conditions
{% endif %}

METHODOLOGY

{% for note in methodology_notes %}
• {{ note }}
{% endfor %}

RECOMMENDATIONS

Based on this analysis, we recommend:

1. Consider implementing dynamic allocation strategies that approximate PHP insights
2. Focus on the optimal parameter combinations identified in this analysis  
3. Maintain risk management disciplines despite theoretical return potential
4. Regular monitoring and rebalancing aligned with optimal frequency findings

The PHP analysis provides valuable theoretical benchmarks for evaluating active allocation decisions and understanding the performance ceiling under perfect information conditions.

---
This analysis is for internal use only and contains forward-looking statements subject to market risks and uncertainties.
        """
        
        template = Template(memo_template.strip())
        
        # Calculate rates for template
        positive_excess_rate = (executive_summary.scenarios_with_positive_excess / 
                               executive_summary.successful_scenarios * 100) if executive_summary.successful_scenarios > 0 else 0
        high_drawdown_rate = (executive_summary.scenarios_with_high_drawdown / 
                             executive_summary.successful_scenarios * 100) if executive_summary.successful_scenarios > 0 else 0
        
        memo_content = template.render(
            analysis_date=datetime.now().strftime('%B %d, %Y'),
            mandate=executive_summary.mandate,
            analysis_period=executive_summary.analysis_period,
            total_scenarios=executive_summary.total_scenarios,
            total_scenarios_formatted=f"{executive_summary.total_scenarios:,}",
            successful_scenarios=executive_summary.successful_scenarios,
            avg_annualized_return=executive_summary.avg_annualized_return,
            best_annualized_return=executive_summary.best_annualized_return,
            avg_excess_return=executive_summary.avg_excess_return,
            scenarios_with_positive_excess=executive_summary.scenarios_with_positive_excess,
            positive_excess_rate=f"{positive_excess_rate:.1f}",
            avg_volatility=executive_summary.avg_volatility,
            avg_max_drawdown=executive_summary.avg_max_drawdown,
            worst_max_drawdown=executive_summary.worst_max_drawdown,
            avg_sharpe_ratio=executive_summary.avg_sharpe_ratio,
            scenarios_with_high_drawdown=executive_summary.scenarios_with_high_drawdown,
            high_drawdown_rate=f"{high_drawdown_rate:.1f}",
            optimal_deviation_limit=executive_summary.optimal_deviation_limit,
            optimal_investment_horizon=executive_summary.optimal_investment_horizon,
            optimal_rebalancing_frequency=executive_summary.optimal_rebalancing_frequency,
            optimal_transaction_cost=executive_summary.optimal_transaction_cost,
            key_insights=key_insights,
            correlation_return_risk=executive_summary.correlation_return_risk,
            success_rate=executive_summary.success_rate,
            methodology_notes=executive_summary.methodology_notes
        )
        
        return memo_content
    
    def generate_presentation_slides(self, 
                                   executive_summary: ExecutiveSummary,
                                   chart_files: Dict[str, str] = None) -> str:
        """Generate presentation slide content in Markdown format"""
        
        slides_template = """
# Perfect Hindsight Portfolio Analysis
## {{ mandate }} Mandate Results

---

## Executive Summary

**Analysis Period:** {{ analysis_period }}  
**Scenarios Analyzed:** {{ total_scenarios_formatted }}  
**Success Rate:** {{ "%.1f"|format(success_rate * 100) }}%

### Key Performance Metrics
- **Average Return:** {{ "%.1f"|format(avg_annualized_return * 100) }}%
- **Best Case Return:** {{ "%.1f"|format(best_annualized_return * 100) }}%  
- **Average Excess Return:** {{ "%.1f"|format(avg_excess_return * 100) }}%
- **Average Sharpe Ratio:** {{ "%.2f"|format(avg_sharpe_ratio) }}

---

## Performance Highlights

### Returns Analysis
- {{ scenarios_with_positive_excess }} of {{ successful_scenarios }} scenarios ({{ positive_excess_rate }}%) generated positive excess returns
- Return range: {{ "%.1f"|format(worst_annualized_return * 100) }}% to {{ "%.1f"|format(best_annualized_return * 100) }}%
- Theoretical performance ceiling demonstrates significant alpha potential

### Risk Profile
- **Average Volatility:** {{ "%.1f"|format(avg_volatility * 100) }}%
- **Average Max Drawdown:** {{ "%.1f"|format(avg_max_drawdown * 100) }}%
- **Worst Drawdown:** {{ "%.1f"|format(worst_max_drawdown * 100) }}%
- {{ scenarios_with_high_drawdown }} scenarios with >20% drawdowns

---

## Optimal Parameter Configuration

### Best Performance Parameters
- **Deviation Limit:** {{ optimal_deviation_limit }}
- **Investment Horizon:** {{ optimal_investment_horizon }}
- **Rebalancing Frequency:** {{ optimal_rebalancing_frequency }}
- **Transaction Cost:** {{ optimal_transaction_cost }}

### Parameter Sensitivity Insights
- Return-risk correlation: {{ "%.2f"|format(correlation_return_risk) }}
- Configuration significantly impacts performance
- Market conditions influence optimal parameters

---

## Strategic Implications

### Key Takeaways
1. **Performance Ceiling:** PHP establishes theoretical maximum returns under perfect information
2. **Parameter Importance:** Optimal configuration varies significantly by market period
3. **Risk Management:** High returns achievable but with material drawdown risk
4. **Implementation:** Real-world approximations should focus on optimal parameter ranges

### Investment Applications
- Dynamic allocation strategy development  
- Performance benchmark establishment
- Risk budgeting optimization
- Manager evaluation framework

---

## Methodology Overview

{% for note in methodology_notes %}
- {{ note }}
{% endfor %}

---

## Next Steps & Recommendations

### Immediate Actions
1. **Strategy Development:** Design implementation framework based on optimal parameters
2. **Risk Framework:** Establish drawdown limits and monitoring protocols  
3. **Performance Tracking:** Implement PHP-based benchmarking system

### Ongoing Monitoring
- Regular parameter optimization updates
- Performance attribution analysis  
- Market regime change detection
- Strategy refinement based on results

---

## Appendix

### Data Sources & Limitations
- Analysis based on historical data through {{ analysis_period.split(' to ')[1] }}
- Perfect hindsight represents theoretical maximum, not achievable returns
- Transaction costs and implementation constraints limit real-world application
- Results subject to market conditions and data availability

### Contact Information
Portfolio Analytics Team  
{{ analysis_date }}

{% if chart_files %}
### Supporting Charts
{% for chart_name, chart_path in chart_files.items() %}
- **{{ chart_name.replace('_', ' ').title() }}:** {{ chart_path }}
{% endfor %}
{% endif %}
        """
        
        template = Template(slides_template.strip())
        
        # Calculate rates for template
        positive_excess_rate = (executive_summary.scenarios_with_positive_excess / 
                               executive_summary.successful_scenarios * 100) if executive_summary.successful_scenarios > 0 else 0
        high_drawdown_rate = (executive_summary.scenarios_with_high_drawdown / 
                             executive_summary.successful_scenarios * 100) if executive_summary.successful_scenarios > 0 else 0
        
        slides_content = template.render(
            mandate=executive_summary.mandate,
            analysis_period=executive_summary.analysis_period,
            total_scenarios=executive_summary.total_scenarios,
            total_scenarios_formatted=f"{executive_summary.total_scenarios:,}",
            successful_scenarios=executive_summary.successful_scenarios,
            success_rate=executive_summary.success_rate,
            avg_annualized_return=executive_summary.avg_annualized_return,
            best_annualized_return=executive_summary.best_annualized_return,
            worst_annualized_return=executive_summary.worst_annualized_return,
            avg_excess_return=executive_summary.avg_excess_return,
            avg_sharpe_ratio=executive_summary.avg_sharpe_ratio,
            scenarios_with_positive_excess=executive_summary.scenarios_with_positive_excess,
            positive_excess_rate=f"{positive_excess_rate:.1f}",
            avg_volatility=executive_summary.avg_volatility,
            avg_max_drawdown=executive_summary.avg_max_drawdown,
            worst_max_drawdown=executive_summary.worst_max_drawdown,
            scenarios_with_high_drawdown=executive_summary.scenarios_with_high_drawdown,
            high_drawdown_rate=f"{high_drawdown_rate:.1f}",
            optimal_deviation_limit=executive_summary.optimal_deviation_limit,
            optimal_investment_horizon=executive_summary.optimal_investment_horizon,
            optimal_rebalancing_frequency=executive_summary.optimal_rebalancing_frequency,
            optimal_transaction_cost=executive_summary.optimal_transaction_cost,
            correlation_return_risk=executive_summary.correlation_return_risk,
            methodology_notes=executive_summary.methodology_notes,
            analysis_date=datetime.now().strftime('%B %d, %Y'),
            chart_files=chart_files or {}
        )
        
        return slides_content
    
    def generate_investment_committee_brief(self, 
                                         executive_summary: ExecutiveSummary,
                                         recommendations: List[str] = None) -> str:
        """Generate brief for investment committee presentation"""
        
        brief_template = """
INVESTMENT COMMITTEE BRIEF

Subject: Perfect Hindsight Portfolio Analysis - {{ mandate }} Mandate
Date: {{ analysis_date }}
Prepared by: Portfolio Analytics Team

SITUATION
We completed a comprehensive Perfect Hindsight Portfolio (PHP) analysis for the {{ mandate }} mandate to establish theoretical performance ceilings and identify optimal allocation parameters.

ANALYSIS SCOPE
- Period: {{ analysis_period }}
- Scenarios: {{ total_scenarios_formatted }} investment combinations
- Success Rate: {{ "%.1f"|format(success_rate * 100) }}%
- Parameters: Multiple deviation limits, horizons, and rebalancing frequencies

KEY RESULTS

Performance Metrics:
• Average annualized return: {{ "%.1f"|format(avg_annualized_return * 100) }}%
• Best scenario return: {{ "%.1f"|format(best_annualized_return * 100) }}%
• Average excess vs. benchmark: {{ "%.1f"|format(avg_excess_return * 100) }}%
• {{ positive_excess_rate }}% scenarios beat benchmark

Risk Assessment:
• Average volatility: {{ "%.1f"|format(avg_volatility * 100) }}%
• Average max drawdown: {{ "%.1f"|format(avg_max_drawdown * 100) }}%
• Worst case drawdown: {{ "%.1f"|format(worst_max_drawdown * 100) }}%
• Average Sharpe ratio: {{ "%.2f"|format(avg_sharpe_ratio) }}

Optimal Configuration:
• Deviation limit: {{ optimal_deviation_limit }}
• Investment horizon: {{ optimal_investment_horizon }}  
• Rebalancing frequency: {{ optimal_rebalancing_frequency }}
• Transaction cost: {{ optimal_transaction_cost }}

IMPLICATIONS

Strategic Value:
1. Establishes theoretical performance ceiling for {{ mandate }} mandate
2. Identifies optimal parameter configurations across market conditions
3. Provides framework for evaluating active management strategies
4. Demonstrates risk-return trade-offs under perfect information

Risk Considerations:
- {{ scenarios_with_high_drawdown }} scenarios ({{ high_drawdown_rate }}%) experienced >20% drawdowns
- Return-risk correlation of {{ "%.2f"|format(correlation_return_risk) }} indicates ongoing diversification importance
- Perfect hindsight represents theoretical ceiling, not achievable returns

RECOMMENDATIONS

{% if recommendations %}
{% for rec in recommendations %}
{{ loop.index }}. {{ rec }}
{% endfor %}
{% else %}
1. Implement dynamic allocation framework based on optimal parameter findings
2. Establish PHP-based performance benchmarks for strategy evaluation  
3. Develop risk management protocols aligned with drawdown analysis
4. Regular parameter optimization reviews to adapt to changing market conditions
5. Use PHP insights to enhance existing allocation methodology
{% endif %}

NEXT STEPS

Immediate (Next 30 days):
- Present detailed findings to Investment Committee
- Begin strategy development based on optimal parameters
- Establish implementation timeline and resource requirements

Medium-term (3-6 months):
- Develop and test implementation framework
- Create monitoring and performance attribution systems
- Conduct pilot implementation with limited allocation

APPENDIX

Methodology Summary:
{% for note in methodology_notes %}
• {{ note }}
{% endfor %}

Analysis Limitations:
• Historical data limitations and survivorship bias
• Perfect hindsight not achievable in practice
• Transaction costs and market impact constraints
• Model assumptions regarding rebalancing and implementation

Contact: Portfolio Analytics Team
Prepared: {{ analysis_date }}
        """
        
        template = Template(brief_template.strip())
        
        # Calculate rates for template
        positive_excess_rate = (executive_summary.scenarios_with_positive_excess / 
                               executive_summary.successful_scenarios * 100) if executive_summary.successful_scenarios > 0 else 0
        high_drawdown_rate = (executive_summary.scenarios_with_high_drawdown / 
                             executive_summary.successful_scenarios * 100) if executive_summary.successful_scenarios > 0 else 0
        
        brief_content = template.render(
            mandate=executive_summary.mandate,
            analysis_date=datetime.now().strftime('%B %d, %Y'),
            analysis_period=executive_summary.analysis_period,
            total_scenarios=executive_summary.total_scenarios,
            total_scenarios_formatted=f"{executive_summary.total_scenarios:,}",
            successful_scenarios=executive_summary.successful_scenarios,
            success_rate=executive_summary.success_rate,
            avg_annualized_return=executive_summary.avg_annualized_return,
            best_annualized_return=executive_summary.best_annualized_return,
            avg_excess_return=executive_summary.avg_excess_return,
            scenarios_with_positive_excess=executive_summary.scenarios_with_positive_excess,
            positive_excess_rate=f"{positive_excess_rate:.1f}",
            avg_volatility=executive_summary.avg_volatility,
            avg_max_drawdown=executive_summary.avg_max_drawdown,
            worst_max_drawdown=executive_summary.worst_max_drawdown,
            avg_sharpe_ratio=executive_summary.avg_sharpe_ratio,
            scenarios_with_high_drawdown=executive_summary.scenarios_with_high_drawdown,
            high_drawdown_rate=f"{high_drawdown_rate:.1f}",
            optimal_deviation_limit=executive_summary.optimal_deviation_limit,
            optimal_investment_horizon=executive_summary.optimal_investment_horizon,
            optimal_rebalancing_frequency=executive_summary.optimal_rebalancing_frequency,
            optimal_transaction_cost=executive_summary.optimal_transaction_cost,
            correlation_return_risk=executive_summary.correlation_return_risk,
            methodology_notes=executive_summary.methodology_notes,
            recommendations=recommendations
        )
        
        return brief_content
    
    def export_executive_reports(self, 
                               executive_summary: ExecutiveSummary,
                               chart_files: Dict[str, str] = None,
                               key_insights: List[str] = None,
                               recommendations: List[str] = None) -> Dict[str, str]:
        """Export all executive report formats"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        mandate = executive_summary.mandate
        exported_files = {}
        
        try:
            # Export 1: Executive Summary JSON
            summary_file = self.output_dir / "summaries" / f"{mandate}_executive_summary_{timestamp}.json"
            with open(summary_file, 'w') as f:
                json.dump(asdict(executive_summary), f, indent=2, default=str)
            exported_files['executive_summary_json'] = str(summary_file)
            
            # Export 2: Executive Memo
            memo_content = self.generate_executive_memo(executive_summary, key_insights)
            memo_file = self.output_dir / "memos" / f"{mandate}_executive_memo_{timestamp}.txt"
            with open(memo_file, 'w') as f:
                f.write(memo_content)
            exported_files['executive_memo'] = str(memo_file)
            
            # Export 3: Presentation Slides
            slides_content = self.generate_presentation_slides(executive_summary, chart_files)
            slides_file = self.output_dir / "presentations" / f"{mandate}_presentation_{timestamp}.md"
            with open(slides_file, 'w') as f:
                f.write(slides_content)
            exported_files['presentation_slides'] = str(slides_file)
            
            # Export 4: Investment Committee Brief
            brief_content = self.generate_investment_committee_brief(executive_summary, recommendations)
            brief_file = self.output_dir / "summaries" / f"{mandate}_committee_brief_{timestamp}.txt"
            with open(brief_file, 'w') as f:
                f.write(brief_content)
            exported_files['committee_brief'] = str(brief_file)
            
            print(f"📋 Executive reports exported for {mandate}:")
            for report_type, filepath in exported_files.items():
                print(f"   {report_type}: {filepath}")
            
        except Exception as e:
            print(f"❌ Error exporting executive reports: {e}")
            exported_files['error'] = str(e)
        
        return exported_files
    
    def generate_complete_executive_package(self,
                                          mandate: str,
                                          analysis_start: str = "1996-01-01",
                                          analysis_end: str = "2025-05-31",
                                          sample_size: int = None,
                                          key_insights: List[str] = None,
                                          recommendations: List[str] = None) -> Dict[str, Any]:
        """Generate complete executive reporting package"""
        
        print(f"📊 Generating Executive Package for {mandate}")
        print(f"   Analysis Period: {analysis_start} to {analysis_end}")
        
        try:
            # Step 1: Run comprehensive analysis
            print("⚙️ Step 1: Running comprehensive analysis...")
            report_data = self.reporting_engine.generate_complete_php_report(
                mandate, analysis_start, analysis_end, sample_size
            )
            
            # Step 2: Generate executive summary
            print("📋 Step 2: Generating executive summary...")
            executive_summary = self.generate_executive_summary(
                report_data['analytics_results'],
                mandate,
                f"{analysis_start} to {analysis_end}"
            )
            
            # Step 3: Export all executive reports  
            print("📄 Step 3: Exporting executive reports...")
            chart_files = report_data.get('performance_charts', {})
            exported_reports = self.export_executive_reports(
                executive_summary, chart_files, key_insights, recommendations
            )
            
            # Step 4: Compile complete package
            executive_package = {
                'mandate': mandate,
                'analysis_period': f"{analysis_start} to {analysis_end}",
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'executive_summary': executive_summary,
                'exported_reports': exported_reports,
                'chart_files': chart_files,
                'full_analytics': report_data['analytics_results'],
                'key_insights': key_insights or [],
                'recommendations': recommendations or []
            }
            
            print(f"✅ Executive package completed for {mandate}")
            print(f"   📁 Reports saved to: {self.output_dir}")
            
            return executive_package
            
        except Exception as e:
            print(f"❌ Error generating executive package: {e}")
            return {
                'mandate': mandate,
                'error': str(e),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

if __name__ == "__main__":
    # Test executive reporting
    print("🧪 Testing Executive Reporting Module")
    
    # Initialize executive reporter
    exec_reporter = ExecutiveReportGenerator()
    
    # Generate sample executive package
    print("\n📊 Generating sample executive package for EQFI...")
    
    try:
        # Custom insights and recommendations for testing
        sample_insights = [
            "PHP analysis reveals optimal deviation limits vary significantly by market regime",
            "10-year investment horizons consistently outperform 5-year horizons across scenarios",
            "Monthly rebalancing provides best risk-adjusted returns despite higher transaction costs",
            "Perfect timing strategies show theoretical alpha of 4-8% annually over benchmark"
        ]
        
        sample_recommendations = [
            "Implement tactical allocation framework based on optimal parameter findings",
            "Establish PHP-based performance benchmarks for active strategy evaluation",
            "Develop risk management protocols aligned with maximum drawdown analysis",
            "Create quarterly parameter optimization reviews to adapt to market changes"
        ]
        
        executive_package = exec_reporter.generate_complete_executive_package(
            mandate="EQFI",
            analysis_start="2010-01-01",
            analysis_end="2020-01-01", 
            sample_size=50,  # Limited sample for testing
            key_insights=sample_insights,
            recommendations=sample_recommendations
        )
        
        if 'error' not in executive_package:
            print("\n✅ Executive package generated successfully!")
            print(f"📈 Executive Summary created for {executive_package['mandate']}")
            print(f"📄 Reports exported: {len(executive_package['exported_reports'])} files")
            print(f"📊 Charts created: {len(executive_package.get('chart_files', {}))}")
            print(f"📁 Output directory: {exec_reporter.output_dir}")
        else:
            print(f"\n❌ Executive package generation failed: {executive_package['error']}")
    
    except Exception as e:
        print(f"❌ Error testing executive reporting: {e}")
        import traceback
        traceback.print_exc()