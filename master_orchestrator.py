import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import warnings
from pathlib import Path
import argparse
import sys
from dataclasses import dataclass, asdict

# Import all PHP modules
from data_mapping import DataMapper
from portfolio_config import PortfolioConfig
from perfect_weight_calculator import PerfectWeightCalculator
from portfolio_simulation import PortfolioSimulation, SimulationConfig, ParallelConfig
from performance_analytics import PerformanceAnalytics
from reporting_visualization import PHPReporting
from interactive_dashboard import InteractiveDashboard
from executive_reporting import ExecutiveReportGenerator, ExecutiveSummary

@dataclass
class PHPAnalysisConfig:
    """Complete PHP analysis configuration"""
    mandate: str
    analysis_start: str
    analysis_end: str
    sample_size: Optional[int] = None
    output_formats: List[str] = None
    generate_dashboard: bool = True
    generate_executive_reports: bool = True
    custom_insights: List[str] = None
    custom_recommendations: List[str] = None
    parallel_config: Optional[ParallelConfig] = None
    
    def __post_init__(self):
        if self.output_formats is None:
            self.output_formats = ['json', 'excel', 'html', 'pdf']
        if self.custom_insights is None:
            self.custom_insights = []
        if self.custom_recommendations is None:
            self.custom_recommendations = []
        if self.parallel_config is None:
            self.parallel_config = ParallelConfig()

class PHPMasterOrchestrator:
    """Master orchestrator for complete PHP analysis pipeline"""
    
    def __init__(self, base_output_dir: str = "reports"):
        """Initialize the master orchestrator with all modules"""
        
        print("🚀 Initializing PHP Master Orchestrator...")
        
        # Core modules
        self.data_mapper = DataMapper()
        
        # Clear any existing forward-fill flags for fresh start
        self.data_mapper.clear_forward_fill_flags()
        self.portfolio_config = PortfolioConfig(self.data_mapper)
        self.perfect_weight_calc = PerfectWeightCalculator(
            self.data_mapper, self.portfolio_config)
        self.simulation_engine = PortfolioSimulation(
            self.data_mapper, self.portfolio_config, self.perfect_weight_calc)
        
        # Analytics and reporting modules  
        self.analytics_engine = PerformanceAnalytics()
        self.reporting_engine = PHPReporting(
            self.data_mapper, self.portfolio_config, self.perfect_weight_calc,
            self.simulation_engine, self.analytics_engine)
        self.dashboard_engine = InteractiveDashboard()
        self.executive_reporter = ExecutiveReportGenerator(
            self.data_mapper, self.portfolio_config, self.perfect_weight_calc,
            self.simulation_engine, self.analytics_engine, self.reporting_engine)
        
        # Output management
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        
        # Analysis cache
        self.analysis_cache = {}
        
        print("✅ PHP Master Orchestrator initialized successfully")
        print(f"📁 Base output directory: {self.base_output_dir}")
    
    def run_complete_php_analysis(self, config: PHPAnalysisConfig) -> Dict[str, Any]:
        """Run complete end-to-end PHP analysis pipeline"""
        
        print(f"\n🎯 Starting Complete PHP Analysis")
        print(f"   Mandate: {config.mandate}")
        print(f"   Period: {config.analysis_start} to {config.analysis_end}")
        print(f"   Sample Size: {config.sample_size or 'All scenarios'}")
        
        start_time = datetime.now()
        analysis_results = {
            'config': config,
            'start_time': start_time,
            'success': False,
            'error': None
        }
        
        try:
            # Phase 1: Data validation and setup
            print("\n📊 Phase 1: Data Validation")
            validation_results = self._validate_analysis_setup(config)
            if not validation_results['valid']:
                raise ValueError(f"Validation failed: {validation_results['errors']}")
            analysis_results['validation'] = validation_results
            
            # Phase 2: Core analysis (simulation + analytics)
            print("\n⚙️ Phase 2: Core Analysis")
            core_results = self._run_core_analysis(config)
            analysis_results['core_analysis'] = core_results
            
            # Phase 3: Comprehensive reporting
            print("\n📋 Phase 3: Comprehensive Reporting")
            reporting_results = self._generate_comprehensive_reports(config, core_results)
            analysis_results['reporting'] = reporting_results
            
            # Phase 4: Interactive dashboards (if requested)
            if config.generate_dashboard:
                print("\n📊 Phase 4: Interactive Dashboards")
                dashboard_results = self._generate_interactive_dashboards(config, core_results)
                analysis_results['dashboards'] = dashboard_results
            
            # Phase 5: Executive reporting (if requested)
            if config.generate_executive_reports:
                print("\n👔 Phase 5: Executive Reporting") 
                executive_results = self._generate_executive_package(config, core_results)
                analysis_results['executive'] = executive_results
            
            # Phase 6: Export and finalization
            print("\n💾 Phase 6: Export and Finalization")
            export_results = self._export_final_package(config, analysis_results)
            analysis_results['exports'] = export_results
            
            # Mark as successful
            analysis_results['success'] = True
            analysis_results['end_time'] = datetime.now()
            analysis_results['total_duration'] = analysis_results['end_time'] - start_time
            
            print(f"\n✅ Complete PHP Analysis Finished Successfully!")
            print(f"   Duration: {analysis_results['total_duration']}")
            print(f"   Output Directory: {self.base_output_dir}")
            
            return analysis_results
            
        except Exception as e:
            print(f"\n❌ PHP Analysis Failed: {str(e)}")
            analysis_results['success'] = False
            analysis_results['error'] = str(e)
            analysis_results['end_time'] = datetime.now()
            
            import traceback
            analysis_results['traceback'] = traceback.format_exc()
            
            return analysis_results
    
    def _validate_analysis_setup(self, config: PHPAnalysisConfig) -> Dict[str, Any]:
        """Validate analysis configuration and data availability"""
        
        validation = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'data_summary': {}
        }
        
        try:
            # Check mandate validity
            valid_mandates = ['6040', 'EQFI', 'EQFILA', 'EQFILAIA']
            if config.mandate not in valid_mandates:
                validation['errors'].append(f"Invalid mandate: {config.mandate}")
            
            # Check date range validity
            try:
                start_date = pd.to_datetime(config.analysis_start)
                end_date = pd.to_datetime(config.analysis_end)
                
                if start_date >= end_date:
                    validation['errors'].append("Start date must be before end date")
                
                if (end_date - start_date).days < 365:
                    validation['warnings'].append("Analysis period less than 1 year")
                    
            except Exception as e:
                validation['errors'].append(f"Invalid date format: {e}")
            
            # Check data availability
            if config.mandate in valid_mandates:
                try:
                    data_summary = self.data_mapper.validate_data_availability()
                    required_indices = self.data_mapper.get_required_indices(config.mandate)
                    
                    missing_data = [idx for idx in required_indices if not data_summary.get(idx, False)]
                    if missing_data:
                        validation['errors'].append(f"Missing data for indices: {missing_data}")
                    
                    validation['data_summary'] = {
                        'required_indices': len(required_indices),
                        'available_indices': sum(data_summary.values()),
                        'missing_indices': len(missing_data),
                        'mandate_weights': self.portfolio_config.get_mandate_weights(config.mandate)
                    }
                    
                except Exception as e:
                    validation['errors'].append(f"Data validation error: {e}")
            
            # Set overall validity
            validation['valid'] = len(validation['errors']) == 0
            
            # Print validation summary
            if validation['valid']:
                print("✅ Validation passed")
                print(f"   Required indices: {validation['data_summary'].get('required_indices', 0)}")
                print(f"   Available indices: {validation['data_summary'].get('available_indices', 0)}")
            else:
                print("❌ Validation failed:")
                for error in validation['errors']:
                    print(f"   • {error}")
            
            if validation['warnings']:
                print("⚠️ Validation warnings:")
                for warning in validation['warnings']:
                    print(f"   • {warning}")
            
        except Exception as e:
            validation['valid'] = False
            validation['errors'].append(f"Validation exception: {str(e)}")
        
        return validation
    
    def _run_core_analysis(self, config: PHPAnalysisConfig) -> Dict[str, Any]:
        """Run core simulation and analytics"""
        
        # Generate scenarios
        print("   Generating scenarios...")
        all_scenarios = self.simulation_engine.generate_all_php_scenarios(
            config.mandate, config.analysis_start, config.analysis_end
        )
        
        # Apply sample size limit
        if config.sample_size and len(all_scenarios) > config.sample_size:
            scenarios = all_scenarios[:config.sample_size]
            print(f"   Using {len(scenarios)} of {len(all_scenarios)} scenarios")
        else:
            scenarios = all_scenarios
            print(f"   Processing all {len(scenarios)} scenarios")
        
        # Run simulations
        print("   Running simulations...")
        simulation_results = self.simulation_engine.simulate_scenarios_parallel(
            scenarios, config.parallel_config
        )
        
        # Run analytics
        print("   Running comprehensive analytics...")
        analytics_results = self.analytics_engine.run_comprehensive_analysis_from_results(
            simulation_results, f"{config.mandate}_master"
        )
        
        core_results = {
            'total_scenarios_generated': len(all_scenarios),
            'scenarios_analyzed': len(scenarios),
            'simulation_results': simulation_results,
            'analytics_results': analytics_results,
            'successful_scenarios': len([r for r in simulation_results if r.get('success', False)])
        }
        
        success_rate = core_results['successful_scenarios'] / core_results['scenarios_analyzed']
        print(f"   ✅ Core analysis complete: {success_rate:.1%} success rate")
        
        return core_results
    
    def _generate_comprehensive_reports(self, 
                                      config: PHPAnalysisConfig,
                                      core_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive reports and visualizations"""
        
        analytics_results = core_results['analytics_results']
        
        # Generate complete reporting package
        reporting_package = {
            'mandate': config.mandate,
            'analysis_period': f"{config.analysis_start} to {config.analysis_end}",
            'total_scenarios': core_results['scenarios_analyzed'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'analytics_results': analytics_results
        }
        
        # Generate executive summary
        reporting_package['executive_summary'] = self.reporting_engine._generate_executive_summary(
            analytics_results, config.mandate
        )
        
        # Generate visualizations
        print("   Creating performance charts...")
        reporting_package['performance_charts'] = self.reporting_engine._create_performance_visualizations(
            analytics_results, config.mandate
        )
        
        # Generate comparison tables
        print("   Creating comparison tables...")
        reporting_package['comparison_tables'] = self.reporting_engine._create_comparison_tables(
            analytics_results, config.mandate
        )
        
        # Generate scenario deep dive
        print("   Creating scenario analysis...")
        reporting_package['scenario_deep_dive'] = self.reporting_engine._create_scenario_deep_dive(
            analytics_results, config.mandate
        )
        
        # Generate risk analysis
        print("   Creating risk analysis...")
        reporting_package['risk_analysis'] = self.reporting_engine._create_risk_analysis_report(
            analytics_results, config.mandate
        )
        
        print("   ✅ Comprehensive reporting complete")
        
        return reporting_package
    
    def _generate_interactive_dashboards(self,
                                       config: PHPAnalysisConfig,
                                       core_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate interactive dashboards"""
        
        analytics_results = core_results['analytics_results']
        dashboard_results = {}
        
        try:
            # Main performance dashboard
            print("   Creating performance dashboard...")
            main_dashboard = self.dashboard_engine.create_interactive_performance_dashboard(
                analytics_results, config.mandate
            )
            
            dashboard_path = self.dashboard_engine.save_dashboard_html(
                main_dashboard,
                f"{config.mandate}_performance_dashboard.html",
                f"{config.mandate} PHP Performance Dashboard"
            )
            dashboard_results['performance_dashboard'] = dashboard_path
            
            # Parameter sensitivity dashboard
            print("   Creating sensitivity analysis...")
            sensitivity_dashboard = self.dashboard_engine.create_parameter_sensitivity_analysis(
                analytics_results, config.mandate
            )
            
            sensitivity_path = self.dashboard_engine.save_dashboard_html(
                sensitivity_dashboard,
                f"{config.mandate}_sensitivity_analysis.html",
                f"{config.mandate} Parameter Sensitivity Analysis"
            )
            dashboard_results['sensitivity_dashboard'] = sensitivity_path
            
            # Scenario comparison dashboard (top performers)
            top_performers = analytics_results.get('top_performers', {}).get('top_10', [])
            if top_performers:
                print("   Creating scenario comparison...")
                comparison_dashboard = self.dashboard_engine.create_scenario_comparison_chart(
                    top_performers[:5], "performance"
                )
                
                comparison_path = self.dashboard_engine.save_dashboard_html(
                    comparison_dashboard,
                    f"{config.mandate}_scenario_comparison.html", 
                    f"{config.mandate} Top Scenarios Comparison"
                )
                dashboard_results['comparison_dashboard'] = comparison_path
            
            print("   ✅ Interactive dashboards complete")
            
        except Exception as e:
            print(f"   ⚠️ Dashboard generation warning: {e}")
            dashboard_results['error'] = str(e)
        
        return dashboard_results
    
    def _generate_executive_package(self,
                                  config: PHPAnalysisConfig,
                                  core_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive reporting package"""
        
        try:
            analytics_results = core_results['analytics_results']
            
            # Generate executive summary
            print("   Creating executive summary...")
            executive_summary = self.executive_reporter.generate_executive_summary(
                analytics_results,
                config.mandate,
                f"{config.analysis_start} to {config.analysis_end}"
            )
            
            # Generate all executive report formats
            print("   Generating executive reports...")
            exported_reports = self.executive_reporter.export_executive_reports(
                executive_summary,
                chart_files={},  # Charts handled separately
                key_insights=config.custom_insights,
                recommendations=config.custom_recommendations
            )
            
            executive_results = {
                'executive_summary': executive_summary,
                'exported_reports': exported_reports,
                'custom_insights': config.custom_insights,
                'custom_recommendations': config.custom_recommendations
            }
            
            print("   ✅ Executive package complete")
            
            return executive_results
            
        except Exception as e:
            print(f"   ⚠️ Executive reporting warning: {e}")
            return {'error': str(e)}
    
    def _export_final_package(self,
                            config: PHPAnalysisConfig,
                            analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Export final analysis package in requested formats"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        mandate = config.mandate
        export_results = {}
        
        try:
            # Create final output directory
            final_output_dir = self.base_output_dir / f"{mandate}_complete_analysis_{timestamp}"
            final_output_dir.mkdir(exist_ok=True)
            
            # Export master summary JSON
            if 'json' in config.output_formats:
                print("   Exporting JSON summary...")
                
                # Prepare JSON-safe analysis results
                json_summary = {
                    'analysis_config': asdict(config),
                    'validation_results': analysis_results.get('validation', {}),
                    'core_analysis_summary': {
                        'total_scenarios': analysis_results.get('core_analysis', {}).get('scenarios_analyzed', 0),
                        'successful_scenarios': analysis_results.get('core_analysis', {}).get('successful_scenarios', 0),
                        'success_rate': analysis_results.get('core_analysis', {}).get('successful_scenarios', 0) / 
                                      max(1, analysis_results.get('core_analysis', {}).get('scenarios_analyzed', 1))
                    },
                    'executive_summary': asdict(analysis_results.get('executive', {}).get('executive_summary', ExecutiveSummary(
                        mandate='', analysis_period='', total_scenarios=0, successful_scenarios=0, success_rate=0.0,
                        avg_annualized_return=0.0, best_annualized_return=0.0, worst_annualized_return=0.0, avg_excess_return=0.0,
                        avg_volatility=0.0, avg_max_drawdown=0.0, worst_max_drawdown=0.0, avg_sharpe_ratio=0.0, best_sharpe_ratio=0.0,
                        optimal_deviation_limit='', optimal_investment_horizon='', optimal_rebalancing_frequency='', optimal_transaction_cost='',
                        scenarios_with_positive_excess=0, scenarios_with_high_drawdown=0, correlation_return_risk=0.0,
                        methodology_notes=[]
                    ))),
                    'analysis_metadata': {
                        'start_time': analysis_results['start_time'].strftime('%Y-%m-%d %H:%M:%S'),
                        'end_time': analysis_results.get('end_time', datetime.now()).strftime('%Y-%m-%d %H:%M:%S'),
                        'total_duration_seconds': analysis_results.get('total_duration', timedelta(0)).total_seconds(),
                        'success': analysis_results['success']
                    }
                }
                
                json_file = final_output_dir / f"{mandate}_master_summary.json"
                with open(json_file, 'w') as f:
                    json.dump(json_summary, f, indent=2, default=str)
                export_results['json_summary'] = str(json_file)
            
            # Create README file
            print("   Creating documentation...")
            readme_content = f"""
# {mandate} PHP Analysis Results

Generated: {timestamp}
Period: {config.analysis_start} to {config.analysis_end}
Duration: {analysis_results.get('total_duration', 'Unknown')}

## Contents

### Core Analysis
- Total scenarios: {analysis_results.get('core_analysis', {}).get('scenarios_analyzed', 0):,d}
- Successful scenarios: {analysis_results.get('core_analysis', {}).get('successful_scenarios', 0):,d}
- Success rate: {analysis_results.get('core_analysis', {}).get('successful_scenarios', 0) / max(1, analysis_results.get('core_analysis', {}).get('scenarios_analyzed', 1)):.1%}

### Generated Reports
"""
            
            if 'reporting' in analysis_results:
                readme_content += f"- Executive summary: Yes\n"
                readme_content += f"- Performance charts: {len(analysis_results.get('reporting', {}).get('performance_charts', {}))}\n" 
                readme_content += f"- Comparison tables: {len(analysis_results.get('reporting', {}).get('comparison_tables', {}))}\n"
            
            if 'dashboards' in analysis_results:
                readme_content += f"- Interactive dashboards: {len(analysis_results.get('dashboards', {}))}\n"
            
            if 'executive' in analysis_results:
                readme_content += f"- Executive reports: {len(analysis_results.get('executive', {}).get('exported_reports', {}))}\n"
            
            readme_content += f"""
### Methodology
- Perfect Hindsight Portfolio analysis
- Rolling monthly investment scenarios
- Multiple parameter combinations tested
- Comprehensive risk-adjusted performance metrics

### Files
- Master summary: {mandate}_master_summary.json
- Documentation: README.md
"""
            
            readme_file = final_output_dir / "README.md"
            with open(readme_file, 'w') as f:
                f.write(readme_content.strip())
            export_results['documentation'] = str(readme_file)
            
            # Copy key files to final directory
            print("   Organizing final files...")
            export_results['final_directory'] = str(final_output_dir)
            export_results['exported_formats'] = config.output_formats
            
            print(f"   ✅ Final package exported to: {final_output_dir}")
            
        except Exception as e:
            print(f"   ⚠️ Export warning: {e}")
            export_results['error'] = str(e)
        
        return export_results
    
    def run_mandate_comparison(self, 
                             mandates: List[str],
                             analysis_start: str = "2010-01-01",
                             analysis_end: str = "2020-01-01",
                             sample_size: int = 50) -> Dict[str, Any]:
        """Run comparative analysis across multiple mandates"""
        
        print(f"\n🏆 Starting Mandate Comparison Analysis")
        print(f"   Mandates: {mandates}")
        print(f"   Period: {analysis_start} to {analysis_end}")
        print(f"   Sample Size: {sample_size}")
        
        comparison_results = {
            'mandates': mandates,
            'analysis_period': f"{analysis_start} to {analysis_end}",
            'sample_size': sample_size,
            'start_time': datetime.now(),
            'mandate_results': {},
            'comparison_analysis': {},
            'success': False
        }
        
        try:
            # Run analysis for each mandate
            for mandate in mandates:
                print(f"\n📊 Analyzing {mandate} mandate...")
                
                config = PHPAnalysisConfig(
                    mandate=mandate,
                    analysis_start=analysis_start,
                    analysis_end=analysis_end,
                    sample_size=sample_size,
                    generate_dashboard=False,  # Skip dashboards for comparison
                    generate_executive_reports=False,
                    output_formats=['json']
                )
                
                mandate_results = self.run_complete_php_analysis(config)
                comparison_results['mandate_results'][mandate] = mandate_results
            
            # Generate comparison analysis
            print(f"\n⚖️ Generating cross-mandate comparison...")
            comparison_analysis = self._generate_mandate_comparison_analysis(comparison_results['mandate_results'])
            comparison_results['comparison_analysis'] = comparison_analysis
            
            comparison_results['success'] = True
            comparison_results['end_time'] = datetime.now()
            comparison_results['total_duration'] = comparison_results['end_time'] - comparison_results['start_time']
            
            print(f"\n✅ Mandate Comparison Complete!")
            print(f"   Duration: {comparison_results['total_duration']}")
            
            return comparison_results
            
        except Exception as e:
            print(f"\n❌ Mandate Comparison Failed: {str(e)}")
            comparison_results['success'] = False
            comparison_results['error'] = str(e)
            return comparison_results
    
    def _generate_mandate_comparison_analysis(self, mandate_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparative analysis across mandates"""
        
        comparison_analysis = {
            'performance_comparison': {},
            'risk_comparison': {},
            'efficiency_comparison': {},
            'optimal_parameters': {}
        }
        
        # Extract summary metrics for each mandate
        mandate_summaries = {}
        
        for mandate, results in mandate_results.items():
            if results.get('success') and 'executive' in results:
                exec_summary = results['executive'].get('executive_summary')
                if exec_summary:
                    mandate_summaries[mandate] = {
                        'avg_return': exec_summary.avg_annualized_return,
                        'best_return': exec_summary.best_annualized_return,
                        'avg_volatility': exec_summary.avg_volatility,
                        'avg_sharpe': exec_summary.avg_sharpe_ratio,
                        'avg_max_drawdown': exec_summary.avg_max_drawdown,
                        'success_rate': exec_summary.success_rate,
                        'optimal_deviation': exec_summary.optimal_deviation_limit,
                        'optimal_horizon': exec_summary.optimal_investment_horizon
                    }
        
        if mandate_summaries:
            # Performance ranking
            performance_ranking = sorted(mandate_summaries.items(), 
                                       key=lambda x: x[1]['avg_return'], reverse=True)
            comparison_analysis['performance_comparison']['return_ranking'] = [
                {'rank': i+1, 'mandate': mandate, 'avg_return': data['avg_return']}
                for i, (mandate, data) in enumerate(performance_ranking)
            ]
            
            # Risk-adjusted performance ranking
            risk_adj_ranking = sorted(mandate_summaries.items(),
                                    key=lambda x: x[1]['avg_sharpe'], reverse=True)
            comparison_analysis['efficiency_comparison']['sharpe_ranking'] = [
                {'rank': i+1, 'mandate': mandate, 'avg_sharpe': data['avg_sharpe']}
                for i, (mandate, data) in enumerate(risk_adj_ranking)
            ]
            
            # Risk comparison
            risk_ranking = sorted(mandate_summaries.items(),
                                key=lambda x: x[1]['avg_max_drawdown'])
            comparison_analysis['risk_comparison']['drawdown_ranking'] = [
                {'rank': i+1, 'mandate': mandate, 'avg_drawdown': data['avg_max_drawdown']}
                for i, (mandate, data) in enumerate(risk_ranking)
            ]
            
            # Summary statistics
            comparison_analysis['summary_statistics'] = {
                'best_performing_mandate': performance_ranking[0][0],
                'most_risk_adjusted_mandate': risk_adj_ranking[0][0],
                'lowest_risk_mandate': risk_ranking[0][0],
                'average_return_across_mandates': np.mean([data['avg_return'] for data in mandate_summaries.values()]),
                'average_volatility_across_mandates': np.mean([data['avg_volatility'] for data in mandate_summaries.values()]),
                'average_sharpe_across_mandates': np.mean([data['avg_sharpe'] for data in mandate_summaries.values()])
            }
        
        return comparison_analysis
    
    def vs_bm(self,
             mandate: str,
             analysis_start: str = "2010-01-01",
             analysis_end: str = "2020-01-01",
             sample_size: Optional[int] = None,
             output_mode: str = "both",
             include_advanced_stats: bool = True) -> Dict[str, Any]:
        """
        Run detailed mandate vs 6040 benchmark comparison analysis
        
        Args:
            mandate: Target mandate to compare (EQFI, EQFILA, EQFILAIA)
            analysis_start: Analysis start date
            analysis_end: Analysis end date
            sample_size: Optional limit for testing
            output_mode: "individual", "aggregated", or "both"
            include_advanced_stats: Include Information Ratio, Alpha, Beta calculations
            
        Returns:
            Comprehensive comparison analysis results
        """
        
        # Validate mandate
        valid_mandates = ['EQFI', 'EQFILA', 'EQFILAIA']
        if mandate not in valid_mandates:
            raise ValueError(f"Mandate must be one of {valid_mandates}")
        
        benchmark = "6040"
        
        print(f"\n🎯 Starting Mandate vs Benchmark Analysis")
        print(f"   Mandate: {mandate}")
        print(f"   Benchmark: {benchmark}")
        print(f"   Period: {analysis_start} to {analysis_end}")
        print(f"   Output Mode: {output_mode}")
        
        start_time = datetime.now()
        analysis_results = {
            'mandate': mandate,
            'benchmark': benchmark,
            'analysis_period': f"{analysis_start} to {analysis_end}",
            'start_time': start_time,
            'success': False,
            'output_mode': output_mode
        }
        
        try:
            # Phase 1: Generate mandate scenarios
            print("\n📊 Phase 1: Generating mandate scenarios...")
            mandate_scenarios = self.simulation_engine.generate_all_php_scenarios(
                mandate, analysis_start, analysis_end
            )
            
            if sample_size and len(mandate_scenarios) > sample_size:
                mandate_scenarios = mandate_scenarios[:sample_size]
                print(f"   Using {len(mandate_scenarios)} of total scenarios (sample)")
            else:
                print(f"   Generated {len(mandate_scenarios)} total scenarios")
            
            # Phase 2: Generate unique benchmark scenarios (with optimization)
            print("\n📊 Phase 2: Generating optimized benchmark scenarios...")
            unique_benchmark_scenarios, benchmark_mapping = self._generate_benchmark_scenarios(mandate_scenarios, benchmark)
            
            # Phase 3: Run parallel simulations
            print("\n⚙️ Phase 3: Running parallel simulations...")
            
            # Configure parallel processing
            parallel_config = ParallelConfig(
                use_parallel=True,
                max_workers=None,  # Auto-detect
                progress_reporting=True
            )
            
            print(f"   Simulating mandate scenarios ({len(mandate_scenarios)} total)...")
            mandate_results = self.simulation_engine.simulate_scenarios_parallel(
                mandate_scenarios, parallel_config
            )
            
            print(f"   Simulating unique benchmark scenarios ({len(unique_benchmark_scenarios)} total, {len(mandate_scenarios)/len(unique_benchmark_scenarios) if unique_benchmark_scenarios else 1:.1f}x optimization)...")
            unique_benchmark_results = self.simulation_engine.simulate_scenarios_parallel(
                unique_benchmark_scenarios, parallel_config
            )
            
            # Phase 3.5: Map unique benchmark results back to mandate scenarios
            print("   Mapping benchmark results to mandate scenarios...")
            expanded_benchmark_results = self._expand_benchmark_results(
                unique_benchmark_results, benchmark_mapping
            )
            print(f"   Expanded {len(unique_benchmark_results)} -> {len(expanded_benchmark_results)} benchmark results")
            
            # Phase 4: Perform scenario-by-scenario comparison
            print("\n📈 Phase 4: Performing detailed comparison analysis...")
            comparison_results = self._perform_scenario_comparison(
                mandate_results, 
                expanded_benchmark_results, 
                include_advanced_stats
            )
            
            # Phase 5: Generate outputs based on mode
            print(f"\n📋 Phase 5: Generating {output_mode} output...")
            
            output_results = {}
            
            if output_mode in ["individual", "both"]:
                output_results['individual_comparisons'] = comparison_results['individual_comparisons']
                print(f"   Generated {len(comparison_results['individual_comparisons'])} individual comparisons")
            
            if output_mode in ["aggregated", "both"]:
                output_results['aggregated_analysis'] = self._generate_aggregated_analysis(
                    comparison_results, mandate, benchmark, include_advanced_stats
                )
                print("   Generated aggregated statistical analysis")
            
            # Phase 6: Create visualizations
            print("\n📊 Phase 6: Creating comparison visualizations...")
            visualization_results = self._create_comparison_visualizations(
                comparison_results, mandate, benchmark
            )
            
            # Calculate total duration first
            analysis_results.update({
                'mandate_scenarios_count': len(mandate_scenarios),
                'unique_benchmark_scenarios_count': len(unique_benchmark_scenarios),
                'benchmark_optimization_factor': len(mandate_scenarios) / len(unique_benchmark_scenarios) if unique_benchmark_scenarios else 1,
                'successful_comparisons': comparison_results['successful_comparisons'],
                'comparison_summary': comparison_results['summary_stats'],
                'results': output_results,
                'visualizations': visualization_results,
                'success': True,
                'end_time': datetime.now()
            })
            
            analysis_results['total_duration'] = analysis_results['end_time'] - start_time
            
            # Phase 7: Save analysis documents
            print("\n💾 Phase 7: Saving analysis documents...")
            document_results = self._save_vs_bm_documents(
                mandate, benchmark, output_results, comparison_results, analysis_results, visualization_results
            )
            
            analysis_results['documents'] = document_results
            
            print(f"\n✅ Mandate vs Benchmark Analysis Complete!")
            print(f"   Duration: {analysis_results['total_duration']}")
            print(f"   Successful comparisons: {comparison_results['successful_comparisons']}")
            print(f"   Benchmark optimization: {analysis_results['benchmark_optimization_factor']:.1f}x speedup")
            
            return analysis_results
            
        except Exception as e:
            print(f"\n❌ Analysis failed: {str(e)}")
            analysis_results.update({
                'success': False,
                'error': str(e),
                'end_time': datetime.now()
            })
            return analysis_results
    
    def _generate_benchmark_scenarios(self, 
                                    mandate_scenarios: List[SimulationConfig],
                                    benchmark: str) -> Tuple[List[SimulationConfig], Dict[int, str]]:
        """
        Generate unique benchmark scenarios with caching optimization
        
        Returns:
            - List of unique benchmark scenarios (much smaller than mandate scenarios)
            - Mapping from mandate scenario index to benchmark scenario key
        """
        
        unique_scenarios = {}
        scenario_mapping = {}
        
        print(f"   Optimizing benchmark scenarios...")
        
        for i, mandate_scenario in enumerate(mandate_scenarios):
            # Create unique key based on period (ignore varying parameters like deviation, rebalancing, costs)
            period_key = f"{mandate_scenario.start_date}_{mandate_scenario.end_date}_{mandate_scenario.investment_horizon_years}Y"
            
            # Only create benchmark scenario once per unique period
            if period_key not in unique_scenarios:
                benchmark_scenario = SimulationConfig(
                    mandate=benchmark,
                    start_date=mandate_scenario.start_date,
                    end_date=mandate_scenario.end_date,
                    investment_horizon_years=mandate_scenario.investment_horizon_years,
                    permitted_deviation=0.0,  # No deviations - pure benchmark weights
                    rebalancing_frequency='daily',  # Daily rebalancing (no drift)
                    transaction_cost_bps=0,  # No transaction costs
                    scenario_id=f"{benchmark}_benchmark_{period_key}"
                )
                unique_scenarios[period_key] = benchmark_scenario
            
            # Map mandate scenario index to benchmark period key
            scenario_mapping[i] = period_key
        
        unique_benchmark_scenarios = list(unique_scenarios.values())
        
        redundancy_factor = len(mandate_scenarios) / len(unique_benchmark_scenarios)
        print(f"   Benchmark optimization: {len(mandate_scenarios)} -> {len(unique_benchmark_scenarios)} scenarios ({redundancy_factor:.1f}x reduction)")
        
        return unique_benchmark_scenarios, scenario_mapping
    
    def _expand_benchmark_results(self,
                                unique_benchmark_results: List[Dict[str, Any]],
                                benchmark_mapping: Dict[int, str]) -> List[Dict[str, Any]]:
        """
        Expand unique benchmark results to match each mandate scenario
        
        Args:
            unique_benchmark_results: Results from simulating unique benchmark scenarios
            benchmark_mapping: Maps mandate scenario index -> benchmark period key
            
        Returns:
            List of benchmark results matching each mandate scenario
        """
        
        # Create lookup from period key to benchmark result
        result_lookup = {}
        for result in unique_benchmark_results:
            if result.get('success') and result.get('config'):
                config = result['config']
                period_key = f"{config.start_date}_{config.end_date}_{config.investment_horizon_years}Y"
                result_lookup[period_key] = result
        
        # Expand results to match mandate scenarios
        expanded_results = []
        for mandate_index in sorted(benchmark_mapping.keys()):
            period_key = benchmark_mapping[mandate_index]
            
            if period_key in result_lookup:
                # Use the cached benchmark result
                expanded_results.append(result_lookup[period_key])
            else:
                # Handle missing results (failed benchmark simulations)
                expanded_results.append({
                    'success': False,
                    'error': f'Benchmark simulation failed for period {period_key}',
                    'config': None,
                    'simulation_results': None
                })
        
        return expanded_results
    
    def _perform_scenario_comparison(self,
                                   mandate_results: List[Dict[str, Any]],
                                   benchmark_results: List[Dict[str, Any]],
                                   include_advanced_stats: bool) -> Dict[str, Any]:
        """
        Perform detailed scenario-by-scenario comparison analysis
        """
        
        individual_comparisons = []
        summary_stats = {
            'total_scenarios': len(mandate_results),
            'successful_comparisons': 0,
            'mandate_wins': 0,
            'benchmark_wins': 0,
            'ties': 0
        }
        
        # Risk-free rate for alpha calculations (2% annually)
        risk_free_rate = 0.02
        
        for i, (mandate_result, benchmark_result) in enumerate(zip(mandate_results, benchmark_results)):
            
            # Skip failed scenarios
            if not mandate_result.get('success', False) or not benchmark_result.get('success', False):
                continue
            
            mandate_sim = mandate_result['simulation_results']
            benchmark_sim = benchmark_result['simulation_results']
            mandate_config = mandate_result['config']
            
            # Basic comparison metrics
            excess_return = mandate_sim['annualized_return'] - benchmark_sim['annualized_return']
            
            # Calculate tracking error (need daily returns for this)
            tracking_error = self._calculate_tracking_error(mandate_result, benchmark_result)
            
            # Max drawdown difference
            drawdown_difference = mandate_sim['max_drawdown'] - benchmark_sim['max_drawdown']
            
            # Determine winner
            if excess_return > 0.001:  # 0.1% threshold
                winner = 'mandate'
                summary_stats['mandate_wins'] += 1
            elif excess_return < -0.001:
                winner = 'benchmark'
                summary_stats['benchmark_wins'] += 1
            else:
                winner = 'tie'
                summary_stats['ties'] += 1
            
            # Basic comparison result
            comparison = {
                'scenario_id': mandate_config.scenario_id,
                'mandate_config': {
                    'start_date': mandate_config.start_date,
                    'end_date': mandate_config.end_date,
                    'horizon_years': mandate_config.investment_horizon_years,
                    'deviation_limit': mandate_config.permitted_deviation,
                    'rebalancing_freq': mandate_config.rebalancing_frequency,
                    'transaction_cost_bps': mandate_config.transaction_cost_bps
                },
                'mandate_performance': {
                    'total_return': mandate_sim['total_return'],
                    'annualized_return': mandate_sim['annualized_return'],
                    'volatility': mandate_sim['volatility'],
                    'max_drawdown': mandate_sim['max_drawdown'],
                    'sharpe_ratio': mandate_sim['sharpe_ratio']
                },
                'benchmark_performance': {
                    'total_return': benchmark_sim['total_return'],
                    'annualized_return': benchmark_sim['annualized_return'],
                    'volatility': benchmark_sim['volatility'],
                    'max_drawdown': benchmark_sim['max_drawdown'],
                    'sharpe_ratio': benchmark_sim['sharpe_ratio']
                },
                'asset_class_analysis': {
                    'mandate_weights': {
                        'neutral_weights': mandate_result.get('neutral_weights', {}),
                        'perfect_weights': mandate_result.get('perfect_weights', {}),
                        'weight_deviations': {asset: mandate_result.get('perfect_weights', {}).get(asset, 0) - 
                                            mandate_result.get('neutral_weights', {}).get(asset, 0)
                                            for asset in mandate_result.get('neutral_weights', {}).keys()}
                    },
                    'benchmark_weights': {
                        'neutral_weights': benchmark_result.get('neutral_weights', {}),
                        'perfect_weights': benchmark_result.get('perfect_weights', {}),
                        'weight_deviations': {asset: benchmark_result.get('perfect_weights', {}).get(asset, 0) - 
                                            benchmark_result.get('neutral_weights', {}).get(asset, 0)
                                            for asset in benchmark_result.get('neutral_weights', {}).keys()}
                    },
                    'asset_returns': {
                        'mandate_asset_performance': mandate_result.get('asset_performance', {}),
                        'benchmark_asset_performance': benchmark_result.get('asset_performance', {}),
                        'asset_return_differences': {asset: mandate_result.get('asset_performance', {}).get(asset, 0) - 
                                                   benchmark_result.get('asset_performance', {}).get(asset, 0)
                                                   for asset in mandate_result.get('asset_performance', {}).keys()}
                    }
                },
                'relative_metrics': {
                    'excess_return': excess_return,
                    'tracking_error': tracking_error,
                    'drawdown_difference': drawdown_difference,
                    'sharpe_difference': mandate_sim['sharpe_ratio'] - benchmark_sim['sharpe_ratio'],
                    'winner': winner
                }
            }
            
            # Add advanced statistics if requested
            if include_advanced_stats:
                advanced_stats = self._calculate_advanced_stats(
                    mandate_result, benchmark_result, risk_free_rate
                )
                comparison['advanced_stats'] = advanced_stats
            
            individual_comparisons.append(comparison)
            summary_stats['successful_comparisons'] += 1
        
        return {
            'individual_comparisons': individual_comparisons,
            'summary_stats': summary_stats,
            'successful_comparisons': summary_stats['successful_comparisons']
        }
    
    def _calculate_tracking_error(self, 
                                mandate_result: Dict[str, Any], 
                                benchmark_result: Dict[str, Any]) -> float:
        """
        Calculate tracking error between mandate and benchmark
        
        Note: This is a simplified calculation. For full accuracy, we would need 
        daily return series from both portfolios.
        """
        
        # Simplified tracking error estimate based on volatility difference
        mandate_vol = mandate_result['simulation_results']['volatility']
        benchmark_vol = benchmark_result['simulation_results']['volatility']
        
        # Correlation assumption (could be improved with actual return series)
        correlation_estimate = 0.85  # Reasonable estimate for related portfolios
        
        # Tracking error formula: sqrt(vol_p^2 + vol_b^2 - 2*corr*vol_p*vol_b)
        tracking_error = np.sqrt(
            mandate_vol**2 + benchmark_vol**2 - 2 * correlation_estimate * mandate_vol * benchmark_vol
        )
        
        return tracking_error
    
    def _calculate_advanced_stats(self,
                                mandate_result: Dict[str, Any],
                                benchmark_result: Dict[str, Any],
                                risk_free_rate: float) -> Dict[str, Any]:
        """
        Calculate advanced statistics: Information Ratio, Alpha, Beta
        """
        
        mandate_sim = mandate_result['simulation_results']
        benchmark_sim = benchmark_result['simulation_results']
        
        excess_return = mandate_sim['annualized_return'] - benchmark_sim['annualized_return']
        tracking_error = self._calculate_tracking_error(mandate_result, benchmark_result)
        
        # Information Ratio
        information_ratio = excess_return / tracking_error if tracking_error > 0 else 0
        
        # Beta calculation (simplified - using volatility ratio as proxy)
        beta = mandate_sim['volatility'] / benchmark_sim['volatility'] if benchmark_sim['volatility'] > 0 else 1.0
        
        # Alpha calculation using CAPM
        expected_return = risk_free_rate + beta * (benchmark_sim['annualized_return'] - risk_free_rate)
        alpha = mandate_sim['annualized_return'] - expected_return
        
        # Up/Down capture ratios (simplified estimates)
        up_capture = mandate_sim['annualized_return'] / benchmark_sim['annualized_return'] if benchmark_sim['annualized_return'] > 0 else 1.0
        down_capture = mandate_sim['max_drawdown'] / benchmark_sim['max_drawdown'] if benchmark_sim['max_drawdown'] > 0 else 1.0
        
        return {
            'information_ratio': information_ratio,
            'alpha': alpha,
            'beta': beta,
            'up_capture_ratio': up_capture,
            'down_capture_ratio': down_capture,
            'correlation_estimate': 0.85  # Static estimate - could be improved
        }
    
    def _generate_aggregated_analysis(self,
                                    comparison_results: Dict[str, Any],
                                    mandate: str,
                                    benchmark: str,
                                    include_advanced_stats: bool) -> Dict[str, Any]:
        """
        Generate aggregated statistical analysis across all comparisons
        """
        
        comparisons = comparison_results['individual_comparisons']
        summary_stats = comparison_results['summary_stats']
        
        if not comparisons:
            return {'error': 'No successful comparisons for aggregation'}
        
        # Extract metrics for statistical analysis
        excess_returns = [comp['relative_metrics']['excess_return'] for comp in comparisons]
        tracking_errors = [comp['relative_metrics']['tracking_error'] for comp in comparisons]
        drawdown_diffs = [comp['relative_metrics']['drawdown_difference'] for comp in comparisons]
        sharpe_diffs = [comp['relative_metrics']['sharpe_difference'] for comp in comparisons]
        
        # Calculate percentiles
        def calculate_percentiles(data):
            return {
                'p10': np.percentile(data, 10),
                'p25': np.percentile(data, 25),
                'p50': np.percentile(data, 50),  # median
                'p75': np.percentile(data, 75),
                'p90': np.percentile(data, 90)
            }
        
        # Win rate calculation
        total_comparisons = summary_stats['successful_comparisons']
        win_rate = summary_stats['mandate_wins'] / total_comparisons if total_comparisons > 0 else 0
        
        aggregated_analysis = {
            'mandate': mandate,
            'benchmark': benchmark,
            'summary': {
                'total_scenarios_compared': total_comparisons,
                'mandate_win_rate': win_rate,
                'benchmark_win_rate': summary_stats['benchmark_wins'] / total_comparisons if total_comparisons > 0 else 0,
                'tie_rate': summary_stats['ties'] / total_comparisons if total_comparisons > 0 else 0
            },
            
            'excess_return_analysis': {
                'mean': np.mean(excess_returns),
                'median': np.median(excess_returns),
                'std_dev': np.std(excess_returns),
                'min': np.min(excess_returns),
                'max': np.max(excess_returns),
                'percentiles': calculate_percentiles(excess_returns),
                'positive_excess_rate': len([x for x in excess_returns if x > 0]) / len(excess_returns),
                'statistically_significant': abs(np.mean(excess_returns)) / np.std(excess_returns) > 1.96  # t-test approximation
            },
            
            'tracking_error_analysis': {
                'mean': np.mean(tracking_errors),
                'median': np.median(tracking_errors),
                'std_dev': np.std(tracking_errors),
                'min': np.min(tracking_errors),
                'max': np.max(tracking_errors),
                'percentiles': calculate_percentiles(tracking_errors)
            },
            
            'relative_drawdown_analysis': {
                'mean_difference': np.mean(drawdown_diffs),
                'median_difference': np.median(drawdown_diffs),
                'std_dev': np.std(drawdown_diffs),
                'better_drawdown_rate': len([x for x in drawdown_diffs if x < 0]) / len(drawdown_diffs),
                'percentiles': calculate_percentiles(drawdown_diffs)
            },
            
            'risk_return_efficiency': {
                'avg_excess_return': np.mean(excess_returns),
                'avg_tracking_error': np.mean(tracking_errors),
                'information_ratio_portfolio': np.mean(excess_returns) / np.mean(tracking_errors) if np.mean(tracking_errors) > 0 else 0,
                'avg_sharpe_difference': np.mean(sharpe_diffs)
            }
        }
        
        # Add advanced statistics aggregation if included
        if include_advanced_stats:
            info_ratios = [comp.get('advanced_stats', {}).get('information_ratio', 0) for comp in comparisons if comp.get('advanced_stats')]
            alphas = [comp.get('advanced_stats', {}).get('alpha', 0) for comp in comparisons if comp.get('advanced_stats')]
            betas = [comp.get('advanced_stats', {}).get('beta', 1) for comp in comparisons if comp.get('advanced_stats')]
            
            if info_ratios:  # Only add if we have advanced stats
                aggregated_analysis['advanced_statistics'] = {
                    'information_ratios': {
                        'mean': np.mean(info_ratios),
                        'median': np.median(info_ratios),
                        'percentiles': calculate_percentiles(info_ratios)
                    },
                    'alpha_analysis': {
                        'mean_alpha': np.mean(alphas),
                        'median_alpha': np.median(alphas),
                        'positive_alpha_rate': len([x for x in alphas if x > 0]) / len(alphas),
                        'percentiles': calculate_percentiles(alphas)
                    },
                    'beta_analysis': {
                        'mean_beta': np.mean(betas),
                        'median_beta': np.median(betas),
                        'percentiles': calculate_percentiles(betas)
                    }
                }
        
        # Parameter sensitivity analysis
        aggregated_analysis['parameter_sensitivity'] = self._analyze_parameter_sensitivity(comparisons)
        
        return aggregated_analysis
    
    def _analyze_parameter_sensitivity(self, comparisons: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze which parameters drive outperformance
        """
        
        # Group by parameters
        param_groups = {}
        
        for comp in comparisons:
            config = comp['mandate_config']
            
            # Group by key parameters
            for param_name, param_value in [
                ('deviation_limit', config['deviation_limit']),
                ('rebalancing_freq', config['rebalancing_freq']),
                ('transaction_cost_bps', config['transaction_cost_bps']),
                ('horizon_years', config['horizon_years'])
            ]:
                if param_name not in param_groups:
                    param_groups[param_name] = {}
                
                if param_value not in param_groups[param_name]:
                    param_groups[param_name][param_value] = []
                
                param_groups[param_name][param_value].append(comp['relative_metrics']['excess_return'])
        
        # Calculate statistics for each parameter group
        sensitivity_analysis = {}
        
        for param_name, param_values in param_groups.items():
            param_stats = {}
            
            for value, excess_returns in param_values.items():
                if excess_returns:
                    param_stats[str(value)] = {
                        'avg_excess_return': np.mean(excess_returns),
                        'win_rate': len([x for x in excess_returns if x > 0]) / len(excess_returns),
                        'count': len(excess_returns)
                    }
            
            sensitivity_analysis[param_name] = param_stats
        
        return sensitivity_analysis
    
    def _create_comparison_visualizations(self,
                                        comparison_results: Dict[str, Any],
                                        mandate: str,
                                        benchmark: str) -> Dict[str, Any]:
        """
        Create visualization components for the comparison analysis
        """
        
        try:
            from interactive_dashboard import InteractiveDashboard
            dashboard = InteractiveDashboard()
            
            comparisons = comparison_results.get('individual_comparisons', [])
            
            if not comparisons:
                return {
                    'error': 'No individual comparisons available for visualization. Use output_mode="individual" or "both" to generate charts.',
                    'success': False
                }
            
            # Prepare data for visualization
            import pandas as pd
            
            vis_data = []
            for comp in comparisons:
                vis_data.append({
                    'scenario_id': comp['scenario_id'],
                    'excess_return': comp['relative_metrics']['excess_return'],
                    'tracking_error': comp['relative_metrics']['tracking_error'],
                    'drawdown_difference': comp['relative_metrics']['drawdown_difference'],
                    'deviation_limit': comp['mandate_config']['deviation_limit'],
                    'rebalancing_freq': comp['mandate_config']['rebalancing_freq'],
                    'transaction_cost_bps': comp['mandate_config']['transaction_cost_bps'],
                    'horizon_years': comp['mandate_config']['horizon_years'],
                    'winner': comp['relative_metrics']['winner']
                })
            
            df = pd.DataFrame(vis_data)
            
            # Create and save actual visualization files
            import matplotlib.pyplot as plt
            import seaborn as sns
            from pathlib import Path
            
            # Create charts directory
            charts_dir = Path("reports") / "vs_bm" / "charts"
            charts_dir.mkdir(parents=True, exist_ok=True)
            
            saved_charts = []
            
            # 1. Excess Return Distribution
            plt.figure(figsize=(10, 6))
            plt.hist(df['excess_return'], bins=20, alpha=0.7, edgecolor='black')
            plt.title(f'{mandate} vs {benchmark}: Excess Return Distribution')
            plt.xlabel('Excess Return')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            hist_file = charts_dir / f"{mandate}_vs_{benchmark}_excess_return_distribution.png"
            plt.savefig(hist_file, dpi=300, bbox_inches='tight')
            plt.close()
            saved_charts.append(str(hist_file))
            
            # 2. Risk-Return Scatter Plot
            plt.figure(figsize=(10, 8))
            colors = ['green' if w == 'mandate' else 'red' for w in df['winner']]
            plt.scatter(df['tracking_error'], df['excess_return'], c=colors, alpha=0.6, s=60)
            plt.title(f'{mandate} vs {benchmark}: Risk-Return Trade-off')
            plt.xlabel('Tracking Error')
            plt.ylabel('Excess Return')
            plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            plt.grid(True, alpha=0.3)
            
            # Add legend
            import matplotlib.patches as mpatches
            green_patch = mpatches.Patch(color='green', label='Mandate Wins')
            red_patch = mpatches.Patch(color='red', label='Benchmark Wins')
            plt.legend(handles=[green_patch, red_patch])
            
            scatter_file = charts_dir / f"{mandate}_vs_{benchmark}_risk_return_scatter.png"
            plt.savefig(scatter_file, dpi=300, bbox_inches='tight')
            plt.close()
            saved_charts.append(str(scatter_file))
            
            # 3. Win Rate by Parameter  
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Deviation Limit
            dev_win_rates = df.groupby('deviation_limit')['winner'].apply(lambda x: (x == 'mandate').mean())
            axes[0,0].bar(dev_win_rates.index, dev_win_rates.values, color='steelblue', alpha=0.7)
            axes[0,0].set_title('Win Rate by Deviation Limit')
            axes[0,0].set_xlabel('Deviation Limit')
            axes[0,0].set_ylabel('Mandate Win Rate')
            axes[0,0].set_ylim(0, 1)
            axes[0,0].grid(True, alpha=0.3)
            
            # Rebalancing Frequency
            freq_win_rates = df.groupby('rebalancing_freq')['winner'].apply(lambda x: (x == 'mandate').mean())
            axes[0,1].bar(range(len(freq_win_rates)), freq_win_rates.values, color='forestgreen', alpha=0.7)
            axes[0,1].set_xticks(range(len(freq_win_rates)))
            axes[0,1].set_xticklabels(freq_win_rates.index, rotation=45)
            axes[0,1].set_title('Win Rate by Rebalancing Frequency')
            axes[0,1].set_xlabel('Rebalancing Frequency')
            axes[0,1].set_ylabel('Mandate Win Rate')
            axes[0,1].set_ylim(0, 1)
            axes[0,1].grid(True, alpha=0.3)
            
            # Transaction Cost
            cost_win_rates = df.groupby('transaction_cost_bps')['winner'].apply(lambda x: (x == 'mandate').mean())
            axes[1,0].bar(cost_win_rates.index, cost_win_rates.values, color='darkorange', alpha=0.7)
            axes[1,0].set_title('Win Rate by Transaction Cost (bps)')
            axes[1,0].set_xlabel('Transaction Cost (bps)')
            axes[1,0].set_ylabel('Mandate Win Rate')
            axes[1,0].set_ylim(0, 1)
            axes[1,0].grid(True, alpha=0.3)
            
            # Investment Horizon
            horizon_win_rates = df.groupby('horizon_years')['winner'].apply(lambda x: (x == 'mandate').mean())
            axes[1,1].bar(horizon_win_rates.index, horizon_win_rates.values, color='purple', alpha=0.7)
            axes[1,1].set_title('Win Rate by Investment Horizon')
            axes[1,1].set_xlabel('Investment Horizon (Years)')
            axes[1,1].set_ylabel('Mandate Win Rate')
            axes[1,1].set_ylim(0, 1)
            axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            params_file = charts_dir / f"{mandate}_vs_{benchmark}_parameter_analysis.png"
            plt.savefig(params_file, dpi=300, bbox_inches='tight')
            plt.close()
            saved_charts.append(str(params_file))
            
            # 4. Parameter-specific scatter plots (Excess Return vs Tracking Error)
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Create color maps for each parameter
            import matplotlib.cm as cm
            import numpy as np
            
            # Deviation Limit scatter
            deviation_values = df['deviation_limit'].unique()
            colors_dev = cm.viridis(np.linspace(0, 1, len(deviation_values)))
            color_map_dev = {val: colors_dev[i] for i, val in enumerate(deviation_values)}
            
            for deviation in deviation_values:
                subset = df[df['deviation_limit'] == deviation]
                if not subset.empty:
                    axes[0,0].scatter(subset['tracking_error'], subset['excess_return'], 
                                    c=[color_map_dev[deviation]], alpha=0.7, s=60, 
                                    label=f'{deviation:.1%}')
            axes[0,0].set_title('Risk-Return by Deviation Limit')
            axes[0,0].set_xlabel('Tracking Error')
            axes[0,0].set_ylabel('Excess Return')
            axes[0,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[0,0].legend(title='Deviation Limit', bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[0,0].grid(True, alpha=0.3)
            
            # Rebalancing Frequency scatter
            freq_values = df['rebalancing_freq'].unique()
            colors_freq = cm.plasma(np.linspace(0, 1, len(freq_values)))
            color_map_freq = {val: colors_freq[i] for i, val in enumerate(freq_values)}
            
            for freq in freq_values:
                subset = df[df['rebalancing_freq'] == freq]
                if not subset.empty:
                    axes[0,1].scatter(subset['tracking_error'], subset['excess_return'], 
                                    c=[color_map_freq[freq]], alpha=0.7, s=60, 
                                    label=freq)
            axes[0,1].set_title('Risk-Return by Rebalancing Frequency')
            axes[0,1].set_xlabel('Tracking Error')
            axes[0,1].set_ylabel('Excess Return')
            axes[0,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[0,1].legend(title='Rebalancing', bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[0,1].grid(True, alpha=0.3)
            
            # Transaction Cost scatter
            cost_values = df['transaction_cost_bps'].unique()
            colors_cost = cm.coolwarm(np.linspace(0, 1, len(cost_values)))
            color_map_cost = {val: colors_cost[i] for i, val in enumerate(cost_values)}
            
            for cost in cost_values:
                subset = df[df['transaction_cost_bps'] == cost]
                if not subset.empty:
                    axes[1,0].scatter(subset['tracking_error'], subset['excess_return'], 
                                    c=[color_map_cost[cost]], alpha=0.7, s=60, 
                                    label=f'{cost}bps')
            axes[1,0].set_title('Risk-Return by Transaction Cost')
            axes[1,0].set_xlabel('Tracking Error')
            axes[1,0].set_ylabel('Excess Return')
            axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1,0].legend(title='Transaction Cost', bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1,0].grid(True, alpha=0.3)
            
            # Investment Horizon scatter
            horizon_values = df['horizon_years'].unique()
            colors_horizon = cm.Set1(np.linspace(0, 1, len(horizon_values)))
            color_map_horizon = {val: colors_horizon[i] for i, val in enumerate(horizon_values)}
            
            for horizon in horizon_values:
                subset = df[df['horizon_years'] == horizon]
                if not subset.empty:
                    axes[1,1].scatter(subset['tracking_error'], subset['excess_return'], 
                                    c=[color_map_horizon[horizon]], alpha=0.7, s=60, 
                                    label=f'{horizon}Y')
            axes[1,1].set_title('Risk-Return by Investment Horizon')
            axes[1,1].set_xlabel('Tracking Error')
            axes[1,1].set_ylabel('Excess Return')
            axes[1,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1,1].legend(title='Investment Horizon', bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            param_scatter_file = charts_dir / f"{mandate}_vs_{benchmark}_parameter_scatter_analysis.png"
            plt.savefig(param_scatter_file, dpi=300, bbox_inches='tight')
            plt.close()
            saved_charts.append(str(param_scatter_file))
            
            print(f"   📊 Visualization files saved:")
            for chart in saved_charts:
                print(f"      - {Path(chart).name}")
            
            return {
                'visualization_data': df.to_dict('records'),
                'saved_charts': saved_charts,
                'charts_directory': str(charts_dir),
                'success': True
            }
            
        except Exception as e:
            return {
                'error': f'Visualization creation failed: {str(e)}',
                'success': False
            }
    
    def _calculate_win_rates_by_parameter(self, df) -> Dict[str, Any]:
        """Calculate win rates grouped by parameters"""
        
        win_rates = {}
        
        for param in ['deviation_limit', 'rebalancing_freq', 'transaction_cost_bps']:
            param_win_rates = {}
            
            for value in df[param].unique():
                subset = df[df[param] == value]
                win_rate = len(subset[subset['winner'] == 'mandate']) / len(subset) if len(subset) > 0 else 0
                param_win_rates[str(value)] = {
                    'win_rate': win_rate,
                    'count': len(subset)
                }
            
            win_rates[param] = param_win_rates
        
        return win_rates
    
    def _save_vs_bm_documents(self,
                             mandate: str,
                             benchmark: str,
                             output_results: Dict[str, Any],
                             comparison_results: Dict[str, Any],
                             analysis_results: Dict[str, Any],
                             visualization_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Save vs_bm analysis documents to /reports/vs_bm directory
        """
        
        import json
        import os
        from pathlib import Path
        
        try:
            # Create vs_bm reports directory
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            vs_bm_dir = Path("reports") / "vs_bm" / f"{mandate}_vs_{benchmark}_{timestamp}"
            vs_bm_dir.mkdir(parents=True, exist_ok=True)
            
            saved_files = []
            
            # 1. Save comprehensive JSON results
            json_file = vs_bm_dir / f"{mandate}_vs_{benchmark}_complete_analysis.json"
            
            # Prepare JSON-serializable data
            json_data = {
                'metadata': {
                    'mandate': mandate,
                    'benchmark': benchmark,
                    'analysis_period': analysis_results['analysis_period'],
                    'generated_timestamp': timestamp,
                    'total_duration': str(analysis_results['total_duration']),
                    'optimization_factor': analysis_results.get('benchmark_optimization_factor', 1)
                },
                'summary': {
                    'mandate_scenarios_count': analysis_results['mandate_scenarios_count'],
                    'unique_benchmark_scenarios_count': analysis_results['unique_benchmark_scenarios_count'],
                    'successful_comparisons': analysis_results['successful_comparisons'],
                    'comparison_summary': comparison_results['summary_stats']
                },
                'results': output_results
            }
            
            with open(json_file, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            saved_files.append(str(json_file))
            
            # 2. Save aggregated analysis summary (if available)
            if 'aggregated_analysis' in output_results:
                agg_analysis = output_results['aggregated_analysis']
                
                if 'error' not in agg_analysis:
                    summary_file = vs_bm_dir / f"{mandate}_vs_{benchmark}_executive_summary.txt"
                    
                    with open(summary_file, 'w') as f:
                        f.write(f"MANDATE vs BENCHMARK ANALYSIS SUMMARY\n")
                        f.write(f"{'=' * 50}\n\n")
                        
                        f.write(f"Analysis Details:\n")
                        f.write(f"   Mandate: {mandate}\n")
                        f.write(f"   Benchmark: {benchmark}\n")
                        f.write(f"   Analysis Period: {analysis_results['analysis_period']}\n")
                        f.write(f"   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                        
                        summary = agg_analysis['summary']
                        f.write(f"Performance Summary:\n")
                        f.write(f"   Scenarios Compared: {summary['total_scenarios_compared']:,}\n")
                        f.write(f"   Mandate Win Rate: {summary['mandate_win_rate']:.1%}\n")
                        f.write(f"   Benchmark Win Rate: {summary['benchmark_win_rate']:.1%}\n")
                        f.write(f"   Tie Rate: {summary['tie_rate']:.1%}\n\n")
                        
                        excess = agg_analysis['excess_return_analysis']
                        f.write(f"Excess Return Analysis:\n")
                        f.write(f"   Mean Excess Return: {excess['mean']:.3%}\n")
                        f.write(f"   Median Excess Return: {excess['median']:.3%}\n")
                        f.write(f"   Excess Return Volatility: {excess['std_dev']:.3%}\n")
                        f.write(f"   Positive Excess Rate: {excess['positive_excess_rate']:.1%}\n")
                        f.write(f"   Statistical Significance: {'Yes' if excess.get('statistically_significant', False) else 'No'}\n\n")
                        
                        tracking = agg_analysis['tracking_error_analysis']
                        f.write(f"Tracking Error Analysis:\n")
                        f.write(f"   Mean Tracking Error: {tracking['mean']:.3%}\n")
                        f.write(f"   Median Tracking Error: {tracking['median']:.3%}\n\n")
                        
                        dd = agg_analysis['relative_drawdown_analysis']
                        f.write(f"Relative Drawdown Analysis:\n")
                        f.write(f"   Mean Drawdown Difference: {dd['mean_difference']:.3%}\n")
                        f.write(f"   Better Drawdown Rate: {dd['better_drawdown_rate']:.1%}\n\n")
                        
                        efficiency = agg_analysis['risk_return_efficiency']
                        f.write(f"Risk-Return Efficiency:\n")
                        f.write(f"   Portfolio Information Ratio: {efficiency['information_ratio_portfolio']:.3f}\n")
                        f.write(f"   Average Sharpe Difference: {efficiency['avg_sharpe_difference']:.3f}\n\n")
                        
                        # Parameter sensitivity
                        if 'parameter_sensitivity' in agg_analysis:
                            f.write(f"Parameter Sensitivity (Top Performers):\n")
                            param_sens = agg_analysis['parameter_sensitivity']
                            
                            for param_name, param_data in param_sens.items():
                                if param_data:
                                    f.write(f"   {param_name}:\n")
                                    sorted_params = sorted(param_data.items(), 
                                                         key=lambda x: x[1]['avg_excess_return'], reverse=True)
                                    for value, stats in sorted_params[:3]:  # Top 3
                                        f.write(f"      {value}: {stats['avg_excess_return']:.3%} excess, {stats['win_rate']:.1%} win rate\n")
                                    f.write("\n")
                    
                    saved_files.append(str(summary_file))
            
            # 3. Save individual comparisons (if available and not too large)
            if 'individual_comparisons' in output_results:
                individual_comps = output_results['individual_comparisons']
                
                if len(individual_comps) <= 1000:  # Only save if reasonable size
                    individual_file = vs_bm_dir / f"{mandate}_vs_{benchmark}_individual_comparisons.json"
                    
                    with open(individual_file, 'w') as f:
                        json.dump(individual_comps, f, indent=2, default=str)
                    saved_files.append(str(individual_file))
                else:
                    # Save a sample if too large
                    sample_file = vs_bm_dir / f"{mandate}_vs_{benchmark}_sample_comparisons.json"
                    sample_data = {
                        'note': f'Sample of {len(individual_comps)} total comparisons',
                        'sample_size': 100,
                        'sample_comparisons': individual_comps[:100]
                    }
                    
                    with open(sample_file, 'w') as f:
                        json.dump(sample_data, f, indent=2, default=str)
                    saved_files.append(str(sample_file))
            
            # 4. Create analysis metadata file
            metadata_file = vs_bm_dir / "analysis_metadata.txt"
            
            with open(metadata_file, 'w') as f:
                f.write(f"vs_bm ANALYSIS METADATA\n")
                f.write(f"{'=' * 30}\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Mandate: {mandate}\n")
                f.write(f"Benchmark: {benchmark}\n")
                f.write(f"Analysis Period: {analysis_results['analysis_period']}\n")
                f.write(f"Duration: {analysis_results['total_duration']}\n")
                f.write(f"Optimization Factor: {analysis_results.get('benchmark_optimization_factor', 1):.1f}x\n")
                f.write(f"Output Mode: {analysis_results.get('output_mode', 'unknown')}\n\n")
                
                f.write(f"Document Files:\n")
                for file_path in saved_files:
                    f.write(f"   - {Path(file_path).name}\n")
                
                # Add visualization information
                if visualization_results and visualization_results.get('success') and 'saved_charts' in visualization_results:
                    f.write(f"\nVisualization Files:\n")
                    for chart_path in visualization_results['saved_charts']:
                        f.write(f"   - {Path(chart_path).name}\n")
                    f.write(f"\nCharts Location: {visualization_results.get('charts_directory', 'N/A')}\n")
            
            saved_files.append(str(metadata_file))
            
            # Print confirmation
            print(f"   📁 Documents saved to: {vs_bm_dir}")
            print(f"   📄 Files created: {len(saved_files)}")
            
            return {
                'success': True,
                'output_directory': str(vs_bm_dir),
                'files_created': saved_files,
                'file_count': len(saved_files)
            }
            
        except Exception as e:
            print(f"   ⚠️ Document saving warning: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'files_created': []
            }

def main():
    """Command-line interface for PHP Master Orchestrator"""
    
    parser = argparse.ArgumentParser(description='PHP Master Orchestrator')
    parser.add_argument('--mandate', type=str,
                       choices=['6040', 'EQFI', 'EQFILA', 'EQFILAIA'],
                       help='Portfolio mandate to analyze (not required if using --compare-mandates)')
    parser.add_argument('--start', type=str, default='2010-01-01',
                       help='Analysis start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2020-01-01',
                       help='Analysis end date (YYYY-MM-DD)')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Limit number of scenarios (default: all)')
    parser.add_argument('--output-formats', type=str, nargs='+',
                       default=['json', 'html'], 
                       help='Output formats')
    parser.add_argument('--skip-dashboard', action='store_true',
                       help='Skip dashboard generation')
    parser.add_argument('--skip-executive', action='store_true',
                       help='Skip executive reports')
    parser.add_argument('--compare-mandates', type=str, nargs='+',
                       help='Run comparison across multiple mandates')
    parser.add_argument('--vs-bm', type=str,
                       help='Run detailed mandate vs 6040 benchmark comparison (specify mandate: EQFI, EQFILA, EQFILAIA)')
    parser.add_argument('--output-mode', type=str, default='both',
                       choices=['individual', 'aggregated', 'both'],
                       help='Output mode for vs-bm analysis (default: both)')
    parser.add_argument('--skip-advanced-stats', action='store_true',
                       help='Skip advanced statistics in vs-bm analysis')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.compare_mandates and not args.mandate and not args.vs_bm:
        parser.error("Either --mandate, --compare-mandates, or --vs-bm must be specified")
    
    # Initialize orchestrator
    orchestrator = PHPMasterOrchestrator()
    
    if args.compare_mandates:
        # Run mandate comparison
        results = orchestrator.run_mandate_comparison(
            args.compare_mandates,
            args.start,
            args.end, 
            args.sample_size or 50
        )
    elif args.vs_bm:
        # Run mandate vs benchmark analysis
        results = orchestrator.vs_bm(
            mandate=args.vs_bm,
            analysis_start=args.start,
            analysis_end=args.end,
            sample_size=args.sample_size,
            output_mode=args.output_mode,
            include_advanced_stats=not args.skip_advanced_stats
        )
    else:
        # Run single mandate analysis
        config = PHPAnalysisConfig(
            mandate=args.mandate,
            analysis_start=args.start,
            analysis_end=args.end,
            sample_size=args.sample_size,
            output_formats=args.output_formats,
            generate_dashboard=not args.skip_dashboard,
            generate_executive_reports=not args.skip_executive
        )
        
        results = orchestrator.run_complete_php_analysis(config)
    
    # Print final results
    if results.get('success'):
        print(f"\n🎉 Analysis completed successfully!")
        if 'exports' in results:
            print(f"📁 Results saved to: {results['exports'].get('final_directory', 'reports/')}")
    else:
        print(f"\n💥 Analysis failed: {results.get('error', 'Unknown error')}")
        sys.exit(1)

if __name__ == "__main__":
    # For testing, run a sample analysis
    if len(sys.argv) == 1:
        print("🧪 Testing PHP Master Orchestrator")
        
        orchestrator = PHPMasterOrchestrator()
        
        # Test configuration
        test_config = PHPAnalysisConfig(
            mandate="EQFI",
            analysis_start="2015-01-01",
            analysis_end="2020-01-01",
            sample_size=20,  # Small sample for testing
            output_formats=['json', 'html'],
            generate_dashboard=True,
            generate_executive_reports=True,
            custom_insights=[
                "PHP analysis demonstrates theoretical alpha potential",
                "Parameter optimization critical for performance"
            ],
            custom_recommendations=[
                "Implement dynamic allocation based on PHP insights",
                "Establish regular parameter optimization reviews"
            ]
        )
        
        # Run test analysis
        results = orchestrator.run_complete_php_analysis(test_config)
        
        if results['success']:
            print("\n✅ Test completed successfully!")
        else:
            print(f"\n❌ Test failed: {results.get('error')}")
    else:
        main()