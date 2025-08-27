#!/usr/bin/env python3
"""
Example usage of the vs_bm (mandate vs benchmark) analysis function
"""

from master_orchestrator import PHPMasterOrchestrator
import json
from datetime import datetime

def example_eqfila_vs_6040():
    """Example: Compare EQFILA to 6040 benchmark over 2015-2020"""
    
    print("=" * 60)
    print("EXAMPLE: EQFILA vs 6040 BENCHMARK ANALYSIS")
    print("=" * 60)
    
    # Initialize orchestrator
    orchestrator = PHPMasterOrchestrator()
    
    # Run the analysis
    print("🚀 Running EQFILA vs 6040 analysis...")
    
    results = orchestrator.vs_bm(
        mandate="EQFILA",
        analysis_start="2015-01-01",
        analysis_end="2020-01-01", 
        sample_size=50,  # Limit for demonstration
        output_mode="both",
        include_advanced_stats=True
    )
    
    if results.get('success'):
        print("✅ Analysis completed successfully!")
        print(f"⏱️  Duration: {results['total_duration']}")
        print(f"📊 Successful comparisons: {results['successful_comparisons']}")
        
        # Show aggregated results
        if 'aggregated_analysis' in results['results']:
            agg = results['results']['aggregated_analysis']
            
            if 'error' not in agg:
                print(f"\n📈 KEY FINDINGS:")
                summary = agg['summary']
                print(f"   EQFILA Win Rate: {summary['mandate_win_rate']:.1%}")
                
                excess = agg['excess_return_analysis']
                print(f"   Average Excess Return: {excess['mean']:.2%}")
                print(f"   Excess Return Range: {excess['min']:.2%} to {excess['max']:.2%}")
                print(f"   Positive Excess Rate: {excess['positive_excess_rate']:.1%}")
                
                track = agg['tracking_error_analysis']
                print(f"   Average Tracking Error: {track['mean']:.2%}")
                
                dd = agg['relative_drawdown_analysis']
                print(f"   Better Drawdown Rate: {dd['better_drawdown_rate']:.1%}")
                
                # Advanced stats if available
                if 'advanced_statistics' in agg:
                    adv = agg['advanced_statistics']
                    print(f"   Average Information Ratio: {adv['information_ratios']['mean']:.3f}")
                    print(f"   Average Alpha: {adv['alpha_analysis']['mean_alpha']:.3%}")
                
                # Parameter sensitivity
                if 'parameter_sensitivity' in agg:
                    print(f"\n🎯 PARAMETER INSIGHTS:")
                    param_sens = agg['parameter_sensitivity']
                    
                    if 'deviation_limit' in param_sens:
                        print("   Best Deviation Limits:")
                        dev_data = param_sens['deviation_limit']
                        sorted_devs = sorted(dev_data.items(), 
                                           key=lambda x: x[1]['avg_excess_return'], reverse=True)
                        for dev, stats in sorted_devs[:3]:
                            print(f"      {float(dev):.0%}: {stats['avg_excess_return']:.3%} excess, {stats['win_rate']:.1%} win rate")
            else:
                print(f"⚠️  No comparison data: {agg['error']}")
        
        # Save results to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"reports/EQFILA_vs_6040_analysis_{timestamp}.json"
        
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n💾 Results saved to: {output_file}")
        except Exception as e:
            print(f"⚠️  Could not save results: {e}")
    
    else:
        print(f"❌ Analysis failed: {results.get('error')}")
        return False
    
    return True

def example_command_line_usage():
    """Show command line usage examples"""
    
    print("\n" + "=" * 60)
    print("COMMAND LINE USAGE EXAMPLES")
    print("=" * 60)
    
    print("\n🖥️  Basic vs benchmark analysis:")
    print("python3 master_orchestrator.py --vs-bm EQFILA --start 2015-01-01 --end 2020-01-01")
    
    print("\n🖥️  With sampling and specific output mode:")
    print("python3 master_orchestrator.py --vs-bm EQFI --sample-size 100 --output-mode aggregated")
    
    print("\n🖥️  Individual comparisons only, no advanced stats:")
    print("python3 master_orchestrator.py --vs-bm EQFILAIA --output-mode individual --skip-advanced-stats")
    
    print("\n📚 Available options:")
    print("   --vs-bm: Mandate to compare (EQFI, EQFILA, EQFILAIA)")
    print("   --start: Analysis start date (default: 2010-01-01)")
    print("   --end: Analysis end date (default: 2020-01-01)")
    print("   --sample-size: Limit scenarios for testing")
    print("   --output-mode: individual, aggregated, or both (default: both)")
    print("   --skip-advanced-stats: Skip Information Ratio, Alpha, Beta calculations")

if __name__ == "__main__":
    print("🏁 EQFILA vs 6040 Benchmark Analysis Example")
    print("=" * 60)
    
    success = example_eqfila_vs_6040()
    
    if success:
        example_command_line_usage()
        print("\n🎉 Example completed successfully!")
    else:
        print("\n❌ Example failed")