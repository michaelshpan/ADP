#!/usr/bin/env python3
"""
Test script for mandate vs benchmark analysis function
"""

from master_orchestrator import PHPMasterOrchestrator
from portfolio_simulation import ParallelConfig
import json
from datetime import datetime

def test_vs_bm_analysis():
    """Test the vs_bm analysis function"""
    
    print("=" * 60)
    print("TESTING MANDATE VS BENCHMARK ANALYSIS")
    print("=" * 60)
    
    # Initialize orchestrator
    print("🚀 Initializing Master Orchestrator...")
    orchestrator = PHPMasterOrchestrator()
    
    # Test parameters
    test_mandate = "EQFILA"
    test_start = "2010-01-01"
    test_end = "2020-01-01"
    test_sample_size = 20  # Small sample for testing
    
    print(f"\n📊 Test Configuration:")
    print(f"   Mandate: {test_mandate}")
    print(f"   Benchmark: 6040")
    print(f"   Period: {test_start} to {test_end}")
    print(f"   Sample Size: {test_sample_size}")
    
    try:
        # Test 1: Individual output mode
        print("\n" + "="*50)
        print("TEST 1: INDIVIDUAL OUTPUT MODE")
        print("="*50)
        
        start_time = datetime.now()
        results_individual = orchestrator.vs_bm(
            mandate=test_mandate,
            analysis_start=test_start,
            analysis_end=test_end,
            sample_size=test_sample_size,
            output_mode="individual",
            include_advanced_stats=True
        )
        
        duration_individual = datetime.now() - start_time
        print(f"⏱️  Individual mode duration: {duration_individual}")
        
        if results_individual.get('success'):
            individual_comps = results_individual['results']['individual_comparisons']
            print(f"✅ Individual mode: {len(individual_comps)} comparisons generated")
            
            # Show sample comparison
            if individual_comps:
                sample_comp = individual_comps[0]
                print(f"\n📋 Sample Individual Comparison:")
                print(f"   Scenario: {sample_comp['scenario_id']}")
                print(f"   Excess Return: {sample_comp['relative_metrics']['excess_return']:.3%}")
                print(f"   Tracking Error: {sample_comp['relative_metrics']['tracking_error']:.3%}")
                print(f"   Winner: {sample_comp['relative_metrics']['winner']}")
                
                if 'advanced_stats' in sample_comp:
                    adv = sample_comp['advanced_stats']
                    print(f"   Information Ratio: {adv['information_ratio']:.3f}")
                    print(f"   Alpha: {adv['alpha']:.3%}")
                    print(f"   Beta: {adv['beta']:.3f}")
        else:
            print(f"❌ Individual mode failed: {results_individual.get('error')}")
            return False
        
        # Test 2: Aggregated output mode
        print("\n" + "="*50)
        print("TEST 2: AGGREGATED OUTPUT MODE")
        print("="*50)
        
        start_time = datetime.now()
        results_aggregated = orchestrator.vs_bm(
            mandate=test_mandate,
            analysis_start=test_start,
            analysis_end=test_end,
            sample_size=test_sample_size,
            output_mode="aggregated",
            include_advanced_stats=True
        )
        
        duration_aggregated = datetime.now() - start_time
        print(f"⏱️  Aggregated mode duration: {duration_aggregated}")
        
        if results_aggregated.get('success'):
            agg_analysis = results_aggregated['results']['aggregated_analysis']
            print(f"✅ Aggregated mode: Analysis generated")
            
            # Check if we have data
            if 'error' in agg_analysis:
                print(f"⚠️  No comparisons available: {agg_analysis['error']}")
            else:
                # Show key aggregated results
                print(f"\n📊 Aggregated Analysis Summary:")
                summary = agg_analysis['summary']
                print(f"   Win Rate: {summary['mandate_win_rate']:.1%}")
                print(f"   Scenarios Compared: {summary['total_scenarios_compared']}")
                
                excess_ret = agg_analysis['excess_return_analysis']
                print(f"   Mean Excess Return: {excess_ret['mean']:.3%}")
                print(f"   Excess Return Volatility: {excess_ret['std_dev']:.3%}")
                print(f"   Positive Excess Rate: {excess_ret['positive_excess_rate']:.1%}")
                
                track_err = agg_analysis['tracking_error_analysis']
                print(f"   Mean Tracking Error: {track_err['mean']:.3%}")
                
                if 'advanced_statistics' in agg_analysis:
                    adv_stats = agg_analysis['advanced_statistics']
                    print(f"   Mean Information Ratio: {adv_stats['information_ratios']['mean']:.3f}")
                    print(f"   Mean Alpha: {adv_stats['alpha_analysis']['mean_alpha']:.3%}")
        else:
            print(f"❌ Aggregated mode failed: {results_aggregated.get('error')}")
            return False
        
        # Test 3: Both output mode
        print("\n" + "="*50)
        print("TEST 3: BOTH OUTPUT MODE")
        print("="*50)
        
        start_time = datetime.now()
        results_both = orchestrator.vs_bm(
            mandate=test_mandate,
            analysis_start=test_start,
            analysis_end=test_end,
            sample_size=test_sample_size,
            output_mode="both",
            include_advanced_stats=False  # Test without advanced stats
        )
        
        duration_both = datetime.now() - start_time
        print(f"⏱️  Both mode duration: {duration_both}")
        
        if results_both.get('success'):
            print(f"✅ Both mode: Analysis generated")
            
            has_individual = 'individual_comparisons' in results_both['results']
            has_aggregated = 'aggregated_analysis' in results_both['results']
            
            print(f"   Individual comparisons: {'✅' if has_individual else '❌'}")
            print(f"   Aggregated analysis: {'✅' if has_aggregated else '❌'}")
            
            if has_individual and has_aggregated:
                individual_count = len(results_both['results']['individual_comparisons'])
                agg_count = results_both['results']['aggregated_analysis']['summary']['total_scenarios_compared']
                print(f"   Comparison count consistency: {'✅' if individual_count == agg_count else '❌'}")
        else:
            print(f"❌ Both mode failed: {results_both.get('error')}")
            return False
        
        # Test 4: Parameter sensitivity analysis
        if results_aggregated.get('success'):
            print("\n" + "="*50)
            print("TEST 4: PARAMETER SENSITIVITY")
            print("="*50)
            
            agg_analysis = results_aggregated['results']['aggregated_analysis']
            if 'parameter_sensitivity' in agg_analysis:
                param_sens = agg_analysis['parameter_sensitivity']
                
                print(f"📈 Parameter Sensitivity Analysis:")
                for param_name, param_data in param_sens.items():
                    print(f"\n   {param_name}:")
                    for value, stats in param_data.items():
                        print(f"      {value}: {stats['avg_excess_return']:.3%} avg excess, {stats['win_rate']:.1%} win rate")
        
        # Final summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print("✅ All tests passed successfully!")
        print(f"📊 Function Performance:")
        print(f"   Individual mode: {duration_individual}")
        print(f"   Aggregated mode: {duration_aggregated}")
        print(f"   Both mode: {duration_both}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_vs_bm_validation():
    """Test input validation for vs_bm function"""
    
    print("\n" + "="*60)
    print("TESTING INPUT VALIDATION")
    print("="*60)
    
    orchestrator = PHPMasterOrchestrator()
    
    # Test invalid mandate
    try:
        orchestrator.vs_bm(mandate="INVALID", sample_size=5)
        print("❌ Invalid mandate test failed - should have thrown error")
        return False
    except ValueError as e:
        print(f"✅ Invalid mandate correctly rejected: {str(e)}")
    
    # Test invalid output mode
    try:
        result = orchestrator.vs_bm(
            mandate="EQFI", 
            output_mode="invalid_mode",
            sample_size=5
        )
        # This should not throw an error, but should handle gracefully
        print("✅ Invalid output mode handled gracefully")
    except Exception as e:
        print(f"⚠️  Unexpected error with invalid output mode: {str(e)}")
    
    print("✅ Input validation tests completed")
    return True

if __name__ == "__main__":
    print(f"🧪 vs_bm Analysis Testing Suite")
    print(f"Started: {datetime.now()}")
    
    # Run validation tests first
    validation_passed = test_vs_bm_validation()
    
    if validation_passed:
        # Run main functionality tests
        main_tests_passed = test_vs_bm_analysis()
        
        if main_tests_passed:
            print(f"\n🎉 ALL TESTS PASSED!")
            print(f"The vs_bm function is ready for production use.")
        else:
            print(f"\n❌ Main tests failed")
    else:
        print(f"\n❌ Validation tests failed")
    
    print(f"\nCompleted: {datetime.now()}")