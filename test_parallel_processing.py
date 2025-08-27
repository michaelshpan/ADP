#!/usr/bin/env python3
"""
Test script for parallel processing implementation
Compares parallel vs sequential results for consistency
"""

import sys
import time
import numpy as np
from portfolio_simulation import PortfolioSimulation, ParallelConfig
from multiprocessing import cpu_count

def test_parallel_vs_sequential():
    """Test that parallel and sequential processing produce identical results"""
    
    print("=" * 60)
    print("PHP PARALLEL PROCESSING TEST")
    print("=" * 60)
    
    # Initialize simulation engine
    sim_engine = PortfolioSimulation()
    
    # Generate a small test set of scenarios
    test_scenarios = sim_engine.generate_all_php_scenarios(
        "EQFI", "2010-01-01", "2020-01-01"  # Longer date range for scenario generation
    )
    
    # Limit to first 20 scenarios for quick testing
    test_scenarios = test_scenarios[:20]
    print(f"Testing with {len(test_scenarios)} scenarios")
    
    if not test_scenarios:
        print("❌ No scenarios generated for testing")
        return False
    
    # Test 1: Sequential processing
    print("\n1️⃣ Running sequential simulation...")
    start_time = time.time()
    
    sequential_config = ParallelConfig(use_parallel=False, progress_reporting=True)
    sequential_results = sim_engine.simulate_scenarios_parallel(
        test_scenarios, sequential_config
    )
    
    sequential_time = time.time() - start_time
    print(f"   Sequential time: {sequential_time:.2f} seconds")
    
    # Test 2: Parallel processing  
    print("\n2️⃣ Running parallel simulation...")
    start_time = time.time()
    
    parallel_config = ParallelConfig(
        use_parallel=True, 
        max_workers=min(4, cpu_count()-1),  # Use max 4 workers for testing
        progress_reporting=True
    )
    parallel_results = sim_engine.simulate_scenarios_parallel(
        test_scenarios, parallel_config
    )
    
    parallel_time = time.time() - start_time
    print(f"   Parallel time: {parallel_time:.2f} seconds")
    
    # Test 3: Compare results
    print("\n3️⃣ Comparing results...")
    
    if len(sequential_results) != len(parallel_results):
        print(f"❌ Result count mismatch: {len(sequential_results)} vs {len(parallel_results)}")
        return False
    
    # Compare each scenario result
    mismatches = 0
    for i, (seq_result, par_result) in enumerate(zip(sequential_results, parallel_results)):
        
        # Check success status
        if seq_result['success'] != par_result['success']:
            print(f"❌ Scenario {i}: Success status mismatch")
            mismatches += 1
            continue
            
        # Skip comparison if either failed
        if not seq_result['success'] or not par_result['success']:
            continue
        
        # Compare simulation results
        seq_sim = seq_result['simulation_results']
        par_sim = par_result['simulation_results']
        
        # Check key metrics with tolerance for floating point differences
        tolerance = 1e-10
        
        metrics_to_check = [
            'total_return', 'annualized_return', 'volatility', 
            'max_drawdown', 'sharpe_ratio'
        ]
        
        for metric in metrics_to_check:
            seq_val = seq_sim.get(metric, 0)
            par_val = par_sim.get(metric, 0)
            
            if abs(seq_val - par_val) > tolerance:
                print(f"❌ Scenario {i}: {metric} mismatch: {seq_val} vs {par_val}")
                mismatches += 1
                break
    
    # Summary
    successful_seq = sum(1 for r in sequential_results if r['success'])
    successful_par = sum(1 for r in parallel_results if r['success'])
    
    print("\n📊 TEST RESULTS:")
    print(f"   Scenarios tested: {len(test_scenarios)}")
    print(f"   Sequential successful: {successful_seq}")
    print(f"   Parallel successful: {successful_par}")
    print(f"   Mismatches found: {mismatches}")
    
    if parallel_time > 0:
        speedup = sequential_time / parallel_time
        print(f"   Speedup: {speedup:.2f}x")
    
    # Final verdict
    if mismatches == 0 and successful_seq == successful_par:
        print("✅ PARALLEL PROCESSING TEST PASSED")
        print("   Results are identical between sequential and parallel processing")
        return True
    else:
        print("❌ PARALLEL PROCESSING TEST FAILED")
        print("   Results differ between sequential and parallel processing")
        return False

def test_performance_scaling():
    """Test how performance scales with different worker counts"""
    
    print("\n" + "=" * 60)
    print("PERFORMANCE SCALING TEST")
    print("=" * 60)
    
    sim_engine = PortfolioSimulation()
    
    # Generate scenarios for performance testing
    test_scenarios = sim_engine.generate_all_php_scenarios(
        "EQFI", "2010-01-01", "2015-01-01"  # 5 year period for scenario generation
    )
    
    # Limit scenarios for reasonable test time
    test_scenarios = test_scenarios[:50]
    print(f"Testing performance scaling with {len(test_scenarios)} scenarios")
    
    max_workers = min(8, cpu_count())
    worker_counts = [1, 2, 4] + ([max_workers] if max_workers > 4 else [])
    
    results = {}
    
    for workers in worker_counts:
        print(f"\n🔄 Testing with {workers} workers...")
        
        config = ParallelConfig(
            use_parallel=(workers > 1),
            max_workers=workers,
            progress_reporting=False  # Reduce output
        )
        
        start_time = time.time()
        simulation_results = sim_engine.simulate_scenarios_parallel(test_scenarios, config)
        elapsed_time = time.time() - start_time
        
        successful = sum(1 for r in simulation_results if r['success'])
        results[workers] = {
            'time': elapsed_time,
            'successful': successful
        }
        
        print(f"   Time: {elapsed_time:.2f}s, Successful: {successful}")
    
    # Calculate and display speedups
    print("\n📈 SCALING RESULTS:")
    baseline_time = results[1]['time']
    
    for workers in worker_counts:
        time_taken = results[workers]['time']
        speedup = baseline_time / time_taken if time_taken > 0 else 0
        efficiency = (speedup / workers) * 100 if workers > 0 else 0
        
        print(f"   {workers} workers: {time_taken:.2f}s, {speedup:.2f}x speedup, {efficiency:.1f}% efficiency")

if __name__ == "__main__":
    print(f"System CPU count: {cpu_count()}")
    
    # Run tests
    consistency_passed = test_parallel_vs_sequential()
    
    if consistency_passed:
        test_performance_scaling()
    else:
        print("\n⚠️ Skipping performance test due to consistency issues")
        sys.exit(1)
    
    print("\n🎉 All tests completed!")