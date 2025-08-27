#!/usr/bin/env python3
"""
Realistic test of parallel processing with larger scenario counts
"""

import time
from portfolio_simulation import PortfolioSimulation, ParallelConfig
from multiprocessing import cpu_count

def test_realistic_parallel():
    """Test parallel processing with a realistic number of scenarios"""
    
    print("=" * 60)
    print("REALISTIC PARALLEL PROCESSING TEST")
    print("=" * 60)
    
    sim_engine = PortfolioSimulation()
    
    # Generate more scenarios (realistic workload)
    test_scenarios = sim_engine.generate_all_php_scenarios(
        "EQFI", "2015-01-01", "2020-01-01"  # 5-year analysis period
    )
    
    print(f"Generated {len(test_scenarios)} total scenarios")
    
    # Test with different scenario counts
    scenario_counts = [50, 100, 200]
    
    for count in scenario_counts:
        if count > len(test_scenarios):
            continue
            
        scenarios = test_scenarios[:count]
        print(f"\n{'='*40}")
        print(f"Testing with {count} scenarios")
        print(f"{'='*40}")
        
        # Sequential
        print("\n🔄 Sequential processing...")
        start_time = time.time()
        
        sequential_config = ParallelConfig(
            use_parallel=False, 
            progress_reporting=False
        )
        sequential_results = sim_engine.simulate_scenarios_parallel(
            scenarios, sequential_config
        )
        
        sequential_time = time.time() - start_time
        sequential_successful = sum(1 for r in sequential_results if r['success'])
        
        print(f"   Time: {sequential_time:.2f}s")
        print(f"   Successful: {sequential_successful}/{count}")
        
        # Parallel
        print("\n🚀 Parallel processing...")
        start_time = time.time()
        
        parallel_config = ParallelConfig(
            use_parallel=True,
            max_workers=min(6, cpu_count()-1),  # Leave some cores free
            progress_reporting=False
        )
        parallel_results = sim_engine.simulate_scenarios_parallel(
            scenarios, parallel_config
        )
        
        parallel_time = time.time() - start_time
        parallel_successful = sum(1 for r in parallel_results if r['success'])
        
        print(f"   Time: {parallel_time:.2f}s")
        print(f"   Successful: {parallel_successful}/{count}")
        
        # Calculate speedup
        if parallel_time > 0:
            speedup = sequential_time / parallel_time
            print(f"\n📊 Speedup: {speedup:.2f}x")
            
            if speedup > 1:
                print("✅ Parallel processing is faster!")
            else:
                print("⚠️  Sequential processing is still faster (overhead dominates)")
        
        # Verify results are identical (sample check)
        print("\n🔍 Quick consistency check...")
        sample_size = min(5, len(sequential_results))
        consistent = True
        
        for i in range(sample_size):
            seq_result = sequential_results[i]
            par_result = parallel_results[i]
            
            if seq_result['success'] != par_result['success']:
                consistent = False
                break
                
            if seq_result['success']:
                seq_return = seq_result['simulation_results']['total_return']
                par_return = par_result['simulation_results']['total_return']
                
                if abs(seq_return - par_return) > 1e-10:
                    consistent = False
                    break
        
        if consistent:
            print("✅ Results are consistent")
        else:
            print("❌ Results differ - parallel processing has issues!")

if __name__ == "__main__":
    print(f"System specs:")
    print(f"  CPU cores: {cpu_count()}")
    
    test_realistic_parallel()