from perfect_weight_calculator import PerfectWeightCalculator

def analyze_weight_logic():
    """Analyze the perfect weight calculation logic with detailed breakdown"""
    
    calc = PerfectWeightCalculator()
    
    print("=== PERFECT WEIGHT LOGIC ANALYSIS ===\n")
    
    # Test the original example with detailed breakdown
    mandate = "EQFI"
    start_date = "2010-01-01"
    end_date = "2020-01-01"
    deviation = 0.05
    
    # Get performance and neutral weights
    performance = calc.calculate_asset_performance(mandate, start_date, end_date)
    neutral_weights = calc.portfolio_config.get_mandate_weights(mandate)
    
    print(f"ORIGINAL EXAMPLE BREAKDOWN:")
    print(f"Russell 3000: {performance['Russell 3000']:.1%}")
    print(f"ICE BofA US Treasury: {performance['ICE BofA US Treasury']:.1%}")  
    print(f"ICE BofAML US Corporate: {performance['ICE BofAML US Corporate']:.1%}")
    
    # Calculate proportions if purely performance-based
    total_perf = sum(performance.values())
    print(f"\nTotal Performance: {total_perf:.1%}")
    print("\nPure Performance-Based Proportions (before constraints):")
    for asset, perf in performance.items():
        prop = perf / total_perf
        print(f"{asset}: {prop:.1%}")
    
    # Now show how constraints affect this
    perfect_weights = calc.calculate_perfect_weights(mandate, performance, deviation)
    print(f"\nAfter applying ±{deviation:.0%} absolute constraints:")
    for asset in neutral_weights:
        neutral = neutral_weights[asset]
        perfect = perfect_weights.get(asset, 0)
        min_wgt = max(0.0, neutral - deviation)
        max_wgt = min(1.0, neutral + deviation)
        print(f"{asset}: {perfect:.1%} (bounds: {min_wgt:.1%} - {max_wgt:.1%})")
    
    print(f"\nTotal after normalization: {sum(perfect_weights.values()):.1%}")
    
    print("\n" + "="*80)
    
def test_multiple_scenarios():
    """Test 5 different scenarios to check robustness"""
    
    calc = PerfectWeightCalculator()
    
    test_cases = [
        {
            'name': 'Tech Boom Period',
            'mandate': 'EQFI',
            'start': '1995-01-01',
            'end': '2000-01-01',
            'deviation': 0.20
        },
        {
            'name': 'Dot-Com Crash Recovery',
            'mandate': 'EQFI', 
            'start': '2003-01-01',
            'end': '2008-01-01',
            'deviation': 0.10
        },
        {
            'name': 'Financial Crisis Period',
            'mandate': 'EQFILA',
            'start': '2008-01-01',
            'end': '2013-01-01',
            'deviation': 0.50
        },
        {
            'name': 'Low Interest Rate Era',
            'mandate': 'EQFILA',
            'start': '2012-01-01',
            'end': '2017-01-01',
            'deviation': 0.05
        },
        {
            'name': 'Recent Bull Market',
            'mandate': 'EQFILAIA',
            'start': '2016-01-01',
            'end': '2021-01-01',
            'deviation': 0.20
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"=== TEST CASE {i}: {test_case['name']} ===")
        print(f"Mandate: {test_case['mandate']}")
        print(f"Period: {test_case['start']} to {test_case['end']}")
        print(f"Deviation: ±{test_case['deviation']:.0%}")
        
        try:
            # Calculate performance
            performance = calc.calculate_asset_performance(
                test_case['mandate'], test_case['start'], test_case['end']
            )
            
            # Get neutral weights
            neutral_weights = calc.portfolio_config.get_mandate_weights(test_case['mandate'])
            
            # Calculate perfect weights
            perfect_weights = calc.calculate_perfect_weights(
                test_case['mandate'], performance, test_case['deviation']
            )
            
            print(f"\nAsset Performance:")
            sorted_perf = sorted(performance.items(), key=lambda x: x[1], reverse=True)
            for asset, perf in sorted_perf:
                print(f"  {asset:<35}: {perf:>8.1%}")
            
            print(f"\nWeight Allocation:")
            print(f"{'Asset':<35} {'Neutral':<8} {'Perfect':<8} {'Change':<8} {'Bounds'}")
            print("-" * 85)
            
            for asset in neutral_weights:
                neutral = neutral_weights[asset]
                perfect = perfect_weights.get(asset, 0)
                change = perfect - neutral
                min_wgt = max(0.0, neutral - test_case['deviation'])
                max_wgt = min(1.0, neutral + test_case['deviation'])
                bounds = f"{min_wgt:.0%}-{max_wgt:.0%}"
                
                print(f"{asset:<35} {neutral:>7.1%} {perfect:>7.1%} {change:>7.1%} {bounds:>8}")
            
            print(f"\nTotal Weight: {sum(perfect_weights.values()):.1%}")
            
            # Check if best performer hit max constraint
            best_performer = max(performance.items(), key=lambda x: x[1])
            best_asset, best_perf = best_performer
            best_weight = perfect_weights.get(best_asset, 0)
            max_allowed = min(1.0, neutral_weights[best_asset] + test_case['deviation'])
            
            print(f"Best performer ({best_asset}): {best_perf:.1%} return → {best_weight:.1%} weight")
            print(f"Hit max constraint? {'Yes' if abs(best_weight - max_allowed) < 0.001 else 'No'}")
            
        except Exception as e:
            print(f"Error in calculation: {e}")
        
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    analyze_weight_logic()
    print("\n" + "="*100 + "\n")
    test_multiple_scenarios()