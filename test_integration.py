"""
Test integration between Person B's simulation and Person A's GA framework
"""

import numpy as np
from signal_simulator import calculate_coverage, generate_signal_heatmap
from create_test_grid import create_test_grid  # Or load from file

def test_baseline_placements():
    """Test random vs uniform vs manual router placements."""
    
    # Load or create grid
    grid = np.load("grid.npy")  # From your test grid
    
    print(f"Grid loaded: {grid.shape}")
    
    # Test random placement
    np.random.seed(42)
    random_routers = [
        (np.random.randint(1, grid.shape[1]-1), 
         np.random.randint(1, grid.shape[0]-1))
        for _ in range(3)
    ]
    
    # Test uniform placement
    uniform_routers = [
        (5, 5),   # Top-left room
        (15, 5),  # Top-right room  
        (5, 15),  # Bottom-left room
        (15, 15)  # Bottom-right room
    ][:3]  # Only use 3 routers
    
    # Test manual "good" placement
    manual_routers = [
        (7, 7),   # Center of top-left
        (13, 7),  # Center of top-right
        (7, 13),  # Center of bottom-left
    ]
    
    # Calculate coverage for each
    print("\n" + "="*50)
    print("BASELINE COVERAGE COMPARISON")
    print("="*50)
    
    strategies = [
        ("Random", random_routers),
        ("Uniform", uniform_routers),
        ("Manual", manual_routers)
    ]
    
    results = {}
    for name, routers in strategies:
        coverage, signal_map, coverage_map = calculate_coverage(grid, routers)
        results[name] = {
            'coverage': coverage,
            'routers': routers,
            'signal_map': signal_map
        }
        print(f"{name}: {coverage:.2f}% coverage")
        print(f"  Router positions: {routers}")
    
    # Find best baseline
    best_strategy = max(results, key=lambda x: results[x]['coverage'])
    print(f"\nBest baseline: {best_strategy} with {results[best_strategy]['coverage']:.2f}% coverage")
    
    return results

if __name__ == "__main__":
    print("INTEGRATION TEST - Person A + Person B")
    print("="*60)
    
    # First, make sure we have a grid
    try:
        grid = np.load("grid.npy")
    except FileNotFoundError:
        print("grid.npy not found. Creating test grid...")
        from create_test_grid import create_test_grid
        grid = create_test_grid()
        np.save("grid.npy", grid)
    
    # Run tests
    results = test_baseline_placements()
    
    print("\n" + "="*60)
    print("Integration test completed!")
    print("Next step: Implement Genetic Algorithm optimizer.")