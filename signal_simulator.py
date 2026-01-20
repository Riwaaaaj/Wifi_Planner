"""
Fixed and completed signal simulation code for WiFi optimization
Integrated with Person C's grid format (0,1,2,3 codes)
"""

import numpy as np

# CONSTANTS AND PARAMETERS
S0 = -30.0           # Reference signal strength at router (dBm)
S_THRESHOLD = -70    # Minimum usable signal strength (dBm)
D_LOSS_K = 2.0       # Distance attenuation constant (dB per meter)

# Person C's grid uses integer codes: 0=Free, 1=Wall, 2=Door, 3=Window
# So we map integer codes to attenuation values
ATTENUATION = {
    0: 0.0,   # FREE
    1: 8.0,   # WALL
    2: 3.0,   # DOOR
    3: 2.0    # WINDOW
}

# GEOMETRY FUNCTIONS
def distance(router, cell):
    """Calculate Euclidean distance between router and cell."""
    x_r, y_r = router
    x_p, y_p = cell
    return np.sqrt((x_p - x_r)**2 + (y_p - y_r)**2)

def distance_loss(distance_val, k=D_LOSS_K):
    """Calculate distance-based signal loss."""
    return k * distance_val

def obstacle_loss(router, cell, grid):
    """
    Calculate obstacle-based signal loss.
    Uses Bresenham's line algorithm to trace path.
    """
    x_r, y_r = router
    x_p, y_p = cell
    
    loss = 0.0
    steps = max(abs(x_p - x_r), abs(y_p - y_r))
    
    if steps == 0:
        return 0.0
    
    for step in range(1, steps):
        t = step / steps
        x = int(round(x_r + t * (x_p - x_r)))
        y = int(round(y_r + t * (y_p - y_r)))
        
        if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
            cell_type = grid[y, x]
            loss += ATTENUATION.get(cell_type, 0.0)
    
    return loss

def signal_strength(router, cell, grid):
    """Calculate signal strength from one router to one cell."""
    d = distance(router, cell)
    loss_d = distance_loss(d)
    loss_o = obstacle_loss(router, cell, grid)
    signal = S0 - loss_d - loss_o
    return signal

# COMPLETED MISSING FUNCTIONS

def best_signal_at_cell(cell, routers, grid):
    """
    Calculate the best signal at a cell from multiple routers.
    
    Args:
        cell: (x, y) tuple
        routers: list of (x, y) router positions
        grid: 2D numpy array
    
    Returns:
        best_signal (dBm), best_router_index
    """
    best_signal = -np.inf
    best_router_idx = -1
    
    for i, router in enumerate(routers):
        sig = signal_strength(router, cell, grid)
        if sig > best_signal:
            best_signal = sig
            best_router_idx = i
    
    return best_signal, best_router_idx

def calculate_coverage(grid, routers):
    """
    Calculate coverage percentage and signal map.
    
    Args:
        grid: 2D numpy array with codes
        routers: list of router positions
    
    Returns:
        coverage_percentage (0-100), signal_map (2D array), coverage_map (2D bool array)
    """
    height, width = grid.shape
    signal_map = np.full((height, width), -np.inf)  # Start with very low signal
    coverage_map = np.zeros((height, width), dtype=bool)
    
    free_cells = 0
    covered_cells = 0
    
    # Calculate signal for each free cell
    for y in range(height):
        for x in range(width):
            if grid[y, x] == 0:  # Only free cells matter for coverage
                free_cells += 1
                cell = (x, y)
                best_sig, _ = best_signal_at_cell(cell, routers, grid)
                signal_map[y, x] = best_sig
                
                if best_sig >= S_THRESHOLD:
                    coverage_map[y, x] = True
                    covered_cells += 1
    
    # Calculate coverage percentage
    if free_cells > 0:
        coverage_percentage = (covered_cells / free_cells) * 100
    else:
        coverage_percentage = 0.0
    
    return coverage_percentage, signal_map, coverage_map

def generate_signal_heatmap(grid, routers):
    """
    Generate data for visualization.
    Returns signal map and coverage map.
    """
    coverage_percentage, signal_map, coverage_map = calculate_coverage(grid, routers)
    return coverage_percentage, signal_map, coverage_map

# Quick test function
def test_simulation():
    """Test the signal simulation with a simple setup."""
    print("Testing signal simulation...")
    
    # Create a small test grid
    test_grid = np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0]
    ])
    
    # Test router positions
    routers = [(0, 0), (4, 4)]
    
    # Calculate coverage
    coverage_percentage, signal_map, coverage_map = calculate_coverage(test_grid, routers)
    
    print(f"Grid shape: {test_grid.shape}")
    print(f"Number of routers: {len(routers)}")
    print(f"Coverage percentage: {coverage_percentage:.2f}%")
    print(f"Signal map shape: {signal_map.shape}")
    
    # Test individual cell
    cell = (4, 0)
    best_sig, router_idx = best_signal_at_cell(cell, routers, test_grid)
    print(f"\nCell (4,0): Best signal = {best_sig:.2f} dBm from router {router_idx}")
    
    return coverage_percentage

if __name__ == "__main__":
    test_simulation()