import numpy as np
import matplotlib.pyplot as plt

def create_test_grid():
    """
    Create a simple 20x20 test floor plan for development
    Layout: 4 rooms with doors connecting them
    """
    # Create empty grid (20x20)
    grid = np.zeros((20, 20), dtype=np.uint8)  # 0 = free
    
    # Create outer walls (top, bottom, left, right)
    grid[0, :] = 1   # Top wall
    grid[-1, :] = 1  # Bottom wall  
    grid[:, 0] = 1   # Left wall
    grid[:, -1] = 1  # Right wall
    
    # Create inner walls (room dividers)
    grid[10, 5:15] = 1   # Horizontal wall
    grid[5:15, 10] = 1   # Vertical wall
    
    # Create doors (openings in walls)
    grid[10, 7] = 2   # Door in horizontal wall
    grid[7, 10] = 2   # Door in vertical wall
    
    # Create windows
    grid[0, 5] = 3    # Window in top wall
    grid[19, 15] = 3  # Window in bottom wall
    
    return grid

if __name__ == "__main__":
    # Generate test grid
    test_grid = create_test_grid()
    
    # Save for Person B's code
    np.save("grid.npy", test_grid)
    
    # Visualize it
    plt.figure(figsize=(10, 8))
    plt.imshow(test_grid, cmap='viridis')
    plt.colorbar(label='0=Free, 1=Wall, 2=Door, 3=Window')
    plt.title("Test Grid for Development")
    plt.grid(True, alpha=0.3)
    plt.savefig("test_grid.png")
    plt.show()
    
    print(f"Test grid created: shape = {test_grid.shape}")
    print("Cell counts:")
    print(f"  Free cells: {np.sum(test_grid == 0)}")
    print(f"  Wall cells: {np.sum(test_grid == 1)}")
    print(f"  Door cells: {np.sum(test_grid == 2)}")
    print(f"  Window cells: {np.sum(test_grid == 3)}")
    print("\nSaved as 'grid.npy' and 'test_grid.png'")