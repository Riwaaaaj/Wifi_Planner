# Testing Person B's code

import numpy as np
import matplotlib.pyplot as plt

# grid ko code herna baki cha
grid = np.load("grid.npy")  
print(f"Grid shape: {grid.shape}")
print(f"Grid unique values: {np.unique(grid)}")

# Visualize the grid to understand it
plt.figure(figsize=(10, 8))
plt.imshow(grid, cmap='viridis')
plt.colorbar(label='0=Free, 1=Wall, 2=Door, 3=Window')
plt.title("Loaded Grid")
plt.savefig("grid_preview.png")
plt.show()

# Count free cells
free_cells = np.sum(grid == 0)
print(f"Free cells: {free_cells}")
print(f"Wall cells: {np.sum(grid == 1)}")