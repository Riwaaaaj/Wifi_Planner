import numpy as np
import random
from typing import List, Tuple
import math

class WiFiGeneticAlgorithm:
    def __init__(self, grid: np.ndarray, n_routers: int = 3):
        """
        Initialize GA for WiFi router placement.
        
        Args:
            grid: 2D numpy array from Person 1
            n_routers: Number of routers to place
        """
        self.grid = grid
        self.n_routers = n_routers
        self.grid_height, self.grid_width = grid.shape
        
        # Get all free cell positions
        self.free_positions = []
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if grid[y, x] == 0:  # Free cell
                    self.free_positions.append((x, y))
        
        print(f"Total free positions: {len(self.free_positions)}")
        
        # GA Parameters
        self.population_size = 50
        self.generations = 100
        self.mutation_rate = 0.1
        self.elite_size = 5
        
    def create_chromosome(self) -> List[Tuple[int, int]]:
        """Create a random chromosome (router placement)."""
        return random.sample(self.free_positions, self.n_routers)
    
    def create_population(self) -> List[List[Tuple[int, int]]]:
        """Create initial population."""
        return [self.create_chromosome() for _ in range(self.population_size)]
    
    def fitness_function(self, chromosome: List[Tuple[int, int]]) -> float:
        """
        Calculate fitness of a chromosome.
        TODO: Integrate with Person B's signal simulation
        For now, use simple coverage estimation
        """
        # Simple heuristic: spread routers apart
        total_distance = 0
        for i in range(len(chromosome)):
            for j in range(i+1, len(chromosome)):
                x1, y1 = chromosome[i]
                x2, y2 = chromosome[j]
                dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                total_distance += dist
        
        # Also avoid placing routers near walls? We'll enhance later
        return total_distance  # Higher distance = better spread
    
    def run(self):
        """Run the genetic algorithm."""
        population = self.create_population()
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = [self.fitness_function(chromosome) for chromosome in population]
            
            # Get best chromosome
            best_idx = np.argmax(fitness_scores)
            best_chromosome = population[best_idx]
            best_fitness = fitness_scores[best_idx]
            
            if generation % 10 == 0:
                print(f"Generation {generation}: Best fitness = {best_fitness:.2f}")
                print(f"Best router positions: {best_chromosome}")
        
        return best_chromosome

# Test the GA
if __name__ == "__main__":
    # Load grid
    grid = np.load("grid.npy")
    
    # Create and run GA
    ga = WiFiGeneticAlgorithm(grid, n_routers=3)
    best_solution = ga.run()
    
    print("\n" + "="*50)
    print("FINAL OPTIMIZED ROUTER PLACEMENT:")
    for i, (x, y) in enumerate(best_solution, 1):
        print(f"Router {i}: Grid position ({x}, {y})")