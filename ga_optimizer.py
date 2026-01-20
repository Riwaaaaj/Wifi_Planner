"""
Genetic Algorithm for WiFi Router Placement Optimization
Person A - Main GA Implementation
"""

import numpy as np
import random
from typing import List, Tuple, Dict
import math
from signal_simulator import calculate_coverage

class WiFiGeneticAlgorithm:
    def __init__(self, grid: np.ndarray, n_routers: int = 3, 
                 population_size: int = 50, generations: int = 100):
        """
        Initialize GA for WiFi router placement.
        
        Args:
            grid: 2D numpy array from Person C (0=free, 1=wall, 2=door, 3=window)
            n_routers: Number of routers to place
            population_size: GA population size
            generations: Number of GA generations
        """
        self.grid = grid
        self.n_routers = n_routers
        self.population_size = population_size
        self.generations = generations
        
        # Get valid positions (only in free cells)
        self.valid_positions = []
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                if grid[y, x] == 0:  # Free space
                    self.valid_positions.append((x, y))
        
        print(f"GA Initialized:")
        print(f"  Grid size: {grid.shape}")
        print(f"  Valid positions: {len(self.valid_positions)}")
        print(f"  Routers to place: {n_routers}")
        print(f"  Population: {population_size}, Generations: {generations}")
        
        # GA Parameters
        self.mutation_rate = 0.1
        self.elite_size = max(2, population_size // 10)  # Keep top 10%
        
    def create_chromosome(self) -> List[Tuple[int, int]]:
        """Create a random chromosome (router placement)."""
        return random.sample(self.valid_positions, self.n_routers)
    
    def create_population(self) -> List[List[Tuple[int, int]]]:
        """Create initial population."""
        return [self.create_chromosome() for _ in range(self.population_size)]
    
    def fitness_function(self, chromosome: List[Tuple[int, int]]) -> float:
        """
        Calculate fitness of a chromosome.
        Fitness = Coverage percentage - Overlap penalty
        """
        # Calculate coverage using Person B's simulation
        coverage_percentage, _, _ = calculate_coverage(self.grid, chromosome)
        
        # Penalty for routers too close together (wasteful placement)
        overlap_penalty = self._calculate_overlap_penalty(chromosome)
        
        # Final fitness (coverage minus penalty)
        fitness = coverage_percentage - overlap_penalty
        
        return fitness
    
    def _calculate_overlap_penalty(self, chromosome: List[Tuple[int, int]]) -> float:
        """Penalize routers placed too close together."""
        penalty = 0.0
        min_distance = 3.0  # Minimum recommended distance between routers (meters)
        
        for i in range(len(chromosome)):
            for j in range(i + 1, len(chromosome)):
                x1, y1 = chromosome[i]
                x2, y2 = chromosome[j]
                distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                if distance < min_distance:
                    # Quadratic penalty for very close routers
                    penalty += (min_distance - distance) ** 2
        
        return penalty
    
    def selection(self, population: List, fitness_scores: List[float]) -> List:
        """Select parents using tournament selection."""
        selected = []
        tournament_size = 3
        
        for _ in range(len(population)):
            # Random tournament
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            
            # Select best from tournament
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx])
        
        return selected
    
    def crossover(self, parent1: List, parent2: List) -> List:
        """Single-point crossover."""
        if random.random() < 0.7:  # 70% crossover rate
            point = random.randint(1, self.n_routers - 1)
            child = parent1[:point] + parent2[point:]
        else:
            child = parent1.copy()  # No crossover
        
        return child
    
    def mutation(self, chromosome: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Mutate a chromosome by changing router positions."""
        mutated = chromosome.copy()
        
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                # Replace this router with a random valid position
                mutated[i] = random.choice(self.valid_positions)
        
        return mutated
    
    def run(self, verbose: bool = True) -> Dict:
        """Run the genetic algorithm."""
        # Initialize population
        population = self.create_population()
        best_fitness_history = []
        avg_fitness_history = []
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = [self.fitness_function(chromosome) for chromosome in population]
            
            # Track statistics
            best_fitness = max(fitness_scores)
            avg_fitness = np.mean(fitness_scores)
            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(avg_fitness)
            
            # Get best chromosome
            best_idx = np.argmax(fitness_scores)
            best_chromosome = population[best_idx]
            
            if verbose and generation % 10 == 0:
                print(f"Generation {generation:3d}: "
                      f"Best Fitness = {best_fitness:.2f}, "
                      f"Avg Fitness = {avg_fitness:.2f}")
            
            # Selection
            selected = self.selection(population, fitness_scores)
            
            # Create next generation
            next_generation = []
            
            # Elitism: keep best chromosomes
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            next_generation.extend([population[i] for i in elite_indices])
            
            # Crossover and mutation
            while len(next_generation) < self.population_size:
                # Select parents
                parent1 = random.choice(selected)
                parent2 = random.choice(selected)
                
                # Crossover
                child = self.crossover(parent1, parent2)
                
                # Mutation
                child = self.mutation(child)
                
                # Ensure no duplicate routers in chromosome
                child = list(set(child))
                while len(child) < self.n_routers:
                    child.append(random.choice(self.valid_positions))
                
                next_generation.append(child[:self.n_routers])
            
            population = next_generation
        
        # Final evaluation
        final_fitness = [self.fitness_function(chromosome) for chromosome in population]
        best_idx = np.argmax(final_fitness)
        best_solution = population[best_idx]
        best_fitness = final_fitness[best_idx]
        
        # Calculate final coverage (without penalty for reporting)
        final_coverage, _, _ = calculate_coverage(self.grid, best_solution)
        
        results = {
            'best_solution': best_solution,
            'best_fitness': best_fitness,
            'best_coverage': final_coverage,
            'fitness_history': best_fitness_history,
            'avg_fitness_history': avg_fitness_history
        }
        
        return results

def main():
    """Main function to run the GA."""
    print("="*60)
    print("WiFi ROUTER OPTIMIZATION - GENETIC ALGORITHM")
    print("="*60)
    
    # Load grid
    try:
        grid = np.load("grid.npy")
    except FileNotFoundError:
        print("Creating test grid...")
        from create_test_grid import create_test_grid
        grid = create_test_grid()
        np.save("grid.npy", grid)
    
    # Initialize GA
    ga = WiFiGeneticAlgorithm(
        grid=grid,
        n_routers=3,
        population_size=30,  # Smaller for faster testing
        generations=50
    )
    
    print("\nRunning Genetic Algorithm...")
    print("-"*40)
    
    # Run GA
    results = ga.run(verbose=True)
    
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Best router positions:")
    for i, (x, y) in enumerate(results['best_solution'], 1):
        print(f"  Router {i}: ({x}, {y})")
    print(f"\nCoverage achieved: {results['best_coverage']:.2f}%")
    print(f"Fitness score: {results['best_fitness']:.2f}")
    
    return results

if __name__ == "__main__":
    results = main()