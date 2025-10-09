from context import Context
import pandas as pd
import numpy as np
import math, random


# Step 1: Load Data and Build FCA Context

dataframe2 = pd.read_csv("test2.csv")
dataframe2.drop(columns=dataframe2.columns[0], inplace=True)

context2 = Context(dataframe2)
concepts2 = context2.extract_concepts()
Q = concepts2.set_cover()
Q_np = np.array(Q, dtype=float)

num_concepts = Q_np.shape[0]
print("QUBO matrix shape:", Q_np.shape)


# Step 2: Objective Function

def objective_function(x, Q):
    return x.T @ Q @ x

def neighbor_solution(x):
    """Flip one random bit"""
    neighbor = x.copy()
    flip_index = random.randint(0, len(x) - 1)
    neighbor[flip_index] = 1 - neighbor[flip_index]
    return neighbor


# Step 3: Simulated Annealing

def simulated_annealing(Q, initial_temp=1000000, cooling_rate=0.99995, 
                       min_temp=1e-6, max_iterations=50000):
    
    num_concepts = Q.shape[0]
    current_solution = np.random.randint(0, 2, num_concepts)
    current_energy = objective_function(current_solution, Q)
    
    best_solution = current_solution.copy()
    best_energy = current_energy
    
    temperature = initial_temp
    
    print(f"Initial energy: {current_energy}, Initial solution: {current_solution}")
    
    for iteration in range(1, max_iterations + 1):
        neighbor = neighbor_solution(current_solution)
        neighbor_energy = objective_function(neighbor, Q)
        
        energy_diff = neighbor_energy - current_energy
        
        # Acceptance criteria
        if energy_diff < 0 or random.random() < math.exp(-energy_diff / temperature):
            current_solution = neighbor
            current_energy = neighbor_energy
            if current_energy < best_energy:
                best_solution = current_solution.copy()
                best_energy = current_energy
        
        # Cool down
        temperature *= cooling_rate
        
        # Print results every 100 iterations (or every iteration if you prefer)
        if iteration % 100 == 0 or iteration == 1:
            print(f"Iter {iteration}: Temp={temperature:.5f}, "
                  f"Current={current_energy}, Best={best_energy}")
    
    print("\nFinal Results:")
    print(f"Best Energy: {best_energy}")
    print(f"Best Solution: {best_solution}")
    print(f"Concepts Selected: {np.sum(best_solution)} / {num_concepts}")
    
    return best_solution, best_energy


# Step 4: Run Simulated Annealing

print(simulated_annealing(Q_np))
