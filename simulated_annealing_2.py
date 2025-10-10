import pandas as pd
import numpy as np
import random, math
from context import Context
import time

# Step 1: Load data & build Q

dataframe2 = pd.read_csv("test4.csv")
dataframe2.drop(columns=dataframe2.columns[0], inplace=True)

context2 = Context(dataframe2)
concepts2 = context2.extract_concepts()
print(f"number of concepts: {concepts2.get_number_of_concepts()}")
Q = np.array(concepts2.set_cover(), dtype=float)

num_concepts = Q.shape[0]
linear_terms = np.diag(Q)
quadratic_matrix = Q.copy()
np.fill_diagonal(quadratic_matrix, 0)

print(f"Loaded {num_concepts} concepts.")
print(f"Q matrix shape: {Q.shape}\n")


# Step 2: Helper functions

def objective(weights, linear, quad, penalty_w=1000):
    """Total energy = main + penalty"""
    main = np.dot(linear, weights) + weights.T @ quad @ weights
    penalty = penalty_w * (np.sum(weights) - num_concepts) ** 2
    return main + penalty

def neighbor(weights, step_frac=0.05):
    """Generate neighbor by transferring fraction of total range between two variables"""
    i, j = random.sample(range(len(weights)), 2)
    new = weights.copy()
    step = step_frac * num_concepts  # scale step to range size
    transfer = random.uniform(-step, step)
    new[i] = np.clip(new[i] + transfer, 0, num_concepts)
    new[j] = np.clip(new[j] - transfer, 0, num_concepts)
    return new

def repair(weights):
    """Rescale weights so they sum to num_concepts"""
    s = np.sum(weights)
    if s == 0:
        return np.full_like(weights, num_concepts / len(weights))
    w = weights * (num_concepts / s)
    return np.clip(w, 0, num_concepts)


# Step 3: Simulated Annealing

def simulated_annealing(linear, quad, T0=1000, cool=0.95, minT=1e-6,
                        outer_iters=100, inner_iters=20, step_frac=0.05):
    
    start_time = time.time()
    w = repair(np.random.uniform(0, num_concepts, len(linear)))
    e = objective(w, linear, quad)
    best_w, best_e = w.copy(), e
    T = T0
    total_iter = 0

    print("Starting simulated annealing...\n")

    for outer in range(outer_iters):
        for _ in range(inner_iters):
            total_iter += 1
            new_w = repair(neighbor(w, step_frac))
            new_e = objective(new_w, linear, quad)
            delta = new_e - e

            if delta < 0 or random.random() < math.exp(-delta / T):
                w, e = new_w, new_e
                if e < best_e:
                    best_w, best_e = w.copy(), e

        # ✅ Print once after each inner iteration loop ends
        print(f"After inner loop {outer+1:3d}: "
              f"Temp={T:.4f} | Energy={e:.4f} | Best={best_e:.4f} | Sum={np.sum(w):.4f}")

        # Cooling
        T *= cool
        if T < minT:
            print("\nTemperature too low — stopping early.")
            break
    runtime = time.time() - start_time
    print("\n=== Final Results ===")
    print(f"RuneTime: {runtime:.2f} sec")
    print(f"Best energy: {best_e:.4f}")
    print(f"Best weights sum: {np.sum(best_w):.4f} (target: {num_concepts})")
    print(f"Best weights: {np.round(best_w, 4)}")

    return best_w, best_e


# Step 4: Run it
weights, energy = simulated_annealing(
    linear_terms, quadratic_matrix,
    T0=10000, cool = 0.9995, outer_iters=1000, inner_iters=10000, step_frac= 0.1
)
