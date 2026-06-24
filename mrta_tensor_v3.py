import numpy as np
import quimb.tensor as qtn
import time
import random

# =========================================================
# 1. SCALABLE MRTA PROBLEM GENERATOR
# =========================================================

def generate_mrta(n, seed=1):
    np.random.seed(seed)
    return np.random.randint(1, 20, size=(n, n))

# problem size (SCALE THIS UP)
N = 6
cost = generate_mrta(N)

print("\n=== MRTA PROBLEM (N =", N, ") ===")
print(cost)

# =========================================================
# 2. TENSOR NETWORK REPRESENTATION
# =========================================================

circ = qtn.Circuit(N)

for q in range(N):
    circ.h(q)

for q in range(N):
    angle = np.mean(cost[:, q]) / 10.0
    circ.ry(q, angle)

psi = circ.psi

print("\n=== TENSOR NETWORK STATE ===")
print(psi)

# =========================================================
# 3. ENERGY FUNCTION (HYBRID MODEL)
# =========================================================

def energy(assignment):
    classical = 0

    for r in range(N):
        classical += cost[r, assignment[r]]

    # constraint penalty (no task collisions)
    penalty = 0
    if len(set(assignment)) < len(assignment):
        penalty += 10

    # smooth tensor-inspired modulation
    tensor_term = np.sin(classical / 7.0)

    return classical + penalty + tensor_term


# =========================================================
# 4. SIMULATED ANNEALING OPTIMIZER (KEY UPGRADE)
# =========================================================

def simulated_annealing(iterations=2000, temp=10.0, cooling=0.995):

    current = list(range(N))
    random.shuffle(current)

    current_energy = energy(current)

    best = current[:]
    best_energy = current_energy

    start = time.time()

    for i in range(iterations):

        # swap two tasks (neighbor solution)
        candidate = current[:]
        a, b = random.sample(range(N), 2)
        candidate[a], candidate[b] = candidate[b], candidate[a]

        candidate_energy = energy(candidate)

        delta = candidate_energy - current_energy

        # acceptance probability
        if delta < 0 or random.random() < np.exp(-delta / temp):
            current = candidate
            current_energy = candidate_energy

            if current_energy < best_energy:
                best = current[:]
                best_energy = current_energy

        temp *= cooling

    end = time.time()

    return best, best_energy, end - start


# =========================================================
# 5. RUN EXPERIMENT
# =========================================================

best_solution, best_energy, runtime = simulated_annealing()

print("\n=== OPTIMIZATION RESULT ===")
print("Best Assignment:", best_solution)
print("Best Energy:", best_energy)
print("Runtime (s):", runtime)