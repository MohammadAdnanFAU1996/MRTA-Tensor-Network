import numpy as np
import quimb.tensor as qtn

# =========================================================
# 1. MULTI-ROBOT TASK ALLOCATION (MRTA) INSTANCE
# =========================================================

cost = np.array([
    [3, 7, 2],
    [5, 1, 4],
    [6, 8, 3]
])

robots, tasks = cost.shape

print("\n=== MRTA PROBLEM ===")
print("Cost Matrix:")
print(cost)

# =========================================================
# 2. TENSOR NETWORK REPRESENTATION (QUANTUM-INSPIRED LAYER)
# =========================================================

circ = qtn.Circuit(tasks)

# uniform superposition-like initialization
for q in range(tasks):
    circ.h(q)

# encode task difficulty into rotations
for q in range(tasks):
    avg_cost = np.mean(cost[:, q])
    angle = avg_cost / 10.0
    circ.ry(q, angle)

psi = circ.psi

print("\n=== TENSOR NETWORK STATE ===")
print(psi)

# =========================================================
# 3. COST FUNCTION (CLASSICAL + TENSOR-INSPIRED TERM)
# =========================================================

def energy(assignment):
    """
    Energy function:
    - classical assignment cost
    - small tensor-inspired smooth penalty term
    """

    classical = 0.0

    for r in range(robots):
        classical += cost[r, assignment[r]]

    # tensor-inspired global coupling term
    coupling = 0.0

    for i in range(len(assignment)):
        for j in range(i + 1, len(assignment)):

            # encourages diversity in assignments
            if assignment[i] == assignment[j]:
                coupling += 2.5

    # smooth nonlinear modulation (tensor-like interference effect)
    tensor_effect = 0.5 * np.cos(classical / 5.0)

    return classical + coupling + tensor_effect


# =========================================================
# 4. SIMPLE EXHAUSTIVE COMPARISON (CONTROLLED BASELINE)
# =========================================================

from itertools import permutations

best_assignment = None
best_energy = float("inf")

for a in permutations(range(tasks)):
    e = energy(a)

    if e < best_energy:
        best_energy = e
        best_assignment = a

# =========================================================
# 5. RESULTS
# =========================================================

print("\n=== OPTIMIZATION RESULT ===")
print("Best Assignment:", best_assignment)
print("Best Energy:", best_energy)