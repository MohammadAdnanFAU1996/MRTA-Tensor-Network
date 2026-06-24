import numpy as np
import quimb.tensor as qtn
from itertools import permutations

# -----------------------------
# MRTA COST MATRIX
# -----------------------------
cost = np.array([
    [3, 7, 2],
    [5, 1, 4],
    [6, 8, 3]
])

robots = cost.shape[0]
tasks = cost.shape[1]

print("Robot-Task Cost Matrix")
print(cost)
print()

# -----------------------------
# Tensor-network style circuit
# -----------------------------
circ = qtn.Circuit(tasks)

for q in range(tasks):
    circ.h(q)

for q in range(tasks):
    angle = np.mean(cost[:, q]) / 10
    circ.ry(q, angle)

psi = circ.psi

print("Tensor Network State")
print(psi)
print()

# -----------------------------
# Classical optimization (MRTA solution)
# -----------------------------
best_cost = float("inf")
best_assignment = None

for assignment in permutations(range(tasks)):
    total = 0
    for r in range(robots):
        total += cost[r][assignment[r]]

    if total < best_cost:
        best_cost = total
        best_assignment = assignment

print("Best Assignment:", best_assignment)
print("Minimum Cost:", best_cost)