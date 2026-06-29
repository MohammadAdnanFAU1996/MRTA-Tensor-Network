import random
import time
from itertools import permutations

import matplotlib.pyplot as plt
import numpy as np
import quimb.tensor as qtn


# =========================================================
# 1. MRTA PROBLEM SETTING
# =========================================================

cost = np.array([
    [9, 10, 15, 19, 1],
    [19, 5, 6, 17, 9],
    [5, 8, 13, 11, 2],
    [15, 16, 11, 16, 7],
    [3, 6, 3, 9, 19],
])

robots, tasks = cost.shape


def assignment_cost(assignment):
    return float(sum(cost[r, assignment[r]] for r in range(robots)))


# =========================================================
# 2. TENSOR LAYER
# =========================================================

def build_tensor_layer():
    circ = qtn.Circuit(tasks)

    for q in range(tasks):
        circ.h(q)

    task_avg_cost = np.mean(cost, axis=0)
    max_cost = np.max(task_avg_cost)

    # Lower average task cost gets a larger rotation.
    angles = (max_cost - task_avg_cost + 1.0) / max_cost

    for q, angle in enumerate(angles):
        circ.ry(q, float(angle))

    psi = circ.psi

    # Extract real probabilities from the tensor-network state.
    amplitudes = np.asarray(psi.to_dense()).reshape(-1)
    probabilities = np.abs(amplitudes) ** 2
    probabilities = probabilities / probabilities.sum()

    task_scores = np.zeros(tasks)

    for basis_index, probability in enumerate(probabilities):
        bitstring = format(basis_index, f"0{tasks}b")

        for q, bit in enumerate(reversed(bitstring)):
            if bit == "1":
                task_scores[q] += probability

    return circ, psi, task_scores


circ, psi, task_scores = build_tensor_layer()


def tensor_guided_assignment():
    start = time.perf_counter()

    remaining_tasks = set(range(tasks))
    assignment = []

    for r in range(robots):
        selected_task = min(
            remaining_tasks,
            key=lambda t: cost[r, t] / (task_scores[t] + 1e-12),
        )

        assignment.append(selected_task)
        remaining_tasks.remove(selected_task)

    runtime = time.perf_counter() - start
    return tuple(assignment), assignment_cost(assignment), runtime


# =========================================================
# 3. THREE CLASSICAL ALGORITHMS
# =========================================================

def exhaustive_search():
    start = time.perf_counter()

    best_assignment = None
    best_cost = float("inf")

    for assignment in permutations(range(tasks)):
        current_cost = assignment_cost(assignment)

        if current_cost < best_cost:
            best_cost = current_cost
            best_assignment = tuple(assignment)

    runtime = time.perf_counter() - start
    return best_assignment, best_cost, runtime


def greedy_assignment():
    start = time.perf_counter()

    remaining_tasks = set(range(tasks))
    assignment = []

    for r in range(robots):
        selected_task = min(remaining_tasks, key=lambda t: cost[r, t])
        assignment.append(selected_task)
        remaining_tasks.remove(selected_task)

    runtime = time.perf_counter() - start
    return tuple(assignment), assignment_cost(assignment), runtime


def random_search(samples=1000, seed=1):
    rng = random.Random(seed)
    start = time.perf_counter()

    best_assignment = None
    best_cost = float("inf")

    for _ in range(samples):
        assignment = list(range(tasks))
        rng.shuffle(assignment)

        current_cost = assignment_cost(assignment)

        if current_cost < best_cost:
            best_cost = current_cost
            best_assignment = tuple(assignment)

    runtime = time.perf_counter() - start
    return best_assignment, best_cost, runtime


def simulated_annealing(iterations=2000, temp=10.0, cooling=0.995, seed=2):
    rng = random.Random(seed)
    start = time.perf_counter()

    current = list(range(tasks))
    rng.shuffle(current)
    current_cost = assignment_cost(current)

    best_assignment = current[:]
    best_cost = current_cost

    for _ in range(iterations):
        candidate = current[:]

        i, j = rng.sample(range(tasks), 2)
        candidate[i], candidate[j] = candidate[j], candidate[i]

        candidate_cost = assignment_cost(candidate)
        delta = candidate_cost - current_cost

        if delta < 0 or rng.random() < np.exp(-delta / temp):
            current = candidate
            current_cost = candidate_cost

            if current_cost < best_cost:
                best_assignment = current[:]
                best_cost = current_cost

        temp *= cooling

    runtime = time.perf_counter() - start
    return tuple(best_assignment), best_cost, runtime


# =========================================================
# 4. RUN EXPERIMENT
# =========================================================

results = []

for name, solver in [
    ("Exhaustive Search", exhaustive_search),
    ("Greedy Assignment", greedy_assignment),
    ("Random Search", random_search),
    ("Simulated Annealing", simulated_annealing),
    ("Tensor-Guided", tensor_guided_assignment),
]:
    assignment, total_cost, runtime = solver()

    results.append({
        "name": name,
        "assignment": assignment,
        "cost": total_cost,
        "runtime": runtime,
    })


# =========================================================
# 5. PRINT RESULTS
# =========================================================

print("\n=== 1. MRTA PROBLEM SETTING ===")
print("Rows = robots, columns = tasks, values = assignment costs")
print(cost)

print("\n=== 2. TENSOR LAYER ===")
print("Number of task qubits:", tasks)
print("Tensor-network state type:", type(psi).__name__)
print("Task scores from tensor-state probabilities:")
print(np.round(task_scores, 4))

print("\n=== 3. COMPARISON WITH CLASSICAL ALGORITHMS ===")

for row in results:
    print(
        f"{row['name']:20s} | "
        f"assignment={row['assignment']} | "
        f"cost={row['cost']:.2f} | "
        f"runtime={row['runtime']:.6f}s"
    )


# =========================================================
# 6. PLOT
# =========================================================

names = [row["name"] for row in results]
costs = [row["cost"] for row in results]
runtimes = [row["runtime"] for row in results]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"]

axes[0].bar(names, costs, color=colors)
axes[0].set_title("MRTA Allocation Cost")
axes[0].set_ylabel("Total Cost")
axes[0].tick_params(axis="x", rotation=25)

axes[1].bar(names, runtimes, color=colors)
axes[1].set_title("Runtime")
axes[1].set_ylabel("Seconds")
axes[1].tick_params(axis="x", rotation=25)

fig.suptitle("MRTA: Tensor Layer and Classical Algorithm Comparison")
fig.tight_layout()

plt.show()