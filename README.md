# MRTA-Tensor-Network
Tensor-network-inspired MRTA optimization framework
# MRTA Tensor-Network Simulation Framework

This project implements a Multi-Robot Task Allocation (MRTA) system using tensor-network-inspired quantum circuit encoding and classical/stochastic optimization.

Tensor networks are built using the quimb library.

## Versions

- V1: Exact MRTA solver (brute force)
- V2: Energy-based MRTA formulation
- V3: Scalable stochastic optimization (simulated annealing)

## Run

pip install -r requirements.txt

python v1/mrta_v1.py
python v2/mrta_v2.py
python v3/mrta_v3.py
