"""Shared location for all synthetic-benchmark artifacts.

Each benchmark gets its own sub-directory under `DATA_DIR` holding its cached
oracle curve and its `results/` folder, e.g. `data/synthetic/ackley4d/`.
"""
from pathlib import Path

DATA_DIR = Path("data/synthetic")
