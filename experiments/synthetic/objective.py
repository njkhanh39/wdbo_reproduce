"""Build a WDBO synthetic-benchmark objective: ``f(x, t)`` plus ``oracle(t)``.

Unlike the temperature benchmark there is no data and no interpolation - the
objective *is* a closed-form function (see `benchmarks.py`). The work here is:

1. map the optimizer's normalized clock ``t in [0, 1]`` onto the function's
   last input axis. ``temporal_span`` controls that map: the default
   ``(0.0, 1.0)`` is the identity (paper H.1's "temporal domain normalized in
   [0, 1]"); passing e.g. ``(-32.0, 32.0)`` instead puts time on the same box
   as the spatial axes (the literal "optimized on [lo, hi]^d'" reading), which
   makes oscillatory benchmarks change much faster in time;
2. sign-flip minimized benchmarks so the maximizing DBO loop chases their
   minimum ("activate the hottest point" <-> "find the deepest well");
3. precompute ``oracle(t)`` = best achievable value at each time by a dense
   spatial grid search, cached to disk like the temperature oracle.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from benchmarks import Benchmark

Span = tuple[float, float]


@dataclass
class SyntheticObjective:
    """Closed-form dynamic objective ``f(x, t)`` with a cached oracle curve."""

    benchmark: Benchmark
    oracle_times: np.ndarray   # normalized clock samples in [0, 1]
    oracle_values: np.ndarray  # best achievable value, already sign-flipped
    noise_std: float
    spatial_domain: np.ndarray  # (spatial_dim, 2); == benchmark.spatial_domain
    temporal_span: Span = (0.0, 1.0)  # normalized clock [0, 1] is mapped onto this

    @property
    def _sign(self) -> float:
        return -1.0 if self.benchmark.minimize else 1.0

    def _scale_time(self, t: float) -> float:
        lo, hi = self.temporal_span
        return lo + t * (hi - lo)

    def evaluate(self, x: np.ndarray, t: float) -> float:
        """Noise-free objective value; higher is better (already sign-flipped)."""
        z = np.concatenate([np.asarray(x, dtype=float), [self._scale_time(t)]])[None, :]
        return self._sign * float(self.benchmark.func(z)[0])

    def oracle(self, t: float) -> float:
        """Best achievable noise-free value at time ``t`` (used for regret).

        ``t`` is the normalized clock in ``[0, 1]``; the cached curve already
        accounts for ``temporal_span``.
        """
        return float(np.interp(t, self.oracle_times, self.oracle_values))


def _spatial_grid(benchmark: Benchmark, resolution: int) -> np.ndarray:
    """Regular ``(resolution ** spatial_dim, spatial_dim)`` grid over the box."""
    axes = [np.linspace(lo, hi, resolution) for lo, hi in benchmark.spatial_domain]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=1)


def compute_oracle_curve(
    benchmark: Benchmark,
    n_time_points: int,
    grid_resolution: int,
    temporal_span: Span = (0.0, 1.0),
):
    """``oracle(t)`` = max over the spatial grid of the sign-flipped function.

    ``times`` is returned as the normalized clock in ``[0, 1]``; the function
    is evaluated with that clock mapped through ``temporal_span``.

    A dense grid search (rather than a numerical optimizer) keeps this exact
    up to grid resolution, is trivially vectorized, and only runs once before
    being cached. Prefer an ODD ``grid_resolution`` so a symmetric-domain
    optimum (e.g. Ackley's origin) lands exactly on a grid node.

    Only practical while ``spatial_dim`` is small (<= ~3): the grid has
    ``grid_resolution ** spatial_dim`` nodes. Higher-dimensional benchmarks
    need an optimizer-based oracle instead.
    """
    sign = -1.0 if benchmark.minimize else 1.0
    grid = _spatial_grid(benchmark, grid_resolution)
    lo, hi = temporal_span

    times = np.linspace(0.0, 1.0, n_time_points)
    best = np.empty(n_time_points)
    for i, t in enumerate(times):
        z = np.column_stack([grid, np.full(len(grid), lo + t * (hi - lo))])
        best[i] = (sign * benchmark.func(z)).max()
    return times, best


def estimate_noise_std(
    benchmark: Benchmark,
    temporal_span: Span = (0.0, 1.0),
    fraction: float = 0.05,
    n_samples: int = 200_000,
    seed: int = 0,
) -> float:
    """``sigma`` such that ``Var(noise) = fraction * signal variance`` (paper H.1).

    Signal variance is estimated by sampling the function uniformly over its
    ``d'`` box: the spatial box, plus the temporal axis over ``temporal_span``.
    Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    lows = np.append(benchmark.spatial_domain[:, 0], temporal_span[0])
    highs = np.append(benchmark.spatial_domain[:, 1], temporal_span[1])
    z = rng.uniform(lows, highs, size=(n_samples, benchmark.dim))
    return float(np.sqrt(fraction * np.var(benchmark.func(z))))


def build_objective(
    benchmark: Benchmark,
    temporal_span: Span = (0.0, 1.0),
    oracle_time_points: int = 1000,
    oracle_grid_resolution: int = 33,
    oracle_cache_path: Path | None = None,
) -> SyntheticObjective:
    """Assemble the objective, computing + caching the oracle curve on first run."""
    temporal_span = (float(temporal_span[0]), float(temporal_span[1]))

    if oracle_cache_path is not None and oracle_cache_path.exists():
        cached = np.load(oracle_cache_path)
        oracle_times, oracle_values = cached["times"], cached["values"]
    else:
        oracle_times, oracle_values = compute_oracle_curve(
            benchmark, oracle_time_points, oracle_grid_resolution, temporal_span
        )
        if oracle_cache_path is not None:
            oracle_cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(oracle_cache_path, times=oracle_times, values=oracle_values)

    return SyntheticObjective(
        benchmark=benchmark,
        oracle_times=oracle_times,
        oracle_values=oracle_values,
        noise_std=estimate_noise_std(benchmark, temporal_span),
        spatial_domain=benchmark.spatial_domain,
        temporal_span=temporal_span,
    )
