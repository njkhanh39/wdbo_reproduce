"""Build a WDBO synthetic-benchmark objective: ``f(x, t)`` plus ``oracle(t)``.

Unlike the temperature benchmark there is no data and no interpolation - the
objective *is* a closed-form function (see `benchmarks.py`). The work here is:

1. sign-flip minimized benchmarks so the maximizing DBO loop chases their
   minimum ("activate the hottest point" <-> "find the deepest well");
2. precompute ``oracle(t)`` = best achievable value at each time by a dense
   spatial grid search, cached to disk like the temperature oracle.

Absolute time
-------------
``t`` here is the **environment clock**, in the function's own time units -
for ``ackley4d`` that is the ``[-32, 32]`` axis of Appendix H.2, not a
normalized ``[0, 1]``. `common.run_once` advances it as
``env_start + env_speed * elapsed_seconds`` and hands it straight to
`evaluate` and `oracle`.

An earlier revision indexed the cached oracle by the *run's* normalized clock
and folded the map onto the function's axis (``lo + t * (hi - lo)``) into the
table at build time. That made the table a property of the run rather than of
the function: every environment span needed its own cache file, and because
the sample count was fixed per file, the table's resolution *per unit of
environment time* silently changed whenever the span did. On an oscillatory
benchmark that is an aliasing bug, and it biases ``f*`` downwards -- making
regret look better than it is.

Indexing by absolute time fixes both. `oracle_density` is samples per unit of
environment time, so resolution is a property of the function; and one table
covering a wide span serves every ``(env_speed, duration)`` pair whose
interval falls inside it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from benchmarks import Benchmark

Span = tuple[float, float]

# Samples per unit of environment time in the cached oracle curve. Ackley
# oscillates roughly once per unit of time, so this is ~64 samples per period:
# enough to resolve the peaks a grid search has to find, and the value the
# defaults here are calibrated at. Raise it and check that ``f*`` does not move
# before trusting a faster environment.
DEFAULT_ORACLE_DENSITY = 64.0


@dataclass
class SyntheticObjective:
    """Closed-form dynamic objective ``f(x, t)`` with a cached oracle curve."""

    benchmark: Benchmark
    oracle_times: np.ndarray   # absolute environment times, ascending
    oracle_values: np.ndarray  # best achievable value there, already sign-flipped
    noise_std: float
    spatial_domain: np.ndarray  # (spatial_dim, 2); == benchmark.spatial_domain

    @property
    def _sign(self) -> float:
        return -1.0 if self.benchmark.minimize else 1.0

    @property
    def env_span(self) -> Span:
        """The environment interval the cached oracle covers."""
        return float(self.oracle_times[0]), float(self.oracle_times[-1])

    def evaluate(self, x: np.ndarray, t: float) -> float:
        """Noise-free objective value; higher is better (already sign-flipped).

        ``t`` is an absolute environment time, on the function's own axis.
        """
        z = np.concatenate([np.asarray(x, dtype=float), [float(t)]])[None, :]
        return self._sign * float(self.benchmark.func(z)[0])

    def oracle(self, t: float) -> float:
        """Best achievable noise-free value at environment time ``t``.

        Interpolated from the cached curve. `np.interp` clamps outside the
        cached span rather than raising, so `assert_covers` is what actually
        guards against a run wandering past the table.
        """
        return float(np.interp(t, self.oracle_times, self.oracle_values))

    def assert_covers(self, env_start: float, env_end: float) -> None:
        """Fail loudly if a run would need oracle values the cache does not hold.

        Outside the cached span `oracle` silently returns the nearest endpoint,
        which would report a plausible-looking but wrong regret -- exactly the
        kind of error that survives a whole experiment unnoticed. Checked once
        per run, before any compute is spent.
        """
        lo, hi = self.env_span
        if env_start < lo - 1e-9 or env_end > hi + 1e-9:
            raise SystemExit(
                f"The run covers environment time [{env_start:g}, {env_end:g}], but the cached "
                f"oracle only covers [{lo:g}, {hi:g}].\n"
                f"Either lower --env-speed / --duration-seconds, or widen the table with "
                f"--env-span {min(lo, env_start):g} {max(hi, env_end):g} (this rebuilds the cache)."
            )


def _spatial_grid(benchmark: Benchmark, resolution: int) -> np.ndarray:
    """Regular ``(resolution ** spatial_dim, spatial_dim)`` grid over the box."""
    axes = [np.linspace(lo, hi, resolution) for lo, hi in benchmark.spatial_domain]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=1)


def oracle_time_grid(env_span: Span, density: float) -> np.ndarray:
    """Sample times for the oracle table: ``density`` samples per unit of time.

    Endpoints included, and the sample count rounded up so the requested span
    is always covered at no less than the requested density.
    """
    lo, hi = float(env_span[0]), float(env_span[1])
    if hi <= lo:
        raise ValueError(f"env_span must be increasing, got ({lo}, {hi})")
    return np.linspace(lo, hi, int(np.ceil((hi - lo) * density)) + 1)


def compute_oracle_curve(
    benchmark: Benchmark,
    env_span: Span,
    density: float = DEFAULT_ORACLE_DENSITY,
    grid_resolution: int = 33,
):
    """``oracle(t)`` = max over the spatial grid of the sign-flipped function.

    ``times`` are absolute environment times covering ``env_span`` at
    ``density`` samples per unit.

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

    times = oracle_time_grid(env_span, density)
    best = np.empty(len(times))
    for i, t in enumerate(times):
        z = np.column_stack([grid, np.full(len(grid), t)])
        best[i] = (sign * benchmark.func(z)).max()
    return times, best


def estimate_noise_std(
    benchmark: Benchmark,
    env_span: Span,
    fraction: float = 0.05,
    n_samples: int = 200_000,
    seed: int = 0,
) -> float:
    """``sigma`` such that ``Var(noise) = fraction * signal variance`` (paper H.1).

    Signal variance is estimated by sampling the function uniformly over its
    ``d'`` box: the spatial box, plus the temporal axis over ``env_span``.
    Deterministic given ``seed``.

    Note this depends on ``env_span`` -- the benchmark's whole temporal domain
    -- and not on the sub-interval a particular run visits, so every run of a
    benchmark gets the same noise level and stays comparable across durations
    and environment speeds.
    """
    rng = np.random.default_rng(seed)
    lows = np.append(benchmark.spatial_domain[:, 0], env_span[0])
    highs = np.append(benchmark.spatial_domain[:, 1], env_span[1])
    z = rng.uniform(lows, highs, size=(n_samples, benchmark.dim))
    return float(np.sqrt(fraction * np.var(benchmark.func(z))))


def oracle_cache_name(env_span: Span, density: float, grid_resolution: int) -> str:
    """Filename encoding everything the cached table depends on.

    All three matter: two tables built over different spans, densities or grid
    resolutions are different tables, and reusing one for another is a silent
    wrong answer rather than an error.
    """
    lo, hi = float(env_span[0]), float(env_span[1])
    return f"oracle_t{lo:g}_{hi:g}_d{density:g}_g{grid_resolution}.npz"


def build_objective(
    benchmark: Benchmark,
    env_span: Span | None = None,
    oracle_density: float = DEFAULT_ORACLE_DENSITY,
    oracle_grid_resolution: int = 33,
    oracle_cache_path: Path | None = None,
) -> SyntheticObjective:
    """Assemble the objective, computing + caching the oracle curve on first run.

    ``env_span`` is the environment interval the oracle table covers; it
    defaults to the benchmark's own ``env_span`` (Appendix H.2's box). It is a
    property of the *table*, not of a run: a run may cover any sub-interval of
    it, and `SyntheticObjective.assert_covers` checks that it does.
    """
    span = (float(env_span[0]), float(env_span[1])) if env_span is not None else benchmark.env_span

    if oracle_cache_path is not None and oracle_cache_path.exists():
        cached = np.load(oracle_cache_path)
        oracle_times, oracle_values = cached["times"], cached["values"]
    else:
        oracle_times, oracle_values = compute_oracle_curve(
            benchmark, span, oracle_density, oracle_grid_resolution
        )
        if oracle_cache_path is not None:
            oracle_cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(oracle_cache_path, times=oracle_times, values=oracle_values)

    return SyntheticObjective(
        benchmark=benchmark,
        oracle_times=oracle_times,
        oracle_values=oracle_values,
        noise_std=estimate_noise_std(benchmark, span),
        spatial_domain=benchmark.spatial_domain,
    )
