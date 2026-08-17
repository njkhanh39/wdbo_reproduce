"""Build the WDBO "Temperature" benchmark objective.

The preprocessed sensor readings (see preprocess.py) are an irregular point
cloud: real sensors only exist at ~50 fixed locations and report at uneven
times. The paper turns this into a benchmark DBO can actually query anywhere
by interpolating the data in space-time (Appendix H.2), which is what
`build_objective` below does.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import RBFInterpolator


@dataclass
class TemperatureObjective:
    """A continuous, noisy stand-in for the dynamic objective f(x, t)."""

    interpolator: RBFInterpolator
    oracle_times: np.ndarray
    oracle_values: np.ndarray
    noise_std: float
    spatial_domain: np.ndarray  # (2, 2): preprocess.py already normalizes space to [0, 1]^2

    def evaluate(self, x: np.ndarray, t: float) -> float:
        """Noise-free interpolated temperature at spatial point `x` and time `t`."""
        query = np.concatenate([x, [t]])[None, :]
        return float(self.interpolator(query)[0])

    def oracle(self, t: float) -> float:
        """Best achievable (noise-free) temperature at time `t`, used for regret."""
        return float(np.interp(t, self.oracle_times, self.oracle_values))


def load_processed(path: Path) -> dict:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def compute_oracle_curve(interpolator: RBFInterpolator, n_time_points: int, grid_resolution: int):
    """Grid-search the spatial maximum of `interpolator` over a series of times.

    A dense grid search (rather than a numerical optimizer) keeps this exact
    up to grid resolution and simple: the interpolated surface is cheap to
    batch-evaluate, and this only runs once to build the objective.
    """
    grid_axis = np.linspace(0.0, 1.0, grid_resolution)
    grid_x, grid_y = np.meshgrid(grid_axis, grid_axis)
    grid_xy = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    times = np.linspace(0.0, 1.0, n_time_points)
    best_values = np.empty(n_time_points)
    for i, t in enumerate(times):
        query = np.column_stack([grid_xy, np.full(len(grid_xy), t)])
        best_values[i] = interpolator(query).max()

    return times, best_values


def build_objective(
    processed_path: Path,
    smoothing: float = 1.0,
    oracle_time_points: int = 200,
    oracle_grid_resolution: int = 25,
    oracle_cache_path: Path | None = None,
) -> TemperatureObjective:
    """Build the interpolated objective and its oracle curve.

    The oracle curve is the expensive part (a dense space-time grid search),
    so it is cached to `oracle_cache_path` and reused across runs; the
    interpolator itself is refit every call since it must live in memory to
    answer per-query evaluations.
    """
    data = load_processed(processed_path)
    interpolator = RBFInterpolator(data["points"], data["temperature"], kernel="thin_plate_spline", smoothing=smoothing)
    noise_std = float(np.std(data["temperature"]) * np.sqrt(0.05))

    if oracle_cache_path is not None and oracle_cache_path.exists():
        cached = np.load(oracle_cache_path)
        oracle_times, oracle_values = cached["times"], cached["values"]
    else:
        oracle_times, oracle_values = compute_oracle_curve(interpolator, oracle_time_points, oracle_grid_resolution)
        if oracle_cache_path is not None:
            oracle_cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(oracle_cache_path, times=oracle_times, values=oracle_values)

    return TemperatureObjective(
        interpolator=interpolator,
        oracle_times=oracle_times,
        oracle_values=oracle_values,
        noise_std=noise_std,
        spatial_domain=np.array([[0.0, 1.0], [0.0, 1.0]]),
    )
