"""Build the WDBO "Temperature" benchmark objective.

The preprocessed sensor readings (see preprocess.py) are an irregular point
cloud: real sensors only exist at ~50 fixed locations and report at uneven
times. The paper turns this into a benchmark DBO can actually query anywhere
by interpolating the data in space-time (Appendix H.2), which is what
`build_objective` below does.

Environment time here is already absolute and already `[0, 1]`: preprocess.py
normalizes the first calendar day of readings onto that interval, so
`ENV_SPAN` is a hard fact about the data, not a convention. A run must stay
inside it -- `RBFInterpolator` will happily extrapolate past the last reading
and return confident nonsense -- which is what `assert_covers` enforces. That
caps how far this benchmark can be pushed: at the default environment speed
one 600s run consumes the whole day, so a longer run needs a proportionally
lower `--env-speed`, not more data.
"""
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import RBFInterpolator

# The environment interval the data supports, fixed by preprocess.py's
# normalization of the first calendar day onto [0, 1].
ENV_SPAN = (0.0, 1.0)

# Oracle table samples per unit of environment time. The interpolated surface
# is far smoother in time than Ackley is, so it needs far fewer; this matches
# the 200-point table the earlier revision built over [0, 1].
DEFAULT_ORACLE_DENSITY = 200.0

# Points per axis of the spatial grid the oracle searches over [0, 1]^2.
DEFAULT_ORACLE_GRID_RESOLUTION = 25


@dataclass
class TemperatureObjective:
    """A continuous, noisy stand-in for the dynamic objective f(x, t)."""

    interpolator: RBFInterpolator
    oracle_times: np.ndarray
    oracle_values: np.ndarray
    noise_std: float
    spatial_domain: np.ndarray  # (2, 2): preprocess.py already normalizes space to [0, 1]^2

    @property
    def env_span(self) -> tuple[float, float]:
        """The environment interval the cached oracle covers."""
        return float(self.oracle_times[0]), float(self.oracle_times[-1])

    def evaluate(self, x: np.ndarray, t: float) -> float:
        """Noise-free interpolated temperature at spatial point `x` and time `t`."""
        query = np.concatenate([x, [t]])[None, :]
        return float(self.interpolator(query)[0])

    def oracle(self, t: float) -> float:
        """Best achievable (noise-free) temperature at time `t`, used for regret."""
        return float(np.interp(t, self.oracle_times, self.oracle_values))

    def assert_covers(self, env_start: float, env_end: float) -> None:
        """Fail loudly if a run would run off the end of the sensor data.

        Unlike the synthetic benchmark, this cannot be fixed by rebuilding a
        wider table: there is no more data. `RBFInterpolator` extrapolates
        silently and `np.interp` clamps, so without this check a run past
        `t = 1` would report regret against a frozen, fictional environment.
        """
        lo, hi = self.env_span
        if env_start < lo - 1e-9 or env_end > hi + 1e-9:
            raise SystemExit(
                f"The run covers environment time [{env_start:g}, {env_end:g}], but the sensor data "
                f"only covers [{lo:g}, {hi:g}], and there is no more of it.\n"
                f"Lower --env-speed to about {(hi - lo) / (env_end - env_start):.4g} times its current "
                f"value, or shorten --duration-seconds by the same factor."
            )


def load_processed(path: Path) -> dict:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def compute_oracle_curve(interpolator: RBFInterpolator, density: float, grid_resolution: int):
    """Grid-search the spatial maximum of `interpolator` over a series of times.

    `density` is samples per unit of environment time, matching the synthetic
    benchmark's convention so the two tables mean the same thing. Since
    `ENV_SPAN` is `[0, 1]` here, it is also simply the sample count minus one.

    A dense grid search (rather than a numerical optimizer) keeps this exact
    up to grid resolution and simple: the interpolated surface is cheap to
    batch-evaluate, and this only runs once to build the objective.
    """
    grid_axis = np.linspace(0.0, 1.0, grid_resolution)
    grid_x, grid_y = np.meshgrid(grid_axis, grid_axis)
    grid_xy = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    lo, hi = ENV_SPAN
    times = np.linspace(lo, hi, int(np.ceil((hi - lo) * density)) + 1)
    best_values = np.empty(len(times))
    for i, t in enumerate(times):
        query = np.column_stack([grid_xy, np.full(len(grid_xy), t)])
        best_values[i] = interpolator(query).max()

    return times, best_values


def oracle_cache_name(density: float, grid_resolution: int, smoothing: float) -> str:
    """Filename encoding every setting the cached table depends on.

    The span is fixed by the data, so it is not in the name. Smoothing is:
    it changes the RBF surface itself, so a table built under one smoothing
    and read under another grades `f` against a different function's
    maximum -- silently wrong regret, not an error.
    """
    return f"oracle_d{density:g}_g{grid_resolution}_s{smoothing:g}.npz"


def data_fingerprint(data: dict) -> str:
    """SHA-256 of the point cloud the RBF is fitted to.

    A filename cannot tell that `processed.npz` was regenerated with other
    preprocessing options, so the table stores this and `build_objective`
    checks it on load.
    """
    digest = hashlib.sha256()
    for key in ("points", "temperature"):
        digest.update(np.ascontiguousarray(data[key], dtype=np.float64).tobytes())
    return digest.hexdigest()


def _check_cache(path: Path, cached, expected: dict) -> None:
    """Refuse a cached table built under different settings or data.

    Tables written before these keys were stored carry none of them and are
    accepted unchecked, so older runs whose `run.json` names them still
    re-score.
    """
    stale = [f"{key}: cached {cached[key].item()!r}, requested {value!r}"
             for key, value in expected.items()
             if key in cached.files and cached[key].item() != value]
    if stale:
        raise SystemExit(
            f"The cached oracle {path} was built under different settings:\n  "
            + "\n  ".join(stale)
            + "\nDelete it, or pass a different --oracle-cache, and the table will be rebuilt."
        )


def build_objective(
    processed_path: Path,
    smoothing: float = 1.0,
    oracle_density: float = DEFAULT_ORACLE_DENSITY,
    oracle_grid_resolution: int = DEFAULT_ORACLE_GRID_RESOLUTION,
    oracle_cache_path: Path | None = None,
) -> TemperatureObjective:
    """Build the interpolated objective and its oracle curve.

    The oracle curve is the expensive part (a dense space-time grid search),
    so it is cached to `oracle_cache_path` and reused across runs; the
    interpolator itself is refit every call since it must live in memory to
    answer per-query evaluations.

    The table is saved with the settings and data fingerprint it was built
    under, and a cache that disagrees with the requested ones is refused
    rather than read (see `_check_cache`).
    """
    data = load_processed(processed_path)
    interpolator = RBFInterpolator(data["points"], data["temperature"], kernel="thin_plate_spline", smoothing=smoothing)
    noise_std = float(np.std(data["temperature"]) * np.sqrt(0.05))
    settings = {
        "density": float(oracle_density),
        "grid_resolution": int(oracle_grid_resolution),
        "smoothing": float(smoothing),
        "data_sha256": data_fingerprint(data),
    }

    if oracle_cache_path is not None and oracle_cache_path.exists():
        cached = np.load(oracle_cache_path)
        _check_cache(oracle_cache_path, cached, settings)
        oracle_times, oracle_values = cached["times"], cached["values"]
    else:
        oracle_times, oracle_values = compute_oracle_curve(interpolator, oracle_density, oracle_grid_resolution)
        if oracle_cache_path is not None:
            oracle_cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(oracle_cache_path, times=oracle_times, values=oracle_values, **settings)

    return TemperatureObjective(
        interpolator=interpolator,
        oracle_times=oracle_times,
        oracle_values=oracle_values,
        noise_std=noise_std,
        spatial_domain=np.array([[0.0, 1.0], [0.0, 1.0]]),
    )
