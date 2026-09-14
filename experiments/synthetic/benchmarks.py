"""Analytic synthetic benchmarks from Appendix H.2 of the WDBO paper.

Every synthetic benchmark in the paper is a standard global-optimization test
function whose LAST input axis is reinterpreted as time: the paper writes each
one as ``f(z)`` with ``z = (x_1, ..., x_d, t)`` and ``d' = d + 1``.

Two scale conventions, both from Appendix H.1:

* the ``d`` spatial axes use the function's natural box (e.g. Ackley on
  ``[-32, 32]^d``);
* the temporal axis is **normalized to ``[0, 1]``** - it is *not* rescaled to
  the spatial box. (Rescaling it, as an over-literal reading of "optimized on
  the domain ``[lo, hi]^d'``" would suggest, makes oscillatory benchmarks
  like Ackley flip sign many times across the horizon, destroying the
  temporal correlation W-DBO exists to exploit - and contradicts the
  moderate dataset sizes W-DBO shows in the paper's figures.)

So a benchmark here is a vectorized function, its ``d`` spatial box, and
whether it is minimized. `objective.py` turns one into the ``f(x, t)`` /
``oracle(t)`` pair the experiment loop needs; `run_experiment.py` selects one
with ``--benchmark``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

FuncT = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Benchmark:
    """A synthetic test function lifted into a spatio-temporal objective.

    Attributes:
        name: registry key, also the results sub-directory name.
        spatial_domain: ``(d, 2)`` natural box for the spatial axes.
        func: vectorized, maps an ``(n, d + 1)`` array to ``(n,)``. The last
            column is time in ``[0, 1]`` (already on the same footing as the
            other columns numerically - no internal rescale).
        minimize: ``True`` if the task is to find the function's minimum. The
            paper's synthetic functions are all minimized while the DBO loop
            maximizes, so `objective.py` negates the objective accordingly.
    """

    name: str
    spatial_domain: np.ndarray
    func: FuncT
    minimize: bool = True

    @property
    def spatial_dim(self) -> int:
        return self.spatial_domain.shape[0]

    @property
    def dim(self) -> int:
        """``d'`` = spatial dimensions + 1 temporal dimension."""
        return self.spatial_dim + 1

    @property
    def domain(self) -> np.ndarray:
        """``(d', 2)``: the spatial box rows plus the temporal ``[0, 1]`` row."""
        return np.vstack([self.spatial_domain, [[0.0, 1.0]]])


def ackley(z: np.ndarray, a: float = 20.0, b: float = 0.2, c: float = 2.0 * np.pi) -> np.ndarray:
    """Ackley function, averaged over all ``d'`` axes (WDBO paper H.2 form).

    Global minimum ``0`` at the origin; a nearly flat outer region riddled
    with local minima surrounding one deep central well. The paper highlights
    Ackley because most DBO baselines never find that well.

    With time normalized to ``[0, 1]`` while space spans ``[-32, 32]``, the
    spatial argmin stays at the origin for every ``t`` (both terms are
    minimized there regardless of the fixed time coordinate); the time axis
    enters only as a smooth, single-period ripple in the achievable value.
    """
    z = np.atleast_2d(np.asarray(z, dtype=float))
    radial = np.sqrt(np.mean(z ** 2, axis=1))
    oscillation = np.mean(np.cos(c * z), axis=1)
    return -a * np.exp(-b * radial) - np.exp(oscillation) + a + np.e


ACKLEY4D = Benchmark(
    name="ackley4d",
    spatial_domain=np.array([[-32.0, 32.0]] * 3),
    func=ackley,
    minimize=True,
)


BENCHMARKS: dict[str, Benchmark] = {b.name: b for b in (ACKLEY4D,)}


def get_benchmark(name: str) -> Benchmark:
    try:
        return BENCHMARKS[name]
    except KeyError:
        known = ", ".join(sorted(BENCHMARKS)) or "(none registered)"
        raise SystemExit(f"Unknown benchmark {name!r}. Available: {known}")
