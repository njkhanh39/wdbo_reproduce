"""Analytic synthetic benchmarks from Appendix H.2 of the WDBO paper.

Every synthetic benchmark in the paper is a standard global-optimization test
function whose LAST input axis is reinterpreted as time: the paper writes each
one as ``f(z)`` with ``z = (x_1, ..., x_d, t)`` and ``d' = d + 1``.

Scale conventions. Appendix H.2 gives each benchmark a single box covering
**all** ``d'`` axes - for Ackley, "we optimized the function on the domain
``[-32, 32]^d'``" with ``d' = 4`` and ``z = (x_1, ..., x_d, t)``. So the
function's time argument spans the same box as its spatial ones, and that is
what ``env_span`` records here: the benchmark's whole temporal domain, in the
function's own units.

Appendix H.1's "the temporal domain is normalized in ``[0, 1]``" describes a
convenience of the reference implementation, not the function. `WDBOOptimizer`
normalizes space itself but takes time as given, and this repo now runs it on
absolute environment time (see `common.py`'s two-clock note), so ``env_span``
is the axis the experiment really moves along. Its width divided by 600 s is
the environment speed the paper's setting implies, which is what
`common.default_env_speed` returns.

(Reading H.1 as a statement about the function instead - time literally in
``[0, 1]`` while space spans ``[-32, 32]`` - makes Ackley far smoother in
time than in space. Pass ``--env-span 0 1`` to run that variant; it is a
sensitivity check, not the paper's setting. Note it changes the default
environment speed too, since that is derived from the span.)

So a benchmark here is a vectorized function, its ``d`` spatial box, the range
its time axis spans, and whether it is minimized. `objective.py` turns one into
the ``f(x, t)`` / ``oracle(t)`` pair the experiment loop needs;
`run_experiment.py` selects one with ``--benchmark``.
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
            column is absolute environment time, in the function's own units.
        env_span: the range the function's time axis covers (Appendix
            H.2's box, applied to the ``(d + 1)``th axis). Sets the default
            oracle table and, via its width, the default environment speed.
        minimize: ``True`` if the task is to find the function's minimum. The
            paper's synthetic functions are all minimized while the DBO loop
            maximizes, so `objective.py` negates the objective accordingly.
    """

    name: str
    spatial_domain: np.ndarray
    func: FuncT
    env_span: tuple[float, float] = (0.0, 1.0)
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
        """``(d', 2)``: the spatial box rows plus the temporal row."""
        return np.vstack([self.spatial_domain, [list(self.env_span)]])


def ackley(z: np.ndarray, a: float = 20.0, b: float = 0.2, c: float = 2.0 * np.pi) -> np.ndarray:
    """Ackley function, averaged over all ``d'`` axes (WDBO paper H.2 form).

    Global minimum ``0`` at the origin; a nearly flat outer region riddled
    with local minima surrounding one deep central well. The paper highlights
    Ackley because most DBO baselines never find that well.

    The spatial argmin stays at the origin for every ``t`` - both terms are
    minimized there whatever the fixed time coordinate - so the time axis
    only modulates the best achievable value, not where it is found.
    """
    z = np.atleast_2d(np.asarray(z, dtype=float))
    radial = np.sqrt(np.mean(z ** 2, axis=1))
    oscillation = np.mean(np.cos(c * z), axis=1)
    return -a * np.exp(-b * radial) - np.exp(oscillation) + a + np.e


ACKLEY4D = Benchmark(
    name="ackley4d",
    spatial_domain=np.array([[-32.0, 32.0]] * 3),
    func=ackley,
    env_span=(-32.0, 32.0),  # Appendix H.2: the [-32, 32] box covers all d' = 4 axes
    minimize=True,
)


BENCHMARKS: dict[str, Benchmark] = {b.name: b for b in (ACKLEY4D,)}


def get_benchmark(name: str) -> Benchmark:
    try:
        return BENCHMARKS[name]
    except KeyError:
        known = ", ".join(sorted(BENCHMARKS)) or "(none registered)"
        raise SystemExit(f"Unknown benchmark {name!r}. Available: {known}")
