"""PyTorch implementations of the synthetic benchmarks from Appendix H.2.

The paper defines each function on the complete native spatio-temporal vector
``z = (x_1, ..., x_d, t)``.  :class:`DynamicBenchmark` keeps that convention,
while providing helpers for the normalized ``[0, 1]`` coordinates used by the
BoTorch optimization loop.

The formulas below are losses (smaller is better), exactly as printed in the
paper.  W-DBO uses an upper-confidence-bound acquisition function, so use
``benchmark.objective(...)`` when passing observations to the maximizer; it
returns the negative loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module


TensorFunction = Callable[[Tensor], Tensor]


def _check_last_dimension(z: Tensor, expected: Optional[int] = None) -> Tensor:
	if not torch.is_tensor(z):
		z = torch.as_tensor(z, dtype=torch.get_default_dtype())
	if not (z.is_floating_point() or z.is_complex()):
		z = z.to(dtype=torch.get_default_dtype())
	if z.ndim == 0:
		raise ValueError("benchmark inputs must have at least one dimension")
	if expected is not None and z.shape[-1] != expected:
		raise ValueError(
			f"expected an input with last dimension {expected}, got {z.shape[-1]}"
		)
	return z


def ackley(z: Tensor) -> Tensor:
	"""Ackley with ``a=20``, ``b=0.2`` and ``c=2*pi``."""
	z = _check_last_dimension(z)
	d = z.shape[-1]
	mean_square = z.square().mean(dim=-1)
	mean_cosine = torch.cos(2.0 * torch.pi * z).mean(dim=-1)
	return -20.0 * torch.exp(-0.2 * torch.sqrt(mean_square)) - torch.exp(
		mean_cosine
	) + 20.0 + torch.exp(z.new_tensor(1.0))


def rastrigin(z: Tensor) -> Tensor:
	"""Rastrigin with ``a=10``."""
	z = _check_last_dimension(z)
	return 10.0 * z.shape[-1] + (
		z.square() - 10.0 * torch.cos(2.0 * torch.pi * z)
	).sum(dim=-1)


def schwefel(z: Tensor) -> Tensor:
	z = _check_last_dimension(z)
	return 418.9829 * z.shape[-1] - (
		z * torch.sin(torch.sqrt(torch.abs(z)))
	).sum(dim=-1)


def styblinski_tang(z: Tensor) -> Tensor:
	z = _check_last_dimension(z)
	return 0.5 * (z.pow(4) - 16.0 * z.square() + 5.0 * z).sum(dim=-1)


def eggholder(z: Tensor) -> Tensor:
	z = _check_last_dimension(z, expected=2)
	z1, z2 = z.unbind(dim=-1)
	return -(z2 + 47.0) * torch.sin(
		torch.sqrt(torch.abs(z2 + z1 / 2.0 + 47.0))
	) - z1 * torch.sin(torch.sqrt(torch.abs(z1 - z2 - 47.0)))


def rosenbrock(z: Tensor) -> Tensor:
	z = _check_last_dimension(z)
	if z.shape[-1] < 2:
		raise ValueError("Rosenbrock requires at least two dimensions")
	return (
		100.0 * (z[..., 1:] - z[..., :-1].square()).square()
		+ (z[..., :-1] - 1.0).square()
	).sum(dim=-1)


_SHEKEL_BETA = (0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5)
_SHEKEL_C = (
	(4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0),
	(4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6),
	(4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0),
	(4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6),
)


def shekel(z: Tensor) -> Tensor:
	z = _check_last_dimension(z, expected=4)
	c = z.new_tensor(_SHEKEL_C)
	beta = z.new_tensor(_SHEKEL_BETA)
	denominator = (z.unsqueeze(-1) - c).square().sum(dim=-2) + beta
	return -(1.0 / denominator).sum(dim=-1)


_H_ALPHA = (1.0, 1.2, 3.0, 3.2)
_H3_A = (
	(3.0, 10.0, 30.0),
	(0.1, 10.0, 35.0),
	(3.0, 10.0, 30.0),
	(0.1, 10.0, 35.0),
)
_H3_P = (
	(0.3689, 0.1170, 0.2673),
	(0.4699, 0.4387, 0.7470),
	(0.1091, 0.8732, 0.5547),
	(0.0381, 0.5743, 0.8828),
)
_H6_A = (
	(10.0, 3.0, 17.0, 3.5, 1.7, 8.0),
	(0.05, 10.0, 17.0, 0.1, 8.0, 14.0),
	(3.0, 3.5, 1.7, 10.0, 17.0, 8.0),
	(17.0, 8.0, 0.05, 10.0, 0.1, 14.0),
)
_H6_P = (
	(0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886),
	(0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991),
	(0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650),
	(0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381),
)


def _hartmann(z: Tensor, a_values: Tuple[Tuple[float, ...], ...], p_values: Tuple[Tuple[float, ...], ...]) -> Tensor:
	z = _check_last_dimension(z, expected=len(a_values[0]))
	a = z.new_tensor(a_values)
	p = z.new_tensor(p_values)
	alpha = z.new_tensor(_H_ALPHA)
	exponent = -(a * (z.unsqueeze(-2) - p).square()).sum(dim=-1)
	return -(alpha * torch.exp(exponent)).sum(dim=-1)


def hartmann3(z: Tensor) -> Tensor:
	return _hartmann(z, _H3_A, _H3_P)


def hartmann6(z: Tensor) -> Tensor:
	return _hartmann(z, _H6_A, _H6_P)


def powell(z: Tensor) -> Tensor:
	z = _check_last_dimension(z)
	if z.shape[-1] % 4:
		raise ValueError("Powell requires a dimension divisible by four")
	groups = z.reshape(*z.shape[:-1], -1, 4)
	a, b, c, d = groups.unbind(dim=-1)
	return (
		(a + 10.0 * b).square()
		+ 5.0 * (c - d).square()
		+ (b - 2.0 * c).pow(4)
		+ 10.0 * (a - d).pow(4)
	).sum(dim=-1)


@dataclass(eq=False)
class DynamicBenchmark(Module):
	"""A differentiable, batched, BoTorch-compatible dynamic test problem.

	``evaluate`` accepts normalized spatial points and normalized time and returns
	the paper's loss. ``objective`` returns its negative for W-DBO's maximization
	API. ``forward`` evaluates full *native* spatio-temporal vectors, matching the
	usual BoTorch test-function calling convention.
	"""

	name: str
	spatial_dimension: int
	lower: float
	upper: float
	function: TensorFunction
	known_spatial_optimum: Optional[Tuple[float, ...]] = None

	def __post_init__(self) -> None:
		Module.__init__(self)
		if self.spatial_dimension < 1:
			raise ValueError("spatial_dimension must be positive")
		if self.lower >= self.upper:
			raise ValueError("lower must be smaller than upper")
		self.register_buffer(
			"_bounds",
			torch.tensor(
				[[self.lower] * self.dim, [self.upper] * self.dim],
				dtype=torch.double,
			),
		)

	@property
	def dim(self) -> int:
		"""Total dimension ``d + 1``, including time."""
		return self.spatial_dimension + 1

	@property
	def bounds(self) -> Tensor:
		"""Native BoTorch bounds with shape ``2 x (d + 1)``."""
		return self._bounds

	@property
	def spatial_bounds(self) -> Tensor:
		"""Native spatial bounds with shape ``2 x d``."""
		return self.bounds[:, : self.spatial_dimension]

	def to_native_space(self, normalized_x: Tensor) -> Tensor:
		x = _check_last_dimension(normalized_x, expected=self.spatial_dimension)
		return self.lower + x * (self.upper - self.lower)

	def temporal_coordinate(self, normalized_time: Tensor, *, like: Optional[Tensor] = None) -> Tensor:
		if like is None:
			time = torch.as_tensor(normalized_time, dtype=torch.get_default_dtype())
		else:
			time = torch.as_tensor(normalized_time, dtype=like.dtype, device=like.device)
		return self.lower + time * (self.upper - self.lower)

	def forward(self, z: Tensor) -> Tensor:
		"""Evaluate the paper's loss on full native spatio-temporal vectors."""
		z = _check_last_dimension(z, expected=self.dim)
		return self.function(z)

	def evaluate(self, normalized_x: Tensor, normalized_time: Tensor) -> Tensor:
		"""Evaluate loss at normalized spatial point(s) and time(s)."""
		x = _check_last_dimension(normalized_x, expected=self.spatial_dimension)
		native_x = self.to_native_space(x)
		time = torch.as_tensor(normalized_time, dtype=x.dtype, device=x.device)
		if time.ndim == x.ndim and time.shape[-1] == 1:
			time = time.squeeze(-1)
		try:
			time = torch.broadcast_to(time, x.shape[:-1])
		except RuntimeError as error:
			raise ValueError(
				"normalized_time must broadcast over the input batch dimensions"
			) from error
		native_time = self.temporal_coordinate(time, like=x).unsqueeze(-1)
		return self.function(torch.cat((native_x, native_time), dim=-1))

	def objective(self, normalized_x: Tensor, normalized_time: Tensor) -> Tensor:
		"""Return the maximization objective consumed by ``WDBOOptimizer.tell``."""
		return -self.evaluate(normalized_x, normalized_time)

	def optimum_value(
		self,
		normalized_time: float,
		*,
		n_candidates: int = 8192,
		dtype: torch.dtype = torch.double,
		device: Optional[torch.device] = None,
	) -> Tensor:
		"""Deterministic Sobol approximation of the minimum at a fixed time.

		The entire candidate set is evaluated in one tensor operation, so CUDA is
		supported and no Python/NumPy loop sits on the benchmark hot path.
		"""
		if n_candidates < 1:
			raise ValueError("n_candidates must be positive")
		t = min(max(float(normalized_time), 0.0), 1.0)
		seed = 2024 + int(t * 100000)
		engine = torch.quasirandom.SobolEngine(
			dimension=self.spatial_dimension, scramble=True, seed=seed
		)
		candidates = engine.draw(n_candidates, dtype=dtype).to(device=device)
		if self.known_spatial_optimum is not None:
			native = candidates.new_tensor(self.known_spatial_optimum)
			normalized = (native - self.lower) / (self.upper - self.lower)
			candidates = torch.cat((candidates, normalized.unsqueeze(0)), dim=0)
		with torch.no_grad():
			return self.evaluate(candidates, t).amin()


BENCHMARKS: Dict[str, DynamicBenchmark] = {
	"rastrigin": DynamicBenchmark("rastrigin", 4, -4.0, 4.0, rastrigin, (0.0,) * 4),
	"schwefel": DynamicBenchmark(
		"schwefel", 3, -500.0, 500.0, schwefel, (420.968746,) * 3
	),
	"styblinski_tang": DynamicBenchmark(
		"styblinski_tang", 3, -5.0, 5.0, styblinski_tang, (-2.903534,) * 3
	),
	"eggholder": DynamicBenchmark("eggholder", 1, -512.0, 512.0, eggholder),
	"ackley": DynamicBenchmark("ackley", 3, -32.0, 32.0, ackley, (0.0,) * 3),
	"rosenbrock": DynamicBenchmark(
		"rosenbrock", 2, -1.0, 1.5, rosenbrock, (1.0, 1.0)
	),
	"shekel": DynamicBenchmark("shekel", 3, 0.0, 10.0, shekel),
	"hartmann3": DynamicBenchmark("hartmann3", 2, 0.0, 1.0, hartmann3),
	"hartmann6": DynamicBenchmark("hartmann6", 5, 0.0, 1.0, hartmann6),
	"powell": DynamicBenchmark("powell", 3, -4.0, 5.0, powell, (0.0,) * 3),
}


__all__ = [
	"BENCHMARKS",
	"DynamicBenchmark",
	"ackley",
	"eggholder",
	"hartmann3",
	"hartmann6",
	"powell",
	"rastrigin",
	"rosenbrock",
	"schwefel",
	"shekel",
	"styblinski_tang",
]
