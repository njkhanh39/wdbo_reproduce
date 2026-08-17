"""Command-line runner for the Appendix H.2 synthetic W-DBO experiments."""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import gpytorch
import numpy as np
import torch
from torch import Tensor

from wdbo_algo.benchmarks import BENCHMARKS, DynamicBenchmark


def _parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Run W-DBO on one synthetic benchmark from paper Appendix H.2."
	)
	parser.add_argument("benchmark", choices=tuple(BENCHMARKS))
	parser.add_argument("--duration-seconds", type=float, default=600.0)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--alpha", type=float, default=0.25)
	parser.add_argument("--initial-observations", type=int, default=15)
	parser.add_argument("--signal-samples", type=int, default=4096)
	parser.add_argument("--oracle-candidates", type=int, default=8192)
	parser.add_argument(
		"--max-iterations",
		type=int,
		default=None,
		help="Optional safety cap; the paper uses only the 600-second horizon.",
	)
	parser.add_argument("--log-every", type=int, default=10)
	parser.add_argument("--device", default="cpu")
	parser.add_argument("--output", type=Path, default=None)
	return parser


def _normalized_spatial(benchmark: DynamicBenchmark, native_x: np.ndarray) -> Tensor:
	x = torch.as_tensor(native_x, dtype=torch.double, device=benchmark.bounds.device)
	return (x - benchmark.lower) / (benchmark.upper - benchmark.lower)


def _loss(
	benchmark: DynamicBenchmark, native_x: np.ndarray, normalized_time: float
) -> float:
	normalized_x = _normalized_spatial(benchmark, native_x)
	with torch.no_grad():
		return float(benchmark.evaluate(normalized_x, normalized_time).cpu())


def _signal_standard_deviation(
	benchmark: DynamicBenchmark, n_samples: int, seed: int
) -> float:
	if n_samples < 2:
		raise ValueError("signal_samples must be at least two")
	engine = torch.quasirandom.SobolEngine(benchmark.dim, scramble=True, seed=seed)
	points = engine.draw(n_samples, dtype=torch.double).to(benchmark.bounds.device)
	with torch.no_grad():
		losses = benchmark.evaluate(points[..., :-1], points[..., -1])
	return float(losses.std(unbiased=False).cpu().clamp_min(1e-12))


def _write_results(path: Path, rows: List[Dict[str, object]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", newline="", encoding="utf-8") as stream:
		writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
		writer.writeheader()
		writer.writerows(rows)


def run(args: argparse.Namespace) -> Path:
	# Import lazily so ``wdbo-synthetic --help`` works before the native criterion
	# extension is installed.
	from wdbo_algo.optimizer import WDBOOptimizer

	if args.duration_seconds <= 0:
		raise ValueError("duration_seconds must be positive")
	if args.initial_observations < 1:
		raise ValueError("initial_observations must be positive")
	if args.oracle_candidates < 1:
		raise ValueError("oracle_candidates must be positive")

	device = torch.device(args.device)
	benchmark = BENCHMARKS[args.benchmark].to(device=device)
	rng = np.random.default_rng(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	# Appendix H.1: spatial Matérn-5/2, temporal Matérn-3/2, 15 initial
	# observations, and Gaussian-noise variance equal to 5% signal variance.
	spatial_domain = np.tile(
		np.array([benchmark.lower, benchmark.upper], dtype=float),
		(benchmark.spatial_dimension, 1),
	)
	optimizer = WDBOOptimizer(
		spatial_domain,
		gpytorch.kernels.MaternKernel,
		gpytorch.kernels.MaternKernel,
		spatial_kernel_args=[2.5],
		temporal_kernel_args=[1.5],
		n_initial_observations=args.initial_observations,
		alpha=args.alpha,
	)
	signal_std = _signal_standard_deviation(
		benchmark, args.signal_samples, 2024 + args.seed
	)
	noise_std = math.sqrt(0.05) * signal_std
	rows: List[Dict[str, object]] = []

	# The paper samples the initial design uniformly from S' x [0, 1/40].
	initial_x = rng.uniform(size=(args.initial_observations, benchmark.spatial_dimension))
	initial_t = rng.uniform(0.0, 1.0 / 40.0, size=args.initial_observations)
	for index, (normalized_x, normalized_time) in enumerate(zip(initial_x, initial_t)):
		native_x = benchmark.lower + normalized_x * (benchmark.upper - benchmark.lower)
		loss = _loss(benchmark, native_x, float(normalized_time))
		objective = -loss
		observed = objective + rng.normal(0.0, noise_std)
		optimizer.tell(native_x, float(normalized_time), observed)
		rows.append(
			{
				"benchmark": args.benchmark,
				"method": "W-DBO",
				"seed": args.seed,
				"duration_seconds": args.duration_seconds,
				"phase": "initial",
				"iteration": index,
				"normalized_time": float(normalized_time),
				"loss": loss,
				"objective": objective,
				"observed_objective": observed,
				"optimum_loss": "",
				"instantaneous_regret": "",
				"response_seconds": 0.0,
				"dataset_size": optimizer.dataset_size(),
				"spatial_x": " ".join(map(str, native_x)),
			}
		)

	# Only algorithm response time advances the normalized experimental clock.
	# This prevents CSV/oracle bookkeeping from changing the objective trajectory.
	elapsed = args.duration_seconds / 40.0
	iteration = 0
	while elapsed < args.duration_seconds:
		if args.max_iterations is not None and iteration >= args.max_iterations:
			break
		normalized_time = elapsed / args.duration_seconds
		started = time.perf_counter()
		native_x = optimizer.next_query(normalized_time)
		loss = _loss(benchmark, native_x, normalized_time)
		objective = -loss
		observed = objective + rng.normal(0.0, noise_std)
		optimizer.tell(native_x, normalized_time, observed)
		optimizer.clean(normalized_time)
		response_seconds = time.perf_counter() - started
		elapsed += response_seconds
		rows.append(
			{
				"benchmark": args.benchmark,
				"method": "W-DBO",
				"seed": args.seed,
				"duration_seconds": args.duration_seconds,
				"phase": "optimization",
				"iteration": iteration,
				"normalized_time": normalized_time,
				"loss": loss,
				"objective": objective,
				"observed_objective": observed,
				"optimum_loss": "",
				"instantaneous_regret": "",
				"response_seconds": response_seconds,
				"dataset_size": optimizer.dataset_size(),
				"spatial_x": " ".join(map(str, native_x)),
			}
		)
		iteration += 1
		if args.log_every > 0 and iteration % args.log_every == 0:
			print(
				f"iteration={iteration} t={normalized_time:.4f} "
				f"response={response_seconds:.3f}s n={optimizer.dataset_size()}",
				flush=True,
			)

	# Regret computation is intentionally post-hoc and therefore never changes
	# the continuous-time trajectory used during optimization.
	for row in rows:
		t = float(row["normalized_time"])
		optimum = float(
			benchmark.optimum_value(
				t,
				n_candidates=args.oracle_candidates,
				device=device,
			).cpu()
		)
		row["optimum_loss"] = optimum
		row["instantaneous_regret"] = max(0.0, float(row["loss"]) - optimum)

	output = args.output or Path("results") / (
		f"{args.benchmark}_wdbo_seed{args.seed}.csv"
	)
	_write_results(output, rows)
	return output.resolve()


def main(argv: Optional[List[str]] = None) -> None:
	args = _parser().parse_args(argv)
	output = run(args)
	print(f"Wrote {output}")


if __name__ == "__main__":
	main()
