"""Paper-style plots for W-DBO synthetic benchmark CSV results."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_METHOD_STYLES = {
	"GP-UCB": ("#1f77b4", "x"),
	"ABO": ("#ff7f0e", "D"),
	"ET-GP-UCB": ("#2ca02c", "o"),
	"R-GP-UCB": ("#d62728", "*"),
	"TV-GP-UCB": ("#9467bd", "^"),
	"W-DBO": ("#8c564b", "v"),
}

_BENCHMARK_TITLES = {
	"rastrigin": "Rastrigin5d",
	"schwefel": "Schwefel4d",
	"styblinski_tang": "StyblinskiTang4d",
	"eggholder": "Eggholder2d",
	"ackley": "Ackley4d",
	"rosenbrock": "Rosenbrock3d",
	"shekel": "Shekel",
	"hartmann3": "Hartmann3d",
	"hartmann6": "Hartmann6d",
	"powell": "Powell4d",
}


@dataclass(frozen=True)
class RunResult:
	path: Path
	benchmark: str
	method: str
	seed: int
	duration_seconds: float
	average_response: float
	average_regret: float
	times: np.ndarray
	dataset_sizes: np.ndarray


def _parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description=(
			"Draw the Appendix H.2 response/regret and dataset-size panels from "
			"one or more benchmark CSV files."
		)
	)
	parser.add_argument("files", nargs="+", type=Path)
	parser.add_argument("--benchmark", default=None)
	parser.add_argument("--duration-seconds", type=float, default=600.0)
	parser.add_argument("--grid-points", type=int, default=121)
	parser.add_argument("--title", default=None)
	parser.add_argument("--output", type=Path, default=None)
	parser.add_argument(
		"--linear-response-scale",
		action="store_true",
		help="Use a linear response-time axis instead of the paper's log scale.",
	)
	parser.add_argument(
		"--linear-dataset-scale",
		action="store_true",
		help="Use a linear dataset-size axis instead of the paper's log scale.",
	)
	return parser


def _infer_benchmark(path: Path) -> str:
	stem = path.stem.lower()
	for benchmark in _BENCHMARK_TITLES:
		if benchmark in stem:
			return benchmark
	return stem.split("_", 1)[0]


def _infer_method(path: Path) -> str:
	name = path.stem.lower().replace("_", "-")
	for method in _METHOD_STYLES:
		if method.lower() in name:
			return method
	return "W-DBO"


def _infer_seed(path: Path) -> int:
	match = re.search(r"seed[-_]?([0-9]+)", path.stem.lower())
	return int(match.group(1)) if match else 0


def _last_nonempty(rows: Sequence[Dict[str, str]], key: str, fallback: str) -> str:
	for row in rows:
		value = row.get(key, "").strip()
		if value:
			return value
	return fallback


def load_run(path: Path, fallback_duration: float = 600.0) -> RunResult:
	with path.open(newline="", encoding="utf-8") as stream:
		rows = list(csv.DictReader(stream))
	if not rows:
		raise ValueError(f"result file contains no rows: {path}")

	benchmark = _last_nonempty(rows, "benchmark", _infer_benchmark(path))
	method = _last_nonempty(rows, "method", _infer_method(path))
	seed = int(_last_nonempty(rows, "seed", str(_infer_seed(path))))
	duration = float(
		_last_nonempty(rows, "duration_seconds", str(fallback_duration))
	)
	if duration <= 0:
		raise ValueError(f"duration must be positive in {path}")

	optimization = [
		row
		for row in rows
		if row.get("phase", "optimization") == "optimization"
		and row.get("instantaneous_regret", "").strip()
		and row.get("response_seconds", "").strip()
	]
	if not optimization:
		raise ValueError(f"no completed optimization rows in {path}")
	responses = np.asarray(
		[float(row["response_seconds"]) for row in optimization], dtype=float
	)
	regrets = np.asarray(
		[float(row["instantaneous_regret"]) for row in optimization], dtype=float
	)

	trajectory = sorted(
		(
			float(row["normalized_time"]) * duration,
			float(row["dataset_size"]),
		)
		for row in rows
		if row.get("normalized_time", "").strip()
		and row.get("dataset_size", "").strip()
	)
	times = np.asarray([point[0] for point in trajectory], dtype=float)
	dataset_sizes = np.asarray([point[1] for point in trajectory], dtype=float)
	return RunResult(
		path=path,
		benchmark=benchmark,
		method=method,
		seed=seed,
		duration_seconds=duration,
		average_response=float(responses.mean()),
		average_regret=float(regrets.mean()),
		times=times,
		dataset_sizes=dataset_sizes,
	)


def _sem(values: np.ndarray, axis: int = 0) -> np.ndarray:
	counts = np.sum(np.isfinite(values), axis=axis)
	mean = np.divide(
		np.nansum(values, axis=axis),
		counts,
		out=np.zeros_like(counts, dtype=float),
		where=counts > 0,
	)
	centered = values - np.expand_dims(mean, axis=axis)
	squared_error = np.nansum(centered * centered, axis=axis)
	variance = np.divide(
		squared_error,
		counts - 1,
		out=np.zeros_like(squared_error, dtype=float),
		where=counts > 1,
	)
	mean_variance = np.divide(
		variance,
		counts,
		out=np.zeros_like(variance, dtype=float),
		where=counts > 1,
	)
	return np.sqrt(mean_variance)


def _nanmean(values: np.ndarray, axis: int = 0) -> np.ndarray:
	counts = np.sum(np.isfinite(values), axis=axis)
	return np.divide(
		np.nansum(values, axis=axis),
		counts,
		out=np.full(np.shape(counts), np.nan, dtype=float),
		where=counts > 0,
	)


def _step_interpolate(run: RunResult, grid: np.ndarray) -> np.ndarray:
	indices = np.searchsorted(run.times, grid, side="right") - 1
	values = np.full(grid.shape, np.nan, dtype=float)
	# Dataset size is piecewise constant between optimizer events. Carrying the
	# final value forward also makes intentionally capped smoke runs plottable.
	valid = indices >= 0
	values[valid] = run.dataset_sizes[indices[valid]]
	return values


def _group_runs(runs: Iterable[RunResult]) -> Dict[str, List[RunResult]]:
	groups: Dict[str, List[RunResult]] = {}
	for run in runs:
		groups.setdefault(run.method, []).append(run)
	return groups


def _style(method: str, index: int) -> Tuple[str, str]:
	if method in _METHOD_STYLES:
		return _METHOD_STYLES[method]
	colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
	markers = ("o", "s", "^", "D", "P", "X")
	return colors[index % len(colors)], markers[index % len(markers)]


def plot_runs(
	files: Sequence[Path],
	*,
	benchmark: Optional[str] = None,
	fallback_duration: float = 600.0,
	grid_points: int = 121,
	title: Optional[str] = None,
	output: Optional[Path] = None,
	log_response: bool = True,
	log_dataset: bool = True,
) -> Tuple[Path, Dict[str, Dict[str, float]]]:
	if grid_points < 2:
		raise ValueError("grid_points must be at least two")
	runs = [load_run(path, fallback_duration) for path in files]
	selected_benchmark = benchmark or runs[0].benchmark
	runs = [run for run in runs if run.benchmark == selected_benchmark]
	if not runs:
		raise ValueError(f"no files matched benchmark {selected_benchmark!r}")
	groups = _group_runs(runs)
	max_duration = max(run.duration_seconds for run in runs)
	grid = np.linspace(0.0, max_duration, grid_points)

	with plt.rc_context(
		{
			"font.family": "serif",
			"font.size": 10,
			"axes.titlesize": 11,
			"axes.labelsize": 10,
			"legend.fontsize": 8,
			"figure.dpi": 150,
		}
	):
		figure, (performance_axis, size_axis) = plt.subplots(
			1, 2, figsize=(10.5, 4.1), constrained_layout=True
		)
		summary: Dict[str, Dict[str, float]] = {}
		for index, (method, method_runs) in enumerate(groups.items()):
			color, marker = _style(method, index)
			response = np.asarray(
				[run.average_response for run in method_runs], dtype=float
			)
			regret = np.asarray([run.average_regret for run in method_runs], dtype=float)
			performance_axis.scatter(
				response,
				regret,
				color=color,
				marker=marker,
				alpha=0.28,
				s=24,
				linewidths=0.8,
			)
			mean_response = float(response.mean())
			mean_regret = float(regret.mean())
			response_sem = float(_sem(response))
			regret_sem = float(_sem(regret))
			performance_axis.errorbar(
				mean_response,
				mean_regret,
				xerr=response_sem,
				yerr=regret_sem,
				fmt=marker,
				color=color,
				markerfacecolor=color,
				markeredgecolor="white",
				markeredgewidth=0.7,
				markersize=8,
				capsize=3,
				elinewidth=7,
				alpha=0.88,
				label=method,
			)

			trajectories = np.vstack(
				[_step_interpolate(run, grid) for run in method_runs]
			)
			mean_size = _nanmean(trajectories, axis=0)
			size_sem = _sem(trajectories, axis=0)
			finite = np.isfinite(mean_size)
			size_axis.plot(
				grid[finite],
				mean_size[finite],
				color=color,
				marker=marker,
				markevery=max(1, grid_points // 12),
				markersize=4.5,
				linewidth=1.5,
				label=method,
			)
			if len(method_runs) > 1:
				size_axis.fill_between(
					grid[finite],
					mean_size[finite] - size_sem[finite],
					mean_size[finite] + size_sem[finite],
					color=color,
					alpha=0.18,
					linewidth=0,
				)
			summary[method] = {
				"runs": float(len(method_runs)),
				"average_response": mean_response,
				"average_regret": mean_regret,
			}

		plot_title = title or _BENCHMARK_TITLES.get(
			selected_benchmark, selected_benchmark
		)
		performance_axis.set_title(plot_title)
		performance_axis.set_xlabel("Average Response Time (s)")
		performance_axis.set_ylabel("Average Regret")
		if log_response:
			performance_axis.set_xscale("log")
		performance_axis.grid(True, which="both", alpha=0.2, linewidth=0.6)
		performance_axis.legend(frameon=True, ncol=min(3, len(groups)))

		size_axis.set_title(plot_title)
		size_axis.set_xlabel("Duration (s)")
		size_axis.set_ylabel("Dataset Size")
		size_axis.set_xlim(0.0, max_duration)
		if log_dataset:
			size_axis.set_yscale("log")
		size_axis.grid(True, which="both", alpha=0.2, linewidth=0.6)
		size_axis.legend(frameon=True, ncol=min(3, len(groups)))

		output_path = output or Path("plots") / f"{selected_benchmark}_paper.png"
		output_path.parent.mkdir(parents=True, exist_ok=True)
		figure.savefig(output_path, bbox_inches="tight")
		plt.close(figure)
	return output_path.resolve(), summary


def main(argv: Optional[List[str]] = None) -> None:
	args = _parser().parse_args(argv)
	output, summary = plot_runs(
		args.files,
		benchmark=args.benchmark,
		fallback_duration=args.duration_seconds,
		grid_points=args.grid_points,
		title=args.title,
		output=args.output,
		log_response=not args.linear_response_scale,
		log_dataset=not args.linear_dataset_scale,
	)
	for method, values in summary.items():
		print(
			f"{method}: runs={int(values['runs'])} "
			f"response={values['average_response']:.4g}s "
			f"regret={values['average_regret']:.4g}"
		)
	print(f"Wrote {output}")


if __name__ == "__main__":
	main()
