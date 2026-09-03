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
	query_times: np.ndarray
	responses: np.ndarray
	response_regrets: np.ndarray
	instantaneous_regrets: np.ndarray
	running_regrets: np.ndarray
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
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=None,
		help="Directory for regret.csv, summary.csv, and the two paper plots.",
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

	optimization = sorted(
		[
		row
		for row in rows
		if row.get("phase", "optimization") == "optimization"
		and row.get("instantaneous_regret", "").strip()
		and row.get("response_seconds", "").strip()
		],
		key=lambda row: float(row["normalized_time"]),
	)
	if not optimization:
		raise ValueError(f"no completed optimization rows in {path}")
	responses = np.asarray(
		[float(row["response_seconds"]) for row in optimization], dtype=float
	)
	regrets = np.asarray(
		[float(row["instantaneous_regret"]) for row in optimization], dtype=float
	)
	query_times = np.asarray(
		[float(row["normalized_time"]) * duration for row in optimization],
		dtype=float,
	)

	all_queries = sorted(
		[
			row
			for row in rows
			if row.get("instantaneous_regret", "").strip()
			and row.get("normalized_time", "").strip()
		],
		key=lambda row: float(row["normalized_time"]),
	)
	all_query_times = np.asarray(
		[float(row["normalized_time"]) * duration for row in all_queries],
		dtype=float,
	)
	all_regrets = np.asarray(
		[float(row["instantaneous_regret"]) for row in all_queries], dtype=float
	)
	running_regrets = np.cumsum(all_regrets) / np.arange(1, len(all_regrets) + 1)

	initial_size = max(
		(
			float(row["dataset_size"])
			for row in rows
			if row.get("phase", "") == "initial"
			and row.get("dataset_size", "").strip()
		),
		default=0.0,
	)
	trajectory = [(0.0, initial_size)] if initial_size > 0 else []
	trajectory.extend(sorted(
		(
			float(row["normalized_time"]) * duration,
			float(row["dataset_size"]),
		)
		for row in rows
		if row.get("phase", "optimization") == "optimization"
		if row.get("normalized_time", "").strip()
		and row.get("dataset_size", "").strip()
	))
	times = np.asarray([point[0] for point in trajectory], dtype=float)
	dataset_sizes = np.asarray([point[1] for point in trajectory], dtype=float)
	return RunResult(
		path=path,
		benchmark=benchmark,
		method=method,
		seed=seed,
		duration_seconds=duration,
		average_response=float(responses.mean()),
		average_regret=float(running_regrets[-1]),
		query_times=all_query_times,
		responses=responses,
		response_regrets=regrets,
		instantaneous_regrets=all_regrets,
		running_regrets=running_regrets,
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


def _nanstd(values: np.ndarray, axis: int = 0) -> np.ndarray:
	counts = np.sum(np.isfinite(values), axis=axis)
	mean = _nanmean(values, axis=axis)
	centered = values - np.expand_dims(mean, axis=axis)
	variance = np.divide(
		np.nansum(centered * centered, axis=axis),
		counts,
		out=np.zeros_like(counts, dtype=float),
		where=counts > 0,
	)
	return np.sqrt(variance)


def _step_values(
	times: np.ndarray, values: np.ndarray, grid: np.ndarray
) -> np.ndarray:
	if len(times) == 0:
		return np.full(grid.shape, np.nan, dtype=float)
	indices = np.searchsorted(times, grid, side="right") - 1
	# The first observation represents the initial design at the left boundary;
	# after the last event, a running metric remains constant.
	indices = np.clip(indices, 0, len(values) - 1)
	return values[indices]


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


def _aggregate_method(
	runs: Sequence[RunResult], grid: np.ndarray
) -> Dict[str, np.ndarray]:
	instantaneous = np.vstack(
		[
			_step_values(run.query_times, run.instantaneous_regrets, grid)
			for run in runs
		]
	)
	running = np.vstack(
		[_step_values(run.query_times, run.running_regrets, grid) for run in runs]
	)
	dataset = np.vstack([_step_interpolate(run, grid) for run in runs])
	return {
		"instantaneous_mean": _nanmean(instantaneous, axis=0),
		"instantaneous_std": _nanstd(instantaneous, axis=0),
		"running_mean": _nanmean(running, axis=0),
		"running_std": _nanstd(running, axis=0),
		"dataset_mean": _nanmean(dataset, axis=0),
		"dataset_std": _nanstd(dataset, axis=0),
	}


def write_paper_outputs(
	files: Sequence[Path],
	*,
	benchmark: Optional[str] = None,
	fallback_duration: float = 600.0,
	grid_points: int = 121,
	title: Optional[str] = None,
	output_dir: Optional[Path] = None,
) -> Tuple[Tuple[Path, Path, Path, Path], Dict[str, Dict[str, float]]]:
	"""Write the two CSVs and two figures described by expected_output/help.md."""
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
	aggregates = {
		method: _aggregate_method(method_runs, grid)
		for method, method_runs in groups.items()
	}
	destination = output_dir or Path("plots") / selected_benchmark
	destination.mkdir(parents=True, exist_ok=True)
	regret_csv = destination / "regret.csv"
	summary_csv = destination / "summary.csv"
	duration_plot = destination / "regret_and_size_vs_duration.png"
	response_plot = destination / "regret_vs_response_time.png"

	with regret_csv.open("w", newline="", encoding="utf-8") as stream:
		fieldnames = [
			"benchmark",
			"method",
			"time_seconds",
			"instantaneous_regret_mean",
			"instantaneous_regret_std",
			"running_average_regret_mean",
			"running_average_regret_std",
			"dataset_size_mean",
			"dataset_size_std",
		]
		writer = csv.DictWriter(stream, fieldnames=fieldnames)
		writer.writeheader()
		for method, aggregate in aggregates.items():
			for index, time_value in enumerate(grid):
				writer.writerow(
					{
						"benchmark": selected_benchmark,
						"method": method,
						"time_seconds": time_value,
						"instantaneous_regret_mean": aggregate[
							"instantaneous_mean"
						][index],
						"instantaneous_regret_std": aggregate[
							"instantaneous_std"
						][index],
						"running_average_regret_mean": aggregate[
							"running_mean"
						][index],
						"running_average_regret_std": aggregate[
							"running_std"
						][index],
						"dataset_size_mean": aggregate["dataset_mean"][index],
						"dataset_size_std": aggregate["dataset_std"][index],
					}
				)

	summary: Dict[str, Dict[str, float]] = {}
	with summary_csv.open("w", newline="", encoding="utf-8") as stream:
		writer = csv.DictWriter(
			stream, fieldnames=["method", "metric", "mean", "variance", "n_runs"]
		)
		writer.writeheader()
		for method, method_runs in groups.items():
			responses = np.asarray(
				[run.average_response for run in method_runs], dtype=float
			)
			regrets = np.asarray(
				[run.average_regret for run in method_runs], dtype=float
			)
			writer.writerow(
				{
					"method": method,
					"metric": "average_regret_at_duration",
					"mean": regrets.mean(),
					"variance": regrets.var(),
					"n_runs": len(method_runs),
				}
			)
			writer.writerow(
				{
					"method": method,
					"metric": "average_response_time",
					"mean": responses.mean(),
					"variance": responses.var(),
					"n_runs": len(method_runs),
				}
			)
			summary[method] = {
				"runs": float(len(method_runs)),
				"average_response": float(responses.mean()),
				"average_regret": float(regrets.mean()),
			}

	plot_title = title or _BENCHMARK_TITLES.get(selected_benchmark, selected_benchmark)
	with plt.rc_context(
		{
			"font.family": "sans-serif",
			"font.size": 10,
			"axes.titlesize": 12,
			"axes.labelsize": 10,
			"legend.fontsize": 8,
			"figure.dpi": 180,
		}
	):
		figure, (regret_axis, size_axis) = plt.subplots(
			1, 2, figsize=(9.6, 3.8), constrained_layout=True
		)
		for index, (method, method_runs) in enumerate(groups.items()):
			color, marker = _style(method, index)
			aggregate = aggregates[method]
			regret_axis.plot(
				grid,
				aggregate["running_mean"],
				color=color,
				marker=marker,
				markevery=max(1, grid_points // 12),
				markersize=4,
				linewidth=1.6,
				label=method,
			)
			regret_axis.fill_between(
				grid,
				aggregate["running_mean"] - aggregate["running_std"],
				aggregate["running_mean"] + aggregate["running_std"],
				color=color,
				alpha=0.18,
				linewidth=0,
			)
			finite = np.isfinite(aggregate["dataset_mean"])
			size_axis.plot(
				grid[finite],
				aggregate["dataset_mean"][finite],
				color=color,
				marker=marker,
				markevery=max(1, grid_points // 12),
				markersize=4,
				linewidth=1.6,
				label=method,
			)
			size_axis.fill_between(
				grid[finite],
				np.maximum(
					aggregate["dataset_mean"][finite]
					- aggregate["dataset_std"][finite],
					1.0,
				),
				aggregate["dataset_mean"][finite]
				+ aggregate["dataset_std"][finite],
				color=color,
				alpha=0.18,
				linewidth=0,
			)
		regret_axis.set_title(plot_title)
		regret_axis.set_xlabel("Time (s)")
		regret_axis.set_ylabel(r"Average $R_t/t$")
		regret_axis.set_xlim(0.0, max_duration)
		regret_axis.grid(True, alpha=0.2, linewidth=0.6)
		regret_axis.legend(frameon=True, ncol=min(3, len(groups)))
		size_axis.set_title(plot_title)
		size_axis.set_xlabel("Duration (s)")
		size_axis.set_ylabel("Dataset Size")
		size_axis.set_xlim(0.0, max_duration)
		size_axis.set_yscale("log")
		size_axis.grid(True, which="both", alpha=0.2, linewidth=0.6)
		size_axis.legend(frameon=True, ncol=min(3, len(groups)))
		figure.savefig(duration_plot, bbox_inches="tight")
		plt.close(figure)

		figure, response_axis = plt.subplots(
			figsize=(4.6, 4.0), constrained_layout=True
		)
		for index, (method, method_runs) in enumerate(groups.items()):
			color, marker = _style(method, index)
			for run in method_runs:
				response_axis.scatter(
					run.responses,
					run.response_regrets,
					color=color,
					alpha=0.20,
					s=11,
					edgecolors="none",
				)
			responses = np.asarray(
				[run.average_response for run in method_runs], dtype=float
			)
			regrets = np.asarray(
				[run.average_regret for run in method_runs], dtype=float
			)
			response_axis.errorbar(
				responses.mean(),
				regrets.mean(),
				xerr=responses.std(),
				yerr=regrets.std(),
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
		response_axis.set_title(plot_title)
		# Faint points are individual queries; the bold marker is the mean over
		# replication-level averages. Keep the axis labels valid for both layers.
		response_axis.set_xlabel("Response Time (s)")
		response_axis.set_ylabel("Regret")
		response_axis.set_xscale("log")
		response_axis.grid(True, which="both", alpha=0.2, linewidth=0.6)
		response_axis.legend(frameon=True, ncol=min(3, len(groups)))
		figure.savefig(response_plot, bbox_inches="tight")
		plt.close(figure)

	paths = tuple(
		path.resolve()
		for path in (regret_csv, summary_csv, duration_plot, response_plot)
	)
	return paths, summary


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
	outputs, summary = write_paper_outputs(
		args.files,
		benchmark=args.benchmark,
		fallback_duration=args.duration_seconds,
		grid_points=args.grid_points,
		title=args.title,
		output_dir=args.output_dir,
	)
	for method, values in summary.items():
		print(
			f"{method}: runs={int(values['runs'])} "
			f"response={values['average_response']:.4g}s "
			f"regret={values['average_regret']:.4g}"
		)
	for output in outputs:
		print(f"Wrote {output}")


if __name__ == "__main__":
	main()
