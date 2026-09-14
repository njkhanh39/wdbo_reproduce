"""Reproduce the WDBO paper's synthetic-function benchmarks (Appendix H.1 & H.2).

Runs WDBOOptimizer against a closed-form test function whose last axis is time
(see benchmarks.py / objective.py), optionally averaged over several
independent replications like the paper's 10-seed averaging, and reproduces
both halves of its per-benchmark figures (e.g. Figure 13 for Ackley4d):
regret & dataset size vs. elapsed duration, and regret vs. response time.

Usage:
    python experiments/synthetic/run_experiment.py --benchmark ackley4d --n-seeds 10

This mirrors experiments/temperature/run_experiment.py; only the objective
(analytic instead of interpolated real data) differs.
"""
import argparse
import csv
import time
from pathlib import Path

import gpytorch
import numpy as np
import torch

from benchmarks import get_benchmark
from objective import build_objective
from paths import DATA_DIR
from wdbo_algo.optimizer import WDBOOptimizer


def print_progress(current_time: float, dataset_size: int, response_time: float, prefix: str = "", bar_width: int = 30):
    """Render a one-line, in-place progress bar for the current replication."""
    filled = int(bar_width * current_time)
    bar = "#" * filled + "-" * (bar_width - filled)
    end = "\n" if current_time >= 1.0 else ""
    print(f"\r{prefix}[{bar}] {current_time * 100:5.1f}% | dataset size={dataset_size:4d} | response time={response_time:5.2f}s", end=end, flush=True)


def run_once(objective, duration_seconds: float, n_initial_observations: int, alpha: float, seed: int, progress_prefix: str = ""):
    """Run a single WDBO replication and return a list of per-iteration log rows."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    optimizer = WDBOOptimizer(
        objective.spatial_domain,
        gpytorch.kernels.MaternKernel, gpytorch.kernels.MaternKernel,
        spatial_kernel_args=[2.5], temporal_kernel_args=[1.5],
        n_initial_observations=n_initial_observations,
        alpha=alpha,
    )

    log = []
    start = time.time()
    current_time = 0.0
    while current_time < 1.0:
        step_start = time.time()
        x = optimizer.next_query(current_time)
        true_value = objective.evaluate(x, current_time)
        y = true_value + rng.normal(0.0, objective.noise_std)
        optimizer.tell(x, current_time, y)
        optimizer.clean(current_time)
        step_end = time.time()

        log.append({
            "sim_time": current_time,
            "wall_time": step_end - start,
            "response_time": step_end - step_start,
            "regret": objective.oracle(current_time) - true_value,
            "dataset_size": optimizer.dataset_size(),
        })

        current_time = min(1.0, (step_end - start) / duration_seconds)
        print_progress(current_time, log[-1]["dataset_size"], log[-1]["response_time"], prefix=progress_prefix)

    return log


def average_over_seeds(runs: list[list[dict]], duration_seconds: float, n_points: int) -> list[dict]:
    """Interpolate each replication onto a common wall-clock-duration grid and average.

    Mirrors the paper's Section H.1 methodology of averaging metrics across
    independent replications (it uses 10; see --n-seeds), and its per-benchmark
    figure convention of plotting against real elapsed duration (0 to
    `duration_seconds`) rather than the optimizer's internal normalized clock.
    """
    grid = np.linspace(0.0, duration_seconds, n_points)
    regrets = np.stack([np.interp(grid, [row["wall_time"] for row in run], [row["regret"] for row in run]) for run in runs])
    sizes = np.stack([np.interp(grid, [row["wall_time"] for row in run], [row["dataset_size"] for row in run]) for run in runs])

    return [
        {"wall_time": t, "regret": r, "dataset_size": s}
        for t, r, s in zip(grid, regrets.mean(axis=0), sizes.mean(axis=0))
    ]


def running_average(values: np.ndarray) -> np.ndarray:
    """Cumulative mean: element i is the average of values[0..i]."""
    return np.cumsum(values) / np.arange(1, len(values) + 1)


def compute_duration_stats(runs: list[list[dict]], duration_seconds: float, n_points: int) -> dict:
    """Interpolate the running (cumulative) average regret and the dataset
    size of every replication onto a common wall-clock grid, and return their
    per-grid-point mean and standard deviation across replications.

    "Average regret up to t" here follows the convention R_t / t = the mean
    of every query's instantaneous regret among queries made up to time t
    (a cumulative mean over queries, not a division by wall-clock t).
    """
    grid = np.linspace(0.0, duration_seconds, n_points)

    running_regret = np.stack([
        np.interp(grid, [row["wall_time"] for row in run], running_average(np.array([row["regret"] for row in run])))
        for run in runs
    ])
    dataset_size = np.stack([
        np.interp(grid, [row["wall_time"] for row in run], [row["dataset_size"] for row in run])
        for run in runs
    ])

    return {
        "wall_time": grid,
        "regret_mean": running_regret.mean(axis=0),
        "regret_std": running_regret.std(axis=0),
        "dataset_size_mean": dataset_size.mean(axis=0),
        "dataset_size_std": dataset_size.std(axis=0),
    }


def summarize_runs(runs: list[list[dict]]) -> dict:
    """Per-replication summary statistics.

    Each replication contributes one number (its own average regret over the
    whole run, and its own average response time); we report the mean and
    variance of those numbers across replications, matching the paper's
    "independent replications" convention used throughout Appendix H.
    """
    final_avg_regret = np.array([np.mean([row["regret"] for row in run]) for run in runs])
    avg_response_time = np.array([np.mean([row["response_time"] for row in run]) for run in runs])
    ddof = 1 if len(runs) > 1 else 0

    return {
        "n_runs": len(runs),
        "avg_regret_mean": float(final_avg_regret.mean()),
        "avg_regret_var": float(final_avg_regret.var(ddof=ddof)),
        "avg_response_time_mean": float(avg_response_time.mean()),
        "avg_response_time_var": float(avg_response_time.var(ddof=ddof)),
    }


def save_csv(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_summary_csv(summary: dict, duration_seconds: float, path: Path):
    """A second, small summary file: one headline number (mean + variance
    across replications) each for regret and response time."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "variance", "n_runs"])
        writer.writerow([f"average_regret_up_to_t={duration_seconds:g}s", summary["avg_regret_mean"], summary["avg_regret_var"], summary["n_runs"]])
        writer.writerow(["average_response_time_s", summary["avg_response_time_mean"], summary["avg_response_time_var"], summary["n_runs"]])


def save_duration_plot(stats: dict, path: Path, title: str = ""):
    """Average regret and dataset size over elapsed real duration, with a
    shaded +/- 1 std. dev. band across replications, mirroring the paper's
    per-benchmark figure (right panel).
    """
    import matplotlib.pyplot as plt

    wall_time = stats["wall_time"]
    fig, (regret_ax, size_ax) = plt.subplots(1, 2, figsize=(10, 4))

    regret_mean, regret_std = stats["regret_mean"], stats["regret_std"]
    regret_ax.plot(wall_time, regret_mean)
    regret_ax.fill_between(wall_time, regret_mean - regret_std, regret_mean + regret_std, alpha=0.25)
    regret_ax.set_xlabel("Duration (s)")
    regret_ax.set_ylabel("Average regret up to t")
    regret_ax.set_title("Regret")

    size_mean, size_std = stats["dataset_size_mean"], stats["dataset_size_std"]
    size_ax.plot(wall_time, size_mean)
    size_ax.fill_between(wall_time, np.clip(size_mean - size_std, 1e-6, None), size_mean + size_std, alpha=0.25)
    size_ax.set_yscale("log")
    size_ax.set_xlabel("Duration (s)")
    size_ax.set_ylabel("Dataset size")
    size_ax.set_title("Dataset size")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def save_regret_vs_response_time_plot(runs: list[list[dict]], summary: dict, path: Path, title: str = ""):
    """Per-query regret vs. response time, mirroring the paper's per-benchmark
    figure (left panel).

    The paper's version compares several algorithms (one box per algorithm);
    since this script only runs W-DBO, we instead scatter every individual
    query across all replications and mark the mean +/- 1 std. dev. across
    replications, which is the single-algorithm equivalent of the same
    average-regret-vs-average-response-time trade-off.
    """
    import matplotlib.pyplot as plt

    response_times = np.array([row["response_time"] for run in runs for row in run])
    regrets = np.array([row["regret"] for run in runs for row in run])

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(response_times, regrets, s=10, alpha=0.3, label="Individual queries")
    ax.errorbar(
        summary["avg_response_time_mean"], summary["avg_regret_mean"],
        xerr=summary["avg_response_time_var"] ** 0.5, yerr=summary["avg_regret_var"] ** 0.5,
        fmt="X", color="black", markersize=10, capsize=4, label="Mean +/- 1 std (across runs)", zorder=3,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Response Time (s)")
    ax.set_ylabel("Regret")
    ax.set_title(title or "Regret vs. Response Time")
    ax.legend()

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--benchmark", default="ackley4d", help="Synthetic benchmark name (see benchmarks.py).")
    parser.add_argument("--duration-seconds", type=float, default=600.0, help="Real wall-clock budget per replication (paper default: 600s).")
    parser.add_argument("--n-initial-observations", type=int, default=15)
    parser.add_argument("--time-span", type=float, nargs=2, default=(0.0, 1.0), metavar=("LO", "HI"),
                        help="Range the normalized clock [0,1] is mapped onto for the function's time axis. "
                             "Default (0 1) = paper H.1's normalized time; (-32 32) puts time on the spatial box.")
    parser.add_argument("--alpha", type=float, default=1.0 / 3.0, help="WDBO removal-budget hyperparameter (paper's Table 2 value: 1/3).")
    parser.add_argument("--n-seeds", type=int, default=10, help="Number of independent replications to average (paper uses 10).")
    parser.add_argument("--seed", type=int, default=0, help="Base seed; replication i uses seed + i.")
    parser.add_argument("--same-seed", action="store_true", help="Use the same --seed for every replication instead of seed + i.")
    parser.add_argument("--oracle-time-points", type=int, default=1000, help="Time samples in the cached oracle curve.")
    parser.add_argument("--oracle-grid-resolution", type=int, default=33, help="Per-axis spatial grid nodes for the oracle search (use an odd number).")
    parser.add_argument("--oracle-cache", type=Path, default=None, help="Defaults to data/synthetic/<benchmark>/oracle.npz.")
    parser.add_argument("--results-dir", type=Path, default=None, help="Defaults to data/synthetic/<benchmark>/results.")
    args = parser.parse_args()

    benchmark = get_benchmark(args.benchmark)
    time_span = (float(args.time_span[0]), float(args.time_span[1]))
    default_time = time_span == (0.0, 1.0)
    span_tag = "" if default_time else f"_t{time_span[0]:g}_{time_span[1]:g}"

    base_dir = DATA_DIR / benchmark.name
    oracle_cache = args.oracle_cache or base_dir / f"oracle{span_tag}.npz"
    results_dir = args.results_dir or base_dir / f"results{span_tag}"
    lo, hi = benchmark.spatial_domain[0]

    print(f"Benchmark: {benchmark.name} (d'={benchmark.dim}: spatial d={benchmark.spatial_dim} in [{lo:g}, {hi:g}]^{benchmark.spatial_dim}, time in [{time_span[0]:g}, {time_span[1]:g}])")
    print(f"Building objective (oracle grid search {args.oracle_grid_resolution}^{benchmark.spatial_dim} x {args.oracle_time_points} times, cached to {oracle_cache})...")
    objective = build_objective(
        benchmark,
        temporal_span=time_span,
        oracle_time_points=args.oracle_time_points,
        oracle_grid_resolution=args.oracle_grid_resolution,
        oracle_cache_path=oracle_cache,
    )
    print(f"Noise std (5% of signal variance): {objective.noise_std:.4f}")

    runs = []
    for i in range(args.n_seeds):
        prefix = f"seed {i + 1}/{args.n_seeds} " if args.n_seeds > 1 else ""
        seed = args.seed if args.same_seed else args.seed + i
        runs.append(run_once(objective, args.duration_seconds, args.n_initial_observations, args.alpha, seed, progress_prefix=prefix))

    rows = average_over_seeds(runs, args.duration_seconds, n_points=200) if args.n_seeds > 1 else runs[0]
    duration_stats = compute_duration_stats(runs, args.duration_seconds, n_points=200)
    summary = summarize_runs(runs)

    plot_title = benchmark.name if default_time else f"{benchmark.name} (time in [{time_span[0]:g}, {time_span[1]:g}])"
    save_csv(rows, results_dir / "regret.csv")
    save_summary_csv(summary, args.duration_seconds, results_dir / "summary.csv")
    save_duration_plot(duration_stats, results_dir / "regret_and_size_vs_duration.png", title=plot_title)
    save_regret_vs_response_time_plot(runs, summary, results_dir / "regret_vs_response_time.png", title=f"{plot_title}: Regret vs. Response Time")

    print(f"Average regret up to t={args.duration_seconds:g}s: {summary['avg_regret_mean']:.4f} (var={summary['avg_regret_var']:.4f}) across {summary['n_runs']} run(s)")
    print(f"Average response time: {summary['avg_response_time_mean']:.4f}s (var={summary['avg_response_time_var']:.4f}) across {summary['n_runs']} run(s)")
    print(f"Saved results under {results_dir}")


if __name__ == "__main__":
    main()
