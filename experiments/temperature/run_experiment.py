"""Reproduce the WDBO paper's "Temperature" benchmark (Appendix H.1 & H.2).

Runs WDBOOptimizer against the real-data objective built by objective.py,
optionally averaged over several independent replications like the paper's
own 10-seed averaging, and reproduces both halves of its Figure 20:
regret & dataset size vs. elapsed duration, and regret vs. response time.

Usage:
    python experiments/temperature/preprocess.py
    python experiments/temperature/run_experiment.py --n-seeds 10
"""
import argparse
import csv
import time
from pathlib import Path

import gpytorch
import numpy as np
import torch

from objective import build_objective
from paths import DATA_DIR
from wdbo_algo.optimizer import WDBOOptimizer

# Paper H.1: initial observations are drawn from S' x [0, 1/40].
INITIAL_TIME_FRACTION = 1.0 / 40.0


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

    # Paper H.1: the 15 initial observations are sampled uniformly in S' x [0, 1/40],
    # i.e. spread over the first fortieth of the horizon -- not all at t = 0. Gathered
    # at a single instant they carry no information about the temporal lengthscale lT,
    # which the removal budget (1 + alpha) ** (dt / lT) divides by.
    for t0 in np.sort(rng.uniform(0.0, INITIAL_TIME_FRACTION, n_initial_observations)):
        x = optimizer.next_query(t0)
        optimizer.tell(x, t0, objective.evaluate(x, t0) + rng.normal(0.0, objective.noise_std))

    log = []
    # That initial window is charged against the experiment budget: back-date the clock
    # so the optimization loop starts at t = 1/40 and advances continuously from there.
    start = time.time() - duration_seconds * INITIAL_TIME_FRACTION
    current_time = INITIAL_TIME_FRACTION
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
    independent replications (it uses 10; see --n-seeds), and its Figure 20
    convention of plotting against real elapsed duration (0 to `duration_seconds`)
    rather than the optimizer's internal normalized clock.
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


def save_duration_plot(stats: dict, path: Path):
    """Average regret and dataset size over elapsed real duration, with a
    shaded +/- 1 std. dev. band across replications, mirroring Figure 20 (right).
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

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def save_regret_vs_response_time_plot(runs: list[list[dict]], summary: dict, path: Path):
    """Per-query regret vs. response time, mirroring Figure 20 (left).

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
    ax.set_title("Regret vs. Response Time")
    ax.legend()

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=DATA_DIR / "processed.npz")
    parser.add_argument("--oracle-cache", type=Path, default=DATA_DIR / "oracle.npz")
    parser.add_argument("--smoothing", type=float, default=1.0, help="RBF interpolation smoothing (in temperature units^2).")
    parser.add_argument("--duration-seconds", type=float, default=600.0, help="Real wall-clock budget per replication (paper default: 600s).")
    parser.add_argument("--n-initial-observations", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--n-seeds", type=int, default=10, help="Number of independent replications to average (paper uses 10).")
    parser.add_argument("--seed", type=int, default=0, help="Base seed; replication i uses seed + i.")
    parser.add_argument("--same-seed", action="store_true", help="Use the same --seed for every replication instead of seed + i.")
    parser.add_argument("--output-csv", type=Path, default=DATA_DIR / "results" / "regret.csv")
    parser.add_argument("--output-summary-csv", type=Path, default=DATA_DIR / "results" / "summary.csv")
    parser.add_argument("--output-duration-plot", type=Path, default=DATA_DIR / "results" / "regret_and_size_vs_duration.png")
    parser.add_argument("--output-response-time-plot", type=Path, default=DATA_DIR / "results" / "regret_vs_response_time.png")
    args = parser.parse_args()

    print("Building objective (RBF fit + oracle grid search, ~30-60s, cached afterwards)...")
    objective = build_objective(args.processed, smoothing=args.smoothing, oracle_cache_path=args.oracle_cache)

    runs = []
    for i in range(args.n_seeds):
        prefix = f"seed {i + 1}/{args.n_seeds} " if args.n_seeds > 1 else ""
        seed = args.seed if args.same_seed else args.seed + i
        runs.append(run_once(objective, args.duration_seconds, args.n_initial_observations, args.alpha, seed, progress_prefix=prefix))

    rows = average_over_seeds(runs, args.duration_seconds, n_points=200) if args.n_seeds > 1 else runs[0]
    duration_stats = compute_duration_stats(runs, args.duration_seconds, n_points=200)
    summary = summarize_runs(runs)

    save_csv(rows, args.output_csv)
    save_summary_csv(summary, args.duration_seconds, args.output_summary_csv)
    save_duration_plot(duration_stats, args.output_duration_plot)
    save_regret_vs_response_time_plot(runs, summary, args.output_response_time_plot)

    print(f"Average regret up to t={args.duration_seconds:g}s: {summary['avg_regret_mean']:.4f} (var={summary['avg_regret_var']:.4f}) across {summary['n_runs']} run(s)")
    print(f"Average response time: {summary['avg_response_time_mean']:.4f}s (var={summary['avg_response_time_var']:.4f}) across {summary['n_runs']} run(s)")
    print(f"Saved logs to {args.output_csv} and {args.output_summary_csv}")
    print(f"Saved plots to {args.output_duration_plot} and {args.output_response_time_plot}")


if __name__ == "__main__":
    main()
