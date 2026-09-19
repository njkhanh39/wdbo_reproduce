"""Shared machinery for the WDBO reproduction experiments.

`synthetic/run_experiment.py` and `temperature/run_experiment.py` differ only
in how they build their objective. The optimization loop, the per-query log,
the summary statistics, the run metadata and the plots all live here, so the
two benchmarks cannot drift apart.

An objective, for our purposes, is anything exposing `spatial_domain`,
`noise_std`, `evaluate(x, t)` and `oracle(t)` -- see either benchmark's
`objective.py`.

Design note: `run_once` writes a *raw* log, one row per real query. It is the
only irreplaceable artifact a run produces; `per_seed.csv`, `summary.csv` and
every plot are views over it, recomputable by `plot.py` without re-running the
experiment (which costs 10 minutes per seed).
"""
from __future__ import annotations

import csv
import json
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import gpytorch
import numpy as np
import torch

from wdbo_algo.mi_optimizer import MIDBOOptimizer
from wdbo_algo.optimizer import WDBOOptimizer

# Paper H.1: the initial observations are sampled uniformly in S' x [0, 1/40].
INITIAL_TIME_FRACTION = 1.0 / 40.0

# Which removal criterion an arm uses. "none" is the ablation: clean() is never
# called, so the dataset grows monotonically.
CRITERIA = ("wasserstein", "mi", "none")

# Resolution of the common wall-clock grid the per-seed curves are averaged on.
# A plotting concern only: seeds make different numbers of queries at different
# wall times, so they have to be resampled onto a shared grid before averaging.
GRID_POINTS = 200

QUERY_FIELDS = [
    "seed", "iteration", "sim_time", "wall_time",
    "t_response", "t_clean", "regret", "dataset_size", "n_removed",
    "lambda", "lS", "lT", "noise", "removal_budget",
    "min_criterion", "criterion_lT", "budget_spent",
]


# --------------------------------------------------------------------------
# The optimization loop
# --------------------------------------------------------------------------

def print_progress(current_time: float, row: dict, prefix: str = "", bar_width: int = 30):
    """Render a one-line, in-place progress bar for the current replication."""
    filled = int(bar_width * current_time)
    bar = "#" * filled + "-" * (bar_width - filled)
    end = "\n" if current_time >= 1.0 else ""
    print(
        f"\r{prefix}[{bar}] {current_time * 100:5.1f}%"
        f" | size={row['dataset_size']:4d} | resp={row['t_response']:5.2f}s"
        f" | clean={row['t_clean']:5.2f}s | lT={row['lT']:.3g}",
        end=end, flush=True,
    )


def build_optimizer(objective, n_initial_observations: int, alpha: float, criterion: str,
                    min_dataset_size: int, mi_options: dict | None, seed: int):
    """Construct the optimizer for one arm.

    Every arm gets the same kernels, the same initial-observation count and the
    same removal floor. Only the cleaning rule differs, which is the point: at a
    fixed wall-clock budget the arms are then comparable.

    `criterion="none"` still builds a `WDBOOptimizer`; `run_once` simply never
    calls `clean()` on it.
    """
    if criterion not in CRITERIA:
        raise ValueError(f"Unknown criterion {criterion!r} (expected one of {CRITERIA})")

    kwargs = dict(
        spatial_kernel_args=[2.5], temporal_kernel_args=[1.5],
        n_initial_observations=n_initial_observations,
        min_dataset_size=min_dataset_size,
        alpha=alpha,
    )
    kernels = (gpytorch.kernels.MaternKernel, gpytorch.kernels.MaternKernel)

    if criterion == "mi":
        return MIDBOOptimizer(objective.spatial_domain, *kernels, seed=seed,
                              **(mi_options or {}), **kwargs)

    return WDBOOptimizer(objective.spatial_domain, *kernels, **kwargs)


def run_once(objective, duration_seconds: float, n_initial_observations: int, alpha: float,
             seed: int, progress_prefix: str = "", removal: bool = True,
             criterion: str = "wasserstein", min_dataset_size: int = 15,
             mi_options: dict | None = None) -> list[dict]:
    """Run a single replication and return its raw per-query log.

    `criterion` selects the removal rule: "wasserstein" is W-DBO's own, "mi" is
    the mutual-information criterion, and "none" is the ablation in which
    `clean()` is never called, so the dataset grows monotonically -- same model,
    same kernels, same acquisition, same clock, only removal disabled. Under
    "none", `n_removed` and `removal_budget` are trivially 0 and 1.0, and
    `t_clean` is ~0. `removal=False` is kept as an alias for it.

    Note the no-removal arm cannot be compared to the others at a fixed
    iteration count: the dataset grows without bound, GP inference is O(n^3), so
    response time grows superlinearly and fewer iterations fit in the same
    budget. That trade-off is the thing being measured, so compare at fixed
    wall-clock duration.
    """
    if not removal:
        criterion = "none"

    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    optimizer = build_optimizer(objective, n_initial_observations, alpha, criterion,
                                min_dataset_size, mi_options, seed)

    # Paper H.1: the initial observations are sampled uniformly in S' x [0, 1/40],
    # i.e. spread over the first fortieth of the horizon -- not all at t = 0.
    # Gathered at a single instant they carry no information about the temporal
    # lengthscale lT, which the removal budget (1 + alpha) ** (dt / lT) divides by.
    for t0 in np.sort(rng.uniform(0.0, INITIAL_TIME_FRACTION, n_initial_observations)):
        x = optimizer.next_query(t0)
        optimizer.tell(x, t0, objective.evaluate(x, t0) + rng.normal(0.0, objective.noise_std))

    log: list[dict] = []
    # That initial window is charged against the experiment budget: back-date the
    # clock so the loop starts at t = 1/40 and advances continuously from there.
    start = time.time() - duration_seconds * INITIAL_TIME_FRACTION
    current_time = INITIAL_TIME_FRACTION

    while current_time < 1.0:
        # (ii) optimize the acquisition function.
        mark = time.time()
        x = optimizer.next_query(current_time)
        t_acqf = time.time() - mark

        # H.1: "the objective function is immediately sampled" -- querying f is
        # the experiment harness's cost, not the algorithm's, so it is timed out
        # of the response time (it still advances the wall clock).
        true_value = objective.evaluate(x, current_time)
        y = true_value + rng.normal(0.0, objective.noise_std)

        # (i) condition the GP and re-estimate the kernel and noise parameters.
        mark = time.time()
        optimizer.tell(x, current_time, y)
        t_fit = time.time() - mark

        # Removing stale observations advances the wall clock (Algorithm 1 reads
        # the clock after the removal loop) but is NOT part of the response time:
        # H.1 defines that as the sum of (i) and (ii) only.
        size_before = optimizer.dataset_size()
        mark = time.time()
        if criterion != "none":
            optimizer.clean(current_time)
        step_end = time.time()

        # The MLE hyperparameters and removal budget as the next iteration will
        # see them. Read off private attributes: `wdbo_algo` is vendored in this
        # repo (src/) and exposes no accessor, and lT in particular is the whole
        # story behind a dataset collapse.
        log.append({
            "seed": seed,
            "iteration": len(log),
            "sim_time": current_time,
            "wall_time": step_end - start,
            "t_response": t_acqf + t_fit,
            "t_clean": step_end - mark,
            "regret": objective.oracle(current_time) - true_value,
            "dataset_size": optimizer.dataset_size(),
            "n_removed": size_before - optimizer.dataset_size(),
            "lambda": optimizer._lambda,
            "lS": optimizer._lS,
            "lT": optimizer._lT,
            "noise": optimizer._noise,
            "removal_budget": optimizer._budget,
            # What the cheapest observation cost, as the arm's own criterion
            # measures it: a normalized Wasserstein ratio for W-DBO, nats for MI.
            # This is the quantity `--mi-alpha` has to be calibrated against, and
            # `criterion_lT` is the lengthscale it was measured under -- not the
            # same as `lT` above, which is post-cleaning.
            "min_criterion": optimizer._last_min_criterion,
            "criterion_lT": optimizer._last_min_lT,
            "budget_spent": optimizer._budget_spent,
        })

        current_time = min(1.0, (step_end - start) / duration_seconds)
        print_progress(current_time, log[-1], prefix=progress_prefix)

    return log


# --------------------------------------------------------------------------
# Summary statistics
# --------------------------------------------------------------------------

def query_durations(run: list[dict], duration_seconds: float) -> np.ndarray:
    """How long each query's regret stood before the next query replaced it.

    The first query is charged from the end of the initial-observation window.
    """
    wall = np.array([row["wall_time"] for row in run])
    return np.diff(wall, prepend=duration_seconds * INITIAL_TIME_FRACTION)


def summarize_seed(run: list[dict], duration_seconds: float) -> dict:
    """One row of per-replication diagnostics.

    Two regret conventions are reported, because the paper does not say which
    it uses. `avg_regret` is the plain mean over queries. `time_weighted` is
    sum(r_i dt_i) / sum(dt_i), the discrete form of (1/T) integral of r(t) dt:
    it charges each query for as long as it actually stood, which is what makes
    a slow iteration cost something. They agree when response times are stable
    and diverge exactly when cleaning stalls.
    """
    regret = np.array([row["regret"] for row in run])
    dt = query_durations(run, duration_seconds)

    # NaN on every query of the no-removal arm, which never runs a cleaning loop,
    # and on any query whose loop broke before scoring anything.
    min_criterion = np.array([row.get("min_criterion", np.nan) for row in run], dtype=float)
    median_min_criterion = float(np.nanmedian(min_criterion)) if np.any(np.isfinite(min_criterion)) else float("nan")

    return {
        "seed": run[0]["seed"],
        "iterations": len(run),
        "avg_regret": float(regret.mean()),
        "time_weighted_avg_regret": float((regret * dt).sum() / dt.sum()),
        "avg_response_time": float(np.mean([row["t_response"] for row in run])),
        "avg_clean_time": float(np.mean([row["t_clean"] for row in run])),
        "final_dataset_size": run[-1]["dataset_size"],
        "max_dataset_size": max(row["dataset_size"] for row in run),
        "min_dataset_size": min(row["dataset_size"] for row in run),
        "median_lT": float(np.median([row["lT"] for row in run])),
        "median_min_criterion": median_min_criterion,
        "total_removed": sum(row["n_removed"] for row in run),
    }


HEADLINE_METRICS = [
    ("average_regret", "avg_regret"),
    ("time_weighted_average_regret", "time_weighted_avg_regret"),
    ("response_time_s", "avg_response_time"),
    ("clean_time_s", "avg_clean_time"),
    ("iterations", "iterations"),
]


def headline(per_seed: list[dict]) -> list[dict]:
    """Mean and standard error across replications, for the metrics worth quoting.

    Standard error rather than variance: Table 2 underlines algorithms whose
    confidence intervals overlap the best one's, so the SEM is what makes a
    reproduction comparable to it.
    """
    n = len(per_seed)
    rows = []
    for name, key in HEADLINE_METRICS:
        values = np.array([row[key] for row in per_seed], dtype=float)
        sem = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        rows.append({"metric": name, "mean": float(values.mean()), "sem": sem, "n_runs": n})
    return rows


# --------------------------------------------------------------------------
# Run provenance
# --------------------------------------------------------------------------

def _git_state() -> dict:
    def git(*cmd):
        try:
            return subprocess.run(("git",) + cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def run_metadata(args: dict, extra: dict | None = None) -> dict:
    """Everything needed to know whether two runs are comparable.

    On a wall-clock-driven benchmark the machine is an experimental parameter:
    a slower host completes fewer iterations in the same 600 s and scores worse
    regret with no algorithmic difference at all. So the host, the thread count
    and the library versions belong next to the results.
    """
    import botorch

    return {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        # argparse hands back Path objects, which json cannot encode.
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in args.items()},
        "git": _git_state(),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "cpu": _cpu_model(),
            "torch_threads": torch.get_num_threads(),
        },
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "gpytorch": gpytorch.__version__,
            "botorch": botorch.__version__,
        },
        **(extra or {}),
    }


# --------------------------------------------------------------------------
# Reading and writing a results directory
# --------------------------------------------------------------------------

def results_dir(base: Path, label: str | None) -> Path:
    """`base/<timestamp>[-label]/`, so a run never overwrites its predecessor."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return base / (f"{stamp}-{label}" if label else stamp)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_run(out_dir: Path, runs: list[list[dict]], metadata: dict, duration_seconds: float) -> list[dict]:
    """Write the raw log, the per-seed table, the headline table and run.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed = [summarize_seed(run, duration_seconds) for run in runs]

    write_csv(out_dir / "queries.csv", [row for run in runs for row in run], QUERY_FIELDS)
    write_csv(out_dir / "per_seed.csv", per_seed)
    write_csv(out_dir / "summary.csv", headline(per_seed))
    (out_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    return per_seed


def load_run(out_dir: Path) -> tuple[list[list[dict]], dict]:
    """Read `queries.csv` + `run.json` back as one list per replication.

    Replications are split on `iteration == 0`, not grouped by `seed`: under
    `--same-seed` every replication carries the same seed, so grouping by it
    would silently concatenate all ten into a single run.
    """
    metadata = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))

    ints = {"seed", "iteration", "dataset_size", "n_removed"}
    runs: list[list[dict]] = []
    with open(out_dir / "queries.csv", newline="") as f:
        for raw in csv.DictReader(f):
            row = {k: (int(v) if k in ints else float(v)) for k, v in raw.items()}
            if row["iteration"] == 0:
                runs.append([])
            runs[-1].append(row)

    return runs, metadata


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def _resample(runs: list[list[dict]], duration_seconds: float, values) -> tuple[np.ndarray, np.ndarray]:
    """Each seed's `values(run)` piecewise-linearly resampled onto a shared grid.

    Seeds make different numbers of queries at different wall times, so they
    cannot be averaged pointwise without this. Values past a seed's last query
    are held flat.
    """
    grid = np.linspace(0.0, duration_seconds, GRID_POINTS)
    curves = np.stack([
        np.interp(grid, [row["wall_time"] for row in run], np.asarray(values(run), dtype=float))
        for run in runs
    ])
    return grid, curves


def _running_average(values) -> np.ndarray:
    """Cumulative mean: element i is the average of values[0..i]."""
    values = np.asarray(values, dtype=float)
    return np.cumsum(values) / np.arange(1, len(values) + 1)


def _seed_lines(ax, grid, curves, log=False, legend=True):
    """The across-seed mean, with every individual seed drawn faintly behind it.

    The mean is the headline curve, matching the paper's convention and the
    numbers in `summary.csv`. The per-seed lines sit behind it so a split across
    seeds stays visible -- on these benchmarks a run either keeps a healthy
    dataset or collapses to the cleaning floor of 2 and stays there, and the
    mean of the two groups lands in the gap between them (see the temperature
    README's section on the collapse).
    """
    for i, curve in enumerate(curves):
        ax.plot(grid, curve, alpha=0.3, linewidth=0.8, color="tab:blue",
                label=f"individual seeds (n={len(curves)})" if legend and i == 0 else None)
    ax.plot(grid, curves.mean(axis=0), color="black", linewidth=2,
            label="mean" if legend else None)
    if log:
        ax.set_yscale("log")
    if legend:
        ax.legend(fontsize=8)


def save_duration_plot(runs: list[list[dict]], duration_seconds: float, path: Path, title: str = ""):
    """Average regret up to t, and dataset size, against elapsed duration.

    Mirrors the paper's per-benchmark right-hand figure. The regret panel plots
    the *running* average: each seed's cumulative mean regret is computed first,
    then resampled onto the shared grid. Both panels show the across-seed mean
    with the individual seeds faint behind it -- see `_seed_lines`.
    """
    import matplotlib.pyplot as plt

    grid, regret = _resample(runs, duration_seconds, lambda run: _running_average([r["regret"] for r in run]))
    _, size = _resample(runs, duration_seconds, lambda run: [r["dataset_size"] for r in run])

    fig, (regret_ax, size_ax) = plt.subplots(1, 2, figsize=(10, 4))
    _seed_lines(regret_ax, grid, regret)
    regret_ax.set(xlabel="Duration (s)", ylabel="Average regret up to t", title="Regret")
    _seed_lines(size_ax, grid, size, log=True)
    size_ax.set(xlabel="Duration (s)", ylabel="Dataset size", title="Dataset size")
    if title:
        fig.suptitle(title)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_response_time_plot(runs: list[list[dict]], per_seed: list[dict], path: Path, title: str = ""):
    """Per-query regret against response time, mirroring the paper's left panel.

    The paper draws one box per algorithm; running W-DBO alone, we scatter every
    query, mark each seed's own average, and put the mean of those on top.
    """
    import matplotlib.pyplot as plt

    response = np.array([row["t_response"] for run in runs for row in run])
    regret = np.array([row["regret"] for run in runs for row in run])
    seed_response = np.array([row["avg_response_time"] for row in per_seed])
    seed_regret = np.array([row["avg_regret"] for row in per_seed])

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(response, regret, s=10, alpha=0.2, color="tab:blue", label="Individual queries")
    ax.scatter(seed_response, seed_regret, s=45, color="tab:orange", edgecolor="black",
               linewidth=0.5, zorder=3, label="Per-seed average")
    ax.plot(seed_response.mean(), seed_regret.mean(), "X", color="black",
            markersize=12, zorder=4, label="Mean across seeds")
    ax.set_xscale("log")
    ax.set(xlabel="Response time (s)", ylabel="Regret", title=title or "Regret vs. response time")
    ax.legend()

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_diagnostics_plot(runs: list[list[dict]], duration_seconds: float, path: Path, title: str = ""):
    """Temporal lengthscale and removal budget against duration.

    Not in the paper -- this is the diagnostic panel. The removal budget grows as
    (1 + alpha) ** (dt / lT), so an lT the MLE drives towards zero makes the
    budget explode and W-DBO purge its dataset down to the floor of 2. When a run
    misbehaves, these two curves say whether that is what happened.
    """
    import matplotlib.pyplot as plt

    grid, lT = _resample(runs, duration_seconds, lambda run: [r["lT"] for r in run])
    _, budget = _resample(runs, duration_seconds, lambda run: [r["removal_budget"] for r in run])

    fig, (lt_ax, budget_ax) = plt.subplots(1, 2, figsize=(10, 4))
    _seed_lines(lt_ax, grid, lT, log=True)
    lt_ax.set(xlabel="Duration (s)", ylabel="$l_T$ (normalized time units)", title="Temporal lengthscale")

    _seed_lines(budget_ax, grid, budget, log=True, legend=False)
    # b = 1 is the threshold Algorithm 1 removes above; a run whose budget sits
    # orders of magnitude over it is purging every observation it collects.
    budget_ax.axhline(1.0, color="grey", linestyle=":", linewidth=1)
    budget_ax.set(xlabel="Duration (s)", ylabel="Removal budget", title="Removal budget")
    if title:
        fig.suptitle(title)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_plots(out_dir: Path, runs: list[list[dict]], per_seed: list[dict],
               duration_seconds: float, title: str = ""):
    """Render every plot for a results directory. Also the entry point for plot.py."""
    save_duration_plot(runs, duration_seconds, out_dir / "regret_and_size_vs_duration.png", title)
    save_response_time_plot(runs, per_seed, out_dir / "regret_vs_response_time.png", title)
    save_diagnostics_plot(runs, duration_seconds, out_dir / "lengthscale_and_budget.png", title)


def add_criterion_arguments(parser):
    """The removal-rule flags both benchmarks share.

    Kept here rather than duplicated in the two `run_experiment.py` scripts so the
    arms cannot drift apart between benchmarks.
    """
    parser.add_argument("--criterion", choices=CRITERIA, default="wasserstein",
                        help="Removal rule: W-DBO's Wasserstein criterion, the mutual-information "
                             "criterion, or none (the ablation, equivalent to --no-removal).")
    parser.add_argument("--alpha", type=float, default=0.25,
                        help="Removal-budget hyperparameter for --criterion wasserstein "
                             "(paper's Table 2 value: 1/4).")
    parser.add_argument("--mi-alpha", type=float, default=1e-11,
                        help="Removal-budget hyperparameter for --criterion mi, in nats of discardable "
                             "information per temporal lengthscale. NOT comparable to --alpha: the MI "
                             "criterion is unnormalized, and scores on ackley4d run from 1e-24 to 1e-3 nats. "
                             "Sweep it in decades.")
    parser.add_argument("--min-dataset-size", type=int, default=15,
                        help="The cleaning loop never removes below this many observations. Defaults to the "
                             "initial-design size; pass 2 for the original W-DBO behaviour.")
    parser.add_argument("--mi-budget-cap", type=float, default=None,
                        help="Clamp the accrued MI budget to this many nats, so a clean that removes nothing "
                             "cannot bank budget and purge a burst later. Defaults to uncapped.")
    parser.add_argument("--mi-max-removals", type=int, default=None,
                        help="Stop after this many removals per clean. Each removal refits the "
                             "hyperparameters, so this bounds the worst-case cleaning time.")
    parser.add_argument("--mi-times", type=int, default=8, help="Quadrature nodes over future time.")
    parser.add_argument("--mi-horizon", type=float, default=3.0,
                        help="How far ahead the criterion integrates, in temporal lengthscales.")
    parser.add_argument("--mi-max-samples", type=int, default=32, help="Monte-Carlo samples of f*_t per node.")
    parser.add_argument("--mi-candidates", type=int, default=512,
                        help="Candidate points discretizing the space for the max-value CDF.")
    parser.add_argument("--mi-weight", choices=("kernel", "uniform"), default="kernel",
                        help="Weighting of future times: proportional to the temporal kernel, or flat. "
                             "Measured effect is a uniform factor of ~2 in magnitude and almost none on ranking.")
    parser.add_argument("--mi-fstar-full", action="store_true",
                        help="Sample f*_t once from the full-data posterior and share it across every "
                             "observation, instead of the default per-leave-one-out sampling. Cheaper "
                             "(one Gumbel fit per future time instead of n), but conditions f*_t on the "
                             "observation being scored.")
    parser.add_argument("--mi-clip-horizon", action="store_true",
                        help="Cap the criterion's future horizon at the end of the run (t = 1) instead of "
                             "letting it run as far as the model's own lengthscale reaches.")
    parser.add_argument("--no-removal", action="store_true",
                        help="Alias for --criterion none. Compare against a removal arm at the same "
                             "--duration-seconds, never at the same iteration count.")
    return parser


def criterion_settings(args) -> tuple[str, float, dict, str]:
    """Resolve the removal-rule flags into what `run_once` and the run label need.

    Returns:
        (criterion, alpha, mi_options, variant) -- `variant` is the human-readable
        description that goes into the figure title and run.json.
    """
    criterion = "none" if args.no_removal else args.criterion

    if criterion != "mi":
        variant = "no removal" if criterion == "none" else f"wasserstein, alpha={args.alpha:g}"
        return criterion, args.alpha, {}, variant

    mi_options = dict(
        budget_cap=args.mi_budget_cap,
        max_removals_per_clean=args.mi_max_removals,
        n_times=args.mi_times,
        horizon_lengthscales=args.mi_horizon,
        n_max_samples=args.mi_max_samples,
        n_candidates=args.mi_candidates,
        weight=args.mi_weight,
        fstar_source="full" if args.mi_fstar_full else "loo",
        clip_horizon=1.0 if args.mi_clip_horizon else None,
    )
    fstar_note = "" if mi_options["fstar_source"] == "loo" else ", f* from full D"
    return criterion, args.mi_alpha, mi_options, f"mi, alpha={args.mi_alpha:g} nats{fstar_note}"


def print_headline(rows: list[dict], reference: float | None = None):
    """Print the quotable numbers, next to the paper's Table 2 value if known."""
    for row in rows:
        line = f"  {row['metric']:<32} {row['mean']:10.4f} +/- {row['sem']:.4f} (n={row['n_runs']})"
        if reference is not None and row["metric"] == "average_regret":
            line += f"   [paper Table 2: {reference:.2f}]"
        print(line)
