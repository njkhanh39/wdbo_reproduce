"""Shared machinery for the WDBO reproduction experiments.

`synthetic/run_experiment.py` and `temperature/run_experiment.py` differ only
in how they build their objective. The optimization loop, the per-query log,
the summary statistics, the run metadata and the plots all live here, so the
two benchmarks cannot drift apart.

An objective, for our purposes, is anything exposing `spatial_domain`,
`noise_std`, `evaluate(x, t_env)` and `oracle(t_env)` -- see either
benchmark's `objective.py`.

Two clocks
----------
The loop keeps two separate clocks, and keeping them separate is the point:

* the **wall clock**, in real seconds, measured with `time.perf_counter()`.
  It starts at 0 when the optimization loop proper begins and runs to
  `duration_seconds` (`H`). This is the compute budget.
* the **environment clock** `t_env`, in the objective's own time units. It
  advances as `t_env = env_start + env_speed * elapsed_seconds`, where
  `env_speed` is a free parameter in environment-units per real second.
  This is how fast the world moves.

Earlier revisions collapsed the two into `elapsed / duration_seconds`, which
tied them together: shortening a run from 600 s to 200 s did not just cut the
compute budget, it also made the environment drift three times faster, so the
600 s and 200 s arms were not the same problem at different budgets -- they
were different problems. `env_speed` breaks that: change `H` to change the
budget, change `env_speed` to change the difficulty, never both at once.

Design note: `run_once` writes a *raw* log, one row per query plus one opening
`INITIAL_ROW`. It is the only irreplaceable artifact a run produces;
`per_seed.csv`, `summary.csv` and every plot are views over it, recomputable
by `plot.py` without re-running the experiment (which costs 10 minutes per
seed).

The log records the queried point `x`, its reading `y`, the noise-free
`true_value`, and the wall-clock instant each stage happened. Oracle calls and
all regret calculations happen after the timed loop.
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

from wdbo_algo.mi_criterion import FSTAR_SOURCES
from wdbo_algo.mi_optimizer import MIDBOOptimizer
from wdbo_algo.optimizer import WDBOOptimizer
from scoring import score_run

# Paper H.1: the initial observations are sampled uniformly in S' x [0, 1/40]
# of the environment interval the run is about to cover. Gathered at a single
# instant they would carry no information about the temporal lengthscale lT.
INITIAL_TIME_FRACTION = 1.0 / 40.0

# The paper's wall-clock budget, and the one every arm in this repo is tuned
# around. It is no longer what sets the environment speed -- it is only the
# duration `default_env_speed` calibrates that speed against, so that a default
# 600 s run reproduces the paper's setting and a 200 s run is the *same*
# environment observed for less time.
REFERENCE_DURATION = 600.0

# Which removal criterion an arm uses. "none" is the ablation: clean() is never
# called, so the dataset grows monotonically.
CRITERIA = ("wasserstein", "mi", "none")

# Resolution of the common wall-clock grid for per-seed curves. The scorer
# adds configuration changes and oracle-table knots for the integral itself.
GRID_POINTS = 200

# The row index of the pseudo-query describing the configuration the run
# starts with: the last observation of the initial design. It is not a query
# the algorithm chose, so every metric excludes it (see `queries_only`), but
# it is the configuration in force from wall-clock zero until the first real
# query applies, so a post-hoc scorer cannot integrate the opening stretch
# without it.
INITIAL_ROW = -1


def query_fields(spatial_dim: int) -> list[str]:
    """Column order for `queries.csv`, given the benchmark's spatial dimension.

    The spatial dimension varies by benchmark (3 for ackley4d, 2 for
    temperature), so `x` is stored as flat `x_0 .. x_{d-1}` columns rather
    than a packed JSON string: pandas, awk and a spreadsheet can all read the
    flat form, and `load_run` reassembles it with `query_point`.

    Timestamps are all elapsed seconds since the optimization loop's zero, so
    a scorer can rebuild the step function `x_held(t)` without consulting
    anything else, and map it onto environment time with `env_start` and
    `env_speed` from `per_seed.csv` / `run.json`.
    """
    return [
        "seed", "iteration", "env_time", "wall_time",
        # When, in real seconds, each stage of the iteration happened.
        "t_iter_start", "t_apply", "t_result", "t_update_done",
        # How long each stage took.
        "t_acq", "t_eval", "t_fit", "t_acq_fit", "t_clean", "t_response",
        # What was queried and what came back.
        *[f"x_{i}" for i in range(spatial_dim)], "y", "true_value", "regret",
        "dataset_size", "n_removed",
        "lambda", "lS", "lT", "noise", "removal_budget",
        "min_criterion", "criterion_lT", "budget_spent",
    ]


def spatial_dim_of(run: list[dict]) -> int:
    """How many `x_i` columns a log has."""
    return sum(1 for key in run[0] if key.startswith("x_"))


def query_point(row: dict) -> np.ndarray:
    """The configuration a log row queried, reassembled from its `x_i` columns."""
    return np.array([row[f"x_{i}"] for i in range(sum(1 for k in row if k.startswith("x_")))])


def queries_only(run: list[dict]) -> list[dict]:
    """The rows the algorithm actually chose, i.e. everything but `INITIAL_ROW`.

    Every reported metric goes through this. The initial design is a warm start
    handed to the optimizer, so scoring it would credit or blame an arm for
    points it never selected -- and it has no timings to average, since it
    predates the wall clock.
    """
    return [row for row in run if row["iteration"] >= 0]


# --------------------------------------------------------------------------
# The environment clock
# --------------------------------------------------------------------------

def default_env_speed(env_span: tuple[float, float],
                      reference_duration: float = REFERENCE_DURATION) -> float:
    """Environment units per real second, calibrated so a 600 s run fills `env_span`.

    The paper runs every benchmark for 600 s over the whole of its temporal
    domain, so that is the speed we default to. Solving

        warmup_span + env_speed * H == span_width   with H = reference_duration
        warmup_span == INITIAL_TIME_FRACTION * env_speed * H

    for `env_speed` gives the expression below. The `(1 + 1/40)` accounts for
    the initial-design window, which occupies environment time ahead of the
    run: without it a default run would need slightly more environment than
    the benchmark defines, and on `temperature` -- where the data simply stops
    at `t = 1` -- that would mean extrapolating an RBF fit past its support.

    Every other duration then observes *this* environment for longer or
    shorter, which is the comparison section 4 of the benchmark note asks for.
    """
    lo, hi = env_span
    return (float(hi) - float(lo)) / (reference_duration * (1.0 + INITIAL_TIME_FRACTION))


def env_schedule(env_t0: float, env_speed: float, duration_seconds: float,
                 reference_duration: float = REFERENCE_DURATION) -> dict:
    """Resolve the environment times a run will touch, before it starts.

    Returns `warmup_span` (the environment interval the initial design is drawn
    over), `env_start` (the environment time the wall clock's zero corresponds
    to, i.e. no earlier than the last initial observation) and `env_end` (the
    environment time at `elapsed == duration_seconds`).

    `warmup_span` is deliberately sized against `reference_duration` rather
    than against this run's `duration_seconds`: it is H.1's fortieth of the
    *paper's* horizon, in environment units, and so depends only on
    `env_speed`. That makes `env_start` the same for every duration at a given
    speed, which in turn makes a 200 s run an exact prefix of a 600 s run --
    the same environment, watched for less time. Scaling it with
    `duration_seconds` would have reintroduced, in miniature, the coupling
    this whole change exists to remove.

    Callers use `env_end` twice: to check the objective can actually be
    evaluated that far, and as the MI criterion's `clip_horizon`.
    """
    warmup_span = INITIAL_TIME_FRACTION * env_speed * reference_duration
    env_start = env_t0 + warmup_span
    return {
        "env_t0": float(env_t0),
        "env_speed": float(env_speed),
        "warmup_span": float(warmup_span),
        "env_start": float(env_start),
        "env_end": float(env_start + env_speed * duration_seconds),
    }


# --------------------------------------------------------------------------
# The optimization loop
# --------------------------------------------------------------------------

def print_progress(fraction: float, row: dict, prefix: str = "", bar_width: int = 30):
    """Render a one-line, in-place progress bar for the current replication.

    `fraction` is progress through the *wall-clock* budget, `elapsed / H` -- no
    longer the environment clock, which now has units of its own.
    """
    filled = int(bar_width * fraction)
    bar = "#" * filled + "-" * (bar_width - filled)
    print(
        f"\r{prefix}[{bar}] {fraction * 100:5.1f}%"
        f" | size={row['dataset_size']:4d} | resp={row['t_response']:5.2f}s"
        f" | clean={row['t_clean']:5.2f}s | lT={row['lT']:.3g}",
        end="", flush=True,
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
             seed: int, env_t0: float, env_speed: float, progress_prefix: str = "",
             removal: bool = True, criterion: str = "wasserstein", min_dataset_size: int = 15,
             mi_options: dict | None = None) -> tuple[list[dict], dict]:
    """Run a single replication; return its raw per-query log and its run info.

    `criterion` selects the removal rule: "wasserstein" is W-DBO's own, "mi" is
    the mutual-information criterion, and "none" is the ablation in which
    `clean()` is never called, so the dataset grows monotonically -- same model,
    same kernels, same acquisition, same clock, only removal disabled. Under
    "none", `n_removed` and `removal_budget` are trivially 0 and 1.0, and
    `t_clean` is ~0. `removal=False` is kept as an alias for it.

    `env_t0` and `env_speed` fix the environment clock (see the module
    docstring): the initial design is drawn over
    `[env_t0, env_t0 + warmup_span]`, the wall clock starts at zero once that
    design is in hand, and query `i` is issued at
    `env_start + env_speed * elapsed_i`. Nothing about the environment depends
    on `duration_seconds` any more, so two durations at one `env_speed` are the
    same world watched for different lengths of time.

    The returned info dict carries what the log cannot: how long the initial
    design really took (charged to nobody -- it is *not* part of `H`), the
    environment interval actually covered, and whether the last iteration
    overran the deadline.

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

    schedule = env_schedule(env_t0, env_speed, duration_seconds)
    env_start = schedule["env_start"]

    optimizer = build_optimizer(objective, n_initial_observations, alpha, criterion,
                                min_dataset_size, mi_options, seed)

    # Paper H.1: the initial observations are sampled uniformly in S' x [0, 1/40]
    # of the run's environment interval -- spread out, not all at one instant, so
    # they say something about the temporal lengthscale lT, which the removal
    # budget (1 + alpha) ** (dt / lT) divides by.
    #
    # This is a warm start handed to the algorithm, not part of the contest: the
    # real seconds it costs are measured and reported separately rather than
    # back-dated into the budget the way earlier revisions did (which silently
    # assumed the warm-up took exactly H/40 seconds, whatever it really took).
    warmup_mark = time.perf_counter()
    warmup_times = env_t0 + np.sort(
        rng.uniform(0.0, schedule["warmup_span"], n_initial_observations))
    for t_env in warmup_times:
        x = optimizer.next_query(t_env)
        true_value = objective.evaluate(x, t_env)
        y = true_value + rng.normal(0.0, objective.noise_std)
        optimizer.tell(x, t_env, y)
    warmup_seconds = time.perf_counter() - warmup_mark

    # The last initial observation is the configuration the system is running
    # when the clock starts, and it stays in force until the first real query
    # applies. Logged as `INITIAL_ROW` so the opening stretch of the run can be
    # scored; excluded from every metric by `queries_only`. Its stage timings
    # are NaN because it predates the wall clock.
    log: list[dict] = [{
        "seed": seed,
        "iteration": INITIAL_ROW,
        "env_time": float(warmup_times[-1]),
        "wall_time": 0.0,
        "t_iter_start": 0.0, "t_apply": 0.0, "t_result": 0.0, "t_update_done": 0.0,
        "t_acq": float("nan"), "t_eval": float("nan"), "t_fit": float("nan"),
        "t_acq_fit": float("nan"), "t_clean": float("nan"), "t_response": float("nan"),
        **{f"x_{i}": float(v) for i, v in enumerate(np.ravel(x))},
        "y": y,
        "true_value": true_value,
        "regret": float("nan"),
        "dataset_size": optimizer.dataset_size(),
        "n_removed": 0,
        "lambda": optimizer._lambda,
        "lS": optimizer._lS,
        "lT": optimizer._lT,
        "noise": optimizer._noise,
        "removal_budget": optimizer._budget,
        "min_criterion": float("nan"),
        "criterion_lT": float("nan"),
        "budget_spent": 0.0,
    }]

    start = time.perf_counter()
    overran = False

    while True:
        # Check the deadline BEFORE committing to a query: a query started with
        # no budget left would be scored on an environment the run is not
        # supposed to reach. `H` is a hard stop, not a target to overshoot.
        t_iter_start = time.perf_counter() - start
        if t_iter_start >= duration_seconds:
            break
        # (ii) optimize the acquisition function. The previous configuration is
        # still the one in force throughout this: x_i does not exist yet.
        x = optimizer.next_query(env_start + env_speed * t_iter_start)
        t_apply = time.perf_counter() - start
        t_acq = t_apply - t_iter_start
        # Acquisition may itself consume the remaining budget. No measurement
        # is issued outside the scored horizon or the objective's support.
        if t_apply >= duration_seconds:
            overran = True
            break
        t_env = env_start + env_speed * t_apply

        # H.1: "the objective function is immediately sampled" -- querying f is
        # the experiment harness's cost, not the algorithm's, so it is timed out
        # of the response time (it still advances the wall clock).
        #
        # `t_apply` is when x_i takes effect and `t_result` is when its reading
        # is in hand. A post-hoc scorer treats x_i as held over
        # [t_apply_i, t_apply_{i+1}).
        true_value = objective.evaluate(x, t_env)
        y = true_value + rng.normal(0.0, objective.noise_std)
        t_result = time.perf_counter() - start
        t_eval = t_result - t_apply

        # (i) condition the GP and re-estimate the kernel and noise parameters.
        optimizer.tell(x, t_env, y)
        t_update_done = time.perf_counter() - start
        t_fit = t_update_done - t_result

        # (iii) remove stale observations. This advances the wall clock
        # (Algorithm 1 reads the clock after the removal loop), and -- a
        # deliberate departure from H.1, which counts only (i) + (ii) -- it IS
        # part of our response time: the next query cannot start until it ends,
        # so an expensive removal rule must pay for itself in `t_response`.
        size_before = optimizer.dataset_size()
        if criterion != "none" and t_update_done < duration_seconds:
            optimizer.clean(env_start + env_speed * t_update_done)
        step_end = time.perf_counter()
        t_step_end = step_end - start

        # The MLE hyperparameters and removal budget as the next iteration will
        # see them. Read off private attributes: `wdbo_algo` is vendored in this
        # repo (src/) and exposes no accessor, and lT in particular is the whole
        # story behind a dataset collapse.
        log.append({
            "seed": seed,
            "iteration": len(log) - 1,  # the INITIAL_ROW occupies log[0]
            "env_time": t_env,
            "wall_time": t_step_end,
            # When each stage happened, so `x_held(t)` can be rebuilt offline.
            "t_iter_start": t_iter_start,
            "t_apply": t_apply,
            "t_result": t_result,
            "t_update_done": t_update_done,
            # Our response time is (i) + (ii) + (iii). H.1's is (i) + (ii) only,
            # kept as `t_acq_fit` for comparison with the paper. The parts are
            # logged separately so a slow arm can be blamed on the right stage.
            "t_acq": t_acq,
            "t_eval": t_eval,
            "t_fit": t_fit,
            "t_acq_fit": t_acq + t_fit,
            "t_clean": t_step_end - t_update_done,
            "t_response": t_acq + t_fit + (t_step_end - t_update_done),
            **{f"x_{i}": float(v) for i, v in enumerate(np.ravel(x))},
            "y": y,
            "true_value": true_value,
            "regret": float("nan"),
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

        # An iteration that started inside the budget may finish outside it. We
        # let it finish -- killing a half-conditioned GP would corrupt the run --
        # but record the fact, because the scoring pass must integrate only up
        # to H and needs to know the last row runs past the end.
        overran = t_step_end > duration_seconds
        print_progress(min(1.0, t_step_end / duration_seconds), log[-1], prefix=progress_prefix)

    if len(log) > 1:
        print()  # close the in-place progress bar

    info = {
        "seed": seed,
        "warmup_seconds": warmup_seconds,
        "elapsed_seconds": time.perf_counter() - start,
        "overran": overran,
        **schedule,
    }
    return log, info


# --------------------------------------------------------------------------
# Summary statistics
# --------------------------------------------------------------------------

def _mean_of(run: list[dict], key: str) -> float:
    """Mean of `key` over the run, or NaN if a legacy log never recorded it."""
    values = [row[key] for row in run if key in row]
    return float(np.mean(values)) if values else float("nan")


def summarize_seed(run: list[dict], score: dict,
                   info: dict | None = None) -> dict:
    """One row of per-replication diagnostics.

    `time_avg_regret` integrates the held configuration over the full horizon.
    `avg_regret` is the separate plain mean at measurement instants.

    `info` is `run_once`'s second return value; when given, its warm-up cost
    and environment interval are carried into `per_seed.csv`.
    """
    initial = run[0]
    run = queries_only(run)
    regret = np.array([row["regret"] for row in run])

    # NaN on every query of the no-removal arm, which never runs a cleaning loop,
    # and on any query whose loop broke before scoring anything.
    min_criterion = np.array([row.get("min_criterion", np.nan) for row in run], dtype=float)
    median_min_criterion = float(np.nanmedian(min_criterion)) if np.any(np.isfinite(min_criterion)) else float("nan")

    extra = {} if info is None else {
        "warmup_seconds": info["warmup_seconds"],
        "elapsed_seconds": info["elapsed_seconds"],
        # Whether the final iteration ran past the deadline. A scorer clips its
        # integral at H regardless, but this says so without it having to infer
        # the fact from `wall_time`.
        "overran": int(info["overran"]),
        "env_speed": info["env_speed"],
        "env_start": info["env_start"],
        "env_end": info["env_end"],
    }

    return {
        "seed": initial["seed"],
        "iterations": len(run),
        "time_avg_regret": score["time_avg_regret"],
        "avg_regret": float(regret.mean()) if len(run) else float("nan"),
        # acquisition + fit + clean; the paper's (i) + (ii) is `avg_acq_fit_time`.
        "avg_response_time": _mean_of(run, "t_response"),
        "avg_acq_fit_time": _mean_of(run, "t_acq_fit"),
        # NaN on a legacy log, which only recorded the (i) + (ii) total.
        "avg_acq_time": _mean_of(run, "t_acq"),
        "avg_fit_time": _mean_of(run, "t_fit"),
        "avg_eval_time": _mean_of(run, "t_eval"),
        "avg_clean_time": _mean_of(run, "t_clean"),
        "final_dataset_size": (run[-1] if run else initial)["dataset_size"],
        "max_dataset_size": max(row["dataset_size"] for row in [initial, *run]),
        "min_dataset_size": min(row["dataset_size"] for row in [initial, *run]),
        "median_lT": float(np.median([row["lT"] for row in run])) if run else float("nan"),
        "median_min_criterion": median_min_criterion,
        "total_removed": sum(row["n_removed"] for row in run),
        **extra,
    }


HEADLINE_METRICS = [
    ("time_average_regret", "time_avg_regret"),
    ("query_average_regret", "avg_regret"),
    ("response_time_s", "avg_response_time"),
    ("acq_fit_time_s", "avg_acq_fit_time"),
    ("clean_time_s", "avg_clean_time"),
    ("iterations", "iterations"),
]


def headline(per_seed: list[dict]) -> list[dict]:
    """Mean and standard error across replications, for the metrics worth quoting.

    Standard error rather than variance: Table 2 underlines algorithms whose
    confidence intervals overlap the best one's, so the SEM is what makes a
    reproduction comparable to it.
    """
    rows = []
    for name, key in HEADLINE_METRICS:
        values = np.array([row[key] for row in per_seed], dtype=float)
        finite = values[np.isfinite(values)]
        sem = float(finite.std(ddof=1) / np.sqrt(len(finite))) if len(finite) > 1 else 0.0
        rows.append({"metric": name, "mean": float(finite.mean()) if len(finite) else float("nan"),
                     "sem": sem, "n_runs": len(finite)})
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


def score_results(runs: list[list[dict]], metadata: dict, objective,
                  duration_seconds: float, infos: list[dict] | None = None):
    """Produce per-seed scores and curves from raw logs and the run clock."""
    infos = infos or [None] * len(runs)
    if len(infos) != len(runs):
        raise ValueError("One run-info record is required per run")
    schedule = metadata.get("env_schedule")
    if not schedule:
        raise ValueError("Missing environment schedule; old logs must be rerun")
    scores = [score_run(run, objective, duration_seconds,
                        float(schedule["env_start"]), float(schedule["env_speed"]),
                        grid_points=GRID_POINTS)
              for run in runs]
    return [summarize_seed(run, score, info)
            for run, score, info in zip(runs, scores, infos)], scores


def save_run(out_dir: Path, runs: list[list[dict]], metadata: dict, duration_seconds: float,
             objective, infos: list[dict] | None = None) -> tuple[list[dict], list[dict]]:
    """Write the raw log, the per-seed table, the headline table and run.json.

    `infos` is the list of `run_once` info dicts, one per replication; passing
    it adds the warm-up cost and the environment interval to `per_seed.csv`.
    """
    per_seed, scores = score_results(runs, metadata, objective, duration_seconds, infos)
    out_dir.mkdir(parents=True, exist_ok=True)

    fields = query_fields(spatial_dim_of(runs[0]))
    write_csv(out_dir / "queries.csv", [row for run in runs for row in run], fields)
    write_csv(out_dir / "per_seed.csv", per_seed)
    write_csv(out_dir / "summary.csv", headline(per_seed))
    (out_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    return per_seed, scores


def load_run(out_dir: Path) -> tuple[list[list[dict]], dict]:
    """Read `queries.csv` + `run.json` back as one list per replication.

    Replications are split wherever `iteration` stops increasing, not grouped
    by `seed`: under `--same-seed` every replication carries the same seed, so
    grouping by it would silently concatenate all ten into a single run. The
    rule is written this way rather than as `iteration == 0` so that it handles
    both the current logs (which open each replication with `INITIAL_ROW`, i.e.
    -1) and older ones (which start at 0).

    Older logs lacking x or application events cannot be scored and must be
    rerun. Their old summary and plots must not be relabeled as time regret.
    Logs predating the `t_response` column get it rebuilt as
    `t_acq_fit + t_clean`, which is exactly what the loop now writes.
    """
    metadata = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))

    ints = {"seed", "iteration", "dataset_size", "n_removed"}
    runs: list[list[dict]] = []
    with open(out_dir / "queries.csv", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if not {"t_apply", "env_time", "x_0"}.issubset(fields):
            raise ValueError(f"{out_dir}: log lacks x or application times; rerun the experiment")
        for raw in reader:
            row = {k: (int(v) if k in ints else float(v))
                   for k, v in raw.items()}
            # Logs written before response time included cleaning: rebuild it
            # from the stages they did record.
            if "t_response" not in row:
                row["t_response"] = row["t_acq_fit"] + row["t_clean"]
            if not runs or row["iteration"] <= runs[-1][-1]["iteration"]:
                runs.append([])
            runs[-1].append(row)

    return runs, metadata


def _import_benchmark_module(benchmark_dir: Path, filename: str, alias: str):
    """Import `<benchmark_dir>/<filename>` under a unique module name.

    Both benchmarks ship a module literally called `objective`, and each
    `run_experiment.py` reaches it by putting its own directory on `sys.path`.
    That is fine while a process only ever touches one benchmark -- but a
    scoring pass that handles both would `import objective` twice and silently
    get whichever directory came first on the path, scoring one benchmark
    against the other's objective.

    So load each file explicitly, under `alias`, and register it in
    `sys.modules` so repeated calls are cheap. `synthetic/objective.py` does
    `from benchmarks import Benchmark` at import time, so its directory is put
    on the path first and taken off afterwards.
    """
    import importlib.util

    if alias in sys.modules:
        return sys.modules[alias]

    path_entry = str(benchmark_dir)
    added = path_entry not in sys.path
    if added:
        sys.path.insert(0, path_entry)
    try:
        spec = importlib.util.spec_from_file_location(alias, benchmark_dir / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if added:
            sys.path.remove(path_entry)


def load_objective(metadata: dict):
    """Rebuild the exact objective a finished run used, from its `run.json`.

    A scoring pass needs `f(x, t)` and `f*(t)` to re-evaluate the configuration
    held at arbitrary times, so it reconstructs the objective with the same
    oracle density and grid resolution recorded for the run.

    Everything needed is already in `run.json`; this just spares every caller
    from reassembling it (and from the `objective` module-name collision).
    The oracle curve is read from the same cache the run built, so this is
    cheap for the synthetic benchmark; `temperature` still has to refit its
    RBF interpolator, which costs ~30-60 s.
    """
    here = Path(__file__).resolve().parent
    args = metadata["args"]

    if metadata["benchmark"] == "temperature":
        objective = _import_benchmark_module(here / "temperature", "objective.py",
                                             "wdbo_temperature_objective")
        density = float(args.get("oracle_density", objective.DEFAULT_ORACLE_DENSITY))
        # Runs from before the flag existed all searched the default grid.
        grid = int(args.get("oracle_grid_resolution") or objective.DEFAULT_ORACLE_GRID_RESOLUTION)
        smoothing = float(args["smoothing"])
        cache = metadata.get("oracle_cache") or args.get("oracle_cache")
        if cache is None:
            paths = _import_benchmark_module(here / "temperature", "paths.py",
                                             "wdbo_temperature_paths")
            cache = paths.DATA_DIR / objective.oracle_cache_name(density, grid, smoothing)
        return objective.build_objective(
            Path(args["processed"]),
            smoothing=smoothing,
            oracle_density=density,
            oracle_grid_resolution=grid,
            oracle_cache_path=Path(cache),
        )

    benchmarks = _import_benchmark_module(here / "synthetic", "benchmarks.py",
                                          "wdbo_synthetic_benchmarks")
    objective = _import_benchmark_module(here / "synthetic", "objective.py",
                                         "wdbo_synthetic_objective")
    benchmark = benchmarks.get_benchmark(metadata["benchmark"])
    env_span = tuple(args["env_span"]) if args.get("env_span") else benchmark.env_span
    density = float(args.get("oracle_density", objective.DEFAULT_ORACLE_DENSITY))
    grid = int(args.get("oracle_grid_resolution", 33))
    cache = metadata.get("oracle_cache") or args.get("oracle_cache")
    if cache is None:
        paths = _import_benchmark_module(here / "synthetic", "paths.py",
                                         "wdbo_synthetic_paths")
        cache = paths.DATA_DIR / benchmark.name / objective.oracle_cache_name(env_span, density, grid)
    return objective.build_objective(
        benchmark,
        env_span=env_span,
        oracle_density=density,
        oracle_grid_resolution=grid,
        oracle_cache_path=Path(cache),
    )


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def _resample(runs: list[list[dict]], duration_seconds: float, values) -> tuple[np.ndarray, np.ndarray]:
    """Resample logged state, held until the next completed iteration."""
    grid = np.linspace(0.0, duration_seconds, GRID_POINTS)
    curves = []
    for run in runs:
        times = np.asarray([row["wall_time"] for row in run], dtype=float)
        state = np.asarray(values(run), dtype=float)
        indices = np.clip(np.searchsorted(times, grid, side="right") - 1, 0, len(run) - 1)
        curves.append(state[indices])
    curves = np.stack(curves)
    return grid, curves


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


def save_duration_plot(runs: list[list[dict]], scores: list[dict], duration_seconds: float,
                       path: Path, title: str = ""):
    """Plot the integrated time regret and dataset size against elapsed time."""
    import matplotlib.pyplot as plt

    grid = scores[0]["grid"]
    regret = np.stack([score["running_time_regret"] for score in scores])
    _, size = _resample(runs, duration_seconds, lambda run: [r["dataset_size"] for r in run])

    fig, (regret_ax, size_ax) = plt.subplots(1, 2, figsize=(10, 4))
    _seed_lines(regret_ax, grid, regret)
    regret_ax.set(xlabel="Duration (s)", ylabel="Time-average regret up to t", title="Held-configuration regret")
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

    Response time here is acquisition + fit + clean (`t_response`), not the
    paper's acquisition + fit, so an arm's removal cost shows on the x-axis.
    """
    import matplotlib.pyplot as plt
    import textwrap

    response = np.array([row["t_response"] for run in runs for row in queries_only(run)])
    regret = np.array([row["regret"] for run in runs for row in queries_only(run)])
    valid = [row for row in per_seed if row["iterations"] > 0]
    seed_response = np.array([row["avg_response_time"] for row in valid])
    seed_regret = np.array([row["avg_regret"] for row in valid])

    fig, ax = plt.subplots(figsize=(5, 4))
    if len(response):
        ax.scatter(response, regret, s=10, alpha=0.2, color="tab:blue", label="Individual queries")
        ax.scatter(seed_response, seed_regret, s=45, color="tab:orange", edgecolor="black",
                   linewidth=0.5, zorder=3, label="Per-seed average")
        ax.plot(seed_response.mean(), seed_regret.mean(), "X", color="black",
                markersize=12, zorder=4, label="Mean across seeds")
        ax.set_xscale("log")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No queries within this run", ha="center", va="center",
                transform=ax.transAxes)
    ax.set(xlabel="Acquisition + fit + clean time (s)", ylabel="Query regret")
    ax.set_title(textwrap.fill(title or "Query regret vs. response time", width=42), fontsize=9)

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
               scores: list[dict], duration_seconds: float, title: str = ""):
    """Render every plot for a results directory. Also the entry point for plot.py."""
    save_duration_plot(runs, scores, duration_seconds, out_dir / "regret_and_size_vs_duration.png", title)
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
    parser.add_argument("--mi-fstar", choices=FSTAR_SOURCES, default="loo",
                        help="How f*_t is sampled. loo: one Gumbel fit per leave-one-out posterior "
                             "(n per future time). full: one fit under the full data, shared by every "
                             "observation (cheap, but conditions f*_t on the observation being scored). "
                             "is: the full-data samples reweighted towards each leave-one-out posterior "
                             "by importance sampling (Week 3); costs about as much as full. "
                             "See MI_CRITERION.md section 3.")
    parser.add_argument("--mi-fstar-full", action="store_true",
                        help="Deprecated alias for --mi-fstar full.")
    parser.add_argument("--mi-clip-horizon", action="store_true",
                        help="Cap the criterion's future horizon at the environment time the run ends at, "
                             "instead of letting it run as far as the model's own lengthscale reaches.")
    parser.add_argument("--no-removal", action="store_true",
                        help="Alias for --criterion none. Compare against a removal arm at the same "
                             "--duration-seconds, never at the same iteration count.")
    return parser


def criterion_settings(args, env_end: float) -> tuple[str, float, dict, str]:
    """Resolve the removal-rule flags into what `run_once` and the run label need.

    `env_end` is the environment time the run finishes at -- `env_schedule`'s
    `env_end`. It is what `--mi-clip-horizon` caps the criterion's lookahead
    at. This used to be hard-coded to 1.0, which was only ever right while the
    optimizer's clock was the normalized `[0, 1]` one; on an absolute
    environment clock a hard-coded 1.0 would silently clip the horizon to
    nothing (Ackley starts at t = -32) and quietly disable the flag.

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
        fstar_source="full" if args.mi_fstar_full else args.mi_fstar,
        clip_horizon=env_end if args.mi_clip_horizon else None,
    )
    fstar_note = {"loo": "", "full": ", f* from full D", "is": ", f* by importance sampling"}[mi_options["fstar_source"]]
    return criterion, args.mi_alpha, mi_options, f"mi, alpha={args.mi_alpha:g} nats{fstar_note}"


def print_headline(rows: list[dict], reference: float | None = None):
    """Print the quotable numbers, next to the paper's Table 2 value if known."""
    for row in rows:
        line = f"  {row['metric']:<32} {row['mean']:10.4f} +/- {row['sem']:.4f} (n={row['n_runs']})"
        if reference is not None and row["metric"] == "query_average_regret" and row["n_runs"]:
            line += f"   [paper Table 2: {reference:.2f}; convention may differ]"
        print(line)
