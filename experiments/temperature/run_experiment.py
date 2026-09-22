"""Reproduce the WDBO paper's "Temperature" benchmark (Appendix H.1 & H.2).

Runs WDBOOptimizer against the interpolated real-data objective built by
objective.py for a real wall-clock budget, replicated over independent seeds
like the paper's own 10-seed averaging.

Everything downstream of the optimization loop -- the log, the summaries, the
run metadata, the plots -- lives in `../common.py` and is shared with the
synthetic experiment. See ../README.md for the experiment design and
README.md for what each output file means.

Two knobs, not one: `--duration-seconds` buys compute, `--env-speed` sets how
fast the sensor day plays back. They default to the paper's setting together;
move one at a time. Unlike the synthetic benchmark the environment here is
finite -- the data stops at t = 1 -- so a run longer than 600s must slow the
environment down to fit.

Usage:
    python experiments/temperature/preprocess.py
    python experiments/temperature/run_experiment.py --n-seeds 10 --label paper

    # Same environment, a third of the budget.
    python experiments/temperature/run_experiment.py --duration-seconds 200
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))  # for `common`

import common
from objective import DEFAULT_ORACLE_DENSITY, ENV_SPAN, build_objective
from paths import DATA_DIR

# Average regret reported for W-DBO on Temperature in the paper's Table 2.
PAPER_TABLE_2_AVERAGE_REGRET = 0.68


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--processed", type=Path, default=DATA_DIR / "processed.npz")
    parser.add_argument("--oracle-cache", type=Path, default=None,
                        help="Defaults to data/temperature/oracle_d<density>.npz. The density is in the "
                             "name because a table built at another density is a different table.")
    parser.add_argument("--smoothing", type=float, default=1.0, help="RBF interpolation smoothing (in temperature units^2).")
    parser.add_argument("--duration-seconds", type=float, default=common.REFERENCE_DURATION,
                        help="Real wall-clock budget per replication, measured from the end of the initial "
                             "design (paper default: 600s). This buys compute only: it no longer changes how "
                             "fast the environment moves.")
    parser.add_argument("--n-initial-observations", type=int, default=15, help="Initial observations, drawn over S' x the first 1/40 of the run's environment interval, per H.1.")
    parser.add_argument("--env-speed", type=float, default=None,
                        help="Environment units per real second, over the [0, 1] day the sensor data covers. "
                             "Defaults to the speed at which a 600s run consumes exactly that day, i.e. the "
                             "paper's setting. A longer run needs a proportionally LOWER speed -- there is no "
                             "more data past t = 1.")
    parser.add_argument("--env-t0", type=float, default=0.0,
                        help="Environment time the initial design starts at.")
    parser.add_argument("--oracle-density", type=float, default=DEFAULT_ORACLE_DENSITY,
                        help="Oracle table samples per unit of environment time.")
    common.add_criterion_arguments(parser)
    parser.add_argument("--n-seeds", type=int, default=10, help="Number of independent replications (paper uses 10).")
    parser.add_argument("--seed", type=int, default=0, help="Base seed; replication i uses seed + i.")
    parser.add_argument("--same-seed", action="store_true", help="Use the same --seed for every replication instead of seed + i. Diagnostic only.")
    parser.add_argument("--label", default=None, help="Suffix for the timestamped results directory, e.g. --label paper.")
    parser.add_argument("--results-dir", type=Path, default=None, help="Write results here instead of a fresh timestamped directory.")
    args = parser.parse_args()

    out_dir = args.results_dir or common.results_dir(DATA_DIR / "results", args.label)
    oracle_cache = args.oracle_cache or DATA_DIR / f"oracle_d{args.oracle_density:g}.npz"

    # The environment clock, resolved before anything is built: `env_speed`
    # defaults to the paper's 600s-per-day setting whatever --duration-seconds
    # says, which is the whole point of separating the two.
    env_speed = args.env_speed if args.env_speed is not None else common.default_env_speed(ENV_SPAN)
    schedule = common.env_schedule(args.env_t0, env_speed, args.duration_seconds)
    print(f"Environment clock: speed={env_speed:g} units/s, run covers "
          f"[{schedule['env_start']:g}, {schedule['env_end']:g}] over {args.duration_seconds:g}s "
          f"(initial design over [{args.env_t0:g}, {schedule['env_start']:g}])")

    print("Building objective (RBF fit + oracle grid search, ~30-60s, cached afterwards)...")
    objective = build_objective(args.processed, smoothing=args.smoothing,
                                oracle_density=args.oracle_density,
                                oracle_cache_path=oracle_cache)
    # Before spending 10 minutes a seed: check the run stays inside the data.
    objective.assert_covers(args.env_t0, schedule["env_end"])
    print(f"Noise std (5% of signal variance): {objective.noise_std:.4f}")

    criterion, alpha, mi_options, variant = common.criterion_settings(args, schedule["env_end"])

    runs, infos = [], []
    for i in range(args.n_seeds):
        prefix = f"seed {i + 1}/{args.n_seeds} " if args.n_seeds > 1 else ""
        seed = args.seed if args.same_seed else args.seed + i
        run, info = common.run_once(objective, args.duration_seconds, args.n_initial_observations,
                                    alpha, seed, args.env_t0, env_speed, progress_prefix=prefix,
                                    criterion=criterion, min_dataset_size=args.min_dataset_size,
                                    mi_options=mi_options)
        runs.append(run)
        infos.append(info)

    title = (f"temperature (d'=3), {variant}, {args.duration_seconds:g}s "
             f"@ {env_speed:g} units/s x {args.n_seeds} seed(s)")
    metadata = common.run_metadata(vars(args), extra={
        "benchmark": "temperature",
        "criterion": criterion,
        "title": title,
        "noise_std": objective.noise_std,
        "env_schedule": schedule,
        "oracle_cache": str(oracle_cache),
        "paper_table_2_average_regret": PAPER_TABLE_2_AVERAGE_REGRET,
    })

    per_seed, scores = common.save_run(out_dir, runs, metadata, args.duration_seconds, objective, infos)
    common.save_plots(out_dir, runs, per_seed, scores, args.duration_seconds, title)

    print(f"\n{title}")
    common.print_headline(common.headline(per_seed), reference=PAPER_TABLE_2_AVERAGE_REGRET)
    print(f"\nResults in {out_dir}")


if __name__ == "__main__":
    main()
