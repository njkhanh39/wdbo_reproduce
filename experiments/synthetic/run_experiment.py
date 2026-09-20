"""Reproduce the WDBO paper's synthetic benchmarks (Appendix H.1 & H.2).

Runs WDBOOptimizer against a closed-form dynamic objective (see benchmarks.py
and objective.py) for a real wall-clock budget, replicated over independent
seeds like the paper's own 10-seed averaging.

Everything downstream of the optimization loop -- the log, the summaries, the
run metadata, the plots -- lives in `../common.py` and is shared with the
temperature experiment. See ../README.md for the experiment design and
README.md for what each output file means.

Two knobs, not one: `--duration-seconds` buys compute, `--env-speed` sets how
fast the environment moves. They default to the paper's setting together; move
one at a time.

Usage:
    python experiments/synthetic/run_experiment.py --n-seeds 10 --label paper

    # Same environment, a third of the budget -- the comparison the coupled
    # clock used to make impossible.
    python experiments/synthetic/run_experiment.py --duration-seconds 200

    # Same budget, an environment that moves twice as fast.
    python experiments/synthetic/run_experiment.py --env-speed 0.208
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))  # for `common`

import common
from benchmarks import get_benchmark
from objective import DEFAULT_ORACLE_DENSITY, build_objective, oracle_cache_name
from paths import DATA_DIR

# Average regret reported for W-DBO in the paper's Table 2, per benchmark, so a
# finished run can print its headline number next to the target.
PAPER_TABLE_2 = {"ackley4d": 2.24}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--benchmark", default="ackley4d", help="Synthetic benchmark name (see benchmarks.py).")
    parser.add_argument("--duration-seconds", type=float, default=common.REFERENCE_DURATION,
                        help="Real wall-clock budget per replication, measured from the end of the initial "
                             "design (paper default: 600s). This buys compute only: it no longer changes how "
                             "fast the environment moves.")
    parser.add_argument("--n-initial-observations", type=int, default=15, help="Initial observations, drawn over S' x the first 1/40 of the run's environment interval, per H.1.")
    parser.add_argument("--env-span", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                        help="The benchmark's whole temporal domain, in the function's own units. Sets the "
                             "oracle table's range and the default --env-speed and --env-t0. Defaults to the "
                             "benchmark's own span from Appendix H.2 (Ackley: -32 32).")
    parser.add_argument("--env-speed", type=float, default=None,
                        help="Environment units per real second. Defaults to the speed at which a 600s run "
                             "covers --env-span exactly, i.e. the paper's setting. Raise it to make the "
                             "target move faster without touching the compute budget.")
    parser.add_argument("--env-t0", type=float, default=None,
                        help="Environment time the initial design starts at. Defaults to the low end of "
                             "--env-span.")
    common.add_criterion_arguments(parser)
    parser.add_argument("--n-seeds", type=int, default=10, help="Number of independent replications (paper uses 10).")
    parser.add_argument("--seed", type=int, default=0, help="Base seed; replication i uses seed + i.")
    parser.add_argument("--same-seed", action="store_true", help="Use the same --seed for every replication instead of seed + i. Diagnostic only.")
    parser.add_argument("--oracle-density", type=float, default=DEFAULT_ORACLE_DENSITY,
                        help="Oracle table samples per unit of environment time. Fixed per unit rather than "
                             "per run, so table resolution does not drift when --env-speed does. Ackley "
                             "oscillates about once per unit of time, so this is samples per period.")
    parser.add_argument("--oracle-grid-resolution", type=int, default=33, help="Per-axis spatial grid nodes for the oracle search (use an odd number).")
    parser.add_argument("--oracle-cache", type=Path, default=None, help="Defaults to data/synthetic/<benchmark>/oracle_<span>_<density>_<grid>.npz.")
    parser.add_argument("--label", default=None, help="Suffix for the timestamped results directory, e.g. --label paper.")
    parser.add_argument("--results-dir", type=Path, default=None, help="Write results here instead of a fresh timestamped directory.")
    args = parser.parse_args()

    benchmark = get_benchmark(args.benchmark)
    span = args.env_span if args.env_span is not None else benchmark.env_span
    env_span = (float(span[0]), float(span[1]))

    # The environment clock, resolved before anything is built: `env_speed`
    # defaults to the paper's 600s-per-span setting whatever --duration-seconds
    # says, which is the whole point of separating the two.
    env_speed = args.env_speed if args.env_speed is not None else common.default_env_speed(env_span)
    env_t0 = args.env_t0 if args.env_t0 is not None else env_span[0]
    schedule = common.env_schedule(env_t0, env_speed, args.duration_seconds)

    base_dir = DATA_DIR / benchmark.name
    oracle_cache = args.oracle_cache or base_dir / oracle_cache_name(
        env_span, args.oracle_density, args.oracle_grid_resolution)
    out_dir = args.results_dir or common.results_dir(base_dir / "results", args.label)
    lo, hi = benchmark.spatial_domain[0]

    print(f"Benchmark: {benchmark.name} (d'={benchmark.dim}: spatial d={benchmark.spatial_dim} "
          f"in [{lo:g}, {hi:g}]^{benchmark.spatial_dim}, time in [{env_span[0]:g}, {env_span[1]:g}])")
    print(f"Environment clock: speed={env_speed:g} units/s, run covers "
          f"[{schedule['env_start']:g}, {schedule['env_end']:g}] over {args.duration_seconds:g}s "
          f"(initial design over [{env_t0:g}, {schedule['env_start']:g}])")
    print(f"Building objective (oracle grid search {args.oracle_grid_resolution}^{benchmark.spatial_dim} "
          f"x {args.oracle_density:g} samples per unit time, cached to {oracle_cache})...")
    objective = build_objective(
        benchmark,
        env_span=env_span,
        oracle_density=args.oracle_density,
        oracle_grid_resolution=args.oracle_grid_resolution,
        oracle_cache_path=oracle_cache,
    )
    # Before spending 10 minutes a seed: check the run stays inside the table.
    objective.assert_covers(env_t0, schedule["env_end"])
    print(f"Noise std (5% of signal variance): {objective.noise_std:.4f}")

    criterion, alpha, mi_options, variant = common.criterion_settings(args, schedule["env_end"])

    runs, infos = [], []
    for i in range(args.n_seeds):
        prefix = f"seed {i + 1}/{args.n_seeds} " if args.n_seeds > 1 else ""
        seed = args.seed if args.same_seed else args.seed + i
        run, info = common.run_once(objective, args.duration_seconds, args.n_initial_observations,
                                    alpha, seed, env_t0, env_speed, progress_prefix=prefix,
                                    criterion=criterion, min_dataset_size=args.min_dataset_size,
                                    mi_options=mi_options)
        runs.append(run)
        infos.append(info)

    title = (f"{benchmark.name} (d'={benchmark.dim}), {variant}, {args.duration_seconds:g}s "
             f"@ {env_speed:g} units/s x {args.n_seeds} seed(s)")
    metadata = common.run_metadata(vars(args) | {"env_span": list(env_span)}, extra={
        "benchmark": benchmark.name,
        "criterion": criterion,
        "title": title,
        "noise_std": objective.noise_std,
        "env_schedule": schedule,
        "oracle_cache": str(oracle_cache),
        "paper_table_2_average_regret": PAPER_TABLE_2.get(benchmark.name),
    })

    per_seed = common.save_run(out_dir, runs, metadata, args.duration_seconds, infos)
    common.save_plots(out_dir, runs, per_seed, args.duration_seconds, title)

    print(f"\n{title}")
    common.print_headline(common.headline(per_seed), reference=PAPER_TABLE_2.get(benchmark.name))
    print(f"\nResults in {out_dir}")


if __name__ == "__main__":
    main()
