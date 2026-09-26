"""Run paired Temperature methods sequentially through the shared fair harness.

Every method receives the same seed, initial-design convention, environment
clock, deadline, logger, scorer, and plotting path. Method order rotates by
seed to reduce machine warm-up and order effects; methods never run in
parallel because wall-clock latency is part of the experiment.
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import common
from objective import (DEFAULT_ORACLE_DENSITY, DEFAULT_ORACLE_GRID_RESOLUTION,
                       ENV_SPAN, build_objective, oracle_cache_name)
from paths import DATA_DIR


def _rotated(items, offset):
    offset %= len(items)
    return items[offset:] + items[:offset]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--processed", type=Path, default=DATA_DIR / "processed.npz")
    parser.add_argument("--oracle-cache", type=Path, default=None)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--oracle-density", type=float, default=DEFAULT_ORACLE_DENSITY)
    parser.add_argument("--oracle-grid-resolution", type=int,
                        default=DEFAULT_ORACLE_GRID_RESOLUTION)
    parser.add_argument("--duration-seconds", type=float, default=common.REFERENCE_DURATION)
    parser.add_argument("--env-speed", type=float, default=None)
    parser.add_argument("--env-t0", type=float, default=0.0)
    parser.add_argument("--n-initial-observations", type=int, default=15)
    parser.add_argument("--methods", nargs="+", choices=common.CRITERIA,
                        default=["wasserstein", "none", "dual_gate"])
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path,
                        default=DATA_DIR / "comparison_shared_harness")
    common.add_criterion_arguments(parser)
    args = parser.parse_args()

    if args.duration_seconds <= 0 or args.n_seeds <= 0:
        parser.error("duration and n-seeds must be positive")
    if len(set(args.methods)) != len(args.methods):
        parser.error("methods must not contain duplicates")

    env_speed = args.env_speed if args.env_speed is not None else common.default_env_speed(ENV_SPAN)
    schedule = common.env_schedule(args.env_t0, env_speed, args.duration_seconds)
    oracle_cache = args.oracle_cache or DATA_DIR / oracle_cache_name(
        args.oracle_density, args.oracle_grid_resolution, args.smoothing)
    objective = build_objective(
        args.processed, smoothing=args.smoothing,
        oracle_density=args.oracle_density,
        oracle_grid_resolution=args.oracle_grid_resolution,
        oracle_cache_path=oracle_cache,
    )
    objective.assert_covers(args.env_t0, schedule["env_end"])

    runs_by_method = {method: [] for method in args.methods}
    infos_by_method = {method: [] for method in args.methods}
    execution_order = []
    for seed_index in range(args.n_seeds):
        seed = args.seed + seed_index
        for method in _rotated(list(args.methods), seed_index):
            args.criterion = method
            args.no_removal = method == "none"
            criterion, alpha, options, _ = common.criterion_settings(args, schedule["env_end"])
            print(f"Running seed={seed}, method={method} in isolation")
            run, info = common.run_once(
                objective, args.duration_seconds, args.n_initial_observations,
                alpha, seed, args.env_t0, env_speed,
                progress_prefix=f"{method} seed {seed} ", criterion=criterion,
                min_dataset_size=args.min_dataset_size, mi_options=options,
            )
            runs_by_method[method].append(run)
            infos_by_method[method].append(info)
            execution_order.append({"seed": seed, "method": method})

    comparison_rows = []
    for method in args.methods:
        method_dir = args.output_dir / method
        title = (f"temperature, {method}, {args.duration_seconds:g}s @ "
                 f"{env_speed:g} units/s x {args.n_seeds} seed(s)")
        metadata = common.run_metadata(vars(args), extra={
            "benchmark": "temperature", "criterion": method, "title": title,
            "noise_std": objective.noise_std, "env_schedule": schedule,
            "oracle_cache": str(oracle_cache), "execution_order": execution_order,
        })
        per_seed, scores = common.save_run(
            method_dir, runs_by_method[method], metadata, args.duration_seconds,
            objective, infos_by_method[method])
        common.save_plots(method_dir, runs_by_method[method], per_seed, scores,
                          args.duration_seconds, title)
        for row in common.headline(per_seed):
            comparison_rows.append({"method": method, **row})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    common.write_csv(args.output_dir / "comparison_summary.csv", comparison_rows)
    with (args.output_dir / "execution_order.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("seed", "method"))
        writer.writeheader()
        writer.writerows(execution_order)
    print(f"Results in {args.output_dir}")


if __name__ == "__main__":
    main()
