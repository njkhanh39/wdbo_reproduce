"""Reproduce the WDBO paper's synthetic benchmarks (Appendix H.1 & H.2).

Runs WDBOOptimizer against a closed-form dynamic objective (see benchmarks.py
and objective.py) for a real wall-clock budget, replicated over independent
seeds like the paper's own 10-seed averaging.

Everything downstream of the optimization loop -- the log, the summaries, the
run metadata, the plots -- lives in `../common.py` and is shared with the
temperature experiment. See README.md for what each output file means.

Usage:
    python experiments/synthetic/run_experiment.py --n-seeds 10 --label paper
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))  # for `common`

import common
from benchmarks import get_benchmark
from objective import build_objective
from paths import DATA_DIR

# Average regret reported for W-DBO in the paper's Table 2, per benchmark, so a
# finished run can print its headline number next to the target.
PAPER_TABLE_2 = {"ackley4d": 2.24}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--benchmark", default="ackley4d", help="Synthetic benchmark name (see benchmarks.py).")
    parser.add_argument("--duration-seconds", type=float, default=600.0, help="Real wall-clock budget per replication (paper default: 600s).")
    parser.add_argument("--n-initial-observations", type=int, default=15, help="Initial observations, drawn over S' x [0, 1/40] per H.1.")
    parser.add_argument("--time-span", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                        help="Range the normalized clock [0,1] is mapped onto for the function's time axis. "
                             "Defaults to the benchmark's own span from Appendix H.2 (Ackley: -32 32).")
    common.add_criterion_arguments(parser)
    parser.add_argument("--n-seeds", type=int, default=10, help="Number of independent replications (paper uses 10).")
    parser.add_argument("--seed", type=int, default=0, help="Base seed; replication i uses seed + i.")
    parser.add_argument("--same-seed", action="store_true", help="Use the same --seed for every replication instead of seed + i. Diagnostic only.")
    parser.add_argument("--oracle-time-points", type=int, default=4000, help="Time samples in the cached oracle curve. Ackley oscillates once per unit of time, so a [-32, 32] span needs a few thousand samples to avoid aliasing.")
    parser.add_argument("--oracle-grid-resolution", type=int, default=33, help="Per-axis spatial grid nodes for the oracle search (use an odd number).")
    parser.add_argument("--oracle-cache", type=Path, default=None, help="Defaults to data/synthetic/<benchmark>/oracle<span tag>.npz.")
    parser.add_argument("--label", default=None, help="Suffix for the timestamped results directory, e.g. --label paper.")
    parser.add_argument("--results-dir", type=Path, default=None, help="Write results here instead of a fresh timestamped directory.")
    args = parser.parse_args()

    benchmark = get_benchmark(args.benchmark)
    span = args.time_span if args.time_span is not None else benchmark.temporal_span
    time_span = (float(span[0]), float(span[1]))
    span_tag = "" if time_span == (0.0, 1.0) else f"_t{time_span[0]:g}_{time_span[1]:g}"

    base_dir = DATA_DIR / benchmark.name
    oracle_cache = args.oracle_cache or base_dir / f"oracle{span_tag}.npz"
    out_dir = args.results_dir or common.results_dir(base_dir / f"results{span_tag}", args.label)
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

    criterion, alpha, mi_options, variant = common.criterion_settings(args)

    runs = []
    for i in range(args.n_seeds):
        prefix = f"seed {i + 1}/{args.n_seeds} " if args.n_seeds > 1 else ""
        seed = args.seed if args.same_seed else args.seed + i
        runs.append(common.run_once(objective, args.duration_seconds, args.n_initial_observations,
                                    alpha, seed, progress_prefix=prefix, criterion=criterion,
                                    min_dataset_size=args.min_dataset_size, mi_options=mi_options))

    title = f"{benchmark.name} (d'={benchmark.dim}), {variant}, {args.duration_seconds:g}s x {args.n_seeds} seed(s)"
    metadata = common.run_metadata(vars(args) | {"time_span": list(time_span)}, extra={
        "benchmark": benchmark.name,
        "criterion": criterion,
        "title": title,
        "noise_std": objective.noise_std,
        "paper_table_2_average_regret": PAPER_TABLE_2.get(benchmark.name),
    })

    per_seed = common.save_run(out_dir, runs, metadata, args.duration_seconds)
    common.save_plots(out_dir, runs, per_seed, args.duration_seconds, title)

    print(f"\n{title}")
    common.print_headline(common.headline(per_seed), reference=PAPER_TABLE_2.get(benchmark.name))
    print(f"\nResults in {out_dir}")


if __name__ == "__main__":
    main()
