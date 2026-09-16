"""Reproduce the WDBO paper's "Temperature" benchmark (Appendix H.1 & H.2).

Runs WDBOOptimizer against the interpolated real-data objective built by
objective.py for a real wall-clock budget, replicated over independent seeds
like the paper's own 10-seed averaging.

Everything downstream of the optimization loop -- the log, the summaries, the
run metadata, the plots -- lives in `../common.py` and is shared with the
synthetic experiment. See README.md for what each output file means.

Usage:
    python experiments/temperature/preprocess.py
    python experiments/temperature/run_experiment.py --n-seeds 10 --label paper
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))  # for `common`

import common
from objective import build_objective
from paths import DATA_DIR

# Average regret reported for W-DBO on Temperature in the paper's Table 2.
PAPER_TABLE_2_AVERAGE_REGRET = 0.68


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--processed", type=Path, default=DATA_DIR / "processed.npz")
    parser.add_argument("--oracle-cache", type=Path, default=DATA_DIR / "oracle.npz")
    parser.add_argument("--smoothing", type=float, default=1.0, help="RBF interpolation smoothing (in temperature units^2).")
    parser.add_argument("--duration-seconds", type=float, default=600.0, help="Real wall-clock budget per replication (paper default: 600s).")
    parser.add_argument("--n-initial-observations", type=int, default=15, help="Initial observations, drawn over S' x [0, 1/40] per H.1.")
    common.add_criterion_arguments(parser)
    parser.add_argument("--n-seeds", type=int, default=10, help="Number of independent replications (paper uses 10).")
    parser.add_argument("--seed", type=int, default=0, help="Base seed; replication i uses seed + i.")
    parser.add_argument("--same-seed", action="store_true", help="Use the same --seed for every replication instead of seed + i. Diagnostic only.")
    parser.add_argument("--label", default=None, help="Suffix for the timestamped results directory, e.g. --label paper.")
    parser.add_argument("--results-dir", type=Path, default=None, help="Write results here instead of a fresh timestamped directory.")
    args = parser.parse_args()

    out_dir = args.results_dir or common.results_dir(DATA_DIR / "results", args.label)

    print("Building objective (RBF fit + oracle grid search, ~30-60s, cached afterwards)...")
    objective = build_objective(args.processed, smoothing=args.smoothing, oracle_cache_path=args.oracle_cache)
    print(f"Noise std (5% of signal variance): {objective.noise_std:.4f}")

    criterion, alpha, mi_options, variant = common.criterion_settings(args)

    runs = []
    for i in range(args.n_seeds):
        prefix = f"seed {i + 1}/{args.n_seeds} " if args.n_seeds > 1 else ""
        seed = args.seed if args.same_seed else args.seed + i
        runs.append(common.run_once(objective, args.duration_seconds, args.n_initial_observations,
                                    alpha, seed, progress_prefix=prefix, criterion=criterion,
                                    min_dataset_size=args.min_dataset_size, mi_options=mi_options))

    title = f"temperature (d'=3), {variant}, {args.duration_seconds:g}s x {args.n_seeds} seed(s)"
    metadata = common.run_metadata(vars(args), extra={
        "benchmark": "temperature",
        "criterion": criterion,
        "title": title,
        "noise_std": objective.noise_std,
        "paper_table_2_average_regret": PAPER_TABLE_2_AVERAGE_REGRET,
    })

    per_seed = common.save_run(out_dir, runs, metadata, args.duration_seconds)
    common.save_plots(out_dir, runs, per_seed, args.duration_seconds, title)

    print(f"\n{title}")
    common.print_headline(common.headline(per_seed), reference=PAPER_TABLE_2_AVERAGE_REGRET)
    print(f"\nResults in {out_dir}")


if __name__ == "__main__":
    main()
