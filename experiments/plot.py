"""Re-render the plots for an existing results directory.

A replication costs ten minutes of wall clock, so the raw per-query log is
written to `queries.csv` and every figure is derived from it. Change a plot,
re-run this -- not the experiment.

Usage:
    python experiments/plot.py data/synthetic/ackley4d/results_t-32_32/20260914-120000-paper
    python experiments/plot.py data/temperature/results/*/        # several at once
"""
import argparse
import csv
from pathlib import Path

from common import headline, load_objective, load_run, save_plots, score_results, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", type=Path, nargs="+", help="Directory holding queries.csv and run.json.")
    parser.add_argument("--title", default=None, help="Figure suptitle. Defaults to the run's own label.")
    args = parser.parse_args()

    for out_dir in args.results_dir:
        runs, metadata = load_run(out_dir)
        duration = float(metadata["args"]["duration_seconds"])
        objective = load_objective(metadata)
        per_seed, scores = score_results(runs, metadata, objective, duration)
        old_table = out_dir / "per_seed.csv"
        if old_table.exists():
            with old_table.open(newline="") as stream:
                old_rows = list(csv.DictReader(stream))
            if len(old_rows) == len(per_seed):
                provenance = ("warmup_seconds", "elapsed_seconds", "overran",
                              "env_speed", "env_start", "env_end")
                per_seed = [{**new, **{key: old[key] for key in provenance if key in old}}
                            for old, new in zip(old_rows, per_seed)]
        write_csv(out_dir / "per_seed.csv", per_seed)
        write_csv(out_dir / "summary.csv", headline(per_seed))
        save_plots(out_dir, runs, per_seed, scores, duration, args.title or metadata.get("title", ""))
        print(f"Re-scored tables and plots in {out_dir} ({len(runs)} seed(s), {duration:g}s)")


if __name__ == "__main__":
    main()
