"""Re-render the plots for an existing results directory.

A replication costs ten minutes of wall clock, so the raw per-query log is
written to `queries.csv` and every figure is derived from it. Change a plot,
re-run this -- not the experiment.

Usage:
    python experiments/plot.py data/synthetic/ackley4d/results_t-32_32/20260914-120000-paper
    python experiments/plot.py data/temperature/results/*/        # several at once
"""
import argparse
from pathlib import Path

from common import load_run, save_plots, summarize_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", type=Path, nargs="+", help="Directory holding queries.csv and run.json.")
    parser.add_argument("--title", default=None, help="Figure suptitle. Defaults to the run's own label.")
    args = parser.parse_args()

    for out_dir in args.results_dir:
        runs, metadata = load_run(out_dir)
        duration = float(metadata["args"]["duration_seconds"])
        per_seed = [summarize_seed(run, duration) for run in runs]
        save_plots(out_dir, runs, per_seed, duration, args.title or metadata.get("title", ""))
        print(f"Re-rendered plots in {out_dir} ({len(runs)} seed(s), {duration:g}s)")


if __name__ == "__main__":
    main()
