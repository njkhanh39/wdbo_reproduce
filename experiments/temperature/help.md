# Reading the results

Running `run_experiment.py` produces **2 CSVs** and **2 image files (3 plot
panels total)** in `data/temperature/results/`. This page explains what each
number and each plot actually shows.

## 1. The metrics — don't mix these up

- **Instantaneous regret**: the regret of *one single query*. At the moment
  W-DBO picks a point `x` at time `t`, instantaneous regret = `best(t) -
  f(x, t)` (best possible value at that instant, minus what we actually
  got). One value per query.
- **Average (running) regret up to `t`**: the *cumulative mean* of every
  instantaneous regret from all queries made so far, up to time `t`. This is
  the `R_t / t` convention: it keeps averaging in more queries as `t` grows,
  so it smooths out and typically trends toward a stable value. **This is
  the paper's convention and the one to compare against Figure 20.**
- **Response time**: real wall-clock seconds one iteration took (pick a
  point + update the model + clean stale data). Not regret — a speed metric.
- **Dataset size**: how many points W-DBO's internal model currently holds,
  after stale ones are cleaned out.
- **Replications / seeds**: each `--n-seeds` run is an independent rerun
  with a different random seed. "Averaged across runs" means: compute the
  metric once per replication, then average those replication-level numbers.

## 2. The files

### `regret.csv`
Time series, one row per time step. If `--n-seeds 1`, each row is one raw
query (instantaneous regret). If `--n-seeds > 1`, each row is the
**instantaneous** regret at that point in time, averaged across
replications — still instantaneous, *not* the running average. Use this if
you want to re-plot or re-analyze the raw curve yourself.

### `summary.csv`
Two headline numbers, each as `mean, variance` across replications:
1. **Average regret up to `t = duration_seconds`** — i.e. the value the left
   plot below reaches at its rightmost point, for each replication, then
   averaged.
2. **Average response time** — the average per-query response time within
   each replication, then averaged across replications.

This is the file to quote when reporting "one number" for the whole run.

### `regret_and_size_vs_duration.png` (2 panels)
- **Left — Regret**: *average regret up to `t`* (not instantaneous) on the
  y-axis, duration (seconds) on the x-axis. Shaded band = ±1 std. dev.
  across replications. Mirrors the paper's Figure 20 (right), regret side.
- **Right — Dataset size**: dataset size (log scale) vs. duration, same
  shaded-band convention. Mirrors Figure 20 (right), size side.

### `regret_vs_response_time.png` (1 panel)
Scatter of every individual query's `(response time, instantaneous regret)`
across all replications (light dots), plus one bold marker at the mean
response time / mean regret with error bars (±1 std across replications).
Mirrors Figure 20 (left) — the paper overlays one such box per baseline
algorithm; since this script only runs W-DBO, there's just one marker here.
